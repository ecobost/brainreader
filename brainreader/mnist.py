"""Experiments in reconstructing mnist.py digits"""
import datajoint as dj
import numpy as np
from scipy import ndimage
import torch

from brainreader import decoding
from brainreader import mnist_classifier
from brainreader import utils
from brainreader import reconstructions

schema = dj.schema('br_mnist')


@schema
class Digits(dj.Manual):
    definition = """ # MNIST digits and labels
    
    mnist_id:   int         # id of this image (1-60K train, 60001-70K test)
    ---
    image:      longblob    # 28 x 28 uint8 image
    split:      varchar(6)  # split it belongs to
    label:      tinyint     # label of this image (0-9)
    """
    @staticmethod
    def fill():
        """ Download MNIST dataset and fill this table."""
        from torchvision import datasets

        # Fill training images
        dset = datasets.MNIST('/tmp', download=True)
        rows = [{'mnist_id': i, 'image': np.array(im), 'split': 'train', 'label': label }
                for i, (im, label) in enumerate(dset, start=1)]
        Digits.insert(rows)

        # Fill test set
        dset = datasets.MNIST('/tmp', train=False, download=True)
        rows = [{'mnist_id': i, 'image': np.array(im), 'split': 'test', 'label': label }
                for i, (im, label) in enumerate(dset, start=60001)]
        Digits.insert(rows)

    def fill_stimulus(self):
        """ Inserts some digits as stimulus. 
        
        It rotates the image 90 degrees counter-clockwise, enlarges it to 144 x 144 and
        pads 56 pixels to the left and right to make a 144 x 256 (16:9 ratio) image.
        
        It used mnist_id as image_id.
        """
        stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')

        # Fetch images
        images, image_ids = self.fetch('image', 'mnist_id')

        # Process them
        processed = []
        for im in images:
            rotated = np.rot90(im)
            resized = ndimage.zoom(rotated, (144/28 ,144/28))
            padded = np.pad(resized, [(0, 0), ((256-144) // 2, (256-144)//2)])
            processed.append(padded)

        # Insert
        stimulus.StaticImageClass.insert1({'image_class': 'mnist'}, skip_duplicates=True)
        stimulus.StaticImage.insert1({'image_class': 'mnist', 'frame_width': 256,
                                      'frame_height': 144, 'num_channels': 1},
                                     skip_duplicates=True)
        rows = [{'image_class': 'mnist', 'image_id': id_, 'image': im} for id_, im in
                zip(image_ids, processed)]
        stimulus.StaticImage.Image.insert(rows)


#TODO: Maybe move to brainreader.utils
def recons2mnist(recons):
    """ Transform reconstruction to MNIST digits. Crop them and restrict them to [0, 1].
     
    Arguments:
        recons (np.array): Array (num_images, height, width) with reconstruction from 
            MNIST digits.
    """
    h, w = recons.shape[1:]

    # Crop
    cropped = recons[:, :, (w - h) // 2: -(w - h) // 2]

    # Resize
    resized = ndimage.zoom(cropped, (1, 28/h, 28/h))

    # Rotate
    rotated = np.rot90(resized, k=-1, axes=(-2, -1))

    # Binarize
    # binarized = 1 / (1 + np.exp(-rotated)) # sigmoid
    min_per_image = rotated.min(axis=(-1, -2), keepdims=True)
    max_per_image = rotated.max(axis=(-1, -2), keepdims=True)
    binarized = (rotated - min_per_image) / (max_per_image - min_per_image)

    return binarized


@schema
class LinearEvaluation(dj.Computed):
    definition = """ # evluate linear reconstruction of MNIST digits
    
    -> decoding.LinearReconstructions
    ---
    mse:        float           # pixel-wise MSE
    corr:       float           # avg correlation across images
    binary_xent:float           # average of binary cross-entropy between MNIST digit and reconstruction
    class_accuracy:float        # average accuracy of an MNIST classifier on the reconstructed images
    """
    @property
    def key_source(self):
        return (
            decoding.LinearReconstructions &
            (decoding.LinearReconstructions.Reconstruction & {'image_class': 'mnist'}))

    def make(self, key):
        # Get reconstructions
        recons_rel = (decoding.LinearReconstructions.Reconstruction & key &
                      {'image_class': 'mnist'})
        recons = recons_rel.fetch('recons', order_by='image_id')
        recons = recons2mnist(np.stack(recons))

        # Get original digits
        images, labels = (Digits & recons_rel.proj(mnist_id='image_id')).fetch(
            'image', 'label', order_by='mnist_id')
        images = np.stack([im.astype(np.float32) / 255 for im in images])

        # Compute MSE and correlation
        mse = ((images - recons)**2).mean()
        corr = utils.compute_imagewise_correlation(images, recons)

        # Compute binary cross entropy
        xent = images * np.log(recons + 1e-8) + (1 - images) * np.log(1 - recons + 1e-8)
        xent = -xent.mean()

        # Classify the digits
        pred_labels = mnist_classifier.classify(recons)
        accuracy = (labels == pred_labels).mean()

        # Insert
        self.insert1({**key, 'mse': mse, 'corr': corr, 'binary_xent': xent,
                      'class_accuracy': accuracy})


@schema
class MLPEvaluation(dj.Computed):
    definition = """ # evaluate mlp reconstruction of MNIST digits
    
    -> decoding.MLPReconstructions
    ---
    mse:        float           # pixel-wise MSE
    corr:       float           # avg correlation across images
    binary_xent:float           # average of binary cross-entropy between MNIST digit and reconstruction
    class_accuracy:float        # average accuracy of an MNIST classifier on the reconstructed images
    """

    @property
    def key_source(self):
        return (decoding.MLPReconstructions &
                (decoding.MLPReconstructions.Reconstruction & {'image_class': 'mnist'}))

    def make(self, key):
        # Get reconstructions
        recons_rel = (decoding.MLPReconstructions.Reconstruction & key &
                      {'image_class': 'mnist'})
        recons = recons_rel.fetch('recons', order_by='image_id')
        recons = recons2mnist(np.stack(recons))

        # Get original digits
        images, labels = (Digits & recons_rel.proj(mnist_id='image_id')).fetch(
            'image', 'label', order_by='mnist_id')
        images = np.stack([im.astype(np.float32) / 255 for im in images])

        # Compute MSE and correlation
        mse = ((images - recons)**2).mean()
        corr = utils.compute_imagewise_correlation(images, recons)

        # Compute binary cross entropy
        xent = images * np.log(recons + 1e-8) + (1 - images) * np.log(1 - recons + 1e-8)
        xent = -xent.mean()

        # Classify the digits
        pred_labels = mnist_classifier.classify(recons)
        accuracy = (labels == pred_labels).mean()

        # Insert
        self.insert1({
            **key, 'mse': mse, 'corr': corr, 'binary_xent': xent,
            'class_accuracy': accuracy})


@schema
class GaborEvaluation(dj.Computed):
    definition = """ # evaluate gabor reconstruction of MNIST digits
    
    -> decoding.GaborReconstructions
    ---
    mse:        float           # pixel-wise MSE
    corr:       float           # avg correlation across images
    binary_xent:float           # average of binary cross-entropy between MNIST digit and reconstruction
    class_accuracy:float        # average accuracy of an MNIST classifier on the reconstructed images
    """

    @property
    def key_source(self):
        return (decoding.GaborReconstructions &
                (decoding.GaborReconstructions.Reconstruction & {'image_class': 'mnist'}))

    def make(self, key):
        # Get reconstructions
        recons_rel = (decoding.GaborReconstructions.Reconstruction & key &
                      {'image_class': 'mnist'})
        recons = recons_rel.fetch('recons', order_by='image_id')
        recons = recons2mnist(np.stack(recons))

        # Get original digits
        images, labels = (Digits & recons_rel.proj(mnist_id='image_id')).fetch(
            'image', 'label', order_by='mnist_id')
        images = np.stack([im.astype(np.float32) / 255 for im in images])

        # Compute MSE and correlation
        mse = ((images - recons)**2).mean()
        corr = utils.compute_imagewise_correlation(images, recons)

        # Compute binary cross entropy
        xent = images * np.log(recons + 1e-8) + (1 - images) * np.log(1 - recons + 1e-8)
        xent = -xent.mean()

        # Classify the digits
        pred_labels = mnist_classifier.classify(recons)
        accuracy = (labels == pred_labels).mean()

        # Insert
        self.insert1({
            **key, 'mse': mse, 'corr': corr, 'binary_xent': xent,
            'class_accuracy': accuracy})


@schema
class AHPEvaluation(dj.Computed):
    definition = """ # evaluate gabor reconstruction of MNIST digits
    
    -> reconstructions.AHPReconstructions
    ---
    mse:        float           # pixel-wise MSE
    corr:       float           # avg correlation across images
    binary_xent:float           # average of binary cross-entropy between MNIST digit and reconstruction
    class_accuracy:float        # average accuracy of an MNIST classifier on the reconstructed images
    """

    @property
    def key_source(self):
        return (reconstructions.AHPReconstructions &
                (reconstructions.AHPReconstructions.Reconstruction & {'image_class': 'mnist'}))

    def make(self, key):
        # Get reconstructions
        recons_rel = (reconstructions.AHPReconstructions.Reconstruction & key &
                      {'image_class': 'mnist'})
        recons = recons_rel.fetch('recons', order_by='image_id')
        recons = recons2mnist(np.stack(recons))

        # Get original digits
        images, labels = (Digits & recons_rel.proj(mnist_id='image_id')).fetch(
            'image', 'label', order_by='mnist_id')
        images = np.stack([im.astype(np.float32) / 255 for im in images])

        # Compute MSE and correlation
        mse = ((images - recons)**2).mean()
        corr = utils.compute_imagewise_correlation(images, recons)

        # Compute binary cross entropy
        xent = images * np.log(recons + 1e-8) + (1 - images) * np.log(1 - recons + 1e-8)
        xent = -xent.mean()

        # Classify the digits
        pred_labels = mnist_classifier.classify(recons)
        accuracy = (labels == pred_labels).mean()

        # Insert
        self.insert1({
            **key, 'mse': mse, 'corr': corr, 'binary_xent': xent,
            'class_accuracy': accuracy})

from brainreader.encoding import train
from brainreader import params

@schema
class GeneratorEvaluation(dj.Computed):
    definition = """ # evaluate gabor reconstruction of MNIST digits
    
    -> train.Ensemble
    -> params.GeneratorMNISTParams
    ---
    mse:        float           # pixel-wise MSE
    corr:       float           # avg correlation across images
    binary_xent:float           # average of binary cross-entropy between MNIST digit and reconstruction
    class_accuracy:float        # average accuracy of an MNIST classifier on the reconstructed images
    """

    @property
    def key_source(self):
        all_keys = train.Ensemble * params.GeneratorMNISTParams
        return all_keys & (reconstructions.GeneratorMNISTReconstruction & {'image_class': 'mnist'})

    def make(self, key):
        # Get reconstructions
        recons_rel = (reconstructions.GeneratorMNISTReconstruction & key &
                      {'image_class': 'mnist'})
        if len(recons_rel) != 10:
            raise ValueError('Some digits may have not been processed yet')
        recons = recons_rel.fetch('recons_digit', order_by='image_id')
        recons = np.stack(recons)

        # Get original digits
        images, labels = (Digits & recons_rel.proj(mnist_id='image_id')).fetch(
            'image', 'label', order_by='mnist_id')
        images = np.stack([im.astype(np.float32) / 255 for im in images])

        # Compute MSE and correlation
        mse = ((images - recons)**2).mean()
        corr = utils.compute_imagewise_correlation(images, recons)

        # Compute binary cross entropy
        xent = images * np.log(recons + 1e-8) + (1 - images) * np.log(1 - recons + 1e-8)
        xent = -xent.mean()

        # Classify the digits
        pred_labels = mnist_classifier.classify(recons)
        accuracy = (labels == pred_labels).mean()

        # Insert
        self.insert1({
            **key, 'mse': mse, 'corr': corr, 'binary_xent': xent,
            'class_accuracy': accuracy})
