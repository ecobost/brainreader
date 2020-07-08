""" Evaluate reconstructions using the intermediate features from a VGG.  """

from torchvision import models
import torch
from torch import nn

def get_vgg_features(images, device='cuda'):
    """ Gets features after each conv and linear VGG layer. It also includes original 
    input as first element.
    
    Arguments:
        images (np.array): Images (num_images x height x width).
        device (torch.device): Where to run the computations. 
    
    Returns
        features (list): A list of VGG features as outputted by each conv and linear 
        layer. For convenience, it also includes the input (as the first element in this 
        list).
    """
    vgg = models.vgg19_bn(pretrained=True)
    vgg.to(device)
    vgg.eval()

    # Run vgg layers preserving intermediate results
    x = torch.as_tensor(images[:, None, :, :], dtype=torch.float32,
                        device=device).expand(-1, 3, -1, -1)  # add repeat dimension
    features = [x]
    with torch.no_grad():
        # Run vgg convolutional layers preserving intermediate results
        conv_layers = [
            i for i, l in enumerate(vgg.features, start=1) if isinstance(l, nn.Conv2d)]
        for start, end in zip([0, *conv_layers[:-1]], conv_layers):
            x = vgg.features[start:end](x)
            features.append(x)
        x = vgg.features[conv_layers[-1]:](x)  # any layers after the last conv

        # Average pooling
        x = vgg.avgpool(x)
        x = torch.flatten(x, 1)

        # Run the mlp layers
        lin_layers = [
            i for i, l in enumerate(vgg.classifier, start=1) if isinstance(l, nn.Linear)]
        for start, end in zip([0, *lin_layers[:-1]], lin_layers):
            x = vgg.classifier[start:end](x)
            features.append(x)
        x = vgg.classifier[lin_layers[-1]:](x)  # any layers after the last linear layer

    # Transform features back to arrays
    features = [f.cpu().numpy() for f in features]

    return features

########################################################################################
import datajoint as dj
import numpy as np

from brainreader import params
from brainreader import decoding
from brainreader import utils
from brainreader import reconstructions

schema = dj.schema('br_vgg')


@schema
class LinearEvaluation(dj.Computed):
    definition = """ # evaluate linear model reconstructions in natural images
    
    -> decoding.LinearReconstructions
    ---
    mses:       longblob    # MSE (averaged across images) for each vgg layer (including the input)
    corrs:      longblob    # correlation (computed per feature map and averaged across images & feature_maps) for each layer
    mse_avg:    float       # average mse across all vgg layers
    corr_avg:   float       # average corr across all vgg layers
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (decoding.LinearReconstructions.Reconstruction &
                                 key).fetch('recons', 'image_class',
                                            order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for images_f, recons_f in zip(get_vgg_features(images), get_vgg_features(recons)):
            # In conv layers, compute correlation per feature map
            if images_f.ndim > 2:
                images_f = images_f.reshape(-1, *images_f.shape[-2:])
                recons_f = recons_f.reshape(-1, *recons_f.shape[-2:])

            # Compute metrics
            mse = ((images_f - recons_f) ** 2).mean()
            corr = utils.compute_imagewise_correlation(images_f, recons_f)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})


@schema
class MLPEvaluation(dj.Computed):
    definition = """ # evaluate mlp model reconstructions in natural images
    
    -> decoding.MLPReconstructions
    ---
    mses:       longblob    # MSE (averaged across images) for each vgg layer (including the input)
    corrs:      longblob    # correlation (computed per feature map and averaged across images & feature_maps) for each layer
    mse_avg:    float       # average mse across all vgg layers
    corr_avg:   float       # average corr across all vgg layers
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (decoding.MLPReconstructions.Reconstruction & key).fetch(
            'recons', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for images_f, recons_f in zip(get_vgg_features(images), get_vgg_features(recons)):
            # In conv layers, compute correlation per feature map
            if images_f.ndim > 2:
                images_f = images_f.reshape(-1, *images_f.shape[-2:])
                recons_f = recons_f.reshape(-1, *recons_f.shape[-2:])

            # Compute metrics
            mse = ((images_f - recons_f)**2).mean()
            corr = utils.compute_imagewise_correlation(images_f, recons_f)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})


@schema
class DeconvEvaluation(dj.Computed):
    definition = """ # evaluate deconv model reconstructions in natural images
    
    -> decoding.DeconvReconstructions
    ---
    mses:       longblob    # MSE (averaged across images) for each vgg layer (including the input)
    corrs:      longblob    # correlation (computed per feature map and averaged across images & feature_maps) for each layer
    mse_avg:    float       # average mse across all vgg layers
    corr_avg:   float       # average corr across all vgg layers
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (decoding.DeconvReconstructions.Reconstruction &
                                 key).fetch('recons', 'image_class',
                                            order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for images_f, recons_f in zip(get_vgg_features(images), get_vgg_features(recons)):
            # In conv layers, compute correlation per feature map
            if images_f.ndim > 2:
                images_f = images_f.reshape(-1, *images_f.shape[-2:])
                recons_f = recons_f.reshape(-1, *recons_f.shape[-2:])

            # Compute metrics
            mse = ((images_f - recons_f)**2).mean()
            corr = utils.compute_imagewise_correlation(images_f, recons_f)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})


@schema
class GaborEvaluation(dj.Computed):
    definition = """ # evaluate gaborv model reconstructions in natural images
    
    -> decoding.GaborReconstructions
    ---
    mses:       longblob    # MSE (averaged across images) for each vgg layer (including the input)
    corrs:      longblob    # correlation (computed per feature map and averaged across images & feature_maps) for each layer
    mse_avg:    float       # average mse across all vgg layers
    corr_avg:   float       # average corr across all vgg layers
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (decoding.GaborReconstructions.Reconstruction &
                                 key).fetch('recons', 'image_class',
                                            order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for images_f, recons_f in zip(get_vgg_features(images), get_vgg_features(recons)):
            # In conv layers, compute correlation per feature map
            if images_f.ndim > 2:
                images_f = images_f.reshape(-1, *images_f.shape[-2:])
                recons_f = recons_f.reshape(-1, *recons_f.shape[-2:])

            # Compute metrics
            mse = ((images_f - recons_f)**2).mean()
            corr = utils.compute_imagewise_correlation(images_f, recons_f)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})


@schema
class AHPEvaluation(dj.Computed):
    definition = """ # evaluate gaborv model reconstructions in natural images
    
    -> reconstructions.AHPReconstructions
    ---
    mses:       longblob    # MSE (averaged across images) for each vgg layer (including the input)
    corrs:      longblob    # correlation (computed per feature map and averaged across images & feature_maps) for each layer
    mse_avg:    float       # average mse across all vgg layers
    corr_avg:   float       # average corr across all vgg layers
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & {'data_params': key['ensemble_data']}).get_images(
            key['ensemble_dset'], split='test')

        # Get reconstructions
        recons, image_classes = (reconstructions.AHPReconstructions.Reconstruction &
                                 key).fetch('recons', 'image_class',
                                            order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for images_f, recons_f in zip(get_vgg_features(images), get_vgg_features(recons)):
            # In conv layers, compute correlation per feature map
            if images_f.ndim > 2:
                images_f = images_f.reshape(-1, *images_f.shape[-2:])
                recons_f = recons_f.reshape(-1, *recons_f.shape[-2:])

            # Compute metrics
            mse = ((images_f - recons_f)**2).mean()
            corr = utils.compute_imagewise_correlation(images_f, recons_f)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})
