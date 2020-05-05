import datajoint as dj
import itertools
import numpy as np

from brainreader import params
from brainreader import utils
from brainreader import data


schema = dj.schema('br_decoding')
dj.config["enable_python_native_blobs"] = True  # allow blob support in dj 0.12
dj.config['stores'] = {
    'brdata': {'protocol': 'file', 'location': '/mnt/scratch07/ecobost'}, }
dj.config['cache'] = '/tmp'


def compute_correlation(images1, images2):
    """ Compute average correlation between two sets of images
    
    Computes correlation per image (i.e., across all pixels) and averages over images.
    
    Arguments:
        images1, images2 (np.array): Array (num_images, height x width) with images.
    
    Returns:
        corr (float): Average correlation across all images.
    """
    num_images = images1.shape[0]
    corrs = utils.compute_correlation(images1.reshape(num_images, -1),
                                      images2.reshape(num_images, -1))

    return corrs.mean()

def compute_pixelwise_correlation(images1, images2):
    """ Compute correlation per pixel (across all images).
    
    Arguments:
        images1, images2 (np.array): Array (num_images, height x width) with images.
    
    Returns:
        pixel_corrs (np.array): A (height, width) array with the correlations for each 
            pixel.
    """
    num_images, height, width = images1.shape
    corrs = utils.compute_correlation(images1.reshape(num_images, -1).T,
                                      images2.reshape(num_images, -1).T)

    return corrs.reshape(height, width)


#########################################################################################
""" Direct decoding models. 

Trained with scikit-learn.
"""
@schema
class LinearParams(dj.Lookup):
    definition = """ # parameters for direct linear decoding
    
    linear_params:  smallint
    ---
    image_height:   smallint        # height of the image to be reconstructed
    image_width:    smallint        # width of the image to be reconstructed
    l2_weight:      float           # weight for the l2 regularization
    l1_weight:      float           # weight for the l1 regularization
    """
    @property
    def contents(self):
        # no regularization
        dims = [(18, 32), (36, 64), (72, 128), (144, 256)]
        for i, (h, w) in enumerate(dims, start=1):
            yield {'linear_params': i, 'image_height': h, 'image_width': w,
                   'l2_weight': 0, 'l1_weight': 0}

        # l2 regularized
        l2_weights = 10 ** np.arange(2, 7.5, 0.5)
        for i, ((h, w), l2) in enumerate(itertools.product(dims, l2_weights), start= i +1):
            yield {'linear_params': i, 'image_height': h, 'image_width': w,
                   'l2_weight': l2, 'l1_weight': 0}

        # l1 regularized
        l1_weights = 10 ** np.arange(-2.5, 0.5, 0.5)
        for i, ((h, w), l1) in enumerate(itertools.product(dims[:2], l1_weights), start= i +1):
            yield {'linear_params': i, 'image_height': h, 'image_width': w,
                   'l2_weight': 0, 'l1_weight': l1}

    def get_model(self, num_processes=8):
        """ Pick the right scikit model: Linear, Ridge, Lasso or ElasticNet.
        
        Arguments:
            num_processes (int): Number of processes to use during fitting. Used only for
                the least squares linear regression (l1_weight=0, l2_weight=0).
        """
        from sklearn import linear_model

        l1_weight, l2_weight = self.fetch1('l1_weight', 'l2_weight')
        if l1_weight == 0 and l2_weight == 0:
            model = linear_model.LinearRegression(n_jobs=num_processes)
        elif l1_weight != 0 and l2_weight == 0:
            model = linear_model.Lasso(alpha=l1_weight, tol=0.01)
        elif l1_weight == 0 and l2_weight !=0:
            model = linear_model.Ridge(alpha=l2_weight, random_state=1234)
        else:
            alpha = l1_weight + l2_weight
            l1_ratio = l1_weight / alpha
            model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        return model

@schema
class LinearModel(dj.Computed):
    definition = """ # linear model decoding 
    
    -> data.Responses
    -> params.DataParams
    -> LinearParams
    ---
    weight:         blob@brdata     # weight matrix (num_pixels x num_cells) of the linear model
    bias:           blob@brdata     # bias (num_pixels) of the linear model
    train_mse:      float           # average MSE in training set
    train_corr:     float           # average correlation in training set
    """
    @property
    def key_source(self):
        all_keys = data.Scan * params.DataParams * LinearParams.proj()
        return all_keys & {'data_params': 1} & 'linear_params < 49'

    def make(self, key):
        # Get training data
        utils.log('Fetching data')
        dset_id = key['dset_id']
        train_images = (params.DataParams & key).get_images(dset_id, split='train')
        train_responses = (params.DataParams & key).get_responses(dset_id, split='train')

        # Resize images
        h, w = (LinearParams & key).fetch1('image_height', 'image_width')
        train_images = utils.resize(train_images, h, w)
        train_images = train_images.reshape(train_images.shape[0], -1) # num_samples x num_pixels

        #TODO: Normalize responses to be zero-centered per cell?

        # Define model
        model = (LinearParams & key).get_model()

        # Fit
        utils.log('Fitting model')
        model.fit(train_responses, train_images)

        # Evaluate
        pred_images = model.predict(train_responses)
        train_mse = ((pred_images - train_images) ** 2).mean()
        train_corr = compute_correlation(pred_images, train_images)

        # Insert
        utils.log('Inserting results')
        self.insert1({**key, 'weight': model.coef_, 'bias': model.intercept_,
                      'train_mse': train_mse, 'train_corr': train_corr})

    def get_model(self):
        """ Fetch a trained model. """
        model = (LinearParams & self).get_model()

        weight, bias = self.fetch1('weight', 'bias')
        model.coef_ = weight
        model.intercept_ = bias

        return model


@schema
class LinearValEvaluation(dj.Computed):
    definition = """ # evaluation on validation images
    
    -> LinearModel
    ---
    val_mse:            float       # validation MSE computed at the resolution in DataParams
    val_corr:           float       # validation correlation computed at the resolution in DataParams
    resized_val_mse:    float       # validation MSE at the resolution this model was fitted on (LinearParams)
    resized_val_corr:   float       # validation correlation at the resolution this model was fitted on
    """
    def make(self, key):
        # Get data
        images = (params.DataParams & key).get_images(key['dset_id'], split='val')
        responses = (params.DataParams & key).get_responses(key['dset_id'], split='val')

        # Get model
        model = (LinearModel & key).get_model()

        # Create reconstructions
        recons = model.predict(responses)
        h, w = (LinearParams & key).fetch1('image_height', 'image_width')
        recons = recons.reshape(-1, h, w)

        # Compute MSE
        val_mse = ((images - utils.resize(recons, *images.shape[1:])) ** 2).mean()
        resized_val_mse = ((utils.resize(images, *recons.shape[1:]) - recons) ** 2).mean()

        # Compute correlation
        val_corr = compute_correlation(images, utils.resize(recons, *images.shape[1:]))
        resized_val_corr = compute_correlation(utils.resize(images, *recons.shape[1:]),
                                               recons)

        # Insert
        self.insert1({**key, 'val_mse': val_mse, 'resized_val_mse': resized_val_mse,
                      'val_corr': val_corr, 'resized_val_corr': resized_val_corr})


@schema
class LinearReconstructions(dj.Computed):
    definition = """ # reconstructions for test set images (activity averaged across repeats)
    
    -> LinearModel
    """
    class Reconstruction(dj.Part):
        definition = """ # reconstruction for a single image
        
        -> master
        -> data.Scan.Image
        ---
        recons:           longblob
        """

    def make(self, key):
        # Get data
        responses = (params.DataParams & key).get_responses(key['dset_id'], split='test')

        # Get model
        model = (LinearModel & key).get_model()

        # Create reconstructions
        recons = model.predict(responses)

        # Resize to orginal dimensions (144 x 256)
        old_h, old_w = (LinearParams & key).fetch1('image_height', 'image_width')
        new_h, new_w = (params.DataParams & key).fetch1('image_height', 'image_width')
        recons = recons.reshape(-1, old_h, old_w)
        recons = utils.resize(recons, new_h, new_w)

        # Find out image ids of test set images
        image_mask = (params.DataParams & key).get_image_mask(key['dset_id'],
                                                              split='test')
        image_classes, image_ids = (data.Scan.Image & key).fetch('image_class',
                                                                 'image_id',
                                                                 order_by='image_class, image_id')
        image_classes = image_classes[image_mask]
        image_ids = image_ids[image_mask]

        # Insert
        self.insert1(key)
        self.Reconstruction.insert([{**key, 'image_class': ic, 'image_id': id_, 'recons': r}
                                     for ic, id_, r in zip(image_classes, image_ids, recons)])


@schema
class LinearEvaluation(dj.Computed):
    definition = """ # evaluate linear model reconstructions in natural images in test set
    
    -> LinearReconstructions
    ---
    test_mse:       float       # average MSE across all image
    test_corr:      float       # average correlation (computed per image and averaged across images)
    test_pixel_mse: longblob    # pixel-wise MSE (computed per pixel, averaged across images)
    test_pixel_corr:longblob    # pixel-wise correlation (computed per pixel across images)
    """
    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (LinearReconstructions.Reconstruction & key).fetch(
            'recons', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE
        mse = ((images - recons) ** 2).mean()
        pixel_mse = ((images - recons) ** 2).mean(axis=0)

        # Compute corrs
        corr = compute_correlation(images, recons)
        pixel_corr = compute_pixelwise_correlation(images, recons)

        # Insert
        self.insert1({**key, 'test_mse': mse, 'test_corr': corr,
                      'test_pixel_mse': pixel_mse, 'test_pixel_corr': pixel_corr})


########################## MLP ##########################################################

import torch
from torch.utils import data as torchdata
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import functional as F
import copy


class MLP(nn.Module):
    """ Multi-layer perceptron.
    
    Arguments:
        num_features (list): Number of features in each layer including input and output
            layer.
    """
    def __init__(self, num_features):
        super().__init__()
        layers = []
        layers.append(nn.Linear(num_features[0], num_features[1]))
        for in_features, out_features in zip(num_features[1:], num_features[2:]):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(in_features, out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def forward_on_batches(self, x, batch_size=256):
        """ Divide the input in batches, forward batch by batch and concatenate output."""
        return torch.cat([self.forward(x[i:i + batch_size]) for i in range(0, len(x),
                                                                           batch_size)])


@schema
class MLPArchitecture(dj.Lookup):
    definition = """ # architectural params for MLP
    
    mlp_architecture:   smallint
    ---
    layer_sizes:        blob        # number of feature maps in each hidden layer
    """
    contents = [(1, [1000]), (2, [5000]), (3, [15000])]


@schema
class MLPTraining(dj.Lookup):
    definition = """ # training hyperparams for mlp
    
    mlp_training:       smallint
    ---
    learning_rate:      float       # initial learning rate for the optimization
    l2_weight:          float       # weight for the l2 regularization
    """
    @property
    def contents(self):
        lrs = [1e-5, 1e-4]
        l2_weights = [0, 1e-4, 1e-3, 1e-2, 1e-1]
        for i, (lr, l2) in enumerate(itertools.product(lrs, l2_weights), start=1):
            yield {'mlp_training': i, 'learning_rate': lr, 'l2_weight': l2}

@schema
class MLPData(dj.Lookup):
    definition = """ # data specific params
    
    mlp_data:           smallint
    ---
    image_height:       smallint            # height of the image to predict
    image_width:        smallint            # width of the image to predict
    """
    contents = [(1, 18, 32), (2, 36, 64), (3, 72, 128), (4, 144, 256)]


@schema
class MLPParams(dj.Lookup):
    definition = """ # parameters to train an MLP decoding model
    
    mlp_params:         smallint
    ---
    -> MLPData
    -> MLPArchitecture
    -> MLPTraining
    """
    @property
    def contents(self):
        for i, (d, a, t) in enumerate(
                itertools.product(MLPData.fetch('mlp_data'),
                                  MLPArchitecture.fetch('mlp_architecture'),
                                  MLPTraining.fetch('mlp_training')), start=1):
            yield {
                'mlp_params': i, 'mlp_data': d, 'mlp_architecture': a, 'mlp_training': t}


@schema
class MLPModel(dj.Computed):
    definition = """ # mlp decoding
    
    -> data.Responses
    -> params.DataParams
    -> MLPParams
    ---
    model:          blob@brdata     # weight matrices per layer
    train_mse:      float           # average MSE in training set
    train_corr:     float           # average correlation in training set
    """

    @property
    def key_source(self):
        return data.Scan * params.DataParams * MLPParams & {'data_params': 1}
        # all_keys = data.Scan * params.DataParams * MLPParams
        # return all_keys & {'data_params': 1, 'mlp_architecture': 2, 'mlp_data': 4}

    def make(self, key):
        # Get data
        utils.log('Fetching data')
        dset_id = key['dset_id']
        train_images = (params.DataParams & key).get_images(dset_id, split='train')
        train_responses = (params.DataParams & key).get_responses(dset_id, split='train')
        val_images = (params.DataParams & key).get_images(dset_id, split='val')
        val_responses = (params.DataParams & key).get_responses(dset_id, split='val')

        # Resize images
        h, w = (MLPData & (MLPParams & key)).fetch1('image_height', 'image_width')
        train_images = utils.resize(train_images, h, w).reshape(train_images.shape[0], -1)
        val_images = utils.resize(val_images, h, w).reshape(val_images.shape[0], -1)

        # Make them tensors
        train_images = torch.as_tensor(train_images, dtype=torch.float32)
        train_responses = torch.as_tensor(train_responses, dtype=torch.float32)
        val_images = torch.as_tensor(val_images, dtype=torch.float32)
        val_responses = torch.as_tensor(val_responses, dtype=torch.float32)

        # Create train dataloader
        train_dset = torchdata.TensorDataset(train_responses, train_images)
        train_dloader = torchdata.DataLoader(train_dset, batch_size=128, num_workers=4,
                                        shuffle=True)

        # Make initialization and training repeatable
        torch.manual_seed(1234)
        torch.backends.cudnn.deterministic = True

        # Define model
        num_cells = train_responses.shape[-1]
        hidden_features = (MLPArchitecture & (MLPParams & key)).fetch1('layer_sizes')
        num_pixels = train_images.shape[-1]
        model = MLP([num_cells, *hidden_features, num_pixels])
        model.cuda()

        # Declare optimizer
        learning_rate, l2_weight = (MLPTraining & (MLPParams & key)).fetch1(
            'learning_rate', 'l2_weight')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=l2_weight)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                   patience=10, verbose=True)

        # Fit
        utils.log('Starting training')
        best_model = copy.deepcopy(model).cpu()
        best_corr = -1
        best_epoch = 1
        for epoch in range(1, 201):
            # Loop over training set
            train_loss = 0
            for responses, images in train_dloader:
                # Zero the gradients
                model.zero_grad()

                # Move variables to GPU
                responses, images = responses.cuda(), images.cuda()

                # Forward
                pred_images = model(responses)

                # Compute loss
                loss = F.mse_loss(pred_images, images)

                # Check for divergence
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError('Loss diverged')

                # Backprop
                loss.backward()
                optimizer.step()

                train_loss = train_loss + loss.item() * len(responses)
            train_loss = train_loss / len(train_responses)

            # Compute validation metrics
            with torch.no_grad():
                pred_images = model.forward_on_batches(val_responses.cuda())
            val_corr = compute_correlation(val_images.cpu().numpy(),
                                           pred_images.cpu().numpy())

            utils.log(f'Train loss (val corr) at epoch {epoch}: {train_loss:.4f} ',
                      f'({val_corr:.4f})')

            # Reduce learning rate
            scheduler.step(val_corr)

            # Save best model yet (if needed)
            if val_corr > best_corr:
                utils.log('Saving best model')
                best_corr = val_corr
                best_model = copy.deepcopy(model).cpu()
                best_epoch = epoch
            elif epoch - best_epoch > 50:
                utils.log('Stopping training. Validation has not improved in 50 '
                          'epochs.')
                break

        # Evaluate best model on training set
        utils.log('Finished training')
        with torch.no_grad():
            best_model.cuda()
            pred_images = best_model(train_responses.cuda())
            best_model.cpu()
        train_mse = ((pred_images.cpu() - train_images)** 2).mean().item()
        train_corr = compute_correlation(pred_images.cpu().numpy(), train_images.numpy())

        # Insert
        utils.log('Inserting results')
        model = {k: v.cpu().numpy() for k, v in best_model.state_dict().items()}
        self.insert1({**key, 'model': model, 'train_mse': train_mse,
                      'train_corr': train_corr})

    def get_model(self):
        """ Fetch a trained model. """
        # Find num_cells for this dataset (with its data_params)
        dset_id = self.fetch1('dset_id')
        cell_mask = (params.DataParams & self).get_cell_mask(dset_id)
        num_cells = np.count_nonzero(cell_mask)

        # Find num_pixels
        h, w = (MLPData & (MLPParams & self)).fetch1('image_height', 'image_width')
        num_pixels = h * w

        # Get num features in the hidden layers
        hidden_features = (MLPArchitecture & (MLPParams & self)).fetch1('layer_sizes')

        # Get model
        model = MLP([num_cells, *hidden_features, num_pixels])

        # Load saved weights
        state_dict = {k: torch.as_tensor(v) for k, v in self.fetch1('model').items()}
        model.load_state_dict(state_dict)

        return model


@schema
class MLPValEvaluation(dj.Computed):
    definition = """ # evaluation on validation images

    -> MLPModel
    ---
    val_mse:            float       # validation MSE computed at the resolution in DataParams
    val_corr:           float       # validation correlation computed at the resolution in DataParams
    resized_val_mse:    float       # validation MSE at the resolution this model was fitted on
    resized_val_corr:   float       # validation correlation at the resolution this model was fitted on
    """

    def make(self, key):
        # Get data
        images = (params.DataParams & key).get_images(key['dset_id'], split='val')
        responses = (params.DataParams & key).get_responses(key['dset_id'], split='val')

        # Get model
        model = (MLPModel & key).get_model()
        model.cuda()

        # Create reconstructions
        with torch.no_grad():
            recons = model(torch.as_tensor(responses, dtype=torch.float32, device='cuda'))
            recons = recons.cpu().numpy()
        h, w = (MLPData & (MLPParams & key)).fetch1('image_height', 'image_width')
        recons = recons.reshape(-1, h, w)

        # Compute MSE
        val_mse = ((images - utils.resize(recons, *images.shape[1:]))**2).mean()
        resized_val_mse = ((utils.resize(images, *recons.shape[1:]) - recons)**2).mean()

        # Compute correlation
        val_corr = compute_correlation(images, utils.resize(recons, *images.shape[1:]))
        resized_val_corr = compute_correlation(utils.resize(images, *recons.shape[1:]),
                                               recons)

        # Insert
        self.insert1({
            **key, 'val_mse': val_mse, 'resized_val_mse': resized_val_mse,
            'val_corr': val_corr, 'resized_val_corr': resized_val_corr})


@schema
class MLPReconstructions(dj.Computed):
    definition = """ # reconstructions for test set images (activity averaged across repeats)
    
    -> MLPModel
    """

    class Reconstruction(dj.Part):
        definition = """ # reconstruction for a single image
        -> master
        -> data.Scan.Image
        ---
        recons:           longblob
        """

    def make(self, key):
        # Get data
        responses = (params.DataParams & key).get_responses(key['dset_id'], split='test')

        # Get model
        model = (MLPModel & key).get_model()
        model.cuda()

        # Create reconstructions
        with torch.no_grad():
            recons = model(torch.as_tensor(responses, dtype=torch.float32, device='cuda'))
            recons = recons.cpu().numpy()

        # Resize to orginal dimensions (144 x 256)
        old_h, old_w = (MLPData & (MLPParams & key)).fetch1('image_height', 'image_width')
        new_h, new_w = (params.DataParams & key).fetch1('image_height', 'image_width')
        recons = recons.reshape(-1, old_h, old_w)
        recons = utils.resize(recons, new_h, new_w)

        # Find out image ids of test set images
        image_mask = (params.DataParams & key).get_image_mask(key['dset_id'],
                                                              split='test')
        image_classes, image_ids = (data.Scan.Image & key).fetch(
            'image_class', 'image_id', order_by='image_class, image_id')
        image_classes = image_classes[image_mask]
        image_ids = image_ids[image_mask]

        # Insert
        self.insert1(key)
        self.Reconstruction.insert(
            [{**key, 'image_class': ic, 'image_id': id_, 'recons': r}
             for ic, id_, r in zip(image_classes, image_ids, recons)])


@schema
class MLPEvaluation(dj.Computed):
    definition = """ # evaluate mlp model reconstructions in natural images in test set
    
    -> MLPReconstructions
    ---
    test_mse:       float       # average MSE across all image
    test_corr:      float       # average correlation (computed per image and averaged across images)
    test_pixel_mse: longblob    # pixel-wise MSE (computed per pixel, averaged across images)
    test_pixel_corr:longblob    # pixel-wise correlation (computed per pixel across images)
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (MLPReconstructions.Reconstruction & key).fetch(
            'recons', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE
        mse = ((images - recons)**2).mean()
        pixel_mse = ((images - recons)**2).mean(axis=0)

        # Compute corrs
        corr = compute_correlation(images, recons)
        pixel_corr = compute_pixelwise_correlation(images, recons)

        # Insert
        self.insert1({
            **key, 'test_mse': mse, 'test_corr': corr, 'test_pixel_mse': pixel_mse,
            'test_pixel_corr': pixel_corr})

#TODO: DeconvolutionNetwork

########################################################################################
""" Decoding from the nature comms paper. """
#TODO: I could do dictonary lerning with a fixed dictionary
#IDEA: or I could create a fake dataset of images where each image is transformed to its gabor weights and then use
# normal scikit-learn linear_model to learn the mapping. this will work

# maybe try gradient reconstruction from a gabor bank encoding model: sum of gabors plus relu. is this better than direct decoding?
# represent the dataset of images as gabor weights and go from gabor weigths to cell responses
# the one hard part is the relu at the end

# they don't use that many gabor filters:

#TODO: the gabor set will be tied to the size of the image though (or won't?), if so just pick one size (but which one)
# If possible make the gabors be independet of image size (i.e., all measurements hsould be relative to the height and width)
# class GaborSet():
#     definition = """ # create gabors with some params
#     gabor_id
#     ---
#     phase
#     position
#     orientations
#     ...
#     """
#     def get_gabors(height, width):
#         """returns the gabors"""
#         pass

# #IDEA: I could just create a fake dataset of images where each image is transformed to its gabor weights and then use
# # normal scikit-learn to

#TODO: Reorganize this so it agrees with one of the options I have below

# class OkhiParams:
#     image size
#     l2 weight
#     -> GaborSet

# class OkhiModel():
#     deinfition = """ train a l2 regularized linar model to predict gabor weights
#     """

# class OkhiReconstruction():
#     definition = """ # reconstruct with average activity

#     """

# class OkhiEvaluation():
#     definition = """
#     -> OkhiReconstructions
#     """

# class OkhiSingleTrialRecons():
#     definition = """ # creates a rescontruction for images in test set (per trial)
#     -> OkhiModel
#     -trial_idx
#     ---
#     singletrialimage
#     """
#     pass
# OR
# class OkhiSingleTrialReconstruction():
#     definition = """
#     -> OkhiModel
#     ---
#     avg_image
#     """
#     class OneTrial():
#         definition = """
#         -> master
#         -> trial_idx
#         ---
#         singletrialimage
#         """

# class OkhiEvaluationSingleTrial():
#     definition = """
#     -> SingleTrialRecons
#     """

# class OkhiAverageSingleTrialReconstruction():
#     defintion = """ # averages all single trial images
#     """
#     # this is how they do it in their paper
#     pass


#TODO: Careful with storing all validation reconstructions, they may be too big

#######################################################################################
#######################################################################################
#######################################################################################
""" Decoding models that use an encoding model"""
from brainreader import train
from brainreader import data

# @schema
# class BestEnsemble(dj.Lookup):
#     definition = """ # collect the best ensemble models for each dataset
#     -> train.Ensemble
#     """
#     contents = [{'ensemble_dset': 1, 'ensemble_data': 1, 'ensemble_model': 2,
#                  'ensemble_training': 9, 'ensemble_params': 1}]

#########################################################################################
""" Gallant decoding. Pass a bunch of images through the model and average the ones that
produce similar activations to the responses we are tryting to match."""

# class ModelResponses(dj.Computed):
#     definition = """ # model responses to all images.
#     -> data.AllImages
#     -> train.Ensemble
#     """

#     class PerImage(dj.Part):
#         definition = """ # model responses to one image (ordered by unit_id)
#         -> master
#         -> data.AllImages.Image
#         ---
#         model_resps:        blob@brdata
#         """
#     @property
#     def key_source(self, key):
#         return data.AllImages * train.Ensemble & BestEnsemble

#     def make(self, key):
#         # Fetch all images
#         # I have to change them so they agree with the dataparams
#         # TODO: Maybe add a function in dataparams that takes images and returns them as the model expects them
#         # Fetch models
#         # Iterate over images getting responses
#         # Save

#     normalize
#         Opt 1: use mean and std of original input or
#         opt 2: use train_mean and train_std from all_images (if this is the case I
#             don't need to fetch norm_method or know how the original input was normalized as long as it was normalized')





# class GallantParams(dj.Lookup):
#     definition = """
#     gallant_params:     smallint
#     ---
#     num_images:         float           # number of images to use for the reconstruction
#     weight_images:      boolean         # whether to weight each image by the similarity of its model response to target response
#     """
#     num_images 1, 5, 10, 50, 100
#     weight in true, false
#     # TODO: should I test different similarities or jsut compute response correlation


# @schema
# class GallantValEvaluation(dj.Computed):
#     definition = """ # evaluation on validation images

#     -> ModelResponses
#     -> GallantParams
#     ---
#     val_mse:            float       # validation MSE computed at the resolution in DataParams
#     val_corr:           float       # validation correlation computed at the resolution in DataParams
#     resized_val_mse:    float       # validation MSE at the resolution this model was fitted on
#     resized_val_corr:   float       # validation correlation at the resolution this model was fitted on
#     """
#     def make(self, key):
#         #TODO: REconstruct all validation images as desired in GallantParams

#         # Get data
#         images = (params.DataParams & key).get_images(key['dset_id'], split='val')
#         responses = (params.DataParams & key).get_responses(key['dset_id'], split='val')

#         # Get model
#         model = (MLPModel & key).get_model()

#         # Create reconstructions
#         recons = model.predict(responses)
#         h, w = (MLPParams & key).fetch1('image_height', 'image_width')
#         recons = recons.reshape(-1, h, w)

#         # Compute MSE
#         val_mse = ((images - utils.resize(recons, *images.shape[1:])) ** 2).mean()
#         resized_val_mse = ((utils.resize(images, *recons.shape[1:]) - recons) ** 2).mean()

#         # Compute correlation
#         val_corr = compute_correlation(images, utils.resize(recons, *images.shape[1:]))
#         resized_val_corr = compute_correlation(utils.resize(images, *recons.shape[1:]),
#                                                recons)

#         # Insert
#         self.insert1({**key, 'val_mse': val_mse, 'resized_val_mse': resized_val_mse,
#                       'val_corr': val_corr, 'resized_val_corr': resized_val_corr})


# @schema
# class GallantReconstructions(dj.Computed):
#     definition = """ # reconstructions for test set images (activity averaged across repeats)

#     -> ModelResponses
#     -> GallantParams
#     """

#     class Reconstruction(dj.Part):
#         definition = """ # reconstruction for a single image
#         -> master
#         -> data.Scan.Image
#         ---
#         recons:           longblob
#         """

#     def make(self, key):
#         #TODO: Reconstruct all test images as desired.
#         # Get data
#         responses = (params.DataParams & key).get_responses(key['dset_id'], split='test')

#         # Get model
#         model = (MLPModel & key).get_model()

#         # Create reconstructions
#         recons = model.predict(responses)

#         # Resize to orginal dimensions (144 x 256)
#         old_h, old_w = (MLPParams & key).fetch1('image_height', 'image_width')
#         new_h, new_w = (params.DataParams & key).fetch1('image_height', 'image_width')
#         recons = recons.reshape(-1, old_h, old_w)
#         recons = utils.resize(recons, new_h, new_w)

#         # Find out image ids of test set images
#         image_mask = (params.DataParams & key).get_image_mask(key['dset_id'],
#                                                               split='test')
#         image_classes, image_ids = (data.Scan.Image & key).fetch(
#             'image_class', 'image_id', order_by='image_class, image_id')
#         image_classes = image_classes[image_mask]
#         image_ids = image_ids[image_mask]

#         # Insert
#         self.insert1(key)
#         self.Reconstruction.insert(
#             [{**key, 'image_class': ic, 'image_id': id_, 'recons': r}
#              for ic, id_, r in zip(image_classes, image_ids, recons)])


# @schema
# class GallantEvaluation(dj.Computed):
#     definition = """ # evaluate gallant reconstructions in natural images in test set

#     -> ModelResponses
#     -> GallantParams
#     ---
#     test_mse:       float       # average MSE across all image
#     test_corr:      float       # average correlation (computed per image and averaged across images)
#     test_pixel_mse: longblob    # pixel-wise MSE (computed per pixel, averaged across images)
#     test_pixel_corr:longblob    # pixel-wise correlation (computed per pixel across images)
#     """

#     def make(self, key):
#         # Get original images
#         images = (params.DataParams & key).get_images(key['dset_id'], split='test')

#         # Get reconstructions
#         recons, image_classes = (GallantReconstructions.Reconstruction & key).fetch(
#             'recons', 'image_class', order_by='image_class, image_id')
#         recons = np.stack(recons)

#         # Restrict to natural images
#         images = images[image_classes == 'imagenet']
#         recons = recons[image_classes == 'imagenet']

#         # Compute MSE
#         mse = ((images - recons)**2).mean()
#         pixel_mse = ((images - recons)**2).mean(axis=0)

#         # Compute corrs
#         corr = compute_correlation(images, recons)
#         pixel_corr = compute_pixelwise_correlation(images, recons)

#         # Insert
#         self.insert1({
#             **key, 'test_mse': mse, 'test_corr': corr, 'test_pixel_mse': pixel_mse,
#             'test_pixel_corr': pixel_corr})


# ########################################################################################
# """ Gradient decoding """"

# class ReconstructionParamds
#     loss = mse, poisson, l1, linfinity


# class ModelReconstruction():
#     pass
#     # reconstruct based on model responses (this is just to get a upper bound of how good the reconstruction can be)
#     # use model responser as a normalization to report result MSE(neural) / MSE(model) (will gie me a 0-1 range)

# class SingleTrialReconstruction():
#     use Data Params with repears=False
#     pass

# class BlankReconstructions():
#   # maybe add a parameter in dataParams.get_reponses that is blank=False, so it fetches the blank responses
#   # rather than the actual responses (but process them the same).
#
"""    
#TODO: Decide whether I like option 3 or 4 here.



# option 1: (like LinearValEvaluation)
# reconstruct all images and evaluate in the same swope (does not save the reconstructions)
class Evaluation
    ->dset_id
    ---
    # reconstruct and evaluate in the same make
    
# option 2: like LinearReconstruction
# all reconstructions in the same step but it also saves them in the intermediate table
class Reconstructions
    -> dset
    # optionally add a set here if I wanna have the same table for both
    class OneImage
        ->dset
        -> imageid
        ---
        recons

class Evaluation
    -> Reconstructions
    # have to check that the image_ids in OneImage agree witht the ones in cell_mask


# option 3: like 
# reconstructions are done image by image and stored
class OneImageReconstruction
    -> image_id
    ---
     recons

class Reconstructions
    -> dset_id
    class OneImage
        ->dset_id
        -> image_id
    # check that all the images for this set (be it validation or test) are here

class Evaluation
    -> Reconstructions
    
# option 4:
# have a single OneImageReconstruction table that does the reconstructions from test set and validation set images rather than a different and lump them together in te ReconstructionSet table.
# - it mixes validation evaluations with test set evaluations: will have 30 validations with the same set and one test (all the tested validation models plus the final model)
class OneImageReconstruction
    dset_id
    image_id
    has all reconstructions (from test set and validation set)
    #TODO: how do I restrict this to only populate test and validation
    #TODO: Reconsider having a table that record for each image in dset id for any specific data params where it comes from  

class SetId(lookup):
    set: varchar(8)
    {'set': 'val', 'set': 'test'}

class Reconstructions():
    dset_id
    set_id
    class OneImage
    
    def make():
        just create a set as desired by set_id
        # also restrict to only stuff in image_class='imaenet

class Evaluation
    -> Reconstructions
    # restrict to only those in the set (maybe fetch all )
    
# Single Trial wll be like option 3 (or 4) but each OneImageRecons will have a part table with the  trials, i.e., the reconstructions will be done per chunks of 10 (or 40) trials in a single make.

"""