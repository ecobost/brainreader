""" Direct decoding models. """

import datajoint as dj
import numpy as np

from brainreader import params
from brainreader import utils
from brainreader import data


schema = dj.schema('br_decoding')
dj.config["enable_python_native_blobs"] = True  # allow blob support in dj 0.12
dj.config['stores'] = {
    'brdata': {'protocol': 'file', 'location': '/mnt/scratch07/ecobost'}, }
dj.config['cache'] = '/tmp'


################################# Linear decoding #######################################
@schema
class LinearModel(dj.Computed):
    definition = """ # linear model decoding 
    
    -> data.Responses
    -> params.DataParams
    -> params.LinearParams
    ---
    weight:         blob@brdata     # weight matrix (num_pixels x num_cells) of the linear model
    bias:           blob@brdata     # bias (num_pixels) of the linear model
    train_mse:      float           # average MSE in training set
    train_corr:     float           # average correlation in training set
    """
    @property
    def key_source(self):
        all_keys = data.Scan * params.DataParams * params.LinearParams.proj()
        return all_keys & {'data_params': 1} & 'linear_params < 49'

    def make(self, key):
        # Get training data
        utils.log('Fetching data')
        dset_id = key['dset_id']
        train_images = (params.DataParams & key).get_images(dset_id, split='train')
        train_responses = (params.DataParams & key).get_responses(dset_id, split='train')

        # Resize images
        h, w = (params.LinearParams & key).fetch1('image_height', 'image_width')
        train_images = utils.resize(train_images, h, w)
        train_images = train_images.reshape(len(train_images), -1) # num_samples x num_pixels

        #TODO: Normalize responses to be zero-centered per cell?

        # Define model
        model = (params.LinearParams & key).get_model()

        # Fit
        utils.log('Fitting model')
        model.fit(train_responses, train_images)

        # Evaluate
        pred_images = model.predict(train_responses)
        train_mse = ((pred_images - train_images) ** 2).mean()
        train_corr = utils.compute_imagewise_correlation(pred_images, train_images)

        # Insert
        utils.log('Inserting results')
        self.insert1({**key, 'weight': model.coef_, 'bias': model.intercept_,
                      'train_mse': train_mse, 'train_corr': train_corr})

    def get_model(self):
        """ Fetch a trained model. """
        model = (params.LinearParams & self).get_model()

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
        h, w = (params.LinearParams & key).fetch1('image_height', 'image_width')
        recons = recons.reshape(-1, h, w)

        # Compute MSE
        val_mse = ((images - utils.resize(recons, *images.shape[1:])) ** 2).mean()
        resized_val_mse = ((utils.resize(images, *recons.shape[1:]) - recons) ** 2).mean()

        # Compute correlation
        val_corr = utils.compute_imagewise_correlation(images, utils.resize(recons, *images.shape[1:]))
        resized_val_corr = utils.compute_imagewise_correlation(utils.resize(images, *recons.shape[1:]),
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
        old_h, old_w = (params.LinearParams & key).fetch1('image_height', 'image_width')
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
        corr = utils.compute_imagewise_correlation(images, recons)
        pixel_corr = utils.compute_pixelwise_correlation(images, recons)

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
class MLPModel(dj.Computed):
    definition = """ # mlp decoding
    
    -> data.Responses
    -> params.DataParams
    -> params.MLPParams
    ---
    model:          blob@brdata     # weight matrices per layer
    train_mse:      float           # average MSE in training set
    train_corr:     float           # average correlation in training set
    """

    @property
    def key_source(self):
        return data.Scan * params.DataParams * params.MLPParams & {'data_params': 1}
        # all_keys = data.Scan * params.DataParams * params.MLPParams
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
        h, w = (params.MLPData & (params.MLPParams & key)).fetch1('image_height', 'image_width')
        train_images = utils.resize(train_images, h, w).reshape(len(train_images), -1)
        val_images = utils.resize(val_images, h, w).reshape(len(val_images), -1)

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
        hidden_features = (params.MLPArchitecture & (params.MLPParams & key)).fetch1('layer_sizes')
        num_pixels = train_images.shape[-1]
        model = MLP([num_cells, *hidden_features, num_pixels])
        model.cuda()

        # Declare optimizer
        learning_rate, l2_weight = (params.MLPTraining & (params.MLPParams & key)).fetch1(
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
            val_corr = utils.compute_imagewise_correlation(val_images.cpu().numpy(),
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
        train_corr = utils.compute_imagewise_correlation(pred_images.cpu().numpy(), 
                                                         train_images.numpy())

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
        h, w = (params.MLPData & (params.MLPParams & self)).fetch1('image_height', 'image_width')
        num_pixels = h * w

        # Get num features in the hidden layers
        hidden_features = (params.MLPArchitecture & (params.MLPParams & self)).fetch1('layer_sizes')

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
        h, w = (params.MLPData & (params.MLPParams & key)).fetch1('image_height', 'image_width')
        recons = recons.reshape(-1, h, w)

        # Compute MSE
        val_mse = ((images - utils.resize(recons, *images.shape[1:]))**2).mean()
        resized_val_mse = ((utils.resize(images, *recons.shape[1:]) - recons)**2).mean()

        # Compute correlation
        val_corr = utils.compute_imagewise_correlation(images, utils.resize(recons, *images.shape[1:]))
        resized_val_corr = utils.compute_imagewise_correlation(utils.resize(images, *recons.shape[1:]),
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
        old_h, old_w = (params.MLPData & (params.MLPParams & key)).fetch1('image_height',
                                                                           'image_width')
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
        corr = utils.compute_imagewise_correlation(images, recons)
        pixel_corr = utils.compute_pixelwise_correlation(images, recons)

        # Insert
        self.insert1({
            **key, 'test_mse': mse, 'test_corr': corr, 'test_pixel_mse': pixel_mse,
            'test_pixel_corr': pixel_corr})

#TODO: DeconvolutionNetwork

################################## Gabor decoding #######################################
""" Linearly predict weights for a dictionary of gabor filters that are linearly combined 
to produce the reconstruction. Based on Yoshida et Okhi, 2020. """

@schema
class GaborModel(dj.Computed):
    definition = """ # reconstruct constrained to a gabor wavelet set
    
    -> data.Responses
    -> params.DataParams
    -> params.GaborParams
    ---
    weight:         blob@brdata     # weight matrix (num_gabors x num_cells) of the linear regression
    bias:           blob@brdata     # bias (num_gabors) of the linear regression
    train_mse:      float           # average MSE in training set images
    train_corr:     float           # average correlation in training set images
    train_feat_mse: float           # average MSE in Gabor features of training images
    train_feat_corr:float           # average correlation in Gabor features of train images
    """
    @property
    def key_source(self):
        all_keys = data.Scan * params.DataParams * params.GaborParams.proj()
        return all_keys & {'data_params': 1} #&(params.GaborParams & {'gabor_set': 1, 'l1_weight': 0})# & 'gabor_params < 81'

    def make(self, key):
        # Get training data
        utils.log('Fetching data')
        dset_id = key['dset_id']
        train_images = (params.DataParams & key).get_images(dset_id, split='train')
        train_responses = (params.DataParams & key).get_responses(dset_id, split='train')
        min_img_value, max_img_value = train_images.min(), train_images.max()

        # Resize images
        h, w = (params.GaborParams & key).fetch1('image_height', 'image_width')
        train_images = utils.resize(train_images, h, w)
        train_images = train_images.reshape(len(train_images), -1) # num_samples x num_pixels

        # Change image range to [-1, 1]
        rescaled_images = ((train_images - min_img_value) /
                           (max_img_value - min_img_value)) * 2 -1

        # Compute gabor features per image
        gabors = (params.GaborSet & (params.GaborParams & key)).get_gabors(h, w)
        train_features = rescaled_images.reshape(-1, h * w) @ gabors.reshape(-1, h * w).T

        # Define model
        model = (params.GaborParams & key).get_model()

        # Fit
        utils.log('Fitting model')
        model.fit(train_responses, train_features)

        # Evaluate
        pred_features = model.predict(train_responses)
        train_feat_mse = ((pred_features - train_features) ** 2).mean()
        train_feat_corr = utils.compute_correlation(pred_features, train_features).mean()

        # Evaluate on images
        pred_images = (pred_features @ gabors.reshape(-1, h * w))
        pred_images = ((pred_images + 1) * (max_img_value - min_img_value) / 2 +
                       min_img_value)
        train_mse = ((pred_images - train_images) ** 2).mean()
        train_corr = utils.compute_imagewise_correlation(pred_images, train_images)

        # Insert
        utils.log('Inserting results')
        self.insert1({**key, 'weight': model.coef_, 'bias': model.intercept_,
                      'train_mse': train_mse, 'train_corr': train_corr,
                      'train_feat_mse': train_feat_mse,
                      'train_feat_corr': train_feat_corr})

    def get_model(self):
        """ Fetch a trained model. """
        model = (params.GaborParams & self).get_model()

        weight, bias = self.fetch1('weight', 'bias')
        model.coef_ = weight
        model.intercept_ = bias

        return model


@schema
class GaborValEvaluation(dj.Computed):
    definition = """ # evaluation on validation images
    
    -> GaborModel
    ---
    val_mse:            float       # validation MSE computed at the resolution in DataParams
    val_corr:           float       # validation correlation computed at the resolution in DataParams
    resized_val_mse:    float       # validation MSE at the resolution this model was fitted on (GaborParams)
    resized_val_corr:   float       # validation correlation at the resolution this model was fitted on
    """
    def make(self, key):
        # Get data
        images = (params.DataParams & key).get_images(key['dset_id'], split='val')
        responses = (params.DataParams & key).get_responses(key['dset_id'], split='val')

        # Get model
        model = (GaborModel & key).get_model()

        # Create reconstructions
        h, w = (params.GaborParams & key).fetch1('image_height', 'image_width')
        gabors = (params.GaborSet & (params.GaborParams & key)).get_gabors(h, w)
        pred_features = model.predict(responses)
        pred_images = (pred_features @ gabors.reshape(-1, h * w)).reshape(-1, h, w)

        # Rescale back to normalized range (so MSE is comparable with previous models)
        train_images = (params.DataParams & key).get_images(key['dset_id'], split='train')
        min_img_value, max_img_value = train_images.min(), train_images.max()
        recons = ((pred_images + 1) * (max_img_value - min_img_value) / 2 + min_img_value)

        # Compute MSE
        val_mse = ((images - utils.resize(recons, *images.shape[1:])) ** 2).mean()
        resized_val_mse = ((utils.resize(images, *recons.shape[1:]) - recons) ** 2).mean()

        # Compute correlation
        val_corr = utils.compute_imagewise_correlation(images, utils.resize(recons, *images.shape[1:]))
        resized_val_corr = utils.compute_imagewise_correlation(utils.resize(images, *recons.shape[1:]),
                                               recons)

        # Insert
        self.insert1({**key, 'val_mse': val_mse, 'resized_val_mse': resized_val_mse,
                      'val_corr': val_corr, 'resized_val_corr': resized_val_corr})


@schema
class GaborReconstructions(dj.Computed):
    definition = """ # reconstructions for test set images (activity averaged across repeats)
    
    -> GaborModel
    """
    class Reconstruction(dj.Part):
        definition = """ # reconstruction for a single image
        
        -> master
        -> data.Scan.Image
        ---
        features:         longblob  
        recons:           longblob
        """

    def make(self, key):
        # Get data
        responses = (params.DataParams & key).get_responses(key['dset_id'], split='test')

        # Get model
        model = (GaborModel & key).get_model()

        # Create reconstructions
        h, w = (params.GaborParams & key).fetch1('image_height', 'image_width')
        gabors = (params.GaborSet & (params.GaborParams & key)).get_gabors(h, w)
        pred_features = model.predict(responses)
        pred_images = (pred_features @ gabors.reshape(-1, h * w)).reshape(-1, h, w)

        # Rescale to normalized range (so they are in the same range as previous models)
        train_images = (params.DataParams & key).get_images(key['dset_id'], split='train')
        min_img_value, max_img_value = train_images.min(), train_images.max()
        recons = ((pred_images + 1) * (max_img_value - min_img_value) / 2 + min_img_value)

        # Resize to orginal dimensions (144 x 256)
        new_h, new_w = (params.DataParams & key).fetch1('image_height', 'image_width')
        recons = recons.reshape(-1, h, w)
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
        self.Reconstruction.insert([{**key, 'image_class': ic, 'image_id': id_, 'features': f, 'recons': r}
                                     for ic, id_, f, r in zip(image_classes, image_ids, pred_features, recons)])


@schema
class GaborEvaluation(dj.Computed):
    definition = """ # evaluate gabor reconstructions in natural images in test set
    
    -> GaborReconstructions
    ---
    test_mse:       float       # average MSE across all images
    test_corr:      float       # average correlation (computed per image and averaged across images)
    test_pixel_mse: longblob    # pixel-wise MSE (computed per pixel, averaged across images)
    test_pixel_corr:longblob    # pixel-wise correlation (computed per pixel across images)
    """
    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (GaborReconstructions.Reconstruction & key).fetch(
            'recons', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE
        mse = ((images - recons) ** 2).mean()
        pixel_mse = ((images - recons) ** 2).mean(axis=0)

        # Compute corrs
        corr = utils.compute_imagewise_correlation(images, recons)
        pixel_corr = utils.compute_pixelwise_correlation(images, recons)

        # Insert
        self.insert1({**key, 'test_mse': mse, 'test_corr': corr,
                      'test_pixel_mse': pixel_mse, 'test_pixel_corr': pixel_corr})





#TODO: Evaluate by comparing to the reconstructed image too (i..e, reconstruct the image with the gabor bank and correlate to that)

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