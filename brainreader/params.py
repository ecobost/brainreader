import datajoint as dj
import numpy as np
import itertools

from brainreader import data
from brainreader import utils

schema = dj.schema('br_params')
dj.config["enable_python_native_blobs"] = True # allow blob support in dj 0.12


@schema
class ImageNormalization(dj.Lookup):
    definition = """ # methods to normalize images

    img_normalization:  varchar(16)
    ---
    description:        varchar(256)
    """
    contents = [
        {'img_normalization': 'zscore-train',
         'description': 'Normalize images using mean and std calculated in the training '
                        'images'},
    ]


@schema
class ResponseNormalization(dj.Lookup):
    definition = """ # methods to normalize neural responses

    resp_normalization: varchar(16)
    ---
    description:        varchar(256)
    """
    contents = [
        {'resp_normalization': 'zscore-blanks',
         'description': 'Normalize responses (per cell) using mean and std calculated in '
                        'responses to all blank images'},
        # {'resp_normalization': 'stddiv-blanks', 'description': ''}
    ]


@schema
class TestImages(dj.Lookup):
    definition = """ # how to create the test set

    test_set:       varchar(16)
    ---
    description:    varchar(256)
    """
    contents = [
        {'test_set': 'repeats',
         'description': 'Use as test set the images that have been shown more than once'},
    ]


@schema
class DataParams(dj.Lookup):
    definition = """ # data-relevant parameters

    data_params:        smallint    # id of data params
    ---
    -> TestImages                   # how are test images chosen
    split_seed:         smallint    # seed used to get the train/validation split
    train_percentage:   float       # percentage of images (not in test set) used for training, rest are validation
    image_height:       smallint    # height for images
    image_width:        smallint    # width for images
    -> ImageNormalization           # how images are normalized
    only_soma:          boolean     # whether we should restrict to cells that have been calssified as soma
    discard_edge:       smallint    # (um) cells closer to the edge than this will be ignored (negative numbers include everything)
    discard_unknown:    boolean     # whether to discard cells from an unknown area
    -> ResponseNormalization        # how neural responses will be normalized
    """
    contents = [
        {'data_params': 1, 'test_set': 'repeats', 'split_seed': 1234,
         'train_percentage': 0.9, 'image_height': 144, 'image_width': 256,
         'img_normalization': 'zscore-train', 'only_soma': False, 'discard_edge': 8,
         'discard_unknown': True, 'resp_normalization': 'zscore-blanks'
         },
    ]

    def get_images(self, dset_id, split='train'):
        """ Gets images shown during this dataset

        Arguments:
            dset_id (int): A dataset from data.Scan.
            split(str): What split to return: 'train', 'val, 'test'. None will return all
                images.

        Returns:
            images (np.array): An np.float array (num_images x height x width) with the
                images ordered by (image_class, image_id).
        """
        # Get images
        images = (data.Image & (data.Scan.Image & {'dset_id': dset_id})).fetch('image',
                                                                               order_by='image_class, image_id')
        images = np.stack(images).astype(np.float32)

        # Resize
        desired_height, desired_width = self.fetch1('image_height', 'image_width')
        images = utils.resize(images, desired_height, desired_width)

        # Normalize
        img_normalization = self.fetch1('img_normalization')
        if img_normalization == 'zscore-train':
            train_mask = self.get_image_mask(dset_id, 'train')
            img_mean = images[train_mask].mean()
            img_std = images[train_mask].std(axis=(-1, -2)).mean()
        else:
            msg = f'Image normalization {img_normalization} not implemented'
            raise NotImplementedError(msg)
        images = (images - img_mean) / img_std

        # Restrict to right split
        if split is not None:
            img_mask = self.get_image_mask(dset_id, split)
            images = images[img_mask]

        return images

    def get_image_mask(self, dset_id, split='train'):
        """ Creates a boolean mask the same size as the number of images in the scan with
        True for the images in the desired split.

        Arguments:
            dset_id (int): A dataset from data.Scan.
            split (str): What split to return: 'train', 'val, 'test'.

        Returns:
            mask (np.array) A boolean array of size num_images with the desired assignment
                for all images in the dataset (ordered by image_class, image_id).
        """
        # Get test mask
        test_set = self.fetch1('test_set')
        if test_set == 'repeats':
            num_repeats = (data.Scan.Image & {'dset_id': dset_id}).fetch('num_repeats',
                                                                         order_by='image_class, image_id')
            test_mask = num_repeats > 1
        else:
            raise NotImplementedError(f'Test split {test_set} not implemented')

        # Split data
        if split in ['train', 'val']:
            # Set seed for RNG (and preserve previous RNG state)
            prev_state = np.random.get_state()
            np.random.seed(self.fetch1('split_seed'))

            # Compute how many training images we want
            train_percentage = self.fetch1('train_percentage')
            train_images = int(round(np.count_nonzero(~test_mask) * train_percentage))

            # Create a mask with train_images True's in non-test positions at random
            mask = np.zeros(len(test_mask), dtype=bool)
            train_mask = (np.random.permutation(np.count_nonzero(~test_mask)) <
                          train_images)
            mask[~test_mask] = train_mask if split == 'train' else ~train_mask

            # Return random number generator to previous state
            np.random.set_state(prev_state)
        elif split == 'test':
            mask = test_mask
        else:
            raise NotImplementedError(f'Unrecognized {split} split')

        return mask

    def get_responses(self, dset_id, split='train', avg_repeats=True):
        """ Gets responses obtained in this dataset

        Arguments:
            dset_id (int): A dataset from data.Scan.
            split(str): What split to return: 'train', 'val, 'test'. None will return all
                images.
            avg_repeats (bool): Average the responses to an image across repeats. This
                will only work if all images have the same number of repeats.

        Returns:
            responses (np.array): An np.float array (num_images x num_cells if avg_repeats
                else num_images x num_repeats x num_cells) with the responses. Images
                ordered by (image_class, image_id), cells ordered by unit_id.
        """
        # Get all responses
        responses = (data.Responses.PerImage & {'dset_id': dset_id}).fetch(
            'response', order_by='image_class, image_id')

        # Restrict to right split and average
        if split is not None:
            img_mask = self.get_image_mask(dset_id, split)
            responses = responses[img_mask]
        responses = np.stack([r.mean(0) for r in responses] if avg_repeats else responses)
        # will fail if avg_repeats==False and responses have a variable number of repeats

        # Normalize
        resp_normalization = self.fetch1('resp_normalization')
        if resp_normalization == 'zscore-blanks':
            blank_responses = (data.Responses.PerImage & 
                               {'dset_id': dset_id}).fetch('blank_response')
            resp_mean = np.concatenate(blank_responses).mean(0)
            resp_std = np.concatenate(blank_responses).std(0)
        else:
            msg = f'Response normalization {resp_normalization} not implemented'
            raise NotImplementedError(msg)
        responses = (responses - resp_mean) / resp_std

        # Restrict to right cells
        cell_mask = self.get_cell_mask(dset_id)
        responses = responses[..., cell_mask]

        return responses

    def get_cell_mask(self, dset_id):
        """ Creates a boolean mask the same size as the number of cells in the scan with
        True for the cells that fulfill the requirements for this DataParams entry.

        Arguments:
            dset_id (int): A dataset from data.Scan.

        Returns:
            mask (np.array) A boolean array of size num_cells with the cells that fulfill
                the restrictions in this DataParams (ordered by unit_id).
        """
        # Fetch cell properties
        is_somas, edge_distances, areas = (data.Scan.Unit & {'dset_id': dset_id}).fetch(
            'is_soma', 'edge_distance', 'brain_area', order_by='unit_id')

        # Get restrictions
        only_soma, discard_edge, discard_unknown = self.fetch1('only_soma',
                                                               'discard_edge',
                                                               'discard_unknown')

        # Create mask
        mask = np.logical_and(is_somas if only_soma else True,
                              edge_distances > discard_edge,
                              areas != 'unknown' if discard_unknown else True)

        return mask


@schema
class TrainingParams(dj.Lookup):
    definition = """ # training params and hyperparams
    training_params:    smallint    # id of training params
    ---
    seed:               int         # random seed for torch/np
    num_epochs:         int         # number of trainig epochs through the dataset
    val_epochs:         smallint    # run validation every this number of epochs
    batch_size:         smallint    # number of images in each batch
    learning_rate:      decimal(5, 3) # initial learning rate for the optimizer
    momentum:           decimal(3, 2) # momentum factor for SGD updates
    weight_decay:       decimal(7, 5) # weight for l2 regularization
    loss_function:      varchar(8)  # loss function to use ('mse' or 'poisson')
    lr_decay:           decimal(2, 2) # factor multiplying learning rate when decreasing
    decay_epochs:       smallint    # number of epochs to wait before decreasing learning rate if val loss has not improved
    """

    @property
    def contents(self):
        seeds = [1234, 2345, 4567, 5678, 6789]
        lrs = [0.1, 1, 10]
        wds = [0, 1e-4]
        batch_sizes = [32, 64]
        #TODO: losses = ['mse', 'poisson']
        for i, (seed, lr, wd,
                bs) in enumerate(itertools.product(seeds, lrs, wds, batch_sizes),
                                 start=1):
            yield {'training_params': i, 'seed': seed, 'num_epochs': 200, 'val_epochs': 1,
                   'batch_size': bs, 'learning_rate': lr, 'momentum': 0.9,
                   'weight_decay': wd, 'loss_function': 'mse', 'lr_decay': 0.1,
                   'decay_epochs': 10}



############################## MODELS ###############################


@schema
class VGGParams(dj.Lookup):
    definition = """ # vgg inspired feature extractor
    
    core_id:            smallint    # id for vgg nets
    ---
    resized_img_dims:   smallint    # resize the input to this dimension at 1:1 aspect ratio (-1 to avoid it)
    layers_per_block:   tinyblob    # number of layers per block
    features_per_block: tinyblob    # number of feature maps in each block
    use_batchnorm:      boolean     # whether to use batchnorm in the architecture
    """
    contents = [
        {'core_id': 1, 'resized_img_dims': 128, 'layers_per_block': [2, 2, 2, 2, 2],
         'features_per_block': [32, 64, 96, 128, 160], 'use_batchnorm': True},
        {'core_id': 2, 'resized_img_dims': 128, 'layers_per_block': [1, 1, 1, 1, 1],
         'features_per_block': [32, 64, 96, 128, 160], 'use_batchnorm': True},
        {'core_id': 3, 'resized_img_dims': 128, 'layers_per_block': [2, 2, 2],
         'features_per_block': [32, 96, 160], 'use_batchnorm': True}, ]


@schema
class ResNetParams(dj.Lookup):
    definition = """ # resnet inspired feature extractor
    
    core_id:            smallint        # id for resnets
    ---
    resized_img_dims:   smallint        # resize the input to this size (1: 1 aspect ratio), -1 avoids it
    initial_maps:       smallint        # number of feature maps in the very initial layer
    blocks_per_layer:   blob            # how many residual blocks (each 2 conv layers) in each residual "layer"
    compression_factor: float           # how much to decrease/increase feature maps after every residual layer
    use_bottleneck:     boolean         # whether to use bottleneck building blocks
    bottleneck_factor=NULL: float       # how much to reduce feature maps in bottleneck (if used)
    """
    contents = [
        {'core_id': 1, 'resized_img_dims': 128, 'initial_maps': 32,
         'blocks_per_layer': [1, 2, 2, 2, 2, 2], 'compression_factor': 1.4,
         'use_bottleneck': False}, ]


@schema
class DenseNetParams(dj.Lookup):
    definition = """
    
    core_id:            smallint        # id for densenets
    ---
    resized_img_dims:   smallint        # resize the input to this size (1: 1 aspect ratio), -1 avoids it
    initial_maps:       smallint        # number of feature maps in the initial layer
    layers_per_block:   blob            # number of layers in each dense block
    growth_rate:        tinyint         # how many feature maps to add in each layer
    compression_factor: float           # how to increase/decrease feature maps in transition layers
    """
    contents = [
        {'core_id': 1, 'resized_img_dims': 128, 'initial_maps': 32,
         'layers_per_block': [1, 3, 3, 3, 3, 2], 'growth_rate': 32,
         'compression_factor': 0.5},  ]

# class RevNetParams():
#     pass


@schema
class AverageAggParams(dj.Lookup):
    definition = """ # takes an average of each feature across spatial dimensions
    agg_id:             smallint    # id for the aggregator
    """
    contents = [{'agg_id': 1}, ]


@schema
class PointAggParams(dj.Lookup):
    definition = """ # samples features at a single spatial position
    agg_id:             smallint    # id for the aggregator
    """
    contents = [{'agg_id': 1}, ]


@schema
class GaussianAggParams(dj.Lookup):
    definition = """ # samples features with a gaussian mask
    agg_id:             smallint    # id for the aggregator
    ---
    full_covariance:    bool        # whether to use a full covariance matrix or only a diagonal covariance matrix
    """
    contents = [
        {'agg_id': 1, 'full_covariance': True},
        {'agg_id': 2, 'full_covariance': False}, ]


@schema
class FactorizedAggParams(dj.Lookup):
    definition = """ # samples features with a spatially factorized mask
    agg_id:             smallint    # id for the aggregator
    """
    contents = [{'agg_id': 1}, ]


@schema
class LinearAggParams(dj.Lookup):
    definition = """ # samples features with a full (learned) mask
    agg_id:             smallint    # id for the aggregator
    """
    contents = [{'agg_id': 1}, ]


@schema
class MLPParams(dj.Lookup):
    definition = """ # multi layer perceptron applied (separately) to the cell features
    readout_id:             smallint
    ---
    hidden_features:tinyblob    # number of features/units in each hidden layer (ignoring input and output features)
    use_batchnorm:  boolean     # whether to use batchnorm in this mlp
    """
    contents = [{'readout_id': 1, 'hidden_features': [192], 'use_batchnorm': True}, ]


@schema
class NoActParams(dj.Lookup):
    definition = """ # does not apply any activation to the output of the readout
    act_id:                 smallint
    """
    contents = [{'act_id': 1}, ]

# @schema
# class ExponentialActParams(dj.Lookup):
#     definition = """ # final activation applied to the output of the readout
#     act_id:                 smallint
#     """
#     contents = [{'act_id': 1}, ]


@schema
class ModelParams(dj.Lookup):
    definition = """ # parameters to define our model
    
    model_params:       int     # unique id for this network
    ---
    core_type:      varchar(16) # type of feature extractor to use as core of the network
    core_id:        smallint    # what specific instance of the core_type will be used
    agg_type:       varchar(16) # type of aggregator
    agg_id:         smallint    # what specific instance of the aggregator to use
    readout_type:   varchar(16) # type of readout
    readout_id:     smallint    # what specific instance of the readout to use
    act_type:       varchar(16) # type of final activation to use
    act_id:         smallint    # what specific instance of the activation to use   
    """

    @property
    def contents(self):
        cores = [('vgg', 1), ('resnet', 1), ('densenet', 1)]
        aggregators = [('avg', 1), ('point', 1), ('gaussian', 1), ('factorized', 1),
                       ('linear', 1)]
        for i, ((ct, cid), (at, aid)) in enumerate(itertools.product(cores, aggregators),
                                                   start=1):
            yield {'model_params': i, 'core_type': ct, 'core_id': cid, 'agg_type': at,
                   'agg_id': aid, 'readout_type': 'mlp', 'readout_id': 1,
                   'act_type': 'none', 'act_id': 1}

        # Test gaussian aggregator with 4 rather than 5 params
        i = i + 1
        yield {
            'model_params': i, 'core_type': 'vgg', 'core_id': 1, 'agg_type': 'gaussian',
            'agg_id': 2, 'readout_type': 'mlp', 'readout_id': 1, 'act_type': 'none',
            'act_id': 1}

        # Test smaller VGGs (less layers)
        cores = [('vgg', 2), ('vgg', 3)]
        aggregators = [('avg', 1), ('point', 1), ('gaussian', 1), ('factorized', 1),
                       ('linear', 1)]
        for i, ((ct, cid), (at, aid)) in enumerate(itertools.product(cores, aggregators),
                                                   start=i+1):
            yield {'model_params': i, 'core_type': ct, 'core_id': cid, 'agg_type': at,
                   'agg_id': aid, 'readout_type': 'mlp', 'readout_id': 1,
                   'act_type': 'none', 'act_id': 1}


    def get_model(self, num_cells, in_channels=1, out_channels=1):
        """ Builds a network with the desired modules
        
        Arguments:
            num_cells (int): Number of cells to predict
            in_channels (int): Number of channels in the input image. Default: 1 
            out_channels (int): Number of channels in the predicted response. If 1 
                (default), output of network will be a num_cells array, else output will 
                be a num_cells x out_channels array.
            
        Returns
            A nn.Module that receives images and predicts responses per cell.
            
        Note:
            To share models with people with no access to the DB, send them:
                models.py
                core_type, agg_type, readout_type, act_type 
                core_kwargs, readout_kwargs
            and modify this function to use those as input.
        """
        from brainreader import models

        # Build core
        core_type = self.fetch1('core_type')
        if core_type == 'vgg':
            args = ['resized_img_dims', 'layers_per_block', 'features_per_block',
                    'use_batchnorm']
            core_params = (VGGParams & self).fetch1()
            core_kwargs = {k: core_params[k] for k in args}
            core_kwargs['use_batchnorm'] = bool(core_kwargs['use_batchnorm'])
        elif core_type == 'resnet':
            args = ['resized_img_dims', 'initial_maps', 'blocks_per_layer',
                    'compression_factor', 'use_bottleneck', 'bottleneck_factor']
            core_params = (ResNetParams & self).fetch1()
            core_kwargs = {k: core_params[k] for k in args}
            core_kwargs['use_bottleneck'] = bool(core_kwargs['use_bottleneck'])
        elif core_type == 'densenet':
            args = ['resized_img_dims', 'initial_maps', 'layers_per_block', 'growth_rate',
                    'compression_factor']
            core_params = (DenseNetParams & self).fetch1()
            core_kwargs = {k: core_params[k] for k in args}
        else:
            raise NotImplementedError(f'Core {core_type} not implemented')
        core = models.build_extractor(core_type, in_channels=in_channels, **core_kwargs)

        # Build aggregator
        agg_type = self.fetch1('agg_type')
        if agg_type == 'gaussian':
            use_full_cov = (GaussianAggParams & self).fetch1('full_covariance')
            agg_kwargs = {'full_covariance': bool(use_full_cov)}
        elif agg_type in ['factorized', 'linear']:
            agg_kwargs = {'in_height': core.out_height, 'in_width': core.out_width}
        else:
            agg_kwargs = {}
        aggregator = models.build_aggregator(agg_type, num_cells=num_cells, **agg_kwargs)

        # Build readout
        readout_type = self.fetch1('readout_type')
        if readout_type == 'mlp':
            hf, ub = (MLPParams & self).fetch1('hidden_features', 'use_batchnorm')
            num_features = [core.out_channels, *hf, out_channels]  # add input and output channels
            readout_kwargs = {'num_features': num_features, 'use_batchnorm': ub}
        else:
            raise NotImplementedError(f'Readout {readout_type} not implemented.')
        readout = models.build_readout(readout_type, num_cells=num_cells,
                                       **readout_kwargs)

        # Build final activation
        act_type = self.fetch1('act_type')
        final_activation = models.build_activation(act_type)

        # Build final model
        final_model = models.CorePlusReadout(core, aggregator, readout, final_activation)

        return final_model


######################################################################################
@schema
class EnsembleParams(dj.Lookup):
    definition = """ # how to create model ensembles (used for evaluation)
    ensemble_params:    tinyint
    ----
    name:               varchar(16)             # name for this ensemble method
    description:        varchar(256)            # description of the method
    """
    contents = [{'ensemble_params': 1, 'name': 'seeds',
                 'description': ('All models with the same config but different '
                                 'initialization seeds')}]