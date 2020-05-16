import datajoint as dj
import itertools

schema = dj.schema('br_encoding_params')
dj.config["enable_python_native_blobs"] = True # allow blob support in dj 0.12

@schema
class TrainingParams(dj.Lookup):
    definition = """ # training params and hyperparams
    
    training_params:    smallint    # id of training params
    ---
    seed:               int         # random seed for torch/np
    num_epochs:         int         # number of trainig epochs through the dataset
    val_epochs:         smallint    # run validation every this number of epochs
    batch_size:         smallint    # number of images in each batch
    learning_rate:      decimal(8, 4) # initial learning rate for the optimizer
    momentum:           decimal(3, 2) # momentum factor for SGD updates
    weight_decay:       decimal(9, 8) # weight for l2 regularization
    loss_function:      varchar(16) # loss function to use ('mse' or 'poisson')
    lr_decay:           decimal(2, 2) # factor multiplying learning rate when decreasing
    decay_epochs:       smallint    # number of epochs to wait before decreasing learning rate if val correlation has not improved
    stopping_epochs:    smallint    # early stop training after this number of epochs without an improvement in val correlation
    """

    @property
    def contents(self):
        seeds = [1234, 2345, 3456, 5678, 7856]
        wds = [0, 1e-7, 1e-6, 1e-5]
        lrs = [10, 100]
        for i, (seed, lr, wd) in enumerate(itertools.product(seeds, lrs, wds), start=1):
            yield {
                'training_params': i, 'seed': seed, 'num_epochs': 200, 'val_epochs': 1,
                'batch_size': 64, 'learning_rate': lr, 'momentum': 0.9,
                'weight_decay': wd, 'loss_function': 'poisson', 'lr_decay': 0.1,
                'decay_epochs': 5, 'stopping_epochs': 30}



############################## MODELS ###############################

@schema
class KonstiParams(dj.Lookup):
    definition = """ # feature extractor inspired on our previous best one

    core_id:            smallint    # id for vgg nets
    ---
    resized_img_dims:   smallint    # resize the input to this dimension at 1:1 aspect ratio
    features_per_layer: tinyblob    # number of feature maps in each layer
    kernel_sizes:       tinyblob    # kernel size in each layer
    """
    contents = [{'core_id': 1, 'resized_img_dims': 64, 'features_per_layer': [64, ] * 8, 
                 'kernel_sizes': [9, 7, 7, 7, 7, 7, 7, 7]}, ]

@schema
class VGGParams(dj.Lookup):
    definition = """ # vgg inspired feature extractor
    
    core_id:            smallint    # id for vgg nets
    ---
    resized_img_dims:   smallint    # resize the input to this dimension at 1:1 aspect ratio
    layers_per_block:   tinyblob    # number of layers per block
    features_per_block: tinyblob    # number of feature maps in each block
    """
    contents = [{'core_id': 1, 'resized_img_dims': 64, 'layers_per_block': [4, 4],
                 'features_per_block': [64, 64]}, ]


@schema
class ResNetParams(dj.Lookup):
    definition = """ # resnet inspired feature extractor
    
    core_id:            smallint        # id for resnets
    ---
    resized_img_dims:   smallint        # resize the input to this size (1: 1 aspect ratio)
    initial_maps:       smallint        # number of feature maps in the very initial layer
    blocks_per_layer:   blob            # how many residual blocks (each 2 conv layers) in each residual "layer"
    compression_factor: float           # how much to decrease/increase feature maps after every residual layer
    use_bottleneck:     boolean         # whether to use bottleneck building blocks
    bottleneck_factor=NULL: float       # how much to reduce feature maps in bottleneck (if used)
    """
    contents = [
        {'core_id': 1, 'resized_img_dims': 64, 'initial_maps': 64, 
         'blocks_per_layer': [3, 3], 'compression_factor': 1,
         'use_bottleneck': False}, ]


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
    contents = [{'agg_id': 1, 'full_covariance': True}, ]


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
    contents = [{'readout_id': 1, 'hidden_features': [64], 'use_batchnorm': True},]


@schema
class NoActParams(dj.Lookup):
    definition = """ # does not apply any activation to the output of the readout
    act_id:                 smallint
    """
    contents = [{'act_id': 1}, ]


@schema
class ExponentialActParams(dj.Lookup):
    definition = """ # final activation applied to the output of the readout
    
    act_id:                 smallint
    ---
    desired_mean:           float       # assuming input ~ N(0, 1), this will be the mean of the output of this layer
    desired_std:            float       # assuming input ~ N(0, 1), this will be the std of the output of this layer
    """
    contents = [{'act_id': 1, 'desired_mean': 1, 'desired_std': 0.2},]


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
        cores = [('konsti', 1), ('vgg', 1), ('resnet', 1)]
        aggregators = [('avg', 1), ('point', 1), ('gaussian', 1), ('factorized', 1),
                       ('linear', 1)]
        for i, ((ct, cid), (at, aid)) in enumerate(itertools.product(cores, aggregators),
                                                   start=1):
            yield {'model_params': i, 'core_type': ct, 'core_id': cid, 'agg_type': at,
                   'agg_id': aid, 'readout_type': 'mlp', 'readout_id': 1,
                   'act_type': 'exp', 'act_id': 1}
            

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
        from brainreader.encoding import models

        # Build core
        core_type = self.fetch1('core_type')
        if core_type == 'konsti':
            args = ['resized_img_dims', 'features_per_layer', 'kernel_sizes']
            core_params = (KonstiParams & self).fetch1()
            core_kwargs = {k: core_params[k] for k in args}
        if core_type == 'vgg':
            args = ['resized_img_dims', 'layers_per_block', 'features_per_block']
            core_params = (VGGParams & self).fetch1()
            core_kwargs = {k: core_params[k] for k in args}
        elif core_type == 'resnet':
            args = ['resized_img_dims', 'initial_maps', 'blocks_per_layer',
                    'compression_factor', 'use_bottleneck', 'bottleneck_factor']
            core_params = (ResNetParams & self).fetch1()
            core_kwargs = {k: core_params[k] for k in args}
            core_kwargs['use_bottleneck'] = bool(core_kwargs['use_bottleneck'])
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
        if act_type == 'none':
            act_kwargs = {}
        elif act_type == 'exp':
            m, s = (ExponentialActParams & self).fetch1('desired_mean', 'desired_std')
            act_kwargs = {'output_mean': m, 'output_std': s}
        else:
            raise NotImplementedError(f'Activation {act_type} not implemented.')
        final_activation = models.build_activation(act_type, **act_kwargs)

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