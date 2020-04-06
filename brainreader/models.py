"""Pytorch models
Models are conformed of a core and a readout
"""
import torch
from torch import nn
from torch.nn import functional as F

from brainreader import utils


def init_conv(modules):
    """ Initializes all module weights using He initialization and set biases to zero."""
    for module in modules:
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_bn(modules):
    """ Initializes all module weights to N(1, 0.1) and set biases to zero."""
    for module in modules:
        nn.init.normal_(module.weight, mean=1, std=0.1)
        nn.init.constant_(module.bias, 0)


################################# CORES #################################################
# Transform input image into an intermediate representation
# Input: h x w x c (c usually 1 or 3)
# Output h' x w' x c'
class VGG(nn.Module):
    """ A VGG-like network.

     Uses 3 x 3 filters, each block has the same number of feature maps and feature maps 
     are downsampled by a factor of 2 after each block (with an average pooling layer).

    Arguments:
        in_channels (int): Number of channels in the input images.
        resized_img_dims (int): Resize the original input to have 1:1 aspect ratio at this
            size. Skip it if -1.
        layers_per_block (list of ints): Number of layers in each block.
        features_per_block (list of ints): Number of feature maps in each block. All
            layers in one block have the same number of feature maps.
        use_batchnorm (bool): Whether to add a batchnorm layer after every convolution.
    """
    #TODO: Change initial_image_size to somethign that cannot be confuised with the initiali size of the input image
    #TODO: Decide on batchnorm nposition
    # TODO: change maps_per_block to features per block or something else

    def __init__(self, in_channels=1, resized_img_dims=128,
                 layers_per_block=[2, 2, 2, 2, 2],
                 features_per_block=[32, 64, 96, 128, 160], use_batchnorm=True):
        super().__init__()

        # Save some params
        self.resized_img_dims = resized_img_dims
        self.in_channels = in_channels
        self.out_channels = features_per_block[-1]

        # Create the layers
        layers = []
        for i, (num_layers, num_features) in enumerate(zip(layers_per_block,
                                                           features_per_block)):
            prev_num_features = in_channels if i == 0 else features_per_block[i - 1]
            for j in range(num_layers):
                in_features = prev_num_features if j == 0 else num_features

                layers.append(nn.Conv2d(in_features, num_features, 3, padding=1,
                              bias=not use_batchnorm))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(num_features))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, input_):
        if self.resized_img_dims > 0:
            input_ = F.interpolate(input_, size=self.resized_img_dims, mode='bilinear',
                                   align_corners=False)  # align_corners is just to avoid warnings
        return self.layers(input_)

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv2d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))

#TODO: Copy the ResNet/DenseNet from multi21 (or bl3d)
class DenseNet(nn.Module):
    pass
class ResNet(nn.Module):
    pass
class RevNet(nn.Module):
    pass

extractors = {'vgg': VGG}
def build_extractor(type_='vgg', **kwargs):
    """ Build a feature extractor module.
    
    Arguments:
        type_ (string): Type of extractor to build.
        kwargs (dictionary): Attributes to send to the extractor constructor.
    
    Returns:
        A module (nn.Module subclass).
    """
    return extractors[type_](**kwargs)


##################################### AGGREGATOR #######################################
# Takes an intermediate representation and produces a representation vector for each cell
# Input: h x w x c
# Output: num_cells x c
class AverageAggregator(nn.Module):
    """ Averages the feature maps across the spatial dimensions.
    
    Arguments:
        num_cells (int): Number of cells in the output. Each cell has the same feature 
            vector.
    """
    def __init__(self, num_cells):
        super().__init__()
        self.num_cells = num_cells

    def forward(self, input_):
        batch_size, num_channels = input_.shape[:2]
        avg = input_.view(batch_size, num_channels, -1).mean(-1)
        return avg.view(batch_size, 1, num_channels).expand(-1, self.num_cells, -1)

    def init_parameters(self):
        pass

    def get_masks(self, in_height, in_width):
        """ Creates the mask each cell applies to the intermediate representation.
        
        This is a flat mask with the same weight everywhere.
        
        Arguments:
            in_height (int): Height of the input to this aggregator.
            in_width (int): Width of the input_ to this aggregator.
            
        Returns:
            A (num_cells, in_height, in_width) tensor with the mask that each cell applied 
            to the intermediate representation to obtain its feature vector.
        """
        masks = torch.ones(self.num_cells, in_height, in_width) / (in_height * in_width)
        return masks


class PointAggregator(nn.Module):
    """ Takes the channels at a single x, y position.

    Arguments:
        num_cells (int): Number of cells in the output.
        
    Note:
        We use grid_sample with align_corners=False; for info of how the sampling happens, 
        check: https://github.com/pytorch/pytorch/issues/20785#issuecomment-494663754
    """
    def __init__(self, num_cells):
        super().__init__()
        self._xy_positions = nn.Parameter(torch.zeros(num_cells, 2))

    @property
    def xy_positions(self):
        """Transforms the internal self._xy_positions to [-1, 1] range"""
        return torch.tanh(self._xy_positions)

    def forward(self, input_):
        grid = self.xy_positions.expand(input_.shape[0], 1, -1, -1)
        samples = F.grid_sample(input_, grid, padding_mode='border', align_corners=False)
        samples = samples.squeeze(-2).transpose(-1, -2)
        return samples

    def init_parameters(self):
        nn.init.normal_(self._xy_positions, mean=0, std=0.25)

    def get_masks(self, in_height, in_width):
        """ Creates the mask each cell applies to the intermediate representation.
        
        Weights for each coordinate in the mask are zero everywhere except for the four 
        corners surrounding the interpolation point (xy_position). For any of this four
        corners, the weight is the area of the square counterdiagonal to it. Check 
        Wikipedia for an illustration.
        
        Arguments:
            in_height (int): Height of the input to this aggregator.
            in_width (int): Width of the input_ to this aggregator.
            
        Returns:
            A (num_cells, in_height, in_width) tensor with the mask that each cell applied 
            to the intermediate representation to obtain its feature vector.
        """
        # Compute distance of each x, y coordinate in the image to the xy_positions
        x = torch.arange(in_width, dtype=torch.float32) + 0.5
        y = torch.arange(in_height, dtype=torch.float32) + 0.5
        xy_points = ((self.xy_positions + 1) / 2) * torch.tensor([in_width, in_height])  # in [0, h] or [0, w] coords
        x_dists = torch.abs(xy_points[:, 0, None] - x)  # num_cells x in_width
        y_dists = torch.abs(xy_points[:, 1, None] - y)  # num_cells x in_height

        # Compute the weights (restricted to points that are closer than 1)
        x_dists = 1 - torch.clamp(x_dists, max=1)
        y_dists = 1 - torch.clamp(y_dists, max=1)
        masks = y_dists[:, :, None] * x_dists[:, None, :]  # num_cells x in_height x in _width

        # Fix masks for padding_mode=borders (drop if using padding_mode=zero)
        masks = masks / masks.sum(dim=(-1, -2), keepdim=True)
        # this works because all xy_positions are restricted to [-1, 1]

        return masks


class GaussianAggregator(nn.Module):
    """ Uses a (learned) gaussian mask to average the feature maps in the input.
    
    Arguments:
        num_cells (int): Number of cells in the output.
    """
    def __init__(self, num_cells):
        super().__init__()
        self._xy_mean = nn.Parameter(torch.zeros(num_cells, 2))
        self._xy_std = nn.Parameter(torch.zeros(num_cells, 2))
        self._corr_xy = nn.Parameter(torch.zeros(num_cells))

    @property
    def xy_mean(self):
        """ Returns mean of the gaussian window in [-1, 1] range."""
        return torch.tanh(self._xy_mean)

    @property
    def xy_std(self):
        return torch.exp(self._xy_std)

    @property
    def corr_xy(self):
        return torch.tanh(self._corr_xy)

    def forward(self, input_):
        masks = self.get_masks(*input_.shape[-2:])
        samples = (input_.unsqueeze(-3) * masks).sum(dim=(-1, -2)).transpose(-1, -2)
        return samples

    def init_parameters(self):
        nn.init.normal_(self._xy_mean, mean=0, std=0.25)
        nn.init.constant_(self._xy_std, -0.7) # std=0.5 in x and y
        nn.init.constant_(self._corr_xy, 0)

    def get_masks(self, in_height, in_width):
        """ Creates the mask each cell applies to the intermediate representation.
        
        This is a gaussian mask with the desired mean and std.
        
        Arguments:
            in_height (int): Height of the input to this aggregator.
            in_width (int): Width of the input_ to this aggregator.
            
        Returns:
            A (num_cells, in_height, in_width) tensor with the mask that each cell applied 
            to the intermediate representation to obtain its feature vector.
        """
        # Get coordinates of input in [-1, 1] range
        device = self._xy_mean.device
        x = torch.arange(in_width, dtype=torch.float32, device=device) + 0.5
        y = torch.arange(in_height, dtype=torch.float32, device=device) + 0.5
        x_coords = 2 * x / in_width - 1
        y_coords = 2 * y / in_height - 1
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords)
        grid_xy = torch.stack([grid_x, grid_y], -1).view(-1, 2)  # in_height*in_widht x 2

        # Get pdf
        masks = utils.bivariate_gaussian(grid_xy, self.xy_mean, self.xy_std, self.corr_xy)
        masks = masks.view(-1, in_height, in_width)

        # Normalize
        masks = masks / masks.sum(dim=(-1, -2), keepdim=True)

        return masks


class FactorizedAggregator(nn.Module):
    #TODO: Given that the mask will be restricted to sum up to 1, should I restrict the
    # norm of each vector? I am afraid if i let it learn the weights directly they may explode.
    pass

class LinearAggregator(nn.Module):
    pass

aggregators = {'avg': AverageAggregator, 'point': PointAggregator,
               'gaussian': GaussianAggregator}
def build_aggregator(type_='avg', **kwargs):
    """ Build an aggregator module.
    
    Arguments:
        type_ (string): Type of aggregator to build.
        kwargs (dictionary): Attributes to send to the aggregator constructor.
    
    Returns:
        A module (nn.Module subclass).
    """
    return aggregators[type_](**kwargs)


##################################### READOUT ##########################################
# Takes a vector representation for each cell and produces a predicted response per cell.
# Input: num_cells x c
# Output: num_cells


class MultipleLinear(nn.Module):
    """ A modified nn.Linear module that applies separate linear transformations to 
    features from different inputs. It could be thought of as applying a different 
    nn.Linear layer to each input and stacking the results.

    Arguments:
         num_inputs (int): Number of cells
         in_features (int): Number of input features per cell
         out_features (int): Number of output features per cell
         bias (bool): Whether to use a bias term in the linear transformation.

    Shape:
        Input: batch_size x num_inputs x in_features
        Output: batch_size x num_inputs x out_features

    Attributes:
         weight: learnable weights shaped (num_inputs, out_features, in_features)
         bias: learnable bias of the module shaped (num_inputs, out_features).

    Returns:
        output (torch.tensor): A tensor with the linear transformation of the features
            from each input.
    """
    def __init__(self, num_inputs, in_features, out_features, bias=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.zeros(num_inputs, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_inputs, out_features))
        else:
            self.register_parameter('bias', None)  # not sure what this does but it is recommended

        self.init_parameters()

    def forward(self, input_):
        # Wx = torch.matmul(self.weight, input_.unsqueeze(-1)).squeeze(-1) # runs out of memory
        Wx = [torch.bmm(self.weight, ex).squeeze(-1) for ex in input_.unsqueeze(-1)]
        Wx = torch.stack(Wx)  # N x num_inputs x out_features
        output = Wx if self.bias is None else Wx + self.bias
        return output

    def init_parameters(self):
        import math
        with torch.no_grad():
            self.weight.normal_(0, math.sqrt(2 / self.weight.shape[-1]))  # He initialization
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def extra_repr(self):
        return 'num_inputs={}, in_features={}, out_features={}, bias={}'.format(
            self.num_inputs, self.in_features, self.out_features, self.bias is not None)


class MultipleBatchnorm(nn.BatchNorm1d):
    """ A modified batchnorm layer that applies a separate batch normalization to features
    from different inputs. It could be thought of as applying a different nn.BatchNorm1d
    layer to each input and stacking the results.

    Arguments:
        num_inputs (int): Number of cells.
        num_features (int): Number of features per cell
        eps, momentum, affine, track_running_stats: Same as nn.BatchNorm1d

    Shape:
        Input: batch_size x num_inputs x num_features
        Output: batch_size x num_inputs x num_features

    Attributes:
         Same as nn.BatchNorm1d.

    Returns:
        output (torch.tensor): A tensor with the normalized features from each input.

    Notes:
        For further documentation, check nn.BatchNorm1d.
    """
    def __init__(self, num_inputs, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_inputs * num_features, eps, momentum, affine,
                         track_running_stats)

    def forward(self, input_):
        return super().forward(input_.view(input_.shape[0], -1)).view(input_.shape)

    def init_parameters(self):
        init_bn([self, ])


class MultipleMLP(nn.Module):
    """ Applies a MLP model (with relu non-linearities) to the features from each cell.

    Last layer does not have a nonlinearity (or bathcnorm layer).

    Input: num_cells x in_features
    Output: num_cells x out_features

    Arguments:
        num_cells (int): Number of cells to predict.
        num_features (list of ints): How many features/units in each layer (includes input
            features and output features, output features are usually 1).
        use_batchnorm (bool): Whether to use batchnorm after each convolution.

    Returns:
        A [num_cells x out_features] tensor with the predicted responses for each cell.
    """
    def __init__(self, num_cells, num_features=[160, 192, 1], use_batchnorm=True):
        super().__init__()

        # Define layers
        layers = []
        for in_f, out_f in zip(num_features[:-2], num_features[1:]):  # all but last layer
            layers.append(MultipleLinear(num_inputs=num_cells, in_features=in_f,
                                         out_features=out_f, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(MultipleBatchnorm(num_inputs=num_cells, num_features=out_f))
            layers.append(nn.ReLU(inplace=True))
        layers.append(MultipleLinear(num_inputs=num_cells, in_features=num_features[-2],
                                     out_features=num_features[-1]))  # last layer

        self.layers = nn.Sequential(*layers)

    def forward(self, input_):
        return self.layers(input_)

    def init_parameters(self):
        for m in self.layers:
            if isinstance(m, (MultipleLinear, MultipleBatchnorm)):
                m.init_parameters()

readouts = {'mlp': MultipleMLP}
def build_readout(type_='mlp', **kwargs):
    """ Build a readout module
    
    Arguments:
        type_ (string): Type of readout to build.
        kwargs (dictionary): Attributes to send to the readout constructor.
    
    Returns:
        A module (nn.Module subclass).
    """
    return readouts[type_](**kwargs)

################################### FINAL ACTIVATIONS ##################################
# Activation applied to the output of readout
class NoActivation(nn.Module):
    """ Sends input as is. Squeezes (if possible) the features/channel dimension."""
    def forward(self, input_):
        return input_.squeeze(-1)

    def init_parameters(self):
        pass

class ExponentialActivation(nn.Module):
    """ Returns exp(x) and squeezes (if possible) the features/channel dimension.
    
    f: R -> R+
    """
    def forward(self, input_):
        return torch.exp(input_).squeeze(-1)

    def init_parameters(self):
        pass

activations = {'none': NoActivation, 'exp': ExponentialActivation}
def build_activation(type_='exp', **kwargs):
    """ Build a final activation module
    
    Arguments:
        type_ (string): Type of activation to build.
        kwargs (dictionary): Attributes to send to the activation constructor.
    
    Returns:
        A module (nn.Module subclass).
    """
    return activations[type_](**kwargs)


################################ Composite model #######################################
# Takes all separate modules and creates the final model.

class CorePlusReadout(nn.Module):
    """ Our main model
    
    Passes an image (h x w x in_channels) through the extractor to get an intermediate 
    representation (h' x w' x c). Aggregator produces a single feature vector per cell 
    (num_cells x c); usually constrained spatially. Readout passes this feature through an 
    MLP (num_cells x out_channels) and a final activation is applied to those outputs 
    (num_cells x out_channels or num_cells if out_channels=1).
    """
    def __init__(self, extractor, aggregator, readout, final_activation): # shifter = None, modulator = None
        super().__init__()
        self.extractor = extractor
        self.aggregator = aggregator
        self.readout = readout
        self.final_activation = final_activation

    def forward(self, input_):
        features = self.extractor(input_)
        agg_features = self.aggregator(features)
        pred_resps_pre_act = self.readout(agg_features)
        pred_resps = self.final_activation(pred_resps_pre_act)
        return pred_resps

    def init_parameters(self):
        self.extractor.init_parameters()
        self.aggregator.init_parameters()
        self.readout.init_parameters()
        self.final_activation.init_parameters()


############################## Ensemble model ##########################################
# Takes a list of models as input and outputs the average of all of them.
# Used for evaluation and to generate feature visualizations and reconstructions.

class Ensemble(nn.Module):
    """ Takes a list of models, applies them one by one and returns average results.
    
    Arguments:
        models (list): A list of nn.Module models. They should all have the same number of 
            input and output channels.
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, input_):
        return torch.stack([m(input_) for m in self.models]).mean(0)