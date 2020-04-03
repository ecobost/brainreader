"""Pytorch models
Models are conformed of a core and a readout
"""
import torch
from torch import nn
from torch.nn import functional as F


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
        self.out_channels = features_per_block[
            -1]  # TODO: check whether out_channels will be used (otherwise drop it)

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


class SinglePointAggregator(nn.Module):
    """ Takes the channels at a single x, y position.

    Arguments:
        num_cells (int): Number of cells in the output
    """
    def __init__(self, num_cells):
        self.xy_positions = nn.Parameter(torch.zeros(num_cells, 2))

    def forward(self):
        # grid sample each cells x, y
        pass

    def init_parameters(self):
        #TODO: Probably better to be a noirmal around 0
        nn.init.constant_(self.xy_positions, 0)
    #TODO: How about let the variables be in the normal range and transforming them to [-1, 1] with a tanh after
    # + regularization may work better
    # +  I don't need to use clamp when the numbers go above 1 (which kills the gradient)
    #+ cell positions are not supposed to be acccesible anyway, they are internal to this aggregator
    #TODO: Check the implementation from multi21

    #TODO:
    def get_xy(self, height=128, width=28):
        #returns the x, y positions of each cell
        # define something similar for the other aggregators
        # or maybe return a mask of how stuff got aggregated (so this one will have a single point)
        # mask will be harder for this one but maybe i do that rather than using grid sampling.
        # so instead of grid_sampling I create a mask  based on the xy_positions and go from there
        pass

class GaussianAggregator(nn.Module):
    # Create a h x w mask
    # normalize the mask
    # Multiply input_ * mask and then sum over the hxw axes
    #   other option is to flatten the last two dimensions (with view), flatten the mask and
    #   use matrix multiplication to multiply and sum.
    """
    take a window, compute the x, y at each position, pass it through the gaussian window
    and that will give me the weight at each x,y and multiply that by the feature map and average."""
    pass

class FactorizedAggregator(nn.Module):
    pass

class LinearAggregator(nn.Module):
    pass

aggregators = {'avg': AverageAggregator}
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
        #TODO: Maybe add neuron_idx and average_batch
        self.models = models

    def forward(self, input_):
        return torch.stack([m(input_) for m in self.models]).mean(0)