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

class KonstiNet(nn.Module):
    """ Core inspired by the one developed by Konstantin Willeke.
    
    
    A 4 layer core with input kernel 9 x 9 (no padding), hidden kernels 7 x 7 and 64 
    feature maps in each layer. Uses a 36 x 64 input, output is 28 x 54 x 64 block. It 
    uses depthwise separable convolutions in each layer, i.e., instead of a 7 x 7 x 
    in_features kernel it learns a 7 x 7 and a in_features kernel that get (outer) 
    multiplied to create the full kernel (or applied one after the other for efficiency). 
    Uses a batchnorm and elu activation after each convolution.
    """
    def __init__(self, in_channels=1, resized_img_dims=(36, 64),
                 num_features=(64, 64, 64, 64), kernel_sizes=(9, 7, 7, 7),
                 padding=(0, 3, 3, 3), use_elu=True, use_extra_conv=True,
                 use_normal_conv=False, use_pooling=False):
        super().__init__()

        # Create the layers
        layers = []
        for i, (in_f, out_f, ks, p) in enumerate(zip([in_channels, *num_features], num_features,
                                      kernel_sizes, padding)):
            if use_normal_conv:
                layers.append(nn.Conv2d(in_f, out_f, kernel_size=ks, padding=p, bias=False))
            else:
                if use_extra_conv:
                    layers.append(nn.Conv2d(in_f, in_f, kernel_size=1, bias=False))
                layers.append(nn.Conv2d(in_f, in_f, kernel_size=ks, padding=p, groups=in_f,
                                        bias=False))
                layers.append(nn.Conv2d(in_f, out_f, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.ELU(inplace=True) if use_elu else nn.ReLU(inplace=True))
            if use_pooling and i > 0 and i%2 == 0:
                layers.append(nn.AvgPool2d(2))
        self.layers = nn.Sequential(*layers)

        # Save some params
        self.resized_img_dims = resized_img_dims
        self.out_channels = num_features[-1]
        dim_change = 2 * sum(padding) - sum([k-1 for k in kernel_sizes])
        self.out_height = resized_img_dims[0] + dim_change
        self.out_width = resized_img_dims[1] + dim_change

        if use_pooling:
            self.out_height = self.out_height / 2
            self.out_width = self.out_width / 2

    def forward(self, input_):
        input_ = F.interpolate(input_, size=self.resized_img_dims, mode='bilinear',
                               align_corners=False)  # align_corners avoids warnings
        return self.layers(input_)

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv2d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))


class StaticNet(nn.Module):
    """ Simple 3 layer core similar to the one we use in cajal/static-networks.

    The one we use in static-networks, receives a 36 x 64 input, uses an initial
    convolution with a 15 x 15 kernel and no padding to convert it into a 22 x 50 x 32
    block, then uses two 7x7 convolutions (with padding) and 32 features each. It uses
    batchnorm and an ELU activation after each conv. The output of these three layers
    is concatenated to produce a 22 x 50 x 96 block. Each feature map is then
    successively smoothed with a 5 x 5 gaussian filter (5 times) and the original
    feature map plus the features map that result from subtracting the smoothed
    versions from the original are concatenated to produce the final 22 x 50 x 576
    block.
    """
    def __init__(self, in_channels=1, resized_img_dims=(36, 64),
                 num_features=(32, 32, 32), kernel_sizes=(15, 7, 7), padding=(0, 3, 3),
                 num_downsamplings=5):
        super().__init__()

        # Create layers
        layers = []
        for in_f, out_f, ks, p in zip([in_channels, *num_features], num_features,
                                      kernel_sizes, padding):
            layers.append(nn.Sequential(nn.Conv2d(in_f, out_f, kernel_size=ks, padding=p, bias=False),
                          nn.BatchNorm2d(out_f), nn.ELU(inplace=True)))
        self.layers = nn.ModuleList(layers)

        # Create a 7x7 gaussian mask
        grid_xy = utils.create_grid(7, 7) # 7 x 7 x 2
        gaussian_kernel = utils.bivariate_gaussian(grid_xy.view(-1, 2),
                                                   xy_mean=torch.tensor([[0, 0]]),
                                                   xy_std=torch.tensor([[0.42, 0.42]]),
                                                   corr_xy=torch.tensor([0]))
        gaussian_kernel = gaussian_kernel.view(7, 7)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        self.register_buffer('gaussian_kernel', gaussian_kernel)

        # Save some params
        self.resized_img_dims = resized_img_dims
        self.out_channels = num_features[-1]
        dim_change = 2 * sum(padding) - sum([k - 1 for k in kernel_sizes])
        self.out_height = resized_img_dims[0] + dim_change
        self.out_width = resized_img_dims[1] + dim_change
        self.num_downsamplings = num_downsamplings

    def forward(self, input_):
        resized = F.interpolate(input_, size=self.resized_img_dims, mode='bilinear',
                                align_corners=False)  # align_corners avoids warnings
        parts = []  #TODO: right now because of the padding in the first layer, cna't also add the first layer to the output
        prev = resized
        for l in self.layers:
            prev = l(prev)
            parts.append(prev)
        hidden = torch.cat(parts, dim=1)

        # Create laplace pyramid
        parts = [hidden]
        blurred = hidden
        for i in range(self.num_downsamplings):
            # Blur
            h, w = self.gaussian_kernel.shape
            num_channels = blurred.shape[1]
            padded = F.pad(blurred, pad=(h // 2, h // 2, w // 2, w // 2), mode='reflect')
            blurred = F.conv2d(padded, self.gaussian_kernel.repeat(num_channels, 1, 1, 1),
                               groups=num_channels)

            parts.append(hidden - blurred)
        output = torch.cat(parts, dim=1)

        return output

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv2d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))


class VGG(nn.Module):
    """ A VGG-like network.

     Uses 3 x 3 filters, each block has the same number of feature maps and feature maps 
     are downsampled by a factor of 2 after each block (with an average pooling layer).

    Arguments:
        in_channels (int): Number of channels in the input images.
        resized_img_dims (int): Resize the original input to have 1:1 aspect ratio at this
            size.
        layers_per_block (list of ints): Number of layers in each block.
        features_per_block (list of ints): Number of feature maps in each block. All
            layers in one block have the same number of feature maps.
        use_batchnorm (bool): Whether to add a batchnorm layer after every convolution.
    """
    def __init__(self, in_channels=1, resized_img_dims=128,
                 layers_per_block=[2, 2, 2, 2, 2],
                 features_per_block=[32, 64, 96, 128, 160], use_batchnorm=True):
        super().__init__()

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

        # Save some params
        self.resized_img_dims = resized_img_dims
        self.out_channels = features_per_block[-1]
        self.out_height = resized_img_dims // 2**len(layers_per_block)
        self.out_width = self.out_height


    def forward(self, input_):
        resized = F.interpolate(input_, size=self.resized_img_dims, mode='bilinear',
                                align_corners=False)  # align_corners avoids warnings
        return self.layers(resized)

    def init_parameters(self):
        init_conv(m for m in self.layers if isinstance(m, nn.Conv2d))
        init_bn(m for m in self.layers if isinstance(m, nn.BatchNorm2d))


class ResidualBlock(nn.Module):
    """ A single residual block.
    
    Two layers (or three for bottleneck blocks) and a shortcut connection from the input
    to the output. Preserves spatial dimensions and number of feature maps.
    
    Arguments:
        in_channels (int): Number of feature maps in the input (and output)
        use_bottleneck (bool): Whether to use bottleneck blocks.
        bottleneck_factor (float): Factor to reduce feature maps in the bottleneck layer.
    """
    def __init__(self, in_channels, use_bottleneck, bottleneck_factor):
        super().__init__()

        self.out_channels = in_channels
        if use_bottleneck:
            btn_channels = int(in_channels * bottleneck_factor)
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, btn_channels, 1, bias=False),
                nn.BatchNorm2d(btn_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(btn_channels, btn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(btn_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(btn_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels))
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels))

    def forward(self, input_):
        return F.relu(self.layers(input_) + input_)

    def init_parameters(self):
        init_conv(module for module in self.layers if isinstance(module, nn.Conv2d))
        init_bn(module for module in self.layers if isinstance(module, nn.BatchNorm2d))


class DownsamplingBlock(nn.Module):
    """ Residual block that downsamples spatially the feature maps and increases their
    number.
    
    We depart from the original implementation in that we don't use 1x1 convolutions with
    stride 1 to reduce spatial dimensions and number of feature maps: in bottleneck
    blocks, the first 1 x 1 conv reduces the number of feature maps and the following
    3 x 3 conv does the spatial downsampling (stride=2); in projection shortcuts, we
    first average pool to downsample spatial dimensions and then apply a 1x1 conv to
    increase number of feature maps.
    
    Arguments:
        in_channels (int): Number of feature maps in the input.
        compression_factor (float): Factor to increase/decrease the number of feature maps.
        use_bottleneck (bool): Whether to use bottleneck residual blocks.
        bottleneck_factor (float): Factor to reduce feature maps in the bottleneck layer.
            This will also be multiplied by compression_factor to obtain the number of
            bottleneck feature maps.
    """
    def __init__(self, in_channels, compression_factor, use_bottleneck,
                 bottleneck_factor):
        super().__init__()
        self.out_channels = int(in_channels * compression_factor)
        if use_bottleneck:
            btn_channels = int(in_channels * bottleneck_factor * compression_factor)
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, btn_channels, 1, bias=False),
                nn.BatchNorm2d(btn_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(btn_channels, btn_channels, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(btn_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(btn_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels))
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, 3, padding=1, stride=2,
                          bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels))
        self.projection = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True),  # half spatial dimensions
            nn.Conv2d(in_channels, self.out_channels, 1, bias=False),  # increase feature maps
            nn.BatchNorm2d(self.out_channels))

    def forward(self, input_):
        return F.relu(self.layers(input_) + self.projection(input_))

    def init_parameters(self):
        init_conv(module for module in self.layers if isinstance(module, nn.Conv2d))
        init_bn(module for module in self.layers if isinstance(module, nn.BatchNorm2d))
        init_conv([self.projection[1]])
        init_bn([self.projection[2]])


class ResNet(nn.Module):
    """ Residual network from (He et al., 2016).
    
    We use skip connection every couple of layers (or every three when using bottleneck
    building blocks); when the skip connection goes to a smaller feature map spatial size
    (usually half the original size), we downsampled the input volume with average
    pooling and use 1x1 convolutions to increase the number feature maps (ResNet-B in the
    paper).
    
    Arguments:
        resized_img_dims (int): Resize the original input to have 1:1 aspect ratio at this
            size.
        in_channels (int): Number of channels in the inputs.
        blocks_per_layer (list of ints): Number of building blocks (two or three layers)
            per layer. After each layer we spatially downsample and increase the number of
            feature maps.
        initial_maps (int): Initial number of feature maps.
        compression_factor (int): How much to decrease/increase feature maps after every
            residual layer, e.g., 2 will double the feature maps.
        use_bottleneck (bool): Whether each building block is a bottleneck block; i.e., a
            1x1 conv reducing feature maps, a 3x3 conv preserving feature maps and a 1x1
            conv recovering original feature maps (Fig. 5 in He et al., 2016).
        bottleneck_factor (float): How much to reduce feature maps in bottleneck layers.
            Ignored if use_bottleneck=False.
    """
    def __init__(self, in_channels=1, resized_img_dims=128, initial_maps=32,
                 blocks_per_layer=(2, 2, 2, 2, 2), compression_factor=2,
                 use_bottleneck=False, bottleneck_factor=0.25):
        super().__init__()

        # First conv
        self.conv1 = nn.Conv2d(in_channels, initial_maps, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(initial_maps)

        # Residual blocks
        blocks = []
        for _ in range(blocks_per_layer[0]): # first residual layer
            blocks.append(ResidualBlock(initial_maps, use_bottleneck, bottleneck_factor))
        for bpl in blocks_per_layer[1:]:
            blocks.append(DownsamplingBlock(blocks[-1].out_channels, compression_factor,
                                            use_bottleneck, bottleneck_factor))
            for _ in range(bpl - 1):
                blocks.append(ResidualBlock(blocks[-1].out_channels, use_bottleneck,
                                            bottleneck_factor))
        self.blocks = nn.Sequential(*blocks)

        # Save some params
        self.resized_img_dims = resized_img_dims
        self.out_channels = blocks[-1].out_channels
        self.out_height = resized_img_dims // 2**(len(blocks_per_layer) - 1)
        self.out_width = self.out_height

    def forward(self, input_):
        resized = F.interpolate(input_, size=self.resized_img_dims, mode='bilinear',
                                align_corners=False)  # align_corners avoids warnings
        h1 = F.relu(self.bn1(self.conv1(resized)), inplace=True)
        return self.blocks(h1)

    def init_parameters(self):
        init_conv([self.conv1])
        init_bn([self.bn1])
        for block in self.blocks:
            block.init_parameters()


class DenseBlock(nn.Module):
    """ A single dense block.
    
    Input to each layer is the concatenation (in the channel axes) of the output of every 
    previous layer (and the input to the block). The output of the block is the 
    concatenation of the input and feature maps from all layers.
    
    Arguments:
        in_channels (int): Number of channels in the input.
        growth_rate (int): Number of feature maps to add per layer.
        num_layers (int): Number of layers in this block.
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.out_channels = in_channels + num_layers * growth_rate
        modules = []
        for next_in_channels in range(in_channels, self.out_channels, growth_rate):
            modules.append(
                nn.Sequential(
                    nn.BatchNorm2d(next_in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(next_in_channels, growth_rate, 3, padding=1, bias=False)))
        self.modules_ = nn.ModuleList(modules)

    def forward(self, input_):
        x = input_
        for module in self.modules_:
            x = torch.cat([x, module(x)], dim=1)
        return x

    def init_parameters(self):
        init_bn(module[0] for module in self.modules_)
        init_conv(module[2] for module in self.modules_)


class TransitionLayer(nn.Module):
    """ A transition layer compresses the number of feature maps.
    
    This decreases/increases the number of feature maps with a 1x1 conovlution and 
    downsamples the input spatially with a 2x2 average pooling.
    
    Arguments:
        in_channels (int): Number of channels in the input.
        compression_factor (float): How much to reduce feature maps.
    """
    def __init__(self, in_channels, compression_factor):
        super().__init__()
        self.out_channels = int(in_channels * compression_factor)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
            nn.AvgPool2d(2))
    def forward(self, input_):
        return self.layers(input_)

    def init_parameters(self):
        init_bn([self.layers[0]])
        init_conv([self.layers[2]])


class DenseNet(nn.Module):
    """ DenseNet-C from Huang et al, 2016.
    
    Arguments:
        in_channels (int): Number of channels in the input images.
        resized_img_dims (int): Resize the original input to have 1:1 aspect ratio at this
            size.
        initial_maps (int): Number of maps in the initial layer.
        layers_per_block (list of ints): Number of layers in each dense block. Also
            defines the number of dense blocks in the network.
        growth_rate (int): Number of feature maps to add per layer.
        compression_factor (float): Between each pair of dense blocks, the number of
            feature maps are decreased by this factor, e.g., 0.5 produces half as many.
    """
    def __init__(self, in_channels=1, resized_img_dims=128, initial_maps=32,
                 layers_per_block=(2, 3, 3, 3, 3, 2), growth_rate=32,
                 compression_factor=0.5):
        super().__init__()

        # First conv
        self.conv1 = nn.Conv2d(in_channels, initial_maps, 3, padding=1)

        # Dense blocks and transition layers
        layers = []
        layers.append(DenseBlock(initial_maps, growth_rate, layers_per_block[0]))
        for num_layers in layers_per_block[1:]:
            layers.append(TransitionLayer(layers[-1].out_channels, compression_factor))
            layers.append(DenseBlock(layers[-1].out_channels, growth_rate, num_layers))
        self.layers = nn.Sequential(*layers)
        self.last_bias = nn.Parameter(torch.zeros(self.layers[-1].out_channels))  # *
        # * last conv does not have bias or batchnorm afterwards, so I'll add it manually

        # Save some parameters
        self.resized_img_dims = resized_img_dims
        self.out_channels = len(self.last_bias)
        self.out_height = resized_img_dims // 2**(len(layers_per_block) - 1)
        self.out_width = self.out_height

    def forward(self, input_):
        resized = F.interpolate(input_, size=self.resized_img_dims, mode='bilinear',
                                align_corners=False)  # align_corners avoids warnings
        return self.layers(self.conv1(resized)) + self.last_bias.view(1, -1, 1, 1)

    def init_parameters(self):
        nn.init.constant_(self.last_bias, 0)
        for layer in self.layers:
            layer.init_parameters()


class RevNet(nn.Module):
    #TODO:
    pass

extractors = {'konsti': KonstiNet, 'static': StaticNet, 'vgg': VGG, 'resnet': ResNet, 'densenet': DenseNet}
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
        full_covariance(bool): Whether to use a full covariance matrix. Learns 5 
            parameters (2 means, 2 stds and cov_xy).
    """
    def __init__(self, num_cells, full_covariance=True):
        super().__init__()
        self._xy_mean = nn.Parameter(torch.zeros(num_cells, 2))
        self._xy_std = nn.Parameter(torch.zeros(num_cells, 2))
        if full_covariance:
            self._corr_xy = nn.Parameter(torch.zeros(num_cells))
        else:  # don't learn the correlation
            self.register_buffer('_corr_xy', torch.zeros(num_cells))

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
        # Get masks
        masks = self.get_masks(*input_.shape[-2:])

        # Sample intermediate representation
        input_ = input_.view(*input_.shape[:2], -1).transpose(-1, -2) # N x hw x nchannels
        masks = masks.view(masks.shape[0], -1) # num_cells x hw
        samples = torch.matmul(masks, input_) # N x num_cells x num_channels

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
        grid_xy = utils.create_grid(in_height, in_width)
        grid_xy = grid_xy.view(-1, 2).to(self.xy_mean.device) # in_height*in_widht x 2x 2

        # Get pdf
        masks = utils.bivariate_gaussian(grid_xy, self.xy_mean, self.xy_std, self.corr_xy)
        masks = masks.view(-1, in_height, in_width)

        # Normalize
        norm_masks = masks / masks.sum(dim=(-1, -2), keepdim=True)

        return norm_masks


class FactorizedAggregator(nn.Module):
    """ Learns two vectors per cell and creates the aggregator mask with an outer product.
    
    Arguments:
        num_cells (int): Number of cells in the output.
        in_height (int): Expected height of the input.
        in_width (int): Expected width of the input.
    """
    def __init__(self, num_cells, in_height, in_width):
        super().__init__()
        self._mask_x = nn.Parameter(torch.zeros(num_cells, in_height))
        self._mask_y = nn.Parameter(torch.zeros(num_cells, in_width))

    @property
    def mask_x(self):
        """ Restrict to [0, 1] range. """
        return torch.sigmoid(self._mask_x)

    @property
    def mask_y(self):
        """ Restrict to [0, 1] range. """
        return torch.sigmoid(self._mask_y)

    def forward(self, input_):
        masks = self.get_masks()
        input_ = input_.view(*input_.shape[:2], -1).transpose(-1, -2) # N x hw x nchannels
        masks = masks.view(masks.shape[0], -1) # num_cells x hw
        samples = torch.matmul(masks, input_) # N x num_cells x num_channels
        return samples

    def init_parameters(self):
        nn.init.constant_(self._mask_x, 0)
        nn.init.constant_(self._mask_y, 0)

    def get_masks(self):
        """ Creates the mask each cell applies to the intermediate representation.
        
        This is the outer product of two (learned) 1-dimensional tensors.
  
        Returns:
            A (num_cells, in_height, in_width) tensor with the mask that each cell applied 
            to the intermediate representation to obtain its feature vector.
        """
        masks = self.mask_y.unsqueeze(-1) * self.mask_x.unsqueeze(-2)
        masks = masks / masks.sum(dim=(-1, -2), keepdims=True)
        return masks


class LinearAggregator(nn.Module):
    """ Aggregate intermediate representations with a learned mask.
    
    It learns a in_height x in_width mask per cell.
    
    Arguments:
        num_cells (int): Number of cells in the output.
        in_height (int): Expected height of the input.
        in_width (int): Expected width of the input.
    """
    def __init__(self, num_cells, in_height, in_width):
        super().__init__()
        self._masks = nn.Parameter(torch.zeros(num_cells, in_height, in_width))

    @property
    def masks(self):
        return torch.sigmoid(self._masks)

    def forward(self, input_):
        masks = self.get_masks()
        input_ = input_.view(*input_.shape[:2], -1).transpose(-1, -2) # N x hw x nchannels
        masks = masks.view(masks.shape[0], -1) # num_cells x hw
        samples = torch.matmul(masks, input_) # N x num_cells x num_channels
        return samples

    def init_parameters(self):
        nn.init.constant_(self._masks, 0)

    def get_masks(self):
        """ Creates the mask each cell applies to the intermediate representation.
        
        This is just the sigmoid of the learned parameters.
  
        Returns:
            A (num_cells, in_height, in_width) tensor with the mask that each cell applied 
            to the intermediate representation to obtain its feature vector.
        """
        masks = self.masks
        masks = masks / masks.sum(dim=(-1, -2), keepdims=True)
        return masks


aggregators = {'avg': AverageAggregator, 'point': PointAggregator,
               'gaussian': GaussianAggregator, 'factorized': FactorizedAggregator,
               'linear': LinearAggregator}
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
    
    Arguments:
        output_mean (float): Assuming the input is sampled form a N(0, 1) distribution, 
            the output of this module will be a lognormal distribution with this mean.
        output_std (float): Assuming the input is sampled form a N(0, 1) distribution, 
            the output of this module will be a lognormal distribution with this std.
    """
    def __init__(self, output_mean=1, output_std=0.2):
        super().__init__()

        self.rescale = output_std > 0 # send output_std -1 to not rescan
        if not self.rescale:
            return

        # Compute the required mean and std of the input to get the desired output stats
        import math
        input_var = math.log((output_std / output_mean)**2 + 1)
        self._input_mean = math.log(output_mean) - input_var / 2
        self._input_std = math.sqrt(input_var)

    def forward(self, input_):
        if self.rescale:
            rescaled = input_ * self._input_std + self._input_mean
            return torch.exp(rescaled).squeeze(-1)
        else:
            return torch.exp(input_).squeeze(-1)

    def init_parameters(self):
        pass


class ELUPlusOneActivation(nn.Module):
    """ Returns elu(x) + 1
    
    f: R -> R+
    """
    def forward(self, input_):
        return F.elu(input_.squeeze(-1)) + 1

    def init_parameters(self):
        pass


activations = {'none': NoActivation, 'exp': ExponentialActivation,
               'elu': ELUPlusOneActivation}
def build_activation(type_='none', **kwargs):
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