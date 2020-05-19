from scipy import ndimage
import numpy as np
import time
import torch


def resize(images, desired_height=72, desired_width=128, order=1):
    """ Resize images to the desired height and width.

    Arguments:
        images (np.array): Array with images (num_images x height x widht).
        desired_height (int): Desired height for the output images
        desired_width (int): Desired height for the output images
        order (int): Order of the interpolation used by ndimage.zoom (see docs there).

    Returns:
        images (np.array). Array with the resized images (num_images x desired_height x
            desired_width).
    """
    zoom = (desired_height / images.shape[-2], desired_width / images.shape[-1])
    if zoom != (1, 1):  # ndimage does not check for this
        images = np.stack([ndimage.zoom(im, zoom, order=order) for im in images])
    return images


def log(*messages):
    """ Prints a message (with a timestamp next to it).
    
    Arguments:
        message (string): Arguments to be print()'d.
    """
    formatted_time = '[{}]'.format(time.ctime())
    print(formatted_time, *messages, flush=True)


def compute_correlation(x, y, eps=1e-8):
    """ Compute Pearson's correlation for each row in x and y.
    
    Arguments:
        x, y (np.array): A (num_variables, num_observations) array containining the 
            variables to correlate.
        eps (float): Small value to avoid division by zero.
    
    Returns:
        corrs (np.array): A (num_variables) array with the correlations between each 
        variable in x and y.  
    """
    # Compute correlation per variable
    x_res = x - x.mean(axis=-1, keepdims=True)
    y_res = y - y.mean(axis=-1, keepdims=True)
    corrs = (x_res * y_res).sum(axis=-1) / (np.sqrt(
        (x_res**2).sum(axis=-1) * (y_res**2).sum(axis=-1)) + eps)

    return corrs

def create_grid(height, width):
    """ Creates a sampling grid (from -1 to 1) of the desired dimensions.
    
    Arguments:
        height (int): Desired height.
        width (int): Desired width.
    
    Returns:
        grid_xy (torch.tensor): A grid array (height x width x 2) with the positions of 
            each coordinate in x and y.
    """
    x = torch.arange(width, dtype=torch.float32) + 0.5
    y = torch.arange(height, dtype=torch.float32) + 0.5
    x_coords = 2 * x / width - 1
    y_coords = 2 * y / height - 1
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords)
    grid_xy = torch.stack([grid_x, grid_y], -1)

    return grid_xy


def bivariate_gaussian(xy, xy_mean, xy_std, corr_xy, normalize=False, eps=1e-8):
    """ Compute the pdf of a bivariate gaussian distribution.
    
    Arguments:
        xy (torch.tensor): Points. A (M, 2) tensor.
        xy_mean (torch.tensor): Mean of the distributions (in xy). Shape: (N, 2). N is the
            number of different gaussian distributions this points will be evaluated in.
        xy_std (torch tensor): Standard deviation in x and y. Shape: (N, 2).
        corr_xy (torch.tensor): Correlation between x and y. Shape: (N).
        normalize (boolean): Normalize the pdf values so it is a valid pdf (sums to 1).
        eps (float): Small number to avoid division by zero.
    
    Returns:
        pdfs (torch.tensor): A tensor (N, M) with the pdf values at each position.
    """
    # Make all inputs broadcastable (N, M, 2)
    xy = xy.unsqueeze(0)
    xy_mean = xy_mean.unsqueeze(1)
    xy_std = xy_std.unsqueeze(1)
    corr_xy = corr_xy.unsqueeze(-1)  # used after last dimension is reduced (so [N, 1])

    # Compute pdf
    residuals = (xy - xy_mean) / (xy_std + eps)
    numer = 2 * corr_xy * torch.prod(residuals, dim=-1) - torch.sum(residuals**2, dim=-1)
    pdf = torch.exp(numer / (2 * (1 - corr_xy**2) + eps))

    # normalize pdf if needed
    if normalize:
        import math
        divisor = 2 * math.pi * torch.prod(xy_std, dim=-1) * torch.sqrt(1 - corr_xy**2)
        pdf = pdf / (divisor + eps)

    return pdf


def create_gabor(height, width, orientation, phase, wavelength, sigma, dx=0, dy=0):
    """Create a Gabor patch.
    
    Arguments:
        height (int): Height of the gabor in pixels.
        width (int): Width of the gabor in pixels.
        orientation (float): Orientation of the gabor in radians.
        phase (float): Phase of the gabor patch in radians.
        wavelength (float): Wavelength of the gabor expressed as a proportion of height.
        sigma (float): Standard deviation of the gaussian window expressed as a proportion
            of height.
        dx (float): Amount of shifting in x (expressed as a proportion of width)
            [-0.5, 0.5]. Positive moves the gabor to the right.
        dy (float): Amount of shifting in y (expressed as a  proportion of height) 
            [-0.5, 0.5]. Positive moves the gabor down.
            
    Returns:
        A Gabor patch (np.array) with the desired properties.
        
    Note:
        This diverges from the Gabor formulation in https://en.wikipedia.org/wiki/Gabor_filter:
        * theta is the orientation of the gabor rather than "the orientation of the normal
            to the parallel stripes" (i.e., theta = wikipedia_theta - pi/2).
        * rotations are counterclockwise (i.e theta = - wikipedia_theta).
        * for some dx, dy, the gaussian mask will always be in the same place regardless
            of orientation.
        * sigma and wavelength are expressed as proportions of height rather than width.
    """
    # Basic checks
    # orientation = orientation % np.pi # 0-180
    # phase = phase % (2 * np.pi) # 0-360
    if wavelength <= 0:
        raise ValueError('wavelength needs to be positive')
    if sigma <= 0:
        raise ValueError('sigma needs to be positive')

    # Create grid
    y = np.arange(height) + 0.5  # in pixels : 0.5, 1.5, ... w-1.5, w-0.5
    y = (y - height / 2) / height  # from -0.5 to 0.5
    x = np.arange(width) + 0.5  # in pixels : 0.5, 1.5, ... h-1.5, h-0.5
    x = (x - width / 2) / height  # from -(w / 2h) to (w / 2h)
    yp, xp = np.meshgrid(y, x, indexing='ij')

    # Sample gaussian mask
    dx = dx * width / height  # re-express as a proportion of height
    gauss = np.exp(-((xp - dx)**2 + (yp - dy)**2) / (2 * sigma**2))

    # Sample sinusoid
    x_rot = xp * np.sin(orientation) + yp * np.cos(orientation)
    sin = np.cos((2 * np.pi / wavelength) * x_rot + phase)

    # Create gabor
    gabor = gauss * sin

    return gabor