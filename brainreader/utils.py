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
        order (int): Order of the interpolation used by ndimage.zoom (see docs there).desired_width

    Returns:
        images (np.array). Array with the resized images (num_images x desired_height x
            desired_width).
    """
    zoom = (desired_height / images.shape[-2], desired_width / images.shape[-1])
    if zoom != (1, 1): # ndimage does not check for this
        images = np.stack([ndimage.zoom(im, zoom, order=order) for im in images])
    return images


def log(*messages):
    """ Prints a message (with a timestamp next to it).
    
    Arguments:
        message (string): Arguments to be print()'d.
    """
    formatted_time = '[{}]'.format(time.ctime())
    print(formatted_time, *messages, flush=True)


def compute_correlation(x, y, eps=1e-9):
    """ Compute correlation between two sets of neural responses.
    
    Computes correlation per cell (i.e., across images) and returns the average across 
    cells.
    
    Arguments:
        x, y (torch.tensor): A (num_images, num_cells) tensor.
        eps (float): Small value to avoid division by zero.
    
    Returns:
        corr (float): Average correlation coefficient.
    """
    # Compute correlation per cell
    x_res = x - x.mean(dim=0)
    y_res = y - y.mean(dim=0)
    corrs = (x_res * y_res).sum(dim=0) / (torch.sqrt(
        (x_res**2).sum(dim=0) * (y_res**2).sum(dim=0)) + eps)

    # Check that all of corrs are valid
    bad_cells = (corrs < -1) | (corrs > 1) | torch.isnan(corrs) | torch.isinf(corrs)
    if torch.any(bad_cells):
        print('Warning: Unstable correlation (setting to -1).')
        corrs[bad_cells] = -1

    return corrs.mean()


def bivariate_gaussian(xy, xy_mean, xy_std, corr_xy, normalize=False):
    """ Compute the pdf of a bivariate gaussian distribution.
    
    Arguments:
        xy (torch.tensor): Points. A (M, 2) tensor.
        xy_mean (torch.tensor): Mean of the distributions (in xy). Shape: (N, 2). N is the
            number of different gaussian distributions this points will be evaluated in.
        xy_std (torch tensor): Standard deviation in x and y. Shape: (N, 2).
        corr_xy (torch.tensor): Correlation between x and y. Shape: (N).
        normalize (boolean): Normalize the pdf values so it is a valid pdf (sums to 1).
    
    Returns:
        pdfs (torch.tensor): A tensor (N, M) with the pdf values at each position.
    """
    # Make all inputs broadcastable (N, M, 2)
    xy = xy.unsqueeze(0)
    xy_mean = xy_mean.unsqueeze(1)
    xy_std = xy_std.unsqueeze(1)
    corr_xy = corr_xy.unsqueeze(-1)  # used after last dimension is reduced (so [N, 1])

    # Compute pdf
    residuals = (xy - xy_mean) / xy_std
    numer = 2 * corr_xy * torch.prod(residuals, dim=-1) - torch.sum(residuals**2, dim=-1)
    pdf = torch.exp(numer / (2 * (1- corr_xy**2)))

    # normalize pdf if needed
    if normalize:
        import math
        divisor = 2 * math.pi * torch.prod(xy_std, dim=-1) * torch.sqrt(1 - corr_xy**2)
        pdf = pdf / divisor

    return pdf