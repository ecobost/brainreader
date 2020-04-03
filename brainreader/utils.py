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
