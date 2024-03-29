from scipy import ndimage
import numpy as np
import time
import torch
from scipy import signal

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


def compute_imagewise_correlation(images1, images2):
    """ Compute average correlation between two sets of images
    
    Computes correlation per image (i.e., across all pixels) and averages over images.
    
    Arguments:
        images1, images2 (np.array): Array (num_images, height x width) with images.
    
    Returns:
        corr (float): Average correlation across all images.
    """
    num_images = len(images1)
    corrs = compute_correlation(images1.reshape(num_images, -1),
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
    corrs = compute_correlation(images1.reshape(num_images, -1).T,
                                images2.reshape(num_images, -1).T)

    return corrs.reshape(height, width)


def compute_imagewise_psnr(images, recons, eps=1e-8):
    """ Compute average peak-to-signal-noise ratio between images and their reconstructions.
    
    Arguments:
        images (np.array): Array (num_images x height x widht) of images.
        recons (np.array): Array (num_images x height x widht) of reconstructions.
        
    Returns
        psnr (float): Average psnr across all images.
    """
    # Reshape (in case images are sent as 2-d rather than 3-d arrays)
    num_images = len(images)
    images = images.reshape(num_images, -1)
    recons = recons.reshape(num_images, -1)

    # Compute PSNR
    #peak_signal = (recons - recons.min(axis=-1, keepdims=True)).max(axis=-1) ** 2
    peak_signal = (images - images.min(axis=-1, keepdims=True)).max(axis=-1)**2
    noise = ((images - recons)**2).mean(axis=-1)
    psnr_per_image = 10 * np.log10(peak_signal / (noise + eps))
    psnr = psnr_per_image.mean()

    return psnr


def compute_imagewise_ssim(images1, images2):
    """ Compute average structural similarity between images and their reconstructions.
    
     Arguments:
        images1, images2 (np.array): Array (num_images x height x widht) of images.
        
    Returns
        ssim (float): Average structural similarity across all images.
    """
    from skimage.metrics import structural_similarity as ssim

    # Basic checks
    if images1.ndim != 3 or images2.ndim != 3:
        raise ValueError('Images and reconstructions need to be 3-d arrays.')

    # Change range to [0, 1]
    min1 = images1.min(axis=(-1, -2), keepdims=True)
    max1 = images1.max(axis=(-1, -2), keepdims=True)
    images1 = (images1 - min1) / (max1 - min1)
    min2 = images2.min(axis=(-1, -2), keepdims=True)
    max2 = images2.max(axis=(-1, -2), keepdims=True)
    images2 = (images2 - min2) / (max2 - min2)

    # Compute ssim
    ssim_per_image = [
        ssim(im1, im2, data_range=1, gaussian_weights=True, sigma=1.5,
             use_sample_covariance=False) for im1, im2 in zip(images1, images2)]
    ssim = np.mean(ssim_per_image)

    return ssim


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


def bandpass_filter(images, low_freq=0, high_freq=0.1, filt_type='butterworth',
                    butt_order=5, eps=1e-8):
    """ Bandpass filter images to only contain freqs in the desired range.
    
    Arguments:
        images (np.array): Array of images.
        low_freq (float): Lower cutoff for frequency range (should be in [0, 0.5) range).
        high_freq (float): Higher cutoff for frequencies (should be in (0, 0.5] range).
        filt_type (string): 'ideal' or 'butterworth' for the type of filter to use.
        butt_order (int): Order of the butterworth filter (unused for ideal filter).
        eps (float): Small number to avoid division by zero.
    
    Returns:
        filt_images (np.array): Array with the filtered images.
        
    Reference:
        http://faculty.salina.k-state.edu/tim/mVision/freq-domain/freq_filters.html#band-reject-filters
    """
    h, w = images.shape[-2:]

    # Mask images to avoid edge effects
    mask = np.outer(signal.tukey(h, 0.3), signal.tukey(w, 0.3))
    mask = mask / mask.sum()
    masked = (images - images.mean(axis=(-1, -2), keepdims=True)) * mask

    # Create filter
    freqs = np.sqrt(np.fft.fftfreq(h)[:, None]**2 + np.fft.rfftfreq(w)[None, :]**2)
    if filt_type == 'ideal':
        filt = np.zeros_like(freqs)
        filt[np.logical_and(freqs >= low_freq, freqs < high_freq)] = 1
    elif filt_type == 'butterworth':
        if low_freq <= 0:  # low-pass filter
            filt = 1 / (1 + (freqs / high_freq)**(2 * butt_order))
        else:  # band-pass
            d0 = low_freq + (high_freq - low_freq) / 2
            w = high_freq - low_freq
            filt = 1 - (1 / (1 + ((freqs * w) /
                                  (freqs**2 - d0**2 + eps))**(2 * butt_order)))
    else:
        raise NotImplementedError(f'Filter type {filt_type} not recognized')

    # Compute fft, filter and return to spatial domain
    filt_images = [np.fft.irfft2(filt * np.fft.rfft2(im)) for im in masked]

    return np.stack(filt_images)