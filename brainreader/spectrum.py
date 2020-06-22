""" Band pass filter image and recons at diff frequency bins and compute metrics. """
import datajoint as dj
import numpy as np

from brainreader import params
from brainreader import decoding
from brainreader import utils
from brainreader import reconstructions

schema = dj.schema('br_spectrum')


@schema
class LinearEvaluation(dj.Computed):
    definition = """ # evaluate linear model reconstructions in natural images
    
    -> decoding.LinearReconstructions
    -> params.SpectrumParams
    ---
    mses:       longblob    # average MSE across all image at each frequency bin
    corrs:      longblob    # average correlation (computed per image and averaged across images) at each frequency bin
    mse_avg:    float       # average mse across all frequency bins
    corr_avg:   float       # average corr across all frequency bins
    """
    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (decoding.LinearReconstructions.Reconstruction & key).fetch(
            'recons', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Get params
        bin_size, interval = (params.SpectrumParams & key).fetch1('bin_size', 'interval')
        low_freqs = np.arange(0, 0.5 - bin_size + 1e-9, interval)
        high_freqs = np.arange(bin_size, 0.5 + 1e-9, interval)

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for low_f, high_f in zip(low_freqs, high_freqs):
            # Filter images
            filt_images = utils.bandpass_filter(images, low_f, high_f)
            filt_recons = utils.bandpass_filter(recons, low_f, high_f)

            # Compute metrics
            mse = ((filt_images - filt_recons) ** 2).mean()
            corr = utils.compute_imagewise_correlation(filt_images, filt_recons)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({**key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
                      'corr_avg': corrs.mean()})


@schema
class MLPEvaluation(dj.Computed):
    definition = """ # evaluate mlp model reconstructions in natural images
    
    -> decoding.MLPReconstructions
    -> params.SpectrumParams
    ---
    mses:       longblob    # average MSE across all image at each frequency bin
    corrs:      longblob    # average correlation (computed per image and averaged across images) at each frequency bin
    mse_avg:    float       # average mse across all frequency bins
    corr_avg:   float       # average corr across all frequency bins
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (decoding.MLPReconstructions.Reconstruction &
                                 key).fetch('recons', 'image_class',
                                            order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Get params
        bin_size, interval = (params.SpectrumParams & key).fetch1('bin_size', 'interval')
        low_freqs = np.arange(0, 0.5 - bin_size + 1e-9, interval)
        high_freqs = np.arange(bin_size, 0.5 + 1e-9, interval)

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for low_f, high_f in zip(low_freqs, high_freqs):
            # Filter images
            filt_images = utils.bandpass_filter(images, low_f, high_f)
            filt_recons = utils.bandpass_filter(recons, low_f, high_f)

            # Compute metrics
            mse = ((filt_images - filt_recons)**2).mean()
            corr = utils.compute_imagewise_correlation(filt_images, filt_recons)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})


@schema
class DeconvEvaluation(dj.Computed):
    definition = """ # evaluate deconv model reconstructions in natural images
    
    -> decoding.DeconvReconstructions
    -> params.SpectrumParams
    ---
    mses:       longblob    # average MSE across all image at each frequency bin
    corrs:      longblob    # average correlation (computed per image and averaged across images) at each frequency bin
    mse_avg:    float       # average mse across all frequency bins
    corr_avg:   float       # average corr across all frequency bins
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (decoding.DeconvReconstructions.Reconstruction & key).fetch(
            'recons', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Get params
        bin_size, interval = (params.SpectrumParams & key).fetch1('bin_size', 'interval')
        low_freqs = np.arange(0, 0.5 - bin_size + 1e-9, interval)
        high_freqs = np.arange(bin_size, 0.5 + 1e-9, interval)

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for low_f, high_f in zip(low_freqs, high_freqs):
            # Filter images
            filt_images = utils.bandpass_filter(images, low_f, high_f)
            filt_recons = utils.bandpass_filter(recons, low_f, high_f)

            # Compute metrics
            mse = ((filt_images - filt_recons)**2).mean()
            corr = utils.compute_imagewise_correlation(filt_images, filt_recons)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})


@schema
class GaborEvaluation(dj.Computed):
    definition = """ # evaluate gaborv model reconstructions in natural images
    
    -> decoding.GaborReconstructions
    -> params.SpectrumParams
    ---
    mses:       longblob    # average MSE across all image at each frequency bin
    corrs:      longblob    # average correlation (computed per image and averaged across images) at each frequency bin
    mse_avg:    float       # average mse across all frequency bins
    corr_avg:   float       # average corr across all frequency bins
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & key).get_images(key['dset_id'], split='test')

        # Get reconstructions
        recons, image_classes = (decoding.GaborReconstructions.Reconstruction &
                                 key).fetch('recons', 'image_class',
                                            order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Get params
        bin_size, interval = (params.SpectrumParams & key).fetch1('bin_size', 'interval')
        low_freqs = np.arange(0, 0.5 - bin_size + 1e-9, interval)
        high_freqs = np.arange(bin_size, 0.5 + 1e-9, interval)

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for low_f, high_f in zip(low_freqs, high_freqs):
            # Filter images
            filt_images = utils.bandpass_filter(images, low_f, high_f)
            filt_recons = utils.bandpass_filter(recons, low_f, high_f)

            # Compute metrics
            mse = ((filt_images - filt_recons)**2).mean()
            corr = utils.compute_imagewise_correlation(filt_images, filt_recons)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})


@schema
class AHPEvaluation(dj.Computed):
    definition = """ # evaluate gaborv model reconstructions in natural images
    
    -> reconstructions.AHPReconstructions
    -> params.SpectrumParams
    ---
    mses:       longblob    # average MSE across all image at each frequency bin
    corrs:      longblob    # average correlation (computed per image and averaged across images) at each frequency bin
    mse_avg:    float       # average mse across all frequency bins
    corr_avg:   float       # average corr across all frequency bins
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & {'data_params': key['ensemble_data']}).get_images(
            key['ensemble_dset'], split='test')

        # Get reconstructions
        recons, image_classes = (reconstructions.AHPReconstructions.Reconstruction & key).fetch(
            'recons', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Get params
        bin_size, interval = (params.SpectrumParams & key).fetch1('bin_size', 'interval')
        low_freqs = np.arange(0, 0.5 - bin_size + 1e-9, interval)
        high_freqs = np.arange(bin_size, 0.5 + 1e-9, interval)

        # Compute MSE and correlations across diff frequencies
        mses = []
        corrs = []
        for low_f, high_f in zip(low_freqs, high_freqs):
            # Filter images
            filt_images = utils.bandpass_filter(images, low_f, high_f)
            filt_recons = utils.bandpass_filter(recons, low_f, high_f)

            # Compute metrics
            mse = ((filt_images - filt_recons)**2).mean()
            corr = utils.compute_imagewise_correlation(filt_images, filt_recons)
            mses.append(mse)
            corrs.append(corr)
        mses = np.stack(mses)
        corrs = np.stack(corrs)

        # Insert
        self.insert1({
            **key, 'mses': mses, 'corrs': corrs, 'mse_avg': mses.mean(),
            'corr_avg': corrs.mean()})