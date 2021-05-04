""" Decoding models that use an encoding model"""
import datajoint as dj
import torch
import numpy as np
import featurevis
from featurevis import ops
from featurevis import utils as fvutils

from brainreader.encoding import train
from brainreader import params
from brainreader import data
from brainreader import utils

schema = dj.schema('br_reconstructions')
dj.config["enable_python_native_blobs"] = True  # allow blob support in dj 0.12
dj.config['stores'] = {'brdata': {'protocol': 'file', 'location': '/mnt/scratch07/ecobost'}}
dj.config['cache'] = '/tmp'

@schema
class BestEnsemble(dj.Lookup):
    definition = """ # collect the best ensemble models for each dataset
    -> train.Ensemble
    """
    @property
    def contents(self):
        for i in [*range(1, 9), *range(11, 23)]:
            yield {'ensemble_dset': i, 'ensemble_data': 2, 'ensemble_model': 12,
                   'ensemble_training': 7, 'ensemble_params': 1}
        yield {
            'ensemble_dset': 9, 'ensemble_data': 2, 'ensemble_model': 12,
            'ensemble_training': 3, 'ensemble_params': 1}
        yield {
            'ensemble_dset': 10, 'ensemble_data': 2, 'ensemble_model': 12,
            'ensemble_training': 1, 'ensemble_params': 1}


# Set params
selected_dsets = 'ensemble_dset in (5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)'


###############################Averaged high posterior ##################################
""" Gallant decoding. 

Pass a bunch of images through the model and average the ones that produce similar 
activations to the responses we are tryting to match. Based on (Nishimoto et al., 2011)

This has the interpretation of maximizing the posterior max_im p(im|r), which assuming a 
uniform prior over images is equivalent to max_im p(r|im). In the paper they assume a 
gaussian likelihood (u = r^, Sigma=sample_Sigma).
"""
def resize_and_normalize(images, desired_height, desired_width, norm_method):
    """ Resize and normalize images. 
    
    Arguments:
        images (np.array): Images (num_images x h x w).
        desired_height (int): Desired height of images.
        desired_width (int): Desired width of images.
        norm_method(str): Type of normalization (from data.ImageNormalization).
    
    Returns:
        An array (num_images x desired_height x desired_widht) with the images.
    """
    images = utils.resize(images, desired_height, desired_width)

    if norm_method == 'zscore-train':
        images = (images - images.mean()) / images.std(axis=(-1, -2)).mean()
    else:
        raise NotImplementedError(f'Unknown image normalization {norm_method}')

    return images


@schema
class ModelResponses(dj.Computed):
    definition = """ # model responses to all images in an ImageSet
    
    -> train.Ensemble
    -> data.ImageSet
    ---
    responses:      blob@brdata     # model responses (num_images x num_units); rows ordered as in ImageSet, cols ordered as returned by the model
    """
    @property
    def key_source(self):
        return train.Ensemble * data.ImageSet & BestEnsemble

    def make(self, key):
        # Fetch all images
        utils.log('Fetching images')
        images = (data.ImageSet & key).get_images().astype(np.float32)  # this is ~16 GB

        # Process images (so they agree with input to model)
        dataparams = params.DataParams & {'data_params': key['ensemble_data']}
        h, w, img_normalization = dataparams.fetch1('image_height', 'image_width',
                                                    'img_normalization')
        images = resize_and_normalize(images, h, w, img_normalization)

        # Fetch (ensemble) model
        utils.log('Instantiating model')
        model = (train.Ensemble & key).get_model()
        model.eval()
        model.cuda()

        # Iterate over images and get responses
        utils.log('Getting responses')
        resps = []
        with torch.no_grad():
            batch_size = 512
            for i in range(0, len(images), batch_size):
                ims = torch.as_tensor(images[i:i + batch_size, None], dtype=torch.float32,
                                      device='cuda')
                resp = model(ims)
                resps.append(resp.detach().cpu().numpy())
        resps = np.concatenate(resps, axis=0)

        # Insert
        utils.log('Inserting')
        self.insert1({**key, 'responses': resps})


@schema
class AHPValEvaluation(dj.Computed):
    definition = """ # evaluation on validation images

    -> ModelResponses
    -> params.AHPParams
    ---
    val_mse:            float       # average validation MSE
    val_corr:           float       # average validation correlation
    val_psnr:           float       # average validation peak_signal-to-noise ratio
    val_ssim:           float       # average validation structural similarity
    """

    @property
    def key_source(self):
        # retrict to using simple averaging of images (results on par with weighted average)
        return ModelResponses * params.AHPParams & {'weight_samples': False}

    def make(self, key):
        # Get data to reconstruct
        utils.log('Fetch data to reconstruct')
        dataparams = params.DataParams & {'data_params': key['ensemble_data']}
        images = dataparams.get_images(key['ensemble_dset'], split='val')
        responses = dataparams.get_responses(key['ensemble_dset'], split='val')

        # Get images and responses passed through the model
        utils.log('Fetch images and model responses')
        h, w, img_normalization = dataparams.fetch1('image_height', 'image_width',
                                                    'img_normalization')
        all_images = (data.ImageSet & key).get_images().astype(np.float32)
        all_images = resize_and_normalize(all_images, h, w, img_normalization)
        all_responses = (ModelResponses & key).fetch1('responses')

        # Compute similarity
        utils.log('Compute similarity')
        similarity = (params.AHPParams & key).fetch1('similarity')
        if similarity == 'correlation':
            residuals = responses - responses.mean(axis=-1, keepdims=True)
            all_residuals = all_responses - all_responses.mean(axis=-1, keepdims=True)
            ssr = (residuals**2).sum(axis=-1)
            all_ssr = (all_residuals**2).sum(axis=-1)
            similarity_matrix = (np.matmul(residuals, all_residuals.T) /
                                 np.sqrt(ssr[:, None] * all_ssr[None, :]))
        elif similarity == 'poisson_loglik':
            similarity_matrix = (np.matmul(responses, np.log(all_responses + 1e-8).T) -
                                 all_responses.sum(axis=-1)) # average log likelihood
        else:
            raise NotImplementedError(f'Similarity {similarity} not implemented')

        # Find sample images that will be used for reconstruction
        utils.log('Create reconstructions')
        num_samples = (params.AHPParams & key).fetch1('num_samples')
        best_indices = np.argsort(similarity_matrix, axis=-1)[:, -num_samples:]

        # Create weights
        weight_samples = bool((params.AHPParams & key).fetch1('weight_samples'))
        if weight_samples:
            best_weights = np.stack([
                s[idx] for s, idx in zip(similarity_matrix, best_indices)])
            norm_weights = best_weights / best_weights.sum(axis=-1, keepdims=True)
        else:
            norm_weights = np.full(best_indices.shape, 1 / best_indices.shape[-1])

        # Create recons
        recons = np.stack([(all_images[idx] * ws[:, None, None]).sum(axis=0)
                           for idx, ws in zip(best_indices, norm_weights)])

        # Compute metrics
        val_mse = ((images - recons)**2).mean()
        val_corr = utils.compute_imagewise_correlation(images, recons)
        val_psnr = utils.compute_imagewise_psnr(images, recons)
        val_ssim = utils.compute_imagewise_ssim(images, recons)

        # Insert
        self.insert1({
            **key, 'val_mse': val_mse, 'val_corr': val_corr, 'val_psnr': val_psnr,
            'val_ssim': val_ssim})


@schema
class AHPValBestModel(dj.Computed):
    definition = """ # best model for each scan using val_ssim as metric
    
    -> ModelResponses
    ---
    -> AHPValEvaluation
    """

    @property
    def key_source(self):
        return ModelResponses & AHPValEvaluation

    def make(self, key):
        keys, ssims = (AHPValEvaluation & key).fetch('KEY', 'val_ssim')
        best_model = keys[np.argmax(ssims)]
        self.insert1(best_model)


@schema
class AHPReconstructions(dj.Computed):
    definition = """ # reconstructions for test set images (activity averaged across repeats)

    -> ModelResponses
    -> params.AHPParams
    """

    class Reconstruction(dj.Part):
        definition = """ # reconstruction for a single image
        
        -> master
        -> data.Scan.Image.proj(ensemble_dset='dset_id')
        ---
        recons:             longblob
        """

    class ReconstructionParts(dj.Part):
        definition = """ # all the sample images used to create a single image reconstruction
        
        -> master
        -> data.Scan.Image.proj(ensemble_dset='dset_id')    # reconstructed image
        -> data.ImageSet.Image.proj(sample_class='image_class', sample_id='image_id')  # sample image
        ---
        weight:         float       # weight 
        """

    @property
    def key_source(self):
        # retrict to using simple averaging of images (results on par with weighted average)
        return ModelResponses * params.AHPParams & {'weight_samples': False}

    def make(self, key):
        # Get data to reconstruct
        utils.log('Fetching data to reconstruct')
        dataparams = params.DataParams & {'data_params': key['ensemble_data']}
        responses = dataparams.get_responses(key['ensemble_dset'], split='test')

        # Get images and responses passed through the model
        utils.log('Fetch images and model responses')
        h, w, img_normalization = dataparams.fetch1('image_height', 'image_width',
                                                    'img_normalization')
        all_images = (data.ImageSet & key).get_images().astype(np.float32)
        all_images = resize_and_normalize(all_images, h, w, img_normalization)
        all_responses = (ModelResponses & key).fetch1('responses')
        all_classes, all_ids = (data.ImageSet.Image & key).fetch(
            'image_class', 'image_id', order_by='image_class, image_id')

        # Compute similarity
        utils.log('Computing similarity')
        similarity = (params.AHPParams & key).fetch1('similarity')
        if similarity == 'correlation':
            residuals = responses - responses.mean(axis=-1, keepdims=True)
            all_residuals = all_responses - all_responses.mean(axis=-1, keepdims=True)
            ssr = (residuals**2).sum(axis=-1)
            all_ssr = (all_residuals**2).sum(axis=-1)
            similarity_matrix = (np.matmul(residuals, all_residuals.T) /
                                 np.sqrt(ssr[:, None] * all_ssr[None, :]))
        elif similarity == 'poisson_loglik':
            similarity_matrix = (np.matmul(responses, np.log(all_responses + 1e-8).T) -
                                 all_responses.sum(axis=-1))  # average log likelihood
        else:
            raise NotImplementedError(f'Similarity {similarity} not implemented')

        # Find sample images that will be used for reconstruction
        utils.log('Create reconstructions')
        num_samples = (params.AHPParams & key).fetch1('num_samples')
        best_indices = np.argsort(similarity_matrix, axis=-1)[:, -num_samples:]
        best_classes = all_classes[best_indices]
        best_ids = all_ids[best_indices]

        # Create weights
        weight_samples = bool((params.AHPParams & key).fetch1('weight_samples'))
        best_weights = np.stack([
            s[idx] for s, idx in zip(similarity_matrix, best_indices)])
        if weight_samples:
            norm_weights = best_weights / best_weights.sum(axis=-1, keepdims=True)
        else:
            norm_weights = np.full_like(best_weights, 1 / best_weights.shape[-1])

        # Create recons
        recons = np.stack([(all_images[idx] * ws[:, None, None]).sum(axis=0)
                           for idx, ws in zip(best_indices, norm_weights)])

        # Find out image ids of test set images
        split_rel = (data.Split.PerImage & dataparams &
                     {'dset_id': key['ensemble_dset'], 'split': 'test'})
        image_classes, image_ids = split_rel.fetch('image_class', 'image_id',
                                                   order_by='image_class, image_id')

        # Insert
        utils.log('Inserting reconstructions')
        self.insert1(key)
        self.Reconstruction.insert(
            [{**key, 'image_class': ic, 'image_id': id_, 'recons': r}
             for ic, id_, r in zip(image_classes, image_ids, recons)])
        for ic, iid, best_class, best_id, best_weight in zip(image_classes, image_ids,
                                                             best_classes, best_ids,
                                                             best_weights):
            self.ReconstructionParts.insert([{
                **key, 'image_class': ic, 'image_id': iid, 'sample_class': sic,
                'sample_id': sid,
                'weight': w} for sic, sid, w in zip(best_class, best_id, best_weight)])


@schema
class AHPEvaluation(dj.Computed):
    definition = """ # evaluate ahp reconstruction in natural images in test set

    -> AHPReconstructions
    ---
    test_mse:       float       # average MSE across all image
    test_corr:      float       # average correlation (computed per image and averaged across images)
    test_psnr:      float       # average peak_signal-to-noise ratio across all images
    test_ssim:      float       # average SSIM across all images
    test_pixel_mse: longblob    # pixel-wise MSE (computed per pixel, averaged across images)
    test_pixel_corr:longblob    # pixel-wise correlation (computed per pixel across images)
    """

    def make(self, key):
        # Get original images
        images = (params.DataParams & {'data_params': key['ensemble_data']}).get_images(
            key['ensemble_dset'], split='test')

        # Get reconstructions
        recons, image_classes = (AHPReconstructions.Reconstruction & key).fetch(
            'recons', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE
        mse = ((images - recons)**2).mean()
        pixel_mse = ((images - recons)**2).mean(axis=0)

        # Compute corrs
        corr = utils.compute_imagewise_correlation(images, recons)
        pixel_corr = utils.compute_pixelwise_correlation(images, recons)

        # Compute PSNR and SSIM
        psnr = utils.compute_imagewise_psnr(images, recons)
        ssim = utils.compute_imagewise_ssim(images, recons)

        # Insert
        self.insert1({
            **key, 'test_mse': mse, 'test_corr': corr, 'test_psnr': psnr,
            'test_ssim': ssim, 'test_pixel_mse': pixel_mse,
            'test_pixel_corr': pixel_corr})


@schema
class AHPTestBestModel(dj.Computed):
    definition = """ # best model for each scan using test_ssim as metric
    
    -> ModelResponses
    ---
    -> AHPEvaluation
    """

    @property
    def key_source(self):
        return ModelResponses & AHPEvaluation

    def make(self, key):
        keys, ssims = (AHPEvaluation & key).fetch('KEY', 'test_ssim')
        best_model = keys[np.argmax(ssims)]
        self.insert1(best_model)


@schema
class AHPBestModelByOthers(dj.Computed):
    definition = """ # pick the best parameters out of all best test models (that are not from the same animal)
    
    -> ModelResponses
    ---
    -> AHPEvaluation
    """

    @property
    def key_source(self):
        return ModelResponses & AHPEvaluation & selected_dsets

    def make(self, key):
        from scipy import stats

        # Find best params
        animal_id = (data.Scan & {'dset_id': key['ensemble_dset']}).fetch1('animal_id')
        all_params = (AHPTestBestModel & selected_dsets &
                      (data.Scan.proj('animal_id', ensemble_dset='dset_id') -
                       {'animal_id': animal_id})).fetch('ahp_params')
        best_params = stats.mode(all_params)[0][0]
        self.insert1({**key, 'ahp_params': best_params})


################################ GRADIENT BASED #########################################
""" Gradient decoding."""

#TODO: May not need this class, though I may need it for the OneReconstruction tables below but I only need to populate those tables for test set which I can add as a key_source restriction
class Fillable:
    """ Small class that adds the ability to populate reconstruction classes restricting 
    to only certain split."""
    @classmethod
    def fill_recons(cls, restr, split='val'):
        """ Fill reconstruction images.
        
        Arguments:
            restr (dj restr): Restriction that will be passed to the populate. It will 
                usually restrict to some dataset or GradientParams.
            split (str): Which split to populate ('train', 'val', 'test').
        """
        for key in BestEnsemble.proj():
            dataparams = params.DataParams & {'data_params': key['ensemble_data']}
            im_restr = (data.Split.PerImage & dataparams &
                        {'dset_id': key['ensemble_dset'], 'split': split})
            cls.populate(key, restr, im_restr, reserve_jobs=True)


@schema
class GradientOneReconstruction(dj.Computed, Fillable):
    definition = """ # create gradient reconstructions
    
    -> train.Ensemble
    -> params.GradientParams
    -> data.Scan.Image.proj(ensemble_dset='dset_id')
    ---
    reconstruction:     blob@brdata     # reconstruction for the response to one image
    sim_value:          float           # final similarity measure between target and recorded responses
    """
    @property
    def key_source(self):
        all_keys = (train.Ensemble * params.GradientParams * data.Scan.Image.proj(ensemble_dset='dset_id'))
        return all_keys & BestEnsemble
        #TODO: Could I limit this to validation/test images here

    def make(self, key):
        # Get recorded responses to this image
        utils.log('Get target response')
        dataparams = params.DataParams & {'data_params': key['ensemble_data']}
        responses = dataparams.get_responses(key['ensemble_dset'], split=None)  # all images
        im_classes, im_ids = (data.Scan.Image & {'dset_id': key['ensemble_dset']}).fetch(
            'image_class', 'image_id', order_by='image_class, image_id')
        im_mask = np.logical_and(im_classes == key['image_class'],
                                 im_ids == key['image_id'])
        resp = responses[im_mask] # 1 x num_cells

        # Get model
        utils.log('Instantiating model')
        model = (train.Ensemble & key).get_model()
        model.eval()
        model.cuda()

        # Get some params
        utils.log('Set up optimization')
        h, w = dataparams.fetch1('image_height', 'image_width')
        opt_params = (params.GradientParams & key).fetch1()
        if dataparams.fetch1('img_normalization') != 'zscore-train':
            raise NotImplementedError('Cannot handle images that are not zscored.')

        # Set up initial image
        torch.manual_seed(opt_params['seed'])
        initial_image = torch.randn(1, 1, h, w, device='cuda')

        # Set up optimization function
        neural_resp = torch.as_tensor(resp, dtype=torch.float32, device='cuda')
        if opt_params['similarity'] == 'negmse':
            similarity = fvutils.Compose([ops.MSE(neural_resp), ops.MultiplyBy(-1)])
        elif opt_params['similarity'] == 'cosine':
            similarity = ops.CosineSimilarity(neural_resp)
        elif opt_params['similarity'] == 'poisson_loglik':
            similarity = ops.PoissonLogLikelihood(neural_resp)
        else:
            msg = 'Similarity metric {similarity} not implemented'.format(**opt_params)
            raise NotImplementedError(msg)
        obj_function = fvutils.Compose([model, similarity])

        # Optimize
        utils.log('Optimize')
        #TODO: Drop some stuff from here if I won't use it.
        jitter = ops.Jitter(opt_params['jitter']) if opt_params['jitter'] != 0 else None
        blur = (ops.GaussianBlur(opt_params['gradient_sigma'])
                if opt_params['gradient_sigma'] != 0 else None)
        l2_reg = (ops.LpNorm(p=2, weight=opt_params['l2_weight'])
                  if opt_params['l2_weight'] != 0 else None)
        fix_std = ops.ChangeStd(1) if opt_params['keep_std_fixed'] else None
        recon, fevals, _ = featurevis.gradient_ascent(
            obj_function,
            initial_image,
            step_size=opt_params['step_size'],
            num_iterations=opt_params['num_iterations'],
            transform=jitter,
            gradient_f=blur,
            regularization=l2_reg,
            post_update=fix_std,
        )
        recon = recon.mean(0).squeeze().cpu().numpy()
        final_f = fevals[-1]

        # Check for divergence
        if np.isnan(final_f) or np.isinf(final_f):
            raise ValueError('Objective function diverged!')

        # Insert
        self.insert1({**key, 'reconstruction': recon, 'sim_value': final_f})


@schema
class GradientValEvaluation(dj.Computed):
    definition = """ # evaluate gradient reconstruction on the validation set
    
    -> train.Ensemble
    -> params.GradientParams
    ---
    val_mse:            float       # average validation MSE
    val_corr:           float       # average validation correlation
    val_psnr:           float       # average validation peak_signal-to-noise ratio
    val_ssim:           float       # average validation structural similarity
    """

    @property
    def key_source(self):
        return train.Ensemble * params.GradientParams & GradientOneReconstruction

    def make(self, key):
        # Get original images
        dataparams = (params.DataParams & {'data_params': key['ensemble_data']})
        images = dataparams.get_images(key['ensemble_dset'], split='val')

        # Get recons
        im_restr = (data.Split.PerImage & dataparams &
                    {'dset_id': key['ensemble_dset'], 'split': 'val'})
        recons = (GradientOneReconstruction & key & im_restr).fetch(
            'reconstruction', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Check all validation images have been reconstructed
        if len(images) != len(recons):
            raise ValueError('Some validation images may not be reconstructed.')

        # Compute metrics
        val_mse = ((images - recons)**2).mean()
        val_corr = utils.compute_imagewise_correlation(images, recons)
        val_psnr = utils.compute_imagewise_psnr(images, recons)
        val_ssim = utils.compute_imagewise_ssim(images, recons)

        # Insert
        self.insert1({
            **key, 'val_mse': val_mse, 'val_corr': val_corr, 'val_psnr': val_psnr,
            'val_ssim': val_ssim})

@schema
class GradientEvaluation(dj.Computed):
    definition = """ # evaluate gradient reconstruction in natural images in test set

    -> train.Ensemble
    -> params.GradientParams
    ---
    test_mse:       float       # average MSE across all image
    test_corr:      float       # average correlation (computed per image and averaged across images)
    test_psnr:      float       # average peak_signal-to-noise ratio across all images
    test_ssim:      float       # average SSIM across all images
    test_pixel_mse: longblob    # pixel-wise MSE (computed per pixel, averaged across images)
    test_pixel_corr:longblob    # pixel-wise correlation (computed per pixel across images)
    """

    @property
    def key_source(self):
        return train.Ensemble * params.GradientParams & GradientOneReconstruction

    def make(self, key):
        # Get original images
        dataparams = (params.DataParams & {'data_params': key['ensemble_data']})
        images = dataparams.get_images(key['ensemble_dset'], split='test')

        # Get recons
        im_restr = (data.Split.PerImage & dataparams &
                    {'dset_id': key['ensemble_dset'], 'split': 'test'})
        recons, image_classes = (GradientOneReconstruction & key & im_restr).fetch(
            'reconstruction', 'image_class', order_by='image_class, image_id')
        recons = np.stack(recons)

        # Restrict to natural images
        images = images[image_classes == 'imagenet']
        recons = recons[image_classes == 'imagenet']

        # Compute MSE
        mse = ((images - recons)**2).mean()
        pixel_mse = ((images - recons)**2).mean(axis=0)

        # Compute corrs
        corr = utils.compute_imagewise_correlation(images, recons)
        pixel_corr = utils.compute_pixelwise_correlation(images, recons)

        # Compute PSNR and SSIM
        psnr = utils.compute_imagewise_psnr(images, recons)
        ssim = utils.compute_imagewise_ssim(images, recons)

        # Insert
        self.insert1({
            **key, 'test_mse': mse, 'test_corr': corr, 'test_psnr': psnr,
            'test_ssim': ssim, 'test_pixel_mse': pixel_mse,
            'test_pixel_corr': pixel_corr})



#TODO: Single trial
class GradientSingleTrialOneRecons():
    definition = """
    -> train.Ensemble
    -> GradientParams
    -> image_id
    """
    class OneTrial():
        definiiton = """
        -> master
        trial_idx
        ---
        one_recon
        """
    #TODO: Limit to only the best nmodel
    # use dataparams with repeats False
    pass
class GradientSingleTrialRecons():
    pass
class GradientSingleTrialEval():
    pass

# TODO: Model responses

class GradientModelRecons():
    pass
#     # reconstruct based on model responses (this is just to get a upper bound of how good the reconstruction can be)
#     # use model responser as a normalization to report result MSE(neural) / MSE(model) (will gie me a 0-1 range)
class GradientModelEval():
    pass

##################################### GENERATOR #########################################
""" Reconstruct as in Gradient params but with a generator in the front. """
from brainreader import generators
from torch.nn import functional as F


class MNIST2Image:
    """ Transforms an MNIST image into a normalized image as used for training.
    
    Takes a MNIST image (28 x 28) in [0, 1] range and transforms it into a normalized 
    144 x 256 image (as were used for model training).
    
    Arguments:
        train_mean (float): Mean of training images (used to normalize this image).
        train_std (float): Std of training images (used to normalize this image). 
    """
    def __init__(self, train_mean, train_std):
        self.train_mean = train_mean
        self.train_std = train_std

    @fvutils.varargin
    def __call__(self, x):
        # Rotate, upsample and pad
        rotated = torch.rot90(x, dims=(-2, -1))
        resized = F.interpolate(rotated, 144, mode='bilinear', align_corners=False)
        padded = F.pad(resized, [(256 - 144) // 2, (256 - 144) // 2, 0, 0], value=-1)

        # Normalize
        image = padded * 255
        norm = (image - self.train_mean) / self.train_std

        return norm


@schema
class GeneratorMNISTReconstruction(dj.Computed):
    definition = """ # create gradient reconstruction with a MNIST generator
    
    -> train.Ensemble
    -> params.GeneratorMNISTParams
    -> data.Scan.Image.proj(ensemble_dset='dset_id')
    ---
    recons_z:       longblob        # final hidden vector obtained
    recons_digit:   longblob        # final MNIST digit obtained
    recons_image:   longblob        # final image reconstruction
    sim_value:      float           # final similarity measure between target and recorded responses
    """

    @property
    def key_source(self):
        all_keys = (train.Ensemble * params.GeneratorMNISTParams *
                    data.Scan.Image.proj(ensemble_dset='dset_id'))
        return all_keys & BestEnsemble & {'image_class': 'mnist', 'mnistgen_params': 2}

    def make(self, key):
        import featurevis
        from featurevis import ops
        from featurevis import utils as fvutils

        # Get recorded responses to this image
        utils.log('Get target response')
        dataparams = params.DataParams & {'data_params': key['ensemble_data']}
        responses = dataparams.get_responses(key['ensemble_dset'],
                                             split=None)  # all images
        im_classes, im_ids = (data.Scan.Image & {'dset_id': key['ensemble_dset']}).fetch(
            'image_class', 'image_id', order_by='image_class, image_id')
        im_mask = np.logical_and(im_classes == key['image_class'],
                                 im_ids == key['image_id'])
        resp = responses[im_mask]  # 1 x num_cells

        # Get train stats (needed for normalization below)
        train_images = (
            data.Image &
            (data.Split.PerImage & {'dset_id': key['ensemble_dset'], 'split': 'train'} &
             dataparams)).fetch('image')
        train_images = np.stack(train_images).astype(np.float32)
        train_mean = train_images.mean()
        train_std = train_images.std(axis=(-1, -2)).mean()

        # Get encoding model
        utils.log('Instantiating model')
        model = (train.Ensemble & key).get_model()
        model.eval()
        model.cuda()

        # Get some params
        opt_params = (params.GeneratorMNISTParams & key).fetch1()

        # Get generator
        generator = (generators.get_mnist_gan()
                     if opt_params['generator'] == 'gan' else generators.get_mnist_vae())
        generator.eval()
        generator.cuda()

        # Set up initial hidden vector
        torch.manual_seed(opt_params['seed'])
        z_shape = (1, 100, 1, 1) if opt_params['generator'] == 'gan' else (1, 20)
        initial_z = torch.randn(*z_shape, dtype=torch.float32, device='cuda')

        # Set up optimization function
        utils.log('Set up optimization')
        neural_resp = torch.as_tensor(resp[None, :], dtype=torch.float32, device='cuda')
        similarity = ops.PoissonLogLikelihood(neural_resp)
        mnist2image = MNIST2Image(train_mean, train_std)
        obj_function = fvutils.Compose([generator, mnist2image, model, similarity])

        # Optimize
        utils.log('Optimize')
        z, fevals, _ = featurevis.gradient_ascent(
            obj_function,
            initial_z,
            step_size=opt_params['step_size'],
            num_iterations=opt_params['num_iterations'],
        )
        with torch.no_grad():
            digit = generator(z)
            recon = mnist2image(digit)
        z = z.cpu().numpy()
        digit = digit.mean(0).squeeze().cpu().numpy()
        recon = recon.mean(0).squeeze().cpu().numpy()
        final_f = fevals[-1]

        # Check for divergence
        if np.isnan(final_f) or np.isinf(final_f):
            raise ValueError('Objective function diverged!')

        # Insert
        self.insert1({
            **key, 'recons_z': z, 'recons_digit': digit, 'recons_image': recon,
            'sim_value': final_f})




# TODO: class BlankReconstructions():
#   # maybe add a parameter in dataParams.get_reponses that is blank=False, so it fetches the blank responses
#   # rather than the actual responses (but process them the same).