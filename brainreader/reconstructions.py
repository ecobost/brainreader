""" Decoding models that use an encoding model"""
import datajoint as dj
import torch
import numpy as np

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
    contents = [{'ensemble_dset': 1, 'ensemble_data': 2, 'ensemble_model': 2,
                 'ensemble_training': 1, 'ensemble_params': 1},
                {'ensemble_dset': 5, 'ensemble_data': 2, 'ensemble_model': 12,
                 'ensemble_training': 7, 'ensemble_params': 1},]


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

    # class PerImage(dj.Part):
    #     definition = """ # model responses to one image (ordered by unit_id)

    #     -> master
    #     -> data.ImageSet.Image
    #     ---
    #     model_resps:        blob@brdata
    #     """

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
        # for ic, iid, r in zip(im_classes, im_ids, resps):
        #     self.PerImage.insert1({**key, 'image_class': ic, 'image_id': iid,
        #                           'model_resps': r})


@schema
class AHPValEvaluation(dj.Computed):
    definition = """ # evaluation on validation images

    -> ModelResponses
    -> params.AHPParams
    ---
    val_mse:            float       # validation MSE computed at the original resolution
    val_corr:           float       # validation correlation computed at the original resolution
    """

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

        # Insert
        self.insert1({**key, 'val_mse': val_mse, 'val_corr': val_corr})


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
        image_mask = dataparams.get_image_mask(key['ensemble_dset'], split='test')
        image_classes, image_ids = (data.Scan.Image &
                                    {'dset_id': key['ensemble_dset']}).fetch(
                                        'image_class', 'image_id',
                                        order_by='image_class, image_id')
        image_classes = image_classes[image_mask]
        image_ids = image_ids[image_mask]

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

        # Insert
        self.insert1({
            **key, 'test_mse': mse, 'test_corr': corr, 'test_pixel_mse': pixel_mse,
            'test_pixel_corr': pixel_corr})


################################ GRADIENT BASED #########################################
""" Gradient decoding """

def fill_single_recons():
    """Helper function to fill reconstructions for all validation images and all params"""

    # Fetch all image ids in the validation set

    # Call populate
    # TODO: Maybe receive a restr that cna be send to the populate,
    pass


def fill_single_recons(split='val', restr={}):
    """Helper function to fill reconstructions for all validation images and all params"""

    # Fetch all image ids in the validation set

    # Call populate
    pass

def fill_val_recons():
    pass

def fill_test_recons(params):
    # same but estricting the table to the right params
    #TODO: Maybe I can have a general funciton above that this functions call.
    pass


# Create val_recons, test_recons and use a general fill_recons function.

#option 3:

class GradientOneRecons():
    definition = """ # reconstruct one image 
    
    -> train.Ensemble
    -> params. GradientParams
    -> OneImage
    """
    @property
    def key_source(self, key):
        #TODO: A way to limit this to images of that dataset adn maybe also to validation/test images
        pass

    def fill_val_recons():
        pass

    def fill_test_recons():
        pass

class GradientValReconstructions():
    definition = """ 
    -> train.Ensemble
    -> params.GradientParams
    """
    class OneImage():
        definition = """ # one image in this set
        -> master
        one_image
        """
    def make():
        # Fetch alm image ids, check that all the validation images are populated and create that set.
        #print a warning otherwise.
        pass

class GradientValEvaluation():
    definition = """
    -> GradientValReconstructions
    ---
    ... # normal validation stuff
    ...
    """
    pass

class GradientReconstructions():
    pass

class GradientEvaluation():
    pass



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


#TODO: Repeat the same as above but with VAE on top
class GradientVAEOneRecons():
    pass
class GradientVAEValReconstructions:
    pass
class GradientVAEValEvaluation():
    pass
class GradientVAEReconstructions():
    pass
class GradientVAEEvaluation():
    pass




# TODO: class BlankReconstructions():
#   # maybe add a parameter in dataParams.get_reponses that is blank=False, so it fetches the blank responses
#   # rather than the actual responses (but process them the same).