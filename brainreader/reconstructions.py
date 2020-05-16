""" Decoding models that use an encoding model"""
from brainreader import train
from brainreader import data

# @schema
# class BestEnsemble(dj.Lookup):
#     definition = """ # collect the best ensemble models for each dataset
#     -> train.Ensemble
#     """
#     contents = [{'ensemble_dset': 1, 'ensemble_data': 1, 'ensemble_model': 2,
#                  'ensemble_training': 9, 'ensemble_params': 1}]

#########################################################################################
""" Gallant decoding. Pass a bunch of images through the model and average the ones that
produce similar activations to the responses we are tryting to match."""

# class ModelResponses(dj.Computed):
#     definition = """ # model responses to all images.
#     -> data.AllImages
#     -> train.Ensemble
#     """

#     class PerImage(dj.Part):
#         definition = """ # model responses to one image (ordered by unit_id)
#         -> master
#         -> data.AllImages.Image
#         ---
#         model_resps:        blob@brdata
#         """
#     @property
#     def key_source(self, key):
#         return data.AllImages * train.Ensemble & BestEnsemble

#     def make(self, key):
#         # Fetch all images
#         # I have to change them so they agree with the dataparams
#         # TODO: Maybe add a function in dataparams that takes images and returns them as the model expects them or just do it here, it will almost always be zscore-train, but I won;t be using the mean and std of the dset to normalize here, probably irrelevant. 
# yeah, just use the mean and std fro all images, bringing t from the train is a bother, just chekc that the imagenormalization is zscore-train, otherwise throw an error saying i don;t know how to normalize
#         # Fetch models
#         # Iterate over images getting responses
#         # Save

#     normalize
#         Opt 1: use mean and std of original input or
#         opt 2: use train_mean and train_std from all_images (if this is the case I
#             don't need to fetch norm_method or know how the original input was normalized as long as it was normalized')

# class GallantParams(dj.Lookup):
#     definition = """
#     gallant_params:     smallint
#     ---
#     num_images:         float           # number of images to use for the reconstruction
#     weight_images:      boolean         # whether to weight each image by the similarity of its model response to target response
#     """
#     num_images 1, 5, 10, 50, 100
#     weight in true, false
#     # TODO: should I test different similarities or jsut compute response correlation

# @schema
# class GallantValEvaluation(dj.Computed):
#     definition = """ # evaluation on validation images

#     -> ModelResponses
#     -> GallantParams
#     ---
#     val_mse:            float       # validation MSE computed at the resolution in DataParams
#     val_corr:           float       # validation correlation computed at the resolution in DataParams
#     resized_val_mse:    float       # validation MSE at the resolution this model was fitted on
#     resized_val_corr:   float       # validation correlation at the resolution this model was fitted on
#     """
#     def make(self, key):
#         #TODO: REconstruct all validation images as desired in GallantParams

#         # Get data
#         images = (params.DataParams & key).get_images(key['dset_id'], split='val')
#         responses = (params.DataParams & key).get_responses(key['dset_id'], split='val')

#         # Get model
#         model = (MLPModel & key).get_model()

#         # Create reconstructions
#         recons = model.predict(responses)
#         h, w = (MLPParams & key).fetch1('image_height', 'image_width')
#         recons = recons.reshape(-1, h, w)

#         # Compute MSE
#         val_mse = ((images - utils.resize(recons, *images.shape[1:])) ** 2).mean()
#         resized_val_mse = ((utils.resize(images, *recons.shape[1:]) - recons) ** 2).mean()

#         # Compute correlation
#         val_corr = compute_correlation(images, utils.resize(recons, *images.shape[1:]))
#         resized_val_corr = compute_correlation(utils.resize(images, *recons.shape[1:]),
#                                                recons)

#         # Insert
#         self.insert1({**key, 'val_mse': val_mse, 'resized_val_mse': resized_val_mse,
#                       'val_corr': val_corr, 'resized_val_corr': resized_val_corr})

# @schema
# class GallantReconstructions(dj.Computed):
#     definition = """ # reconstructions for test set images (activity averaged across repeats)

#     -> ModelResponses
#     -> GallantParams
#     """

#     class Reconstruction(dj.Part):
#         definition = """ # reconstruction for a single image
#         -> master
#         -> data.Scan.Image
#         ---
#         recons:           longblob
#         """

#     def make(self, key):
#         #TODO: Reconstruct all test images as desired.
#         # Get data
#         responses = (params.DataParams & key).get_responses(key['dset_id'], split='test')

#         # Get model
#         model = (MLPModel & key).get_model()

#         # Create reconstructions
#         recons = model.predict(responses)

#         # Resize to orginal dimensions (144 x 256)
#         old_h, old_w = (MLPParams & key).fetch1('image_height', 'image_width')
#         new_h, new_w = (params.DataParams & key).fetch1('image_height', 'image_width')
#         recons = recons.reshape(-1, old_h, old_w)
#         recons = utils.resize(recons, new_h, new_w)

#         # Find out image ids of test set images
#         image_mask = (params.DataParams & key).get_image_mask(key['dset_id'],
#                                                               split='test')
#         image_classes, image_ids = (data.Scan.Image & key).fetch(
#             'image_class', 'image_id', order_by='image_class, image_id')
#         image_classes = image_classes[image_mask]
#         image_ids = image_ids[image_mask]

#         # Insert
#         self.insert1(key)
#         self.Reconstruction.insert(
#             [{**key, 'image_class': ic, 'image_id': id_, 'recons': r}
#              for ic, id_, r in zip(image_classes, image_ids, recons)])

# @schema
# class GallantEvaluation(dj.Computed):
#     definition = """ # evaluate gallant reconstructions in natural images in test set

#     -> ModelResponses
#     -> GallantParams
#     ---
#     test_mse:       float       # average MSE across all image
#     test_corr:      float       # average correlation (computed per image and averaged across images)
#     test_pixel_mse: longblob    # pixel-wise MSE (computed per pixel, averaged across images)
#     test_pixel_corr:longblob    # pixel-wise correlation (computed per pixel across images)
#     """

#     def make(self, key):
#         # Get original images
#         images = (params.DataParams & key).get_images(key['dset_id'], split='test')

#         # Get reconstructions
#         recons, image_classes = (GallantReconstructions.Reconstruction & key).fetch(
#             'recons', 'image_class', order_by='image_class, image_id')
#         recons = np.stack(recons)

#         # Restrict to natural images
#         images = images[image_classes == 'imagenet']
#         recons = recons[image_classes == 'imagenet']

#         # Compute MSE
#         mse = ((images - recons)**2).mean()
#         pixel_mse = ((images - recons)**2).mean(axis=0)

#         # Compute corrs
#         corr = compute_correlation(images, recons)
#         pixel_corr = compute_pixelwise_correlation(images, recons)

#         # Insert
#         self.insert1({
#             **key, 'test_mse': mse, 'test_corr': corr, 'test_pixel_mse': pixel_mse,
#             'test_pixel_corr': pixel_corr})

# ########################################################################################
# """ Gradient decoding """"

# class ReconstructionParamds
#     loss = mse, poisson, l1, linfinity

# class ModelReconstruction():
#     pass
#     # reconstruct based on model responses (this is just to get a upper bound of how good the reconstruction can be)
#     # use model responser as a normalization to report result MSE(neural) / MSE(model) (will gie me a 0-1 range)

# class SingleTrialReconstruction():
#     use Data Params with repears=False
#     pass

# class BlankReconstructions():
#   # maybe add a parameter in dataParams.get_reponses that is blank=False, so it fetches the blank responses
#   # rather than the actual responses (but process them the same).
#
"""    
#TODO: Decide whether I like option 3 or 4 here.



# option 1: (like LinearValEvaluation)
# reconstruct all images and evaluate in the same swope (does not save the reconstructions)
class Evaluation
    ->dset_id
    ---
    # reconstruct and evaluate in the same make
    
# option 2: like LinearReconstruction
# all reconstructions in the same step but it also saves them in the intermediate table
class Reconstructions
    -> dset
    # optionally add a set here if I wanna have the same table for both
    class OneImage
        ->dset
        -> imageid
        ---
        recons

class Evaluation
    -> Reconstructions
    # have to check that the image_ids in OneImage agree witht the ones in cell_mask


# option 3: like 
# reconstructions are done image by image and stored
class OneImageReconstruction
    -> image_id
    ---
     recons

class Reconstructions
    -> dset_id
    class OneImage
        ->dset_id
        -> image_id
    # check that all the images for this set (be it validation or test) are here

class Evaluation
    -> Reconstructions
    
# option 4:
# have a single OneImageReconstruction table that does the reconstructions from test set and validation set images rather than a different and lump them together in te ReconstructionSet table.
# - it mixes validation evaluations with test set evaluations: will have 30 validations with the same set and one test (all the tested validation models plus the final model)
class OneImageReconstruction
    dset_id
    image_id
    has all reconstructions (from test set and validation set)
    #TODO: how do I restrict this to only populate test and validation
    #TODO: Reconsider having a table that record for each image in dset id for any specific data params where it comes from  

class SetId(lookup):
    set: varchar(8)
    {'set': 'val', 'set': 'test'}

class Reconstructions():
    dset_id
    set_id
    class OneImage
    
    def make():
        just create a set as desired by set_id
        # also restrict to only stuff in image_class='imaenet

class Evaluation
    -> Reconstructions
    # restrict to only those in the set (maybe fetch all )
    
# Single Trial wll be like option 3 (or 4) but each OneImageRecons will have a part table with the  trials, i.e., the reconstructions will be done per chunks of 10 (or 40) trials in a single make.

"""