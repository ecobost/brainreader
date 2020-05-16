""" Compute MEIs with our models to confirm everything looks ok."""
import datajoint as dj
import numpy as np

from brainreader import data
from brainreader import params as brparams
from brainreader.encoding import params
from brainreader.encoding import train
import featurevis
from featurevis import ops
import torch

schema = dj.schema('br_encoding_meis')


#TODO: Find a better place for this table, drop it and move it there.
@schema
class ScanQuality(dj.Computed):
    definition = """ # compute a couple of qulaity metrics per cell
    
    -> data.Responses
    ---
    avg_snr:            float           # average signal-to-noise ratio across cells
    std_snr:            float           # standard deviation of snr across cells
    avg_oracle:         float           # average oracle across cells
    std_oracle:         float           # standard deviation of oracle across cells
    """
    class Unit(dj.Part):
        definition = """ # a single unit in the scan
        
        -> master
        -> data.Scan.Unit
        ---
        snr:            float       # mean-to-std ratio (over trials) averaged across images
        oracle:         float       # leave-one-out cross-correlation across trials and repeats 
        """
    def make(self, key):
        # Get responses (for images that were shown more than 5 times)
        repeated_images = (data.Scan.Image & 'num_repeats > 5')
        responses = (data.Responses.PerImage & repeated_images & key).fetch(
            'response')
        responses = np.stack(responses)  # num_images x num_repeats x num_cells

        # Compute SNR
        nsr = responses.std(1) / (responses.mean(1) + 1e-9) # 1/snr
        snr = 1 / nsr.mean(0)

        # Compute leave-one-out mean (used below for oracle score)
        num_repeats = responses.shape[1]
        loo_mean = (responses.sum(1, keepdims=True) - responses) / (num_repeats - 1)

        # Compute correlation (per cell) across all images and repeats of loo_mean with
        # left out sample
        x = responses.reshape(-1, responses.shape[-1])
        y = loo_mean.reshape(-1, loo_mean.shape[-1])
        x_res = x - x.mean(0)
        y_res = y - y.mean(0)
        corrs = (x_res * y_res).sum(0) / np.sqrt((x_res**2).sum(0) * (y_res**2).sum(0))
        oracle = corrs

        # Insert
        self.insert1({**key, 'avg_snr': snr.mean(), 'std_snr': snr.std(),
                      'avg_oracle': oracle.mean(), 'std_oracle': oracle.std()})
        self.Unit.insert([{**key, 'unit_id': i, 'snr': s, 'oracle': o}
                          for i, (s, o) in enumerate(zip(snr, oracle), start=1)])


@schema
class MEIParams(dj.Lookup):
    definition = """ # parameters used to optimize MEI
    mei_params:     tinyint
    ---
    height:         smallint        # height of the image to optimize
    width:          smallint        # widht of the image to optimize
    step_size:      float           # step size for the gradient optimization
    num_iterations: smallint        # number of gradient ascent iterations
    contrast_std:   float           # optimize in the isosphere of this contrast
    gradient_blur:  float           # sigma for the blurring in the gradient
    """
    contents = [{'mei_params': 1, 'height': 144, 'width': 256, 'step_size':100,
                'num_iterations': 1000, 'contrast_std': 1, 'gradient_blur': 2},]


@schema
class Seed(dj.Lookup):
    definition = """ # seed for the initial image
    seed:           int
    """
    contents = [{'seed': 1234}]


@schema
class EnsembleMEI(dj.Computed):
    definition = """ # most exciting image
    
    -> data.Scan.Unit
    -> train.Ensemble
    -> MEIParams
    -> Seed
    ---
    mei:            longblob            # MEI
    acts:           longblob            # activations during optimization
    """
    @property
    def key_source(self):
        # only ensembles with the appropiate dset id
        all_keys = data.Scan.Unit * train.Ensemble * MEIParams * Seed
        return all_keys & 'dset_id = ensemble_dset'

    def make(self, key):
        # Get model
        model = (train.Ensemble & key).get_model()
        model.eval()
        model.cuda()

        # Find index of the unit in model
        cell_mask = (brparams.DataParams & (train.Ensemble.OneModel & key)).get_cell_mask(
            key['dset_id'])
        if not cell_mask[key['unit_id'] - 1]:
            raise ValueError('This unit was not used to train the model.')
        idx_in_model = np.count_nonzero(cell_mask[:key['unit_id'] - 1])

        # Define optimization objective
        for param in model.parameters():
            param.requires_grad = False  # (optional) avoids saving intermediate gradients
        def objective(x):
            return model(x)[:, idx_in_model].mean()

        # Get initial (random) image
        torch.manual_seed(key['seed'])
        mei_params = (MEIParams & key).fetch1()
        initial_image = torch.randn(1, 1, mei_params['height'], mei_params['width'],
                                    device='cuda', dtype=torch.float32)

        # Optimize
        opt_x, fevals, _ = featurevis.gradient_ascent(
            objective, initial_image, step_size=mei_params['step_size'],
            num_iterations=mei_params['num_iterations'],
            post_update=ops.ChangeStd(mei_params['contrast_std']),
            gradient_f=ops.GaussianBlur(mei_params['gradient_blur']))
        mei = opt_x.squeeze().detach().cpu().numpy()

        # Insert
        self.insert1({**key, 'mei': mei, 'acts': fevals})