import datajoint as dj
import numpy as np
import itertools

from brainreader import data
from brainreader import utils

schema = dj.schema('br_params')
dj.config["enable_python_native_blobs"] = True  # allow blob support in dj 0.12


@schema
class ImageNormalization(dj.Lookup):
    definition = """ # methods to normalize images

    img_normalization:  varchar(16)
    ---
    description:        varchar(256)
    """
    contents = [
        {
            'img_normalization': 'zscore-train',
            'description': 'Normalize images using mean and std calculated in the training '
            'images'}, ]


@schema
class ResponseNormalization(dj.Lookup):
    definition = """ # methods to normalize neural responses

    resp_normalization: varchar(16)
    ---
    description:        varchar(256)
    """
    contents = [
        {
            'resp_normalization': 'zscore-blanks',
            'description': 'Normalize responses (per cell) using mean and std calculated in '
            'responses to all blank images'},
        {
            'resp_normalization': 'stddev-blanks',
            'description': 'Normalize responses (per cell) by dividing by std calculated '
            'over responses to blank images. Does not subtract the mean.'}, ]


@schema
class DataParams(dj.Lookup):
    definition = """ # data-specific parameters

    data_params:        smallint    # id of data params
    ---
    -> SplitParams                  # how to split the examples into train/val/test
    image_height:       smallint    # height for images
    image_width:        smallint    # width for images
    -> ImageNormalization           # how images are normalized
    only_soma:          boolean     # whether we should restrict to cells that have been calssified as soma
    discard_edge:       smallint    # (um) cells closer to the edge than this will be ignored (negative numbers include everything)
    discard_unknown:    boolean     # whether to discard cells from an unknown area
    -> ResponseNormalization        # how neural responses will be normalized
    """

    @property
    def contents(self):
        resp_norms = ['zscore-blanks', 'stddev-blanks']
        for i, resp_norm in enumerate(resp_norms, start=1):
            yield {
                'data_params': i, 'split_params': 1, 'image_height': 144,
                'image_width': 256, 'img_normalization': 'zscore-train',
                'only_soma': False, 'discard_edge': 8, 'discard_unknown': True,
                'resp_normalization': resp_norm}

    def get_images(self, dset_id, split='train'):
        """ Gets images shown during this dataset

        Arguments:
            dset_id (int): A dataset from data.Scan.
            split(str): What split to return: 'train', 'val, 'test'. None will return all
                images.

        Returns:
            images (np.array): An np.float array (num_images x height x width) with the
                images ordered by (image_class, image_id).
        """
        # Get images
        images = (data.Image & (data.Scan.Image & {'dset_id': dset_id})).fetch(
            'image', order_by='image_class, image_id')
        images = np.stack(images).astype(np.float32)

        # Resize
        desired_height, desired_width = self.fetch1('image_height', 'image_width')
        images = utils.resize(images, desired_height, desired_width)

        # Normalize
        img_normalization = self.fetch1('img_normalization')
        if img_normalization == 'zscore-train':
            train_mask = self.get_image_mask(dset_id, 'train')
            img_mean = images[train_mask].mean()
            img_std = images[train_mask].std(axis=(-1, -2)).mean()
        else:
            msg = f'Image normalization {img_normalization} not implemented'
            raise NotImplementedError(msg)
        images = (images - img_mean) / img_std

        # Restrict to right split
        if split is not None:
            img_mask = self.get_image_mask(dset_id, split)
            images = images[img_mask]

        return images

    def get_image_mask(self, dset_id, split='train'):
        """ Creates a boolean mask the same size as the number of images in the scan with
        True for the images in the desired split.

        Arguments:
            dset_id (int): A dataset from data.Scan.
            split (str): What split to return: 'train', 'val, 'test'.

        Returns:
            mask (np.array) A boolean array of size num_images with the desired assignment
                for all images in the dataset (ordered by image_class, image_id).
        """
        # Check data.Split is populated
        if len(data.Split & self & {'dset_id': dset_id}) != 1:
            raise ValueError(f'Data.Split for dset {dset_id} has not been populated yet.')

        # Create image mask
        assignments = (data.Split.PerImage & self & {'dset_id': dset_id}).fetch(
            'split', order_by='image_class, image_id')
        mask = assignments == split

        return mask

    #TODO: maybe add a parameter to return the blank responses (rather than actual responses)
    def get_responses(self, dset_id, split='train', avg_repeats=True):
        """ Gets responses obtained in this dataset

        Arguments:
            dset_id (int): A dataset from data.Scan.
            split(str): What split to return: 'train', 'val, 'test'. None will return all
                images.
            avg_repeats (bool): Average the responses to an image across repeats. This
                will only work if all images have the same number of repeats.

        Returns:
            responses (np.array): An np.float array (num_images x num_cells if avg_repeats
                else num_images x num_repeats x num_cells) with the responses. Images
                ordered by (image_class, image_id), cells ordered by unit_id.
        """
        # Get all responses
        all_responses = (data.Responses.PerImage & {'dset_id': dset_id}).fetch(
            'response', order_by='image_class, image_id')

        # Restrict to responses for images in desired split (and average repeats)
        if split is None:
            responses = all_responses
        else:
            img_mask = self.get_image_mask(dset_id, split)
            responses = all_responses[img_mask]
        responses = np.stack([r.mean(0) for r in responses] if avg_repeats else responses)
        # will fail if avg_repeats==False and responses have a variable number of repeats

        # Normalize
        resp_normalization = self.fetch1('resp_normalization')
        blank_responses = np.concatenate(
            (data.Responses.PerImage & {'dset_id': dset_id}).fetch('blank_response'))
        if resp_normalization == 'zscore-blanks':
            resp_mean = blank_responses.mean(0)
        elif resp_normalization == 'stddev-blanks':
            resp_mean = 0  # do not subtract the mean
        else:
            msg = f'Response normalization {resp_normalization} not implemented'
            raise NotImplementedError(msg)
        resp_std = blank_responses.std(0)
        responses = (responses - resp_mean) / resp_std

        # Restrict to desired cells
        cell_mask = self.get_cell_mask(dset_id)
        responses = responses[..., cell_mask]

        return responses

    def get_cell_mask(self, dset_id):
        """ Creates a boolean mask the same size as the number of cells in the scan with
        True for the cells that fulfill the requirements for this DataParams entry.

        Arguments:
            dset_id (int): A dataset from data.Scan.

        Returns:
            mask (np.array) A boolean array of size num_cells with the cells that fulfill
                the restrictions in this DataParams (ordered by unit_id).
        """
        # Fetch cell properties
        is_somas, edge_distances, areas = (data.Scan.Unit & {'dset_id': dset_id}).fetch(
            'is_soma', 'edge_distance', 'brain_area', order_by='unit_id')

        # Get restrictions
        only_soma, discard_edge, discard_unknown = self.fetch1('only_soma',
                                                               'discard_edge',
                                                               'discard_unknown')

        # Create mask
        mask = np.logical_and(is_somas if only_soma else True,
                              edge_distances > discard_edge,
                              areas != 'unknown' if discard_unknown else True)

        return mask


@schema
class LinearParams(dj.Lookup):
    definition = """ # parameters for direct linear decoding
    
    linear_params:  smallint
    ---
    image_height:   smallint        # height of the image to be reconstructed
    image_width:    smallint        # width of the image to be reconstructed
    l2_weight:      float           # weight for the l2 regularization
    l1_weight:      float           # weight for the l1 regularization
    """

    @property
    def contents(self):
        dims = [(18, 32), (36, 64), (72, 128), (144, 256)]

        # no regularization
        for i, (h, w) in enumerate(dims, start=1):
            yield {
                'linear_params': i, 'image_height': h, 'image_width': w, 'l2_weight': 0,
                'l1_weight': 0}

        # l2 regularized
        l2_weights = 10**np.arange(2, 7.5, 0.5)
        for i, ((h, w), l2) in enumerate(itertools.product(dims, l2_weights),
                                         start=i + 1):
            yield {
                'linear_params': i, 'image_height': h, 'image_width': w, 'l2_weight': l2,
                'l1_weight': 0}

        # l1 regularized
        l1_weights = 10**np.arange(-2.5, 0.5, 0.5)
        dims = [(18, 32), (36, 64)] # higher res take too long (~days)
        for i, ((h, w), l1) in enumerate(itertools.product(dims, l1_weights),
                                         start=i + 1):
            yield {
                'linear_params': i, 'image_height': h, 'image_width': w, 'l2_weight': 0,
                'l1_weight': l1}

    def get_model(self, num_processes=8):
        """ Pick the right scikit model: Linear, Ridge, Lasso or ElasticNet.
        
        Arguments:
            num_processes (int): Number of processes to use during fitting. Used only for
                the least squares linear regression (l1_weight=0, l2_weight=0).
        """
        from sklearn import linear_model

        l1_weight, l2_weight = self.fetch1('l1_weight', 'l2_weight')
        if l1_weight == 0 and l2_weight == 0:
            model = linear_model.LinearRegression(n_jobs=num_processes)
        elif l1_weight != 0 and l2_weight == 0:
            model = linear_model.Lasso(alpha=l1_weight, tol=0.01)
        elif l1_weight == 0 and l2_weight != 0:
            model = linear_model.Ridge(alpha=l2_weight, random_state=1234)
        else:
            alpha = l1_weight + l2_weight
            l1_ratio = l1_weight / alpha
            model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        return model


@schema
class MLPArchitecture(dj.Lookup):
    definition = """ # architectural params for MLP
    
    mlp_architecture:   smallint
    ---
    layer_sizes:        blob        # number of feature maps in each hidden layer
    """
    contents = [(1, [1000]), (2, [5000]), (3, [15000])]


@schema
class MLPTraining(dj.Lookup):
    definition = """ # training hyperparams for mlp
    
    mlp_training:       smallint
    ---
    learning_rate:      float       # initial learning rate for the optimization
    l2_weight:          float       # weight for the l2 regularization
    """

    @property
    def contents(self):
        lrs = [1e-5, 1e-4]
        l2_weights = [0, 1e-4, 1e-3, 1e-2, 1e-1]
        for i, (lr, l2) in enumerate(itertools.product(lrs, l2_weights), start=1):
            yield {'mlp_training': i, 'learning_rate': lr, 'l2_weight': l2}


@schema
class MLPData(dj.Lookup):
    definition = """ # data specific params
    
    mlp_data:           smallint
    ---
    image_height:       smallint            # height of the image to predict
    image_width:        smallint            # width of the image to predict
    """
    contents = [(1, 18, 32), (2, 36, 64), (3, 72, 128), (4, 144, 256)]


@schema
class MLPParams(dj.Lookup):
    definition = """ # parameters to train an MLP decoding model
    
    mlp_params:         smallint
    ---
    -> MLPData
    -> MLPArchitecture
    -> MLPTraining
    """

    @property
    def contents(self):
        for i, (d, a, t) in enumerate(
                itertools.product(MLPData.fetch('mlp_data'),
                                  MLPArchitecture.fetch('mlp_architecture'),
                                  MLPTraining.fetch('mlp_training')), start=1):
            yield {
                'mlp_params': i, 'mlp_data': d, 'mlp_architecture': a, 'mlp_training': t}

@schema
class DeconvArchitecture(dj.Lookup):
    definition = """ # architectural params for deconvolution network
    
    deconv_architecture:   smallint
    ---
    layer_sizes:        blob        # number of feature maps in each hidden layer
    """
    contents = [(1, [512, 256, 128, 64, 32])]


@schema
class DeconvTraining(dj.Lookup):
    definition = """ # training hyperparams for mlp
    
    deconv_training:       smallint
    ---
    learning_rate:      float       # initial learning rate for the optimization
    l2_weight:          float       # weight for the l2 regularization
    """

    @property
    def contents(self):
        lrs = [1e-4, 1e-3]
        l2_weights = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for i, (lr, l2) in enumerate(itertools.product(lrs, l2_weights), start=1):
            yield {'deconv_training': i, 'learning_rate': lr, 'l2_weight': l2}


@schema
class DeconvData(dj.Lookup):
    definition = """ # data specific params
    
    deconv_data:           smallint
    ---
    image_height:       smallint            # height of the image to predict
    image_width:        smallint            # width of the image to predict
    """
    contents = [(4, 144, 256)]


@schema
class DeconvParams(dj.Lookup):
    definition = """ # parameters to train a deconvolution network
    
    deconv_params:         smallint
    ---
    -> DeconvData
    -> DeconvArchitecture
    -> DeconvTraining
    """

    @property
    def contents(self):
        for i, (d, a, t) in enumerate(
                itertools.product(DeconvData.fetch('deconv_data'),
                                  DeconvArchitecture.fetch('deconv_architecture'),
                                  DeconvTraining.fetch('deconv_training')), start=1):
            yield {
                'deconv_params': i, 'deconv_data': d, 'deconv_architecture': a,
                'deconv_training': t}


@schema
class GaborSet(dj.Lookup):
    definition = """ # params to create a gabor filter bank
    
    gabor_set:      smallint
    ---
    orientations:   blob        # orientations to try (in radians from 0-pi)
    phases:         blob        # phases to try (in radians from 0-2pi)
    wavelengths:    blob        # wavelength of the gabor (as a proportion of height)
    sigmas:         blob        # standard deviation of the gaussian window (i.e., aperture)
    dxs:            blob        # center of gabor in x (from -0.5 to 0.5)
    dys:            blob        # center of gabor in y (from -0.5 to 0.5)
    """
    @property
    def contents(self):
        # Okhi parameters (2256 gabors)
        orientations = [
            np.linspace(0, np.pi, 4, endpoint=False),
            np.linspace(0, np.pi, 4, endpoint=False) + np.pi / 8,
            np.linspace(0, np.pi, 4, endpoint=False),
            np.linspace(0, np.pi, 4, endpoint=False) + np.pi / 8]
        # Okhi does not shift orientations every other wavelet set but it improves recons (see 2.8.1 in Fischer et al., 2007)
        phases = [
            [0, np.pi / 2], ] * 4
        wavelengths = [[1.16], [0.58], [0.26], [0.13]]  # 1 / (np.array([0.02, 0.04, 0.09, 0.18]) * 43)
        sigmas = [[0.5], [0.25], [0.12], [0.07]]  # approx (they only give "size" in pixels)
        dys = [[0],
               np.linspace(-0.5, 0.5, 3),
               np.linspace(-0.5, 0.5, 5),
               np.linspace(-0.5, 0.5, 11)]
        dxs = [[-0.25, 0.25],
               np.linspace(-0.5, 0.5, int(round(3 * 16 / 9))),
               np.linspace(-0.5, 0.5, int(round(5 * 16 / 9))),
               np.linspace(-0.5, 0.5, int(round(11 * 16 / 9)))
               ]  # more points in x because of the 16:9 ratio
        yield {
            'gabor_set': 1, 'orientations': orientations, 'phases': phases,
            'wavelengths': wavelengths, 'sigmas': sigmas, 'dxs': dxs, 'dys': dys}

        # larger range (36992 gabors)
        orientations = [
            np.linspace(0, np.pi, 8, endpoint=False),
            np.linspace(0, np.pi, 8, endpoint=False) + np.pi / 16,
            np.linspace(0, np.pi, 8, endpoint=False),
            np.linspace(0, np.pi, 8, endpoint=False) + np.pi / 16,
            np.linspace(0, np.pi, 8, endpoint=False)]
        phases = [
            np.linspace(0, np.pi, 4, endpoint=False), ] * 5
        wavelengths = [[1], [0.59], [0.35], [0.20],
                       [0.12]]  # 1 / 1.7 ** np.array([0, 1, 2, 3, 4])
        sigmas = [[0.45], [0.26], [0.16], [0.09], [0.05]]
        dys = [
            np.linspace(-0.5, 0.5, 3),
            np.linspace(-0.5, 0.5, 4),
            np.linspace(-0.5, 0.5, 7),
            np.linspace(-0.5, 0.5, 12),
            np.linspace(-0.5, 0.5, 21)]
        dxs = [
            np.linspace(-0.5, 0.5, int(round(3 * 16 / 9))),
            np.linspace(-0.5, 0.5, int(round(4 * 16 / 9))),
            np.linspace(-0.5, 0.5, int(round(7 * 16 / 9))),
            np.linspace(-0.5, 0.5, int(round(12 * 16 / 9))),
            np.linspace(-0.5, 0.5, int(round(21 * 16 / 9)))]
        yield {
            'gabor_set': 2, 'orientations': orientations, 'phases': phases,
            'wavelengths': wavelengths, 'sigmas': sigmas, 'dxs': dxs, 'dys': dys}

    def get_gabors(self, height, width):
        """ Create the Gabor wavelets from this set
        
        Arguments:
            height (int): Height of the gabors
            width (int): Width of the gabors
            
        Returns
            A (num_gabors x height x width) array with all gabors.
        """
        # Fetch params
        orientations, phases, wavelengths, sigmas, dxs, dys = self.fetch1(
            'orientations', 'phases', 'wavelengths', 'sigmas', 'dxs', 'dys')

        # Create gabors
        gabors = []
        for os, ps, ws, ss, xs, ys in zip(orientations, phases, wavelengths, sigmas, dxs,
                                          dys):  # iterate over groups
            for o, p, w, s, x, y in itertools.product(os, ps, ws, ss, xs, ys):
                gabors.append(utils.create_gabor(height, width, orientation=o, phase=p,
                                                 wavelength=w, sigma=s, dx=x, dy=y))
        gabors = np.stack(gabors)

        # Normalize (to get an almost orthogonal set)
        l2norm = np.sqrt((gabors**2).sum(axis=(-1, -2), keepdims=True))
        gabors = gabors / l2norm

        return gabors


@schema
class GaborParams(dj.Lookup):
    definition = """ # parameters for Gabor reconstructions
    
    gabor_params:  smallint
    ---
    image_height:   smallint        # height of the image to be reconstructed
    image_width:    smallint        # width of the image to be reconstructed
    -> GaborSet
    l2_weight:      float           # weight for the l2 regularization
    l1_weight:      float           # weight for the l1 regularization
    """

    @property
    def contents(self):
        dims = [(18, 32), (36, 64), (72, 128), (144, 256)]
        gabor_sets = [1, 2]

        # no regularization
        for i,((h, w), gs) in enumerate(itertools.product(dims, gabor_sets), start=1):
            yield {
                'gabor_params': i, 'image_height': h, 'image_width': w, 'gabor_set': gs,
                'l2_weight': 0, 'l1_weight': 0}

        # l2 regularized
        l2_weights = 10**np.arange(3, 7.5, 0.5)
        for i, ((h, w), gs,
                l2) in enumerate(itertools.product(dims, gabor_sets, l2_weights),
                                 start=i + 1):
            yield {
                'gabor_params': i, 'image_height': h, 'image_width': w, 'gabor_set': gs,
                'l2_weight': l2, 'l1_weight': 0}

        # l1 regularized
        l1_weights = 10**np.arange(-2, 0.5, 0.5)
        gabor_sets = [1] # bigger gabor set takes too long
        for i, ((h, w), gs, l1) in enumerate(itertools.product(dims, gabor_sets, l1_weights),
                                         start=i + 1):
            yield {
                'gabor_params': i, 'image_height': h, 'image_width': w, 'gabor_set': gs,
                'l2_weight': 0, 'l1_weight': l1}


    def get_model(self, num_processes=8):
        """ Pick the right scikit model: Linear, Ridge, Lasso or ElasticNet.
        
        Arguments:
            num_processes (int): Number of processes to use during fitting. Used only for
                the least squares linear regression (l1_weight=0, l2_weight=0).
        """
        from sklearn import linear_model

        l1_weight, l2_weight = self.fetch1('l1_weight', 'l2_weight')
        if l1_weight == 0 and l2_weight == 0:
            model = linear_model.LinearRegression(n_jobs=num_processes)
        elif l1_weight != 0 and l2_weight == 0:
            model = linear_model.Lasso(alpha=l1_weight, tol=0.01)
        elif l1_weight == 0 and l2_weight != 0:
            model = linear_model.Ridge(alpha=l2_weight, random_state=1234)
        else:
            alpha = l1_weight + l2_weight
            l1_ratio = l1_weight / alpha
            model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        return model

@schema
class AHPParams(dj.Lookup):
    definition = """ # params for average high posterior reconstructions
    
    ahp_params:     smallint
    ---
    num_samples:    smallint    # number of images to average for the reconstruction
    similarity:     varchar(16) # similarity measure to use between target and model responses
    weight_samples: boolean     # whether to weight each image by the similarity of 
    """

    @property
    def contents(self):
        num_images = [1, 3, 10, 32, 100, 316]
        similarity = ['correlation', 'poisson_loglik']
        weight_images = [False, True]

        # for num_images 1 weight_images False and True are the same thing
        for i, s in enumerate(similarity, start=1):
            yield {
                'ahp_params': i, 'num_samples': 1, 'similarity': s,
                'weight_samples': False}

        # for num_images 1 weight_images False and True are the same thing
        for i, (ni, s, w) in enumerate(itertools.product(num_images[1:], similarity, weight_images),
                                       start=i + 1):
            yield {
                'ahp_params': i, 'num_samples': ni, 'similarity': s, 'weight_samples': w}

@schema
class GradientParams(dj.Lookup):
    definition = """ # params for gradient based reconstruction
    
    gradient_params:    smallint
    ---
    seed:               smallint    # seed used to get the initial (random estimate)
    num_iterations:     smallint    # number of iterations for the optimization
    step_size:          float       # step size for the optimization
    similarity:         varchar(16) # how to measure similarity between target responses and model responses
    jitter:             tinyint     # (pixels) amount of jittering to apply every iteration
    gradient_sigma:     float       # (pixels) blur the gradient using a gaussian window with this sigma
    l2_weight:          float       # weight for the l2-norm regularization used during optimization
    keep_std_fixed:     boolean     # whether to keep standard deviation of optimized image at 1
    """

    @property
    def contents(self):
        # mse similarity
        step_sizes = [100, 1000]
        jitters = [0, 1, 3, 5, 8]
        sigmas = [0, 3, 5, 10, 15]
        l2_weights = [0, 1e-2]
        fix_stds = [False, True]
        for i, (ss, j, gs, l2, fs) in enumerate(
                itertools.product(step_sizes, jitters, sigmas, l2_weights, fix_stds),
                start=1):
            yield {
                'gradient_params': i, 'seed': 2345, 'num_iterations': 1000,
                'step_size': ss, 'similarity': 'negmse', 'jitter': j,
                'gradient_sigma': gs, 'l2_weight': l2, 'keep_std_fixed': fs}
        """
        good combinations:
        keep_std_fixed = True
        l2_weight = 0 # smaller l2 than 1e-2 seems to have no effect either.
        step_size = 1000 
        Wants higher sigmas 
        maybe drop jitter (does the say as gaussian blurring and seems irrelevant)
        """

        # Cosine similarity
        step_sizes = [1000, 10000]
        jitters = [0, 1, 3, 5, 8]
        sigmas = [0, 3, 5, 10, 15]
        l2_weights = [0]#, 1e-3]
        for i, (ss, j, gs, l2) in enumerate(itertools.product(step_sizes, jitters, sigmas, l2_weights), start=i+1):
            yield {
                'gradient_params': i, 'seed': 2345, 'num_iterations': 1000,
                'step_size': ss, 'similarity': 'cosine', 'jitter': j, 'gradient_sigma': gs,
                'l2_weight': l2, 'keep_std_fixed': True}

        # Poisson log likelihood similarity
        step_sizes = [1000, 10000]
        jitters = [0, 1, 3, 5, 8]
        sigmas = [0, 3, 5, 10, 15]
        l2_weights = [0]#, 1e-3]
        for i, (ss, j, gs, l2) in enumerate(
                itertools.product(step_sizes, jitters, sigmas, l2_weights), start=i+1):
            yield {
                'gradient_params': i, 'seed': 2345, 'num_iterations': 1000,
                'step_size': ss, 'similarity': 'poisson_loglik', 'jitter': j, 'gradient_sigma': gs,
                'l2_weight': l2, 'keep_std_fixed': True}


@schema
class GeneratorMNISTParams(dj.Lookup):
    definition = """ # params for reconstructions with a generator prior
    
    mnistgen_params:    smallint
    ---
    seed:               smallint    # random seed used to obtain initial estimate
    generator:          varchar(8)  # type of generator to use
    step_size:          float       # step size for the optimization
    num_iterations:     smallint    # number of iterations for this model
    """
    contents = [
        {
            'mnistgen_params': 1, 'seed': 6543, 'generator': 'gan', 'step_size': 1,
            'num_iterations': 1000},
        {
            'mnistgen_params': 2, 'seed': 6453, 'generator': 'vae', 'step_size': 1,
            'num_iterations': 1000}, ]


# @schema
# class GeneratorNaturalParams(dj.Lookup):
#     definition = """ # params for reconstructions with a generator prior

#     natgen_params:    smallint
#     ---
#     seed:               smallint    # random seed used to obtain initial estimate
#     step_size:          float       # step size for the optimization
#     num_iterations:     smallint    # number of iterations for this model
#     truncate_at:        float       # truncate the normal noise at this value (between 0-1)
#     """
#     contents = [{'natgen_params': 1, 'seed': 6543, 'step_size': 1, 'num_iterations': 1000,
#                  'truncate_at': 0.7}]

@schema
class SpectrumParams(dj.Lookup):
    definition = """ # parameters to evaluate reconstructions in frequency bins of the fourier spectrum
    
    spectrum_params:    smallint
    ---
    bin_size:           float           # amount of frequency in each bin (0-0.5)
    interval:           float           # how often to sample the bins
    """
    contents = [{'spectrum_params': 1, 'bin_size': 0.05, 'interval': 0.025}]
    
    def get_freqs(self):
        """Returns the halfway point of each frequency bin for one set of params"""
        bin_size, interval = self.fetch1('bin_size', 'interval')
        freqs = np.arange(bin_size/2, 0.5 - bin_size / 2 + 1e-9, interval)
        return freqs
