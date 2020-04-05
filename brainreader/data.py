"""
Bring data from our two-photon pipeline.
This is the only module that interacts with stuff outside this package.
"""
import datajoint as dj
import numpy as np

reso = dj.create_virtual_module('reso', 'pipeline_reso')
meso = dj.create_virtual_module('meso', 'pipeline_meso')
stack = dj.create_virtual_module('stack', 'pipeline_stack')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')

schema = dj.schema('br_data')
dj.config['stores'] = {
    'brdata': {'protocol': 'file', 'location': '/mnt/scratch07/ecobost'}, }

# Scans with training and testing images shown to mice
scans = [
    {'animal_id': 20892, 'session': 10, 'scan_idx': 10}, ]


@schema
class Scan(dj.Computed):
    definition = """ # a single scan (and its properties) from our pipeline
    
    dset_id:        smallint    # id of this dset
    ---
    animal_id:      smallint    # id of the animal (from experiment.Scan)
    session:        tinyint     # session in this animal (from experiment.Scan)
    scan_idx:       tinyint     # number of scan during this session (from experiment.Scan)
    num_fields:     tinyint     # number of fields in this scan
    num_depths:     tinyint     # number of depths in the scan (z-slices)
    fps:            float       # frame rate at which this scan was recorded
    num_frames:     mediumint   # number of frames recorded in this scan 
    num_units:      smallint    # number of units segmented in this field
    um_height:      smallint    # (um) height of the first field in this scan (usually all fields have the same dimensions)        
    um_width:       smallint    # (um) width of the first field in this scan (usually all fields have the same dimensions)
    px_height:      smallint    # (px) height of the first field in this scan (usually all fields have the same dimensions)        
    px_width:       smallint    # (px) width of the first field in this scan (usually all fields have the same dimensions)
    stack_session:  smallint    # session of one stack this scan has been registered to (same as in experiment.Stack) 
    stack_idx:      smallint    # stack_idx for one stack this scans has been registered to (same as in experiment.Stack)
    first_z:        float       # (um) depth of the first field in the scan (from the stack registration)
    last_z:         float       # (um) depth of the last field in the scan (from the stack registration)
    """

    @property
    def key_source(self):
        """Changes the key source to be ('animal_id', 'session', 'scan_idx') rather than
        dset_id.

        Run only for those scans in `scans` list that have not already been populated.
        """
        experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
        return (experiment.Scan - Scan) & scans

    class Unit(dj.Part):
        definition = """ # all recorded units in this scan
        -> master
        unit_id:    smallint    # this matches with the unit id in meso.ScanSet
        ---
        field:      tinyint     # field where this unit was segmented
        mask_id:    smallint    # mask_id in the field this unit was segmented
        stack_x:    float       # x position in stack coordinates
        stack_y:    float       # y position in stack coordinates
        stack_z:    float       # z position in stack coordinates (0 is usually surface depth)
        is_soma:    boolean     # whether it was classified as soma by the 2-d classifier
        brain_area: varchar(5)  # what area is this unit from
        layer:      varchar(5)  # layer to which this cell belongs
        ms_delay:   smallint    # delay in milliseconds from start of the volume
        edge_distance: float    # (um) distance to the closest edge (vertical or horizontal) of the field
        """

    class Image(dj.Part):
        definition = """ # images shown during this scan
        
        -> master 
        image_class: varchar(32) # type of image presented (same as in stimulus.StaticImage)
        image_id:   int         # id of this image (same as in stimulus.StaticImage)           
        ---
        num_repeats: tinyint    # number of repetitions of this image in this scan
        condition_hash: varchar(20) # identifying hash in stimulus pipeline for this image (when shown to this scan).
        """

    def make(self, key):
        # Assign dset id
        dset_id = max([0, *Scan.fetch('dset_id')]) + 1
        tuple_ = key.copy()
        tuple_['dset_id'] = dset_id

        # Fill in scan properties
        pipe = reso if (reso.ScanInfo & key) else meso
        tuple_['num_fields'] = (pipe.ScanInfo & key).fetch1('nfields')
        tuple_['num_depths'] = len(np.unique((pipe.ScanInfo.Field & key).fetch('z')))
        tuple_['fps'] = (pipe.ScanInfo & key).fetch1('fps')
        tuple_['num_frames'] = (pipe.ScanInfo & key).fetch1('nframes')
        tuple_['num_units'] = len(pipe.Activity.Trace & key)
        tuple_['um_height'] = (pipe.ScanInfo.Field & key &
                               {'field': 1}).fetch1('um_height')
        tuple_['um_width'] = (pipe.ScanInfo.Field & key & {'field': 1}).fetch1('um_width')
        tuple_['px_height'] = (pipe.ScanInfo.Field & key &
                               {'field': 1}).fetch1('px_height')
        tuple_['px_width'] = (pipe.ScanInfo.Field & key & {'field': 1}).fetch1('px_width')

        # Find one stack to which all fields in this scan have been registered
        candidate_stacks = dj.U('animal_id', 'stack_session', 'stack_idx').aggr(
            stack.Registration & key & {'scan_session': key['session']},
            nfields='COUNT(*)') & {'nfields': (pipe.ScanInfo & key).fetch1('nfields')}
        if len(candidate_stacks) == 0:
            raise ValueError('Scan {animal_id}-{session}-{scan_idx} has not been fully'
                             'registered into any stack')
        stack_key = candidate_stacks.fetch(order_by='stack_session, stack_idx')[0]
        tuple_['stack_session'] = stack_key['stack_session']
        tuple_['stack_idx'] = stack_key['stack_idx']
        depths = (stack.Registration.Affine & key & stack_key).fetch('reg_z')
        tuple_['first_z'] = min(depths)
        tuple_['last_z'] = max(depths)

        # Insert in scan
        self.insert1(tuple_)

        # Fill in unit properties
        unit_ids, fields, mask_ids = (pipe.ScanSet.Unit & key).fetch(
            'unit_id', 'field', 'mask_id', order_by='unit_id')
        xs, ys, zs = (pipe.StackCoordinates.UnitInfo & key & stack_key).fetch(
            'stack_x', 'stack_y', 'stack_z', order_by='unit_id')
        unit_types = (pipe.MaskClassification.Type * pipe.ScanSet.Unit & key).fetch(
            'type', order_by='unit_id')
        areas = (anatomy.AreaMembership & key).fetch('brain_area', order_by='unit_id')
        layers = (anatomy.LayerMembership & key).fetch('layer', order_by='unit_id')
        ms_delays = (pipe.ScanSet.UnitInfo & key).fetch('ms_delay', order_by='unit_id')

        # Find distance to closest edge of each unit
        px_x, px_y = (pipe.ScanSet.UnitInfo & key).fetch('px_x', 'px_y',
                                                         order_by='unit_id')
        px_x, px_y = px_x + 0.5, px_y + 0.5
        if pipe == reso:
            px_height, px_width, um_height, um_width = (pipe.ScanInfo & key).fetch1(
                'px_height', 'px_width', 'um_height', 'um_width')
        else:
            # height and width can change per field (and thus unit_id)
            px_height, px_width, um_height, um_width = (pipe.ScanInfo.Field & key).fetch(
                'px_height', 'px_width', 'um_height', 'um_width', order_by='field')
            px_height, px_width = px_height[fields - 1], px_width[fields - 1]
            um_height, um_width = um_height[fields - 1], um_width[fields - 1]
        x_dist = np.minimum(px_x, px_width - px_x) * (um_width / px_width)
        y_dist = np.minimum(px_y, px_height - px_y) * (um_height / px_height)
        edge_distances = np.minimum(x_dist, y_dist)

        # Insert units
        units = [{
            'dset_id': dset_id, 'unit_id': unit_id, 'field': field, 'mask_id': mask_id,
            'stack_x': x, 'stack_y': y, 'stack_z': z, 'is_soma': type_ == 'soma',
            'brain_area': area, 'layer': layer, 'ms_delay': ms_delay, 'edge_distance': ed}
                 for (unit_id, field, mask_id, x, y, z, type_, area, layer, ms_delay,
                      ed) in zip(unit_ids, fields, mask_ids, xs, ys, zs, unit_types,
                                 areas, layers, ms_delays, edge_distances)]
        self.Unit.insert(units)

        # Fill in images shown during this scan
        images = dj.U('condition_hash', 'image_class',
                      'image_id').aggr(stimulus.Frame * stimulus.Trial & key,
                                       num_repeats='COUNT(*)').fetch(as_dict=True)
        self.Image.insert([{'dset_id': dset_id, **im} for im in images])


@schema
class Image(dj.Computed):
    definition = """ # images shown during scans
    
    image_class:    varchar(32)   # type of image presented (same as in stimulus.StaticImage)
    image_id:       int           # id of this image (same as in stimulus.StaticImage)      
    ---
    image:          blob@brdata   # image (h x w) as uint8 (as shown during stimulation)
    """

    @property
    def key_source(self):
        return dj.U('image_class', 'image_id') & Scan.Image  # only images shown in a scan

    def make(self, key):
        image = (stimulus.StaticImage.Image & key).fetch1('image')
        self.insert1({**key, 'image': image})


def get_traces(key):
    """ Get spike traces for all cells in these scan (along with their times in stimulus
    clock).

    Arguments:
        key (dict): Key for a scan (or field).

    Returns:
        traces (np.array): A (num_units x num_scan_frames) array with all spike traces.
            Traces are restricted to those classified as soma and ordered by unit_id.
        unit_ids (list): A (num_units) list of unit_ids in traces.
        trace_times (np.array): A (num_units x num_scan_frames) array with the time (in
            seconds) for each unit's trace in stimulus clock (same clock as times in
            stimulus.Trial).

    Note: On notation
        What is called a frametime in stimulus.Sync and stimulus.Trial is actually the
        time each depth of scanning started. So for a scan with 1000 frames and four
        depths per frame/volume, there will be 4000 "frametimes".

    Note 2:
        For a scan with 10 depths, a frame i is considered complete if all 10 depths were
        recorded and saved in the tiff file, frame_times however save the starting time of
        each depth independently (for instance if 15 depths were recorded there will be
        one scan frame but 15 frame times, the last 5 have to be ignored).
    """
    # Pick right pipeline for this scan (reso or meso)
    pipe = reso if (reso.ScanInfo & key) else meso

    # Get traces
    spikes = pipe.Activity.Trace * pipe.ScanSet.UnitInfo & key
    unit_ids, traces, ms_delays = spikes.fetch('unit_id', 'trace', 'ms_delay',
                                               order_by='unit_id')

    # Get time of each scan frame for this scan (in stimulus clock; same as in Trial)
    depth_times = (stimulus.Sync & key).fetch1('frame_times')
    num_frames = (pipe.ScanInfo & key).fetch1('nframes')
    num_depths = len(dj.U('z') & (pipe.ScanInfo.Field.proj('z', nomatch='field') & key))
    if ((len(depth_times) / num_depths < num_frames) or
        (len(depth_times) / num_depths > num_frames + 1)):
        raise ValueError('Mismatch between frame times and tiff frames')
    frame_times = depth_times[:num_depths * num_frames:num_depths]  # one per frame

    # Add per-cell delay to each frame_time
    trace_times = np.add.outer(ms_delays / 1000, frame_times)  # num_traces x num_frames

    return np.stack(traces), np.stack(unit_ids), trace_times


def trapezoid_integration(x, y, x0, xf):
    """ Integrate y (recorded at points x) from x0 to xf.

    Arguments:
        x (np.array): Timepoints (num_timepoints) when y was recorded.
        y (np.array): Signal (num_timepoints).
        x0 (float or np.array): Starting point(s). Could be a 1-d array (num_samples).
        xf (float or np.array): Final point. Same shape as x0.

    Returns:
        Integrated signal from x0 to xf:
            a 0-d array (i.e., float) if x0 and xf are floats
            a 1-d array (num_samples) if x0 and xf are 1-d arrays
    """
    # Basic checks
    if np.any(xf <= x0):
        raise ValueError('xf has to be higher than x0')
    if np.any(x0 < x[0]) or np.any(xf > x[-1]):
        raise ValueError('Cannot integrate outside the original range x of the signal.')

    # Compute area under each trapezoid
    trapzs = np.diff(x) * (y[:-1] + y[1:]) / 2  # index i is trapezoid from point i to point i + 1

    # Find timepoints right before x0 and xf
    idx_before_x0 = np.searchsorted(x, x0) - 1
    idx_before_xf = np.searchsorted(x, xf) - 1

    # Compute y at the x0 and xf points
    slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])  # index i is slope from p_i to p_{i+1}
    y0 = y[idx_before_x0] + slopes[idx_before_x0] * (x0 - x[idx_before_x0])
    yf = y[idx_before_xf] + slopes[idx_before_xf] * (xf - x[idx_before_xf])

    # Sum area of all interior trapezoids
    indices = np.stack([idx_before_x0 + 1, idx_before_xf], axis=-1).ravel()  # interleaved x0 and xf for all samples
    integral = np.add.reduceat(trapzs, indices, axis=-1)[::2].squeeze()

    # Add area of edge trapezoids (ones that go from x0 to first_x_sample and from last_x_sample to xf)
    integral += (x[idx_before_x0 + 1] - x0) * (y0 + y[idx_before_x0 + 1]) / 2
    integral += (xf - x[idx_before_xf]) * (y[idx_before_xf] + yf) / 2

    # Deal with edge case where both x0 and xf are in the same trapezoid
    same_trapezoid = idx_before_x0 == idx_before_xf
    integral[same_trapezoid] = ((xf - x0) * (y0 + yf) / 2)[same_trapezoid]

    return integral


@schema
class Responses(dj.Computed):
    definition = """ # responses recorded during scanning (averaged over the presentation time)
    -> Scan
    """

    class ImageResponses(dj.Part):
        definition = """ # response to a single image
        -> master
        -> Scan.Image
        ---
        response:           blob@brdata     # (num_repeats x num_cells) image responses (repeats ordered by trial_idx)
        blank_response=NULL: blob@brdata    # (num_repeats x num_cells) responses to the blank space before each trial presentation
        """

    def make(self, key):
        # Get all traces for this scan
        print('Getting traces...')
        scan_key = {
            'animal_id': (Scan & key).fetch1('animal_id'),
            'session': (Scan & key).fetch1('session'),
            'scan_idx': (Scan & key).fetch1('scan_idx')}
        traces, unit_ids, trace_times = get_traces(scan_key)

        # Get trial times for all images in scan
        print('Getting onset and offset times for each image...')
        trials_rel = stimulus.Trial * dj.U('condition_hash') * Scan.Image & (Scan & key)
        flip_times, im_classes, im_ids = trials_rel.fetch('flip_times', 'image_class',
                                                          'image_id', squeeze=True,
                                                          order_by='trial_idx')
        if any([len(ft) < 2 or len(ft) > 3 for ft in flip_times]):
            raise ValueError('Only works for frames with 2 or 3 flips')

        # Find start and duration of blank and image frames
        """
        Assumes the trial was a stimulus.Frame condition.
        A single stimulus.Frame is composed of a flip (1/60 secs), a blanking period (0.3 
        - 0.5 secs), another flip, the image (0.5 secs) and another flip. Times in 
        flip_times is the start of each of those flips. During flips (as during blanking) 
        screen is gray so I count the flips before and after the blanking as part of the 
        blanking. There is also another flip after the image and some variable time 
        between trials (t / 60 secs for t > 0, usually t=1) that could be counted as part 
        of the blanking; I ignore those.   
        """
        monitor_fps = 60
        blank_onset = np.stack([ft[0] for ft in flip_times]) - 1 / monitor_fps  # start of blank period
        image_onset = np.stack([ft[1] for ft in flip_times]) + 1 / monitor_fps  # start of image
        blank_duration = image_onset + 1 / monitor_fps - blank_onset
        image_duration = np.stack([ft[2] for ft in flip_times]) - image_onset

        # Add a shift to the onset times to account for the time it takes for the image to
        # travel from the retina to V1
        # Wiskott, L. How does our visual system achieve shift and size invariance?. Problems in Systems Neuroscience, 2003.
        image_onset += 0.04
        blank_onset += 0.04

        # Sample responses (trace by trace) with a rectangular window
        print('Sampling responses (takes some minutes)...')
        image_resps = np.stack([
            trapezoid_integration(tt, t, image_onset, image_onset + image_duration) /
            image_duration for tt, t in zip(trace_times, traces)], axis=-1)
        blank_resps = np.stack([
            trapezoid_integration(tt, t, blank_onset, blank_onset + blank_duration) /
            blank_duration for tt, t in zip(trace_times, traces)], axis=-1)
        image_resps = image_resps.astype(np.float32)
        blank_resps = blank_resps.astype(np.float32)

        # Insert
        print('Inserting...')
        self.insert1(key)
        for im_class, im_id in set(zip(im_classes, im_ids)):
            im_idx = np.logical_and(im_classes == im_class, im_ids == im_id)
            self.ImageResponses.insert1({
                **key, 'image_class': im_class, 'image_id': im_id,
                'response': image_resps[im_idx], 'blank_response': blank_resps[im_idx]})


#TODO: Uncomment ImageSet when needed, I already tested it (and its functions).
# @schema
# class ImageSet(dj.Manual):
#     definition = """ # set of images
#     iset_id:    smallint
#     """
#
#     class Image(dj.Part):
#         definition = """ # one image
#         -> master
#         image_class:    varchar(32)   # type of image presented (same as in stimulus.StaticImage)
#         image_id:       int           # id of this image (same as in stimulus.StaticImage)
#         """
#
#     @staticmethod
#     def fill():
#         imagenet = dj.create_virtual_module('imagenet', 'pipeline_imagenet')
#
#         # Set some parameters
#         iset_id = 1 # id for this image set
#         oracle_class, oracle_collection = 'imagenet', 2 # avoid oracle images from this collection
#         image_class = 'imagenet_v2_gray' # type of images in this image set
#
#         # Find imagenet ids of oracle images (of image_class imagenet and collection 2)
#         oracle_images = imagenet.Album.Oracle & {'image_class': oracle_class,
#                                                  'collection_id': oracle_collection}
#         oracle_ids = dj.U('imagenet_id') & (stimulus.StaticImage.ImageNet & oracle_images)
#
#         # Find all remaining image ids
#         image_ids = (stimulus.StaticImage.ImageNetV2 - oracle_ids).fetch('image_id')
#
#         # Insert
#         ImageSet.insert1({'iset_id': 1})
#         ImageSet.Image.insert([{'iset_id': 1, 'image_class': image_class, 'image_id': i}
#                                for i in image_ids])
#
#     def get_images(self):
#         """ Get all images in this image set (ordered by image id).
#
#         Arguments:
#             self: ImageSet instance restricted to a single entry.
#
#         Returns:
#             An array of images (num_images x height x widht).
#
#         Warning:
#             Output array could be
#         """
#         if len(self) != 1:
#             raise ValueError("Expected use: (ImageSet & {'iset_id': 1}).get_images()")
#
#         # Get images
#         images = (stimulus.StaticImage.Image & (ImageSet.Image & self)).fetch('image',
#                                                                               order_by='image_id')
#         images = np.stack(images)
#
#         return images