import datajoint as dj
import featurevis
import numpy as np
import torch
from scipy import ndimage
from featurevis import models
from featurevis import ops
from neuro_data.static_images import data_schemas

from staticnet_experiments import configs
from staticnet_experiments import models as static_models

from staticnet_analyses import base

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
schema = dj.schema('zhuokun_pilot')

@schema
class BlurMEIParameters(dj.Lookup):
    definition = """  # parameters to generate MEIs
    
    mei_params:         int
    ---
    mei_seed:           int     # random seed used to create the initialization
    use_avg_model:      bool    # use the average model (across different training seeds) to create the MEI
    num_initializations: int    # how many random images to optimize in parallel to create the MEI (output is the average)
    height:             int     # height of the MEI
    width:              int     # width of the MEI 
    contrast:           decimal(5, 3) # contrast to use when generating the MEI
    step_size:          float   # step size to use when generating the MEI
    num_iterations:     int     # number of optimization iterations
    blur_sigma:     decimal(5, 3) # sigma for the gaussian applied to the gradient in each iteration
    """
    contents = [[1, 1234, True, 1, 36, 64, 0.1, 0.1, 1000, 0], 
                [2, 1234, True, 1, 36, 64, 0.1, 0.3, 1000, 1],
                [3, 1234, True, 1, 36, 64, 0.1, 0.4, 1000, 1.5],
                [4, 1234, True, 1, 36, 64, 0.1, 0.6, 1000, 2.5],
                [5, 1234, True, 1, 36, 64, 0.1, 0.15, 1000, 0.25],
                [6, 1234, True, 1, 36, 64, 0.1, 0.2, 1000, 0.5],
                [7, 1234, True, 1, 36, 64, 0.1, 0.25, 1000, 0.75],
                [8, 1234, True, 1, 36, 64, 0.1, 0.35, 1000, 1.25],
                [9, 1234, True, 1, 36, 64, 0.1, 0.45, 1000, 1.75],
                [10, 1234, True, 1, 36, 64, 0.1, 0.5, 1000, 2],
                [11, 1234, True, 1, 36, 64, 0.1, 0.55, 1000, 2.25],
                [12, 1234, True, 1, 36, 64, 0.1, 0.65, 1000, 2.75],
                [13, 1234, True, 1, 36, 64, 0.1, 0.7, 1000, 3],
    ]

@schema
class BlurMEI(dj.Computed):
    definition = """ # a single MEI

    -> static_models.Model
    -> base.Dataset.Unit
    -> BlurMEIParameters
    ---
    mei:                longblob # optimized MEI
    activation:         float   # activation at the MEI 
    """
    @property
    def key_source(self):
        all_keys = static_models.Model * base.Dataset.Unit * BlurMEIParameters
        return all_keys & {'seed': 1009}

    def make(self, key):
        # Get params
        mei_params = (BlurMEIParameters & key).fetch1()

        # Get models
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                   device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Create a random initial image (at desired contrast)
        torch.manual_seed(mei_params['mei_seed'])
        image_shape = (mei_params['num_initializations'], 1, mei_params['height'],
                       mei_params['width'])
        initial_image = torch.randn(image_shape, device='cuda')
        initial_image = initial_image * float(mei_params['contrast'])

        # Optimize
        postup_op = ops.ChangeStd(float(mei_params['contrast']))
        if mei_params['blur_sigma']:
            gradient_f = featurevis.ops.GaussianBlur(float(mei_params['blur_sigma']))
        else:
            gradient_f = None
        mei, fevals, _ = featurevis.gradient_ascent(model, initial_image,
                                                    post_update=postup_op,
                                                    gradient_f = gradient_f,
                                                    step_size=mei_params['step_size'],
                                                    num_iterations=mei_params[
                                                        'num_iterations'])
        mei = mei.mean(0).squeeze().cpu().numpy()
        activation = fevals[-1]

        # Insert
        self.insert1({**key, 'mei': mei, 'activation': activation})

@schema
class BlurMEIEvolve(dj.Computed):
    definition = """ # Temp Table to calculate the activation evolution curve

    -> BlurMEI
    ---
    mei:                longblob # optimized MEI
    activation:         longblob # activation during evolution
    """

    def make(self, key):
        # Get params
        mei_params = (BlurMEIParameters & key).fetch1()

        # Get models
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                   device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Create a random initial image (at desired contrast)
        torch.manual_seed(mei_params['mei_seed'])
        image_shape = (mei_params['num_initializations'], 1, mei_params['height'],
                       mei_params['width'])
        initial_image = torch.randn(image_shape, device='cuda')
        initial_image = initial_image * float(mei_params['contrast'])

        # Optimize
        postup_op = ops.ChangeStd(float(mei_params['contrast']))
        if mei_params['blur_sigma']:
            gradient_f = featurevis.ops.GaussianBlur(float(mei_params['blur_sigma']))
        else:
            gradient_f = None
        mei, fevals, _ = featurevis.gradient_ascent(model, initial_image,
                                                    post_update=postup_op,
                                                    gradient_f = gradient_f,
                                                    step_size=mei_params['step_size'],
                                                    num_iterations=mei_params[
                                                        'num_iterations'])
        mei = mei.mean(0).squeeze().cpu().numpy()
        activation = fevals

        # Insert
        self.insert1({**key, 'mei': mei, 'activation': activation})

@schema
class BlurMEIActThre(dj.Computed):
    definition = """ # threshold BlurMEI by percetage activity
    -> static_models.Model
    -> base.Dataset.Unit
    """
    
    class MEI(dj.Part):
        definition = """
        -> master
        act_threshold                  :decimal(5,3)  # threshold of relative model activity to the maximum model activity, 0 for original MEI
        ---
        -> BlurMEI
        """
    
    @property
    def key_source(self):
        all_keys = static_models.Model * base.Dataset.Unit
        return all_keys & {'seed': 1009}
    
    def make(self, key):
        
        max_activation = dj.U('neuron_id').aggr(BlurMEI & key, max_act='max(activation)').proj(max_act='max_act')
        min_activation = dj.U('neuron_id').aggr(BlurMEI & key, min_act='min(activation)').proj(min_act='min_act')
        neuron_id, max_act, min_act = (max_activation * min_activation).fetch1('neuron_id', 'max_act', 'min_act')
        min_threshold = min_act / max_act
        min_threshold = np.round(min_threshold // .05 * .05, decimals=2)  # round to smaller float with 2 decimal
        thresholds = np.arange(min_threshold, 0.95+1e-14, 0.05)
        self.insert1({**key})
        for t in thresholds:
            thresholded_mei = (BlurMEI & key) & ('activation > %f' % (t*max_act))  # threshold BlurMEI with activation
            mei_key = (thresholded_mei * BlurMEIParameters).fetch('KEY', order_by='blur_sigma DESC', limit=1)  # get the largest blur sigma possible after thresholding
            mei_key = (BlurMEI & mei_key).fetch1('KEY')
            self.MEI.insert1({**mei_key, **key, 'act_threshold':t})
        mei_key = ((BlurMEI & key) * BlurMEIParameters & 'blur_sigma=0').fetch1('KEY')
        self.MEI.insert1({**mei_key, **key, 'act_threshold':0})

@schema
class BlurMaskParameters(dj.Lookup):
    definition = """ # parameters to create an MEI Mask
    
    mask_params:        int
    ---
    zscore_thresh:      float   # threshold used on the absolute normalized image
    closing_iters:      int     # number of dilation/erosion steps performed for binary closing
    gaussian_sigma:     float   # sigma for the gaussian applied to the mask after processing to soften edges
    """
    contents = [[1, 1.5, 2, 1]]


@schema
class BlurMEIMask(dj.Computed):
    definition = """ # finds a mask for an MEI by thresholding the absolute intensity

    -> BlurMEI
    -> BlurMaskParameters
    ---
    mask:               longblob # produced mask
    mask_x:             float    # (px) centroid of the mask in x; (0, 0) is center of image    
    mask_y:             float    # (px) centroid of the mask in y; (0, 0) is center of image
    mask_mean:          float    # mean of MEI inside the mask   
    mask_std:           float    # standard deviation of MEI inside the mask
    """

    def make(self, key):
        from scipy import ndimage
        from skimage import morphology

        # Get mei
        mei = (BlurMEI & key).fetch1('mei')

        # Get params
        params = (BlurMaskParameters & key).fetch1()

        # Normalize and threshold
        norm_mei = (mei - mei.mean()) / mei.std()
        thresholded = np.abs(norm_mei) > params['zscore_thresh']

        # Remove small holes in the thresholded image and connect any stranding pixels
        closed = ndimage.binary_closing(thresholded, iterations=params['closing_iters'])

        # Remove any remaining small objects
        labeled = morphology.label(closed, connectivity=2)
        most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
        oneobject = labeled == most_frequent

        # Create convex hull just to close any remaining holes and so it doesn't look weird
        hull = morphology.convex_hull_image(oneobject)

        # Smooth edges
        smoothed = ndimage.gaussian_filter(hull.astype(np.float32),
                                           sigma=params['gaussian_sigma'])
        mask = smoothed  # final mask

        # Compute mask centroid
        px_y, px_x = (coords.mean() + 0.5 for coords in np.nonzero(hull))
        mask_y, mask_x = px_y - mask.shape[0] / 2, px_x - mask.shape[1] / 2

        # Compute MEI std inside the mask
        mei_mean = (mei * mask).sum() / mask.sum()
        mei_std = np.sqrt((((mei - mei_mean) ** 2) * mask).sum() / mask.sum())

        # Insert
        self.insert1({**key, 'mask': mask, 'mask_x': mask_x, 'mask_y': mask_y,
                      'mask_mean': mei_mean, 'mask_std': mei_std})

def get_latest_gamma_function(rig='2P4'):
    """ Find latest valid MonitorCalibration recording in the given rig (2p4 is meso).
    When the gamma fitting goes wrong during scanning, the luminance values will all be
    the same number so we have to check for that here.
    Arguments:
        rig (string): microscope name. '2P4' for  meso.
    Returns:
        key (dict): Key of the matched calibration
        f (function): Function to go from pixel to luminance.
        f_inv (function): Function to go from luminance to pixel. Inverse of f.
    """
    from scipy import interpolate

    # Get all recorded monitor values
    latest_known_valid = '2019-09-01 00:00:00'  # there is at least one good one from there on
    keys, px_values, luminances = (experiment.MonitorCalibration &
                                   'ts > "{}"'.format(latest_known_valid) &
                                   (experiment.Session & {'rig': rig})).fetch('KEY',
                                                                              'pixel_value',
                                                                              'luminance',
                                                                              order_by='ts DESC')

    # Pick the latest that has recorded luminances
    for key, px_value, luminance in zip(keys, px_values, luminances):
        if np.diff(luminance, axis=-1).mean(-1) > 0.05:
            break

    # Creat f and f_inv
    f = interpolate.interp1d(px_value, luminance, kind='cubic')
    f_inv = interpolate.interp1d(luminance, px_value, kind='cubic')

    return key, f, f_inv

@schema
class DeepDrawMEI(dj.Computed):
    definition = """ # MEI generated with deepdraw method
    -> static_models.Model
    -> base.Dataset.Unit
    -> base.MEIParameters
    ---
    mei:                longblob # optimized MEI
    activation:         float   # activation at the MEI 
    """

    @property
    def key_source(self):
        all_keys = static_models.Model * base.Dataset.Unit * base.MEIParameters
        return all_keys & {'seed': 1009, 'mei_params': 1}

    def make(self, key):
        import featurevis
        from featurevis import utils
        from featurevis import models
        from featurevis import ops

        # Get params
        mei_params = (base.MEIParameters & key).fetch1()

        # Get models
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos')
                       else (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                   device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Create a random initial image (at desired contrast)
        torch.manual_seed(mei_params['mei_seed'])
        image_shape = (mei_params['num_initializations'], 1, mei_params['height'],
                       mei_params['width'])
        initial_image = torch.randn(image_shape, device='cuda')
        initial_image = initial_image

        # Optimize
        lr_decay_factor = (1 / 850 - 1 / 20400) / (1 - 1000) # decays from 1/850 to 1/20400 in 1000 iterations
        walker_gradient = utils.Compose([ops.FourierSmoothing(0.04),  # not exactly the same as fft_smooth(precond=0.1) but close
                                         ops.DivideByMeanOfAbsolute(),
                                         ops.MultiplyBy(1 / 850,
                                                        decay_factor=lr_decay_factor)])
        bias, scale = 111.28329467773438, 60.922306060791016
        blur_decay_factor = (1.5 - 0.01) / (1 - 1000)  # decays from 1.5 to 0.01 in 1000 iterations
        walker_postup = utils.Compose([ops.ClipRange(-bias / scale, (255 - bias) / scale),
                                       ops.GaussianBlur(1.5,
                                                        decay_factor=blur_decay_factor)])
        mei, fevals, reg_values = featurevis.gradient_ascent(model, initial_image,
                                                             step_size=1,
                                                             num_iterations=1000,
                                                             post_update=walker_postup,
                                                             gradient_f=walker_gradient)
        mei = mei.mean(0).squeeze().cpu().numpy()
        activation = fevals[-1]

        # Insert
        self.insert1({**key, 'mei': mei, 'activation': activation})

@schema
class DeepDrawMEIMask(dj.Computed):
    definition = """ # finds a mask for an MEI by thresholding the absolute intensity
    -> DeepDrawMEI
    -> base.MaskParameters
    ---
    mask:               longblob # produced mask
    mask_x:             float    # (px) centroid of the mask in x; (0, 0) is center of image    
    mask_y:             float    # (px) centroid of the mask in y; (0, 0) is center of image
    mask_mean:          float    # mean of MEI inside the mask   
    mask_std:           float    # standard deviation of MEI inside the mask
    """

    def make(self, key):
        from scipy import ndimage
        from skimage import morphology

        # Get mei
        mei = (DeepDrawMEI & key).fetch1('mei')

        # Get params
        params = (base.MaskParameters & key).fetch1()

        # Normalize and threshold
        norm_mei = (mei - mei.mean()) / mei.std()
        thresholded = np.abs(norm_mei) > params['zscore_thresh']

        # Remove small holes in the thresholded image and connect any stranding pixels
        closed = ndimage.binary_closing(thresholded, iterations=params['closing_iters'])

        # Remove any remaining small objects
        labeled = morphology.label(closed, connectivity=2)
        most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
        oneobject = labeled == most_frequent

        # Create convex hull just to close any remaining holes and so it doesn't look weird
        hull = morphology.convex_hull_image(oneobject)

        # Smooth edges
        smoothed = ndimage.gaussian_filter(hull.astype(np.float32),
                                           sigma=params['gaussian_sigma'])
        mask = smoothed  # final mask

        # Compute mask centroid
        px_y, px_x = (coords.mean() + 0.5 for coords in np.nonzero(hull))
        mask_y, mask_x = px_y - mask.shape[0] / 2, px_x - mask.shape[1] / 2

        # Compute MEI std inside the mask
        mei_mean = (mei * mask).sum() / mask.sum()
        mei_std = np.sqrt((((mei - mei_mean) ** 2) * mask).sum() / mask.sum())

        # Insert
        self.insert1({**key, 'mask': mask, 'mask_x': mask_x, 'mask_y': mask_y,
                      'mask_mean': mei_mean, 'mask_std': mei_std})

@schema
class StimulusParameters(dj.Lookup):
    definition = """ # parameters used to produce the images to be shown to the mice
    stim_params: int
    ---
    height:         int         # height of image
    width:          int         # width of image
    mean_lum=NULL:  float       # mean image luminance in cd/m2
    std_lum=NULL:   float       # standard deviation for stimulus images in cd/m2
    background_lum: float       # luminance for the brackground of the masked image in cd/m2
    """
    contents = [{'stim_params': 1, 'height': 144, 'width': 256, 'mean_lum': 10,
                 'std_lum': 7, 'background_lum': 10}]

@schema
class StimulusBlurMEI(dj.Computed):
    definition = """ # BlurMEI processed to be presented to the mice
    -> BlurMEI
    -> StimulusParameters       
    ---
    mean_luminance:     float     # (px) mean intensity inside the mask of original BlurMEI
    std_luminance:      float     # (px) std of intensities inside the mask in original BlurMEI
    masked_px:          longblob  # masked BlurMEI in pixels after standard deviation matching
    unmasked_px:        longblob  # unmasked BlurMEI in pixels after standard deviation matching
    -> experiment.MonitorCalibration  # experiment used to get the monitor calibration
    """

    @property
    def key_source(self):
        return BlurMEI * StimulusParameters & (base.UnitRanking.Unit & 'rank < 150')

    def make(self, key):
        # Get BlurMEI
        mei = (BlurMEI & key).fetch1('mei')
        
        # average mask for meis generated at all different mei params, an estimation for the neuron's receptive field size
        mask = (BlurMEIMask.proj('mask', not_rel='mei_params') & key).fetch('mask').mean()

        # Resize image and mask(and transform into float32)
        target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
        mei = ndimage.zoom(mei.astype(np.float32), target_shape / mei.shape,
                           mode='reflect')
        mask = ndimage.zoom(mask.astype(np.float32), target_shape / mask.shape)

        # Move image to pixel space (i.e., unnormalize it )
        px_mean, px_std = (base.Dataset.TrainStats & key).fetch('mean_img_value',
                                                                'std_img_value')
        px_mei = mei * px_std + px_mean

        # Get gamma function (and inverse)
        gamma_key, f, f_inv = get_latest_gamma_function()

        # Compute image statistics inside the mask (in luminance space)
        lum_mei = f(np.clip(px_mei, 0, 255))
        mei_mean = (lum_mei * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
        mei_std = np.sqrt(np.sum(((lum_mei - mei_mean) ** 2) * mask, axis=(-1, -2),
                                 keepdims=True) / mask.sum())
        mei_mean, mei_std = mei_mean.squeeze(), mei_std.squeeze()

        # Normalize (in luminance space)
        target_mean, target_std = (StimulusParameters & key).fetch1('mean_lum', 'std_lum')
        target_mean = mei_mean if np.isnan(target_mean) else target_mean
        target_std = mei_std if np.isnan(target_std) else target_std
        lum_unmasked = ((lum_mei - mei_mean) / mei_std) * target_std + target_mean
        final_unmasked = f_inv(np.clip(lum_unmasked, f(0), f(255)))

        # Create masked image
        background_lum = (StimulusParameters & key).fetch1('background_lum')
        lum_masked = lum_unmasked * mask + (1 - mask) * background_lum
        final_masked = f_inv(np.clip(lum_masked, f(0), f(255)))
        # final_masked = final_unmasked * mask + (1 - mask) * f_inv(background_lum) #*
        # * commented version applies the mask in pixels rather than luminance space,
        # exactly the same if mask were binary. Pretty similar looking, just be consistent.

        # Make them a valid image
        px_unmasked = np.round(final_unmasked).astype(np.uint8)
        px_masked = np.round(final_masked).astype(np.uint8)

        # Insert
        self.insert1(
            {**key, 'mean_luminance': mei_mean, 'std_luminance': mei_std,
             'unmasked_px': px_unmasked, 'masked_px': px_masked, **gamma_key})


    @staticmethod
    def fill_stimulus(restr={'group_id': 55}, num_desired_neurons=75, num_blur_level=4, experiment='blur_mei_1'):
        """ Fills StaticImage tables (Image, MEI2, BlurMEI) needed to run this stimulus.
        Arguments:
            restr (str): How to restrict StimulusLEI to get the images we want to insert.
            num_desired_images: The number of images to copy over. This is used to check
                the right number of images is copied over
            experiment (str): Name of the experiment. Inserted into MEI2 and BlurMEI; useful
                when running more than one experiment in the same mice.
        """
        # Set some parameters
        mei_table = stimulus.StaticImage.BlurMEI
        mei_class = 'blur_mei'

        # Check that there is enough images processed
        num_images = len(StimulusBlurMEI & restr)

        if num_images != num_desired_neurons * num_blur_level:
            msg = ('Number of images in restricted table ({}) different than desired '
                   'number of images ({})').format(num_images, num_desired_neurons * num_blur_level)
            raise ValueError(msg)

        # Insert in stimulus.StaticImage (if not already there)
        stimulus.StaticImageClass.insert1({'image_class': mei_class},
                                         skip_duplicates=True)
        stimulus.StaticImage.insert1({'image_class': mei_class, 'color': False},
                                    skip_duplicates=True)

        # Fetch images
        mei_keys, meis = (StimulusBlurMEI & restr).fetch('KEY', 'unmasked_px',
                                                           order_by='neuron_id')


        # Insert BlurMEI images
        all_ids = (stimulus.StaticImage.Image & {'image_class': mei_class}).fetch(
            'image_id')
        next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
        for i, (key, img) in enumerate(zip(mei_keys, meis), start=next_id):
            stimulus.StaticImage.Image.insert1({'image_class': mei_class, 'image_id': i,
                                                'image': img})
            mei_table.insert1({'image_class': mei_class, 'image_id': i,
                               'experiment': experiment, **key,
                               'src_table': 'pilot.StimulusBlurMEI'})

@schema
class StimulusBlurMEIThre(dj.Computed):
    definition = """ # BlurMEI processed to be presented to the mice
    -> BlurMEI
    -> StimulusParameters
    -> BlurMEIActThre.MEI       
    ---
    mean_luminance:     float     # (px) mean intensity inside the mask of original BlurMEI
    std_luminance:      float     # (px) std of intensities inside the mask in original BlurMEI
    masked_px:          longblob  # masked BlurMEI in pixels after standard deviation matching
    unmasked_px:        longblob  # unmasked BlurMEI in pixels after standard deviation matching
    -> experiment.MonitorCalibration  # experiment used to get the monitor calibration
    """

    @property
    def key_source(self):
        return BlurMEI * StimulusParameters * BlurMEIActThre.MEI & '`act_threshold` in (0.95, 0.85, 0)'

    def make(self, key):
        # Get BlurMEI
        mei = (BlurMEI & key).fetch1('mei')
        
        # (BlurMEIMask.proj('mask', not_rel='mei_params') & key).fetch('mask').mean()
        mask = (BlurMEIMask & key).fetch1('mask')

        # Resize image and mask(and transform into float32)
        target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
        mei = ndimage.zoom(mei.astype(np.float32), target_shape / mei.shape,
                           mode='reflect')
        mask = ndimage.zoom(mask.astype(np.float32), target_shape / mask.shape)

        # Move image to pixel space (i.e., unnormalize it )
        px_mean, px_std = (base.Dataset.TrainStats & key).fetch('mean_img_value',
                                                                'std_img_value')
        px_mei = mei * px_std + px_mean

        # Get gamma function (and inverse)
        gamma_key, f, f_inv = get_latest_gamma_function()

        # Compute image statistics inside the mask (in luminance space)
        lum_mei = f(np.clip(px_mei, 0, 255))
        mei_mean = (lum_mei * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
        mei_std = np.sqrt(np.sum(((lum_mei - mei_mean) ** 2) * mask, axis=(-1, -2),
                                 keepdims=True) / mask.sum())
        mei_mean, mei_std = mei_mean.squeeze(), mei_std.squeeze()

        # Normalize (in luminance space)
        target_mean, target_std = (StimulusParameters & key).fetch1('mean_lum', 'std_lum')
        target_mean = mei_mean if np.isnan(target_mean) else target_mean
        target_std = mei_std if np.isnan(target_std) else target_std
        lum_unmasked = ((lum_mei - mei_mean) / mei_std) * target_std + target_mean
        final_unmasked = f_inv(np.clip(lum_unmasked, f(0), f(255)))

        # Create masked image
        background_lum = (StimulusParameters & key).fetch1('background_lum')
        lum_masked = lum_unmasked * mask + (1 - mask) * background_lum
        final_masked = f_inv(np.clip(lum_masked, f(0), f(255)))
        # final_masked = final_unmasked * mask + (1 - mask) * f_inv(background_lum) #*
        # * commented version applies the mask in pixels rather than luminance space,
        # exactly the same if mask were binary. Pretty similar looking, just be consistent.

        # Make them a valid image
        px_unmasked = np.round(final_unmasked).astype(np.uint8)
        px_masked = np.round(final_masked).astype(np.uint8)

        # Insert
        self.insert1(
            {**key, 'mean_luminance': mei_mean, 'std_luminance': mei_std,
             'unmasked_px': px_unmasked, 'masked_px': px_masked, **gamma_key})

    @staticmethod
    def fill_stimulus(restr={'group_id': 74}, num_desired_neurons=50, num_thre=3, experiment='blur_mei_act'):
        """ Fills StaticImage tables (Image, MEI2, BlurMEI) needed to run this stimulus.
        Arguments:
            restr (str): How to restrict StimulusLEI to get the images we want to insert.
            num_desired_images: The number of images to copy over. This is used to check
                the right number of images is copied over
            experiment (str): Name of the experiment. Inserted into MEI2 and BlurMEI; useful
                when running more than one experiment in the same mice.
        """
        # Set some parameters
        mei_table = stimulus.StaticImage.BlurMEIActThre
        mei_class = 'blur_mei_act'

        # Check that there is enough images processed
        num_images = len(StimulusBlurMEIThre & restr)

        if num_images != num_desired_neurons * num_thre:
            msg = ('Number of images in restricted table ({}) different than desired '
                   'number of images ({})').format(num_images, num_desired_neurons * num_thre)
            raise ValueError(msg)

        # Insert in stimulus.StaticImage (if not already there)
        stimulus.StaticImageClass.insert1({'image_class': mei_class},
                                         skip_duplicates=True)
        stimulus.StaticImage.insert1({'image_class': mei_class, 'num_channels': 1},
                                    skip_duplicates=True)

        # Fetch images
        mei_keys, meis = (StimulusBlurMEIThre & restr).fetch('KEY', 'unmasked_px',
                                                           order_by='neuron_id')


        # Insert BlurMEI images
        all_ids = (stimulus.StaticImage.Image & {'image_class': mei_class}).fetch(
            'image_id')
        next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
        for i, (key, img) in enumerate(zip(mei_keys, meis), start=next_id):
            stimulus.StaticImage.Image.insert1({'image_class': mei_class, 'image_id': i,
                                                'image': img})
            mei_table.insert1({'image_class': mei_class, 'image_id': i,
                               'experiment': experiment, **key,
                               'src_table': 'blur_mei.StimulusBlurMEIThre'})

@schema
class StimulusDeepDrawMEI(dj.Computed):
    definition = """ # MEI processed to be presented to the mice
    -> DeepDrawMEI
    -> DeepDrawMEIMask
    -> StimulusParameters
    ---
    mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
    std_luminance:      float  # (px) std of intensities inside the mask in original MEI
    masked_px:          longblob  # masked MEI in pixels after standard deviation matching
    unmasked_px:        longblob  # unmasked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibration  # experiment used to get the monitor calibration
    """
    def make(self, key):
        # Get MEI
        mei, mask = (DeepDrawMEI * DeepDrawMEIMask & key).fetch1('mei', 'mask')

        # Resize image and mask(and transform into float32)
        target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
        mei = ndimage.zoom(mei.astype(np.float32),
                           target_shape / mei.shape, mode='reflect')
        mask = ndimage.zoom(mask.astype(np.float32), target_shape / mask.shape)

        # Move image to pixel space (i.e., unnormalize it )
        px_mean, px_std = (base.Dataset.TrainStats & key).fetch('mean_img_value',
                                                                'std_img_value')
        px_mei = mei * (px_std * 0.75) + px_mean # HACK: I only extend to 0.8 std to avoid clipping

        # Get gamma function (and inverse)
        gamma_key, f, f_inv = get_latest_gamma_function()

        # Compute image statistics inside the mask (in luminance space)
        lum_mei = f(np.clip(px_mei, 0, 255))
        mei_mean = (lum_mei * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
        mei_std = np.sqrt(np.sum(((lum_mei - mei_mean) ** 2) * mask, axis=(-1, -2),
                                 keepdims=True) / mask.sum())
        mei_mean, mei_std = mei_mean.squeeze(), mei_std.squeeze()

        # Normalize (in luminance space)
        target_mean, target_std = (StimulusParameters & key).fetch1('mean_lum', 'std_lum')
        target_mean = mei_mean if np.isnan(target_mean) else target_mean
        target_std = mei_std if np.isnan(target_std) else target_std
        lum_unmasked = ((lum_mei - mei_mean) / mei_std) * target_std + target_mean
        final_unmasked = f_inv(np.clip(lum_unmasked, f(0), f(255)))

        # Create masked image
        background_lum = (StimulusParameters & key).fetch1('background_lum')
        lum_masked = lum_unmasked * mask + (1 - mask) * background_lum
        final_masked = f_inv(np.clip(lum_masked, f(0), f(255)))
        # final_masked = final_unmasked * mask + (1 - mask) * f_inv(background_lum) #*
        # * commented version applies the mask in pixels rather than luminance space,
        # exactly the same if mask were binary. Pretty similar looking, just be consistent.

        # Make them a valid image
        px_unmasked = np.round(final_unmasked).astype(np.uint8)
        px_masked = np.round(final_masked).astype(np.uint8)

        # Insert
        self.insert1(
            {**key, 'mean_luminance': mei_mean, 'std_luminance': mei_std,
             'unmasked_px': px_unmasked, 'masked_px': px_masked, **gamma_key})
