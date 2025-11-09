import datajoint as dj
import numpy as np
from scipy import ndimage
import math
from tqdm import tqdm
import torch
from functools import partial
from staticnet_analyses import base
from staticnet_experiments import configs, models as static_models
from neuro_data.static_images import data_schemas, stats
from staticnet_invariance.deis_reconfigured import MEIParameters, MaskStatsParameters, MaskFixedMEI, DEI, DEIGoodRun, DEINatControl, DEIControl, DEINatControlParameters, DEIControlParameters, \
    TextureParameters, Texture, TextureScoreParameters, TextureGoodRun, TextureLookup, LinearImageModel, get_batch, create_whole_mei, load_model, get_variable_masks
from staticnet_invariance.diverse_meis import SimilarityMetric, FeatureSpace
from featurevis import ops, models

imagenet = dj.create_virtual_module('pipeline_imagenet', 'pipeline_imagenet')
stimulus = dj.create_virtual_module('pipeline_stimulus', 'pipeline_stimulus')
experiment = dj.create_virtual_module('pipeline_experiment', 'pipeline_experiment')

dj.config['enable_python_native_blobs'] = True

schema = dj.schema('neurostatic_deis_reconfigured')

# Compute interpolation between pixel and luminance values using monitor calibration scan 
from scipy import interpolate
experiment = dj.create_virtual_module('pipeline_experiment', 'pipeline_experiment')
# MONCALIB_ON_KEY = dict(animal_id=28121, session=3, scan_idx=7)
# MONCALIB_ON_KEY = dict(animal_id=29647, session=13, scan_idx=6)
MONCALIB_ON_KEY = dict(animal_id=31002, session=7, scan_idx=3)
PDCALIB_ON_KEY = dict(rig="2p4", trial=23)
PX, LUM = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('pixel_value', 'luminance')
f = interpolate.interp1d(PX, LUM)
f_inv = interpolate.interp1d(LUM, PX)

def prepare_stimulus(key, image, mask, already_masked=False, component_mask=None):
     # upsample image and mask
    target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
    if mask.sum() == 0:
        image_mean, image_std, pixel_image = 0, 0, (np.ones(target_shape)*128).astype(np.uint8)
    else:
        clip, target_mean, target_std, background, space, mask_image, match_stats, mask_mean_subtraction, restan_already_masked = \
        (StimulusParameters & key).fetch1('clip', 'mean', 'std', 'background', 'match_stats_space', 'mask_image', 'match_stats', 'mask_mean_subtraction', 'restan_already_masked')
        if mask is not None:
            mask = ndimage.zoom(mask.astype(np.float32), target_shape / mask.shape)
        image = ndimage.zoom(image.astype(np.float32), target_shape / image.shape, mode='reflect')
        target_mean = f(128).item() if np.isnan(target_mean) else target_mean
        background = f(128).item() if np.isnan(background) else background
        
        if match_stats is None:
            image = np.clip(image, -clip, clip)
            image = (image + clip) / (clip*2) * 255
            clipped = np.clip(image, 0, 255)
            clipped = np.clip(image, 0, 255)
            lum_image = f(clipped)
            image_mean = (lum_image * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
            image_std = np.sqrt(np.sum(((lum_image - image_mean) ** 2) * mask, axis=(-1, -2),
                                    keepdims=True) / mask.sum())
            image_mean, image_std = image_mean.squeeze(), image_std.squeeze()

            if mask is not None:
                pixel_image = clipped * mask

        elif match_stats == 'mask':
            image = np.clip(image, -clip, clip)
            image = (image + clip) / (clip*2) * 255

            # Compute image statistics inside the mask (in luminance space)
            clipped = np.clip(image, 0, 255)
            lum_image = f(clipped)
            image_mean = (lum_image * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
            image_std = np.sqrt(np.sum(((lum_image - image_mean) ** 2) * mask, axis=(-1, -2),
                                    keepdims=True) / mask.sum())
            image_mean, image_std = image_mean.squeeze(), image_std.squeeze()

            # Match mask std and mean (in luminance space) and create unmasked image
            lum_image = (lum_image - image_mean) / (image_std + 1e-9) * target_std + target_mean

            # Create masked image
            if component_mask is not None:
                component_mask = ndimage.zoom(component_mask.astype(np.float32), target_shape / component_mask.shape)
                mask = component_mask
            if mask_image:
                lum_image = lum_image * mask + (1 - mask) * background

            pixel_image = f_inv(np.clip(lum_image, f(0), f(255)))

        elif match_stats == 'ff': # only match full-field statistics for masked images
            if not already_masked:
                image = ops.standardize_image(image, 0, 0.25, mask, mask_mean_subtraction, True, 'ff')
            else:
                if restan_already_masked:
                    image = ops.standardize_image(image, 0, 0.25, None, False, False, 'ff')
            image = np.clip(image, -clip, clip)
            image = (image + clip) / (clip*2) * 255
        
            # Compute image statistics inside the mask (in luminance space)
            clipped = np.clip(image, 0, 255)
            lum_image = f(clipped)
            image_mean = lum_image.mean()
            image_std = lum_image.std()
            image_mean, image_std = image_mean.squeeze(), image_std.squeeze()
            
            # Match mask std and mean (either in luminance space or in pixel space)
            if space == 'luminance': # match mean and std in luminance space
                lum_image = (lum_image - image_mean) / (image_std + 1e-9) * target_std + target_mean
                pixel_image = f_inv(np.clip(lum_image, f(0), f(255)))
            elif space == 'pixel':
                pixel_image = np.clip((image - image.mean()) / (image.std() + 1e-9) * target_std + target_mean, 0, 255)
            
        # Make them a valid image
        pixel_image = np.round(pixel_image).astype(np.uint8)
    
    return image_mean, image_std, pixel_image

def fill_stimulus_function(source_table, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys):
    # Check that there is enough images processed
    num_images = len(source_table & restr & image_models.proj())

    if num_images != num_desired_images:
        msg = ('Number of images in restricted table ({}) different than desired '
            'number of images ({})').format(num_images, num_desired_images)
        raise ValueError(msg)

    # Insert in stimulus.StaticImage (if not already there)
    stimulus.StaticImageClass.insert([{'image_class': image_class}],
                                    skip_duplicates=True)
    stimulus.StaticImage.insert([{'image_class': image_class}], skip_duplicates=True)

    # Insert images
    all_ids = (stimulus.StaticImage.Image & {'image_class': image_class}).fetch(
        'image_id')
    next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
    for i, (key, img) in enumerate(zip(image_keys, images), start=next_id):
        stimulus.StaticImage.Image.insert1({'image_class': image_class, 'image_id': i,
                                            'image': img})
        image_table.insert1({'image_class': image_class, 'image_id': i,
                        'experiment': experiment, **key,
                        'src_table': '{}.{}'.format(source_table.database, source_table.table_name)}, ignore_extra_fields=True)

from skimage import morphology
from scipy import ndimage

class TukeyLikeComponentMask():
    @staticmethod
    def get_binary_mei_mask(mei, mask_params):
        ## Get MEI binary mask
        # Normalize and threshold
        norm_mei = (mei - mei.mean()) / mei.std()
        thresholded = np.abs(norm_mei) > mask_params['zscore_thresh']
        # Remove small holes in the thresholded image and connect any stranding pixels
        closed = ndimage.binary_closing(thresholded, iterations=mask_params['closing_iters'])
        # Remove any remaining small objects
        labeled = morphology.label(closed, connectivity=2)
        most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
        oneobject = labeled == most_frequent
        # Create convex hull just to close any remaining holes and so it doesn't look weird
        hull = morphology.convex_hull_image(oneobject)
        return hull
    
    @staticmethod
    def get_binary_component_mask(deis, threshold, mei_binary_mask, mask_params, variable=True):
        img = deis.std(axis=0)
        img = img/img.max()
        thresholded = img > threshold
        # Remove small holes in the thresholded image and connect any stranding pixels
        closed = ndimage.binary_closing(thresholded, iterations=mask_params['closing_iters'])
        # Remove any remaining small objects
        labeled = morphology.label(closed, connectivity=2)
        most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
        oneobject = labeled == most_frequent
        overlapped = (oneobject & mei_binary_mask).astype(np.float32)
        if not variable:
            overlapped = mei_binary_mask - overlapped
        return overlapped
    
    @staticmethod
    def get_boundary(label,parameters={'method':'8-connected'}):
        from scipy.ndimage import binary_erosion
        if parameters['method'] == '8-connected':
            k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1
        else:
            k = np.ones((3,3),dtype=int)
        boundary = label-binary_erosion(label,k)
        return boundary

    def soften_edge(self, binary_mask, mask_params):
        sm_m = ndimage.gaussian_filter(binary_mask, sigma=mask_params['gaussian_sigma'])
        border = self.get_boundary(binary_mask)
        if border.sum() > 0:
            # c = border.sum() / sm_mv[np.where(border)].sum()
            c = 1 / sm_m[np.where(border)].min()
            final_m = np.clip(sm_m * c, 0, 1)
        else:
            final_m = np.zeros_like(border)
        return final_m
    
    def __call__(self, key, dei_std_threshold, sigma=1.0):
        mei, deis = (base.MEI * DEI & key).fetch1('mei', 'deis')
        mask_params = (base.MaskParameters & key).fetch1()
        mask_params['gaussian_sigma'] = sigma
        v_mask = (Texture & key).fetch1('variable_mask')
        mei_binary_mask = self.get_binary_mei_mask(mei, mask_params)
        mv_binary = self.get_binary_component_mask(deis, dei_std_threshold, mei_binary_mask, mask_params)
        mv = self.soften_edge(mv_binary, mask_params)
        mf_binary = self.get_binary_component_mask(deis, dei_std_threshold, mei_binary_mask, mask_params, variable=False)
        mf = self.soften_edge(mf_binary, mask_params)
        return mv, mf

@schema
class StimulusParameters(dj.Lookup):
    definition = """ # parameters used to produce the images to be shown to the mice
    stim_params: int
    ---
    height:                 int         # height of image
    width:                  int         # width of image
    clip:                   float       # range of clipping around 0
    mean=NULL:              float       # mean for stimulus images
    std:                    float       # standard deviation for stimulus images 
    background=NULL:        float       # background value for the masked images
    mask_image:             bool        # whether to mask the image with MEI mask
    match_stats:            varchar(45) # where the target luminance stats are set for, can be 'mask' or 'ff'
    match_stats_space:      varchar(16) # which space where statistics of stimulus images is matched, 'pixel' or 'luminance'
    mask_mean_subtraction:  bool        # whether to subtract mask mean or not during standardization
    restan_already_masked:  bool        # Whether to restandardize images that are already masked (since stats could change slightly due to zoom operation)
    component_mask_type:    varchar(16) # Type of component used, options include 'original', 'tukey-like', etc.
    """
    # contents = [(-1, 144, 256, 4., 128.  , 12.  , 128.  , 1, 'ff', 'pixel', 1, 1),
    #             ( 1, 144, 256, 4.,   2.36,  1.  ,   2.36, 1, 'mask', 'luminance', 1, 0),
    #             ( 2, 144, 256, 4.,   2.36,  0.3 ,   2.36, 1, 'ff', 'luminance', 0, 0),
    #             ( 3, 144, 256, 3.,   2.21,  0.4 ,   2.21, 1, 'ff', 'luminance', 0, 0),
    #             ( 4, 144, 256, 3.,   2.21,  0.3 ,   2.21, 1, 'ff', 'luminance', 0, 0),
    #             ( 5, 144, 256, 4.,   2.21,  0.4 ,   2.21, 1, 'ff', 'luminance', 0, 0),
    #             ( 6, 144, 256, 3.,   2.21,  0.35,   2.21, 1, 'ff', 'luminance', 0, 0),
    #             ( 7, 144, 256, 4.,   2.21,  0.35,   2.21, 1, 'ff', 'luminance', 1, 1),
    #             ( 8, 144, 256, 4.,    nan,  0.35,    nan, 1, 'ff', 'luminance', 1, 1),
    #             ( 9, 144, 256, 4.,    nan,  1.  ,    nan, 0, 'mask', 'luminance', 1, 0),
    #             (10, 144, 256, 4.,    nan,  0.25,    nan, 1, 'ff', 'luminance', 1, 1),
    #             (11, 144, 256, 4.,    nan,  0.3 ,    nan, 1, 'ff', 'luminance', 1, 1)]

####################################### Stimulus tables ##########################################################
@schema
class StimulusMEI(dj.Computed):
    definition = """ # the original single MEI processed to be presented to the mice
    -> MaskFixedMEI
    -> StimulusParameters
    ---
    mean_luminance:                          float     #  mean intensity inside the mask of original MEI
    std_luminance:                           float     #  std of intensities inside the mask in original MEI
    pixel_image:                             longblob  # masked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibrationFromH5             # experiment used to get the monitor calibration
    ts:                                      timestamp
    """
    
    @property
    def key_source(self):
        return MaskFixedMEI * StimulusParameters

    def make(self, key):
        mask_stats_params = (MaskStatsParameters & key).fetch1()
        if int(mask_stats_params['fixed_mask_std']) & int(mask_stats_params['fixed_mask_mean']):
            mei, mask = (MaskFixedMEI * base.MEIMask & key).fetch1('mei', 'mask')
        else:
            mei, mask = (base.MEI * base.MEIMask & key).fetch1('mei', 'mask')

        image_mean, image_std, pixel_image = prepare_stimulus(key, mei, mask)
        # Hacking - needs to implement finding most recent calibration
        gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
        self.insert1({**key, 'mean_luminance': image_mean, 'std_luminance': image_std,
                     'pixel_image': pixel_image, **gamma_key})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='dei'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.MaskFixedMEI
        image_class = 'mask_fixed_mei'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
       
        # Fetch images
        images, image_keys = (self & restr & image_models).fetch('pixel_image', 'KEY', order_by='neuron_id')
        fill_stimulus_function(self, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class StimulusDEI(dj.Computed):
    definition = """ # the DEIs processed to be presented to the mice
    -> DEI
    -> StimulusParameters
    """

    class DEI(dj.Part):
        definition = """
        -> master
        dei_id:                                     int
        ---
        mean_luminance:                             float       #  mean intensity inside the mask of original MEI
        std_luminance:                              float       #  std of intensities inside the mask in original MEI
        pixel_image:                                longblob    # masked MEI in pixels after standard deviation matching
        -> experiment.MonitorCalibrationFromH5                  # experiment used to get the monitor calibration
        ts:                                         timestamp
        """

    @property
    def key_source(self):
        return DEI * StimulusParameters & DEIGoodRun

    def make(self, key):
        self.insert1(key)
        deis, mask = (DEI * base.MEIMask & key).fetch1('deis', 'mask')
        gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
        for i, dei in enumerate(deis):
            image_mean, image_std, pixel_image = prepare_stimulus(key, dei, mask)
            self.DEI.insert1({**key, 'dei_id': i, 'mean_luminance': image_mean, 'std_luminance': image_std,
                        'pixel_image': pixel_image, **gamma_key})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='dei'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.MaskedDEI
        image_class = 'masked_dei'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
       
        # Fetch images
        images, image_keys = (self.DEI & restr & image_models).fetch('pixel_image', 'KEY', order_by='neuron_id')
        fill_stimulus_function(self.DEI, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class StimulusDEIControl(dj.Computed):
    definition = """ # the DEI synthesized controls processed to be presented to the mice
    -> DEIControl.Image
    -> StimulusParameters
    ---
    mean_luminance:                         float       #  mean intensity inside the mask of original MEI
    std_luminance:                          float       #  std of intensities inside the mask in original MEI
    pixel_image:                            longblob    # masked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibrationFromH5              # experiment used to get the monitor calibration
    ts:                                     timestamp
    """

    @property
    def key_source(self):
        return DEIControl.Image * StimulusParameters

    def make(self, key):
        image, mask = (DEIControl.Image * base.MEIMask & key).fetch1('image', 'mask')
        match_stats = (DEIControlParameters & key).fetch1('match_stats')
        if match_stats == 'mask':
            image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask)
        elif match_stats == 'ff':
            image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask, already_masked=True) 

        gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
        self.insert1({**key, 'mean_luminance': image_mean, 'std_luminance': image_std,
                     'pixel_image': pixel_image, **gamma_key})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='dei'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.MaskedDEIControl
        image_class = 'masked_control'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
       
        # Fetch images
        images, image_keys = (self & restr & image_models).fetch('pixel_image', 'KEY', order_by='neuron_id')
        for key in image_keys:
            key['control_image_id'] = key.pop('image_id')
        fill_stimulus_function(self, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class StimulusDEINatControl(dj.Computed):
    definition = """ # the DEI natural controls processed to be presented to the mice
    -> DEINatControl.Image
    -> StimulusParameters
    ---
    mean_luminance:                         float           #  mean intensity inside the mask of original MEI
    std_luminance:                          float           #  std of intensities inside the mask in original MEI
    pixel_image:                            longblob        # masked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibrationFromH5                  # experiment used to get the monitor calibration
    ts:                                     timestamp
    """

    @property
    def key_source(self):
        return DEINatControl.Image * StimulusParameters 

    def make(self, key):
        image, mask = (DEINatControl.Image * base.MEIMask & key).fetch1('image', 'mask')
        match_stats = (DEINatControlParameters & key).fetch1('match_stats')
        if match_stats == 'mask':
            image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask)
        elif match_stats == 'ff':
            image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask, already_masked=True)

        gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
        self.insert1({**key, 'mean_luminance': image_mean, 'std_luminance': image_std,
                     'pixel_image': pixel_image, **gamma_key})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='dei'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.MaskedDEINatControl
        image_class = 'masked_nat_control'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
       
        # Fetch images
        images, image_keys = (self & restr & image_models).fetch('pixel_image', 'KEY', order_by='neuron_id')
        for key in image_keys:
            key['control_image_id'] = key.pop('image_id')
        fill_stimulus_function(self, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class StimulusFullTextureDEI(dj.Computed):
    definition = """ # the full texture DEIs processed to be presented to the mice
    -> TextureLookup
    -> StimulusParameters
    """

    class DEI(dj.Part):
        definition = """
        -> master
        dei_id:                                 int
        ---
        group_id: smallint                                      # index of group
        net_hash: varchar(256)                                  # unique identifier for configuration
        seed:                                   int             # random seed
        data_hash:                              varchar(256)    # unique identifier for configuration
        readout_key:                            varchar(50)      
        neuron_id:                              int             # id of this cell in the dataset (starts at 0), same as index of responses
        mei_params:                             int              
        mask_params:                            int             
        mask_stats_params:                      int              
        diverse_params:                         int              
        weight_id:                              int             
        ref_id:                                 int              
        threshold_params:                       int              
        texture_params:                         int             
        texture_id:                             int             # Id of the run
        mean_luminance:                         float           #  mean intensity inside the mask of original MEI
        std_luminance:                          float           #  std of intensities inside the mask in original MEI
        pixel_image:                            longblob        # masked MEI in pixels after standard deviation matching
        -> experiment.MonitorCalibrationFromH5                  # experiment used to get the monitor calibration
        ts:                                     timestamp
        """

    @property
    def key_source(self):
        return TextureLookup * StimulusParameters & 'texture_id = -1'

    def make(self, key):
        self.insert1(key)
        neuron_key = (TextureLookup * StimulusParameters.proj() & key).fetch1()
        samples, mask = (Texture * base.MEIMask & neuron_key).fetch1('samples', 'mask')
        gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
        for i, sample in enumerate(samples):
            image_mean, image_std, pixel_image = prepare_stimulus(neuron_key, sample, mask)
            self.DEI.insert1({**neuron_key, 'dei_id': i, 'mean_luminance': image_mean, 'std_luminance': image_std,
                        'pixel_image': pixel_image, **gamma_key})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='dei'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.MaskedFullTextureDEI
        image_class = 'masked_full_texture_dei'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
       
        # Fetch images
        images = (self.DEI & restr & image_models.proj()).fetch('pixel_image', order_by='synthesis_id, dei_id')
        image_keys = (self.DEI & restr & image_models.proj()).fetch(as_dict=True, order_by='synthesis_id, dei_id')
        fill_stimulus_function(self.DEI, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class StimulusLocalTextureDEI(dj.Computed):
    definition = """ # the full texture DEIs processed to be presented to the mice
    -> TextureLookup
    -> StimulusParameters
    """

    class DEI(dj.Part):
        definition = """
        -> master
        dei_id:             int
        ---
        group_id             : smallint                     # index of group
        net_hash             : varchar(256)                 # unique identifier for configuration
        seed                 : int                          # random seed
        data_hash            : varchar(256)                 # unique identifier for configuration
        readout_key          : varchar(50)                  # 
        neuron_id            : int                          # id of this cell in the dataset (starts at 0), same as index of responses
        mei_params           : int                          # 
        mask_params          : int                          # 
        mask_stats_params    : int                          # 
        diverse_params       : int                          # 
        weight_id            : int                          # 
        ref_id               : int                          # 
        threshold_params     : int                          # 
        texture_params       : int                          # 
        texture_id           : int                          # Id of the run
        mean_luminance:     float  #  mean intensity inside the mask of original MEI
        std_luminance:      float  #  std of intensities inside the mask in original MEI
        pixel_image:          longblob  # masked MEI in pixels after standard deviation matching
        -> experiment.MonitorCalibrationFromH5  # experiment used to get the monitor calibration
        ts:                 timestamp
        """

    @property
    def key_source(self):
        return TextureLookup * StimulusParameters & (Texture & (TextureGoodRun & 'score_params = 2'))

    def make(self, key):
        self.insert1(key)
        neuron_key = (TextureLookup * StimulusParameters.proj() & key).fetch1()
        samples, mask = (Texture * base.MEIMask & neuron_key).fetch1('samples', 'mask')
        for i, sample in enumerate(samples):
            image_mean, image_std, pixel_image = prepare_stimulus(neuron_key, sample, mask)
            # Hacking - needs to implement finding most recent calibration
            gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
            self.DEI.insert1({**neuron_key, 'dei_id': i, 'mean_luminance': image_mean, 'std_luminance': image_std,
                        'pixel_image': pixel_image, **gamma_key})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='dei'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.MaskedLocalTextureDEI
        image_class = 'masked_local_texture_dei'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
       
        # Fetch images
        images = (self.DEI & restr & image_models.proj()).fetch('pixel_image', order_by='synthesis_id, dei_id')
        image_keys = (self.DEI & restr & image_models.proj()).fetch(as_dict=True, order_by='synthesis_id, dei_id')

        fill_stimulus_function(self.DEI, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class SimilarityEnsembleParameters(dj.Lookup):
    definition = """
    ensemble_params:      int
    ---
    -> SimilarityMetric
    -> FeatureSpace
    sample_criterion:     varchar(16)       # method of selecting samples from the optimized texture, 'random', 'closest', or 'diverse'
    n_per_class:          int
    selection_seed:       int
    gamma_key:            longblobg         # moncalib scan key for measuring pixel to luminance space conversion, same key as in experiment.MonitorCalibrationFromH5
    """

    class ImageClass(dj.Part):
        definition = """
        -> master
        class_id:         int
        --- 
        image_class:      varchar(32)
        description:      varchar(256)
        """
# dei_stimulus.SimilarityEnsembleParameters().insert1([1, 'correlation', 'population_resps', 'closest', 1, 1234])
# dei_stimulus.SimilarityEnsembleParameters.ImageClass.insert([[1, 1, 'DEI', ''], 
#                                                             [1, 2, 'local texture DEI', ''],
#                                                             [1, 3, 'full texture DEI', ''],
#                                                             [1, 4, 'fixed part', 'fixed part of local texture DEI, directly masked by (1-mask_v) on final stimuli'],
#                                                             [1, 5, 'random DEI', 'DEI from a random neurons in the same dataset, shifted to the mask center of the target neuron'],])

def select_texture_dei(key, seed, n_sample=1000, batch_size=64, device='cuda'):
    v_mask, eval_texture = (Texture & key).fetch1('variable_mask','eval_texture')
    mei = (base.MEI & key).fetch1('mei')
    torch.manual_seed(seed)
    image_model = LinearImageModel(mei,v_mask, eval_texture = eval_texture).to(device)
    image_model.initialize_t()
    n_batch = math.floor(n_sample / batch_size)
    redundant = n_sample - batch_size * n_batch
    texture_deis = []
    with torch.no_grad():
        for i in range(n_batch):
            texture_deis.append(image_model(batch_size, key='eval').cpu().detach().numpy().squeeze())
        if redundant: texture_deis.append(image_model(redundant, key='eval').cpu().detach().numpy().squeeze())
    return np.vstack(texture_deis)

@schema
class StimulusSimilarityEnsemble(dj.Computed):
    definition = """ # 
    -> TextureLookup
    -> StimulusParameters
    -> SimilarityEnsembleParameters
    """
    class Image(dj.Part):
        definition = """
        -> master
        image_id:                                   int
        ---
        dei_id:                                     int
        image_class:                                varchar(32)
        source_neuron_id:                           int
        mean_luminance:                             float       #  mean intensity inside the mask of original MEI
        std_luminance:                              float       #  std of intensities inside the mask in original MEI
        pixel_image:                                longblob    # masked MEI in pixels after standard deviation matching
        -> experiment.MonitorCalibrationFromH5                  # experiment used to get the monitor calibration
        ts:                                         timestamp
        """

    @property
    def key_source(self):
        return TextureLookup * StimulusParameters.proj() * SimilarityEnsembleParameters.proj() & (Texture * TextureGoodRun & 'score_params = 2')

    def make(self, key):
        self.insert1(key)

        key = (self.key_source & key).fetch1()
        # Get parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble_params = (SimilarityEnsembleParameters & key).fetch1()
        class_ids, image_classes = (SimilarityEnsembleParameters.ImageClass & key & 'class_id > 1').fetch('class_id', 'image_class', order_by='class_id')
        deis, mask, mask_x, mask_y = (DEI * base.MEIMask & key).fetch1('deis', 'mask', 'mask_x', 'mask_y')
        mei_params = (MEIParameters & key).fetch1()
        gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
        assert (len(DEI & key) == 1), 'Found more than one DEI stimulus for a unique neuron key!'

        tuples = []
        # DEI
        np.random.seed(ensemble_params['selection_seed'])
        dei_id = np.random.choice(np.arange(len(deis)), 1).item()
        dei = deis[dei_id]
        image_mean, image_std, pixel_image = prepare_stimulus(key, dei, mask)
        tuples.append({**key, 'image_id':1, 'image_class': 'dei', 'dei_id': dei_id, 'source_neuron_id': key['neuron_id'], 'mean_luminance': image_mean, 'std_luminance': image_std, 'pixel_image': pixel_image, **gamma_key}) 
        dei_tensor = torch.tensor(dei[None, None], dtype=torch.float32, device=device)

        # Get function to calculate similarity to DEI in certain feature space
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                    mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                    (Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        if ensemble_params['features'] == 'pixels':
            embedding = ops.Identity()  # operation that returns x as is
            similarity_mask = torch.tensor(mask[None, None], dtype=torch.float32, device=device)
        elif ensemble_params['features'] == 'feature_vectors':
            # return a feature map matrix in the shape of batch_size x (num_models x feature_vec_length)
            embedding = ops.Feature_Vector_Ensemble(all_models, key['readout_key'], neuron_idx=key['neuron_id'], eye_pos=mean_eyepos, device=device, average_batch=False)
            similarity_mask = None
        elif ensemble_params['features'] == 'single_grid_population_resps':
            embedding = ops.SingleGridResps(all_models, key['readout_key'], eye_pos=mean_eyepos, neuron_idx=key['neuron_id'], all_neurons=True, average_batch=False, device=device)
            similarity_mask = None
        elif ensemble_params['features'] == 'population_resps':
            embedding = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos, device=device, average_batch=False)
            similarity_mask = None

        else:
            raise NotImplementedError('{} feature embedding not implemented'.format(ensemble_params['features']))
        get_sim = partial(DEI.div_regularization, similarity_mask, ensemble_params['similarity'], 1, ops.DoNothing(), embedding, dei_tensor)

        for class_id, image_class in zip(class_ids, image_classes): 
            if image_class == 'local_texture_dei':
                sims = []
                
                all_images = select_texture_dei(key, ensemble_params['selection_seed'])
                for images in tqdm(get_batch(all_images, 20)):
                    images = ops.standardize_image(images, 0, 0.25, mask, 1, 1, 'ff')
                    images = torch.as_tensor(images[:, None], dtype=torch.float32, device='cuda')
                    with torch.no_grad():
                        sims.append(get_sim(images)[:len(images)].cpu().detach().squeeze().numpy())
                image = all_images[np.argmax(sims).item()]
                image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask)
                tuples.append({**key, 'image_id':class_id, 'image_class': image_class, 'dei_id': dei_id, 'source_neuron_id': key['neuron_id'], 'mean_luminance': image_mean, 'std_luminance': image_std, 'pixel_image': pixel_image, **gamma_key}) 
                
            elif image_class == 'full_texture_dei':
                key_copy = key.copy()
                key_copy['texture_id'] = -1
                sims = []
                all_images = select_texture_dei(key_copy, ensemble_params['selection_seed'])
                for images in tqdm(get_batch(all_images, 20)):
                    images = ops.standardize_image(images, 0, 0.25, mask, 1, 1, 'ff')
                    images = torch.as_tensor(images[:, None], dtype=torch.float32, device='cuda')
                    with torch.no_grad():
                        sims.append(get_sim(images)[:len(images)].cpu().detach().squeeze().numpy())
                image = all_images[np.argmax(sims).item()]
                image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask)
                tuples.append({**key, 'image_id':class_id, 'image_class': image_class, 'dei_id': dei_id, 'source_neuron_id': key['neuron_id'], 'mean_luminance': image_mean, 'std_luminance': image_std, 'pixel_image': pixel_image, **gamma_key}) 
                
            elif image_class == 'fixed_part':
                mv = (Texture & key).fetch1('variable_mask')
                target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
                mv = ndimage.zoom(mv.astype(np.float32), target_shape / mv.shape)
                pixel_image = tuples[1]['pixel_image'] * (1 - mv) + mv * 128
                pixel_image = np.round(pixel_image).astype(np.uint8)    
                tuples.append({**key, 'image_id':class_id, 'image_class': image_class, 'dei_id': dei_id, 'source_neuron_id': key['neuron_id'], 'mean_luminance': image_mean, 'std_luminance': image_std, 'pixel_image': pixel_image, **gamma_key}) 

            elif image_class == 'random_dei':
                key_copy = key.copy()
                key_copy.pop('neuron_id')
                key_copy.pop('weight_id')
                neuron_rel = DEI * base.MEIMask & key_copy & (TextureLookup & (Texture & TextureGoodRun)) & 'neuron_id != {}'.format(key['neuron_id'])
                src_neurons = neuron_rel.fetch('neuron_id', order_by='neuron_id')
                np.random.seed(key['neuron_id'])
                src_neuron_id = np.random.choice(src_neurons, 1).item()
                src_deis, src_mask, src_mask_x, src_mask_y = (neuron_rel & {'neuron_id': src_neuron_id}).fetch1('deis', 'mask', 'mask_x', 'mask_y')
                np.random.seed(key['neuron_id'] + ensemble_params['selection_seed'])
                src_dei_id = np.random.choice(np.arange(len(src_deis)), 1).item()
                src_dei = src_deis[src_dei_id]
                center_dei = ops.center_and_crop_image(src_dei, src_mask_x, src_mask_y, (32, 32))
                src_dei = create_whole_mei(center_dei, mask, mask_x, mask_y, (36, 64), normalize_crop=False)
                center_mask = ops.center_and_crop_image(src_mask, src_mask_x, src_mask_y, (32, 32))
                src_mask = create_whole_mei(center_mask, mask, mask_x, mask_y, (36, 64), normalize_crop=False)
                image_mean, image_std, pixel_image = prepare_stimulus(key, src_dei, src_mask)
                tuples.append({**key, 'image_id':class_id, 'image_class': image_class, 'dei_id': dei_id, 'source_neuron_id': src_neuron_id, 'mean_luminance': image_mean, 'std_luminance': image_std, 'pixel_image': pixel_image, **gamma_key}) 
        
        for t in tuples:
            self.Image.insert1(t, ignore_extra_fields=True)
    
    def fill_stimulus(self, restr, num_desired_images, experiment='similarity_ensemble_1'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace

        # Set some parameters
        if restr['image_class'] == 'dei':
            image_table = stimulus.StaticImage.MaskedDEI
            image_class = 'masked_dei'
        elif restr['image_class'] == 'local_texture_dei':
            image_table = stimulus.StaticImage.MaskedLocalTextureDEI
            image_class = 'masked_local_texture_dei'
        elif restr['image_class'] == 'full_texture_dei':
            image_table = stimulus.StaticImage.MaskedFullTextureDEI
            image_class = 'masked_full_texture_dei'
        elif restr['image_class'] == 'fixed_part':
            image_table = stimulus.StaticImage.MaskedDEIFixedPart
            image_class = 'masked_local_texture_dei_fixed_part'
        elif restr['image_class'] == 'random_dei':
            image_table = stimulus.StaticImage.MaskedRandomDEI
            image_class = 'masked_random_dei'
        image_keys = (TextureLookup * self.Image * SimilarityEnsembleAlbum & restr).fetch(as_dict=True, order_by='group_id, neuron_id')
        for key in image_keys:
            key.pop('image_id')
            key.pop('image_class')
            key['dei_id'] = 0
        images = (TextureLookup * self.Image * SimilarityEnsembleAlbum & restr).fetch('pixel_image', order_by='group_id, neuron_id')
        restr = (self.Image * SimilarityEnsembleAlbum & restr).proj()
        fill_stimulus_function(self.Image, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class SimilarityEnsembleAlbum(dj.Lookup):
    definition = """
    album_id:      int
    -> StimulusSimilarityEnsemble
    """

    def fill(self, album_id=1, n_ensembles=50, seed=1234, rest={}):
        exclude = StimulusSimilarityEnsemble & SimilarityEnsembleAlbum
        ensembles = ((StimulusSimilarityEnsemble - exclude) & rest).fetch(as_dict=True, order_by='synthesis_id')
        np.random.seed(seed)
        idxs = np.random.choice(np.arange(len(ensembles)), n_ensembles, replace=False)
        selected = np.array(ensembles)[idxs]
        self.insert([{**s, 'album_id': album_id} for s in selected])


### NOTE: StimulusMEIComponent for group_id=272 was wrong, mistakenly used target_frac_std as threshold on dei std image!!
@schema
class StimulusMEIComponent(dj.Computed):
    definition = """
    -> base.MEIMask
    -> TextureLookup
    -> StimulusParameters
    ---
    mei:                                        longblob
    variable:                                   longblob
    fixed:                                      longblob
    mean_luminance:                             float       # mean intensity of MEI inside the MEI mask
    std_luminance:                              float       # std of intensity of MEI inside the MEI mask
    -> experiment.MonitorCalibrationFromH5        # experiment used to get the monitor calibration
    ts:                                         timestamp
    """

    @property
    def key_source(self):
        return base.MEIMask.proj() * TextureLookup * StimulusParameters.proj() & (Texture * TextureGoodRun & 'score_params = 2')
    
    def make(self, key):
        key = (self.key_source & key).fetch1()

        # compute threshold applied on dei std image
        mei, mask = (base.MEI * base.MEIMask & key).fetch1('mei', 'mask')
        texture_parameters = (TextureParameters * TextureParameters.TargetFractionStd & key).fetch1()
        mask_params = (base.MaskParameters & key).fetch1()
        deis = (DEI * DEIGoodRun & key).fetch1('deis')
        deis = np.stack(deis)
        threshold = get_variable_masks(deis,mei,mask_params,values=[texture_parameters['target_fraction_std']],
                                            params={'closing_iters':texture_parameters['closing_iters'],
                                                    'gaussian_sigma':texture_parameters['gaussian_sigma']})[1][0]

        height, width, component_mask_type, match_stats = (StimulusParameters & key).fetch1('height', 'width', 'component_mask_type', 'match_stats')
        target_shape = np.array([height, width])
        mei, mask = (base.MEI * base.MEIMask & key).fetch1('mei', 'mask')
        v_mask = (Texture & key).fetch1('variable_mask')
        gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
        
        image_mean, image_std, mei_final = prepare_stimulus(key, mei, mask)
        if component_mask_type == 'original':
            assert match_stats == 'mask', 'Soft component mask must be applied on unmasked images created with match_stats = "ff" to avoid double masking!'
            _, _, variable = prepare_stimulus(key, mei, mask, component_mask=v_mask)
            _, _, fixed = prepare_stimulus(key, mei, mask, component_mask=mask-v_mask)
        elif component_mask_type == 'tukey-like': 
            mv, mf = TukeyLikeComponentMask()(key, threshold)
            mv = np.clip(ndimage.zoom(mv.astype(np.float32), target_shape / mv.shape), 0, 1)
            mf = np.clip(ndimage.zoom(mf.astype(np.float32), target_shape / mf.shape), 0, 1)
            variable = np.round(mv * mei_final + (1-mv) * 128).astype(np.uint8)
            fixed = np.round(mf * mei_final + (1-mf) * 128).astype(np.uint8)
        else:
            raise NotImplementedError('Masking method for component mask type {} not implemented!'.format(component_mask_type))

        self.insert1({**key, 'mean_luminance': image_mean, 'std_luminance': image_std,
                'mei': mei_final, 'variable': variable, 'fixed': fixed, **gamma_key}, ignore_extra_fields=True)

    def fill_stimulus(self, restr, num_desired_images, experiment='two_component_necessity'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Fetch images and set some parameters
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
        meis, variables, fixeds, image_keys = (self & restr & image_models).fetch('mei', 'variable', 'fixed', 'KEY', order_by='neuron_id')
        image_tables = [stimulus.StaticImage.MaskFixedMEI, stimulus.StaticImage.MEIVariableComponent, stimulus.StaticImage.MEIFixedComponent]
        image_classes = ['mask_fixed_mei', 'mei_variable', 'mei_fixed']

        # Fill stimulus
        for images, image_table, image_class in zip([meis, variables, fixeds], image_tables, image_classes):
            fill_stimulus_function(self, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)
    
@schema
class TwoComponentSwapParameters(dj.Lookup):
    definition = """
    control_params:    int
    ---
    n_swaps:           int           # number of images for each swap type
    selection_seed:    int
    v_swap_type:       varchar(16)   # image type for swapping for the variable component
    f_swap_type:       varchar(16)   # image type for swapping for the fixed component
    """
    contents = [[1, 20, 1234, 'other_texture', 'natural']]
    
@schema
class StimulusTwoComponentSwap(dj.Computed):
    definition = """
    -> TextureLookup
    -> TextureScoreParameters
    -> StimulusParameters
    -> TwoComponentSwapParameters
    """
    
    class LocalTextureDEI(dj.Part):
        definition = """
        -> master
        image_id:                                   int
        ---
        mean_luminance:                             float       # mean intensity inside the MEI mask
        std_luminance:                              float       # std of intensities inside the MEI mask
        pixel_image:                                longblob    # final stimulus image in pixels
        -> experiment.MonitorCalibrationFromH5                  # experiment used to get the monitor calibration
        ts:                                         timestamp
        """
    
    class VariableSwap(dj.Part):
        definition = """
        -> master
        image_id:                                   int
        ---
        swap_neuron_id:                             int         # neuron id of the swapped texture crop
        mean_luminance:                             float       # mean intensity inside the MEI mask
        std_luminance:                              float       # std of intensities inside the MEI mask
        pixel_image:                                longblob    # final stimulus image in pixels
        -> experiment.MonitorCalibrationFromH5                  # experiment used to get the monitor calibration
        ts:                                         timestamp
        """
        
    class FixedSwap(dj.Part):
        definition = """
        -> master
        image_id:                                   int
        ---
        swap_image_class:                           varchar(16) # image class of the swapped crop
        swap_image_id:                              int         # image id of the swapped crop
        mean_luminance:                             float       # mean intensity inside the MEI mask
        std_luminance:                              float       # std of intensities inside the MEI mask
        pixel_image:                                longblob    # final stimulus image in pixels
        -> experiment.MonitorCalibrationFromH5                  # experiment used to get the monitor calibration
        ts:                                         timestamp
        """

    @property
    def key_source(self):
        return TextureLookup * TextureScoreParameters.proj() * StimulusParameters.proj() * TwoComponentSwapParameters.proj() & (Texture * TextureGoodRun & 'score_params = 2')
    
    def get_swap_samples(self, key, texture, v_mask, fixed_swap=None, sample_criterion='random', n_sample=20, seed=1234, device='cuda'):
        # n_sample is the number of crops sampled from texture
        mei, mask = (base.MEI * base.MEIMask & key).fetch1('mei', 'mask')
        mask_tensor = torch.tensor(mask,dtype=torch.float32,device=device)

        image_model = LinearImageModel(mei,v_mask,eval_texture = texture).to(device)
        image_model.initialize_t()
        if fixed_swap is not None:
            if (v_mask.sum() == mask.sum()): # when fixed part is empty
                image_model.fixed_c.data = torch.zeros_like(mask_tensor,dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).contiguous()
            else:
                f_mask = mask - v_mask
                fc_mean = (mei * f_mask).sum(axis=(-1, -2), keepdims=True) / f_mask.sum()
                fc_std = np.sqrt(np.sum(((mei - fc_mean) ** 2) * f_mask, axis=(-1, -2), keepdims=True) / f_mask.sum())
                stan_fixed_swap = ops.standardize_image(fixed_swap, fc_mean, fc_std, mask=f_mask, mask_mean_subtraction=True, mask_image=False, match_stats='mask')
                image_model.fixed_c.data = torch.tensor(np.array(stan_fixed_swap * (1-v_mask)),dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).contiguous()

        torch.manual_seed(seed)
        samples = Texture.select_texture_dei(image_model, sample_criterion=sample_criterion, n_sample=n_sample)
        return samples

    def make(self, key):
        self.insert1(key)
        key = (self.key_source & key).fetch1()
        swap_params = (TwoComponentSwapParameters & key).fetch1()
        mask = (base.MEIMask & key).fetch1('mask')
        v_mask, texture = (Texture & key).fetch1('variable_mask', 'eval_texture')
        gamma_key = (experiment.MonitorCalibrationFromH5 & MONCALIB_ON_KEY & PDCALIB_ON_KEY).fetch1('KEY')
        seed = swap_params['selection_seed'] + key['group_id'] + key['neuron_id']

        # Get original local texture DEIs
        ldeis = self.get_swap_samples(key, texture, v_mask, fixed_swap=None, sample_criterion='random', n_sample=swap_params['n_swaps'], seed=seed)
        
        # Get variable swaps
        if swap_params['v_swap_type'] == 'other_texture':
            temp_key = key.copy()
            [temp_key.pop(key) for key in ['neuron_id', 'weight_id', 'texture_id']]
            swap_neurons, swaps = (TextureLookup * Texture & (TextureGoodRun & temp_key) & 'neuron_id != {}'.format(key['neuron_id'])).fetch('neuron_id', 'eval_texture', order_by='neuron_id')
            np.random.seed(seed)
            vswap_idxs = np.random.choice(range(len(swaps)), swap_params['n_swaps'], replace=False)
            vswaps = []
            for swap in swaps[vswap_idxs]:
                vswaps.append(self.get_swap_samples(key, swap, v_mask, fixed_swap=None, sample_criterion='random', n_sample=1, seed=seed))
        else: 
            raise NotImplementedError('Variable component swap type not implemented!')
            
        # Get fixed swaps (variable components are the same samples as in local texture DEIs)
        if swap_params['f_swap_type'] == 'natural':
            image_class = "imagenet"
            image_ids = (stimulus.StaticImage.Image & {'image_class': image_class}).fetch('image_id')
            np.random.seed(seed)
            fswap_ids = np.random.choice(image_ids, swap_params['n_swaps'], replace=False)
            swaps = (stimulus.StaticImage.Image & 'image_class = "imagenet"' & [{'image_id': i} for i in fswap_ids]).fetch('image')
            import cv2
            fswaps = []
            for i, swap in enumerate(swaps):
                swap = cv2.resize(swap, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
                fswaps.append(self.get_swap_samples(key, texture, v_mask, fixed_swap=swap, sample_criterion='random', n_sample=swap_params['n_swaps'], seed=seed)[i])
        else: 
            raise NotImplementedError('Fixed component swap type not implemented!')
        
        # Insert stimuli
        for i, image in enumerate(ldeis):
            image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask)
            self.LocalTextureDEI.insert1({**key, 'image_id': i, 'mean_luminance': image_mean, 'std_luminance': image_std,
                        'pixel_image': pixel_image, **gamma_key}, ignore_extra_fields=True)
        
        for i, (nid, image) in enumerate(zip(swap_neurons[vswap_idxs], np.stack(vswaps).squeeze())):
            image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask)
            self.VariableSwap.insert1({**key, 'image_id': i, 'swap_neuron_id': nid, 
                                        'mean_luminance': image_mean, 'std_luminance': image_std, 'pixel_image': pixel_image, **gamma_key}, ignore_extra_fields=True)

        for i, (iid, image) in enumerate(zip(fswap_ids, np.stack(fswaps).squeeze())):
            image_mean, image_std, pixel_image = prepare_stimulus(key, image, mask)
            self.FixedSwap.insert1({**key, 'image_id': i, 'swap_image_class': image_class, 'swap_image_id': iid,
                                    'mean_luminance': image_mean, 'std_luminance': image_std, 'pixel_image': pixel_image, **gamma_key}, ignore_extra_fields=True)

    def fill_stimulus(self, restr, num_desired_images, experiment='two_component_specificity'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Fetch images and set some parameters
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
        source_tables = [self.LocalTextureDEI, self.VariableSwap, self.FixedSwap]
        image_tables = [stimulus.StaticImage.MaskedLocalTextureDEI, stimulus.StaticImage.LocalTextureDEIVariableSwap, stimulus.StaticImage.LocalTextureDEIFixedSwap]
        image_classes = ['masked_local_texture_dei', 'local_texture_dei_variable_swap', 'local_texture_dei_fixed_swap']

        # Fill stimulus
        for src_table, image_table, image_class in zip(source_tables, image_tables, image_classes):
            image_keys = (TextureLookup * src_table & restr).fetch(as_dict=True, order_by='neuron_id')
            images = (TextureLookup * src_table & restr).fetch('pixel_image', order_by='neuron_id')
            for im_key in image_keys:
                im_key['dei_id'] = im_key['image_id']
                im_key.pop('image_id')
            restr = (TextureLookup * src_table & restr).proj()
            fill_stimulus_function(src_table, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)