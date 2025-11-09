import datajoint as dj
import numpy as np
import itertools
from scipy import ndimage
from itertools import product
from staticnet_analyses import base
from staticnet_analyses.base import MEIMask
from staticnet_experiments import configs, models as static_models
from neuro_data.static_images import data_schemas, stats
from staticnet_invariance.deis import MaskStatsParameters, MaskFixedMEI, DEI, DEIGoodRun, DEINatControl, DEIControl, DEINatControlParameters, DEIControlParameters, \
    TextureSynthesis, TextureSynthesisGoodRun
from featurevis import ops

imagenet = dj.create_virtual_module('pipeline_imagenet', 'pipeline_imagenet')
stimulus = dj.create_virtual_module('pipeline_stimulus', 'pipeline_stimulus')
experiment = dj.create_virtual_module('pipeline_experiment', 'pipeline_experiment')

schema = dj.schema('neurostatic_deis')

# Compute interpolation between pixel and luminance values using monitor calibration scan 
from scipy import interpolate
experiment = dj.create_virtual_module('pipeline_experiment', 'pipeline_experiment')
moncalib_on_key = dict(animal_id=26645, session=2, scan_idx=16)
pdcalib_on_key = dict(rig="2p4", trial=11)
PX, LUM = (experiment.MonitorCalibrationFromH5 & moncalib_on_key & pdcalib_on_key).fetch1('pixel_value', 'luminance')
f = interpolate.interp1d(PX, LUM)
f_inv = interpolate.interp1d(LUM, PX)

def prepare_stimulus(key, image, mask, already_masked=False):
    # upsample image and mask
    target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
    clip, target_mean, target_std, background, space, mask_image, match_stats, mask_mean_subtraction, restan_already_masked = \
    (StimulusParameters & key).fetch1('clip', 'mean', 'std', 'background', 'match_stats_space', 'mask_image', 'match_stats', 'mask_mean_subtraction', 'restan_already_masked')
    mask = ndimage.zoom(mask.astype(np.float32), target_shape / mask.shape)
    image = ndimage.zoom(image.astype(np.float32), target_shape / image.shape, mode='reflect')
    target_mean = f(128).item() if np.isnan(target_mean) else target_mean
    background = f(128).item() if np.isnan(background) else background

    if match_stats == 'mask':
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
        pixel_image = f_inv(np.clip(lum_image, f(0), f(255)))

        # Create masked image
        if mask_image:
            pixel_image = pixel_image * mask + (1 - mask) * background
            pixel_image = f_inv(np.clip(pixel_image, f(0), f(255)))    
    
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
                        'src_table': 'deis.{}'.format(source_table.table_name)}, ignore_extra_fields=True)

@schema
class StimulusParameters(dj.Lookup):
    definition = """ # parameters used to produce the images to be shown to the mice
    stim_params: int
    ---
    height:         int         # height of image
    width:          int         # width of image
    clip:           float       # range of clipping around 0
    mean=NULL:      float       # mean for stimulus images
    std:            float       # standard deviation for stimulus images 
    background=NULL:float       # background value for the masked images
    match_stats:    varchar(45) # where the target luminance stats are set for, can be 'mask' or 'ff'
    match_stats_sapce: varchar(16) # which space where statistics of stimulus images is matched, 'pixel' or 'luminance'
    mask_mean_subtraction: bool # whether to subtract mask mean or not during standardization
    restan_already_masked: bool # Whether to restandardize images that are already masked (since stats could change slightly due to zoom operation)
    """
    contents = [(-1, 144, 256, 4., 128.  , 12.  , 128.  , 1, 'ff', 'pixel', 1, 1),
                ( 1, 144, 256, 4.,   2.36,  1.  ,   2.36, 1, 'mask', 'luminance', 1, 0),
                ( 2, 144, 256, 4.,   2.36,  0.3 ,   2.36, 1, 'ff', 'luminance', 0, 0),
                ( 3, 144, 256, 3.,   2.21,  0.4 ,   2.21, 1, 'ff', 'luminance', 0, 0),
                ( 4, 144, 256, 3.,   2.21,  0.3 ,   2.21, 1, 'ff', 'luminance', 0, 0),
                ( 5, 144, 256, 4.,   2.21,  0.4 ,   2.21, 1, 'ff', 'luminance', 0, 0),
                ( 6, 144, 256, 3.,   2.21,  0.35,   2.21, 1, 'ff', 'luminance', 0, 0),
                ( 7, 144, 256, 4.,   2.21,  0.35,   2.21, 1, 'ff', 'luminance', 1, 1),
                ( 8, 144, 256, 4.,    np.nan,  0.35,    np.nan, 1, 'ff', 'luminance', 1, 1),
                ( 9, 144, 256, 4.,    np.nan,  1.  ,    np.nan, 0, 'mask', 'luminance', 1, 0)]

@schema
class MEICollection(dj.Lookup):
    definition = """
    collection_id:  int
    ---
    mask_params:    int
    massk_stats_params: int
    selection_seed:     int # seed for randomly selecting images
    """
    class Oracle(dj.Part):
        definition = """
        -> master
        -> MaskFixedMEI
        """
    class Single(dj.Part):
        definition = """
        -> master
        -> MaskFixedMEI
        """
    def fill(self, rest={'collection_id': 1, 'mask_params':2, 'mask_stats_params': 1, 'selection_seed':1}, collection_type='oracle'):
        self.insert1(rest)

        valid_group = dj.U('group_id').aggr(data_schemas.StaticMultiDataset.Member() & 'preproc_id in (0, 5)', n='COUNT(*)') & 'n=1 and group_id!=179'
        cnn_models = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.GaussianLaplace * static_models.Model)
        valid_model = static_models.Model.UnitTestScores * base.Dataset.Unit & cnn_models & valid_group.proj() & base.UnitRanking() & 'seed = 1009'
        valid_oracle = stats.Oracle.UnitScores * base.Dataset.Unit & valid_group.proj() & base.UnitRanking()
        mei_rel = (valid_model & 'pearson>0.5').proj(test_corr='pearson') * (valid_oracle & 'pearson > 0.5').proj(oracle='pearson')
        valid_net_hash = ['403a616c4761b733cb767a47a6d0e7da', '8b6fe18fa651ebf452db0fbd77d05a01']
        rel = base.MEI.key_source & mei_rel & 'mei_params = 8 and data_hash = "7572eed73113c993e7d1b92f83e270b4"' & [{'net_hash': net} for net in valid_net_hash]
        exclude_groups = [155, 156, 157, 186, 187, 188, 198, 199, 200]
        final_rel = rel.proj() - (rel & [{'group_id': gid} for gid in exclude_groups]).proj()

        valid_group = dj.U('group_id').aggr(data_schemas.StaticMultiDataset.Member() & 'preproc_id in (0, 5)', n='COUNT(*)') & 'n=1 and group_id!=179'
        cnn_models = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.GaussianLaplace * static_models.Model)
        valid_model = static_models.Model.UnitTestScores * base.Dataset.Unit & cnn_models & valid_group.proj() & (base.UnitRanking() & 'ranking_params = 8') & 'seed = 1009'
        valid_oracle = stats.Oracle.UnitScores * base.Dataset.Unit & valid_group.proj() & (base.UnitRanking() & 'ranking_params = 8')
        mei_rel = (valid_model & 'pearson>0.7').proj(test_corr='pearson') * (valid_oracle & 'pearson > 0.5').proj(oracle='pearson')
        valid_net_hash = ['403a616c4761b733cb767a47a6d0e7da', '8b6fe18fa651ebf452db0fbd77d05a01']
        rel = base.MEI.key_source & mei_rel & 'mei_params = 8 and data_hash = "7572eed73113c993e7d1b92f83e270b4"' & [{'net_hash': net} for net in valid_net_hash]
        include_groups = [74, 88, 106, 142, 204, 209]
        good_neurons = rel & [{'group_id': gid} for gid in include_groups]

        # select random images based on collection_id
        np.random.seed(rest['selection_seed'])
        if collection_type == 'oracle':
            n_ims = 100
            keys = np.random.choice(good_neurons.fetch('KEY'), n_ims, replace=False)
            keys = (MaskFixedMEI & rest & keys).fetch('KEY')
            self.Oracle.insert([{'collection_id': rest['collection_id'], **key} for key in keys])
        elif collection_type == 'single':
            n_ims = 5000
            keys = np.random.choice(good_neurons.fetch('KEY'), n_ims, replace=False)
            keys = (MaskFixedMEI & rest & keys).fetch('KEY')
            self.Single.insert([{'collection_id': rest['collection_id'], **key} for key in keys])

####################################### Stimulus tables ##########################################################
@schema
class StimulusMEIOracle(dj.Computed):
    definition = """ # the original single MEI processed to be presented to the mice
    -> MEICollection.Oracle
    -> MaskFixedMEI
    -> StimulusParameters
    ---
    mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
    std_luminance:      float  # (px) std of intensities inside the mask in original MEI
    pixel_image:          longblob  # masked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibrationFromH5  # experiment used to get the monitor calibration
    ts:                 timestamp
    """
    @property
    def key_source(self):
        return MEICollection.Oracle * MaskFixedMEI * StimulusParameters

    def make(self, key):
        mei, mask = (MEICollection.Oracle * MaskFixedMEI * base.MEIMask & key).fetch1('mei', 'mask')
        image_mean, image_std, pixel_image = prepare_stimulus(key, mei, mask)
        # Hacking - needs to implement finding most recent calibration
        gamma_key = (experiment.MonitorCalibrationFromH5 & {'animal_id': 0, 'session': 3195, 'scan_idx': 1}).fetch1('KEY')
        self.insert1({**key, 'mean_luminance': image_mean, 'std_luminance': image_std,
                     'pixel_image': pixel_image, **gamma_key})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='dei'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.MaskedMEIOracle
        image_class = 'mask_fixed_mei'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace

        # Fetch images
        images, image_keys = (self & restr & image_models).fetch('pixel_image', 'KEY', order_by='neuron_id')
        fill_stimulus_function(self, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class StimulusMEI(dj.Computed):
    definition = """ # the original single MEI processed to be presented to the mice
    -> MaskFixedMEI
    -> StimulusParameters
    ---
    mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
    std_luminance:      float  # (px) std of intensities inside the mask in original MEI
    pixel_image:          longblob  # masked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibrationFromH5  # experiment used to get the monitor calibration
    ts:                 timestamp
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
        gamma_key = (experiment.MonitorCalibrationFromH5 & moncalib_on_key).fetch1('KEY')
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
    -> DEI.DEI
    -> StimulusParameters
    ---
    mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
    std_luminance:      float  # (px) std of intensities inside the mask in original MEI
    pixel_image:          longblob  # masked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibrationFromH5  # experiment used to get the monitor calibration
    ts:                 timestamp
    """

    @property
    def key_source(self):
        return DEI.DEI * StimulusParameters & DEIGoodRun

    def make(self, key):
        mei, mask = (DEI.DEI * base.MEIMask & key).fetch1('dei', 'mask')
        image_mean, image_std, pixel_image = prepare_stimulus(key, mei, mask)
        # Hacking - needs to implement finding most recent calibration
        gamma_key = (experiment.MonitorCalibrationFromH5 & moncalib_on_key).fetch1('KEY')
        self.insert1({**key, 'mean_luminance': image_mean, 'std_luminance': image_std,
                     'pixel_image': pixel_image, **gamma_key})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='dei'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.MaskedDEI
        image_class = 'masked_dei'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
       
        # Fetch images
        images, image_keys = (self & restr & image_models).fetch('pixel_image', 'KEY', order_by='neuron_id')
        fill_stimulus_function(self, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

@schema
class StimulusDEIControl(dj.Computed):
    definition = """ # the DEI synthesized controls processed to be presented to the mice
    -> DEIControl.Image
    -> StimulusParameters
    ---
    mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
    std_luminance:      float  # (px) std of intensities inside the mask in original MEI
    pixel_image:          longblob  # masked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibrationFromH5  # experiment used to get the monitor calibration
    ts:                 timestamp
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
        gamma_key = (experiment.MonitorCalibrationFromH5 & moncalib_on_key).fetch1('KEY')
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
    mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
    std_luminance:      float  # (px) std of intensities inside the mask in original MEI
    pixel_image:          longblob  # masked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibrationFromH5  # experiment used to get the monitor calibration
    ts:                 timestamp
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

        # Hacking - needs to implement finding most recent calibration
        gamma_key = (experiment.MonitorCalibrationFromH5 & moncalib_on_key).fetch1('KEY')
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
class TextureSynthesisLookup(dj.Lookup):
    definition = """
    synthesis_id: int 
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
    """

    def fill(self, texture_key):
        dics = (TextureSynthesis.TextureSynthesis & texture_key).fetch('KEY')
        current_max_id = len(self)
        for i, dic in enumerate(dics):
            dic['synthesis_id'] = i + current_max_id
            self.insert1(dic)

@schema
class StimulusFullTextureDEI(dj.Computed):
    definition = """ # the full texture DEIs processed to be presented to the mice
    -> TextureSynthesisLookup
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
        mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
        std_luminance:      float  # (px) std of intensities inside the mask in original MEI
        pixel_image:          longblob  # masked MEI in pixels after standard deviation matching
        -> experiment.MonitorCalibrationFromH5  # experiment used to get the monitor calibration
        ts:                 timestamp
        """

    @property
    def key_source(self):
        return TextureSynthesisLookup * StimulusParameters & 'texture_id = -1'

    def make(self, key):
        self.insert1(key)
        neuron_key = (TextureSynthesisLookup * StimulusParameters.proj() & key).fetch1()
        samples, mask = (TextureSynthesis.TextureSynthesis * base.MEIMask & neuron_key).fetch1('samples', 'mask')
        for i, sample in enumerate(samples):
            image_mean, image_std, pixel_image = prepare_stimulus(neuron_key, sample, mask)
            # Hacking - needs to implement finding most recent calibration
            gamma_key = (experiment.MonitorCalibrationFromH5 & moncalib_on_key).fetch1('KEY')
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
    -> TextureSynthesisLookup
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
        mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
        std_luminance:      float  # (px) std of intensities inside the mask in original MEI
        pixel_image:          longblob  # masked MEI in pixels after standard deviation matching
        -> experiment.MonitorCalibrationFromH5  # experiment used to get the monitor calibration
        ts:                 timestamp
        """

    @property
    def key_source(self):
        return TextureSynthesisLookup * StimulusParameters & (TextureSynthesis.TextureSynthesis & (TextureSynthesisGoodRun & 'score_params = 2'))

    def make(self, key):
        self.insert1(key)
        neuron_key = (TextureSynthesisLookup * StimulusParameters.proj() & key).fetch1()
        samples, mask = (TextureSynthesis.TextureSynthesis * base.MEIMask & neuron_key).fetch1('samples', 'mask')
        for i, sample in enumerate(samples):
            image_mean, image_std, pixel_image = prepare_stimulus(neuron_key, sample, mask)
            # Hacking - needs to implement finding most recent calibration
            gamma_key = (experiment.MonitorCalibrationFromH5 & moncalib_on_key).fetch1('KEY')
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


# ========================================================= ARCHIVED ========================================================================

