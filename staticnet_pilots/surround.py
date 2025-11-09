""" This is code for the pilot center surround experiment

This is meant to be single use code.

Documentation of experiments ran
22620-4-15  Oct 7 2019  Imagenet (collection 2), source
22620-4-17  Oct 7 2019  ImageNet (collection 2), repeat
22620-5-11  Oct 8 2019  ImageNet (collection 2), repeat
22620-5-13  Oct 8 2019  Masked vs unmasked imagenet
22620-6-9   Oct 9 2019  MEI vs LEI, sync error (up to 80 msecs off)
22620-6-11  Oct 9 2019  Masked vs unmasked MEI
22620-7-10  Oct 10 2019 ImageNet2
22620-7-13  Oct 10 2019 Masked vs UmaskedImageNet, syncing up to 300 msecs off, ignore

stacks
22620-4-16
22620-5-14
22620-6-10
22620-7-14
"""
import datajoint as dj
import torch
import featurevis
from featurevis import ops
from featurevis import utils
from featurevis import models
import numpy as np
from scipy import ndimage

from staticnet_analyses import base
from staticnet_experiments import models as static_models

imagenet = dj.create_virtual_module('imagenet', 'pipeline_imagenet')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')

schema = dj.schema('neurostatic_pilot') # TODO: Change the name here to neurostatic_pilot_surround


# ImageNet collections are sets of distinct 5000 images (+ 100 oracles) sampled at random from the ImageNet database
# Good collections are: 2, 3, 6, 7. 2 is the default one we use for our experiments.

@schema
class PreprocessedAlbum(dj.Computed):
    definition = """ # single frames for collection processed as in neuro_data.static_images.data_schemas.Frame (just resized)
    -> imagenet.Album
    """

    @property
    def key_source(self):
        return imagenet.Album & 'collection_id in (2, 3, 6, 7)'

    class Single(dj.Part):
        definition = """ # single frames
        -> master
        -> imagenet.Album.Single
        ---
        frame:      longblob
        """

    class Oracle(dj.Part):
        definition = """ # oracle frames
        -> master
        -> imagenet.Album.Oracle
        ---
        frame:      longblob 
        """

    def make(self, key):
        from neuro_data.static_images import data_schemas
        stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')

        # Insert key
        self.insert1(key)

        # Insert all single frames
        keys, frames = (stimulus.StaticImage.Image & (imagenet.Album.Single & key)).fetch(
            'KEY', 'image')
        for frame_key, frame in zip(keys, frames):
            processed_frame = data_schemas.process_frame({'preproc_id': 0}, frame)
            self.Single.insert1({**key, **frame_key, 'frame': processed_frame})

        # Insert all oracle frames
        keys, frames = (stimulus.StaticImage.Image & (imagenet.Album.Oracle & key)).fetch(
            'KEY', 'image')
        for frame_key, frame in zip(keys, frames):
            processed_frame = data_schemas.process_frame({'preproc_id': 0}, frame)
            self.Oracle.insert1({**key, **frame_key, 'frame': processed_frame})


@schema
class ImageNetParameters(dj.Lookup):
    definition = """ # some parameters to compute responses to ImageNet models

    imagenet_params:    int
    ---
    desired_mean:   float   # desired mean of the output images
    desired_std:    float   # desired contrast of the output images
    std_thresh:     float   # any images whose original std (inside the mask) is below this will be ignored 
    """
    contents = [
        {'imagenet_params': 1, 'desired_mean': 0, 'desired_std': 0.3, 'std_thresh': 8},
        {'imagenet_params': 2, 'desired_mean': 0, 'desired_std': 0.3, 'std_thresh': 15}, ]


@schema
class ImageNetResponses(dj.Computed):
    definition = """ # model responses to 5000 single images in an imagenet collection

    -> base.MEIMask   
    -> PreprocessedAlbum
    -> ImageNetParameters
    ---
    unmasked_activations:   longblob    # activation to all images in the collection (ordered by image_ids)
    masked_activations:     longblob    # activation to all masked images in the collection (ordered by image_ids)
    """

    @property
    def key_source(self):
        return base.MEIMask * PreprocessedAlbum * ImageNetParameters & {
            'imagenet_params': 2}

    def make(self, key):
        # Get models
        model_key = {'group_id': key['group_id'], 'net_hash': key['net_hash']}
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos')
                       else (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                   device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], average_batch=False,
                                device='cuda')

        # Get images
        images = (PreprocessedAlbum.Single & key).fetch('frame', order_by='image_id')
        images = np.stack(images)

        # normalize images
        # if (base.Dataset.TrainStats & key).fetch1('norm_per_image'):
        #     images = (images - images.mean(axis=(-1, -2), keepdims=True)) / images.std(
        #         axis=(-1, -2), keepdims=True)
        # else:
        #     train_mean, train_std = (base.Dataset.TrainStats & key).fetch1(
        #         'mean_img_value', 'std_img_value')
        #     images = (images - train_mean) / train_std
        #
        # # Change std to match desired contrast inside the mask
        # mask = (base.MEIMask & key).fetch1('mask')
        # target_std = float((ImageNetParameters & key).fetch1('contrast'))
        # img_mean = (images * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
        # img_std = np.sqrt(np.sum(((images - img_mean) ** 2) * mask, axis=(-1, -2),
        #                          keepdims=True) / mask.sum())
        # good_imgs = img_std.squeeze() > 1e-5  # avoid images with very low standard deviation
        # images[good_imgs] = (images / img_std)[good_imgs] * target_std

        # Change mean and standard deviation inside the mask to match the desired values
        mask = (base.MEIMask & key).fetch1('mask')
        target_mean, target_std = (ImageNetParameters & key).fetch1('desired_mean',
                                                                    'desired_std')
        img_mean = (images * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
        img_std = np.sqrt(np.sum(((images - img_mean) ** 2) * mask, axis=(-1, -2),
                                 keepdims=True) / mask.sum())
        norm_images = (images - img_mean) / img_std
        final_images = (norm_images * target_std) + target_mean

        # Evaluate all images
        unmasked_acts = []
        masked_acts = []
        for i in range(0, len(final_images), 128):
            batch = final_images[i: i + 128, None]

            unmasked = torch.as_tensor(batch, dtype=torch.float32, device='cuda')
            masked = torch.as_tensor(batch * mask, dtype=torch.float32, device='cuda')
            with torch.no_grad():
                unmasked_acts.append(model(unmasked).cpu().numpy())
                masked_acts.append(model(masked).cpu().numpy())
        unmasked_acts = np.concatenate(unmasked_acts)
        masked_acts = np.concatenate(masked_acts)

        # Ignore those with too low std (could cause values to explode)
        std_thresh = (ImageNetParameters & key).fetch1('std_thresh')
        unmasked_acts[img_std.squeeze() < std_thresh] = np.nan
        masked_acts[img_std.squeeze() < std_thresh] = np.nan

        self.insert1({**key, 'unmasked_activations': unmasked_acts,
                      'masked_activations': masked_acts})


@schema
class FacilitationScore(dj.Computed):
    definition = """ # compute a measure of surround facilitation (negative is surround suppreses response)

    -> ImageNetResponses
    ---
    facilitation_score:    float        # median unmasked/masked response expressed in dB
    """

    def make(self, key):
        # Get masked and unmasked activations
        unmasked, masked = (ImageNetResponses & key).fetch1('unmasked_activations',
                                                            'masked_activations')

        # Compute facilitation score
        fac_factor = np.nanmean(unmasked - masked)

        # # Compute facilitation score
        # nan_acts = np.isnan(unmasked)
        # unmasked, masked = unmasked[~nan_acts], masked[~nan_acts]
        # min_value = min(unmasked.min(), masked.min())
        # pos_unmasked = unmasked - min_value
        # pos_masked = masked - min_value
        # fac_factor = 10 * np.log10(np.median(pos_unmasked / (pos_masked + 1e-7)))
        # this correlates highly (0.98 spearman) with the mean of differences

        # Insert
        self.insert1({**key, 'facilitation_score': fac_factor})


@schema
class LEI(dj.Computed):
    definition = """ # least exciting image, (same as MEI but trying to shut down the cell)

    -> static_models.Model
    -> base.Dataset.Unit
    -> base.MEIParameters
    ---
    lei:                longblob # optimized MEI
    activation:         float   # activation at the MEI 
    """

    @property
    def key_source(self):
        all_keys = static_models.Model * base.Dataset.Unit * base.MEIParameters
        return all_keys & {'seed': 1009, 'mei_params': 1}

    def make(self, key):
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
        ensemble = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                   neuron_idx=key['neuron_id'], device='cuda')
        model = utils.Compose([ensemble, ops.MultiplyBy(-1)])

        # Create a random initial image (at desired contrast)
        torch.manual_seed(mei_params['mei_seed'])
        image_shape = (mei_params['num_initializations'], 1, mei_params['height'],
                       mei_params['width'])
        initial_image = torch.randn(image_shape, device='cuda')
        initial_image = initial_image * float(mei_params['contrast'])

        # Optimize
        postup_op = ops.ChangeStd(float(mei_params['contrast']))
        lei, fevals, _ = featurevis.gradient_ascent(model, initial_image,
                                                    post_update=postup_op,
                                                    step_size=mei_params['step_size'],
                                                    num_iterations=mei_params[
                                                        'num_iterations'])
        lei = lei.mean(0).squeeze().cpu().numpy()
        activation = -fevals[-1]

        # Insert
        self.insert1({**key, 'lei': lei, 'activation': activation})


#########################################################################################
#########################################################################################
# Preparing stimulus

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


# Masked ImageNEt vs unmasked imagenet
@schema
class StimulusMaskedImageNet(dj.Computed):
    definition = """ # resize to expected size by our stimulus program match mean and standard deviation inside mask, and save masked and unmasked version

    -> ImageNetResponses
    -> StimulusParameters
    ---
    best_fac_score:     float           # unmasked - masked for the selected image
    best_image_id:      int             # image_id for the best activity (same as in stimulus.StaticImage.Image)
    mean_luminance:     float           # (px) mean intensity inside the mask of original image
    std_luminance:      float           # (px) std of intensities inside the mask in original image
    unmasked_px:        longblob        # unmasked image in pixels after standard deviation matching
    masked_px:          longblob        # masked image in pixels after standard deviation matching
    -> experiment.MonitorCalibration    # experiment used to get the monitor calibration
    """

    @property
    def key_source(self):
        return ImageNetResponses * StimulusParameters & (
                    base.UnitRanking.Unit & 'rank < 150')  # & stim_parameters

    def make(self, key):
        # Pick image with the highest facilitation/suppression effect
        masked, unmasked = (ImageNetResponses & key).fetch1('masked_activations',
                                                            'unmasked_activations')
        fac_score = (FacilitationScore & key).fetch1('facilitation_score')
        best_idx = np.nanargmax(unmasked - masked if fac_score > 0 else masked - unmasked)
        best_fac_score = unmasked[best_idx] - masked[best_idx]

        # Fetch best image
        image_ids = (PreprocessedAlbum.Single & key).fetch('image_id',
                                                           order_by='image_id')
        best_image_id = image_ids[best_idx]
        best_image = (PreprocessedAlbum.Single & key &
                      {'image_id': best_image_id}).fetch1('frame')
        # best_image = (stimulus.StaticImage.Image & key &
        #               {'image_id': best_image_id}).fetch1('image')

        # Get mask
        mask = (base.MEIMask & key).fetch1('mask')

        # Resize image and mask(and transform into float32)
        target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
        best_image = ndimage.zoom(best_image.astype(np.float32),
                                  target_shape / best_image.shape, mode='reflect')
        mask = ndimage.zoom(mask.astype(np.float32), target_shape / mask.shape)

        # Get gamma function (and inverse)
        gamma_key, f, f_inv = get_latest_gamma_function()

        # Compute image statistics inside the mask (in luminance space)
        lum_image = f(np.clip(best_image, 0, 255))
        img_mean = (lum_image * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
        img_std = np.sqrt(np.sum(((lum_image - img_mean) ** 2) * mask, axis=(-1, -2),
                                 keepdims=True) / mask.sum())
        img_mean, img_std = img_mean.squeeze(), img_std.squeeze()

        # Normalize (in luminance space)
        target_mean, target_std = (StimulusParameters & key).fetch1('mean_lum', 'std_lum')
        target_mean = img_mean if np.isnan(target_mean) else target_mean
        target_std = img_std if np.isnan(target_std) else target_std
        lum_unmasked = ((lum_image - img_mean) / img_std) * target_std + target_mean
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
            {**key, 'best_fac_score': best_fac_score, 'best_image_id': best_image_id,
             'mean_luminance': img_mean, 'std_luminance': img_std,
             'unmasked_px': px_unmasked, 'masked_px': px_masked, **gamma_key})

    @staticmethod
    def fill_stimulus(restr={'group_id': 29}, num_desired_images=150,
                      experiment='masked_img'):
        """ Fills StaticImage tables (Image, MaskedImage2, UnmaskedImage2) needed to run
        this stimulus.

        Arguments:
            restr (str): How to restrict StimulusMaskedImageNet to get the images we want
                to insert.
            num_desired_images: The number of images to copy over. This is used to check
                the right number of images is copied over.
            experiment (str): Name of the experiment. Inserted into MaskedImage2 and
                UnmaskedImage2; useful when running more than one experiment in the same
                mice.
        """
        # Set some parameters
        unmasked_table = stimulus.StaticImage.UnmaskedImageNet2
        unmasked_class = 'unmasked_image2'
        masked_table = stimulus.StaticImage.MaskedImageNet2
        masked_class = 'masked_image2'

        # Check that there is enough images processed
        num_images = len(StimulusMaskedImageNet & restr)
        if num_images != num_desired_images:
            msg = ('Number of images in restricted table ({}) different than desired '
                   'number of images ({})').format(num_images, num_desired_images)
            raise ValueError(msg)

        # Insert in stimulus.StaticImage (if not already there)
        stimulus.StaticImageClass.insert([{'image_class': unmasked_class},
                                          {'image_class': masked_class}],
                                         skip_duplicates=True)
        stimulus.StaticImage.insert([{'image_class': unmasked_class},
                                     {'image_class': masked_class}], skip_duplicates=True)

        # Fetch images
        keys, masked_imgs, unmasked_imgs = (StimulusMaskedImageNet & restr).fetch('KEY',
                                                                                  'masked_px',
                                                                                  'unmasked_px',
                                                                                  order_by='neuron_id')
        for key in keys:  # change the name of image_class (class of original images from imagenet.Album)
            key['orig_image_class'] = key['image_class']
            del key['image_class']

        # Insert unmasked images
        all_ids = (stimulus.StaticImage.Image & {'image_class': unmasked_class}).fetch(
            'image_id')
        next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
        for i, (key, img) in enumerate(zip(keys, unmasked_imgs), start=next_id):
            stimulus.StaticImage.Image.insert1({'image_class': unmasked_class,
                                                'image_id': i, 'image': img})
            unmasked_table.insert1({'image_class': unmasked_class, 'image_id': i,
                                    'experiment': experiment, **key,
                                    'src_table': 'pilot.StimulusMaskedImageNet'})

        # Insert masked images
        all_ids = (stimulus.StaticImage.Image & {'image_class': masked_class}).fetch(
            'image_id')
        next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
        for i, (key, img) in enumerate(zip(keys, masked_imgs), start=next_id):
            stimulus.StaticImage.Image.insert1({'image_class': masked_class,
                                                'image_id': i, 'image': img})
            masked_table.insert1({'image_class': masked_class, 'image_id': i,
                                  'experiment': experiment, **key,
                                  'src_table': 'pilot.StimulusMaskedImageNet'})


# Masked MEI vs unmasked MEI
@schema
class StimulusMaskedMEI(dj.Computed):
    definition = """ # MEI processed to be presented to the mice

    -> base.MEIMask
    -> StimulusParameters
    ---
    mean_luminance:     float  # (px) mean intensity inside the mask of original MEI
    std_luminance:      float  # (px) std of intensities inside the mask in original MEI
    masked_px:          longblob  # masked MEI in pixels after standard deviation matching
    unmasked_px:        longblob  # unmasked MEI in pixels after standard deviation matching
    -> experiment.MonitorCalibration  # experiment used to get the monitor calibration
    """

    @property
    def key_source(self):
        return base.MEIMask * StimulusParameters  # & (MaskedRanking.Unit & 'rank < 150')

    def make(self, key):
        # Get MEI
        mei, mask = (base.MEI * base.MEIMask & key).fetch1('mei', 'mask')

        # Resize image and mask(and transform into float32)
        target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
        mei = ndimage.zoom(mei.astype(np.float32),
                           target_shape / mei.shape, mode='reflect')
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
    def fill_stimulus(restr={'group_id': 29}, num_desired_images=150,
                      experiment='masked_mei'):
        """ Fills StaticImage tables (Image, MaskedMEI2, MEI2) needed to run this
        stimulus.

        Arguments:
            restr (str): How to restrict StimulusMaskedMEI to get the images we want to insert.
            experiment (str): Name of the experiment. Inserted into MaskedMEI2 and MEI2;
                useful when running more than one experiment in the same mice.
            num_desired_images: The number of images to copy over. This is used to check
                the right number of images is copied over
        """
        # Set some parameters
        unmasked_table = stimulus.StaticImage.MEI2
        unmasked_class = 'mei2'
        masked_table = stimulus.StaticImage.MaskedMEI2
        masked_class = 'masked_mei2'

        # Check that there is enough images processed
        num_images = len(StimulusMaskedMEI & restr)
        if num_images != num_desired_images:
            msg = ('Number of images in restricted table ({}) different than desired '
                   'number of images ({})').format(num_images, num_desired_images)
            raise ValueError(msg)

        # Insert in stimulus.StaticImage (if not already there)
        stimulus.StaticImageClass.insert([{'image_class': unmasked_class},
                                          {'image_class': masked_class}],
                                         skip_duplicates=True)
        stimulus.StaticImage.insert([{'image_class': unmasked_class},
                                     {'image_class': masked_class}], skip_duplicates=True)

        # Fetch images
        keys, masked_imgs, unmasked_imgs = (StimulusMaskedMEI & restr).fetch('KEY',
                                                                             'masked_px',
                                                                             'unmasked_px',
                                                                             order_by='neuron_id')

        # Insert unmasked images
        all_ids = (stimulus.StaticImage.Image & {'image_class': unmasked_class}).fetch(
            'image_id')
        next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
        for i, (key, img) in enumerate(zip(keys, unmasked_imgs), start=next_id):
            stimulus.StaticImage.Image.insert1({'image_class': unmasked_class,
                                                'image_id': i, 'image': img})
            unmasked_table.insert1({'image_class': unmasked_class, 'image_id': i,
                                    'experiment': experiment, **key,
                                    'src_table': 'pilot.StimulusMaskedMEI'})

        # Insert masked images
        all_ids = (stimulus.StaticImage.Image & {'image_class': masked_class}).fetch(
            'image_id')
        next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
        for i, (key, img) in enumerate(zip(keys, masked_imgs), start=next_id):
            stimulus.StaticImage.Image.insert1({'image_class': masked_class,
                                                'image_id': i, 'image': img})
            masked_table.insert1({'image_class': masked_class, 'image_id': i,
                                  'experiment': experiment, **key,
                                  'src_table': 'pilot.StimulusMaskedMEI'})


# MEI vs LEI
@schema
class StimulusLEI(dj.Computed):
    definition = """ # LEI processed to be presented to the mice

    -> LEI
    -> StimulusParameters
    ---
    mean_luminance:     float  # (px) mean intensity inside the mask of original LEI
    std_luminance:      float  # (px) std of intensities inside the mask in original LEI
    masked_px:          longblob  # masked LEI in pixels after standard deviation matching
    unmasked_px:        longblob  # unmasked LEI in pixels after standard deviation matching
    -> experiment.MonitorCalibration  # experiment used to get the monitor calibration
    """

    @property
    def key_source(self):
        return LEI * StimulusParameters & (base.UnitRanking.Unit & 'rank < 150')

    def make(self, key):
        # Get LEI
        lei, mask = (LEI * base.MEIMask & key).fetch1('lei', 'mask')

        # Resize image and mask(and transform into float32)
        target_shape = np.array((StimulusParameters & key).fetch1('height', 'width'))
        lei = ndimage.zoom(lei.astype(np.float32), target_shape / lei.shape,
                           mode='reflect')
        mask = ndimage.zoom(mask.astype(np.float32), target_shape / mask.shape)

        # Move image to pixel space (i.e., unnormalize it )
        px_mean, px_std = (base.Dataset.TrainStats & key).fetch('mean_img_value',
                                                                'std_img_value')
        px_lei = lei * px_std + px_mean

        # Get gamma function (and inverse)
        gamma_key, f, f_inv = get_latest_gamma_function()

        # Compute image statistics inside the mask (in luminance space)
        lum_lei = f(np.clip(px_lei, 0, 255))
        lei_mean = (lum_lei * mask).sum(axis=(-1, -2), keepdims=True) / mask.sum()
        lei_std = np.sqrt(np.sum(((lum_lei - lei_mean) ** 2) * mask, axis=(-1, -2),
                                 keepdims=True) / mask.sum())
        lei_mean, lei_std = lei_mean.squeeze(), lei_std.squeeze()

        # Normalize (in luminance space)
        target_mean, target_std = (StimulusParameters & key).fetch1('mean_lum', 'std_lum')
        target_mean = lei_mean if np.isnan(target_mean) else target_mean
        target_std = lei_std if np.isnan(target_std) else target_std
        lum_unmasked = ((lum_lei - lei_mean) / lei_std) * target_std + target_mean
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
            {**key, 'mean_luminance': lei_mean, 'std_luminance': lei_std,
             'unmasked_px': px_unmasked, 'masked_px': px_masked, **gamma_key})

    @staticmethod
    def fill_stimulus(restr={'group_id': 29}, num_desired_images=150, experiment='lei'):
        """ Fills StaticImage tables (Image, MEI2, LEI) needed to run this stimulus.

        Arguments:
            restr (str): How to restrict StimulusLEI to get the images we want to insert.
            num_desired_images: The number of images to copy over. This is used to check
                the right number of images is copied over
            experiment (str): Name of the experiment. Inserted into MEI2 and LEI; useful
                when running more than one experiment in the same mice.
        """
        # Set some parameters
        lei_table = stimulus.StaticImage.LEI
        lei_class = 'lei'
        mei_table = stimulus.StaticImage.MEI2
        mei_class = 'mei2'

        # Check that there is enough images processed
        num_images = len(StimulusMaskedMEI & restr)
        num_images2 = len(StimulusLEI & restr)
        if num_images != num_images2:
            raise ValueError('Not same number of MEIs and LEIs')
        if num_images != num_desired_images:
            msg = ('Number of images in restricted table ({}) different than desired '
                   'number of images ({})').format(num_images, num_desired_images)
            raise ValueError(msg)

        # Insert in stimulus.StaticImage (if not already there)
        stimulus.StaticImageClass.insert([{'image_class': lei_class},
                                          {'image_class': mei_class}],
                                         skip_duplicates=True)
        stimulus.StaticImage.insert([{'image_class': lei_class},
                                     {'image_class': mei_class}], skip_duplicates=True)

        # Fetch images
        lei_keys, leis = (StimulusLEI & restr).fetch('KEY', 'unmasked_px',
                                                     order_by='neuron_id')
        mei_keys, meis = (StimulusMaskedMEI & restr).fetch('KEY', 'unmasked_px',
                                                           order_by='neuron_id')

        # Insert LEI images
        all_ids = (stimulus.StaticImage.Image & {'image_class': lei_class}).fetch(
            'image_id')
        next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
        for i, (key, img) in enumerate(zip(lei_keys, leis), start=next_id):
            stimulus.StaticImage.Image.insert1({'image_class': lei_class, 'image_id': i,
                                                'image': img})
            lei_table.insert1({'image_class': lei_class, 'image_id': i,
                               'experiment': experiment, **key,
                               'src_table': 'pilot.StimulusLEI'})

        # Insert MEI images
        all_ids = (stimulus.StaticImage.Image & {'image_class': mei_class}).fetch(
            'image_id')
        next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
        for i, (key, img) in enumerate(zip(mei_keys, meis), start=next_id):
            stimulus.StaticImage.Image.insert1({'image_class': mei_class, 'image_id': i,
                                                'image': img})
            mei_table.insert1({'image_class': mei_class, 'image_id': i,
                               'experiment': experiment, **key,
                               'src_table': 'pilot.StimulusMaskedMEI'})