""" Most Exciting Gabors

We use 4 parameters to define our Gabors (see utils.create_gabor for details) plus 2
more params for x, y location:
    orientation: The angle of the Gabor.
    phase: Phase of the sinusoid.
    wavelength: Wavelength of the sinusoid.
    sigma: Standard deviation of the gaussian mask
    dx, dy: Location in monitor.
Gaussian mask is spherical, i.e., same sigma in y and x. More general definitions allow
for elliptical masks (another parameter to control the ellipticity) and for rotations of
this mask; we do not.
"""

import datajoint as dj
import featurevis
import numpy as np
import torch
from featurevis import models
from featurevis import ops
from featurevis import utils

from staticnet_analyses import base
from staticnet_analyses import utils as static_utils
from staticnet_experiments import models as static_models

segmentation = dj.create_virtual_module('neurostatic_segmentation','neurostatic_segmentation')

schema = dj.schema('neurostatic_gabors', create_tables=True)

dj.config['enable_python_native_blobs'] = True

class GratingGenerator():
    """ Generate a Gabor patch based on input parameters."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    @utils.varargin
    def __call__(self, x):
        """  Create a full-field sinusoidal grating with the params in x.

        Arguments:
            x (torch.Tensor): Examples. Each has 3 parameters: orientation, phase,
                wavelength. See utils.create_gabor() for details.
        """
        gratings = [static_utils.create_gabor(self.height, self.width, *params, only_grating=True) for params
                  in x]
        gratings = torch.stack(gratings)[:, None, :, :]  # add channel dimension
        return gratings

class GaborGenerator():
    """ Generate a Gabor patch based on input parameters."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    @utils.varargin
    def __call__(self, x):
        """  Create a Gabor patch with the params in x.

        Arguments:
            x (torch.Tensor): Examples. Each has 4-6 parameters: orientation, phase,
                wavelength, sigma, dx and dy. See utils.create_gabor() for details.
        """
        gabors = [static_utils.create_gabor(self.height, self.width, *params) for params
                  in x]
        gabors = torch.stack(gabors)[:, None, :, :]  # add channel dimension
        return gabors

class ComboGratingGenerator():
    def __init__(self, image_size=(36, 64), center=(0., 0.), mask_sigma=2, mask1=None, mask2=None, 
                 standardization_params=dict(target_mean=0, target_std=0.25, mask=None, mask_mean_subtraction=True, mask_image=True, match_stats='ff')):

        """
        Curved edge Combined Gabor generator class.
        Overall params:
            image_size (tuple of integers): Image height and width.
            center (tuple of integers): The center position of each of the component plaid.
            mask_sigma: Sigma for gaussian blurring on the edge between two masks.
        Params to fit:
            *** Params need to follow this order ***
            r(float): Radius of the circle that defines the curved edge.
            a(float): Orientation of the curve, as fraction of a full circle.
            shift_x(float): Shift in x position of the curved edge from center x. 
            shift_y(float): Shift in y position of the curved edge from center y.
            theta1 (float): Orientation of the sinusoid (in radian) of grating 1.
            psi1 (float): Phase of the sinusoid of grating 1.
            lambda (float): Sinusoid wavelengh (1/frequency) of grating 1.
            theta2 (float): Orientation of the sinusoid (in radian) of grating 2.
            psi2 (float): Phase of the sinusoid of grating 2.
            lambda2 (float): Sinusoid wavelengh (1/frequency) of grating 2.
        Returns:
            2D torch.tensor: An image as summation of two masked gratings.
        """
        super().__init__()
        self.image_size = image_size
        self.center = center
        self.mask_sigma = mask_sigma
        self.standardization_params = standardization_params
        self.mask1 = mask1
        self.mask2 = mask2
    
    def compute_mask(self, r, a, shift_x, shift_y):
        from scipy import ndimage
        ymax, xmax = self.image_size
        Y,X = np.arange(ymax)[:, None], np.arange(xmax)[None, :]
        a *= 2 * np.pi
        cy, cx = ymax/2 + self.center[1] - r * np.cos(a) + shift_y, xmax/2 + self.center[0] - r * np.sin(a) + shift_x
        d = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        mask1 = ndimage.gaussian_filter((d > r).astype(np.float32), self.mask_sigma)
        mask2 = ndimage.gaussian_filter((d <= r).astype(np.float32), self.mask_sigma)
        return mask1, mask2

    def __call__(self, params):
        # Compute mask if there's no exsiting mask
        if self.mask1 is None and self.mask2 is None:
            r, a, shift_x, shift_y, theta1, psi1, lambda1, theta2, psi2, lambda2 = params
            mask1, mask2  = self.compute_mask(r, a, shift_x, shift_y)
        else:
            theta1, psi1, lambda1, theta2, psi2, lambda2 = params
            mask1 = self.mask1
            mask2 = self.mask2
        
        # Combine components
        g1 = GratingGenerator(self.image_size[0], self.image_size[1])(torch.as_tensor([theta1, psi1, lambda1], dtype=torch.float32, device='cuda').unsqueeze(0))
        g2 = GratingGenerator(self.image_size[0], self.image_size[1])(torch.as_tensor([theta2, psi2, lambda2], dtype=torch.float32, device='cuda').unsqueeze(0))
        image = mask1 * g1.detach().cpu().squeeze().numpy() + mask2 * g2.detach().cpu().squeeze().numpy()
        
        # Standardize image
        if self.standardization_params is not None:
            image = ops.standardize_image(image, *self.standardization_params.values())

        return image

@schema
class SearchRange(dj.Lookup):
    definition = """ # search range for Gabor parameters

    search_range:       int         # id of this search range
    ---
    lower_wavelength:   float
    upper_wavelength:   float
    lower_sigma:        float
    upper_sigma:        float
    lower_dx:           float
    upper_dx:           float
    lower_dy:           float
    upper_dy:           float
    """
    contents = [{'search_range': 1, 'lower_wavelength': 0.12, 'upper_wavelength': 0.6,
                 'lower_sigma': 0.04, 'upper_sigma': 0.3, 'lower_dx': -0.45,
                 'upper_dx': 0.45, 'lower_dy': -0.45, 'upper_dy': 0.45}]
    # orientation always go from 0-pi degrees, and phase from 0-2pi degrees


@schema
class OptimalGaborParameters(dj.Lookup):
    definition = """ # parameters used to find the optimal gabor
    
    optgabor_params: int
    ---
    method:             varchar(16) # name of the optimization method 
    seeds:              longblob    # random seeds used during optimization
    use_avg_model:      bool        # use the average model (across different training seeds) to evaluate the gabor
    height:             int         # height of the image
    width:              int         # width of the image
    num_iterations:     int         # number of optimization iterations for the annealing
    """
    contents = [{'optgabor_params': 1, 'method': 'annealing+local',
                 'seeds': [1, 12, 123, 1234, 12345], 'use_avg_model': True, 'height': 36,
                 'width': 64, 'num_iterations': 300}]


@schema
class OptimalGabor(dj.Computed):
    definition = """ # find parameters that produce an optimal gabor for this unit

    -> static_models.Model
    -> base.Dataset.Unit
    -> OptimalGaborParameters
    -> SearchRange
    ---
    opt_gabor:          longblob    # best gabor image
    opt_seed:           int         # random seed used to obtain the best gabor
    opt_activation:     float       # activation at the best gabor image
    opt_orientation:    float       # (radians) counterclockwise rotation to apply (0 is horizontal, pi/2 vertical)
    opt_phase:          float       # (radians) angle at which to start the sinusoid
    opt_wavelength:     float       # (px/height) wavelength of the sinusoid (1 / spatial frequency)
    opt_sigma:          float       # (px/height) sigma of the gaussian mask used
    opt_dx:             float       # (px/width) amount of translation in x (positive moves right)
    opt_dy:             float       # (px/height) amount of translation in y (positive moves downwards)
    """

    @property
    def key_source(self):
        all_keys = (static_models.Model * base.Dataset.Unit * OptimalGaborParameters *
                    SearchRange)
        return all_keys & {'seed': 1009, 'optgabor_params': 1}

    def make(self, key):
        from scipy import optimize

        # Get parameters
        optgabor_params = (OptimalGaborParameters & key).fetch1()

        # Get models
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     optgabor_params['use_avg_model'] else key)
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

        # Get search range
        search_range = (SearchRange & key).fetch1()
        lower_limits = [0, 0, *(search_range['lower_{}'.format(p)] for p in
                                ['wavelength', 'sigma', 'dx', 'dy'])]
        upper_limits = [np.pi, 2 * np.pi, *(search_range['upper_{}'.format(p)] for p in
                                            ['wavelength', 'sigma', 'dx', 'dy'])]

        # Get gabor generating function
        generator = GaborGenerator(optgabor_params['height'], optgabor_params['width'])

        # Write loss function to be optimized
        def neg_model_activation(params):
            # Create gabor image
            params = [np.clip(p, l, u) for p, l, u in zip(params, lower_limits,
                                                          upper_limits)]
            tensor_params = torch.as_tensor(params, dtype=torch.float32, device='cuda')
            gabor = generator(tensor_params.unsqueeze(0))

            # Compute activation
            with torch.no_grad():
                activation = model(gabor).item()
            return - activation

        # Find best parameters (simulated annealing -> local search)
        best_activation = np.inf
        for seed in optgabor_params['seeds']:  # try 5 diff random seeds
            print('Optimizing with seed =', seed)
            res = optimize.dual_annealing(neg_model_activation,
                                          bounds=list(zip(lower_limits, upper_limits)),
                                          maxiter=optgabor_params['num_iterations'],
                                          seed=seed, no_local_search=True)
            res = optimize.minimize(neg_model_activation, x0=res.x, method='Nelder-Mead')

            if res.fun < best_activation:
                # Save best yet
                best_activation = res.fun
                best_params = res.x
                best_seed = seed
        best_params = [np.clip(p, l, u) for p, l, u in zip(best_params, lower_limits,
                                                           upper_limits)]

        # Create best gabor
        tensor_params = torch.as_tensor(best_params, dtype=torch.float32, device='cuda')
        best_gabor = generator(tensor_params.unsqueeze(0))
        best_activation = -model(best_gabor)

        # Insert
        self.insert1({**key, 'opt_gabor': best_gabor.squeeze().cpu().numpy(),
                      'opt_seed': best_seed, 'opt_activation': best_activation.item(),
                      'opt_orientation': best_params[0], 'opt_phase': best_params[1],
                      'opt_wavelength': best_params[2], 'opt_sigma': best_params[3],
                      'opt_dx': best_params[4], 'opt_dy': best_params[5]})

@schema
class OptimalGrating(dj.Computed):
    definition = """ # find parameters that produce an optimal gabor for this unit
    -> static_models.Model
    -> base.Dataset.Unit
    -> OptimalGaborParameters
    -> SearchRange
    ---
    opt_grating:        longblob    # best grating image
    opt_seed:           int         # random seed used to obtain the best grating
    opt_activation:     float       # activation at the best grating image
    opt_orientation:    float       # (radians) counterclockwise rotation to apply (0 is horizontal, pi/2 vertical)
    opt_phase:          float       # (radians) angle at which to start the sinusoid
    opt_wavelength:     float       # (px/height) wavelength of the sinusoid (1 / spatial frequency)
    """

    @property
    def key_source(self):
        all_keys = (static_models.Model * base.Dataset.Unit * OptimalGaborParameters *
                    SearchRange)
        return all_keys & {'seed': 1009, 'optgrating_params': 1}

    def make(self, key):
        from scipy import optimize

        # Get parameters
        optgrating_params = (OptimalGaborParameters & key).fetch1()

        # Get models
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     optgrating_params['use_avg_model'] else key)
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

        # Get search range
        search_range = (SearchRange & key).fetch1()
        lower_limits = [0, 0, *(search_range['lower_{}'.format(p)] for p in
                                ['wavelength'])]
        upper_limits = [np.pi, 2 * np.pi, *(search_range['upper_{}'.format(p)] for p in
                                            ['wavelength'])]

        # Get grating generating function
        generator = GratingGenerator(optgrating_params['height'], optgrating_params['width'])

        # Write loss function to be optimized
        def neg_model_activation(params):
            # Create grating image
            params = [np.clip(p, l, u) for p, l, u in zip(params, lower_limits,
                                                          upper_limits)]
            tensor_params = torch.as_tensor(params, dtype=torch.float32, device='cuda')
            grating = generator(tensor_params.unsqueeze(0))
            # Match image statistics to mean = 0 and std = 1
            grating = (grating - torch.mean(grating)) / torch.std(grating)
            # Compute activation
            with torch.no_grad():
                activation = model(grating).item()
            return - activation

        # Find best parameters (simulated annealing -> local search)
        best_activation = np.inf
        for seed in optgrating_params['seeds']:  # try 5 diff random seeds
            print('Optimizing with seed =', seed)
            res = optimize.dual_annealing(neg_model_activation,
                                          bounds=list(zip(lower_limits, upper_limits)),
                                          maxiter=optgrating_params['num_iterations'],
                                          seed=seed, no_local_search=True)
            res = optimize.minimize(neg_model_activation, x0=res.x, method='Nelder-Mead')

            if res.fun < best_activation:
                # Save best yet
                best_activation = res.fun
                best_params = res.x
                best_seed = seed
        best_params = [np.clip(p, l, u) for p, l, u in zip(best_params, lower_limits,
                                                           upper_limits)]

        # Create best grating
        tensor_params = torch.as_tensor(best_params, dtype=torch.float32, device='cuda')
        best_grating = generator(tensor_params.unsqueeze(0))
        best_activation = model(best_grating)

        # Insert
        self.insert1({**key, 'opt_grating': best_grating.squeeze().cpu().numpy(),
                      'opt_seed': best_seed, 'opt_activation': best_activation.item(),
                      'opt_orientation': best_params[0], 'opt_phase': best_params[1],
                      'opt_wavelength': best_params[2]})

@schema
class ComboParameters(dj.Lookup):
    definition = """ # search range for parameters used for combining two components 
    combo_params:       int         # id of this search range
    ---
    mask_sigma:         float
    boundary_shift:     varchar(16) # method for constraining shift of component boundary from the center of the MEI mask, e.g. 'mask_radius'   
    lower_r:            float
    upper_r:            float
    exisiting_mask:     boolean     # whether the component masks have been computed previously and can be fetched directly
    mask_src_table:     varchar(128)# name of the source table to fetch component masks from
    mask_src_params:    longblob    # a dictionary of parameters for restricting on mask_src_table
    """
    contents = [[1, 1.0, 'mask_radius', 2, 20]]

@schema
class OptimalComboGrating(dj.Computed):
    definition = """
    -> static_models.Model
    -> base.Dataset.Unit
    -> base.MEIMask
    -> OptimalGaborParameters
    -> SearchRange
    -> ComboParameters
    ---
    opt_params:        longblob # a list of dictionary of optimized parameters for all seeds
    opt_acts:          longblob # a list of activations of the optimizes images for all seeds
    best_params:       longblob # a dictionary of optimized parameters for the seed with the highest activation
    best_image:        longblob # the combo grating image of highest activation across all seeds
    best_activation:   float    # activation of the best_image
    mei_activation:    float    # activation of mei using the same standardization params as the combo grating image
    """
    
    @property
    def key_source(self):
        all_keys = static_models.Model * base.Dataset.Unit * base.MEIMask * OptimalGaborParameters * SearchRange * ComboParameters
        return all_keys & {'seed': 1009, 'optgabor_params': 1, 'search_range': 1}
    
    def make(self, key):
        # Get parameters
        optgrating_params = (OptimalGaborParameters & key).fetch1()
        search_range = (SearchRange & key).fetch1()
        combo_params = (ComboParameters & key).fetch1()
        mei_params = (base.MEIParameters & key).fetch1()
        mei, mei_mask, mask_x, mask_y = (base.MEI * base.MEIMask & key).fetch1('mei', 'mask', 'mask_x', 'mask_y')
        standardization_params = dict(target_mean=float(mei_params['mean']), target_std=float(mei_params['contrast']), mask=mei_mask, mask_mean_subtraction=True, mask_image=True, match_stats='ff')

        # Get predictive model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if \
                    optgrating_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos')
                    else (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')
        
        # Compute MEI activation using the specific standardization params
        stan_mei = torch.as_tensor(ops.standardize_image(mei, *standardization_params.values())[None, None], dtype=torch.float32, device='cuda')
        mei_activation = model(stan_mei).item()
        
        # Set search range for all parameters and get generator model
        image_size = [optgrating_params['height'], optgrating_params['width']]
        center = [mask_x, mask_y]
        if combo_params['existing_mask']:
            # no need to mask the image during standardizarion since the existing mask is already masked by mei_mask
            standardization_params = dict(target_mean=float(mei_params['mean']), target_std=float(mei_params['contrast']), mask=mei_mask, mask_mean_subtraction=True, mask_image=False, match_stats='ff')
            mask_src_table = eval(combo_params['mask_src_table'])
            mask1, mask2 = (mask_src_table() & combo_params['mask_src_params'] & key).fetch1('high_mask', 'low_mask')
            param_keys = ['theta1', 'psi1', 'lambda1', 'theta2', 'psi2', 'lambda2']
            lower_limits = [0, 0, search_range['lower_wavelength'], 0, 0, search_range['lower_wavelength']]
            upper_limits = [np.pi, 2 * np.pi, search_range['upper_wavelength'], np.pi, 2*np.pi, search_range['upper_wavelength']]
        else:
            mask1 = mask2 = None
            shift = np.floor(np.sqrt(mei_mask.sum() / np.pi)) # allow shift smaller than radius of the receptive field
            param_keys = ['r', 'a', 'shift_x', 'shift_y', 'theta1', 'psi1', 'lambda1', 'theta2', 'psi2', 'lambda2']
            lower_limits = [combo_params['lower_r'], 0, -shift, -shift, 0, 0, search_range['lower_wavelength'], 0, 0, search_range['lower_wavelength']]
            upper_limits = [combo_params['upper_r'], 1, shift, shift, np.pi, 2 * np.pi, search_range['upper_wavelength'], np.pi, 2*np.pi, search_range['upper_wavelength']]
        
        generator = ComboGratingGenerator(image_size, center, combo_params['mask_sigma'], mask1, mask2, standardization_params)

        from scipy import optimize
        def neg_model_activation(params):
            # Create image image
            params = [np.clip(p, l, u) for p, l, u in zip(params, lower_limits,
                                                        upper_limits)]
            image = generator(params)
            image = torch.as_tensor(image[None, None], dtype=torch.float32, device='cuda')
            # Compute activation
            with torch.no_grad():
                activation = model(image).item()
            return - activation

        optim_params, optim_images, optim_acts = [], [], []
        for seed in optgrating_params['seeds']:  # try 10 different random seeds
            print('Optimizing with seed =', seed)
            res = optimize.dual_annealing(neg_model_activation,
                                        bounds=list(zip(lower_limits, upper_limits)),
                                        maxiter=optgrating_params['num_iterations'],
                                        seed=seed, no_local_search=True)
            res = optimize.minimize(neg_model_activation, x0=res.x, method='Nelder-Mead')
            params = [np.clip(p, l, u) for p, l, u in zip(res.x, lower_limits, upper_limits)]
            params = np.round(params, 3)
            optim_params.append(dict(zip(param_keys, params)))
            image = generator(params)
            optim_images.append(image)
            image = torch.as_tensor(image[None, None], dtype=torch.float32, device='cuda')
            with torch.no_grad():
                activation = model(image).item()
            optim_acts.append(activation)

        # Create best combo grating 
        best_idx = np.argmin(optim_acts)
        best_params = optim_params[best_idx]
        best_image = optim_images[best_idx]
        best_activation = optim_acts[best_idx]

        # Insert
        self.insert1({**key, 'opt_params': optim_params, 'opt_acts': optim_acts, 
                    'best_params': best_params, 'best_image': best_image, 
                    'best_activation': best_activation, 'mei_activation': mei_activation})


@schema
class MEGParameters(dj.Lookup):
    definition = """ # parameters used to find the most exciting Gabor

    meg_params:         int
    ---  
    meg_seed:           int         # random seed used to create the initialization
    use_avg_model:      bool        # use the average model (across different training seeds) to create the MEG
    num_initializations: bool       # how many random images to optimize in parallel to create the MEG (output is the average)
    height:             int         # height of the MEG
    width:              int         # width of the MEG
    step_size:          float       # step size to use when generating the MEI
    num_iterations:     int         # number of optimization iterations for the annealing
    contrast:           float       # fixed contrast of the gabor image
    mean:               float       # fixed mean  of the gabor image
    fixed_mean:         bool        # whether to fix the mean of the gabor image
    """
    contents = [{'meg_params': 1, 'meg_seed': 12345, 'use_avg_model': True,
                 'num_initializations': 1, 'height': 36, 'width': 64, 'step_size': 0.1,
                 'num_iterations': 1000, 'contrast': 0, 'mean': 0, 'fixed_mean': 0}, 
                 {'meg_params': 2, 'meg_seed': 12345, 'use_avg_model': True,
                 'num_initializations': 1, 'height': 36, 'width': 64, 'step_size': 0.1,
                 'num_iterations': 1000, 'contrast': 0.25, 'mean': 0, 'fixed_mean': 1}]


@schema
class MEG(dj.Computed):
    definition = """ # find parameters that produce an optimal gabor for this unit

    -> static_models.Model
    -> base.Dataset.Unit
    -> MEGParameters
    -> SearchRange
    ---
    meg:                longblob    # best gabor image
    meg_activation:     float       # activation at the best gabor image
    meg_orientation:    float       # (radians) counterclockwise rotation to apply (0 is horizontal, pi/2 vertical)
    meg_phase:          float       # (radians) angle at which to start the sinusoid
    meg_wavelength:     float       # (px/height) wavelength of the sinusoid (1 / spatial frequency)
    meg_sigma:          float       # (px/height) sigma of the gaussian mask used
    meg_dx:             float       # (px/width) amount of translation in x (positive moves right)
    meg_dy:             float       # (px/height) amount of translation in y (positive moves downwards)
    """

    @property
    def key_source(self):
        all_keys = static_models.Model * base.Dataset.Unit * MEGParameters * SearchRange
        return all_keys & {'seed': 1009, 'search_range': 1}

    def make(self, key):
        # Get parameters
        meg_params = (MEGParameters & key).fetch1()

        # Get models
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     meg_params['use_avg_model'] else key)
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

        # Get search range
        search_range = (SearchRange & key).fetch1()
        lower_limits = [0, 0, *(search_range['lower_{}'.format(p)] for p in
                                ['wavelength', 'sigma', 'dx', 'dy'])]
        upper_limits = [np.pi, 2 * np.pi, *(search_range['upper_{}'.format(p)] for p in
                                            ['wavelength', 'sigma', 'dx', 'dy'])]

        # Create a random initial image from N(0, 0.25)
        torch.manual_seed(meg_params['meg_seed'])
        initial_params = 0.25 * torch.randn(meg_params['num_initializations'], 6,
                                            device='cuda')

        # Set up optimization
        change_range = ops.ChangeRange(
            torch.as_tensor(lower_limits, dtype=torch.float32, device='cuda'),
            torch.as_tensor(upper_limits, dtype=torch.float32, device='cuda'))
        generator = GaborGenerator(meg_params['height'], meg_params['width'])

        if meg_params['contrast'] == 0:
            meg_transform = utils.Compose([change_range, generator])
        else:
            if not meg_params['fixed_mean']:
                postup_op = ops.ChangeStd(float(meg_params['contrast']))
            else:
                postup_op = ops.ChangeStats(float(meg_params['contrast']), float(meg_params['mean']))
            meg_transform = utils.Compose([change_range, generator, postup_op])

        if meg_params['decay_factor'] != 0:
            gradient_f = ops.MultiplyBy(meg_params['decay_constant'], meg_params['decay_factor'])
        else:
            gradient_f = None

        # Optimize
        meg, fevals, _ = featurevis.gradient_ascent(model, initial_params,
                                                    transform=meg_transform,
                                                    gradient_f=gradient_f,
                                                    step_size=meg_params['step_size'],
                                                    num_iterations=meg_params[
                                                        'num_iterations'])

        # Create best gabor
        best_params = change_range(meg).mean(0)
        best_gabor = meg_transform(meg)
        best_activation = fevals[-1]

        # Insert
        self.insert1({**key, 'meg': best_gabor.squeeze().cpu().numpy(),
                      'meg_activation': best_activation,
                      'meg_orientation': best_params[0].item(),
                      'meg_phase': best_params[1].item(),
                      'meg_wavelength': best_params[2].item(),
                      'meg_sigma': best_params[3].item(), 'meg_dx': best_params[4].item(),
                      'meg_dy': best_params[5].item()})
    
@schema 
class TuningParameters(dj.Lookup):
    definition = """
    tuning_params: int
    ---
    gabor_parameter:       varchar(16) # single feature of gabor, including orientation, phase, wavelength, sigma, dx and dy
    values:                longblob    # values of the particular gabor feature   
    mask_image:            bool        # whether to mask image or not
    match_stats:           varchar(16) # standardize statistics of images within mask or full-field, can be 'mask' or 'ff' 
    """
    contents = [[1, 'orientation', np.linspace(0, np.pi, 20), 1, 'ff'], 
                [2, 'phase', np.linspace(0, 2*np.pi, 20), 1, 'ff']]

@schema
class GaborTuning(dj.Computed):
    definition = """  # Computed tuning to a particular feature using optimal gabor
    -> OptimalGabor
    -> TuningParameters
    -> base.MEIMask
    ---
    mei_activation:             float       # activation of standardized MEI
    opt_gabor_activation:       float       # activation of standardized optimal gabor
    opt_grating_activation:     float       # activation of full-field grating with theta, sigma, psi of the optimal gabor, standardized to mean 0 and std 1
    gabor_activations:          longblob    # array of activations of standardized gabors with a single varying parameter
    grating_activations:        longblob    # array of activations of standardized full-field gratings with a single varying parameter
    baseline_activation:        float       # activation to a blank image of all zeros
    """
    
    def make(self, key):
        mei_params = (base.MEIParameters & key).fetch1()
        tuning_params = (TuningParameters & key).fetch1()
        mask = (base.MEIMask & key).fetch1('mask')
        target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
        mei = (base.MEI & key).fetch1('mei')
        opt_gabor = (OptimalGabor & key).fetch1('opt_gabor')

        # generate gabors with different parameters
        opt_params = (OptimalGabor & key).fetch1('opt_orientation', 'opt_phase', 'opt_wavelength', 'opt_sigma', 'opt_dx', 'opt_dy')
        params = np.repeat(np.array(opt_params)[None], len(tuning_params['values']), axis=0)
        if tuning_params['gabor_parameter'] == 'orientation':
            params[:, 0] = tuning_params['values']
        elif tuning_params['gabor_parameter'] == 'phase':
            params[:, 1] = tuning_params['values']
        gabor_generator = GaborGenerator(mei_params['height'], mei_params['width'])
        grating_generator = GratingGenerator(mei_params['height'], mei_params['width'])

        # mei
        stan_mei = ops.standardize_image(mei, target_mean, target_std, mask, True, tuning_params['mask_image'], tuning_params['match_stats'])

        # optimal gabor and gabors with varying parameters
        gabors = gabor_generator(torch.as_tensor(params, dtype=torch.float32)).cpu().detach().squeeze().numpy()
        stan_opt_gabor = ops.standardize_image(opt_gabor, target_mean, target_std, mask, True, tuning_params['mask_image'], tuning_params['match_stats'])
        stan_gabors = ops.standardize_image(gabors, target_mean, target_std, mask, True, tuning_params['mask_image'], tuning_params['match_stats'])

        # optimal full-field grating
        opt_grating = grating_generator(torch.as_tensor(np.array(opt_params[:3])[None], dtype=torch.float32)).cpu().detach().squeeze().numpy()
        gratings = grating_generator(torch.as_tensor(params[:, :3], dtype=torch.float32)).cpu().detach().squeeze().numpy()   
        stan_opt_grating = ops.standardize_image(opt_grating, 0, 1, None, True, 0, tuning_params['match_stats'])
        stan_gratings = ops.standardize_image(gratings, 0, 1, None, True, 0, tuning_params['match_stats'])

        # make image tensors
        mei = torch.as_tensor(stan_mei[None, None], dtype=torch.float32, device='cuda').contiguous()    
        opt_gabor = torch.as_tensor(stan_opt_gabor[None, None], dtype=torch.float32, device='cuda').contiguous()
        opt_grating = torch.as_tensor(stan_opt_grating[None, None], dtype=torch.float32, device='cuda').contiguous()
        gabors = torch.as_tensor(stan_gabors[:, None], dtype=torch.float32, device='cuda').contiguous()
        gratings = torch.as_tensor(stan_gratings[:, None], dtype=torch.float32, device='cuda').contiguous()

        # Get models
        mei_params = (base.MEIParameters & key).fetch1()
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                    mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos')
                    else (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.as_tensor(mean_eyepos, dtype=torch.float32,
                                    device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos, neuron_idx=key['neuron_id'], device='cuda', average_batch=False)

        with torch.no_grad():
            mei_activation = model(mei).item()
            opt_gabor_activation = model(opt_gabor).item()
            opt_grating_activation = model(opt_grating).item()
            gabor_activations = model(gabors).cpu().detach().squeeze().numpy()
            grating_activations = model(gratings).cpu().detach().squeeze().numpy()
            baseline_activation = model(torch.zeros(1, 1, 36, 64, device='cuda')).cpu().detach().squeeze().numpy()

        self.insert1({**key, 'mei_activation': mei_activation, 
                             'opt_gabor_activation': opt_gabor_activation, 'opt_grating_activation': opt_grating_activation,
                             'gabor_activations': gabor_activations, 'grating_activations': grating_activations,
                             'baseline_activation': baseline_activation})
        
@schema
class GratingTuning(dj.Computed):
    definition = """  # Computed tuning to a particular feature using optimal grating
    -> OptimalGrating
    -> TuningParameters
    ---
    baseline_activation:  float     # activation to a blank image of all zeros
    activations:          longblob  # array of activations of standardized gratings
    """
    
    def make(self, key):
        optgrating_params = (OptimalGaborParameters & key).fetch1()
        tuning_params = (TuningParameters & key).fetch1()

        # generate gratings with differnt parameters
        opt_params = (OptimalGrating & key).fetch1('opt_orientation', 'opt_phase', 'opt_wavelength')
        params = np.repeat(np.array(opt_params)[None], len(tuning_params['values']), axis=0)
        if tuning_params['gabor_parameter'] == 'orientation':
            params[:, 0] = tuning_params['values']
        elif tuning_params['gabor_parameter'] == 'phase':
            params[:, 1] = tuning_params['values']
        generator = GratingGenerator(optgrating_params['height'], optgrating_params['width'])

        gratings = generator(torch.as_tensor(params, dtype=torch.float32)).cpu().detach().squeeze().numpy()
        stan_gratings = ops.standardize_image(gratings, 0, 1, None, True, 0, tuning_params['match_stats'])
        gratings = torch.as_tensor(stan_gratings[:, None], dtype=torch.float32, device='cuda').contiguous()

        # Get models
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                    optgrating_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos')
                    else (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.as_tensor(mean_eyepos, dtype=torch.float32,
                                    device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos, neuron_idx=key['neuron_id'], device='cuda', average_batch=False)

        with torch.no_grad():
            activations = model(gratings).cpu().detach().squeeze().numpy()
            baseline_activation = model(torch.zeros(1, 1, 36, 64, device='cuda')).cpu().detach().squeeze().numpy()
        self.insert1({**key, 'baseline_activation': baseline_activation, 'activations': activations})


