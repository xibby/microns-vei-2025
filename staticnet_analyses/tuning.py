import datajoint as dj
import numpy as np
from itertools import product
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import featurevis
from featurevis import models
from featurevis import ops
from featurevis import utils
from staticnet_analyses import base
from staticnet_analyses.base import MEIParameters, MaskParameters, MEI, MEIMask
from staticnet_analyses import utils as static_utils
from staticnet_experiments import models as static_models
from staticnet_invariance.deis_reconfigured import EvalParameters
from staticnet_invariance.segmentation import standardize_crops_v2, pass_img

stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
imagenet = dj.create_virtual_module('imagenet', 'pipeline_imagenet')

schema = dj.schema('neurostatic_tuning')
dj.config['enable_python_native_blobs'] = True
dj.config.setdefault('stores', dict())
dj.config['stores'].update({
    'static': dict(
        protocol='file', 
        location='/dj-stor01/neuro-static')
})


@schema
class SizeTuningParameters(dj.Lookup):
    definition = """
    size_params: int
    ---
    use_avg_model: bool             # whether to use the average model activation (across different training seeds)
    mask_type:   varchar(16)        # type of mask applied on images, including mei_mask, tight_mei_mask, etc.
    mask_params: longblob           # a dictionary of parameters used for creating masks
    image_type:  varchar(16)        # type of image used for measure size tuning, including mei, imagenet_oracle, etc.
    """
    contents = [[1, 1, 'mei_mask', dict(mei_params = [10], zscore_thresh = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5], closing_iters=[2], gaussian_sigma=[1.5]), 'mei'],
                [2, 1, 'mei_mask', dict(mei_params = [10], zscore_thresh = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5], closing_iters=[2], gaussian_sigma=[1.5]), 'imagenet_oracle']]

    @staticmethod
    def create_mask(neuron_key, mask_type, mask_params):
        if mask_type == 'mei_mask':
            from scipy import ndimage
            from skimage import morphology

            # Get mei and normalize 
            assert len(mask_params['mei_params']) == 1, 'more than one mei_params used for computing mask!'
            mei = (base.MEI & neuron_key & {'mei_params': mask_params['mei_params'][0]}).fetch1('mei')
            norm_mei = (mei - mei.mean()) / mei.std()
            
            # Get a list of parameter dictionaries
            params_ls = []
            for val_tuple in list(product(*mask_params.values())):
                params = {}
                for k, v in zip(list(mask_params.keys()), val_tuple):
                    params[k] = v
                params_ls.append(params)

            # Create masks
            masks = []
            for params in params_ls:  
                # z_score threshold
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
                
                masks.append(mask)

            return params_ls, masks

        else:
            raise NotImplementedError('This mask type have not been implemented yet!')

@schema
class MEISizeTuning(dj.Computed):
    definition = """
    -> base.MEI
    -> SizeTuningParameters
    ---
    activations:         longblob   # an array of activations to masked MEIs
    """

    @property
    def key_source(self):
         return base.MEI.proj() * (SizeTuningParameters & 'image_type = "mei"').proj()

    def make(self, key):
        # Fetch MEI and create masks
        tuning_params = (SizeTuningParameters & key).fetch1()
        neuron_key = key.copy()
        neuron_key.pop('mei_params')
        mei = (base.MEI & neuron_key).fetch1('mei')
        _, masks = SizeTuningParameters.create_mask(neuron_key, tuning_params['mask_type'], tuning_params['mask_params'])
        
        # Load model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     tuning_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                   device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda', average_batch=False)
        
        # Get masked image activation
        masked = torch.as_tensor(np.stack([mei * mask for mask in masks]), dtype=torch.float32, device='cuda')
        with torch.no_grad():
            activations = model(masked[:, None]).cpu().detach().squeeze().numpy()
        
        self.insert1({**key, 'activations': activations})
        

@schema
class NaturalSizeTuning(dj.Computed):
    definition = """
    -> stimulus.StaticImage.Image
    -> static_models.Model
    -> base.Dataset.Unit
    -> SizeTuningParameters
    ---
    activations:         longblob   # an array of activations to masked natural images
    """
    
    @property
    def key_source(self):
        oracles = stimulus.StaticImage.Image & (imagenet.Album.Oracle() & 'collection_id = 2 and image_class = "imagenet"')
        return oracles.proj() * static_models.Model.proj() * base.Dataset.Unit.proj() * (SizeTuningParameters & 'image_type = "imagenet_oracle"').proj() & (base.MEI & 'mei_params = 10')

    def make(self, key):
        # Fetch image and create masks
        import cv2
        tuning_params = (SizeTuningParameters & key).fetch1()
        image = (stimulus.StaticImage.Image & key).fetch1('image')
        # hardcoded, need to fix!
        image = cv2.resize(image, np.array([64, 36])).astype(np.float32)
        train_mean, train_std = (base.Dataset.TrainStats & key).fetch1('mean_img_value', 'std_img_value')
        image = (image - train_mean) / (train_std + 1e-9)
        _, masks = SizeTuningParameters.create_mask(key, tuning_params['mask_type'], tuning_params['mask_params'])
        
        # Load model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     tuning_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                   device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda', average_batch=False)
        
        # Get masked image activation
        masked = torch.as_tensor(np.stack([image * mask for mask in masks]), dtype=torch.float32, device='cuda')
        with torch.no_grad():
            activations = model(masked[:, None]).cpu().detach().squeeze().numpy()
        
        self.insert1({**key, 'activations': activations})

class GaborGenerator():
    def __init__(self, params, image_size=(36, 64), center=(0., 0.)):
        super().__init__()
        self.image_size = image_size
        self.center = center
        self.params = params
    
    @staticmethod
    def gabor(image_size, center, theta, Lambda, sigma, psi, gamma):
        ymax, xmax = image_size
        xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1), indexing='ij')
        sigma_x = sigma
        sigma_y = sigma / gamma
        # Rotation
        x_theta = (x - center[0]) * np.cos(theta) + (y - center[1]) * np.sin(theta)
        y_theta = -(x - center[0]) * np.sin(theta) + (y - center[1]) * np.cos(theta)
        grating = np.cos(2 * np.pi / Lambda * x_theta + psi)
        gauss = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
        gabor = gauss * grating
        return gabor, gauss, grating

    def __call__(self):
        gabor = self.gabor(self.image_size, self.center, *self.params.values())[0]
        return gabor

class PatternGenerator():
    """
    Combined Plaid generator class.
    Params:
        gabor_params: a dictionary of gabor parameters with keys in [theta, Lambda, sigma, psi, gamma]
            theta (float): Orientation of the sinusoid (in radian).
            sigma (float): std deviation of the Gaussian of a gabor.
            Lambda (float): Sinusoid wavelengh (1/frequency) of a gabor.
            psi (float): Phase of the sinusoid of a gabor.0
            gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts like an aspect ratio of the Gaussian) of a gabor.

        edge_params: a dictionary of the component edge parameters with keys in [edge_center, edge_theta, edge_sigma]
            edge_center (tuple): (x, y) position of the center point around which the edge between the two component plaids rotates.
            edge_theta (float): Orientation of the edge between the two plaids (in radian).
            edge_sigma: the level of Gaussian blurring applied on the edge between the two plaids.
    ---
    Returns: 
        combined plaid pattern (with two components of the plaid always 90 degrees apart)
    """
    
    def __init__(self, gabor1_params, gabor2_params, edge_params, image_type='grating', image_size=(36, 64), center=(0., 0.)):
        super().__init__()
        self.image_size = image_size
        self.center = center
        self.gabor1_params = gabor1_params
        self.gabor2_params = gabor2_params
        self.edge_params = edge_params
        self.image_type = image_type
        
    def plaid(self, gabor_params):
        idx = 0 if self.image_type == 'gabor' else -1
        img1 = GaborGenerator.gabor(self.image_size, self.center, *gabor_params.values())[idx]
        shifted_params = gabor_params
        shifted_params['theta'] += np.pi / 2
        img2 = GaborGenerator.gabor(self.image_size, self.center, *shifted_params.values())[idx]
        return img1 + img2

    def get_mask(self):
        edge_center, edge_theta, edge_sigma = self.edge_params.values()
        ymax, xmax = self.image_size
        xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1), indexing='ij')

        px, py = edge_center
        qx, qy = (px + 1, py - np.tan(edge_theta))
        mask1 = (qx - px) * (y - py) - (qy - py) * (x - px)
        mask1 = 1 / (1 + np.exp(- mask1) + 1e-9)
        if ((edge_theta - np.pi/2) // np.pi) % 2:
            temp = np.ones_like(mask1) - mask1
            mask1 = temp
        mask2 = np.ones_like(mask1) - mask1
        
        # blur edge of the masks
        blur = ops.GaussianBlur(edge_sigma)
        mask1 = blur(torch.as_tensor(mask1[None, None], dtype=torch.float32)).squeeze().numpy()
        mask2 = blur(torch.as_tensor(mask2[None, None], dtype=torch.float32)).squeeze().numpy()
        return mask1, mask2
    
    def comb_plaid(self):    
        idx = 0 if self.image_type == 'gabor' else -1
        plaid1 = self.plaid(self.gabor1_params)
        plaid2 = self.plaid(self.gabor2_params)
        mask1, mask2 = self.get_mask()
        return plaid1 * mask1 + plaid2 * mask2
        
    def __call__(self):
        return self.comb_plaid()

@schema
class TuningParameters(dj.Lookup):
    definition = """
    tuning_params:        int 
    ---
    height:               int
    width:                int
    image_type:           varchar(64)
    component_type:       varchar(64) 
    parameters:           varchar(64)
    range:                longblob
    description:          varchar(255)
    """
    
    @staticmethod
    def get_f_ratios(drop_fold=1.5, n=20, max_ratio=1, min_ratio=0.2):
        n_per_fold = n / 4
        drop = (max_ratio - min_ratio) / (n_per_fold + n_per_fold*drop_fold + n_per_fold*(drop_fold**2) + (n_per_fold-1)*(drop_fold**3))
        f_ratios = []
        temp = [1]
        for i in range(4):
            temp_drop = drop * (1.5**i)
            start = temp[-1]
            temp = [start - j * temp_drop for j in range(6)]
            f_ratios.extend(temp[:-1])
        return f_ratios
    
    @staticmethod
    def lambda_pair(lb=4, ub=20, mid=8, ratios=np.arange(1, 6)):
        """
        lb: lower bound of lambda
        mid: lambda for ratio 1:1
        ratios: a series of lambda ratios 
        """
        k = (mid - ub) / (mid - lb)
        res = ub - k * lb
        lam1s = res / (ratios - k)
        lam2s = lam1s * ratios
        return lam1s, lam2s

    # f_ratios = TuningParameters.get_f_ratios()
    # Lambda1, Lambda2 = TuningParameters.lambda_pair(ratios = 1 / np.array(f_ratios))
    # dic = (2, 36, 36, 'CombPlaid', 'grating', 'theta|Lambda1|Lambda2|psi1|psi2', 
    #             dict(theta = np.linspace(np.pi/2 + 1e-9, np.pi*5/2, 41)[:-1],
    #                 Lambda1 = Lambda1,
    #                 Lambda2 = Lambda2,
    #                 psi1 = np.linspace(0, np.pi*2, 11)[:-1],
    #                 psi2 = np.linspace(0, np.pi*2, 11)[:-1],
    #                 ),
    #             'tuning on 2D space of wavelegnth ratio vs edge orientation'
    #             )
    # TuningParameters.insert1(dic)

@schema
class TuningDataset(dj.Lookup):
    definition = """
    -> TuningParameters
    image_id:      int
    ---
    image:         blob@static
    parameters:    longblob
    """
    
    def fill_CompPlaid(self, key):
        # h, w, component_type, param_range = (TuningParameters & key).fetch1('height', 'width', 'component_type', 'range')
        h, w = 36, 36  
        component_type, param_range = (TuningParameters & key).fetch1('image_type', 'range')
        thetas, Lambda1s, Lambda2s, psi1s, psi2s = param_range.values()

        images = []
        param_combos = []
        for theta in thetas:
            for lam1, lam2 in zip(Lambda1s, Lambda2s):
                for psi1, psi2 in product(psi1s, psi2s):
                    params1 = dict(theta=theta - np.pi/2, Lambda=lam1, sigma=10, psi=psi1, gamma=1)
                    params2 = dict(theta=theta, Lambda=lam2, sigma=10, psi=psi2, gamma=1)
                    edge_params = dict(edge_center=(0, 0), edge_theta=theta, edge_sigma=1.5)
                    generator = PatternGenerator(params1, params2, edge_params, image_type=component_type, image_size=(h, w))
                    images.append(generator())
                    param_combos.append([theta, lam1, lam2, psi1, psi2])
        for i, (im, param) in tqdm(enumerate(zip(images, param_combos))):
            self.insert1(dict(**key, image_id=i, image=im, parameters=np.array(param)))
        

@schema
class CombPlaidTuning2(dj.Computed):
    definition = """
    -> base.MEIMask
    -> TuningParameters
    -> EvalParameters
    ---
    activations:   blob@static   # 1d array of activations to masked standardized images in length of number of parameters (i.e. num_thetas * num_Lambda_ratios * num_psi1s * num_psi2s)
    """
    
    def make(self, key):
        eval_params = (EvalParameters & key).fetch1()
        mei_params = (MEIParameters & key).fetch1()
        
        # get images
        images = (TuningDataset & key).fetch('image', order_by='image_id')
        dataloader = DataLoader(images, batch_size=64, drop_last=False)
        iterator = iter(dataloader)
        
        # load model 
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if \
                    mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos')
                    else (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda', average_batch=False)
        
        # standardize and pass images
        activations = []
        with torch.no_grad():
            for img_batch in tqdm(iterator):
                img_batch = standardize_crops_v2(img_batch.detach().cpu().squeeze().numpy(), eval_params, key)
                activations.extend(pass_img(img_batch, model, 'cuda', batch_size=64))

        # insert result
        self.insert1(dict(**key, activations = np.array(activations)))
                                                                                   