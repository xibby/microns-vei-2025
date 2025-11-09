import datajoint as dj
import featurevis
import numpy as np
import itertools
from functools import partial
from itertools import product, repeat
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from attorch.layers import SpatialTransformerPyramid2d, Elu1
from staticnet.cores import GaussianLaplaceCore
from attorch.losses import PoissonLoss
from staticnet_experiments.utils import gitlog, set_seed
from attorch.train import early_stopping
from attorch.losses import PoissonLoss
from staticnet_experiments.utils import corr

from neuro_data.static_images.data_schemas import ImageNetSplit, Preprocessing, process_frame
from neuro_data.static_images import datasets
from staticnet_experiments import configs, models
from neuro_data import logger as log
from torch.utils.data import DataLoader
from neuro_data.static_images.transforms import Subsample, Normalizer, ToTensor
from neuro_data.utils.sampler import SubsetSequentialSampler, BalancedSubsetSampler
from torch.utils.data.sampler import SubsetRandomSampler

from featurevis import ops
from featurevis import utils
from featurevis.utils import varargin
from staticnet_analyses import base, gabors
from staticnet_analyses.base import MaskParameters
# from staticnet_invariance.deis import HingeLoss, prepare_params, MaskStatsParameters, DEIParameters,  DEIThreshold, \
# CombinedCriterion, get_variable_masks, LinearImageModel, cal_texture_minimum_shape, train_texture, TextureParameters, TextureScoreParameters
from staticnet_vae import vae

imagenet = dj.create_virtual_module('imagenet', 'pipeline_imagenet')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')

schema = dj.schema('neurostatic_toy')

dj.config.setdefault('stores', dict())
dj.config['stores'].update({
    'toy': dict(
        protocol='file', 
        location='/dj-stor01/neuro-static')
})
dj.config['enable_python_native_blobs'] = True

class GaborGenerator():
    def __init__(self, params, image_size=(36, 64), center=(0., 0.)):
        super().__init__()
        self.image_size = image_size
        self.center = center
        self.params = params
    
    @staticmethod
    def gen_gabor(image_size, center, shift_x, shift_y, theta, Lambda, sigma, psi, gamma):
        ymax, xmax = image_size
        xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1), indexing='ij')
        sigma_x = sigma
        sigma_y = sigma / gamma
        # Rotation
        x_theta = (x - (center[0] - shift_x)) * np.cos(theta) + (y - (center[1] - shift_y)) * np.sin(theta)
        y_theta = -(x - (center[0] - shift_x)) * np.sin(theta) + (y - (center[1] - shift_y)) * np.cos(theta)
        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
        return gb

    def __call__(self):
        gb = self.gen_gabor(self.image_size, self.center, *self.params)
        return gb

class CombGaborGenerator():
    def __init__(self, params, image_size=(36, 64), center=(0., 0.), mask_sigma=1):
        super().__init__()
        self.image_size = image_size
        self.center = center
        self.params = params
        self.mask_sigma = mask_sigma

    def gen_comb_gabor(self, gabor1, gabor2, r, a, shift_x, shift_y):    
        ymax, xmax = self.image_size
        Y,X = np.arange(ymax)[:, None], np.arange(xmax)[None, :]
        a *= 2 * np.pi
        cy, cx = ymax/2 + self.center[1] - r * np.cos(a) + shift_y, xmax/2 + self.center[0] - r * np.sin(a) + shift_x
        d = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        mask1 = torch.as_tensor(d > r, dtype=torch.float32, device='cuda')
        mask2 = torch.as_tensor(d <= r, dtype=torch.float32, device='cuda')
        blur = ops.GaussianBlur(self.mask_sigma)
        blur_mask1 = blur(mask1[None, None])
        blur_mask2 = blur(mask2[None, None])
        gabor1 = (torch.as_tensor(gabor1[None, None], dtype=torch.float32, device='cuda') * blur_mask1).cpu().detach().squeeze().numpy()
        gabor2 = (torch.as_tensor(gabor2[None, None], dtype=torch.float32, device='cuda') * blur_mask2).cpu().detach().squeeze().numpy()
        
        return gabor1, gabor2, gabor1 + gabor2

    def __call__(self):
        r, a, shift_x, shift_y, theta1, Lambda1, sigma1, psi1, gamma1, theta2, Lambda2, sigma2, psi2, gamma2 = self.params
        gb1 = GaborGenerator.gen_gabor(self.image_size, self.center, self.center[0], self.center[1], theta1, Lambda1, sigma1, psi1, gamma1)
        gb2 = GaborGenerator.gen_gabor(self.image_size, self.center, self.center[0], self.center[1], theta2, Lambda2, sigma2, psi2, gamma2)
        gb1, gb2, comb_gb = self.gen_comb_gabor(gb1, gb2, r, a, shift_x, shift_y)
        return gb1, gb2, comb_gb

class SimpleModel(nn.Module):
    def __init__(self, params, normalize=False, average_batch=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), image_size=[36, 64], center=[0, 0], scale_by=1, **kwargs):
        super().__init__()
        self.device = device
        if len(params) == 7:
            filter = torch.as_tensor(GaborGenerator(params, image_size, center)()[None, None], dtype=torch.float32, device=self.device)
        elif len(params) == 6:
            filter = gabors.GaborGenerator(*image_size)(torch.as_tensor(params, dtype=torch.float32, device=self.device)[None])
        if normalize:
            self.register_buffer('filter', ops.ChangeStats(1, 0)(filter))
        else:
            self.register_buffer('filter', filter)
        self.average_batch = average_batch
        self.scale_by = scale_by
        
    def forward(self, x):
        self.input = x.clone().to(self.device)
        y = F.relu(F.conv2d(self.input, self.filter, bias=None))
        if self.average_batch:
            y = y.mean(0)
        return y.squeeze() / self.scale_by

class ComplexModel(nn.Module):
    def __init__(self, params, sqrt=True, normalize=False, average_batch=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), image_size=[36, 64], center=[0, 0], scale_by=1):
        super().__init__()
        self.device = device
        if len(params) == 7:
            filter1 = torch.as_tensor(GaborGenerator(params, image_size, center)()[None, None], dtype=torch.float32, device=self.device)
            params_copy = params.copy()
            params_copy[-2] += np.pi/2
            filter2 = torch.as_tensor(GaborGenerator(params_copy, image_size, center)()[None, None], dtype=torch.float32, device=self.device)
        elif len(params) == 6:
            filter1 = gabors.GaborGenerator(*image_size)(torch.as_tensor(params, dtype=torch.float32, device=self.device)[None])       
            params_copy = params.copy()
            params_copy[1] += np.pi/2
            filter2 = gabors.GaborGenerator(*image_size)(torch.as_tensor(params_copy, dtype=torch.float32, device=self.device)[None]) 
        if normalize:
            self.register_buffer('filter1', ops.ChangeStats(1, 0)(filter1))
            self.register_buffer('filter2', ops.ChangeStats(1, 0)(filter2))
        else:
            self.register_buffer('filter1', filter1)
            self.register_buffer('filter2', filter2)
        self.average_batch = average_batch
        self.scale_by = scale_by
        self.sqrt = sqrt

    def forward(self, x):
        self.input = x.clone().to(self.device)
        y1 = F.conv2d(self.input, self.filter1, bias=None)
        y2 = F.conv2d(self.input, self.filter2, bias=None) 
        y = torch.sqrt(y1.pow(2) + y2.pow(2)) if self.sqrt else y1.pow(2) + y2.pow(2)
        if self.average_batch:
            y = y.mean(0)
        return y.squeeze() / self.scale_by

class CombinedFullyOverlappedModel(nn.Module):
    def __init__(self,params,average_batch=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), image_size=[36, 64], center=[0, 0], scale_by=1):
        super().__init__()
        self.simple_fraction =  params[-1].item()
        self.simple_model = SimpleModel(params[:-1],average_batch=average_batch,device=device,image_size=image_size,center=center,scale_by=scale_by)
        self.complex_model = ComplexModel(params[:-1],average_batch=average_batch,device=device,image_size=image_size,center=center,scale_by=scale_by)

    def forward(self,x):
        return self.simple_model(x) * self.simple_fraction + self.complex_model(x) * (1-self.simple_fraction)

class CombGaborModel(nn.Module):
    def __init__(self, params, cell_type='simple-complex', average_batch=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), image_size=[36, 64], center=[0, 0], scale_by=1):
        super().__init__()
        self.device = device
        self.cell_type = cell_type
        filter1, filter2, _ = CombGaborGenerator(params, image_size, center)()
        filter1 = torch.as_tensor(filter1[None, None], dtype=torch.float32, device=self.device)
        filter2 = torch.as_tensor(filter2[None, None], dtype=torch.float32, device=self.device)
        self.register_buffer('filter1', filter1)
        self.register_buffer('filter2', filter2)
        if self.cell_type != 'simple-simple':
            params_copy = params.copy()
            params_copy[-7] += np.pi/2 # hard code to modify psi of gabor1
            params_copy[-2] += np.pi/2 # hard code to modify psi of gabor2
            filter3, filter4, _ = CombGaborGenerator(params_copy, image_size, center)()
            filter3 = torch.as_tensor(filter3[None, None], dtype=torch.float32, device=self.device)
            filter4 = torch.as_tensor(filter4[None, None], dtype=torch.float32, device=self.device)
            self.register_buffer('filter3', filter3)
            self.register_buffer('filter4', filter4)
        self.average_batch = average_batch
        self.scale_by = scale_by

    def forward(self, x):
        self.input = x.clone().to(self.device)
        y1 = F.conv2d(self.input, self.filter1, bias=None)
        y2 = F.conv2d(self.input, self.filter2, bias=None) 
        if self.cell_type == 'simple-simple':
            y = (F.relu(y1) + F.relu(y2)) / 2
        else:
            y3 = F.conv2d(self.input, self.filter3, bias=None)
            y4 = F.conv2d(self.input, self.filter4, bias=None) 
            if self.cell_type == 'simple-complex':
                y = (torch.sqrt(y1.pow(2) + y3.pow(2)) + F.relu(y2)) / 2
            elif self.cell_type == 'complex-complex':
                y = (torch.sqrt(y1.pow(2) + y3.pow(2)) + torch.sqrt(y2.pow(2) + y4.pow(2))) / 2

        if self.average_batch:
            y = y.mean(0)

        return y.squeeze() / self.scale_by

@schema
class CombGaborSearchRange(dj.Lookup):
    definition = """ # search range for CombGabor parameters
    range_id:       int         # id of this search range
    ---
    lower_r:            float
    upper_r:            float
    lower_a:            float
    upper_a:            float
    lower_shift_x:      float
    upper_shift_x:      float
    lower_shift_y:      float
    upper_shift_y:      float
    lower_theta:        float
    upper_theta:        float
    lower_lambda:       float
    upper_lambda:       float
    lower_sigma:        float
    upper_sigma:        float
    lower_psi:          float
    upper_psi:          float
    lower_gamma:        float
    upper_gamma:        float
    """
    contents = [[1, 5, 30, 0, 1, -5, 5, -5, 5, 0, np.pi, 5, 20, 4, 4, 0, 0, 1, 1],
                [2, 5, 30, 0, 1, 0, 3, 0, 3, 0, np.pi, 5, 20, 4, 4, 0, 0, 1, 1]]
    
    @staticmethod
    def select_toy_params(range_id=1, n_neurons=10, seed=1009):
        from itertools import product
        search_range = (CombGaborSearchRange & {'range_id': range_id}).fetch1()
        n_samples = [5, 5, 3, 3, 5, 5, 1, 1, 1, 5, 5, 1, 1, 1]
        lower_limits = [*(search_range['lower_{}'.format(p)] for p in ['r', 'a', 'shift_x', 'shift_y', 'theta', 'lambda', 'sigma', 'psi', 'gamma']), \
            *(search_range['lower_{}'.format(p)] for p in ['theta', 'lambda', 'sigma', 'psi', 'gamma'])]
        upper_limits = [*(search_range['upper_{}'.format(p)] for p in ['r', 'a', 'shift_x', 'shift_y', 'theta', 'lambda', 'sigma', 'psi', 'gamma']), \
            *(search_range['upper_{}'.format(p)] for p in ['theta', 'lambda', 'sigma', 'psi', 'gamma'])] 
        grids = [np.linspace(l, u, n) for l, u, n in zip(lower_limits, upper_limits, n_samples)]
        params = np.stack(list(product(*grids)))
        np.random.seed(seed)
        idxs = np.random.choice(np.arange(len(params)), n_neurons, replace=False)
        selected_params = params[idxs]
        return selected_params

@schema
class ToyNeuron(dj.Lookup):
    definition = """
    neuron_type: varchar(16)
    neuron_id:   int
    ---
    parameters:  longblob
    """
    contents = [['Simple', 1, [0, 0, 0, 5, 3, 3.14, 1]],
                ['Complex', 1, [0, 0, 0, 5, 3, 3.14, 1]],
                ['Simple', 2, [0, 0, 0, 5, 4, 3.14, 1]],
                ['Complex', 2, [0, 0, 0, 5, 4, 3.14, 1]],
                ['Simple', 3, [0, 0, 1.57, 10, 4, 1.57, 1]],
                ['Complex', 3, [0, 0, 1.57, 10, 4, 1.57, 1]],]

@schema
class NoiseType(dj.Lookup):
    definition = """
    noise_type: varchar(16)
    """
    contents = [['poisson'], ['none']]

@schema
class ResponseNormalization(dj.Lookup):
    definition = """
    -> imagenet.Album
    -> Preprocessing
    """

    class Neuron(dj.Part):
        definition = """
        -> master
        -> ToyNeuron
        ----
        response_mean:  float # mean response to all training set imagenet images in a certain album 
        response_std:   float # standard deviation of response to all training set imagenet images in a certain album 
        """

    def fill(self, key=dict(collection_id=3, image_class='imagenet', preproc_id=9)):
        self.insert1(key, skip_duplicates=True)
        images = (ImageNetSplit * stimulus.StaticImage.Image * imagenet.Album.Single & key & 'tier = "train"').fetch('image')
        images = torch.as_tensor(np.stack([process_frame(key, im) for im in images]), dtype=torch.float32, device='cuda')
        if key['preproc_id'] in (5, 9):
            train_mean = images.mean(dim=(1,2)).mean()
            train_std = images.std(dim=(1,2)).mean()
        images = (images - train_mean) / train_std
        for neuron_key in ToyNeuron.fetch(as_dict=True):
            model = eval('{}Model'.format(neuron_key['neuron_type']))(neuron_key['parameters'], average_batch=False)
            resps = model(images[:, None]).cpu().detach().squeeze().numpy()
            response_mean = resps.mean()
            response_std = resps.std()
            self.Neuron.insert1({**key, **neuron_key, 'response_mean': response_mean, 'response_std': response_std}, ignore_extra_fields=True, skip_duplicates=True)

@schema
class InputResponse(dj.Computed):
    definition = """
    -> ToyNeuron
    -> NoiseType
    -> imagenet.Album
    -> Preprocessing
    ---
    responses:  longblob      # vector of noisy responses to all images in certain album    
    """
    
    class Input(dj.Part):
        definition = """
        -> master
        row_id:   int         # row id in the response block
        --- 
        image_id: int         # imagenet image_id
        """

    def make(self, key):
        # # preprocess images
        # tiers = ['train', 'validation', 'test']
        # images, image_ids = [], []
        # for tier in tiers:
        #     if tier == 'test':
        #         ims, ids = (ImageNetSplit * stimulus.StaticImage.Image * imagenet.Album.Oracle & key & {'tier': tier}).fetch('image', 'image_id', order_by='image_id')
        #         images.append(np.tile(np.stack(ims), (10, 1, 1)))
        #         image_ids.append(np.tile(ids, 10))
        #     else:
        #         ims, ids = (ImageNetSplit * stimulus.StaticImage.Image * imagenet.Album.Single & key & {'tier': tier}).fetch('image', 'image_id', order_by='image_id')
        #         images.append(np.stack(ims))
        #         image_ids.append(ids)
        # images = np.vstack(images)
        # image_ids = np.hstack(image_ids)
        # images = torch.as_tensor(np.stack([process_frame(key, im) for im in images]), dtype=torch.float32, device='cuda')
        
        # hack to use preprocessed images
        import pickle
        with open('/dj-stor01/users/zhiwei/album3_preprocessed_images.pickle', 'rb') as handle:
            im_dic = pickle.load(handle)
        images = torch.as_tensor(im_dic['images'], dtype=torch.float32, device='cuda')
        image_ids = im_dic['image_ids']
        
        # get toy neuron responses
        neuron_type, params, normalize, sqrt = (ToyNeuron & key).fetch1('neuron_type', 'parameters', 'normalize', 'sqrt')
        noise_type = (NoiseType & key).fetch1('noise_type')
        if neuron_type in ['Simple', 'Complex']:
            model = eval('{}Model'.format(neuron_type))(params, normalize=normalize, sqrt=sqrt, average_batch=False)
        elif neuron_type in ['simple-simple', 'simple-complex', 'complex-complex']:
            model = CombGaborModel(params, neuron_type, average_batch=False)
        elif neuron_type == 'combined_full_overlap':
            model = CombinedFullyOverlappedModel(params, average_batch=False)
        resps = model(images[:, None]).cpu().detach().squeeze().numpy()
        # add noise to responses
        if noise_type == 'none':
            noisy_resps = resps
        elif noise_type == 'poisson':
            noisy_resps = np.random.poisson(resps).astype(np.float32)

        self.insert1({**key, 'responses': noisy_resps})
        for i, id in enumerate(image_ids):
            self.Input.insert1({**key, 'row_id': i, 'image_id': id})

@schema
class GroupAssignment(dj.Lookup):
    definition = """
    group_id: int
    """
    class Member(dj.Part):
        definition = """
        -> master
        member_id: int
        ---
        -> ToyNeuron
        -> NoiseType
        """

from neuro_data.utils.data import h5cached
@h5cached('/dj-stor01/cache/', mode='array', transfer_to_tmp=False,
          file_format='static-toy-group{group_id}-{noise_type}-{collection_id}-{preproc_id}.h5')
@schema
class DatasetAssignment(dj.Computed): # noise_type is not correct in this table, please refer to GroupAssignment.Member
    definition = """
    -> GroupAssignment
    -> NoiseType
    -> imagenet.Album
    -> Preprocessing
    """
    
    @property
    def key_source(self):
        # noise_type should be a neuron-level property instead of dataset level. Ignore noise_type in this table, refer to GroupAssignment.Member.
        return GroupAssignment * NoiseType * imagenet.Album * Preprocessing & InputResponse #& 'noise_type = "poisson"'

    def make(self, key):
        self.insert1(key)
    
    def compute_data(self, key):
        # key.pop('noise_type')
        member_keys = (InputResponse * GroupAssignment.Member & key).fetch('KEY', order_by='member_id')
        responses = []
        for member_key in member_keys:
            responses.append((InputResponse & member_key).fetch1('responses'))
        responses = np.stack(responses).T

        images, tiers = (ImageNetSplit * stimulus.StaticImage.Image * InputResponse.Input & member_keys[0]).fetch('image', 'tier', order_by='row_id')
        images = np.stack([process_frame(member_key, im) for im in images])[:, None]

        def run_stats(selector, ix, item):
            ret = {}
            data = selector(ix)
            if item == 'input':
                ret['all'] = dict(
                    mean=data.mean(axis=(-1, -2)).mean().astype(np.float32),
                    std=data.std(axis=(-1, -2)).mean().astype(np.float32),
                    min=np.min(data, axis=(-1, -2)).mean().astype(np.float32),
                    max=np.max(data, axis=(-1, -2)).mean().astype(np.float32),
                    median=np.median(data, axis=(-1, -2)).mean().astype(np.float32)
                )
            elif item == 'response':
                ret['all'] = dict(
                        mean=data.mean(axis=0).astype(np.float32),
                        std=data.std(axis=0, ddof=1).astype(np.float32),
                        min=data.min(axis=0).astype(np.float32),
                        max=data.max(axis=0).astype(np.float32),
                        median=np.median(data, axis=0).astype(np.float32)
                    )
            return ret
        # --- compute statistics
        log.info('Computing statistics on training dataset')
        response_statistics = run_stats(lambda ix: responses[ix], tiers == 'train', 'response')
        input_statistics = run_stats(lambda ix: images[ix], tiers == 'train', 'input')
        statistics = dict(
            images=input_statistics,
            responses=response_statistics
        )
        dset = dict(images = images, responses = responses, tiers = tiers.astype('S'), statistics = statistics)
        return dset
    
    def fetch_data(self, key):
        dkey = (self & key).fetch1('KEY')
        data_names = ['images', 'responses'] 
        log.info('Data will be ({})'.format(','.join(data_names)))
        h5filename = DatasetAssignment().get_filename(dkey)
        log.info('Loading dataset --> {}'.format(h5filename))
        dset = datasets.StaticImageSet(h5filename, *data_names)
        return dset

    @staticmethod
    def add_transforms(dset):
        transforms = []
        transforms.append(Normalizer(dset))
        transforms.append(ToTensor())
        dset.transforms = transforms
        return dset

    def load_data(self, key, tier, batch_size=60, balanced=False, cuda=True):
        dset = self.fetch_data(key)
        dset = self.add_transforms(dset)
        for tr in dset.transforms:
            if isinstance(tr, ToTensor):
                tr.cuda = cuda

        if tier == 'train':
            if not balanced:
                Sampler = SubsetRandomSampler
            else:
                Sampler = BalancedSubsetSampler
        else:
            Sampler = SubsetSequentialSampler
            
        ix = np.where(dset.tiers == tier)[0]
        sampler = Sampler(ix)
        data_loader = DataLoader(dset, sampler=sampler, batch_size=batch_size)
        
        return dset, data_loader

#========================= CNN models =================================
class SpatialTransformerPyramid2dRO(SpatialTransformerPyramid2d):
    def __init__(self, gamma_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_features = gamma_features
        
    def initialize_mu(self, initial_mu):
        if self.bias is not None:
            self.bias.data = initial_mu.squeeze() - 1
            
    def regularizer(self):
        return self.feature_l1() * self.gamma_features

class CorePlusRO2d(nn.Module):
    def __init__(self, core, readout, nonlinearity=None):
        super().__init__()
        self.core = core
        self.readout = readout
        self.nonlinearity = log1exp if nonlinearity is None else nonlinearity
        
    @staticmethod
    def get_readout_in_shape(core, in_shape):
        mov_shape = in_shape[1:]
        core.eval()
        tmp = Variable(torch.from_numpy(np.random.randn(1, *mov_shape).astype(np.float32)))
        nout = core(tmp).size()[1:]
        core.train(True)
        return nout
    
    def forward(self, x):
        return self.nonlinearity(self.readout(self.core(x)))
        
def build_network(key, dataset):
    img_shape = dataset.images.shape
    n_neurons = dataset.responses.shape[-1]

    core_params = (configs.CoreConfig.GaussianLaplace() & key).fetch1()
    core_params.pop('core_hash')
    core = GaussianLaplaceCore(input_channels=img_shape[1], **core_params)
    ro_in_shape = CorePlusRO2d.get_readout_in_shape(core, img_shape)

    ro_params = (configs.ReadoutConfig.SpatialTransformerPyramid2d() & key).fetch1()
    ro_params.pop('ro_hash')
    ro = SpatialTransformerPyramid2dRO(in_shape=ro_in_shape, outdims=n_neurons, **ro_params)    

    return CorePlusRO2d(core, ro, nonlinearity=Elu1())

#====================================Train model=====================================
def compute_predictions(model, loader):
    y, y_hat = [], []
    for x_val, y_val in loader:
        y_mod = model(x_val).data.cpu().numpy()
        y.append(y_val.cpu().numpy())
        y_hat.append(y_mod)
    return np.vstack(y), np.vstack(y_hat)

def correlation_closure(model, loader, avg=True, print_val=True):
    train = model.training
    model.eval()
    y, y_hat = compute_predictions(model, loader)
    score = corr(y, y_hat, axis=0)
    if print_val:
        print('correlation: {:.4f}'.format(score.mean()))
    model.train(train)
    if avg:
        return score.mean()
    else:
        return score

class Trainer():
    def objective(self, model, inputs, targets, loss_type='PoissonLoss'):
        criterion = eval(loss_type)()
        outputs = model(inputs)
        return criterion(outputs, targets) \
                + model.core.regularizer() \
                + model.readout.regularizer()
    
    def train_helper(self, model, objective, optimizer, stop_closure, trainloader, epoch=0,
            interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
            restore_best=True, accumulate_gradient=1):
        optimizer.zero_grad()
        iteration = 0

        for epoch, val_obj in early_stopping(model, stop_closure,
                                             interval=interval, patience=patience,
                                             start=epoch, max_iter=max_iter, maximize=maximize,
                                             tolerance=tolerance, restore_best=restore_best):
            for inputs, targets in tqdm(trainloader):
                obj = objective(model, inputs, targets)
                obj.backward()
                optimizer.step()
                optimizer.zero_grad()
                iteration += 1
        return model, epoch
    
    def train(self, train_params, model, trainloader, valloader):
        # initialize
        initial_mu = trainloader.dataset.transformed_mean().responses
        model.readout.initialize_mu(initial_mu)
        model.core.initialize()

        # # hack some parameters
        # key = train_params
        # key['finetune'] = key['max_epoch']
        # key['max_epoch'] = 500
        # key['transfer_group_id'] = 272
        # key['transfer_net_hash'] = "8b6fe18fa651ebf452db0fbd77d05a01"
        # key['transfer_seed'] = 1009

        # # load pre-trained transfer core state_dict
        # init_model_dict = model.state_dict()
        # state_dict = (models.Model & dict(group_id=key['transfer_group_id'], net_hash=key['transfer_net_hash'], seed=key['transfer_seed'])).fetch1('model')
        # try:
        #     state_dict = {k: torch.as_tensor(state_dict[k][0].copy()) for k in state_dict.dtype.names if 'core' in k}
        # except AttributeError:
        #     state_dict = {k: torch.as_tensor(state_dict[k].copy()) for k in state_dict.keys() if 'core' in k}
        # init_model_dict.update(state_dict)
        # model.load_state_dict(init_model_dict, strict=False)

        # if not key['finetune']: # freeze core parameters
        #     print('Freezing pre-trained core')
        #     for param in model.core.parameters():
        #         param.requires_grad = False

        
        # --- train
        log.info('Shipping model to GPU')
        model = model.cuda()
        model.train(True)
        print(model)
        epoch = 0

        schedule = train_params['schedule']
        model.shift = True
        for opt, lr in zip(repeat(torch.optim.Adam), schedule):
            log.info('Training with learning rate {}'.format(lr))

            optimizer = opt(model.parameters(), lr=lr)

            model, epoch = self.train_helper(model, self.objective, optimizer,
                               partial(correlation_closure, loader=valloader), trainloader,
                               epoch=epoch, max_iter=train_params['max_epoch'], patience=10)
        model.eval()
        return model

@schema
class NetworkConfig(dj.Lookup):
    definition = """
    config_id:   int
    ---
    config_type: varchar(32)
    config_params: longblob
    """
    contents = [[1, 'CorePlusRO2d', dict(core_hash='28bc2fa358337c5012278f899b5b6947', ro_hash='a206f6da6a16ea14081062a1e2436b48', train_hash='2c25349fa0a81afbe907882e7b50a4da')]]

@schema
class Seed(dj.Lookup):
    definition = """
    # random seed for training

    seed                 :  int # random seed
    ---
    """
    @property
    def contents(self):
        yield from zip([1009, 1215, 2606, 99999])

@schema
class ToyModel(dj.Computed): # noise_type is not correct in this table, please refer to GroupAssignment.Member
    definition = """
    -> NetworkConfig
    -> DatasetAssignment
    -> Seed
    ---
    val_corr: float              # validation correlation (single trial)
    model:    blob@toy       # stored model
    """

    class UnitTestScores(dj.Part):
        definition = """
        -> master
        -> GroupAssignment.Member
        ---
        pearson: float           # test correlation on single trial responses
        """

    def make(self, key):
        # Set initialization seed
        seed = (Seed() & key).fetch1('seed')
        log.info('Setting seed to {}'.format(seed))
        set_seed(seed)
        config_params = (NetworkConfig & key).fetch1('config_params')
        if isinstance(config_params, dict):
            config_key = config_params
        else:
            config_key = dict()
            for name in config_params.dtype.names:
                config_key[name] = config_params[name].item().item()
        train_params = (configs.TrainConfig.Default & config_key).fetch1()
        # Load data
        dataset, trainloader = DatasetAssignment().load_data(key, 'train', train_params['batch_size'])
        _, valloader = DatasetAssignment().load_data(key, 'validation', train_params['batch_size'])
        _, testloader = DatasetAssignment().load_data(key, 'test', train_params['batch_size'])
        # Build and train model
        model = build_network(config_key, dataset).cuda()
        model = Trainer().train(train_params, model, trainloader, valloader)
        # Evaluate model
        val_corr = correlation_closure(model, valloader)
        test_corr = correlation_closure(model, testloader, avg=False)

        self.insert1({**key, 'val_corr': val_corr, 'model':{k: v.cpu().numpy() for k, v in model.state_dict().items()}})
        for i, tc in enumerate(test_corr):
            self.UnitTestScores.insert1({**key, 'member_id': i+1, 'pearson': tc})

    def load_network(self, key, device='cuda'):
        config_params = (NetworkConfig & key).fetch1('config_params')
        if isinstance(config_params, dict):
            config_key = config_params
        else:
            config_key = dict()
            for name in config_params.dtype.names:
                config_key[name] = config_params[name].item().item()
        dataset, _ = DatasetAssignment().load_data(key, 'train')
        model = build_network(config_key, dataset)
        state_dict = (self & key).fetch1('model')
        try:
            state_dict = {k: torch.as_tensor(state_dict[k][0].copy()) for k in state_dict.dtype.names}
        except AttributeError:
            state_dict = {k: torch.as_tensor(state_dict[k].copy()) for k in state_dict.keys()}
        mod_state_dict = model.state_dict()
        for k in set(mod_state_dict) - set(state_dict):
            log.warning('Could not find paramater {} setting to initialization value'.format(repr(k)))
            state_dict[k] = mod_state_dict[k]
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)
    
class Ensemble():
    def __init__(self, key, member_id, average_batch=True, device='cuda'):
        import copy
        key_copy = key.copy()
        if 'seed' in key_copy:
            key_copy.pop('seed')
        all_keys = (ToyModel & key_copy & 'seed > 1000').fetch('KEY')
        all_models = [ToyModel().load_network(mk) for mk in all_keys]
        self.models = [copy.deepcopy(m) for m in all_models]
        self.neuron_idx = member_id - 1
        self.average_batch = average_batch

    def __call__(self, x):
        resps = [m(x)[:, self.neuron_idx] for m in self.models]
        resps = torch.stack(resps)  # num_models x batch_size x num_neurons
        resp = resps.mean(0).mean(0) if self.average_batch else resps.mean(0)
        return resp

## ========================================= more toy neurons =========================
# class CombGaborGenerator():
#     def __init__(self, params, image_size=(36, 64), center=(0., 0.)):
#         super().__init__()
#         self.image_size = image_size
#         self.center = center
#         self.params = params

#     def __call__(self):
#         x1, y1, x2, y2, theta1, Lambda1, sigma1, psi1, gamma1, theta2, Lambda2, sigma2, psi2, gamma2 = self.params
#         gb1 = GaborGenerator.gen_gabor(self.image_size, self.center, x1, y1, theta1, Lambda1, sigma1, psi1, gamma1)
#         gb2 = self.gen_gabor(self.image_size, self.center, x2, y2, theta2, Lambda2, sigma2, psi2, gamma2)
#         comb_gb = gb1 + gb2
#         return comb_gb
  
# class HighLowSimpleModel(nn.Module):
#     def __init__(self, x1, y1, x2, y2, theta1, Lambda1, sigma1, psi1, gamma1, theta2, Lambda2, sigma2, psi2, gamma2, filter_std= 0.05, average_batch=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), image_size=[36, 64], center=[0, 0], scale_by=1):
#         super().__init__()
#         self.device = device
#         self.filter_std = filter_std
#         high_filter1 = GaborGenerator([x1, y1, theta1, Lambda1, sigma1, psi1, gamma1], image_size, center)().to(self.device)
#         low_filter1 = GaborGenerator([x2, y2, theta2, Lambda2, sigma2, psi2, gamma2], image_size, center)().to(self.device)
#         self.register_buffer('high_filter1', high_filter1 / (high_filter1.std() + 1e-9) * self.filter_std)
#         self.register_buffer('low_filter1', low_filter1 / (low_filter1.std() + 1e-9) * self.filter_std)
#         self.average_batch = average_batch
#         self.scale_by = scale_by

#     def forward(self, x):
#         self.input = x.clone().to(self.device)
#         y = F.relu(F.conv2d(self.input, self.high_filter1, bias=None)) + F.relu(F.conv2d(self.input, self.low_filter1, bias=None))
#         if self.average_batch:
#             y = y.mean(0)
#         return y.squeeze() / self.scale_by
    
# class HighLowComplexModel(nn.Module):
#     def __init__(self, x1, y1, x2, y2, theta1, Lambda1, sigma1, psi1, gamma1, theta2, Lambda2, sigma2, psi2, gamma2, filter_std= 0.05, average_batch=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), image_size=[36, 64], center=[0, 0], scale_by=1):
#         super().__init__()
#         self.device = device
#         self.filter_std = filter_std
#         high_filter1 = GaborGenerator([x1, y1, theta1, Lambda1, sigma1, psi1, gamma1], image_size, center)().to(self.device)
#         high_filter2 = GaborGenerator([x1, y1, theta1, Lambda1, sigma1, psi1+np.pi/2, gamma1], image_size, center)().to(self.device)
#         low_filter1 = GaborGenerator([x2, y2, theta2, Lambda2, sigma2, psi2, gamma2], image_size, center)().to(self.device)
#         self.register_buffer('high_filter1', high_filter1 / (high_filter1.std() + 1e-9) * self.filter_std)
#         self.register_buffer('high_filter2', high_filter2 / (high_filter2.std() + 1e-9) * self.filter_std)
#         self.register_buffer('low_filter1', low_filter1 / (low_filter1.std() + 1e-9) * self.filter_std)
#         self.average_batch = average_batch
#         self.scale_by = scale_by

#     def forward(self, x):
#         self.input = x.clone().to(self.device)
#         y1 = F.conv2d(self.input, self.high_filter1, bias=None)
#         y2 = F.conv2d(self.input, self.high_filter2, bias=None)
#         y = torch.sqrt(y1.pow(2) + y2.pow(2)) + F.relu(F.conv2d(self.input, self.low_filter1, bias=None))
#         if self.average_batch:
#             y = y.mean(0)
#         return y.squeeze() / self.scale_by

# @schema
# class MEIParameters(dj.Lookup):
#     definition = """  # parameters to generate MEIs
    
#     mei_params:         int
#     ---
#     mei_seed:           int     # random seed used to create the initialization
#     num_initializations: int    # how many random images to optimize in parallel to create the MEI (output is the average)
#     height:             int     # height of the MEI
#     width:              int     # width of the MEI 
#     contrast:           decimal(5, 3) # contrast to use when generating the MEI
#     step_size:          float   # step size to use when generating the MEI
#     num_iterations:     int     # number of optimization iterations
#     blur_sigma:         decimal(5, 3) # sigma used for gradient blur
#     fixed_mean:         bool
#     mean:               float
#     """
#     contents = [[1, 0, 1, 36, 64, 0.25, 1, 1000, 1, 1, 0], ]

# @schema
# class MEI(dj.Computed):
#     definition = """
#     -> ToyNeuron
#     -> MEIParameters
#     ---
#     mei:                longblob # optimized MEI
#     activation:         float   # activation at the MEI 
#     """
#     @property
#     def key_source(self):
#         return ToyNeuron * MEIParameters & 'mei_params = 1'

#     def make(self, key):
#         # Get params
#         neuron_params = (ToyNeuron & key).fetch1()
#         mei_params = (MEIParameters & key).fetch1()

#         # Get models
#         scale_by = (ResponseNormalization.Neuron & key).fetch1('response_std')
#         model = eval('{}Model'.format(neuron_params['neuron_type']))(*neuron_params['parameters'], scale_by=scale_by)

#         # Create a random initial image (at desired contrast)
#         torch.manual_seed(mei_params['mei_seed'])
#         image_shape = (mei_params['num_initializations'], 1, mei_params['height'],
#                        mei_params['width'])
#         initial_image = torch.randn(image_shape, device='cuda')
#         if not mei_params['fixed_mean']:
#             initial_image = initial_image * float(mei_params['contrast'])
#         else:
#             initial_image = ops.ChangeStats(float(mei_params['contrast']), float(mei_params['mean']))(initial_image)

#         # Optimize
#         if not mei_params['fixed_mean']:
#             postup_op = ops.ChangeStd(float(mei_params['contrast']))
#         else:
#             postup_op = ops.ChangeStats(float(mei_params['contrast']), float(mei_params['mean']))

#         if mei_params['blur_sigma']:
#             gradient_f = featurevis.ops.GaussianBlur(float(mei_params['blur_sigma']))
#         else:
#             gradient_f = None
#         mei, fevals, _ = featurevis.gradient_ascent(model, initial_image,
#                                                     post_update=postup_op,
#                                                     gradient_f = gradient_f,
#                                                     step_size=mei_params['step_size'],
#                                                     num_iterations=mei_params[
#                                                         'num_iterations'])
#         mei = mei.mean(0).squeeze().cpu().numpy()
#         activation = fevals[-1]

#         # Insert
#         self.insert1({**key, 'mei': mei, 'activation': activation})

# @schema
# class MEIMask(dj.Computed):
#     definition = """ # finds a mask for an MEI by thresholding the absolute intensity

#     -> MEI
#     -> MaskParameters
#     ---
#     mask:               longblob # produced mask
#     mask_x:             float    # (px) centroid of the mask in x; (0, 0) is center of image    
#     mask_y:             float    # (px) centroid of the mask in y; (0, 0) is center of image
#     mask_mean:          float    # mean of MEI inside the mask   
#     mask_std:           float    # standard deviation of MEI inside the mask
#     """

#     @property
#     def key_source(self):
#         return MEI * MaskParameters & 'mask_params = 4'

#     def make(self, key):
#         from scipy import ndimage
#         from skimage import morphology

#         # Get mei
#         mei = (MEI & key).fetch1('mei')

#         # Get params
#         params = (MaskParameters & key).fetch1()

#         # Normalize and threshold
#         norm_mei = (mei - mei.mean()) / mei.std()
#         thresholded = np.abs(norm_mei) > params['zscore_thresh']

#         # Remove small holes in the thresholded image and connect any stranding pixels
#         closed = ndimage.binary_closing(thresholded, iterations=params['closing_iters'])

#         # Remove any remaining small objects
#         labeled = morphology.label(closed, connectivity=2)
#         most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
#         oneobject = labeled == most_frequent

#         # Create convex hull just to close any remaining holes and so it doesn't look weird
#         hull = morphology.convex_hull_image(oneobject)

#         # Smooth edges
#         smoothed = ndimage.gaussian_filter(hull.astype(np.float32),
#                                             sigma=params['gaussian_sigma'])
#         mask = smoothed  # final mask

#         # Compute mask centroid
#         px_y, px_x = (coords.mean() + 0.5 for coords in np.nonzero(hull))
#         mask_y, mask_x = px_y - mask.shape[0] / 2, px_x - mask.shape[1] / 2

#         # Compute MEI std inside the mask
#         mei_mean = (mei * mask).sum() / mask.sum()
#         mei_std = np.sqrt((((mei - mei_mean) ** 2) * mask).sum() / mask.sum())

#         # Insert
#         self.insert1({**key, 'mask': mask, 'mask_x': mask_x, 'mask_y': mask_y,
#                         'mask_mean': mei_mean, 'mask_std': mei_std})

# @schema
# class DEI(dj.Computed):
#     definition = """ # create DEIs
#     -> ToyNeuron
#     -> MEIParameters
#     -> MEIMask
#     -> MaskStatsParameters
#     -> DEIParameters
#     """

#     @property
#     def key_source(self):
#         all_keys = MEIMask * MaskStatsParameters * DEIParameters
#         return all_keys & {'seed': 1009} & 'mask_stats_params = 2'

#     class DEI(dj.Part):
#         definition = """
#         -> master
#         weight_id:  int         # id of this weight (0-indexed, same order as in weights list)
#         ref_id:     int         # id of this reference activation (0-indexed, same order as in ref_levels)
#         dei_id:     int         # id of this DEI (0-indexed)
#         ---
#         dei:        longblob                # DEI image
#         activation: float                   # activation for this DEI
#         latent:     longblob                # latent representation of the image     
#         fevals:     longblob
#         regs:       longblob   
#         """

#     def make(self, key):
#         self.insert1(key)
        
#         # Get params
#         mei_params = (MEIParameters & key).fetch1()
#         mask_stats_params = (MaskStatsParameters & key).fetch1()
#         diverse_params = (DEIParameters & key).fetch1()

#         # Get model
#         neuron_params = (ToyNeuron & key).fetch1()
#         scale_by = (ResponseNormalization.Neuron & key).fetch1('response_std')
#         model = eval('{}Model'.format(neuron_params['neuron_type']))(*neuron_params['parameters'], average_batch=False, scale_by=scale_by)
#         base_model, embedding, similarity_mask, mei, mei_activation, mask, mask_x, mask_y, get_latent, gradient_f, combine_op, postup_op, initial_batch = \
#         prepare_params(key, mei_params, mask_stats_params, diverse_params, toy=True, toy_model=model, mask_table=MEIMask, mei_table=MEI)

#         # Iterate over weights and reference levels
#         for (weight_id, div_weight), (ref_id, div_ref) in itertools.product(enumerate(diverse_params['weights']),
#                                                                             enumerate(diverse_params['ref_levels'])):
#             print('Optimizing with div_weight=', div_weight, 'div_ref=', div_ref)
#             # Set up optimization
#             if diverse_params['loss_type'] == 'HingeLoss':
#                 activation_obj = utils.Compose([base_model, HingeLoss(div_ref, mei_activation), ops.ReverseSign()])
#                 @varargin
#                 def div_regularization(x, iteration):
#                     if diverse_params['combine_op'] == 'average_maximum':
#                         if iteration < int(diverse_params['num_iterations']/2):
#                             similarity = ops.Similarity(div_weight, mask=similarity_mask,
#                                             metric=diverse_params['similarity'],
#                                             combine_op=torch.mean)
#                         else:
#                             similarity = ops.Similarity(div_weight, mask=similarity_mask,
#                                             metric=diverse_params['similarity'],
#                                             combine_op=torch.max)
#                     else:
#                         similarity = ops.Similarity(div_weight, mask=similarity_mask,
#                                             metric=diverse_params['similarity'],
#                                             combine_op=combine_op)
#                     mei_copy = mei.clone()
#                     mei_copy = torch.as_tensor(mei_copy, dtype=torch.float32, device='cuda')
#                     im_set = torch.cat([mei_copy, x], dim=0)

#                     return similarity(embedding(im_set))
#             else: 
#                 raise NotImplementedError('{} loss type not '
#                                         'implemented'.format(diverse_params['loss_type']))

#             # Optimize
#             deis, fevals, regs = featurevis.gradient_ascent(activation_obj, initial_batch,
#                                                             step_size=diverse_params['step_size'],
#                                                             num_iterations=diverse_params['num_iterations'],
#                                                             regularization=div_regularization,
#                                                             post_update=postup_op,
#                                                             gradient_f=gradient_f)

#             # Insert DEIs
#             for dei_id, dei in enumerate(deis):
#                 with torch.no_grad():
#                     activation = base_model(dei[None]).item()  # compute activation
#                     latent = get_latent(dei[None]).detach().cpu().numpy().squeeze()  # compute VAE latent representation
#                 if dei_id == 0:
#                     self.DEI.insert1({**key, 'weight_id': weight_id, 'ref_id': ref_id, 'dei_id': dei_id,
#                                         'dei': dei.cpu().squeeze().numpy(), 'activation': activation, 'latent': latent,
#                                         'fevals': fevals, 'regs': regs})
#                 else:
#                     self.DEI.insert1({**key, 'weight_id': weight_id, 'ref_id': ref_id, 'dei_id': dei_id,
#                                         'dei': dei.cpu().squeeze().numpy(), 'activation': activation, 'latent': latent,
#                                         'fevals': [], 'regs': []})

# @schema
# class DEIEvaluation(dj.Computed):
#     definition = """ # evaluate DEIs
#     -> DEI
#     """

#     class DEISet(dj.Part):
#         definition = """ # evaluation metrics at a single weight and single reference level
#         -> master
#         weight_id:             int
#         ref_id:                int         
#         ---
#         avg_activation_ratio:  float  # average activation / original mei_activation across all DEIs
#         min_activation_ratio:  float  # mininum activation / original mei_activation across all DEIs
#         std_activation:        float  # (raw std activation / raw average activation) across all DEIs
#         avg_sim:               float  # average pair-wise similarity
#         max_sim:               float  # maximum pair-wise similarity
#         min_sim:               float  # minimum pair-wise similarity
#         sims_to_mei:           longblob  # similarity to MEI
#         """

#     def make(self, key):
#         # Get params
#         mei_params = (MEIParameters & key).fetch1()
#         mask_stats_params = (MaskStatsParameters & key).fetch1()
#         diverse_params = (DEIParameters & key).fetch1()

#         # Get model
#         neuron_params = (ToyNeuron & key).fetch1()
#         scale_by = (ResponseNormalization.Neuron & key).fetch1('response_std')
#         model = eval('{}Model'.format(neuron_params['neuron_type']))(*neuron_params['parameters'], average_batch=False, scale_by=scale_by)
#         base_model, embedding, similarity_mask, mei, mei_activation, mask, mask_x, mask_y, get_latent, gradient_f, combine_op, postup_op, initial_batch = \
#         prepare_params(key, mei_params, mask_stats_params, diverse_params, toy=True, toy_model=model, mask_table=MEIMask, mei_table=MEI)

#         # Get all DEIs and activations for this cell
#         deis, activations = (DEI.DEI & key).fetch('dei', 'activation', order_by='weight_id, ref_id, dei_id')

#         # Reshape DEI and activation arrays
#         weights, refs = (DEIParameters & key).fetch1('weights', 'ref_levels')
#         deis = np.stack(deis).reshape(len(weights), len(refs), -1, *deis[0].shape)  # num_weights x num_refs x num_deis x h x w
#         activations = activations.reshape(len(weights), len(refs), -1)  # num_weights x num_refs x num_deis
        
#         # Compute similarity using different metrics
#         avg_ne = utils.Compose([embedding, ops.Similarity(mask=similarity_mask, metric=diverse_params['similarity'], combine_op=torch.mean)])
#         max_ne = utils.Compose([embedding, ops.Similarity(mask=similarity_mask, metric=diverse_params['similarity'], combine_op=torch.max)])
#         min_ne = utils.Compose([embedding, ops.Similarity(mask=similarity_mask, metric=diverse_params['similarity'], combine_op=torch.min)])
#         sim = utils.Compose([embedding, ops.Similarity(mask=similarity_mask, metric=diverse_params['similarity'], combine_op=torch.mean)])

#         if diverse_params['initial_type'] == 'MEI':
#             @varargin
#             def similarity(sim, x):
#                 im_set = torch.cat([mei, x], dim=0)
#                 return sim(im_set)
#             avg_ne = partial(similarity, avg_ne)
#             max_ne = partial(similarity, max_ne)
#             min_ne = partial(similarity, min_ne)

#         # Insert one set at a time
#         self.insert1(key)
#         for weight_id, (dei_set, act_set) in enumerate(zip(deis, activations)):
#             for ref_id, (deis_, acts_) in enumerate(zip(dei_set, act_set)):
#                 deis_ = torch.as_tensor(deis_, dtype=torch.float32, device='cuda').contiguous()
#                 sims_to_mei = np.array([sim(torch.cat([dei[None, None], mei])).item() for dei in deis_])
#                 avg_activation_ratio = np.mean(acts_) / mei_activation
#                 min_activation_ratio = np.min(acts_) / mei_activation
#                 std_activation = np.std(acts_) / np.mean(acts_)
#                 avg_sim = avg_ne(deis_[:, None].cuda()).item()
#                 max_sim = max_ne(deis_[:, None].cuda()).item()
#                 min_sim = min_ne(deis_[:, None].cuda()).item()

#                 self.DEISet.insert1({**key, 'weight_id': weight_id, 'ref_id': ref_id, 
#                                     'avg_activation_ratio': avg_activation_ratio, 'min_activation_ratio': min_activation_ratio, 
#                                     'std_activation': std_activation, 'avg_sim': avg_sim, 'max_sim': max_sim, 'min_sim': min_sim,
#                                     'sims_to_mei': sims_to_mei})

# @schema
# class DEIThreshold(dj.Lookup):
#     definition = """
#     threshold_params: int
#     ---
#     dev_from_ref_threshold: float  # threshold on deviation of average activation from reference level of activation
#     std_threshold:          float  # threshold on std activation / avg activation
#     min_ratio:              float  # threshold for minimum DEI activation ratio
#     selection_criterion:     varchar(16) # criteria for selecting one single run among all valid runs
#     """
#     contents = [{'threshold_params': 1, 'dev_from_ref_threshold': 1, 'std_threshold': 1,'min_ratio': 0.85, 'selection_criterion': 'min_act'},]

# @schema
# class DEIGoodRun(dj.Computed):
#     definition = """
#     -> DEIEvaluation
#     -> DEIThreshold
#     ref_id:                int
#     ---
#     weight_id:             int    # id of this weight (0-indexed, same order as in weights list)
#     """

#     @property
#     def key_source(self):
#         return DEIEvaluation * DEIThreshold
    
#     @staticmethod
#     def get_largest_weight_id(key, ref):
#         threshold_params = (DEIThreshold & key).fetch1()
#         diverse_params = (DEIParameters & key).fetch1()
#         eval_table = DEIEvaluation.DEISet
#         avg_act_ratios, min_act_ratios, stds = (eval_table & key).fetch('avg_activation_ratio', 'min_activation_ratio', 'std_activation')

#         if threshold_params['min_ratio'] == 0:
#             satisfied_weight_ids_1 = (avg_act_ratios > (ref - threshold_params['dev_from_ref_threshold'])) & (avg_act_ratios < (ref + threshold_params['dev_from_ref_threshold']))
#             satisfied_weight_ids_2 = (stds < threshold_params['std_threshold'])
#             satisfied_weight_ids = satisfied_weight_ids_1 & satisfied_weight_ids_2
#         else:
#             satisfied_weight_ids = np.round(min_act_ratios, 2) >= threshold_params['min_ratio'] 

#         idx = np.where(satisfied_weight_ids == True)[0]
#         if len(idx) == 0:
#             return None
#         elif threshold_params['selection_criterion'] == 'min_act':
#             return idx[np.argsort(avg_act_ratios[satisfied_weight_ids])[0]] # Select the weight_id corresponding to the lowest activation above threshold (in case the weights are not ordered monotonically)
#         elif threshold_params['selection_criterion'] == 'largest_weight':
#             return idx[-1]

#     def make(self, key):
#         ref_levels = (DEIParameters & key).fetch1('ref_levels')
#         for ref_id, ref in enumerate(ref_levels):
#             key['ref_id'] = ref_id
#             weight_id = self.get_largest_weight_id(key, ref)
#             if weight_id is not None:
#                 self.insert1({**key, 'weight_id': weight_id})

# @schema
# class TextureSynthesis(dj.Computed):
#     definition = """
#     -> DEIEvaluation.DEISet
#     -> DEIThreshold
#     -> TextureParameters
#     """
    
#     class TextureSynthesis(dj.Part):
#         definition= """ # Result at each fraction_std
#         -> master
#         texture_id: int                      # Id of the run 
#         ---
#         target_fraction_std: float           # target fraction std for DEI of the run
#         fraction_std: float                  # actual fraction stds
#         p: float                             # fraction of variable mask
#         variable_mask: longblob              # variable mask to sample from texture
#         fixed_part: longblob                 # fixed part of the DEI
#         variable_crops: longblob             # variable samples
#         full_texture: longblob               # Full texture
#         eval_texture: longblob               # Texture used for evaluation
#         centered_texture: longblob           # Centered texture to sample from
#         samples: blob@toy             # Samples from the centered texture
#         sample_acts: longblob                # Samples' responses
#         sample_avg_div: float                # Samples average diversity
#         sample_avg_activation_ratio: float   # Samples average activation ratio
#         sample_min_activation_ratio: float   # Sample min activation ratio
#         sample_std_activation: float         # Sample activation ratio sd
#         total_loss_his: longblob             # Total loss history
#         act_his: longblob                    # Average activation ratio history
#         div_reg_his: longblob                # Diversity history
#         texture_his: blob@toy         # Texture history
#         dei_avg_div: float                   # Corresponding DEI average diversity
#         dei_avg_activation_ratio: float      # Corresponding DEI average activation ratio
#         """
    
#     @property
#     def key_source(self):
#         return DEIEvaluation.DEISet * DEIThreshold * TextureParameters & DEIGoodRun
    
#     def make(self,key):
#         self.insert1(key)
        
#         device = 'cuda'
#         texture_parameters = (TextureParameters & key).fetch1()
#         mask = np.array((MEIMask() & key).fetch1('mask'))
#         mei,mei_act = (MEIMask * MEI & (DEIGoodRun & key)).fetch1('mei','activation')
#         mask_params = (MaskParameters & key).fetch1()
#         deis,dei_acts,dei_div,dei_avg_activation_ratio = (base.MEI.proj('mei', mei_act='activation') * DEI.DEI * DEIEvaluation.DEISet * (DEIGoodRun & key)).fetch('dei','activation','avg_sim','avg_activation_ratio')
#         deis = np.stack(deis)
#         dei_div = -dei_div[0]
#         dei_avg_activation_ratio = dei_avg_activation_ratio[0]

#         variable_masks = get_variable_masks(deis,mei,mask_params,values=texture_parameters['fraction_stds'],
#                                             params={'closing_iters':texture_parameters['closing_iters'],
#                                                     'gaussian_sigma':texture_parameters['gaussian_sigma']})

#         # get model
#         neuron_params = (ToyNeuron & key).fetch1()
#         scale_by = (ResponseNormalization.Neuron & key).fetch1('response_std')
#         predictive_model = eval('{}Model'.format(neuron_params['neuron_type']))(*neuron_params['parameters'], average_batch=False, scale_by=scale_by)

#         # Include the full mask with texture_id = -1        
#         for i,(target_fraction_std,variable_mask) in enumerate(zip(np.append(texture_parameters['fraction_stds'],1.0),variable_masks+[{'fraction_std':1.0,'variable_mask':mask}])):
#             # Calculate texture shape and appropriate padding
#             texture_shape, eval_padding,center_padding = cal_texture_minimum_shape(mask=variable_mask['variable_mask'],target_shape =(texture_parameters['target_height'],texture_parameters['target_width']),
#                                                                              threshold=texture_parameters['mask_threshold'],padding=texture_parameters['padding'])
            
#             image_model = LinearImageModel(mei,variable_mask['variable_mask'],t_f=texture_parameters['texture_f'],initial_t = None,t_shape=texture_shape,eval_padding=eval_padding,center_padding=center_padding,
#                                            default_n_crops=texture_parameters['n_crops']).to(device)
            
#             image_model,stats = train_texture(image_model, predictive_model, mei_act, mask, device=device,
#                                               optimizer_args={'optimizer_name':texture_parameters['optimizer_name'],'lr':texture_parameters['lr'], 'smooth_gradient':texture_parameters['smooth_gradient'],
#                                                               'gradient_sigma':texture_parameters['gradient_sigma'],'decay_factor':texture_parameters['decay_factor']},
#                                               criterion_args={'ref_level':texture_parameters['ref_level'],'Lambda':texture_parameters['lambda']},
#                                               training_args={'batch_size':texture_parameters['n_crops'],'n_iter_per_eval':texture_parameters['n_iter_per_eval'],'max_iters':texture_parameters['n_iters']})

#             with torch.no_grad():
#                 variable_crops = image_model.get_v_c(key='eval',n_crops=len(deis)).detach().cpu().numpy().squeeze()
#                 samples = image_model(key='eval',n_crops=len(deis)).detach()
#                 sample_acts = predictive_model(samples).detach().cpu().numpy().squeeze()
#                 samples = samples.detach().cpu().numpy().squeeze()
#                 sample_act_ratios = sample_acts/mei_act
                
#             texture_id = i
#             if texture_id == len(variable_masks):
#                 texture_id = -1
#             result = {**key,'texture_id':texture_id,'target_fraction_std':round(target_fraction_std,5),'fraction_std':variable_mask['fraction_std'],'p':variable_mask['variable_mask'].sum()/mask.sum(),
#                       'variable_mask':variable_mask['variable_mask'],'fixed_part':image_model.fixed_c.detach().cpu().numpy().squeeze(),'variable_crops':variable_crops,
#                       'full_texture':image_model.get_texture('full',True),'eval_texture':image_model.get_texture('eval',True),'centered_texture':image_model.get_texture('centered',True),
#                       'samples':samples,'sample_acts':sample_acts,'sample_avg_div':stats['div_reg'][-1],'sample_avg_activation_ratio':sample_act_ratios.mean(),
#                       'sample_min_activation_ratio': min(sample_act_ratios),'sample_std_activation': sample_act_ratios.std(),'total_loss_his':np.array(stats['total_loss']),
#                       'act_his':np.array(stats['act']),'div_reg_his':np.array(stats['div_reg']),'texture_his':np.stack(stats['texture']),'dei_avg_div':dei_div,
#                       'dei_avg_activation_ratio':dei_avg_activation_ratio}
            
#             self.TextureSynthesis.insert1(result)

# @schema
# class TextureSynthesisGoodRun(dj.Computed):
#     definition = """
#     -> TextureSynthesis
#     -> TextureScoreParameters
#     ---
#     texture_id:             int    # Id of the run 
#     """

#     @property
#     def key_source(self):
#         return TextureSynthesis * TextureScoreParameters

#     @staticmethod
#     def relaxed_score(x, y):
#         return 2 * (x*y) / (x+y)

#     @staticmethod
#     def hard_score(x, c):
#         return - abs (x - c)
    
#     def make(self, key):
#         target_ratio, score_method = (TextureScoreParameters() & key).fetch1('target_ratio', 'score_method')
#         texture_ids, sample_avg_activation_ratios, sample_avg_divs = (TextureSynthesis.TextureSynthesis() & key).fetch('texture_id','sample_avg_activation_ratio','sample_avg_div')
        
#          # Hard score
#         if score_method == 'hard_threshold':
#             score = self.hard_score(sample_avg_activation_ratios, target_ratio)
#         # Relaxed score
#         elif score_method == 'f_measurement':
#             score = self.relaxed_score(sample_avg_activation_ratios, sample_avg_divs / sample_avg_divs.max())
#         idx = np.argmax(score)

#         self.insert1({**key,'texture_id':texture_ids[idx]})
