from attorch.layers import (SpatialXFeatureLinear, elu1,
                            SpatialTransformerPooled2d, SpatialTransformerPyramid2d)
from attorch.module import ModuleDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn import Parameter
from torch.nn import functional as F

from staticnet import logger as log


class Readout:
    def initialize(self, *args, **kwargs):
        raise NotImplementedError('initialize is not implemented for ', self.__class__.__name__)

    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: not x.startswith('_') and
                                     ('gamma' in x or 'pool' in x or 'positive' in x), dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'


class SpatialXFeaturesReadout(Readout, ModuleDict):
    def __init__(self, in_shape, neurons, gamma_readout, positive=True, normalize=True, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self.positive = positive
        self.normalize = normalize
        self.gamma_readout = gamma_readout

        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, SpatialXFeatureLinear(in_shape, neur, normalize=normalize, positive=positive))

    def initialize(self, mu_dict):
        log.info('Initializing ' +  self.__class__.__name__)
        for k, mu in mu_dict.items():
            self[k].initialize(init_noise=1e-6)
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].l1() * self.gamma_readout


class SpatialTransformerPooled2dReadout(Readout, ModuleDict):
    _BaseReadout = None

    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, pool_steps=0, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, SpatialTransformerPooled2d(in_shape, neur, positive=positive, pool_steps=pool_steps))

    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        log.info('Initializing with mu_dict: ' + ', '.join(['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()]))

        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_features

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        self._pool_steps = value
        for k in self:
            self[k].poolsteps = value


class SpatialTransformer2dReadout(SpatialTransformerPooled2dReadout):
    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__(in_shape, neurons, positive=positive,
                         gamma_features=gamma_features,
                         _pool_steps=0, **kwargs)


class PooledReadout(Readout):
    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        log.info('Initializing with mu_dict: ' + ', '.join(['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()]))
        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key, subs_idx=None):
        return self[readout_key].feature_l1(subs_idx=subs_idx) * self.gamma_features

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        self._pool_steps = value
        for k in self:
            self[k].poolsteps = value


class _SpatialTransformerPyramid(Readout, ModuleDict):
    _BaseReadout = None

    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, scale_n=3, downsample=True,
                 type=None, _skip_upsampling=False, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, self._BaseReadout(in_shape, neur, positive=positive, scale_n=scale_n,
                                                 downsample=downsample, _skip_upsampling=_skip_upsampling, type=type))

    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        log.info('Initializing with mu_dict: ' + ', '.join(['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()]))

        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_features
    

class SpatialTransformerPyramid2dReadout(_SpatialTransformerPyramid):
    _BaseReadout = SpatialTransformerPyramid2d

class ModifiedSpatialTransformerPyramid2dReadout(_SpatialTransformerPyramid):
    _BaseReadout = SpatialTransformerPyramid2d

    
class ClonedReadout(nn.Module):
    """
    This readout clones another readout while applying a linear transformation on the output. Used for MultiDatasets
    with matched neurons where the x-y positions in the grid stay the same but the predicted responses are rescaled due
    to varying experimental conditions.
    """

    def __init__(self, original_readout, **kwargs):
        super().__init__()

        self._source = original_readout
        self.alpha = Parameter(torch.ones(self._source.features.shape[-1]))
        self.beta = Parameter(torch.zeros(self._source.features.shape[-1]))

    def forward(self, x, shift=None):
        x = self._source(x, shift=shift) * self.alpha + self.beta
        return x

    def feature_l1(self, average=True):
        """ Regularization is only applied on the scaled feature weights, not on the bias."""
        if average:
            return self._source.features.abs().mean()
#             return (self._source.features * self.alpha).abs().mean()
        else:
            return self._source.features.abs().sum()
#             return (self._source.features * self.alpha).abs().sum()

    def initialize(self):
        self.alpha.data.fill_(1.0)
        self.beta.data.fill_(0.0)
        
class MultipleSpatialTransformerPyramid2dReadout(Readout, ModuleDict):
    _BaseReadout = SpatialTransformerPyramid2d
    
    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, scale_n=3, downsample=True, clone_readout=False,
                 type=None, _skip_upsampling=False, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()
        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self.clone_readout = clone_readout
        
#         self.neurons.move_to_end('group139-23555-6-10-0', last=False) # hack: to change dataset order
        for i, (k, n_neurons) in enumerate(self.neurons.items()):
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            if i == 0 or not clone_readout:
                self.add_module(k, self._BaseReadout(in_shape, n_neurons, positive=positive, scale_n=scale_n,
                                                    downsample=downsample, _skip_upsampling=_skip_upsampling, type=type))
                original_readout = k

            elif i > 0 and clone_readout:
                self.add_module(k, ClonedReadout(self[original_readout], **kwargs))
            
                
    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value
                
    def initialize(self, mu_dict):
        for k, mu in mu_dict.items():
            self[k].initialize()
            if not isinstance(self[k], ClonedReadout):
                self[k].bias.data = mu.squeeze() - 1
                
    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_features


from neuralpredictors.layers.readouts import MultiReadoutSharedParametersBase, FullGaussian2d

class MultipleFullGaussian2dNewReadout(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d
    
# # Reference: https://github.com/sinzlab/ml-utils/blob/cf4869cbb53de68e92aa8deacc69b339d3885c4f/mlutils/layers/readouts.py#L586
# class FullGaussian2dReadout(nn.Module):
#     """
#     A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
#     and covariance of that Gaussian are learned.
#     Args:
#         in_shape (list, tuple): shape of the input feature map [channels, width, height]
#         outdims (int): number of output units
#         bias (bool): adds a bias term
#         init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
#                             [expected: positive value <=1]. Default: 0.1
#         init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
#             'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
#             covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
#         batch_sample (bool): if True, samples a position for each image in the batch separately
#                             [default: True as it decreases convergence time and performs just as well]
#         align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
#                 It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
#                 behavior to pre PyTorch 1.3 functionality for comparability.
#         gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
#         grid_mean_predictor (dict): Parameters for a predictor of the mean grid locations. Has to have a form like
#                         {
#                         'hidden_layers':0,
#                         'hidden_features':20,
#                         'final_tanh': False,
#                         }
#         shared_features (dict): Used when the feature vectors are shared (within readout between neurons) or between
#                 this readout and other readouts. Has to be a dictionary of the form
#                {
#                     'match_ids': (numpy.array),
#                     'shared_features': torch.nn.Parameter or None
#                 }
#                 The match_ids are used to match things that should be shared within or across scans.
#                 If `shared_features` is None, this readout will create its own features. If it is set to
#                 a feature Parameter of another readout, it will replace the features of this readout. It will be
#                 access in increasing order of the sorted unique match_ids. For instance, if match_ids=[2,0,0,1],
#                 there should be 3 features in order [0,1,2]. When this readout creates features, it will do so in
#                 that order.
#         shared_grid (dict): Like `shared_features`. Use dictionary like
#                {
#                     'match_ids': (numpy.array),
#                     'shared_grid': torch.nn.Parameter or None
#                 }
#                 See documentation of `shared_features` for specification.
#         source_grid (numpy.array):
#                 Source grid for the grid_mean_predictor.
#                 Needs to be of size neurons x grid_mean_predictor[input_dimensions]
#     """

#     def __init__(self, in_shape, outdims, bias, init_mu_range=0.1, init_sigma=1, batch_sample=True,
#                  align_corners=True, gauss_type='full', grid_mean_predictor={},
#                  shared_features=None, shared_grid=None, source_grid=None, **kwargs):

#         super().__init__()

#         # determines whether the Gaussian is isotropic or not
#         self.gauss_type = gauss_type

#         if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
#             raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")

#         # store statistics about the images and neurons
#         self.in_shape = in_shape
#         self.outdims = outdims

#         # sample a different location per example
#         self.batch_sample = batch_sample

#         # position grid shape
#         self.grid_shape = (1, outdims, 1, 2)

#         # the grid can be predicted from another grid
#         self._predicted_grid = False
#         self._shared_grid = False
#         self._original_grid = not self._predicted_grid

#         if not grid_mean_predictor and shared_grid is None:
#             self._mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
#         elif grid_mean_predictor and shared_grid is not None:
#             raise ConfigurationError('Shared grid_mean_predictor and shared_grid_mean cannot both be set')
#         elif grid_mean_predictor:
#             # convert grid_mean_predictor from recarray to dictionary
#             grid_mean_predictor = {name: grid_mean_predictor[name].item().item() for name in grid_mean_predictor.dtype.names}
#             self.init_grid_predictor(source_grid=source_grid, **grid_mean_predictor)
#         elif shared_grid is not None:
#             self.initialize_shared_grid(**(shared_grid or {}))

#         if gauss_type == 'full':
#             self.sigma_shape = (1, outdims, 2, 2)
#         elif gauss_type == 'uncorrelated':
#             self.sigma_shape = (1, outdims, 1, 2)
#         elif gauss_type == 'isotropic':
#             self.sigma_shape = (1, outdims, 1, 1)
#         else:
#             raise ValueError(f'gauss_type "{gauss_type}" not known')

#         self.init_sigma = init_sigma
#         self.sigma = Parameter(torch.Tensor(*self.sigma_shape))  # standard deviation for gaussian for each neuron

#         self.initialize_features(**(shared_features or {}))

#         if bias:
#             bias = Parameter(torch.Tensor(outdims))
#             self.register_parameter("bias", bias)
#         else:
#             self.register_parameter("bias", None)

#         self.init_mu_range = init_mu_range
#         self.align_corners = align_corners
#         self.initialize()

#     @property
#     def shared_features(self):
#         return self._features

#     @property
#     def shared_grid(self):
#         return self._mu

#     @property
#     def features(self):
#         if self._shared_features:
#             return self.scales * self._features[..., self.feature_sharing_index]
#         else:
#             return self._features

#     @property
#     def grid(self):
#         return self.sample_grid(batch_size=1, sample=False)

#     def feature_l1(self, average=True):
#         """
#         Returns the l1 regularization term either the mean or the sum of all weights
#         Args:
#             average(bool): if True, use mean of weights for regularization
#         """
#         if self._original_features:
#             if average:
#                 return self._features.abs().mean()
#             else:
#                 return self._features.abs().sum()
#         else:
#             return 0

#     @property
#     def mu(self):
#         if self._predicted_grid:
#             return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
#         elif self._shared_grid:
#             if self._original_grid:
#                 return self._mu[:, self.grid_sharing_index, ...]
#             else:
#                 return self.mu_transform(self._mu.squeeze())[self.grid_sharing_index].view(*self.grid_shape)
#         else:
#             return self._mu

#     def sample_grid(self, batch_size, sample=None):
#         """
#         Returns the grid locations from the core by sampling from a Gaussian distribution
#         Args:
#             batch_size (int): size of the batch
#             sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
#                             or use the mean, mu, of the Gaussian distribution without sampling.
#                            if sample is None (default), samples from the N(mu,sigma) during training phase and
#                              fixes to the mean, mu, during evaluation phase.
#                            if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
#         """
#         with torch.no_grad():
#             self.mu.clamp_(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
#             if self.gauss_type != 'full':
#                 self.sigma.clamp_(min=0)  # sigma/variance is always a positive quantity

#         grid_shape = (batch_size,) + self.grid_shape[1:]

#         sample = self.training if sample is None else sample
#         if sample:
#             norm = self.mu.new(*grid_shape).normal_()
#         else:
#             norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

#         if self.gauss_type != 'full':
#             return torch.clamp(
#                 norm * self.sigma + self.mu, min=-1, max=1
#             )  # grid locations in feature space sampled randomly around the mean self.mu
#         else:
#             return torch.clamp(
#                 torch.einsum('ancd,bnid->bnic', self.sigma, norm) + self.mu, min=-1, max=1
#             )  # grid locations in feature space sampled randomly around the mean self.mu

#     def init_grid_predictor(self, source_grid, hidden_features=20, hidden_layers=0, final_tanh=False, **kwargs):
#         self._original_grid = False
#         layers = [
#             nn.Linear(source_grid.shape[1], hidden_features if hidden_layers > 0 else 2)
#         ]

#         for i in range(hidden_layers):
#             layers.extend([
#                 nn.ELU(),
#                 nn.Linear(hidden_features, hidden_features if i < hidden_layers - 1 else 2)
#             ])

#         if final_tanh:
#             layers.append(
#                 nn.Tanh()
#             )
#         self.mu_transform = nn.Sequential(*layers)

#         source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
#         source_grid = source_grid / np.abs(source_grid).max()
#         self.register_buffer('source_grid', torch.from_numpy(source_grid.astype(np.float32)))
#         self._predicted_grid = True

#     def initialize(self):
#         """
#         Initializes the mean, and sigma of the Gaussian readout along with the features weights
#         """
#         # store gradient [used for debug purpose]
#         self.mu_grad = []
#         self.sigma_grad = []
#         self.feat_grad = []
        
#         if not self._predicted_grid or self._original_grid:
#             self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

#         if self.gauss_type != 'full':
#             self.sigma.data.fill_(self.init_sigma)
#         else:
#             self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
#         self._features.data.fill_(1 / self.in_shape[0])
#         if self._shared_features:
#             self.scales.data.fill_(1.)
#         if self.bias is not None:
#             self.bias.data.fill_(0)

#     def initialize_features(self, match_ids=None, shared_features=None):
#         """
#         The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
#         learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
#         via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
#         """
#         c, w, h = self.in_shape
#         self._original_features = True
#         if match_ids is not None:
#             assert self.outdims == len(match_ids)

#             n_match_ids = len(np.unique(match_ids))
#             if shared_features is not None:
#                 assert shared_features.shape == (1, c, 1, n_match_ids), \
#                     f'shared features need to have shape (1, {c}, 1, {n_match_ids})'
#                 self._features = shared_features
#                 self._original_features = False
#             else:
#                 self._features = Parameter(
#                     torch.Tensor(1, c, 1, n_match_ids))  # feature weights for each channel of the core
#             self.scales = Parameter(torch.Tensor(1, 1, 1, self.outdims))  # feature weights for each channel of the core
#             _, sharing_idx = np.unique(match_ids, return_inverse=True)
#             self.register_buffer('feature_sharing_index', torch.from_numpy(sharing_idx))
#             self._shared_features = True
#         else:
#             self._features = Parameter(
#                 torch.Tensor(1, c, 1, self.outdims))  # feature weights for each channel of the core
#             self._shared_features = False

#     def initialize_shared_grid(self, match_ids=None, shared_grid=None):
#         c, w, h = self.in_shape

#         if match_ids is None:
#             raise ConfigurationError('match_ids must be set for sharing grid')
#         assert self.outdims == len(match_ids), 'There must be one match ID per output dimension'

#         n_match_ids = len(np.unique(match_ids))
#         if shared_grid is not None:
#             assert shared_grid.shape == (1, n_match_ids, 1, 2), \
#                 f'shared grid needs to have shape (1, {n_match_ids}, 1, 2)'
#             self._mu = shared_grid
#             self._original_grid = False
#             self.mu_transform = nn.Linear(2, 2)
#             self.mu_transform.bias.data.fill_(0.)
#             self.mu_transform.weight.data = torch.eye(2)
#         else:
#             self._mu = Parameter(torch.Tensor(1, n_match_ids, 1, 2))  # feature weights for each channel of the core
#         _, sharing_idx = np.unique(match_ids, return_inverse=True)
#         self.register_buffer('grid_sharing_index', torch.from_numpy(sharing_idx))
#         self._shared_grid = True

#     def forward(self, x, sample=None, shift=None, out_idx=None):
#         """
#         Propagates the input forwards through the readout
#         Args:
#             x: input data
#             sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
#                             or use the mean, mu, of the Gaussian distribution without sampling.
#                            if sample is None (default), samples from the N(mu,sigma) during training phase and
#                              fixes to the mean, mu, during evaluation phase.
#                            if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
#             shift (bool): shifts the location of the grid (from eye-tracking data)
#             out_idx (bool): index of neurons to be predicted
#         Returns:
#             y: neuronal activity
#         """
#         N, c, w, h = x.size()
#         c_in, w_in, h_in = self.in_shape
#         if (c_in, w_in, h_in) != (c, w, h):
#             raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
#         feat = self.features.view(1, c, self.outdims)
#         bias = self.bias
#         outdims = self.outdims

#         if self.batch_sample:
#             # sample the grid_locations separately per image per batch
#             grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
#         else:
#             # use one sampled grid_locations for all images in the batch
#             grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 2)

#         if out_idx is not None:
#             if isinstance(out_idx, np.ndarray):
#                 if out_idx.dtype == bool:
#                     out_idx = np.where(out_idx)[0]
#             feat = feat[:, :, out_idx]
#             grid = grid[:, out_idx]
#             if bias is not None:
#                 bias = bias[out_idx]
#             outdims = len(out_idx)

#         if shift is not None:
#             grid = grid + shift[:, None, None, :]

#         y = F.grid_sample(x, grid) # [unexpected argument for pytorch 1.1] , align_corners=self.align_corners)
#         y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

#         if self.bias is not None:
#             y = y + bias
#         return y

#     def __repr__(self):
#         c, w, h = self.in_shape
#         r = self.gauss_type + ' '
#         r += self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
#         if self.bias is not None:
#             r += " with bias"
#         if self._shared_features:
#             r += ", with {} features".format('original' if self._original_features else 'shared')

#         if self._predicted_grid:
#             r += ", with predicted grid"
#         if self._shared_grid:
#             r += ", with {} grid".format('original' if self._original_grid else 'shared')

#         for ch in self.children():
#             r += "  -> " + ch.__repr__() + "\n"
#         return r
    

# # Reference: https://github.com/sinzlab/nnsysident/blob/e1404c9e2d0649832b17dca6277fac4c39d1af04/nnsysident/models/readouts.py#L12
# class MultiReadout:

#     def forward(self, *args, data_key=None, **kwargs):
#         if data_key is None and len(self) == 1:
#             data_key = list(self.keys())[0]
#         return self[data_key](*args, **kwargs)

#     def regularizer(self, data_key):
# #         l1_reg = self.gamma_readout * self[data_key].feature_l1(average=False)
#         l1_reg = self.gamma_readout * self[data_key].feature_l1()

#         return l1_reg


# class MultipleFullGaussian2dReadout(MultiReadout, ModuleDict):
#     def __init__(self, in_shape, neurons, init_mu_range, init_sigma, bias, gamma_readout,
#                  gauss_type, grid_mean_predictor, grid_mean_predictor_type, source_grids,
#                  share_features, share_grid, share_transform, shared_match_ids, init_noise, init_transform_scale, **kwargs):
#         # super init to get the _module attribute
#         super().__init__()
#         self.in_shape = in_shape
#         self.neurons = neurons
        
#         k0 = None
#         for i, (k, n_neurons) in enumerate(self.neurons.items()):
#             k0 = k0 or k
#             if isinstance(self.in_shape, dict):
#                 in_shape = self.in_shape[k]

#             source_grid = None
#             shared_grid = None
#             shared_transform = None
#             if grid_mean_predictor:
#                 if grid_mean_predictor_type == 'cortex':
#                     source_grid = source_grids[k]
#                 else:
#                     raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))
#                 if share_transform:
#                     shared_transform = None if i == 0 else self[k0].mu_transform

#             elif share_grid:
#                 shared_grid = {
#                     'match_ids': shared_match_ids[k],
#                     'shared_grid': None if i == 0 else self[k0].shared_grid
#                 }

#             if share_features:
#                 shared_features = {
#                     'match_ids': shared_match_ids[k],
#                     'shared_features': None if i == 0 else self[k0].shared_features
#                 }
#             else:
#                 shared_features = None

#             self.add_module(k, FullGaussian2dReadout(
#                 in_shape=in_shape,
#                 outdims=n_neurons,
#                 init_mu_range=init_mu_range,
#                 init_sigma=init_sigma,
#                 bias=bias,
#                 gauss_type=gauss_type,
#                 grid_mean_predictor=grid_mean_predictor,
#                 shared_features=shared_features,
#                 shared_grid=shared_grid,
#                 source_grid=source_grid,
#                 shared_transform=shared_transform,
#                 init_noise=init_noise,
#                 init_transform_scale=init_transform_scale,
#             )
#                             )
#         self.gamma_readout = gamma_readout
        
#     def initialize(self, mu_dict):
#         log.info('Initializing with mu_dict: ' + ', '.join(['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()]))

#         for k, mu in mu_dict.items():
#             self[k].initialize()
#             self[k].bias.data = mu.squeeze() - 1

            
# ####### MultipleGaussian2d readout zhuokun previously used ##########
# from collections import OrderedDict

# class MultipleGaussian2d(Readout, ModuleDict):
#     """
#     Instantiates multiple instances of Gaussian2d Readouts
#     usually used when dealing with more than one dataset sharing the same core.
#     Args:
#         in_shape (list): shape of the input feature map [channels, width, height]
#         loaders (list):  a list of dataloaders
#         gamma_readout (float): regularizer for the readout
#     """

#     def __init__(self, in_shape, loaders, gamma_readout, **kwargs):
#         super().__init__()

#         self.in_shape = in_shape
#         self.neurons = OrderedDict([(k, loader.dataset.n_neurons) for k, loader in loaders.items()])

#         self.gamma_readout = gamma_readout  # regularisation strength

#         for k, n_neurons in self.neurons.items():
#             self.add_module(k, FullGaussian2dReadout(in_shape=in_shape, outdims=n_neurons, **kwargs))

#     def initialize(self, mean_activity_dict):
#         for k, mu in mean_activity_dict.items():
#             self[k].initialize()
# #             if self[k].bias:
#             self[k].bias.data = mu.squeeze() - 1

#     def regularizer(self, readout_key):
#         return self[readout_key].feature_l1() * self.gamma_readout
    
# class Gaussian2d(nn.Module):
#     """
#     Instantiates an object that can used to learn a point in the core feature space for each neuron,
#     sampled from a Gaussian distribution with some mean and variance at train but set to mean at test time, that best predicts its response.
#     The readout receives the shape of the core as 'in_shape', the number of units/neurons being predicted as 'outdims', 'bias' specifying whether
#     or not bias term is to be used and 'init_range' range for initialising the mean and variance of the gaussian distribution from which we sample to
#     uniform distribution, U(-init_range,init_range) and  uniform distribution, U(0.0, 3*init_range) respectively.
#     The grid parameter contains the normalized locations (x, y coordinates in the core feature space) and is clipped to [-1.1] as it a
#     requirement of the torch.grid_sample function. The feature parameter learns the best linear mapping between the feature
#     map from a given location, sample from Gaussian at train time but set to mean at eval time, and the unit's response with or without an additional elu non-linearity.
#     Args:
#         in_shape (list): shape of the input feature map [channels, width, height]
#         outdims (int): number of output units
#         bias (bool): adds a bias term
#         init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
#                             [expected: positive value <=1]
#         init_sigma_range (float): initialises sigma with Uniform([0.0, init_sigma_range]).
#                 It is recommended however to use a fixed initialization, for faster convergence.
#                 For this, set fixed_sigma to True.
#         batch_sample (bool): if True, samples a position for each image in the batch separately
#                             [default: True as it decreases convergence time and performs just as well]
#         align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
#                 It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
#                 behavior to pre PyTorch 1.3 functionality for comparability.
#         fixed_sigma (bool). Recommended behavior: True. But set to false for backwards compatibility.
#                 If true, initialized the sigma not in a range, but with the exact value given for all neurons.
#     """

#     def __init__(self, in_shape, outdims, bias, init_mu_range=0.5, init_sigma_range=0.5, batch_sample=True, align_corners=True, fixed_sigma=False, **kwargs):

#         super().__init__()
#         if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma_range <= 0.0:
#             raise ValueError("either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive")
#         self.in_shape = in_shape
#         c, w, h = in_shape
#         self.outdims = outdims
#         self.batch_sample = batch_sample
#         self.grid_shape = (1, outdims, 1, 2)
#         self.mu = Parameter(torch.Tensor(*self.grid_shape))  # mean location of gaussian for each neuron
#         self.sigma = Parameter(torch.Tensor(*self.grid_shape))  # standard deviation for gaussian for each neuron
#         self.features = Parameter(torch.Tensor(1, c, 1, outdims))  # feature weights for each channel of the core

#         if bias:
#             bias = Parameter(torch.Tensor(outdims))
#             self.register_parameter("bias", bias)
#         else:
#             self.register_parameter("bias", None)

#         self.init_mu_range = init_mu_range
#         self.init_sigma_range = init_sigma_range
#         self.align_corners = align_corners
#         self.fixed_sigma = fixed_sigma
#         self.initialize()


#     def initialize(self):
#         """
#         Initializes the mean, and sigma of the Gaussian readout along with the features weights
#         """
#         # store gradient [used for debug purpose]
# #         self.mu_grad = []
# #         self.sigma_grad = []
# #         self.feat_grad = []
        
#         self.mu.data.uniform_(-self.init_mu_range, self.init_mu_range)
#         if self.fixed_sigma:
#             self.sigma.data.uniform_(self.init_sigma_range, self.init_sigma_range)
#         else:
#             self.sigma.data.uniform_(0, self.init_sigma_range)
#             warnings.warn("sigma is sampled from uniform distribuiton, instead of a fixed value. Consider setting "
#                           "fixed_sigma to True")
#         self.features.data.fill_(1 / self.in_shape[0])

#         if self.bias is not None:
#             self.bias.data.fill_(0)

#     def sample_grid(self, batch_size, sample=None):
#         """
#         Returns the grid locations from the core by sampling from a Gaussian distribution
#         Args:
#             batch_size (int): size of the batch
#             sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
#                             or use the mean, mu, of the Gaussian distribution without sampling.
#                            if sample is None (default), samples from the N(mu,sigma) during training phase and
#                              fixes to the mean, mu, during evaluation phase.
#                            if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
#         """
#         with torch.no_grad():
#             self.mu.clamp_(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
#             self.sigma.clamp_(min=0)  # sigma/variance is always a positive quantity

#         grid_shape = (batch_size,) + self.grid_shape[1:]

#         sample = self.training if sample is None else sample

#         if sample:
#             norm = self.mu.new(*grid_shape).normal_()
#         else:
#             norm = self.mu.new(*grid_shape).zero_()  # for consistency and CUDA capability

#         return torch.clamp(
#             norm * self.sigma + self.mu, min=-1, max=1
#         )  # grid locations in feature space sampled randomly around the mean self.mu

#     @property
#     def grid(self):
#         return self.sample_grid(batch_size=1, sample=False)

#     def feature_l1(self, average=True):
#         """
#         Returns the l1 regularization term either the mean or the sum of all weights
#         Args:
#             average(bool): if True, use mean of weights for regularization
#         """
#         if average:
#             return self.features.abs().mean()
#         else:
#             return self.features.abs().sum()

#     def forward(self, x, sample=None, shift=None, out_idx=None):
#         """
#         Propagates the input forwards through the readout
#         Args:
#             x: input data
#             sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
#                             or use the mean, mu, of the Gaussian distribution without sampling.
#                            if sample is None (default), samples from the N(mu,sigma) during training phase and
#                              fixes to the mean, mu, during evaluation phase.
#                            if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
#             shift (bool): shifts the location of the grid (from eye-tracking data)
#             out_idx (bool): index of neurons to be predicted
#         Returns:
#             y: neuronal activity
#         """
#         # learn the grid from the original scale input
#         # stores gradient [used for debug purpose]
# #         if self.mu.grad is not None:
# #             self.mu_grad.append(self.mu.grad.detach().clone().cpu().numpy())
# #         if self.sigma.grad is not None:
# #             self.sigma_grad.append(self.sigma.grad.detach().clone().cpu().numpy())
# #         if self.features.grad is not None:
# #             self.feat_grad.append(self.features.grad.detach().clone().cpu().numpy())
            
#         N, c, w, h = x.size()
#         c_in, w_in, h_in = self.in_shape
#         if (c_in, w_in, h_in) != (c, w, h):
#             raise ValueError("the specified feature map dimension is not the readout's expected input dimension")
#         feat = self.features.view(1, c, self.outdims)
#         bias = self.bias
#         outdims = self.outdims

#         if self.batch_sample:
#             # sample the grid_locations separately per image per batch
#             grid = self.sample_grid(batch_size=N, sample=sample)  # sample determines sampling from Gaussian
#         else:
#             # use one sampled grid_locations for all images in the batch
#             grid = self.sample_grid(batch_size=1, sample=sample).expand(N, outdims, 1, 2)

#         if out_idx is not None:
#             if isinstance(out_idx, np.ndarray):
#                 if out_idx.dtype == bool:
#                     out_idx = np.where(out_idx)[0]
#             feat = feat[:, :, out_idx]
#             grid = grid[:, out_idx]
#             if bias is not None:
#                 bias = bias[out_idx]
#             outdims = len(out_idx)

#         if shift is not None:
#             y = F.grid_sample(x, grid + shift[:, None, None, :])
#         else: 
#             y = F.grid_sample(x, grid)
        
#         y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)
#         if self.bias is not None:
#             y = y + bias
            
#         return y

#     def __repr__(self):
#         c, w, h = self.in_shape
#         r = self.__class__.__name__ + " (" + "{} x {} x {}".format(c, w, h) + " -> " + str(self.outdims) + ")"
#         if self.bias is not None:
#             r += " with bias"
#         for ch in self.children():
#             r += "  -> " + ch.__repr__() + "\n"
#         return r
