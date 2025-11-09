from collections import OrderedDict, defaultdict
from itertools import count
from warnings import warn

from torch import nn
from torch.nn.init import xavier_normal
import torch
from torch.nn import Parameter
import torch.nn.init as init
from attorch.regularizers import LaplaceL2
from torch.autograd import Variable
from attorch.module import ModuleDict
from torch.hub import load_state_dict_from_url
from . import logger as log

from neuralpredictors.layers.cores import Stacked2dCore as Stacked2dNewCore

# from staticnet.conv_rf import Conv2dRF
# from staticnet.parseval import Parseval


# filters3x3 = Parseval(
#     shape=[3, 3],
#     low_pass_kernel='gauss',
#     first_order=True,
#     second_order=True,
#     type_of_out_frame='frame').pframe()

# filters15x15 = Parseval(
#     shape=[15, 15],
#     low_pass_kernel='gauss',
#     first_order=True,
#     second_order=True,
#     type_of_out_frame='frame').pframe()

# filters7x7 = Parseval(
#     shape=[7, 7],
#     low_pass_kernel='gauss',
#     first_order=True,
#     second_order=True,
#     type_of_out_frame='frame').pframe()

class Core:
    def initialize(self):
        log.info('Not initializing anything')

    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: 'gamma' in x or 'skip' in x, dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'


class Core2d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        if cuda:
            self = self.cuda()

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.fill_(0)


class Core3d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        if cuda:
            self = self.cuda()

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.fill_(0)


# ---------------------- Identity Core ---------------------------
class IdentityCore(nn.Module, Core2d):
    def __init__(self, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

    def forward(self, x):
        return x

    def regularizer(self):
        return 0.0


# ---------------------- Conv2d Cores -----------------------------

class Stacked2dCore(Core2d, nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, layers=3,
                 gamma_hidden=0, gamma_input=0., skip=0, final_nonlinearity=True, bias=False, skip_nonlin=False,
                 momentum=0.1, pad_input=True, batch_norm=True, laplace_padding=0, laplace_weights_fn=None, **kwargs):
        log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        if skip_nonlin:
            print('Skip non-linearity')


        self._input_weights_regularizer = LaplaceL2(padding=laplace_padding)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.skip = skip

        self.features = nn.Sequential()
        # --- first layer
        layer = OrderedDict()
        layer['conv'] = \
            nn.Conv2d(input_channels, hidden_channels, input_kern,
                      padding=input_kern // 2 if pad_input else 0, bias=bias)
        
        if batch_norm:
            layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if (not skip_nonlin) and (layers > 1 or final_nonlinearity):
            layer['nonlin'] = nn.ELU(inplace=True)
        self.features.add_module('layer0', nn.Sequential(layer))

        # --- other layers
        for l in range(1, self.layers):
            layer = OrderedDict()
            layer['conv'] = \
                nn.Conv2d(hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                          hidden_channels, hidden_kern,
                          padding=hidden_kern // 2, bias=bias)

            if batch_norm:
                layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            if (not skip_nonlin) and (final_nonlinearity or l < self.layers - 1):
                layer['nonlin'] = nn.ELU(inplace=True)
            self.features.add_module('layer{}'.format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

        if laplace_weights_fn is not None:
            _, _, h, w = self.features[0].conv.weight.size()
            dist_grid = torch.sqrt(torch.linspace(-1, 1, h)[:, None].pow(2) + torch.linspace(-1, 1, w).pow(2))
            self.register_buffer('laplace_weights', laplace_weights_fn(dist_grid))
        else:
            self.laplace_weights = None


    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
            ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight, weights=self.laplace_weights)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels


# class StackedRFCore(Core2d, nn.Module):
#     def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, division_num_kernals, layers=3,
#                  gamma_hidden=0, gamma_input=0., skip=0, final_nonlinearity=True, bias=False, skip_nonlin=False,
#                  momentum=0.1, pad_input=True, batch_norm=True, laplace_padding=0, laplace_weights_fn=None, **kwargs):
#         log.info('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
#         super().__init__()

#         if skip_nonlin:
#             print('Skip non-linearity')


#         self._input_weights_regularizer = LaplaceL2(padding=laplace_padding)

#         self.layers = layers
#         self.gamma_input = gamma_input
#         self.gamma_hidden = gamma_hidden
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.division_num_kernals = division_num_kernals
#         self.skip = skip

#         self.features = nn.Sequential()
#         # --- first layer
#         layer = OrderedDict()
        
#         input_filters = Parseval(
#             shape=[input_kern, input_kern],
#             low_pass_kernel='gauss',
#             first_order=True,
#             second_order=True,
#             type_of_out_frame='frame').pframe()

#         layer['conv'] = Conv2dRF(
#             input_channels,
#             hidden_channels,
#             padding=input_kern // 2 if pad_input else 0,
#             bias=False,
#             num_kernels=len(input_filters) // self.division_num_kernals, # use all filters
#             filters=input_filters)

#         if batch_norm:
#             layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
#         if (not skip_nonlin) and (layers > 1 or final_nonlinearity):
#             layer['nonlin'] = nn.ELU(inplace=True)
#         self.features.add_module('layer0', nn.Sequential(layer))

#         # --- other layers
#         hidden_filters = Parseval(
#             shape=[hidden_kern, hidden_kern],
#             low_pass_kernel='gauss',
#             first_order=True,
#             second_order=True,
#             type_of_out_frame='frame').pframe()

#         for l in range(1, self.layers):
#             layer = OrderedDict()
    
#             layer['conv'] = Conv2dRF(
#                 hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
#                 hidden_channels,
#                 padding=hidden_kern // 2,
#                 bias=False,
#                 num_kernels=len(hidden_filters) // self.division_num_kernals,
#                 filters=hidden_filters)

#             if batch_norm:
#                 layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
#             if (not skip_nonlin) and (final_nonlinearity or l < self.layers - 1):
#                 layer['nonlin'] = nn.ELU(inplace=True)
#             self.features.add_module('layer{}'.format(l), nn.Sequential(layer))

#         self.apply(self.init_conv)

#         if laplace_weights_fn is not None:
#             _, _, h, w = self.features[0].conv.weight.size()
#             dist_grid = torch.sqrt(torch.linspace(-1, 1, h)[:, None].pow(2) + torch.linspace(-1, 1, w).pow(2))
#             self.register_buffer('laplace_weights', laplace_weights_fn(dist_grid))
#         else:
#             self.laplace_weights = None


#     def forward(self, input_):
#         ret = []
#         for l, feat in enumerate(self.features):
#             do_skip = l >= 1 and self.skip > 1
#             input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
#             ret.append(input_)
#         return torch.cat(ret, dim=1)

#     def laplace(self):
#         return self._input_weights_regularizer(self.features[0].conv.weight, weights=self.laplace_weights)

#     def group_sparsity(self):
#         ret = 0
#         for l in range(1, self.layers):
#             ret = ret + self.features[l].conv.weight.pow(2).sum(2, keepdim=True).sqrt().mean()
#         return ret / ((self.layers - 1) if self.layers > 1 else 1)

#     def regularizer(self):
#     #     return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()
#         return 0
        
#     @property
#     def outchannels(self):
#         return len(self.features) * self.hidden_channels


class LinearCore(Stacked2dCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, layers=1, skip=0, final_nonlinearity=False, bias=False,
                         hidden_kern=0, gamma_hidden=0)


class SigmoidLaplaceCore(Stacked2dCore):
    def __init__(self, sigmoid_scale, sigmoid_center, *args, **kwargs):
        laplace_weights_fn = lambda x: torch.sigmoid(sigmoid_scale * (x - sigmoid_center))
        super().__init__(*args, laplace_weights_fn=laplace_weights_fn, **kwargs)


class LinearSigmoidLaplaceCore(SigmoidLaplaceCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, layers=1, skip=0, final_nonlinearity=False, bias=False,
                         hidden_kern=0, gamma_hidden=0)

class GaussianLaplaceCore(Stacked2dCore):
    def __init__(self, gauss_sigma, gauss_bias, *args,  **kwargs):
        laplace_weights_fn = lambda x: 1 - torch.exp(-x.pow(2) / 2 / gauss_sigma**2) + gauss_bias
        super().__init__(*args, laplace_weights_fn=laplace_weights_fn, **kwargs)

class LinearGaussianLaplaceCore(GaussianLaplaceCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, layers=1, skip=0, final_nonlinearity=False, bias=False,
                         hidden_kern=0, gamma_hidden=0)

class StackedLinearGaussianLaplaceCore(Stacked2dCore):
    def __init__(self, gauss_sigma, gauss_bias, *args,  **kwargs):
        laplace_weights_fn = lambda x: 1 - torch.exp(-x.pow(2) / 2 / gauss_sigma**2) + gauss_bias
        super().__init__(*args, skip_nonlin=True, laplace_weights_fn=laplace_weights_fn, **kwargs)
    
class VGG19Core(nn.Module):
    # VGG19 with batch normalization
    def __init__(self,input_channels, pretrained, fine_tuning, output_layer, gamma, **kwargs):
        super(VGG19Core, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.features = self.make_layers(self.cfg[:output_layer], batch_norm=True)
        assert input_channels == 1, "VGG19 core only accepts one channel input and will expand it to three channels."
        self.fine_tuning = fine_tuning
        self.pretrained = pretrained
        self.output_layer = output_layer
        self.gamma = gamma
        self.initialize()

    def forward(self, x):
        x = x.expand((-1,3,-1,-1))
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, batch_norm):
        input_channels = 3
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(input_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                input_channels = v
        return nn.Sequential(*layers)

    def regularizer(self):
        # L2 norm regularization
        ret = 0
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                ret = ret + layer.weight.pow(2).sum()
        return ret * self.gamma

    def initialize(self):
        if self.pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
                                                progress=True)
            self.load_state_dict(state_dict, strict=False)  # partially load the pretrained parameter
        else:
            self._initialize_weights()
        if not self.fine_tuning:
            for param in self.parameters():
                param.requires_grad = False
        
# alias just for use of modified stacked 2d core
ModStacked2dCore = Stacked2dCore

ModLinearCore = LinearCore


