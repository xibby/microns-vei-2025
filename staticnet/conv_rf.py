import math
import numpy as np
import torch
import torch.nn as nn
from random import sample
import torch.nn.functional as F
    
    
class ConvRF(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        num_kernels=1,
        filters=None):
        
        super(ConvRF, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        self.filters = filters

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)


class Conv2dRF(ConvRF):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        num_kernels=1,
        filters=None):
        
        super(Conv2dRF, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            num_kernels=num_kernels,
            filters=filters)
        
        """
        :param num_kernels: num of randomly selected kernels out of param "filters"
        :param filters: numpy array of filters. It must of shape = (num_filters, h, w)
        """
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        
        assert filters.ndim == 3
        self.kernel_size = (filters.shape[1], filters.shape[2])
        
        # this will choose the filters out of param "filters" for the linear combinations
        kernels = torch.Tensor(self.build_fixed_kernels(
            filters,
            self.num_kernels,
            self.out_channels,
            self.in_channels))
        
        # -----
        # If you have parameters in your model, 
        # which should be saved and restored in the state_dict, 
        # but not trained by the optimizer, 
        # you should register them as buffers.
        # Buffers won’t be returned in model.parameters()
        # -----
        
        self.register_buffer("kernels", kernels)
        # initialize the coefficients for the linear combinations
        self.weight = nn.Parameter(torch.Tensor(
            self.out_channels, 
            self.in_channels // self.groups, 
            self.num_kernels))
        
        # nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / sqrt(in_channels)
        # self.bias.data.uniform_(-stdv, stdv)
        self.reset_parameters()

    def reset_parameters(self):
#         nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def build_fixed_kernels(self, filters, num_kernels, out_channels, in_channels):
        """
        :param filters:  np.array of fixed kernels. Must be of shape (#filters, h, w)
        :param num_kernels: number of randomly selected fixed kernels per channel
        """
        assert filters.ndim == 3
        n = filters.shape[0]
        h = filters.shape[1]
        w = filters.shape[2]
        
        channels = np.zeros((out_channels, in_channels // self.groups, num_kernels, h, w), dtype=np.float32)
        for k in range(out_channels):
            for j in range(in_channels // self.groups):
                channels[k, j] = filters[sample(range(n), num_kernels), :, :]
        return channels

    def forward(self, x):
        self.kernel = torch.einsum(
            "ijk, ijklm -> ijlm", 
            self.weight, 
            self.kernels)

        return F.conv2d(
            input=x,
            weight=self.kernel,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)


class Conv3dRF(ConvRF):

    def __init__(
        self,
        in_channels,
        out_channels,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        num_kernels=1,
        filters=None):
        super(Conv3dRF, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            num_kernels=num_kernels,
            filters=filters)
        """
        :param num_kernels: num of randomly selected kernels out of param "filters"
        :param filters: numpy array of filters. It must of shape = (num_filters, d, h, w)
        """

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        assert filters.ndim == 4
        self.kernel_size = (filters.shape[1], filters.shape[2], filters.shape[3])
        
        # this will choose the filters out of param "filters" for the linear combinations
        kernels = torch.Tensor(self.build_fixed_kernels(
            filters,
            self.num_kernels,
            self.out_channels,
            self.in_channels))
        
        # -----
        # If you have parameters in your model, 
        # which should be saved and restored in the state_dict, 
        # but not trained by the optimizer, 
        # you should register them as buffers.
        # Buffers won’t be returned in model.parameters()
        # -----
        self.register_buffer("kernels", kernels)
        # initialize the coefficients for the linear combinations
        self.weight = nn.Parameter(torch.Tensor(
        	self.out_channels, 
        	self.in_channels // self.groups, 
        	self.num_kernels))
        
        # nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / sqrt(in_channels)
        # self.bias.data.uniform_(-stdv, stdv)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def build_fixed_kernels(self, filters, num_kernels, out_channels, in_channels):
        """
        :param filters:  np.array of fixed kernels must be of shape (#filters, d, h, w)
        :param num_kernels: number of randomly selected fixed kernels per channel
        """
        assert filters.ndim == 4
        n = filters.shape[0]
        d = filters.shape[1]
        h = filters.shape[2]
        w = filters.shape[3]
        
        channels = np.zeros((out_channels, in_channels // self.groups, num_kernels, d, h, w), dtype=np.float32)
        for k in range(out_channels):
            for j in range(in_channels // self.groups):
                channels[k, j] = filters[sample(range(n), num_kernels), :, :, :]
        return channels

    def forward(self, x):
        self.kernel = torch.einsum(
            "ijk, ijklmn -> ijlmn", 
            self.weight, 
            self.kernels)
        
        return F.conv3d(
            input=x,
            weight=self.kernel,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
