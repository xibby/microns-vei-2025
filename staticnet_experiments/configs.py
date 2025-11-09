from collections import OrderedDict, namedtuple
from functools import partial
from inspect import isclass
from itertools import product, repeat
import torch
from tqdm import tqdm

from attorch.layers import Elu1
import attorch.losses as losses
from attorch.losses import PoissonLoss, MSE
import datajoint as dj

import numpy as np

from attorch.train import early_stopping
from attorch.utils import cycle_datasets
from neuro_data.static_images.configs import DataConfig
from neuro_data.static_images import data_schemas
from staticnet import cores, modulators, readouts, shifters, new_readouts
from staticnet.base import CorePlusReadout2d, BehaviorNet, LocomotionNet


from .utils import key_hash, to_native, compute_predictions, corr, correlation_closure, correlation, loss_closure
from .mixins import TrainMixin, TestMixin
from . import logger as log

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')
meso = dj.create_virtual_module('meso', 'pipeline_meso')
crossval = dj.create_virtual_module('neurostatic_crossval', 'neurostatic_crossval')
models = dj.create_virtual_module('neurostatic_models', 'neurostatic_models')
schema = dj.schema('neurostatic_configs', locals())

Datasets = namedtuple('Datasets', ['trainsets', 'valsets', 'testsets', 'mu_dict', 'n_neurons'])
dj.config['enable_python_native_blobs'] = True

class Config:
    _config_type = None

    @property
    def definition(self):
        return """
        # parameters for {cn}

        {ct}_hash                   : varchar(256) # unique identifier for configuration
        {extra_foreign} 
        ---
        {ct}_type                   : varchar(50)  # type
        {ct}_ts=CURRENT_TIMESTAMP : timestamp      # automatic
        """.format(ct=self._config_type, cn=self.__class__.__name__,
                   extra_foreign=self._extra_foreign if hasattr(self, '_extra_foreign') else '')

    def fill(self):
        type_name = self._config_type + '_type'
        hash_name = self._config_type + '_hash'
        for rel in [getattr(self, member) for member in dir(self)
                    if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]:
            log.info('Checking ' + rel.__name__)
            for key in rel().content:
                key[type_name] = rel.__name__
                key[hash_name] = key_hash(key)

                if not key in (rel()).proj():
                    self.insert1(key, ignore_extra_fields=True)
                    log.info('Inserting ' + repr(key))
                    rel().insert1(key, ignore_extra_fields=True)

    def parameters(self, key, selection=None, exclude_timestamp=True):
        type_name = self._config_type + '_type'
        ts_name = self._config_type + '_ts'
        key = (self & key).fetch1()  # complete parameters
        part = getattr(self, key[type_name])
        ret = (self * part() & key).fetch1()
        ret = to_native(ret)
        if exclude_timestamp:
            del ret[ts_name]
        if selection is None:
            return ret
        else:
            if isinstance(selection, list):
                return tuple(ret[k] for k in selection)
            else:
                return ret[selection]

    def select_hashes(self):
        configs = [getattr(self, member) for member in dir(self) if
                   isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]
        print('\n'.join(['({}) {}'.format(i, rel.__name__) for i, rel in enumerate(configs)]))

        choices = input('Please select configuration [comma separated list]: ')
        choices = list(map(int, choices.split(',')))

        hashes = []
        for choice in choices:
            restriction = dict()
            rel = configs[int(choice)]()
            while restriction != '':
                old_restriction = restriction
                print(old_restriction)
                print(rel & old_restriction)
                restriction = input('Please enter a restriction [ENTER for exit]: ')
            hashes.extend((rel & old_restriction).fetch('{}_hash'.format(self._config_type)))
        return '{}_hash'.format(self._config_type), hashes

    def part_table(self, key=None):
        key = {} if key is None else key
        return getattr(self & key, (self & key).fetch1('{}_type'.format(self._config_type))) & (self & key)


@schema
class Seed(dj.Lookup):
    definition = """
    # random seed for training

    seed                 :  int # random seed
    ---
    """

    @property
    def contents(self):
        yield from zip([1009, 1215, 2606, 99999, 101, 102, 103, 104])


@schema
class CoreConfig(Config, dj.Lookup):
    _config_type = 'core'

    class Stacked2d(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels           : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        """

        @property
        def content(self):
            for p in product([32], [7], [3], [3], [.1], [50], [3], [True], [False], [False]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
            for p in product([32], [18], [1], [3], [0], [50], [3], [True], [False], [False]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
            for p in product([32], [7], [7], [3], [0.1], [50], [0], [True], [False], [False]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
            for p in product([32], [13], [7], [3], [0.1], [50], [0], [True], [False], [False]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class Stacked2dNew(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels         : int      # Integer or list of numbers of hidden channels (i.e feature maps) in each hidden layer
        input_kern              : int      # kernel size of the first layer (i.e. the input layer)
        hidden_kern             : int      # kernel size of each hidden layer's kernel
        layers                  : int      # number of layers
        gamma_hidden            : double   # regularizer factor for group sparsity
        gamma_input             : double   # regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
        skip                    : bool     # Adds a skip connection
        stride                  : int      # stride of the 2d conv layer
        final_nonlinearity      : bool     # Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
        elu_shift               : blob     # a tuple to shift the elu in the following way: Elu(x - elu_xshift) + elu_yshift
        bias                    : bool     # Adds a bias layer.
        momentum                : double   # momentum in the batchnorm layer.
        pad_input               : bool     # Boolean, if True, applies zero padding to all convolutions
        hidden_padding          : int      # int or list of int. Padding for hidden layers. Note that this will apply to all the layers except the first (input) layer.
        batch_norm              : bool     # Boolean, if True appends a BN layer after each convolutional layer batch_norm_scale If True, a scaling factor after BN will be learned.
        final_batchnorm_scale   : bool     # If True, the final layer's BN will learn a scale. Defaults to True.
        hidden_dilation         : int      # If set to > 1, will apply dilated convs for all hidden layers
        laplace_padding         : int      # Padding size for the laplace convolution. If padding = None, it defaults to half of the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts, zero is the default however to recreate backwards compatibility.
        input_regularizer       : varchar(16)  # String that must match one of the regularizers in ..regularizers
        stack                   : int      # Int or iterable. Selects which layers of the core should be stacked for the readout. default value will stack all layers on top of each other.
        use_avg_reg             : bool     # Whether to use the averaged value of regularizer(s) or the summed.
        depth_separable         : bool     # Boolean, if True, uses depth-separable convolutions in all layers after the first one.
        attention_conv          : bool     # Boolean, if True, uses self-attention instead of convolution for all layers after the first one.
        linear                  : bool     # Boolean, if True, removes all nonlinearities
        nonlinearity_type       : varchar(16)  # String to set the used nonlinearity type loaded from neuralpredictors.layers.activation
        nonlinearity_config     : longblob # Dict of the nonlinearities __init__ parameters.
        """
        
        @property
        def content(self):
            for p in product([64], [9], [7], [4], [0], [6.3831], [False], [1], [True], [(0, 0)], [True], [0.9], [False], [None], [True], [True], [1], [None], ['LaplaceL2norm'], [-1], [False], [True], [False], [False], ["AdaptiveELU"], [None]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
        
    class ModStacked2d(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels           : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        """

        @property
        def content(self):
            for p in product([32], [11, 15], [3, 7], [3], [0.1, 1.0], [50, 100], [3], [True], [False], [False], [1]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class Linear(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        """

        @property
        def content(self):
            for p in product([32], [13], [50]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class ModLinear(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        laplace_padding       : int      # laplace padding
        """

        @property
        def content(self):
            for p in product([32], [13, 19, 23], [50], [0, 1]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d


    class SigmoidLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        sigmoid_scale         : float    # scaling for distance in sigmoid weighting
        sigmoid_center        : float    # center of sigmoid distance
        """

        @property
        def content(self):
            for p in product([32], [11, 15], [3, 7], [3], [0.1, 1.0], [50, 100], [3],
                             [True], [False], [False], [1], [1, 2], [0.5]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class LinearSigmoidLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        laplace_padding       : int      # laplace padding
        sigmoid_scale         : float    # scaling for distance in sigmoid weighting
        sigmoid_center        : float    # center of sigmoid distance
        """

        @property
        def content(self):
            for p in product([32], [13, 19, 23], [50, 100], [1], [1, 2], [0.5]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class GaussianLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        gauss_sigma           : float    # sigma of Gaussian regularization weight
        gauss_bias            : float    # base line shift
        """

        @property
        def content(self):
            # for p in product([32], [7, 11, 15], [3, 3, 7], [3], [0.1, 1.0, 10], [50, 100, 1000], [3],
            #                  [True], [False], [False], [1], [0.5, 1], [0.0, 0.5]):
            #     d = dict(zip(self.heading.secondary_attributes, p))
            #     yield d
            for p in product([32], [7, 11, 15, 19, 23, 27], [3, 7, 11, 15, 19, 23], [3], 
                             [0.1, 1.0, 10], [50.0, 100.0, 1000.0], [3], [True], [False], [False], 
                             [1], [0.5, 1.0], [0.0, 0.5]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class StackedLinearGaussianLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        gauss_sigma           : float    # sigma of Gaussian regularization weight
        gauss_bias            : float    # base line shift
        """

        @property
        def content(self):
            for p in product([32], [15], [7], [3], [0.1, 1.0], [50, 100], [3],
                             [True], [False], [False], [1], [0.5, 1], [0.0, 0.5]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class LinearGaussianLaplace(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        laplace_padding       : int      # laplace padding
        gauss_sigma           : float    # sigma of Gaussian regularization weight
        gauss_bias            : float    # base line shift
        """

        @property
        def content(self):
            for p in product([32], [13, 19, 23], [50, 100], [1], [0.5, 1], [0.0, 0.5]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
    

    class VGG19(dj.Part):
        definition = """
        -> master
        ---
        pretrained              : bool       # 1 for pretrained params, 0 for randomly initiated params
        fine_tuning             : bool       # 1 for trainable params, 0 for fixed params
        output_layer            : int        # 1-21, indicate which layer to read from
        gamma                   : double     # regularization constant for convolutional layers
        """

        @property
        def content(self):
            for p in product([1, 0], [1, 0], [4, 7, 12, 17], [0]):  # 7: conv3_1, see staticnet.cores.VGG19Core for more info
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    
    class StackedRF(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : int      # introduce skip connections if skip > 1
        final_nonlinearity    : bool     # end last layer of core with nonlinearity
        bias                  : bool     # use bias or not
        pad_input             : bool     # pad input layers
        laplace_padding       : int      # laplace padding
        gauss_sigma           : float    # sigma of Gaussian regularization weight
        gauss_bias            : float    # base line shift
        division_num_kernals  : int      # divide num_kernals by this number to control the number of filters used in each conv layer
        """

        @property
        def content(self):
            for p in product([32], [11, 15, 7], [3, 7, 3], [3], [0.1, 1.0], [50, 100], [3],
                             [True], [False], [False], [1], [0.5, 1], [0.0, 0.5], [1, 4, 10]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d


    def build(self, input_channels, key, **kwargs):
        core_key = self.parameters(key)

        core_type = core_key.pop('core_type')

        # let parameters be specially parsed if `build_key` is defined
        CoreConfig = getattr(self, core_type)
        core_config = CoreConfig()
        if hasattr(core_config, 'build_key'):
            core_key = core_config.build_key(core_key)

        # get the core module
        core_name = '{}Core'.format(core_type)
        assert hasattr(cores, core_name), '''Cannot find core for {core_name}.
                                             Core needs to be names "{core_name}Core"
                                             in architectures.cores'''.format(core_name=core_name)
        Core = getattr(cores, core_name)

        return Core(input_channels=input_channels, **core_key, **kwargs)



@schema
class ReadoutConfig(Config, dj.Lookup):
    _config_type = 'ro'

    def build(self, in_shape, neurons, key):
        ro_key = self.parameters(key)
        # ro_key['grid_mean_predictor'] = None
        # ro_key['init_sigma'] = 0.1
        # ro_key['gamma_readout'] = 0.0076
        ro_type = ro_key.pop('ro_type')
        ro_name = '{}Readout'.format(ro_type)
        ro_table = getattr(self, ro_type)()
        # let ro specific part table modify the key
        if hasattr(ro_table, 'build_key'):
            ro_key['in_shape'] = in_shape
            ro_key = ro_table.build_key(key, ro_key)
            in_shape = ro_key['in_shape']
            del ro_key['in_shape']
        assert hasattr(readouts, ro_name), '''Cannot find readout for {ro_name}.
                                             Core needs to be names "{ro_name}"
                                             in architectures.readout'''.format(ro_name=ro_name)
        Readout = getattr(readouts, ro_name)

        return Readout(in_shape, neurons, **ro_key)

    class SpatialXFeatures(dj.Part):
        definition = """
        -> master
        ---
        gamma_readout          : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        normalize              : bool  # whether the spatial features will be normalized
        """

        @property
        def content(self):
            for p in product([1], [False], [True]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class SpatialTransformerPooled2d(dj.Part):
        definition = """
        -> master
        ---
        gamma_features         : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        pool_steps             : tinyint  # number of pooling steps in the readout
        """

        @property
        def content(self):
            for p in product([20], [False], [4]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
                 
    class ModifiedSpatialTransformerPyramid2d(dj.Part):
        definition = """
        -> master
        ---
        type                   : varchar(20) # regularization constant for features
        gamma_features         : float   # regularization constant for features
        positive               : bool    # whether the features will be restricted to be positive
        scale_n                : tinyint # number of pooling steps in the readout
        downsample             : bool    # whether to downsample lowpass representations
        """

        @property
        def content(self):
            for p in product(['gauss5x5'], [1], [False], [5], [True, False]):
                d = dict(zip(self.heading.secondary_attributes, p))
                d['_skip_upsampling'] = True
                yield d

    class SpatialTransformerPyramid2d(dj.Part):
        definition = """
        -> master
        ---
        type                   : varchar(20) # regularization constant for features
        gamma_features         : float   # regularization constant for features
        positive               : bool    # whether the features will be restricted to be positive
        scale_n                : tinyint # number of pooling steps in the readout
        downsample             : bool    # whether to downsample lowpass representations
        """

        @property
        def content(self):
            for p in product(['gauss5x5'], [1, 10], [False], [5], [True, False]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
                

    class SpatialTransformer2d(dj.Part):
        definition = """
        -> master
        ---
        gamma_features         : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        """

        @property
        def content(self):
            for p in product([20, 1], [False]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
                
    class MultipleSpatialTransformerPyramid2d(dj.Part):
        definition = """
        -> master
        ---
        type                   : varchar(20) # regularization constant for features
        gamma_features         : float   # regularization constant for features
        positive               : bool    # whether the features will be restricted to be positive
        scale_n                : tinyint # number of pooling steps in the readout
        downsample             : bool    # whether to downsample lowpass representations
        clone_readout          : bool    # whether to share weight between readouts
        alpha_reg              : bool    # whether to include alpha in regularization
        """

        @property
        def content(self):
            for p in product(['gauss5x5'], [0.1, 0.5, 1, 5, 10], [False], [5], [True, False], [True, False], [True, False]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
                
    # Reference: https://github.com/sinzlab/nnsysident/blob/e1404c9e2d0649832b17dca6277fac4c39d1af04/nnsysident/models/models.py#L37
    class MultipleFullGaussian2d(dj.Part):
        definition = """
        -> master
        ---
        init_mu_range:              float # initialises the the mean with Uniform([-init_range, init_range]) [expected: positive value <=1]. Default: 0.1
        bias:                       tinyint # whether to add a bias term 
        init_sigma:                 float # the standard deviation of the Gaussian with `init_sigma` when `gauss_type` is 'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        gamma_readout:              float # regularization constant for features
        gauss_type:                 varchar(16) # which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        grid_mean_predictor:        longblob # a dictionary of parameters for a predictor of the mean grid locations. Has to have a form like {'hidden_layers':0, 'hidden_features':20, 'final_tanh': False}
        share_features:             tinyint # whether the feature vectors are shared (within readout between neurons) or between this readout and other readouts
        share_grid:                 tinyint # whehter the grids are shared 
        share_transform:            tinyint # whether the transform is shared 
        init_noise:                 float 
        init_transform_scale:       float
        cell_match_table:           varchar(45)   # table for cell matching 
        """

        @property
        def content(self):
            for p in product([0.1, 0.3, 0.5, 1], [True], [0.1, 0.5], [0.01, 0.05, 0.1, 1, 2], ['full'], [{}, {"type": "cortex", "input_dimensions": 2, "hidden_layers": 0, "hidden_features": 0, "final_tanh": False}], [True, False], [True, False], [True, False], [1e-3], [0.2], ['neurodata_static_configs.MultipleDatasets', 'neurostatic_crossval.UnitMatching']):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

        def build_key(self, key, ro_key): # compute and add additional keys: source_grids, grid_mean_predictor_type, shared_match_ids
            datasets, dataloaders = DataConfig().load_data(key)
            import copy
            ro_key = (self & {'ro_hash': ro_key['ro_hash']}).fetch1()
            grid_mean_predictor = ro_key['grid_mean_predictor']
            source_grids = None
            grid_mean_predictor_type = None
            if grid_mean_predictor:
                grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
                grid_mean_predictor_type = grid_mean_predictor['type'].item().item()
#                 grid_mean_predictor_type = grid_mean_predictor.pop("type")
                if grid_mean_predictor_type == "cortex":
                    input_dim = grid_mean_predictor['input_dimensions'].item()
#                     input_dim = grid_mean_predictor.pop("input_dimensions", 2)
                    source_grids = {}
                    for k, v in dataloaders.items():
                        # real data
                        coordinates = []
                        for unit in v.dataset.neurons.unit_ids: 
                            coordinates.append(((data_schemas.StaticMultiDataset.Member() & {'name': k}) * meso.ScanSet.UnitInfo & {'unit_id': unit}).fetch1('um_x', 'um_y', 'um_z'))
                        coordinates = np.stack(coordinates)
                        source_grids[k] = coordinates[:, :input_dim]
                else:
                    raise ValueError("Grid mean predictor type {} not understood.".format(grid_mean_predictor_type))

            shared_match_ids = None
            if ro_key['share_features'] or ro_key['share_grid']:
                # matched_ids are the unique ids that corresponds to different units that are matched from different datasets. 
                # In neuro_configs.MultipleDatasets.MatchedCells, the matched unit_ids from different datasets are already ordered, so we can just use the unit idx as match_ids
                if ro_key['cell_match_table'] == 'neurodata_static_configs.MultipleDatasets':
                    shared_match_ids = {k: np.arange(len(v.dataset.neurons.unit_ids)) for k, v in dataloaders.items()}
                    all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))
                elif ro_key['cell_match_table'] == 'neurostatic_crossval.UnitMatching':
                    shared_match_ids = {}
                    for k, v in dataloaders.items():
                        animal, session, scan = (data_schemas.StaticMultiDataset.Member & {'name': k}).fetch1('animal_id', 'session', 'scan_idx')
                        match_ids = (crossval.UnitMatching.Match * data_schemas.StaticMultiDataset.Member() & 'match_params = 1' & {'animal_id': animal, 'session': session, 'scan_idx': scan} & [{'unit_id': i} for i in v.dataset.neurons.unit_ids]).fetch('match_id', order_by='unit_id')
                        shared_match_ids[k] = match_ids
                    all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

                for match_id in shared_match_ids.values():
                    assert len(set(match_id) & all_multi_unit_ids) == len(all_multi_unit_ids), "All multi unit IDs must be present in all datasets"
                    
            ro_key['source_grids'] = source_grids
            ro_key['grid_mean_predictor_type'] = grid_mean_predictor_type
            ro_key['shared_match_ids'] = shared_match_ids

            return ro_key
        
        
    class MultipleFullGaussian2dNew(dj.Part):
        definition = """
        -> master
        ---
        gamma_readout          : float     # regularization strength, deprecated
        feature_reg_weight     : float     # regularization strength
        bias                   : bool      # adds a bias term
        init_mu_range          : float     # initialises the the mean with Uniform([-init_range, init_range]) [expected: positive value <=1]. Default: 0.1
        init_sigma             : float     # The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is 'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample           : bool      # if True, samples a position for each image in the batch separately [default: True as it decreases convergence time and performs just as well]
        align_corners          : bool      # Keyword agrument to gridsample for bilinear interpolation. It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type             : varchar(16) # Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        grid_mean_predictor    : longblob  # (dict) Parameters for a predictor of the mean grid locations. 
        shared_features        : longblob  # (dict) Used when the feature vectors are shared (within readout between neurons) or between this readout and other readouts. Has to be a dictionary of the form
        shared_grid            : longblob  # (dict) Like `shared_features`
        source_grid            : longblob  # (ndarray) Source grid for the grid_mean_predictor. Needs to be of size neurons x grid_mean_predictor[input_dimensions]
        """
        
        @property
        def content(self):
            for p in product([0.0076], [None], [True], [0.3], [0.1], [True], [True], ['full'], [None, {"type": "cortex", "input_dimensions": 2, "hidden_layers": 1, "hidden_features": 30, "final_tanh": True}], [None], [None], [None]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
        
        def build_key(self, key, ro_key):
            # pipe_name = 'meso' if (data_schemas.StaticMultiDatasetGroupAssignment & key).fetch('spike_method')[0] >= 0 else 'neuropixel'
            # pipe = dj.create_virtual_module(pipe_name, 'pipeline_' + pipe_name)

            _, dataloaders = DataConfig().load_data(key)
            in_shape = {k:ro_key['in_shape'] for k in dataloaders.keys()}
            # mean_activity_dict = {k:torch.as_tensor(v.dataset.responses[v.dataset.tiers == 'train'].mean(0), dtype=torch.float32, device='cuda') for k, v in dataloaders.items()}
            
            import copy
            grid_mean_predictor = ro_key['grid_mean_predictor']
            if grid_mean_predictor is None:
                grid_mean_predictor_type = None
                source_grids = None
            else:
                grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
                grid_mean_predictor_type = grid_mean_predictor.pop("type")
                source_grids = {}
                if grid_mean_predictor_type == "cortex":
                    input_dim = grid_mean_predictor.pop("input_dimensions", 2)
                    for k, v in dataloaders.items():
                        coordinates = (
                            (data_schemas.StaticMultiDataset.Member() & {'name': k}) * \
                            meso.ScanSet.UnitInfo & \
                            [{'unit_id': unit} for unit in v.dataset.neurons.unit_ids]
                        ).fetch('um_x', 'um_y', 'um_z', order_by='unit_id')
                        assert len(coordinates[0]) == len(v.dataset.neurons.unit_ids), 'number of units does not match!'
                        source_grids[k] = np.stack(coordinates)[:input_dim, :].T # num_neurons * input_dim
            
            ro_key['grid_mean_predictor'] = grid_mean_predictor      
            ro_key['grid_mean_predictor_type'] = grid_mean_predictor_type
            ro_key['source_grids'] = source_grids  
            ro_key['in_shape'] = in_shape
            # ro_key['mean_activity_dict'] = mean_activity_dict
            
            return ro_key
            
@schema
class ModulatorConfig(Config, dj.Lookup):
    _config_type = 'mod'

    def build(self, data_keys, input_features, key):
        mod_key = self.parameters(key)
        mod_name = '{}Modulator'.format(mod_key.pop('mod_type'))
        assert hasattr(modulators, mod_name), '''Cannot find modulator for {mod_name}.
                                             Core needs to be names "{mod_name}"
                                             in architectures.readout'''.format(mod_name=mod_name)
        Modulator = getattr(modulators, mod_name)
        return Modulator(data_keys, input_features, **mod_key)

    class No(dj.Part):
        definition = """
        -> master
        ---
        """

        @property
        def content(self):
            yield dict()

    class MLP(dj.Part):
        definition = """
        -> master
        ---
        layers                    : tinyint  # layers of MLP
        hidden_channels           : int      # hidden channels
        gamma_modulator           : double   # regularization constant for input  convolutional layers
        """

        @property
        def content(self):
            for p in product([2], [10], [0.0]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d


@schema
class ShifterConfig(Config, dj.Lookup):
    _config_type = 'shift'

    def build(self, data_keys, input_features, key):
        shift_key = self.parameters(key)
        shift_name = '{}Shifter'.format(shift_key.pop('shift_type'))
        assert hasattr(shifters, shift_name), '''Cannot find modulator for {shift_name}.
                                             Core needs to be names "{shift_name}"
                                             in architectures.readout'''.format(shift_name=shift_name)
        Shifter = getattr(shifters, shift_name)
        return Shifter(data_keys, input_features, **shift_key)

    class MLP(dj.Part):
        definition = """
          -> master
          ---
          shift_layers            : tinyint  # layers of MLP
          hidden_channels_shifter : int      # hidden channels
          gamma_shifter           : double   # regularization constant for input convolutional layers
          """

        @property
        def content(self):
            for p in product([3], [5], [0.0]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class StaticAffine2d(dj.Part):
        definition = """
        -> master
        ---
        gamma_shifter           : double   # regularization constant affine weights
        bias                    : bool     # whether to include bias term
        """

        @property
        def content(self):
            for p in product([1e-3], [True]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d

    class No(dj.Part):
        definition = """
        -> master
        ---
        """

        @property
        def content(self):
            yield dict()
        

@schema
class NetworkConfig(Config, dj.Lookup):
    _config_type = 'net'

    class CorePlusReadout(dj.Part, TestMixin, TrainMixin):
        definition = """
        -> master
        ---
        -> CoreConfig
        -> ReadoutConfig
        -> ShifterConfig
        -> ModulatorConfig
        -> TrainConfig
        -> DataConfig
        """

        @property
        def content(self):
            yield from (CoreConfig
                        * (ReadoutConfig - ReadoutConfig.SpatialTransformer2d - ReadoutConfig.SpatialXFeatures)
                        * ShifterConfig.StaticAffine2d
                        * ModulatorConfig.MLP
                        * (TrainConfig.Default & dict(batch_size=125))
                        * DataConfig.AreaLayer
                        & dict(normalize=1,
                               stimulus_type='stimulus.Frame',
                               exclude='images,responses')).fetch("KEY")

        @property
        def content_source(self):
            return (CoreConfig * ReadoutConfig * ShifterConfig * ModulatorConfig * TrainConfig * DataConfig).proj()


        def build_network(self, key, trainsets=None):
            """
            Builds a specified model
            Args:
                key:    key for CNNParameters used to load the parameter of the model. If None, (self & key) must
                        be non-empty so that key can be inferred.
                img_shape: image shape to figure out the size of the readouts
                n_neurons: dictionary with readout sizes (number of neurons)

            Returns:
                an uninitialized MultiCNN
            """
            
            key = dict((self & (key or dict())).fetch1(), **key)
            
            # --- load datasets
            if trainsets is None:
                train_key = TrainConfig().train_key(key)
                trainsets, _ = DataConfig().load_data(key, tier='train', **train_key)
            img_shape = list(trainsets.values())[0].img_shape
            n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])

            core = CoreConfig().build(img_shape[1], key)
            ro_in_shape = CorePlusReadout2d.get_readout_in_shape(core, img_shape)
            # ro_in_shape = OrderedDict([(k, ro_in_shape) for k, _ in trainsets.items()])
            readout = ReadoutConfig().build(ro_in_shape, n_neurons, key)
            shifter = ShifterConfig().build(n_neurons, input_features=2, key=key)
            modulator = ModulatorConfig().build(n_neurons, input_features=3, key=key)

            # --- initialize
            return CorePlusReadout2d(core, readout, nonlinearity=Elu1(), shifter=shifter, modulator=modulator)


        @property
        def source_restriction(self):
            restr = []
            for source in  [CoreConfig(), ReadoutConfig(), ShifterConfig(), ModulatorConfig(), TrainConfig(), DataConfig()]:
                hname = '{}_hash'.format(source._config_type)
                print('---', 'Selecting {} '.format(source.__class__.__name__).ljust(76, '-'))
                hash, hashes = source.select_hashes()
                restr.append('{} in ("{}")'.format(hname, '", "'.join(hashes)))
            return dj.AndList(restr)

        def select_hashes(self):
            hashes = []
            restriction = dict()
            rel = self
            while restriction != '':
                old_restriction = restriction
                print(old_restriction)
                print(rel & old_restriction)
                restriction = input('Please enter a restriction [ENTER for exit]: ')
            hashes.extend((rel & old_restriction).fetch('{}_hash'.format(self._config_type)))
            return 'net_hash', hashes


    class CorePlusReadoutNoDownsample(dj.Part, TestMixin, TrainMixin):
        definition = """
        -> master
        ---
        (parent_hash) -> master.CorePlusReadout
        (new_ro_hash)-> ReadoutConfig
        """

        @property
        def content(self):
            yield from self.content_source.fetch("KEY")

        @property
        def content_source(self):
            parents = (NetworkConfig.CorePlusReadout
                       & ReadoutConfig.SpatialTransformerPyramid2d).proj(parent_hash='net_hash')
            return (parents * ReadoutConfig.SpatialTransformer2d.proj(new_ro_hash='ro_hash')).proj()

        def parent_key(self, key):
            pkey = dict(key)
            pkey['net_hash'] = pkey.pop('parent_hash')
            return NetworkConfig().net_key(pkey)

        def complete_with_parent(self, key):
            pkey = self.parent_key(key)
            del pkey['net_hash']
            return dict(key, **pkey)

        def train(self, key):
            return super().train(self.complete_with_parent(key))

        def build_network(self, key, trainsets=None):
            """
            Builds a specified model
            Args:
                key:    key for CNNParameters used to load the parameter of the model. If None, (self & key) must
                        be non-empty so that key can be inferred.
                img_shape: image shape to figure out the size of the readouts
                n_neurons: dictionary with readout sizes (number of neurons)

            Returns:
                an uninitialized MultiCNN
            """
            key = self.complete_with_parent(key)
            from .models import Model

            model = Model().load_network(self.parent_key(key), trainsets=trainsets)

            img_shape = list(trainsets.values())[0].img_shape
            n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])

            core = CoreConfig().build(img_shape[1], key)
            ro_in_shape = CorePlusReadout2d.get_readout_in_shape(core, img_shape)
            readout = ReadoutConfig().build(ro_in_shape, n_neurons, dict(ro_hash=key['new_ro_hash']))
            for rok in model.readout:
                readout[rok].grid = model.readout[rok].grid

            shifter = ShifterConfig().build(n_neurons, input_features=2, key=key)
            modulator = ModulatorConfig().build(n_neurons, input_features=3, key=key)
            # --- initialize
            return CorePlusReadout2d(core, readout, nonlinearity=Elu1(), shifter=shifter, modulator=modulator)

        @property
        def source_restriction(self):
            restr = []

            print('===', 'Selecting NetworkConfig '.ljust(76, '='))
            restr.append('parent_hash in ("{}")'.format('", "'.join((self.master.CorePlusReadout()
                                                & self.master.CorePlusReadout().source_restriction).fetch('net_hash'))))
            print('===', 'Selecting ReadoutConfig '.ljust(76, '='))
            restr.append('new_ro_hash in ("{}")'.format('", "'.join(ReadoutConfig().select_hashes()[1])))

            return dj.AndList(restr)


    class BehaviorNet(dj.Part, TestMixin, TrainMixin):
        definition = """
        -> master
        ---
        -> CoreConfig
        -> ReadoutConfig
        -> ShifterConfig
        -> ModulatorConfig
        -> TrainConfig
        -> DataConfig
        """
        @property
        def content(self):
            yield from (CoreConfig
                        * (ReadoutConfig - ReadoutConfig.SpatialTransformer2d - ReadoutConfig.SpatialXFeatures)
                        * ShifterConfig.StaticAffine2d
                        * ModulatorConfig.MLP
                        * (TrainConfig.Default & dict(batch_size=125))
                        * DataConfig.AreaLayer
                        & dict(normalize=1,
                               stimulus_type='stimulus.Frame',
                               exclude='images,responses')).fetch("KEY")
        @property
        def content_source(self):
            return (CoreConfig * ReadoutConfig * ShifterConfig * ModulatorConfig * TrainConfig * DataConfig).proj()
        def build_network(self, key, trainsets=None):
            """
            Builds a specified model
            Args:
                key:    key for CNNParameters used to load the parameter of the model. If None, (self & key) must
                        be non-empty so that key can be inferred.
                img_shape: image shape to figure out the size of the readouts
                n_neurons: dictionary with readout sizes (number of neurons)
            Returns:
                an uninitialized MultiCNN
            """
            
            key = dict((self & (key or dict())).fetch1(), **key)
            
            # --- load datasets
            if trainsets is None:
                train_key = TrainConfig().train_key(key)
                trainsets, _ = DataConfig().load_data(key, tier='train', **train_key)
            img_shape = list(trainsets.values())[0].img_shape
            beh_shape = list(trainsets.values())[0].behavior.shape
            in_shape = list(img_shape)
            in_shape[1] += beh_shape[1]
            n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])
            core = CoreConfig().build(in_shape[1], key)
            ro_in_shape = BehaviorNet.get_readout_in_shape(core, in_shape)
            readout = ReadoutConfig().build(ro_in_shape, n_neurons, key)
            shifter = ShifterConfig().build(n_neurons, input_features=2, key=key)
            modulator = ModulatorConfig().build(n_neurons, input_features=3, key=key)
            # --- initialize
            return BehaviorNet(core, readout, nonlinearity=Elu1(), shifter=shifter, modulator=modulator)
        @property
        def source_restriction(self):
            restr = []
            for source in  [CoreConfig(), ReadoutConfig(), ShifterConfig(), ModulatorConfig(), TrainConfig(), DataConfig()]:
                hname = '{}_hash'.format(source._config_type)
                print('---', 'Selecting {} '.format(source.__class__.__name__).ljust(76, '-'))
                hash, hashes = source.select_hashes()
                restr.append('{} in ("{}")'.format(hname, '", "'.join(hashes)))
            return dj.AndList(restr)
        def select_hashes(self):
            hashes = []
            restriction = dict()
            rel = self
            while restriction != '':
                old_restriction = restriction
                print(old_restriction)
                print(rel & old_restriction)
                restriction = input('Please enter a restriction [ENTER for exit]: ')
            hashes.extend((rel & old_restriction).fetch('{}_hash'.format(self._config_type)))
            return 'net_hash', hashes

    class LocomotionNet(dj.Part, TestMixin, TrainMixin):
        definition = """
        -> master
        ---
        -> CoreConfig
        -> ReadoutConfig
        -> ShifterConfig
        -> ModulatorConfig
        -> TrainConfig
        -> DataConfig
        """

        @property
        def content(self):
            yield from (CoreConfig
                        * (ReadoutConfig - ReadoutConfig.SpatialTransformer2d - ReadoutConfig.SpatialXFeatures)
                        * ShifterConfig.StaticAffine2d
                        * ModulatorConfig.MLP
                        * (TrainConfig.Default & dict(batch_size=125))
                        * DataConfig.AreaLayer
                        & dict(normalize=1,
                               stimulus_type='stimulus.Frame',
                               exclude='images,responses')).fetch("KEY")

        @property
        def content_source(self):
            return (CoreConfig * ReadoutConfig * ShifterConfig * ModulatorConfig * TrainConfig * DataConfig).proj()


        def build_network(self, key, trainsets=None):
            """
            Builds a specified model
            Args:
                key:    key for CNNParameters used to load the parameter of the model. If None, (self & key) must
                        be non-empty so that key can be inferred.
                img_shape: image shape to figure out the size of the readouts
                n_neurons: dictionary with readout sizes (number of neurons)
            Returns:
                an uninitialized MultiCNN
            """

            key = dict((self & (key or dict())).fetch1(), **key)

            # --- load datasets
            if trainsets is None:
                train_key = TrainConfig().train_key(key)
                trainsets, _ = DataConfig().load_data(key, tier='train', **train_key)
            img_shape = list(trainsets.values())[0].img_shape
            beh_shape = list(trainsets.values())[0].behavior.shape
            in_shape = list(img_shape)
            in_shape[1] += 1  # add the locomotion dimension
            n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])

            core = CoreConfig().build(in_shape[1], key)
            ro_in_shape = LocomotionNet.get_readout_in_shape(core, in_shape)
            readout = ReadoutConfig().build(ro_in_shape, n_neurons, key)
            shifter = ShifterConfig().build(n_neurons, input_features=2, key=key)
            modulator = ModulatorConfig().build(n_neurons, input_features=3, key=key)

            # --- initialize
            return LocomotionNet(core, readout, nonlinearity=Elu1(), shifter=shifter, modulator=modulator)


        @property
        def source_restriction(self):
            restr = []
            for source in  [CoreConfig(), ReadoutConfig(), ShifterConfig(), ModulatorConfig(), TrainConfig(), DataConfig()]:
                hname = '{}_hash'.format(source._config_type)
                print('---', 'Selecting {} '.format(source.__class__.__name__).ljust(76, '-'))
                hash, hashes = source.select_hashes()
                restr.append('{} in ("{}")'.format(hname, '", "'.join(hashes)))
            return dj.AndList(restr)

        def select_hashes(self):
            hashes = []
            restriction = dict()
            rel = self
            while restriction != '':
                old_restriction = restriction
                print(old_restriction)
                print(rel & old_restriction)
                restriction = input('Please enter a restriction [ENTER for exit]: ')
            hashes.extend((rel & old_restriction).fetch('{}_hash'.format(self._config_type)))
            return 'net_hash', hashes


    def net_key(self, key):
        return dict(key, **self.parameters(key))

    def train(self, key, **kwargs):
        net_key = self.net_key(key)
        Network = getattr(self, net_key.pop('net_type'))
        return Network().train(net_key, **kwargs)

    def build_network(self, key, **kwargs):
        net_key = self.net_key(key)
        Network = getattr(self, net_key.pop('net_type'))
        return Network().build_network(net_key, **kwargs)

    def fill(self):
        type_name = self._config_type + '_type'
        hash_name = self._config_type + '_hash'
        configs = [getattr(self, member) for member in dir(self) if
                   isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]
        print('\n'.join(['({}) {}'.format(i, rel.__name__) for i, rel in enumerate(configs)]))
        choice = None
        choices = list(map(str, range(len(configs))))
        while choice not in choices:
            choice = input('Please select configuration: ')
        network_config = configs[int(choice)]()


        for key in network_config.content_source.proj() & network_config.source_restriction:
            key[type_name] = network_config.__class__.__name__
            key[hash_name] = key_hash(key)

            if not key in (network_config).proj():
                self.insert1(key, ignore_extra_fields=True)
                log.info('Inserting ' + repr(key))
                network_config.insert1(key, ignore_extra_fields=True)
            else:
                log.info('{} already defined'.format(key['net_hash']))

@schema
class TrainConfig(Config, dj.Lookup):
    _config_type = 'train'

    class Default(dj.Part):
        definition = """
        -> master
        ---
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        loss_type              : varchar(16) # name of loss function
        avg_loss               : bool     # whether to average or sum the loss across neurons, set True for average and False for sum
        """

        @property
        def content(self):
            yield dict(batch_size=250, schedule=np.array([0.005, 0.001]), max_epoch=500, loss_type='PoissonLoss', avg_loss=True)
            yield dict(batch_size=125, schedule=np.array([0.005, 0.001]), max_epoch=500, loss_type='PoissonLoss', avg_loss=True)
            yield dict(batch_size=60, schedule=np.array([0.009, 0.009*0.3, 0.009*0.3*0.3, 0.009*0.3*0.3*0.3]), max_epoch=500, loss_type='PoissonLoss', avg_loss=False)
            yield dict(batch_size=128, schedule=np.array([0.009, 0.009*0.3, 0.009*0.3*0.3, 0.009*0.3*0.3*0.3]), max_epoch=500, loss_type='PoissonLoss', avg_loss=False)

        def train(self, key, model, trainloaders, valloaders):
            log.info('Training'.ljust(40, '-'))
            # hack some parameters
            key['finetune'] = key['max_epoch']
            key['max_epoch'] = 500
            key['transfer_group_id'] = 272
            key['transfer_net_hash'] = "8b6fe18fa651ebf452db0fbd77d05a01"
            key['transfer_seed'] = 1009

            # set some parameters
            criterion = PoissonLoss(avg=key['avg_loss'])
            # from neuralpredictors.measures.modules import PoissonLoss
            # criterion = PoissonLoss(avg=False)
            def objective(model, readout_key, data):
                if model.shift and model.modulate:
                    outputs = model(data.images, readout_key, eye_pos=data.pupil_center, behavior=data.behavior)
                elif model.shift:
                    outputs = model(data.images, readout_key, eye_pos=data.pupil_center)
                elif model.modulate:
                    outputs = model(data.images, readout_key, behavior=data.behavior)
                else:
                    outputs = model(data.images, readout_key)
                
                # loss_scale = np.sqrt(len(trainloaders[readout_key].dataset) / data.images.shape[0])
                # print('loss_scale = ', loss_scale)
                return criterion(outputs, data.responses) \
                       + (key['finetune'] * model.core.regularizer()) \
                       + model.readout.regularizer(readout_key) \
                       + (model.shifter.regularizer(readout_key) if model.shift else 0) \
                       + (model.modulator.regularizer(readout_key) if model.modulate else 0)

            def run(model, objective, optimizer, stop_closure, trainloaders, epoch=0,
                    interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
                    restore_best=True, accumulate_gradient=1
                    ):
                log.info('Training models with {} and state {}'.format(optimizer.__class__.__name__,
                                                                       repr(model.state)))
                optimizer.zero_grad()
                iteration = 0

                for epoch, val_obj in early_stopping(model, stop_closure,
                                                     interval=interval, patience=patience,
                                                     start=epoch, max_iter=max_iter, maximize=maximize,
                                                     tolerance=tolerance, restore_best=restore_best):
                    for batch_no, (readout_key, data) in tqdm(enumerate(cycle_datasets(trainloaders)),
                                                              desc=self.__class__.__name__.ljust(
                                                                  25) + '  | Epoch {}'.format(epoch)):
                        obj = objective(model, readout_key, data)
                        obj.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        iteration += 1
                return model, epoch

            # --- initialize
            mu_dict = {k: dl.dataset.transformed_mean().responses for k, dl in trainloaders.items()}
            model.readout.initialize(mu_dict)
            model.core.initialize()
            if model.shifter is not None:
                biases = {k: -dl.dataset.transformed_mean().pupil_center for k, dl in trainloaders.items()}
                model.shifter.initialize(bias=biases)
            if model.modulator is not None:
                model.modulator.initialize()

            # load pre-trained transfer core state_dict
            init_model_dict = model.state_dict()
            state_dict = (models.Model & dict(group_id=key['transfer_group_id'], net_hash=key['transfer_net_hash'], seed=key['transfer_seed'])).fetch1('model')
            try:
                state_dict = {k: torch.as_tensor(state_dict[k][0].copy()) for k in state_dict.dtype.names if 'core' in k}
            except AttributeError:
                state_dict = {k: torch.as_tensor(state_dict[k].copy()) for k in state_dict.keys() if 'core' in k}
            init_model_dict.update(state_dict)
            model.load_state_dict(init_model_dict)

            if not key['finetune']: # freeze core parameters
                print('Freezing pre-trained core')
                for param in model.core.parameters():
                    param.requires_grad = False


            # --- train
            log.info('Shipping model to GPU')
            model = model.cuda()
            model.train(True)
            print(model)
            epoch = 0

            schedule = key['schedule']
            model.shift = True
            for opt, lr in zip(repeat(torch.optim.Adam), schedule):
                log.info('Training with learning rate {}'.format(lr))

                optimizer = opt(model.parameters(), lr=lr)

                model, epoch = run(model, objective, optimizer,
                                   partial(correlation_closure, loaders=valloaders), trainloaders,
                                   epoch=epoch, max_iter=key['max_epoch'], patience=10)
            model.eval()
            return model
        
    class Transfer(dj.Part):
        definition = """
        -> master
        ---
        -> models.Model.proj(transfer_group_id='group_id', transfer_net_hash='net_hash', transfer_seed='seed')
        finetune               : bool     # whether to finetune the transfer core, if False, detach the transfer core from gradient backprop      
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        loss_type              : varchar(16) # name of loss function
        avg_loss               : bool     # whether to average or sum the loss across neurons, set True for average and False for sum
        """

        @property
        def content(self):
            yield dict(transfer_group_id=272, transfer_net_hash="8b6fe18fa651ebf452db0fbd77d05a01", transfer_seed=1009, finetune=True, batch_size=60, schedule=np.array([0.005, 0.001]), max_epoch=500, loss_type='PoissonLoss', avg_loss=True)
            yield dict(transfer_group_id=272, transfer_net_hash="8b6fe18fa651ebf452db0fbd77d05a01", transfer_seed=1009, finetune=False, batch_size=60, schedule=np.array([0.005, 0.001]), max_epoch=500, loss_type='PoissonLoss', avg_loss=True)

        def train(self, key, model, trainloaders, valloaders):
            log.info('Training'.ljust(40, '-'))
            # hack some parameters
            # key['finetune'] = key['max_epoch']
            # key['max_epoch'] = 500
            # key['transfer_group_id'] = 272
            # key['transfer_net_hash'] = "8b6fe18fa651ebf452db0fbd77d05a01"
            # key['transfer_seed'] = 1009
            
            # set some parameters
            criterion = PoissonLoss(avg=key['avg_loss'])

            def objective(model, readout_key, data):
                if model.shift and model.modulate:
                    outputs = model(data.images, readout_key, eye_pos=data.pupil_center, behavior=data.behavior)
                elif model.shift:
                    outputs = model(data.images, readout_key, eye_pos=data.pupil_center)
                elif model.modulate:
                    outputs = model(data.images, readout_key, behavior=data.behavior)
                else:
                    outputs = model(data.images, readout_key)
                
                return criterion(outputs, data.responses) \
                       + (key['finetune'] * model.core.regularizer()) \
                       + model.readout.regularizer(readout_key) \
                       + (model.shifter.regularizer(readout_key) if model.shift else 0) \
                       + (model.modulator.regularizer(readout_key) if model.modulate else 0)

            def run(model, objective, optimizer, stop_closure, trainloaders, epoch=0,
                    interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
                    restore_best=True, accumulate_gradient=1
                    ):
                log.info('Training models with {} and state {}'.format(optimizer.__class__.__name__,
                                                                       repr(model.state)))
                optimizer.zero_grad()
                iteration = 0

                for epoch, val_obj in early_stopping(model, stop_closure,
                                                     interval=interval, patience=patience,
                                                     start=epoch, max_iter=max_iter, maximize=maximize,
                                                     tolerance=tolerance, restore_best=restore_best):
                    for batch_no, (readout_key, data) in tqdm(enumerate(cycle_datasets(trainloaders)),
                                                              desc=self.__class__.__name__.ljust(
                                                                  25) + '  | Epoch {}'.format(epoch)):
                        obj = objective(model, readout_key, data)
                        obj.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        iteration += 1
                return model, epoch

            # --- initialize
            mu_dict = {k: dl.dataset.transformed_mean().responses for k, dl in trainloaders.items()}
            model.readout.initialize(mu_dict)
            model.core.initialize()
            if model.shifter is not None:
                biases = {k: -dl.dataset.transformed_mean().pupil_center for k, dl in trainloaders.items()}
                model.shifter.initialize(bias=biases)
            if model.modulator is not None:
                model.modulator.initialize()
                
            # load pre-trained transfer core state_dict
            init_model_dict = model.state_dict()
            state_dict = (models.Model & dict(group_id=key['transfer_group_id'], net_hash=key['transfer_net_hash'], seed=key['transfer_seed'])).fetch1('model')
            try:
                state_dict = {k: torch.as_tensor(state_dict[k][0].copy()) for k in state_dict.dtype.names if 'core' in k}
            except AttributeError:
                state_dict = {k: torch.as_tensor(state_dict[k].copy()) for k in state_dict.keys() if 'core' in k}
            init_model_dict.update(state_dict)
            model.load_state_dict(init_model_dict)

            if not key['finetune']: # freeze core parameters
                print('Freezing pre-trained core')
                for param in model.core.parameters():
                    param.requires_grad = False

            # --- train
            log.info('Shipping model to GPU')
            model = model.cuda()
            model.train(True)
            print(model)
            epoch = 0

            schedule = key['schedule']
            model.shift = True
            for opt, lr in zip(repeat(torch.optim.Adam), schedule):
                log.info('Training with learning rate {}'.format(lr))

                optimizer = opt(model.parameters(), lr=lr)

                model, epoch = run(model, objective, optimizer,
                                   partial(correlation_closure, loaders=valloaders), trainloaders,
                                   epoch=epoch, max_iter=key['max_epoch'], patience=10)
            model.eval()
            return model
        
    class MSEDefault(dj.Part):
        definition = """
        -> master
        ---
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        loss_type              : varchar(16) # name of loss function
        """

        @property
        def content(self):
            yield dict(batch_size=250, schedule=np.array([0.005, 0.001]), max_epoch=500)
            yield dict(batch_size=125, schedule=np.array([0.005, 0.001]), max_epoch=500)
            yield dict(batch_size=60, schedule=np.array([0.005, 0.001]), max_epoch=500)
            yield dict(batch_size=30, schedule=np.array([0.005, 0.001]), max_epoch=500)

        def train(self, key, model, trainloaders, valloaders):
            log.info('Training'.ljust(40, '-'))
            # set some parameters
            criterion = MSE()

            def objective(model, readout_key, inputs, beh, eye_pos, targets):
                outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh)
                return criterion(outputs, targets) \
                       + model.core.regularizer() \
                       + model.readout.regularizer(readout_key) \
                       + (model.shifter.regularizer(readout_key) if model.shift else 0) \
                       + (model.modulator.regularizer(readout_key) if model.modulate else 0)

            def run(model, objective, optimizer, stop_closure, trainloaders, epoch=0,
                    interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
                    restore_best=True, accumulate_gradient=1
                    ):
                log.info('Training models with {} and state {}'.format(optimizer.__class__.__name__,
                                                                       repr(model.state)))
                optimizer.zero_grad()
                iteration = 0

                for epoch, val_obj in early_stopping(model, stop_closure,
                                                     interval=interval, patience=patience,
                                                     start=epoch, max_iter=max_iter, maximize=maximize,
                                                     tolerance=tolerance, restore_best=restore_best):
                    for batch_no, (readout_key, data) in tqdm(enumerate(cycle_datasets(trainloaders)),
                                                              desc=self.__class__.__name__.ljust(
                                                                  25) + '  | Epoch {}'.format(epoch)):
                        if len(data) == 4: # data includes behavior and pupil_center
                            obj = objective(model, readout_key, *data)
                        elif len(data) == 2: # data only includes images and responses
                            obj = objective(model, readout_key, data.images, None, None, data.responses)
                        obj.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        iteration += 1
                return model, epoch

            # --- initialize
            mu_dict = {k: dl.dataset.transformed_mean().responses for k, dl in trainloaders.items()}
            model.readout.initialize(mu_dict)
            model.core.initialize()
            if model.shifter is not None:
                biases = {k: -dl.dataset.transformed_mean().pupil_center for k, dl in trainloaders.items()}
                model.shifter.initialize(bias=biases)
            if model.modulator is not None:
                model.modulator.initialize()
                
            # --- train
            log.info('Shipping model to GPU')
            model = model.cuda()
            model.train(True)
            print(model)
            epoch = 0

            schedule = key['schedule']
            model.shift = True
            for opt, lr in zip(repeat(torch.optim.Adam), schedule):
                log.info('Training with learning rate {}'.format(lr))

                optimizer = opt(model.parameters(), lr=lr)

                model, epoch = run(model, objective, optimizer,
                                   partial(correlation_closure, loaders=valloaders), trainloaders,
                                   epoch=epoch, max_iter=key['max_epoch'], patience=10)
            model.eval()
            return model

#         def train(self, key, model, trainloaders, valloaders):
#             log.info('Training'.ljust(40, '-'))
#             # set some parameters
#             criterion = PoissonLoss()

#             def objective(model, readout_key, inputs, beh, eye_pos, targets):
#                 outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh)
#                 return criterion(outputs, targets) \
#                        + model.core.regularizer() \
#                        + model.readout.regularizer(readout_key) \
#                        + (model.shifter.regularizer(readout_key) if model.shift else 0) \
#                        + (model.modulator.regularizer(readout_key) if model.modulate else 0)

#             def run(model, objective, optimizer, stop_closure, trainloaders, valloaders, running_train_loss, running_train_eval_loss, train_count,
#                     epoch=0, interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
#                     restore_best=True, accumulate_gradient=1
#                     ):
#                 log.info('Training models with {} and state {}'.format(optimizer.__class__.__name__,
#                                                                        repr(model.state)))
#                 optimizer.zero_grad()
#                 iteration = 0
#                 train_loss, val_loss, train_eval_loss, val_eval_loss = [], [], [], []
#                 train_corr, val_corr, train_eval_corr, val_eval_corr = [], [], [], []
                
#                 for epoch, val_obj in early_stopping(model, stop_closure,
#                                                      interval=interval, patience=patience,
#                                                      start=epoch, max_iter=max_iter, maximize=maximize,
#                                                      tolerance=tolerance, restore_best=restore_best):
#                     for batch_no, (readout_key, data) in tqdm(enumerate(cycle_datasets(trainloaders)),
#                                                               desc=self.__class__.__name__.ljust(
#                                                                   25) + '  | Epoch {}'.format(epoch)):
#                         if len(data) == 4: # data includes behavior and pupil_center
#                             obj = objective(model, readout_key, *data)
#                         elif len(data) == 2: # data only includes images and responses
#                             obj = objective(model, readout_key, data.images, None, None, data.responses)
                        
#                         # store train and validation loss over iterations 
#                         # training mode
#                         batchsize = data.images.shape[0]
#                         train_count += batchsize
#                         running_train_loss += obj.item() * batchsize
#                         train_loss.append(running_train_loss / train_count)
                        
#                         running_val_loss = 0
#                         count = 0
#                         for i, valdata in enumerate(valloaders[readout_key]):
#                             running_val_loss += objective(model, readout_key, *valdata).item() * valdata.images.shape[0]
#                             count += valdata.images.shape[0]
#                         val_loss.append(running_val_loss / count)
                        
#                         if batch_no == 0:
#                             train_corr.append(partial(correlation, loaders=trainloaders)(model))
#                             val_corr.append(partial(correlation, loaders=valloaders)(model))
                            
#                         # evaluate mode
#                         model.eval()
# #                         running_train_eval_loss += objective(model, readout_key, *data) * batchsize
# #                         train_eval_loss.append(running_train_eval_loss / train_count)
                        
# #                         running_val_loss = 0
# #                         count = 0
# #                         for i, valdata in enumerate(valloaders[readout_key]):
# #                             running_val_loss += objective(model, readout_key, *valdata).item() * valdata.images.shape[0]
# #                             count += valdata.images.shape[0]
# #                         val_eval_loss.append(running_val_loss / count)
                        
#                         # compute correlation of real and predicted response for training and validation sets
#                         if batch_no == 0:
#                             train_eval_corr.append(partial(correlation, loaders=trainloaders)(model))
#                             val_eval_corr.append(partial(correlation, loaders=valloaders)(model))
                        
#                         model.train(True)
                    
#                         obj.backward()
#                         optimizer.step()
#                         optimizer.zero_grad()
#                         iteration += 1
                        
#                 return model, epoch, train_count, running_train_loss, running_train_eval_loss, train_loss, val_loss, train_eval_loss, val_eval_loss, train_corr, val_corr, train_eval_corr, val_eval_corr

#             # --- initialize
#             mu_dict = {k: dl.dataset.transformed_mean().responses for k, dl in trainloaders.items()}
#             model.readout.initialize(mu_dict)
#             model.core.initialize()
#             if model.shifter is not None:
#                 biases = {k: -dl.dataset.transformed_mean().pupil_center for k, dl in trainloaders.items()}
#                 model.shifter.initialize(bias=biases)
#             if model.modulator is not None:
#                 model.modulator.initialize()
                
#             # --- train
#             log.info('Shipping model to GPU')
#             model = model.cuda()
#             model.train(True)
#             print(model)
#             epoch = 0

#             schedule = key['schedule']
#             model.shift = True
            
#             # number of training samples that have been used
#             running_train_loss = running_train_eval_loss = 0
#             train_count = 0
#             all_train_loss, all_val_loss, all_train_eval_loss, all_val_eval_loss, all_train_corr, all_val_corr, all_train_eval_corr, all_val_eval_corr = [], [], [], [], [], [], [], []

#             for opt, lr in zip(repeat(torch.optim.Adam), schedule):
#                 log.info('Training with learning rate {}'.format(lr))

#                 optimizer = opt(model.parameters(), lr=lr)

#                 model, epoch, train_count, running_train_loss, running_train_eval_loss, train_loss, val_loss, train_eval_loss, val_eval_loss, train_corr, val_corr, train_eval_corr, val_eval_corr = run(model, objective, optimizer, partial(correlation_closure, loaders=valloaders), trainloaders, valloaders, running_train_loss, running_train_eval_loss, train_count, epoch=epoch, max_iter=key['max_epoch'], patience=10)
                
#                 all_train_loss.append(train_loss)
#                 all_val_loss.append(val_loss)
#                 all_train_eval_loss.append(train_eval_loss)
#                 all_val_eval_loss.append(val_eval_loss)
#                 all_train_corr.append(train_corr)
#                 all_val_corr.append(val_corr)
#                 all_train_eval_corr.append(train_eval_corr)
#                 all_val_eval_corr.append(val_eval_corr)
                
#             model.eval()
#             return model, epoch, all_train_loss, all_val_loss, all_train_eval_loss, all_val_eval_loss, all_train_corr, all_val_corr, all_train_eval_corr, all_val_eval_corr

    class ExpPoisson(dj.Part):
        definition = """
        -> master
        ---
        loss_type              : varchar(16) # name of loss class
        lambda                  : float    # exponential scale
        stop_closure           : varchar(45) # stop_closure type for validation set evaluation
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            for p in product(['ExpPoisson'], [0.1, 0.05], ['correlation_closure', 'loss_closure'], [30, 60, 125, 250], [np.array([0.05, 0.01, 0.001]), np.array([0.005, 0.001])], [500]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
            
    class ExpMSE(dj.Part):
        definition = """
        -> master
        ---
        loss_type              : varchar(16) # name of loss class
        lambda                  : float    # exponential scale
        stop_closure           : varchar(45) # stop_closure type for validation set evaluation
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            for p in product(['ExpMSE'], [0.002, 0.001], ['correlation_closure', 'loss_closure'], [30, 60, 125, 250], [np.array([0.05, 0.01, 0.001]), np.array([0.005, 0.001])], [500]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
                
                
    class ExponentialPoisson(dj.Part):
        definition = """
        -> master
        ---
        loss_type              : varchar(32) # name of loss class
        lambda                  : float    # exponential scale
        stop_closure           : varchar(45) # stop_closure type for validation set evaluation
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            for p in product(['ExponentialPoisson'], [10, 5, 1, 0.5, 0.2, 0.1, 0.05, 0.025], ['correlation_closure', 'loss_closure'], [30, 60, 125, 250], [np.array([0.05, 0.01, 0.001]), np.array([0.005, 0.001])], [500]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
            
    class ExponentialMSE(dj.Part):
        definition = """
        -> master
        ---
        loss_type              : varchar(32) # name of loss class
        lambda                  : float    # exponential scale
        stop_closure           : varchar(45) # stop_closure type for validation set evaluation
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            for p in product(['ExponentialMSE'], [10, 5, 1, 0.5, 0.2, 0.1, 0.05, 0.025], ['correlation_closure', 'loss_closure'], [30, 60, 125, 250], [np.array([0.05, 0.01, 0.001]), np.array([0.005, 0.001])], [500]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
                
                
    class PoissonNL(dj.Part):
        definition = """
        -> master
        ---
        loss_type              : varchar(16) # name of loss class
        stop_closure           : varchar(45) # stop_closure type for validation set evaluation
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            for p in product(['PoissonNL'],  ['correlation_closure', 'loss_closure'], [30, 60, 125, 250], [np.array([0.05, 0.01, 0.001]), np.array([0.005, 0.001])], [500]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
                
    class LogCosh(dj.Part):
        definition = """
        -> master
        ---
        loss_type              : varchar(16) # name of loss class
        stop_closure           : varchar(45) # stop_closure type for validation set evaluation
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            for p in product(['LogCosh'],  ['correlation_closure', 'loss_closure'], [30, 60, 125, 250], [np.array([0.05, 0.01, 0.001]), np.array([0.005, 0.001])], [500]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
                
    class XSigmoid(dj.Part):
        definition = """
        -> master
        ---
        loss_type              : varchar(16) # name of loss class
        stop_closure           : varchar(45) # stop_closure type for validation set evaluation
        batch_size             : smallint # 
        schedule               : longblob # learning rate schedule
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            for p in product(['XSigmoid'],  ['correlation_closure', 'loss_closure'], [30, 60, 125, 250], [np.array([0.05, 0.01, 0.001]), np.array([0.005, 0.001])], [500]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
    
    def train_key(self, key):
        return dict(key, **self.parameters(key))

    def train(self, key, **kwargs):
        train_key = self.train_key(key)
        Trainer = getattr(self, train_key.pop('train_type'))
        try:
            return Trainer().train(train_key, **kwargs)
        except AttributeError: # use default_train if no specific train method is defined for a trainconfig part table
            if 'lambda' not in train_key.keys():
                loss = getattr(losses, train_key.pop('loss_type'))() 
            else: 
                loss = getattr(losses, train_key.pop('loss_type'))(lam=train_key['lambda']) 
            return self.default_train(train_key, criterion=loss, **kwargs)
    
    def default_train(self, key, model, trainloaders, valloaders, criterion):
        log.info('Training'.ljust(40, '-'))

        def objective(model, readout_key, inputs, beh, eye_pos, targets):
            outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh)
            return criterion(outputs, targets) \
                    + model.core.regularizer() \
                    + model.readout.regularizer(readout_key) \
                    + (model.shifter.regularizer(readout_key) if model.shift else 0) \
                    + (model.modulator.regularizer(readout_key) if model.modulate else 0)

        def run(model, objective, optimizer, stop_closure, trainloaders, epoch=0,
                interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
                restore_best=True, accumulate_gradient=1
                ):
            log.info('Training models with {} and state {}'.format(optimizer.__class__.__name__,
                                                                    repr(model.state)))
            optimizer.zero_grad()
            iteration = 0

            for epoch, val_obj in early_stopping(model, stop_closure,
                                                    interval=interval, patience=patience,
                                                    start=epoch, max_iter=max_iter, maximize=maximize,
                                                    tolerance=tolerance, restore_best=restore_best):
                for batch_no, (readout_key, data) in tqdm(enumerate(cycle_datasets(trainloaders)),
                                                            desc=self.__class__.__name__.ljust(
                                                                25) + '  | Epoch {}'.format(epoch)):
                    if len(data) == 4: # data includes behavior and pupil_center
                        obj = objective(model, readout_key, *data)
                    elif len(data) == 2: # data only includes images and responses
                        obj = objective(model, readout_key, data.images, None, None, data.responses)
                    obj.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    iteration += 1
            return model, epoch

        # --- initialize
        mu_dict = {k: dl.dataset.transformed_mean().responses for k, dl in trainloaders.items()}
        model.readout.initialize(mu_dict)
        model.core.initialize()
        if model.shifter is not None:
            biases = {k: -dl.dataset.transformed_mean().pupil_center for k, dl in trainloaders.items()}
            model.shifter.initialize(bias=biases)
        if model.modulator is not None:
            model.modulator.initialize()
            
        # --- train
        log.info('Shipping model to GPU')
        model = model.cuda()
        model.train(True)
        print(model)
        epoch = 0

        schedule = key['schedule']
        model.shift = True
        for opt, lr in zip(repeat(torch.optim.Adam), schedule):
            log.info('Training with learning rate {}'.format(lr))

            optimizer = opt(model.parameters(), lr=lr)

            model, epoch = run(model, objective, optimizer,
                                partial(correlation_closure, loaders=valloaders), trainloaders,
                                epoch=epoch, max_iter=key['max_epoch'], patience=10)
        model.eval()
        return model
