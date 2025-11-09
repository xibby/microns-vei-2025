import datajoint as dj
import numpy as np
from itertools import product
from scipy import ndimage
from functools import partial
import math
import torch.optim as optim
from skimage import morphology
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import os
import pickle
import random
import warnings

from staticnet import logger
from staticnet_analyses.base import MEIParameters, MaskParameters, MEI, MEIMask, Dataset, EnsembleEval, UnitRanking, NeuronSet, NewEnsembleEval, SeedSet, TightMEIMask
from staticnet_experiments import models as static_models, utils as static_utils
from staticnet_invariance.diverse_meis import FeatureSpace, CombinationOperation, SimilarityMetric
from staticnet_invariance import deis as old_deis_schema
from staticnet_vae import vae
from staticnet_analyses import closed_loop

import featurevis
from featurevis import models, ops, utils
from featurevis.ops import create_whole_mei, get_batch
from featurevis.utils import varargin
from utils.datajoint.datajoint_utils import files
from neuro_data.static_images.data_schemas import process_frame, Preprocessing

imagenet = dj.create_virtual_module('pipeline_imagenet', 'pipeline_imagenet')
stimulus = dj.create_virtual_module('pipeline_stimulus', 'pipeline_stimulus')

schema = dj.schema('neurostatic_deis_reconfigured')
dj.config.setdefault('stores', dict())
dj.config['stores'].update({
    'static': dict(
        protocol='file', 
        location='/dj-stor01/neuro-static')
})

VAE_PATH_3 = '/external/zhiwei/MEI_VAE_centered_mei_params_8.pt'

class HingeLoss():
    def __init__(self, ref_level, amax):
        super(HingeLoss, self).__init__()
        self.ref_level = ref_level
        self.amax = amax
    @varargin
    def __call__(self, x):
        diff = self.ref_level - x / self.amax
        return torch.mean(F.relu(diff))

def prepare_params(key, mei_params, mask_stats_params=None, diverse_params=None, toy=False, toy_model=None, mask_table=MEIMask, mei_table=MEI, device='cuda'):
    if not toy:
        # Get models
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        if mei_params['use_avg_model']:
            if key['seed'] == 1009:
                ssid = 1
            elif key['seed'] == 101:
                ssid = 4
            seeds = (SeedSet & {'ssid': ssid}).fetch1('seeds')
            seed_rest = [{'seed': s} for s in seeds]
        else:
            seed_rest = {'seed': key['seed']}
        all_keys = (static_models.Model & model_key & seed_rest).fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

        # Get some train stats
        mean_eyepos = ([0, 0] if (Dataset.TrainStats & key).fetch1('norm_eyepos')
                        else (Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.as_tensor(mean_eyepos, dtype=torch.float32,
                                        device=device).unsqueeze(0)
        
    # Get MEI mask and set up shared operations for MaskFixedMEI and DEI generation
    mask, mask_x, mask_y = (mask_table & key).fetch1('mask', 'mask_x', 'mask_y')
    mask = torch.as_tensor(mask, dtype=torch.float32, device=device).contiguous()
    if mask_stats_params is None or not (mask_stats_params['fixed_mask_std'] or mask_stats_params['fixed_mask_mean']):
        postup_op = ops.ChangeStats(float(mei_params['contrast']), float(mei_params['mean']))
    else:
        postup_op = ops.ChangeMaskStats(mask_stats_params['fixed_mask_std'], mask_stats_params['fixed_mask_mean'], mask)

    # set initial batch seed
    torch.manual_seed(mei_params['mei_seed'])
    if diverse_params is None:  # for MEI generation
        if toy:
            base_model = toy_model
        else:
            base_model = models.Ensemble(all_models, key['readout_key'], neuron_idx=key['neuron_id'], eye_pos=mean_eyepos, device=device, average_batch=True)
        # shared operations for toy and neural models
        if mei_params['blur_sigma']:
            gradient_f = ops.GaussianBlur(float(mei_params['blur_sigma']))
        else:
            gradient_f = None
        image_shape = (mei_params['num_initializations'], 1, mei_params['height'], mei_params['width'])
        initial_image = torch.randn(image_shape, device=device)
        initial_image = postup_op(initial_image)

        return base_model, gradient_f, postup_op, initial_image
        
    else: # for DEI generation
        if toy and diverse_params['features'] != 'pixels':
            raise NotImplementedError('Diversity in feature spaces other than pixel space is not implemented for toy neuron models!')
            
        # Set up embedding, mask, and optimization step size for each space
        if toy:
            base_model = toy_model
        else:
            base_model = models.Ensemble(all_models, key['readout_key'], neuron_idx=key['neuron_id'], eye_pos=mean_eyepos, device=device, average_batch=False)
        
        if diverse_params['features'] == 'pixels':
            embedding = ops.Identity()  # operation that returns x as is
            similarity_mask = mask if diverse_params['diversity_mask'] == 'mei_mask' else None
        elif diverse_params['features'] == 'vae_latent':
            get_latent = vae.Latent_Embedding(mask_x=mask_x, mask_y=mask_y, vae_path=VAE_PATH_3) 
            embedding = get_latent
            similarity_mask = None
        elif diverse_params['features'] == 'feature_vectors':
            # return a feature map matrix in the shape of batch_size x (num_models x feature_vec_length)
            embedding = ops.Feature_Vector_Ensemble(all_models, key['readout_key'], neuron_idx=key['neuron_id'], eye_pos=mean_eyepos, average_batch=False, device=device)
            similarity_mask = None
        elif diverse_params['features'] == 'single_grid_population_resps':
            embedding = ops.SingleGridResps(all_models, key['readout_key'], eye_pos=mean_eyepos, neuron_idx=key['neuron_id'], all_neurons=True, average_batch=False, device=device)
            similarity_mask = None   
        else:
            raise NotImplementedError('{} feature embedding not implemented'.format(diverse_params['features']))
        
        if mask_stats_params['fixed_mask_std'] or mask_stats_params['fixed_mask_mean']:
            mei, mei_activation = (MaskFixedMEI & key).fetch1('mei', 'activation')
        else:
            mei, mei_activation = (mei_table & key).fetch1('mei', 'activation')
        mei = torch.as_tensor(mei[None, None], dtype=torch.float32, device=device).contiguous()

        # Set up gradient blurring
        if mei_params['blur_sigma']:
            gradient_f = utils.Compose([ops.GaussianBlur(float(mei_params['blur_sigma'])), ops.MultiplyBy(diverse_params['decay_constant'], diverse_params['decay_factor'], diverse_params['decay_iters'])])
        else:
            gradient_f = ops.MultiplyBy(diverse_params['decay_constant'], diverse_params['decay_factor'])

        # Set up how pairwise similarities will be combined for optimization
        if diverse_params['combine_op'] == 'maximum':
            combine_op = torch.max
        elif diverse_params['combine_op'] == 'average':
            combine_op = torch.mean
        elif diverse_params['combine_op'] == 'average_maximum':
            combine_op = None
        else:
            raise NotImplementedError('{} combination operation not implemented'.format(diverse_params['combine_op']))

        # Prepare initial batch of images
        image_shape = (diverse_params['num_deis'], 1, mei_params['height'],
                        mei_params['width'])
        if diverse_params['initial_type'] == 'MEI':
            initial_batch = torch.randn(image_shape, device=device) * diverse_params['init_noise_scale'] + mei.repeat(diverse_params['num_deis'], 1, 1, 1)
        else:
            initial_batch = torch.randn(image_shape, device=device) * diverse_params['init_noise_scale']
        initial_batch = postup_op(initial_batch)
        
        return base_model, embedding, similarity_mask, mei, mei_activation, mask, mask_x, mask_y, gradient_f, combine_op, postup_op, initial_batch
        

@schema
class MaskStatsParameters(dj.Lookup):
    definition = """
    mask_stats_params: int
    ---
    fixed_mask_mean:   float
    fixed_mask_std:    float
    """
    contents = [[1, 0.0, 0.8],
                [2, 0.0, 0.0]
                ]

@schema
class MaskFixedMEI(dj.Computed):
    definition = """
    -> static_models.Model
    -> Dataset.Unit
    -> MEIParameters
    -> MEIMask
    -> MaskStatsParameters
    ---
    mei:           blob@static 
    activation:    float           # activation for this MEI
    fevals:        blob@static # list of fevals over iterations during optimization
    """

    @property
    def key_source(self):
        return MEIMask * MaskStatsParameters & [{'seed': 1009}, {'seed': 101}]

    def make(self, key):
        # Get params
        mei_params = (MEIParameters & key).fetch1()
        mask_stats_params = (MaskStatsParameters & key).fetch1()
        if mask_stats_params is None or not int(mask_stats_params['fixed_mask_std']) & int(mask_stats_params['fixed_mask_mean']):
            mei, activation = (MEI & key).fetch1('mei', 'activation')
            # Dummy insert: MEI with FF stats constraint
            self.insert1({**key, 'mei': mei, 'activation': activation, 'fevals': np.array([])})
        else:
            model, gradient_f, postup_op, initial_image = prepare_params(key, mei_params, mask_stats_params)
            mei, fevals, _ = featurevis.gradient_ascent(model, initial_image,
                                                        post_update=postup_op,
                                                        gradient_f=gradient_f,
                                                        step_size=mei_params['step_size'],
                                                        num_iterations=mei_params['num_iterations'])
            mei = mei.mean(0).squeeze().cpu().numpy()
            activation = fevals[-1]
            # Insert
            self.insert1({**key, 'mei': mei, 'activation': activation, 'fevals': np.array(fevals)})

@schema
class DEIParameters(dj.Lookup):
    definition = """ # parameters to generate diverse MEIs using the new loss function with std constraint
    diverse_params: int
    ---
    -> FeatureSpace                 # type of embedding used to compute distances
    -> SimilarityMetric             # similarity metric to use between images
    -> CombinationOperation         # how to compute overall similarity from all pairwise similarities
    num_deis:           int         # number of images to generate
    step_size:          float       # step size for optimizing DEIs
    num_iterations:     int         # number of iteration for optimizing DEIs
    loss_type:          varchar(64) # loss function used for optimization
    decay_constant:     float       # initial scaling of optimization step size
    decay_factor:       float       # step size decay over iterations
    decay_iters:        int         # Number of iterations to wait until next decay_factor is applied
    initial_type:       varchar(16) # image type for initial batch
    diversity_mask:     varchar(45) # mask for measuring diversity 
    init_noise_scale:   float       # scaling factor of white noise on the initial images 
    deterministic:      bool        # whether to use deterministic cudnn algorothims for the sake of reproducibility
    """

    class RefLevel(dj.Part):
        definition = """
        -> master
        ref_id: int
        ---
        ref_level: float   # reference activation level relative to the MEI activation that we want DEIs to achieve 
        """

    class Weight(dj.Part):
        definition = """
        -> master
        weight_id: int     
        ---
        weight: float      # weight of diversity regularization term during optimization
        """

    def fill(self, key):
        self.insert1(key, ignore_extra_fields=True)
        for ref_id, ref_level in enumerate(key['ref_levels']):
            self.RefLevel.insert1({'diverse_params': key['diverse_params'], 'ref_id': ref_id, 'ref_level': ref_level})
        for weight_id, weight in enumerate(key['weights']):
            self.Weight.insert1({'diverse_params': key['diverse_params'], 'weight_id': weight_id, 'weight': weight})

@schema
class DEI(dj.Computed):
    definition = """ # create DEIs
    -> static_models.Model
    -> Dataset.Unit
    -> MEIParameters
    -> MEIMask
    -> MaskStatsParameters
    -> DEIParameters.RefLevel
    -> DEIParameters.Weight
    ---
    deis:                  blob@static         # a set of DEI images
    activations:           longblob                # activations for the DEI set
    fevals:                blob@static
    regs:                  blob@static   
    avg_activation_ratio:  float                   # average activation / original mei_activation across all DEIs
    min_activation_ratio:  float                   # mininum activation / original mei_activation across all DEIs
    std_activation:        float                   # (raw std activation / raw average activation) across all DEIs
    avg_sim:               float                   # average pair-wise similarity
    max_sim:               float                   # maximum pair-wise similarity
    min_sim:               float                   # minimum pair-wise similarity
    sims_to_mei:           longblob                # similarity to MEI
    """

    @property
    def key_source(self):
        all_keys = MEIMask * MaskStatsParameters * DEIParameters.RefLevel * DEIParameters.Weight
        return all_keys #& [{'seed': 1009}, {'seed': 101}] 

    @staticmethod
    @varargin
    def div_regularization(similarity_mask, similarity_metric, div_weight, combine_op, embedding, mei, x, iteration=None):
        if combine_op == 'average_maximum':
            if iteration < int(diverse_params['num_iterations']/2):
                similarity = ops.Similarity(div_weight, mask=similarity_mask,
                                metric=similarity_metric,
                                combine_op=torch.mean)
            else:
                similarity = ops.Similarity(div_weight, mask=similarity_mask,
                                metric=similarity_metric,
                                combine_op=torch.max)
        else:
            similarity = ops.Similarity(div_weight, mask=similarity_mask,
                                metric=similarity_metric,
                                combine_op=combine_op)
        im_set = torch.cat([mei, x], dim=0)

        return similarity(embedding(im_set))

    def make(self, key):        
        # Get params
        mei_params = (MEIParameters & key).fetch1()
        mask_stats_params = (MaskStatsParameters & key).fetch1()
        diverse_params = (DEIParameters & key).fetch1()
        div_ref = (DEIParameters.RefLevel & key).fetch1('ref_level')
        div_weight = (DEIParameters.Weight & key).fetch1('weight')
        single_cell_model, embedding, similarity_mask, mei, mei_activation, mask, mask_x, mask_y, gradient_f, combine_op, postup_op, initial_batch = prepare_params(key, mei_params, mask_stats_params, diverse_params)
        mei_copy = mei.detach().clone()

        print('Optimizing with div_weight=', div_weight, 'div_ref=', div_ref)
        # Set up optimization
        if diverse_params['loss_type'] == 'HingeLoss':
            activation_obj = utils.Compose([single_cell_model, HingeLoss(div_ref, mei_activation), ops.ReverseSign()])
        else: 
            raise NotImplementedError('{} loss type not '
                                    'implemented'.format(diverse_params['loss_type']))

        # Set seed for optimization
        seed = mei_params['mei_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if diverse_params['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        # Optimize
        deis, fevals, regs = featurevis.gradient_ascent(activation_obj, initial_batch,
                                                        step_size=diverse_params['step_size'],
                                                        num_iterations=diverse_params['num_iterations'],
                                                        regularization=partial(self.div_regularization, similarity_mask, diverse_params['similarity'], div_weight, combine_op, embedding, mei_copy),
                                                        post_update=postup_op,
                                                        gradient_f=gradient_f)

        # Evaluate the optimized DEIs
        with torch.no_grad():
            activations = single_cell_model(deis).detach().cpu().numpy().squeeze()  # compute activation
            # latents = get_latent(deis).detach().cpu().numpy().squeeze()  # compute VAE latent representation

        get_sim = partial(self.div_regularization, similarity_mask, diverse_params['similarity'], 1, ops.DoNothing(), embedding, mei_copy)
        avg_sim = get_sim(deis).mean().item()
        max_sim = get_sim(deis).max().item()
        min_sim = get_sim(deis).min().item()

        avg_activation_ratio = np.mean(activations) / mei_activation
        min_activation_ratio = np.min(activations) / mei_activation
        std_activation = np.std(activations) / np.mean(activations)
        sims_to_mei = np.array([get_sim(dei[None]).item() for dei in deis])

        # Insert DEIs
        self.insert1({**key, 'deis': deis.detach().cpu().squeeze().numpy(), 'activations': activations,
                    'fevals': np.array(fevals), 'regs': np.array(regs), 'avg_activation_ratio': avg_activation_ratio, 'min_activation_ratio': min_activation_ratio, 
                    'std_activation': std_activation, 'avg_sim': avg_sim, 'max_sim': max_sim, 'min_sim': min_sim,
                    'sims_to_mei': sims_to_mei})

        # # hacky way of populate by directing inserting entries in the old table old_deis_schema.DEI
        # deis, acts = (old_deis_schema.DEI.DEI * old_deis_schema.DEIEvaluation.DEISet & key).fetch('dei', 'activation', order_by='group_id, neuron_id, dei_id')
        # dic = (old_deis_schema.DEI.DEI * old_deis_schema.DEIEvaluation.DEISet & key & 'dei_id = 0').fetch1()
        # self.insert1({**dic, 'deis': np.stack(deis), 'activations': acts}, ignore_extra_fields=True)

@schema
class DEIThreshold(dj.Lookup):
    definition = """
    threshold_params: int
    ---
    dev_from_ref_threshold: float       # threshold on deviation of average activation from reference level of activation
    std_threshold:          float       # threshold on std activation / avg activation
    min_ratio:              float       # threshold for minimum DEI activation ratio
    selection_criterion:    varchar(16) # criteria for selecting one single run among all valid runs
    """
    contents = [{'threshold_params': 1, 'dev_from_ref_threshold': 1, 'std_threshold': 1,'min_ratio': 0.85, 'selection_criterion': 'min_act'},
                {'threshold_params': 2, 'dev_from_ref_threshold': 1, 'std_threshold': 1,'min_ratio': 0.85, 'selection_criterion': 'largest_weight'}]

@schema
class DEIGoodRun(dj.Computed):
    definition = """
    -> static_models.Model
    -> Dataset.Unit
    -> MEIParameters
    -> MEIMask
    -> MaskStatsParameters
    -> DEIParameters.RefLevel
    -> DEIThreshold
    ---
    weight_id:             int    # id of this weight (0-indexed, same order as in weights list)
    """

    @property
    def key_source(self):
        all_keys = MEIMask * MaskStatsParameters * DEIParameters.RefLevel * DEIThreshold & DEI
        return all_keys #& [{'seed': 1009}, {'seed': 101}]
    
    @staticmethod
    def get_largest_weight_id(key, ref, eval_table):
        threshold_params = (DEIThreshold & key).fetch1()
        diverse_params = (DEIParameters & key).fetch1()
        avg_act_ratios, min_act_ratios, stds = (eval_table & key).fetch('avg_activation_ratio', 'min_activation_ratio', 'std_activation', order_by='weight_id')

        if threshold_params['min_ratio'] == 0:
            satisfied_weight_ids_1 = (avg_act_ratios > (ref - threshold_params['dev_from_ref_threshold'])) & (avg_act_ratios < (ref + threshold_params['dev_from_ref_threshold']))
            satisfied_weight_ids_2 = (stds < threshold_params['std_threshold'])
            satisfied_weight_ids = satisfied_weight_ids_1 & satisfied_weight_ids_2
        else:
            satisfied_weight_ids = np.round(min_act_ratios, 2) >= threshold_params['min_ratio'] 

        idx = np.where(satisfied_weight_ids == True)[0]
        if len(idx) == 0:
            return None
        elif threshold_params['selection_criterion'] == 'min_act':
            return idx[np.argsort(avg_act_ratios[satisfied_weight_ids])[0]] # Select the weight_id corresponding to the lowest activation above threshold (in case the weights are not ordered monotonically)
        elif threshold_params['selection_criterion'] == 'largest_weight':
            return idx[-1]

    def make(self, key):
        # pick the good run once all weights have finished populating
        n_weights = len(DEIParameters.Weight & key)
        if len(DEI & key) == n_weights:
            ref = (DEIParameters.RefLevel & key).fetch1('ref_level')
            weight_id = self.get_largest_weight_id(key, ref, DEI)
            if weight_id is not None:
                self.insert1({**key, 'weight_id': weight_id})

@schema
class EvalParameters(dj.Lookup):
    definition = """
    eval_params:                 int
    ---
    mask_image:                  bool              # whether to mask image or not
    match_stats:                 varchar(16)       # method for matching statistics on the final image, 'mask' or 'ff'
    mask_mean_subtraction:       bool              # whether to subtract mask mean or not during standardization
    """
    contents = [[1, 1, 'ff', 1]]

@schema
class PostOpDEIEvaluation(dj.Computed):
    definition = """
    -> DEI
    -> EvalParameters
    ---
    mei_activation:         float               # activation of standardized MEI         
    dei_activations:        longblob            # activation of standardized DEIs
    sims_to_mei:            longblob            # similarity between MEI and each DEI, in the same order as in dei_activations
    """

    @property
    def key_source(self):
        return DEI * EvalParameters & DEIGoodRun

    def make(self, key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get parameters
        mei_params = (MEIParameters & key).fetch1()
        mask_stats_params = (MaskStatsParameters & key).fetch1()
        mask_float = (MEIMask & key).fetch1('mask')
        diverse_params = (DEIParameters & key).fetch1()
        eval_params = (EvalParameters & key).fetch1()
        if mask_stats_params['fixed_mask_std'] or mask_stats_params['fixed_mask_mean']:
            mei = (MaskFixedMEI & key).fetch1('mei')
        else:
            mei = (MEI & key).fetch1('mei')
        if eval_params['match_stats'] == 'mask':
            target_mean, target_std = float(mask_stats_params['fixed_mask_mean']), float(mask_stats_params['fixed_mask_std'])
        if eval_params['match_stats'] == 'ff':
            target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])

        # Get model and feature space embedding
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device=device, average_batch=False)
        if diverse_params['features'] == 'pixels':
            embedding = ops.Identity()  # operation that returns x as is
        elif diverse_params['features'] == 'feature_vectors':
            # return a feature map matrix in the shape of batch_size x (num_models x feature_vec_length)
            embedding = ops.Feature_Vector_Ensemble(all_models, key['readout_key'], neuron_idx=key['neuron_id'], eye_pos=mean_eyepos, device=device, average_batch=False)
        elif diverse_params['features'] == 'single_grid_population_resps':
            embedding = ops.SingleGridResps(all_models, key['readout_key'], eye_pos=mean_eyepos, neuron_idx=key['neuron_id'], all_neurons=True, average_batch=False, device=device)
        else:
            raise NotImplementedError('{} feature embedding not implemented'.format(diverse_params['features']))
        
        # Compute similarity between DEIs and MEI
        mask = torch.as_tensor(mask_float, dtype=torch.float32, device=device).contiguous()[None, None]
        similarity_mask = mask if eval_params['match_stats'] == 'mask' else None
        mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask_float, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
        deis = (DEI & DEIGoodRun & key).fetch1('deis')
        deis = torch.as_tensor(ops.standardize_image(np.stack(deis), target_mean, target_std, mask_float, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[:, None], dtype=torch.float32, device='cuda')
        get_sim = partial(DEI.div_regularization, similarity_mask, diverse_params['similarity'], 1, ops.DoNothing(), embedding, mei)
        sims_to_mei = np.array([get_sim(dei[None]).item() for dei in deis])
        mei_activation = model(mei).item()
        dei_activations = model(deis).cpu().detach().squeeze().numpy()
        self.insert1({**key, 'mei_activation': mei_activation, 'dei_activations': dei_activations, 'sims_to_mei': sims_to_mei})

@schema
class DEINatControlParameters(dj.Lookup):
    definition = """
    control_params: int
    ---
    image_class:                 varchar(16) 
    crop_h:                      int               # height (in pixels) of crops taken from a search image to augment search dataset
    crop_w:                      int               # width (in pixels) of crops taken from a search image to augment search dataset
    crop_stride:                 int               # stride (in pixels) for taking crops from a search image
    features:                    varchar(16)
    similarity:                  varchar(32)
    similarity_min:              float             # valid control image need to have similarity to MEI larger than similarity_min * target_similarity
    similarity_max:              float             # valid control image need to have similarity to MEI smaller than similarity_max * target_similarity
    n_images:                    int               # number of control images needed 
    control_seed:                int               # seed for randomly selecting n_images images from all valid images 
    batch_size:                  int               # batch size for computing similarity to mei
    mask_image:                  bool              # whether to mask image or not
    normalize_crop:              bool              # whether to normalize each crop to mean 0 and std 1 before making it into a full image
    normalize_crop_within_mask:  bool              # normalize each crop to mean 0 and std 1 within the mei mask before making it into a full image
    match_stats:                 varchar(16)       # method for matching statistics on the final image, 'mask' or 'ff'
    mask_mean_subtraction:       bool              # whether to subtract mask mean or not during standardization
    std_type:                    varchar(16)       # type of std to threshold on, 'mask' or 'ff'
    std_thresh:                  float             # threshold on the std of the each imagine crop
    description:                 varchar(256)      # description of the search pool
    """""
    contents = [(1, 'imagenet', 36, 36, 8, 0.8, 1., 20, 1234, 1000, 0, 0, 0, 'mask', 1, 'mask', 0.e+00, 'multiple crops from each of the high resolution imagenet images'),
                (2, 'imagenet', 36, 36, 8, 0.8, 1., 20, 1234,  500, 0, 1, 0, 'mask', 1, 'mask', 0.e+00, 'multiple crops from each of the high resolution imagenet images, except for the crops with extremely low std'),
                (3, 'imagenet', 36, 36, 8, 0.8, 1., 20, 1234,  500, 1, 1, 0, 'ff', 0, 'ff', 0.e+00, 'multiple crops from each of the high resolution imagenet images'),
                (4, 'imagenet', 36, 36, 8, 0.8, 1., 20, 1234,  100, 1, 1, 0, 'ff', 0, 'ff', 0.e+00, 'multiple crops from each of the high resolution imagenet images'),
                (5, 'imagenet', 36, 36, 8, 0.8, 1., 20, 1234,  100, 1, 1, 0, 'ff', 0, 'ff', 1.e-06, 'multiple crops from each of the high resolution imagenet images, except for the crops with extremely low std'),
                (6, 'imagenet', 36, 36, 8, 0.8, 1., 20, 1234,  100, 1, 1, 1, 'ff', 0, 'mask', 1.e-06, 'multiple crops from each of the high resolution imagenet images, except for the crops with extremely low std within mask'),
                (7, 'imagenet', 36, 36, 8, 0.8, 1., 20, 1234,  100, 1, 0, 0, 'ff', 1, 'mask', 1.e-03, 'multiple crops from each of the high resolution imagenet images, except for the crops with extremely low std within mask'),
                (8, 'imagenet', 36, 36, 8, 0.8, 1., 20, 1234,  100, 1, 0, 0, 'ff', 1, 'mask', 1.e+00, 'multiple crops from each of the high resolution imagenet images, except for the crops with extremely low std within mask')]

@schema
class DEINatControl(dj.Computed):
    definition = """
    -> DEI
    -> DEIThreshold
    -> DEINatControlParameters
    ---        
    n_valid_images:    int        # total number of valid images in the search pool
    max_sim_to_mei:    float      # maxinum similarity of images in the search pool to MEI  
    """
    
    @property
    def key_source(self):
        return DEI * DEIThreshold * DEINatControlParameters & DEIGoodRun

    class Image(dj.Part):
        definition = """ # images at each weight 
        -> master
        image_id:          int               # id of this control image (0-indexed)
        ---
        imagenet_image_id: int               # image_id of the imagenet image_class
        crop_id:           int               # idx of the crop from the imagenet image
        image:             blob@static   # control image
        activation:        float             # activation of the selected masked natural images
        sim_to_mei:        float             # similarity of the selected masked natural images to MEI
        """

    def make(self, key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        control_params = (DEINatControlParameters() & {'control_params': key['control_params']}).fetch1()

        # Process imagenet images and take crops 
        IMAGENET_IMAGE_IDS, IMAGENET_IMAGES = (stimulus.StaticImage.Image & 'image_class = "imagenet"').fetch('image_id', 'image')
        if control_params['shuffle_pool']:
            random.seed(int(key['group_id'] + key['neuron_id']))
            c = list(zip(IMAGENET_IMAGE_IDS, IMAGENET_IMAGES))
            random.shuffle(c)
            IMAGENET_IMAGE_IDS, IMAGENET_IMAGES = zip(*c)
            IMAGENET_IMAGE_IDS = np.array(IMAGENET_IMAGE_IDS)
        crop_h, crop_w, crop_stride = (DEINatControlParameters & key).fetch1('crop_h', 'crop_w', 'crop_stride')
        IM_SIZE = (256, 144)
        imagenet_crops = []
        crop_idxs = []
        for image in tqdm(IMAGENET_IMAGES):
            for idx, (h, w) in enumerate(product(np.arange(0, IM_SIZE[1] - crop_h, crop_stride), np.arange(0, IM_SIZE[0] - crop_w, crop_stride))):
                imagenet_crops.append(image[h:h+crop_h, w:w+crop_w])
                crop_idxs.append(idx)

        print('Searching for group {} neuron {} ...'.format(key['group_id'], key['neuron_id']))
        # Get parameters
        mei_params = (MEIParameters & key).fetch1()
        mask_stats_params = (MaskStatsParameters & key).fetch1()
        mask_float, mask_x, mask_y = (MEIMask & key).fetch1('mask', 'mask_x', 'mask_y')
        if mask_stats_params['fixed_mask_std'] or mask_stats_params['fixed_mask_mean']:
            mei = (MaskFixedMEI & key).fetch1('mei')
        else:
            mei = (MEI & key).fetch1('mei')
        if control_params['match_stats'] == 'mask':
            target_mean, target_std = float(mask_stats_params['fixed_mask_mean']), float(mask_stats_params['fixed_mask_std'])
        if control_params['match_stats'] == 'ff':
            target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])

        # Get model and feature space embedding
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device=device, average_batch=False)
        if control_params['features'] == 'pixels':
            embedding = ops.Identity()  # operation that returns x as is
        elif control_params['features'] == 'feature_vectors':
            # return a feature map matrix in the shape of batch_size x (num_models x feature_vec_length)
            embedding = ops.Feature_Vector_Ensemble(all_models, key['readout_key'], neuron_idx=key['neuron_id'], eye_pos=mean_eyepos, device=device, average_batch=False)
        elif control_params['features'] == 'single_grid_population_resps':
            embedding = ops.SingleGridResps(all_models, key['readout_key'], eye_pos=mean_eyepos, neuron_idx=key['neuron_id'], all_neurons=True, average_batch=False, device=device)
        else:
            raise NotImplementedError('{} feature embedding not implemented'.format(control_params['features']))
        
        # Compute similarity between DEIs and MEI
        mask = torch.as_tensor(mask_float, dtype=torch.float32, device=device).contiguous()[None, None]
        similarity_mask = mask if control_params['match_stats'] == 'mask' else None
        mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask_float, control_params['mask_mean_subtraction'], control_params['mask_image'], control_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
        deis = (DEI & DEIGoodRun & key).fetch1('deis')
        deis = torch.as_tensor(ops.standardize_image(np.stack(deis), target_mean, target_std, mask_float, control_params['mask_mean_subtraction'], control_params['mask_image'], control_params['match_stats'])[:, None], dtype=torch.float32, device='cuda')
        get_sim = partial(DEI.div_regularization, similarity_mask, control_params['similarity'], 1, ops.DoNothing(), embedding, mei)
        sims_to_mei = np.array([get_sim(dei[None]).item() for dei in deis])
        
        # Set the target similarity range from the MEI
        similarity_min, similarity_max = control_params['similarity_min'], control_params['similarity_max']
        try: # when similarity_max and min are float strings, apply the floats as scalings of target_sim
            similarity_max = np.float(similarity_max)
            similarity_min = np.float(similarity_min)
            target_sim = np.max(sims_to_mei)
            target_max = target_sim * similarity_max
            target_min = target_sim * similarity_min
        except ValueError:
            if similarity_max == 'max':
                target_max = np.max(sims_to_mei)
            if similarity_min == 'median':
                target_min = np.median(sims_to_mei)

        # Process image and compute similarity to MEI in batches
        mei_nat_sims, stds = [], []
        n_valid = 0
        for crops in tqdm(get_batch(imagenet_crops, control_params['batch_size'])):
            if not control_params['early_stop'] or (control_params['early_stop'] and n_valid < control_params['early_stop']):
                crops = np.stack(crops)
                # compute std of full crops or within the mask
                if control_params['std_type'] == 'mask': # compute std within the mask
                    cropped_mask = ops.center_and_crop_image(mask_float, mask_x, mask_y, (crops.shape[-1], crops.shape[-2]))[0]
                    _, std = ops.get_mask_stats(crops, cropped_mask)
                elif control_params['std_type'] == 'ff':
                    std = crops.std(axis=(-1, -2))
                stds.extend(std.squeeze())
                # pad crop properly to create image of MEI size
                images = create_whole_mei(crops, mask_float, mask_x, mask_y, (len(crops), 36, 64), normalize_crop = control_params['normalize_crop'])
                # match statistics of images, and mask images if applicable
                images = ops.standardize_image(images, target_mean, target_std, mask_float, control_params['mask_mean_subtraction'], control_params['mask_image'], control_params['match_stats'])
                images = torch.as_tensor(images[:, None], dtype=torch.float32, device='cuda')
                # compute similarity between standardized images and MEI
                torch.cuda.empty_cache()
                with torch.no_grad():
                    sims = get_sim(images)[:len(images)].cpu().detach().squeeze().numpy()
                mei_nat_sims.extend(sims)
                valid = (sims < target_max) & \
                        (sims > target_min) & \
                        (std.squeeze() > control_params['std_thresh'])
                n_valid += valid.sum()

        good_idxs = np.where((np.array(mei_nat_sims) < target_max) & \
                    (np.array(mei_nat_sims) > target_min) & \
                    (np.array(stds) > control_params['std_thresh'])
                    )[0]

        self.insert1({**key, 'n_valid_images': len(good_idxs), 'max_sim_to_mei': np.max(mei_nat_sims)})

        # Gather information of the all good crops and insert 
        if len(good_idxs) >= control_params['n_images']:
            print('Inserting {} out of {} valid controls ...'.format(control_params['n_images'], len(good_idxs)))
            n_crops = len(np.arange(0, IM_SIZE[1] - crop_h, crop_stride)) * len(np.arange(0, IM_SIZE[0] - crop_w, crop_stride))
            IMAGENET_IMAGE_IDS = IMAGENET_IMAGE_IDS.repeat(n_crops)
            np.random.seed(control_params['control_seed'])
            selected_idxs = np.random.choice(good_idxs, control_params['n_images'], replace=False)
            for n, i in enumerate(selected_idxs):
                imagenet_image_id = IMAGENET_IMAGE_IDS[i]
                crop_id = crop_idxs[i]
                sim_to_mei = mei_nat_sims[i]
                im = create_whole_mei(imagenet_crops[i], mask_float, mask_x, mask_y, normalize_crop = control_params['normalize_crop'])
                im = ops.standardize_image(im, target_mean, target_std, mask_float, control_params['mask_mean_subtraction'], control_params['mask_image'], control_params['match_stats'])
                with torch.no_grad():
                    im = torch.as_tensor(im, dtype=torch.float32, device=device).contiguous()[None, None]
                    if control_params['match_stats'] == 'mask':
                        activation = model(im * mask).item()
                    elif control_params['match_stats'] == 'ff':
                        activation = model(im).item()
                self.Image.insert1({**key, 'image_id': n, 'imagenet_image_id': imagenet_image_id, 'crop_id': crop_id, 
                                    'image': im.cpu().detach().squeeze().numpy(), 'activation': activation, 'sim_to_mei': sim_to_mei})

@schema
class DEIControlParameters(dj.Lookup):
    definition = """ # parameters to generate DEI controls
    control_params: int
    ---
    -> FeatureSpace                         # type of embedding used to compute distances
    -> SimilarityMetric                     # similarity metric to use between images
    -> CombinationOperation                 # how to compute overall similarity from all pairwise similarities
    num_deis:                   int         # number of images to generate
    step_size:                  float       # step size for optimizing DEIs
    num_iterations:             int         # number of iteration for optimizing DEIs
    weight:                     float       # list of weights to give to the diversity term during optimization
    loss_type:                  varchar(64) # loss function used for optimization
    decay_constant:             float       # initial scaling of optimization step size
    decay_factor:               float       # step size decay over iterations
    decay_iters:                int         # Number of iterations to wait until next decay_factor is applied
    initial_type:               varchar(16) # image type for initial batch
    diversity_mask:             varchar(45) # mask for measuring diversity 
    init_noise_scale:           float       # scaling factor of white noise on the initial images
    match_stats:                varchar(16) # method for matching image statistics
    mask_mean_subtraction:      bool        # whether to subtract mask mean or not during standardization
    """
    contents = [(1, 'pixels', 'neg_euclidean', 'maximum', 20, 1., 1000, 0.5, 'target_sim', 10., -0.00999, 1, 'MEI', 'mei_mask', 0.1, 'mask', 0),
                (2, 'pixels', 'neg_euclidean', 'maximum', 20, 1., 1000, 0.7, 'target_sim', 10., -0.00999, 1, 'MEI', 'mei_mask', 0.1, 'mask', 0),
                (3, 'pixels', 'neg_euclidean', 'average_maximum', 20, 1., 1000, 0.7, 'target_sim', 10., -0.00999, 1, 'MEI', 'mei_mask', 0.5, 'mask', 0),
                (4, 'pixels', 'neg_euclidean', 'maximum', 20, 1., 1000, 0.2, 'target_sim', 50.,  0.     , 1, 'MEI', 'mei_mask', 0.5, 'ff', 0),
                (5, 'pixels', 'neg_euclidean', 'maximum', 20, 1., 1000, 1. , 'target_sim', 10., -0.00999, 1, 'MEI', 'mei_mask', 0.5, 'mask', 0),
                (6, 'pixels', 'neg_euclidean', 'maximum', 20, 1., 1000, 0.2, 'target_sim', 50.,  0.     , 1, 'MEI', 'mei_mask', 0.5, 'ff', 1),
                (7, 'pixels', 'neg_euclidean', 'maximum', 20, 1., 1000, 0.4, 'target_sim', 50.,  0.     , 1, 'MEI', 'mei_mask', 0.5, 'ff', 1)]

@schema
class DEIControl(dj.Computed):
    definition = """
    -> DEI
    -> DEIThreshold
    -> DEIControlParameters
    """

    class ImageSet(dj.Part):
        definition = """
        -> master
        ---
        fevals:     blob@static                # list of fevals over iterations during optimization
        regs:       blob@static                # list of regularization values over iterations during optimization
        """

    class Image(dj.Part):
        definition = """ # images at each weight 
        -> master.ImageSet
        image_id:     int                   # id of this DEI (0-indexed)
        ---
        image:        blob@static           # control image
        activation:   float                 # activation for this image
        sim_to_mei:   float                 # similarity to MEI
        """

    @property
    def key_source(self):
        return DEI * DEIThreshold * DEIControlParameters & DEIGoodRun

    def make(self, key):
        self.insert1(key)

        # Get parameters
        mei_params = (MEIParameters & key).fetch1()
        mask_stats_params = (MaskStatsParameters & key).fetch1()
        control_params = (DEIControlParameters & key).fetch1()

        single_cell_model, embedding, similarity_mask, mei, mei_activation, mask, _, _, gradient_f, combine_op, postup_op, initial_batch = prepare_params(key, mei_params, mask_stats_params, control_params)
        if control_params['match_stats'] == 'mask':
            target_sim = np.max((DEI & key).fetch1('sims_to_mei'))
            transform = None
        elif control_params['match_stats'] == 'ff': # compute distance on masked and FF stats matched images
            target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
            # standardize MEI and DEIs and measure similarity between them
            stan_mei = ops.standardize_image(mei.cpu().detach().squeeze().numpy(), target_mean, target_std, mask.cpu().detach().squeeze().numpy(), control_params['mask_mean_subtraction'], True, control_params['match_stats'])
            mei = torch.as_tensor(stan_mei[None, None], dtype=torch.float32, device='cuda')
            mei_copy = mei.detach().clone()
            deis = (DEI & key).fetch1('deis')
            stan_deis = ops.standardize_image(np.stack(deis), target_mean, target_std, mask.cpu().detach().squeeze().numpy(), control_params['mask_mean_subtraction'], True, control_params['match_stats'])
            deis = torch.as_tensor(stan_deis[:, None], dtype=torch.float32, device='cuda')
            @varargin
            def transform(x):
                mask_mean = torch.sum(x * mask, (-1, -2), keepdim=True) / mask.sum()
                x = (x - mask_mean) * mask
                x = ops.ChangeStats(target_std, target_mean)(x)
                return x
            postup_op = None
            similarity_mask = None
            similarity = partial(DEI.div_regularization, similarity_mask, control_params['similarity'], 1, ops.DoNothing(), embedding, mei_copy)
            target_sim = similarity(deis)[:len(deis)].max().item()
            
        if control_params['loss_type'] == 'target_sim':
            loss = lambda x: - torch.sum((similarity(x)[:len(x)] - target_sim)**2)

        # Diversity regularization
        get_sim = partial(DEI.div_regularization, similarity_mask, control_params['similarity'], control_params['weight'], combine_op, embedding, mei_copy)

        # Optimize
        controls, fevals, regs = featurevis.gradient_ascent(loss, initial_batch,
                                                        transform=transform, 
                                                        step_size=control_params['step_size'],
                                                        num_iterations=control_params['num_iterations'],
                                                        regularization=get_sim,
                                                        post_update=postup_op,
                                                        gradient_f=gradient_f)
        if transform is not None:
            controls = transform(controls)
            
        self.ImageSet.insert1({**key, 'fevals': np.array(fevals), 'regs': np.array(regs)})

        for image_id, image in enumerate(controls):
            with torch.no_grad():
                activation = single_cell_model(image[None]).item()  # compute activation
                sim_to_mei = similarity(image[None]).item()
            self.Image.insert1({**key, 'image_id': image_id, 'image': image.cpu().squeeze().numpy(), 
                                'activation': activation, 'sim_to_mei': sim_to_mei})


def compute_diversity(neuron_key, param_dict, real_neuron=True, ref_param_dict=dict(mei_params=10, mask_params=3, mask_stats_params=2, diverse_params=14, threshold_params=2), neuron_set=dict(set_id=1, method_id=1)):
    # compute linear fit from previous good neurons
    masks, sims = (old_deis_schema.MEIMask * old_deis_schema.DEIEvaluation.DEISet * old_deis_schema.DEIGoodRun & ref_param_dict & (NeuronSet.Neuron & neuron_set)).fetch('mask', 'avg_sim')
    mask_sizes = np.stack(masks).sum(axis=(1,2))
    a, b = np.polyfit(mask_sizes, -sims, deg=1)
    
    # compute diversity as residual to the fit
    if real_neuron:
        neuron_masks, neuron_sims = (MEIMask * DEI & (DEIGoodRun & param_dict) & neuron_key).fetch('mask', 'avg_sim', order_by='group_id, neuron_id')
    else:
        toy_deis = dj.create_virtual_module('neurostatic_toy_deis', 'neurostatic_toy_deis')
        neuron_masks, neuron_sims = (toy_deis.MEIMask * toy_deis.DEINew & (toy_deis.DEINewGoodRun & param_dict) & neuron_key).fetch('mask', 'avg_sim', order_by='group_id, member_id')

    neuron_mask_sizes = np.stack(neuron_masks).sum(axis=(1,2))
    res = - neuron_sims - (a * neuron_mask_sizes + b)
    
    # # linear fit for population dei 
    # rest = 'mei_params = 10 and mask_params = 3 and threshold_params = 2'
    # masks, sims = (base.MEIMask * deis_schema.DEI * deis_schema.DEIGoodRun & 'diverse_params = 21' & (deis_schema.NeuronSet.Neuron & {'set_id': 1, 'method_id': 1}) & rest).fetch('mask', 'avg_sim')
    # mask_sizes = np.stack(masks).sum(axis=(1,2))
    # a, b = np.polyfit(mask_sizes, 1-sims, deg=1)
    # a = 0.0004493586550405008
    # b = 0.2325953463156649

    return a, b, res
    
def convert_score(euclidean_distance, lower_bound=7.72, upper_bound=13.92):
    return (euclidean_distance - lower_bound) / (upper_bound - lower_bound)

def compute_diversity_index(keys):
    dists = - (DEI & keys).fetch('avg_sim', order_by='group_id, neuron_id')
    return convert_score(dists)

# ##################################################### From Dat: Variable component texture generation ###########################################################
class CombinedCriterion():
    def __init__(self,predictive_model,ref_level,amax,mask,Lambda=0.0):
        super(CombinedCriterion,self).__init__()
        self.hinge_loss = HingeLoss(ref_level,amax)
        self.div_reg = ops.Similarity(mask=mask,metric='neg_euclidean',combine_op=ops.DoNothing())
        self.Lambda = Lambda
        self.predictive_model = predictive_model
        
    # Assume x is the image
    @varargin
    def __call__(self,x):
        a = self.predictive_model(x)
        b = self.div_reg(x)
        b = torch.mean(b)
        return self.hinge_loss(a) + self.Lambda * b, a.mean().item()/self.hinge_loss.amax, - b.detach().item()
    
def load_model(key, use_avg_model=True, device='cuda', seed_rest=None):
    if seed_rest is None:
        if use_avg_model:
            if key['seed'] == 1009:
                ssid = 1
            elif key['seed'] == 101:
                ssid = 4
            seeds = (SeedSet & {'ssid': ssid}).fetch1('seeds')
            seed_rest = [{'seed': s} for s in seeds]
        else:
            seed_rest = {'seed': key['seed']}
        
    model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                    use_avg_model else key)
    all_keys = (static_models.Model & model_key & seed_rest).fetch('KEY')
    all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

    # Get some train stats
    mean_eyepos = ([0, 0] if (Dataset.TrainStats & key).fetch1('norm_eyepos')
                else (Dataset.TrainStats & key).fetch1('mean_eyepos'))

    # Create model ensemble
    mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                            device=device).unsqueeze(0)
    model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                            neuron_idx=key['neuron_id'], device=device, average_batch=False)
    return model
        
def get_variable_masks(deis,mei,mask_params,values=np.linspace(0.3,0.7,9),params={'closing_iters':2,'gaussian_sigma':1.5}, flipped_mask=False, std_mask=None):

    img = deis.std(axis=0)
    img = img/img.max()
    
    def get_binary_mei_mask(mei,mask_params):
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
    
    mei_binary_mask = get_binary_mei_mask(mei, mask_params)
    
    def helper(threshold, mei_binary_mask, flipped_mask=False):
        if not flipped_mask:
            thresholded = img > threshold
        else:
            thresholded = img <= threshold

        # Remove small holes in the thresholded image and connect any stranding pixels
        closed = ndimage.binary_closing(thresholded, iterations=params['closing_iters'])

        # Remove any remaining small objects
        labeled = morphology.label(closed, connectivity=2)
        most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
        oneobject = labeled == most_frequent
        overlapped = oneobject & mei_binary_mask
        # Smooth edges
        smoothed = ndimage.gaussian_filter(overlapped.astype(np.float32),sigma=params['gaussian_sigma'])
        return smoothed
    
    result = {'variable_mask':[],'fraction_std':[]}
    for threshold in np.linspace(0.05,0.95,1000):
        result['variable_mask'].append(helper(threshold, mei_binary_mask, flipped_mask))
        if std_mask is None:
            result['fraction_std'].append((deis.std(axis=0) * result['variable_mask'][-1]).sum() / deis.std(axis=0).sum())
        else:
            result['fraction_std'].append((deis.std(axis=0) * result['variable_mask'][-1]).sum() / (deis.std(axis=0) * std_mask).sum())

    idx = np.argmin(abs(np.array(result['fraction_std'])[:,None]-values),axis=0)
    temp = []
    for i in idx:
        temp_dict = {}
        for temp_key in result.keys():
            temp_dict[temp_key] = result[temp_key][i]
        temp.append(temp_dict)
    return temp, [np.linspace(0.05,0.95,1000)[i] for i in idx]

class LinearImageModel(nn.Module):
    # mei, mask, v_mask is numpy array
    def __init__(self,mei,v_mask,t_f='c_f_scaled',t_shape=(72, 128),eval_padding=(0,0,0,0),center_padding=(0,0,0,0),default_n_crops=64, eval_texture=None, device='cuda'):
        
        super(LinearImageModel,self).__init__()
        self.device = device
        self.eval_padding = eval_padding
        self.center_padding = center_padding
        
        self.mei_f = ops.ChangeStats(mean=mei.mean(),std=mei.std())
        fixed_c = mei * (1.0-v_mask)
        var_c = mei * v_mask
        self.var_c_f = ops.ChangeStats(mean = var_c.mean(),std = var_c.std())
        if t_f == 'c_f_scaled':
            # Ratio to compensate for empty space in texture
            ratio_for_empty_space = 1.3
            self.var_t_f = ops.ChangeStats(mean = var_c.mean(),std = var_c.std() * np.sqrt(v_mask.size/v_mask.sum()) * ratio_for_empty_space)
        elif t_f == 'c_f':
            self.var_t_f = self.var_c_f
        self.register_buffer('fixed_c',torch.tensor(np.array(fixed_c),dtype=torch.float32).unsqueeze(0).unsqueeze(0).contiguous())
        self.register_buffer('v_mask',torch.tensor(np.array(v_mask),dtype=torch.float32))
        
        self.image_shape = var_c.shape
        self.t_shape = t_shape
        # self.initialize_t(initial_t)
        self.default_n_crops = default_n_crops

        # Hack for evaluation of a pre-optimized texture
        if eval_texture is not None:
            self.register_buffer('eval_texture',torch.tensor(np.array(eval_texture),dtype=torch.float32).unsqueeze(0).unsqueeze(0).contiguous())
        else: self.eval_texture = eval_texture
        
    def initialize_t(self,initial_t=None):
        if initial_t is not None:
            if not initial_t.shape[-2:] == self.t_shape:
                raise ValueError('Initial t shape is wrong from t_shape')
            self.t = nn.Parameter(torch.tensor(initial_t[None, None], dtype=torch.float32, device=self.device).contiguous())
        else:
            self.t = nn.Parameter(torch.randn(1, 1, self.t_shape[0], self.t_shape[1], device=self.device))
        self.standardize_t_f()

    def standardize_t_f(self):
        with torch.no_grad():
            self.t.data = self.var_t_f(self.t.data)
            
    def get_v_c(self,n_crops=None,key='full'):
        if n_crops is None:
            n_crops = self.default_n_crops
        v_c, crop_x, crop_y = ops.RandomCrop(self.image_shape[0], self.image_shape[1], n_crops)(self.get_texture(key=key,in_numpy_format=False))
        v_c = v_c*self.v_mask[None,None,:,:]
        v_c = self.var_c_f(v_c)
        return v_c, crop_x, crop_y
    
    @varargin
    def forward(self,n_crops=None,key='full'):
        return self.mei_f(self.fixed_c + self.get_v_c(n_crops,key)[0])
    
    def get_texture(self,key='centered',in_numpy_format=True):
        x = self.t
        if key != 'full':
            if key == 'centered':
                padding = self.center_padding
            elif key == 'eval':
                # Hack for evaluation of a pre-optimized texture
                if self.eval_texture is not None:
                    x = self.eval_texture
                    if in_numpy_format:
                        x = x.detach().cpu().numpy().squeeze()
                    return x
                padding = self.eval_padding
            else:
                raise ValueError(key + ' is not specified')
            x = x[:,:,padding[0]:-padding[1],padding[2]:-padding[3]]
            
        if in_numpy_format:
            x = x.detach().cpu().numpy().squeeze()
        return x

def cal_texture_minimum_shape(mask,target_shape = (36,64),threshold=0.1,padding=5):
    x,y = np.where(mask>threshold)
    height, width = mask.shape
    top, bottom = min(x), mask.shape[0] - max(x)
    left, right = min(y), mask.shape[1] - max(y)
    eval_padding = [height - top - bottom + padding, height - top - bottom + padding, width - left - right + padding, width - left - right + padding]
    center_padding = [i+j for i,j in zip(eval_padding,(top,bottom,left,right))]
    texture_shape = target_shape[0] + center_padding[0] + center_padding[1], target_shape[1] + center_padding[2] + center_padding[3]
    return texture_shape, eval_padding, center_padding

def train_texture(image_model, predictive_model, mei_act, mask, start_seed=0, history_save_path='opt_history.pickle', device='cuda',
                  optimizer_args={'optimizer_name':'SGD','lr':1e3,'smooth_gradient':True,'gradient_sigma':2.0,'decay_factor':0.0},
                  criterion_args={'ref_level':1.00,'Lambda':1e-2},
                  training_args={'batch_size':256,'n_iter_per_eval':10,'max_iters':1000}):
    
    # Load out the necessary arguments
    optimizer_name, lr, smooth_gradient, gradient_sigma, decay_constant, decay_factor, decay_iters = [optimizer_args[i] for i in ['optimizer_name', 'lr', 'smooth_gradient', 'gradient_sigma', 'decay_constant', 'decay_factor', 'decay_iters']]
    ref_level,Lambda = [criterion_args[i] for i in ['ref_level','Lambda']]
    batch_size, n_iter_per_eval, max_iters = [training_args[i] for i in ['batch_size','n_iter_per_eval','max_iters']]
    
    # Set up some functions
    if smooth_gradient:
        if not decay_factor:
            t_smooth_f = ops.GaussianBlur(gradient_sigma)
        else:
            t_smooth_f = utils.Compose([ops.GaussianBlur(gradient_sigma), ops.MultiplyBy(decay_constant, decay_factor, decay_iters)])

    amax = mei_act
    criterion = CombinedCriterion(predictive_model,ref_level,amax,torch.tensor(mask,dtype=torch.float32,device=device),Lambda)
    
    torch.manual_seed(start_seed)
    if not os.path.exists(history_save_path):
        logger.info('Initializing training')
        history = {'n_iter':[], 'total_loss':[],'act':[],'div_reg':[], 'texture_iter':[], 'texture':[]}
        # Store data (serialize)
        with open(history_save_path, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # initialize texture with white noise
        image_model.initialize_t()
        
    else:
        logger.info('Loading training history from checkpoint')
        # Load data (deserialize)
        try:
            with open(history_save_path, 'rb') as handle:
                history = pickle.load(handle)
             # load texture from latest history
            image_model.initialize_t(history['texture'][-1]) if history['texture'] else image_model.initialize_t()
       
        except (pickle.UnpicklingError, ValueError, EOFError) as error:
            logger.info(error)
            warnings.warn('Optimization reinitialized due to UnpicklingError or ValueError!')
            history = {'n_iter':[], 'total_loss':[],'act':[],'div_reg':[], 'texture_iter':[], 'texture':[]}
            # Store data (serialize)
            with open(history_save_path, 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # initialize texture with white noise
            image_model.initialize_t()

    n_iter = history['n_iter'][-1] + 1 if history['n_iter'] else 1

    # Set up optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(image_model.parameters(),lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(image_model.parameters(),lr=lr)
    else:
        raise ValueError('Optimizer unspecified')

    # Use forward trick to create large virtual batch so diversity is calculated in batch not as a whole
    # forward_size or actual batch size is determined to maximize speed
    forward_size = 32
    accumulation_steps = math.ceil(batch_size/forward_size)
    
    for i in tqdm(range(n_iter, max_iters+1)):
        seed = i + start_seed + int(1e6)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        optimizer.zero_grad()
        image_model.t.requires_grad_()
        
        temp = {'total_loss':[],'act':[],'div_reg':[]}  
        for j in range(accumulation_steps):
            x = image_model(n_crops=forward_size,key='full')
            loss,act,div_reg = criterion(x)
            temp['total_loss'].append(loss.item())
            temp['act'].append(act)
            temp['div_reg'].append(div_reg)    
            loss.backward()
            
        for key in temp.keys():
            history[key].append(np.mean(temp[key]))
        history['n_iter'].append(i)

        # Smooth the texture gradient
        if smooth_gradient:
            with torch.no_grad():
                image_model.t.grad = t_smooth_f(image_model.t.grad,iteration=i)                    
        
        optimizer.step()

        # Standardize the texture
        image_model.standardize_t_f()
        
        # For efficiency, we only save textures and optimization history every n_iter_per_eval
        if (i % n_iter_per_eval) == 0:
            history['texture'].append(image_model.get_texture(key='full',in_numpy_format=True))
            history['texture_iter'] = i
            # Store data (serialize)
            with open(history_save_path, 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return image_model, history

@schema
class TextureParameters(dj.Lookup):
    definition="""
    texture_params: int
    ---
    ref_level:             float             # ref level for the activation
    lambda:                float             # coefficient for the diversity regularization
    closing_iters:         int               # closing iters parameter for variable mask
    gaussian_sigma:        float             # gaussian sigma to smooth the edge of variable mask
    target_height:         int               # target centered texture height 
    target_width:          int               # target centered texture width
    texture_f:             varchar(16)       # description of texture standardization function
    padding:               int               # padding for the texture to avoid edge effect
    mask_threshold:        int               # mask threshold to initialize full texture shape
    n_crops:               int               # number of random crops taken for each optimization step
    optimizer_name:        varchar(64)       # optimizer type, e.g. 'SGD' or 'Adam'
    lr:                    float             # optimizer step size     
    smooth_gradient:       bool              # whether to smoothen gradient 
    gradient_sigma:        float             # gradient smoothening sigma
    decay_constant:        float             # initial scaling of optimization step size
    decay_factor:          float             # step size decay over iterations
    decay_iters:           int               # number of iterations to wait before the next decay_factor is applied
    n_iter_per_eval:       int               # every number of iterations for evaluation during optimization
    n_iters:               int               # total number of iterations for optimization
    use_avg_model:         bool              # whether to use average model or first model
    sample_criterion:      varchar(16)       # method of selecting samples from the optimized texture, 'random', 'closest', or 'diverse'
    """

    class TargetFractionStd(dj.Part):
        definition = """
        -> master
        texture_id:          int
        ---
        target_fraction_std: float # targeted fraction of DEI std within the variable mask
        """
    
    def fill(self, key):
        self.insert1(key, ignore_extra_fields=True)
        self.TargetFractionStd.insert1({'texture_params': key['texture_params'], 'texture_id': -1, 'target_fraction_std': 1.0})
        for texture_id, fraction_std in enumerate(key['fraction_stds']):
            self.TargetFractionStd.insert1({'texture_params': key['texture_params'], 'texture_id': texture_id, 'target_fraction_std': fraction_std})

@files('static')
@schema
class Texture(dj.Computed):
    definition = """
    -> DEI
    -> DEIThreshold
    -> TextureParameters.TargetFractionStd
    ---
    fraction_std:                  float               # actual fraction stds
    p:                             float               # fraction of variable mask
    variable_mask:                 longblob            # variable mask to sample from texture
    fixed_part:                    longblob            # fixed part of the DEI
    variable_crops:                blob@static     # variable samples
    full_texture:                  longblob            # full texture
    eval_texture:                  longblob            # texture used for evaluation
    centered_texture:              longblob            # centered texture to sample from
    samples:                       blob@static     # samples from the centered texture
    sample_acts:                   longblob            # sample activations
    sample_avg_div:                float               # sample average diversity
    sample_avg_activation_ratio:   float               # sample average activation ratio
    sample_min_activation_ratio:   float               # sample min activation ratio
    sample_std_activation:         float               # sample activation ratio sd
    dei_avg_div:                   float               # corresponding DEI average diversity
    dei_avg_activation_ratio:      float               # corresponding DEI average activation ratio
    history_save_path:             varchar(255)        # file path for the saved dictionary of texture optimization history
    """

    @property
    def key_source(self):
        return DEI * DEIThreshold * TextureParameters.TargetFractionStd & DEIGoodRun
    
    # This method is only applicable if the samples are chosen by closest and sorted
    @staticmethod
    def cal_dist_between_set(x,y,average=True):
        from sklearn.metrics import pairwise_distances
        dist = np.diag(pairwise_distances(x.reshape(len(x),-1),y.reshape(len(y),-1),metric='euclidean'))
        if average:
            return np.mean(dist)
        else: return dist

    @staticmethod
    def select_texture_dei(image_model,dei=None,sample_criterion='random',n_sample=1000,batch_size=64,device='cuda'):
        from sklearn.metrics import pairwise_distances
        n_batch = math.floor(n_sample/batch_size)
        redundant = n_sample - batch_size * n_batch
        texture_dei = []
        with torch.no_grad():
            for i in range(n_batch):
                texture_dei.append(image_model(batch_size,key='eval').cpu().detach().numpy().squeeze())
            if redundant: texture_dei.append(image_model(redundant,key='eval').cpu().detach().numpy().squeeze())
        texture_dei = np.concatenate(texture_dei)
        if len(texture_dei.shape) == 2:
            texture_dei = texture_dei[None]
        
        if sample_criterion == 'random':
            texture_dei = texture_dei[:n_sample]
        else:
            distance_matrix = pairwise_distances(dei.reshape(len(dei),-1),texture_dei.reshape(len(texture_dei),-1),metric='euclidean')

            if sample_criterion == 'closest':
                # Find nearest point
                idx = np.argmin(distance_matrix,axis=1)
                texture_dei = texture_dei[idx]
            elif sample_criterion == 'diverse':
                def greedy_select_points(n_pts,distance_matrix):
                    # Start with the best pair
                    best_pair = np.unravel_index(distance_matrix.argmax(), distance_matrix.shape)
                    P = set(best_pair)
                    while(len(P)<n_pts):
                        maxdist = 0
                        vbest = None
                        for v in range(len(distance_matrix)):
                            if not v in P:
                                for vprime in P:
                                    if distance_matrix[v,vprime]>maxdist:
                                        maxdist = distance_matrix[v,vprime]
                                        vbest = v
                        P.add(vbest)
                    return np.array(list(P))
                
                # Find the most diverse set of points
                idx = greedy_select_points(len(dei),distance_matrix)
                texture_dei = texture_dei[idx]
            else:
                raise ValueError('sample_criterion {} is not defined'.format(sample_criterion))
        
        return texture_dei#,np.min(distance_matrix,axis=1)
    
    # Take in key of neurostatic.Texture
    @staticmethod
    def hacky_method(key, predictive_model, sample_criterion='closest',n_sample=1000,batch_size=64,device='cuda'):
        # For each key return samples, sample_acts, sample_avg_div, sample_avg_activation_ratio, sample_min_activation_ratio, sample_std_activation
        deis = (DEI & key).fetch1('deis')
        deis = np.stack(deis)

        v_mask,eval_texture = (Texture & key).fetch1('variable_mask','eval_texture')
        mei,mei_act = (MEI & key).fetch1('mei','activation')
        image_model = LinearImageModel(mei,v_mask,eval_texture = eval_texture).to(device)
        mask = np.array((MEIMask() & key).fetch1('mask'))
        mask = torch.tensor(mask,dtype=torch.float32,device=device)
        div_reg = utils.Compose([ops.Similarity(mask=mask,metric='neg_euclidean',combine_op=ops.DoNothing()), ops.ReverseSign()])
        torch.manual_seed(1234)
        texture_dei = Texture.select_texture_dei(image_model,deis,sample_criterion,n_sample,batch_size,device)
        with torch.no_grad():
            samples = torch.tensor(texture_dei,dtype=torch.float32,device=device).unsqueeze(1).contiguous()
            sample_avg_div = np.mean(div_reg(samples).cpu().numpy())
            sample_acts = predictive_model(samples).detach().cpu().numpy().squeeze()
            samples = samples.detach().cpu().numpy().squeeze()
            sample_act_ratios = sample_acts/mei_act
            
        return {'samples':samples,'sample_acts':sample_acts,'sample_avg_div':sample_avg_div,'sample_avg_activation_ratio':sample_act_ratios.mean(),
                    'sample_min_activation_ratio': min(sample_act_ratios),'sample_std_activation': sample_act_ratios.std()}

    def make(self, key):
        # Get parameters and original DEIs 
        device = 'cuda'
        texture_parameters = (TextureParameters * TextureParameters.TargetFractionStd & key).fetch1()
        mask = (MEIMask & key).fetch1('mask')
        mei, mei_act = (MEIMask * MEI & (DEIGoodRun & key)).fetch1('mei','activation')
        mei_params = (MEIParameters & key).fetch1()
        mask_params = (MaskParameters & key).fetch1()
        deis, dei_acts, dei_div, dei_avg_activation_ratio = (MEI.proj('mei', mei_act='activation') * DEI & key & DEIGoodRun).fetch1('deis','activations','avg_sim','avg_activation_ratio')
        deis = np.stack(deis)

        # Compute variable masks from original DEIs
        std_mask = None if texture_parameters['full_dei_std'] else mask
        variable_mask = get_variable_masks(deis,mei,mask_params,values=[texture_parameters['target_fraction_std']],
                                            params={'closing_iters':texture_parameters['closing_iters'],
                                                    'gaussian_sigma':texture_parameters['gaussian_sigma']},
                                            std_mask=std_mask)[0][0]

        # Load predictive model of the neuron
        predictive_model = load_model(key, texture_parameters['use_avg_model'], device=device)
        
        # Calculate texture shape and appropriate padding
        texture_shape, eval_padding,center_padding = cal_texture_minimum_shape(mask=variable_mask['variable_mask'],target_shape =(texture_parameters['target_height'],texture_parameters['target_width']),
                                                                            threshold=texture_parameters['mask_threshold'],padding=texture_parameters['padding'])
        
        image_model = LinearImageModel(mei,variable_mask['variable_mask'],t_f=texture_parameters['texture_f'],t_shape=texture_shape,eval_padding=eval_padding,center_padding=center_padding,
                                        default_n_crops=texture_parameters['n_crops']).to(device)
        
        # Optimize texture 
        history_save_path = os.path.join(self.tuple_dir(key, create=True), "opt_history.pickle")
        image_model, history = train_texture(image_model, predictive_model, mei_act, mask, start_seed=mei_params['mei_seed'], device=device,
                                            history_save_path = history_save_path,
                                            optimizer_args={'optimizer_name':texture_parameters['optimizer_name'],
                                                            'lr':texture_parameters['lr'], 
                                                            'smooth_gradient':texture_parameters['smooth_gradient'],
                                                            'gradient_sigma':texture_parameters['gradient_sigma'],
                                                            'decay_constant':texture_parameters['decay_constant'], 
                                                            'decay_factor':texture_parameters['decay_factor'], 
                                                            'decay_iters': texture_parameters['decay_iters']},
                                            criterion_args={'ref_level':texture_parameters['ref_level'],
                                                            'Lambda':texture_parameters['lambda']},
                                            training_args={'batch_size':texture_parameters['n_crops'],
                                                            'n_iter_per_eval':texture_parameters['n_iter_per_eval'],
                                                            'max_iters':texture_parameters['n_iters']})
        
        # Compute similarity within MEI mask       
        mask_tensor = torch.tensor(mask,dtype=torch.float32,device=device)
        div_reg = utils.Compose([ops.Similarity(mask=mask_tensor,metric='neg_euclidean',combine_op=ops.DoNothing()), ops.ReverseSign()])

        # Sample crops from final texture to form texture DEIs, options: random, most diverse, closest to each DEI
        torch.manual_seed(1234)
        texture_dei = self.select_texture_dei(image_model,deis,sample_criterion=texture_parameters['sample_criterion'], n_sample=len(deis))
        with torch.no_grad():
            variable_crops = image_model.get_v_c(key='eval',n_crops=len(deis))[0].detach().cpu().numpy().squeeze()
            samples = torch.tensor(texture_dei,dtype=torch.float32,device=device).unsqueeze(1).contiguous()
            sample_avg_div = np.mean(div_reg(samples).cpu().numpy())
            sample_acts = predictive_model(samples).detach().cpu().numpy().squeeze()
            samples = samples.detach().cpu().numpy().squeeze()
            sample_act_ratios = sample_acts/mei_act
            
        result = {**key, 'fraction_std':variable_mask['fraction_std'], 'p':variable_mask['variable_mask'].sum()/mask.sum(),
                    'variable_mask':variable_mask['variable_mask'], 'fixed_part':image_model.fixed_c.detach().cpu().numpy().squeeze(), 'variable_crops':variable_crops,
                    'full_texture':image_model.get_texture('full',True), 'eval_texture':image_model.get_texture('eval',True), 'centered_texture':image_model.get_texture('centered',True),
                    'samples':samples,'sample_acts':sample_acts, 'sample_avg_div':sample_avg_div, 'sample_avg_activation_ratio':sample_act_ratios.mean(),
                    'sample_min_activation_ratio': min(sample_act_ratios), 'sample_std_activation': sample_act_ratios.std(), 
                    'dei_avg_div':dei_div, 'dei_avg_activation_ratio':dei_avg_activation_ratio, 'history_save_path': history_save_path}
        
        self.insert1(result)

        # # Hack populate - to insert for texture_params that inherits pre-optimized texture from another texture_params but uses a different method for sampling
        # self.insert1(key)
        # sample_criterion = (TextureParameters & key).fetch1('sample_criterion')
        # temp_key = key.copy()
        # temp_key['texture_params'] = 13
        # predictive_model = load_model(key,device='cuda')
        # tid = key['texture_id']
        # tuples = (self & temp_key & {'texture_id': tid}).fetch1()
        # part_key = (self & temp_key & {'texture_id': tid}).fetch1('KEY')
        # sample_tuples = self.hacky_method(part_key, predictive_model, sample_criterion=sample_criterion)
        # tuples['texture_params'] = key['texture_params']
        # for k in list(sample_tuples.keys()):
        #     if k in tuples.keys():
        #         tuples.pop(k)
        # self.insert1({**tuples, **sample_tuples})
        
        # # hack to directly insert populated tuples from old_deis_schema.TextureSynthesis.TextureSynthesis into the current table
        # names = self.heading.names
        # dic = (old_deis_schema.TextureSynthesis.TextureSynthesis.proj(*names[:-1]) & key).fetch1()
        # self.insert1({**dic, 'history_save_path': ''})

@schema
class TextureScoreParameters(dj.Lookup):
    definition = """
    score_params: int
    ---
    target_ratio: float        # Target for sample average activation ratio
    score_method: varchar(45)  # Name of the scoring method
    description:  varchar(256) # Description of the scoring method
    """
    contents = [{'score_params':1, 'target_ratio': 0.85, 'score_method': 'closest_idx', 'description': 'Hard threshold by the target ratio'},
                {'score_params':2, 'target_ratio': 0.85, 'score_method': 'f_measurement', 'description': 'Relaxed score based on F-measurement to balance between sample responses and diversity'},
                {'score_params':3, 'target_ratio': 0.8, 'score_method': 'max_idx', 'description': 'Hard threshold by the target ratio'},]

@schema
class TextureGoodRun(dj.Computed):
    definition = """
    -> DEI
    -> DEIThreshold
    -> TextureParameters
    -> TextureScoreParameters
    ---
    texture_id:             int    # Id of the run 
    """

    @property
    def key_source(self):
        return DEI * DEIThreshold * TextureParameters * TextureScoreParameters & Texture

    @staticmethod
    def f_score(x, y):
        return 2 * (x*y) / (x+y)

    @staticmethod
    def hard_score(x, c):
        return - abs (x - c)

    @staticmethod
    def compute_bipartiteness(p):
        return 1 - 2 * np.abs(p - 0.5)

    @staticmethod
    def compute_p(key, texture_table, texture_goodrun_table, threshold=0.8):
        n_textures = len(TextureParameters.TargetFractionStd & key)
        assert len(texture_table & key) == n_textures, 'Still remains texture to be generated!'
        if (texture_table & key & 'texture_id = -1').fetch1('sample_avg_activation_ratio') > threshold:
            p = 1.0
        elif (texture_table & key & 'texture_id = 0').fetch1('sample_avg_activation_ratio') < threshold:
            p = 0.0
        else:
            texture_id = (texture_goodrun_table & key).fetch1('texture_id')
            p = (texture_table & key & {'texture_id': texture_id}).fetch1('p')
        bipartiteness = 1 - 2 * np.abs(p - 0.5)
        return p, bipartiteness
    
    def make(self, key):
        # pick the good run once all weights have finished populating
        n_textures = len(TextureParameters.TargetFractionStd & key)
        if len(Texture & key) == n_textures:
            target_ratio, score_method, include_full, norm_with_dei_diversity = (TextureScoreParameters & key).fetch1('target_ratio', 'score_method', 'include_full', "norm_with_dei_diversity")
            if not include_full:
                texture_ids, sample_avg_activation_ratios, sample_avg_divs, dei_divs = (Texture & key & 'texture_id >= 0').fetch('texture_id','sample_avg_activation_ratio','sample_avg_div', 'dei_avg_div', order_by='texture_id')
            else:
                texture_ids, sample_avg_activation_ratios, sample_avg_divs, dei_divs = (Texture & key).fetch('texture_id','sample_avg_activation_ratio','sample_avg_div', 'dei_avg_div', order_by='texture_id')

            # Hard score
            if score_method == 'closest_idx':
                score = self.hard_score(sample_avg_activation_ratios, target_ratio)
                tid = texture_ids[np.argmax(score)]
            # f score
            elif score_method == 'f_measurement':
                if norm_with_dei_diversity:
                    score = self.f_score(sample_avg_activation_ratios, sample_avg_divs / (-dei_divs[0]))
                else:
                    score = self.f_score(sample_avg_activation_ratios, sample_avg_divs / sample_avg_divs.max())
                tid = texture_ids[np.argmax(score)]
            # max idx that results in activation higher than threshold, if no valid run then pick id 0
            elif score_method == 'max_idx':
                if (sample_avg_activation_ratios > target_ratio).sum() > 0:
                    tid = texture_ids[sample_avg_activation_ratios > target_ratio].max()
                else:
                    tid = 0
            self.insert1({**key, 'texture_id':tid})

@schema
class TextureLookup(dj.Lookup):
    definition = """
    synthesis_id       : varchar(256)                 # unique identifier of the texture synthesis key
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
        dics = (Texture & texture_key).fetch('KEY')
        for i, dic in enumerate(dics):
            dic['synthesis_id'] = static_utils.key_hash(dic)
            self.insert1(dic, skip_duplicates=True)


from staticnet_analyses.gabors import GratingGenerator, SearchRange, OptimalGaborParameters

class TwoPartGrating(nn.Module):
    # mei, mei_mask, v_mask is numpy array; v_grating, f_grating is tensor
    def __init__(self, mei, mei_mask, v_mask, v_params, f_params, grating_generator, device='cuda'):
        
        super(TwoPartGrating,self).__init__()
        self.device = device
        self.mei_mask = torch.tensor(np.array(mei_mask), dtype=torch.float32, device=self.device)
        self.v_mask = torch.tensor(np.array(v_mask), dtype=torch.float32, device=self.device)
        
        fixed_c = mei * (1.0-v_mask) * mei_mask
        var_c = mei * v_mask * mei_mask
        self.var_c_f = ops.ChangeStats(mean = var_c.mean(),std = var_c.std())
        self.fixed_c_f = ops.ChangeStats(mean = fixed_c.mean(),std = fixed_c.std())
        
        v_params = torch.as_tensor(v_params, dtype=torch.float32, device=self.device)
        self.v_grating = grating_generator(v_params.unsqueeze(0))
        f_params = torch.as_tensor(f_params, dtype=torch.float32, device=self.device)
        self.f_grating = grating_generator(f_params.unsqueeze(0))

    def forward(self):
        v_c = self.v_grating * self.v_mask * self.mei_mask
        f_c = self.f_grating * (1.0-self.v_mask) * self.mei_mask
        return self.var_c_f(v_c) + self.fixed_c_f(f_c)

@schema
class OptimalTwoPartGrating(dj.Computed):
    definition = """  
    -> TextureLookup
    -> SearchRange
    -> OptimalGaborParameters
    ---
    opt_image:           longblob    # best grating image
    opt_seed:            int         # random seed used to obtain the best two-part grating
    opt_activation:      float       # activation at the best two-part grating image
    v_orientation:       float       # (radians) variable component orientation, counterclockwise rotation to apply (0 is horizontal, pi/2 vertical)
    v_phase:             float       # (radians) variable component phase, angle at which to start the sinusoid
    v_wavelength:        float       # (px/height) variable component wavelength of the sinusoid (1 / spatial frequency)
    f_orientation:       float       # (radians) fixed component orientation, counterclockwise rotation to apply (0 is horizontal, pi/2 vertical)
    f_phase:             float       # (radians) fixed component phase, angle at which to start the sinusoid
    f_wavelength:        float       # (px/height) fixed component wavelength of the sinusoid (1 / spatial frequency)
    """
    
    @property
    def key_source(self):
        return TextureLookup * SearchRange * OptimalGaborParameters & (Texture & (TextureGoodRun & {'score_params': 3}))
    
    def make(self, key):
        from scipy import optimize
        neuron_key = (TextureLookup * SearchRange.proj() * OptimalGaborParameters.proj() & key).fetch1()

        # Get parameters
        mei_params = (MEIParameters & neuron_key).fetch1()
        optgrating_params = (OptimalGaborParameters & neuron_key).fetch1()
        mei, mei_mask = (MEI * MEIMask & neuron_key).fetch1('mei', 'mask')
        texture_info = (TextureLookup & neuron_key).fetch1()
        v_mask = (Texture & texture_info).fetch1('variable_mask')

        # Load predictive model of the neuron
        predictive_model = load_model(neuron_key, True, device='cuda')
        
        # Get search range
        search_range = (SearchRange & key).fetch1()
        lower_limits = [0, 0, *(search_range['lower_{}'.format(p)] for p in
                                ['wavelength'])] * 2
        upper_limits = [np.pi, 2 * np.pi, *(search_range['upper_{}'.format(p)] for p in
                                            ['wavelength'])] * 2

        # Get grating generating function
        generator = GratingGenerator(mei_params['height'], mei_params['width'])
    
        # Write loss function to be optimized
        def neg_model_activation(params):
            params = [np.clip(p, l, u) for p, l, u in zip(params, lower_limits, upper_limits)]
            image = TwoPartGrating(mei, mei_mask, v_mask, params[:3], params[3:], generator)()
            # Compute activation
            with torch.no_grad():
                activation = predictive_model(image).item()
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
        
        # Create best image
        best_params = [np.clip(p, l, u) for p, l, u in zip(best_params, lower_limits, upper_limits)]
        best_image = TwoPartGrating(mei, mei_mask, v_mask, best_params[:3], best_params[3:], generator)()
        best_activation = predictive_model(best_image).item()
        vo, vp, vw, fo, fp, fw = best_params
        
        # Insert
        self.insert1({**key, 'opt_image': best_image.squeeze().cpu().numpy(),
                      'opt_seed': best_seed, 'opt_activation': best_activation,
                      'v_orientation': vo, 'v_phase': vp, 'v_wavelength': vw, 
                      'f_orientation': fo, 'f_phase': fp, 'f_wavelength': fw})


@schema
class ControlTextureParameters(dj.Lookup):
    definition = """
    control_texture_params:      int
    ---
    description:                 varchar(128)
    """
    contents = [[1, 'threshold on dei std with flipped sign, same target_fraction_std'],
                [2, 'fixed subfield mask'],
                [3, 'mask symmetrical to the original variable mask with regard to the MEI center']]
    
def binarize_mask(mask,threshold=0.0):
    mask = np.array(mask)
    mask[mask<threshold] = 0.0
    mask[mask>threshold] = 1.0
    return mask

def find_center(mask,threshold=0):
    from skimage import morphology
    mask = binarize_mask(mask,threshold=threshold)
    hull = morphology.convex_hull_image(mask)
    px_y, px_x = (coords.mean() + 0.5 for coords in np.nonzero(hull))
    return int(np.round(px_y)),int(np.round(px_x))

def find_symmetrical_mask(v_mask,mei_mask,threshold=0.1):
    center = find_center(mei_mask)
    new_mask = np.zeros_like(v_mask)
    for i in range(new_mask.shape[-2]):
        for j in range(new_mask.shape[-1]):
            if v_mask[i,j]:
                new_i,new_j = center[-2]*2-i,center[-1]*2-j
                new_i,new_j = new_i%new_mask.shape[-2],new_j%new_mask.shape[-1]
                new_mask[new_i,new_j] = v_mask[i,j]
    if (new_mask * mei_mask).sum()<(v_mask*mei_mask).sum() * threshold:
        print((new_mask * mei_mask).sum(),(v_mask*mei_mask).sum())
        return None
    else:
        temp = new_mask * mei_mask
        temp *= v_mask.max()/temp.max()
        return temp

@files('static')
@schema
class ControlTexture(dj.Computed):
    definition = """
    -> DEI
    -> DEIThreshold
    -> TextureParameters.TargetFractionStd
    -> ControlTextureParameters
    ---
    fraction_std:                  float               # actual fraction stds
    p:                             float               # fraction of variable mask
    variable_mask:                 longblob            # variable mask to sample from texture
    fixed_part:                    longblob            # fixed part of the DEI
    variable_crops:                blob@static     # variable samples
    full_texture:                  longblob            # full texture
    eval_texture:                  longblob            # texture used for evaluation
    centered_texture:              longblob            # centered texture to sample from
    samples:                       blob@static     # samples from the centered texture
    sample_acts:                   longblob            # sample activations
    sample_avg_div:                float               # sample average diversity
    sample_avg_activation_ratio:   float               # sample average activation ratio
    sample_min_activation_ratio:   float               # sample min activation ratio
    sample_std_activation:         float               # sample activation ratio sd
    dei_avg_div:                   float               # corresponding DEI average diversity
    dei_avg_activation_ratio:      float               # corresponding DEI average activation ratio
    history_save_path:             varchar(255)        # file path for the saved dictionary of texture optimization history
    """
    
    @property
    def key_source(self):
        # rel1 = DEI * DEIThreshold * TextureParameters.TargetFractionStd * ControlTextureParameters & Texture & 'texture_id != -1 and control_texture_params = 2'
        # rel2 = DEI * DEIThreshold * TextureParameters.TargetFractionStd * ControlTextureParameters & Texture & 'control_texture_params != 2' & (TextureGoodRun & 'score_params = 2')
        # return rel1.proj() + rel2.proj()
        return DEI * DEIThreshold * TextureParameters.TargetFractionStd * ControlTextureParameters & 'texture_id != -1'
    
    def make(self, key):
        # Get parameters and original DEIs 
        device = 'cuda'
        texture_parameters = (TextureParameters * TextureParameters.TargetFractionStd & key).fetch1()
        control_texture_params = (ControlTextureParameters & key).fetch1()
        mask = (MEIMask & key).fetch1('mask')
        mei, mei_act = (MEIMask * MEI & (DEIGoodRun & key)).fetch1('mei','activation')
        mei_params = (MEIParameters & key).fetch1()
        mask_params = (MaskParameters & key).fetch1()
        deis, dei_acts, dei_div, dei_avg_activation_ratio = (MEI.proj('mei', mei_act='activation') * DEI & key & DEIGoodRun).fetch1('deis','activations','avg_sim','avg_activation_ratio')
        deis = np.stack(deis)

        # Compute variable masks from original DEIs
        if control_texture_params['control_texture_params'] == 1:
            variable_mask = get_variable_masks(deis,mei,mask_params,values=[texture_parameters['target_fraction_std']],
                                                params={'closing_iters':texture_parameters['closing_iters'],
                                                        'gaussian_sigma':texture_parameters['gaussian_sigma']}, flipped_mask=True)[0][0]
            
        elif control_texture_params['control_texture_params'] == 2:
            variable_mask = dict()
            variable_mask['variable_mask'] = np.clip(mask - (Texture & key).fetch1('variable_mask'), 0, 1) # assign variable_mask as the original fixed mask
            std_mask = None if texture_parameters['full_dei_std'] else mask
            if std_mask is None:
                variable_mask['fraction_std'] = (deis.std(axis=0) * variable_mask['variable_mask']).sum() / deis.std(axis=0).sum()
            else:
                variable_mask['fraction_std'] = (deis.std(axis=0) * variable_mask['variable_mask']).sum() / (deis.std(axis=0) * std_mask).sum()
                
        elif control_texture_params['control_texture_params'] == 3:
            variable_mask = dict()
            variable_mask['variable_mask'] = find_symmetrical_mask((Texture & key).fetch1('variable_mask'), mask) # assign variable_mask as a mask symmetrical to the orignal variable mask with regard to MEI center
            std_mask = None if texture_parameters['full_dei_std'] else mask
            if std_mask is None:
                variable_mask['fraction_std'] = (deis.std(axis=0) * variable_mask['variable_mask']).sum() / deis.std(axis=0).sum()
            else:
                variable_mask['fraction_std'] = (deis.std(axis=0) * variable_mask['variable_mask']).sum() / (deis.std(axis=0) * std_mask).sum()

        # Load predictive model of the neuron
        predictive_model = load_model(key, texture_parameters['use_avg_model'], device=device)
        
        # Calculate texture shape and appropriate padding
        texture_shape, eval_padding,center_padding = cal_texture_minimum_shape(mask=variable_mask['variable_mask'],target_shape =(texture_parameters['target_height'],texture_parameters['target_width']),
                                                                            threshold=texture_parameters['mask_threshold'],padding=texture_parameters['padding'])
        
        image_model = LinearImageModel(mei,variable_mask['variable_mask'],t_f=texture_parameters['texture_f'],t_shape=texture_shape,eval_padding=eval_padding,center_padding=center_padding,
                                        default_n_crops=texture_parameters['n_crops']).to(device)
        
        # Optimize texture 
        history_save_path = os.path.join(self.tuple_dir(key, create=True), "opt_history.pickle")
        image_model, history = train_texture(image_model, predictive_model, mei_act, mask, start_seed=mei_params['mei_seed'], device=device,
                                            history_save_path = history_save_path,
                                            optimizer_args={'optimizer_name':texture_parameters['optimizer_name'],
                                                            'lr':texture_parameters['lr'], 
                                                            'smooth_gradient':texture_parameters['smooth_gradient'],
                                                            'gradient_sigma':texture_parameters['gradient_sigma'],
                                                            'decay_constant':texture_parameters['decay_constant'], 
                                                            'decay_factor':texture_parameters['decay_factor'], 
                                                            'decay_iters': texture_parameters['decay_iters']},
                                            criterion_args={'ref_level':texture_parameters['ref_level'],
                                                            'Lambda':texture_parameters['lambda']},
                                            training_args={'batch_size':texture_parameters['n_crops'],
                                                            'n_iter_per_eval':texture_parameters['n_iter_per_eval'],
                                                            'max_iters':texture_parameters['n_iters']})
        
        # Compute similarity within MEI mask       
        mask_tensor = torch.tensor(mask,dtype=torch.float32,device=device)
        div_reg = utils.Compose([ops.Similarity(mask=mask_tensor,metric='neg_euclidean',combine_op=ops.DoNothing()), ops.ReverseSign()])
        
        # Sample crops from final texture to form texture DEIs, options: random, most diverse, closest to each DEI
        torch.manual_seed(1234)
        texture_dei = Texture.select_texture_dei(image_model,deis,sample_criterion=texture_parameters['sample_criterion'], n_sample=len(deis))
        with torch.no_grad():
            variable_crops = image_model.get_v_c(key='eval',n_crops=len(deis))[0].detach().cpu().numpy().squeeze()
            samples = torch.tensor(texture_dei,dtype=torch.float32,device=device).unsqueeze(1).contiguous()
            sample_avg_div = np.mean(div_reg(samples).cpu().numpy())
            sample_acts = predictive_model(samples).detach().cpu().numpy().squeeze()
            samples = samples.detach().cpu().numpy().squeeze()
            sample_act_ratios = sample_acts/mei_act
            
        result = {**key, 'fraction_std':variable_mask['fraction_std'], 'p':variable_mask['variable_mask'].sum()/mask.sum(),
                    'variable_mask':variable_mask['variable_mask'], 'fixed_part':image_model.fixed_c.detach().cpu().numpy().squeeze(), 'variable_crops':variable_crops,
                    'full_texture':image_model.get_texture('full',True), 'eval_texture':image_model.get_texture('eval',True), 'centered_texture':image_model.get_texture('centered',True),
                    'samples':samples,'sample_acts':sample_acts, 'sample_avg_div':sample_avg_div, 'sample_avg_activation_ratio':sample_act_ratios.mean(),
                    'sample_min_activation_ratio': min(sample_act_ratios), 'sample_std_activation': sample_act_ratios.std(), 
                    'dei_avg_div':dei_div, 'dei_avg_activation_ratio':dei_avg_activation_ratio, 'history_save_path': history_save_path}
        
        self.insert1(result)

        
@schema
class RequestMethod(dj.Manual):
    definition = """
    method_id: int
    ---
    description:  varchar(256)
    """

@schema
class NeuronSetRequest(dj.Manual):
    definition = """
    -> RequestMethod
    -> static_models.Model
    -> Dataset.Unit
    """
def process_p_and_act(ps, acts, divs):
    # Sort ps
    idx = np.argsort(ps)
    ps = ps[idx]
    acts = acts[idx]
    divs = divs[idx]
    
    #Clip p
    ps = np.clip(ps,a_min=0.0,a_max=1.0)

    # Make sure the last one is full texture
    ps[-1] = 1.0
    ps = np.clip(ps, 0, 1)
    
    # Add in MEI
    ps = np.insert(ps, 0, 0.0)
    acts = np.insert(acts, 0, 1.0)
    divs = np.insert(divs, 0, 0.0)
    
    # Remove potential duplicate
    unique_ps,counts = np.unique(ps,return_counts=True)
    for idx in np.where(counts > 1)[0]:
        remove_idx = np.where(ps == unique_ps[idx])[0][:-1]
        ps = np.delete(ps, remove_idx)
        acts = np.delete(acts, remove_idx)
        divs = np.delete(divs, remove_idx)
    if len(unique_ps) < 5:
        return ps, acts, divs, False
    else:
        return ps, acts, divs, True

def fit_spline(xs, ys, mode='linear', clip=False):
    if mode == 'linear':
        f = interp1d(xs, ys, kind='linear')
    elif mode == 'spline':
        f = UnivariateSpline(xs, ys, k=3, s=100)
    else:
        raise ValueError('Mode {} not specified'.format(mode))
    X = np.arange(0, 1.005, 0.005)
    Y = f(X)
    if clip:
        y = np.clip(f(x), a_min=0, a_max=1)
    return X, Y

# Assume that ps and acts have been processed by default
# Mode is either linear or spline
def spline_fit(ps,acts,divs,processed=False,clip=True,mode='linear',valid_solution=True):
    from scipy.interpolate import UnivariateSpline,interp1d
    if not processed:
        ps,acts,divs,valid_solution = process_p_and_act(ps,acts,divs)
    try:
        if mode=='linear':
            f = interp1d(ps, acts,kind='linear')
        elif mode=='spline':
            f = UnivariateSpline(ps,acts,k=2,s=100) # k=3
        else:
            raise ValueError('Mode {} not specified'.format(mode))
        x = np.arange(0, 1.005, 0.005)
        y = f(x)
        if clip:
            y = np.clip(f(x),a_min=0,a_max=1)
        return x,y,valid_solution
    except:
        return np.array([-1.0]),np.array([-1.0]),False
    
def find_p_helper(x,y,target=0.8):
    try:
        return x[(y - target)>0][-1]
    except: 
        return -1.0

def find_p(ps,acts,divs,processed=False,clip=True,mode='linear',target=0.8,valid_solution=True):
    x,y,valid_solution = spline_fit(ps,acts,divs,processed=processed,clip=clip,mode=mode)
    return find_p_helper(x,y,target=target),valid_solution

# Threshold should be in (0,1)
def cal_auc_with_threshold(ps,acts,divs,processed=False,clip=True,mode='linear',threshold=0.8,valid_solution=True):
    from sklearn.metrics import auc
    x,y,valid_solution = spline_fit(ps,acts,divs,processed=processed,clip=clip,mode=mode)
    # Stop at the last y that is above the threshold
    try:
        stopped_idx = np.where(y>threshold)[0][-1]
        return auc(x[:stopped_idx],y[:stopped_idx]),valid_solution
    except:
        return -1.0, valid_solution
    
def randomly_select_population(keys,n_per_group=10,seed=1234,metrics=['auc'], bounds = {'simple':(None, 10),'complex':(90, None),'bipartite':(45,55)}):
    rest = {'mei_params':10, 'mask_params':3, 'mask_stats_params': 2, 'threshold_params': 2, 'texture_params': 20, 'score_params': 3}
    diverse_rest = 'diverse_params in (14, 17)'
    texture_neuron_rel = TextureGoodRun & rest & diverse_rest & keys
    new_keys = texture_neuron_rel.fetch(dj.key,order_by='group_id, neuron_id')
    all_idxs = np.arange(len(new_keys))

    vals = []
    if 'residual' in metrics:
        rel = DEIGoodRun & keys & diverse_rest
        keys = rel.fetch(dj.key, order_by='group_id, neuron_id')
        _, _ , res = compute_diversity(rel, {})
        vals.append(res)

    if 'auc' in metrics:
        aucs = []
        for i in new_keys:
            acts, ps = (Texture & i).fetch('sample_avg_activation_ratio', 'p', order_by='texture_id')
            aucs.append(cal_auc_with_threshold(ps,acts,processed=False,clip=True,mode='spline',threshold=0.8))
        aucs = np.array(aucs)
        key_removed = np.where(aucs[:,1]<1)[0]
        aucs = aucs[:,0]
        if len(key_removed)>0:
            key_removed = np.sort(key_removed)[::-1]
            for i in key_removed:
                del new_keys[i]
            aucs = np.delete(aucs,key_removed)
            all_idxs = np.delete(all_idxs, key_removed)
            if len(vals) > 0:
                for i in range(len(vals)):
                    vals[i] = np.delete(vals[i],key_removed)
        vals.append(aucs)
        print('{} keys out of {} left'.format(len(new_keys),len(keys)))

    cell_types = {}
    for i in bounds.keys():
        cell_types[i] = {}
        lower_bound,upper_bound = bounds[i]
        cond = np.full(len(vals[0]), True)
        for val in vals:
            if lower_bound is not None:
                cond = cond & (val >= np.percentile(val, lower_bound))
            if upper_bound is not None:
                cond = cond & (val < np.percentile(val, upper_bound))
        assert cond.sum() >= n_per_group, 'Not enough neurons to sample'
        np.random.seed(seed)
        selected = np.random.choice(np.where(cond)[0],n_per_group,replace=False)
        cell_types[i]['neuron_list_idx'] = all_idxs[selected]
        cell_types[i]['neuron_list'] = [new_keys[i] for i in selected]
    return cell_types

@schema
class PropertySummary(dj.Computed):
    definition = """
    -> TextureLookup
    ---
    rf_size:      float
    mei_frac_var: float 
    dei_frac_var: float
    dei_act:      float
    dei_div:      float
    dei_residual: float
    partial_act:  float
    partial_div:  float
    full_act:     float
    full_div:     float
    auc:          float
    auc_80:       float
    """
    @property
    def key_source(self):
        return TextureLookup & (Texture * TextureGoodRun & 'score_params = 2').proj()
        
    def make(self, key):
        original_key = key.copy()
        key = (TextureLookup & key).fetch1()
        mei, mask, deis, dei_act = (MEI * MEIMask * DEI * DEIGoodRun & key).fetch1('mei', 'mask', 'deis', 'avg_activation_ratio')
        rf_size = mask.sum()

        # Fraction of variance in MEI within MEI mask
        total_var = (mei.std() ** 2) * mask.size
        mask_var = (ops.get_mask_stats(mei, mask)[-1].item() **2) * rf_size
        mei_frac_var = mask_var / total_var

        # Fraction of variance across DEIs within MEI mask
        dei_var = np.stack(deis).var(0)
        total_var = dei_var.sum((-1, -2))
        mask_var = (dei_var * mask).sum((-1, -2))
        dei_frac_var = mask_var / total_var

        # Pixel DEI diversity
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device='cuda').contiguous()
        deis = torch.as_tensor(np.stack(deis), dtype=torch.float32, device='cuda').contiguous()
        div_reg = utils.Compose([ops.Similarity(mask=mask_tensor, metric='neg_euclidean', combine_op=torch.mean), ops.ReverseSign()])
        dei_div = div_reg(deis).cpu().numpy().squeeze()
        _, _, dei_residual = compute_diversity(key, {})
        
        # Full texture DEI and partial texture DEI activation and diversity
        partial_act, partial_div = (Texture & key).fetch1('sample_avg_activation_ratio', 'sample_avg_div')
        del key['texture_id']
        full_act, full_div = (Texture & key & 'texture_id = -1').fetch1('sample_avg_activation_ratio', 'sample_avg_div')

        # AUC of activation vs p curve
        ps, acts, divs = (Texture & key).fetch('p', 'sample_avg_activation_ratio', 'sample_avg_div', order_by='texture_id')
        ps, acts, divs, _ = process_p_and_act(ps, acts, divs)
        x, y, _ = spline_fit(ps, acts, divs, mode='spline')
        auc_80 = cal_auc_with_threshold(ps, acts, divs, processed=True, clip=True, mode='spline', threshold=0.8)[0]
        auc = cal_auc_with_threshold(ps, acts, divs, processed=True, clip=True, mode='spline', threshold=0.0)[0]
        
        self.insert1({**original_key, 'rf_size': rf_size, 'mei_frac_var': mei_frac_var, 'dei_frac_var':dei_frac_var,
                     'dei_act': dei_act, 'dei_div': dei_div, 'dei_residual': dei_residual.item(), 
                     'partial_act': partial_act, 'partial_div': partial_div, 
                     'full_act': full_act, 'full_div': full_div, 
                     'auc': auc, 'auc_80': auc_80})

@schema
class CellInclusionParameters(dj.Lookup):
    definition = """
    include_params:  int
    ---
    parameters:     longblob   
    """
    contents = [(1, {'mei_frac_var': [2, None], 'dei_frac_var': [2, None], 'rf_size': [25, 75]})]

@schema
class CellGroupParameters(dj.Lookup):
    definition = """
    group_params:        int
    ---
    cell_src_table:      varchar(128)
    base_parameters:     longblob
    group_name:          varchar(32)
    n_subgroups:         int
    n_per_subgroup:      int  
    subgroup_parameters: longblob
    selection_seed:      int
    description:         varchar(256)
    """
    
    contents = [(1, 'NeuronSetRequest & "method_id = 3"',
                    [{'mei_params':10, 'mask_params':3, 'mask_stats_params': 2, 'diverse_params': d, 'threshold_params': 2, 'texture_params': 20} for d in [14, 17]],
                     'extreme_10_center_bin', 6, 50, 
                       dict(simple = dict(dei_residual=[None, 10], full_act=[None, 10], avg_rank=True),
                           complex = dict(dei_residual=[90, None], full_act=[90, None], avg_rank=True),
                           bipartite10_90 = dict(dei_residual=[10, 90], auc=[10, 90]),
                           bipartite20_80 = dict(dei_residual=[20, 80], auc=[20, 80]),
                           bipartite30_70 = dict(dei_residual=[30, 70], auc=[30, 70]),
                           bipartite35_65 = dict(dei_residual=[35, 65], auc=[35, 65])), 1234,
                        'extreme 10 percentile for simple and complex cells, then create bipartite cell bins centered at 50 percentile'),
                
                (2, 'NeuronSetRequest & "method_id = 3"',
                    [{'mei_params':10, 'mask_params':3, 'mask_stats_params': 2, 'diverse_params': d, 'threshold_params': 2, 'texture_params': 20} for d in [14, 17]],
                     'extreme_5_center_bin', 6, 25, 
                      dict(simple = dict(dei_residual=[None, 5], full_act=[None, 5], avg_rank=True),
                          complex = dict(dei_residual=[95, None], full_act=[95, None], avg_rank=True),
                          bipartite10_90 = dict(dei_residual=[10, 90], auc=[10, 90]),
                          bipartite20_80 = dict(dei_residual=[20, 80], auc=[20, 80]),
                          bipartite30_70 = dict(dei_residual=[30, 70], auc=[30, 70]),
                          bipartite35_65 = dict(dei_residual=[35, 65], auc=[35, 65])), 1234, 
                        'extreme 5 percentile for simple and complex cells, then create bipartite cell bins centered at 50 percentile'),
                
                (3, 'NeuronSetRequest & "method_id = 3"',
                    [{'mei_params':10, 'mask_params':3, 'mask_stats_params': 2, 'diverse_params': d, 'threshold_params': 2, 'texture_params': 20} for d in [14, 17]],
                     'extreme_10_sequential_bin', 6, 50, 
                       dict(simple = dict(dei_residual=[None, 10], full_act=[None, 10], avg_rank=True),
                           complex = dict(dei_residual=[90, None], full_act=[90, None], avg_rank=True),
                           bipartite10_50 = dict(dei_residual=[10, 50], auc=[10, 50]),
                           bipartite20_60 = dict(dei_residual=[20, 60], auc=[20, 60]),
                           bipartite30_70 = dict(dei_residual=[30, 70], auc=[30, 70]),
                           bipartite40_80 = dict(dei_residual=[40, 80], auc=[40, 80]),
                           bipartite50_90 = dict(dei_residual=[50, 90], auc=[50, 90])), 1234,
                        'extreme 10 percentile for simple and complex cells, then create sequential bipartite bins'),
                
                (4, 'NeuronSetRequest & "method_id = 3"',
                    [{'mei_params':10, 'mask_params':3, 'mask_stats_params': 2, 'diverse_params': d, 'threshold_params': 2, 'texture_params': 20} for d in [14, 17]],
                     'extreme_5_sequential_bin', 6, 25, 
                       dict(simple = dict(dei_residual=[None, 5], full_act=[None, 5], avg_rank=True),
                           complex = dict(dei_residual=[95, None], full_act=[95, None], avg_rank=True),
                           bipartite10_50 = dict(dei_residual=[10, 50], auc=[10, 50]),
                           bipartite20_60 = dict(dei_residual=[20, 60], auc=[20, 60]),
                           bipartite30_70 = dict(dei_residual=[30, 70], auc=[30, 70]),
                           bipartite40_80 = dict(dei_residual=[40, 80], auc=[40, 80]),
                           bipartite50_90 = dict(dei_residual=[50, 90], auc=[50, 90])), 1234,
                        'extreme 10 percentile for simple and complex cells, then create sequential bipartite bins'),]

@schema
class CellGroupAssignment(dj.Lookup):
    definition = """
    -> CellInclusionParameters
    -> CellGroupParameters
    cell_group_id:      int
    """
    
    class Member(dj.Part):
        definition = """
        -> master
        member_id:     int  
        label:         varchar(32)
        ---
        -> PropertySummary
        """
    
    def get_include_keys(self, rest=dict(include_params=1, group_params=1)):
        # Get inclusion criteria
        cell_pool = eval((CellGroupParameters & rest).fetch1('cell_src_table'))
        base_params = (CellGroupParameters & rest).fetch1('base_parameters')
        include_params = (CellInclusionParameters & rest).fetch1('parameters')
        mei_var, dei_var, rf_sizes = PropertySummary.fetch('mei_frac_var', 'dei_frac_var', 'rf_size')
        mei_thresh = np.percentile(mei_var, include_params['mei_frac_var'][0])
        dei_thresh = np.percentile(dei_var, include_params['dei_frac_var'][0])
        low_rf, up_rf = np.percentile(rf_sizes, include_params['rf_size'][0]), np.percentile(rf_sizes, include_params['rf_size'][1])
        include_rest = 'mei_frac_var >= {} and dei_frac_var >= {} and rf_size >= {} and rf_size < {}'.format(mei_thresh, dei_thresh, low_rf, up_rf)
        include_rel = PropertySummary * TextureLookup & cell_pool & base_params & include_rest
        include_keys = include_rel.fetch(dj.key, order_by='group_id, neuron_id')
        return include_rel, include_keys

    def fill(self, rest=dict(include_params=1, group_params=1)):
        cell_group_id = len(self & rest) + 1
        rest = {**rest, 'cell_group_id': cell_group_id}
        self.insert1(rest)
        include_rel, include_keys = self.get_include_keys(rest)

        # Assign cells to subgroups
        subgroup_params, base_seed, n = (CellGroupParameters & rest).fetch1('subgroup_parameters', 'selection_seed', 'n_per_subgroup')
        metrics = ['dei_residual', 'auc', 'full_act']
        lookup = dict()
        for m in metrics:
            lookup[m] = include_rel.fetch(m, order_by='group_id, neuron_id')

        selected = []
        assigned = []
        for i, (subgroup, params) in enumerate(subgroup_params.items()):
            np.random.seed(base_seed + i)
            # exclude keys that have already been assigned to certain subgroup
            exclude = (include_rel & assigned).proj()
            include_rel = include_rel - exclude
            query = []
            if not 'avg_rank' in params:
                for k, bound in params.items():
                    lb, ub = bound
                    if lb is not None:
                        query.append('{} >= {}'.format(k, np.percentile(lookup[k], lb)))
                    if ub is not None:
                        query.append('{} < {}'.format(k, np.percentile(lookup[k], ub)))
                query = ' and '.join(query)
                # all keys satisfying the query
                keys = (include_rel & query).fetch(dj.key)
                # randomly select from all satisfied keys
                assert (len(keys) >= n), 'Not enough cells to select from!'
                selected.extend([{**rest, **key, 'member_id':i, 'label':subgroup} for i, key in enumerate(np.random.choice(keys, n, replace=False))])
            else:
                rank = []
                for k, bound in params.items():
                    if k in metrics:
                        rank.append(np.argsort(np.argsort(-lookup[k])))
                        lb, ub = bound
                avg_rank = np.argsort(np.argsort(np.stack(rank).mean(0)))
                # all keys satisfying the query
                if lb is not None:
                    idxs = np.where(avg_rank <= np.floor(len(include_keys) * (1 - lb/100)))[0]
                if ub is not None:
                    idxs = np.where(avg_rank > np.floor(len(include_keys) * (1 - ub/100)))[0]
                keys = np.array(include_keys)[idxs]
                # randomly select from all satisfied keys
                assert (len(keys) >= n), 'Not enough cells to select from!'
                selected.extend([{**rest, **key, 'member_id':i, 'label':subgroup} for i, key in enumerate(np.random.choice(keys, n, replace=False))])
            
            # print(subgroup, len(selected))

            # store keys that have already been assigned to simple or complex
            if subgroup in ['simple', 'complex']:
                assigned.extend(keys)
        
        self.Member.insert(selected)

@schema
class MEILookup(dj.Lookup):
    definition = """
    synthesis_id       : varchar(256)                 # unique identifier of the MEI key
    ---
    -> MEIMask
    """
    def fill(self, rest):
        dics = (MEIMask & rest).fetch('KEY')
        for i, dic in enumerate(dics):
            dic['synthesis_id'] = static_utils.key_hash(dic)
            self.insert1(dic, skip_duplicates=True, ignore_extra_fields=True)
            
@schema
class MEIPair(dj.Lookup):
    definition = """ # pairs of MEIs generated from a pair of different models for the same neuron 
    -> MEILookup.proj(n1_synthesis_id='synthesis_id')
    -> MEILookup.proj(n2_synthesis_id='synthesis_id')
    """
    
    def fill(self, pair_type='dynamic_static_chain', group_ids=[]):
        collection = dj.create_virtual_module('pipeline_collection', 'pipeline_collection')
        if pair_type == 'dynamic_static_chain':
            target_rel = collection.CuratedScan() & 'study_name = "dynamic_static_validation" and scan_purpose LIKE "platinum%%" and animal_id != 27342'
            src_rel = collection.CuratedScan() & 'study_name = "dynamic_static_validation" and scan_purpose LIKE "static_image%%" and animal_id != 27342'
            src_rel = (UnitRanking.Unit * Dataset.Unit & src_rel).proj('animal_id', src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
            src_rel = src_rel & MEIMask
            keys = src_rel.fetch(as_dict=True)
            pairs = []
            for key in tqdm(keys):
                n1_id = (MEILookup & key).fetch1('synthesis_id')
                n2_unit_key = (closed_loop.ProximityCellMatch.UnitMatch & target_rel & key).fetch1()
                n2_rel = MEILookup & (Dataset.Unit & n2_unit_key & [{'group_id': gid} for gid in group_ids])
                if n2_rel:
                    n2_id = n2_rel.fetch1('synthesis_id')
                    pairs.append({'n1_synthesis_id': n1_id, 'n2_synthesis_id': n2_id})
            self.insert(pairs, skip_duplicates=True)

@schema
class DynamicStaticMEIEvaluation(dj.Computed):
    definition = """ # compare activation of MEI generated from a pair of different models for the same neuron in a new model ensemble, with all stimuli standardized using parameters in the source model
    -> MEIPair
    -> EvalParameters
    -> static_models.Model
    -> Dataset.Unit
    ---
    sta_mei_act:            float     # evaluation model activation to raw static MEI
    dyn_mei_act:            float     # evaluation model activation to raw dynamic static MEI
    sta_mask_mei_act:       float     # evaluation model activation to masked and standardized static MEI
    dyn_mask_mei_act:       float     # evaluation model activation to masked and standardized dynamic static MEI
    match_dist:             float     # distance between the matched dell pair
    sta_cc_abs:             float     # static model absolute correlation coefficient
    sta_cc_max:             float     # static model maximum possible correlation coefficient
    sta_cc_norm:            float     # static model normalized correlation coefficient
    sta_rf_size:            float     # mask size of static MEI
    dyn_cc_abs:             float     # dynamic model absolute correlation coefficient
    dyn_cc_max:             float     # dynamic model maximum possible correlation coefficient
    dyn_cc_norm:            float     # dynamic model normalized correlation coefficient
    dyn_sta_cc_abs:         float     # dynamic static model absolute correlation coefficient
    dyn_sta_cc_max:         float     # dynamic static model maximum possible correlation coefficient
    dyn_sta_cc_norm:        float     # dynamic static model normalized correlation coefficient
    dyn_sta_rf_size:        float     # mask size of dynamic static MEI
    """
    
    @property
    def key_source(self):
        mei_rel = (MEIPair.proj(synthesis_id = 'n1_synthesis_id') * MEILookup).proj('group_id', 'net_hash', 'data_hash', 'neuron_id')
        neuron_rel = (static_models.Model * Dataset.Unit & 'seed = 101' & 'group_id in (280, 249, 269)').proj()
        return (neuron_rel * mei_rel * EvalParameters).proj(n1_synthesis_id='synthesis_id')
    
    def make(self, key):
        # Assert the same parameters
        p1 = (MEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch('mei_params', 'mask_params', as_dict=True)[0]
        p2 = (MEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch('mei_params', 'mask_params', as_dict=True)[0]
        assert (p1 == p2), 'Two set of DEIs were generated using different parameters!'
        
        # Set up parameters
        eval_params = (EvalParameters & key).fetch1()
        mei_params = (MEIParameters & p1).fetch1()
        target_std, target_mean = float(mei_params['contrast']), float(mei_params['mean'])
        n1_key = (MEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch1()
        n2_key = (MEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch1()
        match_dist = (closed_loop.ProximityCellMatch.UnitMatch \
                    & (Dataset.Unit & n1_key).proj('animal_id', src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id') \
                    & (Dataset.Unit & n2_key).proj()).fetch1('match_distance')
            
        # Get static model neuron information 
        sta_cc_abs, sta_cc_max, sta_cc_norm = (NewEnsembleEval.Unit & n1_key).fetch1('cc_abs', 'cc_max', 'cc_norm')
        sta_rf_size = (MEIMask & n1_key).fetch1('mask').sum()
        
        # Get dynamic model neuron information 
        dv_nns_scan = dj.create_virtual_module('dv_nns_scan', 'dv_nns_v10_scan')
        n2_key = (Dataset.Unit & n2_key).fetch1()
        dyn_cc_abs = (dv_nns_scan.ModelScore.Unit & n2_key).fetch1('model_score')
        if len(dv_nns_scan.Reliability & n2_key) > 0 and len(dv_nns_scan.Reliability.Unit & n2_key) == 0: # set value for units with nan reliability
            dyn_cc_max = -1
        else:
            dyn_cc_max = (dv_nns_scan.Reliability.Unit & n2_key).fetch1('reliability')
        dyn_cc_norm = dyn_cc_abs / dyn_cc_max
        
        # Get dynamic static model neuron information 
        dyn_sta_cc_abs, dyn_sta_cc_max, dyn_sta_cc_norm = (NewEnsembleEval.Unit & n2_key).fetch1('cc_abs', 'cc_max', 'cc_norm')
        dyn_sta_rf_size = (MEIMask & n2_key).fetch1('mask').sum()
 
        # Get evaluation model
        model = load_model(key, mei_params['use_avg_model'], device='cuda', seed_rest='seed in (101, 102, 103, 104)')
        
        # Get static MEI activation 
        mei, mask = (MEI * MEIMask & n1_key).fetch1('mei', 'mask')
        mask_mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
        mei = torch.as_tensor(mei[None, None], dtype=torch.float32, device='cuda')
        with torch.no_grad():
            sta_mei_act = model(mei).item()
            sta_mask_mei_act = model(mask_mei).item()
        
        # Get n2 activation 
        mei, mask = (MEI * MEIMask & n2_key).fetch1('mei', 'mask')
        mask_mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
        mei = torch.as_tensor(mei[None, None], dtype=torch.float32, device='cuda')
        with torch.no_grad():
            dyn_mei_act = model(mei).item()
            dyn_mask_mei_act = model(mask_mei).item()
            
        result = dict(sta_mei_act=sta_mei_act, dyn_mei_act=dyn_mei_act, sta_mask_mei_act=sta_mask_mei_act, dyn_mask_mei_act=dyn_mask_mei_act, 
                     match_dist=match_dist,
                     sta_cc_abs=sta_cc_abs, sta_cc_max=sta_cc_max, sta_cc_norm=sta_cc_norm, sta_rf_size=sta_rf_size,
                     dyn_cc_abs=dyn_cc_abs, dyn_cc_max=dyn_cc_max, dyn_cc_norm=dyn_cc_norm, 
                     dyn_sta_cc_abs=dyn_sta_cc_abs, dyn_sta_cc_max=dyn_sta_cc_max, dyn_sta_cc_norm=dyn_sta_cc_norm, dyn_sta_rf_size=dyn_sta_rf_size)
        self.insert1({**key, **result})

@schema
class DEILookup(dj.Lookup):
    definition = """
    synthesis_id       : varchar(256)                 # unique identifier of the DEI key
    ---
    -> DEI
    """
    def fill(self, rest):
        dics = (DEI & rest).fetch('KEY')
        for i, dic in enumerate(dics):
            dic['synthesis_id'] = static_utils.key_hash(dic)
            self.insert1(dic, skip_duplicates=True, ignore_extra_fields=True)

@schema
class DEIPair(dj.Lookup):
    definition = """ # pairs of DEIs generated from a pair of different models for the same neuron 
    -> DEILookup.proj(n1_synthesis_id='synthesis_id')
    -> DEILookup.proj(n2_synthesis_id='synthesis_id')
    """
    
    def fill(self, pair_type='dynamic_static_chain', group_ids=[]):
        collection = dj.create_virtual_module('pipeline_collection', 'pipeline_collection')
        from staticnet_analyses import closed_loop
        if pair_type == 'dynamic_static_chain':
            target_rel = collection.CuratedScan() & 'study_name = "dynamic_static_validation" and scan_purpose LIKE "platinum%%" and animal_id != 27342'
            src_rel = collection.CuratedScan() & 'study_name = "dynamic_static_validation" and scan_purpose LIKE "static_image%%" and animal_id != 27342'
            src_rel = (UnitRanking.Unit * Dataset.Unit & src_rel).proj('animal_id', src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
            src_rel = src_rel & DEIGoodRun
            keys = src_rel.fetch(as_dict=True)
            pairs = []
            for key in tqdm(keys):
                n2_unit_key = (closed_loop.ProximityCellMatch.UnitMatch & target_rel & key).fetch1()
                if len(DEILookup & key) == 1: 
                    n1_id = (DEILookup & key).fetch1('synthesis_id')
                    n2_rel = DEILookup & (Dataset.Unit & n2_unit_key & [{'group_id': gid} for gid in group_ids])
                    if n2_rel:
                        n2_id = n2_rel.fetch1('synthesis_id')
                        pairs.append({'n1_synthesis_id': n1_id, 'n2_synthesis_id': n2_id})
            self.insert(pairs, skip_duplicates=True)

@schema
class DynamicStaticDEIEvaluation(dj.Computed):
    definition = """ # compare activation of MEI and DEIs generated from a pair of different models for the same neuron in a new model ensemble, with all stimuli standardized using parameters in the source model
    -> DEIPair
    -> EvalParameters
    -> static_models.Model
    -> Dataset.Unit
    ---
    sta_dei_acts:           longblob  # evaluation model activation to static model DEIs
    dyn_dei_acts:           longblob  # evaluation model activation to dynamic static model DEIs
    """
    @property
    def key_source(self):
        dei_rel = (DEIPair.proj(synthesis_id = 'n1_synthesis_id') * DEILookup).proj('group_id', 'net_hash', 'data_hash', 'neuron_id')
        neuron_rel = (static_models.Model * Dataset.Unit & 'seed = 101' & 'group_id in (280, 249, 269)').proj()
        return (neuron_rel * dei_rel * EvalParameters).proj(n1_synthesis_id='synthesis_id')
    
    def make(self, key):
        # Assert the same parameters
        p1 = (DEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch('mei_params', 'mask_params', 'mask_stats_params', 'diverse_params', 'ref_id', as_dict=True)[0]
        p2 = (DEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch('mei_params', 'mask_params', 'mask_stats_params', 'diverse_params', 'ref_id', as_dict=True)[0]
        assert (p1 == p2), 'Two set of DEIs were generated using different parameters!'
        
        # Set up parameters
        eval_params = (EvalParameters & key).fetch1()
        mei_params = (MEIParameters & p1).fetch1()
        target_std, target_mean = float(mei_params['contrast']), float(mei_params['mean'])
        n1_key = (DEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch1()
        n2_key = (DEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch1()

        # Get model of n1 neuron and standardize images based on n1 MEI mask
        model = load_model(key, mei_params['use_avg_model'], device='cuda', seed_rest='seed in (101, 102, 103, 104)')
        
        # Get static DEI activation 
        mask, deis = (MEIMask * DEI & n1_key).fetch1('mask', 'deis')
        deis = torch.as_tensor(ops.standardize_image(np.stack(deis), target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[:, None], dtype=torch.float32, device='cuda')
        with torch.no_grad():
            sta_dei_acts = model(deis).cpu().detach().squeeze().numpy()
        
        # Get dynamic static DEI activation 
        mask, deis = (MEIMask * DEI & n2_key).fetch1('mask', 'deis')
        deis = torch.as_tensor(ops.standardize_image(np.stack(deis), target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[:, None], dtype=torch.float32, device='cuda')
        with torch.no_grad():
            dyn_dei_acts = model(deis).cpu().detach().squeeze().numpy()
        
        self.insert1({**key, 'sta_dei_acts': sta_dei_acts, 'dyn_dei_acts': dyn_dei_acts})


@schema
class ImageSet(dj.Lookup):
    definition = """
    image_set_id:   int
    ---
    image_query:    varchar(256)    # table query to fetch images from stimulus.StaticImage.Image
    description:    varchar(256)  
    """
    contents = [[1, 'imagenet.Album.Single & dict(image_class="imagenet", collection_id=6)', 'imagenet album 6']]

@schema
class DynamicStaticImageEvaluation(dj.Computed):
    definition = """ # evaluate responses from static neuron and dynamic static neuron to standardized images of interest 
    -> MEIPair  
    -> static_models.Model
    -> Dataset.Unit
    -> ImageSet
    -> EvalParameters
    ---
    sta_responses:   blob@static    # an array of static neuron responses to the image set
    dyn_responses:   blob@static    # an array of dynamic static neuron responses to the image set
    """

    @property
    def key_source(self):
        return MEIPair * static_models.Model * Dataset.Unit * ImageSet * EvalParameters & DynamicStaticMEIEvaluation

    def make(self, key):
        # Assert the same parameters
        p1 = (MEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch('mei_params', 'mask_params', as_dict=True)[0]
        p2 = (MEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch('mei_params', 'mask_params', as_dict=True)[0]
        assert (p1 == p2), 'Two set of DEIs were generated using different parameters!'
        
        # Set up parameters
        eval_params = (EvalParameters & key).fetch1()
        mei_params = (MEIParameters & p1).fetch1()
        target_std, target_mean = float(mei_params['contrast']), float(mei_params['mean'])
        n1_key = (MEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch1()
        n2_key = (MEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch1()

        # Get evaluation model
        model = load_model(key, mei_params['use_avg_model'], device='cuda', seed_rest='seed in (101, 102, 103, 104)')

        # Get images 
        image_query = (ImageSet & key).fetch1('image_query')
        images = (stimulus.StaticImage.Image & eval(image_query)).fetch('image')
        print('preprocessing frames')
        import cv2
        images = np.stack([cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) for im in images])

        # Get static responses
        mask = (MEIMask & n1_key).fetch1( 'mask')
        stan_images = np.stack([ops.standardize_image(im, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats']) for im in images])
        stan_images = torch.as_tensor(stan_images, dtype=torch.float32, device='cuda')
        print('getting static responses')
        with torch.no_grad():
            sta_responses = np.array([model(im[None, None]).item() for im in stan_images])

        # Get dynamic static responses
        mask = (MEIMask & n2_key).fetch1( 'mask')
        stan_images = np.stack([ops.standardize_image(im, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats']) for im in images])
        stan_images = torch.as_tensor(stan_images, dtype=torch.float32, device='cuda')
        print('getting dynamic static responses')
        with torch.no_grad():
            dyn_responses = np.array([model(im[None, None]).item() for im in stan_images])

        self.insert1({**key, 'sta_responses': sta_responses, 'dyn_responses': dyn_responses})


# @schema
# class DEICrossEvaluation(dj.Computed):
#     definition = """ # compare cross activation of MEI and DEIs generated from a pair of different models for the same neuron, with all stimuli standardized using parameters in the responsive model
#     -> DEIPair
#     -> EvalParameters
#     ---
#     n1_mei_act:            float     # n1 activation to self MEI
#     n1_dei_acts:           longblob  # an array of n1 activations to self DEIs
#     n1_to_n2_mei_act:      float     # n1 activation to n2 MEI
#     n1_to_n2_dei_acts:     longblob  # an array of n1 activations to n2 DEIs
#     n2_mei_act:            float     # n2 activation to self MEI
#     n2_dei_acts:           longblob  # an array of n2 activations to self DEIs
#     n2_to_n1_mei_act:      float     # n2 activation to n1 MEI
#     n2_to_n1_dei_acts:     longblob  # an array of n2 activations to n1 DEIs
#     """
    
#     def make(self, key):
#         # Assert the same parameters
#         p1 = (DEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch('mei_params', 'mask_params', 'mask_stats_params', 'diverse_params', 'ref_id', as_dict=True)[0]
#         p2 = (DEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch('mei_params', 'mask_params', 'mask_stats_params', 'diverse_params', 'ref_id', as_dict=True)[0]
#         assert (p1 == p2), 'Two set of DEIs were generated using different parameters!'
        
#         # Set up parameters
#         eval_params = (EvalParameters & key).fetch1()
#         mei_params = (MEIParameters & p1).fetch1()
#         target_std, target_mean = float(mei_params['contrast']), float(mei_params['mean'])
#         n1_key = (DEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch1()
#         n2_key = (DEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch1()

#         # Get model of n1 neuron and standardize images based on n1 MEI mask
#         model = load_model(n1_key, mei_params['use_avg_model'], device='cuda')
        
#         # Get n1 activation to self MEI and DEIs
#         mei, mask, deis = (MEI * MEIMask * DEI & n1_key).fetch1('mei', 'mask', 'deis')
#         mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
#         deis = torch.as_tensor(ops.standardize_image(np.stack(deis), target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[:, None], dtype=torch.float32, device='cuda')
#         with torch.no_grad():
#             n1_mei_act = model(mei).item()
#             n1_dei_acts = model(deis).cpu().detach().squeeze().numpy()
        
#         # Get n1 activation to n2 MEI and DEIs
#         mei, deis = (MEI * DEI & n2_key).fetch1('mei', 'deis')
#         mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
#         deis = torch.as_tensor(ops.standardize_image(np.stack(deis), target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[:, None], dtype=torch.float32, device='cuda')
#         with torch.no_grad():
#             n1_to_n2_mei_act = model(mei).item()
#             n1_to_n2_dei_acts = model(deis).cpu().detach().squeeze().numpy()
        
#         # Get model of n2 neuron and standardize images based on n2 MEI mask
#         model = load_model(n2_key, mei_params['use_avg_model'], device='cuda')
        
#         # Get n2 activation to self MEI and DEIs
#         mei, mask, deis = (MEI * MEIMask * DEI & n2_key).fetch1('mei', 'mask', 'deis')
#         mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
#         deis = torch.as_tensor(ops.standardize_image(np.stack(deis), target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[:, None], dtype=torch.float32, device='cuda')
#         with torch.no_grad():
#             n2_mei_act = model(mei).item()
#             n2_dei_acts = model(deis).cpu().detach().squeeze().numpy()
        
#         # Get n2 activation to n1 MEI and DEIs
#         mei, deis = (MEI * DEI & n1_key).fetch1('mei',  'deis')
#         mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
#         deis = torch.as_tensor(ops.standardize_image(np.stack(deis), target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[:, None], dtype=torch.float32, device='cuda')
#         with torch.no_grad():
#             n2_to_n1_mei_act = model(mei).item()
#             n2_to_n1_dei_acts = model(deis).cpu().detach().squeeze().numpy()
        
#         result = dict(n1_mei_act=n1_mei_act, n1_dei_acts=n1_dei_acts, n1_to_n2_mei_act=n1_to_n2_mei_act, n1_to_n2_dei_acts=n1_to_n2_dei_acts,
#                       n2_mei_act=n2_mei_act, n2_dei_acts=n2_dei_acts, n2_to_n1_mei_act=n2_to_n1_mei_act, n2_to_n1_dei_acts=n2_to_n1_dei_acts,)
#         self.insert1({**key, **result})

# @schema
# class DynamicStaticMEIEvaluation(dj.Computed):
#     definition = """ # compare activation of MEI generated from a pair of different models for the same neuron in a new model ensemble, with all stimuli standardized using parameters in the source model
#     -> MEIPair
#     -> EvalParameters
#     -> static_models.Model
#     -> Dataset.Unit
#     ---
#     sta_mei_act:            float     # evaluation model activation to raw static MEI
#     dyn_mei_act:            float     # evaluation model activation to raw dynamic static MEI
#     sta_mask_mei_act:       float     # evaluation model activation to masked and standardized static MEI
#     dyn_mask_mei_act:       float     # evaluation model activation to masked and standardized dynamic static MEI
#     match_dist:             float     # distance between the matched dell pair
#     sta_avg_corr:           float     # static model trial-average test correlation
#     sta_oracle:             float     # static leave-one-out oracle correlation
#     sta_rf_size:            float     # mask size of static MEI
#     dyn_avg_corr:           float     # dynamic model trial-average test correlation
#     dyn_split_oracle:       float     # dynamic split-half oracle correlation
#     dyn_sta_avg_corr:       float     # dynamic static model trial-average test correlation
#     dyn_sta_rf_size:        float     # mask size of dynamic static MEI
#     """
    
#     @property
#     def key_source(self):
#         mei_rel = (MEIPair.proj(synthesis_id = 'n1_synthesis_id') * MEILookup).proj('group_id', 'net_hash', 'data_hash', 'neuron_id')
#         neuron_rel = (static_models.Model * Dataset.Unit & 'seed = 101' & 'group_id in (280, 249, 269)').proj()
#         return (neuron_rel * mei_rel * EvalParameters).proj(n1_synthesis_id='synthesis_id')
    
#     def make(self, key):
#         # Assert the same parameters
#         p1 = (MEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch('mei_params', 'mask_params', as_dict=True)[0]
#         p2 = (MEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch('mei_params', 'mask_params', as_dict=True)[0]
#         assert (p1 == p2), 'Two set of DEIs were generated using different parameters!'
        
#         # Set up parameters
#         eval_params = (EvalParameters & key).fetch1()
#         mei_params = (MEIParameters & p1).fetch1()
#         target_std, target_mean = float(mei_params['contrast']), float(mei_params['mean'])
#         n1_key = (MEILookup & {'synthesis_id': key['n1_synthesis_id']}).fetch1()
#         n2_key = (MEILookup & {'synthesis_id': key['n2_synthesis_id']}).fetch1()
#         match_dist = (closed_loop.ProximityCellMatch.UnitMatch \
#                     & (Dataset.Unit & n1_key).proj('animal_id', src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id') \
#                     & (Dataset.Unit & n2_key).proj()).fetch1('match_distance')
            
#         # Get static model neuron information 
#         sta_oracle, sta_avg_corr = (EnsembleEval.Unit & n1_key).fetch1('oracle_score', 'unit_avg_corr')
#         sta_rf_size = (MEIMask & n1_key).fetch1('mask').sum()
        
#         # Get dynamic model neuron information 
#         dv_nns_scan = dj.create_virtual_module('dv_nns_scan', 'dv_nns_v10_scan')
#         n2_key = (Dataset.Unit & n2_key).fetch1()
#         dyn_avg_corr = (dv_nns_scan.ModelScore.Unit & n2_key).fetch1('model_score')
#         dyn_split_oracle = (dv_nns_scan.SplitScore.Unit & n2_key).fetch1('split_score')
        
#         # Get dynamic static model neuron information 
#         dyn_sta_avg_corr = (EnsembleEval.Unit & n2_key).fetch1('unit_avg_corr')
#         dyn_sta_rf_size = (MEIMask & n2_key).fetch1('mask').sum()
 
#         # Get evaluation model
#         model = load_model(key, mei_params['use_avg_model'], device='cuda', seed_rest='seed in (101, 102, 103, 104)')
        
#         # Get static MEI activation 
#         mei, mask = (MEI * MEIMask & n1_key).fetch1('mei', 'mask')
#         mask_mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
#         mei = torch.as_tensor(mei[None, None], dtype=torch.float32, device='cuda')
#         with torch.no_grad():
#             sta_mei_act = model(mei).item()
#             sta_mask_mei_act = model(mask_mei).item()
        
#         # Get n2 activation 
#         mei, mask = (MEI * MEIMask & n2_key).fetch1('mei', 'mask')
#         mask_mei = torch.as_tensor(ops.standardize_image(mei, target_mean, target_std, mask, eval_params['mask_mean_subtraction'], eval_params['mask_image'], eval_params['match_stats'])[None, None], dtype=torch.float32, device='cuda')
#         mei = torch.as_tensor(mei[None, None], dtype=torch.float32, device='cuda')
#         with torch.no_grad():
#             dyn_mei_act = model(mei).item()
#             dyn_mask_mei_act = model(mask_mei).item()
            
#         result = dict(sta_mei_act=sta_mei_act, dyn_mei_act=dyn_mei_act, sta_mask_mei_act=sta_mask_mei_act, dyn_mask_mei_act=dyn_mask_mei_act, 
#                      match_dist=match_dist,
#                      sta_avg_corr=sta_avg_corr, sta_oracle=sta_oracle, sta_rf_size=sta_rf_size,
#                      dyn_avg_corr=dyn_avg_corr, dyn_split_oracle=dyn_split_oracle,
#                      dyn_sta_avg_corr=dyn_sta_avg_corr, dyn_sta_rf_size=dyn_sta_rf_size)
#         self.insert1({**key, **result})


###################### From DAT #################################3
def find_closest_helper(x,y,target=0.8):
    return x[np.argmin(abs(y - target))]

# Find largest x such that y is above the target
def find_largest_helper(x,y,target=0.8):
    return x[y>=target].max()
    
# By convention let the first p to be full texture by design so fetch would need to fetch in order of texture_id
def process_p_and_act(ps,acts):
    # Remove all the point with p larger than 1 except the first one
    keep_idx = np.where(ps<1)[0]
    keep_idx = np.insert(keep_idx,0,0)
    
    ps = ps[keep_idx]
    acts = acts[keep_idx]
    
    # Add in MEI
    ps = np.insert(ps,0,0.0)
    acts = np.insert(acts,0,1.0)
    
    # Remove potential duplicate based on p for fitting purpose
    unique_ps,counts = np.unique(ps,return_counts=True)
    for idx in np.where(counts>1)[0]:
        remove_idx = np.where(ps==unique_ps[idx])[0][:-1]
        ps = np.delete(ps,remove_idx)
        acts = np.delete(acts,remove_idx)
    
    # Sort based on ps
    idx = np.argsort(ps)
    ps = ps[idx]
    acts = acts[idx]
    
    return ps, acts

# Assume that ps and acts have been processed by default
# Mode is either linear or spline
def spline_fit(ps,acts,processed=True,mode='linear'):
    from scipy.interpolate import UnivariateSpline,interp1d
    if not processed:
        ps,acts = process_p_and_act(ps,acts)
    if mode=='linear':
        f = interp1d(ps, acts,kind='linear')
    elif mode=='spline':
        f = UnivariateSpline(ps,acts,k=2,s=100)
    else:
        raise ValueError('Mode {} not specified'.format(mode))
    x = np.arange(0, 1.01, 0.01)
    return x,np.clip(f(x),a_min=0,a_max=1)

def find_closest_ps(ps,acts,processed=True,mode='spline',target=0.85):
    return find_largest_helper(*spline_fit(ps,acts,processed=processed,mode=mode),target=target)

def cal_auc(ps,acts,processed=True,mode='spline'):
    temp = spline_fit(ps,acts,processed=processed,mode=mode)[1]
    return temp.sum()/len(temp)

def concavity(ps,acts,processed=True,mode='spline'):
    temp = spline_fit(ps,acts,processed=processed,mode=mode)[1]
    return np.diff(np.diff(temp)).mean()

def cal_stat_from_key(key):
    rest1 = {'diverse_params': 14, 'texture_params':13,'score_params':2,'mask_params':3}
    rest2 = {'diverse_params':17,'texture_params':20,'score_params':2,'mask_params':3}
    if key['group_id']<=237:
        rest = rest1
    else: 
        rest = rest2
    key = (DEIGoodRun & key).fetch1('KEY')
    ps,acts = (Texture & key & rest).fetch('p','sample_avg_activation_ratio')
    if len(ps)>10:
        raise ValueError('Too many p')
    auc = cal_auc(ps,acts,processed=False,mode='spline')
    p = find_closest_ps(ps,acts,processed=False,mode='spline')
    return p, auc

def auc_cal(x,y):
    return (((y[:-1] + y[1:])/2) * (x[1:]- x[:-1])).sum()
# ps, acts have not been processed
def texture_summary_helper(ps,acts,stat_type):
    ps,acts = process_p_and_act(ps,acts)
    if stat_type == 'p and act':
        return np.stack([ps,acts])
    else:
        if stat_type == 'auc':
            return auc_cal(ps,acts)
        else:
            ps, acts = spline_fit(ps,acts,processed=True,mode='spline')
            if stat_type == 'fitted_p and fitted_act':
                return np.stack([ps,acts])
            elif stat_type == 'fitted_auc':
                return auc_cal(ps,acts)
            elif stat_type.endswith('p fit'):
                target = float(stat_type.split()[0])
                return find_largest_helper(ps,acts,target=target)
            else:
                raise ValueError('{} is not implemented'.format(stat_type))
            
@schema 
class TextureSummaryParameters(dj.Lookup):
    definition = """
    texture_summary_params: int
    ---
    summary_type: varchar(64)
    """
    contents = [(1,'p and act'),(2,'fitted_p and fitted_act'),(3,'auc'),(4,'fitted_auc'),(5,'0.85 p fit')]
    
@schema
class TextureSummary(dj.Computed):
    definition = """
    -> Texture
    -> TextureSummaryParameters
    ---
    stat: longblob
    """
    # Use texture_id = -1 to restrict
    @property
    def key_source(self):
        return (Texture & {'texture_id' : -1}) * TextureSummaryParameters
    
    def make(self,key):
        summary_type = (TextureSummaryParameters & key).fetch1('summary_type')
        # Remove texture_id for fetching
        new_key = dict(key)
        del new_key['texture_id']
        ps,acts = (Texture & new_key).fetch('p','sample_avg_activation_ratio')
        stat = texture_summary_helper(ps,acts,summary_type)
        self.insert1({**key,'stat':stat})


# Different enough from LinearImageModel to require reformat rather than inherit
class LinearImageModelDoubleVariable(nn.Module):
    # mei, mask, v_mask is numpy array
    def __init__(self,mei,v_mask,f_mask,
                 v_texture,f_texture,
                 default_n_crops=64, device='cuda'):
        
        super(LinearImageModelDoubleVariable,self).__init__()
        self.device = device
        
        self.mei_f = ops.ChangeStats(mean=mei.mean(),std=mei.std())
        self.image_height, self.image_width = mei.shape
        fixed_c = mei * f_mask
        var_c = mei * v_mask
        self.var_c_f = ops.ChangeStats(mean = var_c.mean(),std = var_c.std())
        self.fixed_c_f = ops.ChangeStats(mean = fixed_c.mean(),std = fixed_c.std())
        
        self.register_buffer('v_mask',torch.tensor(np.array(v_mask)[None,None,:,:],dtype=torch.float32))
        self.register_buffer('f_mask',torch.tensor(np.array(f_mask)[None,None,:,:],dtype=torch.float32))
        self.register_buffer('v_t',torch.tensor(np.array(v_texture)[None,None,:,:],dtype=torch.float32))
        self.register_buffer('f_t',torch.tensor(np.array(f_texture)[None,None,:,:],dtype=torch.float32))
        
        self.default_n_crops = default_n_crops

    
    def get_c(self,n_crops=None,key='variable'):
        if n_crops is None:
            n_crops = self.default_n_crops
        cs, crop_x, crop_y = ops.RandomCrop(self.image_height, self.image_width, n_crops)(self.get_texture(key=key,in_numpy_format=False))
        if key == 'variable':
            mask = self.v_mask
            f = self.var_c_f
        else:
            mask = self.f_mask
            f = self.fixed_c_f
            
        cs = f(cs*mask)
        return cs, crop_x, crop_y
    
    @varargin
    def forward(self,n_crops=None,*args):
        return self.mei_f(self.get_c(n_crops,'variable')[0] + self.get_c(n_crops,'fixed')[0])
    
    def get_texture(self,key='variable',in_numpy_format=True):
        if key == 'variable':
            x = self.v_t
        else:
            x = self.f_t
        if in_numpy_format:
            x = x.detach().cpu().numpy().squeeze()
        return x

@files('static')
@schema
# Table to evaluate dual texture parameterization based on bipartite model (score_params=2) and subfield-role-flipped model (control_texture_params=2)
class DualTexture(dj.Computed):
    definition = """
    -> ControlTexture
    --- 
    samples:                       blob@static     # samples from the centered texture
    sample_acts:                   longblob            # sample activations
    sample_avg_div:                float               # sample average diversity
    sample_avg_activation_ratio:   float               # sample average activation ratio
    sample_min_activation_ratio:   float               # sample min activation ratio
    sample_std_activation:         float               # sample activation ratio sd
    """
    @property
    def key_source(self):
        return ControlTexture & {'control_texture_params':2}
    def make(self,key):
        # From Texture table
        device = 'cuda'
        texture_parameters = (TextureParameters * TextureParameters.TargetFractionStd & key).fetch1()
        mask = (MEIMask & key).fetch1('mask')
        mei, mei_act = (MEIMask * MEI & (DEIGoodRun & key)).fetch1('mei','activation')
        mei_params = (MEIParameters & key).fetch1()
        mask_params = (MaskParameters & key).fetch1()
        deis, dei_acts, dei_div, dei_avg_activation_ratio = (MEI.proj('mei', mei_act='activation') * DEI & key & DEIGoodRun).fetch1('deis','activations','avg_sim','avg_activation_ratio')
        deis = np.stack(deis)
        
        # New stuff
        variable_mask,variable_texture = (Texture & key).fetch1('variable_mask','full_texture')
        fixed_mask,fixed_texture = (ControlTexture & key).fetch1('variable_mask','full_texture')
        
        predictive_model = load_model(key, texture_parameters['use_avg_model'], device=device)
        
        image_model = LinearImageModelDoubleVariable(mei=mei,v_mask=variable_mask,f_mask=fixed_mask,
                                                     v_texture=variable_texture,f_texture=fixed_texture,
                                                     default_n_crops=texture_parameters['n_crops'], device=device).to(device)
        # From Texture table
        # Compute similarity within MEI mask       
        mask_tensor = torch.tensor(np.array(mask),dtype=torch.float32,device=device)
        div_reg = utils.Compose([ops.Similarity(mask=mask_tensor,metric='neg_euclidean',combine_op=ops.DoNothing()), ops.ReverseSign()])
            
        torch.manual_seed(1234)
        texture_dei = Texture.select_texture_dei(image_model,deis,sample_criterion=texture_parameters['sample_criterion'], n_sample=len(deis))
        with torch.no_grad():
            samples = torch.tensor(texture_dei,dtype=torch.float32,device=device).unsqueeze(1).contiguous()
            sample_avg_div = np.mean(div_reg(samples).cpu().numpy())
            sample_acts = predictive_model(samples).detach().cpu().numpy().squeeze()
            samples = samples.detach().cpu().numpy().squeeze()
            sample_act_ratios = sample_acts/mei_act
        result = {**key,'samples':samples,'sample_acts':sample_acts, 'sample_avg_div':sample_avg_div, 
                  'sample_avg_activation_ratio':sample_act_ratios.mean(),
                  'sample_min_activation_ratio': min(sample_act_ratios), 'sample_std_activation': sample_act_ratios.std()}
        self.insert1(result)
        
@schema
class NaturalDEIParameters(dj.Lookup):
    definition = """
    natual_dei_params: int
    ---
    ref_level:                   float             # reference activation level relative to the MEI activation that we want DEIs to achieve 
    n_images:                    int               # number of control images needed 
    shuffle_pool:                bool              # whether to shuffle the pool before searching
    early_stop:                  bool              # whether to stop search after have achieved n_images with >= target activation
    image_class:                 varchar(16)       # image_class as stored in stimulus.StaticImage.ImageClass
    crop_h:                      int               # height (in pixels) of crops taken from a search image to augment search dataset
    crop_w:                      int               # width (in pixels) of crops taken from a search image to augment search dataset
    crop_stride:                 int               # stride (in pixels) for taking crops from a search image
    match_stats:                 varchar(16)       # method for matching statistics on the final image, 'mask' or 'ff'
    mask_mean_subtraction:       bool              # whether to subtract mask mean or not during standardization
    description:                 varchar(256)      # description of the search pool
    """""
    contents = [(1, 0.85, 20, True, True, 'imagenet', 36, 64, 8, 'ff', True, 'multiple crops from each of the high resolution imagenet images'),
                (2, 0.8, 20, True, True, 'imagenet', 36, 64, 8, 'ff', True, 'multiple crops from each of the high resolution imagenet images'),
                (3, 0.85, 20, True, True, 'imagenet', 36, 64, 8, 'ff', True, 'multiple crops from each of the high resolution imagenet images'),
                (4, 0.85, 20, True, False, 'imagenet', 36, 64, 8, 'ff', True, 'multiple crops from each of the high resolution imagenet images'),
                (5, 0.85, 20, True, False, 'imagenet', 36, 64, 8, 'ff', True, 'multiple crops from each of the high resolution imagenet images, greedy selection of most diverse images'),
                (6, 0.85, 20, True, False, 'imagenet', 36, 64, 8, 'ff', True, 'multiple crops from each of the high resolution imagenet images, greedy selection of most diverse images, use original mask'),
                ]

@schema
class NaturalDEI(dj.Computed):
    definition = """
    -> static_models.Model
    -> Dataset.Unit
    -> MEIParameters
    -> TightMEIMask
    -> NaturalDEIParameters
    ---
    n_searched_images:   int              # total number of images searched when search stopped
    all_activations:         blob@static      # raw activations of all the images searched 
    n_valid_images:      int              # total number of valid images in the search pool or when search stopped (if early_stop)
    natural_deis:        blob@static      # the final natural DEI images
    dei_activations:         longblob         # raw activations of the final natural DEI images
    src_image_ids:       longblob         # image_ids of the source images of the final natural DEIs
    crop_ids:            longblob         # idxs of the crops from the corresponding source images
    """
    
    @property
    def key_source(self):
        return (TightMEIMask * NaturalDEIParameters).proj()

    def make(self, key):
        print('Searching for group {} neuron {} ...'.format(key['group_id'], key['neuron_id']))
        
        # Get parameters
        mei_params = (MEIParameters & key).fetch1()
        dei_params = (NaturalDEIParameters & key).fetch1()
        if dei_params['natual_dei_params'] == 3 or dei_params['natual_dei_params'] == 6:
            mei_activation, mask = (MEI * MEIMask & key & {'mask_params': 3}).fetch1('activation', 'mask')
        else:
            mei_activation, mask = (MEI * TightMEIMask & key).fetch1('activation', 'mask')
        if dei_params['match_stats'] == 'ff':
            target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
        target_activation = mei_activation * dei_params['ref_level']
        crop_h, crop_w, crop_stride = (NaturalDEIParameters & key).fetch1('crop_h', 'crop_w', 'crop_stride')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get predictive model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device=device, average_batch=False)

        # Process imagenet images and take crops 
        IMAGENET_IMAGE_IDS, IMAGENET_IMAGES = (stimulus.StaticImage.Image & {'image_class': dei_params['image_class']}).fetch('image_id', 'image', order_by='image_id')
        if dei_params['shuffle_pool']:
            random.seed(int(key['group_id'] + key['neuron_id']))
            c = list(zip(IMAGENET_IMAGE_IDS, IMAGENET_IMAGES))
            random.shuffle(c)
            IMAGENET_IMAGE_IDS, IMAGENET_IMAGES = zip(*c)
            IMAGENET_IMAGE_IDS = np.array(IMAGENET_IMAGE_IDS)
        IM_SIZE = (256, 144)
        imagenet_crops = []
        crop_idxs = []
        for image in tqdm(IMAGENET_IMAGES):
            for idx, (h, w) in enumerate(product(np.arange(0, IM_SIZE[1] - crop_h, crop_stride), np.arange(0, IM_SIZE[0] - crop_w, crop_stride))):
                imagenet_crops.append(image[h:h+crop_h, w:w+crop_w])
                crop_idxs.append(idx)        
        n_crops = len(np.arange(0, IM_SIZE[1] - crop_h, crop_stride)) * len(np.arange(0, IM_SIZE[0] - crop_w, crop_stride))
        IMAGENET_IMAGE_IDS = IMAGENET_IMAGE_IDS.repeat(n_crops)
            
        # Pass images through the predictive model in batches and store the valid ones
        n_searched, n_valid = 0, 0
        all_activations, valid_imagenet_ids, valid_crop_ids, valid_crops, valid_activations = [], [], [], [], []
        dei_params['batch_size'] = 100
        for crops, imagenet_ids, crop_ids in tqdm(zip(get_batch(imagenet_crops, dei_params['batch_size']),
                                                      get_batch(IMAGENET_IMAGE_IDS, dei_params['batch_size']),
                                                      get_batch(crop_idxs, dei_params['batch_size']))):
            if not dei_params['early_stop'] or (dei_params['early_stop'] and n_valid < dei_params['n_images']):
                # Mask and standardize image crops
                images = ops.standardize_image(np.stack(crops), target_mean, target_std, mask, dei_params['mask_mean_subtraction'], True, dei_params['match_stats'])
                images = torch.as_tensor(images[:, None], dtype=torch.float32, device='cuda')
                with torch.no_grad():
                    activations = model(images).cpu().detach().squeeze().numpy()
                all_activations.extend(activations)
                
                valid_idxs = np.where(activations >= target_activation)[0]
                valid_imagenet_ids.extend(imagenet_ids[valid_idxs])
                valid_crop_ids.extend(np.array(crop_ids)[valid_idxs])
                if len(valid_idxs) != 1:
                    valid_crops.extend(images[valid_idxs].cpu().detach().squeeze().numpy())
                else:
                    valid_crops.extend(images[valid_idxs].cpu().detach().numpy()[0])
                valid_activations.extend(np.array(activations)[valid_idxs])
                n_searched += len(crops)
                n_valid += len(valid_idxs)
                
        # Insert
        if n_valid >= dei_params['n_images']:
            print('Inserting {} out of {} valid natural DEIs ...'.format(dei_params['n_images'], n_valid))
            
            if dei_params['natual_dei_params'] == 5 or dei_params['natual_dei_params'] == 6: ## greedy selection
                from sklearn.metrics import pairwise_distances
                def maximin_select_points(n_pts, distance_matrix):
                    best_pair = np.unravel_index(distance_matrix.argmax(), distance_matrix.shape)
                    P = set(best_pair)
                    while len(P) < n_pts:
                        vbest = None
                        min_dist = -np.inf
                        for v in range(len(distance_matrix)):
                            if v not in P:
                                current_min = min(distance_matrix[v, list(P)])
                                if current_min > min_dist:
                                    min_dist = current_min
                                    vbest = v
                        P.add(vbest)
                    return np.array(list(P))

                images = np.stack(valid_crops)
                distance_matrix = pairwise_distances(images.reshape(images.shape[0], -1), images.reshape(images.shape[0], -1))
                selected_idxs = maximin_select_points(dei_params['n_images'], distance_matrix)
                
            else: ## random selection
                np.random.seed(int(key['group_id'] + key['neuron_id']))
                selected_idxs = np.random.choice(np.arange(n_valid), dei_params['n_images'], replace=False)
            
            assert n_valid == len(valid_crops) == len(valid_activations) == len(valid_imagenet_ids) == len(valid_crop_ids), 'numbers for the valid set do not agree'
            natural_deis = np.stack(valid_crops)[selected_idxs]
            dei_activations = np.array(valid_activations)[selected_idxs]
            src_image_ids = np.array(valid_imagenet_ids)[selected_idxs]
            crop_ids = np.array(valid_crop_ids)[selected_idxs]
        elif n_valid > 0:
            print('Inserting {} valid natual DEIs ...'.format(n_valid))
            natural_deis = np.stack(valid_crops)
            dei_activations = np.array(valid_activations)
            src_image_ids = np.array(valid_imagenet_ids)
            crop_ids = np.array(valid_crop_ids)
        else:
            print('Inserting {} valid natual DEIs ...'.format(n_valid))
            natural_deis = np.array([])
            dei_activations = np.array([])
            src_image_ids = np.array([])
            crop_ids = np.array([])
        
        self.insert1({**key, 'n_searched_images': n_searched, 'all_activations': np.array(all_activations),
                     'n_valid_images': n_valid, 'natural_deis': natural_deis, 'dei_activations': dei_activations,
                     'src_image_ids': src_image_ids, 'crop_ids': crop_ids})
