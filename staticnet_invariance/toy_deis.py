
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
import random
import os

from staticnet_invariance.toy import ToyNeuron, GroupAssignment, ToyModel, Ensemble
from featurevis import ops
from featurevis import utils
from featurevis.utils import varargin
from staticnet_analyses import base
from staticnet_analyses.base import MaskParameters
from staticnet_experiments import utils as static_utils
from staticnet_invariance import deis_reconfigured as deis_schema
from staticnet_invariance.deis_reconfigured import HingeLoss, MaskStatsParameters, DEIParameters,  DEIThreshold, \
CombinedCriterion, get_variable_masks, LinearImageModel, cal_texture_minimum_shape, train_texture, TextureParameters, TextureScoreParameters
from staticnet_vae import vae

from utils.datajoint.datajoint_utils import files

schema = dj.schema('neurostatic_toy_deis')
dj.config.setdefault('stores', dict())
dj.config['stores'].update({
    'toy': dict(
        protocol='file', 
        location='/dj-stor01/neuro-static')
})

VAE_PATH_3 = '/external/zhiwei/MEI_VAE_centered_mei_params_8.pt'

def prepare_params(key, mei_params, mask_stats_params=None, diverse_params=None):
    model = Ensemble(key, key['member_id'], average_batch=False)
    
    # Get MEI mask and set up shared operations for MaskFixedMEI and DEI generation
    if mask_stats_params is None or not int(mask_stats_params['fixed_mask_std']) & int(mask_stats_params['fixed_mask_mean']):
        postup_op = ops.ChangeStats(float(mei_params['contrast']), float(mei_params['mean']))
    else:
        mask, mask_x, mask_y = (MEIMask & key).fetch1('mask', 'mask_x', 'mask_y')
        mask = torch.as_tensor(mask.copy(), dtype=torch.float32, device='cuda').contiguous()
        postup_op = ops.ChangeMaskStats(mask_stats_params['fixed_mask_std'], mask_stats_params['fixed_mask_mean'], mask)

    # set initial batch seed
    torch.manual_seed(mei_params['mei_seed'])
    if diverse_params is None:  # for MEI generation
        # if mei_params['blur_sigma']:
        #     gradient_f = ops.GaussianBlur(float(mei_params['blur_sigma']))
        # else:
        #     gradient_f = None
        # Set up gradient blurring
        if mei_params['blur_sigma']:
            gradient_f = utils.Compose([ops.GaussianBlur(float(mei_params['blur_sigma'])), ops.MultiplyBy(mei_params['decay_constant'], mei_params['decay_factor'], mei_params['decay_iters'])])
        else:
            gradient_f = ops.MultiplyBy(mei_params['decay_constant'], mei_params['decay_factor'])
        image_shape = (mei_params['num_initializations'], 1, mei_params['height'], mei_params['width'])
        initial_image = torch.randn(image_shape, device='cuda')
        initial_image = postup_op(initial_image)

        return model, gradient_f, postup_op, initial_image
        
    else: # for DEI generation
        mask, mask_x, mask_y = (MEIMask & key).fetch1('mask', 'mask_x', 'mask_y')
        mask = torch.as_tensor(mask.copy(), dtype=torch.float32, device='cuda').contiguous()
        # get_latent = vae.Latent_Embedding(mask_x=mask_x, mask_y=mask_y, vae_path=VAE_PATH_3) 
            
        # Set up embedding, mask, and optimization step size for each space
        if diverse_params['features'] == 'pixels':
            embedding = ops.Identity()  # operation that returns x as is
            similarity_mask = mask if diverse_params['diversity_mask'] == 'mei_mask' else None
        else:
            raise NotImplementedError('{} feature embedding not implemented'.format(diverse_params['features']))
        
        if int(mask_stats_params['fixed_mask_std']) & int(mask_stats_params['fixed_mask_mean']):
            mei, mei_activation = (MaskFixedMEI & key).fetch1('mei', 'activation')
        else:
            mei, mei_activation = (MEI & key).fetch1('mei', 'activation')
        mei = torch.as_tensor(mei[None, None].copy(), dtype=torch.float32, device='cuda').contiguous()

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
            initial_batch = torch.randn(image_shape, device='cuda') * diverse_params['init_noise_scale'] * mask + mei.repeat(diverse_params['num_deis'], 1, 1, 1)
        else:
            initial_batch = torch.randn(image_shape, device='cuda') * diverse_params['init_noise_scale']
        initial_batch = postup_op(initial_batch)
        
        return model, embedding, similarity_mask, mei, mei_activation, mask, mask_x, mask_y, gradient_f, combine_op, postup_op, initial_batch

@schema
class MEIParameters(dj.Lookup):
    definition = """  # parameters to generate MEIs
    
    mei_params:         int
    ---
    mei_seed:           int     # random seed used to create the initialization
    num_initializations: int    # how many random images to optimize in parallel to create the MEI (output is the average)
    height:             int     # height of the MEI
    width:              int     # width of the MEI 
    contrast:           decimal(5, 3) # contrast to use when generating the MEI
    step_size:          float   # step size to use when generating the MEI
    num_iterations:     int     # number of optimization iterations
    blur_sigma:         decimal(5, 3) # sigma used for gradient blur
    fixed_mean:         bool
    mean:               float
    """
    contents = [[1, 0, 1, 36, 64, 0.25, 1, 1000, 1, 1, 0], ]

@schema
class MEI(dj.Computed):
    definition = """
    -> GroupAssignment.Member
    -> ToyModel
    -> MEIParameters
    ---
    mei:                longblob # optimized MEI
    activation:         float   # activation at the MEI 
    """
    @property
    def key_source(self):
        return GroupAssignment.Member * ToyModel * MEIParameters & 'seed = 1009'

    def make(self, key):
        # Get params and models
        mei_params = (MEIParameters & key).fetch1()
        model, gradient_f, postup_op, initial_image = prepare_params(key, mei_params)
        mei, fevals, _ = featurevis.gradient_ascent(model, initial_image,
                                                    post_update=postup_op,
                                                    gradient_f = gradient_f,
                                                    step_size=mei_params['step_size'],
                                                    num_iterations=mei_params[
                                                        'num_iterations'])
        mei = mei.mean(0).squeeze().cpu().numpy()
        activation = fevals[-1]
        # Insert
        self.insert1({**key, 'mei': mei, 'activation': activation})

@schema
class MEIMask(dj.Computed):
    definition = """ # finds a mask for an MEI by thresholding the absolute intensity
    -> MEI
    -> MaskParameters
    ---
    mask:               longblob # produced mask
    mask_x:             float    # (px) centroid of the mask in x; (0, 0) is center of image    
    mask_y:             float    # (px) centroid of the mask in y; (0, 0) is center of image
    mask_mean:          float    # mean of MEI inside the mask   
    mask_std:           float    # standard deviation of MEI inside the mask
    """

    @property
    def key_source(self):
        return MEI * MaskParameters #& 'mask_params = 3'

    def make(self, key):
        from scipy import ndimage
        from skimage import morphology

        # Get mei
        mei = (MEI & key).fetch1('mei')

        # Get params
        params = (MaskParameters & key).fetch1()

        # Normalize and threshold
        norm_mei = (mei - mei.mean()) / mei.std()
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

        # Compute mask centroid
        px_y, px_x = (coords.mean() + 0.5 for coords in np.nonzero(hull))
        mask_y, mask_x = px_y - mask.shape[0] / 2, px_x - mask.shape[1] / 2

        # Compute MEI std inside the mask
        mei_mean = (mei * mask).sum() / mask.sum()
        mei_std = np.sqrt((((mei - mei_mean) ** 2) * mask).sum() / mask.sum())

        # Insert
        self.insert1({**key, 'mask': mask, 'mask_x': mask_x, 'mask_y': mask_y,
                        'mask_mean': mei_mean, 'mask_std': mei_std})

@schema
class DEINew(dj.Computed):
    definition = """ # create DEIs
    -> GroupAssignment.Member
    -> ToyModel
    -> MEIParameters
    -> MEIMask
    -> MaskStatsParameters
    -> DEIParameters.RefLevel
    -> DEIParameters.Weight
    ---
    deis:                  blob@toy             # a set of DEI images
    activations:           longblob                # activations for the DEI set
    fevals:                blob@toy
    regs:                  blob@toy   
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
        return all_keys & {'seed': 1009} 

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
                                                        regularization=partial(deis_schema.DEI.div_regularization, similarity_mask, diverse_params['similarity'], div_weight, combine_op, embedding, mei_copy),
                                                        post_update=postup_op,
                                                        gradient_f=gradient_f)

        # Evaluate the optimized DEIs
        with torch.no_grad():
            activations = single_cell_model(deis).detach().cpu().numpy().squeeze()  # compute activation
            # latents = get_latent(deis).detach().cpu().numpy().squeeze()  # compute VAE latent representation

        get_sim = partial(deis_schema.DEI.div_regularization, similarity_mask, diverse_params['similarity'], 1, ops.DoNothing(), embedding, mei_copy)
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

@schema
class DEINewGoodRun(dj.Computed):
    definition = """
    -> GroupAssignment.Member
    -> ToyModel
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
        all_keys = MEIMask * MaskStatsParameters * DEIParameters.RefLevel * DEIThreshold & DEINew
        return all_keys & {'seed': 1009} 
    
    def make(self, key):
        # pick the good run once all weights have finished populating
        n_weights = len(DEIParameters.Weight & key)
        if len(DEINew & key) == n_weights:
            ref = (DEIParameters.RefLevel & key).fetch1('ref_level')
            weight_id = deis_schema.DEIGoodRun.get_largest_weight_id(key, ref, DEINew)
            if weight_id is not None:
                self.insert1({**key, 'weight_id': weight_id})

@schema
class DEINewGoodRunLookup(dj.Lookup):
    definition = """
    synthesis_id       : varchar(256)                 # unique identifier of the texture synthesis key
    ---
    -> DEINewGoodRun
    """
    def fill(self, rest):
        dics = (DEINewGoodRun & rest).fetch(as_dict=True)
        for i, dic in enumerate(dics):
            dic['synthesis_id'] = static_utils.key_hash(dic)
            self.insert1(dic, skip_duplicates=True, ignore_extra_fields=True)

@files('toy')
@schema
class Texture(dj.Computed):
    definition = """
    -> DEINewGoodRunLookup
    -> TextureParameters.TargetFractionStd
    ---
    fraction_std:                  float               # actual fraction stds
    p:                             float               # fraction of variable mask
    variable_mask:                 longblob            # variable mask to sample from texture
    fixed_part:                    longblob            # fixed part of the DEI
    variable_crops:                blob@toy            # variable samples
    full_texture:                  longblob            # full texture
    eval_texture:                  longblob            # texture used for evaluation
    centered_texture:              longblob            # centered texture to sample from
    samples:                       blob@toy            # samples from the centered texture
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
        return DEINewGoodRunLookup * TextureParameters.TargetFractionStd

    def make(self, key):
        key = (DEINewGoodRun * DEINewGoodRunLookup * TextureParameters.TargetFractionStd & key).fetch1(dj.key)

        # Get parameters and original DEIs 
        device = 'cuda'
        texture_parameters = (TextureParameters * TextureParameters.TargetFractionStd & key).fetch1()
        mask = (MEIMask & key).fetch1('mask')
        mei, mei_act = (MEIMask * MEI & (DEINewGoodRun & key)).fetch1('mei','activation')
        mei_params = (MEIParameters & key).fetch1()
        mask_params = (MaskParameters & key).fetch1()
        deis, dei_acts, dei_div, dei_avg_activation_ratio = (MEI.proj('mei', mei_act='activation') * DEINew * DEINewGoodRun & key).fetch1('deis','activations','avg_sim','avg_activation_ratio')
        deis = np.stack(deis)

        # Compute variable masks from original DEIs
        std_mask = None if texture_parameters['full_dei_std'] else mask
        variable_mask = get_variable_masks(deis,mei,mask_params,values=[texture_parameters['target_fraction_std']],
                                            params={'closing_iters':texture_parameters['closing_iters'],
                                                    'gaussian_sigma':texture_parameters['gaussian_sigma']},
                                            std_mask=std_mask)[0][0]

        # Load predictive model of the neuron
        predictive_model = Ensemble(key, key['member_id'], average_batch=False)
        
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
        texture_dei = deis_schema.Texture.select_texture_dei(image_model,deis,sample_criterion=texture_parameters['sample_criterion'], n_sample=len(deis))
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
        
        self.insert1(result, ignore_extra_fields=True)

@schema
class TextureGoodRun(dj.Computed):
    definition = """
    -> DEINewGoodRunLookup
    -> TextureParameters
    -> TextureScoreParameters
    ---
    texture_id:             int    # Id of the run 
    """

    @property
    def key_source(self):
        return DEINewGoodRunLookup * TextureParameters * TextureScoreParameters
    
    def make(self, key):
        key = (DEINewGoodRun * DEINewGoodRunLookup * TextureParameters * TextureScoreParameters & key).fetch1(dj.key)

        # pick the good run once all weights have finished populating
        n_textures = len(TextureParameters.TargetFractionStd & key)
        if len(Texture & key) == n_textures:
            target_ratio, score_method, include_full = (TextureScoreParameters & key).fetch1('target_ratio', 'score_method', 'include_full')
            if not include_full:
                texture_ids, sample_avg_activation_ratios, sample_avg_divs = (Texture & key & 'texture_id >= 0').fetch('texture_id','sample_avg_activation_ratio','sample_avg_div', order_by='texture_id')
            else:
                texture_ids, sample_avg_activation_ratios, sample_avg_divs = (Texture & key).fetch('texture_id','sample_avg_activation_ratio','sample_avg_div', order_by='texture_id')
            
            # Hard score
            if score_method == 'closest_idx':
                score = deis_schema.TextureGoodRun.hard_score(sample_avg_activation_ratios, target_ratio)
                tid = texture_ids[np.argmax(score)]
            # f score
            elif score_method == 'f_measurement':
                score = deis_schema.TextureGoodRun.f_score(sample_avg_activation_ratios, sample_avg_divs / sample_avg_divs.max())
                tid = texture_ids[np.argmax(score)]
            # max idx that results in activation higher than threshold, if no valid run then pick id 0
            elif score_method == 'max_idx':
                if (sample_avg_activation_ratios > target_ratio).sum() > 0:
                    tid = texture_ids[sample_avg_activation_ratios > target_ratio].max()
                else:
                    tid = 0
            self.insert1({**key, 'texture_id':tid}, ignore_extra_fields=True)


# @schema
# class DEI(dj.Computed):
#     definition = """ # create DEIs
#     -> GroupAssignment.Member
#     -> ToyModel
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
#         model, embedding, similarity_mask, mei, mei_activation, mask, mask_x, mask_y, get_latent, gradient_f, combine_op, postup_op, initial_batch = \
#         prepare_params(key, mei_params, mask_stats_params, diverse_params)

#         # Iterate over weights and reference levels
#         for (weight_id, div_weight), (ref_id, div_ref) in itertools.product(enumerate(diverse_params['weights']),
#                                                                             enumerate(diverse_params['ref_levels'])):
#             print('Optimizing with div_weight=', div_weight, 'div_ref=', div_ref)
#             # Set up optimization
#             if diverse_params['loss_type'] == 'HingeLoss':
#                 activation_obj = utils.Compose([model, HingeLoss(div_ref, mei_activation), ops.ReverseSign()])
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
#                     activation = model(dei[None]).item()  # compute activation
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
#         model, embedding, similarity_mask, mei, mei_activation, mask, mask_x, mask_y, get_latent, gradient_f, combine_op, postup_op, initial_batch = \
#         prepare_params(key, mei_params, mask_stats_params, diverse_params)

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
#     contents = [{'threshold_params': 1, 'dev_from_ref_threshold': 1, 'std_threshold': 1,'min_ratio': 0.85, 'selection_criterion': 'largest_weight'},]

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
# class DEIGoodRunLookup(dj.Lookup):
#     definition = """
#     run_id:  int
#     ---
#     -> DEIEvaluation.DEISet
#     -> DEIThreshold
#     """
#     def fill(self, rest):
#         if len(self) == 0:
#             max_id = 0
#         else:
#             max_id = self.fetch('run_id').max()
#         # exclude the runs that have already been inserted
#         exclude = ((DEIEvaluation.DEISet * DEIGoodRun & rest).proj()) & self
#         dics = ((DEIGoodRun & rest) - exclude).fetch(as_dict=True)
#         for i, dic in enumerate(dics):
#             self.insert1({'run_id': i+1+max_id, **dic})

# @schema
# class TextureSynthesis(dj.Computed):
#     definition = """
#     -> DEIGoodRunLookup
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
#         return DEIGoodRunLookup * TextureParameters
    
#     def make(self,key):
#         self.insert1(key)
#         key = (DEIGoodRunLookup * DEIEvaluation.DEISet * DEIThreshold * TextureParameters & key).fetch1('KEY')
        
#         device = 'cuda'
#         texture_parameters = (TextureParameters & key).fetch1()
#         mask = np.array((MEIMask() & key).fetch1('mask'))
#         mei,mei_act = (MEIMask * MEI & (DEIGoodRun & key)).fetch1('mei','activation')
#         mask_params = (MaskParameters & key).fetch1()
#         deis,dei_acts,dei_div,dei_avg_activation_ratio = (DEI.DEI * DEIEvaluation.DEISet * (DEIGoodRun & key)).fetch('dei','activation','avg_sim','avg_activation_ratio')
#         deis = np.stack(deis)
#         dei_div = -dei_div[0]
#         dei_avg_activation_ratio = dei_avg_activation_ratio[0]

#         variable_masks = get_variable_masks(deis,mei,mask_params,values=texture_parameters['fraction_stds'],
#                                             params={'closing_iters':texture_parameters['closing_iters'],
#                                                     'gaussian_sigma':texture_parameters['gaussian_sigma']})

#         # get model
#         predictive_model = Ensemble(key, key['member_id'], average_batch=False)

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
            
#             self.TextureSynthesis.insert1(result, ignore_extra_fields=True)

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
