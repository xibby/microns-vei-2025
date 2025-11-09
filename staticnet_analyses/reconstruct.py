from staticnet_analyses.base import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr,kurtosis
from collections import Counter
from scipy.spatial.distance import pdist,cdist
import torch.nn as nn
from staticnet_experiments import configs
from neuro_data.static_images import data_schemas
from neuro_data.static_images.configs import DataConfig
from featurevis import ops, models, utils
import cv2
from scipy import ndimage
from staticnet_analyses import multi_mei

meso = dj.create_virtual_module("pipeline_meso", "pipeline_meso")
experiment = dj.create_virtual_module('pipeline_experiment','pipeline_experiment')
stimulus = dj.create_virtual_module('pipeline_stimulus', 'pipeline_stimulus')

schema = dj.schema('neurostatic_reconstruction')
@schema
class UpstateParameters(dj.Lookup):
    definition = """
    upstate_params: int
    ---
    method: varchar(64)     # Either average or kurtosis
    threshold: float    # Threshold after normalize on average
    min_duration: int   # How many traces minimally, assume in 8 Hz
    max_duration: int   # How many traces maximally, assume in 8 Hz
    tolerance: int          # How many not up state allowed
    """
    contents = [(1,'average',0.5,3,5,0),(2,'average',0.3,3,5,0)]
    
@schema
class UpstateTrace(dj.Computed):
    definition = """
    -> UpstateParameters
    -> experiment.Scan
    """
    
    class Trace(dj.Part):
        definition = """
        -> master
        trace_mean_idx: int
        ---
        trace_idxs            : longblob                     # Trace idx
        trace_mean           : longblob
        """
        
    @property
    def key_source(self):
        return experiment.Scan * UpstateParameters
    
    def make(self, key):
        self.insert1(key)
        upstate_params = (UpstateParameters & key).fetch1()
        activities = (meso.Activity.Trace() & key).fetch('trace',order_by='unit_id')
        activities = np.stack(activities)

        normalized_activities = activities/activities.std(axis=1)[:,None]
        if upstate_params['method']=='average':
            x = normalized_activities.mean(axis=0)
        else:
            raise ValueError('Method {} has not been implemented'.format(upstate_params['method']))
            
        ups = x>=upstate_params['threshold']
        idxs = np.where(ups)[0]
        traces = []
        temp = []

        for trace_mean_idx,i in tqdm(enumerate(idxs)):
            if len(temp) == 0 or i <= temp[-1] + 1 + upstate_params['tolerance']:
                temp.append(i)
            else:
                if len(temp)>=upstate_params['min_duration'] and len(temp)<=upstate_params['max_duration']:
                    trace_idxs = np.array(temp)
                    trace_mean = activities[:,trace_idxs].mean(axis=1)
                    traces.append({**key,'trace_mean_idx':trace_mean_idx,'trace_idxs':trace_idxs,'trace_mean':trace_mean})
                temp = [i]
        self.Trace.insert(traces)

@schema
class TargetResponseParameters(dj.Lookup):
    definition = """
    target_response_params: int
    ---
    distance_threshold: float
    """
    contents = [(1,5)]
    
# Table contain the target response after transforming it to the source scan of image net model
@schema
class TargetResponse(dj.Computed):
    definition = """
    -> UpstateTrace
    -> TargetResponseParameters
    source_animal_id     : int                          # id number
    source_session       : smallint                     # session index for the mouse
    source_scan_idx      : smallint                     # number of TIFF stack file
    ---
    response_mask: longblob
    """     
    class Response(dj.Part):
        definition = """
        -> master
        -> UpstateTrace.Trace
        ---
        response          : longblob
        """
    @property
    def key_source(self):
        return UpstateTrace * experiment.Scan.proj('animal_id','session','scan_idx',source_animal_id='animal_id',\
                                                   source_session='session',source_scan_idx='scan_idx') * TargetResponseParameters
    
    def make(self,key):
        
        if not key['animal_id'] == key['source_animal_id']:
            raise Exception('Not the same animal, cannot be matched')
        distance_threshold = (TargetResponseParameters & key).fetch1('distance_threshold')
        trace_mean_idxs,trace_means = (UpstateTrace.Trace & key).fetch('trace_mean_idx','trace_mean')
        trace_means = np.stack(trace_means)
        # Convert random traces to match unit_id from imagenet_scan
        from staticnet_analyses import closed_loop
        target_scan = {'animal_id':key['animal_id'],'session':key['session'],'scan_idx':key['scan_idx']}
        source_scan = {'animal_id':key['source_animal_id'],'session':key['source_session'],'scan_idx':key['source_scan_idx']}
        
        src_unit_ids,unit_ids,match_distances = (closed_loop.ProximityCellMatch.UnitMatch & target_scan).fetch('src_unit_id','unit_id','match_distance',order_by='src_unit_id')
        # Assert src_unit_ids
        if not src_unit_ids[-1] == len(set(src_unit_ids)):
            raise Exception('Source unit id is not completed')


        # Convert to neuron_id from model
        neuron_ids,new_unit_ids = (Dataset.Unit() & source_scan).fetch('neuron_id','unit_id',order_by='neuron_id')
        # Assert src_unit_ids
        if not neuron_ids[-1]+1 == len(set(neuron_ids)):
            raise Exception('Neuron id is not completed')

        transform_idx = np.array((unit_ids-1)[(new_unit_ids-1)],dtype=int)
        std_responses = (Dataset.TrainStats * data_schemas.StaticMultiDatasetGroupAssignment() & source_scan).fetch1('std_responses')
        distance_mask = match_distances[new_unit_ids-1]<=distance_threshold
        self.insert1({**key,'response_mask':distance_mask})
        trace_means = trace_means[:,transform_idx]/std_responses[None,:]
        temp = [{**key,'trace_mean_idx':i,'response':j} for i,j in zip(trace_mean_idxs,trace_means)]
        
        self.Response.insert(temp)

# Setting up loss
# Define the models that will match MSE to either the neural or model responses
# y is the target with the same shape as the model prediction, which ever slot empty is filled with nan
# x is the produced response
class DoNothing():
    def __call__(self, x):
        return x
    
class MatchingLoss(nn.Module):
    def __init__(self, y,mask=None,loss_type='euclidean'):
        super(MatchingLoss, self).__init__()
        if mask is None:
            mask = np.ones(len(y.squeeze())).astype(bool)
        else:
            mask = np.array(mask)
        self.length = mask.sum()
        self.register_buffer('y', torch.tensor(y[mask]))
        self.register_buffer('mask', torch.tensor(mask))
        self.loss_type = loss_type
        
    def forward(self, x):
        x = x.squeeze()[self.mask]
        if self.loss_type == 'euclidean':
            return ((x-self.y)**2).sum()/self.length
        elif self.loss_type == 'correlation':
            temp = torch.stack([x,self.y])
            residuals = temp - temp.mean(-1, keepdim=True)
            numer = torch.mm(residuals, residuals.t())
            ssr = (residuals ** 2).sum(-1)
            sim_matrix = numer / (torch.sqrt(torch.ger(ssr, ssr)) + 1e-9)
            #triu_idx = torch.triu(torch.ones(len(temp), len(temp)), diagonal=1) == 1
            return -sim_matrix[0,1]
            
        else:
            raise ValueError('Loss type {} is not implemented'.format(self.loss_type))

@schema
class ReconstructParameters(dj.Lookup):
    definition = """
    reconstruct_params: int
    ---
    height: int
    width: int
    initial_mode: varchar(64)
    step_size: float
    num_iterations: int
    jitter: int
    blur_sigma: float
    fixed_std: float
    """
    contents = [(1,36,64,'white',10,500,1,0.75,0.2)]
    
@schema
class ReconstructedImageKey(dj.Computed):
    definition = """
    -> ReconstructParameters
    -> TargetResponse
    -> static_models.Model
    """
    class IndividualKey(dj.Part):
        definition = """
        -> master
        -> TargetResponse.Response
        """
    @property
    def key_source(self):
        return TargetResponse * static_models.Model * ReconstructParameters
    def make(self,key):
        self.insert1(key)
        keys = (TargetResponse.Response * static_models.Model * ReconstructParameters & key).fetch('KEY')
        self.IndividualKey.insert(keys)
@schema
class ReconstructedImage(dj.Computed):
    definition = """
    -> ReconstructedImageKey.IndividualKey
    ---
    image: longblob
    predicted_response: longblob
    corr: float   # Correlation between target and invo prediction
    """
    @property
    def key_source(self):
        return ReconstructedImageKey.IndividualKey
    
    def make(self,key):
        device='cuda'
        reconstruct_params = (ReconstructParameters & key).fetch1()
        trace_mask = (TargetResponse & key).fetch1('response_mask')
        trace_mean = (TargetResponse.Response & key).fetch1('response')
        # Load model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']})
        all_keys = (static_models.Model & model_key & 'seed > 0').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        std_responses,data_hash,readout_key = (Dataset.TrainStats & key).fetch1('std_responses','data_hash','readout_key')
        mean_eyepos = torch.tensor([0,0], dtype=torch.float32, device='cuda').unsqueeze(0)
        key['data_hash'] = data_hash
        key['readout_key'] = readout_key
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        dset, (_, _, height, width), train_mean, mean_behavior, mean_eyepos, train_std = train_stats
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos, device=device,average_batch=False)
        
        if reconstruct_params['initial_mode'] == 'white':
            initial_image = torch.zeros(1, 1, reconstruct_params['height'], reconstruct_params['width'], device=device)
        elif reconstruct_params['initial_mode'] == 'random':
            initial_image = torch.randn(1, 1, reconstruct_params['height'], reconstruct_params['width'], device=device)
        else:
            raise ValueError('Initial mode {} not specified'.format(reconstruct_params['initial_mode']))
        
        if reconstruct_params['jitter']>0:
            transform = ops.Jitter(reconstruct_params['jitter'])
        else:
            transform = None
        
        if reconstruct_params['blur_sigma']>0:
            gradient_f = ops.GaussianBlur(reconstruct_params['blur_sigma'])
        else:
            gradient_f = None
            
        if reconstruct_params['fixed_std']>0:
            post_update = ops.ChangeStd(reconstruct_params['fixed_std'])
        else:
            post_update = None
        neural_mse = utils.Compose([model, MatchingLoss(trace_mean,mask = trace_mask,loss_type=reconstruct_params['loss_type']).to(device), ops.MultiplyBy(-1)])

        opt_x, fevals, reg_terms = featurevis.gradient_ascent(neural_mse,initial_image,
                                                              step_size=reconstruct_params['step_size'], 
                                                              num_iterations=reconstruct_params['num_iterations'], 
                                                              print_iters=1000000,transform=transform,
                                                              gradient_f=gradient_f,
                                                              post_update=post_update)
        image = opt_x.detach().cpu().numpy().squeeze()
        with torch.no_grad():
            predicted_response = model(opt_x).detach().cpu().numpy().squeeze()
        corr = pearsonr(predicted_response[trace_mask],trace_mean[trace_mask])[0]
        del key['data_hash']
        del key['readout_key']
        self.insert1({**key,'image':image,'predicted_response':predicted_response,'corr':corr})
        
@schema
class ReconstructedStimuliParameters(dj.Lookup):
    definition = """
    reconstructed_stimuli_params: int
    ---
    height:     int
    width:      int
    target_std: float
    target_mean: float
    upper_bound: float
    lower_bound: float
    """
    contents = [(1,144,256,0.2,0,255.0,0.0)]

@schema
class ReconstructedStimuli(dj.Computed):
    definition = """
    -> ReconstructedImage
    -> ReconstructedStimuliParameters
    ---
    pixel_image: longblob
    frac_clipped: float
    """
    @property
    def key_source(self):
        return ReconstructedImage * ReconstructedStimuliParameters
    def make(self,key):
        reconstructed_stimuli_params = (ReconstructedStimuliParameters & key).fetch1()
        target_shape = np.array((ReconstructedStimuliParameters & key).fetch1('height', 'width'))

        image = (ReconstructedImage & key).fetch1('image')
        # rescale
        image = ndimage.zoom(image.astype(np.float32), target_shape / image.shape, mode='reflect')
        # standardize
        image = ((image-image.mean())*reconstructed_stimuli_params['target_std']/image.std()+1e-9)+reconstructed_stimuli_params['target_mean']
        
        train_mean, train_std = (Dataset.TrainStats & key).fetch1('mean_img_value','std_img_value')
        image = image * train_std + train_mean
        frac_clipped = len(np.where((image.ravel()>reconstructed_stimuli_params['upper_bound']) \
                                    | (image.ravel()<reconstructed_stimuli_params['lower_bound']))[0])/image.size
        image = np.clip(np.round(image),a_max=reconstructed_stimuli_params['upper_bound'],\
                        a_min=reconstructed_stimuli_params['lower_bound']).astype(np.uint8)
        self.insert1({**key,'pixel_image':image,'frac_clipped':frac_clipped})
    
    def fill_stimulus(self, restr, num_desired_images, experiment='xai_recon'):
        """ Fills StaticImage tables needed to run this stimulus.
        """
        # Set some parameters
        image_table = stimulus.StaticImage.XAIRecon
        image_class = 'recon_img'
        image_models = configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace
        
        # Fetch images for a specific diverse set 
        key = (ReconstructedImageKey * ReconstructedStimuliParameters * ReconstructStimuliDiverseSetParameters & restr).fetch1(dj.key)
        params = (ReconstructStimuliDiverseSetParameters & key).fetch1()
        restriction = ReconstructScore().diverse_restrict(n_target=params['n_target'],
                                                          key=key,
                                                          score_threshold=params['score_threshold'])
        images, image_keys = (self & restriction & image_models).fetch('pixel_image', 'KEY', order_by='trace_mean_idx')
        for im_key in image_keys:
            im_key['reconstruct_stimuli_diverse_set_params'] = key['reconstruct_stimuli_diverse_set_params']
        
        # Fill stimulus into stimulus pipeline
        fill_stimulus_function(self, restriction, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys)

def fill_stimulus_function(source_table, restr, num_desired_images, experiment, image_table, image_class, image_models, images, image_keys):
    # Check that there is enough images processed
    num_images = len(source_table & restr & image_models.proj())

    if num_images != num_desired_images:
        msg = ('Number of images in restricted table ({}) different than desired '
            'number of images ({})').format(num_images, num_desired_images)
        raise ValueError(msg)

    # Insert in stimulus.StaticImage (if not already there)
    stimulus.StaticImageClass.insert([{'image_class': image_class}],
                                    skip_duplicates=True)
    stimulus.StaticImage.insert([{'image_class': image_class}], skip_duplicates=True)

    # Insert images
    all_ids = (stimulus.StaticImage.Image & {'image_class': image_class}).fetch(
        'image_id')
    next_id = max(all_ids) + 1 if len(all_ids) > 0 else 0
    for i, (key, img) in enumerate(zip(image_keys, images), start=next_id):
        stimulus.StaticImage.Image.insert1({'image_class': image_class, 'image_id': i,
                                            'image': img})
        image_table.insert1({'image_class': image_class, 'image_id': i,
                        'experiment': experiment, **key,
                        'src_table': '{}.{}'.format(source_table.database, source_table.table_name)}, ignore_extra_fields=True)

# First way is to select based on closest to natural images from Imagenet scan
# Second way is to select based on reconstruction correlation
@schema
class ReconstructScoreParameters(dj.Lookup):
    definition = """
    reconstruct_score_params: int
    ---
    method: varchar(64)
    """
    contents = [(1,'natural_distance'),(2,'model_correlation'),(3,'kurtosis')]

@schema
class ReconstructScore(dj.Computed):
    definition = """
    -> ReconstructScoreParameters
    -> ReconstructedImage
    ---
    score: float
    """
    @property
    def key_source(self):
        return ReconstructScoreParameters * ReconstructedImage
    
    def make(self,key):
        method = (ReconstructScoreParameters & key).fetch1('method')
        if method == 'model_correlation':
            score = (ReconstructedImage & key).fetch1('corr')
        elif method == 'natural_distance':
            std_responses,data_hash,readout_key = (Dataset.TrainStats & key).fetch1('std_responses','data_hash','readout_key')
            key['data_hash'] = data_hash
            key['readout_key'] = readout_key
            train_stats = multi_mei.prepare_data(key, key['readout_key'])
            dset, (_, _, _, _), _, _, _, _ = train_stats
            resps = dset.responses#[:, dset.transforms[1].idx] 
            norm_resps = (resps/ resps.std(0)) 
            
            trace_mask = (TargetResponse & key).fetch1('response_mask')
            trace_mean = (TargetResponse.Response & key).fetch1('response')
            trace_mean = trace_mean[trace_mask]
            norm_resps = norm_resps[:,trace_mask]
            score = np.max([pearsonr(trace_mean,i)[0] for i in norm_resps])
        elif method == 'kurtosis':
            trace_mask = (TargetResponse & key).fetch1('response_mask')
            trace_mean = (TargetResponse.Response & key).fetch1('response')
            score = kurtosis(trace_mean[trace_mask])
        else:
            raise ValueError('Method {} is not specified'.format(method))
        self.insert1({**key,'score':score}, ignore_extra_fields=True)
    
    # Return restrict by a score threshold then diverse select
    def diverse_restrict(self,n_target,key,score_threshold=0.9):
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
        
        # Remove run with score below threshold
        trace_mean_idxs,scores = (self & key).fetch('trace_mean_idx','score')
        
        if len(scores)<=n_target:
            print('Too few responses to threshold')
            return key
        else:
            thresholded_trace_mean_idxs = set([i for i,j in zip(trace_mean_idxs,scores) if j>=score_threshold])
            if len(thresholded_trace_mean_idxs)<=n_target:
                print('Too few responses to diverse select')
                # Order them by scores and return
                temp = [{**key,'trace_mean_idx':trace_mean_idxs[i]} for i in np.argsort(scores)[::-1][:n_target]]
                return temp
            
            else:
                temp = [{**key,'trace_mean_idx':i} for i in thresholded_trace_mean_idxs]

                # Filter them out
                trace_mask = (TargetResponse & key).fetch1('response_mask')
                trace_mean_idxs,trace_mean = (TargetResponse.Response & temp).fetch('trace_mean_idx','response')

                temp = [{**key,'trace_mean_idx':i} for i in trace_mean_idxs]
                trace_mean = np.stack(trace_mean)[:,trace_mask]
                distance_matrix = cdist(trace_mean,trace_mean,metric='correlation')

                # Find the most diverse set of points
                idx = greedy_select_points(n_target,distance_matrix)
                return [temp[i] for i in idx]
    
@schema 
class ReconstructStimuliDiverseSetParameters(dj.Lookup):
    definition = """
    reconstruct_stimuli_diverse_set_params: int
    ---
    reconstruct_score_params: int
    score_threshold: float
    n_target: int
    """
    contents = [(1,1,0.3,150),(2,3,70,150),(3,1,0,150),(4,1,0.3,20),(5,3,70,20)]

# Double saving the set to make it more convenient to get the stimuli based on different diverse select and make diverse select reproducible
@schema 
class ReconstructStimuliDiverseSet(dj.Computed):
    definition = """
    -> ReconstructStimuliDiverseSetParameters
    -> ReconstructedImageKey
    -> ReconstructedStimuliParameters
    """
    class Stimuli(dj.Part):
        definition = """
        -> master
        -> ReconstructedImageKey.IndividualKey
        ---
        stimuli: longblob
        """
    @property
    def key_source(self):
        return ReconstructStimuliDiverseSetParameters * ReconstructedImageKey * ReconstructedStimuliParameters
    
    def make(self,key):
        self.insert1(key)
        params = (ReconstructStimuliDiverseSetParameters & key).fetch1()
        
        restriction = {'reconstruct_score_params': params['reconstruct_score_params'],
                       'reconstruct_params': key['reconstruct_params'],
                       'upstate_params': key['upstate_params']}
        restriction = ReconstructScore().diverse_restrict(n_target=params['n_target'],
                                                          key=restriction,
                                                          score_threshold=params['score_threshold'])
        temp = (ReconstructedStimuli & restriction).fetch(as_dict=True)
        for i in temp:
            i['reconstruct_stimuli_diverse_set_params'] = key['reconstruct_stimuli_diverse_set_params']
            del i['frac_clipped']
            
        self.Stimuli.insert(temp)

@schema
class ReconstructRank(dj.Computed):
    definition = """
    -> ReconstructedImageKey
    -> ReconstructScoreParameters
    """
    class IndividualRank(dj.Part):
        definition = """
        -> master
        -> ReconstructedImage
        ---
        rank: int
        """
    @property
    def key_source(self):
        return ReconstructedImageKey * ReconstructScoreParameters
    
    def make(self,key):
        self.insert1(key)
        keys, scores = (ReconstructScore & key).fetch('KEY','score')
        temp = []
        for rank,index in enumerate(np.argsort(scores)[::-1]):
            temp.append({**keys[index],'rank':rank})
        self.IndividualRank.insert(temp)

@schema
class ReconstructedImageControlParameters(dj.Lookup):
    definition = """
    reconstructed_image_control_params: int
    ---
    shuffle_seed: int
    """
    contents = [(1,99)]
@schema
class ReconstructedImageControl(dj.Computed):
    definition = """
    -> ReconstructedImageControlParameters
    -> ReconstructedImageKey.IndividualKey
    ---
    image: longblob
    predicted_response: longblob
    corr: float # Correlation between target and invo prediction
    """
    @property
    def key_source(self):
        return ReconstructedImageControlParameters * ReconstructedImageKey.IndividualKey
    
    def make(self,key):
        device='cuda'
        reconstruct_params = (ReconstructParameters & key).fetch1()
        trace_mask = (TargetResponse & key).fetch1('response_mask')
        trace_mean = (TargetResponse.Response & key).fetch1('response')
        
        # Only place with difference in control
        np.random.seed((ReconstructedImageControlParameters & key).fetch1('shuffle_seed'))
        trace_mean = np.array(trace_mean)
        np.random.shuffle(trace_mean)
        
        # Load model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']})
        all_keys = (static_models.Model & model_key & 'seed > 0').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        std_responses,data_hash,readout_key = (Dataset.TrainStats & key).fetch1('std_responses','data_hash','readout_key')
        mean_eyepos = torch.tensor([0,0], dtype=torch.float32, device='cuda').unsqueeze(0)
        key['data_hash'] = data_hash
        key['readout_key'] = readout_key
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        dset, (_, _, height, width), train_mean, mean_behavior, mean_eyepos, train_std = train_stats
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos, device=device,average_batch=False)
        
        if reconstruct_params['initial_mode'] == 'white':
            initial_image = torch.zeros(1, 1, reconstruct_params['height'], reconstruct_params['width'], device=device)
        elif reconstruct_params['initial_mode'] == 'random':
            initial_image = torch.randn(1, 1, reconstruct_params['height'], reconstruct_params['width'], device=device)
        else:
            raise ValueError('Initial mode {} not specified'.format(reconstruct_params['initial_mode']))
        
        if reconstruct_params['jitter']>0:
            transform = ops.Jitter(reconstruct_params['jitter'])
        else:
            transform = None
        
        if reconstruct_params['blur_sigma']>0:
            gradient_f = ops.GaussianBlur(reconstruct_params['blur_sigma'])
        else:
            gradient_f = None
            
        if reconstruct_params['fixed_std']>0:
            post_update = ops.ChangeStd(reconstruct_params['fixed_std'])
        else:
            post_update = None
            
        neural_mse = utils.Compose([model, MatchingLoss(trace_mean,mask = trace_mask,loss_type=reconstruct_params['loss_type']).to(device), ops.MultiplyBy(-1)])

        opt_x, fevals, reg_terms = featurevis.gradient_ascent(neural_mse,initial_image,
                                                              step_size=reconstruct_params['step_size'], 
                                                              num_iterations=reconstruct_params['num_iterations'], 
                                                              print_iters=1000000,transform=transform,
                                                              gradient_f=gradient_f,
                                                              post_update=post_update)
        image = opt_x.detach().cpu().numpy().squeeze()
        with torch.no_grad():
            predicted_response = model(opt_x).detach().cpu().numpy().squeeze()
        corr = pearsonr(predicted_response[trace_mask],trace_mean[trace_mask])[0]
        del key['data_hash']
        del key['readout_key']
        self.insert1({**key,'image':image,'predicted_response':predicted_response,'corr':corr})

@schema
class ReconstructedControlStimuli(dj.Computed):
    definition = """
    -> ReconstructedImageControl
    -> ReconstructedStimuliParameters
    ---
    stimuli: longblob
    frac_clipped: float
    """
    @property
    def key_source(self):
        return ReconstructedImageControl * ReconstructedStimuliParameters
    def make(self,key):
        reconstructed_stimuli_params = (ReconstructedStimuliParameters & key).fetch1()
        target_shape = np.array((ReconstructedStimuliParameters & key).fetch1('height', 'width'))

        image = (ReconstructedImageControl & key).fetch1('image')
        # rescale
        image = ndimage.zoom(image.astype(np.float32), target_shape / image.shape, mode='reflect')
        # standardize
        image = ((image-image.mean())*reconstructed_stimuli_params['target_std']\
                 /image.std()+1e-9)+reconstructed_stimuli_params['target_mean']
        
        data_hash,readout_key = (Dataset.TrainStats & key).fetch1('data_hash','readout_key')
        key['data_hash'] = data_hash
        key['readout_key'] = readout_key
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, _, _), train_mean, _, _, train_std = train_stats
        
        image = image * train_std + train_mean
        frac_clipped = len(np.where((image.ravel()>reconstructed_stimuli_params['upper_bound']) \
                                    | (image.ravel()<reconstructed_stimuli_params['lower_bound']))[0])/image.size
        image = np.clip(np.rint(image),a_max=reconstructed_stimuli_params['upper_bound'],a_min=reconstructed_stimuli_params['lower_bound']).astype(np.uint8)
        del key['data_hash']
        del key['readout_key']
        self.insert1({**key,'stimuli':image,'frac_clipped':frac_clipped})