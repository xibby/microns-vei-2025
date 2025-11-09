""" Basic tables to be shared by everybody: UnitRanking, MEI, MEIMask"""
import datajoint as dj
import featurevis
import numpy as np
import torch
import pandas as pd
from featurevis import models
from featurevis import ops, utils
from neuro_data.static_images import data_schemas, stats

from staticnet_experiments import configs
from staticnet_experiments import models as static_models
from staticnet_experiments.utils import compute_predictions, compute_scores
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from itertools import count

fnn = dj.create_virtual_module("fnn", "foundation_fnn")
recording = dj.create_virtual_module("recording", "foundation_recording")
fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')

schema = dj.schema('neurostatic_base')

# similar to multi_mei.TargetDataset
@schema
class Dataset(dj.Computed):
    definition = """ # single dataset (with possibly multiple scans) at a single data configuration
    
    -> data_schemas.StaticMultiDataset
    -> configs.DataConfig   
    """

    class TrainStats(dj.Part):
        definition = """ # some training statistic; these are RAW (i.e., pre any normalization)
        -> master
        readout_key     :varchar(50)
        ---
        num_images:     int     # number of images/trials in the training set
        num_cells:      int     # number of cells in the training set   
        height:         int     # height of training images
        width:          int     # width of training images
        num_channels:   int     # number of channels of training images
        mean_img_value: float   # mean intensity value of all training images
        std_img_value:  float   # std of intensity values across all training images
        mean_responses: longblob # mean responses per cell
        std_responses:  longblob  # std of responses per cell
        mean_eyepos:    blob    # mean eyepos (in x, y) across trials
        std_eyepos:     blob    # std of eyepos (in x, y) across trials
        mean_behavior:  blob    # mean behavior (pupil_dilation, dpupil_dilation/dt, treadmill) across trials
        std_behavior:   blob    # std of behavior (pupil_dilation, dpupil_dilation/dt, treadmill) across trials
        norm_images:    bool    # whether images were z-scored during training
        norm_per_image: bool    # whether the mean for z-scoring images was computed per image rather than using mean_img_value
        norm_responses: bool    # whether responses were divided by std_responses during training (mean_responses is never used)
        norm_eyepos:    bool    # whether eyepos was z-scored during training 
        norm_behavior:  bool    # whether behavior was divided by std_behavior during training (mean_behavior is never used)
        """

    class Unit(dj.Part):
        definition = """ # a neuron from the dataset and some info about it
        -> master
        readout_key:    varchar(50)
        neuron_id:      int     # id of this cell in the dataset (starts at 0), same as index of responses
        ---
        -> data_schemas.StaticMultiDataset.Member
        -> data_schemas.StaticScan.Unit
        brain_area:     varchar(16) # brain area assigned to this cell
        layer:          varchar(16) # cortical layer assigned to this cell
        """

    def make(self, key):
        # Get the training set
        trainsets, _ = configs.DataConfig().load_data(key, tier='train')

        # Insert each scan
        self.insert1(key)
        for readout_key, trainset in trainsets.items():
            # Insert training statistics
            num_images = len(trainset.condition_hashes)
            num_cells = trainset.n_neurons
            _, num_channels, height, width = trainset.img_shape
            stats_source = trainset.stats_source
            mean_intensity = trainset.statistics['images'][stats_source]['mean'][()]
            std_intensity = trainset.statistics['images'][stats_source]['std'][()]
            mean_responses = trainset.statistics['responses'][stats_source]['mean'][()]
            std_responses = trainset.statistics['responses'][stats_source]['std'][()]
            if 'pupil_center' in trainset.data_keys:
                mean_eyepos = trainset.statistics['pupil_center'][stats_source]['mean'][()]
                std_eyepos = trainset.statistics['pupil_center'][stats_source]['std'][()]
            else:
                mean_eyepos, std_eyepos = None, None
            if 'behavior' in trainset.data_keys:
                mean_behavior = trainset.statistics['behavior'][stats_source]['mean'][()]
                std_behavior = trainset.statistics['behavior'][stats_source]['std'][()]
            else:
                mean_behavior, std_behavior = None, None

            normalizer = trainset.transforms[0]
            norm_images = 'images' not in normalizer.exclude
            norm_per_image = normalizer.normalize_per_image
            norm_responses = 'responses' not in normalizer.exclude
            norm_eyepos = 'pupil_center' not in normalizer.exclude
            norm_behavior = 'behavior' not in normalizer.exclude

            self.TrainStats.insert1({**key, 'readout_key': readout_key,
                                     'num_images': num_images, 'num_cells': num_cells,
                                     'num_channels': num_channels, 'height': height,
                                     'width': width, 'mean_img_value': mean_intensity,
                                     'std_img_value': std_intensity,
                                     'mean_responses': mean_responses,
                                     'std_responses': std_responses,
                                     'mean_eyepos': mean_eyepos, 'std_eyepos': std_eyepos,
                                     'mean_behavior': mean_behavior,
                                     'std_behavior': std_behavior,
                                     'norm_images': norm_images,
                                     'norm_per_image': norm_per_image,
                                     'norm_responses': norm_responses,
                                     'norm_eyepos': norm_eyepos,
                                     'norm_behavior': norm_behavior})

            # Insert cells
            scan_key = (data_schemas.StaticMultiDataset.Member() & key & {
                'name': readout_key}).fetch1('KEY')
            for neuron_id, (unit_id, brain_area, layer) in enumerate(zip(
                    trainset.neurons.unit_ids, trainset.neurons.area,
                    trainset.neurons.layer)):
                self.Unit().insert1({**key, **scan_key, 'readout_key': readout_key,
                                     'neuron_id': neuron_id, 'unit_id': unit_id,
                                     'brain_area': brain_area, 'layer': layer})


@schema
class RankingParameters(dj.Lookup):
    definition = """ # parameters used to rank neurons
    ranking_params:     int
    ---
    discard_edge:       int          # (microns) how much of the edge to ignore
    distance_thresh:    int          # (microns) how close do cells have to be to be considered for joining
    correlation_thresh: float        # minimum amount of correlation between cells to be considered the same
    oracle_type:        varchar(16)  # pearson or spearman correlation, or average of the two
    oracle_thresh:      float        # hard threshold on oracle score, use the oracle type indicated in oracle_type, does not threshold if 0
    rank_by:            varchar(45)  # the score used to rank units
    oracle_table:       varchar(45)  # table to fetch oracle scores from 
    ensemble_eval_table:varchar(45)  # table to fetch model performance scores from
	"""
    contents = [(1, 10, 25, 0.4, 'pearson', 0. , 'oracle and test corr', 'stats.Oracle', 'EnsembleEval'),
                (2, 50, 25, 0.4, 'pearson', 0. , 'oracle and test corr', 'stats.Oracle', 'EnsembleEval'),
                (3, 10, 25, 0.4, 'average', 0. , 'oracle and test corr', 'stats.Oracle', 'EnsembleEval'),
                (4, 10, 25, 0.4, 'pearson', 0. , 'avg_corr', 'stats.Oracle', 'EnsembleEval'),
                (5, 10, 25, 0.4, 'pearson', 0.4, 'avg_corr', 'stats.Oracle', 'EnsembleEval'),
                (6, 10, 25, 0.4, 'pearson', 0.5, 'avg_corr', 'stats.OracleMultiTier', 'EnsembleEval'),
                (7, 10, 25, 0.4, 'pearson', 0.4, 'avg_corr', 'stats.OracleMultiTier', 'EnsembleEval'),
                (8, 10, 25, 0.4, 'pearson', 0.5, 'avg_corr', 'stats.Oracle', 'EnsembleEval'),
                (9, 10, 25, 0.4, 'pearson', 0.5, 'avg_corr', 'stats.OracleMultiTier', 'EnsembleEvalTestMEI'),
                (10, 10, 25, 0.4, 'pearson', 0.0, 'oracle and avg_corr', 'stats.Oracle', 'EnsembleEval'),]

# Models in here have already been selected to be the best models for each group id
@schema
class UnitRanking(dj.Computed):
    definition = """  # rank cells (after deduplicating them based on distance and trace correlation)
    
    -> Dataset
    -> static_models.Model
    -> RankingParameters
    """

    @property
    def key_source(self):
        cnn_models = (configs.NetworkConfig & configs.CoreConfig.GaussianLaplace)
        all_keys = Dataset * static_models.Model * RankingParameters
        return all_keys & cnn_models & {'seed': 1009}

    class Unit(dj.Part):
        definition = """ # ranking per unit
        -> master
        -> Dataset.Unit
        ---
        rank:           int     # ranking (starts at 0)
        f:              float   # average of oracle and model correlation ranking
        """

    def make(self, key):
        """
        After excluding cells withing 10 microns of the edge of any field, we order cells
        using the average ranking across oracle and model test correlation and iteratively
        discard those that are less than 20 microns apart and correlate above 0.4.
        """
        fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')
        pipe_name = (fuse.ScanDone & (data_schemas.StaticMultiDatasetGroupAssignment & key)).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe_name, 'pipeline_' + pipe_name)
        self.insert1(key)

        ro_keys = (dj.U('readout_key') & (Dataset.Unit & key)).fetch('readout_key')
        for ro_key in ro_keys:
            dataset_rel = Dataset.Unit & {'readout_key': ro_key}
            # Get some params
            rank_params = (RankingParameters & key).fetch1()
            units_rel = dataset_rel & key
            num_units = len(units_rel)
            oracle_table = eval(rank_params['oracle_table'])

            # Create mask of units to keep after checking for cells closer to the field border
            field_info = pipe.ScanInfo.Field if pipe_name == 'meso' else pipe.ScanInfo
            coords_rel = pipe.ScanSet.UnitInfo * pipe.ScanSet.Unit * field_info
            x_centroid, y_centroid, px_h, px_w, um_h, um_w = (coords_rel & units_rel).fetch(
                'px_x', 'px_y', 'px_height', 'px_width', 'um_height', 'um_width',
                order_by='unit_id')
            px_thresh = rank_params['discard_edge'] * px_h / um_h  # in pixels
            to_keep = np.logical_and(
                np.abs(y_centroid + 0.5 - px_h / 2) < (px_h / 2 - px_thresh),
                np.abs(x_centroid + 0.5 - px_w / 2) < (px_w / 2 - px_thresh))

            print(np.count_nonzero(to_keep), 'out of', len(to_keep), 'units remaining after',
                'removing those close to the edge of their field (<= {} microns).'.format(
                    rank_params['discard_edge']))

            if rank_params['oracle_thresh']:
                if rank_params['oracle_type'] in ('pearson', 'spearman'):
                    alloracle = (oracle_table.UnitScores & units_rel).fetch(rank_params['oracle_type'], order_by='unit_id')
                    to_keep = np.logical_and(to_keep, alloracle > rank_params['oracle_thresh'])
                elif rank_params['oracle_type'] == 'average':
                    oracle_pearson, oracle_spearman = (oracle_table.UnitScores & units_rel).fetch('pearson', 'spearman', order_by='unit_id')
                    to_keep = to_keep & (oracle_pearson > rank_params['oracle_thresh']) & (oracle_spearman > rank_params['oracle_thresh'])

            if 'oracle' in rank_params['rank_by']:
                # Get oracle ranking
                if rank_params['oracle_type'] == 'pearson':
                    oracle_scores = (oracle_table.UnitScores & units_rel).fetch('pearson', order_by='unit_id')[to_keep]
                    oracle_rank = np.argsort(np.argsort(-oracle_scores))
                elif rank_params['oracle_type'] == 'spearman':
                    oracle_scores = (oracle_table.UnitScores & units_rel).fetch('spearman', order_by='unit_id')[to_keep]
                    oracle_rank = np.argsort(np.argsort(-oracle_scores))
                elif rank_params['oracle_type'] == 'average':
                    pearson = (oracle_table.UnitScores & units_rel).fetch('pearson', order_by='unit_id')[to_keep]
                    spearman = (oracle_table.UnitScores & units_rel).fetch('spearman', order_by='unit_id')[to_keep]
                    p_rank = np.argsort(np.argsort(-pearson))
                    s_rank = np.argsort(np.argsort(-spearman))
                    oracle_rank = np.argsort(np.argsort((p_rank + s_rank)/2))

                if rank_params['rank_by'] == 'oracle and test corr':
                    # Get model ranking
                    model_key = {'group_id': key['group_id'], 'net_hash': key['net_hash']}  # ignore seed
                    model_corrs = (static_models.Model.UnitTestScores & model_key & units_rel).fetch(
                        'pearson', order_by='unit_id, seed')
                    model_corrs = model_corrs.reshape(num_units, -1).mean(-1)
                    model_corrs = model_corrs[to_keep]
                    model_rank = np.argsort(np.argsort(-model_corrs))
                elif rank_params['rank_by'] == 'oracle and avg_corr':
                    avg_corr = (eval(rank_params['ensemble_eval_table']).Unit * units_rel & key).fetch('unit_avg_corr', order_by='unit_id')[to_keep]
                    model_rank = np.argsort(np.argsort(-avg_corr))

                # Create average ranking
                fs = (oracle_rank + model_rank) / 2
                avg_rank = np.argsort(np.argsort(fs))
            
            elif rank_params['rank_by'] == 'avg_corr':
                avg_corr = (eval(rank_params['ensemble_eval_table']).Unit * units_rel & key).fetch('unit_avg_corr', order_by='unit_id')[to_keep]
                avg_rank = np.argsort(np.argsort(-avg_corr))
                fs = -avg_corr

            # Filter cells that are closer than 20 microns and correlate above 0.4
            ## Get distances
            print('Computing distances')
            xs, ys, zs = (pipe.StackCoordinates.UnitInfo & 'stack_session = session' & units_rel).fetch('stack_x',
                                                                                                        'stack_y',
                                                                                                        'stack_z',
                                                                                                        order_by='unit_id, stack_session, stack_idx')
            assert len(xs) == len(ys) == len(zs) == num_units, 'Number of units in meso.StackCoordinates and Dataset.Unit do not match!'
            xs = xs.reshape(num_units, -1)[to_keep]
            ys = ys.reshape(num_units, -1)[to_keep]
            zs = zs.reshape(num_units, -1)[to_keep]
            distances = np.sqrt((xs[None, :, :] - xs[:, None, :]) ** 2 +
                                (ys[None, :, :] - ys[:, None, :]) ** 2 +
                                (zs[None, :, :] - zs[:, None, :]) ** 2)
            distances = distances.mean(-1)  # num_units x num_units

            ## Get traces
            print('Fetching traces')
            traces = (pipe.Fluorescence.Trace * pipe.ScanSet.Unit & units_rel).fetch('trace',
                                                                                    order_by='unit_id')
            traces = np.stack(traces[to_keep])  # num_units x num_frames
            norm_traces = ((traces - traces.mean(-1, keepdims=True)) /
                        (traces.std(-1, keepdims=True) + 1e-9))

            ## Reorder matrices
            distances[avg_rank] = distances.copy()  # copy needed for the inplace sorting
            distances[:, avg_rank] = distances.copy()
            norm_traces[avg_rank] = norm_traces.copy()

            ## Iterate from best to worst deleting duplicates
            print('Discarding duplicates')
            too_close = distances < rank_params['distance_thresh']
            too_close[np.tril_indices_from(too_close, k=1)] = False  # so only lower ranking units are affected
            to_keep2 = np.ones(len(distances), dtype=bool)
            for i, (to_drop, trace) in enumerate(zip(too_close, norm_traces)):
                if to_keep2[i] and np.any(to_drop):
                    corrs = (norm_traces[to_drop] * trace).mean(-1)
                    to_drop[to_drop] = corrs > rank_params['correlation_thresh']  # only drop those that are correlated
                    to_keep2[to_drop] = False
            to_keep2 = to_keep2[avg_rank]  # return it to original order
            print(np.count_nonzero(to_keep2), 'out of', len(to_keep2), 'remaining after',
                'duplicate check.')

            ## Update rank to ignore the newly dropped cells
            to_keep[to_keep] = to_keep2
            fs = fs[to_keep2]
            final_rank = np.argsort(np.argsort(fs))

            # Insert
            print('Inserting')
            unit_keys = units_rel.fetch('KEY', order_by='unit_id')
            unit_keys = [uk for uk, keep in zip(unit_keys, to_keep) if keep]
            for unit_key, f, rank in zip(unit_keys, fs, final_rank):
                self.Unit.insert1({**key, **unit_key, 'f': f, 'rank': rank})

@schema
class FoundationUnitRanking(dj.Computed):
    definition = """
    -> fnn.Model
    -> RankingParameters
    """
    
    @property
    def key_source(self):
        return fnn.Model * RankingParameters & {"network_id": "89e60439a60a69b534673310f5948108", "instance_id": "eeb3d45ad2aae80a10153cd18e365b26"}
    
    class Unit(dj.Part):
        definition = """
        -> master
        -> fuse.ScanSet.Unit
        ---
        rank         : int   # ranking (starts at 0)
        f            : float # metric value used for ranking
        """
       
    def make(self, key):
        pipe_name = (fuse.ScanDone & (fnn.Data.VisualScan & key)).fetch1('pipe')
        pipe = dj.create_virtual_module(pipe_name, 'pipeline_' + pipe_name)
        self.insert1(key)

        # Get some params
        rank_params = (RankingParameters & key).fetch1()
        # TODO: replace hard-coded params
        corr_params =  {'trial_filterset_id': 'd00bbb175d63398818ca652391c18856',
                        'videoset_id': 'acb04adeca72c460a2c5849c22630b14',
                        'correlation_id': '9d7c0855cf673c141729b0af5844857b',
                        'burnin': 10,
                        'perspective': 1,
                        'modulation': 1}
        units_rel = pd.DataFrame(
                                ((fnn.VisualRecordingCorrelation & key & corr_params) * 
                                (recording.Trace.ScanUnit * recording.ScanUnitOrder * (fnn.Data.VisualScan & key)
                                ).proj('unit_id', unit='trace_order')
                                ).fetch(order_by='unit_id')
                                )
        num_units = len(units_rel)
        oracle_table = eval(rank_params['oracle_table'])

        # Create mask of units to keep after checking for cells closer to the field border
        field_info = pipe.ScanInfo.Field if pipe_name == 'meso' else pipe.ScanInfo
        coords_rel = pipe.ScanSet.UnitInfo * pipe.ScanSet.Unit * field_info
        x_centroid, y_centroid, px_h, px_w, um_h, um_w = (coords_rel & units_rel).fetch(
            'px_x', 'px_y', 'px_height', 'px_width', 'um_height', 'um_width',
            order_by='unit_id')
        px_thresh = rank_params['discard_edge'] * px_h / um_h  # in pixels
        to_keep = np.logical_and(
            np.abs(y_centroid + 0.5 - px_h / 2) < (px_h / 2 - px_thresh),
            np.abs(x_centroid + 0.5 - px_w / 2) < (px_w / 2 - px_thresh))

        print(np.count_nonzero(to_keep), 'out of', len(to_keep), 'units remaining after',
            'removing those close to the edge of their field (<= {} microns).'.format(
                rank_params['discard_edge']))

        if rank_params['rank_by'] == 'avg_corr':
            avg_corr = units_rel.correlation.to_numpy()[to_keep]
            avg_rank = np.argsort(np.argsort(-avg_corr))
            fs = -avg_corr
        else:
            raise NotImplementedError('ranking by {} has not been implemented yet!'.format(rank_params['rank_by'])) 

        # Filter cells that are closer than 20 microns and correlate above 0.4
        ## Get distances
        print('Computing distances')
        xs, ys, zs = (pipe.StackCoordinates.UnitInfo & 'stack_session = session' & units_rel).fetch('stack_x',
                                                                                                    'stack_y',
                                                                                                    'stack_z',
                                                                                                    order_by='unit_id, stack_session, stack_idx')
        assert len(xs) == len(ys) == len(zs) == num_units, 'Number of units in meso.StackCoordinates and Dataset.Unit do not match!'
        xs = xs.reshape(num_units, -1)[to_keep]
        ys = ys.reshape(num_units, -1)[to_keep]
        zs = zs.reshape(num_units, -1)[to_keep]
        distances = np.sqrt((xs[None, :, :] - xs[:, None, :]) ** 2 +
                            (ys[None, :, :] - ys[:, None, :]) ** 2 +
                            (zs[None, :, :] - zs[:, None, :]) ** 2)
        distances = distances.mean(-1)  # num_units x num_units

        ## Get traces
        print('Fetching traces')
        traces = (pipe.Fluorescence.Trace * pipe.ScanSet.Unit & units_rel).fetch('trace',
                                                                                order_by='unit_id')
        traces = np.stack(traces[to_keep])  # num_units x num_frames
        norm_traces = ((traces - traces.mean(-1, keepdims=True)) /
                    (traces.std(-1, keepdims=True) + 1e-9))
        
        ## Reorder matrices
        distances[avg_rank] = distances.copy()  # copy needed for the inplace sorting
        distances[:, avg_rank] = distances.copy()
        norm_traces[avg_rank] = norm_traces.copy()

        ## Iterate from best to worst deleting duplicates
        print('Discarding duplicates')
        too_close = distances < rank_params['distance_thresh']
        too_close[np.tril_indices_from(too_close, k=1)] = False  # so only lower ranking units are affected
        to_keep2 = np.ones(len(distances), dtype=bool)
        for i, (to_drop, trace) in enumerate(zip(too_close, norm_traces)):
            if to_keep2[i] and np.any(to_drop):
                corrs = (norm_traces[to_drop] * trace).mean(-1)
                to_drop[to_drop] = corrs > rank_params['correlation_thresh']  # only drop those that are correlated
                to_keep2[to_drop] = False
        to_keep2 = to_keep2[avg_rank]  # return it to original order
        print(np.count_nonzero(to_keep2), 'out of', len(to_keep2), 'remaining after',
            'duplicate check.')

        ## Update rank to ignore the newly dropped cells
        to_keep[to_keep] = to_keep2
        fs = fs[to_keep2]
        final_rank = np.argsort(np.argsort(fs))

        # Insert
        print('Inserting')
        unit_keys = units_rel[['animal_id', 'session', 'scan_idx', 'pipe_version', 'segmentation_method', 'unit_id']].to_dict('records')
        unit_keys = [uk for uk, keep in zip(unit_keys, to_keep) if keep]
        for unit_key, f, rank in zip(unit_keys, fs, final_rank):
            self.Unit.insert1({**key, **unit_key, 'f': f, 'rank': rank})


@schema
class MEIParameters(dj.Lookup):
    definition = """  # parameters to generate MEIs
    
    mei_params:         int
    ---
    mei_seed:           int     # random seed used to create the initialization
    use_avg_model:      bool    # use the average model (across different training seeds) to create the MEI
    num_initializations: int    # how many random images to optimize in parallel to create the MEI (output is the average)
    height:             int     # height of the MEI
    width:              int     # width of the MEI 
    contrast:           decimal(5, 3) # contrast to use when generating the MEI
    fixed_mean:         bool    # whether to fix mean when generating the MEI
    mean:               decimal(5, 3) # mean to use when generating the MEI
    step_size:          float   # step size to use when generating the MEI
    num_iterations:     int     # number of optimization iterations
    blur_sigma:         decimal(5, 3) # sigma used for gradient blur
    """
    contents = [( 1, 1234, 1,  1, 36, 64, 0.10, 0, 0.0,  1., 1000, 0.0),
                ( 2, 1234, 1,  1, 36, 64, 0.10, 0, 0.0, 10., 1000, 0.0),
                ( 3, 1234, 1,  1, 36, 64, 0.10, 0, 0.0,  1., 1000, 1.0),
                ( 4, 1234, 1,  1, 36, 64, 0.10, 0, 0.0, 10., 1000, 1.0),
                ( 5, 1234, 1,  1, 36, 64, 0.10, 0, 0.0,  1., 1000, 1.0),
                ( 6, 1234, 1,  1, 36, 64, 0.25, 0, 0.0,  1., 1000, 1.0),
                ( 7, 1234, 1,  1, 36, 64, 0.30, 0, 0.0,  1., 1000, 1.0),
                ( 8, 1234, 1,  1, 36, 64, 0.25, 1, 0.0, 10., 1000, 1.0),
                ( 9, 1234, 1, 20, 36, 64, 0.25, 1, 0.0, 10., 1000, 1.0),
                (10, 1234, 1,  1, 36, 64, 0.25, 1, 0.0,  1., 1000, 1.0)]
    

@schema
class MEI(dj.Computed):
    definition = """ # a single MEI
    
    -> static_models.Model
    -> Dataset.Unit
    -> MEIParameters
    ---
    mei:                longblob # optimized MEI
    activation:         float   # activation at the MEI 
    """
    @property
    def key_source(self):
        all_keys = static_models.Model * Dataset.Unit * MEIParameters
        return all_keys #& [{'seed': 1009}, {'seed': 101}]

    def make(self, key):
        # Get params
        mei_params = (MEIParameters & key).fetch1()

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
        mean_eyepos = ([0, 0] if (Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (Dataset.TrainStats & key).fetch1('mean_eyepos'))

        # Create model ensemble
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                                   device='cuda').unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Create a random initial image (at desired contrast)
        torch.manual_seed(mei_params['mei_seed'])
        image_shape = (mei_params['num_initializations'], 1, mei_params['height'],
                       mei_params['width'])
        initial_image = torch.randn(image_shape, device='cuda')
        if not mei_params['fixed_mean']:
            postup_op = ops.ChangeStd(float(mei_params['contrast']))
        else:
            postup_op = ops.ChangeStats(float(mei_params['contrast']), float(mei_params['mean']))
        initial_image = postup_op(initial_image)

        # Optimize
        if mei_params['blur_sigma']:
            gradient_f = utils.Compose([ops.GaussianBlur(float(mei_params['blur_sigma'])), ops.MultiplyBy(mei_params['decay_constant'], mei_params['decay_factor'], mei_params['decay_iters'])])
        else:
            gradient_f = ops.MultiplyBy(mei_params['decay_constant'], mei_params['decay_factor'])
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
class MaskParameters(dj.Lookup):
    definition = """ # parameters to create an MEI Mask
    
    mask_params:        int
    ---
    zscore_thresh:      float   # threshold used on the absolute normalized image
    closing_iters:      int     # number of dilation/erosion steps performed for binary closing
    gaussian_sigma:     float   # sigma for the gaussian applied to the mask after processing to soften edges
    """
    contents = [[1, 1.5, 2, 1], [2, 1.0, 2, 2.0], [3, 1.5, 2, 1.5],
                [4, 1. , 2, 1.5], [5, 0.5, 2, 1.5]]


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
#TODO: Maybe move mask_centroid, mask_mean, mask_std (and any other relevant properties area?)) to a MaskProperties table.
#TODO: Create centered MEI using the shift estimate of the mask

@schema
class TightMaskParameter(dj.Lookup): # same design as multi_mei.TightMaskParameter, with an additional parameter "erosion_footprint"
    """
    Parameter for "tight" mask on MEIs
    """
    definition = """
    tight_mask_param_id: int
    ---
    stdev_size_thr:         float  # fraction of standard dev threshold for size of blobs
    filter_sigma:           float  # sigma for final gaussian blur
    erosion_footprint:      int    # footprint parameter used in the erosion function, larger footprint corresponds to faster erosion
    target_reduction_ratio: float  # reduction ratio to achieve for tightening the mask
    """
    contents = [[1, 1.0, 2.0, 0.95]]

@schema
class TightMEIMask(dj.Computed): # same design as multi_mei.TightMEIMask
    definition = """
    -> MEI
    -> TightMaskParameter
    ---
    mask: longblob   # mask for mei
    reduction_ratio:  float  # achieved reduction in activation from the baseline mask
    """

    def make(self, key):
        from skimage.morphology import convex_hull_image
        from scipy.ndimage.filters import gaussian_filter
        from skimage.morphology import erosion, square
        from staticnet_analyses.utils import process
        from staticnet_analyses.multi_mei import prepare_data, get_multi_model, get_adj_model

        # get the MEI
        mei = (MEI() & key).fetch1('mei')

        # set in "c" contiguous
        img = mei.copy(order='c')

        stdev_size_thr, filter_sigma, erosion_footprint, target_reduction_ratio = (TightMaskParameter & key).fetch1('stdev_size_thr',
                                                                                                 'filter_sigma',
                                                                                                 'erosion_footprint',
                                                                                                 'target_reduction_ratio')


        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        models = get_multi_model(key)
        adj_model = get_adj_model(models, readout_key, neuron_id, mu_eye=mu_eye)

        def get_activation(mei):
            with torch.no_grad():
                img = torch.as_tensor(mei[None, None],dtype=torch.float32, device='cuda')
                activation = adj_model(img).data.cpu().numpy()[0]
            return activation

        delta = img - img.mean()
        fluc = np.abs(delta)
        thr = np.std(fluc) * stdev_size_thr

        # original mask
        mask = convex_hull_image((fluc > thr).astype(float))
        fm = gaussian_filter(mask.astype(float), sigma=filter_sigma)
        masked_img = fm * img + (1 - fm) * img.mean()
        activation = base_line = get_activation(masked_img)

        print('Baseline:', base_line)
        count = 0
        while (activation > base_line * target_reduction_ratio):
            mask = erosion(mask, square(erosion_footprint))
            fm = gaussian_filter(mask.astype(float), sigma=filter_sigma)
            masked_img = fm * img + (1 - fm) * img.mean()
            activation  = get_activation(masked_img)
            print('Activation:', activation)
            count += 1

            if count > 100:
                print('This has been going on for too long! - aborting')
                raise ValueError('The activation does not reduce for the given setting')

        key['reduction_ratio'] = activation / base_line
        key['mask'] = fm

        self.insert1(key)

@schema
class SeedSet(dj.Lookup):
    definition = """ 
    # Seed set used for ensemble models
    ssid                    : int  # seed set id
    ---
    seeds                   : blob
    """

    content = [[1, [1009, 1215, 2606, 99999]],
               [2, [1009, 1215, 2606, 99999, 101, 102, 103, 104, 105, 106]],
               [3, [1009]],
               [4, [101, 102, 103, 104]],
    ]

@schema
class EnsembleEval(dj.Computed):
    """
    Evaluate the test_score with and without behavior input and calculate the fraction oracle.
    For fraction_oracle_nb, test_score_nb, the behavior input of the model is 'freezed' when evaluating. It makes use of the mean eye position and behavior state
    everywhere. This gives a fair comparison to Oracle.
    """
    definition = """
    -> Dataset
    -> configs.NetworkConfig
    -> SeedSet
    ---
    n_models               : int    # number of models that are combined
    test_score             : float  # average testset score
    test_score_nb          : float  # average testset score with mean behavior input
    fraction_oracle        : float  # the slope of the linear fit from oracle scores to test_scores with zero intercept
    fraction_oracle_nb     : float  # the slope of the linear fit from oracle scores to test_scores_nb with zero intercept
    avg_corr               : float  # correlation of mean neural responses and mean model predictions over repeats
    """

    @property
    def key_source(self):
        cnn_models = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.GaussianLaplace * static_models.Model).proj()
        sensorium_models = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.Stacked2dNew * static_models.Model).proj()
        linear_model = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.StackedLinearGaussianLaplace * static_models.Model).proj()
        beh_models = (configs.NetworkConfig.BehaviorNet * configs.CoreConfig.GaussianLaplace * static_models.Model).proj()

        return Dataset * configs.NetworkConfig * SeedSet & [cnn_models, sensorium_models, linear_model, beh_models]

    class Unit(dj.Part):
        definition = """
        -> master
        -> Dataset.Unit
        ---
        unit_score            : float  # correlation score    
        unit_score_nb         : float  # correlation score with mean behavior input
        oracle_score          : float  # oracle correlation
        unit_fraction_oracle  : float  # fraction oracle
        unit_fraction_oracle_nb  : float  # fraction oracle with mean behavior input
        unit_avg_corr         : float  # correlation of mean neural responses and mean model predictions over repeats
        """

    @staticmethod
    def get_multi_model(key):
        # load the model
        seeds = (SeedSet & key).fetch1('seeds')
        model_keys = (static_models.Model & key & [dict(seed=seed) for seed in seeds]).fetch('KEY')
        assert len(seeds) == len(model_keys), 'Number of seeds and models do not match!'
        
        ensemble = []
        for mk in model_keys:
            model = (static_models.Model & mk).load_network().to('cuda')
            model.eval()
            ensemble.append(model)
        return ensemble

    @staticmethod
    def multi_model_wrapper(ensemble):
        def compute(x, readout_key, eye_pos=None, behavior=None):
            resp = 0
            for m in ensemble:
                resp = resp + m(x, readout_key, eye_pos=eye_pos, behavior=behavior)
            return resp / len(ensemble)
        return compute


    def make(self, key):
        # complete the key
        key = configs.NetworkConfig().net_key(key)
        # get test dataloaders
        testsets, testloaders = configs.DataConfig().load_data(key, tier='test', cuda=True, batch_size=10)
        # get ensemble model
        ensemble = self.get_multi_model(key)
        comb_model = self.multi_model_wrapper(ensemble)
        # get unit oracle scores
        if not len(stats.Oracle.UnitScores * Dataset.Unit & key):
                oracle_units = stats.OracleMultiTier.UnitScores * Dataset.Unit & key
        else:
            oracle_units = stats.Oracle.UnitScores * Dataset.Unit & key

        # compute scores for each readout key
        scores, scores_nb, avg_corrs, unit_scores = [], [], [], []
        for readout_key, testloader in testloaders.items():
            ## compute scores without behavior input
            # fetch mean eye position and behavior traces
            trainsets, _ = configs.DataConfig().load_data(key, tier='train')
            trainset = trainsets[readout_key]
            mus = trainset.transformed_mean()
            if 'pupil_center' in trainset.data_keys and 'behavior' in trainset.data_keys:
                mu_eye = mus.pupil_center[None, :].to('cuda')
                mu_beh = mus.behavior[None, :].to('cuda')
                # override behavioral information with average behavioral values
                y, y_hat = compute_predictions(testloader, comb_model, readout_key, eye_pos=mu_eye, behavior=mu_beh)
                perf_scores_nb = compute_scores(y, y_hat).pearson
                scores_nb.append(perf_scores_nb)

                ## compute scores with behavior input
                y, y_hat = compute_predictions(testloader, comb_model, readout_key)
                perf_scores = compute_scores(y, y_hat).pearson
                scores.append(perf_scores)

            else:
                ## for models trained without behavior input, use None as behavior input and insert scores to both test_score and test_score_nb
                y, y_hat = compute_predictions(testloader, comb_model, readout_key)
                perf_scores = compute_scores(y, y_hat).pearson
                scores.append(perf_scores)
                perf_scores_nb = perf_scores
                scores_nb.append(perf_scores)

            ## compute mean corr
            ### re-organize the responses and predictions
            y_dict = defaultdict(list)
            y_hat_dict = defaultdict(list)
            test_cond = testloader.dataset.condition_hashes[testloader.dataset.tiers=='test']
            for cond, y_, y_hat_ in zip(test_cond, y, y_hat):
                y_dict[cond].append(y_)
                y_hat_dict[cond].append(y_hat_)
            _y = np.array([np.stack(v).mean(axis=0) for v in y_dict.values()])
            _y_hat = np.array([np.stack(v).mean(axis=0) for v in y_hat_dict.values()])
            avg_corr = compute_scores(_y, _y_hat).pearson
            avg_corrs.append(avg_corr)

            unit_scores.extend(
                [dict(key, readout_key=readout_key, unit_score_nb=c_nb, unit_score=c, unit_avg_corr=a, neuron_id=n) for c_nb, c, a, n in zip(perf_scores_nb, perf_scores, avg_corr, count())])

        for unit_key in unit_scores:
            unit_key['oracle_score'] = (oracle_units & unit_key).fetch1('pearson')
            unit_key['unit_fraction_oracle'] = unit_key['unit_score'] / unit_key['oracle_score']
            unit_key['unit_fraction_oracle_nb'] = unit_key['unit_score_nb'] / unit_key['oracle_score']

        # compute fraction oracle (with behavior)
        x = np.array([unit_key['oracle_score'] for unit_key in unit_scores]).reshape(-1,1)
        y = np.array([unit_key['unit_score'] for unit_key in unit_scores])
        fo = LinearRegression(fit_intercept=False).fit(x,y)

        # compute fraction oracle (with mean behavior)
        x = np.array([unit_key['oracle_score'] for unit_key in unit_scores]).reshape(-1,1)
        y = np.array([unit_key['unit_score_nb'] for unit_key in unit_scores])
        fo_nb = LinearRegression(fit_intercept=False).fit(x,y)      

        key['n_models'] = len(ensemble)
        key['test_score'] = np.concatenate(scores).mean()
        key['test_score_nb'] = np.concatenate(scores_nb).mean()
        key['fraction_oracle'] = float(fo.coef_)
        key['fraction_oracle_nb'] = float(fo_nb.coef_)
        key['avg_corr'] = np.concatenate(avg_corrs).mean()

        self.insert1(key, ignore_extra_fields=True)
        self.Unit.insert(unit_scores, ignore_extra_fields=True)

@schema
class EnsembleEvalTestMEI(dj.Computed):
    """
    Same design as EnsembleEval, except that all scores are computed on images with tier "test_mei" instead of "test"
    """
    definition = """
    -> Dataset
    -> configs.NetworkConfig
    -> SeedSet
    ---
    n_models               : int    # number of models that are combined
    test_score             : float  # average testset score
    test_score_nb          : float  # average testset score with mean behavior input
    fraction_oracle        : float  # the slope of the linear fit from oracle scores to test_scores with zero intercept
    fraction_oracle_nb     : float  # the slope of the linear fit from oracle scores to test_scores_nb with zero intercept
    avg_corr               : float  # correlation of mean neural responses and mean model predictions over repeats
    """

    @property
    def key_source(self):
        cnn_models = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.GaussianLaplace * static_models.Model)
        linear_model = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.StackedLinearGaussianLaplace * static_models.Model)
        beh_models = (configs.NetworkConfig.BehaviorNet * configs.CoreConfig.GaussianLaplace * static_models.Model)

        return Dataset * configs.NetworkConfig * SeedSet & [cnn_models, linear_model, beh_models]

    class Unit(dj.Part):
        definition = """
        -> master
        -> Dataset.Unit
        ---
        unit_score            : float  # correlation score    
        unit_score_nb         : float  # correlation score with mean behavior input
        oracle_score          : float  # oracle correlation
        unit_fraction_oracle  : float  # fraction oracle
        unit_fraction_oracle_nb  : float  # fraction oracle with mean behavior input
        unit_avg_corr         : float  # correlation of mean neural responses and mean model predictions over repeats
        """

    @staticmethod
    def get_multi_model(key):
        # load the model
        seeds = (SeedSet & key).fetch1('seeds')
        model_keys = (static_models.Model & key & [dict(seed=seed) for seed in seeds]).fetch('KEY')
        assert len(seeds) == len(model_keys), 'Number of seeds and models do not match!'
        
        ensemble = []
        for mk in model_keys:
            model = (static_models.Model & mk).load_network().to('cuda')
            model.eval()
            ensemble.append(model)
        return ensemble

    @staticmethod
    def multi_model_wrapper(ensemble):
        def compute(x, readout_key, eye_pos=None, behavior=None):
            resp = 0
            for m in ensemble:
                resp = resp + m(x, readout_key, eye_pos=eye_pos, behavior=behavior)
            return resp / len(ensemble)
        return compute


    def make(self, key):
        # complete the key
        key = configs.NetworkConfig().net_key(key)
        # get test dataloaders
        testsets, testloaders = configs.DataConfig().load_data(key, tier='test_mei', cuda=True, batch_size=10)
        # get ensemble model
        ensemble = self.get_multi_model(key)
        comb_model = self.multi_model_wrapper(ensemble)
        # get unit oracle scores
        if not len(stats.Oracle.UnitScores * Dataset.Unit & key):
                oracle_units = stats.OracleMultiTier.UnitScores * Dataset.Unit & key
        else:
            oracle_units = stats.Oracle.UnitScores * Dataset.Unit & key

        # compute scores for each readout key
        scores, scores_nb, avg_corrs, unit_scores = [], [], [], []
        for readout_key, testloader in testloaders.items():
            ## compute scores without behavior input
            # fetch mean eye position and behavior traces
            trainsets, _ = configs.DataConfig().load_data(key, tier='train')
            trainset = trainsets[readout_key]
            mus = trainset.transformed_mean()
            mu_eye = mus.pupil_center[None, :].to('cuda')
            mu_beh = mus.behavior[None, :].to('cuda')
            # override behavioral information with average behavioral values
            y, y_hat = compute_predictions(testloader, comb_model, readout_key, eye_pos=mu_eye, behavior=mu_beh)
            perf_scores_nb = compute_scores(y, y_hat).pearson
            scores_nb.append(perf_scores_nb)

            ## compute scores with behavior input
            y, y_hat = compute_predictions(testloader, comb_model, readout_key)
            perf_scores = compute_scores(y, y_hat).pearson
            scores.append(perf_scores)

            ## compute mean corr
            ### re-organize the responses and predictions
            y_dict = defaultdict(list)
            y_hat_dict = defaultdict(list)
            test_cond = testloader.dataset.condition_hashes[testloader.dataset.tiers=='test_mei']
            for cond, y_, y_hat_ in zip(test_cond, y, y_hat):
                y_dict[cond].append(y_)
                y_hat_dict[cond].append(y_hat_)
            _y = np.array([np.stack(v).mean(axis=0) for v in y_dict.values()])
            _y_hat = np.array([np.stack(v).mean(axis=0) for v in y_hat_dict.values()])
            avg_corr = compute_scores(_y, _y_hat).pearson
            avg_corrs.append(avg_corr)

            unit_scores.extend(
                [dict(key, readout_key=readout_key, unit_score_nb=c_nb, unit_score=c, unit_avg_corr=a, neuron_id=n) for c_nb, c, a, n in zip(perf_scores_nb, perf_scores, avg_corr, count())])

        for unit_key in unit_scores:
            unit_key['oracle_score'] = (oracle_units & unit_key).fetch1('pearson')
            unit_key['unit_fraction_oracle'] = unit_key['unit_score'] / unit_key['oracle_score']
            unit_key['unit_fraction_oracle_nb'] = unit_key['unit_score_nb'] / unit_key['oracle_score']

        # compute fraction oracle (with behavior)
        x = np.array([unit_key['oracle_score'] for unit_key in unit_scores]).reshape(-1,1)
        y = np.array([unit_key['unit_score'] for unit_key in unit_scores])
        fo = LinearRegression(fit_intercept=False).fit(x,y)

        # compute fraction oracle (with mean behavior)
        x = np.array([unit_key['oracle_score'] for unit_key in unit_scores]).reshape(-1,1)
        y = np.array([unit_key['unit_score_nb'] for unit_key in unit_scores])
        fo_nb = LinearRegression(fit_intercept=False).fit(x,y)      

        key['n_models'] = len(ensemble)
        key['test_score'] = np.concatenate(scores).mean()
        key['test_score_nb'] = np.concatenate(scores_nb).mean()
        key['fraction_oracle'] = float(fo.coef_)
        key['fraction_oracle_nb'] = float(fo_nb.coef_)
        key['avg_corr'] = np.concatenate(avg_corrs).mean()

        self.insert1(key, ignore_extra_fields=True)
        self.Unit.insert(unit_scores, ignore_extra_fields=True)

@schema
class NeuronSetMethod(dj.Lookup):
    definition = """     # method for random neuron selection across multiple groups for any customized purposes
    method_id: int
    ---
    ranking_params:  int            # ranking parameter in RankingParameters
    oracle_thresh:   float          # threshold on oracle score in EnsembleEval
    avg_corr_thresh: float          # threshold on avg_corr computed in EnsembleEval
    selection_seed:  int            # seed for random selecting neurons from all valid neurons
    n_neurons_per_group:       int           
    description:     varchar(128)
    """
    contents = [(1, 8, 0.5, 0.7, 1234, 125, 'random unique neurons with high oracle and model performance'),
                (2, 4, 0.2, 0.4, 1234, 100, 'random unique neurons with oracle and model performance that typically forms top 80 percentile of the population')]

@schema
class NeuronSet(dj.Lookup):
    definition = """
    -> NeuronSetMethod
    set_id:     int
    ---
    group_id:   longblob  # group_ids belonging to this neuron set
    """

    class Neuron(dj.Part):
        definition = """
        -> master
        -> Dataset.Unit
        -> static_models.Model
        """
    
    def fill_neuron(self, set_key=dict(set_id=1, method_id=1, group_id=np.array([142, 204, 209, 215, 222, 223, 224, 225]))):
        self.insert1(set_key, skip_duplicates=True)
        method_params = (NeuronSetMethod & set_key).fetch1()
        np.random.seed(method_params['selection_seed'])
        for gid in set_key['group_id']:
            dics = (EnsembleEval.Unit * UnitRanking.Unit & 
                     {'group_id': gid, 'ranking_params': method_params['ranking_params']} & 
                     'oracle_score > {} and unit_avg_corr > {}'.format(method_params['oracle_thresh'], method_params['avg_corr_thresh'])
                     ).fetch(as_dict=True)
            if len(dics) >= method_params['n_neurons_per_group']:
                selected_dics = np.random.choice(np.array(dics), method_params['n_neurons_per_group'], replace=False)
            else:
                selected_dics = dics
            self.Neuron.insert([{**set_key, **dic} for dic in selected_dics], ignore_extra_fields=True, skip_duplicates=True)


@schema
class NewEnsembleEval(dj.Computed):
    definition = """
    -> Dataset
    -> configs.NetworkConfig
    -> SeedSet
    ---
    n_models                      : int    # number of models that are combined
    median_cc_abs                 : float  # median absolute correlation coefficient
    median_cc_max                 : float  # median maximum possible correlation coefficient
    median_cc_norm                : float  # median cc_abs/cc_max
    """
    @property
    def key_source(self):
        sensorium_models = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.Stacked2dNew * static_models.Model)
        cnn_models = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.GaussianLaplace * static_models.Model)
        linear_model = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.StackedLinearGaussianLaplace * static_models.Model)
        beh_models = (configs.NetworkConfig.BehaviorNet * configs.CoreConfig.GaussianLaplace * static_models.Model)
        return Dataset * configs.NetworkConfig * SeedSet & [sensorium_models, cnn_models, linear_model, beh_models]
    
    class Unit(dj.Part):
        definition = """
        -> master
        -> Dataset.Unit
        ---
        cc_abs                 : float  #  absolute correlation coefficient
        cc_max                 : float  #  maximum possible correlation coefficient
        cc_norm                : float  #  cc_abs/cc_max
        """
        
    @staticmethod
    # responses is a list of [n_repeats,n_units]
    def cal_reliability(responses): 
        # Fill missing trial with NaNs for convenience
        agg_ns = np.array([len(r) for r in responses])
        max_n = max(agg_ns)
        for i, r in enumerate(responses):
            add_shape = list(r.shape)
            if add_shape[0] < max_n:
                add_shape[0] = max_n - add_shape[0]
                nan = np.full_like(r, np.nan, shape=add_shape)
                responses[i] = np.concatenate([r, nan])
        v = 1 / agg_ns**2
        w = agg_ns - 1
        z = agg_ns.sum() - len(agg_ns)
        n = np.sqrt(z.sum() / (w * v).sum())
        y = np.stack(responses, axis=0)  

        # Mean for each stimuli
        y_m = np.nanmean(y,axis=1)

        # Power: variance of mean of stimuli
        P = np.var(y_m,axis=0,ddof=1)

        # Total power: mean of variance across repeats
        TP = np.mean(np.nanvar(y, axis=0, ddof=1), axis=0)

        # Signal power: 
        SP = (n * P - TP) / (n - 1)
        # variance of response mean
        y_m_v = np.var(y_m, axis=0, ddof=0)

        # correlation coefficient ceiling
        cc_max = np.sqrt(SP / y_m_v)
        return cc_max

    def make(self, key):
        # complete the key
        key = configs.NetworkConfig().net_key(key)
        # get test dataloaders
        testsets, testloaders = configs.DataConfig().load_data(key, tier='test', cuda=True, batch_size=10)
        # get ensemble model
        ensemble = EnsembleEval.get_multi_model(key)
        comb_model = EnsembleEval.multi_model_wrapper(ensemble)

        # get unit oracle scores
        oracle_units = stats.Oracle.UnitScores * Dataset.Unit & key

        # compute scores for each readout key
        unit_scores = []
        all_cc_abs, all_cc_max, all_cc_norm = [], [], []

        for readout_key, testloader in testloaders.items():

            ## compute scores without behavior input
            # fetch mean eye position and behavior traces
            trainsets, _ = configs.DataConfig().load_data(key, tier='train')
            trainset = trainsets[readout_key]
            mus = trainset.transformed_mean()
            
            if 'pupil_center' in trainset.data_keys and 'behavior' in trainset.data_keys:
                mu_eye = mus.pupil_center[None, :].to('cuda')
                mu_beh = mus.behavior[None, :].to('cuda')
                # override behavioral information with average behavioral values
                y, y_hat = compute_predictions(testloader, comb_model, readout_key, eye_pos=mu_eye, behavior=mu_beh)
            else:
                ## for models trained without behavior input, use None as behavior input and insert scores to both test_score and test_score_nb
                y, y_hat = compute_predictions(testloader, comb_model, readout_key)
                
            y_dict = defaultdict(list)
            y_hat_dict = defaultdict(list)
            test_cond = testloader.dataset.condition_hashes[testloader.dataset.tiers=='test']
            for cond, y_, y_hat_ in zip(test_cond, y, y_hat):
                y_dict[cond].append(y_)
                y_hat_dict[cond].append(y_hat_)

            for cond in y_dict.keys():
                y_dict[cond] = np.stack(y_dict[cond])
                y_hat_dict[cond] = np.stack(y_hat_dict[cond])

            # Average across trial from y_dict
            _y = np.array([np.stack(v).mean(axis=0) for v in y_dict.values()])
            _y_hat = np.array([v[0] for v in y_hat_dict.values()])
            cc_abs = compute_scores(_y, _y_hat).pearson
            cc_max = self.cal_reliability(list(y_dict.values()))
            cc_norm = cc_abs/cc_max
            
            # Replace nan value with 0.0
            cc_abs = np.nan_to_num(cc_abs,nan=0.0)
            cc_max = np.nan_to_num(cc_max,nan=0.0)
            cc_norm = np.nan_to_num(cc_norm,nan=0.0)
            
            all_cc_abs.extend(cc_abs)
            all_cc_max.extend(cc_max)
            all_cc_norm.extend(cc_norm)

            for i,(j,k,l) in enumerate(zip(cc_abs,cc_max,cc_norm)):
                unit_scores.extend([{**key,'readout_key':readout_key,'cc_abs':j,'cc_max':k,'cc_norm':l,'neuron_id':i}])

        median_scores = {**key,'median_cc_abs':np.nanmedian(all_cc_abs),
                        'median_cc_max':np.nanmedian(all_cc_max),'median_cc_norm':np.nanmedian(all_cc_norm),
                        'n_models':len(ensemble)}

        self.insert1(median_scores, ignore_extra_fields=True)
        self.Unit.insert(unit_scores, ignore_extra_fields=True)


@schema
class DynamicStaticEnsembleEval(dj.Computed):
    definition = """ Evaluate dynamic static model prediction against the groundtruth in vivo responses to static images presented in the dynamic scan
    -> Dataset
    -> configs.NetworkConfig
    -> SeedSet
    ---
    n_models                      : int    # number of models that are combined
    median_cc_abs                 : float  # median absolute correlation coefficient between dynamic static model prediction and in vivo static oracle responses in the dynamic scan
    median_cc_max                 : float  # median maximum possible correlation coefficient based on in vivo static oracle responses
    median_cc_norm                : float  # median cc_abs/cc_max
    """
    @property
    def key_source(self):
        cnn_models = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.GaussianLaplace * static_models.Model)
        linear_model = (configs.NetworkConfig.CorePlusReadout * configs.CoreConfig.StackedLinearGaussianLaplace * static_models.Model)
        beh_models = (configs.NetworkConfig.BehaviorNet * configs.CoreConfig.GaussianLaplace * static_models.Model)
        dynamic_dataset = data_schemas.StaticMultiDataset.Member & 'preproc_id = 8'
        return (Dataset & dynamic_dataset) * configs.NetworkConfig * SeedSet & [cnn_models, linear_model, beh_models]
    
    class Unit(dj.Part):
        definition = """
        -> master
        -> Dataset.Unit
        ---
        cc_abs                 : float  #  absolute correlation coefficient
        cc_max                 : float  #  maximum possible correlation coefficient
        cc_norm                : float  #  cc_abs/cc_max
        """

    def make(self, key):
        # complete the key
        key = configs.NetworkConfig().net_key(key)
        # get test dataloaders of the corresponding in vivo static oracle dataset
        temp_key = (data_schemas.StaticMultiDataset.Member & key).fetch1(dj.key)
        temp_key.pop('group_id')
        temp_key['preproc_id'] = 14
        dyn_group = (data_schemas.StaticMultiDataset.Member & temp_key).fetch1('group_id')
        dyn_key = key.copy()
        dyn_key['group_id'] = dyn_group
        _, testloaders = configs.DataConfig().load_data(dyn_key, tier='test', cuda=True, batch_size=10)
        # get ensemble model
        ensemble = EnsembleEval.get_multi_model(key)
        comb_model = EnsembleEval.multi_model_wrapper(ensemble)

        # compute scores for each readout key
        median_scores, unit_scores = [], []

        for _, testloader in testloaders.items():

            ## compute scores without behavior input
            # fetch mean eye position and behavior traces
            trainsets, _ = configs.DataConfig().load_data(key, tier='train')
            readout_key = list(trainsets.keys())[0]
            trainset = trainsets[readout_key]
            mus = trainset.transformed_mean()
            
            if 'pupil_center' in trainset.data_keys and 'behavior' in trainset.data_keys:
                mu_eye = mus.pupil_center[None, :].to('cuda')
                mu_beh = mus.behavior[None, :].to('cuda')
                # override behavioral information with average behavioral values
                y, y_hat = compute_predictions(testloader, comb_model, readout_key, eye_pos=mu_eye, behavior=mu_beh)
            else:
                ## for models trained without behavior input, use None as behavior input and insert scores to both test_score and test_score_nb
                y, y_hat = compute_predictions(testloader, comb_model, readout_key)
                
            y_dict = defaultdict(list)
            y_hat_dict = defaultdict(list)
            test_cond = testloader.dataset.condition_hashes[testloader.dataset.tiers=='test']
            for cond, y_, y_hat_ in zip(test_cond, y, y_hat):
                y_dict[cond].append(y_)
                y_hat_dict[cond].append(y_hat_)

            for cond in y_dict.keys():
                y_dict[cond] = np.stack(y_dict[cond])
                y_hat_dict[cond] = np.stack(y_hat_dict[cond])

            # Average across trial from y_dict
            _y = np.array([np.stack(v).mean(axis=0) for v in y_dict.values()])
            _y_hat = np.array([v[0] for v in y_hat_dict.values()])
            cc_abs = compute_scores(_y, _y_hat).pearson
            cc_max = NewEnsembleEval.cal_reliability(list(y_dict.values()))
            cc_norm = cc_abs/cc_max
            
            # Replace nan value with 0.0
            cc_abs = np.nan_to_num(cc_abs,nan=0.0)
            cc_max = np.nan_to_num(cc_max,nan=0.0)
            cc_norm = np.nan_to_num(cc_norm,nan=0.0)
            
            median_scores = {**key,'median_cc_abs':np.nanmedian(cc_abs),
                             'median_cc_max':np.nanmedian(cc_max),'median_cc_norm':np.nanmedian(cc_norm),
                             'n_models':len(ensemble)}
            
            for i,(j,k,l) in enumerate(zip(cc_abs,cc_max,cc_norm)):
                unit_scores.append({**key,'readout_key':readout_key,'cc_abs':j,'cc_max':k,'cc_norm':l,'neuron_id':i})
            self.insert1(median_scores, ignore_extra_fields=True)
            self.Unit.insert(unit_scores, ignore_extra_fields=True)
