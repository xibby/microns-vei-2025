import os
import numpy as np
from numpy.linalg import LinAlgError
from scipy import interpolate
from tqdm import tqdm
import datajoint as dj
from pprint import pformat
import warnings
from collections import defaultdict
from neuro_data.static_images import data_schemas
from neuro_data.static_images.data_schemas import StaticScan, InputResponse, BehaviorMixin, StaticMultiDataset
from neuro_data.utils.data import SplineCurve, FilterMixin, NaNSpline
from neuro_data import logger

warnings.filterwarnings("ignore")
fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')
meso = dj.create_virtual_module('meso', 'pipeline_meso')
experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')
pupil = dj.create_virtual_module('pupil', 'pipeline_eye')

dj.config.setdefault('stores', dict())
dj.config['stores'].update({
    'static': dict(
        protocol='file', 
        location='/dj-stor01/neuro-static')
})

schema = dj.schema('neurostatic_rf')
dj.config['enable_python_native_blobs'] = True


#-------------------------------------------- Dotmapping linear RF ---------------------------------------------
@schema
class Preprocessing(dj.Lookup):
    definition = """
    preproc_id       : tinyint # preprocessing ID
    ---
    offset           : decimal(6,4) # offset to stimulus onset in s
    duration         : decimal(6,4) # window length in s
    row              : smallint     # row size of movies
    col              : smallint     # col size of movie
    filter           : varchar(24)  # filter type for window extraction
    """
    contents = [(0, 0.05, 0.25, 36, 64, 'hamming')]

@schema 
class Eye(dj.Computed, FilterMixin, BehaviorMixin):
    definition = """
    -> experiment.Scan
    -> Preprocessing
    ---
    -> pupil.FittedPupil             # tracking_method as a secondary attribute
    trial_idxs         : blob@data   # idxs of single dot trials
    pupil              : blob@data   # pupil dilation trace
    dpupil             : blob@data   # derivative of pupil dilation trace
    center             : blob@data   # center position of the eye
    """
    
    def make(self, scan_key):
        scan_key = {**scan_key, 'tracking_method': 2}
        logger.info('Populating '+ pformat(scan_key))
        radius, xy, eye_time = self.load_eye_traces(scan_key)
        frame_times = self.load_frame_times(scan_key)
        behavior_clock = self.load_behavior_timing(scan_key)

        if len(frame_times) - len(behavior_clock) != 0:
            assert abs(len(frame_times) - len(behavior_clock)) < 2, 'Difference bigger than 2 time points'
            l = min(len(frame_times), len(behavior_clock))
            logger.info('Frametimes and stimulus.BehaviorSync differ in length! Shortening it.')
            frame_times = frame_times[:l]
            behavior_clock = behavior_clock[:l]

        fr2beh = NaNSpline(frame_times, behavior_clock, k=1, ext=3)

        duration, offset = map(float, (Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        logger.info('Downsampling eye signal to {}Hz'.format(1 / duration))
        deye = np.nanmedian(np.diff(eye_time))
        h_eye = self.get_filter(duration, deye, 'hamming', warning=True)
        h_deye = self.get_filter(duration, deye, 'dhamming', warning=True)
        pupil_spline = NaNSpline(eye_time,
                                 np.convolve(radius, h_eye, mode='same'), k=1, ext=0)

        dpupil_spline = NaNSpline(eye_time,
                                  np.convolve(radius, h_deye, mode='same'), k=1, ext=0)
        center_spline = SplineCurve(eye_time,
                                    np.vstack([np.convolve(coord, h_eye, mode='same') for coord in xy]),
                                    k=1, ext=0)

        trial_idxs, flip_times = (stimulus.SingleDot * stimulus.Trial & scan_key).fetch('trial_idx', 'flip_times', order_by='trial_idx')

        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            logger.warning('No static frames were present to be processed for {}'.format(scan_key))
            return

        stimulus_onset = InputResponse.stimulus_onset(flip_times, duration)
        t = fr2beh(stimulus_onset + sample_point)
        pupil = pupil_spline(t)
        dpupil = dpupil_spline(t)
        center = center_spline(t)

        self.insert1(dict(scan_key, trial_idxs=trial_idxs, pupil=pupil, dpupil=dpupil, center=center))  
    

@schema
class ExclusionMetric(dj.Lookup):
    definition = """
    metric_id    : int
    ---
    n_std        : float  # exclude trials where the eye position is more than n_std away from the mean eye position of the entire scan
    """
    contents = [(0, 2)]
    
    def exclude_trials_by_eye(self, scan_key, n_std=2):
        positions = (pupil.FittedPupil.Circle & scan_key).fetch('center')
        positions = np.vstack(p for p in positions if p is not None)
        center = np.mean(positions, 0, keepdims=True)
        # std_thresh = n_std * np.linalg.norm(np.std(positions, 0)) # Jiakun's approach
        std_thresh = n_std * np.std(np.linalg.norm(positions - np.mean(positions, 0, keepdims=True), axis=1))
        
        trial_idxs, p, dp, pos = (Eye & scan_key).fetch1('trial_idxs', 'pupil', 'dpupil', 'center')
        d = np.linalg.norm(pos - center.T, axis=0)
        invalid = np.isnan(p + dp + pos.sum(axis=0)) | (d > std_thresh)
        
        return trial_idxs[invalid]
        

@schema
class ExcludedTrial(dj.Manual):
    definition = """
    # trials to be excluded from analysis
    -> ExclusionMetric
    -> stimulus.Trial
    """

    def fill(self, scan_key, metric_id=0):
        n_std = (ExclusionMetric & dict(metric_id=metric_id)).fetch1('n_std')
        excluded = ExclusionMetric().exclude_trials_by_eye(scan_key, n_std=n_std)
        excluded = (ExclusionMetric * stimulus.Trial & scan_key & [{'trial_idx': t} for t in excluded]).fetch(dj.key, order_by='trial_idx')
        self.insert(excluded, skip_duplicates=True)
            
    
@schema 
class TrialResponse(dj.Computed):
    definition = """
    -> experiment.Scan
    -> Preprocessing
    ---
    responses    : longblob    # array of responses (num_neurons, num_trials)
    """
    
    def make(self, scan_key):        
        # integration window size for responses
        duration, offset = map(float, (Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        logger.info('Sampling neural responses at {}s intervals'.format(duration))

        trace_spline, trace_keys, ftmin, ftmax = InputResponse().get_trace_spline(scan_key, duration)
        # exclude trials marked in ExcludedTrial
        logger.info('Excluding {} trials based on ExcludedTrial'.format(len(ExcludedTrial() & scan_key)))
        flip_times, trial_keys = (stimulus.SingleDot * (stimulus.Trial - ExcludedTrial) & scan_key).fetch('flip_times', dj.key,
                                                                           order_by='trial_idx')
        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            logger.warning('No single dot frames were present to be processed for {}'.format(scan_key))
            return

        valid = np.array([ft.min() >= ftmin and ft.max() <= ftmax for ft in flip_times], dtype=bool)
        if not np.all(valid):
            logger.warning('Dropping {} trials with dropped frames or flips outside the recording interval'.format(
                (~valid).sum()))

        stimulus_onset = InputResponse.stimulus_onset(flip_times, duration)
        logger.info('Sampling {} responses {}s after stimulus onset'.format(valid.sum(), sample_point))
        R = trace_spline(stimulus_onset[valid] + sample_point, log=True).T

        self.insert1(dict(scan_key, responses=R))


@schema
class SparseRFParams(dj.Lookup):
    definition = """
    rf_params          : int
    ---
    std                : float       # standard deviation for RF boundary
    """
    contents = [(0, 1.5)]


@schema
class SparseRF(dj.Computed):
    definition = """
    -> TrialResponse
    -> SparseRFParams
    ---    
    """

    class Unit(dj.Part):
        definition = """
        -> master
        -> fuse.Activity.Trace
        ---
        rawmap         : blob        # average response for each location, first axis: [off, on]
        fmap           : blob        # interpolated and smoothened map
        on_params      : blob        # gaussian fit parameters (height, center_x, center_y, width_x, width_y, rotation)
        off_params     : blob        # gaussian fit parameters (height, center_x, center_y, width_x, width_y, rotation)
        on_center      : blob        # center location in stimulus unit (width, height)
        off_center     : blob        # center location in stimulus unit (width, height)
        on_boundary    : blob        # rf boundary in stimulus unit (width, height)
        off_boundary   : blob        # rf boundary in stimulus unit (width, height)
        on_p_value     : float       # bootstrap significance
        off_p_value    : float       # bootstrap significance
        """

    def _make_tuples(self, key):
        print('Populating ', key)
        self.insert1(key)

        pipe = (fuse.Activity() & key).fetch('pipe')[0]
        pipe = dj.create_virtual_module(pipe, 'pipeline_' + pipe)
        trace_keys = ((pipe.MaskClassification.Type & 'type = "soma"') * pipe.ScanSet.Unit * fuse.Activity.Trace & key
                    ).fetch(dj.key, order_by='unit_id')
        responses = (TrialResponse & key).fetch1('responses').T
        dot_x, dot_y, dot_size, dot_level = (stimulus.SingleDot * (stimulus.Trial - ExcludedTrial) & key).fetch('dot_x', 'dot_y', 'dot_xsize', 'dot_level', order_by='trial_idx')
        nx = len(np.unique(dot_x))
        ny = len(np.unique(dot_y))
        on = np.unique(dot_level)[1]
        off = np.unique(dot_level)[0]
        xrange = np.max(dot_x) - np.min(dot_x)
        dot_size = np.unique(dot_size)[0]
        std = (SparseRFParams & key).fetch1('std')

        def anova(location, response):
            import pandas as pd
            from statsmodels.formula.api import ols
            import statsmodels.api as sm
            d = {'loc': location, 'res': response}
            df = pd.DataFrame(data=d)
            fit = ols('res ~ C(loc)', data=df).fit()
            aov_table = sm.stats.anova_lm(fit, typ=2)
            p_value = aov_table['PR(>F)'][0]
            return p_value

        for tk, resp in tqdm(zip(trace_keys, responses)):
            responses_on = np.array(
                [resp[(dot_x == x) * (dot_y == y) * (dot_level == on)] for x in np.unique(dot_x) for y in np.unique(dot_y)])
            responses_off = np.array(
                [resp[(dot_x == x) * (dot_y == y) * (dot_level == off)] for x in np.unique(dot_x) for y in np.unique(dot_y)])

            loc = np.concatenate([i * np.ones(len(res)) for i, res in enumerate(responses_on)])
            try:
                p_on = anova(loc, np.concatenate(responses_on))
            except LinAlgError:
                continue
            loc = np.concatenate([i * np.ones(len(res)) for i, res in enumerate(responses_off)])
            try:
                p_off = anova(loc, np.concatenate(responses_off))
            except LinAlgError:
                continue

            on_map = np.array([r.mean() for r in responses_on]).reshape(nx, ny)
            off_map = np.array([r.mean() for r in responses_off]).reshape(nx, ny)

            try:
                on_fmap, on_fit, on_params, on_center_x, on_center_y = fit_rf(on_map, dot_size)
                off_fmap, off_fit, off_params, off_center_x, off_center_y = fit_rf(off_map, dot_size)

                on_center_x, on_center_y = convert_unit(on_center_x, on_center_y, on_fmap, xrange)
                on_cirx, on_ciry = compute_rf(*on_params, std)
                on_cirx, on_ciry = convert_unit(on_cirx, on_ciry, on_fmap, xrange)
                off_center_x, off_center_y = convert_unit(off_center_x, off_center_y, off_fmap, xrange)
                off_cirx, off_ciry = compute_rf(*off_params, std)
                off_cirx, off_ciry = convert_unit(off_cirx, off_ciry, off_fmap, xrange)
                self.Unit.insert1(dict(key, **tk, 
                                        rawmap=np.stack([off_map, on_map]), fmap=np.stack([off_fmap, on_fmap]),
                                        on_params=on_params, off_params=off_params, 
                                        on_center=np.array([on_center_y, on_center_x]), off_center=np.array([off_center_y, off_center_x]), 
                                        on_boundary=np.array([on_ciry, on_cirx]), off_boundary=np.array([off_ciry, off_cirx]),
                                        on_p_value=p_on, off_p_value=p_off), 
                                        ignore_extra_fields=True
                                )
            except ValueError:
                continue


# ------------------------------------ Regularized pseudoinverse linear RF from natural image responses --------------------------------------
@schema 
class RegPseudoInverseParams(dj.Lookup):
    definition = """
    rpi_params          : int
    ---
    lambdas             : longblob
    valid_only          : tinyint
    """
    contents = [(0, np.logspace(1, 4, 20), 1)]

@schema 
class RegPseudoInverseRF(dj.Computed):
    definition = """
    -> fuse.Activity.Trace
    -> data_schemas.Preprocessing
    -> RegPseudoInverseParams
    ---
    best_l_rf            : longblob     # linear rf with the highest pearson between gt and linearly predicted response
    best_l_pearson       : float        # pearson corresponding to the best_l_rf
    best_nl_rf           : longblob     # linear rf with the highest pearson between gt and non-linearly predicted response
    best_nl_pearson      : float        # pearson corresponding to the best_nl_rf
    rfs                  : blob@static  # linear rfs computed from a series of lambdas
    sigmoid_params       : longblob     # fitted paramters from the nonlinear sigmoid functioms
    l_pearsons           : longblob     # pearsons between gt and linearly predicted responses from rfs computed from a series of lambdas
    nl_pearsons          : longblob     # pearsons between gt and non-linearly predicted responses from rfs computed from a series of lambdas
    """
    
    @property
    def key_source(self):
        soma_units = meso.ScanSet.Unit * meso.MaskClassification.Type & 'type="soma"'
        return fuse.Activity.Trace * data_schemas.Preprocessing * RegPseudoInverseParams & StaticMultiDataset.Member & soma_units & 'preproc_id = 9'
    
    def make(self, key):
        img_size = (data_schemas.Preprocessing & key).fetch1('row', 'col')
        lambdas = (RegPseudoInverseParams & key).fetch1('lambdas')
        scan_key = (StaticMultiDataset.Member & key).fetch(dj.key, limit=1)[0]
        dset = get_dataset_helper(scan_key, 'reg_pseudoinv')
        
        i = np.argwhere(dset['unit_ids'] == key['unit_id']).item()
        rfs = [self.compute_rf(dset['X_train'], dset['Y_train'][:, i], img_size=img_size, Lambda=lam) for lam in tqdm(lambdas)]
        # rfs = [(rf - rf.mean()) / rf.std() for rf in rfs]

        l_pearsons, nl_pearsons, sigmoid_params = [], [], []
        for rf in rfs:
            rf_eval = rf_evaluation(dset['X_test'], dset['Y_test'][:, i], rf.ravel())
            l_pearson, nl_pearson, params = rf_eval.evaluate()
            sigmoid_params.append(params)
            l_pearsons.append(l_pearson)
            nl_pearsons.append(nl_pearson)
        
        self.insert1({**key, 'best_l_rf': rfs[np.argmax(l_pearsons).item()], 'best_l_pearson': np.max(l_pearsons),
                      'best_nl_rf': rfs[np.nanargmax(nl_pearsons).item()], 'best_nl_pearson': np.nanmax(nl_pearsons),
                      'rfs': rfs, 'sigmoid_params': sigmoid_params, 
                      'l_pearsons': l_pearsons, 'nl_pearsons': nl_pearsons})

    # X: stimuli (n_stimuli * n_pixels) Y: responses from one neuron (n_stimuli,)
    def compute_rf(self, X, Y, img_size=(36, 64), Lambda=10):
        assert X.shape[-1] == np.prod(img_size)
        reg = Lambda * self.create_laplacian_matrix(img_size)
        Y_reg = np.concatenate([Y, np.zeros(len(reg))])
        X_reg = np.concatenate([X, reg],axis=0)
        rf = self.solve_rf(X_reg, Y_reg).reshape(img_size)
        return rf
    
    # Create padding then delete
    @staticmethod
    def create_laplacian_matrix(img_size):
        overall_matrix = np.empty((np.prod(img_size), np.prod(img_size)))
        counter = 0
        # Choose a top_left
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                temp = np.zeros([i+2 for i in img_size])
                temp[i:i+3, j:j+3] = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
                temp = temp[1:-1, 1:-1]
                overall_matrix[counter] = temp.ravel()
                counter = counter + 1
        return overall_matrix

    @staticmethod
    def solve_rf(X_reg, Y_reg):
        U, S, Vh = np.linalg.svd(X_reg, full_matrices=False)
        X_reg_inv = np.dot(np.dot(Vh.conj().T, np.diag(1/(S+1e-9))), U.T)
        rf = np.dot(X_reg_inv, Y_reg)
        return rf
    

# ------------------------------------ Reduced-rank regression linear RF from natural image responses --------------------------------------
@schema 
class ReducedRankParams(dj.Lookup):
    definition = """
    rrr_params          : int
    ---
    n_pc                : int
    n_rank              : int
    lambda              : float
    """
    contents = [(0, 100, 25, 5e-3)]
    
@schema 
class ReducedRankRF(dj.Computed):
    definition = """
    -> StaticScan
    -> data_schemas.Preprocessing
    -> ReducedRankParams
    ---
    """
    class Unit(dj.Part):
        definition = """
        -> master
        -> fuse.Activity.Trace
        ---
        rf                  : longblob  # linear rf 
        l_pearson           : float     # pearson between gt and linearly predicted response
        nl_pearson          : float     # between gt and non-linearly predicted response
        sigmoid_params      : longblob  # fitted paramters from the nonlinear sigmoid function
        """
        
    @property
    def key_source(self):
        return StaticScan * data_schemas.Preprocessing * ReducedRankParams & 'preproc_id = 9'

    def make(self, key):
        self.insert1(key)
        img_size = (data_schemas.Preprocessing & key).fetch1('row', 'col')
        n_pc, n_rank, lam = (ReducedRankParams & key).fetch1('n_pc', 'n_rank', 'lambda')
        scan_key = (StaticMultiDataset.Member & key).fetch(dj.key, limit=1)[0]
        dset = get_dataset_helper(scan_key, 'rrr')
        
        # Reconstruct Y from n_pc top PCs
        _U_Y, _S_Y, V_Y = np.linalg.svd(np.round(dset['Y_train'], 4))
        reduced_Y = np.dot(_U_Y[:, :n_pc], np.diag(_S_Y)[:n_pc, :n_pc])
        
        # Reduced-rank regression
        rrr = ReducedRankRegressor(dset['X_train'], reduced_Y, n_rank, lam)
        rfs = np.dot(np.dot(rrr.b[:, :n_rank], rrr.a[:, :n_rank].T), V_Y[:n_pc, :])
        rfs_real = rfs.real
        
        # Evaluation of the RF fits
        rf_eval = rf_evaluation(dset['X_test'], dset['Y_test'], rfs_real)
        l_pearson, nl_pearson, params = rf_eval.evaluate()        
        tuples = []
        for uid, rf, _l_pearson, _nl_pearson, _params in tqdm(zip(dset['unit_ids'], rfs_real.T, l_pearson, nl_pearson, params)):
            unit_key = (fuse.Activity.Trace & key & {'unit_id': uid}).fetch1()
            tuples.append({**key, **unit_key, 'rf': rf.reshape(*img_size),
                          'l_pearson': _l_pearson, 'nl_pearson': _nl_pearson, 'sigmoid_params': np.array(_params)})
        self.Unit.insert(tuples)


# ---------------------------------------- Utility functions for natural image linear RF fit ---------------------------------------
def get_dataset_helper(key, method): #, valid_only=True):
    dsets = StaticMultiDataset().fetch_data(key)
    dset = list(dsets.values())[0]  # if there is only one scan in this dataset (usually this is the case)
    unit_ids = dset.neurons.unit_ids
    Y = dset.responses
    X = dset.images.squeeze().reshape(Y.shape[0], -1)

    if method == "reg_pseudoinv":
        img_mean = dset.statistics['images/{}/mean'.format('all')][()]
        img_std = dset.statistics['images/{}/std'.format('all')][()]
        resp_mean = dset.statistics['responses/{}/mean'.format('all')][()]
        resp_std = dset.statistics['responses/{}/std'.format('all')][()]

        X = (X - img_mean) / img_std
        Y = (Y - resp_mean) / resp_std
    
    elif method == 'rrr':
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y = Y - Y.mean(0, keepdims=True) 

    # if valid_only:
    #     valid_eye = (data_schemas.Eye & key).fetch1('valid')
    #     valid_treadmill = (data_schemas.Treadmill & key).fetch1('valid')            
    #     X = X[valid_eye & valid_treadmill]
    #     Y = Y[valid_eye & valid_treadmill]

    X_train = X[dset.tiers == "train"]
    Y_train = Y[dset.tiers == "train"]    

    test_hashes = dset.condition_hashes[dset.tiers=='test']  # find the oracle condition hashes
    Y_test = Y[dset.tiers=='test']
    X_test = X[dset.tiers=='test']

    # group responses according to hashes 
    Y_test_dict = defaultdict(list)
    X_test_dict = {}
    for h, r, img in zip(test_hashes, Y_test, X_test):
        Y_test_dict[h].append(r)
        X_test_dict[h] = img
    Y_test = np.stack([np.stack(v).mean(axis=0) for v in Y_test_dict.values()])  # number of oracle images x neurons
    X_test = np.stack([v for v in X_test_dict.values()])

    return {'unit_ids': unit_ids, 'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test, 'Y_test': Y_test}

class rf_evaluation(object):
    def __init__(self, X, Y, rf):
        # X: num_images * num_pixels
        # Y: num_images * num_neurons or num_images,
        # rf: num_pixels * num_neurons or num_pixels,
        self.X = X
        self.Y = Y
        self.rf = rf
    
    def evaluate(self):
        from scipy.stats import pearsonr
        from scipy import optimize
        l_r = self.linear_response()
        if len(self.rf.shape) == 1:
            l_pearson = pearsonr(self.Y, l_r)[0]
            # fit nonlinear parameters
            try:
                popt, _ = optimize.curve_fit(self.sigmoid_f, l_r, self.Y, maxfev=1000000000)
                nl_pearson = pearsonr(self.Y, self.sigmoid_f(l_r, *popt))[0]
            except RuntimeError:
                popt = nl_pearson = np.nan
        else:
            l_pearson, nl_pearson, popt = [], [], []
            for idx in tqdm(range(l_r.shape[-1])):
                _l_pearson = pearsonr(self.Y[:, idx], l_r[:, idx])[0]
                # fit nonlinear parameters
                try:
                    _popt, _ = optimize.curve_fit(self.sigmoid_f, l_r[:, idx], self.Y[:, idx], maxfev=1000000000)
                    _nl_pearson = pearsonr(self.Y[:, idx], self.sigmoid_f(l_r[:, idx], *_popt))[0]
                except RuntimeError:
                    _popt = _nl_pearson = np.nan
                l_pearson.append(_l_pearson)
                nl_pearson.append(_nl_pearson)
                popt.append(_popt)
        return l_pearson, nl_pearson, popt

    def linear_response(self):
        if len(self.rf.shape) == 1:
            return (self.X * self.rf[None]).sum(axis=-1)
        else:
            return self.X @ self.rf
        
    @staticmethod
    def sigmoid_f(r, A, alpha, beta):
        return A / (1 + np.exp(r * (-alpha) + beta))

"""
Reduced rank regression class.
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Optimal linear 'bottlenecking' or 'multitask learning'.

code reference: https://github.com/riscy/machine_learning_linear_models/blob/master/reduced_rank_regressor.py
maths reference: https://andrewcharlesjones.github.io/journal/reduced-rank-regression.html
adapted from Stringer et al.2018: https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/79850dba7a4e66e213245105c00131f4d6c84e03/fitRFs/fitLowRankRFs.m
"""
class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, X, Y, rank, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0
        self.rank = rank
        
        xsize = X.shape[1]
        bigcov = np.cov(np.concatenate((X, Y), axis=1).T)
        CXX = bigcov[:xsize, :xsize] + reg * np.eye(xsize)
        CYX = bigcov[xsize:, :xsize]
        from scipy.linalg import fractional_matrix_power
        CXXMH = fractional_matrix_power(CXX, -0.5)
        M = np.dot(CYX, CXXMH)
        M[np.isnan(M)] = 0
        
        _U, _S, V = np.linalg.svd(M, full_matrices=False)
        _S = np.diag(_S)
        b = np.dot(CXXMH, V.T)
        if np.isnan(_U[:, -1]).sum() > 0:
            _U = _U[:, :-1]
            _S = _S[:-1, :-1]
            b = b[:, :-1]
        a = np.dot(_U, _S)
        self.a = a
        self.b = b
        

# -------------------------------------- Utility functions for Dotmapping reverse correlation RF fit ---------------------------------------
from math import *
import numpy as np
from scipy import optimize, interpolate

def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)

    #     center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    #     center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x, y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height * np.exp(
            -(((center_x - xp) / width_x) ** 2 +
              ((center_y - yp) / width_y) ** 2) / 2.)
        return g

    return rotgauss


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(data - gaussian(*p)(*np.indices(data.shape)))
    p, success = optimize.leastsq(errorfunction, params, factor=1)

    return p


def rotate_center(height, center_x, center_y, width_x, width_y, rotation):
    rot = np.deg2rad(rotation)
    rmat = np.array([[cos(rot), sin(rot)], [-sin(rot), cos(rot)]])
    center_y, center_x = (np.array([center_y, center_x]) @ rmat)
    return height, center_x, center_y, width_x, width_y, rotation


def fit_rf(rmap, dot_size):
    """

    :param rmap: raw response map same size as stimuli
    :param dot_size: size of dots fmap pixel unit.
    :return: fitted parameters in fmap pixel unit.
    """
    map_size = np.array(rmap.shape)
    target_size = np.ceil((map_size - 1) * (dot_size * 120) + 1).astype(int)
    target_x = np.linspace(0, 1, target_size[0])
    target_y = np.linspace(0, 1, target_size[1])
    map_x = np.linspace(0, 1, map_size[0])
    map_y = np.linspace(0, 1, map_size[1])
    grid = np.meshgrid(map_x, map_y)

    # ---interpolate and smoothen---
    tck = interpolate.bisplrep(grid[0], grid[1], rmap.T)
    fmap = interpolate.bisplev(target_x, target_y, tck)

    # ---threshold at 90 percentile---
    # thres = fmap.mean() + 2 * np.std(fmap)
    thres = np.percentile(fmap, 90)
    fmap[fmap < thres] = 0

    # ---fit---
    params = fitgaussian(fmap.T)
    fit = gaussian(*params)
    (_, center_x, center_y, _, _, _) = rotate_center(*params)
    return fmap, fit, params, center_x, center_y


def compute_rf(height, center_x, center_y, width_x, width_y, rot, sd):
    params = rotate_center(height, center_x, center_y, width_x, width_y, rot)
    (height, center_x, center_y, width_x, width_y, rot) = params
    phi = np.arange(0, 2 * np.pi, 0.01)
    ciry = sd * np.cos(phi) * width_y
    cirx = sd * np.sin(phi) * width_x

    # add rotation
    rot = np.deg2rad(rot)
    rmat = np.array([[cos(rot), sin(rot)], [-sin(rot), cos(rot)]])
    newy, newx = (np.array([ciry, cirx]).T @ rmat).T
    cirx = newx + center_x  # height
    ciry = newy + center_y  # width

    return cirx, ciry


def convert_unit(x, y, fmap, xrange):
    """
    :return: x from -0.5 to 0.5, y from -0.0.2941 to 0.2941
    :param fmap: interpolated map fitted on
    :param x: original height in pixel units
    :param y: original width in pixel units

    """
    nx, ny = fmap.shape
    unit_x = (x / (nx - 1) - 0.5) * xrange
    unit_y = (y - (ny - 1) / 2) / (nx - 1) * xrange

    return unit_x, unit_y


def convert2map(unit_x, unit_y, rmap, xrange=1, pad_size=None):  # inverse function of conver_unit.
    nx, ny = rmap.shape
    newx = unit_x * (nx - 1) / xrange + 0.5 * (nx - 1)
    newy = unit_y * (nx - 1) / xrange + 0.5 * (ny - 1)

    if pad_size:
        xpad, ypad = pad_size
        newx += xpad
        newy += ypad

    return newx, newy


def pad_rmap(rmap, xrange):
    x_step = xrange / (rmap.shape[0] - 1)
    half_xsize = int((1 / x_step + 1) / 2)

    aspect = 1.7
    y_step = xrange * rmap.shape[1] / rmap.shape[0] / (rmap.shape[1] - 1)
    half_ysize = int((1 / aspect / y_step + 1) / 2)

    new_map = np.concatenate([np.zeros((half_xsize, rmap.shape[1])), rmap, np.zeros((half_xsize, rmap.shape[1]))], 0)
    new_map = np.concatenate(
        [np.zeros((new_map.shape[0], half_ysize)), new_map, np.zeros((new_map.shape[0], half_ysize))], 1)
    return new_map, half_xsize, half_ysize


def smooth_combine(fmap, percent=90):
    """
    fmap: off and on maps.
    """
    dmap = (fmap[0] / fmap[0].max() + fmap[1] / fmap[1].max()) / 2
    threshold = np.percentile(dmap, percent)
    dmap = dmap > threshold
    return dmap

