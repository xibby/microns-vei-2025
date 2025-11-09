import numpy as np
import datajoint as dj 
import warnings
from staticnet_analyses import closed_loop, base#, zd_model_base as zd_base
from staticnet_invariance import deis_reconfigured as deis_schema, dei_stimulus_reconfigured as dei_stimulus
from neuro_data.static_images import data_schemas, configs as neuro_configs
from neuro_data.utils.measures import corr
from staticnet_experiments import models as static_models
from staticnet_experiments import configs
from featurevis import models, ops
from functools import partial
from scipy import ndimage
import torch
from torch.nn import functional as F
from matplotlib.patches import Circle

fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')
reso = dj.create_virtual_module('reso', 'pipeline_reso')
meso = dj.create_virtual_module('meso', 'pipeline_meso')
stack = dj.create_virtual_module('stack', 'pipeline_stack')

stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')

schema = dj.schema('neurostatic_pilot_analyses')

# Matching of cells is done in staticnet_analyses closed_loop.py: ClosedLoopScan->ProximityCellMatch->BestProximityCelllMatch

@schema
class StimSubtype(dj.Lookup):
    definition = """  # for each StimType, record how stimuli is divided into different groups and the table query for each group

    -> closed_loop.StimType
    subtype    :varchar(16)           # group name
    ---
    table         :varchar(1024)           # table query to fetch the stimuli for a certain group
    """
    contents = [{'stim_type': 'dei_multi_area', 'subtype': 'v1_diverse1', 'table': "base.Dataset.Unit.proj('brain_area') * stimulus.StaticImage.DiverseMEI & 'mei_params in (3, 4) and diverse_params = 27 and loop_params = 12 and mei_id = 0' & [dict(experiment='diverse_meis'), dict(experiment='diverse_meis_1')] & dict(brain_area='V1')"},
          {'stim_type': 'dei_multi_area', 'subtype': 'v1_diverse2', 'table': "base.Dataset.Unit.proj('brain_area') * stimulus.StaticImage.DiverseMEI & 'mei_params in (3, 4) and diverse_params = 27 and loop_params = 12 and mei_id = 1' & [dict(experiment='diverse_meis'), dict(experiment='diverse_meis_1')] & dict(brain_area='V1')"},
          {'stim_type': 'dei_multi_area', 'subtype': 'hva_diverse1', 'table': "base.Dataset.Unit.proj('brain_area') * stimulus.StaticImage.DiverseMEI & 'mei_params in (3, 4) and diverse_params = 27 and loop_params = 12 and mei_id = 0' & [dict(experiment='diverse_meis'), dict(experiment='diverse_meis_1')] & [dict(brain_area='LM'), dict(brain_area='AL'), dict(brain_area='RL'), dict(brain_area='PM')]"},
          {'stim_type': 'dei_multi_area', 'subtype': 'hva_diverse2', 'table': "base.Dataset.Unit.proj('brain_area') * stimulus.StaticImage.DiverseMEI & 'mei_params in (3, 4) and diverse_params = 27 and loop_params = 12 and mei_id = 1' & [dict(experiment='diverse_meis'), dict(experiment='diverse_meis_1')] & [dict(brain_area='LM'), dict(brain_area='AL'), dict(brain_area='RL'), dict(brain_area='PM')]"},
          {'stim_type': 'dei_multi_area', 'subtype': 'v1_mei', 'table': "base.Dataset.Unit.proj('brain_area') * stimulus.StaticImage.MEI2 & 'mei_params in (3, 4)' & [dict(experiment='diverse_meis'), dict(experiment='diverse_meis_1')] & dict(brain_area='V1')"},
          {'stim_type': 'dei_multi_area', 'subtype': 'hva_mei', 'table': "base.Dataset.Unit.proj('brain_area') * stimulus.StaticImage.MEI2 & 'mei_params in (3, 4)' & [dict(experiment='diverse_meis'), dict(experiment='diverse_meis_1')] & [dict(brain_area='LM'), dict(brain_area='AL'), dict(brain_area='RL'), dict(brain_area='PM')]"},
          {'stim_type': 'dei_multi_area', 'subtype': 'hva_oracle', 'table': "stimulus.StaticImage.ImageNet"},
          {'stim_type': 'dei_multi_area', 'subtype': 'v1_oracle', 'table': "stimulus.StaticImage.ImageNet"},
        ('actlearn_dei', 'diverse1', "stimulus.StaticImage.ActlearnDEI & {'experiment': 'actlearn_dei', 'mei_params': 3, 'diverse_params': 1, 'loop_params': 7, 'mei_id': 0}"),
        ('actlearn_dei', 'diverse2', "stimulus.StaticImage.ActlearnDEI & {'experiment': 'actlearn_dei', 'mei_params': 3, 'diverse_params': 1, 'loop_params': 7, 'mei_id': 1}"),
        ('actlearn_dei', 'mei2', "stimulus.StaticImage.MEI2 & {'experiment': 'actlearn_dei', 'mei_params': 3}"),
        ('actlearn_dei', 'oracle', "stimulus.StaticImage.ImageNet"),
        ('blurdei_pixel', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 1 and loop_params = 7 and mei_params = 5 and mei_id = 0'"),
        ('blurdei_pixel', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 1 and loop_params = 7 and mei_params = 5 and mei_id = 1'"),
        ('blurdei_pixel', 'mei2', "stimulus.StaticImage.MEI2 & 'mei_params = 5'"),
        ('blurdei_pixel', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('blurdei_pixel2', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 1 and loop_params = 7 and mei_params = 3 and mei_id = 0'"),
        ('blurdei_pixel2', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 1 and loop_params = 7 and mei_params = 3 and mei_id = 1'"),
        ('blurdei_pixel2', 'mei2', "stimulus.StaticImage.MEI2 & {'experiment': 'diverse_meis', 'mei_params': 3}"),
        ('blurdei_pixel2', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('pixeldei_loop9', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 1 and loop_params = 9 and mei_params = 3 and mei_id = 0'"),
        ('pixeldei_loop9', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 1 and loop_params = 9 and mei_params = 3 and mei_id = 1'"),
        ('pixeldei_loop9', 'mei2', "stimulus.StaticImage.MEI2 & {'experiment': 'diverse_meis', 'mei_params': 3}"),
        ('pixeldei_loop9', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('pixeldei_loop10', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 26 and loop_params = 10 and mei_params = 3 and mei_id = 0'"),
        ('pixeldei_loop10', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 26 and loop_params = 10 and mei_params = 3 and mei_id = 1'"),
        ('pixeldei_loop10', 'mei2', "stimulus.StaticImage.MEI2 & {'experiment': 'diverse_meis', 'mei_params': 3}"),
        ('pixeldei_loop10', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('divmeiHVpixel', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 1 and loop_params = 4 and mei_id = 0'"),
        ('divmeiHVpixel', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 1 and loop_params = 4 and mei_id = 1'"),
        ('divmeiHVpixel', 'mei2', 'stimulus.StaticImage.MEI2'),
        ('divmeiHVpixel', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('divmeiV1pixel', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 8 and loop_params = 2 and mei_id = 0'"),
        ('divmeiV1pixel', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 8 and loop_params = 2 and mei_id = 1'"),
        ('divmeiV1pixel', 'mei2', 'stimulus.StaticImage.MEI2'),
        ('divmeiV1pixel', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('divmeiV1vae', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 12 and loop_params = 3 and mei_id = 0'"),
        ('divmeiV1vae', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 12 and loop_params = 3 and mei_id = 1'"),
        ('divmeiV1vae', 'mei2', 'stimulus.StaticImage.MEI2'),
        ('divmeiV1vae', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('divmeiV1pixelv2', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 19 and loop_params = 5 and mei_id = 0'"),
        ('divmeiV1pixelv2', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 19 and loop_params = 5 and mei_id = 1'"),
        ('divmeiV1pixelv2', 'mei2', 'stimulus.StaticImage.MEI2'),
        ('divmeiV1pixelv2', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('divmeiV1vaev2', 'diverse1', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 20 and loop_params = 5 and mei_id = 0'"),
        ('divmeiV1vaev2', 'diverse2', "stimulus.StaticImage.DiverseMEI & 'diverse_params = 20 and loop_params = 5 and mei_id = 1'"),
        ('divmeiV1vaev2', 'mei2', 'stimulus.StaticImage.MEI2'),
        ('divmeiV1vaev2', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('lei', 'lei', 'stimulus.StaticImage.LEI'),
        ('lei', 'mei2', 'stimulus.StaticImage.MEI2'),
        ('lei', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('maskunmaskmei', 'masked_mei2', 'stimulus.StaticImage.MaskedMEI2'),
        ('maskunmaskmei', 'mei2', 'stimulus.StaticImage.MEI2'),
        ('maskunmaskmei', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('maskunmasknat', 'masked_image2', 'stimulus.StaticImage.MaskedImageNet2'),
        ('maskunmasknat', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('maskunmasknat', 'unmasked_image2', 'stimulus.StaticImage.UnmaskedImageNet2'),
        ('meidiffblur', 'blur1', "stimulus.StaticImage.BlurMEI & {'mei_params': 1, 'experiment': 'blur_mei_1'}"),
        ('meidiffblur', 'blur2', "stimulus.StaticImage.BlurMEI & {'mei_params': 2, 'experiment': 'blur_mei_1'}"),
        ('meidiffblur', 'blur3', "stimulus.StaticImage.BlurMEI & {'mei_params': 3, 'experiment': 'blur_mei_1'}"),
        ('meidiffblur', 'blur4', "stimulus.StaticImage.BlurMEI & {'mei_params': 4, 'experiment': 'blur_mei_1'}"),
        ('meidiffblur', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('meidiffbluract', '85', "stimulus.StaticImage.BlurMEIActThre & {'act_threshold': 0.85, 'experiment': 'blur_mei_act'}"),
        ('meidiffbluract', '95', "stimulus.StaticImage.BlurMEIActThre & {'act_threshold': 0.95, 'experiment': 'blur_mei_act'}"),
        ('meidiffbluract', 'noblur', "stimulus.StaticImage.BlurMEIActThre & {'act_threshold': 0, 'experiment': 'blur_mei_act'}"),
        ('meidiffbluract', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('maskunmasknat', 'unmasked_image2', 'stimulus.StaticImage.UnmaskedImageNet2'),
        ('maskunmasknat', 'masked_image2', 'stimulus.StaticImage.MaskedImageNet2'),
        ('maskunmasknat', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('meiversions', 'rf', "stimulus.StaticImage.MultiMEI & {'image_class': 'multi_lin_rf'}"),
        ('meiversions', 'mei', "stimulus.StaticImage.MultiMEI & {'image_class': 'multi_cnn_mei'}"),
        ('meiversions', 'rf2', 'stimulus.StaticImage.LinRF2'),
        ('meiversions', 'mei2', 'stimulus.StaticImage.MEI2'),
        ('meiversions', 'oracle', 'stimulus.StaticImage.ImageNet'),
        ('vaevspixel', 'vae_mei', "stimulus.StaticImage.VAEMEI & 'mei_params = 3 and mei_vae_params = 2'"),
        ('vaevspixel', 'mei2', "stimulus.StaticImage.MEI2 & {'experiment': 'vaevspixel', 'mei_params': 3}"),
        ('vaevspixel', 'oracle', "stimulus.StaticImage.ImageNet"),
        ('pxdei_params37', 'diverse1', "stimulus.StaticImage.DiverseMEI & {'experiment': 'diverse_meis', 'diverse_params': 37, 'stim_params': 3, 'mei_params': 8, 'mask_params': 7, 'mei_id': 0}"),
        ('pxdei_params37', 'diverse2', "stimulus.StaticImage.DiverseMEI & {'experiment': 'diverse_meis', 'diverse_params': 37, 'stim_params': 3, 'mei_params': 8, 'mask_params': 7, 'mei_id': 1}"),
        ('pxdei_params37', 'mei2', "stimulus.StaticImage.MEI2 & {'experiment': 'diverse_meis', 'stim_params': 3, 'mei_params': 8, 'mask_params': 7}"),
        ('pxdei_params37', 'oracle', "stimulus.StaticImage.ImageNet"),
        ('natvsmei_dei', 'mei_dei1', "stimulus.StaticImage.DiverseMEI & {'experiment': 'diverse_meis', 'diverse_params': 37, 'group_id': 186, 'stim_params': 4, 'mei_params': 8, 'mask_params': 7, 'mei_id': 0}"),
        ('natvsmei_dei', 'mei_dei2', "stimulus.StaticImage.DiverseMEI & {'experiment': 'diverse_meis', 'diverse_params': 37, 'group_id': 186, 'stim_params': 4, 'mei_params': 8, 'mask_params': 7, 'mei_id': 1}"),
        ('natvsmei_dei', 'nat_dei1', "stimulus.StaticImage.DiverseMEI & {'experiment': 'diverse_meis', 'diverse_params': 37, 'group_id': 187, 'stim_params': 4, 'mei_params': 8, 'mask_params': 7, 'mei_id': 0}"),
        ('natvsmei_dei', 'nat_dei2', "stimulus.StaticImage.DiverseMEI & {'experiment': 'diverse_meis', 'diverse_params': 37, 'group_id': 187, 'stim_params': 4, 'mei_params': 8, 'mask_params': 7, 'mei_id': 1}"),
        ('natvsmei_dei', 'mei_mei', "stimulus.StaticImage.MEI2 & {'experiment': 'diverse_meis', 'group_id': 186, 'stim_params': 4, 'mei_params': 8, 'mask_params': 7}"),
        ('natvsmei_dei', 'nat_mei', "stimulus.StaticImage.MEI2 & {'experiment': 'diverse_meis', 'group_id': 187, 'stim_params': 4, 'mei_params': 8, 'mask_params': 7}"),
        ('natvsmei_dei', 'oracle', "stimulus.StaticImage.ImageNet"),
        ('natvsmei_dei','oracle_mei', "stimulus.StaticImage.MEIOracle & {'collection_id': 2}"),
        ('meiversion3', 'deepdraw', "stimulus.StaticImage.MEI2 & [{'experiment': 'mei_version3'}, {'experiment': 'mei_version3_2'}]"),
        ('meiversion3', 'fv_fixblur', "stimulus.StaticImage.BlurMEI &  [{'experiment': 'mei_version3'}, {'experiment': 'mei_version3_2'}]"),
        ('meiversion3', 'fv_adapblur', "stimulus.StaticImage.BlurMEIActThre & [{'experiment': 'mei_version3'}, {'experiment': 'mei_version3_2'}]"),
        ('meiversion3', 'oracle', "stimulus.StaticImage.ImageNet"),
        ('pm_mei', 'oracle', "stimulus.StaticImage.ImageNet"),
        ('pm_mei', 'pm_com', "stimulus.StaticImage.MEI2 & {'experiment': 'PMcom'}"),
        ('pm_mei', 'pm_sep', "stimulus.StaticImage.MEI2 & {'experiment': 'PMsep'}"),
        ('pm_mei_1', 'oracle', "stimulus.StaticImage.ImageNet"),
        ('pm_mei_1', 'pm_com', "stimulus.StaticImage.MEI2 & [{'experiment': 'PMcom1'}, {'experiment': 'PMcom2'}]"),
        ('pm_mei_1', 'pm_sep', "stimulus.StaticImage.MEI2 & [{'experiment': 'PMsep1'}, {'experiment': 'PMsep2'}]"),
        ]

    def get_stim_tables(self, order_by='subtype'):
        """ Return a list of subtypes (strings) and a list of stimulus tables (datajoint user tables) for a single StimType"""
        groups, table_strs = self.fetch('subtype','table', order_by=order_by)
        tables = [eval(table_str) for table_str in table_strs]  # TODO: try not use eval for this
        return groups, tables

@schema
class ExcludedScan(dj.Manual):
    definition = """  # scans excluded from analyses
    -> closed_loop.ClosedLoopScan
    ---
    exclusion_comment='': varchar(255)   # reasons for exclusion
    """
    

#TODO: add a method to exclude problematic scans
@schema
class Scan(dj.Computed):
    definition = """  # a scan where our stimulus of interest was presented

    -> fuse.ScanDone
    -> closed_loop.ClosedLoopScan
    """

    class Frame(dj.Part):
        definition = """ # a single image shown during scanning
        -> master
        -> stimulus.Frame               # condition shown during scan
        ---
        -> stimulus.StaticImage.Image    # image corresponding to the condition above (1:1 mapping)
        """

    class Unit(dj.Part):
        definition = """ # a single unit in the scan (restricted to somas and right segmentation)
          -> master        
          unit_id              : int                          # unique per scan & segmentation method
          ---
          -> fuse.ScanSet.Unit
          """

    key_source = (fuse.ScanDone() - ExcludedScan) * closed_loop.ClosedLoopScan & 'spike_method in (5,6) AND segmentation_method=6' & StimSubtype

    def make(self, key):
        for scan_key in (fuse.ScanDone & key).fetch('KEY'):
            print(f'Populating{scan_key}')
            self.insert1(scan_key)

            # Insert frames
            _, stim_tables = (StimSubtype & key).get_stim_tables()
            frame_rel = stimulus.Frame & (stimulus.Trial & key) & stim_tables
            hashes, img_classes, img_ids = frame_rel.fetch('condition_hash',
                                                           'image_class', 'image_id')
            for hash, img_class, img_id in zip(hashes, img_classes, img_ids):
                self.Frame.insert1({**scan_key, 'condition_hash': hash,
                                    'image_class': img_class, 'image_id': img_id})

            # Insert units
            pipe_name = (fuse.ScanDone() & key).fetch1('pipe')
            pipe = reso if pipe_name == 'reso' else meso
            self.Unit().insert(
                fuse.ScanDone * pipe.ScanSet.Unit * pipe.MaskClassification.Type &
                key & {'pipe_version': 1, 'segmentation_method': 6,
                       'type': 'soma'}, ignore_extra_fields=True)


def get_traces(key):
    """ Get spike traces for all cells in these scan (along with their times in stimulus
    clock).

    Arguments:
        key (dict): Key for a scan (or field).

    Returns:
        traces (np.array): A (num_units x num_scan_frames) array with all spike traces.
            Traces are restricted to those classified as soma and ordered by unit_id.
        unit_ids (list): A (num_units) list of unit_ids in traces.
        trace_times (np.array): A (num_units x num_scan_frames) array with the time (in
            seconds) for each unit's trace in stimulus clock (same clock as times in
            stimulus.Trial).

    Note: On notation
        What is called a frametime in stimulus.Sync and stimulus.Trial is actually the
        time each depth of scanning started. So for a scan with 1000 frames and four
        depths per frame/volume, there will be 4000 "frametimes".

    Note 2:
        For a scan with 10 depths, a frame i is considered complete if all 10 depths were
        recorded and saved in the tiff file, frame_times however save the starting time of
        each depth independently (for instance if 15 depths were recorded there will be
        one scan frame but 15 frame times, the last 5 have to be ignored).
    """
    # Pick right pipeline for this scan (reso or meso)
    pipe_name = (fuse.ScanDone & key).fetch1('pipe')
    pipe = reso if pipe_name == 'reso' else meso

    # Get traces
    units = pipe.ScanSet.Unit() & key & (pipe.MaskClassification.Type & {'type': 'soma'})
    spikes = pipe.Activity.Trace() * pipe.ScanSet.UnitInfo() & units.proj()
    unit_ids, traces, ms_delays = spikes.fetch('unit_id', 'trace', 'ms_delay',
                                               order_by='unit_id')

    # Get time of each scan frame for this scan (in stimulus clock; same as in Trial)
    if key['animal_id'] == 25312 and key['session'] == 4 and key['scan_idx'] == 15:
        depth_times = np.load('/external/zhiwei/25312-4-15-stimulus_sync_frame_times.npy')
    else:
        depth_times = (stimulus.Sync & key).fetch1('frame_times')
    num_frames = (pipe.ScanInfo & key).fetch1('nframes')
    num_depths = len(dj.U('z') & (pipe.ScanInfo.Field.proj('z', nomatch='field') & key))
    # if len(depth_times) / num_depths < num_frames or (len(depth_times) / num_depths >
    #                                                   num_frames + 1):
    #     raise ValueError('Mismatch between frame times and tiff frames')
    frame_times = depth_times[:num_depths * num_frames:num_depths]  # one per frame

    # Add per-cell delay to each frame_time
    trace_times = np.add.outer(ms_delays / 1000, frame_times)  # num_traces x num_frames

    return np.stack(traces), np.stack(unit_ids), trace_times


def trapezoid_integration(x, y, x0, xf):
    """ Integrate y (recorded at points x) from x0 to xf.

    Arguments:
        x (np.array): Timepoints (num_timepoints) when y was recorded.
        y (np.array): Signal (num_timepoints).
        x0 (float or np.array): Starting point(s). Could be a 1-d array (num_samples).
        xf (float or np.array): Final point. Same shape as x0.

    Returns:
        Integrated signal from x0 to xf:
            a 0-d array (i.e., float) if x0 and xf are floats
            a 1-d array (num_samples) if x0 and xf are 1-d arrays
    """
    # Basic checks
    if np.any(xf <= x0):
        raise ValueError('xf has to be higher than x0')
    if np.any(x0 < x[0]) or np.any(xf > x[-1]):
        raise ValueError('Cannot integrate outside the original range x of the signal.')

    # Compute area under each trapezoid
    trapzs = np.diff(x) * (y[:-1] + y[1:]) / 2  # index i is trapezoid from point i to point i + 1

    # Find timepoints right before x0 and xf
    idx_before_x0 = np.searchsorted(x, x0) - 1
    idx_before_xf = np.searchsorted(x, xf) - 1

    # Compute y at the x0 and xf points
    slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])  # index i is slope from p_i to p_{i+1}
    y0 = y[idx_before_x0] + slopes[idx_before_x0] * (x0 - x[idx_before_x0])
    yf = y[idx_before_xf] + slopes[idx_before_xf] * (xf - x[idx_before_xf])

    # Sum area of all interior trapezoids
    indices = np.stack([idx_before_x0 + 1, idx_before_xf], axis=-1).ravel()  # interleaved x0 and xf for all samples
    integral = np.add.reduceat(trapzs, indices, axis=-1)[::2].squeeze()

    # Add area of edge trapezoids (ones that go from x0 to first_x_sample and from last_x_sample to xf)
    integral += (x[idx_before_x0 + 1] - x0) * (y0 + y[idx_before_x0 + 1]) / 2
    integral += (xf - x[idx_before_xf]) * (y[idx_before_xf] + yf) / 2

    # Deal with edge case where both x0 and xf are in the same trapezoid
    same_trapezoid = idx_before_x0 == idx_before_xf
    integral[same_trapezoid] = ((xf - x0) * (y0 + yf) / 2)[same_trapezoid]

    return integral


#TODO: Put the response matrices in external if the table gets too big
@schema
class Responses(dj.Computed):
    definition = """ # responses from each cell to all (relevant) trials
    
    -> Scan
    ---
    image_resps:        longblob        # responses to all images (num_trials x num_cells)
    blank_resps:        longblob        # responses to blanks before each image (num_trials x num_cells)
    """

    class Trial(dj.Part):
        definition = """ # a single trial in the response block
        -> master
        row_id      :int
        ---
        -> Scan.Frame
        -> stimulus.Trial
        """

    class Unit(dj.Part):
        definition = """ # a single unit in the response block
        -> master
        col_id              : int
        ---
        -> Scan.Unit
        """

    def make(self, key):
        # Get all traces for this scan
        print('Getting traces...')
        traces, unit_ids, trace_times = get_traces(key)

        # Get trial times for frames in Scan.Frame (excluding bad trials)
        print('Getting onset and offset times for each image (and blank)...')
        trials_rel = stimulus.Trial * Scan.Frame - data_schemas.ExcludedTrial & key
        flip_times, trial_ids, cond_hashes = trials_rel.fetch('flip_times', 'trial_idx',
                                                              'condition_hash',
                                                              order_by='trial_idx',
                                                              squeeze=True)
        if any([len(ft) < 2 or len(ft) > 3 for ft in flip_times]):
            raise ValueError('Only works for frames with 2 or 3 flips')

        # Find start and duration of blank and image frames
        monitor_fps = 60
        blank_onset = np.stack([ft[0] for ft in flip_times]) - 1 / monitor_fps  # start of blank period
        image_onset = np.stack([ft[1] for ft in flip_times]) + 1 / monitor_fps  # start of image
        blank_duration = image_onset + 1 / monitor_fps - blank_onset
        image_duration = 0.5  # np.stack([ft[2] for ft in flip_times]) - image_onset
        """
        Each trial is a stimulus.Frame.
        A single stimulus.Frame is composed of a flip (1/60 secs), a blanking period (0.3 
        - 0.5 secs), another flip, the image (0.5 secs) and another flip. During flips 
        screen is gray (as during blanking) so I count the flips before and after the 
        blanking as part of the blanking. There is also another flip after the image and 
        some time between trials (t / 60, t > 0, usually 1) that could be counted as part 
        of the blanking; I ignore those.
        """

        # Add a shift to the onset times to account for the time it takes for the image to
        # travel from the retina to V1
        image_onset += 0.03
        blank_onset += 0.03
        # Wiskott, L. How does our visual system achieve shift and size invariance?. Problems in Systems Neuroscience, 2003.

        # Sample responses (trace by trace) with a rectangular window
        print('Sampling responses...')
        image_resps = np.stack([trapezoid_integration(tt, t, image_onset, image_onset +
                                                      image_duration) / image_duration for
                                tt, t in zip(trace_times, traces)], axis=-1)
        blank_resps = np.stack([trapezoid_integration(tt, t, blank_onset, blank_onset +
                                                      blank_duration) / blank_duration for
                                tt, t in zip(trace_times, traces)], axis=-1)

        # Insert
        print('Inserting...')
        self.insert1({**key, 'image_resps': image_resps.astype(np.float32),
                      'blank_resps': blank_resps.astype(np.float32)})
        self.Unit.insert([{**key, 'unit_id': unit_id, 'col_id': i} for i, unit_id in
                          enumerate(unit_ids)])
        self.Trial.insert([{**key, 'trial_idx': trial_idx, 'condition_hash': cond_hash,
                            'row_id': i} for i, (trial_idx, cond_hash) in enumerate(zip(
            trial_ids, cond_hashes))])


@schema
class ResponseNormalization(dj.Lookup):
    definition = """ # how are responses normalized per scan (so we can compare across cells/scans)
    
    norm_method:        tinyint         # id
    ---
    name:               varchar(16)     # name of the normalization method
    description = "":   varchar(256)    # description of the normalization method
    """
    contents = [
        {'norm_method': 1, 'name': 'blank_std',
         'description': 'divide responses by the standard deviation of blank responses'},
        {'norm_method': 2, 'name': 'blank_zscore',
         'description': 'z-score responses using the mean and standard deviation '
                        'calculated from the responses to blank/gray images'},
        {'norm_method': 3, 'name': 'image_std',
         'description': 'normalize by dividing to the std calculated over all image '
                        'responses'}
    ]


@schema
class ConfusionMatrix(dj.Computed):
    definition = """ # response of each selected cell to the stimulus images for a specific stimulation group
    
    -> Responses
    -> ResponseNormalization
    """
    
    @property
    def key_source(self):
        return (Responses * ResponseNormalization & closed_loop.BestProximityCellMatch &
                {'norm_method': 3})

    class Unit(dj.Part):
        definition = """ # units in current scan matched with source units (two source units could match to the same unit in current scan)
        -> master
        -> StimSubtype
        col_id:          int
        ---
        -> Scan.Unit 
        """

    class Frame(dj.Part):
        definition = """ # frames shown for specific group
        -> master
        -> StimSubtype
        row_id:          int
        ---
        -> Scan.Frame  
        num_trials:      int
        """


    class Matrix(dj.Part):
        definition = """
        -> master
        -> StimSubtype
        ---
        all_response:      longblob        # num_frames x num_cells x num_trials (single trial responses)
        """


    def make(self, key):
        self.insert1(key)

        # Get response block
        response_block = (Responses & key).fetch1('image_resps') # num_trials x num_cells

        # Normalize the response
        if key['norm_method'] == 1:
            blank_resps = (Responses & key).fetch1('blank_resps')
            blank_std = blank_resps.std(axis=0) # per cell
            response_block = response_block / blank_std
        elif key['norm_method'] == 2:
            blank_resps = (Responses & key).fetch1('blank_resps')
            blank_mean =  blank_resps.mean(axis=0) # per cell
            blank_std = blank_resps.std(axis=0) # per cell
            response_block = (response_block - blank_mean) / blank_std
        elif key['norm_method'] == 3:
            image_std = response_block.std(axis=0)  # per cell
            response_block = response_block / image_std
        else:
            raise NotImplementedError('Normalization method {} not recognized'.format(
                key['norm_method']))

        # Get stimulus tables for the subtypes to be compared
        stim_type = (closed_loop.ClosedLoopScan & key).fetch1('stim_type')
        if 'subset' in stim_type:
            base_table = zd_base
        else:
            base_table = base
        subtypes, stim_tables = (StimSubtype & (closed_loop.ClosedLoopScan & key)).get_stim_tables()  

        unit_numbers = [len(base_table.Dataset.Unit.proj('unit_id') & (stim_tables[i] & (Scan.Frame & key))) for i in range(len(stim_tables))]
        non_oracle_units = [u for u, s in zip(unit_numbers, subtypes) if 'oracle' not in s]
        if not non_oracle_units.count(non_oracle_units[0]) == len(non_oracle_units): # if some non-oracle subtypes have fewer units than others
            least_unit_table_idx = np.argmin(unit_numbers)
            overlap_units = (base_table.Dataset.Unit.proj('unit_id') & (stim_tables[least_unit_table_idx] & (Scan.Frame & key))).proj('unit_id', unrel1='neuron_id', unrel2='data_hash')
            group_id = overlap_units.fetch('group_id')[0]
            
        for subtype, stim_table in zip(subtypes, stim_tables):
            stim_key = (StimSubtype & (closed_loop.ClosedLoopScan & key) & dict(subtype=subtype)).fetch1('KEY')
            if 'oracle' in subtype:
                frames = stimulus.Frame & stim_table & (Scan.Frame.proj() & key)
                frame_keys = frames.fetch('condition_hash', order_by='condition_hash')
                non_oracle_tables = [t for t, s in zip(stim_tables, subtypes) if 'oracle' not in s]
                all_units_rel = []
                for rel in non_oracle_tables:
                    all_units_rel += (base_table.Dataset.Unit & (rel & (Scan.Frame & key))).proj()
                source_units = base_table.Dataset.Unit.proj('unit_id') & all_units_rel
            else:
                frames = stimulus.Frame & (stimulus.Trial * Scan.Frame & key & stim_table).proj()
                # frames = stimulus.Frame & (stim_table & (Scan.Frame.proj() & key) & {'group_id': group_id})
                # order by neuron_id of source units so it matches order of unit_keys
                frame_keys = (frames * stim_table).fetch('condition_hash',
                                                            order_by='neuron_id')
                # Find units for which the two image classes to compare were generated
                source_units = base_table.Dataset.Unit.proj('unit_id') & (stim_table & (Scan.Frame & key))
            
            if len(frame_keys) == 0: # skip subtype if there is no frame belonging to this subtype
                warnings.warn('Processing for subtype {} skipped: no frames found'.format(subtype))
                continue
            else:
                # Find unit keys and col_id for the units in the target scan
                unit_match = closed_loop.BestProximityCellMatch * source_units.proj(
                    src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
                if len(dj.U('group_id') & source_units) > 1:
                    # To get unique unit_ids (when there are more than one group_ids involved and potentially multiple occurance of the same unit_id)
                    unit_rel = ['animal_id', 'session', 'scan_idx', 'pipe_version', 'segmentation_method', 'spike_method', 'unit_id', 'col_id']
                    res = (dj.U(*unit_rel) & (dj.U('unit_id') * Responses.Unit * unit_match & key)).fetch(  # dj.U needed to allow joining
                        'animal_id', 'session', 'scan_idx', 'pipe_version', 'segmentation_method',
                        'spike_method', 'unit_id', 'col_id')  # units are randomly ordered!
                else: 
                    res = (dj.U('unit_id') * Responses.Unit * unit_match & key).fetch(  # dj.U needed to allow joining
                                'animal_id', 'session', 'scan_idx', 'pipe_version', 'segmentation_method',
                                'spike_method', 'unit_id', 'col_id',  order_by='neuron_id')  # order_by neuron_id from source units

                unit_keys = [{'animal_id': a, 'session': s, 'scan_idx': si, 'pipe_version': p,
                            'segmentation_method': sm, 'spike_method': spm, 'unit_id': ui} for
                            a, s, si, p, sm, spm, ui in zip(*res[:-1])] # keys in the target scan
                unit_idx = res[-1]
                # Restrict response block to the specified units
                response_block_group = response_block[:, unit_idx]
                
                # Insert unit order
                self.Unit.insert([{**uk, **key, **stim_key, 'col_id': i} for i, uk in enumerate(unit_keys)])

                # Iterate over each frame and get the related reponses

                all_responses = []
                for i, frame_key in enumerate(frame_keys):
                    trial_idx = (Responses.Trial & key & {'condition_hash': frame_key}).fetch('row_id')

                    # Find the related responses
                    all_responses.append(response_block_group[trial_idx].T)

                    # Insert frame key
                    self.Frame.insert1({**key, **stim_key, 'condition_hash': frame_key, 'row_id': i,
                                        'num_trials': len(trial_idx)})
                
                self.Matrix.insert1({**key, **stim_key, 
                                    'all_response': np.stack(all_responses)})

from neuro_data.utils.data import FilterMixin
from neuro_data import logger as log
from itertools import compress

@schema
class InputResponse(dj.Computed, FilterMixin):
    definition = """
    -> fuse.ScanDone
    -> data_schemas.Preprocessing
    ---
    """

    class Input(dj.Part):
        definition = """
            -> master
            -> stimulus.Trial
            -> stimulus.Frame
            ---
            row_id           : int             # row id in the response block
            """

    class ResponseBlock(dj.Part):
        definition = """
            -> master
            ---
            responses           : longblob   # num of trials * num of neurons
            """

    class ResponseKeys(dj.Part):
        definition = """
            -> master.ResponseBlock
            -> fuse.Activity.Trace
            ---
            col_id           : int             # col id in the response block
            """

    def make(self, scan_key):
        self.insert1(scan_key)
        # integration window size for responses
        duration, offset = map(float, (data_schemas.Preprocessing() & scan_key).fetch1('duration', 'offset'))
        sample_point = offset + duration / 2

        log.info('Sampling neural responses at {}s intervals'.format(duration))

        trace_spline, trace_keys, ftmin, ftmax = data_schemas.InputResponse().get_trace_spline(scan_key, duration)
        # exclude trials marked in ExcludedTrial
        log.info('Excluding {} trials based on ExcludedTrial'.format(len(data_schemas.ExcludedTrial() & scan_key)))
        flip_times, trial_keys = (stimulus.Frame * (stimulus.Trial - data_schemas.ExcludedTrial) & scan_key).fetch('flip_times', dj.key,
                                                                           order_by='trial_idx')
        flip_times = [ft.squeeze() for ft in flip_times]

        # If no Frames are present, skip this scan
        if len(flip_times) == 0:
            log.warning('No static frames were present to be processed for {}'.format(scan_key))
            return

        valid = np.array([ft.min() >= ftmin and ft.max() <= ftmax for ft in flip_times], dtype=bool)
        if not np.all(valid):
            log.warning('Dropping {} trials with dropped frames or flips outside the recording interval'.format(
                (~valid).sum()))

        stimulus_onset = data_schemas.InputResponse.stimulus_onset(flip_times, duration)
        log.info('Sampling {} responses {}s after stimulus onset'.format(valid.sum(), sample_point))
        R = trace_spline(stimulus_onset[valid] + sample_point, log=True).T

        self.ResponseBlock.insert1(dict(scan_key, responses=R))
        self.ResponseKeys.insert([dict(scan_key, **trace_key, col_id=i) for i, trace_key in enumerate(trace_keys)])
        self.Input.insert([dict(scan_key, **trial_key, row_id=i)
                           for i, trial_key in enumerate(compress(trial_keys, valid))])


@schema
class NewConfusionMatrix(dj.Computed):
    definition = """ # response of each selected cell to the stimulus images for a specific stimulation group
    -> InputResponse
    -> ResponseNormalization
    """
    
    @property
    def key_source(self):
        return (InputResponse * ResponseNormalization & closed_loop.BestProximityCellMatch &
                {'norm_method': 3})

    class Unit(dj.Part):
        definition = """ # units in current scan matched with source units (two source units could match to the same unit in current scan)
        -> master
        -> StimSubtype
        col_id:          int
        ---
        -> Scan.Unit 
        """

    class Frame(dj.Part):
        definition = """ # frames shown for specific group
        -> master
        -> StimSubtype
        row_id:          int
        ---
        -> Scan.Frame  
        num_trials:      int
        """


    class Matrix(dj.Part):
        definition = """
        -> master
        -> StimSubtype
        ---
        all_response:      longblob        # num_frames x num_cells x num_trials (single trial responses)
        """


    def make(self, key):
        self.insert1(key)

        response_block = (InputResponse.ResponseBlock & key).fetch1('responses') # num_trials x num_cells

        # Normalize the response
        if key['norm_method'] == 1:
            blank_resps = (Responses & key).fetch1('blank_resps')
            blank_std = blank_resps.std(axis=0) # per cell
            response_block = response_block / blank_std
        elif key['norm_method'] == 2:
            blank_resps = (Responses & key).fetch1('blank_resps')
            blank_mean =  blank_resps.mean(axis=0) # per cell
            blank_std = blank_resps.std(axis=0) # per cell
            response_block = (response_block - blank_mean) / blank_std
        elif key['norm_method'] == 3:
            image_std = response_block.std(axis=0)  # per cell
            response_block = response_block / image_std
        else:
            raise NotImplementedError('Normalization method {} not recognized'.format(
                key['norm_method']))

        stim_type = (closed_loop.ClosedLoopScan & key).fetch1('stim_type')
        if 'subset' in stim_type:
            base_table = zd_base
        else:
            base_table = base
        subtypes, stim_tables = (StimSubtype & (closed_loop.ClosedLoopScan & key)).get_stim_tables()  

        # TODO: fix this part to deal with different number of neurons used for different stimulus subtypes
        # non_oracle_units = [len(base_table.Dataset.Unit.proj('group_id', 'neuron_id') & (stim_table & (Scan.Frame & key))) for s, stim_table in zip(subtypes, stim_tables) if "oracle" not in s]
        # if not non_oracle_units.count(non_oracle_units[0]) == len(non_oracle_units): # if some non-oracle subtypes have fewer units than others
        #     least_unit_table_idx = np.argmin(non_oracle_units)
        #     overlap_units = (base_table.Dataset.Unit.proj('group_id', 'neuron_id') & (stim_tables[least_unit_table_idx] & (Scan.Frame & key))).proj('unit_id', unrel1='neuron_id', unrel2='data_hash')
        #     group_id = overlap_units.fetch('group_id')[0]
            
        for subtype, stim_table in zip(subtypes, stim_tables):
            stim_key = (StimSubtype & (closed_loop.ClosedLoopScan & key) & dict(subtype=subtype)).fetch1('KEY')
            unit_rel = base_table.Dataset.Unit.proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')

            if 'oracle' in subtype:
                frames = stimulus.Frame & stim_table & (Scan.Frame.proj() & key)
                frame_keys = frames.fetch('condition_hash', order_by='condition_hash')
                non_oracle_tables = [t for t, s in zip(stim_tables, subtypes) if 'oracle' not in s]
                all_units_rel = []
                for rel in non_oracle_tables:
                    all_units_rel += (base_table.Dataset.Unit & (rel & (Scan.Frame & key))).proj()
                source_units = unit_rel & all_units_rel
            else:
                # Find units for which the two image classes to compare were generated
                source_units = unit_rel & (stim_table & (Scan.Frame & key))
                # Find frames corresponding to units targets for stimulys optimization
                # frames = stimulus.Frame & (stimulus.Trial * Scan.Frame & key & stim_table).proj()
                # frames = stimulus.Frame & (stim_table & (Scan.Frame.proj() & key) & {'group_id': group_id})
                # order by neuron_id of source units so it matches order of unit_keys
                frame_keys = (Scan.Frame * stim_table * unit_rel & key).fetch('condition_hash', order_by='animal_id, src_session, src_scan_idx, src_unit_id')
                
            if len(frame_keys) == 0: # skip subtype if there is no frame belonging to this subtype
                warnings.warn('Processing for subtype {} skipped: no frames found'.format(subtype))
                continue
            else:
                unit_match = closed_loop.BestProximityCellMatch & key & source_units
                rel_keys = ['animal_id', 'session', 'scan_idx', 'pipe_version', 'segmentation_method', 'spike_method', 'unit_id', 'col_id']
                unit_keys = (InputResponse.ResponseKeys * unit_match).fetch(*rel_keys, as_dict=True, order_by='animal_id, src_session, src_scan_idx, src_unit_id')
                col_idxs = np.array([k['col_id'] for k in unit_keys])

                # Restrict response block to the specified units
                response_block_group = response_block[:, col_idxs]

                # Insert unit order
                self.Unit.insert([{**uk, **key, **stim_key, 'col_id': i} for i, uk in enumerate(unit_keys)])

                # Iterate over each frame and get the related responses
                all_responses = []
                for i, frame_key in enumerate(frame_keys):
                    row_idx = (InputResponse.Input & key & {'condition_hash': frame_key}).fetch('row_id')

                    # Find the related responses
                    all_responses.append(response_block_group[row_idx].T)

                    # Insert frame key
                    self.Frame.insert1({**key, **stim_key, 'condition_hash': frame_key, 'row_id': i,
                                        'num_trials': len(row_idx)})
                self.Matrix.insert1({**key, **stim_key, 
                                    'all_response': np.stack(all_responses)})

@schema
class DEIClosedLoopParameters(dj.Lookup):
    definition = """
    params_id:   int
    ---
    -> base.MEIParameters
    -> base.MaskParameters
    -> deis_schema.MaskStatsParameters
    -> deis_schema.DEIParameters
    -> deis_schema.DEIThreshold
    -> deis_schema.TextureParameters
    -> deis_schema.TextureScoreParameters
    -> dei_stimulus.StimulusParameters
    """
    contents = [[1, 10, 3, 2, 14, 2, 13, 2, 8], 
                [2, 10, 3, 2, 14, 2, 13, 2, 10], # correct texture_params to 13 in database 
                [3, 10, 3, 2, 17, 2, 20, 2, 10],
                [4, 10, 3, 2, 21, 2, 20, 2, 10],]

@schema
class DEIClosedLoopSummaryResults(dj.Computed):
    definition = """
    -> closed_loop.ClosedLoopScan
    -> closed_loop.StimType
    -> DEIClosedLoopParameters
    """
    
    class Neuron(dj.Part):
        definition = """
        -> master
        -> base.Dataset.Unit         
        ---
        matched_corr:     float    # oracle correlation between matched cell in target and source scan
        matched_distance: float    # anatomical distance between matched cell in target and source scan
        """

    class Subtype(dj.Part):
        definition = """
        -> master.Neuron
        -> StimSubtype
        ---
        avg_sim:               float    # average pair-wise similarity of the diverse image batch in certain feature space
        sims_to_mei:           longblob # similarity to MEI in certain feature space
        real_response:         longblob # an array of the target neuron's real responses to its target stimuli
        model_response:        longblob # an array of the target neuron's model predicted responses to its target stimuli converted back to z-score space
        """
    
    @property
    def key_source(self):
        return closed_loop.ClosedLoopScan * closed_loop.StimType * DEIClosedLoopParameters & 'stim_type != "imagenet"'
    
    def make(self, key):
        dj.config['enable_python_native_blobs'] = True

        n_repeats = 20
        mei_key = (closed_loop.ClosedLoopScan & key).fetch1()
        stim_type = (closed_loop.ClosedLoopScan & key).fetch1('stim_type')
        mei_table = (StimSubtype & dict(stim_type=stim_type) & 'subtype not like "%%oracle%%"').get_stim_tables()[1][0]  # get a stimulus table that contains model and neuron info
        single_mats, subtypes = (NewConfusionMatrix.Matrix & mei_key & 'norm_method=3' & {'stim_type': stim_type} & 'subtype not LIKE "oracle%%"').fetch('all_response', 'subtype', order_by='subtype')
        total_neurons = (dj.U('subtype').aggr(NewConfusionMatrix.Unit & mei_key & 'norm_method=3' & {'stim_type': stim_type} & 'subtype not LIKE "oracle%%"', n='count(*)')).fetch('n', order_by='subtype')
        unit_rel = base.Dataset.Unit.proj(src_animal_id='animal_id', src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')

        # get all groups and models used for stimulus optimization
        group_keys = data_schemas.StaticMultiDataset.Member * \
                    (closed_loop.ClosedLoopScan & {'loop_group': mei_key['loop_group']} & 'mei_source = 1') & \
                    'spike_method in (5, 6) and preproc_id in (5, 8, 9)'
        rel_keys = ['group_id', 'net_hash', 'data_hash', 'readout_key', 'animal_id', 'session', 'scan_idx', 'pipe_version', 'segmentation_method', 'spike_method']
        model_keys = (dj.U(*rel_keys) & (mei_table * group_keys)).fetch(as_dict=True, order_by='animal_id, session, scan_idx')

        neuron_tuples, subtype_tuples = [], []
        for model_key in model_keys:
            if stim_type == 'dei_texture_swap':
                subtype_rest = 'subtype not like "%%oracle%%"'
            else: 
                subtype_rest = 'subtype like "%%mei%%"'
            _, neurons, matched_corr, matched_distance = get_exclude_cols(mei_key, model_key, model_key, stim_type, subtype_rest, corr_thresh=-1, matching_table='closed_loop', comb=False)
            neuron_keys = (unit_rel & model_key & [{'neuron_id': nid} for nid in neurons]).fetch(as_dict=True, order_by='src_animal_id, src_session, src_scan_idx, src_unit_id')
            _neuron_tuples = [dict(**key, **nk, matched_corr=mc, matched_distance=md) for nk, mc, md in zip(neuron_keys, matched_corr, matched_distance)]
            neuron_tuples.extend(_neuron_tuples)

            # rows where frames optimized from a specific model_key were presented
            if stim_type == 'dei_texture_swap':
                rows = np.arange(len(neuron_keys))
            else:
                rows = np.stack([(stimulus.StaticImage.MaskFixedMEI * Scan.Frame * NewConfusionMatrix.Frame & tup & key).fetch1('row_id') for tup in neuron_keys])
            
            for mat, subtype, n_total in zip(single_mats, subtypes, total_neurons):
                # get model predicted responses
                images, neurons = get_stimulus(mei_key, model_key, dict(subtype = subtype))
                n_neurons = len(neurons)
                all_models, readout_key, mean_eyepos = load_model(model_key)
                r = get_model_resps(images, neurons, all_models, readout_key, mean_eyepos)
                
                if subtype in ['mei', 'dei1', 'dei2', 'dei3', 'dei4', 'dei5', 'dei6', 'dei7', 'dei8', 'dei9', 'dei10', 'local_dei1', 'local_dei2', 'fixed_part', 'variable_part']:
                    # real responses
                    real_resps = np.stack([np.diag(mat[rows][:, rows][..., i]) for i in range(n_repeats)]).T # num_neurons * num_repeats
                    # model responses         
                    model_resps = np.diag(r.reshape(n_neurons, n_neurons)) # num_neurons
                    for nkey, r, m in zip(neuron_keys, real_resps, model_resps): 
                        subtype_tuples.append(dict(**key, **nkey, subtype=subtype, real_response=r, model_response=m, avg_sim=None, sims_to_mei=None))
                else: 
                    # real responses
                    real_resps = np.stack([np.diag(mat.reshape(n_total, n_repeats, n_total)[rows][:, i][..., rows]) for i in range(n_repeats)]).T # num_neurons * num_repeats
                    # model responses         
                    model_resps = np.stack([np.diag(r.reshape(n_neurons, n_repeats, n_neurons)[:, i, :]) for i in range(n_repeats)]).T # num_neurons * num_repeats
                    
                    if stim_type == 'dei_texture_swap':
                        for nkey, r, m in zip(neuron_keys, real_resps, model_resps): 
                            subtype_tuples.append(dict(**key, **nkey, subtype=subtype, real_response=r, model_response=m, avg_sim=None, sims_to_mei=None))
                    else: 
                        # compute similarity of the image batch and similarity to MEI
                        diverse_params = (deis_schema.DEIParameters & (DEIClosedLoopParameters & key)).fetch1()
                        avg_sims, sims_to_mei = [], []
                        
                        meis, _ = get_stimulus(mei_key, model_key, dict(subtype = 'mei'))
                        for nid, ims, mei in zip(neurons, np.stack(images).reshape(n_neurons, n_repeats, 36, 64), meis):
                            if diverse_params['features'] == 'pixels':
                                embedding = ops.Identity()  # operation that returns x as is
                            elif diverse_params['features'] == 'feature_vectors':
                                # return a feature map matrix in the shape of batch_size x (num_models x feature_vec_length)
                                embedding = ops.Feature_Vector_Ensemble(all_models, model_key['readout_key'], neuron_idx=nid, eye_pos=mean_eyepos, average_batch=False)
                            elif diverse_params['features'] == 'single_grid_population_resps':
                                embedding = ops.SingleGridResps(all_models, model_key['readout_key'], eye_pos=mean_eyepos, neuron_idx=nid, all_neurons=True, average_batch=False)
                            else:
                                raise NotImplementedError('{} feature embedding not implemented'.format(diverse_params['features']))
                            
                            ims = torch.as_tensor(ims[:, None], dtype=torch.float32, device='cuda')
                            mei = torch.as_tensor(mei[None, None], dtype=torch.float32, device='cuda')
                            get_sim = partial(deis_schema.DEI.div_regularization, None, diverse_params['similarity'], 1, ops.DoNothing(), embedding, mei)
                            sims = get_sim(ims).cpu().detach().squeeze().numpy()
                            avg_sims.append(sims[len(ims):].mean())
                            sims_to_mei.append(sims[:len(ims)])
                        
                        for nkey, r, m, a, s in zip(neuron_keys, real_resps, model_resps, avg_sims, sims_to_mei):
                            subtype_tuples.append(dict(**key, **nkey, subtype=subtype, real_response=r, model_response=m, avg_sim=a, sims_to_mei=s))

        self.insert1(key)
        self.Neuron.insert(neuron_tuples, ignore_extra_fields=True)
        self.Subtype.insert(subtype_tuples, ignore_extra_fields=True)

@schema
class DynamicStaticClosedLoopSummary(dj.Lookup):
    definition = """
    -> closed_loop.ClosedLoopScan
    dyn_session           : int
    dyn_scan_idx          : int
    dyn_unit_id           : int
    dyn_group_id          : int
    dyn_neuron_id         : int
    sta_session           : int
    sta_scan_idx          : int
    sta_unit_id           : int
    sta_group_id          : int
    sta_neuron_id         : int
    ---
    src_match_corr      : float
    src_match_distance  : float
    dyn_match_unit_id   : int
    dyn_match_corr      : float
    dyn_match_distance  : float
    sta_match_unit_id   : int
    sta_match_corr      : float
    sta_match_distance  : float
    """
    
    def fill(self, dynamic_key, static_key, mei_scan_key):
        if dynamic_key['session'] == static_key['session']: # if the two source scans were collected on the same day and BestProximityCellMatch is not needed
            dyn_sta_match = (closed_loop.ProximityCellMatch.UnitMatch & static_key & 'session = stack_session' & 'stack_session = src_session').proj('unit_id', mean_distance='match_distance')
        else:
            dyn_sta_match = closed_loop.BestProximityCellMatch & static_key & {'src_session': dynamic_key['session'], 'src_scan_idx': dynamic_key['scan_idx']} & 'total_stacks = 2'
        
        sta_units = \
        (dj.U('animal_id', 'session', 'scan_idx', 'unit_id') & \
        ((base.Dataset.Unit & static_key) * \
        (stimulus.StaticImage.MaskFixedMEI * Scan.Frame * NewConfusionMatrix.Frame & \
         mei_scan_key & {'subtype': "mei"}).proj('group_id', 'neuron_id', target_session='session', target_scan_idx='scan_idx', dummy='preproc_id')
        )).fetch(as_dict=True)

        sta_rel = (dyn_sta_match & sta_units) * (closed_loop.BestProximityCellMatch & mei_scan_key).proj(session='src_session', scan_idx='src_scan_idx', unit_id='src_unit_id', target_session='session', target_scan_idx='scan_idx', sta_match_unit_id='unit_id', sta_match_distance='mean_distance')

        dyn_units = \
        ((dj.U('animal_id', 'session', 'scan_idx', 'unit_id') & \
        ((base.Dataset.Unit & dynamic_key) * \
        (stimulus.StaticImage.MaskFixedMEI * Scan.Frame * NewConfusionMatrix.Frame & \
         mei_scan_key & {'subtype': "mei"}).proj('group_id', 'neuron_id', target_session='session', target_scan_idx='scan_idx', dummy='preproc_id')
        )).proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
        ).fetch(as_dict=True)

        dyn_rel = (dyn_sta_match & dyn_units) * (closed_loop.BestProximityCellMatch & mei_scan_key).proj(target_session='session', target_scan_idx='scan_idx', dyn_match_unit_id='unit_id', dyn_match_distance='mean_distance')

        sta_keys = \
        (DEIClosedLoopSummaryResults.Neuron.proj(target_session='session', target_scan_idx='scan_idx', sta_match_corr='matched_corr') * \
        base.Dataset.Unit * sta_rel).proj('sta_match_unit_id', 'sta_match_corr', 'sta_match_distance', 
                                          sta_session='session', sta_scan_idx='scan_idx', sta_unit_id='unit_id', sta_group_id='group_id', sta_neuron_id='neuron_id',
                                          src_match_distance='mean_distance'
                                         ).fetch('sta_session', 'sta_scan_idx', 'sta_unit_id', 'sta_group_id', 'sta_neuron_id', 
                                                 'sta_match_unit_id', 'sta_match_corr', 'sta_match_distance', 'src_match_distance', as_dict=True, order_by='src_session, src_scan_idx, src_unit_id')

        dyn_keys = \
        (DEIClosedLoopSummaryResults.Neuron.proj(target_session='session', target_scan_idx='scan_idx', dyn_match_corr='matched_corr') * \
        base.Dataset.Unit.proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id') * \
        dyn_rel).proj('dyn_match_unit_id', 'dyn_match_corr', 'dyn_match_distance', 
                      dyn_session='src_session', dyn_scan_idx='src_scan_idx', dyn_unit_id='src_unit_id', dyn_group_id='group_id', dyn_neuron_id='neuron_id'
                     ).fetch('dyn_session', 'dyn_scan_idx', 'dyn_unit_id', 'dyn_group_id', 'dyn_neuron_id', 
                             'dyn_match_unit_id', 'dyn_match_corr', 'dyn_match_distance', as_dict=True, order_by='dyn_session, dyn_scan_idx, dyn_unit_id')
        
        # Compute src_match_corr
        src_match_corrs = get_static_dynamic_matched_corr(sta_keys, dyn_keys, dynamic_key, static_key, mei_scan_key)

        for sta_key, dyn_key, src_match_corr in zip(sta_keys, dyn_keys, src_match_corrs):
            self.insert1(dict(**mei_scan_key, **sta_key, **dyn_key, src_match_corr=src_match_corr), ignore_extra_fields=True)


def get_static_dynamic_matched_corr(sta_keys, dyn_keys, dynamic_key, static_key, mei_scan_key):
    sta_multi_rel = (dj.U('image_class', 'image_id').aggr(stimulus.Trial * stimulus.Frame & static_key, n='count(*)') & 'n > 1').proj()
    dyn_multi_rel = (dj.U('image_class', 'image_id').aggr(stimulus.Trial * stimulus.Frame & dynamic_key, n='count(*)') & 'n > 1').proj()
    if sta_multi_rel & dyn_multi_rel:
        # get static source scan dataset
        ordered_sta_units = np.array([k['sta_unit_id'] for k in sta_keys])
        sta_data_key = (DEIClosedLoopSummaryResults.Neuron & {'group_id': sta_keys[0]['sta_group_id'], 'neuron_id': sta_keys[0]['sta_neuron_id']}).fetch('group_id', 'data_hash', as_dict=True)[0]
        sta_ro_key = (data_schemas.StaticMultiDataset.Member() & sta_data_key & 'preproc_id = 9').fetch1('name')
        sta_dset = neuro_configs.DataConfig().load_data(sta_data_key)[0][sta_ro_key]
        sta_idxs = np.array([np.where(sta_dset.neurons.unit_ids == u)[0].item() for u in ordered_sta_units])

        # get dynamic source scan dataset
        ordered_dyn_units = np.array([k['dyn_unit_id'] for k in dyn_keys])
        dyn_data_key = (DEIClosedLoopSummaryResults.Neuron & {'group_id': dyn_keys[0]['dyn_group_id'], 'neuron_id': dyn_keys[0]['dyn_neuron_id']}).fetch('group_id', 'data_hash', as_dict=True)[0]
        if (data_schemas.StaticMultiDataset.Member & dyn_data_key).fetch1('preproc_id') == 8:
            data_config = configs.DataConfig.CorrectedAreaLayer & \
                            {'stimulus_type': 'stimulus.Frame', 'exclude': '', 'layer': 'L2/3',
                            'normalize_per_image': False, 'normalize': True} & 'brain_area in ("V1")'
            dyn_data_key, dyn_ro_key = (data_schemas.StaticMultiDataset.Member * data_config & dynamic_key & {'preproc_id': 14}).fetch1(dj.key, 'name')
        dyn_dset = neuro_configs.DataConfig().load_data(dyn_data_key)[0][dyn_ro_key]
        dyn_idxs = np.array([np.where(dyn_dset.neurons.unit_ids == u)[0].item() for u in ordered_dyn_units])

        # get oracle responses for target neurons in static and dynamic source datasets
        stim_type = (closed_loop.ClosedLoopScan & mei_scan_key).fetch1('stim_type')
        oracle_subtypes, oracle_tables = (StimSubtype & {'stim_type': stim_type} & 'subtype = "oracle"').get_stim_tables()
        for oracle_subtype, oracle_table in zip(oracle_subtypes, oracle_tables):
            # Find image ids of oracle images as ordered in the oracle confusion matrix
            oracle_image_id, oracle_image_class = (((stimulus.Trial & mei_scan_key) * stimulus.Frame & oracle_table) \
                                                * (NewConfusionMatrix.Frame & mei_scan_key)).fetch('image_id', 'image_class', order_by='row_id')
            oracle_image_id = oracle_image_id[::10]
            oracle_image_class = oracle_image_class[::10]

            # Static dataset oracle responses for the target units
            resps = sta_dset.responses[:, sta_dset.transforms[1].idx]
            norm_resps = (resps/ resps.std(0, ddof=1)) 

            sta_oracle_std = []
            sta_oracle_mean = []
            for im_c, im_id in zip(oracle_image_class, oracle_image_id):
                class_str = np.array([c.astype('str') for c in sta_dset.item_info['frame_image_class']])
                images_idx = np.where((class_str == im_c) & \
                            (np.array(list(sta_dset.item_info['frame_image_id'])) == im_id))[0]
                sta_oracle_std.append(norm_resps[images_idx][:, sta_idxs].std(axis=0))
                sta_oracle_mean.append(norm_resps[images_idx][:, sta_idxs].mean(axis=0))

            # Dynamic dataset oracle responses for the target units
            resps = dyn_dset.responses[:, dyn_dset.transforms[1].idx]
            norm_resps = (resps/ resps.std(0, ddof=1)) 

            dyn_oracle_std = []
            dyn_oracle_mean = []
            for im_c, im_id in zip(oracle_image_class, oracle_image_id):
                class_str = np.array([c.astype('str') for c in dyn_dset.item_info['frame_image_class']])
                images_idx = np.where((class_str == im_c) & \
                            (np.array(list(dyn_dset.item_info['frame_image_id'])) == im_id))[0]
                dyn_oracle_std.append(norm_resps[images_idx][:, dyn_idxs].std(axis=0))
                dyn_oracle_mean.append(norm_resps[images_idx][:, dyn_idxs].mean(axis=0))

        sta_oracle_mean = np.stack(sta_oracle_mean)
        sta_oracle_std = np.stack(sta_oracle_std)
        dyn_oracle_mean = np.stack(dyn_oracle_mean)
        dyn_oracle_std = np.stack(dyn_oracle_std)

        src_match_corrs = corr(sta_oracle_mean, dyn_oracle_mean, axis=0)

    else:
        src_match_corrs = np.ones_like(np.array(sta_keys))
        
    return src_match_corrs


def best_model(key):
    # Pick the dataset
    dataset = data_schemas.StaticMultiDataset & (data_schemas.StaticMultiDataset.Member & key)
    if len(data_schemas.StaticMultiDataset.Member & key) == 1:
        dataconfig_one = configs.DataConfig.CorrectedAreaLayer() & {'data_hash': key['data_hash']}
        # dataconfig_multi = configs.DataConfig.CorrectedAreaLayerMatchedCell() & {'data_hash': key['data_hash']}
        if len(static_models.Model & key & (configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace & dataconfig_one)) > 0:  # V1
            # if any one in the key has been trained for only V1 I pick V1
            dataconfig = dataconfig_one
        # else: # higher areas
        #     dataconfig = dataconfig_multi
    else: 
        dataconfig = configs.DataConfig.CorrectedAreaLayerMatchedCell() & {'data_hash': key['data_hash']}
    cond = (dataset * dataconfig).proj()

    for one_cond in cond.fetch('KEY', order_by='group_id'):
        all_models = static_models.Model * configs.NetworkConfig.CorePlusReadout & configs.CoreConfig.GaussianLaplace & one_cond & key & 'seed >1000' # trained models with right config

        # Find best model config (best average correlation across seeds) and the best seed in that config
        keys, corrs = dj.U('group_id', 'net_hash').aggr(all_models, avg_corr='AVG(val_corr)').fetch('KEY', 'avg_corr') # average across seeds
        best_config = keys[np.argmax(corrs)]
        keys, corrs = (all_models & best_config).fetch('KEY', 'val_corr')
        
    return keys[np.argmax(corrs)] 


def get_matched_corr(mei_key, src_key, model_key, stim_type, subtype_rest, matching_table='closed_loop'):
    
    oracle_subtypes, oracle_tables = (StimSubtype & {'stim_type': stim_type} & 'subtype = "oracle"').get_stim_tables()
        
    # model_mean and model_std is the oracle response mean and std in the scan used for training the predictive model
    model_mean, model_std, target_mean, target_std = [], [], [], []
    for oracle_subtype, oracle_table in zip(oracle_subtypes, oracle_tables):

        # Get training dataset and some training stats
        if (data_schemas.StaticMultiDataset.Member & model_key).fetch1('preproc_id') == 8:
            data_config = configs.DataConfig.CorrectedAreaLayer & \
                            {'stimulus_type': 'stimulus.Frame', 'exclude': '', 'layer': 'L2/3',
                            'normalize_per_image': False, 'normalize': True} & 'brain_area in ("V1")'
            source_key, readout_key = (data_schemas.StaticMultiDataset.Member * data_config & \
                        {'preproc_id': 14, 'animal_id': model_key['animal_id'], 'session': model_key['session'], 'scan_idx': model_key['scan_idx']}
                        ).fetch1(dj.key, 'name')
        else:
            source_key = model_key
            readout_key = (data_schemas.StaticMultiDataset.Member() & source_key).fetch1('name')
        dset = neuro_configs.DataConfig().load_data(source_key)[0][readout_key]

        # Find image ids of oracle images as ordered in the oracle confusion matrix
        oracle_image_id, oracle_image_class = (((stimulus.Trial & mei_key) * stimulus.Frame & oracle_table) \
                                            * (NewConfusionMatrix.Frame & mei_key)).fetch('image_id', 'image_class', order_by='row_id')
        oracle_image_id = oracle_image_id[::10]
        oracle_image_class = oracle_image_class[::10]

        # Oracle image ordered by image_id in stimulus.Frame, neuron in confusion matrix ordered by neuron id in base.Dataset.Unit
        # Source cell responses to oracle images
        resps = dset.responses[:, dset.transforms[1].idx]
        norm_resps = (resps/ resps.std(0, ddof=1)) 

        oracle_std = []
        oracle_mean = []
        for im_c, im_id in zip(oracle_image_class, oracle_image_id):
            class_str = np.array([c.astype('str') for c in dset.item_info['frame_image_class']])
            images_idx = np.where((class_str == im_c) & \
                        (np.array(list(dset.item_info['frame_image_id'])) == im_id))[0]
            oracle_std.append(norm_resps[images_idx].std(axis=0))
            oracle_mean.append(norm_resps[images_idx].mean(axis=0))
        oracle_mean = np.array(oracle_mean)
        oracle_std = np.array(oracle_std)

        # compute mean oracle response for source cells
        table = eval((StimSubtype & (NewConfusionMatrix.Unit() & mei_key & subtype_rest)).fetch('table')[0])
        rel = stimulus.Frame * stimulus.Trial & mei_key & table
        target_rel = table & rel
        if model_key == src_key: # when we use the MEI source scan model
            model_units = (base.Dataset.Unit & target_rel & model_key).fetch('unit_id', order_by='group_id, unit_id')
        else: # when model_key is another training image scan that is not the MEI source scan (usually the imagenet scan on the same day of the mei scan)
            if matching_table == 'closed_loop':
                match_rel = closed_loop.BestProximityCellMatch & {'animal_id': model_key['animal_id'], 'src_session': model_key['session'], 'src_scan_idx': model_key['scan_idx']}
            elif matching_table == 'loop':
                match_rel = loop.TempBestProximityCellMatch2 & {'animal_id': model_key['animal_id'], 'src_session': model_key['session'], 'src_scan_idx': model_key['scan_idx']}
            # neurons ordered by neuron id in source scan
            model_units = ((base.Dataset.Unit & target_rel & model_key).proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id') * (match_rel & model_key)).fetch('unit_id', order_by='group_id, unit_id')
        model_cell_oracle_mean = np.hstack([oracle_mean[:, dset.neurons.unit_ids == u] for u in model_units])
        model_cell_oracle_std = np.hstack([oracle_std[:, dset.neurons.unit_ids == u] for u in model_units])

        # compute mean oracle response for target cells
        target_cell_oracle_resps = (NewConfusionMatrix.Matrix & mei_key & {'subtype': oracle_subtype}).fetch1('all_response')
        # restrict to neurons from the desired model_key
        if stim_type == "dei_texture_swap":
            rows = (NewConfusionMatrix.Unit & 'subtype = "oracle"' & mei_key).fetch('col_id', order_by='col_id')
        else:
            mei_table = (StimSubtype & {'stim_type': stim_type} & 'subtype = "mei"').get_stim_tables()[1][0]
            rows = (NewConfusionMatrix.Frame & (mei_table * Scan.Frame & mei_key &  
                                                {'group_id': model_key['group_id'], 'net_hash': model_key['net_hash'], 
                                                'data_hash': model_key['data_hash'], 'readout_key': model_key['readout_key']}
                                            )).fetch('row_id', order_by='row_id')
        target_cell_oracle_mean = target_cell_oracle_resps[:, rows].mean(axis=2) 
        target_cell_oracle_std = target_cell_oracle_resps[:, rows].std(axis=2)
        model_mean.extend(model_cell_oracle_mean)
        model_std.extend(model_cell_oracle_std)
        target_mean.extend(target_cell_oracle_mean)
        target_std.extend(target_cell_oracle_std)

    model_mean = np.stack(model_mean)
    model_std = np.stack(model_std)
    target_mean = np.stack(target_mean)
    target_std = np.stack(target_std)

    matched_cell_corr = corr(model_mean, target_mean, axis=0)
            
    return matched_cell_corr, model_cell_oracle_mean, model_cell_oracle_std, target_cell_oracle_mean, target_cell_oracle_std, oracle_image_id

def get_exclude_cols(mei_key, src_key, model_key, stim_type, subtype_rest, corr_thresh=0.6, comb_matched_units=[], matching_table='closed_loop', comb=False):  # comb_matched_units are units that matched well across different training scans

    table = (StimSubtype & (closed_loop.ClosedLoopScan & mei_key) & subtype_rest).get_stim_tables()[1][0]
    mei_neurons = np.unique((Scan.Frame() * table & mei_key & \
                            {'group_id': model_key['group_id'], 'net_hash': model_key['net_hash'], 
                                'data_hash': model_key['data_hash'], 'readout_key': model_key['readout_key']}
                            ).fetch('neuron_id', order_by='neuron_id')) # source scan neurons
    # np.unique((stimulus.StaticImage.Image * (Scan.Frame() * table & mei_key)).fetch('neuron_id', order_by='neuron_id')) # source scan neurons
    mei_units = (base.Dataset.Unit & model_key & [{'neuron_id': neuron} for neuron in mei_neurons]).fetch('unit_id', order_by='neuron_id') # source scan units
    if comb:
        matched_mei_units = list(set(comb_matched_units).intersection(mei_units))
        matched_mei_neurons = (base.Dataset.Unit & model_key & [{'unit_id': unit} for unit in matched_mei_units]).fetch('neuron_id', order_by='neuron_id') # source scan neurons that are matched well across days
        comb_include_cols = [np.argwhere(mei_neurons == n).item() for n in matched_mei_neurons]
        comb_exclude_cols = set(range(len(mei_neurons))) - set(comb_include_cols)
    else:
        comb_exclude_cols = set([])
    
    target_multi_rel = (dj.U('image_class', 'image_id').aggr(stimulus.Trial * stimulus.Frame & mei_key, n='count(*)') & 'n > 1').proj()
    src_multi_rel = (dj.U('image_class', 'image_id').aggr(stimulus.Trial * stimulus.Frame & model_key, n='count(*)') & 'n > 1').proj()
    if target_multi_rel & src_multi_rel:
        matched_cell_corr, _, _, _, _, _ = get_matched_corr(mei_key, model_key, model_key, stim_type, subtype_rest, matching_table=matching_table)
    else:
        warnings.warn('Functional correlation critria skipped: no common oracle stimuli in the source and target scan!')
        matched_cell_corr = np.ones_like(mei_neurons)

    mei_exclude_cols = np.where(np.round(matched_cell_corr, 2) < corr_thresh)[0]
    exclude_cols = np.array(list(comb_exclude_cols.union(set(mei_exclude_cols))))
    include_cols = np.array(list(set(range(len(mei_neurons))) - set(exclude_cols)))
    include_neurons = mei_neurons[include_cols]
    include_units = mei_units[include_cols]
    include_neuron_oracle_corr = matched_cell_corr[include_cols]
    
    src_rel = {'animal_id':model_key['animal_id'], 'src_session':model_key['session'], 'src_scan_idx': model_key['scan_idx']}
    if matching_table == 'closed_loop':
        matched_distance = (closed_loop.BestProximityCellMatch & src_rel & mei_key & [{'src_unit_id':unit} for unit in include_units]).fetch('mean_distance', order_by='src_unit_id')
    elif matching_table == 'loop':
        matched_distance = (loop.TempBestProximityCellMatch2 & src_rel & mei_key & [{'src_unit_id':unit} for unit in include_units]).fetch('mean_distance', order_by='src_unit_id')

    return exclude_cols, include_neurons, include_neuron_oracle_corr, matched_distance

def get_confusion_matrix(stim_type, mei_key, src_key, subtype_rest={}, all_exclude_cols=[]): 
    '''
    The returned matrix 50 neurons by 150 frames. Each set of frames is in the order of original MEI, diverse1, diverse2
    '''
    
    all_resps, subtypes = (NewConfusionMatrix.Matrix() & mei_key & 'norm_method=3' & {'stim_type': stim_type} & subtype_rest).fetch('all_response', 'subtype', order_by='subtype')
    
    all_matrices = np.stack([np.mean(resp, 2) for resp in all_resps])
    all_std_matrices = np.stack([np.std(resp, 2) for resp in all_resps])
    if len(all_exclude_cols) > 0:
        matrices = np.delete(np.delete(all_matrices, all_exclude_cols, 1), all_exclude_cols, 2)
        std_matrices = np.delete(np.delete(all_std_matrices, all_exclude_cols, 1), all_exclude_cols, 2)
    else:
        matrices = all_matrices
        std_matrices = all_matrices

    comb_mat = []
    std_comb_mat = []
    for k in range(matrices.shape[1]):
        comb_block = np.concatenate([matrices[i][k].reshape(-1, 1) for i in range(matrices.shape[0])], axis=1)
        std_comb_block = np.concatenate([std_matrices[i][k].reshape(-1, 1) for i in range(std_matrices.shape[0])], axis=1)
        comb_mat.append(comb_block)
        std_comb_mat.append(std_comb_block)
    comb_mat = np.hstack(comb_mat)
    std_comb_mat = np.hstack(std_comb_mat)

    return matrices, std_matrices, comb_mat, std_comb_mat, subtypes, stim_type

def get_stimulus(mei_key, model_key, subtype_rest={}, exclude_cols=[]):
    # get neurons
    subtypes, stim_tables = (StimSubtype & (closed_loop.ClosedLoopScan & mei_key) & subtype_rest).get_stim_tables()
    unit_rel = base.Dataset.Unit.proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')

    # get images
    batch = []
    for table in stim_tables:
        stim_rel = unit_rel * stimulus.StaticImage.Image * Scan.Frame * table & mei_key & \
                   {'group_id': model_key['group_id'], 
                    'net_hash': model_key['net_hash'], 
                    'data_hash': model_key['data_hash'], 
                    'readout_key': model_key['readout_key']}
        neurons = (dj.U('neuron_id') & stim_rel).fetch('neuron_id', order_by='neuron_id')
        neurons = np.unique(neurons)
        if len(exclude_cols) > 0:
            neurons = np.delete(neurons, exclude_cols)

        images = (stim_rel & [{'neuron_id': nid} for nid in neurons]).fetch('image', order_by='src_session, src_scan_idx, src_unit_id')
        batch.append(images)
    batch = np.vstack(batch)
    import itertools
    images = list(itertools.chain(*zip(*batch)))

    # normalize images
    import cv2
    imgsize = (data_schemas.Preprocessing & (data_schemas.StaticMultiDataset.Member & model_key)).fetch1('col', 'row')
    resized = np.stack([cv2.resize(i, imgsize, interpolation=cv2.INTER_AREA).astype(np.float32) for i in images])
    norm_batch = [((image - image.mean()) / (image.std() + 1e-9)) * 0.25 for image in resized]

    return norm_batch, neurons

def load_model(model_key):
    readout_key = (data_schemas.StaticMultiDataset.Member() & model_key).fetch1('name')
    all_keys = (static_models.Model & model_key).fetch('KEY')
    all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
    # Get some train stats
    mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & model_key).fetch1('norm_eyepos') else
                   (base.Dataset.TrainStats & model_key).fetch1('mean_eyepos'))
    mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
                device='cuda').unsqueeze(0)
    return all_models, readout_key, mean_eyepos

def get_model_resps(images, neurons, all_models, readout_key, mean_eyepos):
    images = torch.as_tensor(np.stack(images), dtype=torch.float32, device='cuda')
    model = models.Ensemble(all_models, readout_key, eye_pos=mean_eyepos,
                            neuron_idx=list(neurons), average_batch=False, device='cuda')
    with torch.no_grad():
        model_resps = [model(im[None, None]).cpu().numpy().squeeze() for im in images]
    model_resps = np.stack(model_resps)#.T.reshape(len(neurons), len(neurons), -1)  # num_images * num_neurons
    return model_resps

stack = dj.create_virtual_module('stack', 'pipeline_stack')
from scipy import ndimage
import numpy as np
import torch
from torch.nn import functional as F
from matplotlib.patches import Circle

def lcn(image, sigmas=(12, 12)):
    """ From Erick Cobos. Local contrast normalization.
    Normalize each pixel using mean and stddev computed on a local neighborhood.
    We use gaussian filters rather than uniform filters to compute the local mean and std
    to soften the effect of edges. Essentially we are using a fuzzy local neighborhood.
    Equivalent using a hard defintion of neighborhood will be:
        local_mean = ndimage.uniform_filter(image, size=(32, 32))
    :param np.array image: Array with raw two-photon images.
    :param tuple sigmas: List with sigmas (one per axis) to use for the gaussian filter.
        Smaller values result in more local neighborhoods. 15-30 microns should work fine
    """
    local_mean = ndimage.gaussian_filter(image, sigmas)
    local_var = ndimage.gaussian_filter(image ** 2, sigmas) - local_mean ** 2
    local_std = np.sqrt(np.clip(local_var, a_min=0, a_max=None))
    norm = (image - local_mean) / (local_std + 1e-7)

    return norm


def sharpen_2pimage(image, laplace_sigma=0.7, low_percentile=3, high_percentile=99.9):
    """ From Erick Cobos. Apply a laplacian filter, clip pixel range and normalize.
    :param np.array image: Array with raw two-photon images.
    :param float laplace_sigma: Sigma of the gaussian used in the laplace filter.
    :param float low_percentile, high_percentile: Percentiles at which to clip.
    :returns: Array of same shape as input. Sharpened image.
    """
    sharpened = image - ndimage.gaussian_laplace(image, laplace_sigma)
    clipped = np.clip(sharpened, *np.percentile(sharpened, [low_percentile, high_percentile]))
    norm = (clipped - clipped.mean()) / (clipped.max() - clipped.min() + 1e-7)
    return norm

def resize(original, um_sizes, desired_res, mode='bilinear'):
    """ From Erick Cobos. Resize array originally of um_sizes size to have desired_res resolution.
    We preserve the center of original and resized arrays exactly in the middle. We also
    make sure resolution is exactly the desired resolution. Given these two constraints,
    we cannot hold FOV of original and resized arrays to be exactly the same.
    :param np.array original: Array to resize.
    :param tuple um_sizes: Size in microns of the array (one per axis).
    :param int or tuple desired_res: Desired resolution (um/px) for the output array.
    :return: Output array (np.float32) resampled to the desired resolution. Size in pixels
        is round(um_sizes / desired_res).
    """

    # Create grid to sample in microns
    grid = create_grid(um_sizes, desired_res) # d x h x w x 3

    # Re-express as a torch grid [-1, 1]
    um_per_px = np.array([um / px for um, px in zip(um_sizes, original.shape)])
    torch_ones = np.array(um_sizes) / 2 - um_per_px / 2  # sample position of last pixel in original
    grid = grid / torch_ones[::-1].astype(np.float32)

    # Resample
    input_tensor = torch.from_numpy(original.reshape(1, 1, *original.shape).astype(
        np.float32))
    grid_tensor = torch.from_numpy(grid.reshape(1, *grid.shape))
    resized_tensor = F.grid_sample(input_tensor, grid_tensor, padding_mode='border', mode=mode, align_corners=True)
    resized = resized_tensor.numpy().squeeze()

    return resized

def create_grid(um_sizes, desired_res=1):
    """ From Erick Cobos. Create a grid corresponding to the sample position of each pixel/voxel in a FOV of
     um_sizes at resolution desired_res. The center of the FOV is (0, 0, 0).
    In our convention, samples are taken in the center of each pixel/voxel, i.e., a volume
    centered at zero of size 4 will have samples at -1.5, -0.5, 0.5 and 1.5; thus edges
    are NOT at -2 and 2 which is the assumption in some libraries.
    :param tuple um_sizes: Size in microns of the FOV, .e.g., (d1, d2, d3) for a stack.
    :param float or tuple desired_res: Desired resolution (um/px) for the grid.
    :return: A (d1 x d2 x ... x dn x n) array of coordinates. For a stack, the points at
    each grid position are (x, y, z) points; (x, y) for fields. Remember that in our stack
    coordinate system the first axis represents z, the second, y and the third, x so, e.g.,
    p[10, 20, 30, 0] represents the value in x at grid position 10, 20, 30.
    """
    # Make sure desired_res is a tuple with the same size as um_sizes
    if np.isscalar(desired_res):
        desired_res = (desired_res,) * len(um_sizes)

    # Create grid
    out_sizes = [int(round(um_s / res)) for um_s, res in zip(um_sizes, desired_res)]
    um_grids = [np.linspace(-(s - 1) * res / 2, (s - 1) * res / 2, s, dtype=np.float32)
                for s, res in zip(out_sizes, desired_res)] # *
    full_grid = np.stack(np.meshgrid(*um_grids, indexing='ij')[::-1], axis=-1)
    # * this preserves the desired resolution by slightly changing the size of the FOV to
    # out_sizes rather than um_sizes / desired_res.

    return full_grid

def get_matched_plot(dic):
    pipe_name = (fuse.ScanDone & dic).fetch1('pipe')
    pipe = dj.create_virtual_module(pipe_name, 'pipeline_' + pipe_name)
    field_rel = pipe.ScanInfo.Field if pipe_name == 'meso' else pipe.ScanInfo

    a, ses, si, f, ui = (pipe.ScanSet.Unit() & {'animal_id':dic['animal_id'], 'session': dic['src_session'], 'scan_idx': dic['src_scan_idx'], 'unit_id': dic['src_unit_id']}).fetch1('animal_id', 'session', 'scan_idx', 'field', 'unit_id')
    unit_key_1 = {'animal_id':a,  'session': ses, 'scan_session': ses, 'scan_idx': si, 'field': f, 'unit_id': ui}
    field_key_1 = {'animal_id':a,  'session': ses, 'scan_session': ses, 'scan_idx': si, 'field': f}

    a, ses, si, f, ui = (pipe.ScanSet.Unit() & dic).fetch1('animal_id', 'session', 'scan_idx', 'field', 'unit_id')
    unit_key_2 = {'animal_id':a,  'session': ses, 'scan_session': ses, 'scan_idx': si, 'field': f, 'unit_id': ui}
    field_key_2 = {'animal_id':a,  'session': ses, 'scan_session': ses, 'scan_idx': si, 'field': f}
    
    average_image_1 =  (pipe.SummaryImages.Average() & unit_key_1).fetch1("average_image")
    correlation_image_1 = (pipe.SummaryImages.Correlation() & unit_key_1).fetch1("correlation_image")
    scan_field_1 = average_image_1 * correlation_image_1
    scan_field_1 = sharpen_2pimage(lcn(scan_field_1, 2.5))
    scan_field_1 = resize(scan_field_1, (field_rel & unit_key_1).fetch1('um_height', 'um_width'), desired_res=1)

    xy1 = np.array((pipe.ScanSet.UnitInfo & unit_key_1).fetch1('px_x', 'px_y'))
    cents1 = np.stack((pipe.ScanSet.UnitInfo*pipe.ScanSet.Unit & field_key_1).fetch('px_x', 'px_y'), -1)

    average_image_2 =  (pipe.SummaryImages.Average() & unit_key_2).fetch1("average_image")
    correlation_image_2 = (pipe.SummaryImages.Correlation() & unit_key_2).fetch1("correlation_image")
    scan_field_2 = average_image_2 * correlation_image_2
    scan_field_2 = sharpen_2pimage(lcn(scan_field_2, 2.5))
    scan_field_2 = resize(scan_field_2, (field_rel & unit_key_2).fetch1('um_height', 'um_width'), desired_res=1)

    xy2 = np.array((pipe.ScanSet.UnitInfo & unit_key_2).fetch1('px_x', 'px_y'))
    cents2 = np.stack((pipe.ScanSet.UnitInfo*pipe.ScanSet.Unit & field_key_2).fetch('px_x', 'px_y'), -1)
    
    return unit_key_1, scan_field_1, xy1, cents1, unit_key_2, scan_field_2, xy2, cents2

# DEI pair decoding analysis
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from collections import Counter
from tqdm import tqdm

def fit_helper(dei_repeats,Cs=np.linspace(1e-6,1e-4,10),cv=5):
    y = np.concatenate([[0+i]*dei_repeats.shape[1] for i in range(len(dei_repeats))])
    X = dei_repeats.reshape(-1,dei_repeats.shape[-1])
    clf = LogisticRegressionCV(Cs=Cs,cv=cv,penalty='l2',fit_intercept=True,refit=False,n_jobs=5).fit(X,y)
    return clf
def make_helper(dei1_resps,dei2_resps,Cs=[1.2e-5],cv=5):
    accuracy = []
    for neuron_idx in range(len(dei1_resps)):
        dei_repeats = np.stack([dei1_resps[neuron_idx],dei2_resps[neuron_idx]])
        clf = fit_helper(dei_repeats,Cs=Cs,cv=cv)
        accuracy.append(clf.scores_[1].mean())
    return np.array(accuracy)
def sanity_check(dei1_resps,dei2_resps,Cs=[1.2e-5],cv=5):
    between_accuracy = []
    for neuron_idx in range(len(dei1_resps)):
        dei_repeats = np.stack([dei1_resps[neuron_idx],dei2_resps[neuron_idx]])
        dei_repeats = dei_repeats[:,:dei_repeats.shape[1]//2,:]
        clf = fit_helper(dei_repeats,Cs=Cs,cv=cv)
        between_accuracy.append(clf.scores_[1].mean())
    within_accuracy = []
    for neuron_idx in range(len(dei1_resps)):
        dei_repeats = dei1_resps[neuron_idx]
        np.random.shuffle(dei_repeats)
        dei_repeats = dei_repeats.reshape(2,len(dei_repeats)//2,-1)
        clf = fit_helper(dei_repeats,Cs=Cs,cv=cv)
        within_accuracy.append(clf.scores_[1].mean())
    return np.array(between_accuracy),np.array(within_accuracy)

def scatterplot_helper(x, y, x_label, y_label, figsize=(5, 5), label=None, color='k', get_bound_from_data=False, min_val=0, max_val=1, fit=True, robust_fit=True, x_std=None, y_std=None, alpha=1, plot_identity=True, n_repeats=20, exclude_idxs_for_scatter=None):
    plt.figure(figsize=figsize, dpi=300)
    if exclude_idxs_for_scatter is not None:
        x_scatter, y_scatter = np.delete(x, exclude_idxs_for_scatter), np.delete(y, exclude_idxs_for_scatter) 
    else: 
        x_scatter, y_scatter = x, y

    plt.scatter(x_scatter, y_scatter, color=color, s=5, alpha=alpha, label=label)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    
    if get_bound_from_data:
        min_val = np.min(np.concatenate([x, y]))
        max_val = np.max(np.concatenate([x, y]))
    
    if plot_identity:
        plt.plot([min_val, max_val], [min_val, max_val], color='k', linestyle='dashed', label='x=y')

    p_wc = scipy.stats.wilcoxon(x, y)[1]

    if x_std is not None and y_std is not None:
        ps = np.array([scipy.stats.ttest_ind_from_stats(m1,s1,n_repeats,m2,s2,n_repeats, equal_var=False)[1] for m1,s1,m2,s2 in zip(x, x_std, y, y_std)])

    if fit:
        # Linear fit 
        from sklearn import linear_model
        x = x[:, np.newaxis]
        if robust_fit:
            # robust
            coefs = []
            for i in range(1000):
                ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=False), max_trials=1000)
                ransac.fit(x, y)
                coefs.append(ransac.estimator_.coef_.item())
            coef = np.mean(coefs)
            std_coef = np.std(coefs)
        else:
            # non-robust
            lr = linear_model.LinearRegression(fit_intercept=False)
            lr.fit(x, y)
            coef = lr.coef_.item()
        x_range = np.linspace(min_val, max_val, 100)
        plt.plot(x_range, coef*x_range, color='k', label='linear fit: y = {:.2f}+/-{:.2f}x'.format(coef, std_coef))

    plt.title('p = {}'.format(np.format_float_scientific(p_wc, precision=2)))
    tick_labels = np.round(np.linspace(np.floor(min_val), np.ceil(max_val), 6), 1)
    plt.xticks(tick_labels, tick_labels)
    plt.yticks(tick_labels, tick_labels)

    plt.legend(loc=2)
    sns.despine()
    plt.tight_layout()