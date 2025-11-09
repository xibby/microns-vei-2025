import torch
import datajoint as dj
from featurevis import models
from featurevis import ops
from featurevis import utils
import featurevis

from staticnet_analyses import multi_mei
from staticnet_experiments import configs
from staticnet_experiments import models as static_models


schema = dj.schema('neurostatic_crossval', create_tables=True)


# No clipping, no std matching
@schema
class MEISimple(dj.Computed):
    definition = """ # MEI saved at different iterations, created with no clipping and no standard deviation

    -> multi_mei.TargetModel
    -> multi_mei.TargetDataset.Unit
    ---
    meis                : longblob  # most exciting images
    iters               : longblob  # list of iterations corresponding to the iteration each MEI comes from
    activations         : longblob  # activations of the models during each iteration
    """
    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return multi_mei.TargetModel * multi_mei.TargetDataset.Unit & configs.NetworkConfig.CorePlusReadout

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed > 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get initial image
        initial_image = torch.randn(1, 1, 36, 64, dtype=torch.float32, device='cuda')

        # Optimize (simple SGD)
        opt_x, fevals, _ = featurevis.gradient_ascent(model, initial_image, step_size=1,
                                                      num_iterations=2000, save_iters=50)
        opt_x = [initial_image, *opt_x] # initial image is not returned by featurevis

        self.insert1({**key, 'meis': [m.squeeze().cpu().numpy() for m in opt_x],
                      'iters': list(range(0, 2001, 50)), 'activations': fevals[::50]})


@schema
class OneStd(dj.Lookup):
    definition = """ # a standard deviation to use below
    contrast         : decimal(3, 2)
    """
    contents = [{'contrast': 0.1}, {'contrast': 0.25}, {'contrast': 0.5}, {'contrast': 1},
                {'contrast': 5}]


# No clipping, std matching
@schema
class MEIConstantStd(dj.Computed):
    definition = """ # MEI saved at different iterations, created with no clipping and standard deviation constrained to be 1

    -> multi_mei.TargetModel
    -> multi_mei.TargetDataset.Unit
    -> OneStd
    ---
    meis                : longblob  # most exciting images
    iters               : longblob  # list of iterations corresponding to the iteration each MEI comes from
    activations         : longblob  # activations of the models during each iteration
    """
    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return multi_mei.TargetModel * multi_mei.TargetDataset.Unit * OneStd & configs.NetworkConfig.CorePlusReadout

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed > 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get initial image (at desired contrast)
        initial_image = torch.randn(1, 1, 36, 64, dtype=torch.float32, device='cuda')
        initial_image = initial_image * float(key['contrast'])

        # Optimize
        postup_op = ops.ChangeStd(float(key['contrast']))
        opt_x, fevals, _ = featurevis.gradient_ascent(model, initial_image, step_size=1,
                                                      num_iterations=2000, save_iters=50,
                                                      post_update=postup_op)
        opt_x = [initial_image, *opt_x] # initial image is not returned by featurevis

        self.insert1({**key, 'meis': [m.squeeze().cpu().numpy() for m in opt_x],
                      'iters': list(range(0, 2001, 50)), 'activations': fevals[::50]})


# Clipping, std matching
@schema
class MEIConstantStdAndClipped(dj.Computed):
    definition = """ # MEI saved at different iterations, created with clipping and standard deviation constrained to be 1

    -> multi_mei.TargetModel
    -> multi_mei.TargetDataset.Unit
    -> OneStd
    ---
    meis                : longblob  # most exciting images
    iters               : longblob  # list of iterations corresponding to the iteration each MEI comes from
    activations         : longblob  # activations of the models during each iteration
    """
    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return multi_mei.TargetModel * multi_mei.TargetDataset.Unit * OneStd & configs.NetworkConfig.CorePlusReadout

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed > 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get initial image
        initial_image = torch.randn(1, 1, 36, 64, dtype=torch.float32, device='cuda')
        initial_image = initial_image * float(key['contrast'])

        # Optimize
        train_mean, train_std = 111.28329467773438, 60.922306060791016
        postup_op = utils.Compose([ops.ChangeStd(float(key['contrast'])),
                                   ops.ClipRange(-train_mean / train_std,
                                                 (255 - train_mean) / train_std)])
        opt_x, fevals, _ = featurevis.gradient_ascent(model, initial_image, step_size=1,
                                                      num_iterations=2000, save_iters=50,
                                                      post_update=postup_op)
        opt_x = [initial_image, *opt_x] # initial image is not returned by featurevis

        self.insert1({**key, 'meis': [m.squeeze().cpu().numpy() for m in opt_x],
                      'iters': list(range(0, 2001, 50)), 'activations': fevals[::50]})


########################################################################################
# Test meis at diff iterations in other models

@schema
class AcrossSeedXVal(dj.Computed):
    definition = """ # evaluate the MEIs 
    -> MEIConstantStdAndClipped
    ---
    xactivations        : longblob  # activations of the cross validation models for each mei
    """
    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed < 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Get ensemble model
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], average_batch=False,
                                device='cuda')

        # Evaluate
        meis = (MEIConstantStdAndClipped & key).fetch1('meis')
        with torch.no_grad():
            batch = torch.as_tensor(meis, dtype=torch.float32, device='cuda').unsqueeze(1)
            xevals = model(batch)

        self.insert1({**key, 'xactivations': xevals.cpu().numpy()})


@schema
class VGGXVal(dj.Computed):
    definition = """ # evaluate the MEIs 
    -> MEIConstantStdAndClipped
    -> static_models.Model.proj(vgg_net_hash='net_hash', vgg_seed='seed')
    ---
    xactivations        : longblob  # activations of the cross validation models for each mei
    """
    @property
    def key_source(self):
        # vgg_config = configs.NetworkConfig.CorePlusReadout & (configs.CoreConfig() & 'core_type="VGG19"')
        # vgg_models = (models.Model & vgg_config).proj(vgg_net_hash='net_hash', vgg_seed='seed')
        vgg_models = (static_models.Model & {'net_hash': 'cf229e30b96249ceeff8424e06fb7c07', 'seed': 1009}).proj(vgg_net_hash='net_hash', vgg_seed='seed')
        return MEIConstantStdAndClipped.proj() * vgg_models

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['vgg_net_hash']}
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Get ensemble model
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], average_batch=False,
                                device='cuda')

        # Evaluate
        meis = (MEIConstantStdAndClipped & key).fetch1('meis')
        with torch.no_grad():
            batch = torch.as_tensor(meis, dtype=torch.float32, device='cuda').unsqueeze(1)
            xevals = model(batch)

        self.insert1({**key, 'xactivations': xevals.cpu().numpy()})


#########################################################################################
# Alternative MEIs
# MEIs generated in models other than the four we usually use. Keeping the std constant

@schema
class AcrossSeedMEIConstantStd(dj.Computed):
    definition = """ # MEI saved at different iterations, created with no clipping and standard deviation constrained to be some number

    -> multi_mei.TargetModel
    -> multi_mei.TargetDataset.Unit
    -> OneStd
    ---
    meis                : longblob  # most exciting images
    iters               : longblob  # list of iterations corresponding to the iteration each MEI comes from
    activations         : longblob  # activations of the models during each iteration
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return multi_mei.TargetModel * multi_mei.TargetDataset.Unit * OneStd & configs.NetworkConfig.CorePlusReadout

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key[
                                                    'net_hash']} & 'seed < 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get initial image (at desired contrast)
        initial_image = torch.randn(1, 1, 36, 64, dtype=torch.float32, device='cuda')
        initial_image = initial_image * float(key['contrast'])

        # Optimize
        postup_op = ops.ChangeStd(float(key['contrast']))
        opt_x, fevals, _ = featurevis.gradient_ascent(model, initial_image,
                                                      step_size=1,
                                                      num_iterations=2000,
                                                      save_iters=50,
                                                      post_update=postup_op)
        opt_x = [initial_image, *opt_x]  # initial image is not returned by featurevis

        self.insert1({**key, 'meis': [m.squeeze().cpu().numpy() for m in opt_x],
                      'iters': list(range(0, 2001, 50)), 'activations': fevals[::50]})


# Generate MEIS on the VGG model
@schema
class VGGMEIConstantStd(dj.Computed):
    definition = """ # MEI saved at different iterations, created in the VGG models

    -> static_models.Model.proj(vgg_net_hash='net_hash', vgg_seed='seed')
    -> multi_mei.TargetDataset.Unit
    -> OneStd
    ---
    meis                : longblob  # most exciting images
    iters               : longblob  # list of iterations corresponding to the iteration each MEI comes from
    activations         : longblob  # activations of the models during each iteration
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        vgg_models = (static_models.Model & {'net_hash': 'cf229e30b96249ceeff8424e06fb7c07', 'seed': 1009}).proj(vgg_net_hash='net_hash',  vgg_seed='seed')
        return vgg_models * multi_mei.TargetDataset.Unit * OneStd & configs.NetworkConfig.CorePlusReadout

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['vgg_net_hash']}
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get initial image (at desired contrast)
        initial_image = torch.randn(1, 1, 36, 64, dtype=torch.float32, device='cuda')
        initial_image = initial_image * float(key['contrast'])

        # Optimize
        postup_op = ops.ChangeStd(float(key['contrast']))
        opt_x, fevals, _ = featurevis.gradient_ascent(model, initial_image,
                                                      step_size=1,
                                                      num_iterations=2000,
                                                      save_iters=50,
                                                      post_update=postup_op)
        opt_x = [initial_image, *opt_x]  # initial image is not returned by featurevis

        self.insert1({**key, 'meis': [m.squeeze().cpu().numpy() for m in opt_x],
                      'iters': list(range(0, 2001, 50)), 'activations': fevals[::50]})


# Same as MEIConstantSTD but with blurring (which hypothetically gets rid of the background stuff)
@schema
class BlurredMEIConstantStd(dj.Computed):
    definition = """ # MEI saved at different iterations, created with no clipping and standard deviation constrained to be 1

    -> multi_mei.TargetModel
    -> multi_mei.TargetDataset.Unit
    -> OneStd
    ---
    meis                : longblob  # most exciting images
    iters               : longblob  # list of iterations corresponding to the iteration each MEI comes from
    activations         : longblob  # activations of the models during each iteration
    """
    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return multi_mei.TargetModel * multi_mei.TargetDataset.Unit * OneStd & configs.NetworkConfig.CorePlusReadout

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed > 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get initial image (at desired contrast)
        initial_image = torch.randn(1, 1, 36, 64, dtype=torch.float32, device='cuda')
        initial_image = initial_image * float(key['contrast'])

        # Optimize
        postup_op = utils.Compose([ops.GaussianBlur(1.5, decay_factor=(1.5 - 0.01) /(1-2000)),
                                   ops.ChangeStd(float(key['contrast']))])
        opt_x, fevals, _ = featurevis.gradient_ascent(model, initial_image, step_size=1,
                                                      num_iterations=2000, save_iters=50,
                                                      post_update=postup_op)
        opt_x = [initial_image, *opt_x] # initial image is not returned by featurevis

        self.insert1({**key, 'meis': [m.squeeze().cpu().numpy() for m in opt_x],
                      'iters': list(range(0, 2001, 50)), 'activations': fevals[::50]})

# Same as MEIConstantSTD but with blurring (which hypothetically gets rid of the background stuff)
@schema
class DeepDrawMEI(dj.Computed):
    definition = """ # MEIs as generated by deepdraw
    
    -> multi_mei.TargetModel
    -> multi_mei.TargetDataset.Unit
    ---
    meis                : longblob  # most exciting images
    iters               : longblob  # list of iterations corresponding to the iteration each MEI comes from
    activations         : longblob  # activations of the models during each iteration
    """
    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return multi_mei.TargetModel * multi_mei.TargetDataset.Unit & configs.NetworkConfig.CorePlusReadout

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed > 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get initial image (at desired contrast)
        initial_image = torch.randn(1, 1, 36, 64, dtype=torch.float32, device='cuda')

        # Set up optimization
        walker_gradient = utils.Compose([ops.FourierSmoothing(0.04),  # not exactly the same as fft_smooth(precond=0.1) but close
                                         ops.DivideByMeanOfAbsolute(),
                                         ops.MultiplyBy(1 / 850, decay_factor=(1 / 850 - 1 / 20400) / (1 - 1000))])  # decays from 1/850 to 1/20400 in 1000 iterations
        bias, scale = 111.28329467773438, 60.922306060791016
        walker_postup = utils.Compose([ops.ClipRange(-bias / scale, (255 - bias) / scale),
                                       ops.GaussianBlur(1.5, decay_factor=(1.5 - 0.01) / (1 - 1000))])  # decays from 1.5 to 0.01 in 1000 iterations

        # Optimize
        opt_x, fevals, reg_values = featurevis.gradient_ascent(model, initial_image,
                                                               step_size=1, save_iters=50,
                                                               num_iterations=1000,
                                                               post_update=walker_postup,
                                                               gradient_f=walker_gradient)
        opt_x = [initial_image, *opt_x] # initial image is not returned by featurevis

        self.insert1({**key, 'meis': [m.squeeze().cpu().numpy() for m in opt_x],
                      'iters': list(range(0, 1001, 50)), 'activations': fevals[::50]})


@schema
class TestAtDiffContrast(dj.Computed):
    definition = """ # tests MEIs generated at diff resolutions at a new resolution
    -> MEIConstantStd
    -> OneStd.proj(target_contrast='contrast')
    ---
    target_activation       :float      #activation at the target contrast
    """
    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed > 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get mei
        mei = (MEIConstantStd & key).fetch1('meis')[20] # at iteration 1000
        mei = mei / float(key['contrast']) * float(key['target_contrast'])
        mei = torch.as_tensor(mei[None, None, ...], dtype=torch.float32, device='cuda')

        # Get activations
        activation = model(mei).item()

        # Insert
        self.insert1({**key, 'target_activation': activation})


@schema
class TestDeepDraw(dj.Computed):
    definition = """ # tests MEIs generated from deepdraw at the target contrast
    -> DeepDrawMEI
    -> OneStd.proj(target_contrast='contrast')
    ---
    target_activation       :float      #activation at the target contrast
    """
    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed > 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get mei
        mei = (DeepDrawMEI & key).fetch1('meis')[-1]
        mei = mei / mei.std() * float(key['target_contrast'])
        mei = torch.as_tensor(mei[None, None, ...], dtype=torch.float32, device='cuda')

        # Get activations
        activation = model(mei).item()

        # Insert
        self.insert1({**key, 'target_activation': activation})

# Same as MEIConstantStd but with a linear model
@schema
class RFConstantStd(dj.Computed):
    definition = """ # MEI saved at different iterations, created with no clipping and standard deviation constrained to be 1

    -> multi_mei.TargetModel
    -> multi_mei.TargetDataset.Unit
    -> OneStd
    ---
    meis                : longblob  # most exciting images
    iters               : longblob  # list of iterations corresponding to the iteration each MEI comes from
    activations         : longblob  # activations of the models during each iteration
    """
    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return multi_mei.TargetModel * multi_mei.TargetDataset.Unit * OneStd & configs.NetworkConfig.CorePlusReadout

    def make(self, key):
        # Get train statistics
        train_stats = multi_mei.prepare_data(key, key['readout_key'])
        _, (_, _, height, width), _, mean_behavior, mean_eyepos, _ = train_stats

        # Get all models
        all_models_rel = static_models.Model & {'group_id': key['group_id'],
                                                'net_hash': key['net_hash']} & 'seed > 1000'
        all_models = [(static_models.Model & mk).load_network() for mk in
                      all_models_rel.fetch('KEY', order_by='seed')]

        # Create model ensemble
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device='cuda')

        # Get initial image (at desired contrast)
        initial_image = torch.randn(1, 1, 36, 64, dtype=torch.float32, device='cuda')
        initial_image = initial_image * float(key['contrast'])

        # Optimize
        postup_op = ops.ChangeStd(float(key['contrast']))
        opt_x, fevals, _ = featurevis.gradient_ascent(model, initial_image, step_size=1,
                                                      num_iterations=2000, save_iters=50,
                                                      post_update=postup_op)
        opt_x = [initial_image, *opt_x] # initial image is not returned by featurevis

        self.insert1({**key, 'meis': [m.squeeze().cpu().numpy() for m in opt_x],
                      'iters': list(range(0, 2001, 50)), 'activations': fevals[::50]})


#########################################################################################
# Check that MEIs generated on different days are the same for the same neuron
import numpy as np
from neuro_data.static_images import stats
from staticnet_experiments import models as static_models

meso = dj.create_virtual_module('pipeline_meso', 'pipeline_meso')

# scans = [{'animal_id': 22564, 'session': 2, 'scan_idx': 12, 'group_id': 46, 'net_hash': '8b6fe18fa651ebf452db0fbd77d05a01'},  # collection_id 2
#          {'animal_id': 22564, 'session': 2, 'scan_idx': 13, 'group_id': 47, 'net_hash': '403a616c4761b733cb767a47a6d0e7da'},  # collection id 3
#          {'animal_id': 22564, 'session': 3, 'scan_idx': 8, 'group_id': 48, 'net_hash': '8b6fe18fa651ebf452db0fbd77d05a01'},  # collection id 4
#          {'animal_id': 22564, 'session': 3, 'scan_idx': 12, 'group_id': 49, 'net_hash': '8b6fe18fa651ebf452db0fbd77d05a01'},  # collection id 6
#          ]

# stacks = [{'animal_id': 22564, 'session': 2, 'stack_idx': 14},
#           {'animal_id': 22564, 'session': 3, 'stack_idx': 10},
#           {'animal_id': 22564, 'session': 4, 'stack_idx': 1},
#           {'animal_id': 22564, 'session': 5, 'stack_idx': 15},
#           ]

stacks = [{'animal_id': 24867, 'session': 3, 'stack_idx': 17},
              {'animal_id': 24867, 'session': 4, 'stack_idx': 14},
              {'animal_id': 24867, 'session': 5, 'stack_idx': 16},
              {'animal_id': 24867, 'session': 6, 'stack_idx': 12},
             ]
scans = [{'animal_id': 24867, 'session': 6, 'scan_idx': 11, 'group_id': 210},
         {'animal_id': 24867, 'session': 6, 'scan_idx': 13, 'group_id': 211},
            ]

@schema
class MatchingParameters(dj.Lookup):
    definition = """ # parameters used for cell matching
    match_params:    int     
    ---
    discard_nonsoma:    bool        # whether to discard those that were not classified as soma
    edge_thresh:        float       # any cell whose centroid is < number of microns to the edge will be discarded (<=0 to not discard)
    oracle_thresh:     float        # any cells with less than this oracle will be discarded
    cnnperf_thresh:    float        # any cells with less than this cnn performance will be discarded
    max_height:         int         # maximum height (in z) allowed for a joint cell
    inscan_distance_thresh: float   # distance in microns that two cells in the same scan have to be to be candidates to be joined together
    inscan_corr_thresh: float       # minimum correlation for two cells to be joined together in the same scan
    outscan_distance_thresh: float  # distance in microns that two cells in diff scans have to be to be candidates to be joined together
    outscan_one_per_scan: bool      # whether a cell in scan 1 can only match with one cell in any of the other scans (cells in the same scan can still be joined in the first stage)
    """
    contents = [{'match_params': i, 'discard_nonsoma': True, 'edge_thresh': 10,
                 'oracle_thresh': -1, 'cnnperf_thresh': -1, 'max_height': 25,
                 'inscan_distance_thresh': 10, 'inscan_corr_thresh': 0.4,
                 'outscan_distance_thresh': d, 'outscan_one_per_scan': True} for i, d in
                enumerate([25, 15], start=1)]


class _MatchedCell():
    """ Coordinates for a set of masks that form a single cell."""

    def __init__(self, group_id, unit_id, coord, trace, oracle_score=None, cnn_score=None):
        self.group_ids = [group_id]
        self.unit_ids = [unit_id]
        self.coords = [coord] # num_stacks x 3
        self.traces = [trace]
        self.centroid = coord # num_stacks x 3

    def join_with(self, other):
        self.group_ids += other.group_ids
        self.unit_ids += other.unit_ids
        self.coords += other.coords
        self.traces += other.traces
        self.centroid = np.mean(self.coords, axis=0)

    def __lt__(self, other):
        """ Used for sorting. """
        return True

def match_cells(cells, max_distance, max_height, min_corr=None,
                allow_same_scan_matching=False):
    """ Match cells based on distance. It iteratively matches the closest pair of cells
    that meet the requirements.

    Arguments:
        cells (list): List of _MatchedCell objects.
        max_distance (float): Maximum distance between two cells to be joined together
        max_height (float): Maximum height (in z) for cells can be matched.
        min_corr (float): Minimum correlation between traces for two cells to be joined.
            If None, ignore correlation.
        allow_same_scan_matching (bool): Whether a cell can be joined with another cell
            in the same scan.

    Returns:
        out_cells: A list of _MatchedCell objects. Cells after all matching.

    Note:
        based on stack.StackSet, bit inefficient but simple.
    """
    import bisect

    # Compute distance matrix (had to do it stack by stack to keep memory manageable)
    coords = np.stack([c.centroid for c in cells]) # num_units x num_stacks x 3
    #distance_matrix = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1)).mean(-1)
    distance_matrix = [np.sqrt(((coords[:, None, i] - coords[None, :, i]) ** 2).sum(-1))
                       for i in range(coords.shape[-1])]
    distance_matrix = np.mean(distance_matrix, axis=0)

    # Order all* pair of cells by distance (not really all, only all that are less than max_distance)
    close_pairs = distance_matrix < max_distance
    close_pairs = np.triu(close_pairs, k=1) # zero diagonal and lower triangle
    candidate_pairs = [(d, cells[i1], cells[i2]) for d, i1, i2 in zip(distance_matrix[close_pairs], *np.nonzero(close_pairs))]
    candidate_pairs = sorted(candidate_pairs)

    # Iteratively join
    out_cells = cells.copy() # shallow copy just to not directly delete stuff from the input list
    while (len(candidate_pairs) > 0):
        # Get next pair of units
        d, unit1, unit2 = candidate_pairs.pop(0)

        # Check whether it's a valid pair
        is_distance_ok = d < max_distance
        is_height_ok = abs(unit1.centroid - unit2.centroid)[:, -1].mean() < max_height
        if min_corr is None:
            is_corr_ok = True
        else:
            traces1 = np.stack(unit1.traces)  # num_traces x num_frames
            traces2 = np.stack(unit2.traces)  # num_traces x num_frames
            corrs = (np.dot(traces1 - traces1.mean(-1), (traces2 - traces2.mean(-1)).T) /
                     np.outer(traces1.std(-1), traces2.std(-1)))
            corrs = corrs / traces1.shape[-1]

            is_corr_ok = corrs.mean() > min_corr
        is_scan_ok = True if allow_same_scan_matching else all([g not in unit2.group_ids
                                                                for g in unit1.group_ids])
        is_valid = is_distance_ok and is_height_ok and is_corr_ok and is_scan_ok

        if is_valid:
            # Remove them from lists
            out_cells.remove(unit1)
            out_cells.remove(unit2)
            f = lambda x: (unit1 not in x[1:]) and (unit2 not in x[1:])
            candidate_pairs = list(filter(f, candidate_pairs))

            # Join them
            unit1.join_with(unit2)

            # Recalculate distances
            coords = np.stack([c.centroid for c in out_cells])  # num_units x num_stacks x 3
            distances = ((coords - unit1.centroid) ** 2).sum(-1).mean(-1)
            for idx in np.nonzero(distances < max_distance)[0]:
                bisect.insort(candidate_pairs, (distances[idx], unit1, out_cells[idx]))

            # Insert new unit
            out_cells.append(unit1)

    return out_cells


# this is pretty ad hoc, will need to be rewritten to be general
@schema
class UnitMatching(dj.Computed):
    definition = """ # match cells from all scans
    -> MatchingParameters
    ---
    matching_ts=CURRENT_TIMESTAMP:  timestamp
    """

    class MatchedCell(dj.Part):
        definition = """ # one matched cell (could be one or more unit_ids)
        -> master
        match_id:       int             # identifier for this match (starts at 1)
        ---
        num_units:      smallint        # number of unit_ids that form this cell
        centroid:       longblob        # centroid in all stacks (num_stacks x 3)
        """

    class Match(dj.Part):
        definition = """ # store the match of all unit ids
        -> master
        group_id:       smallint        # identifier for the scan
        unit_id:        int             # identifier for the unit (start at 1)
        ---
        -> master.MatchedCell
        """

    def make(self, key):
        params = (MatchingParameters &  key).fetch1()

        # Get all cells
        cells = []
        for scan_key in scans:
            # Get unit coordinates in stack space
            xs, ys, zs = (meso.StackCoordinates.UnitInfo & scan_key).fetch('stack_x',
                                                                            'stack_y',
                                                                            'stack_z',
                                                                            order_by='unit_id, stack_session, stack_idx')
            coords = np.stack([xs, ys, zs], axis=-1).reshape(-1, len(stacks), 3) # num_cells x num_stacks x 3
            coords = coords.astype(np.float32)  # save memory when computing distance matrix
            unit_ids = np.arange(1, len(coords) + 1)

            # Get coordinates in pixels (will be used below to discard stuff near the edges)
            px_xs, px_ys = (meso.ScanSet.UnitInfo & scan_key).fetch('px_x', 'px_y',
                                                                    order_by='unit_id')
            px_coords = np.stack([px_xs, px_ys], axis=-1) + 0.5 # so they are in the middle of the pixel, e.g., (0.5,1.5, ... h-1.5, h-0.5)

            # Get traces
            traces = (meso.Fluorescence.Trace * meso.ScanSet.Unit & scan_key).fetch(
                'trace', order_by='unit_id')
            traces = np.stack(traces) # num_cells x num_frames
            traces = traces[:, 2000:-2000] # delete some frames at start and finish just in case

            # Ignore non-somas
            if params['discard_nonsoma']:
                mask_class = (meso.MaskClassification.Type * meso.ScanSet.Unit & scan_key).fetch('type', order_by='unit_id')
                coords = coords[mask_class == 'soma']
                unit_ids = unit_ids[mask_class == 'soma']
                px_coords = px_coords[mask_class == 'soma']
                traces = traces[mask_class == 'soma']
                print(np.count_nonzero(mask_class == 'soma'), 'out of', len(mask_class),
                      'cells remaining after discarding non-soma')

            # Ignore stuff in the wrong cortical area or layer
            pass # all cells are in V1 and layer 2/3 so nothing to do here

            # Get oracle scores (only for masks classified as soma)
            if params['oracle_thresh'] > -1:
                oracle_units, oracle_scores = (stats.Oracle.UnitScores & scan_key).fetch(
                    'unit_id', 'pearson', order_by='unit_id')
                if np.any(unit_ids != oracle_units):
                    # oracle is calculated for the masks classified as soma so if
                    # params['discard_nonsoma'] = False, this will break.
                    raise ValueError('Different set of units from meso and stats.Oracle')
            else:
                oracle_units = unit_ids
                oracle_scores = np.ones_like(oracle_units)

            # Get cnn performance
            if params['cnn_thresh'] > -1:
                cnn_units, cnn_scores = (static_models.Model.UnitTestScores & scan_key).fetch(
                    'unit_id', 'pearson', order_by='unit_id')
                cnn_scores = cnn_scores.reshape(-1, 4).mean(-1) # 4 comes because we train 4 seeds of the same model
                cnn_units = cnn_units.reshape(-1, 4)[:, 0]
                if np.any(unit_ids != cnn_units):
                    raise ValueError('Different set of units from meso and models.Model')
            else:
                cnn_units = unit_ids
                cnn_scores = np.ones_like(cnn_units)

            # Ignore stuff near the edges
            px_h, um_h, px_w, um_w = (meso.ScanInfo.Field & scan_key & {'field': 1}).fetch1(
                'px_height', 'um_height', 'px_width', 'um_width')
            px_thresh = params['edge_thresh'] * np.stack([px_w / um_w, px_h / um_h])
            px_mask = np.logical_and(px_coords > px_thresh,
                                     px_coords < np.array([px_w, px_h]) - px_thresh)
            px_mask = np.logical_and(px_mask[:, 0], px_mask[:, 1])
            coords = coords[px_mask]
            px_coords = px_coords[px_mask]
            unit_ids = unit_ids[px_mask]
            traces = traces[px_mask]
            cnn_scores = cnn_scores[px_mask]
            oracle_scores = oracle_scores[px_mask]
            print(np.count_nonzero(px_mask), 'out of', len(px_mask),
                  'cells remaining after deleting those close to the edge')

            # Ignore if oracle score is too small
            oracle_mask = oracle_scores > params['oracle_thresh']
            coords = coords[oracle_mask]
            px_coords = px_coords[oracle_mask]
            unit_ids = unit_ids[oracle_mask]
            traces = traces[oracle_mask]
            cnn_scores = cnn_scores[oracle_mask]
            oracle_scores = oracle_scores[oracle_mask]
            print(np.count_nonzero(oracle_mask), 'out of', len(oracle_mask),
                  'cells remaining after oracle thresholding')

            # Ignore if cnn is too small
            cnn_mask = cnn_scores > params['cnnperf_thresh']
            coords = coords[cnn_mask]
            px_coords = px_coords[cnn_mask]
            traces = traces[cnn_mask]
            unit_ids = unit_ids[cnn_mask]
            cnn_scores = cnn_scores[cnn_mask]
            oracle_scores = oracle_scores[cnn_mask]
            print(np.count_nonzero(cnn_mask), 'out of', len(cnn_mask),
                  'cells remaining after cnnperf thresholding')

            # Join close (and correlated) cells.
            scan_cells = [_MatchedCell(scan_key['group_id'], u, c, t) for u, c, t in
                          zip(unit_ids, coords, traces)]
            scan_cells = match_cells(scan_cells,
                                     max_distance=params['inscan_distance_thresh'],
                                     max_height=params['max_height'],
                                     min_corr=params['inscan_corr_thresh'],
                                     allow_same_scan_matching=True)
            print(len(scan_cells), 'out of', len(unit_ids),
                  'cells remaining after in-scan matching')

            cells.extend(scan_cells)

        # Join again with those in a different scan (this will take hours(?))
        final_cells = match_cells(cells, max_distance=params['outscan_distance_thresh'],
                                  max_height=params['max_height'],
                                  allow_same_scan_matching=params['outscan_one_per_scan'])
        print(len(final_cells), 'out of', len(cells),
              'cells remaining after scan-to-scan matching')

        # Save stuff
        print('Inserting')
        self.insert1(key)
        for match_id, cell in enumerate(final_cells, start=1):
            self.MatchedCell.insert1({**key, 'match_id': match_id,
                                      'num_units': len(cell.unit_ids),
                                      'centroid': cell.centroid})
            for group_id, unit_id in zip(cell.group_ids, cell.unit_ids):
                self.Match.insert1({**key, 'group_id': group_id, 'unit_id': unit_id,
                                    'match_id': match_id})


@schema
class MatchRanking(dj.Computed):
    definition = """ # rank matches using average oracle and average cnn_performance
    
     -> UnitMatching
    """
    class MatchedCell(dj.Part):

        definition = """
        -> master
        -> UnitMatching.MatchedCell
        ---
        rank:           int             # ranking (starts at 0, low is better)
        f:              float           # value used to produce the rank: (average oracle rank + average cnn rank)/ 2
        avg_oracle:     float           # average oracle value across unit_ids
        std_oracle:     float           # std of oracle value across unit_ids
        avg_cnnscore:   float           # average cnn score value across unit_ids
        std_cnnscore:   float           # std of cnn score value across unit_ids
        """

    def make(self, key):
        # Get all scores for each scan
        oracle_scores = {}
        cnn_scores = {}
        unit_ids = {}
        for scan_key in scans:
            # Get oracle scores for all units
            oracle_units, oracle_score = (stats.Oracle.UnitScores & scan_key).fetch(
                'unit_id', 'pearson', order_by='unit_id')
            unit_ids[scan_key['group_id']] = oracle_units
            oracle_scores[scan_key['group_id']] = oracle_score

            # Get cnn performance
            cnn_score = (static_models.Model.UnitTestScores & scan_key).fetch('pearson',
                                                                              order_by='unit_id')
            cnn_score = cnn_score.reshape(-1, 4).mean(-1)  # 4 comes because we train 4 seeds of the same model
            cnn_scores[scan_key['group_id']] = cnn_score

        # Find the oracle scores and cnn_scores for each match (this takes a minute)
        oracle_scores_ = []
        cnn_scores_ = []
        num_matches = len(UnitMatching.MatchedCell & key)
        for match_id in range(1, num_matches + 1):
            group_ids, units = (UnitMatching.Match & key & {'match_id': match_id}).fetch(
                'group_id', 'unit_id')

            os = []
            cs = []
            for group_id, unit in zip(group_ids, units):
                unit_mask = unit_ids[group_id] == unit
                os.append(oracle_scores[group_id][unit_mask])
                cs.append(cnn_scores[group_id][unit_mask])
            oracle_scores_.append(os)
            cnn_scores_.append(cs)

        # Compute average and std oracle (and cnn) scores
        avg_oracles = [np.mean(os) for os in oracle_scores_]
        std_oracles = [np.std(os) for os in oracle_scores_]
        avg_cnn = [np.mean(cs) for cs in cnn_scores_]
        std_cnn = [np.std(cs) for cs in cnn_scores_]

        # Compute ranking
        oracle_rank = np.argsort(np.argsort(-np.array(avg_oracles)))
        cnn_rank = np.argsort(np.argsort(-np.array(avg_cnn)))
        combined_rank = (oracle_rank + cnn_rank) / 2
        final_rank = np.argsort(np.argsort(combined_rank))

        # Insert
        print('Inserting')
        self.insert1(key)
        for match_id, (ao, so, ac, sc, r, f) in enumerate(
                zip(avg_oracles, std_oracles, avg_cnn, std_cnn, final_rank,
                    combined_rank), start=1):
            self.MatchedCell.insert1({**key, 'match_id': match_id, 'rank': r, 'f': f,
                                      'avg_oracle': ao, 'std_oracle': so,
                                      'avg_cnnscore': ac, 'std_cnnscore': sc})