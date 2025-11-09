
import datajoint as dj
import torch
import numpy as np

from .utils import list_hash, key_hash, deepdraw, process, unprocess, SpatialTransformerPyramid2d, roll

from attorch.regularizers import Laplace

from neuro_data.static_images.data_schemas import StaticMultiDataset, InputResponse, StaticScan, meso
from neuro_data.static_images.configs import DataConfig
from neuro_data.static_images.stats import Oracle

from staticnet_experiments.configs import CoreConfig, ReadoutConfig, ShifterConfig, ModulatorConfig, NetworkConfig
from staticnet_experiments.models import Model


from itertools import count, product, chain
from tqdm import tqdm




#meso = dj.create_virtual_module('meso', 'pipeline_meso')

#schema = dj.schema('eywalker_mei_test')
schema = dj.schema('neurostatic_mei')

#schema = dj.schema('cortex_ex_machina_mesonet_mei_20180503')
#schema = dj.schema('cortex_ex_machina_mesonet_mei')


def best_model(m, aggr=None):
    """
    Given model `m`, returns the best scoring network for each combination of StaticMultiDataset and DataConfig
    :param m: potentially restricted models.Model instance
    :return: best network model for each combination of StaticMultiDataset and DataConfig
    """

    target = m * NetworkConfig.CorePlusReadout()
    aggr_target = StaticMultiDataset() * DataConfig()
    if aggr is not None:
        aggr_target = aggr_target * aggr
    return target * aggr_target.aggr(target, best_corr='max(val_corr)') & 'best_corr = val_corr'


def prepare_data(key, readout_key):
    """
    Given a key to uniquely identify a dataset and a readout key corresponding to a single component within the
    scan, returns information pertinent to generating MEIs

    Args:
        key: a key that can uniquely identify a single entry from StaticMultiDataset * DataConfig
        readout_key: a specific readout key

    Returns:
        trainset, img_shape, mu, mu_beh, mu_eye, s - where mu and s are mean and stdev of input images.
    """
    trainsets, _ = DataConfig().load_data(key)
    trainset = trainsets[readout_key]
    img_shape = trainset.img_shape
    mu = trainset.statistics['images'][trainset.stats_source]['mean'].value.item()
    s = trainset.statistics['images'][trainset.stats_source]['std'].value.item()
    mus = trainset.transformed_mean()
    mu_eye = mus.pupil_center[None, :].to('cuda')
    mu_beh = mus.behavior[None, :].to('cuda')
    return trainset, img_shape, mu, mu_beh, mu_eye, s



@schema
class TargetModel(dj.Computed):
    """
    All models for which we want to generate MEIs
    """
    definition = """
    -> Model
    """

    def make(self, key):
        self.insert1(key)

    def load_model(self, key):
        return Model().load_model(key)

    def fill(self, group_id, dryrun=False):
        restr = {'group_id': group_id}
        core_types = [CoreConfig.Stacked2d, CoreConfig.Linear]
        # check if downsample should be made 0
        ro_types = [ReadoutConfig.ModifiedSpatialTransformerPyramid2d & 'downsample=1',
                    ReadoutConfig.ModifiedSpatialTransformerPyramid2d & 'downsample=0',
                    ReadoutConfig.SpatialTransformerPooled2d,
                    ReadoutConfig.SpatialTransformerPyramid2d & 'downsample=1',
                    ReadoutConfig.SpatialTransformerPyramid2d & 'downsample=0']
        shifter_types = [ShifterConfig.MLP, ShifterConfig.StaticAffine2d]
        mod_types = [ModulatorConfig.MLP]
        data_types = [DataConfig.AreaLayer & dict(stimulus_type='stimulus.Frame', exclude='images,responses')]

        conds = [dj.AndList(comb) for comb in chain(
            product(core_types, ro_types, shifter_types, mod_types, data_types)
        )]


        targets = []
        for cond in conds:
            keys = best_model(Model & (NetworkConfig.CorePlusReadout & cond) & restr).fetch('KEY')
            if len(keys):
              # work around for DataJoint bug #492
              targets.append(keys)

        if not dryrun:
            self.populate(targets)

        return targets

    # def fill(self, group_id=8, dryrun=False):
    #     restr = {'group_id': group_id}
    #     core_types = [CoreConfig.Stacked2d, CoreConfig.Linear]
    #     ro_types = [ReadoutConfig.SpatialTransformerPyramid2d & 'downsample=0']
    #     alex_ro = [ReadoutConfig.SpatialXFeatures]
    #     shifter_types = [ShifterConfig.MLP]
    #     no_shifter = [ShifterConfig.No]
    #     mod_types = [ModulatorConfig.MLP]
    #
    #     conds = [dj.AndList(comb) for comb in chain(
    #         product(core_types, ro_types, shifter_types, mod_types),
    #         product(core_types, alex_ro, no_shifter, mod_types),
    #         (
    #             (CoreConfig.VGG & 'core_hash = "49fbd0b012a1e83ae56e72d1db630b0f"', ReadoutConfig.SpatialTransformerPyramid2d & 'downsample=0', ShifterConfig.MLP, ModulatorConfig.MLP),
    #         )
    #     )]
    #     targets = MesoNetMultiDataset() * DataConfig.AreaLayer() & restr
    #
    #     all_rels = []
    #     for cond in conds:
    #         rels = (Encoder() & cond & targets).best_model()
    #         all_rels.append(rels)
    #         if not dryrun:
    #             self.populate(rels)
    #             TopModel.populate(rels)
    #
    #     # add seed variants
    #     if not dryrun:
    #         self.populate(Encoder & self.proj(seed_old='seed'))
    #     return conds, all_rels


@schema
class TargetDataset(dj.Computed):
    """
    Register target dataset and all of their units
    """

    definition = """
    -> StaticMultiDataset
    -> DataConfig
    """

    class Unit(dj.Part):
        definition = """
        -> master
        readout_key : varchar(128)
        neuron_id  : smallint
        ---
        -> StaticMultiDataset.Member
        -> StaticScan.Unit
        brain_area : varchar(16)   # brain area
        layer:   varchar(16)       # layer
        """

    def make(self, key):
        # get the model
        self.insert1(key)

        # get the training set
        trainsets, _ = DataConfig().load_data(key, tier='train')

        for readout_key in trainsets.keys():
            ro_tuple = dict(key)
            ro_tuple['readout_key'] = readout_key
            member_key = (StaticMultiDataset.Member() & key & 'name="{}"'.format(readout_key)).fetch1('KEY')

            ro_tuple = dict(ro_tuple, **member_key)

            trainset = trainsets[readout_key]

            unit_ids = trainset.neurons.unit_ids
            areas = trainset.neurons.area
            layers = trainset.neurons.layer

            for neuron_id, unit_id, brain_area, layer in zip(count(), unit_ids, areas, layers):
                tuple = dict(ro_tuple)
                tuple['neuron_id'] = neuron_id
                tuple['unit_id'] = unit_id.item()
                tuple['brain_area'] = brain_area.item()
                tuple['layer'] = layer.item()
                self.Unit().insert1(tuple)


@schema
class HighOracleUnits(dj.Computed):
    """
    Fill with top 25% units for each dataset group
    """
    definition = """
    -> TargetDataset.Unit
    """

    key_source = TargetDataset()

    def make(self, key):
        corr = (Oracle.UnitScores & key).fetch('pearson')
        thr = np.percentile(corr, 75)
        self.insert((TargetDataset.Unit & (Oracle.UnitScores & 'pearson > {}'.format(thr) & key)).proj())
        #
        # target_units = TargetDataset.Unit() & key
        # units = Oracle.UnitScores() & target_units
        #
        # corr = units.fetch('pearson')
        # cutoff = np.percentile(corr, 75)
        # r = units & 'pearson > {}'.format(cutoff)
        # self.insert((TargetDataset.Unit() & key & r).proj())

@schema
class OracleRankedUnit(dj.Computed):
    definition = """
    # ranked by the oracle score
    -> TargetDataset.Unit
    ---
    rank: int    # ranking
    oracle_corr: float   # oracle correlation
    top_ptile: float     # percentile from the top
    """
    key_source = TargetDataset()
    def make(self, key):
        keys, corr = (TargetDataset.Unit * Oracle.UnitScores & key).fetch('KEY', 'pearson', order_by='pearson DESC')
        N = len(keys)
        unit_keys = []
        for i, k, c in zip(count(), keys, corr):
            unit_key = dict(k)
            unit_key['rank'] = i
            unit_key['top_ptile'] = i / N
            unit_key['oracle_corr'] = c
            unit_keys.append(unit_key)
        self.insert(unit_keys, ignore_extra_fields=True)



@schema
class ModelGroup(dj.Computed):
    definition = """
    -> TargetDataset
    """

    @property
    def key_source(self):
        return TargetDataset & (NetworkConfig.CorePlusReadout * TargetModel)

    def make(self, key):
        best_cnn = best_model(Model & key & (NetworkConfig.CorePlusReadout & TargetModel & key & [CoreConfig.SigmoidLaplace, CoreConfig.GaussianLaplace]))
        best_lin = best_model(Model & key & (NetworkConfig.CorePlusReadout & TargetModel & key & [CoreConfig.LinearSigmoidLaplace, CoreConfig.LinearGaussianLaplace, CoreConfig.StackedLinearGaussianLaplace]))

        self.insert1(key)
        self.CNNModel().insert1(best_cnn.fetch1('KEY'))
        self.LinearModel().insert1(best_lin.fetch1('KEY'))
        #self.VGGModel().insert1(best_vgg)


    class CNNModel(dj.Part):
        definition = """
        -> master
        -> TargetModel
        """

    class LinearModel(dj.Part):
        definition = """
        -> master
        -> TargetModel
        """

    # class VGGModel(dj.Part):
    #     definition = """
    #     -> master
    #     -> CoreConfig.VGG
    #     -> TargetModel
    #     """


@schema
class HighUnitSelection(dj.Computed):
    definition = """
    -> TargetDataset.Unit
    ---
    hu_rank: int  # ranking
    """
    # Selection criteria
    # top 25 % oracle
    # top 25 % in fraction oracle performance on CNN and on Linear
    # top 50 neurons with largest performance diff CNN > Linear
    # excluding units that are closer than 25um in x,y plane


    key_source = ModelGroup()

    def make(self, key):

        g = (TargetDataset.Unit & key & HighOracleUnits)
        r = Oracle.UnitScores() & g

        cnn_scores = Model.UnitTestScores & (ModelGroup.CNNModel() & key)
        lin_scores = Model.UnitTestScores & (ModelGroup.LinearModel() & key)

        all_data = r * cnn_scores.proj(cnn_hash='net_hash', cnn_corr='pearson', cnn_seed='seed') * lin_scores.proj(
            lin_hash='net_hash', lin_corr='pearson', lin_seed='seed') * meso.ScanSet.UnitInfo()


        keys, oracle_corr, cnn_corr, lin_corr, posx, posy, posz = all_data.fetch('KEY', 'pearson', 'cnn_corr', 'lin_corr',
                                                                           'um_x', 'um_y', 'um_z')
        keys = np.array(keys)

        cnn_fraction = cnn_corr / oracle_corr
        lin_fraction = lin_corr / oracle_corr
        cnn_thr = np.percentile(cnn_fraction, 75)
        lin_thr = np.percentile(lin_fraction, 75)
        cnn_selection = cnn_fraction > cnn_thr
        lin_selection = lin_fraction > lin_thr
        pos = np.where(cnn_selection & lin_selection)[0]
        print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        print('%d units remaining' % len(pos))

        delta = (cnn_fraction - lin_fraction)[pos]

        sorted_ordering = pos[np.argsort(delta)[::-1]]

        px = posx[sorted_ordering]
        py = posy[sorted_ordering]
        pz = posz[sorted_ordering]
        good_keys = keys[sorted_ordering]
        xy_nearby = np.sqrt((px[:, None] - px) ** 2 + (py[:, None] - py) ** 2) < 25
        z_nearby = np.abs(pz[:, None] - pz) < 50
        nearby = xy_nearby & z_nearby

        checks = np.tril(nearby, k=-1)

        N = len(sorted_ordering)

        include = np.ones(N).astype(bool)

        #
        for i in range(N):
            if (checks[i] * include).sum() > 0:
                include[i] = False

        print('After distance check, {} units remain'.format(sum(include)))
        assert not np.any(checks[include][:, include]), 'Overlap still remains. Check the algorithm'

        final_keys = good_keys[include]

        for i, k in enumerate(final_keys):
            unit_key = (TargetDataset.Unit & k).fetch1('KEY')
            unit_key['hu_rank'] = i
            self.insert1(unit_key)


        #
        # g = (TargetDataset.Unit & key & HighOracleUnits)
        #
        # r = MesoNetAllOracle.UnitScores() & g
        #
        # common = ['group_id', 'data_hash', 'train_hash', 'animal_id', 'session', 'scan_idx', 'window', 'unit_id']
        # cnn_scores = rename(Encoder.UnitScores() & (ModelGroup.CNNModel() & g), prefix='cnn_', exclude=common)
        # lin_scores = rename(Encoder.UnitScores() & (ModelGroup.LinearModel() & g), prefix='lin_', exclude=common)
        # a = cnn_scores * lin_scores * r
        # keys, cnn_corr, lin_corr, oracle = a.fetch(dj.key, 'cnn_test_corr', 'lin_test_corr', 'pearson')
        #
        # cnn_fraction = cnn_corr / oracle
        # lin_fraction = lin_corr / oracle
        # cnn_thr = np.percentile(cnn_fraction, 25)
        # # lin_thr = np.percentile(lin_fraction, 50)
        # cnn_selection = cnn_fraction > cnn_thr
        # #lin_selection = lin_fraction > lin_thr
        # #pos = np.where(cnn_selection & lin_selection)[0]
        # pos = np.where(cnn_selection)[0]
        # #print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        # print('%d units remaining' % len(pos))
        #
        # delta = (cnn_fraction - lin_fraction)[pos]
        # delta_thr = np.percentile(delta, 75)
        # pos = pos[delta >= delta_thr]
        #
        # sub = [keys[p] for p in pos]
        #
        # print('Adding %d units' % len(sub))
        # self.insert(TargetDataset.Unit() & sub, ignore_extra_fields=True)

@schema
class CorrectedHighUnitSelection(dj.Computed):
    definition = """
    -> TargetDataset.Unit
    ---
    hu_rank: int  # ranking
    cnn_fraction_oracle: float # CNN model fraction oracle score
    lin_fraction_oracle: float # Linear model fraction oracle score
    """
    # Selection criteria
    # top 50 % oracle
    # top 30 % in fraction oracle performance on CNN and on Linear
    # top neurons with largest performance diff CNN > Linear
    # excluding units that are closer than 25um in x,y plane


    key_source = ModelGroup()

    def make(self, key):

        g = (TargetDataset.Unit & key & (OracleRankedUnit & 'top_ptile < 0.50'))
        r = Oracle.UnitScores() & g

        h, w, cx, cy = (meso.ScanInfo.Field & (TargetDataset.Unit & key) & 'field = 1').fetch1('um_height', 'um_width',
                                                                                               'x', 'y')

        cnn_scores = Model.UnitTestScores & (ModelGroup.CNNModel() & key)
        lin_scores = Model.UnitTestScores & (ModelGroup.LinearModel() & key)

        all_data = r * cnn_scores.proj(cnn_hash='net_hash', cnn_corr='pearson', cnn_seed='seed') * lin_scores.proj(
            lin_hash='net_hash', lin_corr='pearson', lin_seed='seed') * meso.ScanSet.UnitInfo()

        # exclude neurons close to the edge of the field
        dist_thr = 10  # distance for exclusion
        n_before = len(all_data)
        all_data = all_data & 'ABS(um_y - {}) < {}'.format(cy, h / 2 - dist_thr) & 'ABS(um_x - {}) < {}'.format(cx,
                                                                                                                w / 2 - dist_thr)
        n_after = len(all_data)

        print('Excluded {} / {} neurons lying within {} of edge'.format(n_before - n_after, n_before, dist_thr))

        keys, oracle_corr, cnn_corr, lin_corr, posx, posy, posz = all_data.fetch('KEY', 'pearson', 'cnn_corr',
                                                                                 'lin_corr',
                                                                                 'um_x', 'um_y', 'um_z')
        keys = np.array(keys)

        cnn_fraction = cnn_corr / oracle_corr
        lin_fraction = lin_corr / oracle_corr
        cnn_thr = np.percentile(cnn_fraction, 70)
        lin_thr = np.percentile(lin_fraction, 70)
        cnn_selection = cnn_fraction > cnn_thr
        lin_selection = lin_fraction > lin_thr
        pos = np.where(cnn_selection & lin_selection)[0]
        print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        print('%d units remaining' % len(pos))

        delta = (cnn_fraction - lin_fraction)[pos]

        sorted_ordering = pos[np.argsort(delta)[::-1]]

        px = posx[sorted_ordering]
        py = posy[sorted_ordering]
        pz = posz[sorted_ordering]
        good_keys = keys[sorted_ordering]
        xyz_dist = np.sqrt((px[:, None] - px) ** 2 + (py[:, None] - py) ** 2 + (pz[:, None] - pz) ** 2)
        nearby = xyz_dist < 20

        checks = np.tril(nearby, k=-1)

        N = len(sorted_ordering)

        include = np.ones(N).astype(bool)

        #
        for i in range(N):
            if (checks[i] * include).sum() > 0:
                include[i] = False

        final_index = sorted_ordering[include]

        print('After distance check, {} units remain'.format(sum(include)))
        assert not np.any(checks[include][:, include]), 'Overlap still remains. Check the algorithm'

        final_keys = good_keys[include]
        cnn_f = cnn_fraction[final_index]
        lin_f = lin_fraction[final_index]

        for i, k, cf, lf in zip(count(), final_keys, cnn_f, lin_f):
            unit_key = (TargetDataset.Unit & k).fetch1('KEY')
            unit_key['hu_rank'] = i
            unit_key['cnn_fraction_oracle'] = cf
            unit_key['lin_fraction_oracle'] = lf
            self.insert1(unit_key)


        #
        # g = (TargetDataset.Unit & key & HighOracleUnits)
        #
        # r = MesoNetAllOracle.UnitScores() & g
        #
        # common = ['group_id', 'data_hash', 'train_hash', 'animal_id', 'session', 'scan_idx', 'window', 'unit_id']
        # cnn_scores = rename(Encoder.UnitScores() & (ModelGroup.CNNModel() & g), prefix='cnn_', exclude=common)
        # lin_scores = rename(Encoder.UnitScores() & (ModelGroup.LinearModel() & g), prefix='lin_', exclude=common)
        # a = cnn_scores * lin_scores * r
        # keys, cnn_corr, lin_corr, oracle = a.fetch(dj.key, 'cnn_test_corr', 'lin_test_corr', 'pearson')
        #
        # cnn_fraction = cnn_corr / oracle
        # lin_fraction = lin_corr / oracle
        # cnn_thr = np.percentile(cnn_fraction, 25)
        # # lin_thr = np.percentile(lin_fraction, 50)
        # cnn_selection = cnn_fraction > cnn_thr
        # #lin_selection = lin_fraction > lin_thr
        # #pos = np.where(cnn_selection & lin_selection)[0]
        # pos = np.where(cnn_selection)[0]
        # #print('Overlap = %.2f %%' % (100 * len(pos) / (cnn_selection.sum() + lin_selection.sum() - len(pos))))
        # print('%d units remaining' % len(pos))
        #
        # delta = (cnn_fraction - lin_fraction)[pos]
        # delta_thr = np.percentile(delta, 75)
        # pos = pos[delta >= delta_thr]
        #
        # sub = [keys[p] for p in pos]
        #
        # print('Adding %d units' % len(sub))
        # self.insert(TargetDataset.Unit() & sub, ignore_extra_fields=True)



def compute_rf(model, readout_key, dataset, neuron_id, post_process=True, base=None, eye_pos=None, behavior=None):
    """
    Computes gradient based receptive field for the readout neuron of the model
    as specified by readout_key and neuron_id. dataset is used to determine the input image size
    and to adjust the mean and standard deviation of the generated RF image.
    Args:
        model: CorePlusReadout model
        readout_key: target readout layer key
        dataset: Tensor dataset object
        neuron_id: target neuron id to compute RF for
        post_process: if True will adjust the RF to match dataset statistics (mean and std)

    Returns:
        Gradient based RF image
    """
    rf_shape = list(dataset.img_shape[1:])

    if base is None:
        base = torch.zeros(1, *rf_shape, requires_grad=True).cuda()
    else:
        base = base.requires_grad_()

    if eye_pos is not None:
        eye_pos = eye_pos.cuda()
    if behavior is not None:
        behavior = behavior.cuda()

    output = model(base, readout_key, eye_pos=eye_pos, behavior=behavior)
    output[:, neuron_id].backward()
    gradrf = base.grad.data.cpu().numpy().squeeze()
    if post_process:
        mu, s = map(lambda x: float(np.array(x).squeeze()),
                    [dataset.input_statistics['mean'], dataset.input_statistics['std']])
        return (gradrf * s + mu).squeeze()
    else:
        return gradrf


def rand_images(img_shape=(1, 36, 64), n_images=1):
    """
    Generates n_images random images of shape img_shape
    Args:
        img_shape: shape of the image. Needs to at least specificy the [channel, height, width]. If larger than length
        3, only the last three dimensions are used to infer necessary image size information.
        n_images: number of random images to generate. Defaults to 1

    Returns:
        A numpy array representing a stack of random images, with dimension [n_images, channel, height, width].

    """
    return np.random.randn(n_images, *img_shape[-3:]).astype(np.float32)


def compute_mei(model, readout_key, neuron_id, x0, radius=25, l0=2, iterations=5000, samples=None,
                laplace0=True, gamma0=0.1, gamma_distance=0, lr=0.005, constrain=True, norm_type='L2'):
    """
    Given a model, compute the most exciting image (MEI) for the readout neuron as specified by the readout_key and
    neuron_id. MEI is generated by performing gradient ascent on the input image to the model with respect to the
    activity of the target neuron. If constrain=True (default), then the generated image(s) are constrained to have either
    a fixed L2 norm on the image or a fixed L2 norm on Laplacian on the image. `radius` specifies the norm magnitude.
    During the optimization, the `l0` norm of either the laplacian on the image (`laplace0=True`) or the `l0` norm of the
    image(s) as is (`lapace0=False`) is incorporated as a regularizer
    with weight `gamma0`.
    If `x0` specifies more than one image, then average dot products between all images is added as a regularizer with
    weight `gamma_distance`.
    The optimization is performed for `iterations` iterations with learning rate `lr`. `samples` can be used to capture
    the in progress image(s) at the specified iterations.

    Args:
        model: CorePlusReadout model
        readout_key: target readout layer key
        neuron_id: target neuron within the readout layer to generate MEI for
        x0: starting image
        radius: constrain norm radius
        l0: order of the image norm regularizer
        iterations: how many iterations to perform optimization
        samples: A list of iterations at which to sample the image during the MEI optimization. If left to None
        (default), no samples will be taken.
        laplace0: whether to take norm of lapacian on the image as the regularizer
        gamma0: weight on the image norm regularizer
        gamma_distance: weight on the image similarity (dot product) regularizer
        lr: learning rate for the optimization
        constrain: if True, images norm is constrained during optimization
        norm_type: The type of image norm to be used when constraining image norm. May be `l2` or `laplace`

    Returns:
        (mei_images, mei_path, activations)
            mei_images: the final stack of MEIs
            mei_path: sequence of MEIs sampled at iterations specified by `samples`
            activations: activations of the target neuron at all images in mei_path
    """
    if samples is None:
        samples = []
    # --- Compute Most Exciting Image

    laplace = Laplace().cuda()

    def normalize(x):
        x = x - x.mean(2, keepdim=True).mean(3, keepdim=True)
        if constrain:
            if norm_type == 'laplace':
                # Constrain the L2 over Laplace filtered image
                x = x / laplace(x).pow(2).sum(2, keepdim=True).sum(3, keepdim=True).sqrt() * radius
            else:
                # Constrain the L2 norm over the image
                x = x / x.pow(2).sum(2, keepdim=True).sum(3, keepdim=True).sqrt() * radius
        return x

    if x0.ndim < 4:
        x0 = x0.reshape((1,) * (4 - x0.ndim) + x0.shape)
    n_images = x0.shape[0]
    X = torch.tensor(x0, device='cuda', requires_grad=True)
    X.data = normalize(X.data)

    optimizer = torch.optim.Adam([X], lr=lr)

    def closure():
        optimizer.zero_grad()
        X_norm = normalize(X)

        r = model(X_norm, readout_key)

        if laplace0:
            # take Laplace over image
            X_reg = laplace(X_norm)
        else:
            X_reg = X_norm
        activation = r[:, neuron_id]
        loss = - activation.mean() + gamma0 * X_reg.abs().pow(l0).mean()

        if n_images > 1:
            X_flat = X_norm.view(n_images, -1)
            G = X_flat @ X_flat.t()
            loss = loss + gamma_distance * G.mean()

        # update the gradients
        loss.backward()
        return activation, loss

    mei_path = []
    activations = []
    for i in tqdm(range(iterations)):
        activation, _ = closure()
        if i in samples:
            mei_path.append(X.data.cpu().numpy())
            activations.append(activation.data.cpu().numpy())
        optimizer.step()
        # perform in-place normalization
        X.data = normalize(X.data)
    mei_path = np.stack(mei_path)
    activations = np.stack(activations)
    return X.data.cpu(), mei_path, activations


def contrast_tuning(model, img, bias, scale, min_contrast=0.01, n=1000, linear=True, use_max_lim=False):
    mu = img.mean()
    delta = img - img.mean()
    vmax = delta.max()
    vmin = delta.min()

    min_pdist = delta[delta > 0].min()
    min_ndist = (-delta[delta < 0]).min()

    max_lim_gain = max((255 - mu) / min_pdist, mu / min_ndist)

    base_contrast = img.std()

    lim_contrast = 255 / (vmax - vmin) * base_contrast # maximum possible reachable contrast without clipping
    min_gain = min_contrast / base_contrast
    max_gain = min((255 - mu) / vmax, -mu / vmin)

    def run(x):
        with torch.no_grad():
            img = torch.Tensor(process(x[..., None], mu=bias, sigma=scale)[None, ...]).cuda()
            result = model(img)
        return result

    target = max_lim_gain if use_max_lim else max_gain

    if linear:
        gains = np.linspace(min_gain, target, n)
    else:
        gains = np.logspace(np.log10(min_gain), np.log10(target), n)
    vals = []
    cont = []

    for g in tqdm(gains):
        img = delta * g + mu
        img = np.clip(img, 0, 255)
        c = img.std()
        v = run(img).data.cpu().numpy()[0]
        cont.append(c)
        vals.append(v)

    vals = np.array(vals)
    cont = np.array(cont)

    return cont, vals, lim_contrast


def adjust_contrast(img, contrast=-1, mu=-1, force=False, verbose=False, steps=1000):
    current_contrast = img.std()

    if contrast < 0:
        gain = 1   # no adjustment of contrast
    else:
        gain = contrast / current_contrast

    delta = img - img.mean()
    if mu is None or mu < 0:
        mu = img.mean()

    min_pdist = delta[delta > 0].min()
    min_ndist = (-delta[delta < 0]).min()

    max_lim_gain = max((255 - mu) / min_pdist, mu / min_ndist)


    vmax = delta.max()
    vmin = delta.min()

    max_gain = min((255 - mu) / vmax, -mu / vmin)
    clipped = gain > max_gain
    v = np.linspace(0, 50, 100)
    if clipped and force:
        if verbose:
            print('Adjusting...')
        cont = []
        imgs = []
        gains = np.logspace(np.log10(gain), np.log10(max_lim_gain), steps)
        # for each gain, perform offset adjustment such that the mean is equal to the set value
        for g in gains:
            img = delta * g + mu
            img = np.clip(img, 0, 255)
            offset = img.mean() - mu
            if offset < 0:
                offset = -offset
                mask = (255-img < v[:, None, None])
                nlow = mask.sum(axis=(1, 2))
                nhigh = img.size - nlow
                va = ((mask * (255-img)).sum(axis=(1, 2)) + v * nhigh) / (nlow + nhigh)
                pos = np.argmin(np.abs(va - offset))
                actual_offset = -v[pos]
            else:
                mask = (img < v[:, None, None])
                nlow = mask.sum(axis=(1, 2))
                nhigh = img.size - nlow
                va = ((mask * img).sum(axis=(1, 2)) + v * nhigh) / (nlow + nhigh)
                pos = np.argmin(np.abs(va - offset))
                actual_offset = v[pos]
            img = img - actual_offset
            img = np.clip(img, 0, 255)
            c = img.std()
            cont.append(c)
            imgs.append(img)
            if c > contrast:
                break
        loc = np.argmin(np.abs(np.array(cont) - contrast))
        adj_img = imgs[loc]
    else:
        adj_img = delta * gain + mu
        adj_img = np.clip(adj_img, 0, 255)
    actual_contrast = adj_img.std()
    return adj_img, clipped, actual_contrast


def normalize(img, radius=1):
    """
    Normalize image to L2 norm of radius
    """
    img = img - img.mean(-2, keepdim=True).mean(-1, keepdim=True)
    norms = img.pow(2).sum(-2, keepdim=True).sum(-1, keepdim=True).sqrt()
    return img / norms * radius



@schema
class TopModel(dj.Computed):
    definition = """
    -> TargetModel
    """
    def make(self, key):
        self.insert1(key)



@schema
class TargetModelAlias(dj.Manual):
    definition = """
    model_hash: varchar(128)
    ---
    -> TargetModel
    """

    def fill(self):
        keys = (TargetModel() - self).fetch('KEY')
        self.insert([dict(k, model_hash=key_hash(k)) for k in keys])


@schema
class MEIParameter(dj.Lookup):
    definition = """
    # parameters for the MEI generation

    mei_param_id        : varchar(64)  # id
    ---
    iter_n              : int   # number of iterations to run
    start_sigma         : float # starting sigma value
    end_sigma           : float # ending sigma value
    start_step_size     : float # starting step size
    end_step_size       : float # ending step size
    precond             : float # strength of gradient preconditioning filter falloff
    step_gain           : float # scaling of gradient steps
    jitter              : int   #size of translational jittering
    blur                : bool  # whether to apply bluring or not
    """

    @property
    def contents(self):
        yield from map(lambda x: (list_hash(x),) + x,
           (
               (600, 1.5, 0.01, 3.0, 0.125, 0.5, 0.1, 0, True),
               (1000, 1.5, 0.01, 3.0, 0.125, 0.5, 0.1, 0, True),
               (1000, 1.5, 0.01, 3.0, 0.125, 0.2, 0.1, 0, True),
               (600, 1.5, 0.01, 3.0, 0.125, 0.1, 0.1, 0, True),
               (1000, 1.5, 0.01, 3.0, 0.125, 0.1, 0.1, 0, True),
               (1000, 1.5, 0.01, 3.0, 0.125, 0.0, 0.1, 0, False),
               (1000, 1.5, 0.01, 3.0, 0.125, 0.1, 0.1, 0, False),
               (600, 1.5, 0.01, 3.0, 0.125, 0.0, 0.1, 0, False),
               (800, 1.5, 0.01, 3.0, 0.125, 0.5, 0.1, 0, False),
               (800, 1.5, 0.01, 3.0, 0.125, 0.3, 0.1, 0, False),
               (800, 1.5, 0.01, 3.0, 0.125, 0.2, 0.1, 0, False),
               (800, 0.5, 0.01, 3.0, 0.125, 0.1, 0.1, 0, True),
               (800, 1.0, 0.01, 3.0, 0.125, 0.1, 0.1, 0, True),

           )
        )


@schema
class MEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    mei                 : longblob  # most exciting images
    activation          : float     # activation at the MEI
    monotonic           : bool      # does activity increase monotonically with contrast
    max_contrast        : float     # contrast at which maximum activity is achived
    max_activation      : float     # activation at the maximum contrast
    sat_contrast        : float     # contrast at which image would start saturating
    img_mean            : float     # mean luminance of the image
    lim_contrast        : float     # max reachable contrast without clipping
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return TargetModel() * MEIParameter() * TargetDataset.Unit & NetworkConfig.CorePlusReadout


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model

        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        print('Working with images with mu={}, sigma={}'.format(bias, scale))

        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)


        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(adj_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale)

        mei = gen_image.squeeze()

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')
            activation = adj_model(img).data.cpu().numpy()[0]

        cont, vals, lim_contrast = contrast_tuning(adj_model, mei, bias, scale)

        key['mei'] = mei
        key['activation'] = activation
        key['monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['max_activation'] = np.max(vals)
        key['max_contrast'] = cont[np.argmax(vals)]
        key['sat_contrast'] = np.max(cont)
        key['img_mean'] = mei.mean()
        key['lim_contrast'] = lim_contrast

        self.insert1(key)


@schema
class MultiSeedMEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    n_seeds             : int       # number of distinct seeded models used
    mei                 : longblob  # most exciting images
    activation          : float     # activation at the MEI
    monotonic           : bool      # does activity increase monotonically with contrast
    max_contrast        : float     # contrast at which maximum activity is achived
    max_activation      : float     # activation at the maximum contrast
    sat_contrast        : float     # contrast at which image would start saturating
    img_mean            : float     # mean luminance of the image
    lim_contrast        : float     # max reachable contrast without clipping
    """

    @property
    def key_source(self):
        # restriction by CorePlusReadout is needed to link dataconfig with the appropriate model
        return TargetModel() * MEIParameter() * TargetDataset.Unit & NetworkConfig.CorePlusReadout

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model
        key_seed = dict(key)
        key_seed.pop('seed')
        model_keys =  (Model() & key_seed).fetch('KEY')

        models = []
        for mk in model_keys:
            model = (Model() & mk).load_network().to('cuda')
            model.eval()
            models.append(model)

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        print('Working with images with mu={}, sigma={}'.format(bias, scale))

        def adj_model(x):
            resp = 0
            for m in models:
                resp = resp + m(x, readout_key, eye_pos=mu_eye)[:, neuron_id]
            return resp / len(models)

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
            {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size': float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(adj_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale)

        mei = gen_image.squeeze()

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')
            activation = adj_model(img).data.cpu().numpy()[0]

        cont, vals, lim_contrast = contrast_tuning(adj_model, mei, bias, scale)

        key['n_seeds'] = len(models)
        key['mei'] = mei
        key['activation'] = activation
        key['monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['max_activation'] = np.max(vals)
        key['max_contrast'] = cont[np.argmax(vals)]
        key['sat_contrast'] = np.max(cont)
        key['img_mean'] = mei.mean()
        key['lim_contrast'] = lim_contrast

        self.insert1(key)


@schema
class ImageConfig(dj.Lookup):
    definition = """
    img_config_id: int
    ---
    img_mean: float   # image mean to use. -1 would use original image mean.
    img_contrast: float   # image contrast to use. -1 would use original image contrast.
    force_stats: bool     # whether to make forcible adjustment on the stats
    """
    contents = [
        (0, 111.0, 16.0, True),
       # (1, 112.0, 6.0, False),
    ]


#(0, 112.0, 6.0, False),
#(1, 112.0, 35.0, True),


@schema
class MEIActivation(dj.Computed):
    definition = """
    -> MEI
    -> ImageConfig
    ---
    mei_activation: float   # activation on mei
    mei_clipped: bool       # whether image was clipped
    mei_contrast: float     # actual contrast of mei
    """

    def make(self, key):
        mei = (MEI() & key).fetch1('mei')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast', 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)


        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activation = adj_model(img).data.cpu().numpy()[0]

        key['mei_activation'] = activation
        key['mei_clipped'] = bool(clipped)
        key['mei_contrast'] = actual_contrast

        self.insert1(key)


@schema
class JitterConfig(dj.Lookup):
    definition = """
    jitter_config_id:   int
    ---
    jitter_size:   int
    """
    contents = [(0, 5)]


@schema
class JitterInPlace(dj.Computed):
    definition = """
    -> MEI
    -> ImageConfig
    -> JitterConfig
    ---
    jitter_activations: longblob      # activation resulting from jitter
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        mei = (MEI() & key).fetch1('mei')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        jitter_size = int((JitterConfig & key).fetch1('jitter_size'))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]

        key['jitter_activations'] = activations

        self.insert1(key)

@schema
class TruncScale(dj.Lookup):
    definition = """
    scale_idx: int   # index of truncation scale
    """
    contents = zip([0, 1, 2])


@schema
class TruncCenterMEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    -> TruncScale
    ---
    disc_mei                 : longblob  # center position based MEI
    disc_activation          : float     # activation of the center unit
    disc_monotonic           : bool      # does activity increase monotonically with contrast
    disc_max_contrast        : float     # contrast at which maximum activity is achieved
    disc_max_activation      : float     # activation at the maximum contrast
    disc_sat_contrast        : float     # contrast at which image would start saturating
    disc_mean                : float     # mean luminance of the image
    disc_lim_contrast        : float     # max reachable contrast without clipping
    """
    #
    # key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
    #              & ReadoutConfig.SpatialTransformerPyramid2d()

    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit * TruncScale &\
                 (NetworkConfig.CorePlusReadout &
                  [ReadoutConfig.SpatialTransformerPyramid2d, ReadoutConfig.ModifiedSpatialTransformerPyramid2d])


    def make(self, key):
        scale_idx = (TruncScale & key).fetch1('scale_idx')
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)


        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)


        # generate class visualization via octavewise gradient ascent

        with SpatialTransformerPyramid2d.trunc_center_readout(scale_idx):
            disc_mei = deepdraw(adj_model, gen_image, octaves, clip=True,
                                 random_crop=False, blur=blur, jitter=jitter,
                                 precond=precond, step_gain=step_gain,
                                 bias=bias, scale=scale)

            with torch.no_grad():
                center_img = torch.Tensor(process(disc_mei, mu=bias, sigma=scale)[None, ...]).to('cuda')
                activation = adj_model(center_img).data.cpu().numpy()[0]
                disc_mei = disc_mei.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, disc_mei, bias, scale)

        key['disc_mei'] = disc_mei
        key['disc_activation'] = activation
        key['disc_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['disc_max_activation'] = np.max(vals)
        key['disc_max_contrast'] = cont[np.argmax(vals)]
        key['disc_sat_contrast'] = np.max(cont)
        key['disc_mean'] = disc_mei.mean()
        key['disc_lim_contrast'] = lim_contrast


        self.insert1(key)


@schema
class SimpleMEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    disc_mei                 : longblob  # center position based MEI
    disc_activation          : float     # activation of the center unit
    disc_monotonic           : bool      # does activity increase monotonically with contrast
    disc_max_contrast        : float     # contrast at which maximum activity is achieved
    disc_max_activation      : float     # activation at the maximum contrast
    disc_sat_contrast        : float     # contrast at which image would start saturating
    disc_mean                : float     # mean luminance of the image
    disc_lim_contrast        : float     # max reachable contrast without clipping
    """
    #
    # key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
    #              & ReadoutConfig.SpatialTransformerPyramid2d()

    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit &\
                 (NetworkConfig.CorePlusReadout &
                  [ReadoutConfig.SpatialTransformerPyramid2d, ReadoutConfig.ModifiedSpatialTransformerPyramid2d])


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)


        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)


        # generate class visualization via octavewise gradient ascent

        with SpatialTransformerPyramid2d.simple_readout():
            disc_mei = deepdraw(adj_model, gen_image, octaves, clip=True,
                                 random_crop=False, blur=blur, jitter=jitter,
                                 precond=precond, step_gain=step_gain,
                                 bias=bias, scale=scale)

            with torch.no_grad():
                center_img = torch.Tensor(process(disc_mei, mu=bias, sigma=scale)[None, ...]).to('cuda')
                activation = adj_model(center_img).data.cpu().numpy()[0]
                disc_mei = disc_mei.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, disc_mei, bias, scale)

        key['disc_mei'] = disc_mei
        key['disc_activation'] = activation
        key['disc_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['disc_max_activation'] = np.max(vals)
        key['disc_max_contrast'] = cont[np.argmax(vals)]
        key['disc_sat_contrast'] = np.max(cont)
        key['disc_mean'] = disc_mei.mean()
        key['disc_lim_contrast'] = lim_contrast


        self.insert1(key)

@schema
class FixedMEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    disc_mei                 : longblob  # center position based MEI
    disc_activation          : float     # activation of the center unit
    disc_monotonic           : bool      # does activity increase monotonically with contrast
    disc_max_contrast        : float     # contrast at which maximum activity is achieved
    disc_max_activation      : float     # activation at the maximum contrast
    disc_sat_contrast        : float     # contrast at which image would start saturating
    disc_mean                : float     # mean luminance of the image
    disc_lim_contrast        : float     # max reachable contrast without clipping
    """
    #
    # key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
    #              & ReadoutConfig.SpatialTransformerPyramid2d()

    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit &\
                 (NetworkConfig.CorePlusReadout &
                  [ReadoutConfig.SpatialTransformerPyramid2d, ReadoutConfig.ModifiedSpatialTransformerPyramid2d])


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)


        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)


        # generate class visualization via octavewise gradient ascent

        with SpatialTransformerPyramid2d.fixed_readout():
            disc_mei = deepdraw(adj_model, gen_image, octaves, clip=True,
                                 random_crop=False, blur=blur, jitter=jitter,
                                 precond=precond, step_gain=step_gain,
                                 bias=bias, scale=scale)

            with torch.no_grad():
                center_img = torch.Tensor(process(disc_mei, mu=bias, sigma=scale)[None, ...]).to('cuda')
                activation = adj_model(center_img).data.cpu().numpy()[0]
                disc_mei = disc_mei.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, disc_mei, bias, scale)

        key['disc_mei'] = disc_mei
        key['disc_activation'] = activation
        key['disc_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['disc_max_activation'] = np.max(vals)
        key['disc_max_contrast'] = cont[np.argmax(vals)]
        key['disc_sat_contrast'] = np.max(cont)
        key['disc_mean'] = disc_mei.mean()
        key['disc_lim_contrast'] = lim_contrast


        self.insert1(key)

@schema
class DiscCenterMEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    disc_mei                 : longblob  # center position based MEI
    disc_activation          : float     # activation of the center unit
    disc_monotonic           : bool      # does activity increase monotonically with contrast
    disc_max_contrast        : float     # contrast at which maximum activity is achieved
    disc_max_activation      : float     # activation at the maximum contrast
    disc_sat_contrast        : float     # contrast at which image would start saturating
    disc_mean                : float     # mean luminance of the image
    disc_lim_contrast        : float     # max reachable contrast without clipping
    """
    #
    # key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
    #              & ReadoutConfig.SpatialTransformerPyramid2d()

    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit &\
                 (NetworkConfig.CorePlusReadout &
                  [ReadoutConfig.SpatialTransformerPyramid2d, ReadoutConfig.ModifiedSpatialTransformerPyramid2d])


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)


        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)


        # generate class visualization via octavewise gradient ascent

        with SpatialTransformerPyramid2d.disc_center_readout():
            disc_mei = deepdraw(adj_model, gen_image, octaves, clip=True,
                                 random_crop=False, blur=blur, jitter=jitter,
                                 precond=precond, step_gain=step_gain,
                                 bias=bias, scale=scale)

            with torch.no_grad():
                center_img = torch.Tensor(process(disc_mei, mu=bias, sigma=scale)[None, ...]).to('cuda')
                activation = adj_model(center_img).data.cpu().numpy()[0]
                disc_mei = disc_mei.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, disc_mei, bias, scale)

        key['disc_mei'] = disc_mei
        key['disc_activation'] = activation
        key['disc_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['disc_max_activation'] = np.max(vals)
        key['disc_max_contrast'] = cont[np.argmax(vals)]
        key['disc_sat_contrast'] = np.max(cont)
        key['disc_mean'] = disc_mei.mean()
        key['disc_lim_contrast'] = lim_contrast


        self.insert1(key)


@schema
class DiscretizedMEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    disc_mei                 : longblob  # center position based MEI
    disc_activation          : float     # activation of the center unit
    disc_monotonic           : bool      # does activity increase monotonically with contrast
    disc_max_contrast        : float     # contrast at which maximum activity is achieved
    disc_max_activation      : float     # activation at the maximum contrast
    disc_sat_contrast        : float     # contrast at which image would start saturating
    disc_mean                : float     # mean luminance of the image
    disc_lim_contrast        : float     # max reachable contrast without clipping
    """
    #
    # key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
    #              & ReadoutConfig.SpatialTransformerPyramid2d()

    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit &\
                 (NetworkConfig.CorePlusReadout &
                  [ReadoutConfig.SpatialTransformerPyramid2d, ReadoutConfig.ModifiedSpatialTransformerPyramid2d])


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)


        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)


        # generate class visualization via octavewise gradient ascent

        with SpatialTransformerPyramid2d.discrete_readout():
            disc_mei = deepdraw(adj_model, gen_image, octaves, clip=True,
                                 random_crop=False, blur=blur, jitter=jitter,
                                 precond=precond, step_gain=step_gain,
                                 bias=bias, scale=scale)

            with torch.no_grad():
                center_img = torch.Tensor(process(disc_mei, mu=bias, sigma=scale)[None, ...]).to('cuda')
                activation = adj_model(center_img).data.cpu().numpy()[0]
                disc_mei = disc_mei.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, disc_mei, bias, scale)

        key['disc_mei'] = disc_mei
        key['disc_activation'] = activation
        key['disc_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['disc_max_activation'] = np.max(vals)
        key['disc_max_contrast'] = cont[np.argmax(vals)]
        key['disc_sat_contrast'] = np.max(cont)
        key['disc_mean'] = disc_mei.mean()
        key['disc_lim_contrast'] = lim_contrast


        self.insert1(key)


@schema
class CenteredMEI(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    ctr_mei                 : longblob  # center position based MEI
    ctr_activation          : float     # activation of the center unit
    ctr_monotonic           : bool      # does activity increase monotonically with contrast
    ctr_max_contrast        : float     # contrast at which maximum activity is achieved
    ctr_max_activation      : float     # activation at the maximum contrast
    ctr_sat_contrast        : float     # contrast at which image would start saturating
    ctr_mean                : float     # mean luminance of the image
    ctr_lim_contrast        : float     # max reachable contrast without clipping
    """
    #
    # key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
    #              & ReadoutConfig.SpatialTransformerPyramid2d()

    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit &\
                 (NetworkConfig.CorePlusReadout &
                  [ReadoutConfig.SpatialTransformerPyramid2d, ReadoutConfig.ModifiedSpatialTransformerPyramid2d])


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)


        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)


        # generate class visualization via octavewise gradient ascent

        with SpatialTransformerPyramid2d.center_readout():
            ctr_mei = deepdraw(adj_model, gen_image, octaves, clip=True,
                                 random_crop=False, blur=blur, jitter=jitter,
                                 precond=precond, step_gain=step_gain,
                                 bias=bias, scale=scale)

            with torch.no_grad():
                center_img = torch.Tensor(process(ctr_mei, mu=bias, sigma=scale)[None, ...]).to('cuda')
                activation = adj_model(center_img).data.cpu().numpy()[0]
                ctr_mei = ctr_mei.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, ctr_mei, bias, scale)

        key['ctr_mei'] = ctr_mei
        key['ctr_activation'] = activation
        key['ctr_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['ctr_max_activation'] = np.max(vals)
        key['ctr_max_contrast'] = cont[np.argmax(vals)]
        key['ctr_sat_contrast'] = np.max(cont)
        key['ctr_mean'] = ctr_mei.mean()
        key['ctr_lim_contrast'] = lim_contrast


        self.insert1(key)


@schema
class JitterAnalysis(dj.Computed):
    definition = """
    -> CenteredMEI
    -> ImageConfig
    -> JitterConfig
    ---
    jitter_activations: longblob      # activation resulting from jitter
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        mei = (CenteredMEI() & key).fetch1('ctr_mei')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        jitter_size = int((JitterConfig & key).fetch1('jitter_size'))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)


        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')

            with SpatialTransformerPyramid2d.center_readout():
                for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                    jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                    jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                    activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]

        key['jitter_activations'] = activations

        self.insert1(key)


@schema
class GradRF(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    point_rf            : longblob  # single gradient RF
    rf                  : longblob  # most exciting images
    activation          : float     # activation at the MEI
    monotonic           : bool      # does activity increase monotonically with contrast
    max_activation      : float     # activation at the maximum contrast
    max_contrast        : float     # contrast at which maximum activity is archived
    sat_contrast        : float     # contrast at which image would start saturating
    """

    key_source =  TargetModel() * MEIParameter() * TargetDataset.Unit & NetworkConfig.CorePlusReadout

    @staticmethod
    def init_rf_image(stimulus_shape=(1, 36, 64)):
        return torch.zeros(1, *stimulus_shape, device='cuda', requires_grad=True)


    def make(self, key):
        model = (Model() & key).load_network().to('cuda')
        model.train(False)

        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        # --- Compute gradient receptive field
        X = self.init_rf_image(img_shape[1:])
        y = adj_model(X)
        y.backward()
        point_rf = X.grad.data.cpu().numpy().squeeze()
        rf = X.grad.data


        def linear_model(x):
            return (x * rf).sum()

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
         {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(linear_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale)

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')
            activation = adj_model(img).data.cpu().numpy()[0]

        rf = gen_image.squeeze()
        cont, vals, lim_contrast = contrast_tuning(adj_model, rf, bias, scale)
        key['point_rf'] = point_rf
        key['monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['max_activation'] = np.max(vals)
        key['max_contrast'] = cont[np.argmax(vals)]
        key['sat_contrast'] = np.max(cont)
        key['rf'] = rf
        key['activation'] = activation
        self.insert1(key)


@schema
class SimpleGradRF(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    point_rf            : longblob  # single gradient RF
    ctr_rf                  : longblob  # rf achieved by stepping through deep dreaming
    rf_activation          : float     # activation at the MEI
    rf_monotonic           : bool      # does activity increase monotonically with contrast
    rf_max_activation      : float     # activation at the maximum contrast
    rf_max_contrast        : float     # contrast at which maximum activity is archived
    rf_sat_contrast        : float     # contrast at which image would start saturating
    rf_mean                : float     # mean luminance of the image
    rf_lim_contrast        : float     # max reachable contrast without clipping
    """


    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() &  NetworkConfig.CorePlusReadout\
                 & [ReadoutConfig.SpatialTransformerPyramid2d(), ReadoutConfig.ModifiedSpatialTransformerPyramid2d()]

    @staticmethod
    def init_rf_image(stimulus_shape=(1, 36, 64)):
        return torch.zeros(1, *stimulus_shape, requires_grad=True, device='cuda')


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        model = (Model() & key).load_network().to('cuda')
        model.eval()

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        # --- Compute gradient receptive field
        with SpatialTransformerPyramid2d.simple_readout():
            X = self.init_rf_image(img_shape[1:])
            y = adj_model(X)
            y.backward()
            point_rf = X.grad.data.cpu().numpy().squeeze()
            rf = X.grad.data


        def linear_model(x):
            return (x * rf).sum()

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
            {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(linear_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale)

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')

            with SpatialTransformerPyramid2d.simple_readout():
                activation = adj_model(img).data.cpu().numpy()[0]
                rf = gen_image.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, rf, bias, scale)

        key['point_rf'] = point_rf
        key['rf_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['rf_max_activation'] = np.max(vals)
        key['rf_max_contrast'] = cont[np.argmax(vals)]
        key['rf_sat_contrast'] = np.max(cont)
        key['rf_mean'] = rf.mean()
        key['rf_lim_contrast'] = lim_contrast
        key['ctr_rf'] = rf
        key['rf_activation'] = activation
        self.insert1(key)


@schema
class FixedGradRF(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    point_rf            : longblob  # single gradient RF
    ctr_rf                  : longblob  # rf achieved by stepping through deep dreaming
    rf_activation          : float     # activation at the MEI
    rf_monotonic           : bool      # does activity increase monotonically with contrast
    rf_max_activation      : float     # activation at the maximum contrast
    rf_max_contrast        : float     # contrast at which maximum activity is archived
    rf_sat_contrast        : float     # contrast at which image would start saturating
    rf_mean                : float     # mean luminance of the image
    rf_lim_contrast        : float     # max reachable contrast without clipping
    """


    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() &  NetworkConfig.CorePlusReadout\
                 & [ReadoutConfig.SpatialTransformerPyramid2d(), ReadoutConfig.ModifiedSpatialTransformerPyramid2d()]

    @staticmethod
    def init_rf_image(stimulus_shape=(1, 36, 64)):
        return torch.zeros(1, *stimulus_shape, requires_grad=True, device='cuda')


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        model = (Model() & key).load_network().to('cuda')
        model.eval()

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        # --- Compute gradient receptive field
        with SpatialTransformerPyramid2d.fixed_readout():
            X = self.init_rf_image(img_shape[1:])
            y = adj_model(X)
            y.backward()
            point_rf = X.grad.data.cpu().numpy().squeeze()
            rf = X.grad.data


        def linear_model(x):
            return (x * rf).sum()

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
            {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(linear_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale)

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')

            with SpatialTransformerPyramid2d.fixed_readout():
                activation = adj_model(img).data.cpu().numpy()[0]
                rf = gen_image.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, rf, bias, scale)

        key['point_rf'] = point_rf
        key['rf_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['rf_max_activation'] = np.max(vals)
        key['rf_max_contrast'] = cont[np.argmax(vals)]
        key['rf_sat_contrast'] = np.max(cont)
        key['rf_mean'] = rf.mean()
        key['rf_lim_contrast'] = lim_contrast
        key['ctr_rf'] = rf
        key['rf_activation'] = activation
        self.insert1(key)


@schema
class CenteredGradRF(dj.Computed):
    definition = """
    -> TargetModel
    -> MEIParameter
    -> TargetDataset.Unit
    ---
    point_rf            : longblob  # single gradient RF
    ctr_rf                  : longblob  # rf achieved by stepping through deep dreaming
    rf_activation          : float     # activation at the MEI
    rf_monotonic           : bool      # does activity increase monotonically with contrast
    rf_max_activation      : float     # activation at the maximum contrast
    rf_max_contrast        : float     # contrast at which maximum activity is archived
    rf_sat_contrast        : float     # contrast at which image would start saturating
    rf_mean                : float     # mean luminance of the image
    rf_lim_contrast        : float     # max reachable contrast without clipping
    """


    key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
                 & [ReadoutConfig.SpatialTransformerPyramid2d(), ReadoutConfig.ModifiedSpatialTransformerPyramid2d()]

    @staticmethod
    def init_rf_image(stimulus_shape=(1, 36, 64)):
        return torch.zeros(1, *stimulus_shape, requires_grad=True, device='cuda')


    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        model = (Model() & key).load_network().to('cuda')
        model.eval()

        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key)[:, neuron_id]

        # --- Compute gradient receptive field
        with SpatialTransformerPyramid2d.center_readout():
            X = self.init_rf_image(img_shape[1:])
            y = adj_model(X)
            y.backward()
            point_rf = X.grad.data.cpu().numpy().squeeze()
            rf = X.grad.data


        def linear_model(x):
            return (x * rf).sum()

        params = (MEIParameter() & key).fetch1()
        blur = bool(params['blur'])
        jitter = int(params['jitter'])
        precond = float(params['precond'])
        step_gain = float(params['step_gain'])

        octaves = [
            {
                'iter_n': int(params['iter_n']),
                'start_sigma': float(params['start_sigma']),
                'end_sigma': float(params['end_sigma']),
                'start_step_size': float(params['start_step_size']),
                'end_step_size':float(params['end_step_size']),
            },
        ]

        # prepare initial image
        channels, original_h, original_w = img_shape[-3:]

        # the background color of the initial image
        background_color = np.float32([128] * channels)
        # generate initial random image
        gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
        gen_image = np.clip(gen_image, 0, 255)

        # generate class visualization via octavewise gradient ascent
        gen_image = deepdraw(linear_model, gen_image, octaves, clip=True,
                             random_crop=False, blur=blur, jitter=jitter,
                             precond=precond, step_gain=step_gain,
                             bias=bias, scale=scale)

        with torch.no_grad():
            img = torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).to('cuda')

            with SpatialTransformerPyramid2d.center_readout():
                activation = adj_model(img).data.cpu().numpy()[0]
                rf = gen_image.squeeze()
                cont, vals, lim_contrast = contrast_tuning(adj_model, rf, bias, scale)

        key['point_rf'] = point_rf
        key['rf_monotonic'] = bool(np.all(np.diff(vals) >= 0))
        key['rf_max_activation'] = np.max(vals)
        key['rf_max_contrast'] = cont[np.argmax(vals)]
        key['rf_sat_contrast'] = np.max(cont)
        key['rf_mean'] = rf.mean()
        key['rf_lim_contrast'] = lim_contrast
        key['ctr_rf'] = rf
        key['rf_activation'] = activation
        self.insert1(key)


@schema
class RFJitterInPlace(dj.Computed):
    definition = """
    -> GradRF
    -> ImageConfig
    -> JitterConfig
    ---
    rf_jitter_activations: longblob      # activation resulting from jitter
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        rf = (GradRF() & key).fetch1('rf')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        rf, clipped, actual_contrast = adjust_contrast(rf, target_contrast, mu=target_mean, force=force_stats)

        jitter_size = int((JitterConfig & key).fetch1('jitter_size'))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(process(rf[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]

        key['rf_jitter_activations'] = activations

        self.insert1(key)

@schema
class RFJitterAnalysis(dj.Computed):
    definition = """
    -> CenteredGradRF
    -> ImageConfig
    -> JitterConfig
    ---
    rf_jitter_activations: longblob      # activation resulting from jitter
    """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        mei = (CenteredGradRF() & key).fetch1('ctr_rf')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        jitter_size = int((JitterConfig & key).fetch1('jitter_size'))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')

            with SpatialTransformerPyramid2d.center_readout():
                for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                    jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                    jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                    activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]

        key['rf_jitter_activations'] = activations

        self.insert1(key)


@schema
class ImageShifts(dj.Lookup):
    definition = """
    x_shift: int    # shift in the width dimension
    y_shift: int    # shift in the hieght dimension
    """
    contents = product([-1, 0, 1], [-1, 0, 1])

@schema
class ShiftedRF(dj.Computed):
    definition = """
       -> GradRF
       -> ImageConfig
       -> ImageShifts
       ---
       shifted_rf_activation: float      # activation resulting from shift
       shifted_rf: longblob               # copy of the shifted RF
       """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        rf = (GradRF() & key).fetch1('rf')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        rf, clipped, actual_contrast = adjust_contrast(rf, target_contrast, mu=target_mean, force=force_stats)


        # shift the image
        x_shift, y_shift = key['x_shift'], key['y_shift']
        shifted_rf = np.roll(np.roll(rf, x_shift, 1), y_shift, 0)


        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(shifted_rf[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['shifted_rf_activation'] = activations
        key['shifted_rf'] = shifted_rf

        self.insert1(key)

@schema
class ShiftedMEI(dj.Computed):
    definition = """
       -> MEI
       -> ImageConfig
       -> ImageShifts
       ---
       shifted_mei_activation: float      # activation resulting from shift
       shifted_mei: longblob               # copy of the shifted RF
       """

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        mei = (MEI() & key).fetch1('mei')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)


        # shift the image
        x_shift, y_shift = key['x_shift'], key['y_shift']
        shifted_mei = np.roll(np.roll(mei, x_shift, 1), y_shift, 0)


        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(shifted_mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['shifted_mei_activation'] = activations
        key['shifted_mei'] = shifted_mei

        self.insert1(key)


@schema
class CrossShiftedMEIActivation(dj.Computed):
    definition = """
    -> MEI
    -> ImageConfig
    -> ImageShifts
    ---
    cross_shifted_mei_activation: float   # activation of CNN from shifted Lin model MEI
    """
    key_source = MEI * ImageConfig.proj() * ImageShifts & ModelGroup.CNNModel

    def make(self, key):
        # readout_key = key['readout_key']
        # neuron_id = key['neuron_id']
        # print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
        #
        # mei = (MEI() & key).fetch1('mei')

        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        lin_key = (ModelGroup.LinearModel & (ModelGroup & (ModelGroup.CNNModel & key))).fetch1('KEY')

        key_lin = dict(key, **lin_key)

        mei = (MEI() & key_lin).fetch1('mei')

        # adjust the contrast and mean luminance of the image
        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        # shift the image
        x_shift, y_shift = key['x_shift'], key['y_shift']
        shifted_mei = np.roll(np.roll(mei, x_shift, 1), y_shift, 0)

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        # compute the activation on the shifted image
        with torch.no_grad():
            img = torch.Tensor(process(shifted_mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')
            activations = adj_model(img).data.cpu().numpy()[0]

        key['cross_shifted_mei_activation'] = activations

        self.insert1(key)


@schema
class CrossJitterInPlace(dj.Computed):
    definition = """
    -> MEI
    -> ImageConfig
    -> JitterConfig
    ---
    cross_jitter_activations: longblob      # activation resulting from jitter
    """

    key_source = MEI * ImageConfig.proj() * JitterConfig & ModelGroup.CNNModel()

    def make(self, key):
        readout_key = key['readout_key']
        neuron_id = key['neuron_id']
        print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))

        lin_key = (ModelGroup.LinearModel & (ModelGroup & (ModelGroup.CNNModel & key))).fetch1('KEY')

        key_lin = dict(key, **lin_key)

        mei = (MEI() & key_lin).fetch1('mei')

        target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
                                                                                 'force_stats')
        mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)

        jitter_size = int((JitterConfig & key).fetch1('jitter_size'))

        # load the model
        model = (Model() & key).load_network().to('cuda')
        model.eval()

        # get input statistics
        _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)

        def adj_model(x):
            return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]

        shift = list(enumerate(range(-jitter_size, jitter_size+1)))
        activations = np.empty((len(shift), len(shift)))

        with torch.no_grad():
            img = torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).to('cuda')

            for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
                jitter_y, jitter_x = int(jitter_y), int(jitter_x)
                jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
                activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]

        key['cross_jitter_activations'] = activations

        self.insert1(key)


# @schema
# class FeatureMatch(dj.Computed):
#     """
#     Find pairs of Units that are *feature matched* - for each unit, a feature matched unit is unit:
#      1) that is from another scan
#      2) has readout pattern of features that is most similar to the target unit
#     """
#     definition = """
#     -> TargetModel
#     """
#
#     class UnitPair(dj.Part):
#         definition = """
#         -> TargetModel
#         -> TargetDataset.Unit
#         ---
#         (matched_readout_key, matched_neuron_id) -> TargetDataset.Unit(readout_key, neuron_id)
#         score: float       # cosine distance between feature vector
#         """
#
#     def make(self, key):
#         model = (Encoder() & key).load_model()
#         # unbelievably dirty...
#         ro1_f = model.readout['11521-7-1'].features
#         ro2_f = model.readout['11521-7-2'].features
#         assert ro1_f.size(1) == ro2_f.size(1)
#         c = ro1_f.size(1)
#         ro1_f = ro1_f.view(c, -1, 1)
#         ro2_f = ro2_f.view(c, 1, -1)
#         ro1_f = ro1_f / (ro1_f ** 2).sum(0, keepdim=True).sqrt()
#         ro2_f = ro2_f / (ro2_f ** 2).sum(0, keepdim=True).sqrt()
#
#         cos_d = (ro1_f * ro2_f).sum(0)
#         f1_score, f1_match = cos_d.max(1)
#         f2_score, f2_match = cos_d.max(0)
#
#         self.insert1(key)
#
#
#         for i, match, score in zip(count(), f1_match.data.cpu().numpy(), f1_score.data.cpu().numpy()):
#             tuple = dict(key)
#             tuple['readout_key'] = '11521-7-1'
#             tuple['neuron_id'] = i
#             tuple['matched_readout_key'] = '11521-7-2'
#             tuple['matched_neuron_id'] = match
#             tuple['score'] = score
#             self.UnitPair().insert1(tuple)
#
#         for i, match, score in zip(count(), f2_match.data.cpu().numpy(), f2_score.data.cpu().numpy()):
#             tuple = dict(key)
#             tuple['readout_key'] = '11521-7-2'
#             tuple['neuron_id'] = i
#             tuple['matched_readout_key'] = '11521-7-1'
#             tuple['matched_neuron_id'] = match
#             tuple['score'] = score
#             self.UnitPair().insert1(tuple)
#
#
# @schema
# class HighOracleUnits(dj.Computed):
#     definition = """
#     -> TargetDataset.Unit
#     """
#
#     key_source = TargetDataset()
#
#     def make(self, key):
#         target_units = TargetDataset.Unit() & key
#         units = MesoNetAllOracle.UnitScores() & target_units
#
#         corr = units.fetch('pearson')
#         cutoff = np.percentile(corr, 90)
#         r = units & 'pearson > {}'.format(cutoff)
#         self.insert((TargetDataset.Unit() & key & r).proj())
#
#
# @schema
# class ModelGroup(dj.Computed):
#     definition = """
#     -> TargetDataset
#     """
#
#     @property
#     def key_source(self):
#         return TargetDataset & TargetModel
#
#     def _make_tuples(self, key):
#         best_cnn = (Encoder & (TargetModel & key & CoreConfig.Stacked2d & ReadoutConfig.SpatialTransformerPyramid2d)).best.fetch1('KEY')
#         best_lin = (Encoder & (TargetModel & key & CoreConfig.Linear & ReadoutConfig.SpatialTransformerPyramid2d)).best.fetch1('KEY')
#         best_vgg = (Encoder & (TargetModel & key & CoreConfig.VGG & ReadoutConfig.SpatialTransformerPyramid2d)).best.fetch1('KEY')
#
#
#         self.insert1(key)
#         self.CNNModel().insert1(best_cnn)
#         self.LinearModel().insert1(best_lin)
#         self.VGGModel().insert1(best_vgg)
#
#
#     class CNNModel(dj.Part):
#         definition = """
#         -> master
#         -> CoreConfig.Stacked2d
#         -> TargetModel
#         """
#
#     class LinearModel(dj.Part):
#         definition = """
#         -> master
#         -> CoreConfig.Linear
#         -> TargetModel
#         """
#
#     class VGGModel(dj.Part):
#         definition = """
#         -> master
#         -> CoreConfig.VGG
#         -> TargetModel
#         """
#
#
# @schema
# class UnitSelection(dj.Manual):
#     definition = """
#     -> TargetDataset.Unit
#     """
#
#     def fill(self):
#         for k in TargetDataset().fetch('KEY'):
#             units = (TargetDataset.Unit() & k & HighOracleUnits()).aggr(
#                 Encoder.UnitScores() & TargetModel() & CoreConfig.Stacked2d, min_test_corr='min(test_corr)').fetch(
#                 order_by='min_test_corr DESC')
#             self.insert(TargetDataset.Unit() & units[:100], ignore_extra_fields=True)
#
#
# @schema
# class RankedUnit(dj.Manual):
#     definition = """
#     -> TargetDataset.Unit
#     ---
#     rank: int    # ranking
#     """
#     def fill(self):
#         for k in TargetDataset().fetch('KEY'):
#             units = (TargetDataset.Unit() & k & HighOracleUnits()).aggr(
#                 Encoder.UnitScores() & TargetModel() & CoreConfig.Stacked2d, min_test_corr='min(test_corr)').fetch(
#                 order_by='min_test_corr DESC')
#             for i, u in enumerate(units):
#                 key = (TargetDataset.Unit & u).fetch1('KEY')
#                 key['rank'] = i
#                 self.insert1(key)
#
#

#
#
#
# # octaves0 = [
# #  {
# #         'iter_n':600,
# #         'start_sigma':1.5,
# #         'end_sigma':0.01,
# #         'start_step_size': 12.*0.25,
# #         'end_step_size':0.5*0.25,
# #     },
# # ]
#
#
#
#
#

#
# @schema
# class MultiModelMEI(dj.Computed):
#     definition = """
#     -> CoreConfig
#     -> ReadoutConfig
#     -> ShifterConfig
#     -> ModulatorConfig
#     -> MEIParameter
#     -> TargetDataset.Unit
#     ---
#     multi_mei                  : longblob  # most exciting images
#     avg_multi_activation       : float     # activation at the MEI
#     multi_activations          : longblob  # activations across models
#     """
#
#     key_source = CoreConfig() * ReadoutConfig() * ShifterConfig() * ModulatorConfig() * MEIParameter() \
#                  * TargetDataset.Unit() & TargetModel()
#
#     def make(self, key):
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         # load the models
#         model_keys = (Encoder() & key).fetch('KEY')
#         n_models = len(model_keys)
#         assert n_models == 3, 'Seed variations missing'
#
#         models = []
#         with silent():
#             for mk in model_keys:
#                 model = (Encoder() & mk).load_model().cuda()
#                 model.eval()
#                 models.append(model)
#
#         # get input statistics
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#
#         def adj_model(x, aggr=True):
#             activations = [m(x, readout_key, eye_pos=mu_eye, behavior=mu_beh)[:, neuron_id] for m in models]
#             if aggr:
#                 return sum(activations) / n_models
#             else:
#                 return activations
#
#         params = (MEIParameter() & key).fetch1()
#         blur = bool(params['blur'])
#         jitter = int(params['jitter'])
#         precond = float(params['precond'])
#         step_gain = float(params['step_gain'])
#
#         octaves = [
#          {
#                 'iter_n': int(params['iter_n']),
#                 'start_sigma': float(params['start_sigma']),
#                 'end_sigma': float(params['end_sigma']),
#                 'start_step_size': float(params['start_step_size']),
#                 'end_step_size':float(params['end_step_size']),
#             },
#         ]
#
#         # prepare initial image
#         channels, original_h, original_w = img_shape[-3:]
#
#         # the background color of the initial image
#         background_color = np.float32([128] * channels)
#         # generate initial random image
#         gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
#         gen_image = np.clip(gen_image, 0, 255)
#
#
#         # generate class visualization via octavewise gradient ascent
#         gen_image = deepdraw(adj_model, gen_image, octaves, clip=True,
#                              random_crop=False, blur=blur, jitter=jitter,
#                              precond=precond, step_gain=step_gain,
#                              bias=bias, scale=scale)
#
#         mei = gen_image.squeeze()
#
#         img = Variable(torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).cuda(), volatile=True)
#         activation = adj_model(img).data.cpu().numpy()[0]
#         multi_activations = np.array([x.data.cpu().numpy()[0] for x in adj_model(img, aggr=False)])
#
#         key['multi_mei'] = mei
#         key['avg_multi_activation'] = activation
#         key['multi_activations'] = multi_activations
#
#         self.insert1(key)
#
#
# @schema
# class AvgMEI(dj.Computed):
#     definition = """
#     -> TargetDataset.Unit
#     -> MEIParameter
#     ---
#     avg_mei: longblob  # average mei
#     """
#
#     @property
#     def key_source(self):
#         target = TargetDataset.Unit() * MEIParameter()
#         return target.aggr(MEI() & CoreConfig.Stacked2d(), count='count(*)') & 'count = 6'
#
#     def make(self, key):
#         meis = (MEI() & key).fetch('mei')
#         assert len(meis) == 6, 'It appears that not all MEIs were computed!'
#         key['avg_mei'] = np.stack(meis).mean(axis=0)
#         self.insert1(key)
#
#
# @schema
# class AvgMEIActivation(dj.Computed):
#     definition = """
#     -> AvgMEI
#     -> MEI
#     -> ImageConfig
#     ---
#     avg_mei_activation: float   # activaton on average mei
#     avg_mei_clipped: bool       # whether image was clipped
#     avg_mei_contrast: float     # actual contrast of average mei
#     """
#
#     def make(self, key):
#         avg_mei = (AvgMEI() & key).fetch1('avg_mei')
#
#         target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast', 'force_stats')
#         avg_mei, clipped, actual_contrast = adjust_contrast(avg_mei, target_contrast, mu=target_mean, force=force_stats)
#
#
#
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         # load the model
#         with silent():
#             model = (Encoder() & key).load_model().cuda()
#             model.eval()
#
#         # get input statistics
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#
#         def adj_model(x):
#             return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]
#
#         img = Variable(torch.Tensor(process(avg_mei[..., None], mu=bias, sigma=scale)[None, ...]).cuda(), volatile=True)
#         activation = adj_model(img).data.cpu().numpy()[0]
#
#         key['avg_mei_activation'] = activation
#         key['avg_mei_clipped'] = bool(clipped)
#         key['avg_mei_contrast'] = actual_contrast
#
#         self.insert1(key)
#
#
# @schema
# class CenteredMEI(dj.Computed):
#     definition = """
#     -> TargetModel
#     -> MEIParameter
#     -> TargetDataset.Unit
#     ---
#     ctr_mei                 : longblob  # center position based MEI
#     ctr_activation          : float     # activation of the center unit
#     ctr_monotonic           : bool      # does activity increase monotonically with contrast
#     ctr_max_contrast        : float     # contrast at which maximum activity is achieved
#     ctr_max_activation      : float     # activation at the maximum contrast
#     ctr_sat_contrast        : float     # contrast at which image would start saturating
#     ctr_mean                : float     # mean luminance of the image
#     ctr_lim_contrast        : float     # max reachable contrast without clipping
#     """
#
#     key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
#                  & ReadoutConfig.SpatialTransformerPyramid2d()
#
#     def make(self, key):
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         # load the model
#         with silent():
#             model = (Encoder() & key).load_model().cuda()
#             model.eval()
#
#         # get input statistics
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#
#
#         def adj_model(x):
#             return model(x, readout_key)[:, neuron_id]
#
#         params = (MEIParameter() & key).fetch1()
#         blur = bool(params['blur'])
#         jitter = int(params['jitter'])
#         precond = float(params['precond'])
#         step_gain = float(params['step_gain'])
#
#         octaves = [
#          {
#                 'iter_n': int(params['iter_n']),
#                 'start_sigma': float(params['start_sigma']),
#                 'end_sigma': float(params['end_sigma']),
#                 'start_step_size': float(params['start_step_size']),
#                 'end_step_size':float(params['end_step_size']),
#             },
#         ]
#
#         # prepare initial image
#         channels, original_h, original_w = img_shape[-3:]
#
#         # the background color of the initial image
#         background_color = np.float32([128] * channels)
#         # generate initial random image
#         gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
#         gen_image = np.clip(gen_image, 0, 255)
#
#
#         # generate class visualization via octavewise gradient ascent
#
#         with SpatialTransformerPyramid2d.center_readout():
#             ctr_mei = deepdraw(adj_model, gen_image, octaves, clip=True,
#                                  random_crop=False, blur=blur, jitter=jitter,
#                                  precond=precond, step_gain=step_gain,
#                                  bias=bias, scale=scale)
#
#             center_img = Variable(torch.Tensor(process(ctr_mei, mu=bias, sigma=scale)[None, ...]).cuda(),
#                                  volatile=True)
#             activation = adj_model(center_img).data.cpu().numpy()[0]
#
#             ctr_mei = ctr_mei.squeeze()
#             cont, vals, lim_contrast = contrast_tuning(adj_model, ctr_mei, bias, scale)
#
#         key['ctr_mei'] = ctr_mei
#         key['ctr_activation'] = activation
#         key['ctr_monotonic'] = bool(np.all(np.diff(vals) >= 0))
#         key['ctr_max_activation'] = np.max(vals)
#         key['ctr_max_contrast'] = cont[np.argmax(vals)]
#         key['ctr_sat_contrast'] = np.max(cont)
#         key['ctr_mean'] = ctr_mei.mean()
#         key['ctr_lim_contrast'] = lim_contrast
#
#
#         self.insert1(key)
#
#
# @schema
# class JitterConfig(dj.Lookup):
#     definition = """
#     jitter_config_id:   int
#     ---
#     jitter_size:   int
#     """
#     contents = [(0, 5)]
#
#
# @schema
# class JitterAnalysis(dj.Computed):
#     definition = """
#     -> CenteredMEI
#     -> ImageConfig
#     -> JitterConfig
#     ---
#     jitter_activations: longblob      # activation resulting from jitter
#     """
#
#     def make(self, key):
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Jitter analysis: Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         mei = (CenteredMEI() & key).fetch1('ctr_mei')
#
#         target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
#                                                                                  'force_stats')
#         mei, clipped, actual_contrast = adjust_contrast(mei, target_contrast, mu=target_mean, force=force_stats)
#
#         jitter_size = int((JitterConfig & key).fetch1('jitter_size'))
#
#         # load the model
#         with silent():
#             model = (Encoder() & key).load_model().cuda()
#             model.eval()
#
#         # get input statistics
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#
#
#         def adj_model(x):
#             return model(x, readout_key)[:, neuron_id]
#
#         shift = list(enumerate(range(-jitter_size, jitter_size+1)))
#         activations = np.empty((len(shift), len(shift)))
#         img = Variable(torch.Tensor(process(mei[..., None], mu=bias, sigma=scale)[None, ...]).cuda(),
#                        volatile=True)
#
#         with SpatialTransformerPyramid2d.center_readout():
#             for (iy, jitter_y), (ix, jitter_x) in product(shift, shift):
#                 jitter_y, jitter_x = int(jitter_y), int(jitter_x)
#                 jittered_img = roll(roll(img, jitter_y, -2), jitter_x, -1)
#                 activations[iy, ix] = adj_model(jittered_img).data.cpu().numpy()[0]
#
#         key['jitter_activations'] = activations
#
#         self.insert1(key)
#
#
# @schema
# class ConditionedGradRF(dj.Computed):
#     definition = """
#     -> TargetModel
#     -> MEIParameter
#     -> TargetDataset.Unit
#     ---
#     point_rf            : longblob  # single gradient RF
#     rf                  : longblob  # most exciting images
#     activation          : float     # activation at the MEI
#     monotonic           : bool      # does activity increase monotonically with contrast
#     max_activation      : float     # activation at the maximum contrast
#     max_contrast        : float     # contrast at which maximum activity is archived
#     sat_contrast        : float     # contrast at which image would start saturating
#     """
#
#     key_source = TargetModel() * MEIParameter() * TargetDataset.Unit()
#
#     @staticmethod
#     def init_rf_image(stimulus_shape=(1, 36, 64)):
#         return Variable(torch.zeros(1, *stimulus_shape).cuda(), requires_grad=True)
#
#
#     def make(self, key):
#         with silent():
#             model = (Encoder() & key).load_model().cuda()
#         model.train(False)
#
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#
#         def adj_model(x):
#             return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]
#
#         # --- Compute gradient receptive field
#         X = self.init_rf_image(img_shape[1:])
#         y = adj_model(X)
#         y.backward()
#         point_rf = X.grad.data.cpu().numpy().squeeze()
#         rf = Variable(X.grad.data)
#
#
#         def linear_model(x):
#             return (x * rf).sum()
#
#         params = (MEIParameter() & key).fetch1()
#         blur = bool(params['blur'])
#         jitter = int(params['jitter'])
#         precond = float(params['precond'])
#         step_gain = float(params['step_gain'])
#
#         octaves = [
#          {
#                 'iter_n': int(params['iter_n']),
#                 'start_sigma': float(params['start_sigma']),
#                 'end_sigma': float(params['end_sigma']),
#                 'start_step_size': float(params['start_step_size']),
#                 'end_step_size':float(params['end_step_size']),
#             },
#         ]
#
#         # prepare initial image
#         channels, original_h, original_w = img_shape[-3:]
#
#         # the background color of the initial image
#         background_color = np.float32([128] * channels)
#         # generate initial random image
#         gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
#         gen_image = np.clip(gen_image, 0, 255)
#
#         # generate class visualization via octavewise gradient ascent
#         gen_image = deepdraw(linear_model, gen_image, octaves, clip=True,
#                              random_crop=False, blur=blur, jitter=jitter,
#                              precond=precond, step_gain=step_gain,
#                              bias=bias, scale=scale)
#
#         img = Variable(torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).cuda(), volatile=True)
#         activation = adj_model(img).data.cpu().numpy()[0]
#
#         rf = gen_image.squeeze()
#         cont, vals, lim_contrast = contrast_tuning(adj_model, rf, bias, scale)
#         key['point_rf'] = point_rf
#         key['monotonic'] = bool(np.all(np.diff(vals) >= 0))
#         key['max_activation'] = np.max(vals)
#         key['max_contrast'] = cont[np.argmax(vals)]
#         key['sat_contrast'] = np.max(cont)
#         key['rf'] = rf
#         key['activation'] = activation
#         self.insert1(key)
#
#
# @schema
# class CenteredGradRF(dj.Computed):
#     definition = """
#     -> TargetModel
#     -> MEIParameter
#     -> TargetDataset.Unit
#     ---
#     point_rf            : longblob  # single gradient RF
#     ctr_rf                  : longblob  # rf achieved by stepping through deep dreaming
#     rf_activation          : float     # activation at the MEI
#     rf_monotonic           : bool      # does activity increase monotonically with contrast
#     rf_max_activation      : float     # activation at the maximum contrast
#     rf_max_contrast        : float     # contrast at which maximum activity is archived
#     rf_sat_contrast        : float     # contrast at which image would start saturating
#     rf_mean                : float     # mean luminance of the image
#     rf_lim_contrast        : float     # max reachable contrast without clipping
#     """
#
#
#     key_source = TargetModel() * MEIParameter() * TargetDataset.Unit() \
#                  & ReadoutConfig.SpatialTransformerPyramid2d()
#
#     @staticmethod
#     def init_rf_image(stimulus_shape=(1, 36, 64)):
#         return Variable(torch.zeros(1, *stimulus_shape).cuda(), requires_grad=True)
#
#
#     def make(self, key):
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         with silent():
#             model = (Encoder() & key).load_model().cuda()
#             model.eval()
#
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#
#         def adj_model(x):
#             return model(x, readout_key)[:, neuron_id]
#
#         # --- Compute gradient receptive field
#         with SpatialTransformerPyramid2d.center_readout():
#             X = self.init_rf_image(img_shape[1:])
#             y = adj_model(X)
#             y.backward()
#             point_rf = X.grad.data.cpu().numpy().squeeze()
#             rf = Variable(X.grad.data)
#
#
#         def linear_model(x):
#             return (x * rf).sum()
#
#         params = (MEIParameter() & key).fetch1()
#         blur = bool(params['blur'])
#         jitter = int(params['jitter'])
#         precond = float(params['precond'])
#         step_gain = float(params['step_gain'])
#
#         octaves = [
#             {
#                 'iter_n': int(params['iter_n']),
#                 'start_sigma': float(params['start_sigma']),
#                 'end_sigma': float(params['end_sigma']),
#                 'start_step_size': float(params['start_step_size']),
#                 'end_step_size':float(params['end_step_size']),
#             },
#         ]
#
#         # prepare initial image
#         channels, original_h, original_w = img_shape[-3:]
#
#         # the background color of the initial image
#         background_color = np.float32([128] * channels)
#         # generate initial random image
#         gen_image = np.random.normal(background_color, 8, (original_h, original_w, channels))
#         gen_image = np.clip(gen_image, 0, 255)
#
#         # generate class visualization via octavewise gradient ascent
#         gen_image = deepdraw(linear_model, gen_image, octaves, clip=True,
#                              random_crop=False, blur=blur, jitter=jitter,
#                              precond=precond, step_gain=step_gain,
#                              bias=bias, scale=scale)
#
#         img = Variable(torch.Tensor(process(gen_image, mu=bias, sigma=scale)[None, ...]).cuda(), volatile=True)
#
#         with SpatialTransformerPyramid2d.center_readout():
#             activation = adj_model(img).data.cpu().numpy()[0]
#             rf = gen_image.squeeze()
#             cont, vals, lim_contrast = contrast_tuning(adj_model, rf, bias, scale)
#
#         key['point_rf'] = point_rf
#         key['rf_monotonic'] = bool(np.all(np.diff(vals) >= 0))
#         key['rf_max_activation'] = np.max(vals)
#         key['rf_max_contrast'] = cont[np.argmax(vals)]
#         key['rf_sat_contrast'] = np.max(cont)
#         key['rf_mean'] = rf.mean()
#         key['rf_lim_contrast'] = lim_contrast
#         key['ctr_rf'] = rf
#         key['rf_activation'] = activation
#         self.insert1(key)
#
#


#
# @schema
# class MEIConfusion(dj.Computed):
#     definition = """
#     -> TargetDataset.Unit
#     -> MEIParameter
#     -> ImageConfig
#     (src_model_hash) -> TargetModelAlias
#     (dest_model_hash) -> TargetModelAlias
#     ---
#     activation: float   # activation
#     contrast_clipped: bool # whether image was clipped during contrast adjustment
#     actual_contrast: float # actual contrast of the image used
#     """
#
#     @property
#     def key_source(self):
#         rel = TargetDataset.Unit() * MEIParameter() * ImageConfig() * TargetModelAlias() & MEI().proj()
#         return rel.proj(src_model_hash='model_hash') * rel.proj(dest_model_hash='model_hash')
#
#     def make(self, key):
#         src_key = dict(key)
#         src_key['model_hash'] = src_key['src_model_hash']
#         for e in ('src_model_hash', 'dest_model_hash'):
#             src_key.pop(e, None)
#
#         src_mei = (MEI() * TargetModelAlias() & src_key).fetch1('mei')
#
#         target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast', 'force_stats')
#         src_mei, clipped, actual_contrast = adjust_contrast(src_mei, target_contrast, mu=target_mean, force=force_stats)
#
#
#
#
#         dest_key = dict(key)
#         dest_key['model_hash'] = dest_key['dest_model_hash']
#         for e in ('src_model_hash', 'dest_model_hash'):
#             dest_key.pop(e, None)
#
#         dest_model_key = (TargetModel() * TargetModelAlias() & dest_key).fetch1('KEY')
#
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         # load the model
#         with silent():
#             model = (Encoder() & dest_model_key).load_model().cuda()
#             model.eval()
#
#         # get input statistics
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(dest_model_key, readout_key)
#
#         def adj_model(x):
#             return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]
#
#         img = Variable(torch.Tensor(process(src_mei[..., None], mu=bias, sigma=scale)[None, ...]).cuda(), volatile=True)
#         activation = adj_model(img).data.cpu().numpy()[0]
#
#         key['activation'] = activation
#         key['contrast_clipped'] = bool(clipped)
#         key['actual_contrast'] = actual_contrast
#
#         self.insert1(key)
#
#
# @schema
# class RFConfusion(dj.Computed):
#     definition = """
#     -> TargetDataset.Unit
#     -> MEIParameter
#     -> ImageConfig
#     (src_model_hash) -> TargetModelAlias
#     (dest_model_hash) -> TargetModelAlias
#     ---
#     activation: float   # activation
#     contrast_clipped: bool # whether image was clipped during contrast adjustment
#     actual_contrast: float # actual contrast of the image used
#     """
#
#     @property
#     def key_source(self):
#         rel = TargetDataset.Unit() * MEIParameter() * ImageConfig() * TargetModelAlias() & ConditionedGradRF()
#         return rel.proj(src_model_hash='model_hash') * rel.proj(dest_model_hash='model_hash')
#
#     def make(self, key):
#         src_key = dict(key)
#         src_key['model_hash'] = src_key['src_model_hash']
#         for e in ('src_model_hash', 'dest_model_hash'):
#             src_key.pop(e, None)
#
#         src_rf = (ConditionedGradRF() * TargetModelAlias() & src_key).fetch1('rf')
#
#         target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast', 'force_stats')
#
#         # adjust the contrast of the receptive field
#         src_rf, clipped, actual_contrast = adjust_contrast(src_rf, target_contrast, mu=target_mean, force=force_stats)
#
#
#         dest_key = dict(key)
#         dest_key['model_hash'] = dest_key['dest_model_hash']
#         for e in ('src_model_hash', 'dest_model_hash'):
#             dest_key.pop(e, None)
#
#         dest_model_key = (TargetModel() * TargetModelAlias() & dest_key).fetch1('KEY')
#
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         # load the model
#         with silent():
#             model = (Encoder() & dest_model_key).load_model().cuda()
#             model.eval()
#
#         # get input statistics
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(dest_model_key, readout_key)
#
#         def adj_model(x):
#             return model(x, readout_key, eye_pos=mu_eye)[:, neuron_id]
#
#         img = Variable(torch.Tensor(process(src_rf[..., None], mu=bias, sigma=scale)[None, ...]).cuda(), volatile=True)
#         activation = adj_model(img).data.cpu().numpy()[0]
#
#         key['activation'] = activation
#         key['contrast_clipped'] = bool(clipped)
#         key['actual_contrast'] = actual_contrast
#
#         self.insert1(key)
#
#
# @schema
# class UnitConfusion(dj.Computed):
#     definition = """
#     -> MEIParameter
#     -> ImageConfig
#     -> TargetModel
#     -> TargetDataset.Unit
#     (src_readout_key, src_neuron_id) -> CenteredMEI(readout_key, neuron_id)
#     ---
#     activation: float # activation on destination neuron
#     contrast_clipped: bool # whether image was clipped during contrast adjustment
#     actual_contrast: float # actual contrast of the image used
#     actual_mean: float     # actual mean of the image used
#     """
#
#     @property
#     def key_source(self):
#         targets = CenteredMEI & ModelGroup.CNNModel#& (RankedUnit & 'rank < 200')
#         src = targets.proj(src_readout_key='readout_key', src_neuron_id='neuron_id')
#         #dest = targets.proj(dest_readout_key='readout_key', dest_neuron_id='neuron_id')
#         return src * ImageConfig
#
#     def make(self, key):
#         src_key = dict(key)
#         src_key['readout_key'] = src_key['src_readout_key']
#         src_key['neuron_id'] = src_key['src_neuron_id']
#
#         exclusion = ['src_readout_key', 'src_neuron_id', 'dest_readout_key', 'dest_neuron_id']
#         for e in exclusion:
#             src_key.pop(e, None)
#
#         src_mei = (CenteredMEI() & src_key).fetch1('ctr_mei')
#
#         target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast', 'force_stats')
#         src_mei, clipped, actual_contrast = adjust_contrast(src_mei, target_contrast, mu=target_mean, force=force_stats)
#         actual_mean = src_mei.mean()
#
#         key['actual_contrast'] = actual_contrast
#         key['actual_mean'] = actual_mean
#         key['contrast_clipped'] = bool(clipped)
#
#
#         # load the model
#         with silent():
#             model = (Encoder() & key).load_model().cuda()
#             model.eval()
#
#         # get input statistics
#         readout_keys = list(model.readout.keys())
#
#         def adj_model(x, rokey):
#             return model(x, rokey)
#
#         with SpatialTransformerPyramid2d.center_readout():
#             for readout_key in readout_keys:
#                 _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#                 img = Variable(torch.Tensor(process(src_mei[..., None], mu=bias, sigma=scale)[None, ...]).cuda(), volatile=True)
#                 activations = adj_model(img, readout_key).data.cpu().numpy()[0]
#
#                 tuple = dict(key)
#                 tuple['readout_key'] = readout_key
#                 for i, activation in tqdm(enumerate(activations)):
#                     tuple['neuron_id'] = i
#                     tuple['activation'] = activation
#                     self.insert1(tuple)
#
#
# @schema
# class FeatureMatchedMEIActivation(dj.Computed):
#     definition = """
#     -> FeatureMatch.UnitPair
#     -> MEIParameter
#     -> ImageConfig
#     ---
#     own_mei_activation: float    # activation of the unit by its own MEI
#     own_rf_activation: float     # activation of the unit by its own RF
#     own_lin_activation: float    # activation of the unit by linear model fit
#     matched_mei_activation: float   # activation of the unit by the matched unit's MEI
#     matched_rf_activation: float    # activation of the unit by the matched unit's RF
#     matched_lin_activation: float   # activation of the unit by the linear model fit to the matched unit
#     """
#
#     @property
#     def key_source(self):
#         rel = ImageConfig() * FeatureMatch.UnitPair() * MEIParameter() & ModelGroup.CNNModel & CenteredMEI() & CenteredMEI().proj(matched_readout_key='readout_key', matched_neuron_id='neuron_id')
#
#         lin_mei = TargetDataset.Unit & (CenteredMEI & ModelGroup.LinearModel)
#         lin_mei_matched = lin_mei.proj(matched_readout_key='readout_key', matched_neuron_id='neuron_id')
#
#         return rel & lin_mei & lin_mei_matched
#
#
#
#     def make(self, key):
#         # get image config
#         target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast', 'force_stats')
#
#         with silent():
#             model = (Encoder() & key).load_model().cuda()
#             model.eval()
#
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#
#         # get input statistics
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#
#         def adj_model(x):
#             return model(x, readout_key)[:, neuron_id]
#
#
#
#         # get own MEI and RF
#         own_mei = (CenteredMEI() & key).fetch1('ctr_mei')
#         own_mei, *_ = adjust_contrast(own_mei, target_contrast, mu=target_mean, force=force_stats)
#
#
#         gen_target = (TargetDataset.Unit * MEIParameter) & key
#         own_linear = (CenteredMEI() & gen_target & ModelGroup.LinearModel).fetch1('ctr_mei')
#         own_linear, *_ = adjust_contrast(own_linear, target_contrast, mu=target_mean, force=force_stats)
#
#         # get matched MEI and RF
#         tuple = dict(key)
#         tuple['readout_key'], tuple['neuron_id'] = (FeatureMatch.UnitPair & key).fetch1('matched_readout_key', 'matched_neuron_id')
#
#         matched_mei = (CenteredMEI() & tuple).fetch1('ctr_mei')
#         matched_mei, *_ = adjust_contrast(matched_mei, target_contrast, mu=target_mean, force=force_stats)
#
#         matched_rf = (CenteredGradRF() & tuple).fetch1('ctr_rf')
#         matched_rf, *_ = adjust_contrast(matched_rf, target_contrast, mu=target_mean, force=force_stats)
#
#         gen_target = (TargetDataset.Unit * MEIParameter) & tuple
#         matched_linear = (CenteredMEI() & gen_target & ModelGroup.LinearModel).fetch1('ctr_mei')
#         matched_linear, *_ = adjust_contrast(matched_linear, target_contrast, mu=target_mean, force=force_stats)
#
#         with SpatialTransformerPyramid2d.center_readout():
#             own_mei_img = Variable(torch.Tensor(process(own_mei[..., None], mu=bias, sigma=scale)[None, ...]).cuda(),
#                                   volatile=True)
#             own_mei_activation = adj_model(own_mei_img).data.cpu().numpy()[0]
#
#
#             own_lin_img = Variable(torch.Tensor(process(own_linear[..., None], mu=bias, sigma=scale)[None, ...]).cuda(),
#                                    volatile=True)
#             own_lin_activation = adj_model(own_lin_img).data.cpu().numpy()[0]
#
#
#             matched_mei_img = Variable(torch.Tensor(process(matched_mei[..., None], mu=bias, sigma=scale)[None, ...]).cuda(),
#                                    volatile=True)
#             matched_mei_activation = adj_model(matched_mei_img).data.cpu().numpy()[0]
#
#             matched_rf_img = Variable(torch.Tensor(process(matched_rf[..., None], mu=bias, sigma=scale)[None, ...]).cuda(),
#                                   volatile=True)
#             matched_rf_activation = adj_model(matched_rf_img).data.cpu().numpy()[0]
#
#             matched_lin_img = Variable(torch.Tensor(process(matched_linear[..., None], mu=bias, sigma=scale)[None, ...]).cuda(),
#                                    volatile=True)
#             matched_lin_activation = adj_model(matched_lin_img).data.cpu().numpy()[0]
#
#
#         key['own_mei_activation'] = own_mei_activation
#         key['own_lin_activation'] = own_lin_activation
#         key['matched_mei_activation'] = matched_mei_activation
#         key['matched_lin_activation'] = matched_lin_activation
#
#         self.insert1(key)
#
#
# @schema
# class CNNCrossAreaUnitConfusion(dj.Computed):
#     definition = """
#     -> MEIParameter
#     -> ImageConfig
#     -> TargetDataset.Unit
#     (src_data_hash, src_readout_key, src_neuron_id) -> TargetDataset.Unit(data_hash, readout_key, neuron_id)
#     ---
#     activation: float    #  activation on the neuron
#     """
#
#     @property
#     def key_source(self):
#         field_maps = dict(src_data_hash='data_hash', src_readout_key='readout_key', src_neuron_id='neuron_id')
#         rel = TargetDataset.Unit & (CenteredMEI & ModelGroup.CNNModel) & HighUnitSelection()
#         return ImageConfig * MEIParameter * rel * rel.proj(**field_maps)
#
#     def make(self, key):
#         src_key = dict(key)
#         src_key['data_hash'] = src_key['src_data_hash']
#         src_key['readout_key'] = src_key['src_readout_key']
#         src_key['neuron_id'] = src_key['src_neuron_id']
#
#         src_mei = (CenteredMEI() & ModelGroup.CNNModel & src_key).fetch1('ctr_mei')
#
#         target_mean, target_contrast, force_stats = (ImageConfig() & key).fetch1('img_mean', 'img_contrast',
#                                                                                  'force_stats')
#
#         src_mei, clipped, actual_contrast = adjust_contrast(src_mei, target_contrast, mu=target_mean, force=force_stats)
#
#         actual_mean = src_mei.mean()
#
#
#
#         readout_key = key['readout_key']
#         neuron_id = key['neuron_id']
#         print('Generating Unit confusion entry')
#         print('Working on neuron_id={}, readout_key={}'.format(neuron_id, readout_key))
#
#         # load the model
#         with silent():
#             model = (Encoder() & ModelGroup.CNNModel & key).load_model().cuda()
#             model.eval()
#
#         # get input statistics
#         _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
#
#         def adj_model(x):
#             return model(x, readout_key)[:, neuron_id]
#
#         with SpatialTransformerPyramid2d.center_readout():
#             img = Variable(torch.Tensor(process(src_mei[..., None], mu=bias, sigma=scale)[None, ...]).cuda(),
#                            volatile=True)
#             activation = adj_model(img).data.cpu().numpy()[0]
#
#         key['activation'] = activation
#
#         self.insert1(key)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#





















