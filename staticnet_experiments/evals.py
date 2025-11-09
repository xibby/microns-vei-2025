import datajoint as dj
from staticnet_analyses import base
from staticnet_experiments import models as static_models, configs as static_configs, utils
from neuro_data.static_images.configs import DataConfig
from neuro_data.static_images import stats
import numpy as np
from itertools import count

schema = dj.schema('zhiwei_evals')


@schema
class SeedSet(dj.Lookup):
    definition = """
    ssid                    : int  # seed set id
    ---
    seeds                   : blob
    """
    content = [[1, [1009, 1215, 2606, 99999]]
    ]

def get_multi_model(key):
    # load the model
    seeds = (SeedSet & key).fetch1('seeds')
    model_keys = (static_models.Model() & key & [dict(seed=seed) for seed in seeds]).fetch('KEY')

    models = []
    for mk in model_keys:
        model = (static_models.Model() & mk).load_network().to('cuda')
        model.eval()
        models.append(model)
    return models

def multi_model_wrapper(models):
    def compute(x, readout_key, eye_pos=None, behavior=None):
        resp = 0
        for m in models:
            resp = resp + m(x, readout_key, eye_pos=eye_pos, behavior=behavior)
        return resp / len(models)
    return compute

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

def compute_predictions(loader, model, readout_key, eye_pos=None, behavior=None):
    y, y_hat = [], []
    for x_val, beh_val, eye_val, y_val in loader:
        if eye_pos is not None:
            eye_val = eye_pos
        if behavior is not None:
            beh_val = behavior
        y_mod = model(x_val, readout_key, eye_pos=eye_val, behavior=beh_val).data.cpu().numpy()
        y.append(y_val.cpu().numpy())
        y_hat.append(y_mod)
    return np.vstack(y), np.vstack(y_hat)

@schema
class FrozenModelEval(dj.Computed):
    """
    "Freeze" the behavior input of the model when evaluating. It makes use of the mean eye position and behavior state
    everywhere. This gives a fair comparison to Oracle.
    """
    definition = """
    -> base.Dataset
    -> static_configs.NetworkConfig
    -> SeedSet
    ---
    n_models: int         # number of models that are combined
    test_score: float    # average testset score
    """

    @property
    def key_source(self):
        cnn_models = (static_configs.NetworkConfig.CorePlusReadout & static_configs.CoreConfig.GaussianLaplace).proj()
#         cnn_models = (static_configs.NetworkConfig.CorePlusReadout & static_configs.CoreConfig.StackedRF).proj()

        return base.Dataset * static_configs.NetworkConfig * SeedSet & cnn_models

    class Unit(dj.Part):
        definition = """
        -> master
        -> base.Dataset.Unit
        ---
        unit_score: float   # correlation score
        oracle_score: float # oracle correlation
        fraction_oracle: float  # fraction oracle
        """

    def make(self, key):
        testsets, testloaders = DataConfig().load_data(key, tier='test', cuda=True, batch_size=30)

        models = get_multi_model(key)
        comb_model = multi_model_wrapper(models)

        scores, unit_scores = [], []
        oracle_units = stats.Oracle.UnitScores * base.Dataset.Unit & key
        for readout_key, testloader in testloaders.items():
            _, img_shape, bias, mu_beh, mu_eye, scale = prepare_data(key, readout_key)
            # override behavioral information with average behavioral values
            y, y_hat = compute_predictions(testloader, comb_model, readout_key, eye_pos=mu_eye, behavior=mu_beh)
            perf_scores = utils.corr(y, y_hat, axis=0)

            scores.append(perf_scores)
            unit_scores.extend(
                [dict(key, readout_key=readout_key, neuron_id=u, unit_score=c) for u, c in zip(count(), perf_scores)])

        for unit_key in unit_scores:
            unit_key['oracle_score'] = (oracle_units & unit_key).fetch1('pearson')
            unit_key['fraction_oracle'] = unit_key['unit_score'] / unit_key['oracle_score']

        key['n_models'] = len(models)
        key['test_score'] = np.concatenate(scores).mean()

        self.insert1(key)
        self.Unit.insert(unit_scores)
