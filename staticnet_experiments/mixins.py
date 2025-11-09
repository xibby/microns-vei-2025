from functools import partial

import datajoint as dj
import numpy as np

from neuro_data.static_images.configs import DataConfig
from neuro_data.static_images.data_schemas import StaticMultiDataset
# from neuro_data.static_images import zd_neurodata

from . import logger as log
from .utils import correlation_closure, compute_predictions, compute_scores

active_learning = dj.create_virtual_module('neurostatic_active_learning', 'neurostatic_active_learning')

class TrainMixin:
    def train(self, key):
        from .configs import TrainConfig
        log.info('Training ' + repr(key))
        # --- load data
        train_key = TrainConfig().train_key(key)        
        trainsets, trainloaders = DataConfig().load_data(key, tier='train', cuda=True, **train_key)

        if len(trainsets) == 0:
            log.warning("Empty dataset. Adding this key into BadConfig")
            from .models import BadConfig, BadConfigException
            BadConfig.insert1(dict(key, reason='Empty dataset'), ignore_extra_fields=True)
            raise BadConfigException
        
        valsets, valloaders = DataConfig().load_data(key, tier='validation', cuda=True, **train_key,
                                                     key_order=trainsets)

        testsets, testloaders = DataConfig().load_data(key, tier='test', cuda=True, **train_key,
                                                       key_order=trainsets)

        for k, ts in trainsets.items():
            log.info('Trainingset {}\n{}'.format(k, repr(ts)))

        model = self.build_network(key, trainsets=trainsets)

        model = TrainConfig().train(key, model=model, trainloaders=trainloaders,
                                    valloaders=valloaders)
#         model, lr_shift_epoch, lr_shift_batch, all_train_loss, all_train_corr, all_val_loss, all_val_corr = TrainConfig().train(key, model=model, trainloaders=trainloaders,
#                                     valloaders=valloaders)

        # --- test
        stop_closure = partial(correlation_closure, loaders=valloaders)
        updated_key = dict(key,
                           val_corr=np.nanmean(stop_closure(model, avg=False)),
                           model={k: v.cpu().numpy() for k, v in model.state_dict().items()})
        # updated_key = dict(key,
        #                    val_corr=np.nanmean(stop_closure(model, avg=False)),
        #                    model={k: v.cpu().numpy() for k, v in model.state_dict().items()},
        #                    lr_shift_epochs=lr_shift_epoch,
        #                    lr_shift_batchs=lr_shift_batch,
        #                    all_train_losses=all_train_loss,
        #                    all_train_corrs=all_train_corr,
        #                    all_val_losses=all_val_loss, 
        #                    all_val_corrs=all_val_corr)

        num_test = len(list(testloaders.items())[0][-1])
        if num_test == 0:
            log.warning("Empty test dataset.")
            scores, unit_scores = [], []
        else:
            scores, unit_scores = self.compute_test_score_tuples(key, testloaders, model)
        return updated_key, scores, unit_scores


class TestMixin:
    def compute_test_score_tuples(self, key, testloaders, model):
        scores, unit_scores = [], []
        for readout_key, testloader in testloaders.items():
            log.info('Computing test scores for ' + readout_key)

            y, y_hat = compute_predictions(testloader, model, readout_key)
            perf_scores = compute_scores(y, y_hat)

            if 'multi_session' in key:
                member_key = (active_learning.MultiSessionDataset & key & dict(name=readout_key)).fetch1(dj.key)
                unit_ids = testloader.dataset.neurons.neuron_ids
            # elif 'subset_id' in key:
            #     member_key = (zd_neurodata.StaticMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
            #     unit_ids = testloader.dataset.neurons.unit_ids
            # elif len(zd_neurodata.StaticMultiDataset2 & key) > 0:
            #     member_key = (zd_neurodata.StaticMultiDataset2.Member() & key & dict(name=readout_key)).fetch1(dj.key)
            #     unit_ids = testloader.dataset.neurons.unit_ids
            else:
                member_key = (StaticMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
                unit_ids = testloader.dataset.neurons.unit_ids

            member_key.update(key)

            member_key['neurons'] = len(unit_ids)
            member_key['pearson'] = perf_scores.pearson.mean()

            scores.append(member_key)
            unit_scores.extend(
                [dict(member_key, unit_id=u, pearson=c) for u, c in zip(unit_ids, perf_scores.pearson)])
        return scores, unit_scores