import datajoint as dj
from neuro_data.static_images.data_schemas import StaticMultiDataset
from staticnet_experiments import models as static_models, zd_models
from staticnet_analyses import base

schema = dj.schema('zhiwei_neurostatic_base')

@schema
class TestNeurons(dj.Lookup):
    definition = """
    -> StaticMultiDataset
    neuron_subset_id:         int
    ---
    neuron_ids:               longblob # list of neuron ids 
    """

@schema
class BenchmarkActivation(dj.Computed):
    definition = """ # benchmark predicted activation to the same images from same models trained in different environments or machines
    -> TestNeurons
    -> zd_models.TempModel
    -> base.MEIParameters
    -> static_models.Model.proj(mei_net_hash='net_hash', mei_model_seed='seed')
    ---
    responses_matrix:         longblob # matrix of test neuron responses to test neuron meis in shape of num_images * num_neurons
    """

    # @property
    # def key_source(self):
    #     mei_model_rel = (static_models.Model * base.MEIParameters & base.MEI).proj(mei_net_hash='net_hash', mei_model_seed='seed')
    #     return mei_model_rel * zd_models.TempModel * TestNeurons

    # def make(self, key):
    #     neuron_ids = (TestNeurons & key).fetch1('neuron_ids')
    #     mei_keys, meis = (base.MEI & key & [{'neuron_id': n} for n in neuron_ids]).fetch('KEY', 'mei', order_by='neuron_id')
    #     meis = torch.as_tensor(meis[:, None], dtype=torch.float32, device='cuda')
        
    #     # Get params
    #     mei_params = (base.MEIParameters & key).fetch1()

    #     # Get models
    #     all_keys = (static_models.Model & key).fetch('KEY')
    #     all_models = [(static_models.Model & mk).load_network() for mk in all_keys]

    #     # Get some train stats
    #     mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & mei_keys[0]).fetch1('norm_eyepos') else
    #                    (base.Dataset.TrainStats & mei_keys[0]).fetch1('mean_eyepos'))

    #     # Create model ensemble
    #     mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32,
    #                                device='cuda').unsqueeze(0)
    #     #import pdb; pdb.set_trace()
    #     model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
    #                             neuron_idx=neuron_ids, device='cuda', average_batch=False)
    #     resp = model(meis).detach().cpu().squeeze().numpy()

    #     self.insert1({**key, 'response_matrix': resp})