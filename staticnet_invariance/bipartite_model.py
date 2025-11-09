import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import datajoint as dj
from tqdm import tqdm

from attorch.losses import PoissonLoss, MSE
from attorch.layers import elu1

from neuro_data.static_images.configs import DataConfig
from staticnet_experiments.utils import corr, set_seed
from staticnet_experiments.configs import TrainConfig
from staticnet_analyses import base
from featurevis import ops
from staticnet_invariance import deis_reconfigured as deis_schema

schema = dj.schema('neurostatic_bipartite_model')

dj.config.setdefault('stores', dict())
dj.config['stores'].update({
    'static': dict(
        protocol='file', 
        location='/dj-stor01/neuro-static')
})

dj.config["enable_python_native_blobs"] = True

class LinearTextureModel(nn.Module):
    def __init__(self, texture, n_crops, seed, f, image_shape=(36, 64), device='cuda'):        
        super(LinearTextureModel, self).__init__()
        self.device = device
        self.image_shape = image_shape
        self.texture = torch.as_tensor(texture[None, None], dtype=torch.float32, device=self.device).contiguous()
        self.n_crops = n_crops
        self.v_crops = ops.RandomCrop(self.image_shape[0], self.image_shape[1], self.n_crops)(self.texture)
        self.seed = seed
        self.f = torch.as_tensor(f[None, None], dtype=torch.float32, device=self.device).contiguous()
        self.mlp = nn.Sequential(nn.Linear(2, 100), nn.Linear(100, 1))
        # self.mlp = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
        self.v_scale = nn.Parameter(torch.Tensor([100.0]))
        self.f_scale = nn.Parameter(torch.Tensor([100.0]))
        
    def feature_l1(self, average=True):
        if average:
            return self.weights.abs().mean()
        else:
            return self.weights.abs().sum()
        
    def forward(self, x):
        y_v = F.relu(F.conv2d(x, self.v_crops)).mean(1).squeeze() / self.v_scale
        y_f = F.relu(F.conv2d(x, self.f)).squeeze() / self.f_scale
        y = torch.cat([y_f[:, None], y_v[:, None]], dim=-1)
        y = self.mlp(y).squeeze()
        y = elu1(y)  #F.relu(y)
        return y
        
def objective(model, inputs, targets):
    criterion = PoissonLoss()
    # inputs in batch_size * 1 * h * w
    # targets in batch_size
    outputs = model(inputs)
    return criterion(outputs, targets)

def get_correlation(model, loader, neuron_id):
    y, y_hat = [], []
    for inputs, _, _, targets in loader:
        outputs = model(inputs)
        y.append(targets[:, neuron_id].cpu().numpy())
        y_hat.append(outputs.data.cpu().numpy())
    y = np.hstack(y)
    y_hat = np.hstack(y_hat)
    return corr(y, y_hat, axis=0)


@schema
class ModelParameters(dj.Lookup):
    definition = """
    model_params: int
    ---
    param_dict: longblob  # a dictionary of model parameters
    """
    contents = [[1, dict(n_crops=20, seed=1234)]]

@schema
class TrainingParameters(dj.Lookup):
    definition = """
    train_params: int
    ---
    param_dict: longblob  # a dictionary of model training parameters
    """
    contents = [[1, dict(optim_name="SGD", lr=0.001, max_epoch=100)]]

@schema
class Seed(dj.Lookup):
    definition = """
    # random seed for training
    train_seed                 :  int # random seed
    ---
    """

    @property
    def contents(self):
        yield from zip([1009, 1215, 2606, 99999])

@schema
class BipartiteModel(dj.Computed):
    definition = """
    -> deis_schema.TextureLookup
    -> ModelParameters
    -> TrainingParameters
    -> Seed
    ---
    history:    blob@static
    model:      blob@static
    val_corr:   float
    test_corr:  float
    """
    
    @property
    def key_source(self):
        return ModelParameters.proj() * TrainingParameters.proj() * Seed * \
               deis_schema.TextureLookup & (deis_schema.Texture * deis_schema.TextureGoodRun & {'score_params': 3})
    
    def make(self, key):
        # --- set seed
        seed = (Seed() & key).fetch1('train_seed')
        set_seed(seed)

        model_params = (ModelParameters & key).fetch1('param_dict')
        train_params = (TrainingParameters & key).fetch1('param_dict')
        neuron_key = (deis_schema.TextureLookup & key).fetch1()
        texture, f, deis = (deis_schema.Texture & neuron_key).fetch1('eval_texture', 'fixed_part', 'samples')
        train_key = (TrainConfig * TrainConfig.Default & {'batch_size': 60}).fetch1()
        
        trainsets, trainloaders = DataConfig().load_data(neuron_key, tier='train', cuda=True, **train_key)
        _, valloaders = DataConfig().load_data(neuron_key, tier='validation', cuda=True, **train_key, key_order=trainsets)
        _, testloaders = DataConfig().load_data(neuron_key, tier='test', cuda=True, **train_key, key_order=trainsets)
        
        model = LinearTextureModel(texture, model_params['n_crops'], model_params['seed'], f).cuda()
        model, history, val_corr, test_corr = self.train_model(model, neuron_key, train_params, trainloaders, valloaders, testloaders)
        model={k: v.cpu().numpy() for k, v in model.state_dict().items()}
        self.insert1({**key, 'history': history, 'model': model, 'val_corr': val_corr, 'test_corr': test_corr})

    @staticmethod
    def train_model(model, key, train_params, trainloaders, valloaders, testloaders):
        if train_params['optim_name'] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=train_params['lr'])
        elif train_params['optim_name'] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=train_params['lr'])

        optimizer.zero_grad()

        train_loss, val_loss, train_corr, val_corr = [], [], [], []
        for epoch in tqdm(range(train_params['max_epoch'])):
            model.train()
            loss = []
            for inputs, _, _, targets in trainloaders[key['readout_key']]:
                obj = objective(model, inputs, targets[:, key['neuron_id']]) #+ model.feature_l1(True)
                loss.append(obj.item())
                obj.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_loss.append(np.stack(loss).mean())

            model.eval()
            loss = []
            for inputs, _, _, targets in valloaders[key['readout_key']]:
                obj = objective(model, inputs, targets[:, key['neuron_id']])
                loss.append(obj.item())
            val_loss.append(np.stack(loss).mean())

            tcorr = get_correlation(model, trainloaders[key['readout_key']], key['neuron_id'])
            vcorr = get_correlation(model, valloaders[key['readout_key']], key['neuron_id'])
            train_corr.append(tcorr)
            val_corr.append(vcorr)

            print('train correlation = {:.4f} validation correlation = {:.4f}'.format(tcorr, vcorr))

        model.eval()
        test_corr = get_correlation(model, testloaders[key['readout_key']], key['neuron_id'])
        
        history = dict(train_loss=train_loss, val_loss=val_loss, train_corr=train_corr, val_corr=val_corr)
        
        return model, history, val_corr[-1], test_corr
    
    