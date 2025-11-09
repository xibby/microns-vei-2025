import hashlib
import random

import torch
import numpy as np
from collections import OrderedDict, namedtuple
import git
import inspect
import datajoint as dj
import os
import datetime
import warnings


def gitlog(cls):
    """
    Decorator that equips a DataJoint class with an additional dj.Part table that stores the current sha1 git hash,
    the branch, the date of the head commit,and whether the code was modified since the last commit,
    for the class representing the master table. Use the instantiated version of the decorator.
    Here is an example:
    .. code-block:: python
       :linenos:
        import datajoint as dj
        from djaddon import gitlog
        schema = dj.schema('mydb',locals())
        @schema
        @gitlog
        class MyRelation(dj.Computed):
            definition = ...
    """
    class GitKey(dj.Part):
        definition = """
        ->master
        ---
        sha1        : varchar(40)
        branch      : varchar(50)
        modified    : int   # whether there are modified files or not
        head_date   : datetime # authored date of git head
        """

    def log_key(self, key):
        key = dict(key) # copy key
        path = inspect.getabsfile(cls).split('/')
        for i in reversed(range(len(path))):
            if os.path.exists('/'.join(path[:i]) + '/.git'):
                repo = git.Repo('/'.join(path[:i]))
                break
        else:
            raise KeyError("%s.GitKey could not find a .git directory for %s" % (cls.__name__, cls.__name__))
        sha1, branch = repo.head.commit.name_rev.split()
        modified = (repo.git.status().find("modified") > 0) * 1
        if modified:
            warnings.warn('You have uncommited changes. Consider committing the changes before running populate.')
        key['sha1'] = sha1
        key['branch'] = branch
        key['modified'] = modified
        key['head_date'] = datetime.datetime.fromtimestamp(repo.head.commit.authored_date)
        self.GitKey().insert1(key, skip_duplicates=True, ignore_extra_fields=True)
        return key


    cls.GitKey = GitKey
    cls.log_git = log_key

    return cls


def set_seed(seed, cuda=True):
    """
    Sets the random generator seed for both NumPy and PyTorch. If cuda=True, will set the seed
    for GPU device under PyTorch as well.
    Args:
        seed: seed value
        cuda: if True, set GPU seed as well
    """
    print('Setting numpy and torch seed to', seed, flush=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(int(seed))
    if cuda:
        torch.cuda.manual_seed(int(seed))


def list_hash(values):
    """
    Returns MD5 digest hash values for a list of values
    """
    hashed = hashlib.md5()
    for v in values:
        hashed.update(str(v).encode())
    return hashed.hexdigest()


def key_hash(key):
    """
    32-byte hash used for lookup of primary keys of jobs
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()


def corr(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two matrices along certain dimensions.

    Args:
        y1:      first matrix
        y2:      second matrix
        axis:    dimension along which the correlation is computed.
        eps:     offset to the standard deviation to make sure the correlation is well defined (default 1e-8)
        **kwargs passed to final `mean` of standardized y1 * y2

    Returns: correlation vector

    """
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=1) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=1) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)


def ptcorr(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two matrices along certain dimensions.

    Args:
        y1:      first matrix
        y2:      second matrix
        axis:    dimension along which the correlation is computed.
        eps:     offset to the standard deviation to make sure the correlation is well defined (default 1e-8)
        **kwargs passed to final mean of standardized y1 * y2

    Returns: correlation vector

    """
    y1 = (y1 - y1.mean(dim=axis, keepdim=True)) / (y1.std(dim=axis, keepdim=True) + eps)
    y2 = (y2 - y2.mean(dim=axis, keepdim=True)) / (y2.std(dim=axis, keepdim=True) + eps)
    return (y1 * y2).mean(dim=axis, **kwargs)


def to_native(key):
    """
    Given a mapping (e.g. dictionary) or a iterable (e.g. list) with 0d NumPy array or Torch tensor values,
    convert these into native Python data type object (e.g. ints and floats).

    Args:
        key: a mapping or an iterable object

    Returns:
        Same object as `key` but with 0d NumPy array or Torch tensor value replaced with native scalar value
    """
    if isinstance(key, (dict, OrderedDict)):
        for k, v in key.items():
            if hasattr(v, 'dtype'):
                try:   
                    if len(v) < 2: 
                        key[k] = v.item() 
                except TypeError:   # when there's only one value inf v and the value is None, TypeError would be raised
                        key[k]= v.item()
    else:
        for k, v in enumerate(key):
            if hasattr(v, 'dtype') and len(v) < 2:
                key[k] = v.item()
    return key


def compute_predictions(loader, model, readout_key, eye_pos=None, behavior=None):
    y, y_hat = [], []
    for data_point in loader:
        if eye_pos is not None:
            eye_val = eye_pos
        elif hasattr(data_point, 'pupil_center'):
            eye_val = data_point.pupil_center
        else:
            eye_val = None

        if behavior is not None:
            beh_val = behavior
        elif hasattr(data_point, 'behavior'):
            beh_val = data_point.behavior
        else:
            beh_val = None

        y_mod = model(data_point.images, readout_key, eye_pos=eye_val, behavior=beh_val).data.cpu().numpy()
        y.append(data_point.responses.cpu().numpy())
        y_hat.append(y_mod)
    return np.vstack(y), np.vstack(y_hat)


    # if len(loader.dataset.data_keys) == 2: # when dataset only contrains images and responses
    #     for x_val, y_val in loader:
    #         y_mod = model(x_val, readout_key, eye_pos=eye_pos, behavior=behavior).data.cpu().numpy()
    #         y.append(y_val.cpu().numpy())
    #         y_hat.append(y_mod)
    # elif len(loader.dataset.data_keys) == 4:  # when dataset also contains behavior and pupil_center
    #     for x_val, beh_val, eye_val, y_val in loader: 
    #         # when use mean_behavior and mean_eyepos to override behavior and eyepos in loader
    #         if eye_pos is not None:
    #             eye_val = eye_pos
    #         if behavior is not None:
    #             beh_val = behavior
    #         y_mod = model(x_val, readout_key, eye_pos=eye_val, behavior=beh_val).data.cpu().numpy()
    #         y.append(y_val.cpu().numpy())
    #         y_hat.append(y_mod)

    # return np.vstack(y), np.vstack(y_hat)

def loss_closure(mod, objective, loaders, avg=True, print_val=True):
    ret = []
    train = mod.training
    mod.eval()
    for readout_key, loader in loaders.items():
        y, y_hat = compute_predictions(loader, mod, readout_key)
        y = torch.as_tensor(y, device='cuda')
        y_hat = torch.as_tensor(y_hat, device='cuda')
        obj = objective(mod, readout_key, y_hat, y)
        if print_val:
            print(readout_key + 'loss: {:.4f}'.format(obj.item()))
        ret.append(obj.item())
    ret = np.hstack(ret)
    mod.train(train)

    if avg:
        return ret.mean()
    else:
        return ret

def correlation_closure(mod, loaders, avg=True, print_val=True):
    ret = []
    train = mod.training
    mod.eval()
    for readout_key, loader in loaders.items():
        y, y_hat = compute_predictions(loader, mod, readout_key)
        co = corr(y, y_hat, axis=0)
        if print_val:
            print(readout_key + 'correlation: {:.4f}'.format(co.mean()))
        ret.append(co)
    ret = np.hstack(ret)
    mod.train(train)

    if avg:
        return ret.mean()
    else:
        return ret
    
def correlation(mod, loaders, avg=True):
    ret = []
#     train = mod.training
#     mod.eval()
    for readout_key, loader in loaders.items():
        y, y_hat = compute_predictions(loader, mod, readout_key)
        co = corr(y, y_hat, axis=0)
        print(readout_key + 'correlation: {:.4f}'.format(co.mean()))
        ret.append(co)
    ret = np.hstack(ret)
#     mod.train(train)

    if avg:
        return ret.mean()
    else:
        return ret

PerformanceScores = namedtuple('PerformanceScores', ['pearson'])

def compute_scores(y, y_hat, axis=0):
    pearson = corr(y, y_hat, axis=axis)
    return PerformanceScores(pearson=pearson)