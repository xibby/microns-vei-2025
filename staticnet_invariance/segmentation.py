# baseline pytorch framework
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

# plotting and data libraries
import numpy as np
import datajoint as dj
import os
import pandas as pd
import copy
import math
from PIL import Image
import skimage.io as io
from itertools import chain
from scipy.stats import pearsonr
import seaborn as sns
from tqdm import tqdm
import random
import time
import pickle
from typing import Any, Callable, Optional, Tuple, List

# internal schemas
from utils.datajoint.datajoint_utils import files
from staticnet import logger
from staticnet_analyses import base
from staticnet_experiments import models as static_models, configs
import staticnet_invariance.deis_reconfigured as deis_schema
from featurevis import ops, models, utils
from staticnet_experiments import utils as static_utils
from scipy import ndimage

schema = dj.schema('neurostatic_segmentation')
dj.config.setdefault('stores', dict())
dj.config['stores'].update({
    'static': dict(
        protocol='file', 
        location='/dj-stor01/neuro-static')
})

dj.config["enable_python_native_blobs"] = True

# install perlin-noise
def install(package):
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("perlin-noise")

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, img_path, label_path, root = '', train=True, img_transform=None, mask_transform = None,loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.loader = default_loader
        self.train = train
        self.img_path = img_path
        self.label_path = label_path
        if download:
            self._download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, self.img_path, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        self.data = images
#         image_class_labels = pd.read_csv(os.path.join(self.root, self.img_path, 'image_class_labels.txt'),
#                                          sep=' ', names=['img_id', 'target'])
#         train_test_split = pd.read_csv(os.path.join(self.root, self.img_path, 'train_test_split.txt'),
#                                        sep=' ', names=['img_id', 'is_training_img'])

#         data = images.merge(image_class_labels, on='img_id')
        #self.data = data.merge(train_test_split, on='img_id')

        # if self.train:
        #     self.data = self.data[self.data.is_training_img == 1]
        # else:
        #     self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.img_path,'images', row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.img_path,'images', sample.filepath)

        # modify the path the get the mask
        pre, ext = os.path.splitext(sample.filepath)
        mask_file_path = pre + '.png'
        label_path = os.path.join(self.root,self.label_path,mask_file_path)

        #target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        mask = self.loader(label_path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            mask[mask>0.5] = 1.0
            mask[mask<1.0] = -1.0
        return img, mask

class PascalDataset(Dataset):
    """Encapsulates the Pascal VOC segmentation dataset
    Read more about PASCAL here: http://host.robots.ox.ac.uk/pascal/VOC/
    """
    def __init__(
        self,
        root: str,
        img_set: str,
        input_transform: Optional[Callable] = lambda x: x,
        target_transform: Optional[Callable] = lambda x: x
    ) -> None:
        
        self.root = root
        list_path = os.path.join(self.root, 'ImageSets','Segmentation','{}.txt'.format(img_set))
        with open(list_path,'r') as list_file:
            proto_list = [line.strip() for line in list_file]
            self.img_list = ['{}.jpg'.format(filename) for filename in proto_list]
            self.mask_list = ['{}.png'.format(filename) for filename in proto_list]
        self.input_transform = input_transform
        self.target_transform = target_transform
                                                                   
    def _load_image(self, img_id: int) -> Image.Image:
        """Loads image into memory
        Parameters
        ----------
        img_id : int
        
        Returns
        -------
        PIL image in RGB format 
        """

        path = os.path.join(self.root, 'JPEGImages', self.img_list[img_id]) 
        return Image.open(path).convert("RGB")

    def _load_target(self, img_id: int) -> Image.Image:
        """Loads mask into memory
        Parameters
        ----------
        img_id : int
        
        Returns
        -------
        PIL image
        """

        path = os.path.join(self.root, 'SegmentationClass', self.mask_list[img_id]) 
        return Image.open(path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        target = self._load_target(index)
        
        image = self.input_transform(image)
        target = self.target_transform(target)
        return image, target

    def __len__(self) -> int:
        return len(self.img_list)
    
class BinaryPascalDataset(PascalDataset):
    def __init__(
        self,
        root: str,
        img_set: str,
        input_transform: Optional[Callable] = lambda x: x,
        target_transform: Optional[Callable] = lambda x: x
    ) -> None:
        
        super().__init__(root, img_set, input_transform, target_transform)   

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        target = self._load_target(index)
        
        image = self.input_transform(image)
        target = self.target_transform(target) 
        binary_target = torch.bitwise_and(target > 0, target != 255).type(target.dtype)
        
        binary_target[binary_target>0.0] = 1.0
        binary_target[binary_target<1.0] = -1.0
        return image, binary_target

    def __len__(self) -> int:
        return len(self.img_list)

class PerlinDatasetLoader():
    def __init__(self,perlin_params,batch_size):
        self.my_table = PerlinImage() & {'perlin_params':perlin_params}
        self.batch_size = batch_size
    def __iter__(self):
        self.img_id = 0
        return self
    def __next__(self):
        # Don't handle exception to stop iteration
        imgs, labels = (self.my_table & 'img_id>={:} and img_id<{:}'.format(self.img_id,self.img_id+self.batch_size)).fetch('img','label')
        if len(imgs) == 0:
            return None, None
        else:
            self.img_id += self.batch_size
            return np.stack(imgs), np.stack(labels)
        
 
def create_grating(sf, ori, phase, wave, imsize):
    import scipy.signal as signal
    """
    :param sf: spatial frequency (in pixels)
    :param ori: wave orientation (in degrees, [0-360])
    :param phase: wave phase (in degrees, [0-360])
    :param wave: type of wave ('sqr' or 'sin')
    :param imsize: image size (integer)
    :return: numpy array of shape (imsize, imsize)
    """
    # Get x and y coordinates
    x, y = np.meshgrid(np.arange(imsize), np.arange(imsize))

    # Get the appropriate gradient
    gradient = np.sin(ori * math.pi / 180) * x - np.cos(ori * math.pi / 180) * y

    # Plug gradient into wave function
    if wave == 'sin':
        grating = np.sin((2 * math.pi * gradient) / sf + (phase * math.pi) / 180)
    elif wave =='sqr':
        grating = signal.square((2 * math.pi * gradient) / sf + (phase * math.pi) / 180)
    else:
        raise NotImplementedError

    return grating

def create_cub_based_dataset(image_mode='monet_same_freq',new_img_path = '/external/datt/CUB_modified',original_img_path='/external/datt/CUB_200_2011',label_path = '/external/datt/segmentations',height=64,width=64,make_new=False,seed=0):
    import shutil
    new_img_path = os.path.join(new_img_path,image_mode)
    transform = transforms.Compose([transforms.Resize((height,width)),
                                    transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
    all_data = Cub2011(img_path = original_img_path,label_path = label_path,img_transform = transform,mask_transform=transform,train=True,download=False)
    if make_new and os.path.exists(new_img_path):
        # This requires no file is read-only
        shutil.rmtree(new_img_path)

    if not os.path.exists(new_img_path):
        os.makedirs(os.path.join(new_img_path,'images'))
        
        shutil.copyfile(os.path.join(original_img_path, 'images.txt'),os.path.join(new_img_path, 'images.txt'))
        with open(os.path.join(new_img_path, 'images.txt')) as file:
            lines = file.readlines()
            lines = [line.rstrip().replace('jpg','png') for line in lines]
        with open(os.path.join(new_img_path, 'images.txt'), 'w') as f:
            for line in lines:
                f.write(f"{line}\n")
        # Make all subfolders in images
        for i in os.listdir(os.path.join(original_img_path,'images')):
            os.makedirs(os.path.join(new_img_path,'images',i))
        if 'monet' in image_mode:
            if 'same_freq' in image_mode:
                boundaries = [[0,5],[0,5]]
            elif 'same_low_freq' in image_mode:
                persistences = [[3,5],[3,5]]
            elif 'high_freq_inside' in image_mode:
                boundaries = [[0,2],[3,5]]
            elif 'high_freq_outside' in image_mode:
                boundaries = [[3,5],[0,2]]
            else:
                raise ValueError()
        elif 'perlin' in image_mode:
            if 'same_freq' in image_mode:
                persistences = [1.0,1.0]
            elif 'same_low_freq' in image_mode:
                persistences = [0.1,0.1]
            elif 'high_freq_inside' in image_mode:
                persistences = [1.0,0.1]
            elif 'high_freq_outside' in image_mode:
                persistences = [0.1,1.0]
            else:
                raise ValueError()
        elif 'grating' in image_mode:
            if 'same_freq' in image_mode:
                freqs = [3,8]
            elif 'same_low_freq' in image_mode:
                freqs = [8,30]
            elif 'high_freq_inside' in image_mode:
                freqs = [[3,8],[8,30]]
            elif 'high_freq_outside' in image_mode:
                freqs = [[8,30],[3,8]]
            else:
                raise ValueError()
        else:
            raise ValueError() 
            
        r = np.random.default_rng(seed)
        
        for i in tqdm(range(len(all_data.data))):
            img,mask = all_data.__getitem__(i)
            img = img.numpy().squeeze()
            mask = mask.numpy().squeeze()
            mask = ((mask+1)/2)
            if image_mode.endswith('smooth_boundary'):
                mask = ndimage.gaussian_filter(mask.astype(np.float32),sigma=1.5)
            elif image_mode.endswith('no_boundary'):
                mask = np.ones_like(mask)
            if 'monet' in image_mode:
                in_text, out_text = [make_monet_texture(height = img.shape[-2],width=img.shape[-1], ori=r.uniform(low=0,high=np.pi),
                                 sigma=r.uniform(low=i,high=j),mean_intensity=r.uniform(low=0.2,high=0.8),
                                 ori_coherence=r.uniform(low=2,high=12),seed=r.integers(0,1000000000)) for (i,j) in boundaries]
    
            elif 'perlin' in image_mode:
                random.seed(r.integers(0,1000000000))
                # This is highly sensitive to change in size since shape need to be multiple of 2^(octaves-1)*res
                assert ((height % (2**(4-1)*8))==0) and ((width % (2**(4-1)*8))==0), 'Violate Perlin noise condition'
                in_text, out_text = [generate_fractal_noise_2d(shape=(height, width), res=(8, 8), octaves =4,persistence=i) for i in persistences]
            elif 'grating' in image_mode:
                # For grating only atm, if same_freq or same_low_freq then both will have same sf
                # option same_orientation is only for grating atm
                if 'same_orientation' in image_mode:
                    ori = r.uniform(low=0,high=360)
                    ori = (ori,ori)
                else:
                    ori = (r.uniform(low=0,high=360),r.uniform(low=0,high=360))
                if 'same_freq' in image_mode or 'same_low_freq' in image_mode:
                    sf = r.uniform(low=freqs[0],high=freqs[1])
                    sf = (sf,sf)
                else:
                    sf = (r.uniform(low=freqs[0][0],high=freqs[0][1]),r.uniform(low=freqs[1][0],high=freqs[1][1]))
                in_text = create_grating(sf=sf[0], ori=ori[0],phase=r.uniform(low=0,high=360), wave='sin', imsize=img.shape[0])
                out_text = create_grating(sf=sf[1], ori=ori[1],phase=r.uniform(low=0,high=360), wave='sin', imsize=img.shape[0])
            else:
                raise ValueError()
            
            if 'same_contrast' in image_mode:
                in_text = in_text /(in_text.std()+1e-9)
                out_text = out_text/(out_text.std()+1e-9)
                
            if 'same_mean' in image_mode:
                in_text = in_text - in_text.mean()
                out_text = out_text - out_text.mean()          
            
            img = in_text * mask + out_text * (1-mask)

            img = np.clip(img,a_min=np.percentile(img.ravel(),5),a_max = np.percentile(img.ravel(),95))
            img = (img - img.min())/(img.max()-img.min()+1e-9)
            img = Image.fromarray(np.uint8(img*255),'L')
            file_name = all_data.data['filepath'][i].replace('jpg','png')
            img.save(os.path.join(new_img_path,'images',file_name))            
def get_dataloader(search_params):
    if search_params['dataset_name'].startswith('cub'):
        if search_params['dataset_name'] == 'cub':
            img_path = '/external/datt/CUB_200_2011'
            img_transform = transforms.Compose([transforms.Resize((search_params['transform_height'],search_params['transform_width'])),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor()])
        else:
            image_mode = search_params['dataset_name'].replace('cub_','')
            img_path = os.path.join('/external/datt/CUB_modified',image_mode)
            # If it doesn't exist then make the images
            if not os.path.exists(img_path):
                create_cub_based_dataset(image_mode=image_mode,new_img_path = '/external/datt/CUB_modified',
                                         original_img_path='/external/datt/CUB_200_2011',height=search_params['transform_height'],
                                         width=search_params['transform_width'],make_new=True,seed=0)
            img_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
            
        mask_transform = transforms.Compose([transforms.Resize((search_params['transform_height'],search_params['transform_width'])),
                                             transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
        label_path = '/external/datt/segmentations'
        all_data = Cub2011(img_path = img_path,label_path = label_path,img_transform = img_transform,mask_transform=mask_transform,train=False,download=False)
        all_dl = DataLoader(all_data, batch_size=search_params['batch_size'], drop_last=False)
    
    # Todo modify pascal to accomodate modification if needed
    elif search_params['dataset_name'].startswith('pascal'):
        transform = transforms.Compose([transforms.Resize((search_params['transform_height'],search_params['transform_width'])),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor()])
        
        img_path = '/external/datt/segmentation_data/VOC2012'
        all_data = BinaryPascalDataset(img_path, 'train', transform, transform)
        all_dl = DataLoader(all_data, batch_size=search_params['batch_size'], drop_last=False)
    
    # Perlin is kept now to avoid conflict with previous code
    elif search_params['dataset_name'].startswith('perlin'):
        perlin_params = int(search_params['dataset_name'].split('_')[1])
        all_dl = PerlinDatasetLoader(perlin_params,batch_size=search_params['batch_size'])
    return all_dl


        
# Crop the image for sharing across neurons
def crop_img(images,masks,search_params):
    from itertools import product
    crop_h, crop_w, crop_stride = [search_params[i] for i in ['crop_height', 'crop_width', 'crop_stride']]
    IM_SIZE = images.shape[-2:]
    results = []
    for img,mask in zip(images,masks):
        temp = {'original_image': img if img.dtype == 'float32' else img.detach().cpu().numpy().squeeze(),
                'original_label': mask if mask.dtype == 'float32' else mask.detach().cpu().numpy().squeeze(),'x':[],'y':[],'crop':[],'mask':[]}
        for (h, w) in product(np.arange(0, IM_SIZE[0] - crop_h, crop_stride), np.arange(0, IM_SIZE[1] - crop_w, crop_stride)):
            temp['x'].append(h)
            temp['y'].append(w)
            temp['crop'].append(img[...,h:h+crop_h, w:w+crop_w].squeeze())
            temp['mask'].append(mask[...,h:h+crop_h, w:w+crop_w].squeeze())
        for i in ['x','y','crop', 'mask']:
            temp[i] = np.stack(temp[i])
        results.append(temp)
    return results

# Standardize based per neuron
def standardize_crops(crop_results,search_params,key):
    mask_float, mask_x, mask_y = (base.MEIMask & key).fetch1('mask', 'mask_x', 'mask_y')
    mei_params = (base.MEIParameters & key).fetch1()
    image_results = []
    for result in crop_results:
        images = ops.create_whole_mei(result['crop'], mask_float, mask_x, mask_y, 
                                                                (len(result['crop']), 36, 64), 
                                                                normalize_crop = False)
        if search_params['match_stats'] == 'ff':
            target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
        else:
            raise ValueError()
        images = ops.standardize_image(images, target_mean, target_std, mask_float, 
                                       search_params['mask_mean_subtraction'],search_params['mask_image'], 
                                       search_params['match_stats'])
        image_results.append(images)
    return image_results

def pass_img(imgs,model,device,batch_size=32):
    import math
    n_batch = int(math.ceil(len(imgs)/batch_size))
    y = []
    with torch.no_grad():
        for x in np.array_split(imgs,n_batch):
            x = torch.tensor(x,dtype=torch.float32,device=device).unsqueeze(1)
            y.append(model(x).cpu().numpy().squeeze())
    return np.concatenate(y)

# To increase amount of crops, sacrifice speed by processing batches of images at a time instead of sharing


# foreground is the parameter to get the proper persistence
def perlin_noise_helper(perlin_params,noise_fs = None,foreground=True):
    from perlin_noise import PerlinNoise

    if noise_fs is None:
        noise_fs = [PerlinNoise(octaves=i) for i in perlin_params['octaves']]
    xpix, ypix = perlin_params['transform_height'],perlin_params['transform_width']
    img = []
    if foreground:
        persistence = perlin_params['foreground_persistence']
    else:
        persistence = perlin_params['background_persistence']
    for i in range(xpix):
        row = []
        for j in range(ypix):
            noise_val = 0.0
            for k,noise_f in enumerate(noise_fs):
                noise_val += (persistence**k) * noise_f([i/xpix, j/ypix])
            row.append(noise_val)
        img.append(row)
    img = np.array(img,dtype=np.float32)
    if perlin_params['standardize']:
        img = (img - img.mean())/img.std()
    return img

@schema
class PerlinNoiseParameters(dj.Lookup):
    definition = """
    perlin_params: int
    ---
    seed: int                     # Seed
    foreground_persistence: float # Persistence of the foreground
    background_persistence: float # Persistence of the background
    octaves: longblob             # List of noise components with their octaves
    transform_height: int         # Height of the resized image
    transform_width: int          # Width of the resized image
    batch_size: int               # Batch size for processing
    sigma: float                  # Sigma for blurring the edges
    standardize: bool             # Whether to match contrast and mean between foreground and background despite having different frequency
    dataset_name: varchar(16)     # Dataset specified to load ('cub' or 'pascal')
    """
    contents = [(1,1,1.0,0.25,[3,6,12,24],64,64,32,1.5,1,'cub'),(2,2,1.0,0.5,[3,6,12,24],64,64,32,1.5,1,'cub'),
                (3,3,1.0,0.75,[3,6,12,24],64,64,32,1.5,1,'cub'),(4,4,1.0,1.0,[3,6,12,24],64,64,32,1.5,1,'cub'),
                (5,5,0.25,1.0,[3,6,12,24],64,64,32,1.5,1,'cub'),(6,6,0.5,1.0,[3,6,12,24],64,64,32,1.5,1,'cub'),
                (7,7,0.75,1.0,[3,6,12,24],64,64,32,1.5,1,'cub'),(8,8,0.25,1.0,[3,6,12,24],64,64,32,1.5,0,'cub'),
                (9,9,1.0,0.25,[3,6,12,24],64,64,32,1.5,0,'cub')]
@schema
class PerlinImage(dj.Computed):
    definition = """
    -> PerlinNoiseParameters
    img_id: int              # Id of the crop
    ---
    original_img: longblob   # Original full image    
    label:        longblob   # Original label of the image
    img:          longblob   # Perlin noise image
    """
    @property
    def key_source(self):
        return PerlinNoiseParameters

    def make(self,key):
        perlin_params = (PerlinNoiseParameters() & {'perlin_params':key['perlin_params']}).fetch1()
        
        # Get dataloader
        all_dl = get_dataloader(perlin_params)
        iterator = iter(all_dl)
        random.seed(perlin_params['seed'])
        
        img_id = 0
        for imgs,masks in iterator:
            if imgs is None:
                break
            else:
                for img,mask in zip(imgs,masks):
                    img = img.numpy().squeeze()
                    mask = mask.numpy().squeeze()
                    temp = np.array(mask)
                    temp[temp<1.0] = 0.0
                    temp[temp>0.0] = 1.0
                    temp = ndimage.gaussian_filter(temp.astype(np.float32),sigma=perlin_params['sigma'])

                    new_img = perlin_noise_helper(perlin_params,foreground=True) * temp +\
                    (1.0-temp) * perlin_noise_helper(perlin_params,foreground=False)
                    result = {**key,'img_id':img_id,'original_img':img,'label':mask,'img':new_img}
                    self.insert1(result)
                    img_id +=1

@schema
class CropSearchParameters(dj.Lookup):
    definition = """
    search_params: int
    ---
    transform_height: int  # Height of the resized image
    transform_width: int   # Width of the resized image
    crop_height: int
    crop_width: int
    crop_stride: int       # Stride to generate crop
    batch_size: int        # Batch size of images to process
    n_target_images: int   # Number of crops to search
    n_keep : int           # Number of crops to keep
    mask_image: bool       # Whether to mask the image
    match_stats: varchar(16) # Method for matching statistics on the image ('mask' or 'ff')
    mask_mean_subtraction: bool # Subtract mask mean
    dataset_name: varchar(16)   # Dataset specified to load ('cub' or 'pascal')
    """
    contents = [(1,64,64,36,36,2,128,1000000,100,1,'ff',1,'cub'),(2,64,64,36,36,2,128,1000000,100,1,'ff',1,'pascal'),
                (3,128,128,36,36,4,128,1000000,100,1,'ff',1,'cub'),(4,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_1'),
                (5,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_2'),(6,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_3'),
                (7,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_4'),(8,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_5'),
                (9,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_6'),(10,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_7'),
               (11,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_8'),(12,64,64,36,36,2,128,1000000,100,1,'ff',1,'perlin_9')]

@schema
class HighlyActivatingCrop(dj.Computed):
    definition = """ # For multiple neurons
    -> base.MEIMask
    -> CropSearchParameters
    crop_id: int               # Id of the crop
    ---
    original_image: longblob   # Original full image    
    original_label: longblob    # Original label of the image
    x: int                     # Height coordinate of the crop
    y: int                     # Width coordinate of the crop
    crop: longblob             # Crop pre standardization
    crop_masked: longblob      # Crop post standardization
    act : float                # In silico activation
    n_searched: int            # Number of crops searched
    """
    @property
    def key_source(self):
        return base.MEIMask * CropSearchParameters

    def make(self,key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        search_params = (CropSearchParameters() & {'search_params': key['search_params']}).fetch1()
        search_params['n_target_images'] = 30000
        mei_params = (base.MEIParameters & key).fetch1()

        # Get model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device=device, average_batch=False)

        # Get a batch of images
        all_dl = get_dataloader(search_params)
            
        iterator = iter(all_dl)

        counter = 0
        selected_results = []
        min_act = 0

        # # track progress in an infinite while loop
        # def generator(counter, iterator):
        #     while counter < search_params['n_target_images']:
        #         yield next(iterator)

        # for imgs, masks in tqdm(generator(counter, iterator)):
        while counter < search_params['n_target_images']:
            imgs, masks = next(iterator)
            if imgs is None:
                break
            crop_results = crop_img(imgs,masks,search_params)
            imgs = standardize_crops(crop_results,search_params,key)
            acts = [pass_img(i,model,device) for i in imgs]

            for crop_result,img,act in zip(crop_results,imgs,acts):
                counter += len(img)
                crop_result['crop_masked'] = img
                crop_result['act'] = act
                for index,i in enumerate(act):
                    if i > min_act:
                        temp = dict(crop_result)
                        for j in ['crop','crop_masked','x','y','act']:
                            temp[j] = temp[j][index]
                        selected_results.append(temp)
                        if len(selected_results) < search_params['n_keep']:
                            min_act = 0
                        else:
                            selected_results = sorted(selected_results,key=lambda x: x['act'],reverse=True)
                            selected_results = selected_results[:search_params['n_keep']]
                            min_act = selected_results[-1]['act']
                                
        for i,result in enumerate(selected_results):
            temp = {**key,'crop_id':i,**result,'n_searched':counter}
            self.insert1(temp)


def search_crops(img,label,fourier_params):
    fourier_params['target_crop_size'] = 16
    search_params = {'crop_height':fourier_params['target_crop_size'], 'crop_width':fourier_params['target_crop_size'], 'crop_stride':1}
    while True:
        crop_result = crop_img(torch.tensor(np.array(img)).unsqueeze(0).unsqueeze(0),torch.tensor(np.array(label)).unsqueeze(0).unsqueeze(0),search_params)[0]
        crop_size = search_params['crop_height']*search_params['crop_width']
        coordinates = np.stack([crop_result['x'],crop_result['y']]).T
        f_sizes = np.array([(crop_result['original_label'][i:i+search_params['crop_height'],j:j+search_params['crop_width']]+1.0).sum()/(2*crop_size) for i,j in coordinates])
        f_crop_idxs = np.where(f_sizes>0.9999)[0]
        b_crop_idxs = np.where(f_sizes<0.0001)[0]
        if (len(f_crop_idxs)>fourier_params['n_crop']) & (len(b_crop_idxs)>fourier_params['n_crop']):
            select_f_crop_idxs = np.random.choice(f_crop_idxs,fourier_params['n_crop'],replace=False)
            select_b_crop_idxs = np.random.choice(b_crop_idxs,fourier_params['n_crop'],replace=False)
            return {'crop_size':search_params['crop_height'],'f_crop_coordinates':coordinates[select_f_crop_idxs],'f_crops':crop_result['crop'][select_f_crop_idxs],
                    'b_crop_coordinates':coordinates[select_b_crop_idxs],'b_crops':crop_result['crop'][select_b_crop_idxs]}
        else:
            # Reduce the crop size
            search_params['crop_height'] = search_params['crop_height'] - 1
            search_params['crop_width'] = search_params['crop_height']


def cal_power_spectrum(img,fourier_params):
    # Process the image to remove artifact
    img = np.clip(img,a_min=np.percentile(img.ravel(),fourier_params['min_percentile']),a_max = np.percentile(img.ravel(),fourier_params['max_percentile']))
    img = (img-img.mean())
    img = np.fft.fftshift(np.fft.fft2(img))
    return np.abs(img)**2

def radially_avg(matrix, bins=None, title='Radially Averaged Power Spectrum', scatter=True, **kwargs):
    import scipy.stats as stats
    h, w = matrix.shape
    if not bins:
        bins = np.max((h,w))
    center_h, center_w = h/2, w/2
    # calculate the radial average
    z = []
    v = []
    for i in range(h):
        for j in range(w):
            z.append(np.sqrt((i - center_h)**2 + (j - center_w)**2))
            v.append(matrix[i,j])
    avg, edges, bin_idx = stats.binned_statistic(z, v, bins=bins)
    data = np.array((z, v))
    return edges, avg, data


@schema 
class FourierRadialParameters(dj.Lookup):
    definition = """
    fourier_params: int 
    ---
    n_crop: int
    target_crop_size: int
    min_percentile: float #Minimum percentile to clip from 0 to 100
    max_percentile: float #Maximum percentile to clip from 0 to 100
    """
    contents = [(1,10,32,5,95)]

@schema
class FourierRadial(dj.Computed):
    definition = """
    -> FourierRadialParameters
    -> HighlyActivatingCrop
    original_img_hash: varchar(128)               # Hash of the original image
    ---
    crop_size: int       # Actual crop size
    
    f_crop_coordinates: longblob   # Coordinate for f crop   
    f_crops: longblob    # f crops
    f_crop_radials: longblob  # Radial of f crops
    
    b_crop_coordinates: longblob   # Coordinate for b crop    
    b_crops: longblob    # b crops
    b_crop_radials: longblob  # Radial of b crops
    """
    @property
    def key_source(self):
        return FourierRadialParameters * HighlyActivatingCrop

    def make(self,key):
        fourier_params = (FourierRadialParameters() & {'fourier_params':key['fourier_params']}).fetch1()
    
        original_image, original_label = (HighlyActivatingCrop() & key).fetch1('original_image','original_label')
        # Check if the image is already processed in the table (note this assume the only thing change is the frac cut off
        import hashlib
        original_img_hash = hashlib.sha256(np.ascontiguousarray(original_image)).hexdigest()
        if len(FourierRadial() & {'original_img_hash':original_img_hash}) > 0:
            # Simply copy the result 
            old_result = (FourierRadial() & {'original_img_hash':original_img_hash}).fetch(as_dict=True)[0]
            temp = {**key,'original_img_hash':original_img_hash,'crop_size':old_result['crop_size'],
                   'f_crop_coordinates':old_result['f_crop_coordinates'],'f_crops':old_result['f_crops'],
                   'f_crop_radials':old_result['f_crop_radials'],'b_crop_coordinates':old_result['b_crop_coordinates'],
                    'b_crops':old_result['b_crops'],'b_crop_radials':old_result['b_crop_radials']}
        else:
            crop_result = search_crops(original_image,original_label,fourier_params)
            temp = {**key,'original_img_hash':original_img_hash,**crop_result}
            
            for i,j in [['f_crops','f_crop_radials'],['b_crops','b_crop_radials']]:
                radials = np.stack([radially_avg(cal_power_spectrum(l,fourier_params),scatter=False)[1] for l in crop_result[i]])
                temp[j] = radials
        self.insert1(temp)

def get_variable_masks(deis,mei,mei_mask,mask_params,values=np.linspace(0.3,0.7,9),params={'closing_iters':2,'gaussian_sigma':1.5}, flipped_mask=False):
    from skimage import morphology
    img = deis.std(axis=0)
    img = img/img.max()
    
    def get_binary_mei_mask(mei,mask_params):
        # Normalize and threshold
        norm_mei = (mei - mei.mean()) / mei.std()
        thresholded = np.abs(norm_mei) > mask_params['zscore_thresh']

        # Remove small holes in the thresholded image and connect any stranding pixels
        closed = ndimage.binary_closing(thresholded, iterations=mask_params['closing_iters'])

        # Remove any remaining small objects
        labeled = morphology.label(closed, connectivity=2)
        most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
        oneobject = labeled == most_frequent

        # Create convex hull just to close any remaining holes and so it doesn't look weird
        hull = morphology.convex_hull_image(oneobject)

        return hull
    
    mei_binary_mask = get_binary_mei_mask(mei, mask_params)
    
    def helper(threshold, mei_binary_mask, flipped_mask=False):
        if not flipped_mask:
            thresholded = img > threshold
        else:
            thresholded = img <= threshold

        # Remove small holes in the thresholded image and connect any stranding pixels
        closed = ndimage.binary_closing(thresholded, iterations=params['closing_iters'])

        # Remove any remaining small objects
        labeled = morphology.label(closed, connectivity=2)
        most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
        oneobject = labeled == most_frequent
        overlapped = oneobject & mei_binary_mask
        # Smooth edges
        smoothed = ndimage.gaussian_filter(overlapped.astype(np.float32),sigma=params['gaussian_sigma'])
        return smoothed
    
    result = {'variable_mask':[],'fraction_std':[]}
    for threshold in np.linspace(0.05,0.95,1000):
        result['variable_mask'].append(helper(threshold, mei_binary_mask, flipped_mask))
        result['fraction_std'].append((deis.std(axis=0) * result['variable_mask'][-1]).sum() / (deis.std(axis=0)*mei_mask).sum())

    idx = np.argmin(abs(np.array(result['fraction_std'])[:,None]-values),axis=0)
    temp = []
    for i in idx:
        temp_dict = {}
        for temp_key in result.keys():
            temp_dict[temp_key] = result[temp_key][i]
        temp.append(temp_dict)
    return temp

@schema
class TwoComponentMaskParameters(dj.Lookup):
    definition ="""
    two_component_mask_params: int
    ---
    name: varchar(16) # Description to get appropriate params (texture-based, std-based)
    params: longblob # Params to get the appropriate masks
    """
    contents = [(1,'std_based',{'diverse_params':[14,17],'threshold_params':2,'target_fraction_std':0.6}),
                (2,'std_based',{'diverse_params':[14,17],'threshold_params':2,'target_fraction_std':0.9}),
                (3,'texture_based',{'texture_params':[13,20],'score_params':3}),
                (4,'mei_mask_only',{})]

# This table requires careful populating from either deis_schema.TextureGoodRun or deis_schema.DEIGoodRun
@schema
class TwoComponentMask(dj.Computed):
    definition = """
    -> base.MEIMask
    -> TwoComponentMaskParameters
    ---
    variable_mask: longblob  
    fixed_mask: longblob
    """
    @property
    def key_source(self):
        return base.MEIMask * TwoComponentMaskParameters
    
    def make(self,key):
        two_component_params = (TwoComponentMaskParameters & {'two_component_mask_params':key['two_component_mask_params']}).fetch1()
        mei_mask = np.array((base.MEIMask & key).fetch1('mask'))

        if two_component_params['name'] == 'std_based':
            # Get all the mei and deis
            mei = (base.MEI & key).fetch1('mei')
            temp = 'diverse_params in {}'.format(str(tuple(two_component_params['params']['diverse_params'])))

            deis = (deis_schema.DEIGoodRun * deis_schema.DEI & {'threshold_params':two_component_params['params']['threshold_params']} & key & temp).fetch('deis',order_by='diverse_params')[0]
            mei_params = (base.MEIParameters & key).fetch1()
            mask_params = (deis_schema.MaskParameters & key).fetch1()
            variable_mask = get_variable_masks(deis,mei,mei_mask,mask_params,values = [two_component_params['params']['target_fraction_std']])[0]['variable_mask']
            fixed_mask = mei_mask - variable_mask
            
        elif two_component_params['name'] == 'texture_based':
            temp = 'texture_params in {}'.format(str(tuple(two_component_params['params']['texture_params'])))
            variable_mask = np.array((deis_schema.TextureGoodRun * deis_schema.Texture & {'score_params':two_component_params['params']['score_params']} & key & temp).fetch('variable_mask',order_by='texture_params')[0])
            fixed_mask = mei_mask - variable_mask
            
        elif two_component_params['name'] == 'mei_mask_only':
            variable_mask = mei_mask
            fixed_mask = np.zeros_like(variable_mask)
        
        else:
            raise ValueError('{} is not defined'.format(two_component_params['name']))
            
        self.insert1({**key,'variable_mask':variable_mask,'fixed_mask':fixed_mask})
        
@schema
class PerlinNoiseFullFieldParameters(dj.Lookup):
    definition = """
    perlin_noise_params: int
    ---
    height: int
    width: int
    n_stimuli: int
    persistence: float # Persistence of the noise
    octaves: longblob   # List of noise components with their octaves
    standardize: bool  # Whether to standardize the image
    """
    contents = [(1,36,64,1000000,1.0,[3,6,12,24],1),(2,36,64,1000000,0.25,[3,6,12,24],1),
               (3,36,64,1000000,0.5,[3,6,12,24],1)]

@schema
class PerlinNoiseFullFieldId(dj.Computed):
    definition = """
    -> PerlinNoiseFullFieldParameters
    noise_id: int
    ---
    """
    @property 
    def key_source(self):
        return PerlinNoiseFullFieldParameters
    def make(self,key):
        perlin_noise_params = (PerlinNoiseFullFieldParameters & {'perlin_noise_params':key['perlin_noise_params']}).fetch1()
        for i in range(perlin_noise_params['n_stimuli']):
            self.insert1({**key,'noise_id':i})

@schema
class PerlinNoiseFullField(dj.Computed):
    definition = """
    -> PerlinNoiseFullFieldId
    ---
    img: blob@static
    """
    @property
    def key_source(self):
        return PerlinNoiseFullFieldId()
    
    def make(self, key):
        # Can't specify seed to take advantage of parallelism 
        perlin_noise_params = (PerlinNoiseFullFieldParameters & {'perlin_noise_params': key['perlin_noise_params']}).fetch1()
        img = perlin_noise_helper({'octaves':perlin_noise_params['octaves'],'transform_height':perlin_noise_params['height'],
                             'transform_width':perlin_noise_params['width'],'foreground_persistence':perlin_noise_params['persistence'],
                             'standardize':perlin_noise_params['standardize']})
        self.insert1({**key, 'img':img})

@schema
class TwoComponentPerlinNoiseParameters(dj.Lookup):
    definition = """
    perlin_params: int 
    ---
    variable_perlin_noise_params: int # Parameters to identify noise from PerlinNoiseFullField
    fixed_perlin_noise_params: int # Parameters to identify noise from PerlinNoiseFullField
    mask_image: bool       # Whether to mask the image
    match_stats: varchar(16) # Method for matching statistics on the image ('mask' or 'ff')
    mask_mean_subtraction: bool # Subtract mask mean
    batch_size: int
    """
    contents = [(1,1,2,0,'ff',1,64),
                (2,2,1,0,'ff',1,64),
                (3,1,3,0,'ff',1,64)]

@schema
class TwoComponentPerlinNoiseStimuli(dj.Computed):
    definition= """
    -> TwoComponentMask
    -> TwoComponentPerlinNoiseParameters
    stimuli_id: int
    ---
    variable_id: int
    fixed_id: int
    act: float
    """
    
    @property
    def key_source(self):
        return TwoComponentMask * TwoComponentPerlinNoiseParameters
    
    def make(self,key):            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        perlin_params = (TwoComponentPerlinNoiseParameters() & {'perlin_params': key['perlin_params']}).fetch1()
        mei_params = (base.MEIParameters & key).fetch1()
        mei_mask = np.array((base.MEIMask & key).fetch1('mask'))

        # Get model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device=device, average_batch=False)

        # Create the a fixed and a variable mask
        variable_mask,fixed_mask = (TwoComponentMask & key).fetch1('variable_mask','fixed_mask')
        
        # Get the stimuli
        variable_keys = (PerlinNoiseFullFieldId & {'perlin_noise_params':perlin_params['variable_perlin_noise_params']}).fetch('KEY',order_by='noise_id')
        fixed_keys = (PerlinNoiseFullFieldId & {'perlin_noise_params':perlin_params['fixed_perlin_noise_params']}).fetch('KEY',order_by='noise_id')
        
        # If the length is not equal then discard the last one
        if len(variable_keys) == len(fixed_keys):
            drop_last = False
        else:
            drop_last = True
        variable_keys = [variable_keys[i:i+perlin_params['batch_size']] for i in range(0, len(variable_keys), perlin_params['batch_size'])]
        fixed_keys = [fixed_keys[i:i+perlin_params['batch_size']] for i in range(0, len(fixed_keys), perlin_params['batch_size'])]
        
        if drop_last:
            variable_keys = variable_keys[:-1]
            fixed_keys = fixed_keys[:-1]
        
        all_acts = []
        for counter,(i,j) in tqdm(enumerate(zip(variable_keys,fixed_keys))):
            v_imgs = variable_mask[None,...] * np.stack((PerlinNoiseFullField & i).fetch('img'))
            f_imgs = fixed_mask[None,...] * np.stack((PerlinNoiseFullField & j).fetch('img'))
            imgs = v_imgs + f_imgs
            
            if perlin_params['match_stats'] == 'ff':
                target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
            else:
                raise ValueError()
            imgs = ops.standardize_image(imgs, target_mean, target_std, mei_mask, 
                                       perlin_params['mask_mean_subtraction'],perlin_params['mask_image'], 
                                       perlin_params['match_stats'])
            with torch.no_grad():
                acts = model(torch.tensor(imgs,dtype=torch.float32,device=device).unsqueeze(1)).cpu().numpy()
            
            all_acts.extend(acts.copy())

        for i, (variable_key,fixed_key,act) in enumerate(zip(np.concatenate(variable_keys), np.concatenate(fixed_keys), all_acts)):
            self.insert1({**key, 'stimuli_id':i,'variable_id':variable_key['noise_id'], 'fixed_id':fixed_key['noise_id'], 'act':act})     


@schema
class TwoComponentPerlinNoiseResponses(dj.Computed):
    definition= """
    -> TwoComponentMask
    -> TwoComponentPerlinNoiseParameters
    ---
    variable_ids:       blob@static
    fixed_ids:          blob@static
    activations:        blob@static
    """
    
    @property
    def key_source(self):
        return TwoComponentMask * TwoComponentPerlinNoiseParameters
    
    def make(self,key):            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        perlin_params = (TwoComponentPerlinNoiseParameters() & {'perlin_params': key['perlin_params']}).fetch1()
        mei_params = (base.MEIParameters & key).fetch1()
        mei_mask = np.array((base.MEIMask & key).fetch1('mask'))

        # Get model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device=device, average_batch=False)

        # Create the a fixed and a variable mask
        variable_mask,fixed_mask = (TwoComponentMask & key).fetch1('variable_mask','fixed_mask')
        
        # Get the stimuli
        variable_keys = (PerlinNoiseFullFieldId & {'perlin_noise_params':perlin_params['variable_perlin_noise_params']}).fetch('KEY',order_by='noise_id')
        fixed_keys = (PerlinNoiseFullFieldId & {'perlin_noise_params':perlin_params['fixed_perlin_noise_params']}).fetch('KEY',order_by='noise_id')
        
        # If the length is not equal then discard the last one
        if len(variable_keys) == len(fixed_keys):
            drop_last = False
        else:
            drop_last = True
        variable_keys = [variable_keys[i:i+perlin_params['batch_size']] for i in range(0, len(variable_keys), perlin_params['batch_size'])]
        fixed_keys = [fixed_keys[i:i+perlin_params['batch_size']] for i in range(0, len(fixed_keys), perlin_params['batch_size'])]
        
        if drop_last:
            variable_keys = variable_keys[:-1]
            fixed_keys = fixed_keys[:-1]
        
        all_acts = []
        for counter,(i,j) in tqdm(enumerate(zip(variable_keys,fixed_keys))):
            v_imgs = variable_mask[None,...] * np.stack((PerlinNoiseFullField & i).fetch('img'))
            f_imgs = fixed_mask[None,...] * np.stack((PerlinNoiseFullField & j).fetch('img'))
            imgs = v_imgs + f_imgs
            
            if perlin_params['match_stats'] == 'ff':
                target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
            else:
                raise ValueError()
            imgs = ops.standardize_image(imgs, target_mean, target_std, mei_mask, 
                                       perlin_params['mask_mean_subtraction'],perlin_params['mask_image'], 
                                       perlin_params['match_stats'])
            with torch.no_grad():
                acts = model(torch.tensor(imgs,dtype=torch.float32,device=device).unsqueeze(1)).cpu().numpy()
            
            all_acts.extend(acts.copy())

        vks = [vk['noise_id'] for vk in np.concatenate(variable_keys)]
        fks = [fk['noise_id'] for fk in np.concatenate(fixed_keys)]
        self.insert1({**key, 'variable_ids': vks, 'fixed_ids': fks, 'activations': all_acts})


@schema
class DatasetParameters(dj.Lookup):
    definition = """
    dataset_params: int
    ---
    transform_height: int  # Height of the resized image
    transform_width: int   # Width of the resized image
    dataset_name: varchar(16)
    """
    contents = [(1,64,64,'cub'),(2,64,64,'cub_2'),(3,64,64,'cub_5'),(4,64,64,'cub_monet'),
                (5,64,64,'cub_monet_same_freq_same_contrast_same_mean'),
               (6,64,64,'cub_perlin_same_freq_same_contrast_same_mean')]

def blur_background(image,label,sigma,border_sigma=1.0):
    label = ((label+1)/2)
    label = ndimage.gaussian_filter(label,sigma=border_sigma)
    foreground = image * label

    background = image * (1-label) + np.ones_like(img)*image.mean() * label
    background = ndimage.gaussian_filter(background,sigma=sigma)
    background = background * (1-label)
    return foreground+background
def hann(q):
    return (0.5 + 0.5 * np.cos(q)) * (np.abs(q) < np.pi)

def make_monet_texture(height=64,width=64,ori=np.pi,sigma=0.0,ori_coherence=5,mean_intensity=0.5,std_intensity = 0.25,pad=10,seed=None):
    # ori from 0 to np.pi
    # sigma from 0 to 2
    # mean_intensity from 0.2 to 0.8
    # std_intensity stays 0.1
    import numpy as np
    from numpy import fft
    from scipy import signal
    from scipy.ndimage import gaussian_filter
    r = np.random.default_rng(seed)
    m = r.normal(size=[height+2*pad,width+2*pad])

    m = gaussian_filter(m,sigma)

    fy, fx = np.meshgrid(
        fft.ifftshift(np.arange(-np.floor(m.shape[0]/2), m.shape[0]/2))*2*np.pi/m.shape[0],
        fft.ifftshift(np.arange(-np.floor(m.shape[1]/2), m.shape[1]/2))*2*np.pi/m.shape[1],
        indexing='ij')

    finterp = np.exp(-(fy ** 2 + fx ** 2) / 2)
    m = fft.fft2(m, axes=(0, 1))

    theta = np.mod(np.arctan2(fx, fy) + ori, np.pi) - np.pi / 2
    fmask = finterp * (np.sqrt(ori_coherence) * hann(theta * ori_coherence))
    m = fmask * m
    m = np.real(fft.ifft2(m, axes=(0, 1)))
    
    m = np.clip(m,a_min=np.percentile(m.ravel(),1),a_max=np.percentile(m.ravel(),99))
    m = (m-m.mean())*(std_intensity/m.std()) + mean_intensity
    m = np.clip(m,a_min=0.0,a_max=1.0)
    return m[pad:height+pad,pad:width+pad]

def make_random_monet_texture(seed=None):
    r = np.random.default_rng(seed)
    return make_monet_texture(ori=r.uniform(low=0,high=np.pi),
                              sigma=r.uniform(low=0,high=5),
                              mean_intensity=r.uniform(low=0.2,high=0.8),
                             ori_coherence=r.uniform(low=2,high=12))
@schema  
class ImageDataset(dj.Computed):
    definition = """
    -> DatasetParameters
    image_id: int
    ---
    image: blob@static
    label: blob@static
    """
    @property
    def key_source(self):
        return DatasetParameters
    
    def make(self,key):
        dataset_params = (DatasetParameters() & {'dataset_params': key['dataset_params']}).fetch1()
        dataset_params['batch_size'] = 1
        all_dl = get_dataloader(dataset_params)
        iterator = iter(all_dl)
        for i,(img, mask) in tqdm(enumerate(iterator)):
            if img is None:
                break
            img = img.numpy().squeeze()
            mask = mask.numpy().squeeze()
            self.insert1({**key,'image_id':i,'image':img,'label':mask})
        
        # Todo: Integrate old way into new more systematic way
        
#         if len(dataset_params['dataset_name'].split('_')) <2:
#             sigma = 0
#         else:
#             try:
#                 sigma = float(dataset_params['dataset_name'].split('_')[-1])
#             except:
#                 sigma = 0
#         all_dl = get_dataloader(dataset_params)
#         iterator = iter(all_dl)
#         monet_mode = dataset_params['dataset_name'].split('_')[1] == 'monet'

#         for i,(img, mask) in tqdm(enumerate(iterator)):
#             if img is None:
#                 break
#             img = img.numpy().squeeze()
#             mask = mask.numpy().squeeze()
#             if monet_mode:
#                 label = np.array(mask)
#                 label = ((label+1)/2)
#                 img = make_random_monet_texture() * label + make_random_monet_texture() * (1-label)
#             # Blur the image just the background
#             if sigma>0:
#                 img = blur_background(img,mask,sigma=sigma,border_sigma=1.0)
#             self.insert1({**key,'image_id':i,'image':img,'label':mask})
            

from attorch.layers import SpatialTransformerPyramid2d, Elu1
from staticnet.cores import GaussianLaplaceCore
class ChannelSpatialTransformerPyramid2d(SpatialTransformerPyramid2d):
    def __init__(self, readout_state_dict,neuron_id,in_shape=(96,64,64), scale_n=5, positive=False, bias=True, downsample=False, _skip_upsampling=False, type='gauss5x5',*args,**kwargs):
        super().__init__(in_shape=in_shape, outdims=np.array(neuron_id).size,scale_n=scale_n, positive=positive, bias=bias, downsample=downsample, _skip_upsampling=_skip_upsampling, type=type)
        self.neuron_id = np.array(neuron_id)
        self.n_readouts = in_shape[1]*in_shape[2]
        self.load_state_dict(readout_state_dict)
        
    def load_state_dict(self,readout_state_dict):
        nx,ny = self.in_shape[1:]
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        xx,yy = np.meshgrid(x,y,indexing='ij')
        coordinates = torch.tensor(np.stack([xx.ravel(), yy.ravel()]).T,dtype=torch.float32).unsqueeze(1).unsqueeze(0)[...,[1,0]]
        self.grid = nn.Parameter(coordinates)
        self.bias.data = readout_state_dict['bias'][self.neuron_id]
        self.features.data = readout_state_dict['features'][:,:,:,self.neuron_id]
        
    def forward(self, x):
        if self.positive:
            positive(self.features)
        self.grid.data = torch.clamp(self.grid.data, -1, 1)
        N, c, w, h = x.size()
        m = self.gauss_pyramid.scale_n + 1
        feat = self.features.view(m * c,self.outdims)
        feat = feat.permute(1,0)
        grid = self.grid.expand(N, self.n_readouts, 1, 2)
        pools = [F.grid_sample(xx, grid, align_corners=True) for xx in self.gauss_pyramid(x)]
        y = torch.cat(pools, dim=1).squeeze(-1)
        y = (y[:,None,:,:]*feat[None,:,:,None]).sum(2).view(N,self.outdims,self.in_shape[1],self.in_shape[2])
        if self.bias is not None:
            if self.neuron_id.size >1:
                y = y + self.bias[None,:,None,None]
            else:
                y = y + self.bias
        return y
    
class Ensemble(nn.Module):
    def __init__(self,models):
        super(Ensemble,self).__init__()
        self.models = models
        
    def forward(self, x):
        resps = [m(x) for m in self.models]
        resps = torch.stack(resps)
        resp = resps.mean(0)
        return resp

    
def convolution_helper(key,in_shape,use_avg_model=True,device='cuda'):
    model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if use_avg_model else key)
    all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
    all_models = [(static_models.Model & mk).load_network().eval() for mk in all_keys]
    core_hash, ro_hash = (configs.NetworkConfig.CorePlusReadout & {'net_hash':key['net_hash']}).fetch1('core_hash','ro_hash')
    core_params = dict((configs.CoreConfig.GaussianLaplace() & {'core_hash':core_hash}).fetch1())
    ro_params = dict((configs.ReadoutConfig.SpatialTransformerPyramid2d() & {'ro_hash':ro_hash}).fetch1())
    model_dict = [{'core_state_dict':i.core.state_dict(),'core_params':core_params,'readout_state_dict':i.readout[key['readout_key']].state_dict()
                   ,'ro_params':ro_params,'neuron_idx':key['neuron_id']} for i in all_models]
    all_models = []
    for i in model_dict:
        core = GaussianLaplaceCore(input_channels=1,**i['core_params']).eval().to(device)
        readout = ChannelSpatialTransformerPyramid2d(i['readout_state_dict'],i['neuron_idx'],**i['ro_params'],in_shape=in_shape).eval().to(device)
        core.load_state_dict(i['core_state_dict'])
        all_models.append(utils.Compose([core,readout,Elu1()]))
    model = Ensemble(models=all_models)
    return model

def model_helper(key,use_avg_model=True,device='cuda'):
    # Get model
    model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                 use_avg_model else key)
    all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
    all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
    mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                   (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))
    mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
    model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                            neuron_idx=key['neuron_id'], device=device, average_batch=False)
    return model

def save_helper(result,save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

@schema
class WholeImageResponseParameters(dj.Lookup):
    definition = """
    whole_image_params: int
    ---
    params: longblob
    """
    contents = [(1,{'dataset_params':1,'method_name':'crop_based','crop_height':36,'crop_width':36,'crop_stride':1,'padding':18,'padding_mode':'constant',
                    'constant_value':0.5,'mask_image':1,'match_stats':'ff','mask_mean_subtraction':1,'save_frequency':1000}),
               (2,{'dataset_params':1,'method_name':'full_convolution','save_frequency':1000}),
               (3,{'dataset_params':2,'method_name':'full_convolution','save_frequency':1000}),
               (4,{'dataset_params':3,'method_name':'full_convolution','save_frequency':1000}),
               (5,{'dataset_params':4,'method_name':'full_convolution','save_frequency':1000})]
    
@files('static')
@schema
class WholeImageResponse(dj.Computed):
    definition = """
    -> base.MEIMask
    -> WholeImageResponseParameters
    ---
    response:      blob@static
    response_path: varchar(256)
    """
    @property
    def key_source(self):
        return base.MEIMask * WholeImageResponseParameters

    def make(self,key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        whole_image_params = (WholeImageResponseParameters() & {'whole_image_params': key['whole_image_params']}).fetch1()['params']
        mei_params = (base.MEIParameters & key).fetch1()
        full_image_height, full_image_width = (DatasetParameters & {'dataset_params':whole_image_params['dataset_params']}).fetch1('transform_height','transform_width')
        response_path = os.path.join(self.tuple_dir(key, create=True), "response.pickle")
        image_ids = set((ImageDataset & {'dataset_params':whole_image_params['dataset_params']}).fetch('image_id'))
        response = {}
        if os.path.exists(response_path):
            logger.info('Loading training history from checkpoint')
            try:
                # Load image_id that has been populated
                with open(response_path, 'rb') as handle:
                    response = pickle.load(handle)
                # Get all the image_id that hasn't been populated
                image_ids = image_ids - set(response.keys())
            except (pickle.UnpicklingError, ValueError, EOFError) as error:
                response = {}

        # Method full convolution
        if whole_image_params['method_name'] == 'full_convolution':
            model = convolution_helper(key,(96,full_image_height,full_image_width),mei_params['use_avg_model'],device=device)
            
            from staticnet_analyses import multi_mei
            # Get training dataset and some training stats
            train_stats = multi_mei.prepare_data(key, key['readout_key'])
            _, (_, _, _, _), train_mean, _, _, train_std = train_stats
            train_mean,train_std = train_mean/225.0, train_std/225.0
            
            counter = 0
            for image_id in tqdm(image_ids):
                counter += 1
                image = np.array((ImageDataset & {'dataset_params':whole_image_params['dataset_params'],'image_id':image_id}).fetch1('image'),dtype=float).squeeze()
                
                image = np.pad(image,((7,7),(7,7)),'constant',constant_values=image.mean())
                
                image = (image-train_mean)/train_std
                image = torch.tensor(image,dtype=torch.float32,device=device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    response[image_id] = model(image).cpu().numpy().squeeze()
                    
                if counter>whole_image_params['save_frequency']:
                    save_helper(response,response_path)
                    counter = 0
        
        elif whole_image_params['method_name'] == 'crop_based':
            model = model_helper(key,mei_params['use_avg_model'],device=device)
            counter = 0
            for image_id in tqdm(image_ids):
                counter += 1
                image = np.array((ImageDataset & {'dataset_params':whole_image_params['dataset_params'],'image_id':image_id}).fetch1('image'),dtype=float).squeeze()
                image = torch.tensor(image,dtype=torch.float32)
                import torch.nn.functional as F
                if whole_image_params['padding_mode'] == 'constant':
                    image = F.pad(image,[whole_image_params['padding']]*4,whole_image_params['padding_mode'],value=whole_image_params['constant_value'])
                else:
                    image = F.pad(image,[whole_image_params['padding']]*4,whole_image_params['padding_mode'])
                image = image.unsqueeze(0)
                crop_results = crop_img(image[None,...],image[None,...],whole_image_params)
                imgs = standardize_crops(crop_results,whole_image_params,key)
                acts = [pass_img(i,model,device) for i in imgs][0].squeeze()
                assert acts.size == full_image_height * full_image_width, "Can't resize to original image size"
                response[image_id] = np.reshape(acts,[full_image_height,full_image_width])
                
                if counter>whole_image_params['save_frequency']:
                    save_helper(response,response_path)
                    counter = 0
        else:
            raise ValueError('Method {} is not specified'.format(whole_image_params['method_name']))
        save_helper(response,response_path)
        self.insert1({**key,'response':response,'response_path':response_path})

@schema
class DatasetCropParameters(dj.Lookup):
    definition = """
    dataset_crop_params: int
    ---
    target_crop_size: int
    n_crop: int
    low_mix_f: float # Between 0 to 1 to select crops that contain both
    high_mix_f: float
    centered_mix: bool # To ensure the border go through the middle of the crop
    one_region: bool   # To ensure the mix have only 1 connected foreground region
    one_region_min_thres: float # From 0 to 1 as the minimum size of the biggest region over the sum of all regions 
    """
    contents = [(1,16,10,0.3,0.7,1,1,0.9),(2,16,10,0.3,0.7,1,0,0.9)]



def label_contain_border_center(imgs):
    img_size = imgs.shape[-1]
    if img_size%2 == 0:
        imgs = imgs[...,img_size//2-1:1+(img_size//2),img_size//2-1:1+(img_size//2)]
    else:
        imgs = imgs[...,img_size//2-1:2+(img_size//2),img_size//2-1:2+(img_size//2)]
    return imgs.prod(axis=(-1,-2))==0

def label_contain_one_region(imgs,thres):
    from scipy import ndimage
    from skimage import morphology
    result = []
    for img in imgs:
        labeled = morphology.label(img,connectivity=2)
        counts = np.bincount(labeled.ravel())[1:]
        result.append(counts[np.argmax(counts)]/counts.sum()>thres)
    return np.array(result)

def search_crop_with_mix(img,label,dataset_crop_params):
    dataset_crop_params['target_crop_size'] = 16
    search_params = {'crop_height':dataset_crop_params['target_crop_size'], 'crop_width':dataset_crop_params['target_crop_size'], 'crop_stride':1}
    while True:
        crop_result = crop_img(torch.tensor(np.array(img)).unsqueeze(0).unsqueeze(0),torch.tensor(np.array(label)).unsqueeze(0).unsqueeze(0),search_params)[0]
        crop_size = search_params['crop_height']*search_params['crop_width']
        coordinates = np.stack([crop_result['x'],crop_result['y']]).T
        label_crops = np.stack([(crop_result['original_label'][i:i+search_params['crop_height'],j:j+search_params['crop_width']]+1.0)/2 for i,j in coordinates])
        f_sizes = label_crops.sum(axis=(-1,-2))/crop_size
        f_crop_idxs = np.where(f_sizes>0.9999)[0]
        b_crop_idxs = np.where(f_sizes<0.0001)[0]
        
        # Solve for mix crop is a bit more complicated
        mix_crop_idxs = np.where(np.logical_and((f_sizes>=dataset_crop_params['low_mix_f']),(f_sizes<=dataset_crop_params['high_mix_f'])))[0]
        
        if dataset_crop_params['centered_mix']:
            mix_crop_idxs = mix_crop_idxs[np.where(label_contain_border_center(label_crops[mix_crop_idxs]))[0]]
        
        if dataset_crop_params['one_region']:
            mix_crop_idxs = mix_crop_idxs[np.where(label_contain_one_region(label_crops[mix_crop_idxs],thres=dataset_crop_params['one_region_min_thres']))[0]]
            
        if (len(f_crop_idxs)>dataset_crop_params['n_crop']) & (len(b_crop_idxs)>dataset_crop_params['n_crop']) & (len(mix_crop_idxs)>dataset_crop_params['n_crop']):
            select_f_crop_idxs = np.random.choice(f_crop_idxs,dataset_crop_params['n_crop'],replace=False)
            select_b_crop_idxs = np.random.choice(b_crop_idxs,dataset_crop_params['n_crop'],replace=False)
            select_mix_crop_idxs = np.random.choice(mix_crop_idxs,dataset_crop_params['n_crop'],replace=False)
            return {'crop_size':search_params['crop_height'],'f_crop_coordinates':coordinates[select_f_crop_idxs],
                    'f_crops':crop_result['crop'][select_f_crop_idxs],'b_crop_coordinates':coordinates[select_b_crop_idxs],
                    'b_crops':crop_result['crop'][select_b_crop_idxs],'mix_crop_coordinates':coordinates[select_mix_crop_idxs],
                   'mix_crops':crop_result['crop'][select_mix_crop_idxs]}
        elif search_params['crop_height'] == 1:
            return {'crop_size':0,'f_crop_coordinates':np.array([]),
                    'f_crops':np.array([]),'b_crop_coordinates':np.array([]),
                    'b_crops':np.array([]),'mix_crop_coordinates':np.array([]),
                   'mix_crops':np.array([])}
        else:
            # Reduce the crop size
            search_params['crop_height'] = search_params['crop_height'] - 1
            search_params['crop_width'] = search_params['crop_height']

@schema
class DatasetCrop(dj.Computed):
    definition = """
    -> DatasetCropParameters
    -> ImageDataset
    ---
    crop_size: int       # Actual crop size
    
    f_crop_coordinates: longblob   # Coordinate for f crop   
    f_crops: longblob    # f crops
    
    b_crop_coordinates: longblob   # Coordinate for b crop    
    b_crops: longblob    # b crops
    
    mix_crop_coordinates: longblob  # Coordinate for mix crop
    mix_crops: longblob  # mix crops
    """
    @property
    def key_source(self):
        return DatasetCropParameters * ImageDataset

    def make(self,key):
        dataset_crop_params = (DatasetCropParameters() & {'dataset_crop_params':key['dataset_crop_params']}).fetch1()
        image,label = (ImageDataset & key).fetch1('image','label')
        crop_result = search_crop_with_mix(image,label,dataset_crop_params)
        self.insert1({**key,**crop_result})


@schema
class DatasetFourierRadial(dj.Computed):
    definition = """
    -> CropSearchParameters
    -> FourierRadialParameters
    original_img_hash: varchar(128)               # Hash of the original image
    ---
    crop_size: int       # Actual crop size
    
    f_crop_coordinates: longblob   # Coordinate for f crop   
    f_crops: longblob    # f crops
    f_crop_radials: longblob  # Radial of f crops
    
    b_crop_coordinates: longblob   # Coordinate for b crop    
    b_crops: longblob    # b crops
    b_crop_radials: longblob  # Radial of b crops
    """
    @property
    def key_source(self):
        return FourierRadialParameters * CropSearchParameters

    def make(self,key):
        fourier_params = (FourierRadialParameters() & {'fourier_params':key['fourier_params']}).fetch1()
        search_params = (CropSearchParameters() & {'search_params':key['search_params']}).fetch1()
        all_dl = get_dataloader(search_params)
        iterator = iter(all_dl)
        for imgs, masks in tqdm(iterator):
            if imgs is None:
                break
            for img,mask in zip(imgs,masks):
                original_image = img.numpy().squeeze()
                original_label = mask.numpy().squeeze()
                # Check if the image is already processed in the table (note this assume the only thing change is the frac cut off
                import hashlib
                original_img_hash = hashlib.sha256(np.ascontiguousarray(original_image)).hexdigest()
                # if len(DatasetFourierRadial() & {'original_img_hash':original_img_hash}) > 0:
                #     # Simply copy the result 
                #     old_result = (DatasetFourierRadial() & {'original_img_hash':original_img_hash}).fetch(as_dict=True)[0]
                #     temp = {**key,'original_img_hash':original_img_hash,'crop_size':old_result['crop_size'],
                #            'f_crop_coordinates':old_result['f_crop_coordinates'],'f_crops':old_result['f_crops'],
                #            'f_crop_radials':old_result['f_crop_radials'],'b_crop_coordinates':old_result['b_crop_coordinates'],
                #             'b_crops':old_result['b_crops'],'b_crop_radials':old_result['b_crop_radials']}
                # else:
                crop_result = search_crops(original_image,original_label,fourier_params)
                temp = {**key,'original_img_hash':original_img_hash,**crop_result}

                for i,j in [['f_crops','f_crop_radials'],['b_crops','b_crop_radials']]:
                    radials = np.stack([radially_avg(cal_power_spectrum(l,fourier_params),scatter=False)[1] for l in crop_result[i]])
                    temp[j] = radials
                print(len(temp))
                self.insert1(temp)

@schema
class HighLowMaskParameters(dj.Lookup):
    definition = """
    high_low_mask_params: int
    ---
    binarized_mei_thres: float
    thres: float  # Frequency of the pixel belong to the high frequency region
    closing_iters: int
    gaussian_sigma: float
    """
    
    contents = [(1,0.3,0.5,2,1.0),(2,0.3,0.6,2,1.0),(3,0.3,0.7,2,1.0),(4,0.3,0.8,2,1.0)]
    
# Note that only use search_params = 4,5,6
@schema
class HighLowMask(dj.Computed):
    definition = """
    -> HighLowMaskParameters
    -> CropSearchParameters
    -> base.MEIMask
    ---
    high_mask: longblob
    low_mask: longblob
    p: float # Ratio of the high_mask over the mei_mask
    """
    @property
    def key_source(self):
        return base.MEIMask * HighLowMaskParameters * CropSearchParameters
    
    def make(self,key):
        from scipy import ndimage
        from skimage import morphology
        from featurevis import ops
        high_low_mask_params = (HighLowMaskParameters & key).fetch1()
        search_params = (CropSearchParameters & key).fetch1()
        mask_float, mask_x, mask_y = (base.MEIMask & key).fetch1('mask', 'mask_x', 'mask_y')
        binarized_mei_mask = np.array(mask_float)
        
        #Binarized mei mask
        binarized_mei_mask[binarized_mei_mask<high_low_mask_params['binarized_mei_thres']] = 0.0
        binarized_mei_mask[binarized_mei_mask>0.0] = 1.0
        
        # Get the crop label
        original_labels,xs,ys = (HighlyActivatingCrop & {'search_params':search_params['search_params']} & key).fetch('original_label','x','y')
        masked_labels = np.stack([original_label[x:x+search_params['crop_height'],y:y+search_params['crop_width']] for original_label,x,y in zip(original_labels,xs,ys)])
        masked_labels = (masked_labels + 1.0)/2
        masked_labels = ops.create_whole_mei(masked_labels,binarized_mei_mask, mask_x, mask_y, output_size=(len(masked_labels),36, 64), normalize_crop=False) * binarized_mei_mask[None,...]
        # Threshold, fill holes and blur
        high_frequency_mask = masked_labels.sum(axis=0)/len(masked_labels)
        #thresholded = high_frequency_mask > np.percentile(high_frequency_mask[mask_float>0.0].ravel(),high_low_mask_params['thres'])
        thresholded = high_frequency_mask > high_low_mask_params['thres']
        try:
            closed = ndimage.binary_closing(thresholded, iterations=high_low_mask_params['closing_iters'])
            labeled = morphology.label(closed, connectivity=2)
            most_frequent = np.argmax(np.bincount(labeled.ravel())[1:]) + 1
            oneobject = labeled == most_frequent
            smoothed = ndimage.gaussian_filter(oneobject.astype(np.float32),sigma=high_low_mask_params['gaussian_sigma'])
        except:
            smoothed = np.zeros_like(mask_float)
        
        self.insert1({**key,'high_mask':smoothed*mask_float,'low_mask':(1.0-smoothed)*mask_float,'p':(smoothed*mask_float).sum()/mask_float.sum()})

@schema
class ImageBoundaryParameters(dj.Lookup):
    definition = """
    image_boundary_params: int
    ---
    sigma: float
    method: varchar(24)
    """
    contents = [(1,1.5,'8-connected'),
               (2,1.5,'8-connected_95_scaled')]
@schema 
class ImageBoundary(dj.Computed):
    definition = """
    -> ImageBoundaryParameters
    -> ImageDataset
    ---
    boundary: blob@static
    """
    @property
    def key_source(self):
        return ImageBoundaryParameters * ImageDataset
    def make(self,key):
        from scipy.ndimage import binary_erosion,gaussian_filter
        image_boundary_params = (ImageBoundaryParameters & key).fetch1()
        label = (ImageDataset & key).fetch1('label')
        label = (np.array(label)+1.0)/2
        if image_boundary_params['method'].startswith('8-connected'):
            k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1
        elif image_boundary_params['method'].startswith('4-connected'):
            k = np.ones((3,3),dtype=int)
        else:
            raise ValueError('Method {} is not specified'.format(image_boundary_params['method']))
        boundary = label-binary_erosion(label,k)
        boundary = gaussian_filter(boundary,sigma=image_boundary_params['sigma'])
        
        if image_boundary_params['method'].endswith('scaled'):
            percentile =  float(image_boundary_params['method'].split('_')[-2])
            boundary = boundary * (1.0/(np.percentile(boundary.ravel(),percentile)+1e-9))
            boundary = np.clip(boundary,a_min=0.0,a_max=1.0)
        self.insert1({**key,'boundary':boundary})

@schema
class MultipleNeuronWholeImageResponseParameters(dj.Lookup):
    definition = """
    multiple_neuron_params: int
    ---
    neuron_src_table: varchar(128)
    n_subset: int
    n_neuron: int
    seed: int
    """
    contents = [(1,'deis_schema.NeuronSetRequest & {"method_id": 2}',10,400,0),
               (2,'deis_schema.NeuronSetRequest & {"method_id": 3}',10,400,0),
               (3,'deis_schema.NeuronSetRequest & {"method_id": 3}',10,20,0),
               (4,'deis_schema.NeuronSetRequest & {"method_id": 3}',10,100,0),
               (5,'deis_schema.NeuronSetRequest & {"method_id": 3}',7,10,0),
               (6,'deis_schema.NeuronSetRequest & {"method_id": 3}',7,10,1234),
               (7,'deis_schema.NeuronSetRequest & {"method_id": 3}',3,100,1234),
               (8,'deis_schema.NeuronSetRequest & {"method_id": 3}',3,50,1234),]

@schema
class MultipleNeuronWholeImageResponseList(dj.Computed):
    definition = """
    -> MultipleNeuronWholeImageResponseParameters
    set_id: int
    ---
    neuron_list_idx: longblob
    neuron_list: longblob
    """
    
    def make(self,key):
        multiple_neuron_params = (MultipleNeuronWholeImageResponseParameters & key).fetch1()
        neuron_list = eval(multiple_neuron_params['neuron_src_table']).fetch(dj.key, as_dict=True, order_by='group_id, neuron_id')
        np.random.seed(multiple_neuron_params['seed'])
        self.insert1({**key,'set_id':-1,'neuron_list_idx':np.arange(len(neuron_list)),'neuron_list':neuron_list})
        
        # Cheat to sample differently without touching previous result designate what multiple_neuron_params to be
        # sampled based on high-low bipartite here
        # -1 is still the entire population, even is classical and odd is bipartite
        if multiple_neuron_params['multiple_neuron_params'] in [3,4]:
            rest = {'mei_params':10, 'mask_params':3, 'mask_stats_params': 2, 'threshold_params': 2, 'texture_params': 20, 'score_params': 3}
            diverse_rest = 'diverse_params in (14, 17)'
            texture_neuron_rel = deis_schema.TextureGoodRun & rest & diverse_rest & neuron_list
            bpts = []
            keys = texture_neuron_rel.fetch(dj.key)
            for i in keys:
                p, bpt = deis_schema.TextureGoodRun().compute_p(i)
                bpts.append(bpt)
            bpts = np.array(bpts)

            # Classical index
            classical_idx = np.where(bpts<=1e-9)[0]
            assert len(classical_idx) != 0,"No classical cell."
            # Bipartite index
            bipartite_idx = np.where(bpts>1e-9)[0]
            assert len(bipartite_idx) != 0,"No bipartite cell."
            # To guarantee more distinction, only take the top 50 percentile
            thres = np.percentile(bpts[bipartite_idx],50)
            bipartite_idx = np.where(bpts>thres)[0]
            for i in range(multiple_neuron_params['n_subset']):
                if i%2==0:
                    subset_idx = np.random.choice(classical_idx,multiple_neuron_params['n_neuron'], replace=False)
                else:
                    subset_idx = np.random.choice(bipartite_idx,multiple_neuron_params['n_neuron'], replace=False)
                subset_list = [neuron_list[i] for i in subset_idx]
                self.insert1({**key,'set_id':i,'neuron_list_idx':subset_idx,'neuron_list':subset_list})
        
        elif multiple_neuron_params['multiple_neuron_params'] in [5, 6]:
            # percentile bounds on AUC
            bound_ls = [(None, 1), (99, None), (None, 10), (90, None), (20, 40), (40, 60), (60, 80)]
            bounds = dict()
            for i, tup in enumerate(bound_ls):
                bounds[i] = tup
            cell_types = deis_schema.randomly_select_population(neuron_list, metrics=['auc'], n_per_group=multiple_neuron_params['n_neuron'], seed=multiple_neuron_params['seed'], bounds=bounds)
            for i in cell_types.keys():            
                subset_idx = cell_types[i]['neuron_list_idx']
                subset_list = cell_types[i]['neuron_list']
                self.insert1({**key,'set_id':i,'neuron_list_idx':subset_idx,'neuron_list':subset_list})
        
        elif multiple_neuron_params['multiple_neuron_params'] in [7]:
            # percentile bounds on AUC
            bound_ls = [(None, 25), (75, None), (25, 75)]
            bounds = dict()
            for i, tup in enumerate(bound_ls):
                bounds[i] = tup
            cell_types = deis_schema.randomly_select_population(neuron_list, metrics=['residual', 'auc'], n_per_group=multiple_neuron_params['n_neuron'], seed=multiple_neuron_params['seed'], bounds=bounds)
            for i in cell_types.keys():            
                subset_idx = cell_types[i]['neuron_list_idx']
                subset_list = cell_types[i]['neuron_list']
                self.insert1({**key,'set_id':i,'neuron_list_idx':subset_idx,'neuron_list':subset_list})

        # get cell subgroups directly from deis_schema.CellGroupAssignment.Member
        elif multiple_neuron_params['multiple_neuron_params'] in [8]:
            groups, neurons = eval(multiple_neuron_params['neuron_src_table']).fetch('group_id', 'neuron_id', order_by='group_id, neuron_id')
            cell_group_params = {'include_params':1, 'group_params':1}
            labels = ['simple', 'complex', 'bipartite30_70']
            for i, label in enumerate(labels):
                subset_list = (deis_schema.TextureLookup * deis_schema.CellGroupAssignment.Member & cell_group_params & {'label': label}).fetch('group_id', 'neuron_id', as_dict=True, order_by='group_id, neuron_id')
                subset_idx = np.array([np.where((groups == dic['group_id']) & (neurons == dic['neuron_id']))[0].item() for dic in subset_list])
                self.insert1({**key,'set_id':i,'neuron_list_idx':subset_idx,'neuron_list':subset_list})
            # all neurons included after filtering
            _, subset_list = deis_schema.CellGroupAssignment().get_include_keys(cell_group_params)
            subset_list = (deis_schema.TextureLookup & subset_list).fetch('group_id', 'neuron_id', as_dict=True, order_by='group_id, neuron_id')
            subset_idx = np.array([np.where((groups == dic['group_id']) & (neurons == dic['neuron_id']))[0].item() for dic in subset_list])
            self.insert1({**key,'set_id':-2,'neuron_list_idx':subset_idx,'neuron_list':subset_list})
        else:            
            for i in range(multiple_neuron_params['n_subset']):
                subset_idx = np.random.choice(range(len(neuron_list)),multiple_neuron_params['n_neuron'], replace=False)
                subset_list = [neuron_list[i] for i in subset_idx]
                self.insert1({**key,'set_id':i,'neuron_list_idx':subset_idx,'neuron_list':subset_list})

@files('static')
@schema
class MultipleNeuronWholeImageResponse(dj.Computed):
    definition = """
    -> MultipleNeuronWholeImageResponseParameters
    -> WholeImageResponseParameters
    -> ImageDataset
    ---
    response: blob@static
    """
    @property
    def key_source(self):
        # Not entirely correct since WholeImageResponseParamters contain information about what ImageDataset to use but they have same number of images regardless
        # Restriction for keys of interested is: MultipleNeuronWholeImageResponseParameters * WholeImageResponseParameters * ImageDataset & {'dataset_params':3,'whole_image_params':4}
        return MultipleNeuronWholeImageResponseParameters * WholeImageResponseParameters * ImageDataset & 'image_id%500=0'
    
    def make(self,key):
        n_entry_per_make = 500
        image_ids = np.array((ImageDataset & {'dataset_params': key['dataset_params']} & \
                            'image_id>={} and image_id<{}'.format(key['image_id'],key['image_id']+n_entry_per_make)).fetch('image_id'),dtype=int)
        neuron_src_table = (MultipleNeuronWholeImageResponseParameters & key).fetch1('neuron_src_table')
        neuron_list = eval(neuron_src_table).fetch(dj.key, as_dict=True, order_by='group_id, neuron_id')
        whole_image_params = key['whole_image_params']
    
        assert len(WholeImageResponse & neuron_list & {'whole_image_params':whole_image_params}) == len(neuron_list), "There are not enough neurons in WholeImageResponse with respect to neurons from MultipleNeuronWholeImageResponseParameters"
    
        response = {i:[] for i in image_ids}
            
        for i in tqdm(neuron_list):
            neuron_response = (WholeImageResponse & i & {'whole_image_params':whole_image_params}).fetch1('response')
            for j in response.keys():
                response[j].append(np.array(neuron_response[j]))
        
        for i in response.keys():
            temp = dict(key)
            temp['image_id'] = i
            self.insert1({**temp,'response':np.stack(response[i],axis=0)})

@schema
class SegmentationDataloaderParameter(dj.Lookup):
    definition = """
    dataloader_params: int
    ---
    params: longblob 
    """
    contents = [(1,{'image_boundary_params':1,'multiple_neuron_params':1,'dataset_params':3,'whole_image_params':4,'train_ratio':0.8,
                    'batch_size':32,'seed':0}),
               (2,{'image_boundary_params':1,'multiple_neuron_params':2,'dataset_params':3,'whole_image_params':4,'train_ratio':0.8,
                    'batch_size':32,'seed':0}),
               (3,{'image_boundary_params':1,'multiple_neuron_params':2,'dataset_params':1,'whole_image_params':2,'train_ratio':0.8,
                    'batch_size':32,'seed':0}),
               (4,{'image_boundary_params':2,'multiple_neuron_params':2,'dataset_params':3,'whole_image_params':4,'train_ratio':0.8,
                    'batch_size':32,'seed':0}),
               (5,{'image_boundary_params':2,'multiple_neuron_params':2,'dataset_params':1,'whole_image_params':2,'train_ratio':0.8,
                    'batch_size':32,'seed':0}),
               (6, {'image_boundary_params':2,'multiple_neuron_params':4,'dataset_params':4,'whole_image_params':5,'train_ratio':0.8, 
                    'batch_size':32,'seed':0})]

@schema
class SegmentationTrainingParameters(dj.Lookup):
    definition = """
    segmentation_training_params: int
    ---
    params: longblob
    """
    contents = [(1,{'optimizer_name':'SGD','lr':1e-2,'l1_lambda':1e1,'n_epoch':200,'weight_scale':20.0,'saving_frequency':1}),
               (2,{'optimizer_name':'SGD','lr':1e-2,'l1_lambda':0.0,'n_epoch':200,'weight_scale':20.0,'saving_frequency':1}),
               (3,{'optimizer_name':'SGD','lr':1e-3,'l1_lambda':0.0,'n_epoch':200,'weight_scale':20.0,'saving_frequency':1}),
               (4,{'optimizer_name':'SGD','lr':1e-4,'l1_lambda':0.0,'n_epoch':200,'weight_scale':20.0,'saving_frequency':1}),
               (5,{'optimizer_name':'SGD','lr':1e-1,'l1_lambda':0.0,'n_epoch':200,'weight_scale':20.0,'saving_frequency':1}),
               (6,{'optimizer_name':'SGD','lr':1e0,'l1_lambda':0.0,'n_epoch':200,'weight_scale':20.0,'saving_frequency':1})]

@schema
class SegmentationModelParameters(dj.Lookup):
    definition = """
    segmentation_model_params: int
    ---
    params: longblob
    """
    contents = [(1,{'in_channels':1000,'use_affine':False,'weight_boundary':(0,None),
                    'shift_boundary':(-0.2,0.2),'seed':0}),
               (2,{'in_channels':1200,'use_affine':False,'weight_boundary':(0,None),
                    'shift_boundary':(-0.2,0.2),'seed':0})]
    
class NeuronConvolutionDataset(Dataset):
    def __init__(self, params,phase='train'):
        
        import random
        self.params = params
        self.phase = phase
        random.seed(self.params['seed'])
        all_keys = self.get_matched_keys()
        n_train = int(np.round(len(all_keys) * self.params['train_ratio']))
        random.shuffle(all_keys)
        self.all_keys = {'train':all_keys[:n_train],'test':all_keys[n_train:]}
        
    def get_matched_keys(self):
        return (MultipleNeuronWholeImageResponse * ImageBoundary & self.params).fetch('KEY')
     
    
    def __len__(self):
        return len(self.all_keys[self.phase])

    def __getitem__(self, idx):
        single_key = self.all_keys[self.phase][idx]
        image = np.array((MultipleNeuronWholeImageResponse & single_key).fetch1('response'),dtype=np.float32)
        label = np.array((ImageBoundary & single_key).fetch1('boundary'),dtype=np.float32)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return image,label
    
def create_dataloader(key):
    params = (SegmentationDataloaderParameter & key).fetch1('params')
    all_dl = {}
    for phase in ['train','test']:
        train_data = NeuronConvolutionDataset(params,phase=phase)
        all_dl[phase] = DataLoader(train_data, batch_size=params['batch_size'])
    return all_dl


# Output has form of either n * 2 * h * w or n * 1 * h * w
# Target has form of n * h * w or n * 1 * h * w
def custom_loss(output, target,device,weight_scale=20.0):
    # Penalize for missing out heavier
    target = target.view(len(target),-1)
    # Use only the first channel as probability of boundary
    output = output.view(len(output),-1)
    
    averaged_loss = 0.0
    # Loop through each image because weight of loss is dependent on the target
    for i,j in zip(target,output):
        weight = 1.0 + abs(i)*weight_scale
        loss_f = nn.BCELoss(weight=weight).to(device)
        averaged_loss += loss_f(j,i)
    averaged_loss = averaged_loss/len(output)
    return averaged_loss


class AffineTransform(nn.Module):
    def __init__(self, transX=0.0, transY=0.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([[1, 0, transX],[0, 1, transY]]))
        
    def forward(self, x):
        
        thetas = self.theta.expand(len(x),-1,-1)
        grid = F.affine_grid(thetas, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)
    # To avoid learning shearing or rotation
    def zero_out_shear_rotation(self):
        self.theta.grad[:2, :2] = 0.0
        
    def clip_shift(self,a_min=-1,a_max=1):
        self.theta[:,2].data = torch.clamp(self.theta[:,2].data,min=a_min,max=a_max)
        
# channel_used is a array of neuron index to train and the rest is masked out
class BoundaryDecoder(nn.Module):
    def __init__(self,params):
        super(BoundaryDecoder,self).__init__()
        self.params = params
        torch.manual_seed(self.params['seed'])
        self.in_channels = self.params['in_channels']
        self.use_affine = self.params['use_affine']
        self.weight_min, self.weight_max = self.params['weight_boundary']
        self.shift_min, self.shift_max = self.params['shift_boundary']
        self.channel_used = self.params['channel_used']
        if self.params['use_affine']:
            self.affine_transforms = nn.ModuleList([AffineTransform() for i in range(in_channels)])
        
        self.channel_weight = nn.Parameter(torch.ones(self.in_channels,dtype=torch.float32))
        self.channel_bias = nn.Parameter(torch.ones(self.in_channels,dtype=torch.float32)*(-0.4))
        if self.channel_used is not None:
            self.channel_masked = np.array(list(set(range(self.in_channels)) - set(self.channel_used)))
            self.channel_weight.data[self.channel_masked] = 0.0
            self.channel_bias.data[self.channel_masked] = 0.0
        
    def forward(self,x):
        # Affine transform each channel independently
        if self.use_affine:
            transformed_x = []
            for i,f in enumerate(self.affine_transforms):
                transformed_x.append(f(x[:,i,...].unsqueeze(1)))
            x = torch.cat(transformed_x,axis=1)
        #x = self.batch_norm(x)
        x = self.channel_weight[None,:,None,None] * x + self.channel_bias[None,:,None,None]
        x = torch.sum(x,axis=1)
        x = torch.sigmoid(x)
        return x
    
    def get_weight(self):
        if self.channel_used is not None:
            return self.channel_weight[self.channel_used]
        else:
            return self.channel_weight
    
    def clip_weight(self):
        self.channel_weight.data = torch.clamp(self.channel_weight.data,min=self.weight_min,max=self.weight_max)
        if self.use_affine:
            for f in self.affine_transforms:
                f.clip_shift(a_min=self.shift_min,a_max=self.shift_max)
            
    def zero_out_frozen_paramters_grad(self):
        if self.use_affine:
            for f in self.affine_transforms:
                f.zero_out_shear_rotation()
        if self.channel_used is not None:
            self.channel_weight.grad[self.channel_masked] = 0.0
            self.channel_bias.grad[self.channel_masked] = 0.0

            
def p_to_binary(x,threshold=0.5):
    if torch.is_tensor(x):
        x = x.detach()
    elif isinstance(x,np.ndarray):
        x = np.array(x)
    else:
        raise ValueError('x is not tensor or array')
    x[x<=threshold] = 0.0
    x[x>threshold] = 1.0
    if torch.is_tensor(x):
        return x.int()
    else:
        return x.astype(int)
    
def train_model(model,dataloaders,params,training_save_path,device='cuda'):
    import torch.optim as optim
    import os
    
    if params['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    elif params['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    else:
        raise ValueError('Optimizer name {} is not specified'.format(params['optimizer_name']))
        
    # training_save_path have 'history', 'model_state_dict','optimizer_state_dict'
    if os.path.exists(training_save_path):
        with open(training_save_path,'rb') as handle:
            temp = pickle.load(handle)
        history = temp['history']
        model_state_dicts = temp['model_state_dict']
        optimizer_state_dicts = temp['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state_dicts[-1])
        model.load_state_dict(model_state_dicts[-1])
        current_epoch = len(history['test']['loss'])
    else:
        history = {'train':{'accuracy':[],'loss':[],'l1':[]},'test':{'accuracy':[],'loss':[],'l1':[]}}
        model_state_dicts,optimizer_state_dicts = [],[]
        current_epoch = 0
        
    model = model.to(device)
    optimizer.zero_grad()
    
    for epoch in tqdm(range(current_epoch,params['n_epoch'])): 
        for phase in ['train','test']:
            stat = {'accuracy':[],'loss':[],'l1':[]}
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for i, data in tqdm(enumerate(dataloaders[phase], 0)):
                inputs, labels = data
                inputs, labels = inputs.to(device),labels.to(device)

                outputs = model(inputs)
                loss = custom_loss(outputs, labels,device,weight_scale=params['weight_scale'])
                l1_loss = abs(model.get_weight()).mean()
                loss_combined = loss + params['l1_lambda'] * l1_loss
                
                stat['loss'].append(loss.item())
                stat['l1'].append(l1_loss.item())
            
                if phase == 'train':
                    loss_combined.backward()
                    model.zero_out_frozen_paramters_grad()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.clip_weight()
                with torch.no_grad():
                    accuracy = (p_to_binary(outputs.data) == labels.int()).sum()/outputs.numel()
                stat['accuracy'].append(accuracy.item())
                
            for i in stat.keys():
                history[phase][i].append(np.mean(stat[i]))
        if epoch%params['saving_frequency']==0:
            model_state_dicts.append(model.state_dict())
            optimizer_state_dicts.append(optimizer.state_dict())
            
            # Save in training_save_path
            result = {'history':history,'model_state_dict':model_state_dicts,'optimizer_state_dict':optimizer_state_dicts}
            with open(training_save_path, 'wb') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    return result

@files('static')
@schema
class SegmentationModel(dj.Computed):
    definition = """
    -> SegmentationDataloaderParameter
    -> SegmentationTrainingParameters
    -> SegmentationModelParameters
    -> MultipleNeuronWholeImageResponseList
    ---
    training_save_path: varchar(244) 
    """
    @property 
    def key_source(self):
        return SegmentationDataloaderParameter.proj(params_1='params') *  SegmentationTrainingParameters.proj(params_2='params') * SegmentationModelParameters.proj(params_3='params') * MultipleNeuronWholeImageResponseList
    
    def make(self,key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        training_params = (SegmentationTrainingParameters & key).fetch1('params')
        model_params = (SegmentationModelParameters & key).fetch1('params')
        channel_used = np.array((MultipleNeuronWholeImageResponseList & key).fetch1('neuron_list_idx'),dtype=int)
        model_params['channel_used'] = channel_used
        # Instanciate dataloaders, model
        dataloaders = create_dataloader(key)
        model = BoundaryDecoder(model_params)
        training_save_path = os.path.join(self.tuple_dir(key, create=True), "training_save_path.pickle")
        self.insert1({**key,'training_save_path':training_save_path})
        _ = train_model(model,dataloaders,training_params,training_save_path,device=device)
    
    def load_save_file(self,key):
        save_file_path = (self & key).fetch1('training_save_path')
        with open(save_file_path,'rb') as handle:
            temp = pickle.load(handle)
        return temp
    
    def load_model(self,key,model_idx=-1):
        model_params = (SegmentationModelParameters & key).fetch1('params')
        channel_used = np.array((MultipleNeuronWholeImageResponseList & key).fetch1('neuron_list_idx'),dtype=int)
        model_params['channel_used'] = channel_used
        model = BoundaryDecoder(model_params)
        state_dict = self.load_save_file(key)['model_state_dict'][model_idx]
        # try:
        #     state_dict = {k: torch.as_tensor(state_dict[k][0].copy()) for k in state_dict.dtype.names}
        # except AttributeError:
        #     state_dict = {k: torch.as_tensor(state_dict[k].copy()) for k in state_dict.keys()}
        # mod_state_dict = model.state_dict()
        # for k in set(mod_state_dict) - set(state_dict):
        #     log.warning('Could not find paramater {} setting to initialization value'.format(repr(k)))
        #     state_dict[k] = mod_state_dict[k]
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def load_history(self,key):
        return self.load_save_file(key)['history']
    
    def get_prediction_sample(self,key,model_idx=-1,device='cuda'):
        model = self.load_model(key,model_idx).to(device)
        dataloaders = create_dataloader(key)
        result = {}
        for phase in ['train','test']:
            x,y = next(iter(dataloaders[phase]))
            with torch.no_grad():
                x = x.to(device)
                y_pred = model(x).detach().cpu().numpy()
            result[phase] = {'x':x.detach().cpu().numpy(),'y':y.numpy(),'y_pred':y_pred}
        return result
# @files('static')
# @schema
# class MultipleNeuronWholeImageResponse(dj.Computed):
#     definition = """
#     -> MultipleNeuronWholeImageResponseParameters
#     -> WholeImageResponseParameters
#     -> ImageDataset
#     ---
#     response: blob@static
#     response_path: varchar(255)
#     """
#     @property
#     def key_source(self):
#         # Not entirely correct since WholeImageResponseParamters contain information about what ImageDataset to use but they have same number of images regardless
#         # Restriction for keys of interested is: MultipleNeuronWholeImageResponseParameters * WholeImageResponseParameters * ImageDataset & {'dataset_params':3,'whole_image_params':4}
#         return MultipleNeuronWholeImageResponseParameters * WholeImageResponseParameters * ImageDataset & 'whole_image_params=4' & 'dataset_params=3'
    
#     def make(self,key):
#         saving_frequency = 50
#         neuron_src_table = (MultipleNeuronWholeImageResponseParameters & key).fetch1('neuron_src_table')
#         neuron_list = eval(neuron_src_table).fetch(dj.key, as_dict=True, order_by='group_id, neuron_id')
#         whole_image_params = key['whole_image_params']
#         image_id = key['image_id']
#         response_path = os.path.join(self.tuple_dir(key, create=True), "response.pickle")

#         assert len(WholeImageResponse & neuron_list & {'whole_image_params':whole_image_params}) == len(neuron_list), "There are not enough neurons in WholeImageResponse with respect to neurons from MultipleNeuronWholeImageResponseParameters"
        
#         if os.path.exists(response_path):
#             try:
#                 with open(response_path,'rb') as handle:
#                     response = pickle.load(handle)
#                     starting_idx = len(response)
#             except (pickle.UnpicklingError, ValueError, EOFError) as error:
#                 logger.info(error)
#                 response = []
#                 starting_idx = 0
#                 with open(response_path,'wb') as handle:
#                     pickle.dump(response,handle,protocol=pickle.HIGHEST_PROTOCOL)
#         else:
#             response = []
#             starting_idx = 0
#             with open(response_path,'wb') as handle:
#                     pickle.dump(response,handle,protocol=pickle.HIGHEST_PROTOCOL)

#         # To avoid memory issue load one response at a time
#         for counter,i in tqdm(enumerate(neuron_list[starting_idx:500])):
#             response.append(np.array((WholeImageResponse & i & {'whole_image_params':whole_image_params}).fetch1('response')[image_id]))
#             # response.append((WholeImageResponse & i & {'whole_image_params':whole_image_params}).fetch1('response')[image_id])
#             if counter%saving_frequency==0:
#                 with open(response_path,'wb') as handle:
#                     pickle.dump(response,handle,protocol=pickle.HIGHEST_PROTOCOL)
    
#         if not isinstance(response, np.ndarray):
#             response = np.stack(response,axis=0)
            
#         with open(response_path, 'wb') as handle:
#             pickle.dump(response, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
#         self.insert1({**key,'response':response,'response_path':response_path})
def find_cross(images,upsize_factor=4, mask=None):
    from tqdm import tqdm
    import cv2
    # Multiple images
    if len(images.shape)>2:
        new_images = []
        crosses = []
        for i in tqdm(images):
            j,k = find_cross(i, upsize_factor, mask)
            new_images.append(k)
            crosses.append(j)
        new_images = np.stack(new_images)
        crosses = np.stack(crosses)
        return crosses,new_images
    else:
        images = cv2.resize(images,(images.shape[1]*upsize_factor,images.shape[0]*upsize_factor))
        cross = np.logical_or(find_cross_helper(images,True),find_cross_helper(images,False))
        if mask is not None:
            cross = cross.astype(int) * mask
        return cross,images
        
def find_cross_helper(image,positive = True,std_thres_percentile=90):
    image = image - image.mean()    
    h,w = image.shape
    label = np.zeros_like(image)
    stds = np.zeros_like(image)
    for i in range(1,h):
        for j in range(1,w):
            stds[i,j] = image[i-1:i+1,j-1:j+1].ravel().std()
            if positive and image[i,j]>0.0 and len(np.where(image[i-1:i+1,j-1:j+1]<0.0)[0])>0:
                label[i,j] = 1.0
            elif not positive and image[i,j]<0.0 and len(np.where(image[i-1:i+1,j-1:j+1]>0.0)[0])>0:
                label[i,j] = 1.0
    std_thres = np.percentile(stds.ravel(),std_thres_percentile)
    label[stds<std_thres] = 0.0
    return label

def find_furthest_mask(v_mask,mei_mask,overlap_fraction=0.8):
    def translate_mask(mask,shift_x=0,shift_y=0):
        return np.roll(np.roll(mask, shift_x, axis=0),shift_y,axis=1)
    def still_within(new_small_mask,original_small_mask,big_mask,overlap_fraction):
        return (new_small_mask * big_mask).sum() >= overlap_fraction * (original_small_mask * big_mask).sum()
    
    furthest_mask = None
    dist = 0
    for shift_x in range(-25,25):
        for shift_y in range(-25,25):
            temp = translate_mask(v_mask,shift_x,shift_y)
            new_dist = np.sqrt(shift_x**2+shift_y**2)
            if still_within(temp,v_mask,mei_mask,overlap_fraction) and new_dist>=dist:
                dist = new_dist
                furthest_mask = temp
                chosen_shift_x = shift_x
                chosen_shift_y = shift_y
    return {'original_v_mask':v_mask,'new_furthest_v_mask':furthest_mask,'shift_x':chosen_shift_x,'shift_y':chosen_shift_y}

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

@schema
class DatasetLabelParameters(dj.Lookup):
    definition = """
    label_params: int
    ---
    transform_height: int  # Height of the resized image
    transform_width: int   # Width of the resized image
    dataset_name: varchar(16)
    """
    contents = [(1, 64, 64, 'cub')]
    
@schema
class CropParameters(dj.Lookup):
    definition = """
    crop_params: int
    ---
    crop_height: int
    crop_width: int
    crop_stride: int            # Stride to generate crop
    batch_size: int             # Batch size of images to process
    mask_image: bool            # for image standardization: whether to mask the crop when passing to the model
    match_stats: varchar(16)    # for image standardization: method for matching statistics on the image ('mask' or 'ff')
    mask_mean_subtraction: bool # for image standardization: whether to subtract mask mean 
    """
    contents = [(1, 36, 36, 8, 128, 1, 'ff', 1)]

@schema
class LabelCrops(dj.Computed):
    definition = """
    -> DatasetLabelParameters
    -> CropParameters
    crop_id:              int
    ---
    image_idx:            int
    x:                    int   # Height coordinate of the crop
    y:                    int   # Width coordinate of the crop
    fg_ratio_full:        float # size ratio of foreground in the full crop
    fg_ratio_center:      float # size ratio of foreground in the center circle representative of average MEI mask
    """
    @property
    def key_source(self):
        return DatasetLabelParameters * CropParameters
    
    def make(self, key):
        params = (DatasetLabelParameters * CropParameters & key).fetch1()
        center_mask = create_circular_mask(36, 36, radius = 7.75).astype(np.float32)
        # Get a batch of images
        all_dl = get_dataloader(params)
        iterator = iter(all_dl)
        all_dics = []
        for i, (imgs, labels) in enumerate(tqdm(iterator)):
            crop_results = crop_img(imgs,labels,params)
            for j, result in enumerate(crop_results):
                all_dics.extend([dict(image_idx = i * params['batch_size'] + j, x = x, y = y, 
                                      fg_ratio_full = (m == 1).sum() / np.ones_like(m).sum(),
                                      fg_ratio_center = ((m * center_mask) == 1).sum() / center_mask.sum()) 
                                for x, y, m in zip(result['x'], result['y'], result['mask'])])
        self.insert([dict(crop_id=i, **key, **dic) for i, dic in enumerate(all_dics)], ignore_extra_fields=True)

@schema
class CropSet(dj.Lookup):
    definition = """
    set_id:       int
    ---
    n_crops:         int 
    seed:            int          # seed for randomly selection among qualified crops
    query:           varchar(128) # query for restricting LabelCrops to get qualified crops
    description:     varchar(256) # description of selection criteria
    """
    
    class Crops(dj.Part):
        definition = """
        -> master
        -> LabelCrops
        """

    def fill(self, set_id=1, n_crops=500, seed=1e6, query='foreground_ratio = 0', description=None):
        keys = (LabelCrops & query).fetch(dj.key, order_by='crop_id')
        np.random.seed(int(seed + set_id))
        selected = np.random.choice(np.array(keys), n_crops, replace=False)
        self.insert1(dict(set_id=set_id, n_crops=n_crops, seed=seed, query=query, description=description))
        self.Crops.insert([dict(set_id=set_id, **key) for key in selected])

@schema
class CropSetResponses(dj.Computed):
    definition = """
    -> base.MEIMask
    -> CropParameters
    -> CropSet
    -> DatasetParameters
    ---
    responses:   longblob     # a vector of number of crops in CropSet
    """
    
    @property
    def key_source(self):
        neuron_rel = base.MEIMask & (deis_schema.TextureGoodRun & 'score_params = 3')
        params_rel = CropParameters * CropSet * DatasetParameters & ImageDataset & 'dataset_params = 4'
        return neuron_rel.proj() * params_rel.proj()
    
    def make(self, key):
        # load parameters
        mask, mask_x, mask_y = (base.MEIMask & key).fetch1('mask', 'mask_x', 'mask_y')
        mei_params = (base.MEIParameters & key).fetch1()
        crop_params = (CropParameters & key).fetch1()
        if crop_params['match_stats'] == 'ff':
            target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
        else:
            raise ValueError()
            
        # load crops
        crops = []
        images, xs, ys = (ImageDataset * (CropSet.Crops * LabelCrops).proj('x', 'y', image_id='image_idx') & key).fetch('image', 'x', 'y', order_by='crop_id')
        for im, x, y in zip(images, xs, ys):
            crop = im[x:x+crop_params['crop_height'], y:y+crop_params['crop_width']]
            crop = ops.create_whole_mei(crop, mask, mask_x, mask_y, normalize_crop = False)
            crop = ops.standardize_image(crop, target_mean, target_std, mask, 
                                         crop_params['mask_mean_subtraction'], crop_params['mask_image'], crop_params['match_stats'])
            crops.append(crop)
            
        # load predictive model and get responses
        model = deis_schema.load_model(key)
        responses = pass_img(crops, model, 'cuda')
        
        self.insert1(dict(**key, responses = responses))
        
@schema
class MostExcitingCropParameters(dj.Lookup):
    definition = """
    most_exciting_crop_params: int
    ---
    crop_height: int
    crop_width: int
    crop_stride: int       # Stride to generate crop
    batch_size: int        # Batch size of images to process
    n_target_images: int   # Number of crops to search
    n_keep : int           # Number of crops to keep
    mask_image: bool       # Whether to mask the image
    match_stats: varchar(16) # Method for matching statistics on the image ('mask' or 'ff')
    mask_mean_subtraction: bool # Subtract mask mean
    balanced="balanced": varchar(64)                     
    hetero_bounds=null   : varchar(16)                   
    homo_bounds=null     : varchar(16)
    mask_threshold=0     : float
    """
    
    contents = [(1, 36, 36, 2, 128, 1000000, 100, 1, 'ff', 1, 'balanced', '[0.3, 0.7]', '[0.95, 1.0]', 0),
                (2, 36, 36, 2, 128, 1000000, 100, 1, 'ff', 1, 'balanced', '[0.2, 0.8]', '[0.8, 1.0]', 0),
                (3, 36, 36, 2, 128, 1000000, 100, 1, 'ff', 1, 'balanced', '[0.3, 0.7]', '[0.95, 1.0]', 0.3),
                (4, 36, 36, 2, 128, 1000000, 100, 1, 'ff', 1, 'balanced', '[0.2, 0.8]', '[0.8, 1.0]', 0.3),
                (5, 36, 36, 2, 128, 1000000, 100, 1, 'ff', 1, 'random', '[None, None]', '[None, None]', 0.3),
                (6, 36, 36, 2, 128, 1000000, 100, 1, 'ff', 1, 'heter_only', '[0.2, 0.8]', '[0.8, 1.0]', 0.3),
                (7, 36, 36, 2, 128, 1000000, 100, 1, 'ff', 1, 'homo_only', '[0.2, 0.8]', '[0.8, 1.0]', 0.3)]

def get_patch_type(crop_label,mask_float,mask_x,mask_y,homo_lower,homo_upper,
                  heter_lower,heter_upper):
    if len(crop_label.shape) == 2:
        masked_label = deis_schema.create_whole_mei(crop_label, mask_float, mask_x, mask_y,(36, 64),
                                                    normalize_crop = False)
        masked_label = masked_label * mask_float
        # Calculate the positive part first
        sizes = masked_label[masked_label>0.0].sum()/(mask_float.sum()+1e-9)
        sizes = np.array([sizes, 1-sizes])
        if ((sizes>=heter_lower) & (sizes<=heter_upper)).any():
            return 'heter'
        elif ((sizes>=homo_lower) & (sizes<=homo_upper)).any():
            return 'homo'
        else:
            return 'invalid'
    elif len(crop_label.shape)==3:
        masked_label = deis_schema.create_whole_mei(crop_label, mask_float, mask_x, mask_y,(len(crop_label),36, 64),
                                                    normalize_crop = False)
        masked_label = masked_label * mask_float[None,...]
        # Calculate the positive part first
        sizes = (masked_label*(masked_label>0.0)).sum(axis=(-1,-2))/(mask_float.sum()+1e-9)
        sizes = np.stack([sizes,1-sizes],axis=0).T
        result = np.full((len(sizes)), ['invalid'],dtype=np.dtype('<U5'))
        result[((sizes>=heter_lower) & (sizes<=heter_upper)).any(axis=1)] = 'heter'
        result[((sizes>=homo_lower) & (sizes<=homo_upper)).any(axis=1)] = 'homo'
        return result
    else:
        raise ValueError('Number of dimension of input can only be 2 or 3')

def binarize_mask(mask,threshold=0.0):
    mask = np.array(mask)
    mask[mask<threshold] = 0.0
    mask[mask>threshold] = 1.0
    return mask

def crop_img_v2(images,masks,image_ids,most_exciting_crop_params):
    from itertools import product
    crop_h, crop_w, crop_stride = [most_exciting_crop_params[i] for i in ['crop_height', 'crop_width', 'crop_stride']]
    IM_SIZE = images.shape[-2:]
    images = images.detach().cpu().numpy().squeeze() if torch.is_tensor(images) else images
    masks = masks.detach().cpu().numpy().squeeze() if torch.is_tensor(masks) else masks
    
    results = {'crop':[],'mask':[]}
    xs,ys = np.arange(0, IM_SIZE[0] - crop_h, crop_stride),np.arange(0, IM_SIZE[1] - crop_w, crop_stride)
    for (h, w) in product(xs,ys):
        results['crop'].append(images[...,h:h+crop_h, w:w+crop_w])
        results['mask'].append(masks[...,h:h+crop_h, w:w+crop_w])
        
    results['x'] = np.repeat(np.meshgrid(xs,ys)[0].T,len(images))
    results['y'] = np.repeat(np.meshgrid(xs,ys)[1].T,len(images))
    results['image_id'] = np.repeat(image_ids[None,:],len(xs)*len(ys),axis=0).ravel()
    for i in ['crop','mask']:
        results[i] = np.concatenate(results[i],axis=0)
    return results

# Standardize based per neuron
def standardize_crops_v2(crops,most_exciting_crop_params,key):
    mask_float, mask_x, mask_y = (base.MEIMask & key).fetch1('mask', 'mask_x', 'mask_y')
    mei_params = (base.MEIParameters & key).fetch1()
    crops = ops.create_whole_mei(crops, mask_float, mask_x, mask_y, (len(crops), 36, 64), normalize_crop = False)
    if most_exciting_crop_params['match_stats'] == 'ff':
        target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
    else:
        raise ValueError()
    crops = ops.standardize_image(crops, target_mean, target_std, mask_float,most_exciting_crop_params['mask_mean_subtraction'],
                                   most_exciting_crop_params['mask_image'],most_exciting_crop_params['match_stats'])
    return crops


@schema
class MostExcitingCrop(dj.Computed):
    definition = """ # For multiple neurons
    -> base.MEIMask
    -> MostExcitingCropParameters
    -> DatasetParameters
    ---
    image_id:          blob
    x:                 blob
    y:                 blob
    crop_masked:       blob@static # Crop standardized by the neuron mask
    crop_label:        blob@static # Crop label to make it convenient
    act:               blob
    n_searched:         int
    heter_searched:     int
    homo_searched:      int
    """
    @property
    def key_source(self):
        return base.MEIMask * MostExcitingCropParameters * DatasetParameters

    def make(self,key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        most_exciting_crop_params = (MostExcitingCropParameters() & key).fetch1()
        dataset_params = (DatasetParameters & key).fetch1()

        mei_params = (base.MEIParameters & key).fetch1()

        # Get model
        model_key = ({'group_id': key['group_id'], 'net_hash': key['net_hash']} if
                     mei_params['use_avg_model'] else key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=key['neuron_id'], device=device, average_batch=False)

        # Since we can match the fetching from ImageDataset, make won't use ImageDataset but assume the id match up
        # To do this trick without much complication, no shuffling will be done
        all_dl = get_dataloader({**most_exciting_crop_params,**dataset_params})
        iterator = iter(all_dl)
        image_id_offset = 0

        selected_results = []
        counter,min_act,heter_counter,homo_counter = 0,0,0,0
        max_heter = max_homo = None
        # current_image_batch,n_image_batch = 0, len(image_keys)
        balanced,n_target_images,hetero_bounds,homo_bounds,mask_threshold = [most_exciting_crop_params[i] for i in ['balanced','n_target_images','hetero_bounds','homo_bounds','mask_threshold']]

        if balanced == 'heter_only':
            max_heter = n_target_images
            max_homo = 0
        elif balanced == 'homo_only':
            max_heter = 0
            max_homo = n_target_images
        elif balanced == 'balanced':
            max_heter = n_target_images//2
            max_homo = n_target_images//2

        # Deal with balancing between number of homo and heter patches
        if not balanced=='random':
            heter_lower,heter_upper = float(hetero_bounds.split(',')[0][1:]),float(hetero_bounds.split(',')[1][:-1])
            homo_lower,homo_upper = float(homo_bounds.split(',')[0][1:]),float(homo_bounds.split(',')[1][:-1])
            mask_float,mask_x,mask_y = (base.MEIMask & key).fetch1('mask','mask_x','mask_y')

            if mask_threshold > 0.0:
                mask_float = binarize_mask(mask_float,mask_threshold)

        def continue_search_condition(balanced,counter,n_target_images,heter_counter,homo_counter,max_heter=None,max_homo=None):
            if balanced=='random':
                if counter < n_target_images:
                    return True
                else:
                    return False
            else:
                if heter_counter < max_heter or homo_counter < max_homo:
                    return True
                else:
                    return False

        while continue_search_condition(balanced,counter,n_target_images,heter_counter,homo_counter,max_heter,max_homo):
            try:
                images,masks = next(iterator)
                if (len(images) == 0)  or (len(masks)==0):
                    break
            except:
                break
            images, masks = images.detach().cpu().numpy().squeeze(), masks.detach().cpu().numpy().squeeze()
            image_ids = np.arange(len(images)) + image_id_offset
            image_id_offset += len(images)

            # Assume here that all images have the same size for convenience
            crop_results = crop_img_v2(images,masks,image_ids,most_exciting_crop_params)

            # Create filter
            if not balanced=='random':
                patch_types = get_patch_type(crop_results['mask'],mask_float,mask_x,mask_y,homo_lower,homo_upper,heter_lower,heter_upper)
                n_heter_needed = np.clip(max_heter - heter_counter,a_min=0,a_max=len(patch_types))
                n_homo_needed = np.clip(max_homo - homo_counter,a_min=0,a_max=len(patch_types))
                heter_idx,homo_idx = [np.where(patch_types==i)[0][:j] for i,j in zip(['heter','homo'],[n_heter_needed,n_homo_needed])]
                idxs = np.concatenate([heter_idx,homo_idx])
                heter_counter += len(heter_idx)
                homo_counter += len(homo_idx)

            else:
                # If not balanced still need to restrict to the number of crops accepted generally
                n_processed = np.clip(n_target_images - counter,a_min=0,a_max=len(crop_results['crop']))
                idxs = np.random.choice(np.arange(len(crop_results['crop'])),n_processed,replace=False)

            if len(idxs)>0:
                # Filter out imgs, acts, x, y, image_id
                for i in crop_results.keys():
                    crop_results[i] = crop_results[i][idxs]

                crop_results['crop'] = standardize_crops_v2(crop_results['crop'],most_exciting_crop_params,key)
                crop_results['act'] = pass_img(crop_results['crop'],model,device,batch_size=32)
                counter += len(crop_results['act'])

                # Only select if the act is higher than minimum
                idxs = np.where(crop_results['act']>min_act)[0]

            if len(idxs)>0:
                new_results = [{j:crop_results[j][i] for j in crop_results.keys()} for i in idxs]
                combined_results = selected_results + new_results
                selected_results = sorted(combined_results,key=lambda x: x['act'],reverse=True)[:most_exciting_crop_params['n_keep']]
                min_act = selected_results[-1]['act']

        temp = {i:[] for i in ['crop','mask','image_id','x','y','act']}
        # Merge the result back
        for i in ['crop','mask','image_id','x','y','act']:
            temp[i] = [j[i] for j in selected_results]
            if i=='crop':
                temp[i] = np.stack(temp[i])
            else:
                temp[i] = np.array(temp[i])
        temp['crop_masked'] = temp['crop']
        del temp['crop']
        temp['crop_label'] = temp['mask']
        del temp['mask']
        temp = {**key,**temp,'heter_searched':heter_counter,'homo_searched':homo_counter,'n_searched':counter}
        self.insert1(temp)
        
# key should be from base.MEIMask
def get_patch_fraction(keys,most_exciting_crop_params=4,dataset_params=1):
    patch_types = []
    temp = (MostExcitingCropParameters & {'most_exciting_crop_params':most_exciting_crop_params}).fetch1()
    heter_lower,heter_upper = float(temp['hetero_bounds'].split(',')[0][1:]),float(temp['hetero_bounds'].split(',')[1][:-1])
    homo_lower,homo_upper = float(temp['homo_bounds'].split(',')[0][1:]),float(temp['homo_bounds'].split(',')[1][:-1])
    
    for key in tqdm(keys):
        mask_float,mask_x,mask_y = (base.MEIMask & key).fetch1('mask','mask_x','mask_y')
        if temp['mask_threshold'] > 0.0:
            mask_float = binarize_mask(mask_float,temp['mask_threshold'])
        
        crop_label = (MostExcitingCrop &  {'most_exciting_crop_params':most_exciting_crop_params,'dataset_params':dataset_params} & key).fetch1('crop_label')
        patch_type = get_patch_type(crop_label,mask_float,mask_x,mask_y,homo_lower,homo_upper,heter_lower,heter_upper)
        patch_types.append(patch_type)
    return np.stack(patch_types)


# Assume that mask_1 is a smooth mask or binary (0-1)
# Assume that mask_2 is composed of target strictly catergory -1, 1.
# This is done to avoid chance of double masking
# Return fraction of positive matching, negative matching, and 0 matching
def cal_match(mask_1,mask_2):
    combined_mask = mask_1 * mask_2
    mask_1_sum = mask_1.sum()
    overlap = combined_mask[combined_mask>0.0].sum()/(mask_1_sum+1e-9)
    overlap = np.array([overlap,1.0-overlap])
    return overlap

def find_center(mask,threshold=0):
    from skimage import morphology
    mask = binarize_mask(mask,threshold=threshold)
    hull = morphology.convex_hull_image(mask)
    px_y, px_x = (coords.mean() + 0.5 for coords in np.nonzero(hull))
    return int(np.round(px_y)),int(np.round(px_x))

# The magnitude of the v_mask doesn't influence the result of matching. However will return None if the new mask overlap less than 10% of the original v_mask 
def find_symmetrical_mask(v_mask,mei_mask,threshold=0.1):
    center = find_center(mei_mask)
    new_mask = np.zeros_like(v_mask)
    for i in range(new_mask.shape[-2]):
        for j in range(new_mask.shape[-1]):
            if v_mask[i,j]:
                new_i,new_j = center[-2]*2-i,center[-1]*2-j
                new_i,new_j = new_i%new_mask.shape[-2],new_j%new_mask.shape[-1]
                new_mask[new_i,new_j] = v_mask[i,j]
    if (new_mask * mei_mask).sum()<(v_mask*mei_mask).sum() * threshold:
        print((new_mask * mei_mask).sum(),(v_mask*mei_mask).sum())
        return None
    else:
        temp = new_mask * mei_mask
        temp *= v_mask.max()/temp.max()
        return temp

def find_furthest_mask(v_mask,mei_mask):
    def translate_mask(mask,shift_x=0,shift_y=0):
        return np.roll(np.roll(mask, shift_x, axis=0),shift_y,axis=1)
    def still_within(new_small_mask,original_small_mask,big_mask):
        return (new_small_mask *big_mask).sum()>(original_small_mask * big_mask).sum() * 0.95
    furthest_mask = None
    dist = 0
    for shift_x in range(-18,19):
        for shift_y in range(-18,19):
            temp = translate_mask(v_mask,shift_x,shift_y)
            new_dist = np.sqrt(shift_x**2+shift_y**2)
            if still_within(temp,v_mask,mei_mask) and new_dist>=dist:
                dist = new_dist
                furthest_mask = temp
                chosen_shift_x = shift_x
                chosen_shift_y = shift_y
    #return {'original_v_mask':v_mask,'new_furthest_v_mask':furthest_mask,'shift_x':chosen_shift_x,'shift_y':chosen_shift_y}
    return furthest_mask

# Todo: Check for p before do the matching and check to see if there are enough heter crops
def valid_cell_for_matching(key,rest='diverse_params in (14, 17) and texture_params = 20 and score_params=2',most_exciting_crop_params=4,dataset_params=1,p_thres=(0.05,0.95),heter_thres=(0.1,None),*args,**kwargs):
    variable_mask,p = (deis_schema.Texture * deis_schema.TextureGoodRun & rest & key).fetch1('variable_mask','p')
    if (p_thres[0] is not None) and (p<p_thres[0]):
        return False, None
    elif (p_thres[1] is not None) and (p>p_thres[1]):
        return False, None                        
    temp = (MostExcitingCropParameters & {'most_exciting_crop_params':most_exciting_crop_params}).fetch1()
    heter_lower,heter_upper = float(temp['hetero_bounds'].split(',')[0][1:]),float(temp['hetero_bounds'].split(',')[1][:-1])
    homo_lower,homo_upper = float(temp['homo_bounds'].split(',')[0][1:]),float(temp['homo_bounds'].split(',')[1][:-1])
    
    mask_float,mask_x,mask_y = (base.MEIMask & key).fetch1('mask','mask_x','mask_y')
    if temp['mask_threshold'] > 0.0:
        mask_float = binarize_mask(mask_float,temp['mask_threshold'])
        

    crop_label = (MostExcitingCrop &  {'most_exciting_crop_params':most_exciting_crop_params,'dataset_params':dataset_params} & key).fetch1('crop_label')
    patch_type = get_patch_type(crop_label,mask_float,mask_x,mask_y,homo_lower,homo_upper,heter_lower,heter_upper)
    crop_label = crop_label[patch_type=='heter',...]
    full_crop_label = deis_schema.create_whole_mei(crop_label, mask_float, mask_x, mask_y,(len(crop_label),36, 64),normalize_crop = False)
    
    if (heter_thres[0] is not None) and (len(full_crop_label) < heter_thres[0]*len(patch_type)):
        return False, None
    elif (heter_thres[1] is not None) and (len(full_crop_label) > heter_thres[1]*len(patch_type)):
        return False, None
    else:
        return True, full_crop_label

# Need to run valid_cell_for_matching first to guarantee get_match can run so crop_labels is a default argument (a list of crop_labels)
# rest='diverse_params = 14 and texture_params = 13 and group_id <= 237 and score_params=2', 'diverse_params = 17 and texture_params = 20 and group_id > 237 and score_params=2' for close loop neurons
# rest = 'diverse_params in (14, 17) and texture_params = 20 and score_params=2' for 1200 neurons
# Order is [[v,foreground]/v,[v,background]/v,[f,foreground]/f,[f,background]/f]
def get_match_helper(key,full_crop_label,rest='diverse_params in (14, 17) and texture_params = 20 and score_params=2',most_exciting_crop_params=4,dataset_params=1,mask_mode='original'):
    temp = (MostExcitingCropParameters & {'most_exciting_crop_params':most_exciting_crop_params}).fetch1()

    mask_float,mask_x,mask_y = (base.MEIMask & key).fetch1('mask','mask_x','mask_y')
    variable_mask = (deis_schema.Texture * deis_schema.TextureGoodRun & rest & key).fetch1('variable_mask')
    if temp['mask_threshold'] > 0.0:
        mask_float = binarize_mask(mask_float,temp['mask_threshold'])
        variable_mask = binarize_mask(variable_mask,temp['mask_threshold'])

    if mask_mode == 'furthest':
        variable_mask = find_furthest_mask(variable_mask,mask_float)
    elif mask_mode == 'symmetrical':
        variable_mask = find_symmetrical_mask(variable_mask,mask_float)
    fixed_mask = mask_float-variable_mask
    # Clip to guarantee that fixed_mask is a valid mask
    fixed_mask = np.clip(fixed_mask,a_min=0.0,a_max=1.0)
    
    result = np.stack([np.stack(([cal_match(variable_mask,i),cal_match(fixed_mask,i)])) for i in full_crop_label])
    return result

def get_match(keys,rest='diverse_params in (14, 17) and texture_params = 20 and score_params=2',most_exciting_crop_params=4,
              dataset_params=1, mask_mode='original',p_thres=(0.05,0.95),heter_thres=(0.1,None)):
    
    full_crop_labels = [valid_cell_for_matching(key,rest=rest,most_exciting_crop_params=most_exciting_crop_params,
                                                dataset_params=dataset_params,p_thres=p_thres,heter_thres=heter_thres)[1] for key in keys]
    result, valid_keys = [],[]
    for key,full_crop_label in tqdm(zip(keys,full_crop_labels)):
        if full_crop_label is not None:
            valid_keys.append(key)
            result.append(get_match_helper(key,full_crop_label=full_crop_label,rest=rest,most_exciting_crop_params=most_exciting_crop_params,dataset_params=dataset_params,mask_mode=mask_mode))
    return result,valid_keys

def summarize_match_result(match_result,region='variable_match_foreground',operation='mean'):
    result = []
    for i in match_result:
        if region == 'variable_match_foreground':
            temp = i[:,0,0]
        elif region == 'variable_match_background':
            temp = i[:,0,1]
        elif region == 'fixed_match_foreground':
            temp = i[:,1,0]
        else:
            temp = i[:,1,1]
            
        if operation == 'mean':
            temp = np.mean(temp)
        elif operation == 'median':
            temp = np.median(temp)
        elif operation == 'std':
            temp = np.std(temp)
        elif operation == 'count':
            temp = len(temp)
        else:
            raise ValueError()
        result.append(temp)
    return np.array(result)

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise
# One number match in order
# variable_mask is variable or fixed mask
# label_mask contains 1,-1, and 0
def cal_match_one_number(variable_mask,mei_mask,label_mask):
    fixed_mask = mei_mask-variable_mask
    fixed_mask = np.clip(fixed_mask,a_min=0.0,a_max=1.0)
    combined_mask = variable_mask - fixed_mask
    combined_mask = combined_mask * label_mask
    return (combined_mask).sum()/abs(combined_mask).sum()

# @schema
# class TextureGoodRunReducedParameters(dj.Lookup):
#     definition = """
#     texture_good_run_reduced_params: int
#     ---
#     table_restriction: varchar(256)
#     """
#     contents = [(0,'(deis_schema.TextureGoodRun * deis_schema.NeuronSetRequest & \'diverse_params in (14, 17) and texture_params = 20 and score_params=2 and method_id=3\')')]
    
# @schema
# class TextureGoodRunReduced(dj.Computed):
#     definition = """
#     -> TextureGoodRunReducedParameters
#     key_hash: varchar(256)
#     ---
#     key: longblob
#     """
#     @property
#     def key_source(self):
#         return TextureGoodRunReducedParameters
#     def make(self,key):
#         texture_good_run_reduced_params = (TextureGoodRunReducedParameters & key).fetch1()
#         temp = eval(texture_good_run_reduced_params['table_restriction'])
#         temps = temp.fetch('KEY',as_dict=True, order_by='group_id, neuron_id')
#         for i,j in enumerate(temps):
#             temp = {'texture_good_run_reduced_params':texture_good_run_reduced_params['texture_good_run_reduced_params'],'key_hash':static_utils.key_hash(j),'key':j}
#             self.insert1(temp)
  
# Look-up table for variable mask
@schema
class BipartiteMaskParameters(dj.Lookup):
    definition = """
    bipartite_mask_params: int
    ---
    mask_type: varchar(64) #Mask type ('original','symmetrical','furthest')
    """
    contents = [(1,'original'),(2,'symmetrical'),(3,'furthest')]

@schema
class BipartiteMask(dj.Computed):
    definition = """
    -> deis_schema.TextureLookup
    -> BipartiteMaskParameters
    ---
    variable_mask: longblob
    fixed_mask: longblob
    """
    @property
    def key_source(self):
        rest = 'mei_params = 10 and mask_params = 3 and diverse_params in (14, 17) and texture_params = 20 and score_params = 2'
        neuron_rel = deis_schema.NeuronSetRequest & 'method_id = 3'
        all_keys = deis_schema.TextureLookup & (deis_schema.Texture * deis_schema.TextureGoodRun & rest & neuron_rel)
        return all_keys * BipartiteMaskParameters

    def make(self, key):
        mask_type = (BipartiteMaskParameters & key).fetch1()['mask_type']
        neuron_key = (deis_schema.TextureLookup & key).fetch1()
        variable_mask = (deis_schema.Texture & neuron_key).fetch1('variable_mask')
        mei_mask = (base.MEIMask & neuron_key).fetch1('mask')
        
        if mask_type == 'furthest':
            variable_mask = find_furthest_mask(variable_mask,mei_mask)
        elif mask_type == 'symmetrical':
            variable_mask = find_symmetrical_mask(variable_mask,mei_mask)
        fixed_mask = mei_mask - variable_mask
        fixed_mask = np.clip(fixed_mask,a_min=0.0,a_max=1.0)
        self.insert1({**key,'variable_mask':variable_mask,'fixed_mask':fixed_mask})
     
@schema 
class BipartiteFullFieldParameters(dj.Lookup):
    definition = """
    bipartite_full_field_params: int
    ---
    image_type: varchar(128)
    n_image: int
    height: int
    width: int
    standardized: bool
    save_location: varchar(128)
    params: longblob # Params to create noise
    """
    contents = [(1,'perlin_low_freq',1000000, 64,64,1,'/dj-stor01/datt/perlin_low_freq_1',{'res':(8,8),'octaves':4,'persistence':0.25,'seed':0}),
               (2,'perlin_low_freq',1000000, 64,64,1,'/dj-stor01/datt/perlin_low_freq_2',{'res':(8,8),'octaves':4,'persistence':0.25,'seed':1}),
               (3,'perlin_high_freq',1000000,64,64,1,'/dj-stor01/datt/perlin_high_freq_1',{'res':(8,8),'octaves':4,'persistence':1.0,'seed':2}),
               (4,'perlin_high_freq',1000000,64,64,1,'/dj-stor01/datt/perlin_high_freq_2',{'res':(8,8),'octaves':4,'persistence':1.0,'seed':3})]
    
    def populate_bipartite_full_field(key,make_new=False):
        bipartite_full_field_params = (BipartiteFullFieldParameters & key).fetch1()
        n_image,height,width,standardized,save_location,params = [bipartite_full_field_params[i] for i in ['n_image','height','width','standardized','save_location','params']]
        assert ((max(height,width) % (2**(params['octaves']-1)*params['res'][0]))==0), 'Violate Perlin noise condition'
        import shutil
        if make_new and os.path.exists(save_location):
            # This requires no file is read-only
            shutil.rmtree(save_location)
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        random.seed(params['seed'])
        for i in tqdm(range(n_image)):
            img = generate_fractal_noise_2d(shape=(max(height,width), max(height,width)), res=params['res'], octaves =params['octaves'],persistence=params['persistence'])
            if standardized:
                img = (img - img.mean()) /(img.std()+1e-9)
            img = np.clip(img,a_min=np.percentile(img.ravel(),5),a_max = np.percentile(img.ravel(),95))

            img = (img - img.min())/(img.max()-img.min()+1e-9)
            img = Image.fromarray(np.uint8(img*255),'L')
            file_name = '{}.png'.format(i)
            img.save(os.path.join(save_location,file_name))
        text = os.listdir(save_location)
        with open(os.path.join(save_location,'images.csv'),'w') as file:
            for line in text:
                file.write(line)
                file.write('\n')
    
class NoiseLoader(Dataset):
    def __init__(self, img_path, img_transform=None, loader=default_loader):
        self.img_transform = img_transform
        self.loader = default_loader
        self.img_path = img_path
        self._load_metadata()

    def _load_metadata(self):
        file_names = os.listdir(self.img_path)
        df = pd.DataFrame(file_names, columns= ['file_name'])
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx].file_name
        path = os.path.join(self.img_path,file_name)

        img = self.loader(path)
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img
    
def get_noise_dataloader(key):
    height,width,save_location = (BipartiteFullFieldParameters & key).fetch1('height','width','save_location')
    img_transform = transforms.Compose([transforms.Resize((height,width)),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
    all_data = NoiseLoader(img_path = save_location,img_transform = img_transform)
    all_dl = DataLoader(all_data, batch_size=32, drop_last=False)
    return all_dl

@schema
class MostExcitingBipartiteStimuliParameters(dj.Lookup):
    definition = """
    most_exciting_bipartite_stimuli_params: int
    ---
    image_mode: varchar(128) #String to describe image type
    n_target_images: int
    n_keep: int
    batch_size: int
    mask_image: bool       # Whether to mask the image
    match_stats: varchar(16) # Method for matching statistics on the image ('mask' or 'ff')
    mask_mean_subtraction: bool # Subtract mask mean
    variable_bipartite_full_field_params: int
    fixed_bipartite_full_field_params: int
    """
    contents = [(1,'perlin_high_freq_variable_high_freq_fixed_same_contrast_same_mean',1000000,100,32,1,'ff',1,3,4),
                (2,'perlin_low_freq_variable_low_freq_fixed_same_contrast_same_mean',1000000,100,32,1,'ff',1,1,2),
               (3,'perlin_high_freq_variable_low_freq_fixed_same_contrast_same_mean',1000000,100,32,1,'ff',1,3,1),
               (4,'perlin_low_freq_variable_high_freq_fixed_same_contrast_same_mean',1000000,100,32,1,'ff',1,2,4)]


@schema
class MostExcitingBipartiteStimuli(dj.Computed):
    definition = """
    -> MostExcitingBipartiteStimuliParameters
    -> BipartiteMask
    ---
    img: blob@static
    act: blob@static
    """
    @property
    def key_source(self):
        return MostExcitingBipartiteStimuliParameters * BipartiteMask

    def make(self,key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bipartite_stimuli_params = (MostExcitingBipartiteStimuliParameters() & key).fetch1()
        
        # Get neuron key
        neuron_key = (deis_schema.TextureLookup & key).fetch1()
        
        # Check condition to create noise
        mei_mask = np.array((base.MEIMask & neuron_key).fetch1('mask'))
        height,width = mei_mask.shape[-2:]
        
        variable_mask,fixed_mask = (BipartiteMask & key).fetch1('variable_mask','fixed_mask')
        
        # Get model
        mei_params = (base.MEIParameters & neuron_key).fetch1()
        
        model_key = ({'group_id': neuron_key['group_id'], 'net_hash': neuron_key['net_hash']} if
                     mei_params['use_avg_model'] else neuron_key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & neuron_key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & neuron_key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, neuron_key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=neuron_key['neuron_id'], device=device, average_batch=False)
        
                
        selected_results = []
        min_act,counter = 0,0
        
        variable_dl = get_noise_dataloader({'bipartite_full_field_params':bipartite_stimuli_params['variable_bipartite_full_field_params']})
        fixed_dl = get_noise_dataloader({'bipartite_full_field_params':bipartite_stimuli_params['fixed_bipartite_full_field_params']})
        variable_iter = iter(variable_dl)
        fixed_iter = iter(fixed_dl)
        while True:
            if counter>bipartite_stimuli_params['n_target_images']:
                break
            try:
                variable_texture = next(variable_iter)[...,:height,:width].cpu().numpy().squeeze()
                fixed_texture = next(fixed_iter)[...,:height,:width].cpu().numpy().squeeze()
                imgs = variable_texture * variable_mask[None,...] + fixed_texture * fixed_mask[None,...]
                counter += len(variable_texture)
            except:
                break  
            
            if bipartite_stimuli_params['match_stats'] == 'ff':
                target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
            else:
                raise ValueError()
            imgs = ops.standardize_image(imgs, target_mean, target_std, mei_mask, 
                                       bipartite_stimuli_params['mask_mean_subtraction'],bipartite_stimuli_params['mask_image'], 
                                       bipartite_stimuli_params['match_stats'])
            
            with torch.no_grad():
                acts = model(torch.tensor(imgs,dtype=torch.float32,device=device).unsqueeze(1)).cpu().numpy().squeeze()
            
            # Only select if the act is higher than minimum
            idxs = np.where(acts>min_act)[0]

            if len(idxs)>0:
                new_results = [{'img':img,'act':act} for img,act in zip(imgs,acts)]
                combined_results = selected_results + new_results
                selected_results = sorted(combined_results,key=lambda x: x['act'],reverse=True)[:bipartite_stimuli_params['n_keep']]
                min_act = selected_results[-1]['act']
        temp = {**key,'img':np.stack([i['img'] for i in selected_results]),'act':np.stack([i['act'] for i in selected_results])}
        self.insert1(temp)

def simple_load(x, height=36, width=64):
    return x,np.asarray(Image.open(x[0]))[:height,:width],np.asarray(Image.open(x[1]))[:height,:width]

# Twin of MostExcitingBipartiteStimuli
# Built based on a few assumptions to optimize for speed
@schema
class BipartiteStimuli(dj.Computed):
    definition = """
    -> MostExcitingBipartiteStimuliParameters
    -> BipartiteMask
    ---
    filename: blob@static
    act: blob@static
    """
    @property
    def key_source(self):
        return MostExcitingBipartiteStimuliParameters * BipartiteMask

    def make(self,key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bipartite_stimuli_params = (MostExcitingBipartiteStimuliParameters() & key).fetch1()
        # bipartite_stimuli_params['n_target_images'] = 1000

        # Get neuron key
        neuron_key = (deis_schema.TextureLookup & key).fetch1()

        # Check condition to create noise
        mei_mask = np.array((base.MEIMask & neuron_key).fetch1('mask'))
        height,width = mei_mask.shape[-2:]

        variable_mask,fixed_mask = (BipartiteMask & key).fetch1('variable_mask','fixed_mask')

        # Get model
        mei_params = (base.MEIParameters & neuron_key).fetch1()

        model_key = ({'group_id': neuron_key['group_id'], 'net_hash': neuron_key['net_hash']} if
                     mei_params['use_avg_model'] else neuron_key)
        all_keys = (static_models.Model & model_key & 'seed > 1000').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & neuron_key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & neuron_key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, neuron_key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=neuron_key['neuron_id'], device=device, average_batch=False)

        since = time.time()
        ## Assume the order is the same from the images.csv file for convenience
        variable_save_location = (BipartiteFullFieldParameters & {'bipartite_full_field_params':bipartite_stimuli_params['variable_bipartite_full_field_params']}).fetch1('save_location')
        fixed_save_location = (BipartiteFullFieldParameters & {'bipartite_full_field_params':bipartite_stimuli_params['fixed_bipartite_full_field_params']}).fetch1('save_location')

        filenames = list(pd.read_csv(os.path.join(variable_save_location,'images.csv'), sep=' ',names=['filepath'])[:bipartite_stimuli_params['n_target_images']].filepath)
        full_filenames = [(os.path.join(variable_save_location,i),os.path.join(fixed_save_location,i)) for i in filenames]

        since = time.time()
        from multiprocessing import Process, Pool,cpu_count
        import gc
        pool = Pool(8)
        results = pool.map(simple_load, full_filenames)
        pool.close()        
        print('finished loading')

        filenames,variable_textures,fixed_textures = [],[],[]
        for i in results:
            filenames.append(i[0])
            variable_textures.append(i[1])
            fixed_textures.append(i[2])

        del results
        gc.collect()

        filenames = np.array([os.path.basename(i[0]) for i in filenames])
        variable_textures = np.stack(variable_textures)
        fixed_textures = np.stack(fixed_textures)

        acts = []
        idxs = np.array_split(np.arange(len(variable_textures)),len(variable_textures)//bipartite_stimuli_params['batch_size'])
        for idx in tqdm(idxs):
            imgs = variable_textures[idx] * variable_mask[None,...] + fixed_textures[idx] * fixed_mask[None,...]
            if bipartite_stimuli_params['match_stats'] == 'ff':
                target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
            else:
                raise ValueError()
            imgs = ops.standardize_image(imgs, target_mean, target_std, mei_mask, 
                                       bipartite_stimuli_params['mask_mean_subtraction'],bipartite_stimuli_params['mask_image'], 
                                       bipartite_stimuli_params['match_stats'])
            with torch.no_grad():
                x = torch.tensor(imgs,dtype=torch.float32,device=device).unsqueeze(1)
                acts.append(model(x).cpu().numpy().squeeze())
        acts = np.concatenate(acts)
        temp = {**key, 'filename':filenames,'act':acts}
        self.insert1(temp)
@schema
class BipartiteGratingStimuliParameters(dj.Lookup):
    definition="""
    bipartite_grating_stimuli_params: int
    ---
    n_target_images: int
    image_size: int # Create square image for simplicity
    batch_size: int
    mask_image: bool       # Whether to mask the image
    match_stats: varchar(16) # Method for matching statistics on the image ('mask' or 'ff')
    mask_mean_subtraction: bool # Subtract mask mean
    grating_param_bounds: blob # Bounds for params to allow flexibility
    seed: int
    """
    contents = [(1,1000000,64,32,1,'ff',1,{'freq':[3,30],'ori':[0,360],'phase':[0,360]},2107),(2,2000000,64,32,1,'ff',1,{'freq':[3,30],'ori':[0,360],'phase':[0,360]},2107)]

# We do not save image but we keep all responses and parameter to generate the image
@schema
class BipartiteGratingStimuli(dj.Computed):
    definition = """
    -> BipartiteGratingStimuliParameters
    -> BipartiteMask
    ---
    img_param: blob@static
    act: blob@static
    """
    
    @property
    def key_source(self):
        return BipartiteGratingStimuliParameters.proj() * BipartiteMask
    
    # Each param should contain a pair
    @staticmethod
    def create_grating_image(param,variable_mask,fixed_mask):
        return (variable_mask * create_grating(sf=param['freq'][0], ori=param['ori'][0], 
                                               phase=param['phase'][0],wave='sin', 
                                               imsize=max(variable_mask.shape))[:variable_mask.shape[0],:variable_mask.shape[1]] + \
                fixed_mask * create_grating(sf=param['freq'][1], ori=param['ori'][1], 
                                               phase=param['phase'][1],wave='sin', 
                                               imsize=max(variable_mask.shape))[:fixed_mask.shape[0],:fixed_mask.shape[1]])
        
    def make(self,key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bipartite_grating_stimuli_params = (BipartiteGratingStimuliParameters() & key).fetch1()
        #bipartite_grating_stimuli_params['n_target_images'] = 10000

        # Get neuron key
        neuron_key = (deis_schema.TextureLookup & key).fetch1()

        # Check condition to create noise
        mei_mask = np.array((base.MEIMask & neuron_key).fetch1('mask'))
        height,width = mei_mask.shape[-2:]

        variable_mask,fixed_mask = (BipartiteMask & key).fetch1('variable_mask','fixed_mask')

        # Get model
        mei_params = (base.MEIParameters & neuron_key).fetch1()

        model_key = ({'group_id': neuron_key['group_id'], 'net_hash': neuron_key['net_hash']} if
                     mei_params['use_avg_model'] else neuron_key)
        all_keys = (static_models.Model & model_key & 'seed > 0').fetch('KEY')
        all_models = [(static_models.Model & mk).load_network() for mk in all_keys]
        mean_eyepos = ([0, 0] if (base.Dataset.TrainStats & neuron_key).fetch1('norm_eyepos') else
                       (base.Dataset.TrainStats & neuron_key).fetch1('mean_eyepos'))
        mean_eyepos = torch.tensor(mean_eyepos, dtype=torch.float32, device=device).unsqueeze(0)
        model = models.Ensemble(all_models, neuron_key['readout_key'], eye_pos=mean_eyepos,
                                neuron_idx=neuron_key['neuron_id'], device=device, average_batch=False)
        
        if bipartite_grating_stimuli_params['match_stats'] == 'ff':
            target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
        else:
            raise ValueError()
            
        grating_bounds = bipartite_grating_stimuli_params['grating_param_bounds']
        acts = []
        img_params = {i:[] for i in grating_bounds.keys()}
        
        # Process images and pass
        n_batch = bipartite_grating_stimuli_params['n_target_images']//bipartite_grating_stimuli_params['batch_size'] + 1
        
        # Specify seed
        np.random.seed(bipartite_grating_stimuli_params['seed'])
        
        for _ in tqdm(range(n_batch)):
            # Create the params
            temp_params = {}
            for i,j in grating_bounds.items():
                temp_params[i] = np.random.uniform(low=j[0],high=j[1],size=(bipartite_grating_stimuli_params['batch_size'],2))
            for i in temp_params.keys():
                img_params[i].append(temp_params[i])
                
            # Create images
            imgs = []
            for i in range(bipartite_grating_stimuli_params['batch_size']):
                param = {j:temp_params[j][i] for j in temp_params.keys()}
                imgs.append(BipartiteGratingStimuli.create_grating_image(param,variable_mask,fixed_mask))
            imgs = np.stack(imgs)
            imgs = ops.standardize_image(imgs, target_mean, target_std, mei_mask, 
                           bipartite_grating_stimuli_params['mask_mean_subtraction'],bipartite_grating_stimuli_params['mask_image'], 
                           bipartite_grating_stimuli_params['match_stats'])
            with torch.no_grad():
                x = torch.tensor(imgs,dtype=torch.float32,device=device).unsqueeze(1)
                acts.append(model(x).cpu().numpy().squeeze())
        for i in img_params.keys():
            img_params[i] = np.concatenate(img_params[i],axis=0)[:bipartite_grating_stimuli_params['n_target_images']]
        acts = np.concatenate(acts)[:bipartite_grating_stimuli_params['n_target_images']]
        self.insert1({**key, 'img_param':img_params,'act':acts})

from staticnet_invariance import toy_deis, toy

def toy_standardize_crops_v2(crops,most_exciting_crop_params,key):
    mask_float, mask_x, mask_y = (toy_deis.MEIMask & key).fetch1('mask', 'mask_x', 'mask_y')
    mei_params = (base.MEIParameters & key).fetch1()
    crops = ops.create_whole_mei(crops, mask_float, mask_x, mask_y, (len(crops), 36, 64), normalize_crop = False)
    if most_exciting_crop_params['match_stats'] == 'ff':
        target_mean, target_std = float(mei_params['mean']), float(mei_params['contrast'])
    else:
        raise ValueError()
    crops = ops.standardize_image(crops, target_mean, target_std, mask_float,most_exciting_crop_params['mask_mean_subtraction'],
                                   most_exciting_crop_params['mask_image'],most_exciting_crop_params['match_stats'])
    return crops

@schema
class ToyMostExcitingCrop(dj.Computed):
    definition = """ # For multiple neurons
    -> toy_deis.MEIMask
    -> MostExcitingCropParameters
    -> DatasetParameters
    ---
    image_id:          blob
    x:                 blob
    y:                 blob
    crop_masked:       blob@static # Crop standardized by the neuron mask
    crop_label:        blob@static # Crop label to make it convenient
    act:               blob
    n_searched:         int
    heter_searched:     int
    homo_searched:      int
    
    """
    @property
    def key_source(self):
        return toy_deis.MEIMask * MostExcitingCropParameters * DatasetParameters

    def make(self,key):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        most_exciting_crop_params = (MostExcitingCropParameters() & key).fetch1()
        dataset_params = (DatasetParameters & key).fetch1()

        # Get toy model
        model = toy.Ensemble(member_id=key['member_id'],key=key,device=device,average_batch=False)
        for i in model.models:
            i.to(device)



        # Since we can match the fetching from ImageDataset, make won't use ImageDataset but assume the id match up
        # To do this trick without much complication, no shuffling will be done
        all_dl = get_dataloader({**most_exciting_crop_params,**dataset_params})
        iterator = iter(all_dl)
        image_id_offset = 0

        selected_results = []
        counter,min_act,heter_counter,homo_counter = 0,0,0,0
        max_heter = max_homo = None
        # current_image_batch,n_image_batch = 0, len(image_keys)
        balanced,n_target_images,hetero_bounds,homo_bounds,mask_threshold = [most_exciting_crop_params[i] for i in ['balanced','n_target_images','hetero_bounds','homo_bounds','mask_threshold']]

        n_target_images=10000

        if balanced == 'heter_only':
            max_heter = n_target_images
            max_homo = 0
        elif balanced == 'homo_only':
            max_heter = 0
            max_homo = n_target_images
        elif balanced == 'balanced':
            max_heter = n_target_images//2
            max_homo = n_target_images//2

        # Deal with balancing between number of homo and heter patches
        if not balanced=='random':
            heter_lower,heter_upper = float(hetero_bounds.split(',')[0][1:]),float(hetero_bounds.split(',')[1][:-1])
            homo_lower,homo_upper = float(homo_bounds.split(',')[0][1:]),float(homo_bounds.split(',')[1][:-1])
            mask_float,mask_x,mask_y = (toy_deis.MEIMask & key).fetch1('mask','mask_x','mask_y')

            if mask_threshold > 0.0:
                mask_float = binarize_mask(mask_float,mask_threshold)

        def continue_search_condition(balanced,counter,n_target_images,heter_counter,homo_counter,max_heter=None,max_homo=None):
            if balanced=='random':
                if counter < n_target_images:
                    return True
                else:
                    return False
            else:
                if heter_counter < max_heter or homo_counter < max_homo:
                    return True
                else:
                    return False

        while continue_search_condition(balanced,counter,n_target_images,heter_counter,homo_counter,max_heter,max_homo):
            try:
                images,masks = iterator.next()
                if (len(images) == 0)  or (len(masks)==0):
                    break
            except:
                break
            images, masks = images.detach().cpu().numpy().squeeze(), masks.detach().cpu().numpy().squeeze()
            image_ids = np.arange(len(images)) + image_id_offset
            image_id_offset += len(images)

            # Assume here that all images have the same size for convenience
            crop_results = crop_img_v2(images,masks,image_ids,most_exciting_crop_params)

            # Create filter
            if not balanced=='random':
                patch_types = get_patch_type(crop_results['mask'],mask_float,mask_x,mask_y,homo_lower,homo_upper,heter_lower,heter_upper)
                n_heter_needed = np.clip(max_heter - heter_counter,a_min=0,a_max=len(patch_types))
                n_homo_needed = np.clip(max_homo - homo_counter,a_min=0,a_max=len(patch_types))
                heter_idx,homo_idx = [np.where(patch_types==i)[0][:j] for i,j in zip(['heter','homo'],[n_heter_needed,n_homo_needed])]
                idxs = np.concatenate([heter_idx,homo_idx])
                heter_counter += len(heter_idx)
                homo_counter += len(homo_idx)

            else:
                # If not balanced still need to restrict to the number of crops accepted generally
                n_processed = np.clip(n_target_images - counter,a_min=0,a_max=len(crop_results['crop']))
                idxs = np.random.choice(np.arange(len(crop_results['crop'])),n_processed,replace=False)

            if len(idxs)>0:
                # Filter out imgs, acts, x, y, image_id
                for i in crop_results.keys():
                    crop_results[i] = crop_results[i][idxs]

                crop_results['crop'] = toy_standardize_crops_v2(crop_results['crop'],most_exciting_crop_params,key)
                crop_results['act'] = pass_img(crop_results['crop'],model,device,batch_size=32)
                counter += len(crop_results['act'])

                # Only select if the act is higher than minimum
                idxs = np.where(crop_results['act']>min_act)[0]

            if len(idxs)>0:
                new_results = [{j:crop_results[j][i] for j in crop_results.keys()} for i in idxs]
                combined_results = selected_results + new_results
                selected_results = sorted(combined_results,key=lambda x: x['act'],reverse=True)[:most_exciting_crop_params['n_keep']]
                min_act = selected_results[-1]['act']

        temp = {i:[] for i in ['crop','mask','image_id','x','y','act']}
        # Merge the result back
        for i in ['crop','mask','image_id','x','y','act']:
            temp[i] = [j[i] for j in selected_results]
            if i=='crop':
                temp[i] = np.stack(temp[i])
            else:
                temp[i] = np.array(temp[i])
        temp['crop_masked'] = temp['crop']
        del temp['crop']
        temp['crop_label'] = temp['mask']
        del temp['mask']
        temp = {**key,**temp,'heter_searched':heter_counter,'homo_searched':homo_counter,'n_searched':counter}
        self.insert1(temp)
