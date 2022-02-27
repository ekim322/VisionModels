from pathlib import Path
from tqdm import tqdm

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

import albumentations as A

from utils.utils_fn import np_to_tensor

def UNET_DataLoader(img_dir, mask_dir, batch_size, img_size, split_ratio=0):
    """
    If split ratio is 0 - return dataset 
    Else - split train/val
    """
    transform = A.Compose([
        A.Resize(width=img_size, height=img_size)
    ])

    dataset = UNET_Dataset(img_dir, mask_dir, transform)
    # dataset = Nucleus_Dataset(img_dir, transform)
    if split_ratio==False:
        ds_sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(ds_sampler, batch_size, drop_last=False)
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)
        return data_loader
    else:
        num_val = int(len(dataset)*split_ratio)
        num_train = len(dataset) - num_val
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=False)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=4)

        val_sampler = torch.utils.data.RandomSampler(val_dataset)
        val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, batch_size, drop_last=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=4)

        return train_dataloader, val_dataloader
    

class UNET_Dataset(Dataset):
    """
    Define img_paths, mask_paths based on dataset
        - each paths 
    """
    def __init__(self, img_dir, mask_dir, transform=None, rgb=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transfrom = transform
        self.rgb = rgb

        # Find first file extension and set as data type to look for
        img_data_type = str(next(Path(img_dir).rglob('*.*'))).split('.')[-1]
        mask_data_type = str(next(Path(mask_dir).rglob('*.*'))).split('.')[-1]

        # Set paths
        self.img_paths = [str(x) for x in Path(img_dir).rglob('*.{}'.format(img_data_type))]
        self.mask_paths = [str(x) for x in Path(mask_dir).rglob('*.{}'.format(mask_data_type))]
        
        assert len(self.img_paths)==len(self.mask_paths), "Image folder and Annot folder have different sizes!"
        
        # Sort paths so that data can be accessed with index
        self.img_paths.sort()
        self.mask_paths.sort()
        
    def __len__(self):
        return len(self.img_paths) 

    def get_num_classes(self):
        """
        Loops through all the masks and finds all unique classes
        """
        unique_classes = set()
        data_loop = tqdm(range(len(self.img_paths)))
        data_loop.set_description("Searching for all unique classes...")
        for i in data_loop:
            mask_path = self.mask_paths[i]
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # Gray scale
            unique_classes.update(np.unique(mask))
        return len(unique_classes)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        if self.rgb:
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = np.array(Image.open(img_path).convert("L"))
            image = np.expand_dims(image, axis=-1)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # Gray scale
        image = image/np.max(image)
        if len(np.unique(mask))==2:
            mask = mask/np.max(mask)

        augmentations = self.transfrom(image=image, mask=mask)
        image = augmentations['image']
        mask = augmentations['mask']

        image = np_to_tensor(image, is_mask=False)
        mask = np_to_tensor(mask, is_mask=True)

        return {'images':image.contiguous(), 'masks':mask.contiguous()}

class Nucleus_Dataset(Dataset):
    """
    Define img_paths, mask_paths based on dataset
        - each paths 
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = [str(x) for x in Path(img_dir).rglob('*/images/*.*')]
        self.img_paths.sort()

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = np.array(Image.open(img_path).convert("L"))
        img = np.expand_dims(image, axis=-1)
        mask_dir = img_path.replace('images', 'masks').rsplit('/', 1)[0]
        mask_paths = [str(x) for x in Path(mask_dir).rglob('*.*')]

        mask = np.zeros(img.shape[:2])
        for mask_path in mask_paths:
            cur_mask = np.array(Image.open(mask_path).convert("L"))
            mask = np.maximum(mask, cur_mask)

        image = image/np.max(image)
        if len(np.unique(mask))==2:
            mask = mask/np.max(mask)
        mask = mask.astype(np.float32)

        augmentations = self.transform(image=image, mask=mask)
        image = augmentations['image']
        mask = augmentations['mask']

        image = np_to_tensor(image, is_mask=False)
        mask = np_to_tensor(mask, is_mask=True)

        return {'images':image.contiguous(), 'masks':mask.contiguous()}