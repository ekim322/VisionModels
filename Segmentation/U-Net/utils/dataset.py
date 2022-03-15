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

    type1 = [A.Blur(blur_limit = 50, p = 0.15), 
         A.GaussianBlur(blur_limit=(15, 51), p=0.15), 
         A.GlassBlur(sigma=1, max_delta=7, iterations=1, p=0.15), 
         A.HorizontalFlip(p=0.2), 
         A.MedianBlur(blur_limit=51, p=0.15), 
         A.MotionBlur(blur_limit = 47, p=0.15)]

    type2 = [A.CLAHE(clip_limit = 10, p = 0.2), 
            A.Emboss(alpha=(0.5, 0.9), strength=(0.5, 0.9), p = 0.2), 
            A.Equalize(p = 0.2), 
            A.InvertImg(p = 0.01), 
            A.MultiplicativeNoise(multiplier=(1, 3), p=0.2),
            A.Solarize(p=0.2)]

    type3 = [A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p = 0.2), 
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, p=0.2), 
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2), 
            A.RandomToneCurve(scale = 0.3, p=0.2),
            A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.2),
            A.ToGray(p=0.2), 
            A.ToSepia(p=0.2)]

    type4 = [A.GaussNoise(var_limit=(45.0, 95.0), p = 0.2), 
            A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.3, 0.7), p = 0.2),
            A.Sharpen(alpha=(0.4, 0.8), lightness=(0.5, 1.0), p=0.2)]

    type5 = [A.GridDistortion(num_steps=1, distort_limit=1, p=0.1), 
            A.OpticalDistortion(distort_limit=0.7, shift_limit=1, p=0.1), 
            A.RandomGridShuffle(grid=(3, 3), p=0.1),
            A.Superpixels(p=0.05)]

    type6 = [A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08, p=0.1), 
            A.RandomRain(p=0.1), 
            A.RandomShadow(p=0.1), 
            A.RandomSnow(p=0.1), 
            A.RandomSunFlare(p=0.1)]

    transform = A.Compose([
        A.Resize(width=img_size, height=img_size),
        np.random.choice(type1), 
        np.random.choice(type2), 
        np.random.choice(type3),
        np.random.choice(type4), 
        np.random.choice(type5), 
        np.random.choice(type6)
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
        self.transform = transform
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

        augmentations = self.transform(image=image, mask=mask)
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