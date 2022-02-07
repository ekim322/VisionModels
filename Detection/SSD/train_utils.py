import json
import numpy as np
import os
from PIL import Image

import torch
from torch.optim import lr_scheduler
import torch.distributed as dist

from torch.utils.data import Dataset

# from data_loader.TickDataset import GaugeDataset
from torch.utils.data.sampler import BatchSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloader(img_folder, label_path, batch_size, type='train'):
    dataset = GaugeDataset(img_folder, label_path)
    if type=='train':
        ds_sampler = torch.utils.data.RandomSampler(dataset)
    else:
        ds_sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(ds_sampler, batch_size, drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, 
                                              num_workers=4, collate_fn=collate_fn)
    return data_loader

def exp_loader(img_folder, label_path, batch_size):
 
    def get_goKart(root, image_set): 
        label_path = os.path.join(root, "{}_labels.json".format(image_set))
        img_path = os.path.join(root, "{}_imgs".format(image_set))
        
        return goKartDataset(label_path=label_path, img_path=img_path)
    def get_dataset(name, image_set, data_path):
        paths = {
            "goKart": (data_path, get_goKart, 1)
        }
        p, ds_fn, num_classes = paths[name]

        ds = ds_fn(p, image_set=image_set)
        return ds, num_classes
    
    dataset, num_classes = get_dataset("goKart", "train", "goKart_data")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(train_sampler, 4, drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, 
                                                num_workers=4, collate_fn=collate_fn)
    return data_loader

class goKartDataset(Dataset):
    def __init__(self, label_path, img_path, transform=None):
        json_file = open(label_path, 'r')
        labels_json = json.load(json_file)
        json_file.close()
        self.labels_json = labels_json
        self.img_path = img_path
        
    def __len__(self):
        return len(self.labels_json)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
            
        labels = self.labels_json[idx]
        imgID = labels['imgID']
    
        for key in labels:
            if key=="boxes" or key=="labels":
                labels[key] = torch.Tensor(np.asarray(labels[key])).type(torch.long)

        img_file = os.path.join(self.img_path, str(imgID)+".jpg")
        img_raw = Image.open(img_file)
        img = np.array(img_raw) / 255
        img_raw.close()
        img = np.transpose(img, (2, 0, 1))
        return_labels = {k:v for k,v in labels.items() if k=='boxes' or k=='labels'}

        tensor_img = torch.from_numpy(img).type(torch.float32)
        return tensor_img, return_labels