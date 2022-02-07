import json
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class TickDataset(Dataset):
    def __init__(self, img_folder, label_path):
        self.img_folder = img_folder
        json_file = open(label_path, 'r')
        self.labels_json = json.load(json_file)
        json_file.close()

    def __len__(self):
        return len(self.labels_json)

    def __getitem__(self, idx):
        labels = self.labels_json[idx]
        coordinates = labels['bbox']#[labels['x'], labels['y'], labels['w'], labels['h']]
        coordinates = torch.Tensor([coordinates]).type(torch.long)

        imgID = labels['imgID']
        img_path = os.path.join(self.img_folder, str(imgID)+".jpg")
        img_raw = Image.open(img_path)
        img = np.array(img_raw) / np.max(img_raw)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).type(torch.float32)

        return img, {'boxes':coordinates, 'labels': torch.Tensor([1]).to(torch.int64)}

def collate_fn(batch):
    return tuple(zip(*batch))

def Tick_dataloader(img_folder, label_path, batch_size, type='train'):
    dataset = TickDataset(img_folder, label_path)
    if type=='train':
        ds_sampler = torch.utils.data.RandomSampler(dataset)
    else:
        ds_sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(ds_sampler, batch_size, drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, 
                                              num_workers=4, collate_fn=collate_fn)
    return data_loader