import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing

from model.ssdlite import create_ssdlite

# from train_utils import exp_loader, get_dataloader
from data_loader.TickDataset import Tick_dataloader

# parser = argparse.ArgumentParser(description='FindStuffNet Training')
# parser.add_argument('-c', '--config', required=True, type=str,
#                     help='location of config file')
# args = parser.parse_args()
class Params():
    def __init__(self):
        self.config = 'model_configs/config.yaml'
args = Params()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')

def train_step(model, optimizer, data_loader, epoch, cfg, device=device):
    model.train()

    if epoch==0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader)-1)

        # TODO: figure out LR scheduler
        # lr_scheduler = torch.optim.lr_scheduler.LinearLR( 
        #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        # )

    epoch_loss = 0.
    epoch_cls_loss = 0.
    epoch_reg_loss = 0.

    data_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)

    for i, (images, targets) in data_loop:
        images = list(image.to(device) for image in images) # List of images, length batch size [(c,w,h), (c,w,h)]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # List of targets dictionaries

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) # TODO: loss ratio between classification vs regression
        loss_value = losses.item()
        cls_loss_value = loss_dict['classification'].item()
        reg_loss_value = loss_dict['bbox_regression'].item()

        epoch_loss += loss_value
        epoch_cls_loss += cls_loss_value
        epoch_reg_loss += reg_loss_value

        losses.backward()
        if ((i+1)%cfg['batch_accum'])==0: # TODO: check if implementation correct
            optimizer.step()
            optimizer.zero_grad()

        loop_description = "Epoch - {}/{}".format(epoch, cfg['epochs'])
        data_loop.set_description(loop_description)
        data_loop.set_postfix(loss=loss_value)
    epoch_loss = epoch_loss/len(data_loader)
    epoch_cls_loss = epoch_cls_loss/len(data_loader)
    epoch_reg_loss = epoch_reg_loss/len(data_loader)
        
    return epoch_loss, epoch_cls_loss, epoch_reg_loss


def val_step(model, data_loader, epoch, device=device):
    model.eval()

    pix_off_lst = np.array([0, 0, 0, 0]) 
    total_four_off = 0
    total_eight_off = 0
    total_tw_off = 0

    data_loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for i, (images, targets) in data_loop:
        images = list(image.to(device) for image in images) # List of images, length batch size [(c,w,h), (c,w,h)]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # List of targets dictionaries

        loss_dict = model(images) #len batch size (boxes, scores, labels)
        top_preds = [res['boxes'][0].tolist() for res in loss_dict]
        # top_confs = [res['scores'][0].item() for res in loss_dict]
        
        target_bbox = [res['boxes'][0].tolist() for res in targets]
        abs_pix_off = np.abs(np.array(target_bbox) - np.array(top_preds))
        mean_pix_off = np.mean(abs_pix_off, axis=0)
        pix_off_lst = pix_off_lst + mean_pix_off

        total_four_off += np.sum(np.sum(abs_pix_off, axis=1) > 4)
        total_eight_off += np.sum(np.sum(abs_pix_off, axis=1) > 8)
        total_tw_off += np.sum(np.sum(abs_pix_off, axis=1) > 20)
        loop_description = "Validation"
        data_loop.set_description(loop_description)
    
    pix_off_lst = pix_off_lst / i
    print("Total data:", len(data_loader)*len(targets))
    print("  More than 4 off:", total_four_off)
    print("  More than 8 off:", total_eight_off)
    print("  More than 20 off:", total_tw_off)
    return pix_off_lst, total_four_off, total_eight_off, total_tw_off

def train_model(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, cfg['lr']) # TODO: experiment values   

    train_dataloader = Tick_dataloader(cfg['img_folder_train'], cfg['label_path_train'], cfg['batch_size'])
    val_dataloader = Tick_dataloader(cfg['img_folder_val'], cfg['label_path_val'], cfg['batch_size'])
    # train_dataloader = exp_loader(cfg['img_folder_train'], cfg['label_path_train'], cfg['batch_size'])
    # val_dataloader = exp_loader(cfg['img_folder_val'], cfg['label_path_val'], cfg['batch_size'])
    
    BEST_TRAIN_LOSS = 10000
    BEST_VAL_OFF = 10000

    writer = SummaryWriter(os.path.join('logs', cfg['exp_name']))    
    for epoch in range(cfg['epochs']):
        train_loss, train_cls_loss, train_reg_loss = train_step(model, optimizer, train_dataloader, epoch, cfg)
        writer.add_scalar('train_loss/total_loss', train_loss, global_step=epoch)
        writer.add_scalar('train_loss/cls_loss', train_cls_loss, global_step=epoch)
        writer.add_scalar('train_loss/reg_loss', train_reg_loss, global_step=epoch)
        if train_loss < BEST_TRAIN_LOSS:
            torch.save(model.state_dict(), os.path.join(MODEL_CKPT, 'train.pt'))
            BEST_TRAIN_LOSS = train_loss
        
        val_off, four, eight, twenty = val_step(model, val_dataloader, epoch)
        writer.add_scalar('val_off/mean', np.mean(val_off), global_step=epoch)
        writer.add_scalar('val_off/four', four, global_step=epoch)
        writer.add_scalar('val_off/eight', eight, global_step=epoch)
        writer.add_scalar('val_off/twenty', twenty, global_step=epoch)
        print("Val off: {} - Mean: {}".format(val_off, np.mean(val_off)))
        if np.mean(val_off) < BEST_VAL_OFF:
            torch.save(model.state_dict(), os.path.join(MODEL_CKPT, 'best.pt'))
            BEST_VAL_OFF = np.mean(val_off)

if __name__=='__main__':
    with open(args.config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    MODEL_CKPT = 'model_weights/{}'.format(cfg['exp_name'])
    if not os.path.exists(MODEL_CKPT):
        os.makedirs(MODEL_CKPT)

    model = create_ssdlite()
    # model.load_state_dict(torch.load('model_weights/TDN_1e4/best.pt'))
    model.to(device)

    train_model(model, cfg)