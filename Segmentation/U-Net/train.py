import os
import argparse
import yaml
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import UNET_DataLoader, UNET_Dataset
from utils.utils_fn import check_config_key
from model.unet import UNet
from model.loss import cross_entropy_loss_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/nuc_train_config.yaml', type=str)

    return parser.parse_args()


def train_step(model, optimizer, grad_scaler, loss_fn, dataloader, epoch, cfg, writer):
    model.train()
    running_loss = 0.

    data_loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for i, data in data_loop:
        images = data['images'].to(device=device, dtype=torch.float32)
        if model.n_classes==1: # If single class, use BCELoss -> target needs to be float if input is float
            masks = data['masks'].to(device=device, dtype=torch.float32)
        else:
            masks = data['masks'].to(device=device, dtype=torch.long)
        
        if model.n_classes==1: # If single class expand dimension 
            masks = masks.unsqueeze(1)

        # Use mixed precision
        with torch.cuda.amp.autocast(enabled=cfg['mixed_precision']):
            masks_preds = model(images)
            loss = loss_fn(masks_preds, masks)
            running_loss += loss.item()

        grad_scaler.scale(loss).backward()
        # Only update after batch_accum iterations -> same thing as training bigger batch size
        if (i+1)%cfg['batch_accum'] == 0:
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        loop_description = "Epoch {}/{} - (train)".format(epoch, cfg['epochs'])
        data_loop.set_description(loop_description)
        data_loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss/i
    writer.add_scalar('loss/train_loss', epoch_loss, global_step=epoch)
    
    return epoch_loss

def eval_step(model, loss_fn, dataloader, epoch, cfg, writer, eval_type='val'):
    """
    Args 
        eval_type (str): 'val' or 'test'
    """
    model.eval()
    running_loss = 0.

    data_loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for i, data in data_loop:
        images = data['images'].to(device=device, dtype=torch.float32)
        if model.n_classes==1: # If single class, use BCELoss -> target needs to be float if input is float
            masks = data['masks'].to(device=device, dtype=torch.float32)
        else:
            masks = data['masks'].to(device=device, dtype=torch.long)

        if model.n_classes==1: # If single class expand dimension 
            masks = masks.unsqueeze(1)

        with torch.no_grad():
            masks_preds = model(images)
            loss = loss_fn(masks_preds, masks)
            running_loss += loss.item()

        loop_description = "Epoch {}/{} - ({})".format(epoch, cfg['epochs'], eval_type)
        data_loop.set_description(loop_description)
        data_loop.set_postfix(loss=loss.item())

    eval_loss = running_loss/i
    writer.add_scalar('loss/{}_loss'.format(eval_type), eval_loss, global_step=epoch)

    return eval_loss

def train_model(model, cfg):
    if not os.path.exists(cfg['model_save_dir']):
        os.makedirs(cfg['model_save_dir'])
    
    TRAIN_SAVE_PATH = os.path.join(cfg['model_save_dir'], cfg['exp_name']+"_train.pt")
    VAL_SAVE_PATH = os.path.join(cfg['model_save_dir'], cfg['exp_name']+"_val.pt")
    TEST_SAVE_PATH = os.path.join(cfg['model_save_dir'], cfg['exp_name']+"_test.pt")

    # Load train/val dataset
    if check_config_key(cfg, 'val_img_dir'): # If validation set exists
        train_dataloader = UNET_DataLoader(cfg['train_img_dir'], cfg['train_mask_dir'], cfg['batch_size'], cfg['img_size'])
        val_dataloader = UNET_DataLoader(cfg['val_img_dir'], cfg['val_mask_dir'], cfg['batch_size'], cfg['img_size'])
    else: # Split train set  to train/val
        print("Splitting training and validation set")
        train_dataloader, val_dataloader = UNET_DataLoader(cfg['train_img_dir'], cfg['train_mask_dir'], cfg['batch_size'], cfg['img_size'], split_ratio=0.1)

    # Load test set if exist
    if check_config_key(cfg, 'test_img_dir'):
        test_dataloader = UNET_DataLoader(cfg['test_img_dir'], cfg['test_mask_dir'], cfg['batch_size'], cfg['img_size'])

    if check_config_key(cfg, 'cont_epoch'):
        start_epoch = cfg['cont_epoch']
        print("Continue training at epoch: {}".format(start_epoch))
    else:
        start_epoch = 0
        print("Starting training from scratch...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = cfg['lr'],
        betas = (0.9, 0.999),
        weight_decay=cfg['wd']
    )
    loss_fn = torch.nn.CrossEntropyLoss()
#     loss_fn = torch.nn.BCEWithLogitsLoss()
    writer = SummaryWriter(os.path.join('logs', cfg['exp_name']))
    grad_scaler = torch.cuda.amp.GradScaler()

    best_train_loss = 1e5
    best_val_loss = 1e5
    best_test_loss = 1e5

    patience_counter = 0

    for epoch in range(start_epoch, start_epoch+cfg['epochs']):
        train_loss = train_step(model, optimizer, grad_scaler, loss_fn, train_dataloader, epoch, cfg, writer)
        val_loss = eval_step(model, loss_fn, val_dataloader, epoch, cfg, writer, eval_type='val')

        print("Epoch {} - Train Loss: {} - Val Loss: {}".format(epoch, train_loss, val_loss))

        if check_config_key(cfg, 'test_img_dir'):
            test_loss = eval_step(model, loss_fn, test_dataloader, epoch, cfg, writer, eval_type='test')
            if test_loss<best_test_loss:
                torch.save(model.state_dict(), TEST_SAVE_PATH)
        
        if train_loss<best_train_loss:
            torch.save(model.state_dict(), TRAIN_SAVE_PATH)
        if val_loss<best_val_loss:
            torch.save(model.state_dict(), VAL_SAVE_PATH)
        else:
            patience_counter += 1
            # Early stopping
            if patience_counter >= cfg['patience']:
                print("Early stopping at epoch {}".format(epoch))
                break
                
    return

if __name__=='__main__':
    args = get_args()

    # Read config file
    with open(args.config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    # Get number of classes
    if check_config_key(cfg, 'num_classes'):
        num_classes = cfg['num_classes']
    else:
        # Manually search number of classes
        dataset = UNET_Dataset(cfg['train_img_dir'], cfg['train_mask_dir'])
        num_classes = dataset.get_num_classes()
    in_channels = 3 if cfg['img_rgb'] else 1
 
    model = UNet(n_channels=in_channels, n_classes=num_classes)
    if check_config_key(cfg, 'model_load_path'):
        model.load_state_dict(torch.load(cfg['model_load_path']))
    model.to(device)

    train_model(model, cfg)