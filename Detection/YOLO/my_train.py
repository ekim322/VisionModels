import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class Train():
    
    def __init__(self, 
                 hyp, # path/to/hyp.yaml or hyp dictionary
                 opt,  
                 device,
                 callbacks = Callbacks()):
        self.hyp = hyp
        self.opt = opt
        self.callbacks = callbacks
        self.device = device
        
        self.save_dir, self.epochs, self.batch_size, self.weights, self.single_cls, self.evolve, self.data, self.cfg, self.resume, self.noval, self.nosave, self.workers, self.freeze = \
        Path(self.opt.save_dir), self.opt.epochs, self.opt.batch_size, self.opt.weights, self.opt.single_cls, self.opt.evolve, 
        self.opt.data, self.opt.cfg, self.opt.resume, self.opt.noval, self.opt.nosave, self.opt.workers, self.opt.freeze
        
        # Directories
        self.w = self.save_dir / 'weights'  # weights dir
        (self.w.parent if self.evolve else self.w).mkdir(parents=True, exist_ok=True)  # make dir
        self.last, self.best = self.w / 'last.pt', self.w / 'best.pt'
        
        # Hyperparameters
        if isinstance(self.hyp, str):
            with open(self.hyp, errors='ignore') as f:
                self.hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in self.hyp.items()))
        
        # Save run settings
        if not self.evolve:
            with open(self.save_dir / 'hyp.yaml', 'w') as f:
                yaml.safe_dump(self.hyp, f, sort_keys=False)
            with open(self.save_dir / 'opt.yaml', 'w') as f:
                yaml.safe_dump(vars(self.opt), f, sort_keys=False)
        
        # Loggers
        self.data_dict = None
        if RANK in [-1, 0]:
            self.loggers = Loggers(self.save_dir, weights, self.opt, self.hyp, LOGGER)  # loggers instance
            if self.loggers.wandb:
                self.data_dict = self.loggers.wandb.data_dict
                if self.resume:
                    weights, epochs, self.hyp, batch_size = self.opt.weights, self.opt.epochs, self.opt.hyp, self.opt.batch_size

            # Register actions
            for k in methods(self.loggers):
                self.callbacks.register_action(k, callback=getattr(self.loggers, k))
                
        # Config
        self.plots = not self.evolve  # create plots
        self.cuda = self.device.type != 'cpu'
        init_seeds(1 + RANK)
        with torch_distributed_zero_first(LOCAL_RANK):
            self.data_dict = self.data_dict or check_dataset(self.data)  # check if None
        train_path, val_path = self.data_dict['train'], self.data_dict['val']
        self.nc = 1 if self.single_cls else int(self.data_dict['nc'])  # number of classes
        names = ['item'] if self.single_cls and len(self.data_dict['names']) != 1 else self.data_dict['names']  # class names
        assert len(names) == self.nc, f'{len(names)} names found for nc={self.nc} dataset in {self.data}'  # check
        self.is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
        
        # Freeze
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                LOGGER.info(f'freezing {k}')
                v.requires_grad = False

        # Image size
        self.gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        self.imgsz = check_img_size(self.opt.imgsz, self.gs, floor=self.gs * 2)  # verify imgsz is gs-multiple

        # Batch size
        if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
            batch_size = check_train_batch_size(model, self.imgsz)
            self.loggers.on_params_update({"batch_size": batch_size})

        # Optimizer
        self.nbs = 64  # nominal batch size
        accumulate = max(round(self.nbs / batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= batch_size * accumulate / self.nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {self.hyp['weight_decay']}")

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        if self.opt.optimizer == 'Adam':
            self.optimizer = Adam(g0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        elif self.opt.optimizer == 'AdamW':
            self.optimizer = AdamW(g0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = SGD(g0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': g1, 'weight_decay': self.hyp['weight_decay']})  # add g1 with weight_decay
        self.optimizer.add_param_group({'params': g2})  # add g2 (biases)
        LOGGER.info(f"{colorstr('optimizer:')} {type(self.optimizer).__name__} with parameter groups "
                    f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
        del g0, g1, g2

        # Scheduler
        if self.opt.linear_lr:
            self.lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        else:
            self.lf = one_cycle(1, self.hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        self.ema = ModelEMA(model) if RANK in [-1, 0] else None

        # Resume
        self.start_epoch, self.best_fitness = 0, 0.0
        if weights.endswith('.pt'):
            # Optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if self.ema and ckpt.get('ema'):
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                self.ema.updates = ckpt['updates']

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.resume:
                assert self.start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
            if epochs < self.start_epoch:
                LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, csd

        # DP mode
        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        self.train_loader, self.dataset = create_dataloader(train_path, self.imgsz, batch_size // WORLD_SIZE, self.gs, self.single_cls,
                                                hyp=self.hyp, augment=True, cache=self.opt.cache, rect=self.opt.rect, rank=LOCAL_RANK,
                                                workers=self.workers, image_weights=self.opt.image_weights, quad=self.opt.quad,
                                                prefix=colorstr('train: '), shuffle=True)
        mlc = int(np.concatenate(self.dataset.labels, 0)[:, 0].max())  # max label class
        self.nb = len(self.train_loader)  # number of batches
        assert mlc < self.nc, f'Label class {mlc} exceeds nc={self.nc} in {self.data}. Possible class labels are 0-{self.nc - 1}'

        # Process 0
        if RANK in [-1, 0]:
            self.val_loader = create_dataloader(val_path, self.imgsz, batch_size // WORLD_SIZE * 2, self.gs, self.single_cls,
                                        hyp=self.hyp, cache=None if self.noval else self.opt.cache, rect=True, rank=-1,
                                        workers=self.workers, pad=0.5,
                                        prefix=colorstr('val: '))[0]

            if not self.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                # c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                if self.plots:
                    plot_labels(labels, names, self.save_dir)

                # Anchors
                if not self.opt.noautoanchor:
                    check_anchors(self.dataset, model=model, thr=self.hyp['anchor_t'], imgsz=self.imgsz)
                model.half().float()  # pre-reduce anchor precision

            self.callbacks.run('on_pretrain_routine_end')

        # DDP mode
        if self.cuda and RANK != -1:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        
        # Model attributes
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        self.hyp['box'] *= 3 / nl  # scale to layers
        self.hyp['cls'] *= self.nc / 80 * 3 / nl  # scale to classes and layers
        self.hyp['obj'] *= (self.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.hyp['label_smoothing'] = self.opt.label_smoothing
        model.nc = self.nc  # attach number of classes to model
        model.hyp = self.hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(self.device) * self.nc  # attach class weights
        model.names = names
        

    def train_step(self, model, epoch):
        
        # save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        #     Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        #     opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
        
        # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / self.nc  # class weights
            iw = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=cw)  # image weights
            self.dataset.indices = random.choices(range(self.dataset.n), weights=iw, k=self.dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        self.mloss = torch.zeros(3, device=self.device)  # mean losses
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + self.nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(self.device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= self.nw:
                xi = [0, self.nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [self.hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=self.cuda):
                pred = model(imgs)  # forward
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            self.scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                self.scaler.step(self.optimizer)  # optimizer.step
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                self.mloss = (self.mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{self.epochs - 1}', mem, *self.mloss, targets.shape[0], imgs.shape[-1]))
                self.callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, self.plots, opt.sync_bn)
                if self.callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
        self.scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            self.callbacks.run('on_train_epoch_end', epoch=epoch)
            self.ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop
            if not self.noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(self.data_dict,
                                            batch_size=self.batch_size // WORLD_SIZE * 2,
                                            imgsz=self.imgsz,
                                            model=self.ema.ema,
                                            single_cls=self.single_cls,
                                            dataloader=self.val_loader,
                                            save_dir=self.save_dir,
                                            plots=False,
                                            callbacks=self.callbacks,
                                            compute_loss=self.compute_loss)

            # Update best mAP
            self.fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if self.fi > best_fitness:
                best_fitness = self.fi
            log_vals = list(self.mloss) + list(results) + self.lr
            self.callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, self.fi)

            # Save model
            if (not self.nosave) or (final_epoch and not self.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(self.ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict(),
                        'wandb_id': self.loggers.wandb.wandb_run.id if self.loggers.wandb else None,
                        'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, self.last)
                if best_fitness == self.fi:
                    torch.save(ckpt, self.best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, self.w / f'epoch{epoch}.pt')
                del ckpt
                self.callbacks.run('on_model_save', self.last, epoch, final_epoch, best_fitness, self.fi)

            # Stop Single-GPU
            if RANK == -1 and self.stopper(epoch=epoch, fitness=self.fi):
                return False # break loop

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch
        
        return True # continue loop


    def train_model(self, model):
                
        # Start training
        t0 = time.time()
        self.nw = max(round(self.hyp['warmup_epochs'] * self.nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(self.nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        self.stopper = EarlyStopping(patience=self.opt.patience)
        self.compute_loss = ComputeLoss(model)  # init loss class
        LOGGER.info(f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        
        writer = SummaryWriter(os.path.join('logs', self.opt.name)) # should use cfg
        
        for epoch in range(self.start_epoch, self.epochs):
            loop = self.train_step(model, epoch)
            
            if not loop:
                break
        # end training -----------------------------------------------------------------------------------------------------
        
        if RANK in [-1, 0]:
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
            for f in self.last, self.best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
                    if f is self.best:
                        LOGGER.info(f'\nValidating {f}...')
                        results, _, _ = val.run(self.data_dict,
                                                batch_size=self.batch_size // WORLD_SIZE * 2,
                                                imgsz=self.imgsz,
                                                model=attempt_load(f, self.device).half(),
                                                iou_thres=0.65 if self.is_coco else 0.60,  # best pycocotools results at 0.65
                                                single_cls=self.single_cls,
                                                dataloader=self.val_loader,
                                                save_dir=self.save_dir,
                                                save_json=self.is_coco,
                                                verbose=True,
                                                plots=True,
                                                callbacks=self.callbacks,
                                                compute_loss=self.compute_loss)  # val best model with plots
                        if self.is_coco:
                            self.callbacks.run('on_fit_epoch_end', list(self.mloss) + list(results) + self.lr, epoch, self.best_fitness, self.fi)

            self.callbacks.run('on_train_end', self.last, self.best, self.plots, epoch, results)
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        torch.cuda.empty_cache()
        return results

    def create_model(self):
    
        # save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        # Path(self.opt.save_dir), self.opt.epochs, self.opt.batch_size, self.opt.weights, self.opt.single_cls, self.opt.evolve, 
        # self.opt.data, self.opt.cfg, self.opt.resume, self.opt.noval, self.opt.nosave, self.opt.workers, self.opt.freeze

        # # Config
        # plots = not evolve  # create plots
        # cuda = device.type != 'cpu'
        # init_seeds(1 + RANK)
        # with torch_distributed_zero_first(LOCAL_RANK):
            # data_dict = data_dict or check_dataset(self.data)  # check if None
        # train_path, val_path = data_dict['train'], data_dict['val']
        # nc = 1 if self.single_cls else int(data_dict['nc'])  # number of classes
        # names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        # assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
        # is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

        # Model
        check_suffix(weights, '.pt')  # check weights
        pretrained = weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            model = Model(self.cfg or ckpt['model'].yaml, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
            exclude = ['anchor'] if (self.cfg or self.hyp.get('anchors')) and not self.resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        else:
            model = Model(self.cfg, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
        
        return model

def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=['thop'])

    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        
        train_obj = Train(opt.hyp, opt, device, callbacks)        
        model = train_obj.create_model()            
        train_obj.train_model(model) 

        # train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            train_obj = Train(opt.hyp, opt, device, callbacks)
            model = train_obj.create_model()
            results = train_obj.train_model(model)
            
            # results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    
    # MODEL_CKPT = 'model_weights/{}'.format(opt.name) # should use cfg
    # if not os.path.exists(MODEL_CKPT):
    #     os.makedirs(MODEL_CKPT)
        
    main(opt)