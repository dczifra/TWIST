# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import numpy as np
import os
import time
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
from objective import *
from timm.scheduler import create_scheduler
from timm.utils import get_state_dict

from datasets import ImageNet, ImageNetLMDB, get_datasets
from augmentation import get_augmentations
from engine import train_one_epoch, eval_one_epoch, eval_knn, inference
from model import TWIST

import utils
import warnings
from tensorboardX import SummaryWriter
import torchvision
#from torchvision.models import resnet50
#from widen_resnet import resnet50w2, resnet50w4, resnet50w5, resnet200, resnet200w2
import widen_resnet
from lars import *
import vision_transformer as vit
from timm.utils import ModelEmaV2 as ModelEma
from functools import partial
from torchvision import transforms
warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Self-Supervised', add_help=False)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--lam1', type=float, default=0.0, metavar='LR')
    parser.add_argument('--lam2', type=float, default=1.0, metavar='LR')
    parser.add_argument('--tau', type=float, default=1.0, metavar='LR')
    parser.add_argument('--lbn_type', type=str, default='bn')
    parser.add_argument('--determine', type=int, default=0)
    parser.add_argument('--aug', type=str, default='multicrop')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--img_size_small', type=int, default=96)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--clip_norm', type=float, default=0.0)
    parser.add_argument('--EPS', type=float, default=1e-5, help='episillon')
    parser.add_argument('--reduce_mean', type=float, default=0)
    parser.add_argument('--loss_type', type=str, default="ENTLoss")

    parser.add_argument('--eval_only', type=int, default=0)
    parser.add_argument('--inference_only', type=int, default=0)
    parser.add_argument('--img_path', type=str, default='./test.jpg')
    parser.add_argument('--match_path', type=str, default='')

    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--quantile', type=float, default=0.5)
    parser.add_argument('--quantile_end', type=float, default=0.6)
    parser.add_argument('--enable_watch', type=int, default=1)

    parser.add_argument('--use_momentum_encoder', type=int, default=0)
    parser.add_argument('--momentum_start', default=0.996, type=float)
    parser.add_argument('--momentum_end', default=1.0, type=float)
    parser.add_argument('--freeze_embedding', default=0, type=int)

    # self-label relevant
    parser.add_argument('--mme_epochs', type=int, default=800)
    parser.add_argument('--sl_warmup_epochs', type=int, default=5)
    parser.add_argument('--lr_sl', type=float, default=0.05, metavar='LR')

    # maybe different between cnn and vit.
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--backbone', default='resnet50', type=str, metavar='BACKBONEMODEL', help='Name of model to train')
    parser.add_argument('--weight-decay', type=float, default=1.5e-6, help='weight decay (default: 1e-4)')
    parser.add_argument('--weight-decay-end', type=float, default=1.5e-6)
    parser.add_argument('--optim', default='lars', type=str)
    parser.add_argument('--lr', type=float, default=0.5, metavar='LR')
    parser.add_argument('--proj_trunc_init', type=int, default=0)
    parser.add_argument('--proj_norm', type=str, default='bn', choices=['bn', 'ln', 'none'])
    parser.add_argument('--drop_path', type=float, default=0.0)
    
    
    # multi-crop enabled only the aug is set to multicrop
    parser.add_argument('--local_crops_number', type=int, default=12,)
    parser.add_argument('--crops_interact_style', type=str, default='sparse')
    parser.add_argument('--min1', type=float, default=0.4, metavar='LR')
    parser.add_argument('--max1', type=float, default=1.0, metavar='LR')
    parser.add_argument('--min2', type=float, default=0.05, metavar='LR')
    parser.add_argument('--max2', type=float, default=0.4, metavar='LR')


    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--bunch-size', default=256, type=int)

    parser.add_argument('--epochs', default=850, type=int)
    parser.add_argument('--dim', default=4096, type=int)
    parser.add_argument('--hid_dim', default=4096, type=int)

    parser.add_argument('--eval-only', default=0, type=int)

    # Model parameters

    parser.add_argument('--exclude-bias-weight-decay', type=int, default=1)

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr_wbr', type=float, default=1.0)
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')


    # Dataset parameters
    parser.add_argument('--data-path', default='/nothing/', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--use-lmdb', action='store_true')
    parser.add_argument('--amp', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(args):
    if args.epochs <=20:
        args.warmup_epochs = 3 
    elif args.epochs <= 50:
        args.warmup_epochs = 5 
    else:
        args.warmup_epochs = 10
    if args.mme_epochs > args.epochs:
        args.mme_epochs = args.epochs
    if args.crops_interact_style == 'self_label':
        assert args.mme_epochs == args.epochs

    if(args.dataset not in ["imagenet", "imagenet_lmdb", "cifar10"]):
        assert(1, "dataset not found")

    utils.init_distributed_mode(args)
    print(args)

    output_dir = Path(args.output_dir)
    args.global_crops_scale = (args.min1, args.max1)
    args.local_crops_scale = (args.min2, args.max2)
    print('global crops: {}'.format(args.global_crops_scale))
    print('local  crops: {}'.format(args.local_crops_scale))

    cudnn.benchmark = True
    device = torch.device(args.device)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.determine:
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # =================== Data Preparation =================== 
    dataset_train, dataset_val, data_loader_train, \
        data_loader_val, sampler_train, sampler_val = get_datasets(args, num_tasks, global_rank)

    # ===================   Model   =================== 
    model = TWIST(args)
    model.to(device)

    if args.use_momentum_encoder:
        teacher_model = TWIST(args)
        teacher_model.to(device)
    else:
        teacher_model = None

    n_bb_parameters = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print('number of backbone params:{:.2f} M'.format(n_bb_parameters/1e6))
    n_ph_parameters = sum(p.numel() for p in model.projection_heads.parameters() if p.requires_grad)
    print('number of head params:{:.2f} M'.format(n_ph_parameters/1e6))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of total params:{:.2f} M'.format(n_parameters/1e6))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    if args.use_momentum_encoder:
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu])
        teacher_model_without_ddp = teacher_model.module
        teacher_model_without_ddp.load_state_dict(model_without_ddp.state_dict())
        for p in teacher_model.parameters():
            p.requires_grad = False
        momentum_schedule = utils.cosine_scheduler(args.momentum_start, args.momentum_end, args.mme_epochs, len(data_loader_train))
    else:
        momentum_schedule = None

    # =================== Optimizer =================== 
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 256.0
    args.lr = linear_scaled_lr
    linear_scaled_lr_sl = args.lr_sl * args.batch_size * utils.get_world_size() / 256.0
    args.lr_sl = linear_scaled_lr_sl

    param_weights = []
    param_biases = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print('{} is not optimized'.format(name))
            continue
        skip = ['pos_embed', 'cls_token', 'dist_token']
        if len(param.shape) == 1 or name.endswith(".bias") or sum([sk in name for sk in skip]):
            print('{} has been excluded for weight decay'.format(name))
            param_biases.append(param)
        else:
            param_weights.append(param)
    
    if args.optim == 'sgd':
        bias_weight_decay = 0.0 if args.exclude_bias_weight_decay else args.weight_decay
        parameters = [{'params': param_weights, 'weight_decay': args.weight_decay}, 
                      {'params': param_biases,  'weight_decay': bias_weight_decay}]
        optimizer = torch.optim.SGD(parameters, lr=0, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim in ['lars', 'lars_oss']:
        bias_weight_decay = 0.0 if args.exclude_bias_weight_decay else args.weight_decay
        parameters = [{'params': param_weights, 'weight_decay': args.weight_decay, 'lars_exclude': False}, 
                      {'params': param_biases,  'weight_decay': bias_weight_decay, 'lars_exclude': True}]
        optimizer = LARS_OPENSELF(parameters, lr=0, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optim == 'admw':
        bias_weight_decay = 0.0 if args.exclude_bias_weight_decay else args.weight_decay
        parameters = [{'params': param_weights, 'weight_decay': args.weight_decay}, 
                      {'params': param_biases,  'weight_decay': bias_weight_decay}]
        optimizer = torch.optim.AdamW(parameters)

    # weight decay scheduler, used for vision transformer proposed by dino.
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, 
        len(data_loader_train),
    )
    qt_schedule = utils.cosine_scheduler(
        args.quantile,
        args.quantile_end,
        args.epochs if args.crops_interact_style=='self_label' else (args.epochs-args.mme_epochs),
        len(data_loader_train),
    )

    # =================== Loss Function =================== 
    if(args.loss_type == "ENTLoss"):
        criterion = EntLoss(args, args.lam1, args.lam2, pqueue=None)
    else:
        criterion = PAWSLoss(args)
    
    loss_scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'runs_{}'.format( args.output_dir.replace('/','').replace('.','') ))
        writer = SummaryWriter(logdir=local_runs)

    # =================== Traning =================== 
    if args.resume: # Resume traning from checkpoint
        checkpoint = torch.load(args.resume, map_location='cpu')
        try:
            model_without_ddp.load_state_dict(checkpoint['model'])
        except:
            ckpt = checkpoint['model']
            unexpected = {"projection_heads.cls_heads.0.0.weight": "projection_heads.last_linear.weight", 
                          "projection_heads.cls_heads.0.0.bias":  "projection_heads.last_linear.bias",
                          "projection_heads.cls_heads.0.1.running_mean": "projection_heads.norm.running_mean", 
                          "projection_heads.cls_heads.0.1.running_var": "projection_heads.norm.running_var", 
                          "projection_heads.cls_heads.0.1.num_batches_tracked": "projection_heads.norm.num_batches_tracked"
                         }
            for k, v in unexpected.items():
                ckpt[v] = ckpt[k]
                del ckpt[k]
            model_without_ddp.load_state_dict(ckpt)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        if 'teacher' in checkpoint:
            teacher_model_without_ddp.load_state_dict(checkpoint['teacher'])
    
    if args.eval_only:
        eval_stats = eval_one_epoch(args, 
                model, data_loader_val, device, 
                logfn=os.path.join(output_dir, 'detail_log.txt')
        )
        np.save(os.path.join(args.output_dir, 'match.npy'), eval_stats['match'])
        np.save(os.path.join(args.output_dir, 'mapped_preds.npy'), eval_stats['mapped_preds'])
        print({k: v for k,v in eval_stats.items() if k not in ['match', 'mapped_preds']})
        return

    if args.inference_only:
        inference(args, model, args.img_path, device, np.load(args.match_path))
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if epoch == args.mme_epochs:
            print("Switching to Harder Multi Crop")
            args.global_crops_scale = (0.14, 1.0)
            args.local_crops_scale = (0.05, 0.14)
            args.aug = 'multicrop'
            data_loader_train.dataset.aug = get_augmentations(args)

        data_loader_train.sampler.set_epoch(epoch)
        
        # train for one epoch
        train_stats = train_one_epoch(args,
            model, criterion, data_loader_train,
            optimizer, device, epoch, 
            set_training_mode=True,
            scaler=loss_scaler,
            logfn=os.path.join(output_dir, 'detail_log.txt'),
            wd_schedule=wd_schedule,
            qt_schedule=qt_schedule,
            teacher_model=teacher_model,
            momentum_schedule=momentum_schedule,
        )

        if args.crops_interact_style == 'label':
            torch.save(args.pseudo_labels, 'pseudo_labels.pth')
            break
        
        # only dim=1000 evaluate unsupervised classification
        if args.dim == 1000:
            eval_stats = eval_one_epoch(args, 
                    model, data_loader_val, device, 
                    logfn=os.path.join(output_dir, 'detail_log.txt')
            )
            print(eval_stats)

        eval_results = eval_knn(args, model, data_loader_val, device, epoch, num_tasks, global_rank)
        
        
        # saving checkpoint
        checkpoint_path = os.path.join(output_dir,'checkpoint.pth')
        save_dict = {
                'model': model_without_ddp.state_dict(),
                'backbone': model_without_ddp.backbone_weights(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
        }
        if args.use_momentum_encoder:
            save_dict.update({
                'teacher': teacher_model_without_ddp.state_dict(),
                'teacher_backbone': teacher_model_without_ddp.backbone_weights(),
            })
        if loss_scaler: # save state dict of loss scaler if using mix precision.
            save_dict.update({'scaler': loss_scaler.state_dict()})
        utils.save_on_master(save_dict, checkpoint_path)

        if (epoch + 1) == args.mme_epochs:
            utils.save_on_master(save_dict, os.path.join(output_dir,'mme_ckpt.pth'))
        
        # logging train stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                if epoch == 0: # write the arguments parameters
                    try:
                        f.write(json.dumps(args.__dict__) + "\n")
                    except:
                        pass
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TWIST', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
