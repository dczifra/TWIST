# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from imagenet_lmdb import ImageNetLMDB as lmdb
from PIL import Image
from PIL import ImageFile
import random
import os
import glob
import torchvision
from torchvision.datasets.folder import default_loader
from collections import defaultdict
ImageFile.LOAD_TRUNCATED_IMAGES = True

from augmentation import get_augmentations, TestAugmentation

class ImageNetLMDB(lmdb):
    def __init__(self, root, list_file, aug):
        super(ImageNetLMDB, self).__init__(root, list_file, ignore_label=False)
        self.aug = aug

    def __getitem__(self, index):
        img, target = super(ImageNetLMDB, self).__getitem__(index)
        imgs = self.aug(img)
        return imgs, target, index

class ImageNet(datasets.ImageFolder):
    def __init__(self, root, aug, train=True):
        print(os.path.join(root, 'train' if train else 'val'))
        super(ImageNet, self).__init__(os.path.join(root, 'train' if train else 'val'))
        self.aug = aug

    def __getitem__(self, index):
        img, target = super(ImageNet, self).__getitem__(index)
        imgs = self.aug(img)
        return imgs, target, index

class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, aug, train=True):
        super(CIFAR10, self).__init__(root, train)
        self.aug = aug

    def __getitem__(self, index):
        img, target = super(CIFAR10, self).__getitem__(index)
        imgs = self.aug(img)
        return imgs, target, index

def get_train_with_val_augmentation(args, num_tasks, global_rank):
    val_aug = TestAugmentation(args).aug

    if args.dataset == "imagenet_lmdb":
        dataset_train = ImageNetLMDB(args.data_path, 'train.lmdb', val_aug)
    elif args.dataset == "imagenet":
        dataset_train = ImageNet(args.data_path, val_aug, train=True)
    elif args.dataset == "cifar10":
        dataset_train = CIFAR10(args.data_path, val_aug, train=True)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        )

    return data_loader_train
    
def get_datasets(args, num_tasks, global_rank):
    dataset_train, dataset_val = None,None
    data_loader_train, data_loader_val = None,None
    sampler_train, sampler_val = None,None

    if args.dataset == "imagenet_lmdb":
        dataset_train = ImageNetLMDB(args.data_path, 'train.lmdb', get_augmentations(args))
    if args.dataset == "imagenet":
        dataset_train = ImageNet(args.data_path, get_augmentations(args))
    elif args.dataset == "cifar10":
        dataset_train = CIFAR10(args.data_path, get_augmentations(args))


    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # prepare evaluation data loader to make unsupervised classification
    if args.dim == 1000 or args.dim == 10 or args.eval_only or 1:
        val_aug = TestAugmentation(args).aug

        if args.dataset == "imagenet_lmdb":
            dataset_val = ImageNetLMDB(args.data_path, 'val.lmdb', val_aug)
        elif args.dataset == "imagenet":
            dataset_val = ImageNet(args.data_path, val_aug, train=False)
        elif args.dataset == "cifar10":
            dataset_val = CIFAR10(args.data_path, val_aug, train=False)

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    
    return dataset_train, dataset_val, data_loader_train, data_loader_val, sampler_train, sampler_val
    
