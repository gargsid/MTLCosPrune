import os, sys
import argparse
from tqdm import tqdm
import pickle 
import time  

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.optim as optim

def logprint(log, logs_fp=None):
    if not isinstance(logs_fp, type(None)):
        with open(logs_fp, 'a') as f:
            f.write(log + '\n')
    print(log)

def create_dirs(logs_dir, logs_fp=None):
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
        # os.mkdir(f'{logs_dir}/plots')
        log = f'{logs_dir} created!'
        logprint(log, logs_fp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_pruned_model_sparsity(model):
    zeros = 0. 
    total = 0.
    for k, v in model.state_dict().items():
        if 'mask' in k:
            zeros += (torch.numel(v) - torch.sum(v))
            total += torch.numel(v)
    if total == 0:
        return 0.
    return zeros / total

def calculate_unpruned_model_sparsity(model):
    total_elements = 0.
    zeros = 0.
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            zeros += float(torch.sum(layer.weight == 0))
            total_elements += float(layer.weight.nelement())
        if isinstance(layer, torch.nn.Linear):
            zeros += float(torch.sum(layer.weight == 0))
            total_elements += float(layer.weight.nelement())
    return zeros / total_elements

def save_pickle_obj(pkl_obj, path):
    with open(path, 'wb') as f:
        pickle.dump(pkl_obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        objj = pickle.load(f)
    return objj

def plot_graphs(entries, path, xlabel='', ylabel='', xticks=None):

    plt.figure()

    x = np.arange(len(entries))
    plt.plot(x, entries)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if not isinstance(xticks, type(None)):
        plt.xticks(x, xticks, rotation=80)
        
    plt.tight_layout()

    plt.savefig(path)
    plt.close()
    # print(f'saved to {path}!')


def get_optimizer_with_cosine_schedule(optim_type, model_parameters, learning_rate, epochs):
    if optim_type=='pcgrad':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        optimizer = PCGrad(optim.AdamW(model_parameters, lr=learning_rate), scheduler=scheduler, epochs=epochs) 
        return optimizer, scheduler

    if optim_type=='adamW':
        optimizer = optim.AdamW(model_parameters, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        return optimizer, scheduler

    if optim_type == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0)
        return optimizer, scheduler

def get_cityscapes_data(tasks, batch_size, num_workers):
    dataroot = '/work/siddhantgarg_umass_edu/datasets/cityscapes' # change to your data root
    semantic_task = 'segment_semantic'
    depth_task = "depth_zbuffer"

    dataset = CityScapes(dataroot, 'train', crop_h=224, crop_w=224)
    trainDataLoader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

    dataset = CityScapes(dataroot, 'val', crop_h=224, crop_w=224)
    valDataLoader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

    semantic_loss_fn = CityScapesCriterions(semantic_task)
    depth_loss_fn = CityScapesCriterions(depth_task)

    semantic_metric = CityScapesMetrics(semantic_task)
    depth_metric = CityScapesMetrics(depth_task)

    criterion = {
        k : CityScapesCriterions(k) for k in tasks.keys() 
    }

    metrics = {
        k : CityScapesMetrics(k) for k in tasks.keys
    }

    return trainDataLoader, valDataLoader, criterion, metrics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def get_new_sparsity(current_model_sparsity, prune_steps=0.2):
#     unpruned_ratio = (1. - current_model_sparsity)
#     to_prune = unpruned_ratio * prune_steps 
#     new_unpruned = unpruned_ratio - to_prune 
#     new_sparsity = 1. - new_unpruned
#     return new_sparsity

# sparsity_set = ['sparsity_0']
# sparse_list = [0.]
# current_sparsity = 0

# while current_sparsity < 0.99:
#     new_sparsity = get_new_sparsity(current_sparsity)
#     k = f'sparsity_{new_sparsity:.4f}'
#     sparsity_set.append(k)
#     current_sparsity = new_sparsity
#     sparse_list.append(round(new_sparsity, 4))

# print(sparse_list)
# sys.exit()
