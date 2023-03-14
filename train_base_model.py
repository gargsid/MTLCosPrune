from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# ++++++for pruning
import os, sys
import time
sys.path.append('..')
import time 

import numpy as np

from models.jumbo_vgg import MTLNet, VGG

from nyuv2_data.nyuv2_dataloader_adashare import NYU_v2
from nyuv2_data.pixel2pixel_loss import NYUCriterions
from nyuv2_data.pixel2pixel_metrics import NYUMetrics

from taylor_utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def createdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created!')

def torch_save(obj, name, path):
    torch.save({name:obj}, path)
    print(f'{path} saved!')

def train_mtl(model, train_dataloader, optimizer, scheduler, device):
    model.train()  
    loss_list = {task: [] for task in tasks}
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        x = data['input'].to(device)
        output = model(x)

        loss = 0
        for task in tasks:
            y = data[task].to(device)
            if task + '_mask' in data:
                tloss = criterion_dict[task](output[task], y, data[task + '_mask'].to(device))
            else:
                tloss = criterion_dict[task](output[task], y)
            loss_list[task].append(tloss.item())
            loss += tloss

        loss.backward()
        optimizer.step()

        if args.testing:
            break

    # scheduler.step()
    for task in tasks:
        # logprint(f'Task {} Train Loss: {:.4f}'.format(task[:4], np.mean(loss_list[task])), logs_fp)
        logprint('Task {} Train Loss: {:.4f}'.format(task[:4], np.mean(loss_list[task])), logs_fp)

def validate_mtl(model, val_dataloader, device):
    model.eval()  
    loss_list = {task: [] for task in tasks}
    for i, data in enumerate(val_dataloader):
        x = data['input'].to(device)
        output = model(x)
        for task in tasks:
            y = data[task].to(device)
            if task + '_mask' in data:
                loss = criterion_dict[task](output[task], y, data[task + '_mask'].to(device))
                metric_dict[task](output[task], y, data[task + '_mask'].to(device))
            else:
                loss = criterion_dict[task](output[task], y)
                metric_dict[task](output[task], y)
            loss_list[task].append(loss.item())
        
        if args.testing:
            break
    
    ret_results = {}
    for task in tasks:
        val_results = metric_dict[task].val_metrics()
        ret_results[task] = val_results.copy()
        logprint('Task {} Val Loss: {:.4f}'.format(task[:4], np.mean(loss_list[task])), logs_fp)
        logprint('{}'.format(val_results), logs_fp)
    return np.mean([np.mean(loss_list[task]) for task in tasks]), ret_results

def plot_results(all_results, plots_dir):
    epochs =  all_results['epoch']
    x = np.arange(len(epochs))

    if len(x) > 10:
        step = len(x) // 10
        x_xticks = np.arange(0, len(x), step)
    else:
        x_xticks = x 
    xticks = [epochs[i] for i in x_xticks]


    for m in all_results.keys():
        v = all_results[m]

        plt.figure()
        plt.plot(x, v, label=m)

        plt.xlabel('epochs')
        plt.ylabel(m)
        plt.xticks(x_xticks, xticks, rotation=50)

        plt.tight_layout()
        plt.legend(loc='best')

        plot_path = os.path.join(plots_dir, f'{m}.png')
        plt.savefig(plot_path)
        print('saved to', plot_path)
        plt.close()

parser = argparse.ArgumentParser()

parser.add_argument('--jobid', type=int, default=0)
parser.add_argument('--gpu', type=str, default='gypsum-rtx8000')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--testing', action='store_true')
parser.add_argument('--validate', action='store_true')
parser.add_argument('--logs_dir', type=str, default='assets/experiments/MTLNet_random_init_nyu_trained/run_1')
parser.add_argument('--metric', type=str, default='loss') # 'pixel'
parser.add_argument('--ckpt', type=str, default=None)
args = parser.parse_args() 

logs_dir = args.logs_dir
logs_fp = os.path.join(logs_dir, 'logs.txt')
createdirs(logs_dir)

logprint(f'device:{device}', logs_fp)

tasks = ['segment_semantic','normal','depth_zbuffer']
T = len(tasks)
cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

metrics_names = ['Pixel Acc', 'mIoU', 'abs_err', 'rel_err', 'sigma_1.25', 'sigma_1.25^2', 'sigma_1.25^3', 'Angle Mean', 'Angle Median', 'Angle 11.25', 'Angle 22.5', 'Angle 30']
all_results = {}
for m in metrics_names:
    all_results[m] = []
all_results['epoch'] = []

plots_dir = os.path.join(logs_dir, 'train_plots')
createdirs(plots_dir)

batch_size = 16

dataroot = '/work/siddhantgarg_umass_edu/datasets/nyu_v2/nyu_v2' # change to your data root
dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
test_loader = DataLoader(dataset, 8, shuffle=True, num_workers=args.num_workers)

criterion_dict = {}
metric_dict = {}
for task in tasks:
    criterion_dict[task] = NYUCriterions(task)
    metric_dict[task] = NYUMetrics(task)

s_config = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
model = MTLNet(tasks, cls_num, s_config).to(device)

if not isinstance(args.ckpt, type(None)):
    saved_weights = torch.load(args.ckpt)
    model.load_state_dict(saved_weights)
    logprint(f'ckpt loaded from {args.ckpt}')

gate_modules = ['2', '6', '9', '13', '16', '19', '23', '26', '29', '33', '36', '39', '43']
gate_modules_names = [f'backbone.{m}.weight' for m in gate_modules]

parameters_for_update = []
for name, m in model.named_parameters():
    if name not in gate_modules_names:
        parameters_for_update.append(m)
    else:
        print('skipping', name)
        # print(m)

# sys.exit()

optimizer = torch.optim.Adam(parameters_for_update, lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs*2))
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

save_name = os.path.join(logs_dir, 'trained_model.pth')

if args.validate:
    pass 
else:
    T = args.epochs
    best_val_loss = 1e10
    best_pixel_acc = 0.
    for t in range(T):
        logprint('Epoch {}/{}'.format(t+1, T), logs_fp)
        train_mtl(model, train_loader, optimizer, scheduler, device)
        if (t+1) % 10 == 0 or args.testing:
            val, results = validate_mtl(model, test_loader, device)
            vpx = results['segment_semantic']['Pixel Acc']
            # if val < best_val_loss:
            if args.metric == 'pixel':
                if best_pixel_acc < vpx:
                    # best_val_loss = val 
                    best_pixel_acc = vpx 
                    torch.save(model.state_dict(), save_name)
                    logprint(f'ckpt updated with pixel acc: {best_pixel_acc}', logs_fp)
            else:
                if val < best_val_loss:
                    best_val_loss = val 
                    torch.save(model.state_dict(), save_name)
                    logprint(f'ckpt updated with pixel acc: {best_pixel_acc}', logs_fp) 
                    
            logprint('-'*10, logs_fp)

            for task in tasks:
                for m, v in results[task].items():
                    all_results[m].append(v) 
            all_results['epoch'].append(t+1)
            plot_results(all_results, plots_dir)

        if args.testing:
            break   
        scheduler.step()
        
logprint('---- Validation ----', logs_fp)
model.load_state_dict(torch.load(save_name))
loss, results = validate_mtl(model, test_loader, device)
torch_save(results, 'results', os.path.join(logs_dir, 'results.pth'))