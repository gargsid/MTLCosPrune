"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import os, sys
import time
import time 
import numpy as np

from taylor_utils.taylor_args import get_talyor_args
from taylor_pruning.cosine_pruning_engine import pytorch_pruning, PruningConfigReader, prepare_pruning_list

from models.jumbo_vgg import MTLNet, VGG

from nyuv2_data.nyuv2_dataloader_adashare import NYU_v2
from nyuv2_data.pixel2pixel_loss import NYUCriterions
from nyuv2_data.pixel2pixel_metrics import NYUMetrics

from taylor_utils.utils import *

from ptflops import get_model_complexity_info

def str2bool(v):
    # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_validation_stats(task_losses, pruned_s_config, average_cosine_score, args):

    validation_stats['pruned_s_config'].append(pruned_s_config)
    validation_stats['average_cosine_score'].append(average_cosine_score)

    logprint(f'{"-"*10} best stats for current config {"-"*10}', logs_fp)
    for task in tasks:
        logprint('{}'.format(best_metrics[task]), logs_fp)
    logprint(f'best_val_loss:{best_metrics["best_val_loss"]}', logs_fp)

    for task in tasks:
        validation_stats[task]['loss'].append(task_losses[task])
        for metric, val in best_metrics[task].items():
            if metric not in validation_stats[task].keys():
                validation_stats[task][metric] = [val]
            else:
                validation_stats[task][metric].append(val)

    path = os.path.join(logs_dir, 'validation_stats.pkl')
    save_pickle_obj(validation_stats, path)
    # print('after ending:', validation_stats)
    

def train(args, model, device, train_loader, test_loader, optimizer, epoch, criterion_dict, metric_dict, pruning_engine=None):
# def train(args, model, device, train_loader, optimizer, epoch, criterion, train_writer=None, pruning_engine=None):
    """Train for one epoch on the training set also performs pruning"""
    global global_iteration
    global best_val_loss
    global pruning_iterations

    losses = AverageMeter()

    model.train()

    # for multi-task code
    loss_list = {task: [] for task in criterion_dict.keys()}

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        x = data['input'].to(device)
        output = model(x)

        loss = 0. # total gradient 
        objectives = {} # to calculate param gradient with respect to each objective
        for task in criterion_dict.keys():
            y = data[task].to(device)
            if task + '_mask' in data:
                task_loss = criterion_dict[task](output[task], y, batch_data[task + '_mask'].to(device))
            else:
                task_loss = criterion_dict[task](output[task], y)

            loss_list[task].append(task_loss.item())
            loss += task_loss
            objectives[task] = task_loss

        # losses.update(loss.item(), x.size(0))
        # logprint(f'train batch_idx: {batch_idx}/{len(train_loader)} losses: {losses.val:.4f}, {losses.avg:.4f}', logs_fp)

        global_iteration = global_iteration + 1
        # print('global_iteration:', global_iteration)

        # checking if pruning will happen in this step to store the best results of the current config before pruning
        if global_iteration % pruning_engine.frequency == 0 and global_iteration != 0:
            task_losses, val_metrics = validate(args, test_loader, model, optimizer, device, criterion_dict, metric_dict, 0)
            
            pruned_s_config = pruning_engine._get_pruned_s_config()
            average_cosine_score = pruning_engine._get_average_cosine_scores() # layer-wise

            get_validation_stats(task_losses, pruned_s_config.copy(), average_cosine_score.copy(), args)

            # resetting best_val_metrics after pruning
            best_val_loss = 1000.
            best_metrics = {task: {} for task in tasks}
            best_metrics['best_val_loss'] = best_val_loss
            logprint(f'#filters just before pruning: {sum(pruned_s_config)}', logs_fp)

            # model will be in eval() state after validation
            model.train()

        if args.pruning:
            pruning_engine.do_cosine_step(objectives, model, optimizer, loss=loss.item())
            all_neuron_units, unpruned_neuron_units = pruning_engine._count_number_of_neurons()
            
            if global_iteration % pruning_engine.frequency == 0 and global_iteration != 0:
                pruned_s_config = pruning_engine._get_pruned_s_config()
                logprint(f'#filters remaining after pruning:{sum(pruned_s_config)}', logs_fp)
                pruning_iterations += 1

                unpruned_neuron_units = sum(pruned_s_config)
            
            if args.cosine_ranking:
                if unpruned_neuron_units == 65:
                    logprint(f'65 filters remaining for cos-rank. exiting...', logs_fp)
                    sys.exit()
            elif unpruned_neuron_units == 0:
                logprint(f'No filters remaining for total-rank. exiting...', logs_fp)
                sys.exit()

        loss.backward()
        optimizer.step()
        # print('pruning_iterations', pruning_iterations)
        
        if args.testing:
            break
    for task in tasks:
        logprint('Task {} Train Loss: {:.4f}'.format(task[:4], np.mean(loss_list[task])), logs_fp)

def validate(args, test_loader, model, optimizer, device, criterion_dict, metric_dict, epoch, train_writer=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    loss_list = {task: [] for task in criterion_dict.keys()}

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # data, target = data_test
            # data = data_test['input'] # for multi-task code
            x = data['input'].to(device)
            # print('x:', x.shape)
            output = model(x)

            # for multi-task code 
            total_loss = 0. 
            for task in criterion_dict.keys():
                y = data[task].to(device)
                if task + '_mask' in data:
                    loss = criterion_dict[task](output[task], y, data[task + '_mask'].to(device))
                    metric_dict[task](output[task], y, data[task + '_mask'].to(device))
                else:
                    loss = criterion_dict[task](output[task], y)
                    # metric_dict[task](output[task].detach().cpu(), y.cpu())
                    metric_dict[task](output[task], y)

                total_loss += loss.item()
                loss_list[task].append(loss.item())

            losses.update(total_loss, x.size(0))

            # logprint(f'val batch_idx: {i}/{len(test_loader)} losses: {losses.val:.4f}, {losses.avg:.4f}', logs_fp)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            torch.cuda.empty_cache()

            if args.testing:
                break

    task_loss = {task: np.mean(loss_list[task]) for task in criterion_dict.keys()}

    combined_loss = 0
    for task in tasks:
        combined_loss += task_loss[task]

    current_metrics = {}

    logprint(f'{"-"*10} validating {"-"*10}', logs_fp)
    for task in criterion_dict.keys():
        val_results = metric_dict[task].val_metrics()
        logprint('Task {} Val Loss: {:.4f}'.format(task[:4], task_loss[task]), logs_fp)
        logprint('{}'.format(val_results), logs_fp)
        current_metrics[task] = val_results.copy()
    logprint(f'combined_loss:{combined_loss}', logs_fp)

    global best_val_loss

    pruned_checkpoint_path = os.path.join(logs_dir, 'pruned_models', f'pruned_model_iter_{pruning_iterations}.pth')
    create_dirs(os.path.join(logs_dir, 'pruned_models'), logs_fp)

    if combined_loss < best_val_loss:

        best_val_loss = combined_loss
        for task in tasks:
            best_metrics[task] = current_metrics[task].copy()
            best_metrics['best_val_loss'] = combined_loss

        checkpoint = model.state_dict()
        torch.save(checkpoint, pruned_checkpoint_path)
        # torch.save(optimizer.state_dict(), os.path.join(logs_dir, 'optim_state.pth'))
        
        logprint(f'best_val_loss: {best_val_loss} (ckpt saved): {pruned_checkpoint_path}', logs_fp)
        del checkpoint
        torch.cuda.empty_cache()

    return task_loss, metric_dict

def main():

    best_prec1 = 0
    global global_iteration
    global group_wd_optimizer

    global pruning_iterations 
    
    global_iteration = 0
    pruning_iterations = 0

    # args = parser.parse_args()
    args = get_talyor_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    global logs_dir 
    global logs_fp 

    logs_dir = args.logs_dir
    logs_fp = os.path.join(logs_dir, 'logs.txt')

    create_dirs(logs_dir, logs_fp)
    
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    logprint(f'device:{device}', logs_fp)

    global tasks
    global cls_num
    tasks = ['segment_semantic','normal','depth_zbuffer']
    T = len(tasks)
    cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

    # # dataset loading section
    dataroot = '/work/siddhantgarg_umass_edu/datasets/nyu_v2/nyu_v2' # change to your data root

    dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
    print('#train', len(dataset.triples))
    # train_loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

    dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
    print('#test', len(dataset.triples))
    # test_loader = DataLoader(dataset, 8, shuffle=True, num_workers=args.num_workers)

    sys.exit()

    s_config = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model = MTLNet(tasks, cls_num, s_config)
    
    # trained_weights = '/work/siddhantgarg_umass_edu/MTLCosPrune/assets/experiments/MTLNet_random_init_nyu_trained/run_1K_lr0.0001_half_cosdecay/trained_model.pth'
    # checkpoint = torch.load(trained_weights)
    # model.load_state_dict(checkpoint)

    trained_weights = '/work/siddhantgarg_umass_edu/pruning/assets/experiments/taylor_pruning/vgg_base_mt_nyu/trained_nyu_mtl_base_vgg_pretrained_true.pth'
    checkpoint = torch.load(trained_weights)
    module_map = {'0': '0', '2' : '3', '5':'7', '7':'10', '10':'14', '12':'17', '14':'20', '17':'24', '19':'27', '21':'30', '24':'34', '26':'37','28':'40'}
    for k, v in checkpoint.items():
        if 'backbone' in k:
            new_k = k.split('.')
            if new_k[1] in module_map.keys():
                new_k[1] = module_map[new_k[1]]
                new_k = '.'.join(new_k)
                model.state_dict()[new_k].copy_(checkpoint[k])
                # logprint(f'copied tensor from {k} to {new_k}')
        else:
            model.state_dict()[k].copy_(checkpoint[k])
            # logprint(f'copied tensor from {k} to {k}')
    del checkpoint 
    torch.cuda.empty_cache()
    logprint(f'trained weights loaded from {trained_weights}')
    
    # standard to gated vgg
    module_map = {'0': '0', '2' : '3', '5':'7', '7':'10', '10':'14', '12':'17', '14':'20', '17':'24', '19':'27', '21':'30', '24':'34', '26':'37','28':'40'}
    
    logprint(f'trained weights loaded from {trained_weights}')

    

    
    criterion_dict = {}
    metric_dict = {}
    for task in tasks:
        criterion_dict[task] = NYUCriterions(task)
        metric_dict[task] = NYUMetrics(task)

    print("model is defined")

    if use_cuda and not args.mgpu:
        model = model.to(device)

    # remove updates from gate layers, because we want them to be 0 or 1 constantly
    gate_modules = ['2', '6', '9', '13', '16', '19', '23', '26', '29', '33', '36', '39', '43']
    gate_modules_names = [f'backbone.{m}.weight' for m in gate_modules]
    
    filter_param_counts = {}

    parameters_for_update = []
    parameters_for_update_named = []
    for name, m in model.named_parameters():
        if name not in gate_modules_names:
            parameters_for_update.append(m)
            parameters_for_update_named.append((name, m))
            # print('pushed for updates:', name, m.shape)
        else:
            print("skipping parameter", name, "shape:", m.shape)
        
        if 'backbone' in name and 'weight' in name and name not in gate_modules_names:
            filter_param_counts[name] = m.shape[1] * m.shape[2] * m.shape[3]
            # print(f'filter_param_counts[{name}]:', filter_param_counts[name])
    # print('filter weight counts:', list(filter_param_counts.values()), len(list(filter_param_counts.values())))
    per_filter_params_count = list(filter_param_counts.values())

    total_size_params = sum([np.prod(par.shape) for par in parameters_for_update])
    print("Total number of parameters, w/o usage of bn consts: ", total_size_params)

    # optimizer = optim.SGD(parameters_for_update, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)
    optimizer = optim.AdamW(parameters_for_update, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*5)

    # initializing parameters for pruning
    # we can add weights of different layers or we can add gates (multiplies output with 1, useful only for gradient computation)
    pruning_engine = None
    if args.pruning:
        pruning_settings = dict()
        if not (args.pruning_config is None):
            pruning_settings_reader = PruningConfigReader()
            pruning_settings_reader.read_config(args.pruning_config)
            pruning_settings = pruning_settings_reader.get_parameters()

        has_attribute = lambda x: any([x in a for a in sys.argv])

        if has_attribute('pruning-momentum'):
            pruning_settings['pruning_momentum'] = vars(args)['pruning_momentum']
        if has_attribute('pruning-method'):
            pruning_settings['method'] = vars(args)['pruning_method']

        pruning_parameters_list = prepare_pruning_list(pruning_settings, model, model_name=args.model, pruning_mask_from=args.pruning_mask_from, name=args.name)
        print("Total pruning layers:", len(pruning_parameters_list))

        log_folder = None
        pruning_engine = pytorch_pruning(pruning_parameters_list, arguments=args, pruning_settings=pruning_settings, tasks=tasks, per_filter_params_count=per_filter_params_count, logs_dir=logs_dir, logs_fp=logs_fp)

        pruning_engine.dataset = args.dataset
        pruning_engine.model = args.model
        pruning_engine.pruning_mask_from = args.pruning_mask_from

    train_writer = None

    global best_val_loss
    global best_metrics

    best_val_loss = 1000.
    best_metrics = {task: {} for task in tasks}
    best_metrics['best_val_loss'] = best_val_loss

    global validation_stats # to store the losses and val_metrics along with the model FLOPs, and #params
    validation_stats = {}
    for task in tasks:
        validation_stats[task] = {}
        validation_stats[task]['loss'] = []

    validation_stats['flops'] = []
    validation_stats['params'] = []
    validation_stats['pruned_s_config'] = []
    validation_stats['average_cosine_score'] = []

    global pruned_checkpoint_path
    # pruned_checkpoint_path = os.path.join(logs_dir, 'pruned_gated_model.pth')

    print('validation_stats: ',validation_stats)

    # print('validating loaded model')
    # task_losses, val_metrics = validate(args, test_loader, model, optimizer, device, criterion_dict, metric_dict, 0)
    
    for epoch in range(1, args.epochs + 1):

        logprint(f'epoch:{epoch}', logs_fp)
        
        start = time.time()
        train(args, model, device, train_loader, test_loader, optimizer, epoch, criterion_dict, metric_dict, pruning_engine=pruning_engine)

        # evaluate on validation set
        task_losses, val_metrics = validate(args, test_loader, model, optimizer, device, criterion_dict, metric_dict, epoch)
        # written to logs only.
        scheduler.step()

        end = time.time()
        time_taken = f'{ (end - start)//3600} hrs , {(end - start) // 60} mins; avg: {((end-start)/60)/epoch} mins/epoch'
        logprint(f'time_elapsed after epoch {epoch}: {time_taken}', logs_fp)

        if (epoch % 10) == 0:
            optimizer = optim.AdamW(parameters_for_update, lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs*5)
            logprint(f'epoch:{epoch}/{args.epochs}.. resetting optimizer and scheduler', logs_fp)

if __name__ == '__main__':
    main()
