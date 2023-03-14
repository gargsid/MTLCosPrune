import os, sys
import argparse
from tqdm import tqdm
import pickle 
import time  

import torch
import pandas as pd 

from ptflops import get_model_complexity_info

from models.jumbo_vgg import MTLNet, VGG
from taylor_utils.utils import * 

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--s_config", nargs="+", default='64 64 128 128 256 256 256 512 512 512 512 512 512')
parser.add_argument('--compare_plots', action='store_true')
parser.add_argument('--write_to_csv', action='store_true')
parser.add_argument('--mean_std_plots_by_metric', action='store_true')
parser.add_argument('--mean_std_plots', action='store_true')
parser.add_argument('--config_plots', action='store_true')
parser.add_argument('--plot_params', action='store_true')
parser.add_argument('--analyzing_metrics', action='store_true')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

s_config = args.s_config
s_config = s_config[0].split(' ')
s_config = [int(s) for s in s_config]

tasks = ['segment_semantic','normal','depth_zbuffer']
T = len(tasks)
cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

metrics_names = ['Pixel Acc', 'mIoU', 'abs_err', 'rel_err', 'sigma_1.25', 'sigma_1.25^2', 'sigma_1.25^3', 'Angle Mean', 'Angle Median', 'Angle 11.25', 'Angle 22.5', 'Angle 30']
metric_to_task = {}
for i in range(2):
    metric_to_task[metrics_names[i]] = 'Semantic Segmentation'
for i in range(2,7):
    metric_to_task[metrics_names[i]] = 'Depth Estimation'
for i in range(7,len(metrics_names)):
    metric_to_task[metrics_names[i]] = 'Surface Normal Prediction'

def torch_load(name, path):
    ckpt = torch.load(path)
    return ckpt[name]

def create_validation_stats_with_flops(val_stats_path):
    validation_stats = load_pickle(val_stats_path)

    save_path_dir = val_stats_path.split('/')[:-1]
    save_path_dir = '/'.join(save_path_dir)
    save_path = os.path.join(save_path_dir, 'validation_stats_with_flops.pkl')
    
    all_pruned_s_configs = validation_stats['pruned_s_config']
    all_filters = [np.sum(s) for s in all_pruned_s_configs]

    last_idx = None
    for i, pruned_s_config in enumerate(all_pruned_s_configs):
        if 0 in pruned_s_config:
            last_idx = i 
            break
    if isinstance(last_idx, type(None)):
        last_idx = len(all_filters)

    flops_list = []
    params_list = []

    for i in range(last_idx):
        s_config = all_pruned_s_configs[i]
        en_model = MTLNet(tasks, cls_num, s_config).to(device)
        macs, params = get_model_complexity_info(en_model, (3, 321, 321), print_per_layer_stat=False, verbose=False, as_strings=False)
        del en_model
        torch.cuda.empty_cache()
        print(i, macs, params)

        flops_list.append(macs)
        params_list.append(params)

        validation_stats['flops_list'] = flops_list 
        validation_stats['params_list'] = params_list 

        save_pickle_obj(validation_stats, save_path)
        print('saved to', save_path)

def get_params_to_x_map(all_params):
    all_params.sort(reverse=True)

    params_to_x_axis = {}
    for i in range(len(all_params)):
        p = all_params[i]
        if p not in params_to_x_axis.keys():
            params_to_x_axis[p] = i

    # print(params_to_x_axis)
    return params_to_x_axis

def compare_plots():

    mtl_exp = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/'
    total_dir = os.path.join(mtl_exp, 'total_ranking/run_1')
    cos_dir = os.path.join(mtl_exp, 'cosine_ranking/run_1')

    # create_validation_stats_with_flops(os.path.join(total_dir, 'validation_stats.pkl'))
    # create_validation_stats_with_flops(os.path.join(cos_dir, 'validation_stats.pkl'))
    path_to_label = {
        os.path.join(total_dir, 'validation_stats_with_flops.pkl') : 'total',
        os.path.join(cos_dir, 'validation_stats_with_flops.pkl') : 'cos-rank',
    }
    plots_dir = os.path.join(cos_dir, 'plots', 't1_c1')
    create_dirs(plots_dir)

    paths = list(path_to_label.keys())
    val_stats = {}
    all_params = []
    for path in paths:
        val_stats[path] = load_pickle(path)
        params = val_stats[path]['params_list']
        params = [p/1e6 for p in params]
        all_params += params.copy()

    print('all_params:', all_params)

    params_to_x_axis = get_params_to_x_map(all_params.copy())

    all_params = list(params_to_x_axis.keys())
    print('all_params',len(all_params))
    step = len(all_params) // 10
    x_xticks = [i for i in range(0, len(all_params), step)]
    if len(all_params) - 1 not in x_xticks:
        x_xticks.append(len(all_params) - 1)
    print('x_xticks:', x_xticks)
    xticks = [f'{all_params[i]:.2f}M' for i in x_xticks]
    print('xticks:', xticks)

    for task in tasks:
        for metric in val_stats[path][task].keys():
            
            # plt.figure(figsize=(20, 15), dpi=300) 

            for path in paths:
                results = val_stats[path][task][metric]
                params = val_stats[path]['params_list']
                params = [p/1e6 for p in params]

                plot_results = []
                x_locations = []

                for r, p in zip(results, params):
                    # intp = int(p/1e6)
                    if p in params_to_x_axis.keys():
                        x_locations.append(params_to_x_axis[p])
                        plot_results.append(r)
                        # print('r:', r, 'p:', p, 'x:', params_to_x_axis[p])
                
                print(metric)
                print(x_locations)
                print(plot_results)
                plt.plot(x_locations, plot_results, label=path_to_label[path])

            plt.xlabel('Params')
            if metric == 'Pixel Acc':
                plt.ylabel(f'{metric}(%)')
            else: 
                plt.ylabel(f'{metric}')
            plt.title(f'{metric_to_task[metric]}')

            if not isinstance(xticks, type(None)):
                plt.xticks(x_xticks, xticks, rotation=50)
                
            plt.tight_layout()
            plt.legend(loc='best')

            plot_path = os.path.join(plots_dir, f'{task}_{metric}.pdf')
            plt.savefig(plot_path)
            print('saved to', plot_path)
            plt.close()

if args.compare_plots:
    compare_plots()

def transpose(l1):
    l2 =[[row[i] for row in l1] for i in range(len(l1[0]))]
    return l2

# def write_to_csv(base_dir):
def write_to_csv(input_path, output_path):

    import pandas as pd

    for i in range(1, 7):
        # path = os.path.join(base_dir, f'run_{i}', 'validation_stats_with_flops.pkl')
        # val_stats = load_pickle(path)
        val_stats = load_pickle(input_path)
        columns_list = ['params_list']
        data = []

        data.append(val_stats['params_list'])

        for task in tasks:
            for metric in val_stats[task].keys():
                print(f'{task[:4]}_{metric}', len(val_stats[task][metric]))
                data.append(val_stats[task][metric])
                columns_list.append(f'{task[:4]}_{metric}')

        df_data = transpose(data)
        # print('df_data:',df_data)
        df = pd.DataFrame(df_data, columns=columns_list)
        # path = os.path.join(base_dir, f'run_{i}/val_summary.csv')
        # df.to_csv(path)
        df.to_csv(output_path)

if args.write_to_csv:
    # base_dir = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/total_ranking'
    # base_dir = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking'
    # input_path = '/work/siddhantgarg_umass_edu/pruning/assets/experiments/taylor_pruning/mt_base_vgg_nyu/total_gradient_importance/run_6/validation_stats_with_flops.pkl'
    input_path = '/work/siddhantgarg_umass_edu/pruning/assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_5/validation_stats_with_flops.pkl'
    output_path = '/work/siddhantgarg_umass_edu/MTLCosPrune/assets/val_summaries.csv'
    # write_to_csv(base_dir)
    write_to_csv(input_path, output_path)

# def latex_print():
#     methods = ['total', 'cosine']
#     dfs = {
#         'total' : pd.read_csv('assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking_imagenet_pretrained/run_1_6_summaries/report_mean_total.csv'),
#         'cosine' : pd.read_csv('assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking_imagenet_pretrained/run_1_6_summaries/report_mean_cosine.csv')
#     }

#     print_ms = ['params', 'Pixel Acc', 'rel_err', 'Angle Mean']
#     data = []


# def plot_val_summaries(val_summaries, baseline_summaries, plots_dir, metrics_names=metrics_names):
def plot_val_summaries(val_summaries, plots_dir, metrics_names=metrics_names):

    create_dirs(plots_dir)

    all_params = []
    for method in val_summaries.keys():
        params = val_summaries[method]['mean_params']
        params = [p/1e6 for p in params]
        all_params += params.copy()

    print('all_params:', all_params)

    params_to_x_axis = get_params_to_x_map(all_params.copy())

    all_params = list(params_to_x_axis.keys())
    # print('all_params',len(all_params))
    step = len(all_params) // 10
    x_xticks = [i for i in range(0, len(all_params), step)]
    # if len(all_params) - 1 not in x_xticks:
    #     x_xticks.append(len(all_params) - 1)
    # print('x_xticks:', x_xticks)
    xticks = [f'{all_params[i]:.2f}M' for i in x_xticks]
    # print('xticks:', xticks)

    # colors = {
    #     'Taylor' : 'darkorange',
    #     'CosPrune' : 'royalblue'
    # }
    # labels = {
    #     'Taylor' : 'Taylor',
    #     'CosPrune' : 'CosPrune'
    # }

    colors = {
        'total' : 'darkorange',
        'cosine' : 'royalblue'
    }
    labels = {
        'total' : 'Taylor',
        'cosine' : 'CosPrune'
    }

    for metric in metrics_names:

        plt.figure()

        for method in val_summaries.keys():
            
            x_mean = val_summaries[method][f'mean_{metric}']
            x_std = val_summaries[method][f'std_{metric}']
            # x_base = baseline_summaries[method][f'mean_{metric}']
            

            params = val_summaries[method]['mean_params']
            params = [p/1e6 for p in params]

            mean_y = []
            std_y = []
            # base_y = []
            x_locations = []

            # for p, my, sy, by in zip(params, x_mean, x_std, x_base):
            for p, my, sy in zip(params, x_mean, x_std):
                if p in params_to_x_axis.keys():
                    x_locations.append(params_to_x_axis[p])
                    mean_y.append(my)
                    std_y.append(sy)
                    # base_y.append(by)
            # print(method, metric, p, my) 

            x_locations = np.array(x_locations)
            mean_y = np.array(mean_y)
            std_y = np.array(std_y)
            # base_y = np.array(base_y)

            if 'Pixel' in metric:
                mean_y *= 100.
                std_y *= 100.
                # base_y *= 100.

            # print('x:', len(x_locations), x_locations)
            # print('mean:', len(mean_y), mean_y)
            # print('std:', len(std_y), std_y)
            plt.plot(x_locations, mean_y, label=labels[method], color=colors[method])
            plt.fill_between(x_locations, mean_y-std_y, mean_y+std_y, alpha=0.2, color=colors[method])
            # plt.plot(x_locations, base_y, label=method + ' iter. pruning', color=colors[method], linestyle='dashed')

            # break
        
        plt.xlabel('#Params (M)')
        if 'Pixel' in metric:
            plt.ylabel(f'{metric}(%)')
        else:
            plt.ylabel(f'{metric}')
        plt.title(f'{metric_to_task[metric]}')

        if not isinstance(xticks, type(None)):
            plt.xticks(x_xticks, xticks, rotation=50)
            
        plt.tight_layout()
        plt.legend(loc='best')

        plot_path = os.path.join(plots_dir, f'{metric}.pdf')
        plt.savefig(plot_path)
        print('saved to', plot_path)
        plt.close()

def mean_std_plots(base_dir='assets/experiments/taylor_pruning/mt_base_vgg_nyu'):

    methods = ['total', 'cosine']
    val_summaries = {'total' : {}, 'cosine' : {}}
    for method in methods:
        val_summaries[method]['params'] = []
        for m in metrics_names:
            val_summaries[method][m] = []

    for method in methods:
        print('method', method)

        for i in range(1, 7):
            logs_dir = os.path.join(base_dir, f'{method}_ranking/run_{i}')
            # create_validation_stats_with_flops(os.path.join(logs_dir, 'validation_stats.pkl'))

            path = os.path.join(logs_dir, 'validation_stats_with_flops.pkl')

            val_stats = load_pickle(path)
            # print(val_stats.keys())
            # sys.exit()

            val_summaries[method]['params'].append(val_stats['params_list']) 
            n = len(val_stats['params_list'])   

            for task in tasks:
                for metric in val_stats[task].keys():
                    if metric in metrics_names:
                        val_summaries[method][metric].append(val_stats[task][metric][:n])

    mean_summaries = {}
    for method in methods:
        print('method:', method)
        data = []
        column_names = [] 

        mean_summaries[method] = {}
        # print('params:', len(val_summaries[method]['params']))
        for m in val_summaries[method].keys():
            data = np.array(val_summaries[method][m])
            mean_d = np.mean(data, axis=0)
            std_d = np.std(data, axis=0)
            print(m, data.shape, mean_d.shape, std_d.shape, mean_d[-1])

            mean_summaries[method][f'mean_{m}'] = mean_d
            mean_summaries[method][f'std_{m}'] = std_d
        
        mean_summaries[method]['param_red'] = []
        bp = mean_summaries[method]['mean_params'][0]
        for p in mean_summaries[method]['mean_params']:
            red = 100 * ((bp - p)/bp)
            mean_summaries[method]['param_red'].append(red)
        # df = pd.DataFrame(mean_summaries[method])
        # df.to_csv(f'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_1_6_summaries/report_mean_{method}.csv')
        # print(df)
    # sys.exit()

    plots_dir = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_1_6_summaries/report_plots'
    plot_val_summaries(mean_summaries, plots_dir)

if args.mean_std_plots:
    mean_std_plots()

def mean_std_plots_by_metric(val_stats_metric_name, base_dir='assets/experiments/taylor_pruning/mt_base_vgg_nyu'):

    methods = ['total', 'cosine']
    val_summaries = {'total' : {}, 'cosine' : {}}
    for method in methods:
        val_summaries[method]['params'] = []
        val_summaries[method][val_stats_metric_name] = []

    for method in methods:
        print('method', method)

        for i in range(1, 7):
            logs_dir = os.path.join(base_dir, f'{method}_ranking/run_{i}')
            path = os.path.join(logs_dir, 'validation_stats_with_flops.pkl')

            val_stats = load_pickle(path)

            val_summaries[method]['params'].append(val_stats['params_list']) 
            n = len(val_stats['params_list'])   

            val_summaries[method][val_stats_metric_name].append(val_stats[val_stats_metric_name])
            # print(val_stats[val_stats_metric_name])

    mean_summaries = {}
    for method in methods:
        print('method:', method)
        mean_summaries[method] = {}
        # print('params:', len(val_summaries[method]['params']))
        for m in val_summaries[method].keys():
            data = np.array(val_summaries[method][m])
            if m == 'average_cosine_score':
                data = np.clip(data, -1., 1.)
            # print('data:', data.shape)
            mean_d = np.mean(data, axis=0)
            std_d = np.std(data, axis=0)

            if m == 'average_cosine_score':
                mean_d = np.mean(mean_d, axis=-1)
                std_d = np.mean(std_d, axis=-1)
            # print(m, data.shape, mean_d.shape, std_d.shape, mean_d[-1])

            mean_summaries[method][f'mean_{m}'] = mean_d
            mean_summaries[method][f'std_{m}'] = std_d

            print('method:', method, 'm:', m, 'mean:', mean_d.shape, 'std:', std_d.shape )

    plots_dir = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_1_6_summaries/plots'
    plot_val_summaries(mean_summaries, plots_dir, metrics_names=[val_stats_metric_name])

if args.mean_std_plots_by_metric:
    mean_std_plots_by_metric('average_cosine_score')

def get_mean_std_metrics(base_dir, runs=6):
    val_summary = {}
    for m in metrics_names:
        val_summary[m] = []

    for run in range(1, runs+1):
        results_path = os.path.join(base_dir, f'run_{run}', 'results.pth')
        run_results = torch_load('results', results_path)
        for task in tasks:
            for m in run_results[task].keys():
                if m in metrics_names:
                    val_summary[m].append(run_results[task][m])
    
    # print('config_dir', base_dir)
    # for m in ['Pixel Acc']:
    #     print(val_summary[m])

    mean_summary = {}
    for m in metrics_names:
        mean_summary[f'mean_{m}'] = np.mean(np.array(val_summary[m]))
        mean_summary[f'std_{m}'] = np.std(np.array(val_summary[m]))

    # sys.exit()
    
    return mean_summary.copy()


def config_plots():
    methods = ['Taylor', 'CosPrune']
    baselines = ['Iterative Taylor', 'Iterative CosPrune']

    config_dirs = {
        'Taylor' : [(0, 0.0001), (16, 0.0005), (17, 0.0001), (25, 0.001), (29, 0.001), (31, 0.001), (36, 0.001), (39, 0.001), (40, 0.001)],
        'CosPrune' : [(0, 0.0001), (4, 0.0001), (7, 0.0001), (9, 0.0001), (11, 0.0005), (18, 0.0001), (28, 0.001), (29, 0.0001), (32, 0.001)] 
        # 'cosine' : [(0, 0.0001), (4, 0.0001), (9, 0.0001), (11, 0.001), (18, 0.0001), (28, 0.001), (29, 0.0001), (32, 0.001)] 
    }
    base_dirs = {
        'Taylor' : 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/total_ranking/run_3/',
        'CosPrune' : 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_3/'
    }

    val_stats = {
        'Taylor' : load_pickle(os.path.join(base_dirs['Taylor'], 'validation_stats_with_flops.pkl')),
        'CosPrune' : load_pickle(os.path.join(base_dirs['CosPrune'], 'validation_stats_with_flops.pkl'))
    }

    params_dict = {
        'Taylor' : val_stats['Taylor']['params_list'],
        'CosPrune' : val_stats['CosPrune']['params_list'],
    }

    val_summaries = {}
    val_summaries = {'Taylor' : {}, 'CosPrune' : {}}
    baseline_summaries = {'Taylor' : {}, 'CosPrune' : {}}

    for method in methods:
        data = []
        column_names = []
        column_names.append('mean_params')
        column_names.append('lr')
        for m in metrics_names:
            column_names.append(m)

        val_summaries[method]['mean_params'] = []
        # print(val_stats[method].keys())
        # for task in tasks:
        #     print(val_stats[method][task].keys())
        for m in metrics_names:
            
            val_summaries[method][f'mean_{m}'] = []
            val_summaries[method][f'std_{m}'] = []

            baseline_summaries[method][f'mean_{m}'] = []
        
        for config_index, lr in config_dirs[method]:
            val_summaries[method]['mean_params'].append(params_dict[method][config_index])
            row = [params_dict[method][config_index], lr]
            
            if method == 'Taylor':
                runs_dir = os.path.join(base_dirs[method], f'total_ranking_configs_lr_{lr}_epochs_500', f'config_{config_index}')
            elif method == 'CosPrune':
                runs_dir = os.path.join(base_dirs[method], f'cosine_ranking_configs_lr_{lr}_epochs_500', f'config_{config_index}')
            
            run_summary = get_mean_std_metrics(runs_dir)

            for m in metrics_names:
                val_summaries[method][f'mean_{m}'].append(run_summary[f'mean_{m}'])
                val_summaries[method][f'std_{m}'].append(run_summary[f'std_{m}'])
                
                row.append(run_summary[f'mean_{m}'])

                for task in tasks:
                    if m in val_stats[method][task].keys():
                        v = val_stats[method][task][m][config_index]
                        baseline_summaries[method][f'mean_{m}'].append(v)
            data.append(row)
            
            # print('runs_dir:', runs_dir)
            # for m in ['Pixel Acc']:
            #     print(val_summaries[method][f'mean_{m}'])
            #     print(val_summaries[method][f'std_{m}'])
        # df = pd.DataFrame(data, columns=column_names)
        # df.to_csv(f'assets/config_{method}.csv')
        # print(df)
    # sys.exit()

    plots_dir = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_3/config_summaries_report'
    plot_val_summaries(val_summaries, baseline_summaries, plots_dir)
    # plot_val_summaries(val_summaries, plots_dir)

if args.config_plots:
    config_plots()

def plot_params(base_dir='assets/experiments/taylor_pruning/mt_base_vgg_nyu'):
    
    methods = ['total', 'cosine']
    val_summaries = {'total' : {}, 'cosine' : {}}
    for method in methods:
        val_summaries[method]['params'] = []

    for method in methods:
        # print('method', method)

        for i in range(1, 7):
            logs_dir = os.path.join(base_dir, f'{method}_ranking_imagenet_pretrained/run_{i}')
            path = os.path.join(logs_dir, 'validation_stats_with_flops.pkl')

            val_stats = load_pickle(path)

            val_summaries[method]['params'].append(val_stats['params_list']) 
            n = len(val_stats['params_list'])   
    
    labels = {
        'total' : 'Taylor',
        'cosine' : 'CosPrune'
    }

    colors = {
        'total' : 'darkorange',
        'cosine' : 'royalblue'
    }

    mean_summaries = {}
    for method in methods:
        print('method:', method)
        mean_summaries[method] = {}
        # print('params:', len(val_summaries[method]['params']))
        for m in val_summaries[method].keys():
            print(m)
            data = np.array(val_summaries[method][m])
            mean_d = np.mean(data, axis=0)
            std_d = np.std(data, axis=0)
            print(m, data.shape, mean_d.shape, std_d.shape, mean_d[-1])

            mean_summaries[method][f'mean_{m}'] = mean_d
            mean_summaries[method][f'std_{m}'] = std_d

        x = np.arange(len(mean_summaries[method][f'mean_params']))
        mean_y = np.array(mean_summaries[method][f'mean_params'])
        std_y = np.array(mean_summaries[method][f'std_params'])
        plt.plot(x, mean_y, label=labels[method], color=colors[method])
        plt.fill_between(x, mean_y-std_y, mean_y+std_y, alpha=0.2, color=colors[method])

        # break
    
    plt.xlabel('#Pruning iterations')
    plt.ylabel(f'#Remaining Params')

    steps = 10
    x_xticks = [i for i in range(0, len(x), len(x)//steps)]
    xticks = [x[i] for i in x_xticks]

    y_yticks = [i for i in range(0, len(mean_y), len(mean_y)//steps)]
    y_yticks = [mean_y[i] for i in y_yticks]
    # mean_y = mean_y / 1e6
    yticks = [f'{y/1e6:.2f}M' for y in y_yticks]
    print(y_yticks)
    print(yticks)

    y_yticks = [84129904.0, 59669883.5, 32843137.5, 24420361.0, 13385436.5]
    yticks = ['84.13M', '59.67M', '32.84M', '24.42M', '13.39M']

    if not isinstance(xticks, type(None)):
        plt.xticks(x_xticks, xticks, rotation=50)
    plt.yticks(y_yticks, yticks)
        
    plt.tight_layout()
    plt.legend(loc='best')

    plots_dir = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking_imagenet_pretrained/run_1_6_summaries/report_plots'
    plot_path = os.path.join(plots_dir, f'params_v_iters.pdf')
    plt.savefig(plot_path)
    print('saved to', plot_path)
    plt.close()

if args.plot_params:
    plot_params()

def analyzing_metrics():
    loss_dir = 'assets/experiments/MTLNet_random_init_nyu_trained/best_loss'
    pixel_dir = 'assets/experiments/MTLNet_random_init_nyu_trained/best_pixel'

    loss_mean_std = get_mean_std_metrics(loss_dir, 3)
    pixel_mean_std = get_mean_std_metrics(pixel_dir, 3)

    for k, v in loss_mean_std.items():
        loss_mean_std[k] = [v]
        pixel_mean_std[k] = [pixel_mean_std[k]]

    # print(loss_mean_std)
    # print(pixel_mean_std)

    ldf = pd.DataFrame(loss_mean_std)
    pdf = pd.DataFrame(pixel_mean_std)

    print_m = ['Pixel Acc', 'mIoU', 'abs_err', 'rel_err', 'Angle Mean', 'Angle Median']
    pstr = ''
    for m in print_m:
        print(m)
        k = f'mean_{m}'
        v = pixel_mean_std[k][0]
        if m == 'Pixel Acc':
            v *= 100.
        pstr = pstr +  f'& $\mathbf{{{v:.4f}}}$ '
        print(pstr)
    
    # ldf.to_csv('assets/experiments/MTLNet_random_init_nyu_trained/best_loss/mean_sumamry.csv')
    # pdf.to_csv('assets/experiments/MTLNet_random_init_nyu_trained/best_pixel/mean_sumamry.csv')

if args.analyzing_metrics:
    analyzing_metrics()
