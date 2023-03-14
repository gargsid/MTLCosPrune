import os, sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config_index', type=int, default=0)
parser.add_argument('--method', type=str, default='total')
parser.add_argument('--num_runs', type=int, default=10)
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--gpu', type=str, default='gypsum-rtx8000')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--testing', action='store_true')

args = parser.parse_args() 

if args.lr == 0.001:
    setting=0
else:
    setting=1

cmd = '#!/bin/sh\n'
# cmd += f'#SBATCH --job-name={args.method[0]}/{args.config_index}/{args.run}\n'
cmd += f'#SBATCH --job-name={args.method[0]}{args.config_index},{setting}{args.run}\n'
cmd += f'#SBATCH -o assets/slurm/slurm-%j.out  # %j = job ID\n'
cmd += f'#SBATCH --partition={args.gpu}\n'
cmd += f'#SBATCH --gres=gpu:1\n'
cmd += f'#SBATCH -c 12\n'
cmd += f'#SBATCH --mem 30GB\n' 

if args.method == 'total':
    base_dir = f'assets/experiments/taylor_pruning/mt_base_vgg_nyu/total_ranking/run_3/total_ranking_configs_lr_{args.lr}_epochs_{args.epochs}'
    val_stats_path = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/total_ranking/run_3/validation_stats_with_flops.pkl'
elif args.method == 'cos':
    base_dir = f'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_3/cosine_ranking_configs_lr_{args.lr}_epochs_{args.epochs}'
    val_stats_path = 'assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_3/validation_stats_with_flops.pkl'

cmd += f'python train_config_models.py --gpu={args.gpu} --method={args.method} --base_dir={base_dir} --validation_stats_path={val_stats_path} --config_index={args.config_index} --run={args.run} --num_workers=3'
cmd += f' --epochs={args.epochs} --lr={args.lr}'

# print(cmd)
# sys.exit()

if args.testing:
    cmd += ' --testing'
    print(cmd)
    os.system(cmd)
else:
    print(cmd)
    with open('job_config_train.sh', 'w') as f:
        f.write(cmd)
    os.system('sbatch job_config_train.sh')