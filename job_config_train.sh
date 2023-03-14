#!/bin/sh
#SBATCH --job-name=c11,15
#SBATCH -o assets/slurm/slurm-%j.out  # %j = job ID
#SBATCH --partition=gypsum-rtx8000
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem 30GB
python train_config_models.py --gpu=gypsum-rtx8000 --method=cos --base_dir=assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_3/cosine_ranking_configs_lr_0.0005_epochs_500 --validation_stats_path=assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_3/validation_stats_with_flops.pkl --config_index=11 --run=5 --num_workers=3 --epochs=500 --lr=0.0005