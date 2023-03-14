#!/bin/sh
#SBATCH --job-name=l1
#SBATCH -o assets/slurm/slurm-%j.out  # %j = job ID
#SBATCH --partition=gypsum-rtx8000
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem 30GB

# CosineDecay
# python train_base_model.py --num_workers=3 --epochs=5000 --logs_dir=assets/experiments/MTLNet_random_init_nyu_trained/run_5K
# python train_base_model.py --num_workers=3 --epochs=1000 --lr=0.0001 --logs_dir=assets/experiments/MTLNet_random_init_nyu_trained/run_1K_lr0.0001

# StepLR schedule
# python train_base_model.py --num_workers=3 --epochs=1000 --lr=0.0001 --logs_dir=assets/experiments/MTLNet_random_init_nyu_trained/run_steplr_1K_lr0.0001

# re-training longer using trained weights
# python train_base_model.py --num_workers=3 --epochs=2500 --lr=0.0001 \
# --logs_dir=assets/experiments/MTLNet_random_init_nyu_trained/run_2.5K_lr0.0001_retraining \
# --ckpt=assets/experiments/MTLNet_random_init_nyu_trained/run_1K_lr0.0001/pruned_model.pth

python train_base_model.py --num_workers=3 --epochs=500 --lr=0.0001 --metric=loss \
--logs_dir=assets/experiments/MTLNet_random_init_nyu_trained/best_loss_1