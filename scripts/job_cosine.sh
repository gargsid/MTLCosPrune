#!/bin/sh
#SBATCH --job-name=cosrank
#SBATCH -o ../assets/slurm/slurm-%j.out  # %j = job ID
#SBATCH --partition=gypsum-rtx8000
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem 20GB

# python mtl_cosine_main.py --pruning=True --batch-size=16 --lr=0.00005 \
# --lr-decay-every=10 --momentum=0.9 --seed=0 --mgpu=False --group_wd_coeff=1e-8 --wd=0.0 --tensorboard=False --pruning-method=22 --no_grad_clip=True \
# --epochs=430 --logs_dir=../assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_sampling/sample_2 \
# --cosine_sampling \
# --num_workers=3 --pruning_config=./configs/cifar10_lenet.json 
# --num_workers=0 --pruning_config=./configs/test_config.json --testing

python mtl_cosine_main.py --pruning=True --batch-size=16 --lr=0.00005 \
--lr-decay-every=10 --momentum=0.9 --seed=0 --mgpu=False --group_wd_coeff=1e-8 --wd=0.0 --tensorboard=False --pruning-method=22 --no_grad_clip=True \
--epochs=430 --logs_dir=../assets/experiments/taylor_pruning/mt_base_vgg_nyu/cosine_ranking/run_5 \
--cosine_ranking \
--num_workers=3 --pruning_config=./configs/cifar10_lenet.json 
# --num_workers=0 --pruning_config=./configs/test_config.json --testing
# --cosine_sampling \

# python mtl_cosine_main.py --pruning=True --batch-size=16 --lr=0.00005 \
# --lr-decay-every=10 --momentum=0.9 --seed=0 --mgpu=False --group_wd_coeff=1e-8 --wd=0.0 --tensorboard=False --pruning-method=22 --no_grad_clip=True \
# --epochs=430 --logs_dir=../assets/experiments/taylor_pruning/mt_base_vgg_nyu/track_cosine_similarity/run_3 \
# --num_workers=3 --pruning_config=./configs/cifar10_lenet.json 
# --num_workers=0 --pruning_config=./configs/test_config.json --testing

# python mtl_max_main.py --pruning=True --batch-size=16 --lr=0.00005 \
# --l1 --num_normalize --nu_rank \
# --epochs=430 --logs_dir=../assets/experiments/taylor_pruning/mt_base_vgg_nyu/max_gradient_importance/l1_norm_nu \
# --lr-decay-every=10 --momentum=0.9 --seed=0 --num_workers=3 \
# --mgpu=False --group_wd_coeff=1e-8 --wd=0.0 --tensorboard=False --pruning-method=22 \
# --no_grad_clip=True --pruning_config=./configs/cifar10_lenet.json 

# python mtl_max_main.py --pruning=True --batch-size=16 --lr=0.00005 --supermodel --l1 --num_normalize \
# --lr-decay-every=10 --momentum=0.9 --seed=0 --mgpu=False --group_wd_coeff=1e-8 --wd=0.0 --tensorboard=False --pruning-method=22 \
# --epochs=1270 --logs_dir=../assets/experiments/taylor_pruning/mt_super_vgg_nyu/max_gradient_importance/run_4 --no_grad_clip=True \
# --num_workers=0 --pruning_config=./configs/test_config.json --testing
# --num_workers=3 --pruning_config=./configs/cifar10_lenet.json 
