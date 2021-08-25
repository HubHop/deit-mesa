#!/bin/bash

#SBATCH --account=dl65
#SBATCH --partition=m3g
# SBATCH --qos=dgx

#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32GB
#SBATCH --time=3:00:00

#SBATCH --mail-user=zizhengpan98@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# SBATCH --exclude=dgx000


# Command to run a gpu job
# For example:
# module load anaconda/2019.03-Python3.7-gcc5
source activate quant
which python

nvidia-smi
cd ../
python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port 1622 --use_env main.py  \
        --model deit_ms_tiny_patch16_224  \
        --batch-size 128 \
        --data-path /projects/dl65/m3_imagenet  \
        --data-set CIFAR \
        --input-size 224  \
        --output_dir ./exp_cuda/full \
        --num_workers 10 \
        --ms_policy config/policy_tiny-8bit.txt 