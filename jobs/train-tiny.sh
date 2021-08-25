#!/bin/bash

#SBATCH --account=dl66
#SBATCH --partition=dgx
#SBATCH --qos=dgx

#SBATCH -n 1
#SBATCH -c 40
#SBATCH --gres=gpu:V100:8
#SBATCH --mem=400GB
#SBATCH --time=24:00:00

#SBATCH --mail-user=zizhengpan98@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# SBATCH --exclude=dgx000


# Command to run a gpu job
# For example:
# module load anaconda/2019.03-Python3.7-gcc5
cd ../
bash scripts/run.sh