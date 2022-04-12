#!/usr/bin/bash
#
#SBATCH --job-name=myTestTaskGPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:3090:1
#SBATCH --output=log_dstc10val_%A_%a.out
#SBATCH --error=log_dstc10val_%A_%a.err
#SBATCH --time=24:00:00

module purge

export PATH=/scratch/tl2564/anaconda3/bin:$PATH
source activate csqa2

python experiment.py