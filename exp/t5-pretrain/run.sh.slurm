#!/usr/bin/bash
#
#SBATCH --job-name=t5-conceptnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2
#SBATCH --output=log/%j_%x.out
#SBATCH --error=log/%j_%x.err
#SBATCH --time=24:00:00

module purge

export PATH=/scratch/tl2564/anaconda3/bin:$PATH
source activate csqa2

cd /scratch/tl2564/senior-capstone-commonsense-reasoning/exp/t5-pretrain
python experiment.py