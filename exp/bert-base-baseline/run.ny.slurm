#!/bin/bash
#SBATCH --job-name=bert
#SBATCH --open-mode=append
#SBATCH --output=log/%j_%x.out
#SBATCH --error=log/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4

singularity exec --nv --bind /scratch/tl2564 --overlay /scratch/tl2564/overlay-25GB-500K.ext3:ro /scratch/tl2564/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/tl2564/senior-capstone-commonsense-reasoning/exp/bert-base
python experiment.py
"
