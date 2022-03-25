#!/bin/bash
#SBATCH --job-name=job_wgpu
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1 
#SBATCH --requeue

singularity exec --nv --bind /scratch/tl2564 --overlay /scratch/tl2564/overlay-25GB-500K.ext3:ro /scratch/tl2564/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /scratch/tl2564/senior-capstone-commonsense-reasoning
python exp/tmp.py
"