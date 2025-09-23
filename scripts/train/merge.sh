#!/bin/bash -l
#SBATCH --job-name=merge_save_qwen
#SBATCH --output=/home/vp1837/data/LLaVA-NeXT/RC_error/merge_out_%j.txt
#SBATCH --error=/home/vp1837/data/LLaVA-NeXT/RC_error/merge_err_%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=1-0:30:00
#SBATCH --mem=48g
#SBATCH --partition=tier3
#SBATCH --account=ai-asl

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4
conda activate llavaov

python /home/vp1837/data/LLaVA-NeXT/scripts/train/merge_save.py

