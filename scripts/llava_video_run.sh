#!/bin/bash -l
#SBATCH --job-name=test-llavavideo
#SBATCH --error=/home/vp1837/data/llava_video/RC_error/test_err_%j.txt
#SBATCH --output=/home/vp1837/data/llava_video/RC_out/test_out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-0:30:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=32g
#SBATCH --account=ai-asl

# Load CUDA module
spack load cuda@12.4.0/obxqih4

conda activate llava
# Set any necessary environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your Python script
python /home/vp1837/data/LLaVA-NeXT/scripts/llava_video_test.py
