#!/bin/bash -l
# NOTE the -l flag!
#
#SBATCH --job-name=test_ssvp
#SBATCH --error=/home/vp1837/data/LLaVA-NeXT/RC_error/test_err_%j.txt
#SBATCH --output=/home/vp1837/data/LLaVA-NeXT/RC_out/test_out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0:30:00
#SBATCH --gpus-per-node=a100:1      
#SBATCH --partition debug
#SBATCH --mem=128g                    
#SBATCH --account=ai-asl

# Load CUDA
spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Activate Conda environment
conda activate llavaov

# Run the test script with torchrun (for single GPU)
torchrun --nproc_per_node=1 --nnodes=1 --master_port=29503 \
  /home/vp1837/data/LLaVA-NeXT/playground/demo/video_demo.py \
  --video_path /home/vp1837/test/raw_videos/ \
  --output_dir ./new_outputs \
  --output_name run1.json \
  --model-path /home/vp1837/data/LLaVA-NeXT/checkpoints/merged_ssvp_fastvit/ \
  --model-base Qwen/Qwen2-7B-Instruct \
  --conv-mode qwen_2 \
  --for_get_frames_num 5 \
  --prompt "What is the signer doing?"
