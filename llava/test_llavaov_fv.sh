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
    -m playground.demo.simpleQA \
    --model-path /home/vp1837/data/LLaVA-NeXT/checkpoints/merged_ssvp_fastvit/ \
    --model-name llava-qwen-lora-0.5b \
    --model-base lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --question-file /home/vp1837/test/rgb_vid_df_test.json \
    --video-folder /home/vp1837/test/raw_videos/ \
    --answers-file test_ssvp.json \
    --out_dir ./new_outputs