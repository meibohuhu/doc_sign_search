#!/bin/bash -l
# NOTE the -l flag!
#
#SBATCH --job-name=rgb_trcpt
#SBATCH --error=/home/vp1837/data/LLaVA-NeXT/RC_error/test_err_%j.txt
#SBATCH --output=/home/vp1837/data/LLaVA-NeXT/RC_out/test_out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-3:30:00
#SBATCH --gpus-per-node=a100:1      
#SBATCH --partition tier3
#SBATCH --mem=128g                    
#SBATCH --account=ai-asl

#### Original SLURM-based script by Vanshika

# Load CUDA
spack load cuda@12.4.0/obxqih4

# Activate Conda environment
conda activate llava

# Run the test script with torchrun (for single GPU) - BASE MODEL ONLY
torchrun --nproc_per_node=1 --nnodes=1 \
    -m playground.demo.simpleQA \
    --model-path lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --model-name llava_qwen \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/rgb_vid_df_test_filtered.json \
    --video-folder /local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/raw_videos/\
    --answers-file base_model_results.json \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/