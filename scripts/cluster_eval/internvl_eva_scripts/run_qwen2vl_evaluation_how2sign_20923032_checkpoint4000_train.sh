#!/bin/bash -l
# InternVL Checkpoint Evaluation Script
# Based on qwen2vl_evaluation_how2sign_claude.py evaluation approach

#SBATCH --job-name=internvl_eval_checkpoint_train
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=64g

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths
# Use qwenvl environment (or adjust to InternVL-specific environment if needed)
export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/InternVL/internvl_chat:/home/mh2803/projects/sign_language_llm/InternVL:/home/mh2803/projects/sign_language_llm/evaluation:$PYTHONPATH"
export OMP_NUM_THREADS=8

# GPU optimization settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_USE_CUDA_DSA=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=1

# Change to project directory
cd /home/mh2803/projects/sign_language_llm

# Configuration
# Update checkpoint path to point to InternVL checkpoints directory
CHECKPOINT_PATH="/home/mh2803/projects/sign_language_llm/InternVL/checkpoints/finetune_internvl2_5_how2sign_18fps/checkpoint-2000"
# You can specify a specific checkpoint subdirectory, e.g.:
# CHECKPOINT_PATH="/home/mh2803/projects/sign_language_llm/InternVL/checkpoints/checkpoint-4000"

MODEL_BASE="OpenGVLab/InternVL2.5-2B"  # Adjust based on your InternVL model
VIDEO_FOLDER="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224/"
QUESTION_FILE="/home/mh2803/projects/sign_language_llm/InternVL/data/how2sign/test_how2sign_internvl.jsonl"
OUT_DIR="/home/mh2803/projects/sign_language_llm/outputs/"

echo "🎬 InternVL Checkpoint Evaluation"
echo "=================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Model Base: $MODEL_BASE"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo ""


# Run evaluation with limited samples for testing
/home/mh2803/miniconda3/envs/qwenvl/bin/python scripts/cluster_eval/internvl_eva_scripts/internvl_evaluation_how2sign.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --model-base "$MODEL_BASE" \
    --video-folder "$VIDEO_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --out-dir "$OUT_DIR" \
    --max-samples 10 \
    --min-num-frames 32 \
    --max-num-frames 160 \
    --sampling-method fps18.0 \
    --image-size 224 \
    --max-new-tokens 128

echo "🎉 Evaluation job completed!"
