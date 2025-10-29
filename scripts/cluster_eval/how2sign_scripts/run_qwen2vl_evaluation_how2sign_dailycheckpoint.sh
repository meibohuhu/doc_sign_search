#!/bin/bash -l
# Qwen2VL Checkpoint-5000 Evaluation Script
# Based on qwen25_metrics.py reference approach

#SBATCH --job-name=qwen2vl_eval_checkpoint5000
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=64g

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths
export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:/home/mh2803/projects/sign_language_llm/evaluation:$PYTHONPATH"
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
# CHECKPOINT_PATH="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/qwen2vl_how2sign_4xa100_filtered_16batchsize/checkpoint-4000"
# CHECKPOINT_PATH="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/1018/qwen2vl_how2sign_4xa100_filtered_32batchsize_freezevisiontower/checkpoint-4000"   ### out_20919470.txt
# CHECKPOINT_PATH="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/qwen2vl_how2sign_4xa100_filtered_32batchsize_fast/checkpoint-3000"

CHECKPOINT_PATH="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/qwen2vl_ssvp_2xa100_12fps_diverse/checkpoint-6000"
MODEL_BASE="Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_FOLDER="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_320x320/"
QUESTION_FILE="/home/mh2803/projects/sign_language_llm/vanshika/asl_test/segmented_videos.json"
OUT_DIR="/home/mh2803/projects/sign_language_llm/outputs/"

echo "🎬 Qwen2VL Checkpoint-5000 Evaluation"
echo "======================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Model Base: $MODEL_BASE"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo ""


# Run evaluation with limited samples for testing
/home/mh2803/miniconda3/envs/qwenvl/bin/python scripts/cluster_eval/how2sign_scripts/qwen2vl_evaluation_how2sign_claude.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --model-base "$MODEL_BASE" \
    --video-folder "$VIDEO_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --out-dir "$OUT_DIR" \
    --max-samples 250 \
    --video-fps 12

echo "🎉 Evaluation job completed!"
