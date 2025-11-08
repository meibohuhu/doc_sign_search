#!/bin/bash
# Qwen2VL Evaluation Script for A6000
# Adapted from cluster evaluation script for local execution

set -e  # Exit on error

# Activate conda environment
source /home/ztao/anaconda3/etc/profile.d/conda.sh
conda activate qwen25_vl_sign

echo "✅ Activated conda environment: qwen25_vl_sign"
echo "🐍 Python: $(which python)"
echo ""

# GPU optimization settings for A6000 (Using BOTH GPUs)
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_USE_CUDA_DSA=1
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=16

# Disable flash attention
export DISABLE_FLASH_ATTN=1

# Change to project directory
cd /local1/mhu/sign_language_llm

# Configuration - Update these paths as needed
CHECKPOINT_PATH="/local1/mhu/sign_language_llm/how2sign/checkpoints/qwen2vl_how2sign_4xa100_filtered_32batchsize_fast/checkpoint-3000"
MODEL_BASE="Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_FOLDER="/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/segmented_clips"
QUESTION_FILE="/local1/mhu/sign_language_llm/vanshika/asl_test/segmented_videos.json"
OUT_DIR="/local1/mhu/sign_language_llm/outputs/a6000_evaluation"

# Create output directory
mkdir -p "$OUT_DIR"

echo "🎬 Qwen2VL Evaluation on A6000"
echo "======================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Model Base: $MODEL_BASE"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo ""

# Check if paths exist
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint path not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$VIDEO_FOLDER" ]; then
    echo "❌ Error: Video folder not found: $VIDEO_FOLDER"
    exit 1
fi

if [ ! -f "$QUESTION_FILE" ]; then
    echo "❌ Error: Question file not found: $QUESTION_FILE"
    exit 1
fi

echo "✅ All paths validated"
echo ""

# Run evaluation
# NOTE: Adjust --max-samples as needed (None for all samples, or specify a number for testing)
# NOTE: Add --freeze-vision-tower flag if vision tower was frozen during training
python scripts/local_eval/qwen2vl_evaluation_a6000.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --model-base "$MODEL_BASE" \
    --video-folder "$VIDEO_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --out-dir "$OUT_DIR" \
    --max-samples 1000 \
    --enable-evaluation \
    --video-fps 12 \
    --max-frames 128 \
    --max-new-tokens 256

echo ""
echo "🎉 Evaluation completed!"

