#!/bin/bash
# Qwen2.5-VL-3B-Instruct BASE Model Evaluation on A6000
# Evaluates pretrained model WITHOUT fine-tuning as baseline comparison

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

# Configuration
MODEL_BASE="Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_FOLDER="/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/segmented_clips"
QUESTION_FILE="/local1/mhu/sign_language_llm/vanshika/asl_test/segmented_videos.json"
OUT_DIR="/local1/mhu/sign_language_llm/outputs/a6000_base_evaluation"

# Create output directory
mkdir -p "$OUT_DIR"

echo "🎬 Qwen2.5-VL-3B-Instruct BASE Model Evaluation on A6000"
echo "=========================================================="
echo "Model: $MODEL_BASE (pretrained, NO fine-tuning)"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo ""

# Check if paths exist
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
python scripts/local_eval/qwen2vl_base_evaluation_a6000.py \
    --model-base "$MODEL_BASE" \
    --video-folder "$VIDEO_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --out-dir "$OUT_DIR" \
    --max-samples 32 \
    --enable-evaluation \
    --video-fps 10 \
    --max-frames 96 \
    --max-new-tokens 256

echo ""
echo "🎉 BASE model evaluation completed!"

