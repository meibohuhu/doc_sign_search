#!/bin/bash

# InternVL 2.5 (8B) Inference Script
# Adapted from run_qwen25_test.sh for InternVL 2.5 (8B) model

echo "🚀 Starting inference with InternVL 2.5 (8B) model..."

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
# Using the same environment as Qwen2.5-VL since InternVL also uses transformers
conda activate llava

# Set Python path to use the qwen25_vl_sign environment
PYTHON_PATH="/home/ztao/anaconda3/envs/qwen25_vl_sign/bin/python"

# Set video folder
VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/segmented_clips/"

echo "📁 Using video folder: $VIDEO_FOLDER"

# Check if max_samples argument is provided
if [ $# -eq 1 ]; then
    MAX_SAMPLES_ARG="--max-samples $1"
    echo "📝 Processing test samples..."
    echo "📊 Processing $1 samples"
else
    MAX_SAMPLES_ARG=""
    echo "📝 Processing all test samples..."
fi

# Run InternVL 2.5 (8B) inference
$PYTHON_PATH -m playground.demo.internvl25_metrics \
    --model-path OpenGVLab/InternVL3-8B \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/segmented_videos.json \
    --image_size 448 \
    --video-folder "$VIDEO_FOLDER" \
    --answers-file internvl25_results.json \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
    --enable_evaluation \
    $MAX_SAMPLES_ARG

echo "✅ Inference completed!"
echo "📄 Results saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/internvl25_results.json"
echo "📊 Evaluation metrics saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/evaluation_metrics.json"



