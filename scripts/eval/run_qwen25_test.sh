#!/bin/bash
# Run inference using Qwen2.5-VL-7B model
##### mhu update 09/22/2025 Qwen2.5-VL-7B inference setup

echo "🚀 Starting inference with Qwen2.5-VL-7B model..."

# Set Python path
export PYTHONPATH=/local1/mhu/LLaVANeXT_RC:$PYTHONPATH

# Use direct Python path from qwen25_vl_sign environment
PYTHON_PATH="/home/ztao/anaconda3/envs/qwen25_vl_sign/bin/python"

# VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/raw_videos/"
VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/segmented_clips/"
# VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/test_videos/"


echo "📁 Using video folder: $VIDEO_FOLDER"
echo "📝 Processing test samples..."

# Get number of samples (default: process all)
MAX_SAMPLES=${1:-""}

if [ -n "$MAX_SAMPLES" ]; then
    echo "📊 Processing $MAX_SAMPLES samples"
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
else
    echo "📊 Processing all samples"
    MAX_SAMPLES_ARG=""
fi

# Run the inference with Qwen2.5-VL-7B model
$PYTHON_PATH -m playground.demo.qwen25_metrics \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/segmented_videos.json \
    --image_size 768 \
    --video-folder "$VIDEO_FOLDER" \
    --answers-file qwen25_vl_results.json \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
    --enable_evaluation \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_new_tokens 512 \
    $MAX_SAMPLES_ARG

echo "✅ Inference completed!"
echo "📄 Results saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/qwen25_vl_results.json"
echo "📊 Evaluation metrics saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/evaluation_metrics.json"
