#!/bin/bash

# GPT-4o API Inference Script
# Adapted from run_internvl25_test.sh for GPT-4o API inference

echo "🚀 Starting inference with GPT-4o API..."

# Activate conda environment
conda activate llava

# Set Python path to use the qwen25_vl_sign environment
PYTHON_PATH="/home/ztao/anaconda3/envs/qwen25_vl_sign/bin/python"

# Set video folder
VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/segmented_clips/"

echo "📁 Using video folder: $VIDEO_FOLDER"

# Check if OpenAI API key is provided
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "🔑 Using OpenAI API key: ${OPENAI_API_KEY:0:10}..."

# Check if max_samples argument is provided
if [ $# -eq 1 ]; then
    MAX_SAMPLES_ARG="--max-samples $1"
    echo "📝 Processing test samples..."
    echo "📊 Processing $1 samples"
else
    MAX_SAMPLES_ARG=""
    echo "📝 Processing all test samples..."
fi

# Run GPT-4o API inference
$PYTHON_PATH -m playground.demo.gpt4o_metrics \
    --openai_api_key "$OPENAI_API_KEY" \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/segmented_videos.json \
    --image_size 1024 \
    --video-folder "$VIDEO_FOLDER" \
    --answers-file gpt4o_results.json \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
    --enable_evaluation \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    $MAX_SAMPLES_ARG

echo "✅ Inference completed!"
echo "📄 Results saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/gpt4o_results.json"
echo "📊 Evaluation metrics saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/evaluation_metrics.json"
