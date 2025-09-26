#!/bin/bash

# Gemini 2.5 Pro Inference Script
# Adapted from run_internvl25_test.sh for Gemini 2.5 Pro API

echo "🚀 Starting inference with Gemini 2.5 Pro model..."

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
# Using the same environment as other models
conda activate qwen25_vl_sign

# Set Python path to use the same environment
PYTHON_PATH="/home/ztao/anaconda3/envs/qwen25_vl_sign/bin/python"

# Set video folder
# VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/segmented_clips/"

VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/segmented_clips/"

echo "📁 Using video folder: $VIDEO_FOLDER"

# Check if API key is provided
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY environment variable is not set!"
    echo "Please set your Gemini API key:"
    echo "export GEMINI_API_KEY='your_api_key_here'"
    exit 1
fi

# Check if max_samples argument is provided
if [ $# -eq 1 ]; then
    MAX_SAMPLES_ARG="--max-samples $1"
    echo "📝 Processing test samples..."
    echo "📊 Processing $1 samples"
else
    MAX_SAMPLES_ARG=""
    echo "📝 Processing all test samples..."
fi

# Run Gemini 2.5 Pro inference
$PYTHON_PATH -m playground.demo.gemini25_metrics \
    --api-key "$GEMINI_API_KEY" \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/segmented_videos.json \
    --image_size 512 \
    --video-folder "$VIDEO_FOLDER" \
    --answers-file gemini25_results.json \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
    --enable_evaluation \
    $MAX_SAMPLES_ARG

echo "✅ Inference completed!"
echo "📄 Results saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/gemini25_results.json"
echo "📊 Evaluation metrics saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/evaluation_metrics.json"

# Display usage instructions
echo ""
echo "📋 Usage Instructions:"
echo "1. Set your Gemini API key: export GEMINI_API_KEY='your_api_key_here'"
echo "2. Run with all samples: bash scripts/eval/run_gemini25_test.sh"
echo "3. Run with limited samples: bash scripts/eval/run_gemini25_test.sh 5"
echo ""
echo "🔗 Get your API key from: https://aistudio.google.com/app/apikey"
