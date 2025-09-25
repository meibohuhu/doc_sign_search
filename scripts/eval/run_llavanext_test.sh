#!/bin/bash

# LLaVA-NeXT Video Inference Script
# Adapted from run_internvl25_test.sh for LLaVA-NeXT Video model

echo "🚀 Starting inference with LLaVA-NeXT Video model..."

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
# Using the llava environment for LLaVA-NeXT
conda activate llava

# Set Python path to use the llava environment
PYTHON_PATH="/home/ztao/anaconda3/envs/llava/bin/python"

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

# Run LLaVA-NeXT Video inference
$PYTHON_PATH -m playground.demo.llavanext_metrics \
    --model-path lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/segmented_videos.json \
    --image_size 336 \
    --video-folder "$VIDEO_FOLDER" \
    --answers-file llavanext_results.json \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
    --enable_evaluation \
    --add_time_instruction \
    --do_sample \
    $MAX_SAMPLES_ARG

echo "✅ Inference completed!"
echo "📄 Results saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/llavanext_results.json"
echo "📊 Evaluation metrics saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/evaluation_metrics.json"

# Display usage instructions
echo ""
echo "📋 Usage Instructions:"
echo "1. Run with all samples: bash scripts/eval/run_llavanext.sh"
echo "2. Run with limited samples: bash scripts/eval/run_llavanext_test.sh 5"
echo ""
echo "🔧 Model Options:"
echo "- Default: lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only"
echo "- Alternative: lmms-lab/LLaVA-Video-7B-Qwen2"
echo ""
echo "📖 Based on: https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_Video_1003.md"
