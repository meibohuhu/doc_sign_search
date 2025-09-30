#!/bin/bash

# LLaVA-OneVision-1.5-4B-Instruct Inference Script
# Based on the LLaVA-OneVision-1.5 framework from EvolvingLMMs-Lab

echo "🚀 Starting inference with LLaVA-OneVision-1.5-4B-Instruct model..."

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
# Using the mh_llava environment for LLaVA-OneVision-1.5
conda activate mh_llava

# Set Python path to use the mh_llava environment
PYTHON_PATH="/home/mh2803/miniconda3/envs/mh_llava/bin/python"

# Set video folder
VIDEO_FOLDER="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips/"

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

# Run LLaVA-OneVision-1.5-4B-Instruct inference
$PYTHON_PATH -m playground.demo.llava_onevision_metrics \
    --model-path liuhaotian/llava-v1.5-7b \
    --model-name llava_llama \
    --question-file /home/mh2803/projects/sign_language_llm/output/asl_test/segmented_videos.json \
    --image_size 336 \
    --video-folder "$VIDEO_FOLDER" \
    --answers-file llava_onevision_results.json \
    --out_dir /home/mh2803/projects/sign_language_llm/outputs/llava_onevision/ \
    --enable_evaluation \
    --add_time_instruction \
    --do_sample \
    --temperature 0.2 \
    --top_p 0.8 \
    --max_new_tokens 512 \
    $MAX_SAMPLES_ARG

echo "✅ Inference completed!"
echo "📄 Results saved to: /home/mh2803/projects/sign_language_llm/outputs/llava_onevision/llava_onevision_results.json"
echo "📊 Evaluation metrics saved to: /home/mh2803/projects/sign_language_llm/outputs/llava_onevision/evaluation_metrics.json"

# Display usage instructions
echo ""
echo "📋 Usage Instructions:"
echo "1. Run with all samples: bash scripts/eval/run_llava_onevision_test.sh"
echo "2. Run with limited samples: bash scripts/eval/run_llava_onevision_test.sh 5"
echo ""
echo "🔧 Model Options:"
echo "- Default: lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
echo "- Alternative models available from the LLaVA-OneVision-1.5 repository"
echo ""
echo "📖 Based on: https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5"
echo "🎯 Evaluation includes: BLEU, ROUGE-L, F1, Precision, Recall, Exact Match"
