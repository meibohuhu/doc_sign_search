#!/bin/bash
# Run inference using the llava conda environment
##### mhu update 09/19/2025 Direct simpleQA approach

echo "🚀 Starting inference with llava conda environment..."

# Activate llava environment and run inference
conda activate llava

# Set Python path
export PYTHONPATH=/local1/mhu/LLaVANeXT_RC:$PYTHONPATH

# VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/raw_videos/"
VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/segmented_clips/"

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

# # Run the inference with the metrics-enabled model
# python -m playground.demo.simpleQA_metrics \
#     --model-path lmms-lab/llava-onevision-qwen2-7b-ov \
#     --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/rgb_vid_df_test.json \
#     --image_size 336 \
#     --video-folder "$VIDEO_FOLDER" \
#     --answers-file llava_base_results.json \
#     --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
#     --enable_evaluation \
#     $MAX_SAMPLES_ARG

# Run the inference with the metrics-enabled model
python -m playground.demo.simpleQA_metrics \
    --model-path lmms-lab/llava-onevision-qwen2-7b-ov \
    --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/segmented_videos.json \
    --image_size 336 \
    --video-folder "$VIDEO_FOLDER" \
    --answers-file llava_base_results.json \
    --out_dir /local1/mhu/LLaVANeXT_RC/new_outputs/ \
    --enable_evaluation \
    $MAX_SAMPLES_ARG

echo "✅ Inference completed!"
echo "📄 Results saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/llava_base_results_0923.json"
echo "📊 Evaluation metrics saved to: /local1/mhu/LLaVANeXT_RC/new_outputs/evaluation_metrics_0923.json"
