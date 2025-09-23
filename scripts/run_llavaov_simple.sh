#!/bin/bash
# Run ASL inference using LLaVA-OneVision 0.5B model
#### mhu update 09/19/2025 Direct llavaov_simple approach, Object-oriented approach with error handling

echo "🚀 Starting ASL inference with LLaVA-OneVision 0.5B..."

# Activate llava environment
eval "$(conda shell.bash hook)"
conda activate llava

# Set Python path
export PYTHONPATH=/local1/mhu/LLaVANeXT_RC:$PYTHONPATH

VIDEO_FOLDER="/local1/mhu/LLaVANeXT_RC/how2sign/video/test_raw_videos/raw_videos/"

echo "📁 Using video folder: $VIDEO_FOLDER"
echo "📝 Processing ASL test samples with LLaVA-OneVision 0.5B model..."

# Get number of samples (default: process all)
MAX_SAMPLES=${1:-"all"}

if [ "$MAX_SAMPLES" = "all" ]; then
    echo "📊 Processing all samples in dataset"
    python scripts/test_llavaov_simple.py \
        --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/rgb_vid_df_test.json \
        --video-folder "$VIDEO_FOLDER" \
        --output-file llavaov_asl_results.json \
        --num-frames 8
else
    echo "📊 Processing $MAX_SAMPLES samples"
    python scripts/test_llavaov_simple.py \
        --question-file /local1/mhu/LLaVANeXT_RC/output/asl_test/rgb_vid_df_test.json \
        --video-folder "$VIDEO_FOLDER" \
        --output-file llavaov_asl_results.json \
        --num-frames 8 \
        --max-samples "$MAX_SAMPLES"
fi

echo "✅ LLaVA-OneVision inference completed!"
echo "📄 Results saved to: llavaov_asl_results.json"
echo "🔍 Use 'python -m json.tool llavaov_asl_results.json' to view formatted results"
