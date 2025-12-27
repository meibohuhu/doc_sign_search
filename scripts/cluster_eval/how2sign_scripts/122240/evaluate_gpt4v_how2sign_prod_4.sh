#!/bin/bash
#
# Azure GPT-5 Vision Evaluation Script for How2Sign
# Uses Azure OpenAI GPT-5 Vision API to process video frames and answer questions
#

# Set up conda environment (if needed)
# source "$HOME/anaconda3/etc/profile.d/conda.sh"
# conda activate your_env_name
# echo "✅ Conda environment activated"

# Essential environment variables
export PYTHONUNBUFFERED=1

# Change to project directory
cd /code/doc_sign_search

# Configuration
MODEL="${MODEL:-gpt-5}"  # Azure deployment model name
VIDEO_FOLDER="${VIDEO_FOLDER:-/mnt/localssd/doc_sign_search/train_crop_videos_224}"
# VIDEO_FOLDER="${VIDEO_FOLDER:-/mnt/localssd/doc_sign_search/how2sign_test_videos_224x224}"
QUESTION_FILE="${QUESTION_FILE:-/code/doc_sign_search/InternVL/data/gpt/122240/gpt_how2sign_internvl_40_4.jsonl}"
# QUESTION_FILE="${QUESTION_FILE:-/code/doc_sign_search/InternVL/data/how2sign/test_how2sign_internvl.jsonl}"
OUT_DIR="${OUT_DIR:-/code/doc_sign_search/outputs/gpt5_eval/122240/122240_4}"


# Azure OpenAI Configuration
AZURE_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://dil-research-3.openai.azure.com/}"
AZURE_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
AZURE_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-$MODEL}"  # Default to MODEL if not set

# Evaluation parameters
MAX_SAMPLES=${MAX_SAMPLES:-1000}  # Set to a number to limit samples, empty for full evaluation
NUM_FRAMES=${NUM_FRAMES:-32}  # Number of frames to extract from video
VIDEO_FPS=${VIDEO_FPS:-}  # Target FPS for frame extraction (if empty, uses num-frames)
IMAGE_DETAIL=${IMAGE_DETAIL:-low}  # Image detail level: low, high, auto
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
TEMPERATURE=${TEMPERATURE:-1}
SAVE_FRAMES=${SAVE_FRAMES:-false}  # Set to "true" to enable frame saving

echo "🎬 Azure GPT-5 Vision How2Sign Evaluation"
echo "=========================================="
echo "Azure Endpoint: $AZURE_ENDPOINT"
echo "Azure API Version: $AZURE_API_VERSION"
echo "Azure Deployment: $AZURE_DEPLOYMENT"
echo "Model: $MODEL"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo "Num Frames: $NUM_FRAMES"
if [ -n "$VIDEO_FPS" ]; then
    echo "Video FPS: $VIDEO_FPS"
else
    echo "Video FPS: (using num-frames instead)"
fi
echo "Image Detail: $IMAGE_DETAIL"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples (limited): $MAX_SAMPLES"
else
    echo "Max Samples: Full evaluation"
fi
if [ "$SAVE_FRAMES" = "true" ]; then
    echo "Save Frames: Enabled (frames will be saved to $OUT_DIR/extracted_frames/)"
else
    echo "Save Frames: Disabled"
fi
echo ""

# Check if Azure API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY (Azure subscription key) environment variable is not set"
    echo "   You can set it with: export OPENAI_API_KEY='your-azure-subscription-key'"
    echo "   Or pass it via --api-key argument"
    echo ""
fi

# Check if video folder exists
if [ ! -d "$VIDEO_FOLDER" ]; then
    echo "❌ Error: Video folder does not exist: $VIDEO_FOLDER"
    exit 1
fi

# Check if question file exists
if [ ! -f "$QUESTION_FILE" ]; then
    echo "❌ Error: Question file does not exist: $QUESTION_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"
echo "📁 Output directory: $OUT_DIR"
echo ""

# Log file with timestamp
LOG_FILE="${OUT_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"
echo ""

# Build command arguments
EVAL_ARGS=(
    --api-type "openai"
    --video-folder "$VIDEO_FOLDER"
    --question-file "$QUESTION_FILE"
    --out-dir "$OUT_DIR"
    --model "$MODEL"
    --num-frames "$NUM_FRAMES"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --image-detail "$IMAGE_DETAIL"
    --azure-endpoint "$AZURE_ENDPOINT"
    --azure-api-version "$AZURE_API_VERSION"
    --azure-deployment "$AZURE_DEPLOYMENT"
)

# Add API key if provided as environment variable
if [ -n "$OPENAI_API_KEY" ]; then
    EVAL_ARGS+=(--api-key "$OPENAI_API_KEY")
fi

# Add video FPS if specified
if [ -n "$VIDEO_FPS" ]; then
    EVAL_ARGS+=(--video-fps "$VIDEO_FPS")
fi

# Add max-samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    EVAL_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# Add save-frames if enabled
if [ "$SAVE_FRAMES" = "true" ]; then
    EVAL_ARGS+=(--save-frames)
fi

# Get python path (use system python or conda python)
PYTHON_CMD="python3"
if command -v python3 &> /dev/null; then
    PYTHON_CMD=$(which python3)
elif command -v python &> /dev/null; then
    PYTHON_CMD=$(which python)
else
    echo "❌ Error: Python not found!"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Run evaluation
echo "🚀 Starting evaluation..."
echo ""

$PYTHON_CMD scripts/cluster_eval/how2sign_scripts/gpt4v_evaluation_how2sign_prod.py \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo "📁 Results saved to: $OUT_DIR"
    echo "📝 Log file: $LOG_FILE"
else
    echo "❌ Evaluation failed with exit code $EVAL_EXIT_CODE."
    echo "Please check the error messages above for details."
    echo "Check log file: $LOG_FILE"
    exit $EVAL_EXIT_CODE
fi

