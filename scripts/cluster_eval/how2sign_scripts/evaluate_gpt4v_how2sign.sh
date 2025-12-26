#!/bin/bash
#
# GPT-4 Vision / Gemini Evaluation Script for How2Sign
# Uses OpenAI GPT-4 Vision API or Google Gemini API to process video frames and answer questions
#

# Set up conda environment (if needed)
# source "$HOME/anaconda3/etc/profile.d/conda.sh"
# conda activate your_env_name
# echo "✅ Conda environment activated"

# Essential environment variables
export PYTHONUNBUFFERED=1

# Change to project directory
cd /local1/mhu/sign_language_llm

# Configuration
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"  # Not used for API, but kept for compatibility
API_TYPE="${API_TYPE:-openai}"  # Options: openai, gemini

# Set model based on API type, ensuring consistency
if [ "$API_TYPE" = "gemini" ]; then
    # For Gemini, ensure we use a gemini model
    # If MODEL env var is set but not a gemini model, override it
    if [[ -n "$MODEL" ]] && [[ ! "$MODEL" =~ ^gemini ]]; then
        echo "⚠️  Warning: MODEL=$MODEL is not a Gemini model. Overriding to gemini-2.5-flash"
        MODEL="gemini-2.5-pro"
    else
        MODEL="${MODEL:-gemini-2.5-pro}"  # Default if not set
    fi
else
    # For OpenAI, use OpenAI model
    MODEL="${MODEL:-gpt-5}"
fi
VIDEO_FOLDER="${VIDEO_FOLDER:-/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224}"
QUESTION_FILE="${QUESTION_FILE:-/local1/mhu/sign_language_llm/InternVL/data/how2sign/train_how2sign_internvl.jsonl}"
OUT_DIR="${OUT_DIR:-/local1/mhu/sign_language_llm/outputs/gpt4v_eval/}"


### /local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224/G3k86AVFwVs_6-5-rgb_front.mp4
### /local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224/FZLxEwsoc1c_8-8-rgb_front.mp4
### /local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224/f_zna_wG3zo_2-10-rgb_front.mp4

# g3kFAmcBpFc_13-3-rgb_front.mp4

# Evaluation parameters
MAX_SAMPLES=${MAX_SAMPLES:-5}  # Set to a number to limit samples, empty for full evaluation
NUM_FRAMES=${NUM_FRAMES:-32}  # Number of frames to extract from video
VIDEO_FPS=${VIDEO_FPS:-}  # Target FPS for frame extraction (if empty, uses num-frames)
IMAGE_DETAIL=${IMAGE_DETAIL:-low}  # Image detail level: low, high, auto
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10240}
TEMPERATURE=${TEMPERATURE:-0.7}
SAVE_FRAMES=${SAVE_FRAMES:-false}  # Set to "true" to enable frame saving
MAX_WORKERS=${MAX_WORKERS:-1}  # Number of concurrent threads for Gemini API (default: 5)

echo "🎬 GPT-4 Vision / Gemini How2Sign Evaluation"
echo "=============================================="
echo "API Type: $API_TYPE"
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
if [ "$API_TYPE" = "gemini" ]; then
    echo "Max Workers (threads): $MAX_WORKERS"
fi
echo ""

# Check if API key is set
if [ "$API_TYPE" = "gemini" ]; then
    if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
        echo "⚠️  Warning: GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set"
        echo "   You can set it with: export GEMINI_API_KEY='your-api-key-here'"
        echo "   Or pass it via --api-key argument"
        echo ""
    fi
else
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set"
        echo "   You can set it with: export OPENAI_API_KEY='your-api-key-here'"
        echo "   Or pass it via --api-key argument"
        echo ""
    fi
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
    --api-type "$API_TYPE"
    --video-folder "$VIDEO_FOLDER"
    --question-file "$QUESTION_FILE"
    --out-dir "$OUT_DIR"
    --model "$MODEL"
    --num-frames "$NUM_FRAMES"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
)

# Add API key if provided as environment variable (will be used if --api-key not provided)
if [ "$API_TYPE" = "gemini" ]; then
    if [ -n "$GEMINI_API_KEY" ]; then
        EVAL_ARGS+=(--api-key "$GEMINI_API_KEY")
    elif [ -n "$GOOGLE_API_KEY" ]; then
        EVAL_ARGS+=(--api-key "$GOOGLE_API_KEY")
    fi
else
    if [ -n "$OPENAI_API_KEY" ]; then
        EVAL_ARGS+=(--api-key "$OPENAI_API_KEY")
    fi
fi

# Add image-detail only for OpenAI (not used by Gemini)
if [ "$API_TYPE" != "gemini" ]; then
    EVAL_ARGS+=(--image-detail "$IMAGE_DETAIL")
fi

# Add max-workers only for Gemini (multithreading)
if [ "$API_TYPE" = "gemini" ]; then
    EVAL_ARGS+=(--max-workers "$MAX_WORKERS")
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

$PYTHON_CMD scripts/cluster_eval/how2sign_scripts/gpt4v_evaluation_how2sign.py \
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

