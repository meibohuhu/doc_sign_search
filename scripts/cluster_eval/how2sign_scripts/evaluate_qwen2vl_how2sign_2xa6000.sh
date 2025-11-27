#!/bin/bash
#
# Qwen2.5-VL-3B How2Sign Evaluation on 2×A6000 GPUs - LOCAL VERSION
# Set up conda environment
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate qwenvl
echo "✅ Conda environment activated: qwenvl"

# Essential environment variables
export PYTHONPATH="/local1/mhu/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /local1/mhu/sign_language_llm

# GPU configuration
GPU_IDS=${GPU_IDS:-"0,1"}  # Default: use GPU 0 and 1
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Configuration
# Set CHECKPOINT_PATH to empty string or unset to use base model only
# Update checkpoint path to point to your trained checkpoint (or leave empty for base model)
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/local1/mhu/sign_language_llm/InternVL/checkpoints/qwen2vl_how2sign_4xa100_filtered_32batchsize_robust/checkpoint-5000}"  # Empty by default - will use base model
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen2.5-VL-3B-Instruct}"
VIDEO_FOLDER="${VIDEO_FOLDER:-/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224/}"
QUESTION_FILE="${QUESTION_FILE:-/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_test_videos_filtered_110_samples.json}"
OUT_DIR="${OUT_DIR:-/local1/mhu/sign_language_llm/outputs/qwen2vl_eval/}"

# Evaluation parameters
MAX_SAMPLES=${MAX_SAMPLES:-55}  # Set to a number to limit samples, empty for full evaluation
MIN_PIXELS=${MIN_PIXELS:-$((224*224))}  # Match training: 224x224
MAX_PIXELS=${MAX_PIXELS:-$((224*224))}  # Match training: 224x224
VIDEO_FPS=${VIDEO_FPS:-18}  # Match training FPS
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

echo "🎬 Qwen2.5-VL-3B How2Sign Evaluation on 2×A6000"
echo "=================================================="
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint: $CHECKPOINT_PATH"
else
    echo "Checkpoint: None (using base model)"
fi
echo "Model Base: $MODEL_BASE"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo "GPU IDs: $GPU_IDS (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
echo "Number of GPUs: $NUM_DEVICES"
echo "Min Pixels: $MIN_PIXELS"
echo "Max Pixels: $MAX_PIXELS"
echo "Video FPS: $VIDEO_FPS"
echo "Max New Tokens: $MAX_NEW_TOKENS"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples (limited): $MAX_SAMPLES"
else
    echo "Max Samples: Full evaluation"
fi
echo ""

# Create output directory
mkdir -p "$OUT_DIR"
echo "📁 Output directory: $OUT_DIR"
echo ""

# Check if checkpoint exists (only if CHECKPOINT_PATH is provided)
if [ -n "$CHECKPOINT_PATH" ]; then
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "⚠️  Warning: Checkpoint path does not exist: $CHECKPOINT_PATH"
        echo "   Will use base model instead."
        CHECKPOINT_PATH=""  # Clear checkpoint path to use base model
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

# Log file with timestamp
LOG_FILE="${OUT_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"
echo ""

# Build command arguments
EVAL_ARGS=(
    --model-base "$MODEL_BASE"
    --video-folder "$VIDEO_FOLDER"
    --question-file "$QUESTION_FILE"
    --out-dir "$OUT_DIR"
    --min-pixels "$MIN_PIXELS"
    --max-pixels "$MAX_PIXELS"
    --video-fps "$VIDEO_FPS"
    --max-new-tokens "$MAX_NEW_TOKENS"
)

# Add checkpoint path only if provided
if [ -n "$CHECKPOINT_PATH" ]; then
    EVAL_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH")
fi

# Add max-samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    EVAL_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# Run evaluation
echo "🚀 Starting evaluation..."
echo ""

# Get python path from conda environment (works even if conda activate fails in non-interactive shell)
CONDA_PYTHON="$HOME/anaconda3/envs/qwenvl/bin/python"
if [ ! -f "$CONDA_PYTHON" ]; then
    # Fallback to system python if conda env doesn't exist
    CONDA_PYTHON=$(which python3)
    echo "⚠️  Warning: Using system python: $CONDA_PYTHON"
else
    echo "Using conda python: $CONDA_PYTHON"
fi
echo ""

$CONDA_PYTHON scripts/cluster_eval/how2sign_scripts/qwen2vl_evaluation_how2sign_claude_local.py \
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

