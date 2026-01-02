#!/bin/bash
#
# Qwen3-VL-2B How2Sign Evaluation on 2×A6000 GPUs - LOCAL VERSION
# Set up conda environment
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate qwenvl
echo "✅ Conda environment activated: qwenvl"

#### /local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224/1L0GSTIb9Vs_4-5-rgb_front.mp4
## /local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224/0rsdwK7sHjE_18-5-rgb_front.mp4
# Essential environment variables
export PYTHONPATH="/code/doc_sign_search/qwenvl/Qwen2-VL-Finetune/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /code/doc_sign_search

# GPU configuration
GPU_IDS=${GPU_IDS:-"3"}  # Default: use GPU 0 and 1
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Configuration
# Set CHECKPOINT_PATH to empty string or unset to use base model only
# Update checkpoint path to point to your trained checkpoint (or leave empty for base model)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen3-VL-8B-Instruct}"
VIDEO_FOLDER="${VIDEO_FOLDER:-/mnt/localssd/doc_sign_search/openasl_test_videos}"
# QUESTION_FILE="${QUESTION_FILE:-/local1/mhu/sign_language_llm/how2sign/video/test_raw_videos/segmented_test_videos_filtered_110_samples.json}"
QUESTION_FILE="${QUESTION_FILE:-/code/doc_sign_search/how2sign/video/test_raw_videos/test_openasl_internvl_converted.json}"
# QUESTION_FILE="${QUESTION_FILE:-/code/doc_sign_search/InternVL/data/how2sign/test_how2sign_internvl.jsonl}"

OUT_DIR="${OUT_DIR:-/code/doc_sign_search/outputs/qwen2vl_eval/}"

# Evaluation parameters
MAX_SAMPLES=${MAX_SAMPLES:-2000}  # Set to a number to limit samples, empty for full evaluation
MIN_PIXELS=${MIN_PIXELS:-$((224*224))}  # Match training: 224x224
MAX_PIXELS=${MAX_PIXELS:-$((224*224))}  # Match training: 224x224
VIDEO_FPS=${VIDEO_FPS:-15}  # Match training FPS
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

USE_FRAMES=${USE_FRAMES:-false}  # Set to true to extract frames and use as images
NUM_FRAMES=${NUM_FRAMES:-}  # Optional: Number of frames to extract (if empty, use fps-based sampling)
SAVE_FRAMES=${SAVE_FRAMES:-false}  # Set to true to save extracted frames to disk

echo "🎬 Qwen3-VL-2B How2Sign Evaluation on 2×A6000"
echo "=================================================="
echo "Checkpoint: ${CHECKPOINT_PATH:-None (using base model)}"
echo "Model Base: $MODEL_BASE"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo "GPU IDs: $GPU_IDS"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Max Samples: ${MAX_SAMPLES:-Full evaluation}"
[ -n "$VIDEO_FPS" ] && echo "Video FPS: $VIDEO_FPS"
[ "$USE_FRAMES" = "true" ] && echo "Use Frames: Yes (extract frames as images)" && [ -n "$NUM_FRAMES" ] && echo "Num Frames: $NUM_FRAMES"
[ "$SAVE_FRAMES" = "true" ] && echo "Save Frames: Yes (frames will be saved to output directory)"
echo ""

mkdir -p "$OUT_DIR"

# Clear checkpoint path if it doesn't exist
[ -n "$CHECKPOINT_PATH" ] && [ ! -d "$CHECKPOINT_PATH" ] && CHECKPOINT_PATH=""

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

LOG_FILE="${OUT_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"

# Build command arguments
EVAL_ARGS=(
    --model-base "$MODEL_BASE"
    --video-folder "$VIDEO_FOLDER"
    --question-file "$QUESTION_FILE"
    --out-dir "$OUT_DIR"
    --max-new-tokens "$MAX_NEW_TOKENS"
)

[ -n "$CHECKPOINT_PATH" ] && EVAL_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH")
[ -n "$MAX_SAMPLES" ] && EVAL_ARGS+=(--max-samples "$MAX_SAMPLES")
[ -n "$VIDEO_FPS" ] && EVAL_ARGS+=(--video-fps "$VIDEO_FPS")
[ "$USE_FRAMES" = "true" ] && EVAL_ARGS+=(--use-frames)
[ -n "$NUM_FRAMES" ] && EVAL_ARGS+=(--num-frames "$NUM_FRAMES")
[ "$SAVE_FRAMES" = "true" ] && EVAL_ARGS+=(--save-frames)

CONDA_PYTHON="${HOME}/anaconda3/envs/qwenvl/bin/python"
[ ! -f "$CONDA_PYTHON" ] && CONDA_PYTHON=$(which python3)

$CONDA_PYTHON scripts/cluster_eval/how2sign_scripts/qwen3vl_evaluation_how2sign.py \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo "📁 Results: $OUT_DIR | 📝 Log: $LOG_FILE"
else
    echo "❌ Evaluation failed (exit code: $EVAL_EXIT_CODE)"
    echo "Check log: $LOG_FILE"
    exit $EVAL_EXIT_CODE
fi

