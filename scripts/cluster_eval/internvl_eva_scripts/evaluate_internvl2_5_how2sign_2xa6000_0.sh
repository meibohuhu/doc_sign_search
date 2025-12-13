#!/bin/bash
#
# InternVL2.5-2B How2Sign Evaluation on 2×A6000 GPUs - LOCAL VERSION
# Set up conda environment
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate internvl
echo "✅ Conda environment activated: internvl"

# Essential environment variables
export PYTHONPATH="/code/doc_sign_search/InternVL/internvl_chat:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /code/doc_sign_search

# GPU configuration
GPU_IDS=${GPU_IDS:-"0"}  # Default: use GPU 0 and 1
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Configuration
# Set CHECKPOINT_PATH to empty string or unset to use base model only
# Update checkpoint path to point to your trained checkpoint (or leave empty for base model)
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"  # Empty by default - will use base model
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/local1/mhu/sign_language_llm/InternVL/checkpoints/finetune_internvl2_5_openasl_16fps_1130/checkpoint-8499}"

# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/code/doc_sign_search/script_adobe/checkpoints/finetune_internvl2_5_how2sign_8b_16fps_1209/checkpoint-2548}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/code/doc_sign_search/script_adobe/checkpoints/finetune_internvl2_5_how2sign_8b_16fps_1209/checkpoint-3054}"

# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/local1/mhu/sign_language_llm/InternVL/checkpoints/finetune_internvl2_5_how2sign_16fps_1130/checkpoint-2399}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/local1/mhu/sign_language_llm/InternVL/checkpoints/finetune_internvl2_5_how2sign_20fps/checkpoint-2874}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/local1/mhu/sign_language_llm/InternVL/checkpoints/finetune_internvl2_5_how2sign_20fps/checkpoint-2399}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/local1/mhu/sign_language_llm/InternVL/checkpoints/finetune_internvl2_5_how2sign_20fps_freeze/checkpoint-2874}"


MODEL_BASE="${MODEL_BASE:-OpenGVLab/InternVL2_5-8B}"
# MODEL_BASE="${MODEL_BASE:-OpenGVLab/InternVL2_5-1B}"
# VIDEO_FOLDER="${VIDEO_FOLDER:-}"
VIDEO_FOLDER="${VIDEO_FOLDER:-/mnt/localssd/doc_sign_search/how2sign_test_videos_224x224}"

QUESTION_FILE="${QUESTION_FILE:-/code/doc_sign_search/InternVL/data/how2sign/test_how2sign_internvl.jsonl}"
# QUESTION_FILE="${QUESTION_FILE:-/local1/mhu/sign_language_llm/InternVL/data/openasl/test_openasl_internvl.jsonl}"
# QUESTION_FILE="${QUESTION_FILE:-/code/doc_sign_search/InternVL/data/how2sign/test_how2sign_internvl_sample550.jsonl}"

OUT_DIR="${OUT_DIR:-/code/doc_sign_search/outputs/internvl_eval/}"

# Evaluation parameters
# MAX_SAMPLES=${MAX_SAMPLES:-20}  # Set to a number to limit samples, empty for full evaluation
MAX_SAMPLES=${MAX_SAMPLES:-2337}
MIN_NUM_FRAMES=${MIN_NUM_FRAMES:-32}  # Minimum number of frames (set to 6 to ensure 6 frames)
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-130}  # Maximum number of frames (set to 6 to fix at 6 frames)
SAMPLING_METHOD=${SAMPLING_METHOD:-fps16.0}  # Sampling method: 'fpsX.X' or 'uniform' for uniform sampling
IMAGE_SIZE=${IMAGE_SIZE:-224}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
EXPORT_FRAMES=${EXPORT_FRAMES:-false}  # Set to "true" to enable frame export

echo "🎬 InternVL2.5-2B How2Sign Evaluation on 2×A6000"
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
echo "Min Num Frames: $MIN_NUM_FRAMES"
echo "Max Num Frames: $MAX_NUM_FRAMES"
echo "Sampling Method: $SAMPLING_METHOD"
echo "Image Size: $IMAGE_SIZE"
echo "Max New Tokens: $MAX_NEW_TOKENS"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max Samples (limited): $MAX_SAMPLES"
else
    echo "Max Samples: Full evaluation"
fi
if [ "$EXPORT_FRAMES" = "true" ]; then
    echo "Export Frames: Enabled (frames will be saved to $OUT_DIR/extracted_frames/)"
else
    echo "Export Frames: Disabled"
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
    --min-num-frames "$MIN_NUM_FRAMES"
    --max-num-frames "$MAX_NUM_FRAMES"
    --sampling-method "$SAMPLING_METHOD"
    --image-size "$IMAGE_SIZE"
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

# Add export-frames if enabled
if [ "$EXPORT_FRAMES" = "true" ]; then
    EVAL_ARGS+=(--export-frames)
fi

# Run evaluation
echo "🚀 Starting evaluation..."
echo ""

# Get python path from conda environment (works even if conda activate fails in non-interactive shell)
CONDA_PYTHON="$HOME/anaconda3/envs/internvl/bin/python"
if [ ! -f "$CONDA_PYTHON" ]; then
    # Fallback to system python if conda env doesn't exist
    CONDA_PYTHON=$(which python3)
    echo "⚠️  Warning: Using system python: $CONDA_PYTHON"
else
    echo "Using conda python: $CONDA_PYTHON"
fi
echo ""

$CONDA_PYTHON scripts/cluster_eval/internvl_eva_scripts/internvl_evaluation_how2sign.py \
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

