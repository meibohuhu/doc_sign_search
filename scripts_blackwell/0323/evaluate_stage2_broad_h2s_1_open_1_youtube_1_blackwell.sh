#!/bin/bash
#
# InternVL2.5-1B How2Sign Evaluation
#
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate internvl

# Essential environment variables
export PYTHONPATH="/home/stu2/s15/mh2803/workspace/doc_sign_search/InternVL/internvl_chat:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"
export WANDB_PROJECT="internvl-sign-search-eval"

cd /home/stu2/s15/mh2803/workspace/doc_sign_search

# GPU configuration
GPU_IDS=${GPU_IDS:-"0"}
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Configuration
# Note: Use checkpoint-1019-merged-v2 (LoRA weights merged, extra base_model wrapper removed)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch/mh2803/checkpoints/sft/finetune_stage2_broad_h2s_1_open_1_youtube_1_blackwell/checkpoint-5-fixed}"
MODEL_BASE="${MODEL_BASE:-OpenGVLab/InternVL2_5-1B}"
VIDEO_FOLDER="${VIDEO_FOLDER:-/scratch/mh2803/how2sign_test_videos_224x224}"
QUESTION_FILE="${QUESTION_FILE:-/home/stu2/s15/mh2803/workspace/doc_sign_search/InternVL/data/how2sign/test_how2sign_internvl.jsonl}"
OUT_DIR="${OUT_DIR:-/home/stu2/s15/mh2803/workspace/doc_sign_search/outputs/blackwell/test}"

# Evaluation parameters
MAX_SAMPLES=${MAX_SAMPLES:-10}
MIN_NUM_FRAMES=${MIN_NUM_FRAMES:-32}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-150}
SAMPLING_METHOD=${SAMPLING_METHOD:-fps16.0}
IMAGE_SIZE=${IMAGE_SIZE:-224}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Evaluating: $CHECKPOINT_PATH"
echo "   Output: $OUT_DIR"
echo "   Frames: $MIN_NUM_FRAMES-$MAX_NUM_FRAMES, Sampling: $SAMPLING_METHOD"
echo ""

python scripts/cluster_eval/internvl_eva_scripts/internvl_evaluation_how2sign_nogate_nogrpo_wandb.py \
    --model-base "$MODEL_BASE" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --video-folder "$VIDEO_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --out-dir "$OUT_DIR" \
    --min-num-frames "$MIN_NUM_FRAMES" \
    --max-num-frames "$MAX_NUM_FRAMES" \
    --sampling-method "$SAMPLING_METHOD" \
    --image-size "$IMAGE_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-samples "$MAX_SAMPLES" \
    --wandb-run-name "testttttt" \
    --wandb-notes "Blackwell evaluation with wandb logging" \
    2>&1 | tee "$LOG_FILE"

echo ""
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Evaluation completed. Results: $OUT_DIR"
else
    echo "❌ Evaluation failed. Log: $LOG_FILE"
    exit 1
fi
