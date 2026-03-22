#!/bin/bash
#SBATCH --job-name=eval_sft_1b_broad
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --qos=a100_genai_interns_high
#SBATCH --account=genai_interns
#SBATCH --error=logs/run_%j.err
#SBATCH --output=logs/run_%j.out

set -x

GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

export PYTHONPATH="/home/zachsun/doc_sign_search/InternVL/internvl_chat:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MASTER_PORT=34223
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_API_KEY="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"
export WANDB_PROJECT="internvl-sign-search-eval"
export HF_HOME=/genai/fsx-project/zachsun

# Configuration change checkpoint path to point to your trained checkpoint (or leave empty for base model)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/genai/fsx-project/zachsun/checkpoint/doc_sign_search/finetune_stage1_broad_h2s_1_open_1_yt_1_0321/checkpoint-6578}"
MODEL_BASE="${MODEL_BASE:-OpenGVLab/InternVL2_5-1B}"
VIDEO_FOLDER="${VIDEO_FOLDER:-/genai/fsx-project/zachsun/dataset/doc_sign_search/how2sign_test_videos_224x224}"
QUESTION_FILE="${QUESTION_FILE:-/home/zachsun/doc_sign_search/InternVL/data/how2sign/test_how2sign_internvl.jsonl}"
OUT_DIR="${OUT_DIR:-/home/stu2/s15/mh2803/workspace/doc_sign_search/scripts_meta_new/0321/eval/outputs/eval_stage1_broad_h2s_1_open_1_yt_1_0321}"

# Evaluation parameters
MAX_SAMPLES=${MAX_SAMPLES:-2355}
MIN_NUM_FRAMES=${MIN_NUM_FRAMES:-32}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-150}
SAMPLING_METHOD=${SAMPLING_METHOD:-fps16.0}
IMAGE_SIZE=${IMAGE_SIZE:-224}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"

srun  \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  ${SRUN_ARGS} \
  python -u scripts/cluster_eval/internvl_eva_scripts/internvl_evaluation_how2sign_nogate_nogrpo_wandb.py \
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
    --wandb-run-name "eval_stage1_broad_h2s_1_open_1_yt_1_0321_checkpoint-6578" \
    --wandb-notes "Evaluation with wandb logging" \
    2>&1 | tee "$LOG_FILE"
