#!/bin/bash -l
# NOTE the -l flag!
#
# InternVL2.5-2B How2Sign Fine-Tuning on 2×A100 GPUs

#SBATCH --job-name=internvl25_how2sign_2xa100
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=a100:2
#SBATCH --partition tier3
#SBATCH --mem=128G

set -euo pipefail

# Load toolchains only if not already present (avoids errors with set -e)
if ! spack find --loaded /lhqcen5 >/dev/null 2>&1; then
    spack load /lhqcen5
fi
if ! spack find --loaded cuda@12.4.0/obxqih4 >/dev/null 2>&1; then
    spack load cuda@12.4.0/obxqih4
fi

# Explicit CUDA environment setup (required before training)
CUDA_ROOT="$(spack location -i /obxqih4)"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Activate fine-tuning environment
source /home/mh2803/miniconda3/bin/activate /home/mh2803/miniconda3/envs/internvl
export PATH="/home/mh2803/miniconda3/envs/internvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/InternVL:/home/mh2803/projects/sign_language_llm/InternVL/internvl_chat:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CUDA_EXT=0
export DS_BUILD_CPU_ADAM=0
export DEEPSPEED_CPU_ADAM=1
if [ -z "${CUDA_HOME:-}" ]; then
    CUDA_HOME="$(spack location -i /obxqih4)"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export DS_BUILD_OPS=0

# Network and Hugging Face configuration
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_TIMEOUT=600
export HF_HUB_DOWNLOAD_TIMEOUT=600
# Fast transfer requires hf_transfer package; disable unless installed
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

# Change to InternVL directory
cd /home/mh2803/projects/sign_language_llm/InternVL

# Model and data configuration
MODEL_NAME="OpenGVLab/InternVL2_5-2B"
OUTPUT_DIR="/home/mh2803/projects/sign_language_llm/InternVL/output/how2sign/internvl2_5_2B_2xa100"
META_PATH="/home/mh2803/projects/sign_language_llm/InternVL/data/how2sign/train_how2sign_under10s_meta.json"
IMAGE_ROOT="/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224/"


GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
NUM_DEVICES=${NUM_DEVICES:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))}

# Memory envelopes (from internvl2_5_2b_dynamic_res_2nd_finetune_lora)
DEEPSPEED_CONFIG="internvl_chat/zero_stage1_config.json"
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
MAX_BUFFER_SIZE=${MAX_BUFFER_SIZE:-20}
NUM_IMAGES_EXPECTED=${NUM_IMAGES_EXPECTED:-128}
MAX_NUM_FRAME=${MAX_NUM_FRAME:-96}

# Video frame sampling method
SAMPLING_METHOD='fps12.0'
# SAMPLING_METHOD='rand'


if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

MODEL_CACHE_DIR="/home/mh2803/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-2B"
if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "✅ Model found in cache: $MODEL_CACHE_DIR"
    echo "📊 Cache size: $(du -sh "$MODEL_CACHE_DIR" | cut -f1)"
else
    echo "⚠️  Model not found in cache, will download during training"
fi

# Generate unique port using SLURM job ID
# Add a small random offset to avoid conflicts with previous jobs that might not have cleaned up
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Use job ID as base, add a small random offset (0-99) to avoid stale port conflicts
    RAND_OFFSET=$((SLURM_JOB_ID % 100))
    # Port formula: 20000 + (job_id % 9900) + random_offset
    # This ensures ports are in range 20000-29999
    BASE_PORT=$((20000 + (SLURM_JOB_ID % 9900)))
    MASTER_PORT=${MASTER_PORT:-$((BASE_PORT + RAND_OFFSET))}
    # Ensure port doesn't exceed 29999
    if [ $MASTER_PORT -gt 29999 ]; then
        MASTER_PORT=$((MASTER_PORT - 10000))
    fi
else
    # Fallback to random port if not running under SLURM
    MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
fi
export MASTER_PORT
echo "MASTER_PORT: $MASTER_PORT (Job ID: ${SLURM_JOB_ID:-N/A})"

deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/internvl_chat_finetune_local.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --meta_path "$META_PATH" \
    --conv_style internvl2_5 \
    --use_fast_tokenizer False \
    --do_train True \
    --num_train_epochs 5 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 4e-5 \
    --vision_select_layer -1 \
    --force_image_size 224 \
    --max_dynamic_patch 6 \
    --dynamic_image_size True \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.0 \
    --freeze_llm True \
    --freeze_backbone False \
    --freeze_mlp False \
    --unfreeze_vit_layers 0 \
    --use_llm_lora 16 \
    --bf16 True \
    --max_seq_length $MAX_SEQ_LENGTH \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 10 \
    --logging_first_step True \
    --evaluation_strategy no \
    --report_to none \
    --grad_checkpoint True \
    --dataloader_num_workers 4 \
    --use_thumbnail True \
    --ps_version v2 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --group_by_length True \
    --use_packed_ds False \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --min_num_frame 32 \
    --max_num_frame $MAX_NUM_FRAME \
    --sampling_method "$SAMPLING_METHOD" \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine

TRAINING_EXIT_CODE=${PIPESTATUS[0]}
