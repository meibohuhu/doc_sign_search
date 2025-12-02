#!/bin/bash -l
# NOTE the -l flag!
#
# InternVL2.5-2B How2Sign Fine-Tuning on 2×A100 GPUs
# Single-node recipe using DeepSpeed ZeRO-2 by default, with optional ZeRO-3 + CPU offload
# Memory guidance:
#   - Default (ZeRO-2): 8k seq / 16k packed / 96 frames works on 2×A100 40 GB.
#   - Enable ZeRO-3 by exporting USE_ZERO_STAGE3=1 to uplift defaults to 12k seq / 20k packed / 128 frames.
#   - You can still override any MAX_* env if you need tighter or looser bounds.

#SBATCH --job-name=internvl25_how2sign_2xa100
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:08:00
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
OUTPUT_DIR="/home/mh2803/projects/sign_language_llm/InternVL/output/how2sign/internvl2_5_2B_2xa100_mae"
META_PATH="/home/mh2803/projects/sign_language_llm/InternVL/data/how2sign/train_how2sign_meta.json"
VIDEO_BASE_PATH="/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224/"

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
NUM_DEVICES=${NUM_DEVICES:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))}


MODEL_CACHE_DIR="/home/mh2803/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-2B"


MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
export MASTER_PORT
echo "MASTER_PORT: $MASTER_PORT"

deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/train_internvl_mae.py \
    --model_id "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$META_PATH" \
    --video_base_path "$VIDEO_BASE_PATH" \
    --batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_epochs 30 \
    --learning_rate 1.5e-4 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --image_size 224 \
    --min_num_frames 8 \
    --max_num_frames 32 \
    --sampling_method random_start_every2 \
    --mask_ratio 0.75 \
    --mask_strategy random \
    --decoder_dim 384 \
    --decoder_depth 6 \
    --decoder_heads 12 \
    --norm_pix_loss True \
    --spacetime_mask True \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_interval 10000 \
    --log_interval 2 \
    --num_workers 2 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed internvl_chat/zero_stage3_config.json \
    --local_rank -1 \

TRAINING_EXIT_CODE=$?


