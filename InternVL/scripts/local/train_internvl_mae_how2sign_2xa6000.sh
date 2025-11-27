#!/bin/bash
set -e

export PYTHONPATH="/local1/mhu/sign_language_llm/InternVL:/local1/mhu/sign_language_llm/InternVL/internvl_chat:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CUDA_EXT=0
export DS_BUILD_CPU_ADAM=0
export DEEPSPEED_CPU_ADAM=1
export PYTHONUNBUFFERED=1

cd /local1/mhu/sign_language_llm/InternVL

MODEL_NAME="OpenGVLab/InternVL2_5-2B"
OUTPUT_DIR="/local1/mhu/sign_language_llm/InternVL/output/how2sign/internvl2_5_2B_mae_2xa6000/checkpoints"
META_PATH="/local1/mhu/sign_language_llm/InternVL/data/how2sign/train_how2sign_meta.json"
VIDEO_BASE_PATH="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224"

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
NUM_DEVICES=${NUM_DEVICES:-2}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

mkdir -p "$OUTPUT_DIR"
[ -f "$META_PATH" ] || { echo "❌ Meta file not found: $META_PATH"; exit 1; }
[ -d "$VIDEO_BASE_PATH" ] || { echo "❌ Video folder not found: $VIDEO_BASE_PATH"; exit 1; }

MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
LOG_FILE="${OUTPUT_DIR}/mae_training_$(date +%Y%m%d_%H%M%S).log"

deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/train_internvl_mae.py \
    --model_id "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$META_PATH" \
    --video_base_path "$VIDEO_BASE_PATH" \
    --batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_epochs 50 \
    --learning_rate 1.5e-4 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --image_size 224 \
    --min_num_frames 8 \
    --max_num_frames 8 \
    --sampling_method random_start_every2 \
    --mask_ratio 0.80 \
    --mask_strategy random \
    --decoder_dim 384 \
    --decoder_depth 6 \
    --decoder_heads 12 \
    --norm_pix_loss True \
    --spacetime_mask True \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_interval 50 \
    --log_interval 2 \
    --num_workers 2 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed internvl_chat/zero_stage3_config.json \
    --local_rank -1 \
    2>&1 | tee "$LOG_FILE"

