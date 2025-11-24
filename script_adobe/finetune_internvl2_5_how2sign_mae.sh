#!/bin/bash
#
# InternVL2.5-2B How2Sign Fine-Tuning on 4*A100
# Set up conda environment - matches setup_internvl_auto.txt
# Anaconda is installed at $HOME/anaconda3 by setup script
# Environment name: internvl
# Initialize and activate conda environment
# source "$HOME/anaconda3/etc/profile.d/conda.sh"
# conda activate internvl
# echo "✅ Conda environment activated: internvl"

# Essential environment variables
export PYTHONPATH="/code/doc_sign_search/InternVL/internvl_chat:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Change to InternVL directory
cd /code/doc_sign_search/InternVL

# GPU configuration
# Specify which GPUs to use (comma-separated, e.g., "0,1,2,3" for GPU 0, 1, 2, and 3)
GPU_IDS=${GPU_IDS:-"0,1,2,3"}  # Default: use GPU 0, 1, 2, 3
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Calculate number of devices from GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Model and data configuration
MODEL_NAME="OpenGVLab/InternVL2_5-2B"
OUTPUT_DIR="/code/doc_sign_search/script_adobe/checkpoints/internvl2_5_2B_mae_how2sign"
META_PATH="/code/doc_sign_search/script_adobe/train_how2sign_meta.json"
VIDEO_BASE_PATH="/mnt/localssd/doc_sign_search/train_crop_videos_224"

# Optimized training configuration
# Note: NUM_DEVICES is automatically calculated from GPU_IDS above
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))}

IMAGE_SIZE=${IMAGE_SIZE:-224}
MIN_NUM_FRAMES=${MIN_NUM_FRAMES:-8}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-64}
SAMPLING_METHOD=${SAMPLING_METHOD:-random_start_every2}
MASK_RATIO=${MASK_RATIO:-0.80}
MASK_STRATEGY=${MASK_STRATEGY:-random}
DECODER_DIM=${DECODER_DIM:-384}
DECODER_DEPTH=${DECODER_DEPTH:-6}
DECODER_HEADS=${DECODER_HEADS:-12}
FREEZE_ENCODER=${FREEZE_ENCODER:-False}
UNFREEZE_TOPK_VISION=${UNFREEZE_TOPK_VISION:-0}

export MASTER_PORT=29500
LOG_FILE="${OUTPUT_DIR}/mae_training_$(date +%Y%m%d_%H%M%S).log"

FREEZE_ARGS=""
[ "$FREEZE_ENCODER" = "True" ] && FREEZE_ARGS="--freeze_encoder True"
[ "$UNFREEZE_TOPK_VISION" -gt 0 ] && [ -z "$FREEZE_ARGS" ] && FREEZE_ARGS="--unfreeze_topk_vision $UNFREEZE_TOPK_VISION"

mkdir -p "$OUTPUT_DIR"
[ -f "$META_PATH" ] || { echo "❌ Meta file not found: $META_PATH"; exit 1; }
[ -d "$VIDEO_BASE_PATH" ] || { echo "❌ Video folder not found: $VIDEO_BASE_PATH"; exit 1; }

deepspeed --include localhost:$GPU_IDS --master_port=$MASTER_PORT \
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
    --image_size $IMAGE_SIZE \
    --min_num_frames $MIN_NUM_FRAMES \
    --max_num_frames $MAX_NUM_FRAMES \
    --sampling_method "$SAMPLING_METHOD" \
    --mask_ratio $MASK_RATIO \
    --mask_strategy "$MASK_STRATEGY" \
    --decoder_dim $DECODER_DIM \
    --decoder_depth $DECODER_DEPTH \
    --decoder_heads $DECODER_HEADS \
    --norm_pix_loss True \
    --spacetime_mask True \
    $FREEZE_ARGS \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_interval  10000 \
    --log_interval 10 \
    --num_workers 4 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed internvl_chat/zero_stage3_config.json \
    --local_rank -1 \
    2>&1 | tee "$LOG_FILE"

