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

# DeepSpeed build configuration (matching local script)
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CUDA_EXT=0
export DS_BUILD_CPU_ADAM=0
export DEEPSPEED_CPU_ADAM=1

# NCCL configuration for distributed training
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes (default is 10 minutes)
export NCCL_DEBUG=INFO    # Enable NCCL debug logging
export NCCL_DEBUG_SUBSYS=ALL  # Log all NCCL subsystems
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_SOCKET_IFNAME=^docker0,lo  # Use all interfaces except docker and loopback
export NCCL_P2P_DISABLE=0  # Enable P2P communication
export NCCL_SHM_DISABLE=0  # Enable shared memory
export OMP_NUM_THREADS=8   # Limit OpenMP threads

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

export MASTER_PORT=29508
LOG_FILE="${OUTPUT_DIR}/mae_training_$(date +%Y%m%d_%H%M%S).log"
NCCL_DEBUG_FILE="${OUTPUT_DIR}/nccl_debug_$(date +%Y%m%d_%H%M%S).log"
export NCCL_DEBUG_FILE="$NCCL_DEBUG_FILE"


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
    --image_size 224 \
    --min_num_frames 8 \
    --max_num_frames 48 \
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
    --save_interval  10000 \
    --log_interval 10 \
    --num_workers 2 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed internvl_chat/zero_stage3_config.json \
    --local_rank -1 \
    2>&1 | tee "$LOG_FILE"

