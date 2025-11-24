#!/bin/bash
#
# InternVL2.5-2B How2Sign Fine-Tuning on 2*A100
# Set up conda environment - matches setup_internvl_auto.txt
# Anaconda is installed at $HOME/anaconda3 by setup script
# Environment name: internvl
# Initialize and activate conda environment
# source "$HOME/anaconda3/etc/profile.d/conda.sh"
# conda activate internvl
# echo "✅ Conda environment activated: internvl"

export PYTHONPATH="/code/doc_sign_search/qwenvl/Qwen2-VL-Finetune/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /code/doc_sign_search/qwenvl/Qwen2-VL-Finetune

# GPU configuration
# Specify which GPUs to use (comma-separated, e.g., "0,1,2,3" for GPU 0, 1, 2, and 3)
GPU_IDS=${GPU_IDS:-"0,1"}  # Default: use GPU 0, 1, 2, 3
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Calculate number of devices from GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Model and data configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/code/doc_sign_search/script_adobe/checkpoints/qwen2_5_vl_how2sign_18fps"
DATA_PATH="/code/doc_sign_search/how2sign/video/segmented_train_videos_corrupted_removed.json"
IMAGE_FOLDER="/mnt/localssd/sign_mllm"

# Optimized training configuration
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export MASTER_PORT=29500
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"
[ -f "$DATA_PATH" ] || { echo "❌ Data file not found: $DATA_PATH"; exit 1; }
[ -d "$IMAGE_FOLDER" ] || { echo "❌ Image folder not found: $IMAGE_FOLDER"; exit 1; }

deepspeed --include localhost:$GPU_IDS --master_port=$MASTER_PORT \
    src/train/train_sft.py \
    --deepspeed scripts/zero3_qwen2vl.json \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 6 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((224 * 224)) \
    --video_max_pixels $((224 * 224)) \
    --fps 18 \
    --max_grad_norm 1.0 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --use_liger True \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --gradient_checkpointing True \
    --lora_enable True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --vision_lr 2e-5 \
    --merger_lr 2e-5 \
    --report_to none \
    2>&1 | tee "$LOG_FILE"

