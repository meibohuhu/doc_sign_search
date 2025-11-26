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

export PYTHONPATH="/code/doc_sign_search/qwenvl/Qwen2-VL-Finetune/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /code/doc_sign_search/qwenvl/Qwen2-VL-Finetune

# GPU configuration
# Specify which GPUs to use (comma-separated, e.g., "0,1,2,3" for GPU 0, 1, 2, and 3)
GPU_IDS=${GPU_IDS:-"0,1,2,3"}  # Default: use GPU 0, 1, 2, 3
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Calculate number of devices from GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Model and data configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/code/doc_sign_search/script_adobe/checkpoints/qwen2_5_vl_how2sign_fbcf_1"
DATA_PATH="/code/doc_sign_search/how2sign/video/segmented_train_videos_corrupted_removed.json"
IMAGE_FOLDER="/mnt/localssd/sign_mllm"
MASK_FOLDER="/mnt/localssd/train_crop_videos_720_mask"

# Optimized training configuration
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export MASTER_PORT=29500
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"
[ -f "$DATA_PATH" ] || { echo "❌ Data file not found: $DATA_PATH"; exit 1; }
[ -d "$IMAGE_FOLDER" ] || { echo "❌ Image folder not found: $IMAGE_FOLDER"; exit 1; }
[ -d "$MASK_FOLDER" ] || { echo "❌ Mask folder not found: $MASK_FOLDER"; exit 1; }

deepspeed --include localhost:$GPU_IDS --master_port=$MASTER_PORT \
    src/train/train_sft.py \
    --deepspeed scripts/zero3_qwen2vl.json \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --mask_folder "$MASK_FOLDER" \
    --mask_file_suffix .npz \
    --mask_key mask \
    --mask_dilation 3 \
    --mask_blur_kernel 5 \
    --fbcf_bg_noise_std 0.05 \
    --enable_fbcf True \
    --fbcf_lambda 0.15 \
    --fg_loss_weight 1.0 \
    --bg_loss_weight 1.0 \
    --fbcf_sampling_mode True \
    --fbcf_sampling_ratio_original 0.40 \
    --fbcf_sampling_ratio_foreground 0.40 \
    --fbcf_sampling_ratio_background 0.20 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 8 \
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
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 4 \
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
    --ddp_find_unused_parameters True \
    --report_to none \
    2>&1 | tee "$LOG_FILE"

