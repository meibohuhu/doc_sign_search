#!/bin/bash
# Qwen2-VL Foreground-Background Consistency Finetuning on 2x A6000 (local debug)

set -e

export PYTHONPATH="/local1/mhu/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:${PYTHONPATH:-}"

cd /local1/mhu/sign_language_llm/qwenvl/Qwen2-VL-Finetune

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/local1/mhu/sign_language_llm/how2sign/video/segmented_train_videos_corrupted_removed.json"
IMAGE_FOLDER="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224"
MASK_FOLDER="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_720_mask"
OUTPUT_DIR="/local1/mhu/sign_language_llm/qwenvl/outputs/qwen2vl_how2sign_2xa6000_fbcf"

GLOBAL_BATCH_SIZE=8
PER_DEVICE_BS=1
NUM_DEVICES=2
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BS * NUM_DEVICES)))
FPS=12

mkdir -p "$OUTPUT_DIR"

# Log file setup
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Training log will be saved to: $LOG_FILE"
echo "📝 Training started at $(date)" | tee -a "$LOG_FILE"

# NCCL settings (reduced logging for better performance)
export NCCL_DEBUG=WARN  # Reduced from INFO to WARN to reduce overhead
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
# Save NCCL logs to file (only warnings/errors)
export NCCL_DEBUG_FILE="$OUTPUT_DIR/nccl_debug_%h_%p.log"

# Enable DeepSpeed logging (reduced for better performance)
export DS_LOG_LEVEL=WARN  # Reduced from INFO to WARN

# Run training with logging (tee to both file and stdout)
deepspeed src/train/train_sft.py \
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
  --num_train_epochs 1 \
  --max_steps 2000 \
  --per_device_train_batch_size $PER_DEVICE_BS \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --video_min_pixels $((224 * 224)) \
  --video_max_pixels $((224 * 224)) \
  --fps $FPS \
  --max_grad_norm 1.0 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --dataloader_num_workers 2 \
  --dataloader_pin_memory True \
  --dataloader_prefetch_factor 2 \
  --dataloader_persistent_workers False \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --use_liger True \
  --freeze_llm True \
  --freeze_vision_tower False \
  --freeze_merger False \
  --bf16 True \
  --disable_flash_attn2 True \
  --gradient_checkpointing True \
  --lora_enable True \
  --lora_rank 16 \
  --lora_alpha 32 \
  --vision_lr 2e-5 \
  --merger_lr 2e-5 \
  --ddp_find_unused_parameters True \
  2>&1 | tee -a "$LOG_FILE"

# Log completion
echo "📝 Training completed at $(date)" | tee -a "$LOG_FILE"
echo "📝 Log file saved to: $LOG_FILE"

