#!/bin/bash
# Qwen2-VL Foreground-Background Consistency Finetuning on 2x A6000 (local debug)

set -euo pipefail

CONDA_ENV_NAME="qwen25_vl_sign"
CONDA_BASE_PATH="${CONDA_BASE:-$HOME/anaconda3}"

activate_env() {
  local env_path="$1/envs/$CONDA_ENV_NAME"
  if [ -d "$env_path" ]; then
    export PATH="$env_path/bin:$PATH"
    echo "✅ Using conda environment: $env_path"
    return 0
  fi
  return 1
}

activate_env "$CONDA_BASE_PATH" || \
activate_env "/home/ztao/anaconda3" || \
activate_env "/local1/mhu/miniconda3" || \
{
  echo "⚠️  Conda env $CONDA_ENV_NAME not found. Activate manually first."
  exit 1
}

export PYTHONPATH="/local1/mhu/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
# Clear GPU cache periodically
export CUDA_LAUNCH_BLOCKING=0

cd /local1/mhu/sign_language_llm/qwenvl/Qwen2-VL-Finetune

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/local1/mhu/sign_language_llm/how2sign/video/segmented_train_videos_with_masks.json"
IMAGE_FOLDER="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224"
MASK_FOLDER="/local1/mhu/sign_language_llm/how2sign/masks"
OUTPUT_DIR="/local1/mhu/sign_language_llm/qwenvl/outputs/qwen2vl_how2sign_2xa6000_fbcf"

GLOBAL_BATCH_SIZE=8
PER_DEVICE_BS=1
NUM_DEVICES=2
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BS * NUM_DEVICES)))
FPS=12

mkdir -p "$OUTPUT_DIR"

# Log file with timestamp
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"

echo "🚀 Qwen2-VL FBCF training on 2xA6000"
echo "Model          : $MODEL_NAME"
echo "Data JSON      : $DATA_PATH"
echo "Video folder   : $IMAGE_FOLDER"
echo "Mask folder    : $MASK_FOLDER"
echo "Output dir     : $OUTPUT_DIR"
echo "Batch/GPU      : $PER_DEVICE_BS"
echo "Grad accum     : $GRAD_ACCUM"
echo "fps            : $FPS"
echo "DeepSpeed      : ZeRO-3 with CPU offload"
echo "FBCF Sampling  : Per-step deterministic (40% original, 40% foreground, 20% background)"
echo "Memory optimizations: FPS=$FPS, LoRA rank=8, unfreeze_topk_vision=2, workers=2"
echo ""

python - <<'PY'
import torch, os
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        print(f"  GPU {idx}: {prop.name} ({prop.total_memory/1024**3:.1f} GB)")
PY

echo ""
echo "🏃 Launching training with DeepSpeed..."
echo "📝 Logging to: $LOG_FILE"
echo ""

# Run training and log both stdout and stderr to file, while also displaying on terminal
# Note: Using zero3_qwen2vl.json which includes CPU offload for optimizer and parameters
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
  --fbcf_lambda 0.2 \
  --fg_loss_weight 1.0 \
  --bg_loss_weight 1.0 \
  --fbcf_sampling_mode True \
  --fbcf_sampling_ratio_original 0.4 \
  --fbcf_sampling_ratio_foreground 0.4 \
  --fbcf_sampling_ratio_background 0.2 \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 1 \
  --max_steps 2000 \
  --per_device_train_batch_size $PER_DEVICE_BS \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --video_min_pixels $((224 * 224)) \
  --video_max_pixels $((224 * 224)) \
  --video_resized_width 224 \
  --video_resized_height 224 \
  --fps $FPS \
  --max_grad_norm 1.0 \
  --learning_rate 2e-5 \
  --dataloader_num_workers 2 \
  --dataloader_pin_memory False \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --use_liger True \
  --freeze_llm True \
  --freeze_vision_tower False \
  --freeze_merger False \
  --unfreeze_topk_vision 8 \
  --bf16 True \
  --disable_flash_attn2 True \
  --gradient_checkpointing True \
  --lora_enable True \
  --lora_rank 8 \
  --lora_alpha 16 \
  --vision_lr 2e-5 \
  --merger_lr 2e-5 \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Training finished. Results in $OUTPUT_DIR"
else
  echo "❌ Training failed (exit $EXIT_CODE)"
fi

