#!/bin/bash
#
# Qwen2VL How2Sign Training on 2xA6000 GPU - LOCAL DEBUG VERSION
# Adapted from 2xA100 cluster script for local A6000 debugging

# Set up environment paths - adjust these based on your local conda environment
# Common locations to check:
# - /home/ztao/anaconda3/envs/qwen25_vl_sign
# - /local1/mhu/miniconda3/envs/qwenvl
# - ~/miniconda3/envs/qwenvl
CONDA_ENV_NAME="qwen25_vl_sign"  # Change this to your conda environment name
CONDA_BASE_PATH="${CONDA_BASE:-$HOME/anaconda3}"  # Adjust if your conda is elsewhere

# Try to find conda environment
if [ -d "$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME" ]; then
    export PATH="$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME/bin:$PATH"
    echo "✅ Using conda environment: $CONDA_BASE_PATH/envs/$CONDA_ENV_NAME"
elif [ -d "/home/ztao/anaconda3/envs/$CONDA_ENV_NAME" ]; then
    export PATH="/home/ztao/anaconda3/envs/$CONDA_ENV_NAME/bin:$PATH"
    echo "✅ Using conda environment: /home/ztao/anaconda3/envs/$CONDA_ENV_NAME"
elif [ -d "/local1/mhu/miniconda3/envs/$CONDA_ENV_NAME" ]; then
    export PATH="/local1/mhu/miniconda3/envs/$CONDA_ENV_NAME/bin:$PATH"
    echo "✅ Using conda environment: /local1/mhu/miniconda3/envs/$CONDA_ENV_NAME"
else
    echo "⚠️  Warning: Conda environment not found at expected locations"
    echo "   Please activate your conda environment manually before running this script"
    echo "   Example: conda activate $CONDA_ENV_NAME"
fi

export PYTHONPATH="/local1/mhu/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:$PYTHONPATH"
export OMP_NUM_THREADS=8
# Use new PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF is deprecated)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Network and Hugging Face configuration
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_TIMEOUT=600
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ENABLE_HF_TRANSFER=1

# Change to Qwen2-VL-Finetune directory
cd /local1/mhu/sign_language_llm/qwenvl/Qwen2-VL-Finetune

# Model configuration - using 3B model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Optimized training configuration for 2xA6000 GPUs (48GB each, similar to A100)
GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))


# Check if model is already cached
MODEL_CACHE_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "✅ Model found in cache: $MODEL_CACHE_DIR"
    echo "📊 Cache size: $(du -sh $MODEL_CACHE_DIR 2>/dev/null | cut -f1 || echo 'N/A')"
else
    echo "⚠️  Model not found in cache, will download during training"
fi


DATA_PATH="/local1/mhu/sign_language_llm/how2sign/video/segmented_train_videos_corrupted_removed.json"
IMAGE_FOLDER="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224"


# Create output directory
OUTPUT_DIR="/local1/mhu/sign_language_llm/qwenvl/outputs/qwen2vl_how2sign_2xa6000_top2_debug"
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""


# Run training with DeepSpeed ZeRO-3 for efficient multi-GPU training
# Enable gradient checkpointing to tame activation memory growth while keeping the top vision blocks trainable.
echo "🏃 Starting training with DeepSpeed..."
deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero3_qwen2vl.json \
    --model_id $MODEL_NAME \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 7 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((224 * 224)) \
    --video_max_pixels $((224 * 224)) \
    --fps 12 \
    --max_grad_norm 1.0 \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 2 \
    --use_liger True \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --gradient_checkpointing True \
    --lora_enable True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --vision_lr 2e-5 \
    --merger_lr 2e-5 \
    --report_to none

TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Output saved to: $OUTPUT_DIR"
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE."
    echo "Please check the error messages above for details."
    exit $TRAINING_EXIT_CODE
fi

