#!/bin/bash -l
# NOTE the -l flag!
#
# Qwen2VL Foreground-Background Consistency Finetuning on 2xA100 GPU

#SBATCH --job-name=qwen2vl_fbcf_2xa100
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=a100:2
#SBATCH --partition tier3
#SBATCH --mem=128g

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths
CONDA_ENV_NAME="qwenvl"
CONDA_BASE_PATH="${CONDA_BASE:-$HOME/miniconda3}"

# Try to find conda environment
if [ -d "$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME" ]; then
    export PATH="$CONDA_BASE_PATH/envs/$CONDA_ENV_NAME/bin:$PATH"
    echo "✅ Using conda environment: $CONDA_BASE_PATH/envs/$CONDA_ENV_NAME"
elif [ -d "/home/mh2803/miniconda3/envs/$CONDA_ENV_NAME" ]; then
    export PATH="/home/mh2803/miniconda3/envs/$CONDA_ENV_NAME/bin:$PATH"
    echo "✅ Using conda environment: /home/mh2803/miniconda3/envs/$CONDA_ENV_NAME"
else
    echo "⚠️  Warning: Conda environment not found at expected locations"
    echo "   Please activate your conda environment manually before running this script"
    echo "   Example: conda activate $CONDA_ENV_NAME"
fi

export PYTHONPATH="/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:$PYTHONPATH"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# Clear GPU cache periodically
export CUDA_LAUNCH_BLOCKING=0

# Network and Hugging Face configuration
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_TIMEOUT=600
export HF_HUB_DOWNLOAD_TIMEOUT=600

# Disable wandb (Weights & Biases) to avoid API key issues in non-interactive environments
export WANDB_DISABLED=1

# Change to Qwen2-VL-Finetune directory
cd /home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Data configuration
DATA_PATH="/home/mh2803/projects/sign_language_llm/how2sign/video/segmented_train_videos_with_masks.json"
IMAGE_FOLDER="/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224/"
MASK_FOLDER="/home/mh2803/projects/sign_language_llm/how2sign/masks/"

# Training configuration for 2xA100 GPU
GLOBAL_BATCH_SIZE=4
PER_DEVICE_BS=1
NUM_DEVICES=2
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BS * NUM_DEVICES)))
FPS=12

# Output configuration
OUTPUT_DIR="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/1119/qwen2vl_fbcf_2xa100"
mkdir -p "$OUTPUT_DIR"

# Log file with timestamp
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"

echo "🚀 Qwen2-VL FBCF training on 2xA100"
echo "Model          : $MODEL_NAME"
echo "Data JSON      : $DATA_PATH"
echo "Video folder   : $IMAGE_FOLDER"
echo "Mask folder    : $MASK_FOLDER"
echo "Output dir     : $OUTPUT_DIR"
echo "Batch/GPU      : $NUM_DEVICES"
echo "Grad accum     : $GRAD_ACCUM"
echo "fps            : $FPS"
echo "DeepSpeed      : ZeRO-3"
echo "FBCF Sampling  : Per-sample random (20% original, 60% foreground, 20% background)"
echo "                Note: Reduced original ratio for two-stage training (FBCF -> Full Video FT)"
echo "FBCF Lambda    : 0.1 (reduced from 0.2 for Stage 1 - gentle background regularization)"
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
  --fbcf_lambda 0.1 \
  --fg_loss_weight 1.0 \
  --bg_loss_weight 1.0 \
  --fbcf_sampling_mode True \
  --fbcf_sampling_ratio_original 0.2 \
  --fbcf_sampling_ratio_foreground 0.6 \
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
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --use_liger True \
  --freeze_llm True \
  --freeze_vision_tower True \
  --freeze_merger False \
  --unfreeze_topk_vision 8 \
  --bf16 True \
  --disable_flash_attn2 False \
  --gradient_checkpointing True \
  --lora_enable True \
  --lora_rank 8 \
  --lora_alpha 16 \
  --vision_lr 1e-5 \
  --merger_lr 2e-5 \
  --report_to none

EXIT_CODE=${PIPESTATUS[0]}
echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Training finished. Results in $OUTPUT_DIR"
else
  echo "❌ Training failed (exit $EXIT_CODE)"
fi

