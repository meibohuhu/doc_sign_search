#!/bin/bash -l
# NOTE the -l flag!
#
# Qwen2VL MAE Pre-training on 1xA100 GPU
# Masked Autoencoder for vision encoder pre-training

#SBATCH --job-name=qwen2vl_mae_1xa100_random
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=64g

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
export OMP_NUM_THREADS=16
# Memory optimization: use expandable segments and set max split size to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false  # Disable tokenizers parallelism warnings in multi-process environment
# Enable memory efficient attention if available
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Network and Hugging Face configuration
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_TIMEOUT=600
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ENABLE_HF_TRANSFER=1

# Disable wandb (Weights & Biases) to avoid API key issues in non-interactive environments
export WANDB_DISABLED=1

# Change to Qwen2-VL-Finetune directory
cd /home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Data configuration
# DATA_PATH should point to a JSON file containing video paths
# For MAE, only video paths are needed (no text/conversations required)
DATA_PATH="/home/mh2803/projects/sign_language_llm/how2sign/video/segmented_train_videos_corrupted_removed.json"

# VIDEO_BASE_PATH: Base path for video files (used when JSON contains relative paths)
# If JSON contains absolute paths, this can be omitted or set to empty
VIDEO_BASE_PATH="/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224/"

# Training configuration
PER_DEVICE_BS=1  # Batch size per device (reduced for memory)
GRAD_ACCUM=4     # Gradient accumulation steps (increased to maintain effective batch size and reduce memory)
FPS=6           # Frames per second (increased from 4, can try 8 or 12 with optimizations below)
MASK_RATIO=0.80 # Mask ratio (0.75 or 0.90)

# MAE model configuration
DECODER_DIM=384  # Reduced from 512 to save memory
DECODER_DEPTH=6  # Reduced from 8 to save memory
DECODER_HEADS=12 # Reduced from 16 to save memory
MLP_RATIO=4.0

# Vision encoder training configuration
# Set UNFREEZE_TOPK_VISION to train only top k layers (0 = train all layers)
# Example: UNFREEZE_TOPK_VISION=4 will train only the last 4 layers of vision encoder
UNFREEZE_TOPK_VISION=4  # 0 = train all layers, >0 = train only top k layers

# Training hyperparameters
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.05
NUM_EPOCHS=100
MAX_GRAD_NORM=1.0


# Output configuration
OUTPUT_DIR="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/1119/qwen2vl_mae_1xa100_random"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Generate log file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/mae_training_${TIMESTAMP}.log"

echo "=========================================="
echo "Qwen2VL MAE Pre-training Configuration"
echo "=========================================="
echo "Model           : $MODEL_NAME"
echo "Data Path       : $DATA_PATH"
echo "Video Base Path : ${VIDEO_BASE_PATH:-'(not set, using JSON file directory)'}"
echo "Output Dir      : $OUTPUT_DIR"
echo "Mask Ratio      : $MASK_RATIO"
echo "Mask Strategy   : mu (mask-unit like Hiera)"
echo "Mask Unit Size  : 4x4 (for block/mu strategies)"
echo "FPS             : $FPS"
echo "Batch Size      : $PER_DEVICE_BS per device"
echo "Grad Accum      : $GRAD_ACCUM"
echo "Learning Rate   : $LEARNING_RATE"
echo "Epochs          : $NUM_EPOCHS"
echo "Decoder Dim     : $DECODER_DIM"
echo "Decoder Depth   : $DECODER_DEPTH"
echo "Decoder Heads   : $DECODER_HEADS"
if [ "$UNFREEZE_TOPK_VISION" -gt 0 ]; then
    echo "Vision Encoder  : Top $UNFREEZE_TOPK_VISION layers trainable"
else
    echo "Vision Encoder  : All layers trainable"
fi
echo "DeepSpeed       : ZeRO-3"
echo "Log File        : $LOG_FILE"
echo "=========================================="

# Run training with DeepSpeed ZeRO-3 for efficient GPU training
echo "🏃 Starting MAE training with DeepSpeed..."
deepspeed src/train/train_qwen_mae.py \
    --deepspeed scripts/zero3_qwen2vl.json \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --video_base_path "$VIDEO_BASE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --mask_ratio $MASK_RATIO \
    --mask_strategy random \
    --mask_unit_size 4 4 \
    --decoder_dim $DECODER_DIM \
    --decoder_depth $DECODER_DEPTH \
    --decoder_heads $DECODER_HEADS \
    --mlp_ratio $MLP_RATIO \
    --norm_pix_loss True \
    --spacetime_mask True \
    --unfreeze_topk_vision $UNFREEZE_TOPK_VISION \
    --batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --video_resized_width 224 \
    --video_resized_height 224 \
    --video_min_pixels $((224 * 224)) \
    --video_max_pixels $((224 * 224)) \
    --fps $FPS \
    --num_workers 2 \
    --log_interval 10 \
    --save_strategy steps \
    --save_interval 1000 \
    --save_total_limit 2 \
    --bf16 \
    --gradient_checkpointing \
    --dataloader_pin_memory False

echo "Training completed! Check logs at: $LOG_FILE"

# ============================================================================
# Memory Optimization Tips for Higher FPS:
# ============================================================================
# If you still get OOM with higher FPS, try these additional optimizations:
#
# 1. Further reduce decoder size:
#    DECODER_DIM=192  (from 256)
#    DECODER_DEPTH=3  (from 4)
#    DECODER_HEADS=6  (from 8)
#
# 2. Reduce trainable vision layers:
#    UNFREEZE_TOPK_VISION=1  (from 2, only train last layer)
#
# 3. Increase gradient accumulation:
#    GRAD_ACCUM=8  (from 4, reduces memory per step)
#
# 4. Reduce video resolution (if acceptable):
#    --video_resized_width 192  (from 224)
#    --video_resized_height 192
#    --video_min_pixels $((192 * 192))
#    --video_max_pixels $((192 * 192))
#
# 5. Use lower mask ratio (processes fewer patches):
#    MASK_RATIO=0.90  (masks more, processes less)
#
# 6. Enable optimizer CPU offload in DeepSpeed config:
#    Set "offload_optimizer": {"device": "cpu"} in zero3_qwen2vl.json
#
# 7. Reduce num_workers to save CPU memory:
#    --num_workers 1  (from 2)
# ============================================================================

