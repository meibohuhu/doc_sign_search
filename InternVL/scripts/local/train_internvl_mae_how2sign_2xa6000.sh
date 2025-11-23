#!/bin/bash
#
# InternVL2.5-2B MAE Pre-Training on 2×A6000 GPUs - LOCAL DEBUG VERSION
# Masked Autoencoder (MAE) pre-training for InternVL vision encoder
# Single-node recipe using DeepSpeed ZeRO-3
# Memory guidance:
#   - MAE training is memory-intensive due to decoder
#   - Default: 32 frames, batch_size=4 per device works on 2×A6000 48 GB
#   - Can increase to 64 frames with smaller batch size or gradient checkpointing

# Set up environment paths - adjust these based on your local conda environment
# Common locations to check:
# - /home/ztao/anaconda3/envs/internvl
# - /local1/mhu/miniconda3/envs/internvl
# - ~/miniconda3/envs/internvl
CONDA_ENV_NAME="internvl"  # Change this to your conda environment name
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

export PYTHONPATH="/local1/mhu/sign_language_llm/InternVL:/local1/mhu/sign_language_llm/InternVL/internvl_chat:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
# Use new PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF is deprecated)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# DeepSpeed build flags (disable custom ops for compatibility)
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CUDA_EXT=0
export DS_BUILD_CPU_ADAM=0
export DEEPSPEED_CPU_ADAM=1

# Network and Hugging Face configuration
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_TIMEOUT=600
export HF_HUB_DOWNLOAD_TIMEOUT=600
# Fast transfer requires hf_transfer package; disable unless installed
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1

# Change to InternVL directory
cd /local1/mhu/sign_language_llm/InternVL

# Model and data configuration
MODEL_NAME="OpenGVLab/InternVL2_5-2B"
OUTPUT_DIR="/local1/mhu/sign_language_llm/InternVL/output/how2sign/internvl2_5_2B_mae_2xa6000"
# Use local data paths - MAE only needs video paths, no text/conversations
META_PATH="/local1/mhu/sign_language_llm/InternVL/data/how2sign/train_how2sign_meta.json"
VIDEO_BASE_PATH="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224"

# Optimized training configuration for 2xA6000 GPUs (48GB each)
# MAE training is memory-intensive, use smaller batch size
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}  # Smaller batch for MAE
NUM_DEVICES=${NUM_DEVICES:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))}

# DeepSpeed config
DEEPSPEED_CONFIG="internvl_chat/zero_stage3_config.json"

# MAE-specific memory configuration
DEFAULT_IMAGE_SIZE=224
DEFAULT_MIN_NUM_FRAMES=8
DEFAULT_MAX_NUM_FRAMES=64

IMAGE_SIZE=${IMAGE_SIZE:-$DEFAULT_IMAGE_SIZE}
MIN_NUM_FRAMES=${MIN_NUM_FRAMES:-$DEFAULT_MIN_NUM_FRAMES}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-$DEFAULT_MAX_NUM_FRAMES}

# Video frame sampling method configuration
# Options:
#   - 'rand' (default): Sample every 2 frames with random start (0 or 1)
#   - 'fpsX.X': FPS-based sampling (e.g., 'fps2.0', 'fps12.0')
#     For 24fps video with fps12.0: samples at 12fps (every 2 frames)
#   - 'random_start_every2': Random start frame, then sample every 2 frames
#     (truncates from end if exceeds max_num_frames)
# Example: export SAMPLING_METHOD='fps12.0'  # for 12fps sampling
#          export SAMPLING_METHOD='random_start_every2'  # for random start + every 2 frames
SAMPLING_METHOD=${SAMPLING_METHOD:-rand}
export SAMPLING_METHOD='random_start_every2'

# MAE-specific parameters
MASK_RATIO=${MASK_RATIO:-0.80}  # 0.75 or 0.85 for video MAE
MASK_STRATEGY=${MASK_STRATEGY:-random}  # 'random', 'tube', 'block', 'mu'
DECODER_DIM=${DECODER_DIM:-384}
DECODER_DEPTH=${DECODER_DEPTH:-6}
DECODER_HEADS=${DECODER_HEADS:-12}
# For freeze_encoder: use 'True' or 'False' (as strings) since argparse type=bool
FREEZE_ENCODER=${FREEZE_ENCODER:-False}  # Set to 'True' to freeze encoder
UNFREEZE_TOPK_VISION=${UNFREEZE_TOPK_VISION:-0}  # 0 = train all layers

echo "🚀 Starting InternVL2.5-2B MAE Pre-Training on 2×A6000 (LOCAL DEBUG)"
echo "======================================================"
echo "Model: $MODEL_NAME"
echo "Meta Path: $META_PATH"
echo "Video Base Path: $VIDEO_BASE_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Per-Device Batch Size: $BATCH_PER_DEVICE"
echo "World Size: $NUM_DEVICES"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "Deepspeed Config: $DEEPSPEED_CONFIG"
echo "Image Size: $IMAGE_SIZE"
echo "Min Num Frames: $MIN_NUM_FRAMES"
echo "Max Num Frames: $MAX_NUM_FRAMES"
echo "Sampling Method: $SAMPLING_METHOD"
echo "Mask Ratio: $MASK_RATIO"
echo "Mask Strategy: $MASK_STRATEGY"
echo "Decoder Dim: $DECODER_DIM"
echo "Decoder Depth: $DECODER_DEPTH"
echo "Decoder Heads: $DECODER_HEADS"
echo "Freeze Encoder: $FREEZE_ENCODER"
echo "Unfreeze TopK Vision: $UNFREEZE_TOPK_VISION"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Verify data paths exist
echo "📂 Verifying data paths..."
if [ -f "$META_PATH" ]; then
    echo "✅ Meta file found: $META_PATH"
    echo "   File size: $(du -sh "$META_PATH" | cut -f1)"
else
    echo "❌ Meta file not found: $META_PATH"
    exit 1
fi

if [ -d "$VIDEO_BASE_PATH" ]; then
    echo "✅ Video folder found: $VIDEO_BASE_PATH"
    VIDEO_COUNT=$(find "$VIDEO_BASE_PATH" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | wc -l)
    echo "   Video files: $VIDEO_COUNT"
else
    echo "❌ Video folder not found: $VIDEO_BASE_PATH"
    exit 1
fi
echo ""

# Check if model is already cached
MODEL_CACHE_DIR="$HOME/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-2B"
if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "✅ Model found in cache: $MODEL_CACHE_DIR"
    echo "📊 Cache size: $(du -sh "$MODEL_CACHE_DIR" 2>/dev/null | cut -f1 || echo 'N/A')"
else
    echo "⚠️  Model not found in cache, will download during training"
fi

# Test network connectivity
echo "🌐 Testing network connectivity..."
if ping -c 1 huggingface.co > /dev/null 2>&1; then
    echo "✅ Network connectivity to Hugging Face is working"
else
    echo "⚠️  Warning: Cannot reach huggingface.co - training will use cached models"
fi
echo ""

# Check and install required dependencies
echo "📦 Checking required dependencies..."
python -c "
import sys
import subprocess

missing_packages = []
# Map of (module_name, package_name) for import checking
required_packages = [
    ('transformers', 'transformers'),
    ('accelerate', 'accelerate'),
    ('deepspeed', 'deepspeed'),
    ('packaging', 'packaging'),
]

for module_name, package_name in required_packages:
    try:
        __import__(module_name)
        print(f'✅ {package_name} is installed')
    except (ImportError, ModuleNotFoundError):
        print(f'❌ {package_name} is missing')
        missing_packages.append(package_name)

if missing_packages:
    print(f'\n⚠️  Missing packages: {missing_packages}')
    print('Please install them manually:')
    print(f'  pip install {\" \".join(missing_packages)}')
else:
    print('✅ All required packages are installed!')
"
echo ""

# Set launcher to pytorch for local training (not SLURM cluster)
export LAUNCHER=pytorch

# Generate MASTER_PORT
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
export MASTER_PORT
echo "MASTER_PORT: $MASTER_PORT"
echo "LAUNCHER: $LAUNCHER (for local training, not SLURM)"
echo ""

# Log file with timestamp
LOG_FILE="${OUTPUT_DIR}/mae_training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"
echo ""

echo "🏃 Starting MAE pre-training with DeepSpeed..."
echo "📝 Logging to: $LOG_FILE"
echo ""

# Run MAE training with DeepSpeed launcher
deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/train_internvl_mae.py \
    --model_id "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --data_path "$META_PATH" \
    --video_base_path "$VIDEO_BASE_PATH" \
    --batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_epochs 100 \
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
    --freeze_encoder $FREEZE_ENCODER \
    --unfreeze_topk_vision $UNFREEZE_TOPK_VISION \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_interval 1000 \
    --log_interval 10 \
    --num_workers 4 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --local_rank -1 \
    2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ MAE pre-training completed successfully!"
    echo "📁 Output saved to: $OUTPUT_DIR"
    echo "📁 Final encoder saved to: $OUTPUT_DIR/mae_encoder_final.pth"
else
    echo "❌ MAE pre-training failed with exit code $TRAINING_EXIT_CODE."
    echo "Please check the error messages above for details."
    echo "Check log file: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi

