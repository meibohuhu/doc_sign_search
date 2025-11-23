#!/bin/bash
#
# InternVL2.5-2B How2Sign Fine-Tuning on 2×A6000 GPUs - LOCAL DEBUG VERSION
# Adapted from 2×A100 cluster script for local A6000 debugging
# Single-node recipe using DeepSpeed ZeRO-2 by default, with optional ZeRO-3
# Memory guidance:
#   - Default (ZeRO-2): 8k seq / 16k packed / 96 frames works on 2×A6000 48 GB.
#   - Enable ZeRO-3 by exporting USE_ZERO_STAGE3=1 to uplift defaults to 12k seq / 20k packed / 128 frames.
#   - You can still override any MAX_* env if you need tighter or looser bounds.

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

# GPU configuration
# Specify which GPUs to use (comma-separated, e.g., "0,1" for GPU 0 and 1)
# Leave empty to use all available GPUs
GPU_IDS=${GPU_IDS:-"0,1"}  # Default: use GPU 0 and 1
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Calculate number of devices from GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "🎯 GPU Configuration:"
echo "   GPU IDs: $GPU_IDS"
echo "   Number of GPUs: $NUM_DEVICES"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Model and data configuration
MODEL_NAME="OpenGVLab/InternVL2_5-2B"
OUTPUT_DIR="/local1/mhu/sign_language_llm/InternVL/output/how2sign/internvl2_5_2B_2xa6000"
# Use local data paths
META_PATH="/local1/mhu/sign_language_llm/InternVL/data/how2sign/train_how2sign_meta.json"
IMAGE_ROOT="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224"

# Optimized training configuration
# Note: NUM_DEVICES is automatically calculated from GPU_IDS above
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}  # Reduced from 8 for A6000
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
# NUM_DEVICES is set above based on GPU_IDS
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))}

# Memory envelopes (defaults can be overridden via env vars)


DEEPSPEED_CONFIG="internvl_chat/zero_stage3_config.json"


DEFAULT_MAX_SEQ_LENGTH=12288
DEFAULT_MAX_PACKED_TOKENS=12288
DEFAULT_MAX_BUFFER_SIZE=20
DEFAULT_NUM_IMAGES_EXPECTED=128
DEFAULT_MAX_NUM_FRAME=128

# DEFAULT_MAX_SEQ_LENGTH=16384
# DEFAULT_MAX_PACKED_TOKENS=16384
# DEFAULT_MAX_BUFFER_SIZE=20
# DEFAULT_NUM_IMAGES_EXPECTED=160
# DEFAULT_MAX_NUM_FRAME=160



MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-$DEFAULT_MAX_SEQ_LENGTH}
MAX_PACKED_TOKENS=${MAX_PACKED_TOKENS:-$DEFAULT_MAX_PACKED_TOKENS}
MAX_BUFFER_SIZE=${MAX_BUFFER_SIZE:-$DEFAULT_MAX_BUFFER_SIZE}
NUM_IMAGES_EXPECTED=${NUM_IMAGES_EXPECTED:-$DEFAULT_NUM_IMAGES_EXPECTED}
MAX_NUM_FRAME=${MAX_NUM_FRAME:-$DEFAULT_MAX_NUM_FRAME}

# Video frame sampling method configuration
# Options:
#   - 'rand' (default): Sample every 2 frames with random start (0 or 1)
#   - 'fpsX.X': FPS-based sampling (e.g., 'fps2.0', 'fps12.0')
#     For 24fps video with fps12.0: samples at 12fps (every 2 frames)
#   - 'random_start_every2': Random start frame, then sample every 2 frames
#     (truncates from end if exceeds max_num_frame)
# Example: export SAMPLING_METHOD='fps12.0'  # for 12fps sampling
#          export SAMPLING_METHOD='random_start_every2'  # for random start + every 2 frames
SAMPLING_METHOD=${SAMPLING_METHOD:-rand}
export SAMPLING_METHOD='fps18.0'

echo "🚀 Starting InternVL2.5-2B How2Sign Training on 2×A6000 (LOCAL DEBUG)"
echo "======================================================"
echo "Model: $MODEL_NAME"
echo "Meta Path: $META_PATH"
echo "Image Root: $IMAGE_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Per-Device Batch Size: $BATCH_PER_DEVICE"
echo "World Size: $NUM_DEVICES"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "Deepspeed Config: $DEEPSPEED_CONFIG"
echo "Max Seq Length: $MAX_SEQ_LENGTH"
echo "Max Packed Tokens: $MAX_PACKED_TOKENS"
echo "Num Images Expected: $NUM_IMAGES_EXPECTED"
echo "Max Num Frame: $MAX_NUM_FRAME"
echo "Sampling Method: $SAMPLING_METHOD"
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

if [ -d "$IMAGE_ROOT" ]; then
    echo "✅ Image folder found: $IMAGE_ROOT"
    IMAGE_COUNT=$(find "$IMAGE_ROOT" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | wc -l)
    echo "   Video files: $IMAGE_COUNT"
else
    echo "❌ Image folder not found: $IMAGE_ROOT"
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
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"
echo ""

echo "🏃 Starting training with DeepSpeed..."
echo "📝 Logging to: $LOG_FILE"
echo ""

# Run training with DeepSpeed launcher (like qwenvl) to avoid no_sync compatibility issues
# with ZeRO Stage 2/3. DeepSpeed launcher handles gradient accumulation correctly.
# Note: CUDA_VISIBLE_DEVICES is already set above, so deepspeed will use the specified GPUs
deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/internvl_chat_finetune_local.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --meta_path "$META_PATH" \
    --conv_style internvl2_5 \
    --do_train True \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 2e-5 \
    --vision_select_layer -1 \
    --force_image_size 224 \
    --dynamic_image_size False \
    --down_sample_ratio 0.5 \
    --pad2square False \
    --freeze_llm True \
    --freeze_backbone True \
    --freeze_mlp False \
    --unfreeze_vit_layers 12 \
    --use_llm_lora 16 \
    --use_backbone_lora 0 \
    --bf16 True \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_steps 8000 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 10 \
    --logging_first_step True \
    --evaluation_strategy no \
    --report_to none \
    --disable_tqdm False \
    --grad_checkpoint True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --group_by_length False \
    --use_packed_ds True \
    --max_packed_tokens $MAX_PACKED_TOKENS \
    --max_buffer_size $MAX_BUFFER_SIZE \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --optim adamw_torch \
    --ddp_find_unused_parameters False \
    --num_images_expected $NUM_IMAGES_EXPECTED \
    --min_num_frame 32 \
    --max_num_frame $MAX_NUM_FRAME \
    --sampling_method ${SAMPLING_METHOD} \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --strict_mode False \
    --loss_reduction square \
    --loss_reduction_all_gather True \
    2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Output saved to: $OUTPUT_DIR"
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE."
    echo "Please check the error messages above for details."
    echo "Check log file: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi

