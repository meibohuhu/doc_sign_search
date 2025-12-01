#!/bin/bash
#
# InternVL2.5-2B OpenASL Fine-Tuning on 4*A100
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

# Change to InternVL directory
cd /code/doc_sign_search/InternVL

# GPU configuration
# Specify which GPUs to use (comma-separated, e.g., "0,1,2,3" for GPU 0, 1, 2, and 3)
GPU_IDS=${GPU_IDS:-"4,5,6,7"}  # Default: use GPU 0, 1, 2, 3
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Calculate number of devices from GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Model and data configuration
MODEL_NAME="OpenGVLab/InternVL2_5-2B"
OUTPUT_DIR="/code/doc_sign_search/script_adobe/checkpoints/finetune_internvl2_5_openasl_20fps_1130"
META_PATH="/code/doc_sign_search/script_adobe/train_openasl_meta.json"
IMAGE_ROOT="/mnt/localssd/sign_mllm_openasl_videos"

# Optimized training configuration
# Note: NUM_DEVICES is automatically calculated from GPU_IDS above
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))}

# Memory envelopes (defaults can be overridden via env vars)
DEEPSPEED_CONFIG="internvl_chat/zero_stage1_config.json"

# DEFAULT_MAX_SEQ_LENGTH=12288
# DEFAULT_MAX_BUFFER_SIZE=20
# DEFAULT_NUM_IMAGES_EXPECTED=128
# DEFAULT_MAX_NUM_FRAME=128

MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16584}
MAX_BUFFER_SIZE=${MAX_BUFFER_SIZE:-20}
NUM_IMAGES_EXPECTED=${NUM_IMAGES_EXPECTED:-160}
MAX_NUM_FRAME=${MAX_NUM_FRAME:-160}

# Video frame sampling method
SAMPLING_METHOD='fps20.0'

echo "🚀 Starting InternVL2.5-2B OpenASL Training on 4*A100"
echo "======================================================"
echo "Model: $MODEL_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "GPU IDs: $GPU_IDS (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
echo "Number of GPUs: $NUM_DEVICES"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Per-Device Batch Size: $BATCH_PER_DEVICE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "Max Seq Length: $MAX_SEQ_LENGTH"
echo "Max Num Frame: $MAX_NUM_FRAME"
echo "Sampling Method: $SAMPLING_METHOD"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Check if model is already cached
MODEL_CACHE_DIR="$HOME/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-2B"
if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "✅ Model found in cache: $MODEL_CACHE_DIR"
    echo "📊 Cache size: $(du -sh "$MODEL_CACHE_DIR" 2>/dev/null | cut -f1 || echo 'N/A')"
else
    echo "⚠️  Model not found in cache, will download during training"
fi

# Set launcher to pytorch for local training (not SLURM cluster)
export LAUNCHER=pytorch

# Generate MASTER_PORT
# MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
# export MASTER_PORT
export MASTER_PORT=29508
echo "MASTER_PORT: $MASTER_PORT"
echo "LAUNCHER: $LAUNCHER (for local training, not SLURM)"
echo ""

# Log file with timestamp
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"
echo ""

# Run training with DeepSpeed launcher (like qwenvl) to avoid no_sync compatibility issues
# with ZeRO Stage 2/3. DeepSpeed launcher handles gradient accumulation correctly.
# Note: CUDA_VISIBLE_DEVICES is already set above, so deepspeed will use the specified GPUs
# deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT \
#     internvl_chat/internvl/train/internvl_chat_finetune_local.py \
deepspeed --include localhost:$GPU_IDS --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/internvl_chat_finetune_local.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --meta_path "$META_PATH" \
    --conv_style internvl2_5 \
    --use_fast_tokenizer False \
    --do_train True \
    --num_train_epochs 6 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 6e-5 \
    --vision_select_layer -1 \
    --force_image_size 224 \
    --max_dynamic_patch 6 \
    --dynamic_image_size True \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.0 \
    --freeze_llm True \
    --freeze_backbone False \
    --freeze_mlp False \
    --unfreeze_vit_layers 0 \
    --use_llm_lora 16 \
    --bf16 True \
    --max_seq_length $MAX_SEQ_LENGTH \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 10 \
    --logging_first_step True \
    --evaluation_strategy no \
    --report_to none \
    --grad_checkpoint True \
    --dataloader_num_workers 4 \
    --use_thumbnail True \
    --ps_version v2 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --group_by_length True \
    --use_packed_ds False \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --min_num_frame 32 \
    --max_num_frame $MAX_NUM_FRAME \
    --sampling_method "$SAMPLING_METHOD" \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=$?

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

