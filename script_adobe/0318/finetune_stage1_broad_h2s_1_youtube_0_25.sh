#!/bin/bash
#
# InternVL2.5-1B Stage 1: Broad Mixed SFT (OpenASL + H2S + YouTube)
# Multi-GPU training with WeightedConcatDataset for balanced batch-level mixing


# Essential environment variables
export PYTHONPATH="/mnt/localssd/workspace/doc_sign_search/InternVL/internvl_chat:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True


# Change to InternVL directory
cd /mnt/localssd/workspace/doc_sign_search/InternVL

# GPU configuration
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Calculate number of devices from GPU_IDS
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# Model and data configuration
MODEL_NAME="OpenGVLab/InternVL2_5-1B"
OUTPUT_DIR="/mnt/localssd/checkpoints/sft/tttttt"
META_PATH="/mnt/localssd/workspace/doc_sign_search/script_adobe/0318/train_stage1_meta_broad_h2s_1_youtube_0_25.json"

# Optimized training configuration
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))}

# Memory envelopes
# DEEPSPEED_CONFIG="internvl_chat/zero_stage1_config.json"
DEEPSPEED_CONFIG="internvl_chat/zero_stage1_config.json"

MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16584}
MAX_BUFFER_SIZE=${MAX_BUFFER_SIZE:-20}
NUM_IMAGES_EXPECTED=${NUM_IMAGES_EXPECTED:-160}
MAX_NUM_FRAME=${MAX_NUM_FRAME:-160}

# Video frame sampling method
SAMPLING_METHOD='fps16.0'

echo "🚀 Starting InternVL2.5-1B Stage 1: Broad Mixed SFT"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Set launcher
export LAUNCHER=pytorch
export MASTER_PORT=29501

# Log file
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Log file: $LOG_FILE"
echo ""

# Run training
deepspeed --include localhost:$GPU_IDS --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/internvl_chat_finetune_local.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --meta_path "$META_PATH" \
    --conv_style internvl2_5 \
    --use_fast_tokenizer False \
    --do_train True \
    --num_train_epochs 8 \
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
    --freeze_mlp True \
    --use_llm_lora 16 \
    --bf16 True \
    --max_seq_length $MAX_SEQ_LENGTH \
    --save_strategy epoch \
    --save_total_limit 5 \
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
    --use_data_resampling True \
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
    echo "✅ Stage 1 training completed successfully!"
    echo "📁 Output saved to: $OUTPUT_DIR"
    echo "Next step: Run Stage 2 with checkpoint from this directory"
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE."
    echo "Check log file: $LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi
