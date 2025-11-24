#!/bin/bash
#
# Qwen2VL MAE Pre-training on 2xA6000 GPU
# Masked Autoencoder for vision encoder pre-training
## RUN_BACKGROUND=1 bash finetune_qwen2vl_mae_2xa6000.sh

# Set up environment paths
CONDA_ENV_NAME="qwen25_vl_sign"
CONDA_BASE_PATH="${CONDA_BASE:-$HOME/anaconda3}"

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
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false  # Disable tokenizers parallelism warnings in multi-process environment

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

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Data configuration
# DATA_PATH should point to a JSON file containing video paths
# For MAE, only video paths are needed (no text/conversations required)
# DATA_PATH="/local1/mhu/sign_language_llm/how2sign/video/debug_two_samples_path.json"  # Change to your MAE training data
DATA_PATH="/local1/mhu/sign_language_llm/how2sign/video/segmented_train_videos_corrupted_removed.json"  # Change to your MAE training data

# VIDEO_BASE_PATH: Base path for video files (used when JSON contains relative paths)
# If JSON contains absolute paths, this can be omitted or set to empty
VIDEO_BASE_PATH="/local1/mhu/sign_language_llm/how2sign/video/train_crop_videos_224"

# Training configuration
PER_DEVICE_BS=1  # Batch size per device (reduced for memory)
GRAD_ACCUM=4     # Gradient accumulation steps (increased to maintain effective batch size)
FPS=4           # Frames per second
MASK_RATIO=0.80  # Mask ratio (0.75 or 0.90)

# Output configuration
OUTPUT_DIR="/local1/mhu/sign_language_llm/qwenvl/outputs/qwen2vl_mae_2xa6000"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Generate log file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/mae_training_${TIMESTAMP}.log"

# Check if running in background mode (set RUN_BACKGROUND=1 to enable)
RUN_BACKGROUND="${RUN_BACKGROUND:-0}"

# Run training with DeepSpeed ZeRO-3 for efficient multi-GPU training
echo "🏃 Starting MAE training with DeepSpeed..."
if [ "$RUN_BACKGROUND" -eq 1 ]; then
    echo "📝 Running in background mode (nohup)"
    echo "📝 Log file: $LOG_FILE"
    echo "📝 Process will continue even if terminal is closed"
    echo "📝 Use 'tail -f $LOG_FILE' to monitor progress"
    echo ""
    
    # Run with nohup in background
    nohup deepspeed src/train/train_qwen_mae.py \
        --deepspeed scripts/zero3_qwen2vl.json \
        --model_id "$MODEL_NAME" \
        --data_path "$DATA_PATH" \
        --video_base_path "$VIDEO_BASE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --mask_ratio $MASK_RATIO \
        --mask_strategy random \
        --mask_unit_size 2 2 \
        --decoder_dim 384 \
        --decoder_depth 6 \
        --decoder_heads 12 \
        --mlp_ratio 4.0 \
        --norm_pix_loss True \
        --spacetime_mask True \
        --unfreeze_topk_vision 8 \
        --batch_size $PER_DEVICE_BS \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs 5 \
        --learning_rate 1e-4 \
        --weight_decay 0.05 \
        --max_grad_norm 1.0 \
        --video_resized_width 224 \
        --video_resized_height 224 \
        --video_min_pixels $((224 * 224)) \
        --video_max_pixels $((224 * 224)) \
        --fps $FPS \
        --num_workers 4 \
        --log_interval 10 \
        --save_strategy steps \
        --save_interval 10000 \
        --save_total_limit 2 \
        --bf16 \
        --gradient_checkpointing \
        >> "$LOG_FILE" 2>&1 &
    
    TRAIN_PID=$!
    echo "✅ Training started in background (PID: $TRAIN_PID)"
    echo "📝 Monitor with: tail -f $LOG_FILE"
    echo "📝 Check process with: ps aux | grep $TRAIN_PID"
    echo "📝 Stop training with: kill $TRAIN_PID"
else
    # Run in foreground (default behavior)
    deepspeed src/train/train_qwen_mae.py \
        --deepspeed scripts/zero3_qwen2vl.json \
        --model_id "$MODEL_NAME" \
        --data_path "$DATA_PATH" \
        --video_base_path "$VIDEO_BASE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --mask_ratio $MASK_RATIO \
        --mask_strategy mu \
        --mask_unit_size 2 2 \
        --decoder_dim 384 \
        --decoder_depth 6 \
        --decoder_heads 12 \
        --mlp_ratio 4.0 \
        --norm_pix_loss True \
        --spacetime_mask True \
        --unfreeze_topk_vision 8 \
        --batch_size $PER_DEVICE_BS \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs 5 \
        --learning_rate 1e-4 \
        --weight_decay 0.05 \
        --max_grad_norm 1.0 \
        --video_resized_width 224 \
        --video_resized_height 224 \
        --video_min_pixels $((224 * 224)) \
        --video_max_pixels $((224 * 224)) \
        --fps $FPS \
        --num_workers 4 \
        --log_interval 10 \
        --save_strategy steps \
        --save_interval 1000 \
        --save_total_limit 2 \
        --bf16 \
        --gradient_checkpointing \
        2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training completed! Check logs at: $LOG_FILE"
    else
        echo "❌ Training failed (exit $EXIT_CODE). Check logs at: $LOG_FILE"
    fi
fi

