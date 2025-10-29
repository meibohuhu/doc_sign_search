#!/bin/bash -l
# NOTE the -l flag!
#
# Qwen2VL How2Sign Training Stage 2: Vision-Text Alignment
# Freeze vision encoder, train projector (merger) and LoRA only
# This script continues training from a previous checkpoint

#SBATCH --job-name=qwen2vl_how2sign_4xa100_stage2_alignment
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=52:00:00
#SBATCH --gpus-per-node=a100:4
#SBATCH --partition tier3
#SBATCH --mem=256G

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths directly
export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:$PYTHONPATH"
export OMP_NUM_THREADS=16

# Network and Hugging Face configuration to handle timeouts
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_TIMEOUT=600
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ENABLE_HF_TRANSFER=1

# Change to Qwen2-VL-Finetune directory
cd /home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune

# Model configuration - using 3B model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# ============================================
# STAGE 2 CONFIGURATION: Alignment Training
# ============================================
# Freeze: Vision Encoder
# Train: Projector (merger) + LoRA
# ============================================

# Optimized training configuration for 4xA100 GPUs (40GB each)
GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=1
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Checkpoint to continue from (modify this to your checkpoint path)
# Example: checkpoint-5000 or checkpoint-6000
STAGE1_CHECKPOINT="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/1018/qwen2vl_how2sign_4xa100_filtered_32batchsize_robust/checkpoint-6000"

echo "🚀 Starting Qwen2VL How2Sign Training - STAGE 2: Alignment"
echo "================================================================"
echo "Model: $MODEL_NAME"
echo "Continuing from: $STAGE1_CHECKPOINT"
echo "Stage 2 Strategy:"
echo "  ✅ Freeze Vision Encoder (keep learned features)"
echo "  ✅ Train Projector/Merger (vision-text bridge)"
echo "  ✅ Train LoRA (text generation)"
echo ""
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Per-Device Batch Size: $BATCH_PER_DEVICE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "Number of GPUs: $NUM_DEVICES"
echo ""

# Check if checkpoint exists
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found at $STAGE1_CHECKPOINT"
    echo "Please modify STAGE1_CHECKPOINT variable in this script"
    exit 1
fi

echo "✅ Found checkpoint: $STAGE1_CHECKPOINT"

# Run training with DeepSpeed ZeRO-3
echo "🏃 Starting Stage 2 training (Alignment)..."
deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero3_qwen2vl.json \
    --model_id $MODEL_NAME \
    --data_path /home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_corrupted_removed.json \
    --image_folder /shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224/ \
    --output_dir /shared/rc/llm-gen-agent/mhu/qwen2.5vl/1018/qwen2vl_how2sign_4xa100_stage2_alignment/ \
    --resume_from_checkpoint $STAGE1_CHECKPOINT \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((224 * 224)) \
    --video_max_pixels $((224 * 224)) \
    --fps 18 \
    --max_grad_norm 1.0 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 200 \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --max_steps 4000 \
    --use_liger True \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --gradient_checkpointing True \
    --lora_enable True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --merger_lr 3e-5 \
    --report_to none

TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Stage 2 training completed successfully!"
    echo ""
    echo "📊 Next steps:"
    echo "  1. Evaluate on training subset to check BLEU improvement"
    echo "  2. Evaluate on inference set"
    echo "  3. Expected BLEU improvement: 6% → 15-25%+"
else
    echo "❌ Stage 2 training failed with exit code $TRAINING_EXIT_CODE."
    echo "Please check the error logs for more details: /home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_${SLURM_JOB_ID}.txt"
    exit $TRAINING_EXIT_CODE
fi

