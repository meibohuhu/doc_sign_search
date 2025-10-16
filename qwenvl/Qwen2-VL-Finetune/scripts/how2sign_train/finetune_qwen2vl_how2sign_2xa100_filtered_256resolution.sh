#!/bin/bash -l
# NOTE the -l flag!
#
# Qwen2VL How2Sign Training on 2xA100 GPUs
# Using DeepSpeed ZeRO-3 for efficient multi-GPU training

#SBATCH --job-name=qwen2vl_how2sign_2xa100_filtered_256resolution
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --gpus-per-node=a100:2
#SBATCH --partition tier3
#SBATCH --mem=256G

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths directly
export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:$PYTHONPATH"
export OMP_NUM_THREADS=16

# Memory optimization for longer videos at 12fps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to Qwen2-VL-Finetune directory
cd /home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune

# Model configuration - using 3B model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Optimized training configuration for 2xA100 GPUs (40GB each)
GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "🚀 Starting Qwen2VL How2Sign Training on 2xA100"
echo "================================================"
echo "Model: $MODEL_NAME"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Per-Device Batch Size: $BATCH_PER_DEVICE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo "Number of GPUs: $NUM_DEVICES"
echo ""

# Run training with DeepSpeed ZeRO-3
deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero3_qwen2vl.json \
    --model_id $MODEL_NAME \
    --data_path /home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_filtered.json \
    --image_folder /shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips/ \
    --output_dir /shared/rc/llm-gen-agent/mhu/qwen2.5vl/qwen2vl_how2sign_2xa100_filtered_256resolution/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((256 * 256)) \
    --video_max_pixels $((256 * 256)) \
    --fps 12 \
    --max_grad_norm 1.0 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_steps 6000 \
    --use_liger True \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --gradient_checkpointing True \
    --lora_enable True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --vision_lr 2e-6 \
    --merger_lr 1e-5 \
    --report_to none

echo ""
echo "✅ Training completed!"

