#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=qwen2vl_how2sign_1xa100
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=256g

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths directly
export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:$PYTHONPATH"
export OMP_NUM_THREADS=8

# Change to Qwen2-VL-Finetune directory
cd /home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune

# Model configuration - using 3B model for faster training on 1xA100
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Optimized training configuration for 1xA100 GPU
GLOBAL_BATCH_SIZE=2
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Run training with optimized settings
torchrun --nproc_per_node=1 --nnodes=1 --master_port=29501 src/train/train_sft.py \
    --model_id $MODEL_NAME \
    --data_path /home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos.json \
    --image_folder /shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips/ \
    --output_dir /home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/output/qwen2vl_how2sign_1xa100 \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((320 * 320)) \
    --video_max_pixels $((320 * 320)) \
    --fps 10 \
    --max_grad_norm 1.0 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --save_steps 500 \
    --save_total_limit 2 \
    --max_steps 1500 \
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
