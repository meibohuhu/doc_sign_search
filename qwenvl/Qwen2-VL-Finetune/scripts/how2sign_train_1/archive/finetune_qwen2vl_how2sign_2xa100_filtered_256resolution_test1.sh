#!/bin/bash -l
# NOTE the -l flag!
#
# Qwen2VL How2Sign Training on 2xA100 GPUs
# Using DeepSpeed ZeRO-3 for efficient multi-GPU training
# Includes network timeout handling, offline fallback, and memory optimization

#SBATCH --job-name=qwen2vl_how2sign_2xa100_short_clips_recipe
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --gpus-per-node=a100:2
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

# Memory optimization for CUDA out of memory issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Change to Qwen2-VL-Finetune directory
cd /home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune

# Model configuration - using 3B model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Optimized training configuration for 2xA100 GPUs (40GB each) - Memory optimized
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
echo "Memory Optimization: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""
echo "📋 TRAINING RECIPE CONFIGURATION:"
echo "• Vision Tower: Frozen with top1 block unfrozen (unfreeze_topk_vision=1)"
echo "• Merger + LoRA: Enabled"
echo "• Learning Rate: 1e-4 (main), 5e-6 (vision), 1e-4 (merger)"
echo "• Scheduler: Cosine with 0.05 warmup ratio"
echo "• Weight Decay: 0.01"
echo "• Label Smoothing: 0.0"
echo "• Max Grad Norm: 1.0"
echo "• Dataset: Short clips under 10s (24,880 clips - cleaned)"
echo ""

# Check if model is already cached
MODEL_CACHE_DIR="/home/mh2803/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "✅ Model found in cache: $MODEL_CACHE_DIR"
    echo "📊 Cache size: $(du -sh $MODEL_CACHE_DIR | cut -f1)"
else
    echo "⚠️  Model not found in cache, will download during training"
fi

# Pre-download model components to avoid timeout issues during training
echo "📥 Pre-downloading model components..."
python -c "
import os
import sys
os.environ['HF_HUB_TIMEOUT'] = '600'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

try:
    from transformers import AutoTokenizer, AutoProcessor
    from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
    
    print('Downloading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
    print('✅ Tokenizer downloaded successfully!')
    
    print('Downloading processor...')
    processor = AutoProcessor.from_pretrained('$MODEL_NAME', trust_remote_code=True)
    print('✅ Processor downloaded successfully!')
    
    print('Downloading model config...')
    model = Qwen2VLForConditionalGeneration.from_pretrained('$MODEL_NAME', trust_remote_code=True, torch_dtype='auto')
    print('✅ Model config downloaded successfully!')
    
    print('🎉 All model components downloaded successfully!')
    
except Exception as e:
    print(f'⚠️  Warning: Some components failed to download: {e}')
    print('This might be due to network issues, but training will continue...')
    print('The model will be downloaded during training if needed.')
"
echo ""

# Test network connectivity
echo "🌐 Testing network connectivity..."
if ping -c 1 huggingface.co > /dev/null 2>&1; then
    echo "✅ Network connectivity to Hugging Face is working"
else
    echo "⚠️  Warning: Cannot reach huggingface.co - training will use cached models"
fi
echo ""

# Run training with DeepSpeed ZeRO-3
echo "🏃 Starting training with memory optimization..."
deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero3_qwen2vl.json \
    --model_id $MODEL_NAME \
    --data_path /home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/short_clips_under_10s.json \
    --image_folder /shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224/ \
    --output_dir /shared/rc/llm-gen-agent/mhu/qwen2.5vl/1018/qwen2vl_how2sign_2xa100_short_clips_recipe/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((224 * 224)) \
    --video_max_pixels $((224 * 224)) \
    --fps 12 \
    --max_grad_norm 1.0 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --label_smoothing 0.0 \
    --logging_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_steps 6000 \
    --use_liger True \
    --freeze_vision_tower False \
    --unfreeze_topk_vision 1 \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --gradient_checkpointing True \
    --lora_enable True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --vision_lr 5e-6 \
    --merger_lr 1e-4 \
    --report_to none

TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE."
    echo "Please check the error logs for more details: /home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_${SLURM_JOB_ID}.txt"
    exit $TRAINING_EXIT_CODE
fi

