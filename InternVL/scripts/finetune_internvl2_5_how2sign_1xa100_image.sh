#!/bin/bash -l
# NOTE the -l flag!
#
# InternVL2.5-4B How2Sign Fine-Tuning on 1×A100 GPU
# Single-node recipe derived from the 1×A100 ZeRO-2 setup

#SBATCH --job-name=internvl25_how2sign_1xa100
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:08:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=64G

set -euo pipefail

# Load toolchains only if not already present (avoids errors with set -e)
if ! spack find --loaded /lhqcen5 >/dev/null 2>&1; then
    spack load /lhqcen5
fi
if ! spack find --loaded cuda@12.4.0/obxqih4 >/dev/null 2>&1; then
    spack load cuda@12.4.0/obxqih4
fi

# Activate fine-tuning environment
source /home/mh2803/miniconda3/bin/activate /home/mh2803/miniconda3/envs/internvl
export PATH="/home/mh2803/miniconda3/envs/internvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/InternVL:/home/mh2803/projects/sign_language_llm/InternVL/internvl_chat:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
cd /home/mh2803/projects/sign_language_llm/InternVL

# Model and data configuration
MODEL_NAME="OpenGVLab/InternVL2_5-4B"
OUTPUT_DIR="/home/mh2803/projects/sign_language_llm/InternVL/output/how2sign/internvl2_5_4b_1xa100"
META_PATH="/home/mh2803/projects/sign_language_llm/InternVL/data/how2sign/train_how2sign_under10s_meta.json"
IMAGE_ROOT="/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224/"

GLOBAL_BATCH_SIZE=4
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "🚀 Starting InternVL2.5-4B How2Sign Training on 1×A100"
echo "======================================================"
echo "Model: $MODEL_NAME"
echo "Meta Path: $META_PATH"
echo "Image Root: $IMAGE_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Per-Device Batch Size: $BATCH_PER_DEVICE"
echo "Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

MODEL_CACHE_DIR="/home/mh2803/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-4B"
if [ -d "$MODEL_CACHE_DIR" ]; then
    echo "✅ Model found in cache: $MODEL_CACHE_DIR"
    echo "📊 Cache size: $(du -sh "$MODEL_CACHE_DIR" | cut -f1)"
else
    echo "⚠️  Model not found in cache, will download during training"
fi

echo "📥 Pre-downloading model components..."
python - <<'PYTHON'
import os
os.environ["HF_HUB_TIMEOUT"] = "600"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
MODEL_NAME = "OpenGVLab/InternVL2_5-4B"
try:
    from transformers import AutoTokenizer, AutoModel
    print("Downloading tokenizer...")
    _ = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    print("✅ Tokenizer ready!")
    print("Downloading model weights (may take a while)...")
    _ = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✅ Model weights cached!")
except Exception as exc:
    print(f"⚠️  Warning: Some components failed to download: {exc}")
    print("They will be downloaded lazily during training if needed.")
PYTHON
echo ""

echo "🌐 Testing network connectivity..."
if ping -c 1 huggingface.co > /dev/null 2>&1; then
    echo "✅ Network connectivity to Hugging Face is working"
else
    echo "⚠️  Warning: Cannot reach huggingface.co - training will rely on cached artifacts"
fi
echo ""

# Launch training (single GPU -> plain python run)
echo "🏃 Starting training..."
python internvl_chat/internvl/train/internvl_chat_finetune.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --meta_path "$META_PATH" \
    --conv_style internvl2_5 \
    --do_train True \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 1e-5 \
    --vision_select_layer -1 \
    --force_image_size 224 \
    --dynamic_image_size False \
    --down_sample_ratio 0.5 \
    --pad2square False \
    --freeze_llm True \
    --freeze_backbone False \
    --freeze_mlp False \
    --unfreeze_vit_layers 2 \
    --use_llm_lora 16 \
    --use_backbone_lora 0 \
    --bf16 True \
    --max_seq_length 8192 \
    --max_steps 10 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 1 \
    --logging_first_step True \
    --evaluation_strategy no \
    --report_to none \
    --disable_tqdm True \
    --grad_checkpoint True \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --group_by_length False \
    --use_packed_ds True \
    --max_packed_tokens 16384 \
    --max_buffer_size 20 \
    --num_images_expected 96 \
    --min_num_frame 32 \
    --max_num_frame 96 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --strict_mode False \
    --log_freq 500 \
    --allow_overflow False \
    --loss_reduction square \
    --loss_reduction_all_gather True

TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE."
    echo "Please check the error logs for more details: /home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_${SLURM_JOB_ID}.txt"
    exit $TRAINING_EXIT_CODE
fi

