#!/bin/bash -l
# NOTE the -l flag!
#
# InternVL2.5-2B How2Sign Fine-Tuning on 2×A100 GPUs
# Single-node recipe using DeepSpeed ZeRO-2 by default, with optional ZeRO-3 + CPU offload
# Memory guidance:
#   - Default (ZeRO-2): 8k seq / 16k packed / 96 frames works on 2×A100 40 GB.
#   - Enable ZeRO-3 by exporting USE_ZERO_STAGE3=1 to uplift defaults to 12k seq / 20k packed / 128 frames.
#   - You can still override any MAX_* env if you need tighter or looser bounds.

#SBATCH --job-name=internvl25_how2sign_2xa100
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=a100:2
#SBATCH --partition tier3
#SBATCH --mem=128G

set -euo pipefail

# Load toolchains only if not already present (avoids errors with set -e)
if ! spack find --loaded /lhqcen5 >/dev/null 2>&1; then
    spack load /lhqcen5
fi
if ! spack find --loaded cuda@12.4.0/obxqih4 >/dev/null 2>&1; then
    spack load cuda@12.4.0/obxqih4
fi

# Explicit CUDA environment setup (required before training)
CUDA_ROOT="$(spack location -i /obxqih4)"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Activate fine-tuning environment
source /home/mh2803/miniconda3/bin/activate /home/mh2803/miniconda3/envs/internvl
export PATH="/home/mh2803/miniconda3/envs/internvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/InternVL:/home/mh2803/projects/sign_language_llm/InternVL/internvl_chat:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CUDA_EXT=0
export DS_BUILD_CPU_ADAM=0
export DEEPSPEED_CPU_ADAM=1
if [ -z "${CUDA_HOME:-}" ]; then
    CUDA_HOME="$(spack location -i /obxqih4)"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export DS_BUILD_OPS=0

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
MODEL_NAME="OpenGVLab/InternVL2_5-2B"
OUTPUT_DIR="/home/mh2803/projects/sign_language_llm/InternVL/output/how2sign/internvl2_5_2B_2xa100"
META_PATH="/home/mh2803/projects/sign_language_llm/InternVL/data/how2sign/train_how2sign_under10s_meta.json"
IMAGE_ROOT="/shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_224x224/"

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-12}
BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-1}
NUM_DEVICES=${NUM_DEVICES:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))}

# Memory envelopes (defaults can be overridden via env vars)
USE_ZERO_STAGE3=${USE_ZERO_STAGE3:-0}
DEFAULT_MAX_SEQ_LENGTH=8192
DEFAULT_MAX_PACKED_TOKENS=16384
DEFAULT_MAX_BUFFER_SIZE=20
DEFAULT_NUM_IMAGES_EXPECTED=96
DEFAULT_MAX_NUM_FRAME=96

DEEPSPEED_CONFIG="internvl_chat/zero_stage2_config.json"
if [ "$USE_ZERO_STAGE3" -eq 1 ]; then
    DEEPSPEED_CONFIG="internvl_chat/zero_stage3_config.json"
    DEFAULT_MAX_SEQ_LENGTH=8192
    DEFAULT_MAX_PACKED_TOKENS=16384
    DEFAULT_MAX_BUFFER_SIZE=20
    DEFAULT_NUM_IMAGES_EXPECTED=96
    DEFAULT_MAX_NUM_FRAME=96
fi

MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-$DEFAULT_MAX_SEQ_LENGTH}
MAX_PACKED_TOKENS=${MAX_PACKED_TOKENS:-$DEFAULT_MAX_PACKED_TOKENS}
MAX_BUFFER_SIZE=${MAX_BUFFER_SIZE:-$DEFAULT_MAX_BUFFER_SIZE}
NUM_IMAGES_EXPECTED=${NUM_IMAGES_EXPECTED:-$DEFAULT_NUM_IMAGES_EXPECTED}
MAX_NUM_FRAME=${MAX_NUM_FRAME:-$DEFAULT_MAX_NUM_FRAME}

echo "🚀 Starting InternVL2.5-2B How2Sign Training on 2×A100 (DeepSpeed ZeRO-2)"
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
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

MODEL_CACHE_DIR="/home/mh2803/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-2B"
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
MODEL_NAME = "OpenGVLab/InternVL2_5-2B"
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

MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}
export MASTER_PORT
echo "MASTER_PORT: $MASTER_PORT"

echo "🏃 Starting training with torchrun + DeepSpeed..."
torchrun --nproc_per_node=$NUM_DEVICES --master_port=$MASTER_PORT \
    internvl_chat/internvl/train/internvl_chat_finetune.py \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
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
    --freeze_backbone False \
    --freeze_mlp False \
    --unfreeze_vit_layers 8 \
    --use_llm_lora 8 \
    --use_backbone_lora 0 \
    --bf16 True \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_steps 20000 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 1 \
    --logging_first_step True \
    --evaluation_strategy no \
    --report_to none \
    --disable_tqdm False \
    --grad_checkpoint True \
    --dataloader_num_workers 4 \
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
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --strict_mode False \
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

