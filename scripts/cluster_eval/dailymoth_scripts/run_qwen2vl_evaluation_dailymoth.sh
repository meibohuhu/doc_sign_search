#!/bin/bash -l
# Qwen2VL Evaluation on DailyMoth-70h Dataset
# Uses improved qwen2vl_evaluation_dailymoth_claude.py with better model loading

#SBATCH --job-name=qwen2vl_eval_dailymoth
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=32g

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths
export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:/home/mh2803/projects/sign_language_llm/evaluation:$PYTHONPATH"
export OMP_NUM_THREADS=8

# GPU optimization settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_USE_CUDA_DSA=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=1

# Change to project directory
cd /home/mh2803/projects/sign_language_llm

# Configuration for DailyMoth-70h dataset
# CHECKPOINT_PATH="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/qwen2vl_ssvp_2xa100_12fps_diverse/checkpoint-6000"
CHECKPOINT_PATH="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/qwen2vl_ssvp_4xa100_20fps/checkpoint-4000"

MODEL_BASE="Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_FOLDER="/home/mh2803/projects/sign_language_llm/dailymoth-70h/dailymoth-70h/unblurred_clips/videos/"
QUESTION_FILE="/home/mh2803/projects/sign_language_llm/vanshika/asl_test/test_ssvp.json"
OUT_DIR="/home/mh2803/projects/sign_language_llm/outputs/dailymoth/"

echo "🎬 Qwen2VL Evaluation on DailyMoth-70h Dataset"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Model Base: $MODEL_BASE"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Available checkpoints in parent directory:"
    ls -la $(dirname "$CHECKPOINT_PATH") | grep checkpoint
    exit 1
fi

echo "✅ Checkpoint found: $CHECKPOINT_PATH"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Video resolution settings (MUST match training!)
# For 224x224 training: min-pixels=50176, max-pixels=50176
# For 320x320 training: min-pixels=102400, max-pixels=102400
RESOLUTION=224
MIN_PIXELS=$((RESOLUTION * RESOLUTION))
MAX_PIXELS=$((RESOLUTION * RESOLUTION))

echo "📐 Video resolution: ${RESOLUTION}x${RESOLUTION} (${MIN_PIXELS} pixels)"
echo ""

# Run evaluation on full test set (4185 samples)
# Adjust --max-samples as needed or remove it to process all samples
/home/mh2803/miniconda3/envs/qwenvl/bin/python scripts/cluster_eval/dailymoth_scripts/qwen2vl_evaluation_dailymoth_claude.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --model-base "$MODEL_BASE" \
    --video-folder "$VIDEO_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --out-dir "$OUT_DIR" \
    --max-samples 250 \
    --video-fps 12 \
    --min-pixels "$MIN_PIXELS" \
    --max-pixels "$MAX_PIXELS"

echo "🎉 DailyMoth-70h evaluation job completed!"

