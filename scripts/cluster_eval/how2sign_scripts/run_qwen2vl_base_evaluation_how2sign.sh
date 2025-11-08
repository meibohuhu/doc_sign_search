#!/bin/bash -l
# Qwen2.5-VL-3B-Instruct BASE Model Evaluation on How2Sign Dataset
# Baseline comparison without fine-tuning

#SBATCH --job-name=qwen2vl_base_eval_how2sign
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=64g

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

# Configuration for How2Sign dataset
MODEL_BASE="Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_FOLDER="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_clips_stable_224x224/"
QUESTION_FILE="/home/mh2803/projects/sign_language_llm/how2sign/video/test_raw_videos/segmented_test_videos_filtered.sample120.json"
OUT_DIR="/home/mh2803/projects/sign_language_llm/outputs/how2sign_base/"

echo "🎬 Qwen2.5-VL-3B-Instruct BASE Model Evaluation on How2Sign"
echo "============================================================"
echo "Model: $MODEL_BASE (pretrained, NO fine-tuning)"
echo "Video Folder: $VIDEO_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Dir: $OUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Run evaluation on test set
# Adjust --max-samples as needed or remove it to process all samples
/home/mh2803/miniconda3/envs/qwenvl/bin/python scripts/cluster_eval/how2sign_scripts/qwen2vl_base_evaluation_how2sign.py \
    --model-base "$MODEL_BASE" \
    --video-folder "$VIDEO_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --out-dir "$OUT_DIR" \
    --max-samples 18 \
    --enable-evaluation \
    --video-fps 18

echo "🎉 BASE model evaluation job completed!"
