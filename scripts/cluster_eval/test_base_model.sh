#!/bin/bash -l
# Test Base Qwen2VL Model (without LoRA) with 10 videos and evaluation metrics

#SBATCH --job-name=test_base_qwen2vl
#SBATCH --error=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/err_%j.txt
#SBATCH --output=/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=32g

spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths
export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"
export PYTHONPATH="/home/mh2803/projects/sign_language_llm/qwenvl/Qwen2-VL-Finetune/src:$PYTHONPATH"
export OMP_NUM_THREADS=8

# GPU optimization settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_USE_CUDA_DSA=1

# Change to project directory
cd /home/mh2803/projects/sign_language_llm

echo "🎬 Testing Base Qwen2VL Model (No LoRA)"
echo "========================================"
echo ""

# Run base model test
/home/mh2803/miniconda3/envs/qwenvl/bin/python scripts/cluster_eval/test_base_model.py

echo "🎉 Base model test completed!"
