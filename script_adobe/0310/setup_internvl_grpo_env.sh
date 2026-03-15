#!/bin/bash
#
# Setup conda environment for InternVL GRPO training
# Environment: internvl_grpo
# PyTorch 2.7.0+cu128 (Blackwell GPU compatible)
#

set -e

CONDA_BASE="$HOME/anaconda3"
ENV_NAME="internvl_grpo"
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"

echo "Setting up conda environment: $ENV_NAME"
echo "========================================="

# Initialize conda
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create environment with Python 3.10 (skip if already exists)
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating conda environment with Python 3.10..."
    conda create -n $ENV_NAME python=3.10 -y
else
    echo "Environment $ENV_NAME already exists, skipping creation."
fi

# Use the conda env's pip and python directly to avoid PATH issues
PIP="$ENV_DIR/bin/pip"
PYTHON="$ENV_DIR/bin/python"

echo "Using pip: $PIP"
echo "Using python: $PYTHON"

# PyTorch 2.7.0+cu128 (native Blackwell sm_120 support)
echo "Installing PyTorch 2.7.0+cu128..."
$PIP install "torch==2.7.0+cu128" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Transformers (latest for TRL compatibility)
echo "Installing transformers, peft, accelerate..."
$PIP install "transformers>=4.48.0" "accelerate>=1.2.0" "peft>=0.13.0"

# TRL (for GRPOConfig, selective_log_softmax, unwrap_model_for_generation)
echo "Installing TRL..."
$PIP install "trl>=0.14.0"

# DeepSpeed
echo "Installing DeepSpeed..."
$PIP install deepspeed

# GRPO reward dependencies
echo "Installing reward function dependencies..."
$PIP install bert-score nltk

# Download NLTK data
$PYTHON -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Video processing
echo "Installing video processing libraries..."
$PIP install decord opencv-python-headless pillow

# Other dependencies (matching internvl requirements)
echo "Installing other dependencies..."
$PIP install einops timm sentencepiece protobuf

# CUDA toolkit (nvcc required by DeepSpeed and Flash Attention)
echo "Installing CUDA toolkit (nvcc)..."
if [ ! -f "$ENV_DIR/bin/nvcc" ]; then
    conda install -n $ENV_NAME -c nvidia cuda-toolkit -y
fi
export CUDA_HOME="$ENV_DIR"
echo "nvcc: $($ENV_DIR/bin/nvcc --version | tail -1)"

# Flash Attention 2 (for InternVL)
echo "Installing Flash Attention 2 (CUDA_HOME=$CUDA_HOME)..."
$PIP install flash-attn --no-build-isolation --no-cache-dir

echo ""
echo "========================================="
echo "Environment setup complete: $ENV_NAME"
echo ""
echo "Verify installation:"
$PYTHON -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import transformers
print(f'Transformers: {transformers.__version__}')
import trl
print(f'TRL: {trl.__version__}')
from trl.trainer.grpo_config import GRPOConfig
print('GRPOConfig: OK')
from trl.trainer.utils import selective_log_softmax
print('selective_log_softmax: OK')
from trl.models import unwrap_model_for_generation
print('unwrap_model_for_generation: OK')
import peft
print(f'PEFT: {peft.__version__}')
import deepspeed
print(f'DeepSpeed: {deepspeed.__version__}')
import bert_score
print('bert_score: OK')
import nltk
print('NLTK: OK')
import decord
print('decord: OK')
print()
print('All dependencies verified!')
"
echo ""
echo "To use: conda activate $ENV_NAME"
echo "To train: GPU_IDS=0,1,2,3 bash script_adobe/0310/grpo_internvl2_5_how2sign_1b_blackwell.sh"
