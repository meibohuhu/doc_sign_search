#!/bin/bash
#
# Setup conda environment for InternVL GRPO training
# Environment: internvl_grpo
# Compatible with standard GPU clusters (A100, V100, H100, etc.)
#
# Usage:
#   bash setup_internvl_grpo_env_regular.sh
#
# Override defaults:
#   CUDA_VERSION=cu118 bash setup_internvl_grpo_env_regular.sh   # for CUDA 11.8
#   CUDA_VERSION=cu121 bash setup_internvl_grpo_env_regular.sh   # for CUDA 12.1 (default)
#   CUDA_VERSION=cu124 bash setup_internvl_grpo_env_regular.sh   # for CUDA 12.4

set -e

CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
ENV_NAME="internvl_grpo"
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"

# Normalize CUDA_VERSION if it's a raw version like "12.4" or "12.4.1"
if [[ "$CUDA_VERSION" =~ ^[0-9]+\.[0-9] ]]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        CUDA_VERSION="cu128"
    elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        CUDA_VERSION="cu124"
    elif [ "$CUDA_MAJOR" -ge 12 ]; then
        CUDA_VERSION="cu121"
    else
        CUDA_VERSION="cu118"
    fi
    echo "Normalized CUDA_VERSION to: ${CUDA_VERSION}"
fi

# CUDA version: auto-detect or override via env var
if [ -z "$CUDA_VERSION" ]; then
    if command -v nvcc &>/dev/null; then
        CUDA_RAW=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        CUDA_MAJOR=$(echo $CUDA_RAW | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_RAW | cut -d. -f2)
        if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
            CUDA_VERSION="cu124"
        elif [ "$CUDA_MAJOR" -ge 12 ]; then
            CUDA_VERSION="cu121"
        else
            CUDA_VERSION="cu118"
        fi
        echo "Auto-detected CUDA ${CUDA_RAW} → using PyTorch index: ${CUDA_VERSION}"
    else
        CUDA_VERSION="cu121"
        echo "nvcc not found, defaulting to ${CUDA_VERSION}"
    fi
fi

# PyTorch version selection based on CUDA
case "$CUDA_VERSION" in
    cu118)
        TORCH_VERSION="2.1.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        ;;
    cu121)
        TORCH_VERSION="2.1.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        ;;
    cu124)
        TORCH_VERSION="2.6.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        ;;
    cu128)
        # Blackwell only
        TORCH_VERSION="2.7.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        ;;
    *)
        echo "Unknown CUDA_VERSION: $CUDA_VERSION. Use cu118, cu121, cu124, or cu128."
        exit 1
        ;;
esac

echo "Setup InternVL GRPO Environment"
echo "================================"
echo "Conda base:    $CONDA_BASE"
echo "Env name:      $ENV_NAME"
echo "CUDA version:  $CUDA_VERSION"
echo "PyTorch:       $TORCH_VERSION"
echo ""

# Initialize conda
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create environment
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating conda environment with Python 3.10..."
    conda create -n $ENV_NAME python=3.10 -y
else
    echo "Environment $ENV_NAME already exists, skipping creation."
fi

PIP="$ENV_DIR/bin/pip"
PYTHON="$ENV_DIR/bin/python"

echo "Using pip:    $PIP"
echo "Using python: $PYTHON"
echo ""

# PyTorch
echo "Installing PyTorch ${TORCH_VERSION}+${CUDA_VERSION}..."
$PIP install "torch==${TORCH_VERSION}+${CUDA_VERSION}" torchvision torchaudio \
    --index-url "$TORCH_INDEX"

# Core training libraries
echo "Installing transformers, peft, accelerate..."
$PIP install "transformers>=4.48.0" "accelerate>=1.2.0" "peft>=0.13.0"

echo "Installing TRL..."
$PIP install "trl>=0.14.0"

echo "Installing DeepSpeed..."
$PIP install deepspeed

# Reward function dependencies
echo "Installing reward function dependencies..."
$PIP install bert-score nltk rouge-score

# Download NLTK data
$PYTHON -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Video processing
echo "Installing video processing libraries..."
$PIP install decord opencv-python-headless pillow

# Other InternVL dependencies
echo "Installing other dependencies..."
$PIP install einops timm sentencepiece protobuf

# CUDA_HOME for DeepSpeed / Flash Attention
# Prefer system CUDA if available, fall back to conda env
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    echo "Using system CUDA_HOME: $CUDA_HOME"
elif [ -f "$ENV_DIR/bin/nvcc" ]; then
    export CUDA_HOME="$ENV_DIR"
    echo "Using conda CUDA_HOME: $CUDA_HOME"
else
    echo "Installing CUDA toolkit into conda env..."
    conda install -n $ENV_NAME -c nvidia cuda-toolkit -y
    export CUDA_HOME="$ENV_DIR"
fi

# Flash Attention 2
echo "Installing Flash Attention 2 (CUDA_HOME=$CUDA_HOME)..."
$PIP install flash-attn --no-build-isolation --no-cache-dir

echo ""
echo "================================"
echo "Verifying installation..."
echo ""
$PYTHON -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:  {torch.version.cuda}')
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
import transformers
print(f'Transformers:  {transformers.__version__}')
import trl
print(f'TRL:           {trl.__version__}')
from trl.trainer.grpo_config import GRPOConfig
print('GRPOConfig:    OK')
from trl.trainer.utils import selective_log_softmax
print('selective_log_softmax: OK')
from trl.models import unwrap_model_for_generation
print('unwrap_model_for_generation: OK')
import peft
print(f'PEFT:          {peft.__version__}')
import deepspeed
print(f'DeepSpeed:     {deepspeed.__version__}')
import bert_score
print('bert_score:    OK')
import nltk
print('NLTK:          OK')
import rouge_score
print('rouge_score:   OK')
import decord
print('decord:        OK')
print()
print('All dependencies verified!')
"

echo ""
echo "================================"
echo "Setup complete: $ENV_NAME"
echo ""
echo "To activate:  conda activate $ENV_NAME"
echo "To train:     GPU_IDS=0,1,2,3 bash script_adobe/0310/grpo_internvl2_5_how2sign_1b_blackwell.sh"
echo ""
echo "To override CUDA version:"
echo "  CUDA_VERSION=cu118 bash setup_internvl_grpo_env_regular.sh   # CUDA 11.8"
echo "  CUDA_VERSION=cu121 bash setup_internvl_grpo_env_regular.sh   # CUDA 12.1"
echo "  CUDA_VERSION=cu124 bash setup_internvl_grpo_env_regular.sh   # CUDA 12.4"