#!/bin/bash
# Setup script for Qwen2.5-VL training environment with CUDA 12.4
# Run this script to set up the complete environment

set -e  # Exit on error

echo "=========================================="
echo "Qwen2.5-VL Environment Setup"
echo "=========================================="

# Step 1: Load CUDA 12.4
echo ""
echo "Step 1: Loading CUDA 12.4..."
echo "Available CUDA versions:"
spack find -l cuda

echo ""
echo "Please run the following command to load CUDA 12.4:"
echo "  spack load cuda@12.4.0/<hash>"
echo ""
read -p "Have you loaded CUDA 12.4? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please load CUDA first, then run this script again."
    exit 1
fi

# Verify CUDA is loaded
echo "Verifying CUDA installation..."
if [ -z "$CUDA_HOME" ]; then
    echo "Warning: CUDA_HOME is not set. Please ensure CUDA is loaded correctly."
    echo "You may need to manually set: export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))"
else
    echo "CUDA_HOME: $CUDA_HOME"
fi

# Step 2: Create conda environment
echo ""
echo "Step 2: Creating conda environment 'qwen' with Python 3.10..."
conda create -n qwen python=3.10 -y

echo ""
echo "Activating qwen environment..."
echo "NOTE: After this script completes, run: conda activate qwen"

# Install packages in the new environment
eval "$(conda shell.bash hook)"
conda activate qwen

# Step 3: Install PyTorch with CUDA 12.4 support
echo ""
echo "Step 3: Installing PyTorch 2.6.0 with CUDA 12.4..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Step 4: Install flash-attn
echo ""
echo "Step 4: Installing flash-attn (this may take a while)..."
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Step 5: Install transformers and accelerate
echo ""
echo "Step 5: Installing transformers and accelerate..."
pip install git+https://github.com/huggingface/transformers accelerate

# Step 6: Install qwen-vl-utils
echo ""
echo "Step 6: Installing qwen-vl-utils..."
pip install qwen-vl-utils[decord]

# Step 7: Install additional required packages
echo ""
echo "Step 7: Installing additional required packages..."
pip install deepspeed
pip install peft
pip install ujson
pip install liger_kernel
pip install datasets  # Note: 'datasets' not 'dataset'
pip install wandb

# Step 8: Install specific transformers version for training
echo ""
echo "Step 8: Installing transformers==4.51.3 for training..."
pip install transformers==4.51.3

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To activate this environment in the future, run:"
echo "  conda activate qwen"
echo ""
echo "Installed packages summary:"
pip list | grep -E "torch|transformers|accelerate|deepspeed|peft|qwen|flash"
echo ""
echo "To verify CUDA is working with PyTorch, run:"
echo "  python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")'"
