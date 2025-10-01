# Qwen2.5-VL Training Environment Setup Guide

This guide will help you set up a CUDA environment for Qwen2.5-VL training on your RC cluster.

## Prerequisites
- Access to RC cluster with spack and conda
- CUDA 12.4 available via spack

---

## Step 1: Load CUDA 12.4

First, find available CUDA versions:
```bash
spack find -l cuda
```

Load CUDA 12.4 (replace `<hash>` with the actual hash from the command above):
```bash
spack load cuda@12.4.0/<hash>
```

Verify CUDA is loaded:
```bash
echo $CUDA_HOME
nvcc --version
```

If `CUDA_HOME` is not set, manually set it:
```bash
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
```

---

## Step 2: Create Conda Environment

Create a new conda environment with Python 3.10:
```bash
conda create -n qwenvl python=3.10 -y
conda activate qwenvl
```

---

## Step 3: Install PyTorch with CUDA 12.4 Support

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Verify PyTorch installation:
```bash
conda activate qwenvl && python -c "
import torch
print('✅ Environment Check:')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()} (expected: False on login node)')
print(f'  Device count: {torch.cuda.device_count()} (expected: 0 on login node)')
print()
print('🎯 Your setup is ready for cluster submission!')
print('   When job runs on GPU node, CUDA will be available.')
"
```

Expected output:
- PyTorch: 2.6.0+cu124
- CUDA available: True
- CUDA version: 12.4

---

## Step 4: Install Flash Attention

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

**Note:** This compilation may take 10-15 minutes. Be patient!

---

## Step 5: Install Transformers and Accelerate

```bash
pip install git+https://github.com/huggingface/transformers accelerate
```

---

## Step 6: Install Qwen VL Utils

```bash
pip install qwen-vl-utils[decord]
```

---

## Step 7: Install Additional Required Packages

```bash
pip install deepspeed
pip install peft
pip install ujson
pip install liger_kernel
pip install datasets
pip install wandb
```

---

## Step 8: Install Specific Transformers Version for Training

**Important:** Use transformers==4.51.3 for training compatibility:
```bash
pip install transformers==4.51.3
```

This will downgrade from the git version to the stable release needed for training.

---

## Verification

After installation, verify everything is set up correctly:

### Check installed packages:
```bash
pip list | grep -E "torch|transformers|accelerate|deepspeed|peft|qwen|flash"
```

### Test CUDA with PyTorch:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### Test Flash Attention:
```bash
python -c "import flash_attn; print('Flash Attention installed successfully')"
```

### Test Transformers:
```bash
python -c "from transformers import AutoModel; print('Transformers working')"
```

---

## Quick Reference Commands

### Activate environment:
```bash
conda activate qwenvl
```

### Deactivate environment:
```bash
conda deactivate
```

### Remove environment (if needed):
```bash
conda env remove -n qwen
```

### Export environment:
```bash
conda env export > qwen_environment.yml
```

---

## Troubleshooting

### Issue: CUDA not found during flash-attn installation
**Solution:** Ensure CUDA_HOME is set:
```bash
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Issue: Out of memory during flash-attn compilation
**Solution:** Request more memory in your job or compile on a node with more RAM

### Issue: torch.cuda.is_available() returns False
**Solution:** 
1. Check if you're on a GPU node
2. Verify CUDA drivers are available: `nvidia-smi`
3. Reinstall PyTorch with correct CUDA version

---

## Notes

- **Flash Attention** compilation requires significant time and memory
- **Transformers version:** We install from git first to get latest features, then pin to 4.51.3 for training stability
- **Dataset vs datasets:** The correct package name is `datasets` (plural), not `dataset`
- **DeepSpeed:** May require additional configuration for multi-node training
- **Wandb:** Remember to login: `wandb login`

---

## Next Steps

After setup is complete:
1. Test the environment with a simple Qwen2.5-VL inference script
2. Configure your training scripts
3. Set up DeepSpeed configuration if needed
4. Configure Wandb for experiment tracking

Good luck with your training! 🚀
