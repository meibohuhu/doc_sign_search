# InternVL Requirements Comparison

## Differences between INSTALLATION.md and Actual Installation

### Key Version Differences

| Package | INSTALLATION.md | Actual Installed | Notes |
|---------|----------------|------------------|-------|
| `python` | 3.9 | **3.10.19** | ⚠️ Newer Python version |
| `flash-attn` | 2.3.6 | **2.5.7** | ⚠️ Newer version installed |
| `deepspeed` | >=0.13.5 | **0.18.2** | ⚠️ Much newer version |
| `accelerate` | <1 | **0.34.2** | ✅ Within constraint |
| `transformers` | 4.37.2 | **4.37.2** | ✅ Matches |
| `peft` | 0.10.0 | **0.10.0** | ✅ Matches |
| `torch` | >=2 | **2.4.0+cu121** | ✅ Meets requirement |
| `torchvision` | >=0.15 | **0.19.0+cu121** | ✅ Meets requirement |

### Additional Packages Not Mentioned in INSTALLATION.md

These packages are installed but not explicitly mentioned in INSTALLATION.md:
- `datasets==4.4.1` - For dataset handling
- `safetensors==0.6.2` - For model serialization
- `triton==3.0.0` - For GPU kernels
- Various CUDA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12, etc.)

### Optional Packages (Not Installed)

According to INSTALLATION.md, these are optional:
- `mmcv-full==1.6.2` - For segmentation (not installed)
- `apex` - For segmentation (not installed)

### Recommendations

1. **flash-attn**: The installed version (2.5.7) is newer than documented (2.3.6). This is likely fine as it's backward compatible, but if you encounter issues, you may want to downgrade to 2.3.6.

2. **deepspeed**: The installed version (0.18.2) is much newer than the minimum requirement (0.13.5). This should be fine, but test thoroughly.

3. **Use actual installed versions**: The `requirements_internvl.txt` file uses the actual installed versions to ensure reproducibility.

## Installation Command

To install using the actual versions:

```bash
# Note: Using Python 3.10 instead of 3.9 (as per actual installation)
conda create -n internvl python=3.10 -y
conda activate internvl
pip install -r requirements_internvl.txt
pip install flash-attn==2.5.7 --no-build-isolation
```

Note: CUDA-enabled packages (torch, torchvision, torchaudio) may need to be installed from PyTorch's index:
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

