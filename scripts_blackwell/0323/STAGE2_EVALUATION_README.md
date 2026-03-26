# Stage2 Evaluation Guide

## Problem Summary

Stage2 checkpoint (`checkpoint-1019`) had compatibility issues when loading:

1. **Weight key structure problem**: Extra `base_model` layer wrapping
   - Original: `language_model.base_model.model.base_model.model.layers...`
   - Should be: `language_model.base_model.model.layers...`

2. **Unmerged LoRA weights**: Checkpoint contained separate LoRA_A and LoRA_B tensors

3. **CUDA errors during inference**: Shape mismatches due to above issues

## Solution

Two processing scripts have been created:

### 1. Merge LoRA Weights + Fix Keys
**Script**: `merge_stage2_weights_v2.py`

- Merges 336 LoRA weight pairs into base weights
- Removes extra `base_model` wrapper from all weight keys
- Creates clean checkpoint: `checkpoint-1019-merged-v2`
- **Status**: ✅ Already completed

### 2. Evaluation Scripts

#### Option A: Final Evaluation (Recommended)
**Script**: `evaluate_stage2_final.sh`

Uses the processed checkpoint with all fixes applied.

```bash
# Default: 10 samples
bash evaluate_stage2_final.sh

# Custom GPU and sample count
GPU_IDS="1" MAX_SAMPLES=50 bash evaluate_stage2_final.sh
```

#### Option B: Original Script (Updated)
**Script**: `evaluate_internvl2_5_how2sign_1b_broad_h2s_1_open_067_youtube_025_stage2_wandb.sh`

Has been updated to use the processed checkpoint.

```bash
bash evaluate_internvl2_5_how2sign_1b_broad_h2s_1_open_067_youtube_025_stage2_wandb.sh
```

## Checkpoints Available

| Path | Description | Status |
|------|-------------|--------|
| `checkpoint-1019` | Original Stage2 checkpoint | ❌ Has issues |
| `checkpoint-1019-merged-v2` | LoRA merged + keys fixed | ✅ Ready for evaluation |

## Environment

- GPU: RTX PRO 6000 Blackwell (sm_120)
- CUDA: 12.8
- PyTorch: 2.7.0+cu128
- DeepSpeed: 0.18.7

## Notes

- The processed checkpoint (`checkpoint-1019-merged-v2`) is self-contained and includes all necessary files
- LoRA weights have been permanently merged into base weights, so no PEFT loading is needed
- Configuration (force_image_size=224, downsample_ratio=0.5) is preserved
- Evaluation results will be saved to `outputs/blackwell/finetune_stage2_h2s_stage1_broad_h2s_1_open_067_youtube_0_25_checkpoint6225/`
