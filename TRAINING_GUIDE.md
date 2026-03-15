# InternVL2.5 How2Sign Training Guide

## Quick Start

### Run training on specific GPUs:
```bash
cd /home/stu2/s15/mh2803/workspace/doc_sign_search/InternVL
GPU_IDS="0,1,2,3" bash ../script_adobe/0310/finetune_internvl2_5_how2sign_1b_unfreeze.sh
```

### Monitor training:
```bash
# Watch the latest log file
tail -f script_adobe/checkpoints/finetune_internvl2_5_how2sign_1b_unfreeze/training_*.log

# Check GPU usage
watch nvidia-smi

# Check specific GPU
nvidia-smi -i 0
```

## Configuration

- **Model**: InternVL2.5-1B
- **Training config**: `script_adobe/0310/finetune_internvl2_5_how2sign_1b_unfreeze.sh`
- **DeepSpeed config**: `InternVL/internvl_chat/zero_stage0_config.json` (Stage 0 for Blackwell compatibility)
- **Data**: `script_adobe/train_how2sign_meta.json`
- **Output**: `script_adobe/checkpoints/finetune_internvl2_5_how2sign_1b_unfreeze/`

## Hardware
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (8x available)
- **GPU Memory per card**: 98GB
- **Batch size**: 2 per device, 64 global (with 32 gradient accumulation steps)

## Common Issues & Fixes

### Issue: "CUDA error: no kernel image is available for execution on the device"
- **Status**: ✅ FIXED (using DeepSpeed Stage 0 config)
- **Details**: RTX PRO 6000 Blackwell (sm_120) incompatible with Stage 1 kernels
- **Solution**: Use `zero_stage0_config.json` instead of `zero_stage1_config.json`

### Issue: GPU not found or CUDA not available
- **Check**: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
- **Expected**: Should show "NVIDIA RTX PRO 6000 Blackwell Server Edition"

### Issue: Out of Memory
- **Reduce**: `--per_device_train_batch_size` in the training script
- **Increase**: `--gradient_accumulation_steps` to maintain effective batch size

## Performance Tips

1. **Use all available GPUs**: `GPU_IDS="0,1,2,3,4,5,6,7"`
2. **Monitor memory**: GPU 0 should use ~30-40GB, others minimal (data loading overhead)
3. **Log location**: Check output for `📝 Log file:` path
4. **Training time**: Monitor progress via log file for loss/accuracy metrics

## Environment
- **Python**: 3.10
- **PyTorch**: 2.5.1+cu121
- **DeepSpeed**: 0.18.2
- **Transformers**: Latest from environment
- **CUDA**: 12.1 (Driver 590.48)
