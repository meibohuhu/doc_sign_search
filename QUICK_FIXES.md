# Quick Fixes for Training Issue

## Current Configuration Analysis

From your training script (`finetune_qwen2vl_how2sign_2xa100_filtered_320resolution_scavenger.sh`):

**Current Settings:**
- ✅ Vision tower: **NOT frozen** (`--freeze_vision_tower False`) - GOOD
- ❌ Vision LR: `2e-6` - **VERY LOW** - may be insufficient
- ⚠️  LLM: **FROZEN** (`--freeze_llm True`) - Only LoRA trainable - limited capacity
- ✅ Merger: **NOT frozen** (`--freeze_merger False`) - GOOD

**Issue**: With vision LR = 2e-6 and only LoRA adapters trainable on LLM, the model has very limited capacity to learn the sign language mapping.

---

## Immediate Fixes to Try

### Fix #1: Increase Vision Learning Rate
```bash
# Change this line in your training script:
--vision_lr 2e-6
# To:
--vision_lr 1e-5  # or even 2e-5 for faster learning
```

**Rationale**: Vision features need to adapt significantly to understand sign language. 2e-6 is likely too conservative.

### Fix #2: Unfreeze More LLM Layers
```bash
# Instead of:
--freeze_llm True

# Try:
--freeze_llm False
# OR use selective unfreezing:
--unfreeze_topk_llm 4  # Unfreeze last 4 layers
```

**Rationale**: Sign language translation is a complex task. Only training LoRA adapters might not provide enough capacity.

### Fix #3: Adjust Learning Rate Ratios
```bash
# Suggested configuration:
--learning_rate 1e-5      # Main LR (for LoRA or unfrozen LLM)
--vision_lr 1e-5          # Increase from 2e-6
--merger_lr 1e-5          # Keep same
```

**Rationale**: Vision and language need to co-adapt. Too large a gap between their learning rates can cause misalignment.

### Fix #4: Add Validation with BLEU
Modify your training script to evaluate BLEU during training:
```bash
--evaluation_strategy steps
--eval_steps 500
--metric_for_best_model bleu4
--load_best_model_at_end True
```

---

## Recommended Configuration

```bash
deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero3_qwen2vl.json \
    --model_id $MODEL_NAME \
    --data_path /home/mh2803/projects/sign_language_llm/how2sign/video/train_videos/segmented_train_videos_corrupted_removed.json \
    --image_folder /shared/rc/llm-gen-agent/mhu/videos/how2sign_train_segment_clips_stable_320x320/ \
    --output_dir /shared/rc/llm-gen-agent/mhu/qwen2.5vl/1018/qwen2vl_how2sign_2xa100_filtered_320resolution_scavenger_v2/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((320 * 320)) \
    --video_max_pixels $((320 * 320)) \
    --fps 12 \
    --max_grad_norm 1.0 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_steps 6000 \
    --use_liger True \
    --freeze_vision_tower False \
    --freeze_llm False \              # CHANGED: Unfreeze LLM
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --gradient_checkpointing True \
    --lora_enable False \              # CHANGED: Disable LoRA if unfreezing LLM
    --vision_lr 1e-5 \                  # CHANGED: Increased from 2e-6
    --merger_lr 1e-5 \
    --report_to none
```

**OR** keep LoRA but increase vision LR:

```bash
--freeze_llm True \                     # Keep LoRA
--lora_enable True \
--lora_rank 64 \                        # INCREASE: More LoRA capacity
--lora_alpha 128 \
--vision_lr 1e-5 \                      # INCREASED: Much higher vision LR
```

---

## What to Monitor

After making changes, monitor:

1. **BLEU Score on Validation Set** (not just loss!)
   - Should increase from 6% to at least 15-20% after 1000 steps
   - Should continue increasing over training

2. **Vision Gradient Norms**
   - Add logging to check if vision encoder receives gradients
   - Should be non-zero throughout training

3. **Training Loss vs BLEU Correlation**
   - Loss might increase slightly as model learns actual task (this is OK!)
   - BLEU should increase as loss decreases

4. **Semantic Similarity** (use analysis script)
   - Run `scripts/analyze_training_issue.py` on new checkpoints
   - Word overlap should increase from 19% to >30%

---

## Expected Timeline

- **Steps 0-1000**: BLEU should start increasing (from 6% to ~10-15%)
- **Steps 1000-3000**: Steady improvement (BLEU 15-25%)
- **Steps 3000-6000**: Further refinement (BLEU 25-35% on training set)

If BLEU doesn't improve after 1000 steps with these changes, check:
1. Training data quality/alignment
2. Video preprocessing pipeline
3. Model architecture compatibility

---

## Quick Test Command

To verify vision encoder is being trained:
```bash
# Add this to check gradients during training
python -c "
import torch
checkpoint = torch.load('checkpoint-1000/pytorch_model.bin', map_location='cpu')
vision_params = [k for k in checkpoint.keys() if 'vision' in k.lower()]
print(f'Vision parameters in checkpoint: {len(vision_params)}')
print('Sample keys:', vision_params[:5])
"
```

If vision parameters exist but are frozen, their gradients will be zero. Check training logs for gradient norms.

