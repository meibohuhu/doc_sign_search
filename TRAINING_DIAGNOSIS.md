# Training Diagnosis: Low BLEU4 (6.12%) Despite Loss Reduction

## Problem Summary

**Symptom**: BLEU4 score of 6.12% on training subset, despite training loss reducing to 1.4 at checkpoint 5000.

**Root Cause**: Model is generating fluent, grammatically correct English but **not learning the sign language-to-text mapping**. The model has learned to generate plausible text patterns but ignores video content.

**Evidence**:
- Only 19.5% average word overlap between predictions and ground truth
- 25% of predictions have <10% semantic similarity
- Grammatical quality is excellent (98.9% proper punctuation), indicating text generation is working
- Topic mismatch: predictions don't match ground truth topics

---

## Key Issues Identified

### 1. **Loss-Task Mismatch**
- **Problem**: Cross-entropy loss optimizes token prediction accuracy, not translation quality
- **Impact**: Loss can decrease while model learns wrong patterns (e.g., generating generic English)
- **Evidence**: Loss 1.4 is "good" for language modeling, but BLEU is terrible

### 2. **Vision-Text Disconnection**
- **Problem**: Model likely not properly attending to or learning from video features
- **Impact**: Model generates text without understanding sign language
- **Evidence**: Low semantic overlap despite good grammar

### 3. **Training Data/Procedure Issues**
- **Problem**: Possible misalignment between training objective and evaluation metric
- **Impact**: Model optimizes for wrong objective

---

## Recommendations

### Immediate Actions

1. **Verify Vision Encoder Status**
   ```bash
   # Check your training script - is vision tower frozen?
   # Should be: --freeze_vision_tower False
   ```
   - If vision tower is frozen, unfreeze it or use a lower learning rate
   - Vision features must be trainable to learn sign language patterns

2. **Check Training Data Alignment**
   - Verify video-text pairs are correctly matched
   - Ensure no data corruption or misaligned pairs
   - Check if video preprocessing matches training (resolution, fps, sampling)

3. **Examine Training Configuration**
   - Review learning rates for vision vs. language components
   - Check if using proper video input format
   - Verify video features are actually used in forward pass


### Training Strategy Improvements

1. **Curriculum Learning**
   - Start with shorter, simpler sequences
   - Gradually increase complexity
   - Begin with high-quality, diverse examples

2. **Multi-Task Learning**
   - Add auxiliary tasks (e.g., sign word classification)
   - Use contrastive learning for vision-text alignment
   - Regularize to prevent mode collapse

3. **Vision Encoder Pre-training**
   - Fine-tune vision encoder on sign language recognition first
   - Use frozen vision features initially, then unfreeze
   - Consider sign language-specific visual augmentations

4. **Loss Function Improvements**
   - Combine cross-entropy with BLEU/ROUGE loss
   - Use sequence-level training (e.g., REINFORCE with BLEU reward)
   - Add contrastive loss for better vision-text alignment

5. **Training Stability**
   - Check gradient flow through vision components
   - Monitor vision encoder gradients (should be non-zero if unfrozen)
   - Use gradient clipping appropriate for vision-text models

---

## Specific Technical Checks

### 1. Vision Encoder Check
```python
# Add to training script to verify vision features are being used
for name, param in model.named_parameters():
    if 'vision' in name.lower() or 'video' in name.lower():
        print(f"{name}: requires_grad={param.requires_grad}, "
              f"grad_norm={param.grad.norm() if param.grad is not None else 0}")
```

### 2. Forward Pass Verification
- Add logging to confirm video tensors are non-empty
- Check video feature dimensions match expected input
- Verify attention masks include video tokens

### 3. Training Data Verification
```python
# Sample check: verify video-text alignment
import random
samples = random.sample(train_data, 5)
for sample in samples:
    print(f"Video: {sample['video']}")
    print(f"Text: {sample['translation']}")
    print(f"Do they match? (manual check)")
```

### 4. Learning Rate Check
- Vision encoder LR should be lower than LLM LR (typically 1/10 to 1/100)
- But should NOT be zero (if vision tower frozen)
- Typical setup: `vision_lr=1e-6`, `llm_lr=1e-5`

---

## Expected Outcomes After Fixes

- **BLEU4 should increase** from 6% to at least 20-30% on training set
- **Loss might increase slightly** as model learns actual task (this is okay!)
- **Semantic similarity should improve** (word overlap >40%)
- **Predictions should match ground truth topics**

---

## Long-Term Improvements

1. **Evaluation Framework**
   - Track multiple metrics (BLEU, ROUGE, METEOR, semantic similarity)
   - Use human evaluation for qualitative assessment
   - Build error taxonomy

2. **Model Architecture**
   - Consider specialized sign language architectures
   - Use temporal attention for video sequences
   - Add sign language-specific tokenization if applicable

3. **Data Quality**
   - Increase training data diversity
   - Add data augmentation specific to sign language videos
   - Ensure balanced coverage of sign language concepts

---

## Conclusion

**Yes, this is a training issue.** The low BLEU score despite loss reduction indicates:
- Model is learning wrong patterns (text generation without visual understanding)
- Training objective doesn't align with evaluation metric
- Vision-text alignment is broken or insufficiently trained

**Primary suspect**: Vision encoder is either frozen, not receiving gradients, or learning rate is too high/low.

**Action**: Start by verifying and fixing vision encoder training, then re-train with BLEU as validation metric.

