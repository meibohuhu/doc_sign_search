# InternVL GRPO Training Notes

## Overview

GRPO (Group Relative Policy Optimization) training for InternVL2.5-1B sign language translation.
Starting from SFT checkpoint, using rule-based rewards (BLEU + ROUGE) to further improve translation quality.

---

## Architecture

```
SFT Checkpoint (checkpoint-2548, BLEU4=0.078)
    │
    ├── InternVisionModel (ViT) — fully unfrozen (freeze_backbone=False)
    ├── MLP Projector — frozen (freeze_mlp=True)
    └── Qwen2ForCausalLM (LLM) — frozen base + LoRA rank 16 trainable
```

**Trainable parameters**: ~312M / 946M (33%)
- ViT: ~300M (full params)
- LLM LoRA: ~8.8M

---

## Key Implementation Details

### 1. LoRA Loading from SFT Checkpoint

The SFT checkpoint already contains trained LoRA weights (PeftModel format).
`internvl_grpo_train.py` detects this and **skips `wrap_llm_lora()`** to avoid
overwriting SFT-learned weights with random LoRA initialization.

```python
if not _has_peft(model.language_model): ##### 本来是在这里错了，然后policy rollout的效果都很差
    model.wrap_llm_lora(...)       # new LoRA (only if not already present)
else:
    model.language_model.enable_input_require_grads()  # reuse existing LoRA
```

### 2. Prompt Format Must Match SFT

SFT training uses: `Frame-1: <img><IMG_CONTEXT>×64</img>\nFrame-2: ...`

The GRPO trainer's `_format_prompt()` must replicate this exactly:
```python
for i in range(num_patches):
    img_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token + IMG_END_TOKEN
    frame_tokens.append(f'Frame-{i + 1}: {img_tokens}')
```

Missing the `Frame-{i+1}:` prefix causes garbage generation.

### 3. model.generate() Returns Only New Tokens

When `model.generate()` uses `inputs_embeds` internally (as InternVL does),
it returns **only generated tokens**, not the full sequence including prompt.

```python
if gen_output.size(1) > prompt_length:
    comp_ids = gen_output[:, prompt_length:]  # standard case
else:
    comp_ids = gen_output  # InternVL case: only new tokens returned
```

### 4. No Cross-GPU Gather for Rewards

Each GPU processes `per_device_batch_size` unique prompts and generates
`num_generations` completions per prompt locally. Rewards and advantages
are computed locally per GPU — no cross-GPU gather needed since each GPU
has complete groups for advantage normalization.

### 5. DeepSpeed Stage 0 + gloo Backend (Blackwell GPUs)

- Blackwell RTX PRO 6000 requires PyTorch 2.7.0+cu128
- Use `gloo` backend (not `nccl`) for distributed training on Blackwell
- DeepSpeed Stage 0 (no sharding) to avoid custom kernel issues
- Set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`

---

## Training Configuration

### Current Settings (first run)

| Parameter | Value | Notes |
|-----------|-------|-------|
| GPUs | 4 × RTX PRO 6000 | Blackwell |
| per_device_batch_size | 1 | 1 unique prompt per GPU per micro-step |
| grad_accum_steps | 4 | |
| num_generations | 4 | completions per prompt |
| **unique prompts per update** | **16** | 4 GPUs × 1 × 4 grad_accum |
| **total completions per update** | **64** | 16 prompts × 4 generations |
| learning_rate | 1e-6 | consistent with VRAG repo |
| temperature | 0.7 | consider 1.0 for more diversity |
| beta (KL) | 0.0 | no KL penalty |
| num_iterations | 1 | single pass per batch (standard for GRPO) |
| max_completion_length | 128 | max new tokens per generation |
| max_num_frame | 130 | video frames |
| min_num_frame | 32 | video frames |
| loss_type | grpo | clipped surrogate loss |
| num_train_epochs | 1 | |

### Batch Size Formula (Custom Trainer)

```
unique_prompts_per_update = num_gpus × per_device_batch_size × grad_accum_steps
total_completions_per_update = unique_prompts × num_generations
```

Note: This differs from TRL's standard GRPOTrainer which uses RepeatSampler.
Our custom trainer generates completions in a loop inside `_generate_and_score_completions()`.

---

## Reward Functions

### Available Rewards (`reward_functions.py`)

| Function | Metric | Speed | Distribution | Notes |
|----------|--------|-------|-------------|-------|
| `bleu_reward` | BLEU-4 | Fast | **59% near 0** | Too sparse for short sentences |
| `bleu1_reward` | BLEU-1 | Fast | 0.1-0.5 range | Stable, unigram precision |
| `rouge_reward` | ROUGE-L F1 | Fast | Uniform | LCS-based, word order aware |
| `bertscore_reward` | BERTScore F1 | **Slow (CPU)** | 0.5-0.9 range | Semantic similarity |

### Recommended Combinations

| Combo | Weights | Pros | Cons |
|-------|---------|------|------|
| BLEU1 + ROUGE-L | 0.5, 0.5 | Fast, complementary | No semantic signal |
| BLEU1 + ROUGE-L + BLEU4 | 0.4, 0.4, 0.2 | Covers all eval metrics | BLEU4 still sparse |
| BERTScore + BLEU1 | 0.6, 0.4 | Best signal quality | Slow (~5-10s per step) |

### Why BLEU4 is Problematic as Primary Reward

- SFT checkpoint has corpus-level BLEU4 = 0.078
- Sentence-level BLEU4 is even lower; 59% of samples score < 0.05
- 34% of groups have tied BLEU4 values (usually all zeros)
- Zero reward → zero advantage → wasted gradient step

### Reward Analysis from First Run (BLEU4 + BLEU1, weights 0.5/0.5)

```
Within-group diversity:
  bleu4 std: mean=0.0566 (low — poor discrimination)
  bleu1 std: mean=0.0923 (better)
  advantage range: mean=2.0886 (acceptable)

Groups with tied values:
  bleu4: 34.3% have ties
  bleu1: 29.2% have ties
```

---

## Training Observations

### First Run Results (187 steps, epoch 0.34)

```
Reward trend (BLEU4 + BLEU1):
  first 1/3:  reward=0.189, bleu4=0.097
  middle 1/3: reward=0.213, bleu4=0.115
  last 1/3:   reward=0.215, bleu4=0.116
```

- **Loss ~1e-8**: Normal for GRPO with num_iterations=1 (ratio always ~1.0)
- **clip_ratio = 0**: Expected when num_iterations=1 (old_logps = new_logps)
- **grad_norm ~4.0**: Healthy, gradients are flowing
- **Reward slowly increasing**: Marginal improvement, mostly plateaued

### Why clip_ratio = 0 is Normal

With `num_iterations=1`, `old_per_token_logps = per_token_logps.detach()`,
so the ratio `exp(logp_new - logp_old) = exp(0) = 1.0` always.
Clipping never triggers. This is standard GRPO behavior (same as VRAG repo).

### Advantage Verification

- `advantage = (reward - group_mean) / (group_std + 1e-4)` — verified correct
- Sum of advantages within each group = 0.0000 — perfect zero-mean
- Combined reward = 0.5 * bleu4 + 0.5 * bleu1 — verified matches logged values

---

## Hyperparameter Tuning Suggestions

### Priority 1: Change Reward Functions
- Replace BLEU4 as primary reward (too sparse)
- Use BLEU1 + ROUGE-L (0.5, 0.5) or BLEU1 + ROUGE-L + BLEU4 (0.4, 0.4, 0.2)

### Priority 2: Increase Temperature
- Current: 0.7 (conservative, low diversity)
- Suggested: 1.0 (matches VRAG repo, increases reward variance within groups)

### Priority 3: Increase Global Batch Size
- Increase grad_accum_steps from 4 to 8 (32 unique prompts, 128 completions)
- Prefer increasing grad_accum over num_generations (no speed penalty)

### Priority 4 (Optional): Add KL Penalty
- beta=0.01 (matches VRAG) to prevent policy from diverging too far from SFT

### Priority 5 (Future): Difficulty-Aware Sampling
- Run SFT checkpoint on training data, score each sample
- Sample by difficulty: 20% hard, 60% medium, 20% easy
- "Medium" samples (0.05 < BLEU4 < 0.3) are the sweet spot for GRPO

---

## File Structure

```
InternVL/internvl_chat/internvl/train/grpo/
├── internvl_grpo_train.py    # Main training script (model loading, LoRA, dataset, trainer init)
├── trainer_grpo.py           # Custom GRPO trainer (generation, rewards, loss, logging)
├── grpo_dataset.py           # Video dataset (frame loading, transforms)
├── reward_functions.py       # BLEU, ROUGE, BERTScore reward functions
└── GRPO_NOTES.md             # This file

script_adobe/0310/
├── grpo_internvl2_5_how2sign_1b_blackwell.sh    # Training launch script
├── setup_internvl_grpo_env.sh                    # Env setup (Blackwell)
└── setup_internvl_grpo_env_regular.sh            # Env setup (A100/V100/H100)
```

---

## Reference: VRAG Repo Hyperparameters

From https://github.com/Alibaba-NLP/VRAG (Qwen2.5-VL-7B GRPO):

| Parameter | Value |
|-----------|-------|
| learning_rate | 1e-6 |
| ppo_epochs (num_iterations) | 1 |
| num_generations (n_agent) | 5 |
| temperature | 1.0 |
| train_batch_size | 32 |
| kl_loss_coef (beta) | 0.01 |
