# MAE Vision Encoder Checkpoint 使用指南

## 概述

本文档说明如何从 MAE (Masked Autoencoder) checkpoint 中提取 vision encoder 权重，并创建可用于标准 InternVL 训练的 checkpoint。

## 两个 Checkpoint 目录的区别

### `checkpoint-30000_vision_encoder`（中间产物）

**用途**: 中间产物，只包含 vision encoder 权重

**内容**:
- 340 个 vision encoder 权重（`vision_model.*`）
- 从 MAE checkpoint 中提取的纯权重
- 格式：`vision_encoder_weights.safetensors` / `.pth`

**特点**:
- 文件大小：~580MB
- 只包含 vision encoder，不包含完整模型
- **不能直接用于训练**
- 需要进一步处理才能使用

**生成方式**:
```bash
python3 InternVL/scripts/local/extract_vision_encoder_from_mae.py \
    --mae-checkpoint InternVL/checkpoints/internvl2_5_2B_2xa100_mae/checkpoint-30000 \
    --output InternVL/checkpoints/internvl2_5_2B_2xa100_mae/checkpoint-30000_vision_encoder
```

**使用场景**:
- 仅用于提取和查看 vision encoder 权重
- 需要进一步处理才能用于训练
- 适合：权重分析、调试、自定义加载方式

---

### `checkpoint-30000_vision_only`（完整 Checkpoint）

**用途**: 完整的训练 checkpoint，可直接用于训练

**内容**:
- 340 个 vision encoder 权重（来自 MAE）
- 171 个 language model 权重（来自 base model）
- 102 个 MLP 权重（来自 base model）
- 总计：517 个权重

**特点**:
- 文件大小：~4.2GB
- 包含完整的 InternVL 模型结构
- Vision encoder 使用 MAE 权重初始化
- **可以直接用于训练**（使用 `--resume_from_checkpoint`）
- 训练从 step 0 开始（不是从 step 30000 继续）

**生成方式**:
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate internvl

python3 InternVL/scripts/local/load_vision_encoder_to_checkpoint.py \
    --vision-encoder-weights InternVL/checkpoints/internvl2_5_2B_2xa100_mae/checkpoint-30000_vision_encoder/vision_encoder_weights.safetensors \
    --base-model OpenGVLab/InternVL2_5-2B \
    --output InternVL/checkpoints/internvl2_5_2B_2xa100_mae/checkpoint-30000_vision_only \
    --trainer-state InternVL/checkpoints/internvl2_5_2B_2xa100_mae/checkpoint-30000/trainer_state.json
```

**使用场景**:
- 直接用于训练（推荐）
- 使用 `--resume_from_checkpoint` 参数
- 适合：使用 MAE vision encoder 权重初始化模型并开始训练

---

## 完整工作流程

```
1. MAE checkpoint-30000 (原始，包含 decoder 和 MAE 组件)
   ↓
   [extract_vision_encoder_from_mae.py]
   ↓
2. checkpoint-30000_vision_encoder (纯 vision encoder 权重，340 keys)
   ↓
   [load_vision_encoder_to_checkpoint.py]
   ↓
3. checkpoint-30000_vision_only (完整模型，517 keys)
   ↓
   [训练脚本使用 --resume_from_checkpoint]
   ↓
4. 开始训练（vision encoder 使用 MAE 权重初始化）
```

## 详细对比表

| 特性 | checkpoint-30000_vision_encoder | checkpoint-30000_vision_only |
|------|--------------------------------|------------------------------|
| **文件大小** | ~580MB | ~4.2GB |
| **权重数量** | 340 keys | 517 keys |
| **包含组件** | 仅 vision encoder | Vision + Language Model + MLP |
| **可直接训练** | ❌ 否 | ✅ 是 |
| **用途** | 中间产物 | 完整 checkpoint |
| **生成脚本** | `extract_vision_encoder_from_mae.py` | `load_vision_encoder_to_checkpoint.py` |

## 使用方法

### 方法 1：使用 Vision-Only Checkpoint（推荐）

```bash
# 脚本已默认设置为使用 vision-only checkpoint
bash InternVL/scripts/local/finetune_internvl2_5_how2sign_2xa6000_2rd_finetune_mae.sh
```

### 方法 2：手动指定 Checkpoint

```bash
RESUME_FROM_CHECKPOINT="/local1/mhu/sign_language_llm/InternVL/checkpoints/internvl2_5_2B_2xa100_mae/checkpoint-30000_vision_only" \
bash InternVL/scripts/local/finetune_internvl2_5_how2sign_2xa6000_2rd_finetune_mae.sh
```

### 方法 3：从头开始训练（不使用 MAE 权重）

```bash
RESUME_FROM_CHECKPOINT="" \
bash InternVL/scripts/local/finetune_internvl2_5_how2sign_2xa6000_2rd_finetune_mae.sh
```

## 重要说明

### ✅ 使用 Vision Encoder 权重时（推荐方式）

- Vision encoder 使用 MAE checkpoint 的权重初始化（通过 `--vision_encoder_weights_path` 参数）
- 训练从 step 0 开始（不是从 checkpoint 的 step 继续）
- 使用 ZeRO Stage 3 配置（匹配原始 MAE checkpoint）
- 不需要完整的 DeepSpeed checkpoint 结构
- 直接在模型初始化后加载权重，避免 DeepSpeed 兼容性问题

### ✅ 使用完整 Checkpoint Resume 时

- 使用 `--resume_from_checkpoint` 参数
- 会恢复训练状态（step, epoch, optimizer 等）
- 需要完整的 DeepSpeed checkpoint 结构（包括 `global_step*/` 目录）

### ⚠️ 注意事项

1. **训练状态**: 使用 `--vision_encoder_weights_path` 时，训练从 step 0 开始，不会继续之前的训练进度
2. **ZeRO Stage 兼容性**: 必须使用与原始 checkpoint 相同的 ZeRO stage（这里是 Stage 3）
3. **权重格式**: Vision encoder 权重已从 `visual.*` 转换为 `vision_model.*` 格式
4. **互斥性**: `VISION_ENCODER_WEIGHTS_PATH` 和 `RESUME_FROM_CHECKPOINT` 是互斥的，只能使用其中一个

## 文件结构

### checkpoint-30000_vision_encoder/
```
checkpoint-30000_vision_encoder/
├── vision_encoder_weights.safetensors  (580MB)
├── vision_encoder_weights.pth          (580MB)
└── trainer_state.json                   (1.8MB, 从原始 checkpoint 复制)
```

### checkpoint-30000_vision_only/
```
checkpoint-30000_vision_only/
├── model.safetensors                    (4.2GB, 完整模型)
├── pytorch_model.bin                    (4.2GB, 完整模型)
└── trainer_state.json                   (452B, 重置为 step 0)
```

## 相关脚本

1. **extract_vision_encoder_from_mae.py**
   - 从 MAE checkpoint 中提取 vision encoder 权重
   - 将 `visual.*` 转换为 `vision_model.*` 格式

2. **load_vision_encoder_to_checkpoint.py**
   - 将 vision encoder 权重合并到标准 InternVL 模型
   - 创建完整的训练 checkpoint

3. **finetune_internvl2_5_how2sign_2xa6000_2rd_finetune_mae.sh**
   - 训练脚本
   - 支持使用 `--vision_encoder_weights_path` 加载 vision encoder 权重（推荐）
   - 支持使用 `--resume_from_checkpoint` 恢复完整训练状态

## 当前配置

训练脚本默认使用：
- **Vision Encoder 权重**: `checkpoint-60_vision_encoder/vision_encoder_weights.safetensors`
- **来源**: `/local1/mhu/sign_language_llm/InternVL/output/how2sign/internvl2_5_2B_mae_2xa6000/checkpoints/checkpoint-60`

## 总结

- **checkpoint-*_vision_encoder**: 中间产物，仅包含 vision encoder 权重，通过 `--vision_encoder_weights_path` 使用（推荐）
- **checkpoint-*_vision_only**: 完整 checkpoint，包含完整模型结构，通过 `--resume_from_checkpoint` 使用（需要 DeepSpeed 结构）

**推荐使用 `--vision_encoder_weights_path` 方式**，它更简单，不需要完整的 DeepSpeed checkpoint 结构，直接在模型初始化后加载权重。

