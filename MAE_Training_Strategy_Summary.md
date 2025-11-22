# MAE (Masked Autoencoder) 训练策略总结

## 📋 核心设计理念

MAE是一种自监督预训练方法，通过masking大部分patches并让模型重建它们来学习视频表示。

### 核心设计原则

1. **Encoder: 只处理Visible Tokens** ⚡
   - 只对未被mask的patches进行编码，大幅节省计算量
   - 计算复杂度：O((1-ratio) × N)，其中ratio是mask比例（默认0.75）

2. **Decoder: 轻量级重建** 🔨
   - 轻量级的transformer decoder（默认8层，512维度）
   - 处理完整序列（visible tokens + mask tokens）
   - 负责重建被mask的patches

3. **Loss: 只在Masked Patches上计算** 🎯
   - 只对masked patches计算重建loss
   - 不计算visible patches的loss（它们已经"看到"了）

## 🎭 Masking策略

### 1. Random Masking (`mask_strategy='random'`)

**特点**: Patch级别的随机mask

- **方式**: 每个patch独立随机决定是否mask
- **优点**: 实现简单，计算高效
- **缺点**: 丢失空间和时间局部性
- **适用场景**: 通用预训练，快速迭代

**Mask模式**:
```
Frame 1: [V M V M M V ...]  ← 随机mask
Frame 2: [M V M V V M ...]  ← 随机mask
Frame 3: [V M M V M V ...]  ← 随机mask
```

### 2. Tube Masking (`mask_strategy='tube'`) ⭐ 推荐用于手语

**特点**: 时间tube masking，保持空间连续性

- **方式**: 同一个空间位置(H, W)在所有时间帧上同时mask/visible
- **优点**: 
  - 保持空间连续性（手势位置完整保留）
  - 适合手语识别（手势位置比时间变化更重要）
- **缺点**: 丢失时间连续性

**Mask模式**:
```
Frame 1: [V M V M M V ...]  ← 空间位置1 visible
Frame 2: [V M V M M V ...]  ← 空间位置1 visible（保持一致）
Frame 3: [V M V M M V ...]  ← 空间位置1 visible（保持一致）
        ↑     ↑  ↑ ↑        ← 同一个空间位置在所有帧上相同
```

### 3. Block Masking (`mask_strategy='block'`)

**特点**: 空间块masking，保持时间连续性

- **方式**: 在空间维度上mask连续的块（如4×4 patches），该块在所有时间帧上保持一致
- **配置**: `mask_unit_size=(H, W)` 指定块大小，如 `(4, 4)`
- **优点**:
  - 保持时间连续性（运动轨迹完整）
  - 适合理解时间运动
- **缺点**: 可能丢失空间细节

**Mask模式** (假设block_size=2×2):
```
Frame 1: [V V M M V V ...]  ← 2×2块masked
Frame 2: [V V M M V V ...]  ← 相同块masked
Frame 3: [V V M M V V ...]  ← 相同块masked
        └─┘ └─┘           ← 空间块在所有帧上保持一致
```

### 4. MU (Mask Unit) Masking (`mask_strategy='mu'`) ⭐⭐ 最适合手语

**特点**: 3D块masking，保持空间和时间局部性

- **方式**: Mask 3D块 (T × H × W)，类似Hiera架构
- **配置**: 
  - `mask_unit_size=(H, W)` 指定空间块大小，如 `(4, 4)`
  - 时间维度：固定为2帧（`block_t=2`）
- **优点**:
  - ✅ **最佳选择用于手语**：同时保持空间结构（手势形状）和时间运动（手势变化）
  - 最大程度保留局部性
- **缺点**: 实现较复杂

**Mask模式** (假设MU=2×2×2):
```
Frame 1: [V V M M V V ...]  ← 2×2空间块
Frame 2: [V V M M V V ...]  ← 相同2×2空间块
Frame 3: [V V V V V V ...]  ← 不同MU
Frame 4: [V V V V V V ...]  ← 不同MU
        └─┘ └─┘           ← 2×2×2的3D块同时mask/visible
```

## 🏗️ 架构详解

### Encoder (Qwen2-VL Vision Encoder)

**完整流程**:
1. **Patch Embedding**: 将visible patches转换为embeddings
2. **Rotary Position Embedding**: 添加位置信息（支持时空位置编码）
3. **Window Reorganization**: 按窗口重组tokens（优化attention计算）
4. **Encoder Blocks**: 通过完整的vision encoder transformer blocks
   - 支持window attention和full attention
   - 每层包含self-attention和MLP
   - 使用rotary position embedding
5. **跳过Merger**: MAE不使用merger（保持encoder原始维度）

**关键优化**:
- 只对visible tokens进行编码（减少75%计算量）
- 保持完整的encoder结构（所有32层blocks）
- 支持gradient checkpointing节省内存

### Decoder (轻量级Transformer)

**结构**:
- `decoder_embed`: 投影层 (encoder_dim → decoder_dim, 默认512)
- `mask_token`: 可学习的mask token参数
- `decoder_blocks`: Transformer blocks（默认8层）
  - Self-attention
  - MLP (mlp_ratio=4.0)
  - Layer normalization
- `decoder_norm`: 最终layer norm
- `decoder_pred`: 预测head (decoder_dim → patch_pixel_dim)

**输入**:
- Visible tokens: 经过encoder编码的visible patches
- Mask tokens: 可学习的mask token（填充到masked位置）

**输出**:
- 重建的patches (所有patches，包括visible和masked)

## 💡 Loss计算

### 重建Loss (MSE)

```python
loss = MSE(pred_masked, target_masked)
```

**关键特性**:
1. **只在Masked Patches上计算**
   ```python
   loss = (loss_per_patch * mask).sum() / mask.sum()
   ```

2. **可选的Pixel Normalization** (`norm_pix_loss=True`)
   - 对target patches进行per-patch归一化
   - 移除patch级别的亮度和对比度变化
   - 让模型专注于学习结构信息
   ```python
   mean = target.mean(dim=-1, keepdim=True)  # Per-patch mean
   var = target.var(dim=-1, keepdim=True)    # Per-patch variance
   target = (target - mean) / (var + 1e-6)**0.5
   ```

## 🔄 完整训练流程

```
输入: [N, patch_pixel_dim] flattened patches
      [num_videos, 3] video_grid_thw (T, H, W)

1. Masking阶段
   ├─ 根据mask_strategy生成mask
   ├─ ids_keep: visible patches的索引（shuffled顺序）
   ├─ ids_restore: 从shuffled恢复到original顺序的映射
   └─ mask: binary mask (1=masked, 0=visible)

2. Encoder阶段
   ├─ 提取visible patches: patches[ids_keep]
   ├─ Patch embedding: visible_embeds
   ├─ 通过完整encoder blocks（保持空间结构）
   └─ 输出: latent [num_visible, encoder_dim]

3. Decoder阶段（按视频分别处理）
   ├─ 投影: latent → decoder_dim
   ├─ 构建完整序列: [visible_tokens + mask_tokens]
   ├─ 通过decoder blocks
   ├─ 预测: pred [num_patches, patch_pixel_dim]
   └─ 恢复到原始顺序

4. Loss计算
   ├─ 只在masked patches上计算MSE
   └─ 可选：对target进行per-patch归一化

输出: loss (scalar), pred [N, patch_pixel_dim], mask [N]
```

## ⚙️ 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mask_ratio` | `0.75` | Mask比例（75%的patches被mask） |
| `mask_strategy` | `'tube'` | Masking策略：`'random'`, `'tube'`, `'block'`, `'mu'` |
| `mask_unit_size` | `(4, 4)` | 用于`'block'`和`'mu'`策略的空间块大小 |
| `decoder_embed_dim` | `512` | Decoder的embedding维度 |
| `decoder_depth` | `8` | Decoder transformer blocks的层数 |
| `decoder_num_heads` | `16` | Decoder attention heads数量 |
| `mlp_ratio` | `4.0` | MLP hidden维度 = embed_dim × mlp_ratio |
| `norm_pix_loss` | `True` | 是否对target patches进行归一化 |

## 🎯 策略选择建议

### 手语视频预训练 ⭐

**推荐策略**: `mask_strategy='mu'` (Mask Unit)

**原因**:
- 手语需要同时理解：
  - **空间结构**：手势的形状和位置
  - **时间运动**：手势的变化轨迹
- MU masking同时保持两种局部性，最适合手语

**配置示例**:
```python
mask_strategy='mu'
mask_unit_size=(4, 4)  # 4×4空间块
mask_ratio=0.75
```

### 通用视频预训练

**推荐策略**: `mask_strategy='tube'`

**原因**:
- 空间连续性通常比时间连续性更重要
- 计算效率高（比MU简单）

### 快速实验/调试

**推荐策略**: `mask_strategy='random'`

**原因**:
- 实现最简单
- 训练最快
- 适合快速验证模型

## 🚀 优化技巧

### 1. 内存优化

- **Gradient Checkpointing**: 启用decoder的gradient checkpointing
  ```python
  model.gradient_checkpointing_enable()
  ```

- **Encoder只处理Visible Tokens**: 自动减少75%计算量

### 2. 训练效率

- **动态patch_pixel_dim检测**: 自动从输入维度推断，支持不同的视频配置
- **Batch处理**: 所有视频一起处理encoder，分别处理decoder（处理不同数量的visible tokens）

### 3. 分布式训练

- 支持DeepSpeed ZeRO-3
- 自动检测分布式环境，只在rank 0打印日志

## 📊 预期效果

### Mask Ratio = 0.75 时的计算节省

- **Encoder计算量**: 减少75% (只处理25%的tokens)
- **Decoder计算量**: 100% (需要重建所有tokens)
- **总体加速**: 约40-50% (取决于encoder/decoder比例)

### 训练目标

通过重建masked patches，模型学习：
- 空间结构理解（物体形状、手势形状）
- 时间运动理解（物体运动、手势变化）
- 上下文推理（从visible patches推断masked patches）

## 🔍 实现细节

### ID索引系统

MAE使用复杂的ID索引系统来管理shuffled和original顺序：

1. **ids_shuffle**: 随机打乱的顺序 `[rand_idx0, rand_idx1, ...]`
2. **ids_restore**: 恢复映射 `ids_restore[shuffled_pos] = original_pos`
3. **ids_keep**: visible tokens在shuffled顺序中的位置 `[0, 1, ..., num_visible-1]`

**关系**:
- `visible_positions_original = ids_restore[ids_keep]`
- Decoder需要恢复完整序列的顺序

### Encoder的完整序列处理

虽然encoder只处理visible tokens，但为了正确计算rotary position embedding和window index，代码构建了完整的序列（masked位置填零），然后从输出中提取visible tokens。这确保了位置编码的正确性。




