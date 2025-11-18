# Sign Language Video Masking 策略指南

基于 SSVP Hiera 的 Mask Unit (MU) masking 思想，我们为 QwenVL ViT 实现了多种 masking 策略，特别针对 Sign Language Video 的特点进行了优化。

## Sign Language Video 的特殊性

1. **空间结构重要**：
   - 手势需要理解手部形状（handshape）、位置（location）
   - 手部和上半身的空间关系很关键
   - 面部表情也包含信息（non-manual markers）

2. **时间连续性重要**：
   - 手势是动作序列，不是静态图像
   - 需要理解动作的时序和运动轨迹
   - 相邻帧之间有很强的相关性

3. **关键区域集中**：
   - 信息主要集中在：手部、手臂、面部、上半身
   - 背景信息相对不重要

## Masking 策略对比

### 1. `random` (Random Patch-level Masking)

**特点**：
- 每个patch独立随机mask
- 实现最简单
- ❌ 破坏空间局部性（手部patch可能被分散mask）
- ❌ 破坏时间连续性（相邻帧的同一位置可能不一致）

**适用场景**：
- 快速实验
- 与原始MAE论文一致

**使用示例**：
```bash
--mask_strategy random
--mask_ratio 0.85
```

### 2. `tube` (Time-tube Masking) ⭐ **推荐用于 Sign Language**

**特点**：
- Mask整个时间tube（所有帧的同一空间位置）
- ✅ 保持空间连续性（手势在空间上完整）
- ✅ 适合理解手部位置和形状
- ✅ 实现相对简单

**适用场景**：
- **Sign Language Video（默认推荐）**
- 手部位置和形状重要的任务

**使用示例**：
```bash
--mask_strategy tube
--mask_ratio 0.85
```

**可视化示例**：
```
Frame 0:  [V][M][V][M]    # V=visible, M=masked
Frame 1:  [V][M][V][M]    # 同一空间位置在所有帧保持相同
Frame 2:  [V][M][V][M]
Frame 3:  [V][M][V][M]
```

### 3. `block` (Spatial Block Masking)

**特点**：
- Mask空间块（如4×4 patches）跨所有帧
- ✅ 保持时间连续性（同一空间块在所有帧可见/隐藏）
- ✅ 适合理解动作序列和运动轨迹

**适用场景**：
- 需要理解时间动态的任务
- 动作序列分析

**使用示例**：
```bash
--mask_strategy block
--mask_unit_size 4 4  # (H, W) patch blocks
--mask_ratio 0.85
```

**可视化示例**：
```
Frame 0:  [MU][MU]    # MU = 4×4 patch block
          [__][MU]    # 整个块在所有帧保持相同
Frame 1:  [MU][MU]
          [__][MU]
Frame 2:  [MU][MU]
          [__][MU]
```

### 4. `mu` (Mask Unit Masking) ⭐⭐ **最佳选择，类似 Hiera**

**特点**：
- Mask 3D块（T×H×W patches，如1×4×4）
- ✅ 同时保持空间和时间局部性
- ✅ 最接近 SSVP Hiera 的实现
- ✅ 最适合 Sign Language（需要空间结构和时间序列）

**适用场景**：
- **Sign Language Video（最佳选择）**
- 需要同时理解空间结构和时间序列的任务

**使用示例**：
```bash
--mask_strategy mu
--mask_unit_size 4 4  # (H, W) = (4, 4) patches per MU
--mask_ratio 0.85
```

**可视化示例**：
```
Frame 0:  [MU][__]    # MU = 1×4×4 patch block
          [__][MU]    # 同时保持空间和时间局部性
Frame 1:  [MU][__]
          [__][MU]
```

## 推荐配置

### 对于 Sign Language Video：

**首选：`tube` 策略**（默认，已设置为默认值）
```bash
--mask_strategy tube
--mask_ratio 0.85
```

**备选：`mu` 策略**（更接近 Hiera，可能效果更好）
```bash
--mask_strategy mu
--mask_unit_size 4 4
--mask_ratio 0.85
```

### 对比实验建议：

1. **基线实验**（原始方法）：
   ```bash
   --mask_strategy random
   ```

2. **推荐实验**（默认）：
   ```bash
   --mask_strategy tube
   ```

3. **最佳实验**（类似 Hiera）：
   ```bash
   --mask_strategy mu
   --mask_unit_size 4 4
   ```

## 与 SSVP Hiera 的对比

| 特性 | SSVP Hiera | QwenVL MAE |
|------|------------|------------|
| Masking 粒度 | Mask Unit (8×8 patches) | 可配置：random/tube/block/mu |
| 局部性 | ✅ 保持（块状masking） | ✅ 保持（tube/block/mu策略） |
| 实现复杂度 | 较复杂（需要Unroll/Reroll） | 较简单（直接实现） |
| 灵活性 | 固定MU大小 | 可配置mask_unit_size |

**关键优势**：
- ✅ `tube` 和 `mu` 策略借鉴了 SSVP Hiera 的局部性思想
- ✅ 适配了 QwenVL ViT 的架构（无需 Unroll/Reroll）
- ✅ 更适合 Sign Language Video 的特点

## 实现细节

### Mask Unit Size 选择

- **4×4**：推荐用于大多数情况（平衡局部性和计算效率）
- **8×8**：更接近 Hiera（更强的局部性，但mask单位更少）
- **2×2**：更细粒度（更接近 random，但仍保持局部性）

### Mask Ratio 建议

- **0.75**：标准（25% visible）
- **0.85**：推荐用于 Sign Language（15% visible，更激进的masking）
- **0.90**：非常激进（10% visible）

## 使用方法

在训练脚本中（如 `finetune_qwen2vl_mae_2xa6000.sh`）：

```bash
deepspeed src/train/train_qwen_mae.py \
    --mask_strategy tube \
    --mask_unit_size 4 4 \
    --mask_ratio 0.85 \
    ...
```

或者在 Python 代码中：

```python
model = QwenViTMAE(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    mask_strategy='tube',  # or 'mu', 'block', 'random'
    mask_unit_size=(4, 4),  # For 'block' and 'mu' strategies
    mask_ratio=0.85,
    ...
)
```

## 预期效果

基于 Sign Language Video 的特点，我们预期：

1. **`tube` 策略**：在保持简单实现的同时，显著提升手部位置和形状的理解
2. **`mu` 策略**：最接近 Hiera，预期效果最好，同时保持空间结构和时间序列
3. **`block` 策略**：适合理解动作序列和运动轨迹
4. **`random` 策略**：作为基线对比

建议先尝试 `tube` 策略（已设置为默认），然后对比 `mu` 策略的效果。
