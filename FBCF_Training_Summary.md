# FBCF (Foreground-Background Consistency Finetuning) 训练总结

## 📋 概述

FBCF是一种针对视频语言模型的训练方法，通过同时利用原始视频、前景视频（只有手势区域）和背景视频（只有背景区域）来增强模型的表示学习能力。

## 🎯 核心思想

1. **原始视频 (Original)**: 包含完整的手势和背景信息，使用标准Cross-Entropy loss
2. **前景视频 (Foreground)**: 只保留手势区域，鼓励模型关注关键的手势信息，使用加权CE loss
3. **背景视频 (Background)**: 只保留背景区域，鼓励模型对背景输出均匀分布（不产生有意义的预测），使用KL散度loss

## 🔧 训练模式

### 模式1: Full Forward Mode (完整前向模式)
- **启用条件**: `fbcf_sampling_mode=False` (默认)
- **特点**: 每个训练step中，对每个样本同时进行三种view的前向传播
- **内存消耗**: 较高（需要3倍前向传播）
- **Loss计算**:
  ```
  total_loss = loss_full + fg_loss_weight * loss_fg + fbcf_lambda * bg_loss_weight * loss_bg
  ```

### 模式2: Sampling Mode (采样模式)
- **启用条件**: `fbcf_sampling_mode=True`
- **特点**: 每个训练step中，随机选择一种view类型，所有样本使用相同的view类型
- **内存消耗**: 较低（只需要1倍前向传播）
- **数据多样性增强**: 
  - ✅ **增加数据多样性**: 同一个视频样本在不同step中会以不同的view（original/foreground/background）出现
  - ✅ **数据增强效果**: 相当于对每个视频进行了3种不同的数据增强（完整视图、前景视图、背景视图）
  - ✅ **跨epoch多样性**: 由于随机采样，同一个样本在不同epoch中可能以不同的view类型训练
  - ✅ **训练效率**: 相比Full Forward Mode，在相同训练步数下，模型会看到更多样化的数据组合
- **采样比例** (默认):
  - Original: 40% (`fbcf_sampling_ratio_original=0.4`)
  - Foreground: 40% (`fbcf_sampling_ratio_foreground=0.4`)
  - Background: 20% (`fbcf_sampling_ratio_background=0.2`)
- **Loss计算**: 根据选中的view类型，只计算对应的loss

## 📊 Loss 详细说明

### 1. Loss_full (原始视频的CE Loss)
- **计算方式**: 标准Cross-Entropy Loss
- **公式**: `loss_full = CE(pred, labels)`
- **权重**: 1.0 (固定)
- **目的**: 保持模型在完整视频上的正常预测能力

### 2. Loss_fg (前景视频的CE Loss)
- **计算方式**: 标准Cross-Entropy Loss
- **公式**: `loss_fg = CE(pred_fg, labels)`
- **权重**: `fg_loss_weight` (默认=1.0)
- **目的**: 强化模型对前景（手势）区域的关注和学习

### 3. Loss_bg (背景视频的KL散度Loss)
- **计算方式**: KL散度，衡量模型输出分布与均匀分布的差异
- **公式**: 
  ```
  KL(p||u) = Σ p_i * log(p_i / u_i)
           = Σ p_i * log(p_i) + log(vocab_size)
  ```
  其中 `u_i = 1/vocab_size` 是均匀分布
- **权重**: `fbcf_lambda * bg_loss_weight` (默认: 0.2 * 1.0 = 0.2)
- **目的**: 鼓励模型在面对背景区域时输出接近均匀分布（表示"不知道"或"不相关"）

### 完整Loss公式

#### Full Forward Mode:
```python
total_loss = loss_full + fg_loss_weight * loss_fg + fbcf_lambda * bg_loss_weight * loss_bg
```

#### Sampling Mode:
- **Original view**: `total_loss = loss_full`
- **Foreground view**: `total_loss = fg_loss_weight * loss_fg`
- **Background view**: `total_loss = fbcf_lambda * bg_loss_weight * loss_bg`

## ⚙️ 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_fbcf` | `False` | 是否启用FBCF训练 |
| `fbcf_lambda` | `0.2` | 背景KL正则化的权重系数 |
| `fg_loss_weight` | `1.0` | 前景CE loss的权重 |
| `bg_loss_weight` | `1.0` | 背景KL loss的额外乘数（在lambda之前应用） |
| `fbcf_sampling_mode` | `False` | 是否使用采样模式（单路径） |
| `fbcf_sampling_ratio_original` | `0.4` | 原始视频的采样比例 |
| `fbcf_sampling_ratio_foreground` | `0.4` | 前景视频的采样比例 |
| `fbcf_sampling_ratio_background` | `0.2` | 背景视频的采样比例 |

## 🔍 实现细节

### 数据准备
- 需要提供mask文件来分离前景和背景
- Mask路径通过 `mask_folder` 参数指定
- Mask文件格式: `.npz` (numpy压缩格式)
- 支持mask dilation和blur处理

### 分布式训练兼容性
- **Sampling Mode**: 使用step-based确定性随机选择
  - 所有进程在同一个step使用相同的随机种子
  - 确保所有进程选择相同的view类型，避免梯度同步问题
  - 代码位置: `sft_trainer.py:354-360`

### 内存优化

#### 1. Background KL Loss 优化
- **实现**: 使用内存优化的实现
  - 立即对vocab维度求和，减少中间tensor的内存占用
  - 代码位置: `sft_trainer.py:222-262`
- **DeepSpeed兼容**: 自动检测是否使用DeepSpeed，避免不必要的cache清理

#### 2. Lazy Loading 策略（Sampling Mode）
- **问题**: 原先在 DataLoader 中预计算所有 fg_pixels 和 bg_pixels，导致：
  - DataLoader workers 内存压力过大（OOM）
  - Workers 频繁崩溃和重启
  - 训练速度逐渐变慢（内存累积）
  
- **解决方案**: 在 Sampling Mode 下实现 Lazy Loading
  - **数据加载阶段** (`sft_dataset.py`):
    - 只存储 `video_mask_paths`（mask文件路径）
    - **不预计算** `fg_pixels` 和 `bg_pixels`
    - 只加载原始 `video_pixels`
    - 代码位置: `sft_dataset.py:244-249`
  
  - **训练阶段** (`sft_trainer.py`):
    - 在 `compute_loss` 中根据随机选择的 `view_type` 按需计算
    - 使用 `_compute_pixels_on_demand` 方法动态计算需要的 pixels
    - 只计算当前 step 需要的 view（original/foreground/background）
    - 计算完成后立即释放临时变量（`del mask_tensor`, `del inv_mask` 等）
    - 代码位置: `sft_trainer.py:271-332`, `sft_trainer.py:450-520`
  
- **优势**:
  - ✅ **大幅减少内存占用**: 不再同时存储3种view的pixels
  - ✅ **避免DataLoader崩溃**: Workers不再因为内存压力而OOM
  - ✅ **训练稳定性提升**: 减少workers重启，训练速度更稳定
  - ✅ **按需计算**: 只计算当前需要的view，节省计算资源
  
- **权衡**:
  - ⚠️ **轻微计算开销**: 每次forward需要动态计算mask和pixels
  - ⚠️ **I/O开销**: 需要从磁盘加载mask文件（但通常mask文件较小）

#### 3. DataLoader 参数优化
- **dataloader_num_workers**: `4 → 2` (减少50%的worker进程)
- **dataloader_prefetch_factor**: `4 → 2` (减少预取数据量)
- **dataloader_persistent_workers**: `False` (workers在每个epoch后重启，定期清理内存)
- **目的**: 进一步减少内存压力，提高训练稳定性

## 📝 训练流程

### Full Forward Mode 流程

1. **数据加载**:
   - 加载原始视频帧
   - 从mask文件夹加载对应的mask文件
   - **预计算**前景视频 (应用mask)
   - **预计算**背景视频 (应用反mask)
   - 同时存储 `video_pixels`, `fg_pixels`, `bg_pixels`

2. **训练循环**:
   - 对三种view都进行前向传播
   - 计算各自的loss并加权求和
   - `total_loss = loss_full + fg_loss_weight * loss_fg + fbcf_lambda * bg_loss_weight * loss_bg`

3. **Loss反向传播**:
   - 计算total_loss
   - 反向传播更新模型参数

### Sampling Mode 流程（当前实现 - Lazy Loading）

1. **数据加载** (`sft_dataset.py`):
   - 加载原始视频帧 → `video_pixels`
   - 从mask文件夹获取mask文件路径 → `video_mask_paths`
   - **不预计算**前景和背景pixels（Lazy Loading）
   - 只存储mask路径，节省内存

2. **训练循环** (`sft_trainer.py`):
   - **Step 1**: 随机选择view类型（基于step的确定性随机种子）
     - 使用 `torch.Generator` 确保分布式训练同步
     - 根据配置的比例选择：Original / Foreground / Background
   
   - **Step 2**: 按需计算pixels（如果需要）
     - **Original view**: 直接使用 `video_pixels`
     - **Foreground view**: 调用 `_compute_pixels_on_demand(..., view_type=1)`
       - 加载mask文件
       - 构建mask tensor
       - 计算 `fg_pixels = video_pixels * mask_tensor`
       - 立即释放临时变量
     - **Background view**: 调用 `_compute_pixels_on_demand(..., view_type=2)`
       - 加载mask文件
       - 构建mask tensor
       - 计算 `bg_pixels = video_pixels * (1 - mask_tensor)`
       - 可选：添加背景噪声
       - 立即释放临时变量
   
   - **Step 3**: 前向传播和Loss计算
     - 使用计算得到的pixels进行前向传播
     - 根据view类型计算对应的loss：
       - Original: `total_loss = loss_full`
       - Foreground: `total_loss = fg_loss_weight * loss_fg`
       - Background: `total_loss = fbcf_lambda * bg_loss_weight * loss_bg`

3. **Loss反向传播**:
   - 计算total_loss
   - 反向传播更新模型参数
   - 释放计算过程中的临时变量

### 关键改进点

- **内存效率**: 不再同时存储3种view的pixels，只在需要时计算
- **稳定性**: 避免DataLoader workers因内存压力而崩溃
- **灵活性**: 根据实际需要的view类型动态计算，节省资源

## 🔄 数据多样性分析

### Sampling Mode 的数据多样性优势

**为什么Sampling Mode能增加数据多样性？**

1. **同一视频的多种视图**:
   - 每个视频样本可以以3种不同的view出现：Original、Foreground、Background
   - 在训练过程中，同一个视频会在不同的step中以不同的view被训练
   - 这相当于将数据集大小**隐式扩大3倍**（虽然每次只使用一种view）

2. **跨epoch的多样性**:
   - 由于随机采样，同一个样本在不同epoch中可能以不同的view类型出现
   - 例如：epoch 1中某个视频可能以Original view训练，epoch 2中可能以Foreground view训练
   - 这提供了**时间维度的数据增强**

3. **与Full Forward Mode的对比**:
   - **Full Forward Mode**: 每个step中，每个样本都以3种view同时训练
     - 优点：每个样本在每一步都看到所有view
     - 缺点：内存消耗高（3倍），训练速度慢
   - **Sampling Mode**: 每个step中，每个样本只以一种view训练
     - 优点：内存消耗低，训练速度快，数据多样性更高（跨step/epoch）
     - 缺点：每个step只看到一种view

4. **实际效果**:
   - 假设训练1000个steps，使用Sampling Mode (40/40/20比例):
     - 每个样本大约400次以Original view训练
     - 每个样本大约400次以Foreground view训练
     - 每个样本大约200次以Background view训练
   - 这比Full Forward Mode（每个样本1000次同时看到3种view）提供了**更多的训练组合多样性**

### 数据多样性总结

| 特性 | Full Forward Mode | Sampling Mode |
|------|------------------|---------------|
| 每个step的view数量 | 3种（同时） | 1种（随机） |
| 内存消耗 | 高（3倍） | 低（1倍） |
| 训练速度 | 慢 | 快 |
| 数据多样性（跨step） | 低（固定组合） | **高（随机组合）** |
| 数据多样性（跨epoch） | 低（固定） | **高（随机）** |
| 每个view的训练频率 | 100% | 按比例（如40/40/20） |

## 🎯 预期效果

- **提高手势识别准确性**: 通过前景训练，模型更关注手势区域
- **减少背景干扰**: 通过背景KL loss，模型学会忽略无关的背景信息
- **更好的泛化能力**: 模型能够区分有意义的信号（手势）和无意义的噪声（背景）

## 🎯 针对Evaluation的训练比例建议

### 重要发现
- **Evaluation时使用的是原始视频（Original）**，没有mask或前景/背景分离
- 因此，训练时应该**主要训练原始视频**，确保模型在原始视频上表现最好
- 前景和背景视频作为**辅助训练**，帮助模型学习区分手势和背景

### 推荐比例（Sampling Mode）

#### 方案1: 保守方案（推荐用于直接evaluation）
```bash
--fbcf_sampling_ratio_original 0.60 \
--fbcf_sampling_ratio_foreground 0.30 \
--fbcf_sampling_ratio_background 0.10 \
```
- **Original: 60%** - 确保模型在原始视频上表现良好
- **Foreground: 30%** - 帮助模型关注手势区域
- **Background: 10%** - 帮助模型忽略背景干扰

#### 方案2: 平衡方案
```bash
--fbcf_sampling_ratio_original 0.50 \
--fbcf_sampling_ratio_foreground 0.35 \
--fbcf_sampling_ratio_background 0.15 \
```
- **Original: 50%** - 保持原始视频的训练
- **Foreground: 35%** - 强化手势关注
- **Background: 15%** - 适度的背景抑制

#### 方案3: 激进方案（更强调前景学习）
```bash
--fbcf_sampling_ratio_original 0.40 \
--fbcf_sampling_ratio_foreground 0.50 \
--fbcf_sampling_ratio_background 0.10 \
```
- **Original: 40%** - 基础训练
- **Foreground: 50%** - 大量前景训练，强化手势识别
- **Background: 10%** - 最小背景训练

### 当前使用的比例
根据代码库中的训练脚本，当前主要使用：
- **Original: 40%**, **Foreground: 40%**, **Background: 20%** (默认值，推荐用于训练)
- **Original: 20%**, **Foreground: 60%**, **Background: 20%** (在部分脚本中，更强调前景学习)

**当前训练脚本配置** (`finetune_qwen2vl_how2sign_2xa6000_fbcf.sh`):
```bash
--fbcf_sampling_mode True \
--fbcf_sampling_ratio_original 0.40 \
--fbcf_sampling_ratio_foreground 0.40 \
--fbcf_sampling_ratio_background 0.20 \
--fbcf_lambda 0.15 \
--fg_loss_weight 1.0 \
--bg_loss_weight 1.0 \
--fbcf_bg_noise_std 0.05 \
--dataloader_num_workers 2 \
--dataloader_prefetch_factor 2 \
--dataloader_persistent_workers False
```

⚠️ **注意**: 如果直接用于evaluation，建议使用**方案1（60/30/10）**，因为evaluation时使用的是原始视频。

### Full Forward Mode
如果使用Full Forward Mode（`fbcf_sampling_mode=False`），所有三种view都会在每个step中训练，比例由loss权重控制：
- `loss_full` 权重: 1.0 (固定)
- `loss_fg` 权重: `fg_loss_weight` (默认=1.0)
- `loss_bg` 权重: `fbcf_lambda * bg_loss_weight` (默认=0.2)

可以通过调整 `fg_loss_weight` 和 `fbcf_lambda` 来控制不同view的相对重要性。

## 📌 注意事项

1. **Mask质量**: Mask的准确性和覆盖范围直接影响训练效果
2. **超参数调整**: `fbcf_lambda`、`fg_loss_weight`、`bg_loss_weight` 需要根据具体任务调整
3. **采样比例**: 在Sampling Mode中，采样比例会影响不同view的训练频率
4. **内存管理**: Full Mode需要更多GPU内存，Sampling Mode更节省内存但训练可能更慢收敛
5. **Evaluation一致性**: 如果模型主要用于evaluation，建议Original比例 ≥ 50%，确保模型在原始视频上表现最佳

## 🔄 当前实现与原先版本的区别

### 版本演进历史

#### 原先版本（v1.0）
**特点**:
- 在 DataLoader 中预计算所有 view 的 pixels
- 同时存储 `video_pixels`, `fg_pixels`, `bg_pixels`
- 内存消耗：3倍 video pixels

**问题**:
- ❌ DataLoader workers 内存压力过大（OOM）
- ❌ Workers 频繁崩溃和重启
- ❌ 训练速度逐渐变慢（内存累积问题）
- ❌ 需要大量GPU内存

**代码位置**:
- `sft_dataset.py`: 在 `__getitem__` 中预计算所有 pixels
- `sft_trainer.py`: 直接使用预计算的 pixels

#### 当前版本（v2.0 - Lazy Loading）
**特点**:
- 在 Sampling Mode 下实现 Lazy Loading
- 只存储 mask 路径，不预计算 pixels
- 在 trainer 中按需计算需要的 view

**改进**:
- ✅ **内存占用大幅减少**: 不再同时存储3种view的pixels
- ✅ **避免DataLoader崩溃**: Workers不再因为内存压力而OOM
- ✅ **训练稳定性提升**: 减少workers重启，训练速度更稳定
- ✅ **按需计算**: 只计算当前需要的view，节省计算资源

**实现细节**:

1. **数据加载阶段** (`sft_dataset.py`):
   ```python
   # Sampling Mode: 只存储mask路径，不计算pixels
   if self.sampling_mode:
       # Lazy loading: only save mask paths, don't compute pixels
       # The trainer will compute fg/bg pixels on-demand based on view_type
       pass  # Skip pixel computation, mask paths are already in video_mask_paths
   ```

2. **训练阶段** (`sft_trainer.py`):
   ```python
   # 根据view_type按需计算
   if view_type == 0:
       pixels = video_pixels  # Original: 直接使用
   elif view_type == 1:
       pixels = self._compute_pixels_on_demand(..., view_type=1)  # Foreground: 按需计算
   else:
       pixels = self._compute_pixels_on_demand(..., view_type=2)  # Background: 按需计算
   ```

3. **按需计算方法** (`_compute_pixels_on_demand`):
   - 加载mask文件
   - 构建mask tensor
   - 根据view_type计算对应的pixels
   - 立即释放临时变量（`del mask_tensor`, `del inv_mask`）

**代码位置**:
- `sft_dataset.py:244-249`: Lazy loading逻辑
- `sft_trainer.py:271-332`: `_compute_pixels_on_demand`方法
- `sft_trainer.py:450-520`: Sampling Mode中的按需计算

**DataLoader参数优化**:
- `dataloader_num_workers`: `4 → 2`
- `dataloader_prefetch_factor`: `4 → 2`
- `dataloader_persistent_workers`: `False`

### 对比总结

| 特性 | 原先版本 (v1.0) | 当前版本 (v2.0) |
|------|----------------|----------------|
| **内存占用** | 高（3倍pixels） | 低（1倍pixels + mask路径） |
| **DataLoader稳定性** | ❌ 频繁崩溃 | ✅ 稳定运行 |
| **训练速度** | ❌ 逐渐变慢 | ✅ 稳定速度 |
| **计算时机** | 数据加载时预计算 | 训练时按需计算 |
| **适用模式** | Full Forward Mode | Sampling Mode |
| **I/O开销** | 低（一次性加载） | 中等（按需加载mask） |
| **计算开销** | 低（预计算） | 中等（按需计算） |

### 使用建议

1. **Sampling Mode（推荐）**: 使用当前版本的 Lazy Loading 实现
   - 内存占用低
   - 训练稳定
   - 适合大规模训练

2. **Full Forward Mode**: 仍使用原先的预计算方式
   - 如果内存充足
   - 需要每个step同时看到所有view

3. **参数调优**:
   - 如果仍然遇到内存问题，可以进一步减少 `dataloader_num_workers` 到 1
   - 如果训练速度可以接受，可以保持当前参数设置




