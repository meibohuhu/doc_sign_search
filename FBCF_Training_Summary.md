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
- **Background KL Loss**: 使用内存优化的实现
  - 立即对vocab维度求和，减少中间tensor的内存占用
  - 代码位置: `sft_trainer.py:222-262`
- **DeepSpeed兼容**: 自动检测是否使用DeepSpeed，避免不必要的cache清理

## 📝 训练流程

1. **数据加载**:
   - 加载原始视频帧
   - 从mask文件夹加载对应的mask文件
   - 生成前景视频 (应用mask)
   - 生成背景视频 (应用反mask)

2. **训练循环**:
   - **Full Mode**: 对三种view都进行前向传播，计算各自的loss并加权求和
   - **Sampling Mode**: 根据配置的比例随机选择一种view，只计算对应的loss

3. **Loss反向传播**:
   - 计算total_loss
   - 反向传播更新模型参数

## 🎯 预期效果

- **提高手势识别准确性**: 通过前景训练，模型更关注手势区域
- **减少背景干扰**: 通过背景KL loss，模型学会忽略无关的背景信息
- **更好的泛化能力**: 模型能够区分有意义的信号（手势）和无意义的噪声（背景）

## 📌 注意事项

1. **Mask质量**: Mask的准确性和覆盖范围直接影响训练效果
2. **超参数调整**: `fbcf_lambda`、`fg_loss_weight`、`bg_loss_weight` 需要根据具体任务调整
3. **采样比例**: 在Sampling Mode中，采样比例会影响不同view的训练频率
4. **内存管理**: Full Mode需要更多GPU内存，Sampling Mode更节省内存但训练可能更慢收敛




