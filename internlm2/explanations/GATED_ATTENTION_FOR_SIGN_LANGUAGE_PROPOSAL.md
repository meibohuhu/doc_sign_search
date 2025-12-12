# Gated Attention for Sign Language Recognition - Proposal

## 问题分析

从当前的 attention 可视化可以看到：
1. **高度稀疏的 attention**：大多数层只关注 1-3 个 patch
2. **不稳定的关注区域**：不同层关注不同的 patch，缺乏一致性
3. **可能错过关键信息**：手部和面部表情可能分散在多个 patch 中，但模型只关注少数 patch

## 总体目标

使用 Gated Attention 机制，引导模型：
1. **优先关注手部区域**：手语的核心信息源
2. **关注面部表情**：手语的重要补充信息（表情、口型等）
3. **保持灵活性**：允许模型学习哪些区域最重要

## 实施策略：两阶段方法

本 proposal 采用**渐进式两阶段方法**：

### Phase 1: 标准 Gated Attention（基础实验）
- 实现标准的 Head-wise 或 Element-wise Gated Attention
- 不添加任何领域特定的引导
- 让模型自主学习门控信号
- **目标**：验证 Gated Attention 机制本身的效果

### Phase 2: 手语特定的区域增强（领域优化）
- 在 Phase 1 的基础上，添加 binary mask 引导
- 使用手部和面部的 binary mask 增强门控信号
- **目标**：针对手语识别任务进一步优化

---

# Phase 1: 标准 Gated Attention 实现

## Phase 1 目标

实现标准的 Gated Attention 机制，验证其在手语识别任务上的基础效果，不添加任何领域特定的引导。

## Phase 1 方案设计

## Phase 1.1: 选择 Gating 模式

**决策点**：选择 Head-wise 还是 Element-wise Gating？

### Head-wise Gating（推荐用于 Phase 1）

**优势**：
- 参数效率高（每个 head 只需一个门控值）
- 计算开销小
- 实现简单，易于调试
- 适合作为第一步实验

**适用场景**：
- 资源受限的环境
- 需要快速验证效果
- 希望模型学习哪些 head 更重要

### Element-wise Gating

**优势**：
- 最细粒度的控制
- 更强的表达能力
- 可以学习更复杂的注意力模式

**劣势**：
- 参数数量多（每个 head 的每个维度都需要门控值）
- 计算开销较大

**建议**：Phase 1 先使用 **Head-wise Gating**，验证效果后再考虑 Element-wise。

## Phase 1.2: 实现标准 Gated Attention

### 1.2.1 配置修改

```python
# configuration_internlm2.py
class InternLM2Config:
    def __init__(
        self,
        # ... 现有参数 ...
        
        # Phase 1: 标准 gating（二选一）
        headwise_attn_output_gate=False,      # 推荐：参数效率高
        elementwise_attn_output_gate=False,   # 可选：更细粒度
        
        # Phase 2 的配置（暂时不启用）
        # region_aware_gating=False,
        # use_binary_mask=False,
    ):
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
```

### 1.2.2 修改 Attention 层（参考 Qwen3 实现）

```python
# modeling_internlm2.py
class InternLM2Attention(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        # ... 现有初始化代码 ...
        
        # Phase 1: 根据 gating 模式调整 q_proj 输出维度
        if config.headwise_attn_output_gate:
            # Head-wise: 增加 num_heads 个维度用于门控信号
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim + self.num_heads,
                bias=config.qkv_bias
            )
        elif config.elementwise_attn_output_gate:
            # Element-wise: 输出维度翻倍
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim * 2,
                bias=config.qkv_bias
            )
        else:
            # 标准模式
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=config.qkv_bias
            )
        
        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        self.elementwise_attn_output_gate = config.elementwise_attn_output_gate
    
    def forward(self, hidden_states, ...):
        # ... 标准 attention 计算 ...
        
        # Phase 1: 提取和应用标准 gating（无领域特定引导）
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            gate_score = self._extract_gate_score(query_states)
            attn_output = attn_output * torch.sigmoid(gate_score)
        
        return attn_output
```

### 1.2.3 Gate Score 提取（Head-wise 示例）

```python
def _extract_gate_score(self, query_states):
    """
    Phase 1: 标准 gate score 提取（无领域特定引导）
    """
    bsz, q_len, _ = query_states.shape
    
    if self.headwise_attn_output_gate:
        # 重塑为 [batch, seq_len, num_key_value_heads, ...]
        query_states = query_states.view(bsz, q_len, self.num_key_value_heads, -1)
        
        # 分离 query 和 gate_score
        query_states, gate_score = torch.split(
            query_states,
            [self.head_dim * self.num_key_value_groups, self.num_key_value_groups],
            dim=-1
        )
        
        # 重塑 gate_score: [batch, seq_len, num_heads, 1]
        gate_score = gate_score.reshape(bsz, q_len, -1, 1)
        
        return query_states, gate_score
    
    elif self.elementwise_attn_output_gate:
        # 类似处理，但 gate_score shape 是 [batch, seq_len, num_heads, head_dim]
        # ... (参考 GATED_ATTENTION_EXPLANATION.md) ...
        pass
```

## Phase 1.3: 训练策略

### 1.3.1 基线对比

**实验设置**：
1. **Baseline**：无 gating 的标准模型
2. **Head-wise Gating**：启用 `headwise_attn_output_gate=True`
3. **Element-wise Gating**（可选）：启用 `elementwise_attn_output_gate=True`

### 1.3.2 评估指标

- **识别准确率**：WER (Word Error Rate) 或 BLEU
- **Attention 模式**：可视化分析 attention 分布
- **计算效率**：推理速度和内存占用
- **收敛速度**：训练 epoch 数

### 1.3.3 预期结果

- ✅ 如果 Gated Attention 有效：准确率提升，attention 更聚焦
- ✅ 如果效果不明显：需要进入 Phase 2，添加领域特定引导
- ✅ 如果效果变差：检查实现或调整超参数

## Phase 1.4: 实施步骤

### Step 1: 代码实现（1 周）

1. ✅ 在 `InternLM2Config` 中添加 gating 配置参数
2. ✅ 修改 `InternLM2Attention.__init__` 调整 `q_proj` 维度
3. ✅ 实现 `_extract_gate_score` 方法
4. ✅ 在 `forward` 中应用 gating
5. ✅ 确保与现有代码兼容

### Step 2: 单元测试（2-3 天）

1. ✅ 测试 Head-wise gating 的形状和计算
2. ✅ 测试 Element-wise gating 的形状和计算
3. ✅ 验证梯度可以正常反向传播
4. ✅ 检查内存占用

### Step 3: 小规模实验（3-5 天）

1. ✅ 在小数据集上训练（如 100 个样本）
2. ✅ 对比 baseline 和 gating 版本
3. ✅ 可视化 attention 模式变化
4. ✅ 分析门控值的分布

### Step 4: 完整评估（1 周）

1. ✅ 在完整数据集上训练
2. ✅ 对比所有指标
3. ✅ 生成详细的实验报告
4. ✅ 决定是否进入 Phase 2

---

# Phase 2: 手语特定的区域增强

## Phase 2 目标

在 Phase 1 验证标准 Gated Attention 有效的基础上，添加手语特定的引导：
- 使用 binary mask 识别手部和面部区域
- 增强这些区域的门控信号
- 进一步优化手语识别性能

## Phase 2 前提条件

- ✅ Phase 1 已完成并验证有效
- ✅ 标准 Gated Attention 已实现并测试通过
- ✅ 有可用的 binary mask（手部和面部区域标注）

## Phase 2.1: 方案设计

### Phase 2.1.1: 使用 Binary Mask 识别手部和面部区域

```python
def identify_hand_face_regions_from_mask(
    binary_mask,  # [H, W] or [num_frames, H, W] binary mask (1=hand/face, 0=background)
    visual_token_index,
    num_patches_h,
    num_patches_w,
    patch_size=14  # ViT patch size (default 14 for InternVL)
):
    """
    从 binary mask 识别哪些 patch 包含手部和面部
    
    Args:
        binary_mask: Binary mask indicating hand/face regions
                    - Shape: [H, W] for single frame or [num_frames, H, W] for video
                    - Values: 1 = hand/face region, 0 = background
        visual_token_index: [start_idx, end_idx] visual token indices
        num_patches_h: Number of patches in height
        num_patches_w: Number of patches in width
        patch_size: Size of each patch (default 14 for ViT)
    
    Returns:
        hand_face_patch_mask: [num_visual_tokens] binary mask for patches
                            1 = patch contains hand/face, 0 = background patch
    """
    import torch
    import torch.nn.functional as F
    
    # Handle different input shapes
    if len(binary_mask.shape) == 2:
        # Single frame: [H, W]
        binary_mask = binary_mask.unsqueeze(0)  # [1, H, W]
    
    num_frames, H, W = binary_mask.shape
    patches_per_frame = num_patches_h * num_patches_w
    total_visual_tokens = num_frames * patches_per_frame
    
    # Initialize patch mask
    hand_face_patch_mask = torch.zeros(total_visual_tokens, dtype=torch.float32)
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame_mask = binary_mask[frame_idx]  # [H, W]
        
        # Downsample mask to patch grid resolution
        # Method 1: Average pooling to patch grid size
        frame_mask_tensor = frame_mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        patch_mask = F.adaptive_avg_pool2d(
            frame_mask_tensor, 
            (num_patches_h, num_patches_w)
        ).squeeze()  # [num_patches_h, num_patches_w]
        
        # Method 2: Threshold - if >50% of patch area is hand/face, mark as 1
        patch_mask = (patch_mask > 0.5).float()
        
        # Flatten to 1D and store
        start_patch = frame_idx * patches_per_frame
        end_patch = start_patch + patches_per_frame
        hand_face_patch_mask[start_patch:end_patch] = patch_mask.flatten()
    
    return hand_face_patch_mask


def identify_hand_face_regions_separate(
    hand_mask,     # [H, W] or [num_frames, H, W] - hand region mask
    face_mask,     # [H, W] or [num_frames, H, W] - face region mask
    visual_token_index,
    num_patches_h,
    num_patches_w,
    patch_size=14
):
    """
    如果手部和面部有分开的 mask，可以分别处理
    
    Returns:
        hand_patch_mask: [num_visual_tokens] - hand patches
        face_patch_mask: [num_visual_tokens] - face patches
    """
    hand_patch_mask = identify_hand_face_regions_from_mask(
        hand_mask, visual_token_index, num_patches_h, num_patches_w, patch_size
    )
    face_patch_mask = identify_hand_face_regions_from_mask(
        face_mask, visual_token_index, num_patches_h, num_patches_w, patch_size
    )
    
    return hand_patch_mask, face_patch_mask
```

### Phase 2.1.2: 使用 Binary Mask 进行区域感知的门控增强

```python
def forward(
    self, 
    hidden_states, 
    attention_mask=None, 
    visual_token_index=None,
    hand_face_mask=None,  # 新增：binary mask [num_visual_tokens] or [num_frames, H, W]
    **kwargs
):
    # 标准 attention 计算
    attn_output = self._compute_attention(hidden_states, ...)
    
    # 标准 gating（如果启用）
    if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
        gate_score = self._extract_gate_score(hidden_states)
        attn_output = attn_output * torch.sigmoid(gate_score)
    else:
        # 如果没有标准 gating，创建一个全 1 的 gate_score 作为基础
        gate_score = torch.ones_like(attn_output[..., 0:1])  # [batch, seq_len, num_heads, 1]
    
    # 区域感知增强（使用 binary mask）
    if self.region_aware_gating and visual_token_index is not None and hand_face_mask is not None:
        # 将 binary mask 转换为 patch-level mask
        if len(hand_face_mask.shape) > 1:
            # 如果是图像 mask [H, W] 或 [num_frames, H, W]，需要转换为 patch mask
            hand_face_patch_mask = identify_hand_face_regions_from_mask(
                hand_face_mask,
                visual_token_index,
                self.num_patches_h,
                self.num_patches_w
            )
        else:
            # 如果已经是 patch-level mask [num_visual_tokens]
            hand_face_patch_mask = hand_face_mask
        
        # 应用区域增强
        enhanced_gate = self._apply_region_enhancement_with_mask(
            gate_score,
            hand_face_patch_mask,
            visual_token_index,
            self.hand_face_prior
        )
        
        attn_output = attn_output * torch.sigmoid(enhanced_gate)
    
    return attn_output
```

### Phase 2.1.3: 使用 Binary Mask 的区域增强函数

```python
def _apply_region_enhancement_with_mask(
    self, 
    gate_score,           # [batch, seq_len, num_heads, 1] or [batch, seq_len, num_heads, head_dim]
    hand_face_patch_mask, # [num_visual_tokens] binary mask (1=hand/face, 0=background)
    visual_token_index,   # [start_idx, end_idx]
    hand_face_prior       # [num_heads, 1] or [num_heads, head_dim] 先验权重
):
    """
    使用 binary mask 增强手部和面部区域的门控值
    
    Args:
        gate_score: [batch, seq_len, num_heads, ...] 标准门控值
        hand_face_patch_mask: [num_visual_tokens] binary mask for visual tokens
        visual_token_index: [start_idx, end_idx] visual token range
        hand_face_prior: [num_heads, 1] or [num_heads, head_dim] 先验权重
    
    Returns:
        enhanced_gate: 增强后的门控值
    """
    batch_size, seq_len, num_heads = gate_score.shape[:3]
    device = gate_score.device
    
    enhanced_gate = gate_score.clone()
    
    # 创建序列级别的 mask
    # 对于 visual tokens (visual_token_index[0] 到 visual_token_index[1])
    visual_start = visual_token_index[0].item()
    visual_end = visual_token_index[1].item() + 1
    num_visual_tokens = visual_end - visual_start
    
    # 确保 mask 长度匹配
    if len(hand_face_patch_mask) != num_visual_tokens:
        # 如果 mask 长度不匹配，进行截断或填充
        if len(hand_face_patch_mask) > num_visual_tokens:
            hand_face_patch_mask = hand_face_patch_mask[:num_visual_tokens]
        else:
            padding = torch.zeros(num_visual_tokens - len(hand_face_patch_mask), device=device)
            hand_face_patch_mask = torch.cat([hand_face_patch_mask, padding])
    
    # 创建完整的序列 mask [seq_len]
    sequence_mask = torch.zeros(seq_len, device=device)
    sequence_mask[visual_start:visual_end] = hand_face_patch_mask.to(device)
    
    # 扩展到 batch 和 head 维度
    # sequence_mask: [seq_len] -> [1, seq_len, 1, 1] (for broadcasting)
    sequence_mask = sequence_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # [1, seq_len, 1, 1]
    
    # 扩展 hand_face_prior 以匹配 gate_score 的形状
    if len(hand_face_prior.shape) == 2:
        # [num_heads, 1] or [num_heads, head_dim]
        hand_face_prior = hand_face_prior.unsqueeze(0).unsqueeze(0)
        # [1, 1, num_heads, 1] or [1, 1, num_heads, head_dim]
    
    # 应用增强：只在 hand_face_patch_mask=1 的位置增强
    # 方式 1: 加法增强（推荐）
    # enhanced_gate = enhanced_gate + sequence_mask * hand_face_prior
    
    # 方式 2: 乘法增强（更温和，推荐用于稳定训练）
    enhancement = sequence_mask * hand_face_prior * self.config.hand_face_prior_weight
    enhanced_gate = enhanced_gate * (1 + enhancement)
    
    # 方式 3: 混合方式（最灵活）
    # base_enhancement = sequence_mask * hand_face_prior * 0.3  # 基础增强
    # adaptive_enhancement = sequence_mask * gate_score * 0.2   # 自适应增强
    # enhanced_gate = enhanced_gate + base_enhancement + adaptive_enhancement
    
    return enhanced_gate
```

## Phase 2.2: 修改 Attention 层（在 Phase 1 基础上扩展）

在 Phase 1 的标准 gating 基础上，添加区域增强：

### Phase 2.2.1: 配置扩展（在 Phase 1 基础上添加）

```python
# configuration_internlm2.py
class InternLM2Config:
    def __init__(
        self,
        # ... 现有参数 ...
        
        # Phase 1: 标准 gating（必须已实现）
        headwise_attn_output_gate=False,
        elementwise_attn_output_gate=False,
        
        # Phase 2: 区域感知 gating（使用 binary mask）
        region_aware_gating=False,              # Phase 2 总开关
        hand_face_prior_weight=0.3,            # 手部/面部的增强权重（建议 0.2-0.5）
        use_binary_mask=True,                   # 是否使用 binary mask（推荐 True）
        mask_enhancement_mode='multiply',       # 'add', 'multiply', or 'hybrid'
    ):
        # Phase 1 配置
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        
        # Phase 2 配置
        self.region_aware_gating = region_aware_gating
        self.hand_face_prior_weight = hand_face_prior_weight
        self.use_binary_mask = use_binary_mask
        self.mask_enhancement_mode = mask_enhancement_mode
```

### Phase 2.2.2: 修改 Attention 层（扩展 Phase 1 的实现）

```python
# modeling_internlm2.py
class InternLM2Attention(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        visual_token_index=None,  # 用于识别 visual tokens
        hand_face_mask=None,      # Phase 2 新增：binary mask
        **kwargs
    ):
        # ... 标准 attention 计算 ...
        
        # Phase 1: 应用标准 gating（必须启用）
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            query_states, gate_score = self._extract_gate_score(query_states)
            # ... 使用 query_states 计算 attention ...
            attn_output = attn_output * torch.sigmoid(gate_score)
        else:
            # 如果没有标准 gating，Phase 2 无法工作
            if self.config.region_aware_gating:
                raise ValueError(
                    "region_aware_gating requires headwise_attn_output_gate or "
                    "elementwise_attn_output_gate to be enabled first"
                )
            gate_score = None
        
        # Phase 2: 应用区域感知增强（使用 binary mask）
        # 注意：这依赖于 Phase 1 的 gate_score
        if (self.config.region_aware_gating and 
            self.config.use_binary_mask and 
            visual_token_index is not None and 
            hand_face_mask is not None and
            gate_score is not None):
            
            enhanced_gate = self._apply_region_aware_gating_with_mask(
                gate_score,           # 从 Phase 1 得到的标准 gate_score
                hand_face_mask,       # binary mask
                visual_token_index,
                self.config.hand_face_prior_weight
            )
            
            # 重新应用增强后的 gating
            attn_output = attn_output * torch.sigmoid(enhanced_gate)
        
        return attn_output
```

### Phase 2.2.3: 在模型 forward 中传递 binary mask

```python
# modeling_internlm2.py - InternLM2Model.forward
def forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    visual_token_index=None,
    hand_face_mask=None,  # 新增：binary mask
    **kwargs
):
    # ... 现有代码 ...
    
    for idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[idx] if past_key_values is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            visual_token_index=visual_token_index,  # 传递 visual_token_index
            hand_face_mask=hand_face_mask,          # 传递 binary mask
            **kwargs
        )
        hidden_states = layer_outputs[0]
        # ...
```

### Phase 2.2.4: 在 InternVL Chat 接口中传递 mask

```python
# internvl_chat/modeling_internvl_chat.py
def chat(
    self,
    tokenizer,
    pixel_values,
    question,
    generation_config,
    num_patches_list,
    history=None,
    hand_face_mask=None,  # 新增：binary mask
    **kwargs
):
    # ... 处理 pixel_values 和 question ...
    
    # 调用 language_model.forward
    outputs = self.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        visual_token_index=visual_token_index,
        hand_face_mask=hand_face_mask,  # 传递 binary mask
        output_attentions=output_attentions,
        **kwargs
    )
```

## Phase 2.3: Binary Mask 的使用方式

### Phase 2.3.1: Mask 格式

Binary mask 可以是以下格式之一：

1. **图像级 mask**：`[H, W]` 或 `[num_frames, H, W]`
   - 值：1 = 手部/面部区域，0 = 背景
   - 需要转换为 patch-level mask

2. **Patch 级 mask**：`[num_visual_tokens]`
   - 值：1 = 包含手部/面部的 patch，0 = 背景 patch
   - 可以直接使用

3. **分离的 mask**：`hand_mask` 和 `face_mask`
   - 可以分别处理手部和面部
   - 可以给它们不同的权重

### Phase 2.3.2: Mask 加载示例

```python
# 方式 1: 从文件加载
import numpy as np
import torch

# 假设 mask 保存在 .npy 或 .png 文件中
hand_face_mask = np.load('path/to/mask.npy')  # [H, W] or [num_frames, H, W]
hand_face_mask = torch.from_numpy(hand_face_mask).float()

# 方式 2: 从视频处理中生成
def load_mask_for_video(video_path, frame_indices):
    """
    为视频的特定帧加载 binary mask
    """
    masks = []
    for frame_idx in frame_indices:
        # 假设每个帧的 mask 保存在单独的文件中
        mask_path = f'masks/frame_{frame_idx:04d}_mask.npy'
        mask = np.load(mask_path)  # [H, W]
        masks.append(mask)
    
    # 堆叠为 [num_frames, H, W]
    return np.stack(masks)

# 方式 3: 实时生成（如果使用检测器）
def generate_mask_with_detector(pixel_values):
    """
    使用检测器实时生成 mask
    """
    import mediapipe as mp
    
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    
    # 检测并生成 mask
    # ... 检测逻辑 ...
    
    return hand_face_mask
```

### Phase 2.3.3: 在训练/推理中使用

```python
# 训练时（必须有 binary mask）
for batch in dataloader:
    pixel_values = batch['pixel_values']
    hand_face_mask = batch['hand_face_mask']  # 从数据加载器获取（必须）
    
    outputs = model(
        pixel_values=pixel_values,
        hand_face_mask=hand_face_mask,  # 传递 mask（训练时必需）
        ...
    )

# Evaluation 时：两种模式

# 模式 1: 有 binary mask（理想情况，更准确）
hand_face_mask = load_mask_for_video(video_path, frame_indices)
response = model.chat(
    tokenizer,
    pixel_values,
    question,
    hand_face_mask=hand_face_mask,  # 使用 mask
    ...
)

# 模式 2: 无 binary mask（实际部署情况）
# 如果 mask 不可用，可以：
# 选项 A: 回退到 Phase 1 模式（只使用标准 gating，不使用区域增强）
response = model.chat(
    tokenizer,
    pixel_values,
    question,
    hand_face_mask=None,  # 不传递 mask，模型自动回退到标准 gating
    ...
)

# 选项 B: 使用检测器实时生成 mask（如果实现了）
hand_face_mask = generate_mask_with_detector(pixel_values)  # 实时生成
response = model.chat(
    tokenizer,
    pixel_values,
    question,
    hand_face_mask=hand_face_mask,  # 使用实时生成的 mask
    ...
)
```

### Phase 2.3.4: Evaluation 策略

**重要考虑**：在实际 evaluation 时，binary mask 可能不可用（标注成本高）。需要设计灵活的机制：

```python
# modeling_internlm2.py - Attention 层的 forward
def forward(self, hidden_states, hand_face_mask=None, ...):
    # Phase 1: 标准 gating（总是执行）
    if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
        gate_score = self._extract_gate_score(query_states)
        attn_output = attn_output * torch.sigmoid(gate_score)
    
    # Phase 2: 区域增强（仅在 mask 可用时执行）
    if (self.config.region_aware_gating and 
        hand_face_mask is not None):  # 关键：mask 是可选的
        enhanced_gate = self._apply_region_aware_gating_with_mask(...)
        attn_output = attn_output * torch.sigmoid(enhanced_gate)
    # 如果没有 mask，就只使用 Phase 1 的标准 gating
    
    return attn_output
```

**Evaluation 建议**：

1. **有 mask 的 evaluation**（如果数据集有标注）：
   - 使用 binary mask，测试 Phase 2 的最佳性能
   - 这是理想情况，展示模型在有先验知识时的上限

2. **无 mask 的 evaluation**（实际部署场景）：
   - 不传递 mask，模型自动回退到 Phase 1 的标准 gating
   - 测试模型在实际部署时的性能（更贴近真实场景）

3. **对比实验**：
   - 同时测试有 mask 和无 mask 两种情况
   - 评估 binary mask 带来的性能提升
   - 如果提升不明显，说明 Phase 1 已经足够好


