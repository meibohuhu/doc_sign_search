# Visual-Conditioned Gated Attention Implementation

## 概述

Visual-Conditioned Gated Attention 是一个将全局视觉上下文信息融入到注意力门控机制中的方法。它通过聚合视觉 token 生成一个全局的视觉摘要（visual summary），然后使用这个摘要来生成一个全局的 head gate，与现有的 token-wise gate 进行组合。

## 核心思想

1. **Visual Summary (z)**: 从所有视觉 token 中聚合得到一个全局表示 `[B, C]`
2. **Visual Gate (g_vis)**: 通过线性投影将 visual summary 映射到每个 attention head 的 gate 值 `[B, num_heads]`
3. **Gate Combination**: 将 visual gate 与 token-wise gate 组合，支持多种组合策略

## 实现细节

### 1. 配置参数

在 `configuration_internlm2.py` 中添加了三个新参数：

```python
class InternLM2Config:
    def __init__(
        self,
        # ... existing parameters ...
        visual_summary_head_gate=False,      # 是否启用 visual-conditioned head gate
        visual_gate_mode='add_logits',        # 组合策略: 'add_logits', 'mul', 'replace'
        visual_summary_aggregation='mean',    # Visual summary 聚合方法: 'mean', 'max', 'sum', 'first'
        **kwargs,
    ):
        # ...
        self.visual_summary_head_gate = visual_summary_head_gate
        self.visual_gate_mode = visual_gate_mode
        self.visual_summary_aggregation = visual_summary_aggregation
```

**参数说明**:
- `visual_summary_head_gate` (bool): 是否启用 visual-conditioned head gate
- `visual_gate_mode` (str): Gate 组合策略
  - `'add_logits'`: `sigmoid(raw_tok + raw_vis)` - 在 raw logits 上相加（推荐，初始化时等价于原模型）
  - `'mul'`: `g_tok * g_vis` - 在 sigmoid 后的 gate 值上相乘
  - `'replace'`: 只用 `g_vis`，忽略 `g_tok`
- `visual_summary_aggregation` (str): Visual summary 聚合方法
  - `'mean'` (默认): 对所有 visual tokens 求平均 - 适合需要全局平均表示的场景
  - `'max'`: 对所有 visual tokens 求最大值 - 适合需要突出最显著特征的场景
  - `'sum'`: 对所有 visual tokens 求和 - 适合需要累积所有信息的场景
  - `'first'`: 取第一个 visual token - 适合第一个 token 包含全局信息（如 CLS token）的场景

### 2. Visual Summary 计算

在 `modeling_internvl_chat.py` 的 `InternVLChatModel.forward` 中计算 visual summary：

```python
# 计算 visual_summary z: 从 vit_embeds 聚合得到 [B, C]
visual_summary = None
if hasattr(self.language_model.config, 'visual_summary_head_gate') and \
   self.language_model.config.visual_summary_head_gate:
    if len(vit_embeds) > 0:
        # vit_embeds: [num_frames, num_patches, C] 或 [num_visual_tokens, C]
        # 先 flatten 到 2 维
        if vit_embeds.dim() == 3:
            # [num_frames, num_patches, C] -> [num_frames * num_patches, C]
            vit_embeds_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        else:
            vit_embeds_flat = vit_embeds
        
        # 根据 aggregation_mode 选择聚合方法
        aggregation_mode = getattr(self.language_model.config, 'visual_summary_aggregation', 'mean')
        if aggregation_mode == 'mean':
            visual_summary_flat = vit_embeds_flat.mean(dim=0, keepdim=True)  # [1, C]
        elif aggregation_mode == 'max':
            visual_summary_flat = vit_embeds_flat.max(dim=0, keepdim=True)[0]  # [1, C]
        elif aggregation_mode == 'sum':
            visual_summary_flat = vit_embeds_flat.sum(dim=0, keepdim=True)  # [1, C]
        elif aggregation_mode == 'first':
            visual_summary_flat = vit_embeds_flat[0:1]  # [1, C] - 取第一个 token
        else:
            # fallback to mean
            visual_summary_flat = vit_embeds_flat.mean(dim=0, keepdim=True)  # [1, C]
        
        visual_summary = visual_summary_flat.expand(B, -1)  # [B, C]
```

**计算流程**:
1. 将 `vit_embeds` flatten 到 `[num_visual_tokens, C]`
2. 根据 `visual_summary_aggregation` 配置选择聚合方法，得到 `[1, C]`
3. 扩展到 batch 维度得到 `[B, C]`

**聚合方法详解**:

| 方法 | 计算方式 | 适用场景 | 特点 |
|------|---------|---------|------|
| `mean` (默认) | `visual_summary = mean(vit_embeds, dim=0)` | 需要全局平均表示 | 平衡所有 visual tokens 的信息，适合大多数场景 |
| `max` | `visual_summary = max(vit_embeds, dim=0)[0]` | 需要突出最显著特征 | 保留每个维度的最大值，适合需要突出关键信息的场景 |
| `sum` | `visual_summary = sum(vit_embeds, dim=0)` | 需要累积所有信息 | 对所有 visual tokens 求和，适合需要累积信息的场景 |
| `first` | `visual_summary = vit_embeds[0]` | 第一个 token 包含全局信息 | 直接使用第一个 visual token，适合 CLS token 场景 |

**选择建议**:
- **mean**: 默认选择，适合大多数场景，提供平衡的全局表示
- **max**: 当需要突出图像中最显著的特征时使用
- **sum**: 当需要累积所有视觉信息时使用（注意：值会随 visual tokens 数量变化）
- **first**: 当第一个 visual token 已经包含全局信息时使用（如某些 ViT 架构的 CLS token）

### 3. Visual Gate Projection

在 `InternLM2Attention.__init__` 中初始化 `visual_gate_proj`：

```python
# Visual-conditioned gated attention
self.visual_summary_head_gate = getattr(config, 'visual_summary_head_gate', False)
self.visual_gate_mode = getattr(config, 'visual_gate_mode', 'add_logits')
if self.visual_summary_head_gate:
    self.visual_gate_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
    nn.init.zeros_(self.visual_gate_proj.weight)
    nn.init.zeros_(self.visual_gate_proj.bias)
```

**初始化策略**:
- 权重和偏置都初始化为 0
- 这样在训练初期，`g_vis_raw = 0`，`sigmoid(0) = 0.5`
- 在 `add_logits` 模式下，如果 `g_tok_raw` 也是 0，则 `gate = sigmoid(0 + 0) = 0.5`
- 这确保了初始化时不会改变模型行为（如果 token gate 也是零初始化）

### 4. Visual Gate 计算

在 `InternLM2FlashAttention2.forward` 中计算 visual gate：

```python
# Compute visual gate g_vis if visual_summary is provided
g_vis = None
if self.visual_summary_head_gate and visual_summary is not None:
    g_vis_raw = self.visual_gate_proj(visual_summary)  # [B, num_heads]
    g_vis = g_vis_raw.unsqueeze(1).unsqueeze(-1)  # [B, 1, num_heads, 1]
```

**维度变换**:
- `visual_summary`: `[B, hidden_size]`
- `g_vis_raw`: `[B, num_heads]`
- `g_vis`: `[B, 1, num_heads, 1]` (用于后续与 token gate 组合)

### 5. Gate 组合策略
**组合策略详解**:

1. **add_logits** (推荐):
   - `gate = sigmoid(gate_tok_raw + g_vis_expanded)`
   - 在 raw logits 上相加，然后统一做 sigmoid
   - 初始化时如果两者都是 0，则 `gate = 0.5`，不会改变模型行为

2. **mul**:
   - `gate = sigmoid(gate_tok_raw) * sigmoid(g_vis_expanded)`
   - 先分别做 sigmoid，然后相乘
   - 两个 gate 都需要激活才能让最终 gate 接近 1

3. **replace**:
   - `gate = sigmoid(g_vis_expanded)`
   - 完全忽略 token-wise gate，只用 visual gate

### 6. 参数传递链

Visual summary 通过以下路径传递：

```
InternVLChatModel.forward
  └─> InternLM2ForCausalLM.forward(visual_summary=visual_summary)
      └─> InternLM2Model.forward(visual_summary=visual_summary)
          └─> InternLM2DecoderLayer.forward(visual_summary=visual_summary)
              └─> InternLM2FlashAttention2.forward(visual_summary=visual_summary)
```

**关键代码位置**:

1. **InternVLChatModel.forward** (`modeling_internvl_chat.py`):
   - 计算 `visual_summary`
   - 传递给 `language_model.forward(visual_summary=visual_summary)`

2. **InternLM2ForCausalLM.forward** (`modeling_internlm2_gate_wqkv.py`):
   - 接收 `visual_summary` 参数
   - 传递给 `self.model.forward(visual_summary=visual_summary)`

3. **InternLM2Model.forward** (`modeling_internlm2_gate_wqkv.py`):
   - 接收 `visual_summary` 参数
   - 在 layer 循环中传递给每个 `decoder_layer`
   - 支持 gradient checkpointing（在 `create_custom_forward` 中包含 `visual_summary`）

4. **InternLM2DecoderLayer.forward** (`modeling_internlm2_gate_wqkv.py`):
   - 接收 `visual_summary` 参数
   - 传递给 `self.attention.forward(visual_summary=visual_summary)`

5. **InternLM2FlashAttention2.forward** (`modeling_internlm2_gate_wqkv.py`):
   - 接收 `visual_summary` 参数
   - 计算 `g_vis` 并与 token gate 组合

### 7. 训练脚本配置

在 `internvl_chat_finetune_gate_wqkv.py` 中添加了命令行参数：


## 使用方法

### 命令行参数

```bash
python internvl_chat/internvl/train/internvl_chat_finetune_gate_wqkv.py \
    --headwise_attn_output_gate True \
    --visual_summary_head_gate True \
    --visual_gate_mode add_logits \
    --visual_summary_aggregation mean \
    # ... other arguments ...
```

### 配置说明

- `--headwise_attn_output_gate True`: 启用 token-wise head gate（必需，visual gate 会与它组合）
- `--visual_summary_head_gate True`: 启用 visual-conditioned head gate
- `--visual_gate_mode add_logits`: 选择组合策略（推荐使用 `add_logits`）
- `--visual_summary_aggregation mean`: 选择 visual summary 聚合方法（可选：`mean`, `max`, `sum`, `first`，默认 `mean`）

### 配置示例

**示例 1: add_logits + mean (推荐)**
```bash
--headwise_attn_output_gate True \
--visual_summary_head_gate True \
--visual_gate_mode add_logits \
--visual_summary_aggregation mean
```

**示例 2: add_logits + sum**
```bash
--headwise_attn_output_gate True \
--visual_summary_head_gate True \
--visual_gate_mode add_logits \
--visual_summary_aggregation sum
```

**示例 3: replace + mean**
```bash
--headwise_attn_output_gate True \
--visual_summary_head_gate True \
--visual_gate_mode replace \
--visual_summary_aggregation mean
```

**示例 4: mul + mean**
```bash
--headwise_attn_output_gate True \
--visual_summary_head_gate True \
--visual_gate_mode mul \
--visual_summary_aggregation mean
```

## 实现文件清单

1. **配置**:
   - `internvl_chat/internvl/model/internlm2/configuration_internlm2.py`
     - 添加 `visual_summary_head_gate`、`visual_gate_mode` 和 `visual_summary_aggregation` 参数

2. **顶层模型**:
   - `internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py`
     - 计算 `visual_summary` 并传递给 language model

3. **语言模型**:
   - `internvl_chat/internvl/model/internlm2/modeling_internlm2_gate_wqkv.py`
     - `InternLM2ForCausalLM.forward`: 接收并传递 `visual_summary`
     - `InternLM2Model.forward`: 接收并传递 `visual_summary`（支持 gradient checkpointing）
     - `InternLM2DecoderLayer.forward`: 接收并传递 `visual_summary`
     - `InternLM2Attention.__init__`: 初始化 `visual_gate_proj`
     - `InternLM2FlashAttention2.forward`: 计算 `g_vis` 并实现组合逻辑

4. **训练脚本**:
   - `internvl_chat/internvl/train/internvl_chat_finetune_gate_wqkv.py`
     - 添加命令行参数
     - 设置配置
     - 确保模型加载后配置正确

## 维度说明

- `visual_summary`: `[B, hidden_size]` (例如 `[1, 2048]`)
- `g_vis_raw`: `[B, num_heads]` (例如 `[1, 16]`)
- `g_vis`: `[B, 1, num_heads, 1]` (例如 `[1, 1, 16, 1]`)  g_vis 是一个全局的、每个 head 一个标量的 gate 值， 它来自 visual summary（所有 visual tokens 的聚合, 在 add_logits 模式下，会与每个 token 的 gate 值相。因此，g_vis 对每个 head 有一个值，表示该 head 受全局视觉信息影响的程度。
- `gate_tok_raw`: `[B, q_len, num_heads, 1]` (headwise) 或 `[B, q_len, num_heads, head_dim]` (elementwise)
- `g_vis_expanded`: `[B, q_len, num_heads, 1]` 或 `[B, q_len, num_heads, head_dim]` (根据 token gate 维度)
- `gate_combined_raw`: `[B, q_len, num_heads, 1]` 或 `[B, q_len, num_heads, head_dim]`
- `gate_sigmoid`: `[B, q_len, num_heads, 1]` 或 `[B, q_len, num_heads, head_dim]`
- `attn_output`: `[B, q_len, num_heads, head_dim]`

## 设计优势

1. **初始化稳定性**: `visual_gate_proj` 初始化为 0，确保训练初期不改变模型行为
2. **灵活组合**: 支持多种组合策略，可以根据任务选择最适合的方式
3. **向后兼容**: 当 `visual_summary_head_gate=False` 时，完全不影响原有功能
4. **内存高效**: Visual summary 是全局的 `[B, C]`，不会随序列长度增长
5. **梯度友好**: 支持 gradient checkpointing，不会影响训练效率

## 注意事项

1. **必须同时启用 token-wise gate**: `visual_summary_head_gate` 需要与 `headwise_attn_output_gate` 或 `elementwise_attn_output_gate` 一起使用
2. **Visual summary 计算**: 如果 `vit_embeds` 为空，`visual_summary` 将为 `None`，visual gate 不会生效
3. **配置传递**: 模型加载后需要手动更新所有 attention 模块的属性，因为模块在初始化时已经读取了配置
4. **聚合方法选择**: 
   - `mean` 是默认且最稳定的选择，适合大多数场景
   - `sum` 的值会随 visual tokens 数量变化，可能需要调整学习率
   - `max` 适合需要突出关键特征的场景
   - `first` 适合第一个 token 包含全局信息的架构




--headwise_attn_output_gate  True \
--visual_summary_head_gate True \
--visual_gate_mode add_logits \
--visual_summary_aggregation sum \


--headwise_attn_output_gate  True \
--visual_summary_head_gate True \
--visual_gate_mode add_logits \
--visual_summary_aggregation mean \



--headwise_attn_output_gate  True \
--visual_summary_head_gate True \
--visual_gate_mode replace \
--visual_summary_aggregation mean \

##### initial loss 5.46
--headwise_attn_output_gate  True \
--visual_summary_head_gate True \
--visual_gate_mode mul \
--visual_summary_aggregation mean \

