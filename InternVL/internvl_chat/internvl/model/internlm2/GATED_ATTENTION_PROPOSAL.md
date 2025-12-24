# Gated Attention with Independent Projection Layer Proposal

## 概述

将 `modeling_internlm2_gate.py` 中的 gated attention 实现改为使用独立的 `gate_proj` layer，而不是在 `wqkv` 中包含 gate 的维度。这种方式更清晰，gate 的计算独立于 QKV 的计算，类似于 ViT 中的实现。

## 当前实现（modeling_internlm2_gate.py）

### 问题
1. `wqkv` 的输出维度会根据 gating 模式动态调整
2. Gate score 从 `wqkv` 的输出中分离出来
3. 代码逻辑复杂，需要从 `wqkv` 输出中切片分离 gate

### 当前代码结构
```python
# __init__ 中
base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
if self.headwise_attn_output_gate:
    gate_dim = self.num_heads
    total_dim = base_qkv_dim + gate_dim
elif self.elementwise_attn_output_gate:
    gate_dim = self.num_heads * self.head_dim
    total_dim = base_qkv_dim + gate_dim
else:
    total_dim = base_qkv_dim

self.wqkv = nn.Linear(self.hidden_size, total_dim, bias=config.bias)

# forward 中
qkv_states = self.wqkv(hidden_states)
qkv_base = qkv_states[:, :, :base_qkv_dim]
gate_score_raw = qkv_states[:, :, base_qkv_dim:]
```

## 新实现方案（使用独立 gate_proj）

### 优势
1. **代码清晰**：`wqkv` 保持原始维度，gate 计算独立
2. **易于维护**：gate 逻辑与 QKV 逻辑分离
3. **灵活性**：可以轻松添加 gate 的 normalization 或其他处理
4. **一致性**：与 ViT 中的实现方式一致

### 实现细节

#### 1. `__init__` 方法修改

```python
def __init__(self, config: InternLM2Config):
    super().__init__()
    # ... 现有代码 ...
    
    # mh 1211: Gated Attention with Independent Projection Layer
    self.headwise_attn_output_gate = config.headwise_attn_output_gate
    self.elementwise_attn_output_gate = config.elementwise_attn_output_gate
    
    # wqkv 保持原始维度（不包含 gate）
    self.wqkv = nn.Linear(
        self.hidden_size,
        (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
        bias=config.bias,
    )
    
    # 独立的 gate_proj layer
    if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
        if self.headwise_attn_output_gate:
            # Headwise gate: [B, q_len, num_heads, 1]
            gate_out_dim = self.num_heads
        else:  # elementwise
            # Elementwise gate: [B, q_len, num_heads, head_dim]
            gate_out_dim = self.num_heads * self.head_dim
        
        self.gate_proj = nn.Linear(self.hidden_size, gate_out_dim, bias=False)
        # Initialize gate_proj with small std to prevent sigmoid saturation
        # Small std ensures gate logits are near 0, so sigmoid outputs are near 0.5
        std = config.initializer_range * 0.1  # Use 10% of initializer_range
        self.gate_proj.weight.data.normal_(mean=0.0, std=std)
    else:
        self.gate_proj = None
    
    self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
    self._init_rope()
```

#### 2. `forward` 方法修改（Eager Attention）

```python
def forward(self, hidden_states, ...):
    bsz, q_len, _ = hidden_states.size()
    
    # 计算 gate（如果启用）
    gate_score = None
    if self.gate_proj is not None:
        # 使用独立的 gate_proj 计算 gate logits
        gate_logits = self.gate_proj(hidden_states)  # [B, q_len, gate_out_dim]
        
        # 根据 gate 类型 reshape
        if self.headwise_attn_output_gate:
            # [B, q_len, num_heads] -> [B, q_len, num_heads, 1]
            gate_score = gate_logits.view(bsz, q_len, self.num_heads, 1)
        else:  # elementwise
            # [B, q_len, num_heads * head_dim] -> [B, q_len, num_heads, head_dim]
            gate_score = gate_logits.view(bsz, q_len, self.num_heads, self.head_dim)
    
    # 标准 QKV 计算（不包含 gate）
    qkv_states = self.wqkv(hidden_states)
    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )
    
    # ... 标准的 attention 计算 ...
    
    # 应用 gate（在 attention output 上）
    if gate_score is not None:
        gate_score = gate_score.transpose(1, 2)  # [B, num_heads, q_len, ...]
        gate_sigmoid = torch.sigmoid(gate_score)
        attn_output = attn_output * gate_sigmoid
    
    # ... 后续处理 ...
```

#### 3. `forward` 方法修改（Flash Attention）

```python
def forward(self, hidden_states, ...):
    bsz, q_len, _ = hidden_states.size()
    
    # 计算 gate（如果启用）
    gate_score = None
    if self.gate_proj is not None:
        gate_logits = self.gate_proj(hidden_states)
        
        if self.headwise_attn_output_gate:
            gate_score = gate_logits.view(bsz, q_len, self.num_heads, 1)
        else:  # elementwise
            gate_score = gate_logits.view(bsz, q_len, self.num_heads, self.head_dim)
    
    # 标准 QKV 计算
    qkv_states = self.wqkv(hidden_states)
    # ... 标准 attention 流程 ...
    
    # Flash Attention 返回 [B, q_len, num_heads, head_dim]
    attn_output = self._flash_attention_forward(...)
    
    # 应用 gate
    if gate_score is not None:
        gate_sigmoid = torch.sigmoid(gate_score)
        attn_output = attn_output * gate_sigmoid
    
    # ... 后续处理 ...
```

## 关键变化总结

### 1. `__init__` 变化
- ✅ `wqkv` 维度固定为 `(num_heads + 2 * num_key_value_heads) * head_dim`
- ✅ 添加独立的 `gate_proj` layer（仅在启用 gating 时）
- ✅ `gate_proj` 初始化使用较小的 std（`initializer_range * 0.1`）

### 2. `forward` 变化
- ✅ Gate 计算独立：`gate_logits = self.gate_proj(hidden_states)`
- ✅ 不再需要从 `wqkv` 输出中切片分离 gate
- ✅ QKV 计算保持标准流程，不涉及 gate
- ✅ Gate 应用逻辑保持不变（在 attention output 上）

### 3. 代码简化
- ✅ 移除了 `base_qkv_dim` 和 `total_dim` 的计算
- ✅ 移除了从 `qkv_states` 中分离 `qkv_base` 和 `gate_score_raw` 的逻辑
- ✅ Gate 计算和 QKV 计算完全解耦

## 配置要求

需要在 `InternLM2Config` 中添加以下配置项（如果还没有）：
- `headwise_attn_output_gate: bool = False`
- `elementwise_attn_output_gate: bool = False`

## 数值稳定性

- Gate logits 使用 `sigmoid` 激活，范围在 [0, 1]
- 初始化使用较小的 std（`initializer_range * 0.1`）确保初始 gate 值接近 0.5
- 可以考虑添加 `torch.clamp(gate_logits, min=-10.0, max=10.0)` 防止 sigmoid 饱和（可选）

## 迁移建议

1. **保持向后兼容**：如果 `gate_proj` 为 `None`，行为与标准 attention 相同
2. **权重迁移**：如果从旧版本迁移，需要：
   - 从 `wqkv` 权重中提取 gate 相关的权重
   - 初始化新的 `gate_proj` layer
3. **测试**：确保 gate 应用的位置和逻辑与原来一致

## 示例代码对比

### 旧方式（从 wqkv 分离）
```python
qkv_states = self.wqkv(hidden_states)  # [B, q_len, total_dim]
qkv_base = qkv_states[:, :, :base_qkv_dim]
gate_score_raw = qkv_states[:, :, base_qkv_dim:]
```

### 新方式（独立 gate_proj）
```python
qkv_states = self.wqkv(hidden_states)  # [B, q_len, base_qkv_dim]
gate_logits = self.gate_proj(hidden_states)  # [B, q_len, gate_dim]
```

## 总结

使用独立的 `gate_proj` layer 的优势：
1. ✅ 代码更清晰，逻辑分离
2. ✅ 易于维护和扩展
3. ✅ 与 ViT 实现方式一致
4. ✅ 不改变 `wqkv` 的原始维度，保持标准结构
5. ✅ 可以独立控制 gate 的初始化和处理




