# Phase 1: 标准 Gated Attention 在 InternLM2 中的实现 Proposal

## 核心差异分析

### Qwen3 vs InternLM2 的架构差异

| 特性 | Qwen3 | InternLM2 |
|------|-------|-----------|
| QKV 投影 | 分离的 `q_proj`, `k_proj`, `v_proj` | 统一的 `wqkv` |
| 投影输出 | `q_proj`: 根据 gating 调整维度 | `wqkv`: `(num_heads + 2 * num_key_value_heads) * head_dim` |
| 重塑方式 | `view` + `transpose` | `einops.rearrange` |
| GQA 支持 | 有 `num_key_value_heads` | 有 `num_key_value_heads` 和 `num_key_value_groups` |

### 关键挑战

1. **统一投影 vs 分离投影**：InternLM2 使用 `wqkv` 统一投影，需要调整整个输出维度
2. **Einops 重塑**：需要修改 `rearrange` 的逻辑来分离 gate_score
3. **GQA 兼容性**：需要确保 gating 与 Grouped Query Attention 兼容

## 实现方案

### 方案 1: 修改 `wqkv` 输出维度（推荐）

**核心思想**：在 `wqkv` 的输出中，为 query 部分增加 gate_score 的维度。

#### 1.1 配置修改

```python
# configuration_internlm2.py
class InternLM2Config:
    def __init__(
        self,
        # ... 现有参数 ...
        
        # Phase 1: 标准 gating（二选一）
        headwise_attn_output_gate=False,      # 推荐：参数效率高
        elementwise_attn_output_gate=False,   # 可选：更细粒度
    ):
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
```

#### 1.2 修改 `InternLM2Attention.__init__`

```python
# modeling_internlm2.py
class InternLM2Attention(nn.Module):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        # ... 现有初始化代码 ...
        
        # Phase 1: 根据 gating 模式调整 wqkv 输出维度
        # 标准模式: (num_heads + 2 * num_key_value_heads) * head_dim
        # Head-wise: query 部分增加 num_heads 个维度
        # Element-wise: query 部分增加 num_heads * head_dim 个维度
        
        base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        
        if config.headwise_attn_output_gate:
            # Head-wise: query 部分从 num_heads * head_dim 增加到 num_heads * head_dim + num_heads
            # 总维度增加: num_heads
            gate_dim = self.num_heads
            self.wqkv = nn.Linear(
                self.hidden_size,
                base_qkv_dim + gate_dim,
                bias=config.bias,
            )
        elif config.elementwise_attn_output_gate:
            # Element-wise: query 部分从 num_heads * head_dim 增加到 num_heads * head_dim * 2
            # 总维度增加: num_heads * head_dim
            gate_dim = self.num_heads * self.head_dim
            self.wqkv = nn.Linear(
                self.hidden_size,
                base_qkv_dim + gate_dim,
                bias=config.bias,
            )
        else:
            # 标准模式
            self.wqkv = nn.Linear(
                self.hidden_size,
                base_qkv_dim,
                bias=config.bias,
            )
        
        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        self.elementwise_attn_output_gate = config.elementwise_attn_output_gate
```

#### 1.3 修改 `forward` 方法 - 分离 gate_score

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # ... 现有代码 ...
    
    bsz, q_len, _ = hidden_states.size()
    
    qkv_states = self.wqkv(hidden_states)  # [bsz, q_len, base_qkv_dim + gate_dim]
    
    # Phase 1: 分离 gate_score 和标准 qkv
    if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
        # 分离标准 qkv 和 gate_score
        base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        
        if self.headwise_attn_output_gate:
            gate_dim = self.num_heads
        else:  # elementwise
            gate_dim = self.num_heads * self.head_dim
        
        # 分离
        qkv_base = qkv_states[:, :, :base_qkv_dim]  # [bsz, q_len, base_qkv_dim]
        gate_score_raw = qkv_states[:, :, base_qkv_dim:]  # [bsz, q_len, gate_dim]
        
        # 处理 qkv_base（标准流程）
        qkv_states = rearrange(
            qkv_base,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )
        
        # 处理 gate_score
        if self.headwise_attn_output_gate:
            # gate_score_raw: [bsz, q_len, num_heads]
            # 需要重塑为 [bsz, q_len, num_heads, 1] 以匹配 attention output
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
        else:  # elementwise
            # gate_score_raw: [bsz, q_len, num_heads * head_dim]
            # 需要重塑为 [bsz, q_len, num_heads, head_dim]
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
    else:
        # 标准模式（无 gating）
        qkv_states = rearrange(
            qkv_states,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )
        gate_score = None
    
    # 继续标准 attention 计算
    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]
    
    query_states = query_states.transpose(1, 2)  # [bsz, num_heads, q_len, head_dim]
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # ... RoPE, attention 计算等 ...
    
    attn_output = torch.matmul(attn_weights, value_states)  # [bsz, num_heads, q_len, head_dim]
    
    # Phase 1: 应用 gating
    if gate_score is not None:
        # gate_score 需要 transpose 以匹配 attn_output 的形状
        # gate_score: [bsz, q_len, num_heads, ...] -> [bsz, num_heads, q_len, ...]
        gate_score = gate_score.transpose(1, 2)  # [bsz, num_heads, q_len, 1] or [bsz, num_heads, q_len, head_dim]
        attn_output = attn_output * torch.sigmoid(gate_score)
    
    attn_output = attn_output.transpose(1, 2).contiguous()  # [bsz, q_len, num_heads, head_dim]
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    
    attn_output = self.wo(attn_output)
    
    # ... 返回 ...
```

### 方案 2: 分离 q_proj（备选，不推荐）

**核心思想**：将 `wqkv` 拆分为 `wq`, `wk`, `wv`，只在 `wq` 中增加 gate_score 维度。

**缺点**：
- 需要大幅修改现有代码结构
- 可能影响性能（三个独立的线性层）
- 与现有架构差异较大

**不推荐使用此方案**。

## 完整实现代码

### 修改后的 `InternLM2Attention.__init__`

```python
def __init__(self, config: InternLM2Config):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.is_causal = True
    
    # Phase 1: Gating 配置
    self.headwise_attn_output_gate = getattr(config, 'headwise_attn_output_gate', False)
    self.elementwise_attn_output_gate = getattr(config, 'elementwise_attn_output_gate', False)
    
    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
            f' and `num_heads`: {self.num_heads}).'
        )
    
    # Phase 1: 根据 gating 模式调整 wqkv 输出维度
    base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
    
    if self.headwise_attn_output_gate:
        gate_dim = self.num_heads
        total_dim = base_qkv_dim + gate_dim
    elif self.elementwise_attn_output_gate:
        gate_dim = self.num_heads * self.head_dim
        total_dim = base_qkv_dim + gate_dim
    else:
        total_dim = base_qkv_dim
    
    self.wqkv = nn.Linear(
        self.hidden_size,
        total_dim,
        bias=config.bias,
    )
    
    self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
    self._init_rope()
```

### 修改后的 `forward` 方法（关键部分）

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # ... 现有代码 ...
    
    bsz, q_len, _ = hidden_states.size()
    
    qkv_states = self.wqkv(hidden_states)
    
    # Phase 1: 分离 gate_score
    base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
    
    if self.headwise_attn_output_gate:
        gate_dim = self.num_heads
        qkv_base = qkv_states[:, :, :base_qkv_dim]
        gate_score_raw = qkv_states[:, :, base_qkv_dim:]
        gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
    elif self.elementwise_attn_output_gate:
        gate_dim = self.num_heads * self.head_dim
        qkv_base = qkv_states[:, :, :base_qkv_dim]
        gate_score_raw = qkv_states[:, :, base_qkv_dim:]
        gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
    else:
        qkv_base = qkv_states
        gate_score = None
    
    # 标准 qkv 处理
    qkv_states = rearrange(
        qkv_base,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )
    
    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]
    
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # ... RoPE, past_key_value, repeat_kv 等 ...
    
    # Attention 计算
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)  # [bsz, num_heads, q_len, head_dim]
    
    # Phase 1: 应用 gating
    if gate_score is not None:
        gate_score = gate_score.transpose(1, 2)  # [bsz, num_heads, q_len, 1] or [bsz, num_heads, q_len, head_dim]
        attn_output = attn_output * torch.sigmoid(gate_score)
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    
    attn_output = self.wo(attn_output)
    
    if not output_attentions:
        attn_weights = None
    
    return attn_output, attn_weights, past_key_value
```

## Flash Attention 2 的兼容性

### 修改 `InternLM2FlashAttention2`

Flash Attention 2 也需要类似的修改：

```python
class InternLM2FlashAttention2(InternLM2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # ... 现有代码 ...
        
        bsz, q_len, _ = hidden_states.size()
        
        qkv_states = self.wqkv(hidden_states)
        
        # Phase 1: 分离 gate_score（与标准 attention 相同）
        base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        
        if self.headwise_attn_output_gate:
            gate_dim = self.num_heads
            qkv_base = qkv_states[:, :, :base_qkv_dim]
            gate_score_raw = qkv_states[:, :, base_qkv_dim:]
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            gate_dim = self.num_heads * self.head_dim
            qkv_base = qkv_states[:, :, :base_qkv_dim]
            gate_score_raw = qkv_states[:, :, base_qkv_dim:]
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
        else:
            qkv_base = qkv_states
            gate_score = None
        
        # 标准 qkv 处理
        qkv_states = rearrange(
            qkv_base,
            'b q (h gs d) -> b q h gs d',
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )
        
        # ... 后续处理 ...
        
        # Flash Attention 计算
        attn_output = self._flash_attention_forward(...)
        
        # Phase 1: 应用 gating（在 reshape 之前）
        if gate_score is not None:
            # attn_output: [bsz, q_len, num_heads, head_dim]
            # gate_score: [bsz, q_len, num_heads, 1] or [bsz, q_len, num_heads, head_dim]
            attn_output = attn_output * torch.sigmoid(gate_score)
        
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.wo(attn_output)
        
        # ... 返回 ...
```

## 关键实现细节

### 1. 维度计算

- **标准模式**：`wqkv` 输出 `(num_heads + 2 * num_key_value_heads) * head_dim`
- **Head-wise**：增加 `num_heads` 个维度
- **Element-wise**：增加 `num_heads * head_dim` 个维度

### 2. Gate Score 形状

- **Head-wise**：`[bsz, q_len, num_heads, 1]` → transpose → `[bsz, num_heads, q_len, 1]`
- **Element-wise**：`[bsz, q_len, num_heads, head_dim]` → transpose → `[bsz, num_heads, q_len, head_dim]`

### 3. 应用时机

- 在 `attn_output = torch.matmul(attn_weights, value_states)` 之后
- 在 `transpose(1, 2)` 和 `reshape` 之前
- 确保 gate_score 的形状与 attn_output 匹配

## 测试要点

1. **形状验证**：确保所有 tensor 的形状正确
2. **GQA 兼容性**：验证与 Grouped Query Attention 的兼容性
3. **Flash Attention 兼容性**：确保 Flash Attention 2 也能正常工作
4. **梯度检查**：确保梯度可以正常反向传播
5. **内存占用**：检查额外的维度是否显著增加内存

## 实施步骤

1. ✅ 在 `InternLM2Config` 中添加 gating 配置参数
2. ✅ 修改 `InternLM2Attention.__init__` 调整 `wqkv` 维度
3. ✅ 修改 `InternLM2Attention.forward` 分离和应用 gate_score
4. ✅ 修改 `InternLM2FlashAttention2.forward`（如果使用 Flash Attention）
5. ✅ 单元测试：验证形状和计算正确性
6. ✅ 集成测试：在小规模数据上测试

## 预期影响

- **参数增加**：
  - Head-wise: `hidden_size * num_heads` 额外参数
  - Element-wise: `hidden_size * num_heads * head_dim` 额外参数
- **计算开销**：增加 sigmoid 计算，但开销很小（< 1%）
- **内存占用**：增加 gate_score 的存储，但可以忽略不计


