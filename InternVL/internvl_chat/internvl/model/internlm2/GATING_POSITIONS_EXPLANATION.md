# Gating Positions Implementation Proposal

## 当前实现：G1 (after SDPA)

当前实现在 SDPA (Scaled Dot-Product Attention) 输出后应用 gating：
```python
# Line 509: SDPA
attn_output = torch.matmul(attn_weights, value_states)

# Line 558: G1 - Apply gating after SDPA
attn_output = attn_output * gate_sigmoid
```

## 实现方案

### 1. 修改 Config 添加 Gating 位置选择

在 `configuration_internlm2.py` 中添加：

```python
# Gating position: 'G1', 'G2', 'G3', 'G4', 'G5', or None
gating_position: Optional[str] = field(
    default=None,
    metadata={'help': 'Gating position: G1 (after SDPA), G2 (after Q), G3 (after K), G4 (after V), G5 (after wo)'}
)
```

### 2. G2: Gate on Query States (after Q projection)

**位置**: 在 Q projection 之后、rotary embedding 之前或之后应用

**实现要点**:
- Gate 维度需要匹配 query_states: `[bsz, num_heads, q_len, head_dim]`
- 应用在 line 472 之后（transpose后）

```python
# After line 472: query_states = query_states.transpose(1, 2)
# Shape: [bsz, num_heads, q_len, head_dim]

if self.gating_position == 'G2':
    # gate_score shape: [bsz, q_len, num_heads, head_dim] or [bsz, q_len, num_heads, 1]
    # Need to transpose to match query_states shape
    if gate_score is not None:
        gate_score_q = gate_score.transpose(1, 2)  # [bsz, num_heads, q_len, gate_dim]
        
        # Expand if headwise gating (dim=1) to match head_dim
        if gate_score_q.shape[-1] == 1:
            gate_score_q = gate_score_q.expand(-1, -1, -1, self.head_dim)
        
        # Combine with visual gate if available
        if g_vis is not None:
            g_vis_q = g_vis.expand(-1, q_len, -1, -1).transpose(1, 2)  # [B, num_heads, q_len, 1]
            if self.visual_gate_mode == 'add_logits':
                gate_score_q = gate_score_q + g_vis_q.expand(-1, -1, -1, self.head_dim)
            # ... other visual gate modes
        
        gate_sigmoid_q = torch.sigmoid(gate_score_q)
        query_states = query_states * gate_sigmoid_q
```

**注意**: Gate 在 rotary embedding 之前或之后都可以，建议在之后（line 480 之后）以保证形状匹配。

### 3. G3: Gate on Key States (after K projection)

**位置**: 在 K projection 之后应用

**实现要点**:
- Gate 维度需要匹配 key_states: `[bsz, num_key_value_heads, kv_seq_len, head_dim]`
- 注意 key_states 的 head 数可能小于 query_states (GQA)

```python
# After line 473: key_states = key_states.transpose(1, 2)
# Shape: [bsz, num_key_value_heads, kv_seq_len, head_dim]

if self.gating_position == 'G3':
    if gate_score is not None:
        # Gate score was computed for num_heads, but key_states has num_key_value_heads
        # Need to handle GQA case: select first num_key_value_heads or average
        gate_score_k = gate_score.transpose(1, 2)  # [bsz, num_heads, q_len, gate_dim]
        
        # For GQA: key has fewer heads, take first num_key_value_heads
        if self.num_key_value_heads < self.num_heads:
            gate_score_k = gate_score_k[:, :self.num_key_value_heads, :, :]
        
        # Match sequence length (q_len vs kv_seq_len)
        if gate_score_k.shape[2] != key_states.shape[2]:
            # If different seq lengths, need to handle appropriately
            # For simplicity, assume same length or use first kv_seq_len elements
            gate_score_k = gate_score_k[:, :, :key_states.shape[2], :]
        
        # Expand if headwise
        if gate_score_k.shape[-1] == 1:
            gate_score_k = gate_score_k.expand(-1, -1, -1, self.head_dim)
        
        # Combine with visual gate
        if g_vis is not None:
            g_vis_k = g_vis[:, :, :self.num_key_value_heads, :]  # [B, 1, num_key_value_heads, 1]
            g_vis_k = g_vis_k.expand(-1, key_states.shape[2], -1, -1).transpose(1, 2)
            if self.visual_gate_mode == 'add_logits':
                gate_score_k = gate_score_k + g_vis_k.expand(-1, -1, -1, self.head_dim)
        
        gate_sigmoid_k = torch.sigmoid(gate_score_k)
        key_states = key_states * gate_sigmoid_k
```

### 4. G4: Gate on Value States (after V projection)

**位置**: 在 V projection 之后应用

**实现要点**:
- 类似 G3，但应用于 value_states
- 注意处理 GQA 情况

```python
# After line 474: value_states = value_states.transpose(1, 2)
# Shape: [bsz, num_key_value_heads, kv_seq_len, head_dim]

if self.gating_position == 'G4':
    if gate_score is not None:
        gate_score_v = gate_score.transpose(1, 2)  # [bsz, num_heads, q_len, gate_dim]
        
        # Handle GQA
        if self.num_key_value_heads < self.num_heads:
            gate_score_v = gate_score_v[:, :self.num_key_value_heads, :, :]
        
        # Match sequence length
        if gate_score_v.shape[2] != value_states.shape[2]:
            gate_score_v = gate_score_v[:, :, :value_states.shape[2], :]
        
        # Expand if headwise
        if gate_score_v.shape[-1] == 1:
            gate_score_v = gate_score_v.expand(-1, -1, -1, self.head_dim)
        
        # Combine with visual gate
        if g_vis is not None:
            g_vis_v = g_vis[:, :, :self.num_key_value_heads, :]
            g_vis_v = g_vis_v.expand(-1, value_states.shape[2], -1, -1).transpose(1, 2)
            if self.visual_gate_mode == 'add_logits':
                gate_score_v = gate_score_v + g_vis_v.expand(-1, -1, -1, self.head_dim)
        
        gate_sigmoid_v = torch.sigmoid(gate_score_v)
        value_states = value_states * gate_sigmoid_v
```

### 5. G5: Gate after Output Projection (after wo)

**位置**: 在 `self.wo()` 投影之后应用

**实现要点**:
- Gate 维度需要匹配最终输出: `[bsz, q_len, hidden_size]`
- 需要 reshape gate_score 来匹配

```python
# After line 575: attn_output = self.wo(attn_output)
# Shape: [bsz, q_len, hidden_size]

if self.gating_position == 'G5':
    if gate_score is not None:
        # gate_score shape: [bsz, q_len, num_heads, gate_dim]
        # Need to reshape to [bsz, q_len, hidden_size]
        
        if self.headwise_attn_output_gate:
            # [bsz, q_len, num_heads, 1] -> [bsz, q_len, num_heads]
            gate_score_flat = gate_score.squeeze(-1)  # [bsz, q_len, num_heads]
            # Expand each head gate to head_dim: [bsz, q_len, num_heads] -> [bsz, q_len, num_heads * head_dim]
            gate_score_expanded = gate_score_flat.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            gate_score_5 = gate_score_expanded.reshape(bsz, q_len, self.hidden_size)
        else:  # elementwise
            # [bsz, q_len, num_heads, head_dim] -> [bsz, q_len, hidden_size]
            gate_score_5 = gate_score.reshape(bsz, q_len, self.hidden_size)
        
        # Combine with visual gate
        if g_vis is not None:
            # g_vis: [B, 1, num_heads, 1]
            if self.headwise_attn_output_gate:
                g_vis_5 = g_vis.squeeze(1).squeeze(-1)  # [B, num_heads]
                g_vis_5 = g_vis_5.unsqueeze(1).expand(-1, q_len, -1)  # [B, q_len, num_heads]
                g_vis_5 = g_vis_5.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  # [B, q_len, num_heads, head_dim]
                g_vis_5 = g_vis_5.reshape(bsz, q_len, self.hidden_size)
            else:
                g_vis_5 = g_vis.squeeze(1).expand(-1, q_len, -1, -1)  # [B, q_len, num_heads, head_dim]
                g_vis_5 = g_vis_5.reshape(bsz, q_len, self.hidden_size)
            
            if self.visual_gate_mode == 'add_logits':
                gate_score_5 = gate_score_5 + g_vis_5
        
        gate_sigmoid_5 = torch.sigmoid(gate_score_5)
        attn_output = attn_output * gate_sigmoid_5
```

## 代码结构修改建议

### 1. 在 `__init__` 中添加 gating position 配置

```python
def __init__(self, config: InternLM2Config):
    # ... existing code ...
    
    # Gating position configuration
    self.gating_position = getattr(config, 'gating_position', None)  # 'G1', 'G2', 'G3', 'G4', 'G5', or None
    
    # Adjust wqkv dimension based on gating position and mode
    base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
    
    # For G2, G3, G4: gate is applied to Q/K/V separately, so gate_dim needs to match each
    # For G1, G5: gate is applied to attention output, same as current implementation
    if self.gating_position in ['G2', 'G3', 'G4', 'G1', 'G5']:
        if self.headwise_attn_output_gate:
            gate_dim = self.num_heads
        elif self.elementwise_attn_output_gate:
            gate_dim = self.num_heads * self.head_dim
        else:
            gate_dim = 0
        total_dim = base_qkv_dim + gate_dim
    else:
        total_dim = base_qkv_dim
    
    self.wqkv = nn.Linear(self.hidden_size, total_dim, bias=config.bias)
```

### 2. 在 forward 中实现不同位置的 gating

需要将当前的 G1 gating 代码提取为函数，并在不同位置调用：

```python
def _apply_gating(self, gate_score, target_tensor, target_shape, g_vis=None, gating_position='G1'):
    """
    Apply gating to target tensor based on gate_score and gating position.
    
    Args:
        gate_score: Raw gate scores [bsz, q_len, num_heads, gate_dim]
        target_tensor: Tensor to apply gating to
        target_shape: Expected shape of target tensor after transpose if needed
        g_vis: Visual gate [B, 1, num_heads, 1]
        gating_position: 'G1', 'G2', 'G3', 'G4', 'G5'
    """
    if gate_score is None:
        return target_tensor
    
    # ... implementation based on gating_position ...
```

## 关键注意事项

1. **GQA (Grouped Query Attention) 处理**: G2, G3, G4 需要特别处理，因为 key/value 的 head 数可能少于 query
2. **Sequence Length 匹配**: G3, G4 中 gate_score 的序列长度需要与 key/value 的序列长度匹配（考虑 past_key_values）
3. **Shape 变换**: 不同位置的 gating 需要不同的 shape 变换
4. **Visual Gate 兼容性**: 所有位置都需要支持 visual gate 的融合
5. **Flash Attention 兼容性**: InternLM2FlashAttention2 也需要实现相同的 gating 位置

## 实现顺序建议

1. 先实现 G5（最简单，只需要 reshape）
2. 然后实现 G2（相对简单）
3. 接着实现 G3, G4（需要处理 GQA）
4. 最后确保 Flash Attention 版本也支持所有位置



