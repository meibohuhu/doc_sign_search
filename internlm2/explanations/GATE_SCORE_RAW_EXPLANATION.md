# gate_score_raw 计算详解

## 什么是 gate_score_raw？

`gate_score_raw` 是**从 `wqkv` 线性层输出中提取的原始 gate 分数**，在应用 sigmoid 激活之前的值。

## 关键问题：gate_score_raw 和 qkv_states 都来自 wqkv，有什么区别？

虽然 `gate_score_raw` 和 `qkv_states` 都调用同一个 `self.wqkv` 线性层，但它们有**三个重要区别**：

### 区别 1: 输入不同（Flash Attention 2 模式）

```python
# Flash Attention 2 中
if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
    # gate_score 使用归一化后的 hidden_states
    hidden_states_normalized = self.gate_norm(hidden_states)
    gate_qkv_states = self.wqkv(hidden_states_normalized)  # ← 归一化输入
    
    # 标准 QKV 使用原始 hidden_states
    qkv_states = self.wqkv(hidden_states)  # ← 原始输入
```

**为什么不同？**
- **gate_score** 需要稳定的输入范围，归一化后计算更稳定
- **标准 QKV** 保持原有行为，使用原始输入

**注意**：Eager Attention 中两者都使用原始输入（归一化被注释掉了）

### 区别 2: 提取的输出维度段不同

`wqkv` 的输出维度是 `[base_qkv_dim + gate_dim]`，被分成两部分：

```python
# wqkv 输出: [bsz, q_len, base_qkv_dim + gate_dim]
#            [2, 512, 3072 + 16]  (Head-wise 示例)

# 从 qkv_states 提取标准 QKV（前半部分）
qkv_base = qkv_states[:, :, :base_qkv_dim]
#         = qkv_states[:, :, :3072]  ← 前 3072 个维度

# 从 gate_qkv_states 提取 gate_score（后半部分）
gate_score_raw = gate_qkv_states[:, :, base_qkv_dim:]
#                = gate_qkv_states[:, :, 3072:]  ← 后 16 个维度
```

**可视化**：
```
wqkv 输出维度布局：
┌─────────────────────────────────────────────────────────┐
│  [0 : base_qkv_dim]      │  [base_qkv_dim : total_dim]  │
│  ← 标准 QKV 部分          │  ← gate_score 部分          │
│  (3072 维)                │  (16 维，Head-wise)         │
└─────────────────────────────────────────────────────────┘
         ↑                              ↑
    qkv_states 提取               gate_qkv_states 提取
```

### 区别 3: 用途不同

- **qkv_states → qkv_base**：用于标准的 Attention 计算（Q @ K^T, softmax, @ V）
- **gate_qkv_states → gate_score_raw**：用于控制 Attention 输出的 gating（`attn_output * sigmoid(gate_score)`）

### 为什么 wqkv 要包含两部分？

`wqkv` 的权重矩阵 `W_qkv` 实际上包含两部分：

```python
# wqkv.weight 的形状: [total_dim, hidden_size]
#                      [3072 + 16, 2048]  (Head-wise 示例)

# 内部结构：
W_qkv = [
    W_qkv_base,  # 前 base_qkv_dim 行: [3072, 2048] - 用于标准 QKV
    W_gate,      # 后 gate_dim 行: [16, 2048] - 用于 gate_score
]
```

**设计原因**：
- 共享输入特征（`hidden_states`），但输出不同用途
- 一次矩阵乘法得到两部分，效率更高
- 两部分可以独立学习（通过不同的权重行）

### 完整对比表

| 特性 | qkv_states | gate_qkv_states |
|------|------------|-----------------|
| **输入** | 原始 `hidden_states` | 归一化后的 `hidden_states_normalized` (Flash) 或原始 `hidden_states` (Eager) |
| **wqkv 调用** | `wqkv(hidden_states)` | `wqkv(hidden_states_normalized)` 或 `wqkv(hidden_states)` |
| **提取维度** | `[:, :, :base_qkv_dim]` (前半部分) | `[:, :, base_qkv_dim:]` (后半部分) |
| **输出形状** | `[bsz, q_len, base_qkv_dim]` | `[bsz, q_len, gate_dim]` |
| **用途** | 标准 Attention 计算 (Q, K, V) | Gating 控制 (gate_score) |
| **后续处理** | 分离为 Q, K, V，计算 Attention | 应用 sigmoid，用于 gating |

### 代码示例对比

```python
# Flash Attention 2 模式
if self.headwise_attn_output_gate:
    # 1. 归一化（仅用于 gate_score）
    hidden_states_normalized = self.gate_norm(hidden_states)
    
    # 2. 两次 wqkv 调用
    gate_qkv_states = self.wqkv(hidden_states_normalized)  # 归一化输入
    qkv_states = self.wqkv(hidden_states)                  # 原始输入
    
    # 3. 从不同位置提取
    base_qkv_dim = 3072  # 示例
    gate_dim = 16        # Head-wise 示例
    
    qkv_base = qkv_states[:, :, :base_qkv_dim]           # 前 3072 维 → QKV
    gate_score_raw = gate_qkv_states[:, :, base_qkv_dim:] # 后 16 维 → gate
    
    # 4. 不同用途
    # qkv_base → 用于 Attention 计算
    # gate_score_raw → 用于 Gating 控制
```

## 计算流程

### 整体流程

```
hidden_states
    │
    ├─→ (可选) LayerNorm → hidden_states_normalized
    │                           │
    │                           ▼
    └─→ wqkv 线性层 ────────────→ gate_qkv_states [bsz, q_len, base_qkv_dim + gate_dim]
                                        │
                                        ├─→ [:base_qkv_dim] → qkv_base (标准 QKV)
                                        │
                                        └─→ [base_qkv_dim:] → gate_score_raw
                                                                    │
                                                                    ├─→ (可选) * gate_scale
                                                                    │
                                                                    └─→ reshape → gate_score
```

## 详细步骤

### Step 1: 输入准备

#### Eager Attention 模式（当前实现）

```python
# modeling_internlm2_gate.py, line 415-417
if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
    # 直接使用原始 hidden_states（没有 LayerNorm）
    gate_qkv_states = self.wqkv(hidden_states)
    qkv_states = self.wqkv(hidden_states)  # 标准 QKV 也用原始 hidden_states
```

**注意**：当前 Eager Attention 实现中，`gate_norm` 和 `gate_scale` 都被设置为 `None`（line 334-335），所以：
- 不使用 LayerNorm
- 不使用缩放因子

#### Flash Attention 2 模式（完整实现）

```python
# modeling_internlm2_gate.py, line 603-609
if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
    # Step 1.1: 归一化 hidden_states（用于计算 gate_score）
    hidden_states_normalized = self.gate_norm(hidden_states)
    
    # Step 1.2: 使用归一化后的 hidden_states 计算 gate_score
    gate_qkv_states = self.wqkv(hidden_states_normalized)
    
    # Step 1.3: 使用原始 hidden_states 计算标准 QKV
    qkv_states = self.wqkv(hidden_states)
```

**关键点**：
- `gate_score` 从**归一化后的** `hidden_states` 计算（更稳定）
- 标准 QKV 从**原始** `hidden_states` 计算（保持原有行为）

### Step 2: wqkv 线性投影

```python
# wqkv 的定义（__init__ 中）
self.wqkv = nn.Linear(
    self.hidden_size,      # 输入: 2048
    total_dim,             # 输出: base_qkv_dim + gate_dim
    bias=config.bias,
)

# 前向传播
gate_qkv_states = self.wqkv(hidden_states_normalized)
# 输出形状: [bsz, q_len, base_qkv_dim + gate_dim]
```

**数学表示**：
```
gate_qkv_states = hidden_states_normalized @ W_qkv^T + b_qkv
```

其中：
- `W_qkv`: 权重矩阵 `[hidden_size, base_qkv_dim + gate_dim]`
- `b_qkv`: 偏置向量 `[base_qkv_dim + gate_dim]`（如果 `bias=True`）

### Step 3: 提取 gate_score_raw

```python
# modeling_internlm2_gate.py, line 432-434 (Eager) 或 656-658 (Flash)
base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim

if self.headwise_attn_output_gate:
    gate_dim = self.num_heads
else:  # elementwise
    gate_dim = self.num_heads * self.head_dim

# 提取 gate_score_raw（从 gate_qkv_states 的后 gate_dim 个维度）
gate_score_raw = gate_qkv_states[:, :, base_qkv_dim:]
# 形状: [bsz, q_len, gate_dim]
```

**维度示例**（假设 `hidden_size=2048`, `num_heads=16`, `num_key_value_heads=4`, `head_dim=128`）：

```python
# gate_qkv_states 形状: [bsz, q_len, 3072 + gate_dim]
#                        [2, 512, 3072 + 16]  (Head-wise)
#                        或
#                        [2, 512, 3072 + 2048]  (Element-wise)

# gate_score_raw 形状:
# Head-wise:   [2, 512, 16]
# Element-wise: [2, 512, 2048]
```

### Step 4: 应用缩放因子（仅 Flash Attention 2）

```python
# modeling_internlm2_gate.py, line 660 (Flash Attention 2)
gate_score_raw = gate_score_raw * self.gate_scale
# gate_scale 通常是 0.1，用于控制 gate_score 的幅度
```

**作用**：
- 减小 `gate_score_raw` 的幅度，使其更接近 0
- 这样 `sigmoid(gate_score_raw)` 会更接近 0.5（中性 gating）
- 避免初始时 gate 值过大或过小

**注意**：Eager Attention 中这步被注释掉了（line 436），所以不应用缩放。

### Step 5: Reshape 为 gate_score

```python
# modeling_internlm2_gate.py, line 447-450 (Eager) 或 675-678 (Flash)
if self.headwise_attn_output_gate:
    gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
    # 形状: [bsz, q_len, num_heads, 1]
else:  # elementwise
    gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
    # 形状: [bsz, q_len, num_heads, head_dim]
```

## 完整代码流程对比

### Eager Attention（当前实现）

```python
# Line 411-434
if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
    # 1. 直接使用原始 hidden_states
    gate_qkv_states = self.wqkv(hidden_states)
    qkv_states = self.wqkv(hidden_states)
    
    # 2. 分离
    base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
    if self.headwise_attn_output_gate:
        gate_dim = self.num_heads
    else:
        gate_dim = self.num_heads * self.head_dim
    
    qkv_base = qkv_states[:, :, :base_qkv_dim]
    gate_score_raw = gate_qkv_states[:, :, base_qkv_dim:]  # 提取
    
    # 3. 不应用缩放（被注释掉）
    # gate_score_raw = gate_score_raw * self.gate_scale
    
    # 4. Reshape
    if self.headwise_attn_output_gate:
        gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
    else:
        gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
```

### Flash Attention 2（完整实现）

```python
# Line 603-678
if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
    # 1. 归一化 hidden_states（用于 gate_score）
    hidden_states_normalized = self.gate_norm(hidden_states)
    
    # 2. 分别计算
    gate_qkv_states = self.wqkv(hidden_states_normalized)  # 归一化后
    qkv_states = self.wqkv(hidden_states)  # 原始
    
    # 3. 分离
    base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
    if self.headwise_attn_output_gate:
        gate_dim = self.num_heads
    else:
        gate_dim = self.num_heads * self.head_dim
    
    qkv_base = qkv_states[:, :, :base_qkv_dim]
    gate_score_raw = gate_qkv_states[:, :, base_qkv_dim:]  # 提取
    
    # 4. 应用缩放因子
    gate_score_raw = gate_score_raw * self.gate_scale  # 例如 0.1
    
    # 5. Reshape
    if self.headwise_attn_output_gate:
        gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
    else:
        gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
```

## 数学公式

### 完整计算链

```
gate_score_raw = (gate_norm(hidden_states) @ W_qkv^T + b_qkv)[:, :, base_qkv_dim:]
                = (normalize(hidden_states) @ W_gate^T + b_gate)
```

其中：
- `W_gate` 是 `W_qkv` 的后 `gate_dim` 列
- `b_gate` 是 `b_qkv` 的后 `gate_dim` 个元素

### 应用缩放后

```
gate_score_raw_scaled = gate_score_raw * gate_scale
```

### 最终 gate_score

```
gate_score = reshape(gate_score_raw_scaled)
```

### 应用 gating

```
attn_output_gated = attn_output * sigmoid(gate_score)
```

## 关键点总结

1. **`gate_score_raw` 的来源**：
   - 从 `wqkv` 线性层的输出中提取
   - 位于输出的后 `gate_dim` 个维度

2. **计算方式**：
   - **Eager Attention**：直接从原始 `hidden_states` 计算
   - **Flash Attention 2**：从归一化后的 `hidden_states` 计算（更稳定）

3. **维度**：
   - **Head-wise**：`[bsz, q_len, num_heads]` → reshape → `[bsz, q_len, num_heads, 1]`
   - **Element-wise**：`[bsz, q_len, num_heads * head_dim]` → reshape → `[bsz, q_len, num_heads, head_dim]`

4. **缩放因子**：
   - Flash Attention 2 中使用 `gate_scale = 0.1` 来控制幅度
   - Eager Attention 中不使用（被注释掉）

5. **与标准 QKV 的区别**：
   - 标准 QKV 从**原始** `hidden_states` 计算
   - `gate_score_raw` 从**归一化后的** `hidden_states` 计算（Flash Attention 2）

## 代码位置

- **定义**：`modeling_internlm2_gate.py`, line 351-355 (`wqkv` 初始化)
- **计算**：`modeling_internlm2_gate.py`, line 415-434 (Eager) 或 605-660 (Flash)
- **提取**：`modeling_internlm2_gate.py`, line 434 (Eager) 或 658 (Flash)
- **应用**：`modeling_internlm2_gate.py`, line 524-528 (Eager) 或 730-734 (Flash)


