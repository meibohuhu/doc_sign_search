# wqkv 详解

## 什么是 wqkv？

`wqkv` 是 InternLM2 中用于**统一投影 Query、Key、Value** 的线性层（Linear Layer）。

### 命名含义

- **w** = weight（权重矩阵）
- **q** = Query（查询向量）
- **k** = Key（键向量）
- **v** = Value（值向量）

`wqkv` 表示"将 Query、Key、Value 的投影合并到一个线性层中"。

## 核心概念

### 标准 Transformer 的分离投影

在标准的 Transformer 架构中（如 GPT、BERT），通常使用**三个独立的线性层**：

```python
# 标准方式（分离投影）
self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)  # Query 投影
self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim)  # Key 投影
self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim)  # Value 投影

# 前向传播
query = self.q_proj(hidden_states)
key = self.k_proj(hidden_states)
value = self.v_proj(hidden_states)
```

### InternLM2 的统一投影（wqkv）

InternLM2 使用**一个统一的线性层**来同时计算 Q、K、V：

```python
# InternLM2 方式（统一投影）
self.wqkv = nn.Linear(
    hidden_size,
    (num_heads + 2 * num_key_value_heads) * head_dim,  # Q + K + V 的总维度
    bias=config.bias,
)

# 前向传播
qkv_states = self.wqkv(hidden_states)  # 一次性得到 Q、K、V 的拼接结果
# 然后通过 rearrange 或 split 分离出 Q、K、V
```

## 为什么使用 wqkv？

### 优势

1. **计算效率更高**：
   - 只需要一次矩阵乘法（`hidden_states @ W_qkv`）
   - 而不是三次（`hidden_states @ W_q`, `hidden_states @ W_k`, `hidden_states @ W_v`）
   - 减少 GPU 内存访问次数

2. **参数共享**：
   - 虽然参数数量相同，但计算路径更统一
   - 便于优化和并行化

3. **内存效率**：
   - 中间结果可以更紧凑地存储
   - 减少内存碎片

### 劣势

1. **灵活性稍低**：
   - 如果需要单独调整 Q、K、V 的投影，不如分离式方便
   - 但在大多数情况下这不是问题

## wqkv 的维度计算

### 标准模式（无 Gating）

```python
# 假设配置
hidden_size = 2048
num_heads = 16
num_key_value_heads = 4  # GQA: Grouped Query Attention
head_dim = hidden_size // num_heads = 128

# wqkv 输出维度
base_qkv_dim = (num_heads + 2 * num_key_value_heads) * head_dim
              = (16 + 2 * 4) * 128
              = 24 * 128
              = 3072

# 分解：
# - Q 部分: num_heads * head_dim = 16 * 128 = 2048
# - K 部分: num_key_value_heads * head_dim = 4 * 128 = 512
# - V 部分: num_key_value_heads * head_dim = 4 * 128 = 512
# 总计: 2048 + 512 + 512 = 3072
```

### 带 Gating 的模式

#### Head-wise Gating

```python
# wqkv 输出维度
total_dim = base_qkv_dim + gate_dim
          = 3072 + num_heads
          = 3072 + 16
          = 3088

# 分解：
# - QKV 部分: 3072
# - gate_score 部分: 16 (每个 head 一个 gate 值)
```

#### Element-wise Gating

```python
# wqkv 输出维度
total_dim = base_qkv_dim + gate_dim
          = 3072 + (num_heads * head_dim)
          = 3072 + (16 * 128)
          = 3072 + 2048
          = 5120

# 分解：
# - QKV 部分: 3072
# - gate_score 部分: 2048 (每个 head 的每个维度一个 gate 值)
```

## wqkv 的工作流程

### 1. 初始化

```python
# modeling_internlm2.py, line 349
self.wqkv = nn.Linear(
    self.hidden_size,           # 输入维度: 2048
    total_dim,                  # 输出维度: 3072 (标准) 或 3088/5120 (Gating)
    bias=config.bias,
)
```

### 2. 前向传播 - 投影

```python
# 输入
hidden_states: [batch_size, seq_len, hidden_size]  # [2, 512, 2048]

# wqkv 投影
qkv_states = self.wqkv(hidden_states)
# 输出: [batch_size, seq_len, total_dim]  # [2, 512, 3072]
```

### 3. 分离 Q、K、V（标准模式）

```python
# 使用 einops.rearrange 重塑
qkv_states = rearrange(
    qkv_states,
    'b q (h gs d) -> b q h gs d',
    gs=2 + self.num_key_value_groups,  # gs = 2 + 4 = 6 (Q有4组, K有1组, V有1组)
    d=self.head_dim,                    # d = 128
)
# 输出: [batch_size, seq_len, num_heads, 6, head_dim]  # [2, 512, 16, 6, 128]

# 提取 Q、K、V
query_states = qkv_states[..., :self.num_key_value_groups, :]  # [2, 512, 16, 4, 128]
query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')  # [2, 512, 64, 128]
key_states = qkv_states[..., -2, :]  # [2, 512, 4, 128]
value_states = qkv_states[..., -1, :]  # [2, 512, 4, 128]
```

### 4. 分离 Q、K、V + gate_score（Gating 模式）

```python
# 先分离 gate_score
base_qkv_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
qkv_base = qkv_states[:, :, :base_qkv_dim]  # [2, 512, 3072]
gate_score_raw = qkv_states[:, :, base_qkv_dim:]  # [2, 512, 16] (Head-wise) 或 [2, 512, 2048] (Element-wise)

# 然后对 qkv_base 进行标准处理（同上）
# 对 gate_score_raw 进行 reshape
if headwise:
    gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)  # [2, 512, 16, 1]
else:  # elementwise
    gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)  # [2, 512, 16, 128]
```

## 与 Qwen3 的对比

| 特性 | Qwen3 | InternLM2 |
|------|-------|-----------|
| **投影方式** | 分离的 `q_proj`, `k_proj`, `v_proj` | 统一的 `wqkv` |
| **计算次数** | 3 次矩阵乘法 | 1 次矩阵乘法 |
| **灵活性** | 高（可单独调整） | 中（需要 reshape 分离） |
| **效率** | 标准 | 更高（减少内存访问） |

## 代码位置

### 定义位置

```python
# InternVL/internvl_chat/internvl/model/internlm2/modeling_internlm2.py
# Line 349-353
self.wqkv = nn.Linear(
    self.hidden_size,
    total_dim,
    bias=config.bias,
)
```

### 使用位置

```python
# InternVL/internvl_chat/internvl/model/internlm2/modeling_internlm2.py
# Line ~420 (forward 方法中)
qkv_states = self.wqkv(hidden_states)
```

## 总结

**wqkv 是一个统一的线性投影层，它将 Query、Key、Value 的投影合并到一个矩阵乘法中，提高了计算效率，是 InternLM2 架构的一个优化设计。**

在 Gated Attention 中，`wqkv` 的输出维度会增加，以包含 `gate_score` 部分，但核心思想不变：**一次投影，然后分离使用**。


