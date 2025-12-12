# Phase 1: Gated Attention 框架图 Proposal

## 整体架构图

### 方案 1: 详细流程图（推荐）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    InternLM2Attention with Gating                        │
└─────────────────────────────────────────────────────────────────────────┘

输入: hidden_states [bsz, q_len, hidden_size]
  │
  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 1: wqkv 投影                                                         │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ wqkv: Linear(hidden_size → base_qkv_dim + gate_dim)         │       │
│  │                                                               │       │
│  │ 标准模式: base_qkv_dim = (num_heads + 2*num_key_value_heads)*head_dim│
│  │ Head-wise: + num_heads 维度                                    │       │
│  │ Element-wise: + num_heads * head_dim 维度                     │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│         qkv_states [bsz, q_len, base_qkv_dim + gate_dim]                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 2: 分离 gate_score 和标准 qkv                                       │
│                                                                           │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐   │
│  │ qkv_base                      │  │ gate_score_raw               │   │
│  │ [bsz, q_len, base_qkv_dim]    │  │ [bsz, q_len, gate_dim]        │   │
│  └──────────────────────────────┘  └──────────────────────────────┘   │
│           │                                    │                         │
│           ▼                                    ▼                         │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐   │
│  │ rearrange(...)                 │  │ Reshape gate_score            │   │
│  │ 'b q (h gs d) -> b q h gs d'  │  │                               │   │
│  └──────────────────────────────┘  │ Head-wise:                    │   │
│           │                         │   view(..., num_heads, 1)      │   │
│           ▼                         │ Element-wise:                 │   │
│  qkv_states [bsz, q_len, h, gs, d]  │   view(..., num_heads, head_dim)│   │
│                                     └──────────────────────────────┘   │
│                                              │                           │
│                                              ▼                           │
│                                     gate_score                           │
│                                     [bsz, q_len, num_heads, ...]        │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 3: 标准 Attention 计算                                               │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ 提取 query, key, value                                        │       │
│  │ query_states: [bsz, num_heads, q_len, head_dim]              │       │
│  │ key_states:   [bsz, num_key_value_heads, q_len, head_dim]    │       │
│  │ value_states: [bsz, num_key_value_heads, q_len, head_dim]   │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ RoPE (Rotary Position Embedding)                            │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ Attention Weights 计算                                        │       │
│  │ attn_weights = softmax(Q @ K^T / sqrt(head_dim))            │       │
│  │ [bsz, num_heads, q_len, kv_len]                              │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ Attention Output                                              │       │
│  │ attn_output = attn_weights @ value_states                     │       │
│  │ [bsz, num_heads, q_len, head_dim]                            │       │
│  └──────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 4: 应用 Gating (关键步骤)                                            │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ gate_score 转置                                              │       │
│  │ [bsz, q_len, num_heads, ...]                                │       │
│  │         ↓ transpose(1, 2)                                   │       │
│  │ [bsz, num_heads, q_len, ...]                                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ Sigmoid 激活                                                 │       │
│  │ sigmoid(gate_score) → [0, 1]                                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ Element-wise 乘法                                           │       │
│  │ attn_output = attn_output * sigmoid(gate_score)             │       │
│  │                                                               │       │
│  │ Head-wise:  广播到 [bsz, num_heads, q_len, head_dim]        │       │
│  │ Element-wise: 直接匹配 [bsz, num_heads, q_len, head_dim]     │       │
│  └──────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Step 5: 输出投影                                                         │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ Reshape & Transpose                                          │       │
│  │ [bsz, num_heads, q_len, head_dim]                           │       │
│  │         ↓ transpose(1, 2)                                   │       │
│  │ [bsz, q_len, num_heads, head_dim]                           │       │
│  │         ↓ reshape                                            │       │
│  │ [bsz, q_len, hidden_size]                                   │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │ wo: Linear(hidden_size → hidden_size)                       │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                              │                                           │
│                              ▼                                           │
│              输出: [bsz, q_len, hidden_size]                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 维度变化图

### Head-wise Gating 模式

```
输入维度流程:
─────────────────────────────────────────────────────────────────────────
hidden_states:     [bsz, q_len, hidden_size]
                              │
                              ▼
wqkv 输出:         [bsz, q_len, base_qkv_dim + num_heads]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        qkv_base: [bsz, q_len, base_qkv_dim]    gate_score_raw: [bsz, q_len, num_heads]
                    │                                   │
                    ▼                                   ▼
        rearrange: [bsz, q_len, h, gs, d]      reshape: [bsz, q_len, num_heads, 1]
                    │                                   │
                    ▼                                   ▼
        query/key/value 提取                    gate_score: [bsz, num_heads, q_len, 1]
                    │                                   │
                    └───────────┬───────────────────────┘
                                ▼
                    attn_output: [bsz, num_heads, q_len, head_dim]
                                │
                                ▼
                    gated_output: [bsz, num_heads, q_len, head_dim]
                                │
                                ▼
                    输出: [bsz, q_len, hidden_size]
```

### Element-wise Gating 模式

```
输入维度流程:
─────────────────────────────────────────────────────────────────────────
hidden_states:     [bsz, q_len, hidden_size]
                              │
                              ▼
wqkv 输出:         [bsz, q_len, base_qkv_dim + num_heads*head_dim]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        qkv_base: [bsz, q_len, base_qkv_dim]    gate_score_raw: [bsz, q_len, num_heads*head_dim]
                    │                                   │
                    ▼                                   ▼
        rearrange: [bsz, q_len, h, gs, d]      reshape: [bsz, q_len, num_heads, head_dim]
                    │                                   │
                    ▼                                   ▼
        query/key/value 提取                    gate_score: [bsz, num_heads, q_len, head_dim]
                    │                                   │
                    └───────────┬───────────────────────┘
                                ▼
                    attn_output: [bsz, num_heads, q_len, head_dim]
                                │
                                ▼
                    gated_output: [bsz, num_heads, q_len, head_dim]
                                │
                                ▼
                    输出: [bsz, q_len, hidden_size]
```

## 对比图：标准 vs Gated Attention

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        标准 Attention                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  hidden_states                                                            │
│      │                                                                    │
│      ▼                                                                    │
│  wqkv (base_qkv_dim)                                                     │
│      │                                                                    │
│      ▼                                                                    │
│  Q, K, V 提取                                                             │
│      │                                                                    │
│      ▼                                                                    │
│  Attention 计算                                                           │
│      │                                                                    │
│      ▼                                                                    │
│  attn_output                                                              │
│      │                                                                    │
│      ▼                                                                    │
│  wo → 输出                                                                │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      Gated Attention (Phase 1)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  hidden_states                                                            │
│      │                                                                    │
│      ▼                                                                    │
│  wqkv (base_qkv_dim + gate_dim)  ← 增加 gate_dim                         │
│      │                                                                    │
│      ├──────────────────┬──────────────────┐                            │
│      ▼                  ▼                  ▼                            │
│  qkv_base          gate_score_raw      (分离)                            │
│      │                  │                                                │
│      ▼                  ▼                                                │
│  Q, K, V 提取      gate_score (reshape)                                   │
│      │                  │                                                │
│      ▼                  │                                                │
│  Attention 计算         │                                                │
│      │                  │                                                │
│      ▼                  │                                                │
│  attn_output ───────┐   │                                                │
│                     │   │                                                │
│                     ▼   ▼                                                │
│              attn_output * sigmoid(gate_score)  ← Gating 应用            │
│                     │                                                    │
│                     ▼                                                    │
│              wo → 输出                                                    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## 关键组件图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    wqkv 线性层结构                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  输入: hidden_states [bsz, q_len, hidden_size]                          │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  Linear(hidden_size → base_qkv_dim + gate_dim)               │      │
│  │                                                               │      │
│  │  ┌──────────────────────────┐  ┌──────────────────────┐   │      │
│  │  │ base_qkv_dim               │  │ gate_dim             │   │      │
│  │  │                            │  │                      │   │      │
│  │  │ Q: num_heads * head_dim    │  │ Head-wise:           │   │      │
│  │  │ K: num_key_value_heads *  │  │   num_heads          │   │      │
│  │  │    head_dim                │  │                      │   │      │
│  │  │ V: num_key_value_heads *   │  │ Element-wise:        │   │      │
│  │  │    head_dim                │  │   num_heads *        │   │      │
│  │  │                            │  │   head_dim           │   │      │
│  │  └──────────────────────────┘  └──────────────────────┘   │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│  输出: qkv_states [bsz, q_len, base_qkv_dim + gate_dim]                 │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Gating 应用细节图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Gating 应用流程                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  attn_output: [bsz, num_heads, q_len, head_dim]                         │
│      │                                                                    │
│      │                                                                    │
│  gate_score (转置后): [bsz, num_heads, q_len, ...]                      │
│      │                                                                    │
│      ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ Head-wise Gating:                                             │      │
│  │                                                               │      │
│  │  gate_score: [bsz, num_heads, q_len, 1]                      │      │
│  │      │                                                        │      │
│  │      ▼                                                        │      │
│  │  sigmoid(gate_score) → [bsz, num_heads, q_len, 1]            │      │
│  │      │                                                        │      │
│  │      │  (广播)                                                │      │
│  │      ▼                                                        │      │
│  │  [bsz, num_heads, q_len, head_dim]  ← 匹配 attn_output       │      │
│  │      │                                                        │      │
│  │      ▼                                                        │      │
│  │  attn_output * sigmoid(gate_score)                            │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ Element-wise Gating:                                          │      │
│  │                                                               │      │
│  │  gate_score: [bsz, num_heads, q_len, head_dim]               │      │
│  │      │                                                        │      │
│  │      ▼                                                        │      │
│  │  sigmoid(gate_score) → [bsz, num_heads, q_len, head_dim]     │      │
│  │      │                                                        │      │
│  │      │  (直接匹配)                                            │      │
│  │      ▼                                                        │      │
│  │  [bsz, num_heads, q_len, head_dim]  ← 匹配 attn_output       │      │
│  │      │                                                        │      │
│  │      ▼                                                        │      │
│  │  attn_output * sigmoid(gate_score)                            │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                           │
│  输出: gated_attn_output [bsz, num_heads, q_len, head_dim]              │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## 简化版框架图（适合演示）

```
┌─────────────────────────────────────────────────────────────────┐
│              InternLM2Attention with Gating                      │
└─────────────────────────────────────────────────────────────────┘

Input: hidden_states
  │
  ├─→ wqkv ──┬─→ qkv_base ──→ Q, K, V ──→ Attention ──┐
  │          │                                            │
  │          └─→ gate_score_raw ──→ gate_score ──────────┼─→ Gating ──→ Output
  │                                                       │
  └───────────────────────────────────────────────────────┘
```

## 数据流示例（数值）

假设配置：
- `hidden_size = 2048`
- `num_heads = 16`
- `num_key_value_heads = 4`
- `head_dim = 128`
- `bsz = 2, q_len = 512`

### Head-wise Gating:

```
hidden_states:        [2, 512, 2048]
                            │
                            ▼
wqkv 输出:            [2, 512, 4096 + 16]  ← base_qkv_dim=4096, gate_dim=16
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
qkv_base:      [2, 512, 4096]    gate_score_raw: [2, 512, 16]
                │                       │
                ▼                       ▼
rearrange:     [2, 512, 16, 4, 128]    reshape: [2, 512, 16, 1]
                │                       │
                ▼                       ▼
Q/K/V:         [2, 16, 512, 128]  gate_score: [2, 16, 512, 1]
                │                       │
                └───────────┬───────────┘
                            ▼
                attn_output: [2, 16, 512, 128]
                            │
                            ▼
                gated_output: [2, 16, 512, 128]
                            │
                            ▼
                输出: [2, 512, 2048]
```

### Element-wise Gating:

```
hidden_states:        [2, 512, 2048]
                            │
                            ▼
wqkv 输出:            [2, 512, 4096 + 2048]  ← base_qkv_dim=4096, gate_dim=2048
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
qkv_base:      [2, 512, 4096]    gate_score_raw: [2, 512, 2048]
                │                       │
                ▼                       ▼
rearrange:     [2, 512, 16, 4, 128]    reshape: [2, 512, 16, 128]
                │                       │
                ▼                       ▼
Q/K/V:         [2, 16, 512, 128]  gate_score: [2, 16, 512, 128]
                │                       │
                └───────────┬───────────┘
                            ▼
                attn_output: [2, 16, 512, 128]
                            │
                            ▼
                gated_output: [2, 16, 512, 128]
                            │
                            ▼
                输出: [2, 512, 2048]
```

## 绘图建议

### 工具推荐：
1. **Draw.io / diagrams.net**: 支持流程图，可以导出为 PNG/SVG
2. **Mermaid**: 代码生成图表，适合文档
3. **PowerPoint / Keynote**: 适合演示
4. **LaTeX TikZ**: 适合论文

### 关键元素：
- ✅ 用不同颜色区分标准流程和 gating 流程
- ✅ 标注关键维度变化
- ✅ 突出显示 gating 应用的位置
- ✅ 对比 Head-wise 和 Element-wise 的区别
- ✅ 标注关键代码位置（行号）

### 颜色方案建议：
- 🔵 蓝色：标准 attention 流程
- 🟢 绿色：gate_score 生成和应用
- 🟡 黄色：关键操作（分离、reshape、gating）
- 🔴 红色：输出结果


