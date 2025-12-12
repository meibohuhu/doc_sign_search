# Attention Map 计算方式对比

## 概述

`demo.py` 和 `internvl2_evaluation_attention_score.py` 计算的是**完全不同类型**的注意力，因此结果不一样是正常的。

## 核心区别

| 特性 | demo.py | internvl2_evaluation_attention_score.py |
|------|---------|------------------------------------------|
| **模型类型** | 纯文本模型 (Qwen3) | 多模态模型 (InternVL2) |
| **输入** | 纯文本 | 文本 + 视频帧 |
| **注意力类型** | 完整注意力矩阵 | 聚合的视觉 token 注意力 |
| **注意力形状** | `[seq_len, seq_len]` | `[num_visual_tokens]` |
| **可视化内容** | Token-to-token 注意力 | 对视觉 token 的聚合注意力 |
| **颜色映射** | `viridis` | `hot` |

## 详细对比

### 1. demo.py - 完整注意力矩阵

#### 计算方式

```python
# 1. 获取原始注意力
attentions = outputs.attentions  
# 形状: tuple of [batch, num_heads, seq_len, seq_len]

# 2. 平均所有 heads
avg_attn = layer_attn.mean(dim=1)  
# 形状: [batch, seq_len, seq_len]

# 3. 取第一个样本
attn_map = avg_attn[0]  
# 形状: [seq_len, seq_len]
```

#### 注意力矩阵含义

```
attn_map[i, j] = token i 对 token j 的注意力权重
```

**示例**（文本："Sparse gating mechanism mitigates attention sink."）:
```
        Sparse  gating  mechanism  mitigates  attention  sink  .
Sparse    0.1     0.2      0.05       0.03       0.02    0.1  0.5
gating    0.15    0.2      0.3        0.1        0.05   0.1  0.1
mechanism 0.05    0.1      0.15       0.4        0.2    0.05 0.05
...
```

- **行 (i)**: Query token（哪个 token 在查询）
- **列 (j)**: Key token（查询哪个 token）
- **值**: 注意力权重（0-1）

#### 可视化

- **2D 热力图**: 显示所有 token 之间的注意力关系
- **对角线**: 自注意力（token 对自己的注意力）
- **非对角线**: 跨 token 注意力

### 2. internvl2_evaluation_attention_score.py - 聚合视觉注意力

#### 计算方式

```python
# 1. 获取原始注意力（在 modeling_internlm2.py 中）
layer_attn = layer_outputs[1]  
# 形状: [batch, num_heads, seq_len, seq_len]

# 2. 提取对视觉 token 的注意力
visual_attn = layer_attn[:, :, :, visual_start:visual_end+1]  
# 形状: [batch, num_heads, seq_len, num_visual_tokens]

# 3. 聚合：对所有 query tokens 和 heads 求和
aggregated = visual_attn.sum(dim=(0, 1, 2))  
# 形状: [num_visual_tokens]

# 4. 累加所有层
aggregated_viusal_token_attention += aggregated
```

#### 注意力向量含义

```
aggregated_attention[j] = 所有 token 和所有层对视觉 token j 的注意力总和
```

**示例**（2 帧视频，每帧 14×14=196 patches）:
```
aggregated_attention = [
    0.15,  # patch 0 的注意力
    0.23,  # patch 1 的注意力
    0.08,  # patch 2 的注意力
    ...
    0.31   # patch 391 的注意力
]
# 形状: [392] (2 * 196)
```

- **索引 (j)**: 视觉 token/patch 的位置
- **值**: 所有文本 token 和所有层对该视觉 patch 的注意力总和

#### 可视化

- **1D 数组**: 每个视觉 patch 的聚合注意力值
- **重塑为 2D**: `[num_frames, num_patches_h, num_patches_w]`
- **Mosaic 图**: 将多帧排列成网格

## 关键差异总结

### 1. **维度不同**

```python
# demo.py
attn_map.shape = [seq_len, seq_len]  # 例如: [10, 10]

# internvl2
aggregated_attention.shape = [num_visual_tokens]  # 例如: [392]
```

### 2. **聚合方式不同**

```python
# demo.py: 只平均 heads，保留完整的 token-to-token 关系
attn_map = layer_attn.mean(dim=1)  # [batch, seq_len, seq_len]

# internvl2: 聚合所有 query tokens、所有 heads、所有层
aggregated = visual_attn.sum(dim=(0, 1, 2))  # [num_visual_tokens]
```

### 3. **关注点不同**

- **demo.py**: 
  - 关注**哪些文本 token 关注哪些文本 token**
  - 用于理解文本内部的注意力模式
  - 例如：查看 "attention" 这个词关注哪些其他词

- **internvl2**:
  - 关注**哪些视觉区域获得了更多注意力**
  - 用于理解模型对视觉内容的关注分布
  - 例如：查看模型是否关注手部、面部等区域

### 4. **计算位置不同**

```python
# demo.py: 在 Python 脚本中计算
def average_heads(attentions):
    for layer_attn in attentions:
        avg_attn = layer_attn.mean(dim=1)  # 平均 heads
        averaged.append(avg_attn[0])

# internvl2: 在模型 forward 中计算（modeling_internlm2.py）
for layer in layers:
    layer_attn = layer_outputs[1]  # [batch, heads, seq, seq]
    visual_attn = layer_attn[:, :, :, visual_start:visual_end+1]
    aggregated += visual_attn.sum(dim=(0, 1, 2))  # 聚合
```

## 为什么不一样？

### 1. **不同的任务目标**

- **demo.py**: 分析**文本模型**的注意力模式
  - 研究 gated attention 如何影响文本 token 之间的注意力
  - 比较 baseline、head-wise、element-wise 三种模式

- **internvl2**: 分析**多模态模型**对视觉内容的关注
  - 研究模型在 ASL 翻译时关注哪些视觉区域
  - 验证模型是否关注关键区域（手部、面部）

### 2. **不同的信息需求**

- **demo.py**: 需要完整的注意力矩阵
  - 查看 token-to-token 的详细关系
  - 识别注意力 sink 问题
  - 分析注意力模式的变化

- **internvl2**: 只需要对视觉 token 的聚合注意力
  - 不需要知道具体哪个文本 token 关注哪个视觉 patch
  - 只需要知道哪些视觉区域整体获得了更多关注
  - 用于生成热力图叠加在原图上

### 3. **不同的可视化方式**

```python
# demo.py: 2D 矩阵热力图
imshow(attn_map, cmap="viridis")  
# 显示: [seq_len, seq_len] 矩阵

# internvl2: 1D 数组重塑为空间热力图
mosaic = attn_1d.reshape(num_frames, num_patches_h, num_patches_w)
imshow(mosaic, cmap="hot")  
# 显示: 空间注意力分布
```

## 如何统一？

如果你想在 `internvl2_evaluation_attention_score.py` 中生成类似 `demo.py` 的完整注意力矩阵，可以这样做：

```python
# 在 extract_attention_with_hook 中
if hasattr(outputs, 'attentions') and outputs.attentions is not None:
    # 获取完整注意力矩阵（类似 demo.py）
    full_attention_maps = []
    for layer_attn in outputs.attentions:
        # layer_attn: [batch, num_heads, seq_len, seq_len]
        avg_attn = layer_attn.mean(dim=0).mean(dim=0)  # [seq_len, seq_len]
        full_attention_maps.append(avg_attn[0].cpu().numpy())
    
    # 可视化（类似 demo.py）
    for layer_idx, attn_map in enumerate(full_attention_maps):
        plt.figure(figsize=(10, 10))
        plt.imshow(attn_map, cmap="viridis")
        plt.colorbar()
        plt.title(f"Layer {layer_idx} - Full Attention Matrix")
        plt.savefig(f"layer_{layer_idx}_full_attention.png")
```

## 总结

| 方面 | demo.py | internvl2_evaluation_attention_score.py |
|------|---------|------------------------------------------|
| **计算内容** | ✅ 完整注意力矩阵 | ✅ 聚合视觉注意力 |
| **适用场景** | 文本模型分析 | 多模态模型视觉分析 |
| **信息量** | 详细（token-to-token） | 简化（区域级聚合） |
| **可视化** | 2D 矩阵热力图 | 空间热力图/Mosaic |
| **用途** | 理解文本内部注意力 | 理解视觉关注分布 |

**结论**: 这两种计算方式服务于不同的分析目标，结果不一样是**预期行为**，不是 bug。





