# InternVL: Text Token 与 Visual Token 融合机制详解

## 概述

InternVL 通过**Embedding 替换**的方式将视觉特征融合到文本序列中，然后统一送入 LLM 进行 self-attention 计算。整个过程分为以下几个阶段：

---

## 阶段 1: 数据预处理（`video_get_item`）

**位置**: `internvl_chat_finetune.py:1020-1089`

```python
# 1. 视频加载：将视频解码为多个帧
image_list = _load_video_locally(video_path, ...)
# image_list: List[Image.Image], 例如 32 帧

# 2. 生成特殊token字符串
special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
# 结果: "Frame-1: <image>\nFrame-2: <image>\nFrame-3: <image>\n..."

# 3. 替换原始prompt中的<video>占位符
data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
    '<video>\n', special_tokens + '\n')
```

**此时状态**:
- 文本: `"Frame-1: <image>\nFrame-2: <image>\n...Translate the ASL..."`
- 图像: `pixel_values.shape = [32, 3, 224, 224]` (32帧)

---

## 阶段 2: Token 替换（`preprocess` 函数）

**位置**: `dataset.py:315-428` 或 `preprocess_internlm` (643-691行)

```python
# 1. 构建对话模板
conversations = conv.get_prompt()
# 结果: "Frame-1: <image>\nFrame-2: <image>\n...Translate the ASL..."

# 2. 将 <image> 替换为特殊token序列
for i in range(num_image):  # num_image = 32 (帧数)
    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
    # IMG_START_TOKEN = '<img>'
    # IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    # IMG_END_TOKEN = '</img>'
    # num_image_token_list[i] = 64 (每帧的patch数)
    conversation = conversation.replace('<image>', image_tokens, 1)
```

**替换结果示例**:
```
Frame-1: <img><IMG_CONTEXT><IMG_CONTEXT>...<IMG_CONTEXT></img>
Frame-2: <img><IMG_CONTEXT><IMG_CONTEXT>...<IMG_CONTEXT></img>
...
Translate the American Sign Language in this video to English.
```

**关键点**:
- `IMG_CONTEXT_TOKEN` 的数量 = `num_image_token * num_patches`
- 例如：每帧64个patch → 64个 `<IMG_CONTEXT>` token
- 32帧 → 32 × 64 = 2048 个 `<IMG_CONTEXT>` token

---

## 阶段 3: Tokenization

**位置**: `dataset.py:681-688`

```python
# Tokenize所有文本和特殊token
input_ids = tokenizer(
    conversations,
    return_tensors='pt',
    padding=True,
    max_length=tokenizer.model_max_length,
    truncation=True,
).input_ids
```

**此时状态**:
- `input_ids.shape = [B, N]`，例如 `[1, 512]`
- 每个token都有对应的ID：
  - `"Frame-1:"` → 文本token IDs
  - `"<img>"` → `IMG_START_TOKEN_ID`
  - `"<IMG_CONTEXT>"` → `IMG_CONTEXT_TOKEN_ID` (重复64次)
  - `"</img>"` → `IMG_END_TOKEN_ID`
  - 其他文本 → 对应的token IDs

**示例**:
```python
input_ids = [
    [token_id("Frame-1:"), token_id(":"), ..., 
     IMG_START_TOKEN_ID, 
     IMG_CONTEXT_TOKEN_ID, IMG_CONTEXT_TOKEN_ID, ..., IMG_CONTEXT_TOKEN_ID,  # 64个
     IMG_END_TOKEN_ID,
     token_id("\n"),
     IMG_START_TOKEN_ID, 
     IMG_CONTEXT_TOKEN_ID, ...,  # 下一帧的64个
     ...]
]
```

---

## 阶段 4: Vision Encoder 提取视觉特征

**位置**: `modeling_internvl_chat.py:347-367` (`extract_feature`)

```python
def extract_feature(self, pixel_values):
    # 1. Vision Encoder (ViT)
    vit_embeds = self.vision_model(
        pixel_values=pixel_values,  # [32, 3, 224, 224]
        output_hidden_states=False,
        return_dict=True
    ).last_hidden_state
    # vit_embeds.shape: [32, 257, 1024]  # 32帧，每帧257个token (256 patches + 1 CLS)
    
    # 2. 移除CLS token
    vit_embeds = vit_embeds[:, 1:, :]  # [32, 256, 1024]
    
    # 3. Reshape为空间维度
    h = w = int(vit_embeds.shape[1] ** 0.5)  # 16
    vit_embeds = vit_embeds.reshape(32, 16, 16, 1024)  # [32, 16, 16, 1024]
    
    # 4. Pixel Shuffle (降采样)
    vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=0.5)
    # 结果: [32, 8, 8, 4096] → reshape → [32, 64, 4096]
    
    # 5. MLP投影到LLM的hidden_size
    vit_embeds = self.mlp1(vit_embeds)  # [32, 64, 2048]
    return vit_embeds
```

**输出**:
- `vit_embeds.shape = [num_frames, num_patches_per_frame, hidden_size]`
- 例如: `[32, 64, 2048]` (32帧，每帧64个patch，每个2048维)

---

## 阶段 5: Embedding 替换（核心融合步骤）

**位置**: `modeling_internvl_chat.py:145-263` (`forward`)

### 步骤 5.1: 获取文本Embeddings

```python
# 步骤 1: 文本token → 文本embeddings
input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
# input_embeds.shape: [B, N, C]
# 例如: [1, 512, 2048]
# 此时 IMG_CONTEXT_TOKEN 位置的 embeddings 是文本token的embedding（随机初始化或学习得到）
```

### 步骤 5.2: 提取视觉特征

```python
# 步骤 2: pixel_values → Vision Encoder → Patch Embeddings
vit_embeds = self.extract_feature(pixel_values)
# vit_embeds.shape: [32, 64, 2048]  # 32帧，每帧64个patch

# 根据image_flags过滤（只保留有效的图像）
vit_embeds = vit_embeds[image_flags == 1]
# vit_embeds.shape: [32, 64, 2048]  # 假设所有32帧都有效
```

### 步骤 5.3: 找到 IMG_CONTEXT_TOKEN 的位置

```python
# 步骤 3: 找到所有 IMG_CONTEXT_TOKEN 的位置
B, N, C = input_embeds.shape  # [1, 512, 2048]
input_ids = input_ids.reshape(B * N)  # [512]
input_embeds = input_embeds.reshape(B * N, C)  # [512, 2048]

selected = (input_ids == self.img_context_token_id)
# selected: [False, False, ..., True, True, ..., False]
#            ↑ 文本token    ↑ IMG_CONTEXT位置 (2048个True)
```

### 步骤 5.4: 替换Embeddings

```python
# 步骤 4: 用视觉特征替换 IMG_CONTEXT_TOKEN 位置的embeddings
vit_embeds_flat = vit_embeds.reshape(-1, C)  # [2048, 2048] (32*64=2048)
input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds_flat
# 等价于: input_embeds[selected] = vit_embeds_flat

# 重新reshape回原始形状
input_embeds = input_embeds.reshape(B, N, C)  # [1, 512, 2048]
```

**关键点**:
- **替换是in-place操作**，不改变序列长度
- 文本token的embeddings保持不变
- 只有`IMG_CONTEXT_TOKEN`位置的embeddings被替换为视觉特征

**替换前后对比**（以2帧视频为例）:

假设prompt为：`"Frame-1: <image>\nFrame-2: <image>\nTranslate the ASL..."`

经过token替换后，实际的token序列是：
```
[Frame-1, :, <img>, <IMG_CONTEXT>, <IMG_CONTEXT>, ..., <IMG_CONTEXT>, </img>, \n,
 Frame-2, :, <img>, <IMG_CONTEXT>, <IMG_CONTEXT>, ..., <IMG_CONTEXT>, </img>, \n,
 Translate, the, ASL, ...]
 ↑文本    ↑文本  ↑特殊  ↑视觉(64个)                    ↑特殊  ↑文本
```

**替换前**（所有token都是文本embedding）:
```python
input_embeds = [
    # 位置 0-1: 文本token "Frame-1:"
    [text_emb("Frame-1"), text_emb(":"), 
     # 位置 2: 特殊token "<img>"
     text_emb("<img>"),
     # 位置 3-66: IMG_CONTEXT token (64个) - 此时还是文本embedding
     text_emb("<IMG_CONTEXT>"), text_emb("<IMG_CONTEXT>"), ..., text_emb("<IMG_CONTEXT>"),
     # 位置 67: 特殊token "</img>"
     text_emb("</img>"),
     # 位置 68: 文本token "\n"
     text_emb("\n"),
     # 位置 69-70: 文本token "Frame-2:"
     text_emb("Frame-2"), text_emb(":"),
     # 位置 71: 特殊token "<img>"
     text_emb("<img>"),
     # 位置 72-135: IMG_CONTEXT token (64个) - 此时还是文本embedding
     text_emb("<IMG_CONTEXT>"), ..., text_emb("<IMG_CONTEXT>"),
     # 位置 136: 特殊token "</img>"
     text_emb("</img>"),
     # 位置 137: 文本token "\n"
     text_emb("\n"),
     # 位置 138+: 文本token "Translate", "the", "ASL", ...
     text_emb("Translate"), text_emb("the"), text_emb("ASL"), ...]
]
```

**替换后**（IMG_CONTEXT位置的embedding被替换为视觉特征）:
```python
input_embeds = [
    # 位置 0-1: 文本token "Frame-1:" - 保持不变
    [text_emb("Frame-1"), text_emb(":"), 
     # 位置 2: 特殊token "<img>" - 保持不变
     text_emb("<img>"),
     # 位置 3-66: IMG_CONTEXT token (64个) - ✅ 被替换为视觉特征！
     visual_emb_frame1_patch1, visual_emb_frame1_patch2, ..., visual_emb_frame1_patch64,
     # 位置 67: 特殊token "</img>" - 保持不变
     text_emb("</img>"),
     # 位置 68: 文本token "\n" - 保持不变
     text_emb("\n"),
     # 位置 69-70: 文本token "Frame-2:" - 保持不变
     text_emb("Frame-2"), text_emb(":"),
     # 位置 71: 特殊token "<img>" - 保持不变
     text_emb("<img>"),
     # 位置 72-135: IMG_CONTEXT token (64个) - ✅ 被替换为视觉特征！
     visual_emb_frame2_patch1, visual_emb_frame2_patch2, ..., visual_emb_frame2_patch64,
     # 位置 136: 特殊token "</img>" - 保持不变
     text_emb("</img>"),
     # 位置 137: 文本token "\n" - 保持不变
     text_emb("\n"),
     # 位置 138+: 文本token "Translate", "the", "ASL", ... - 保持不变
     text_emb("Translate"), text_emb("the"), text_emb("ASL"), ...]
]
```

**关键理解**:
1. **Text embedding和Visual embedding是交错排列的**，不是简单的"text在前，visual在后"
2. **Prompt的结构**：`[文本描述] + [视觉token] + [文本描述] + [视觉token] + [指令文本]`
3. **只有`<IMG_CONTEXT>`位置的embedding被替换**，其他所有token（包括`<img>`、`</img>`、文本token）的embedding都保持不变
4. **视觉特征按照帧的顺序插入**：第1帧的64个patch → 第1个`<IMG_CONTEXT>`块，第2帧的64个patch → 第2个`<IMG_CONTEXT>`块

**什么是"文本token"？**

"文本token"指的是**所有非视觉的token**，包括：

- **Frame描述文本**：`"Frame-1"`, `":"`, `"Frame-2"`, `":"` 等（这些会被tokenize成多个token，比如`"Frame"`和`"-1"`可能被分成不同的token）
- **特殊标记token**：`<img>`, `</img>` (这些是特殊token，但也是文本形式的token，不会被替换)
- **换行符**：`"\n"`
- **指令文本**：`"Translate"`, `"the"`, `"American"`, `"Sign"`, `"Language"`, `"in"`, `"this"`, `"video"`, `"to"`, `"English"` 等
- **其他所有普通文本**：prompt中的任何文字都会被tokenize成文本token

**只有`<IMG_CONTEXT>` token会被替换为视觉特征**，其他所有token（包括`<img>`、`</img>`、`"Frame-1"`、`":"`、`"Translate"`等）的embedding都保持不变，都是文本embedding。

**示例**：
```
Token序列: [Frame-1, :, <img>, <IMG_CONTEXT>×64, </img>, \n, Frame-2, :, <img>, <IMG_CONTEXT>×64, </img>, \n, Translate, the, ASL, ...]
Type:      [文本,   文本, 特殊,  视觉(替换),     特殊,  文本, 文本,  文本, 特殊,  视觉(替换),     特殊,  文本, 文本,    文本, 文本, ...]
```

所以"文本token"不仅仅是`"Frame-1"`或`"Frame-2"`，而是**除了`<IMG_CONTEXT>`之外的所有token**。

**更直观的图示**:

```
原始Prompt结构:
┌─────────────────────────────────────────────────────────────┐
│ Frame-1: <image>                                             │
│ Frame-2: <image>                                             │
│ ...                                                          │
│ Translate the American Sign Language in this video to English│
└─────────────────────────────────────────────────────────────┘

Token替换后:
┌─────────────────────────────────────────────────────────────┐
│ Frame-1: <img><IMG_CONTEXT>×64</img>                        │
│ Frame-2: <img><IMG_CONTEXT>×64</img>                        │
│ ...                                                          │
│ Translate the American Sign Language in this video to English│
└─────────────────────────────────────────────────────────────┘

Token序列（简化，只显示关键部分）:
位置:  0      1     2     3-66       67    68    69     70    71    72-135    136   137   138+
Token: Frame-1  :   <img>  <IMG>×64  </img> \n  Frame-2  :   <img>  <IMG>×64  </img> \n  Translate...
Type:  文本   文本  特殊   视觉(64)   特殊  文本  文本   文本  特殊   视觉(64)   特殊  文本  文本

Embedding替换后:
位置:  0      1     2     3-66       67    68    69     70    71    72-135    136   137   138+
Emb:   text   text  text  visual×64  text  text  text   text  text  visual×64  text  text  text...
       ↑文本  ↑文本 ↑特殊  ↑视觉特征  ↑特殊 ↑文本 ↑文本  ↑文本 ↑特殊  ↑视觉特征  ↑特殊 ↑文本 ↑文本
```

**关键点**:
- **Text embedding在Visual embedding之前**：是的，prompt中的文本描述（如"Frame-1:"）确实在视觉token之前
- **但是它们是交错排列的**：不是"所有text在前，所有visual在后"，而是"text → visual → text → visual → text"的模式
- **Prompt的结构决定了顺序**：因为prompt是 `"Frame-1: <image>\nFrame-2: <image>\n...Translate..."`，所以token顺序就是文本描述 → 视觉token → 文本描述 → 视觉token → 指令文本

---

## 阶段 6: 送入LLM进行Self-Attention

**位置**: `modeling_internvl_chat.py:247-263`

```python
# 步骤 5: 融合后的embeddings进入LLM
outputs = self.language_model(
    inputs_embeds=input_embeds,  # [1, 512, 2048]
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=return_dict,
)
```

### Self-Attention 机制

在LLM内部，**所有token（包括文本和视觉token）统一进行self-attention计算**：

```python
# LLM内部（简化版）
class LLMDecoderLayer:
    def forward(self, hidden_states, attention_mask):
        # hidden_states: [B, N, C] = [1, 512, 2048]
        # 包含: [text_emb, text_emb, ..., visual_emb, visual_emb, ..., text_emb]
        
        # 1. Self-Attention
        q = self.q_proj(hidden_states)  # [1, 512, 2048]
        k = self.k_proj(hidden_states)  # [1, 512, 2048]
        v = self.v_proj(hidden_states)  # [1, 512, 2048]
        
        # 2. Attention计算（所有token之间）
        attn_weights = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
        # attn_weights.shape: [1, num_heads, 512, 512]
        # 每个位置可以attend到所有位置（包括文本和视觉token）
        
        # 3. 应用attention mask（causal mask）
        attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = softmax(attn_weights, dim=-1)
        
        # 4. 加权求和
        attn_output = attn_weights @ v  # [1, 512, 2048]
        
        return attn_output
```

### Self-Attention 的关键特性

1. **统一计算**: 文本token和视觉token在同一个attention矩阵中计算
2. **双向交互**: 
   - 文本token可以attend到视觉token（例如："Frame-1"可以关注到对应的64个视觉patch）
   - 视觉token可以attend到文本token（例如：视觉patch可以关注到"Translate"等指令）
   - 视觉token之间也可以相互attend（例如：不同帧的patch之间可以交互）
3. **位置编码**: `position_ids`确保每个token（包括视觉token）都有正确的位置信息
4. **Causal Mask**: 在生成时，视觉token可以attend到所有之前的token（包括文本和其他视觉token）

### Attention 模式示例

假设序列为: `[Frame-1:, <img>, <IMG_CONTEXT>×64, </img>, Frame-2:, <img>, <IMG_CONTEXT>×64, </img>, Translate, ...]`

**Attention矩阵** (简化，只显示部分):
```
                Frame-1:  <img>  IMG_CTX×64  </img>  Frame-2:  <img>  IMG_CTX×64  </img>  Translate
Frame-1:          ✓        ✓        ✓         ✓         ✗        ✗        ✗         ✗         ✗
<img>             ✓        ✓        ✓         ✓         ✗        ✗        ✗         ✗         ✗
IMG_CTX (frame1)  ✓        ✓        ✓         ✓         ✗        ✗        ✗         ✗         ✗
</img>            ✓        ✓        ✓         ✓         ✗        ✗        ✗         ✗         ✗
Frame-2:          ✓        ✓        ✓         ✓         ✓        ✓        ✓         ✓         ✗
<img>             ✓        ✓        ✓         ✓         ✓        ✓        ✓         ✓         ✗
IMG_CTX (frame2)  ✓        ✓        ✓         ✓         ✓        ✓        ✓         ✓         ✗
</img>            ✓        ✓        ✓         ✓         ✓        ✓        ✓         ✓         ✗
Translate         ✓        ✓        ✓         ✓         ✓        ✓        ✓         ✓         ✓
```

**说明**:
- ✓ 表示可以attend（在causal mask允许的范围内）
- ✗ 表示被mask掉（未来token不能attend到过去token）
- 视觉token可以attend到对应的文本描述（如"Frame-1:"）
- 文本token可以attend到对应的视觉特征

---

## 完整数据流示例

### 输入
- **视频**: 32帧，每帧 224×224
- **Prompt**: `"Frame-1: <video>\nFrame-2: <video>\n...Translate the ASL..."`

### 处理流程

```
1. 视频解码
   pixel_values: [32, 3, 224, 224]
   ↓
2. Token替换
   "Frame-1: <image>" → "Frame-1: <img><IMG_CONTEXT>×64</img>"
   ↓
3. Tokenization
   input_ids: [1, 512]  # 包含文本token和IMG_CONTEXT token
   ↓
4. Vision Encoder
   pixel_values → vit_embeds: [32, 64, 2048]
   ↓
5. Text Embedding
   input_ids → input_embeds: [1, 512, 2048]
   ↓ 
6. Embedding替换
   input_embeds[IMG_CONTEXT_positions] = vit_embeds.reshape(-1, 2048)
   ↓
7. LLM Self-Attention
   input_embeds → [所有token统一进行attention] → outputs
   ↓
8. 生成
   outputs.logits → 下一个token预测
```

---

## 关键设计优势

1. **统一序列**: 文本和视觉token在同一个序列中，LLM可以自然地处理多模态信息
2. **灵活交互**: Self-attention机制允许任意token之间的交互
3. **位置感知**: Position encoding确保视觉token有正确的位置信息
4. **无需修改LLM**: LLM本身不需要知道哪些是视觉token，统一按文本token处理

---

## 代码位置总结

| 阶段 | 文件 | 函数/类 | 行号 |
|------|------|---------|------|
| 视频预处理 | `internvl_chat_finetune.py` | `video_get_item` | 1020-1089 |
| Token替换 | `dataset.py` | `preprocess_internlm` | 643-691 |
| Tokenization | `dataset.py` | `preprocess_internlm` | 681-688 |
| Vision Encoder | `modeling_intern_vit.py` | `InternVisionModel.forward` | 422-464 |
| Feature提取 | `modeling_internvl_chat.py` | `extract_feature` | 347-367 |
| Embedding替换 | `modeling_internvl_chat.py` | `InternVLChatModel.forward` | 145-263 |
| Self-Attention | LLM内部 | `LlamaDecoderLayer` / `InternLM2DecoderLayer` | - |

---

## 总结

InternVL通过**"Token替换 + Embedding替换"**的方式实现多模态融合：

1. **预处理阶段**: 将`<image>`替换为`<IMG_CONTEXT>` token序列
2. **Forward阶段**: 将`IMG_CONTEXT_TOKEN`位置的文本embeddings替换为视觉特征embeddings
3. **LLM阶段**: 所有token（文本+视觉）统一进行self-attention计算

这种设计的核心思想是：**让LLM"看到"的是一串统一的embeddings序列，其中视觉信息已经"伪装"成了文本token的形式**，从而LLM可以自然地处理多模态信息。

