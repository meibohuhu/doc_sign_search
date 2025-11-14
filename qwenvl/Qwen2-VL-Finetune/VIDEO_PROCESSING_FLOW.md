# Qwen2-VL 视频处理流程详解

## 完整流程概览

```
Video File (.mp4)
    ↓
[1] Processor (qwen_vl_utils.process_vision_info)
    ↓ 提取帧 + 预处理
[2] pixel_values_videos + video_grid_thw (T, H, W)
    ↓
[3] model.get_video_features()
    ↓
[4] Vision Encoder (model.visual)
    ├─ [4.1] Patch Embedding (patch_embed)
    ├─ [4.2] Vision Blocks (temporal + spatial attention)
    └─ [4.3] Merger (spatial downsampling)
    ↓
[5] Video Embeddings
    ↓
[6] LLM Input (masked_scatter into text embeddings)
    ↓
[7] Language Model Processing
```

---

## 详细步骤

### 步骤 1: 数据加载 (Dataset)

**文件**: `src/dataset/sft_dataset.py`

```python
# 在 get_video_info() 中
video_input, video_kwargs = get_video_info(
    video_file, 
    min_pixels, max_pixels, 
    width, height, 
    fps, nframes
)
```

**关键函数**: `src/dataset/data_utils.py::get_video_info()`
- 调用 `qwen_vl_utils.process_vision_info()` 
- 返回 `video_input` 和 `video_kwargs`
- `video_kwargs` 包含 `video_grid_thw` 信息

### 步骤 2: Processor 处理

**Processor**: `AutoProcessor.from_pretrained(model_id)`

```python
inputs = processor(
    text=[user_input], 
    videos=videos,  # 视频文件路径或已加载的视频
    padding=False, 
    do_resize=False, 
    return_tensors='pt',
    **video_kwargs  # 包含 fps, nframes 等参数
)
```

**输出**:
- `pixel_values_videos`: `[num_frames, channels, height, width]` - 视频帧的像素值
- `video_grid_thw`: `[num_videos, 3]` - 每个视频的 (Temporal, Height, Width) 网格信息
  - T: 时间维度（帧数）
  - H: 高度（patch 数量）
  - W: 宽度（patch 数量）

**Processor 内部处理**:
1. 使用 `fps` 或 `nframes` 参数从视频中采样帧
2. 将帧调整为指定尺寸（如 224x224）
3. 归一化像素值
4. 计算 `video_grid_thw`（基于 patch size 和帧数）

### 步骤 3: 模型前向传播

**文件**: `src/train/monkey_patch_forward.py`

```python
if pixel_values_videos is not None:
    video_embeds = self.get_video_features(
        pixel_values_videos, 
        video_grid_thw
    )
    video_embeds = torch.cat(video_embeds, dim=0)
    _, video_mask = self.get_placeholder_mask(
        input_ids, 
        inputs_embeds=inputs_embeds, 
        video_features=video_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
```

### 步骤 4: Vision Encoder 处理

**文件**: `src/train/monkey_patch_vision.py`

#### 4.1 Patch Embedding

```python
hidden_states = self.patch_embed(hidden_states)
```

**Qwen2_5_VisionPatchEmbed**:
- 将视频帧切分成 patches
- 每个 patch 通常是 14x14 像素（取决于 `patch_size`）
- 输出: `[seq_len, embed_dim]`
  - `seq_len = T × H × W` (时间 × 高度 × 宽度)
  - `embed_dim = hidden_size` (如 1024)

#### 4.2 Vision Blocks (Temporal + Spatial Attention)

```python
for layer_num, blk in enumerate(self.blocks):
    if layer_num in self.fullatt_block_indexes:
        cu_seqlens_now = cu_seqlens  # Full attention
    else:
        cu_seqlens_now = cu_window_seqlens  # Window attention
    
    hidden_states = blk(
        hidden_states, 
        cu_seqlens=cu_seqlens_now, 
        position_embeddings=position_embeddings
    )
```

**Qwen2_5_VLVisionBlock**:
- **Window Attention**: 在空间维度使用窗口注意力（类似 Swin Transformer）
- **Full Attention**: 在时间维度使用全局注意力
- **Rotary Position Embedding**: 为每个 patch 添加位置编码
- 处理时空信息，提取视频特征

**关键机制**:
- `window_size`: 控制空间窗口大小
- `spatial_merge_size`: 控制空间下采样率
- `temporal_patch_size`: 控制时间维度 patch 大小

#### 4.3 Merger (Spatial Downsampling)

```python
hidden_states = self.merger(hidden_states)
```

**Qwen2_5_VLPatchMerger**:
- 将 Vision Encoder 的输出维度投影到 LLM 的输入维度
- 进行空间下采样（通过 `spatial_merge_size`）
- 输出: `[seq_len_llm, llm_hidden_size]`
  - `seq_len_llm = T × (H/spatial_merge_size) × (W/spatial_merge_size)`

**配置示例**:
- Vision hidden_size: 1024
- LLM hidden_size: 2048
- spatial_merge_size: 2 (2x2 合并)

### 步骤 5: 融合到 LLM

```python
inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
```

**过程**:
1. 在文本中找到 `<video>` token 的位置
2. 用 `video_embeds` 替换这些位置的 embeddings
3. 保持文本和视频 embeddings 在同一序列中

### 步骤 6: LLM 处理

```python
outputs = self.language_model(
    input_ids=None,  # 使用 inputs_embeds 而不是 input_ids
    inputs_embeds=inputs_embeds,  # 包含文本 + 视频 embeddings
    position_ids=position_ids,  # 包含 RoPE 位置编码
    ...
)
```

**LLM**:
- 处理混合模态序列（文本 + 视频）
- 使用 RoPE (Rotary Position Embedding) 处理位置信息
- 生成文本输出

---

## 关键参数说明

### 训练参数

- `--fps 12`: 每秒采样 12 帧
- `--nframes`: 固定采样帧数（如果设置，会覆盖 fps）
- `--video_min_pixels`: 最小像素数（用于调整尺寸）
- `--video_max_pixels`: 最大像素数（用于调整尺寸）
- `--video_resized_width/height`: 固定尺寸（如 224x224）

### Vision Encoder 配置

- `patch_size`: Patch 大小（通常 14）
- `temporal_patch_size`: 时间维度 patch 大小
- `spatial_merge_size`: 空间下采样率（通常 2）
- `window_size`: 窗口注意力窗口大小
- `hidden_size`: Vision encoder 隐藏层维度
- `out_hidden_size`: 输出到 LLM 的维度（通常等于 LLM hidden_size）

### Merger 配置

- `dim`: 输出维度（LLM hidden_size）
- `context_dim`: 输入维度（Vision hidden_size）
- `spatial_merge_size`: 空间合并大小

---

## 数据流示例

假设输入一个 5 秒视频，fps=12:

1. **帧采样**: 5秒 × 12fps = 60 帧
2. **Patch Embedding**: 
   - 每帧 224×224 → 16×16 patches (patch_size=14)
   - 60帧 × 16×16 = 15,360 patches
3. **Vision Blocks**: 
   - 处理时空特征
   - 输出: [15360, 1024]
4. **Merger**: 
   - 空间下采样: 16×16 → 8×8 (spatial_merge_size=2)
   - 输出: [60×8×8, 2048] = [3840, 2048]
5. **LLM**: 
   - 3840 个视频 tokens + 文本 tokens
   - 生成响应

---

## 代码位置总结

| 组件 | 文件位置 |
|------|---------|
| 数据加载 | `src/dataset/sft_dataset.py` |
| 视频信息提取 | `src/dataset/data_utils.py::get_video_info()` |
| Processor | `qwen_vl_utils.process_vision_info()` (外部库) |
| 模型前向 | `src/train/monkey_patch_forward.py` |
| Vision Encoder | `src/train/monkey_patch_vision.py` |
| Patch Embedding | `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed` |
| Vision Blocks | `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock` |
| Merger | `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLPatchMerger` |

---

## 注意事项

1. **内存优化**: 
   - 使用 `gradient_checkpointing` 减少内存
   - 使用 `fps` 控制采样帧数

2. **训练配置**:
   - `freeze_vision_tower`: 是否冻结 Vision Encoder
   - `freeze_merger`: 是否冻结 Merger
   - `vision_lr`: Vision Encoder 学习率
   - `merger_lr`: Merger 学习率

3. **性能优化**:
   - 使用 `use_liger=True` 启用 Liger kernel 加速
   - 使用 `bf16` 或 `fp16` 混合精度训练

