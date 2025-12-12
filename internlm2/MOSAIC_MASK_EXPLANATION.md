# Mosaic Mask 图解释

## 什么是 Mosaic Mask？

Mosaic Mask 是一个**注意力热力图的拼接图**，它将视频中所有帧的注意力分数以网格形式排列在一起，形成一个大的可视化图像。

## 生成过程

### 1. 数据来源

Mosaic Mask 基于模型对视觉 token 的**注意力分数**（attention scores）生成：

```python
# 从模型提取的注意力分数
aggregated_attention = outputs.aggregated_viusal_token_attention
# 形状: [num_frames * num_patches_h * num_patches_w]
# 例如: [2 * 14 * 14] = [392] (2帧，每帧14x14个patch)
```

### 2. 数据预处理

```python
# 1. 归一化到 [0, 1] 范围
attn_1d = minmax_01(attn_1d)

# 2. 重塑为 3D 数组
# [num_frames * num_patches_h * num_patches_w] 
# -> [num_frames, num_patches_h, num_patches_w]
attn_3d = attn_1d.reshape(num_frames, num_patches_h, num_patches_w)
```

### 3. 网格排列

代码将多帧注意力图排列成一个网格（mosaic）：

```python
# 计算网格维度（尽量接近正方形）
cols = int(np.ceil(np.sqrt(num_frames)))  # 列数
rows = int(np.ceil(num_frames / cols))     # 行数

# 例如：2帧 -> 2x1网格，4帧 -> 2x2网格，9帧 -> 3x3网格
```

### 4. 拼接成 Mosaic

```python
# 将每一帧的注意力图放到网格的对应位置
for frame_idx in range(num_frames):
    row = frame_idx // cols
    col = frame_idx % cols
    # 将这一帧的注意力图 [num_patches_h, num_patches_w] 
    # 放到 mosaic 的对应位置
    mosaic[start_h:end_h, start_w:end_w] = attn_3d[frame_idx]
```

### 5. 颜色映射

```python
# 使用 'hot' colormap 将注意力值转换为颜色
# 低注意力 -> 黑色/深色
# 高注意力 -> 红色/黄色/白色
mosaic_rgb = plt.cm.hot(mosaic_mask)[:, :, :3]
mosaic_rgb = (mosaic_rgb * 255).astype(np.uint8)
```

## 可视化示例

假设有 2 帧视频，每帧有 14×14=196 个视觉 patch：

```
原始注意力数据:
Frame 1: [196个注意力值] -> reshape -> [14, 14] 注意力图
Frame 2: [196个注意力值] -> reshape -> [14, 14] 注意力图

Mosaic 排列 (2x1 网格):
┌─────────────┬─────────────┐
│  Frame 1    │  Frame 2    │
│  [14×14]    │  [14×14]    │
└─────────────┴─────────────┘

最终 Mosaic 图: [14, 28] (高度14, 宽度28)
```

如果有 4 帧，会排列成 2×2 网格：
```
┌─────────────┬─────────────┐
│  Frame 1    │  Frame 2    │
├─────────────┼─────────────┤
│  Frame 3    │  Frame 4    │
└─────────────┴─────────────┘
```

## 颜色含义

使用 `hot` colormap（热力图）：
- **黑色/深蓝色**: 注意力分数低（接近 0）
- **深红色**: 中等注意力分数
- **黄色/白色**: 注意力分数高（接近 1）

## 实际意义

### 1. **整体注意力模式**
- 一眼看到整个视频的注意力分布
- 识别哪些帧/区域获得了更多关注

### 2. **时间维度分析**
- 比较不同帧之间的注意力差异
- 发现注意力随时间的变化模式

### 3. **空间注意力分布**
- 每帧内的注意力分布（哪些区域更重要）
- 跨帧的空间注意力一致性

### 4. **模型行为理解**
- 模型在翻译 ASL 时关注哪些视觉区域
- 是否关注手部动作、面部表情等关键区域

## 代码位置

### 生成函数
- **`create_mosaic_mask()`** (第 67-110 行): 核心生成逻辑
- **`process_video()`** (第 538-550 行): 调用生成并保存

### 关键参数

```python
# 从命令行参数
--num-segments 2        # 视频采样帧数（影响 num_frames）
--image-size 224        # 图像尺寸（影响 patch 数量）
--save-mosaic-mask      # 启用 mosaic 生成
```

### 输出文件

```python
# 保存路径
mosaic_path = os.path.join(output_dir, f"{video_file}_mosaic_mask.png")
# 例如: attention_visualizations/abzRFn8xngA_5-3-rgb_front.mp4_mosaic_mask.png
```

## 与其他可视化的区别

| 可视化类型 | 内容 | 用途 |
|-----------|------|------|
| **Mosaic Mask** | 所有帧的注意力图拼接 | 整体模式、时间对比 |
| **Frame Attention** | 单帧的注意力热力图 | 单帧详细分析 |
| **Overlay** | 注意力叠加在原图上 | 直观看到关注区域 |

## 使用场景

1. **快速概览**: 快速了解模型对整段视频的关注模式
2. **对比分析**: 比较不同模型或不同 prompt 的注意力差异
3. **调试**: 检查模型是否正确关注关键区域（手部、面部等）
4. **研究**: 分析注意力在时间维度上的变化

## 注意事项

1. **Patch 数量**: 取决于图像尺寸和模型配置
   - `image_size=224` → 通常 `14×14=196` patches/帧
   - `image_size=448` → 通常 `28×28=784` patches/帧

2. **帧数**: 由 `--num-segments` 参数控制
   - 更多帧 → 更大的 mosaic 图
   - 建议 2-8 帧以获得清晰的视觉效果

3. **归一化**: 所有注意力值被归一化到 [0, 1]
   - 颜色强度反映相对注意力，而非绝对值

## 总结

Mosaic Mask 是一个**多帧注意力热力图的网格拼接图**，用于：
- 📊 **可视化**整个视频的注意力分布
- 🔍 **分析**模型关注的时间-空间模式
- 🎯 **验证**模型是否正确关注关键区域（如 ASL 中的手部和面部）

通过这个图，你可以一眼看出模型在处理视频时，哪些帧和哪些区域获得了最多的注意力。





