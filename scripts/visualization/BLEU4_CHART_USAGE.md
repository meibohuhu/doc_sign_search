# BLEU4 柱状图使用指南

这个脚本可以帮你创建不同模型大小的BLEU4性能对比柱状图，并支持自定义颜色。

## 快速开始

### 1. 从JSON文件读取数据

创建数据文件 `my_data.json`:

```json
{
  "models": [
    {"model_size": "1B", "bleu4": 0.0564},
    {"model_size": "2B", "bleu4": 0.0615},
    {"model_size": "4B", "bleu4": 0.0672},
    {"model_size": "8B", "bleu4": 0.0710}
  ]
}
```

运行脚本：

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output bleu4_chart.png
```

### 2. 从命令行直接输入数据

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --model-sizes "1B" "2B" "4B" "8B" \
    --bleu4 0.0564 0.0615 0.0672 0.0710 \
    --output bleu4_chart.png
```

## 样式自定义

### 使用预设样式（推荐）

预设样式包含颜色、填充图案和边框的完美组合：

- `professional` - 专业风格（类似示例图片，带条纹和点状图案）
- `paper` - 纸张风格（柔和的米色）
- `striped` - 条纹风格（所有柱子都有条纹）
- `academic` - 学术风格（淡蓝色系）

示例（推荐使用这个，效果最好）：

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output bleu4_chart.png \
    --style-preset professional
```

### 使用填充图案（Hatch Patterns）

可以单独指定每个柱子的填充图案：

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output bleu4_chart.png \
    --color-palette paper \
    --hatch-patterns "/" "/" "." "none" "none"
```

可用的填充图案：
- `/` - 对角条纹（向右）
- `\` - 对角条纹（向左）
- `.` - 点状
- `-` - 水平线
- `|` - 垂直线
- `x` - 交叉
- `+` - 加号
- `..` - 密集点
- `xx` - 密集交叉
- `none` - 无填充（实心）

### 使用预设调色板

可用的调色板：
- `default` - 默认灰色系
- `paper` - 纸张米色
- `soft` - 柔和棕色
- `blue` - 蓝色渐变
- `green` - 绿色渐变
- `red` - 红色渐变
- `purple` - 紫色渐变
- `orange` - 橙色渐变
- `cool` - 冷色调
- `warm` - 暖色调
- `rainbow` - 彩虹色
- `academic` - 学术蓝色

示例：

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output bleu4_chart.png \
    --color-palette paper
```

### 使用自定义颜色

使用十六进制颜色代码：

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output bleu4_chart.png \
    --custom-colors "#e8e8e8" "#e8e8e8" "#e8e8e8" "#f5f5dc" "#d4c2a8"
```

### 自定义边框颜色

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output bleu4_chart.png \
    --style-preset professional \
    --edgecolors "#666666" "#666666" "#666666" "#8b7355" "#8b7355"
```

## 高级选项

### 自定义图表标题和标签

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output chart.png \
    --title "不同模型大小的BLEU4性能对比" \
    --xlabel "模型大小" \
    --ylabel "BLEU4分数"
```

### 调整图表大小

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output chart.png \
    --figsize 12 8
```

### 调整柱状图宽度

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output chart.png \
    --bar-width 0.7
```

### 旋转X轴标签

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output chart.png \
    --rotation 45
```

### 设置Y轴范围

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output chart.png \
    --ylim 0 0.1
```

### 隐藏数值标签

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output chart.png \
    --no-values
```

### 设置输出分辨率

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output chart.png \
    --dpi 300
```

## 完整示例

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json my_data.json \
    --output outputs/visualizations/bleu4_comparison.png \
    --title "不同模型大小的BLEU4性能对比" \
    --xlabel "模型大小" \
    --ylabel "BLEU4分数" \
    --color-palette blue \
    --figsize 12 8 \
    --bar-width 0.7 \
    --rotation 0 \
    --dpi 300
```

## 运行示例脚本

```bash
bash scripts/visualization/plot_bleu4_example.sh
```

这会生成多个不同颜色方案的示例图表。

