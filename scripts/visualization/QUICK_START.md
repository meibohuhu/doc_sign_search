# 快速开始 - 专业风格柱状图

## 最简单的方式（推荐）

使用预设的 `professional` 样式，效果类似你提供的示例图片：

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json scripts/visualization/example_bleu4_data.json \
    --output outputs/visualizations/bleu4_chart.png \
    --style-preset professional
```

这会生成一个带有：
- 灰色背景的柱子（前几个带对角条纹和点状图案）
- 米色/棕色的实心柱子（后几个）
- 专业的边框颜色

## 其他预设样式

### 纸张风格（柔和米色）
```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json scripts/visualization/example_bleu4_data.json \
    --output outputs/visualizations/bleu4_chart_paper.png \
    --style-preset paper
```

### 条纹风格
```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json scripts/visualization/example_bleu4_data.json \
    --output outputs/visualizations/bleu4_chart_striped.png \
    --style-preset striped
```

### 学术风格
```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json scripts/visualization/example_bleu4_data.json \
    --output outputs/visualizations/bleu4_chart_academic.png \
    --style-preset academic
```

## 自定义填充图案

如果你想自己控制每个柱子的样式：

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json scripts/visualization/example_bleu4_data.json \
    --output outputs/visualizations/bleu4_chart_custom.png \
    --color-palette paper \
    --hatch-patterns "/" "/" "." "none" "none" \
    --edgecolors "#666666" "#666666" "#666666" "#8b7355" "#8b7355"
```

这个命令会：
- 前两个柱子：对角条纹 `/`
- 第三个柱子：点状 `.`
- 后两个柱子：实心 `none`
- 使用柔和的米色和棕色

## 从命令行直接输入数据

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --model-sizes "1B" "2B" "4B" "8B" \
    --bleu4 0.0564 0.0615 0.0672 0.0710 \
    --output outputs/visualizations/bleu4_chart.png \
    --style-preset professional
```




