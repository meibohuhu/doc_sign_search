# Training Loss Visualization

Scripts to visualize training loss from log files.

## Installation

First, install required packages:

```bash
# Using conda environment
conda activate qwenvl
pip install matplotlib numpy

# Or install in the environment
pip install matplotlib numpy --user
```

## Usage

### Basic Usage

```bash
python scripts/visualization/plot_training_loss.py \
    --log-file scripts/cluster_eval/out_20907455.txt \
    --output outputs/visualizations/training_loss.png \
    --smoothing 0.9
```

### Parameters

- `--log-file`: Path to training log file (required)
- `--output`: Output path for the plot (optional, auto-generated if not provided)
- `--smoothing`: Smoothing weight for curves (0-1, default: 0.9)
  - Higher values = smoother curves
  - 0.0 = no smoothing
  - 0.9 = recommended for noisy training

### Example

```bash
# Use the example script
bash scripts/visualization/plot_loss_example.sh
```

## Output

The script generates two plots:

1. **Comprehensive Plot** (`training_loss_plot.png`):
   - Training loss (raw + smoothed)
   - Loss vs epoch
   - Learning rate schedule
   - Gradient norm
   - Loss distribution histogram
   - Rolling statistics

2. **Simple Plot** (`training_loss_plot_simple.png`):
   - Just the training loss curve
   - Good for presentations/papers

## Training Statistics

The script also prints detailed statistics:
- Loss statistics (min, max, mean, median, std)
- Gradient norm statistics
- Learning rate information
- Total steps and epochs

## Troubleshooting

If you get `ModuleNotFoundError: No module named 'matplotlib'`:

```bash
# Activate your conda environment first
conda activate qwenvl
pip install matplotlib numpy
```

Or install in your base environment:

```bash
pip install matplotlib numpy --user
```

---

# BLEU4 Performance Bar Chart Visualization

Scripts to create bar charts comparing BLEU4 scores across different model sizes.

## Installation

Same as above - ensure matplotlib and numpy are installed.

## Usage

### Method 1: From JSON File

Create a JSON file with your data:

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

Or simpler format:

```json
{
  "1B": 0.0564,
  "2B": 0.0615,
  "4B": 0.0672,
  "8B": 0.0710
}
```

Then run:

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json scripts/visualization/example_bleu4_data.json \
    --output outputs/visualizations/bleu4_chart.png
```

### Method 2: From Command Line

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --model-sizes "1B" "2B" "4B" "8B" \
    --bleu4 0.0564 0.0615 0.0672 0.0710 \
    --output outputs/visualizations/bleu4_chart.png
```

### Color Customization

#### Using Color Palettes

Available palettes: `default`, `blue`, `green`, `red`, `purple`, `orange`, `cool`, `warm`, `rainbow`

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json data.json \
    --output chart.png \
    --color-palette blue
```

#### Using Custom Colors

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json data.json \
    --output chart.png \
    --custom-colors "#3498db" "#2ecc71" "#e74c3c" "#f39c12"
```

### Advanced Options

```bash
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json data.json \
    --output chart.png \
    --title "My Custom Title" \
    --xlabel "Model Size" \
    --ylabel "BLEU4 Score" \
    --figsize 12 8 \
    --bar-width 0.7 \
    --rotation 45 \
    --ylim 0 0.1 \
    --dpi 300
```

### Parameters

- `--json`: JSON file with model sizes and BLEU4 scores
- `--model-sizes`: Model sizes (e.g., 1B 2B 4B) - required if not using --json
- `--bleu4`: BLEU4 scores - required if using --model-sizes
- `--output`: Output path for the plot (required)
- `--title`: Chart title (default: "BLEU4 Performance by Model Size")
- `--xlabel`: X-axis label (default: "Model Size")
- `--ylabel`: Y-axis label (default: "BLEU4 Score")
- `--color-palette`: Color palette name (default: "default")
- `--custom-colors`: Custom colors (hex codes, overrides palette)
- `--figsize`: Figure size WIDTH HEIGHT (default: 10 6)
- `--bar-width`: Bar width 0-1 (default: 0.6)
- `--no-values`: Hide values on top of bars
- `--value-fontsize`: Font size for values (default: 10)
- `--rotation`: Rotation angle for x-axis labels in degrees (default: 0)
- `--ylim`: Y-axis limits MIN MAX
- `--no-grid`: Hide grid
- `--dpi`: Resolution for saved figure (default: 300)

### Example Script

```bash
# Run the example script
bash scripts/visualization/plot_bleu4_example.sh
```

This will generate multiple example charts with different color schemes.

