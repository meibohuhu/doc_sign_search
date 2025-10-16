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

