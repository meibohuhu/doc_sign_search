#!/bin/bash
# Example script to visualize training loss

# Activate conda environment
export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"

# Check if matplotlib is installed
python -c "import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing matplotlib and numpy..."
    pip install matplotlib numpy --quiet
fi

# Set paths
LOG_FILE="/home/mh2803/projects/sign_language_llm/scripts/cluster_eval/out_20907455.txt"
OUTPUT_DIR="/home/mh2803/projects/sign_language_llm/outputs/visualizations"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📊 Visualizing training loss from: $LOG_FILE"

# Run visualization
python /home/mh2803/projects/sign_language_llm/scripts/visualization/plot_training_loss.py \
    --log-file "$LOG_FILE" \
    --output "$OUTPUT_DIR/training_loss_plot.png" \
    --smoothing 0.9

if [ $? -eq 0 ]; then
    echo "✅ Done! Check output in: $OUTPUT_DIR"
    echo "   - Comprehensive plot: $OUTPUT_DIR/training_loss_plot.png"
    echo "   - Simple plot: $OUTPUT_DIR/training_loss_plot_simple.png"
else
    echo "❌ Error occurred during visualization"
fi

