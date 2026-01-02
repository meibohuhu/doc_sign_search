#!/bin/bash
# Example script to plot BLEU4 bar chart

# Activate conda environment (adjust path as needed)
# export PATH="/home/mh2803/miniconda3/envs/qwenvl/bin:$PATH"

# Check if matplotlib is installed
python -c "import matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing matplotlib and numpy..."
    pip install matplotlib numpy --quiet
fi

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="${SCRIPT_DIR}/example_bleu4_data.json"
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/visualizations"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📊 Creating BLEU4 bar chart from: $DATA_FILE"

# Example 1: Professional style with stripes and patterns (like the example image)
python "${SCRIPT_DIR}/plot_bleu4_bar_chart.py" \
    --json "$DATA_FILE" \
    --output "${OUTPUT_DIR}/bleu4_chart_professional.png" \
    --style-preset professional \
    --title "BLEU4 Performance by Model Size"

# Example 2: Paper/academic style (soft beige colors)
python "${SCRIPT_DIR}/plot_bleu4_bar_chart.py" \
    --json "$DATA_FILE" \
    --output "${OUTPUT_DIR}/bleu4_chart_paper.png" \
    --style-preset paper \
    --title "BLEU4 Performance by Model Size"

# Example 3: Striped style (like GPT models in the example)
python "${SCRIPT_DIR}/plot_bleu4_bar_chart.py" \
    --json "$DATA_FILE" \
    --output "${OUTPUT_DIR}/bleu4_chart_striped.png" \
    --style-preset striped \
    --title "BLEU4 Performance by Model Size"

# Example 4: Custom hatch patterns (diagonal stripes, dots, solid)
python "${SCRIPT_DIR}/plot_bleu4_bar_chart.py" \
    --json "$DATA_FILE" \
    --output "${OUTPUT_DIR}/bleu4_chart_custom_hatch.png" \
    --color-palette paper \
    --hatch-patterns "/" "/" "." "none" "none" \
    --edgecolors "#666666" "#666666" "#666666" "#8b7355" "#8b7355" \
    --title "BLEU4 Performance by Model Size"

# Example 5: Academic style
python "${SCRIPT_DIR}/plot_bleu4_bar_chart.py" \
    --json "$DATA_FILE" \
    --output "${OUTPUT_DIR}/bleu4_chart_academic.png" \
    --style-preset academic \
    --title "BLEU4 Performance by Model Size"

# Example 6: From command line with professional style
python "${SCRIPT_DIR}/plot_bleu4_bar_chart.py" \
    --model-sizes "1B" "2B" "4B" "8B" \
    --bleu4 0.0564 0.0615 0.0672 0.0710 \
    --output "${OUTPUT_DIR}/bleu4_chart_cli.png" \
    --style-preset professional \
    --title "BLEU4 Performance Comparison"

if [ $? -eq 0 ]; then
    echo "✅ Done! Check outputs in: $OUTPUT_DIR"
    echo "   - Professional style: ${OUTPUT_DIR}/bleu4_chart_professional.png"
    echo "   - Paper style: ${OUTPUT_DIR}/bleu4_chart_paper.png"
    echo "   - Striped style: ${OUTPUT_DIR}/bleu4_chart_striped.png"
    echo "   - Custom hatch: ${OUTPUT_DIR}/bleu4_chart_custom_hatch.png"
    echo "   - Academic style: ${OUTPUT_DIR}/bleu4_chart_academic.png"
    echo "   - CLI example: ${OUTPUT_DIR}/bleu4_chart_cli.png"
else
    echo "❌ Error occurred during visualization"
fi

