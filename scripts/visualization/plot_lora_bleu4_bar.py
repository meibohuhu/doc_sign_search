#!/usr/bin/env python3
"""
LoRA vs BLEU4 Line Chart Visualization
Creates line charts comparing BLEU4 scores across different LoRA models

Usage:
    # From JSON file:
python scripts/visualization/plot_lora_bleu4_bar.py     --json scripts/visualization/example_lora_data.json     --output outputs/visualizations/lora_bleu4_line.png
    
    # From command line:
    python scripts/visualization/plot_lora_bleu4_bar.py \
        --lora LoRA-1 LoRA-2 LoRA-3 \
        --bleu4 7.12 7.25 7.35 \
        --output outputs/visualizations/lora_bleu4_bar.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def parse_data_from_json(json_file: str) -> Tuple[List[str], List[float]]:
    """
    Parse LoRA names and BLEU4 scores from JSON file
    
    Expected JSON format:
    {
        "data": [
            {"lora": "LoRA-1", "bleu4": 7.12},
            {"lora": "LoRA-2", "bleu4": 7.25},
            {"lora": "LoRA-3", "bleu4": 7.35}
        ]
    }
    OR
    {
        "LoRA-1": 7.12,
        "LoRA-2": 7.25,
        "LoRA-3": 7.35
    }
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    lora_names = []
    bleu4_scores = []
    
    if 'data' in data:
        # Format 1: {"data": [{"lora": "LoRA-1", "bleu4": 7.12}, ...]}
        for item in data['data']:
            lora_names.append(str(item['lora']))
            bleu4_scores.append(float(item['bleu4']))
    elif isinstance(data, dict):
        # Format 2: {"LoRA-1": 7.12, "LoRA-2": 7.25, ...}
        for lora_name, bleu4 in data.items():
            lora_names.append(str(lora_name))
            bleu4_scores.append(float(bleu4))
    else:
        raise ValueError("Unsupported JSON format")
    
    return lora_names, bleu4_scores

def plot_lora_bleu4_bar(
    lora_names: List[str],
    bleu4_scores: List[float],
    output_path: str,
    title: str = "BLEU4 Performance by LoRA Model",
    xlabel: str = "LoRA",
    ylabel: str = "BLEU4 Score",
    figsize: Tuple[float, float] = (6, 3),
    show_values: bool = True,
    value_fontsize: int = 16,
    bar_color: str = '#E85A4A',
    bar_width: float = 0.6,
    grid: bool = True,
    dpi: int = 300,
    ylim: Optional[Tuple[float, float]] = None
):
    """
    Create a line chart comparing BLEU4 scores across different LoRA models
    
    Args:
        lora_names: List of LoRA model names
        bleu4_scores: List of BLEU4 scores
        output_path: Path to save the figure
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        show_values: Whether to show values on markers
        value_fontsize: Font size for values on markers
        bar_color: Color of the line and markers (kept for backward compatibility)
        bar_width: Not used in line chart (kept for backward compatibility)
        grid: Whether to show grid
        dpi: Resolution for saved figure
        ylim: Y-axis limits (min, max)
    """
    # Validate inputs
    if len(lora_names) != len(bleu4_scores):
        raise ValueError(f"Number of LoRA names ({len(lora_names)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x positions
    x_pos = np.arange(len(lora_names))
    
    # Plot line
    line = ax.plot(x_pos, bleu4_scores, 
                   linestyle='-', 
                   linewidth=2.5, 
                   color=bar_color, 
                   alpha=0.8, 
                   zorder=1,
                   label='BLEU4 Score')
    
    # Plot markers
    markers = ax.plot(x_pos, bleu4_scores, 
                     marker='o', 
                     color=bar_color, 
                     markersize=10, 
                     linewidth=0, 
                     markeredgecolor='white',
                     markeredgewidth=1.5,
                     zorder=2)
    
    # Add value labels on markers
    if show_values:
        for x, score in zip(x_pos, bleu4_scores):
            ax.text(x, score,
                   f'{score:.2f}',
                   ha='center', 
                   va='bottom', 
                   fontsize=value_fontsize,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Customize axes
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('', fontsize=16, fontweight='bold', pad=20)  # Remove title
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lora_names, fontsize=16, rotation=0, ha='center')
    
    # Set y-axis limits
    if ylim:
        ax.set_ylim(ylim)
    else:
        # Default Y-axis range: 6-8
        ax.set_ylim(6, 8)
    
    # Set y-axis ticks to integers only
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set y-axis tick label font size
    ax.tick_params(axis='y', labelsize=16)
    
    # Set background color to white
    ax.set_facecolor('white')
    
    # Enable minor ticks for denser grid
    ax.minorticks_on()
    
    # Add grid with darker beige color - both horizontal and vertical, more dense
    # Always show grid, regardless of grid parameter
    # Darker grid for better visibility
    grid_color = '#D4C4A8'  # Darker beige color
    grid_alpha = 0.6  # Higher alpha for darker appearance
    ax.grid(True, alpha=grid_alpha, linestyle='--', axis='both', color=grid_color, linewidth=0.5, zorder=0, which='major')  # Major grid lines
    ax.grid(True, alpha=grid_alpha * 0.8, linestyle='--', axis='both', color=grid_color, linewidth=0.4, zorder=0, which='minor')  # Minor grid lines for denser grid
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Set x-axis limits with padding
    if len(x_pos) > 1:
        padding = (x_pos[-1] - x_pos[0]) * 0.10
        ax.set_xlim(x_pos[0] - padding, x_pos[-1] + padding)
    else:
        padding = 0.5
        ax.set_xlim(x_pos[0] - padding, x_pos[0] + padding)
    
    # Tight layout and save
    plt.tight_layout()
    
    # Determine output format from file extension
    output_path_obj = Path(output_path)
    file_ext = output_path_obj.suffix.lower()
    
    # Set format and DPI based on file extension
    if file_ext == '.pdf':
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📊 Saved line chart (PDF) to: {output_path}")
    elif file_ext in ['.png', '.jpg', '.jpeg', '.svg', '.eps']:
        plt.savefig(output_path, format=file_ext[1:], dpi=dpi, bbox_inches='tight')
        print(f"📊 Saved line chart ({file_ext[1:].upper()}) to: {output_path}")
    else:
        if not file_ext:
            output_path = str(output_path_obj) + '.png'
        plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
        print(f"📊 Saved line chart (PNG) to: {output_path}")
    
    print(f"   LoRA models: {lora_names}")
    print(f"   BLEU4 scores: {[f'{s:.4f}' for s in bleu4_scores]}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Create line chart comparing BLEU4 scores across different LoRA models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON file:
  python plot_lora_bleu4_bar.py --json data.json --output chart.png
  
  # From command line:
  python plot_lora_bleu4_bar.py --lora LoRA-1 LoRA-2 LoRA-3 --bleu4 7.12 7.25 7.35 --output chart.png
  
  # With custom styling:
  python plot_lora_bleu4_bar.py --json data.json --output chart.png --bar-color "#308BE4"
        """
    )
    
    # Data input options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--json', type=str, help='JSON file with LoRA names and BLEU4 scores')
    data_group.add_argument('--lora', nargs='+', type=str, help='LoRA model names (e.g., LoRA-1 LoRA-2 LoRA-3)')
    
    parser.add_argument('--bleu4', nargs='+', type=float, help='BLEU4 scores (required if --lora is used)')
    parser.add_argument('--output', type=str, required=True, help='Output path for the plot')
    
    # Chart customization
    parser.add_argument('--title', type=str, default='BLEU4 Performance by LoRA', help='Chart title')
    parser.add_argument('--xlabel', type=str, default='LoRA', help='X-axis label')
    parser.add_argument('--ylabel', type=str, default='BLEU4 Score', help='Y-axis label')
    parser.add_argument('--figsize', type=float, nargs=2, default=[6, 4], metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size (default: 6 4)')
    parser.add_argument('--no-values', action='store_true', help='Hide values on markers')
    parser.add_argument('--value-fontsize', type=int, default=16, help='Font size for values on markers')
    parser.add_argument('--bar-color', type=str, default='#E85A4A', help='Line and marker color (hex code)')
    parser.add_argument('--bar-width', type=float, default=0.6, help='Not used in line chart (kept for compatibility)')
    parser.add_argument('--no-grid', action='store_true', help='Hide grid lines')
    parser.add_argument('--ylim', type=float, nargs=2, metavar=('MIN', 'MAX'), help='Y-axis limits')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution for saved figure (default: 300)')
    
    args = parser.parse_args()
    
    # Parse data
    if args.json:
        lora_names, bleu4_scores = parse_data_from_json(args.json)
    elif args.lora:
        if not args.bleu4:
            parser.error("--bleu4 is required when using --lora")
        if len(args.lora) != len(args.bleu4):
            parser.error(f"Number of LoRA names ({len(args.lora)}) must match number of BLEU4 scores ({len(args.bleu4)})")
        lora_names = args.lora
        bleu4_scores = args.bleu4
    else:
        parser.error("Either --json or --lora must be provided")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the plot
    plot_lora_bleu4_bar(
        lora_names=lora_names,
        bleu4_scores=bleu4_scores,
        output_path=str(output_path),
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        figsize=tuple(args.figsize),
        show_values=not args.no_values,
        value_fontsize=args.value_fontsize,
        bar_color=args.bar_color,
        bar_width=args.bar_width,
        grid=not args.no_grid,
        dpi=args.dpi,
        ylim=tuple(args.ylim) if args.ylim else None
    )
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main()

