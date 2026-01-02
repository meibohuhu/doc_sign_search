#!/usr/bin/env python3
"""
BLEU4 Performance Bar Chart Visualization
Creates bar charts comparing BLEU4 scores across different model sizes

python scripts/visualization/plot_bleu4_bar_chart.py     --json scripts/visualization/example_bleu4_data.json     --output outputs/visualizations/bleu4_chart.png     --style-preset professional
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default color palettes - softer, more professional colors
COLOR_PALETTES = {
    'default': ['#e8e8e8', '#d4d4d4', '#c0c0c0', '#acacac', '#989898'],
    'paper': ['#f5f5dc', '#e8e8d3', '#dcdcc6', '#d0d0b9', '#c4c4ac'],  # Beige/paper tones
    'soft': ['#e8dcc6', '#d4c2a8', '#c0a88a', '#ac8e6c', '#98744e'],  # Soft browns
    'blue': ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5'],
    'green': ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a'],
    'red': ['#ffebee', '#ffcdd2', '#ef9a9a', '#e57373', '#ef5350'],
    'purple': ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc'],
    'orange': ['#fff3e0', '#ffe0b2', '#ffcc80', '#ffb74d', '#ffa726'],
    'cool': ['#e0f2f1', '#b2dfdb', '#80cbc4', '#4db6ac', '#26a69a'],
    'warm': ['#fff8e1', '#ffecb3', '#ffe082', '#ffd54f', '#ffca28'],
    'rainbow': ['#ffcdd2', '#ffe0b2', '#fff9c4', '#c8e6c9', '#b3e5fc', '#c5cae9', '#f8bbd0'],
    'academic': ['#d4e4f7', '#b8d4f0', '#9cc4e9', '#80b4e2', '#64a4db'],  # Academic paper style
}

# Hatch patterns for different styles
HATCH_PATTERNS = {
    'none': None,
    'diagonal': '/',
    'back_diagonal': '\\',
    'cross': 'x',
    'dots': '.',
    'horizontal': '-',
    'vertical': '|',
    'grid': '+',
    'dense_dots': '..',
    'diagonal_grid': 'xx',
}

# Predefined style combinations (color + hatch) for professional look
STYLE_PRESETS = {
    'paper': {
        'colors': ['#f5f5dc', '#e8e8d3', '#dcdcc6', '#d0d0b9', '#c4c4ac'],
        'hatches': [None, None, None, None, None],
        'edgecolors': ['#8b7355', '#8b7355', '#8b7355', '#8b7355', '#8b7355']
    },
    'striped': {
        'colors': ['#e8e8e8', '#e8e8e8', '#e8e8e8', '#e8e8e8', '#e8e8e8'],
        'hatches': ['/', '/', '.', None, None],
        'edgecolors': ['#666666', '#666666', '#666666', '#8b7355', '#8b7355']
    },
    'professional': {
        'colors': ['#e8e8e8', '#e8e8e8', '#e8e8e8', '#f5f5dc', '#d4c2a8', '#c4b299'],
        'hatches': ['/', '/', '.', None, None, None],
        'edgecolors': ['#666666', '#666666', '#666666', '#8b7355', '#8b7355', '#8b7355']
    },
    'academic': {
        'colors': ['#d4e4f7', '#d4e4f7', '#d4e4f7', '#f5f5dc', '#d4c2a8'],
        'hatches': ['/', '/', '.', None, None],
        'edgecolors': ['#4a90a4', '#4a90a4', '#4a90a4', '#8b7355', '#8b7355']
    },
}

def parse_data_from_json(json_file: str, grouped: bool = False) -> Tuple[List[str], List[float], Optional[List[float]]]:
    """
    Parse model sizes and BLEU scores from JSON file
    
    Expected JSON format (single metric):
    {
        "models": [
            {"model_size": "1B", "bleu4": 0.0564},
            {"model_size": "2B", "bleu4": 0.0615},
            ...
        ]
    }
    OR
    {
        "1B": 0.0564,
        "2B": 0.0615,
        ...
    }
    
    Expected JSON format (grouped - BLEU1 and BLEU4):
    {
        "models": [
            {"model_size": "1B", "bleu1": 0.2823, "bleu4": 0.0564},
            {"model_size": "2B", "bleu1": 0.2884, "bleu4": 0.0615},
            ...
        ]
    }
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    model_sizes = []
    bleu4_scores = []
    bleu1_scores = []
    
    if 'models' in data:
        # Format 1: {"models": [{"model_size": "...", "bleu4": ..., "bleu1": ...}, ...]}
        for model in data['models']:
            model_sizes.append(str(model['model_size']))
            bleu4_scores.append(float(model['bleu4']))
            if grouped and 'bleu1' in model:
                bleu1_scores.append(float(model['bleu1']))
    elif isinstance(data, dict):
        # Format 2: {"1B": 0.0564, "2B": 0.0615, ...} or {"1B": {"bleu1": ..., "bleu4": ...}, ...}
        for model_size, score_data in data.items():
            model_sizes.append(str(model_size))
            if isinstance(score_data, dict):
                # Nested format: {"1B": {"bleu1": 0.2823, "bleu4": 0.0564}, ...}
                bleu4_scores.append(float(score_data['bleu4']))
                if grouped and 'bleu1' in score_data:
                    bleu1_scores.append(float(score_data['bleu1']))
            else:
                # Simple format: {"1B": 0.0564, ...}
                bleu4_scores.append(float(score_data))
    else:
        raise ValueError("Unsupported JSON format")
    
    if grouped and bleu1_scores:
        return model_sizes, bleu1_scores, bleu4_scores
    else:
        return model_sizes, bleu4_scores, None

def parse_data_from_args(model_sizes: List[str], bleu4_scores: List[str]) -> Tuple[List[str], List[float]]:
    """Parse model sizes and BLEU4 scores from command line arguments"""
    if len(model_sizes) != len(bleu4_scores):
        raise ValueError(f"Number of model sizes ({len(model_sizes)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    return model_sizes, [float(score) for score in bleu4_scores]

def get_colors(num_bars: int, color_palette: str, custom_colors: Optional[List[str]] = None) -> List[str]:
    """
    Get colors for bars
    
    Args:
        num_bars: Number of bars
        color_palette: Name of color palette to use
        custom_colors: Custom color list (overrides palette)
    
    Returns:
        List of color strings
    """
    if custom_colors:
        # Repeat colors if needed
        colors = custom_colors * ((num_bars // len(custom_colors)) + 1)
        return colors[:num_bars]
    
    palette = COLOR_PALETTES.get(color_palette, COLOR_PALETTES['default'])
    # Repeat palette if needed
    colors = palette * ((num_bars // len(palette)) + 1)
    return colors[:num_bars]

def get_hatches(num_bars: int, hatch_patterns: Optional[List[str]] = None) -> List[Optional[str]]:
    """
    Get hatch patterns for bars
    
    Args:
        num_bars: Number of bars
        hatch_patterns: List of hatch pattern names (None for no hatch)
    
    Returns:
        List of hatch pattern strings (or None)
    """
    if hatch_patterns is None:
        return [None] * num_bars
    
    # Convert pattern names to actual patterns
    patterns = []
    for pattern in hatch_patterns:
        if pattern is None or pattern.lower() == 'none':
            patterns.append(None)
        else:
            patterns.append(HATCH_PATTERNS.get(pattern.lower(), pattern))
    
    # Repeat if needed
    if len(patterns) < num_bars:
        patterns = patterns * ((num_bars // len(patterns)) + 1)
    
    return patterns[:num_bars]

def plot_bleu4_bar_chart(
    model_sizes: List[str],
    bleu4_scores: List[float],
    output_path: str,
    title: str = "BLEU4 Performance by Model Size",
    xlabel: str = "Model Size",
    ylabel: str = "BLEU4 Score",
    color_palette: str = "default",
    custom_colors: Optional[List[str]] = None,
    hatch_patterns: Optional[List[str]] = None,
    style_preset: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    bar_width: float = 0.25,
    show_values: bool = True,
    value_fontsize: int = 18,
    rotation: int = 0,
    ylim: Optional[Tuple[float, float]] = None,
    grid: bool = False,
    dpi: int = 300,
    edgecolor: Optional[str] = None,
    edgecolors: Optional[List[str]] = None,
    bleu1_scores: Optional[List[float]] = None,
    chart_type: str = 'bar'  # 'bar' or 'line'
):
    """
    Create a bar chart comparing BLEU scores across different model sizes
    
    Args:
        model_sizes: List of model size labels (e.g., ["1B", "2B", "4B"])
        bleu4_scores: List of BLEU4 scores
        output_path: Path to save the figure
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color_palette: Color palette name
        custom_colors: Custom color list (overrides palette)
        hatch_patterns: List of hatch pattern names (e.g., ['/', '/', '.', None, None])
        style_preset: Use predefined style preset ('paper', 'striped', 'professional', 'academic')
        figsize: Figure size (width, height)
        bar_width: Width of bars (0-1)
        show_values: Whether to show values on top of bars
        value_fontsize: Font size for values on bars
        rotation: Rotation angle for x-axis labels
        ylim: Y-axis limits (min, max)
        grid: Whether to show grid
        dpi: Resolution for saved figure
        edgecolor: Single edge color for all bars
        edgecolors: List of edge colors for each bar
        bleu1_scores: Optional list of BLEU1 scores for grouped bar chart
    """
    # Check if grouped bar chart
    is_grouped = bleu1_scores is not None and len(bleu1_scores) > 0
    
    # Validate inputs
    if len(model_sizes) != len(bleu4_scores):
        raise ValueError(f"Number of model sizes ({len(model_sizes)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    if is_grouped and len(bleu1_scores) != len(bleu4_scores):
        raise ValueError(f"Number of BLEU1 scores ({len(bleu1_scores)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if is_grouped:
        # Grouped bar chart: BLEU1 and BLEU4
        # Reduce spacing between columns
        x_pos = np.arange(len(model_sizes)) * 0.80  # Reduce spacing between columns
        # Use a smaller bar width for grouped bars to fit two bars nicely
        bar_width_single = bar_width * 0.4  # Each bar is 40% of the original width
        spacing = bar_width_single * 0.3  # Small gap between the two bars
        
        # Calculate positions for grouped bars (symmetric around x_pos)
        # BLEU1 on the left, BLEU4 on the right
        x_bleu1 = x_pos - bar_width_single / 2 - spacing / 2
        x_bleu4 = x_pos + bar_width_single / 2 + spacing / 2
        
        # Colors for BLEU1 and BLEU4 - pure colors only, no patterns
        if style_preset == 'professional':
            color_bleu1 = '#e8e8e8'  # Light gray
            color_bleu4 = '#f5f5dc'  # Beige
            edge_col_bleu1 = '#666666'
            edge_col_bleu4 = '#8b7355'
        elif style_preset == 'paper':
            color_bleu1 = '#e8e8d3'
            color_bleu4 = '#f5f5dc'
            edge_col_bleu1 = '#8b7355'
            edge_col_bleu4 = '#8b7355'
        else:
            # Default colors
            color_bleu1 = '#d4d4d4'  # Light gray
            color_bleu4 = '#e8e8e8'  # Lighter gray
            edge_col_bleu1 = '#666666'
            edge_col_bleu4 = '#666666'
        
        # Draw BLEU1 bars - no hatch patterns
        bars_bleu1 = ax.bar(x_bleu1, bleu1_scores, width=bar_width_single, 
                           color=color_bleu1, edgecolor=edge_col_bleu1, 
                           linewidth=1.2, alpha=0.9, hatch=None, label='BLEU1')
        
        # Draw BLEU4 bars - no hatch patterns
        bars_bleu4 = ax.bar(x_bleu4, bleu4_scores, width=bar_width_single,
                           color=color_bleu4, edgecolor=edge_col_bleu4,
                           linewidth=1.2, alpha=0.9, hatch=None, label='BLEU4')
        
        bars = list(bars_bleu1) + list(bars_bleu4)
        all_scores = bleu1_scores + bleu4_scores
        
        # Update ylabel
        if ylabel == "BLEU4 Score":
            ylabel = "BLEU Score"
        
        # Add legend
        ax.legend(loc='upper left', fontsize=14, frameon=True)
        
    else:
        # Single bar chart: only BLEU4
        # Use style preset if specified
        if style_preset and style_preset in STYLE_PRESETS:
            preset = STYLE_PRESETS[style_preset]
            # Extend colors/hatches/edgecolors if needed
            num_models = len(model_sizes)
            preset_colors = preset['colors']
            preset_hatches = preset['hatches']
            preset_edges = preset['edgecolors']
            
            # Repeat preset lists if we have more models than preset items
            if num_models > len(preset_colors):
                repeat_factor = (num_models // len(preset_colors)) + 1
                colors = (preset_colors * repeat_factor)[:num_models]
                hatches = (preset_hatches * repeat_factor)[:num_models]
                edge_colors = (preset_edges * repeat_factor)[:num_models]
            else:
                colors = preset_colors[:num_models]
                hatches = preset_hatches[:num_models]
                edge_colors = preset_edges[:num_models]
        else:
            # Get colors
            colors = get_colors(len(model_sizes), color_palette, custom_colors)
            # Get hatch patterns
            hatches = get_hatches(len(model_sizes), hatch_patterns)
            # Get edge colors
            if edgecolors:
                edge_colors = edgecolors[:len(model_sizes)]
                if len(edge_colors) < len(model_sizes):
                    edge_colors = edge_colors * ((len(model_sizes) // len(edge_colors)) + 1)
                    edge_colors = edge_colors[:len(model_sizes)]
            elif edgecolor:
                edge_colors = [edgecolor] * len(model_sizes)
            else:
                edge_colors = ['#333333'] * len(model_sizes)  # Default dark gray
        
        # Set colors based on model name - pure colors only, no patterns
        model_colors = []
        model_edge_colors = []
        
        # Unified colors for each series
        qwenvl_color = '#C5D7C4'  # Unified light green for QwenVL
        internvl_color = '#C6D2E4'  # Unified light blue for InternVL
        
        for model_size in model_sizes:
            if 'QwenVL' in model_size:
                # Unified light green for QwenVL
                model_colors.append(qwenvl_color)
                model_edge_colors.append('#C5D7C4')
            elif 'InternVL' in model_size:
                # Unified light blue for InternVL
                model_colors.append(internvl_color)
                model_edge_colors.append('#C6D2E4')
            else:
                # Default color for other models
                model_colors.append('#e8e8e8')
                model_edge_colors.append('#666666')
        
        # Override colors with model-specific colors
        colors = model_colors
        # No hatch patterns - pure colors only
        hatches = [None] * len(model_sizes)
        edge_colors = model_edge_colors
        
        # Create chart based on chart_type
        x_pos = np.arange(len(model_sizes)) * 0.85  # Reduce spacing between columns
        
        if chart_type == 'line':
            # Create line chart with markers - connect all points with a single line
            markers = ['o', 's', '^', 'D', 'v', 'p']  # Different markers for different models
            
            # Draw the connecting line first - use blue color
            line = ax.plot(x_pos, bleu4_scores, linestyle='-', linewidth=2.5, 
                          color='#2196F3', alpha=0.8, zorder=1)  # Blue color
            
            # Then add markers for each point - also use blue with different shades
            lines = []
            blue_shades = ['#1976D2', '#2196F3', '#42A5F5', '#64B5F6', '#90CAF9', '#BBDEFB']  # Different blue shades
            for i, (x, score) in enumerate(zip(x_pos, bleu4_scores)):
                marker = markers[i % len(markers)]
                blue_color = blue_shades[i % len(blue_shades)]
                point = ax.plot(x, score, marker=marker, color=blue_color, 
                              markersize=10, linewidth=0, 
                              markeredgecolor='#1565C0', markeredgewidth=1.2,
                              label=model_sizes[i], zorder=2)
                lines.append(point[0])
            bars = lines  # For compatibility with value labels
        else:
            # Create bars with modern styling
            bars = []
            for i, (x, score, color, hatch, edge_col) in enumerate(zip(x_pos, bleu4_scores, colors, hatches, edge_colors)):
                # Use lighter edge color for softer look
                edge_color_soft = edge_col if edge_col != '#666666' else '#999999'
                
                # Create bar with modern styling: softer edges, better alpha
                bar = ax.bar(x, score, width=bar_width, color=color, 
                            edgecolor=edge_color_soft, linewidth=1.0, alpha=0.95, hatch=hatch,
                            zorder=2)
                
                # Add subtle shadow effect for depth (optional - can be disabled)
                # Shadow bar slightly offset
                shadow_offset = 0.02
                ax.bar(x + shadow_offset, score, width=bar_width, 
                      color='#000000', alpha=0.05, zorder=1)
                
                bars.append(bar[0])
        
        all_scores = bleu4_scores
    
    # Add value labels on top of bars/points
    if show_values:
        if is_grouped:
            # Labels for grouped bars
            for bar, score in zip(bars_bleu1, bleu1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.2f}',
                       ha='center', va='bottom', fontsize=value_fontsize)
            for bar, score in zip(bars_bleu4, bleu4_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.2f}',
                       ha='center', va='bottom', fontsize=value_fontsize)
        else:
            # Labels for single bars or line points
            if chart_type == 'line':
                # For line chart, label above markers
                for x, score in zip(x_pos, bleu4_scores):
                    ax.text(x, score,
                           f'{score:.2f}',
                           ha='center', va='bottom', fontsize=value_fontsize,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
            else:
                # Labels for bars
                for bar, score in zip(bars, bleu4_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.2f}',
                           ha='center', va='bottom', fontsize=value_fontsize)
    
    # Customize axes
    # Remove x-axis label as requested
    ax.set_xlabel('', fontsize=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    # Remove title as requested
    ax.set_title('', fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis ticks - x_pos is already defined in both branches
    # For grouped bars, x_pos represents the center between the two bars
    ax.set_xticks(x_pos)
    
    # Split model names into two lines: series name and size
    def split_model_name(name):
        """Split model name like 'QwenVL-3B' into ['QwenVL', '3B']"""
        if '-' in name:
            parts = name.split('-', 1)
            return f'{parts[0]}\n{parts[1]}'
        return name
    
    split_labels = [split_model_name(name) for name in model_sizes]
    ax.set_xticklabels(split_labels, rotation=rotation, fontsize=18, ha='center')
    # Increase y-axis tick label font size
    ax.tick_params(axis='y', labelsize=18)
    
    # Set x-axis limits to ensure all bars are visible
    # Add padding on both sides to prevent bars from being cut off
    if is_grouped:
        # For grouped bars, need more space
        x_padding_left = 0.85  # More padding on left to move first bar away from 0
        x_padding_right = 0.6  # More padding on right to ensure last bar is fully visible
    else:
        # For single bars, padding based on bar width
        # More padding on left to move first bar away from 0
        x_padding_left = bar_width * 1.5
        x_padding_right = bar_width * 0.8  # More padding on right to ensure last bar is fully visible
    ax.set_xlim(x_pos[0] - x_padding_left, x_pos[-1] + x_padding_right)
    
    # Set y-axis limits if provided
    if ylim:
        ax.set_ylim(ylim)
    else:
        # Set y-axis to 2-10 for both chart types
        ax.set_ylim(2, 10)
    
    # Set background color to white
    ax.set_facecolor('white')  # White background
    
    # Enable minor ticks for denser grid
    ax.minorticks_on()
    
    # Add grid with light beige color - both horizontal and vertical, more dense
    # Always show grid, regardless of grid parameter
    ax.grid(True, alpha=0.6, linestyle='--', axis='both', color='#E8D5B7', linewidth=1.2, zorder=0, which='major')  # Major grid lines
    ax.grid(True, alpha=0.4, linestyle='--', axis='both', color='#E8D5B7', linewidth=0.8, zorder=0, which='minor')  # Minor grid lines for denser grid
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove top and right borders, keep bottom and left axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Statistics text box removed as requested
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"📊 Saved bar chart to: {output_path}")
    print(f"   Model sizes: {model_sizes}")
    if is_grouped:
        print(f"   BLEU1 scores: {[f'{s:.2f}' for s in bleu1_scores]}")
        print(f"   BLEU4 scores: {[f'{s:.2f}' for s in bleu4_scores]}")
    else:
        print(f"   BLEU4 scores: {[f'{s:.2f}' for s in bleu4_scores]}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Create bar chart comparing BLEU4 scores across different model sizes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON file (single metric):
  python plot_bleu4_bar_chart.py --json data.json --output chart.png
  
  # From JSON file (grouped - BLEU1 and BLEU4):
  python plot_bleu4_bar_chart.py --json data.json --output chart.png --style-preset professional
  
  # From command line (single metric):
  python plot_bleu4_bar_chart.py --model-sizes 1B 2B 4B 8B --bleu4 0.0564 0.0615 0.0672 0.0710 --output chart.png
  
  # From command line (grouped):
  python plot_bleu4_bar_chart.py --model-sizes 1B 2B 4B 8B --bleu1 0.2823 0.2884 0.3117 0.3253 --bleu4 0.0564 0.0615 0.0672 0.0710 --output chart.png --style-preset professional
  
  # With custom colors:
  python plot_bleu4_bar_chart.py --json data.json --output chart.png --custom-colors "#3498db" "#2ecc71" "#e74c3c"
  
  # With color palette:
  python plot_bleu4_bar_chart.py --json data.json --output chart.png --color-palette blue
  
Available color palettes: default, paper, soft, blue, green, red, purple, orange, cool, warm, rainbow, academic
Available style presets: paper, striped, professional, academic
Available hatch patterns: /, \\, x, ., -, |, +, .., xx, none
        """
    )
    
    # Data input options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--json', type=str, help='JSON file with model sizes and BLEU4 scores')
    data_group.add_argument('--model-sizes', nargs='+', help='Model sizes (e.g., 1B 2B 4B)')
    
    parser.add_argument('--bleu4', nargs='+', help='BLEU4 scores (required if --model-sizes is used)')
    parser.add_argument('--bleu1', nargs='+', help='BLEU1 scores for grouped bar chart (optional)')
    parser.add_argument('--output', type=str, required=True, help='Output path for the plot')
    
    # Chart customization
    parser.add_argument('--title', type=str, default='BLEU4 Performance by Model Size', help='Chart title')
    parser.add_argument('--xlabel', type=str, default='Model Size', help='X-axis label')
    parser.add_argument('--ylabel', type=str, default='BLEU4 Score', help='Y-axis label')
    parser.add_argument('--color-palette', type=str, default='default', 
                       choices=list(COLOR_PALETTES.keys()),
                       help='Color palette to use')
    parser.add_argument('--custom-colors', nargs='+', help='Custom colors (hex codes, e.g., #3498db #2ecc71)')
    parser.add_argument('--hatch-patterns', nargs='+', 
                       help='Hatch patterns for bars (e.g., / / . none none). Options: /, \\, x, ., -, |, +, .., xx, none')
    parser.add_argument('--style-preset', type=str, choices=list(STYLE_PRESETS.keys()),
                       help='Use predefined style preset: paper, striped, professional, academic')
    parser.add_argument('--edgecolor', type=str, help='Edge color for all bars')
    parser.add_argument('--edgecolors', nargs='+', help='Edge colors for each bar')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 6], metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size (default: 10 6)')
    parser.add_argument('--bar-width', type=float, default=0.4, help='Bar width (0-1, default: 0.4)')
    parser.add_argument('--no-values', action='store_true', help='Hide values on top of bars')
    parser.add_argument('--value-fontsize', type=int, default=18, help='Font size for values on bars')
    parser.add_argument('--rotation', type=int, default=0, help='Rotation angle for x-axis labels (degrees)')
    parser.add_argument('--ylim', type=float, nargs=2, metavar=('MIN', 'MAX'), help='Y-axis limits')
    parser.add_argument('--grid', action='store_true', help='Show grid lines (default: no grid)')
    parser.add_argument('--chart-type', type=str, default='bar', choices=['bar', 'line'],
                       help='Chart type: bar (default) or line')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution for saved figure (default: 300)')
    
    args = parser.parse_args()
    
    # Parse data
    bleu1_scores = None
    if args.json:
        # Try to parse as grouped first (with BLEU1)
        result = parse_data_from_json(args.json, grouped=True)
        # Check if we got grouped format: result[2] is bleu4_scores (not None)
        # or single format: result[2] is None, result[1] is bleu4_scores
        if result[2] is not None:
            # Grouped format: (model_sizes, bleu1_scores, bleu4_scores)
            model_sizes, bleu1_scores, bleu4_scores = result
        else:
            # Single metric format: (model_sizes, bleu4_scores, None)
            model_sizes, bleu4_scores = result[0], result[1]
    elif args.model_sizes:
        if not args.bleu4:
            parser.error("--bleu4 is required when using --model-sizes")
        model_sizes, bleu4_scores = parse_data_from_args(args.model_sizes, args.bleu4)
        if args.bleu1:
            if len(args.bleu1) != len(bleu4_scores):
                parser.error(f"Number of BLEU1 scores ({len(args.bleu1)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
            bleu1_scores = [float(s) for s in args.bleu1]
    else:
        parser.error("Either --json or --model-sizes must be provided")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Update title if grouped
    title = args.title
    if bleu1_scores and title == "BLEU4 Performance by Model Size":
        title = "BLEU1 and BLEU4 Performance by Model Size"
    
    # Create the plot
    plot_bleu4_bar_chart(
        model_sizes=model_sizes,
        bleu4_scores=bleu4_scores,
        output_path=str(output_path),
        title=title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        color_palette=args.color_palette,
        custom_colors=args.custom_colors,
        hatch_patterns=args.hatch_patterns,
        style_preset=args.style_preset,
        figsize=tuple(args.figsize),
        bar_width=args.bar_width,
        show_values=not args.no_values,
        value_fontsize=args.value_fontsize,
        rotation=args.rotation,
        ylim=tuple(args.ylim) if args.ylim else None,
        grid=args.grid,
        dpi=args.dpi,
        edgecolor=args.edgecolor,
        edgecolors=args.edgecolors,
        bleu1_scores=bleu1_scores,
        chart_type=args.chart_type
    )
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main()

