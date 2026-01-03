#!/usr/bin/env python3
"""
BLEU4 Performance Bar Chart Visualization
Creates bar charts comparing BLEU4 scores across different model sizes

Dataset comparison mode (2 datasets, 3 metrics each):
python scripts/visualization/plot_bleu4_bar_chart.py \
    --dataset-comparison \
    --json scripts/visualization/dataset_comparison.json \
    --output outputs/visualizations/dataset_comparison.pdf

Standard mode:
python scripts/visualization/plot_bleu4_bar_chart.py \
    --json scripts/visualization/example_bleu4_data.json \
    --output outputs/visualizations/bleu4_chart.pdf


python scripts/visualization/plot_bleu4_bar_chart.py \
    --dataset-comparison \
    --json scripts/visualization/example_dataset_comparison.json \
    --output outputs/visualizations/dataset_comparison.pdf \
    --figsize 10 6
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def parse_data_from_json(json_file: str, grouped: bool = False) -> Tuple[List[str], List[float], Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
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
    
    Expected JSON format (with ROUGE):
    {
        "models": [
            {"model_size": "1B", "bleu4": 0.0564, "rouge": 16.32},
            ...
        ]
    }
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    model_sizes = []
    bleu4_scores = []
    bleu1_scores = []
    rouge_scores = []
    bleu1_right_scores = []
    
    if 'models' in data:
        # Format 1: {"models": [{"model_size": "...", "bleu4": ..., "bleu1": ..., "rouge": ..., "bleu1_right": ...}, ...]}
        for model in data['models']:
            model_sizes.append(str(model['model_size']))
            bleu4_scores.append(float(model['bleu4']))
            if grouped and 'bleu1' in model:
                bleu1_scores.append(float(model['bleu1']))
            if 'rouge' in model:
                rouge_scores.append(float(model['rouge']))
            if 'bleu1_right' in model:
                bleu1_right_scores.append(float(model['bleu1_right']))
    elif isinstance(data, dict):
        # Format 2: {"1B": 0.0564, "2B": 0.0615, ...} or {"1B": {"bleu1": ..., "bleu4": ..., "rouge": ...}, ...}
        for model_size, score_data in data.items():
            model_sizes.append(str(model_size))
            if isinstance(score_data, dict):
                # Nested format: {"1B": {"bleu1": 0.2823, "bleu4": 0.0564, "rouge": 16.32}, ...}
                bleu4_scores.append(float(score_data['bleu4']))
                if grouped and 'bleu1' in score_data:
                    bleu1_scores.append(float(score_data['bleu1']))
                if 'rouge' in score_data:
                    rouge_scores.append(float(score_data['rouge']))
                if 'bleu1_right' in score_data:
                    bleu1_right_scores.append(float(score_data['bleu1_right']))
            else:
                # Simple format: {"1B": 0.0564, ...}
                bleu4_scores.append(float(score_data))
    else:
        raise ValueError("Unsupported JSON format")
    
    if grouped and bleu1_scores:
        return model_sizes, bleu1_scores, bleu4_scores, rouge_scores if rouge_scores else None, bleu1_right_scores if bleu1_right_scores else None
    else:
        return model_sizes, bleu4_scores, None, rouge_scores if rouge_scores else None, bleu1_right_scores if bleu1_right_scores else None

def parse_data_from_args(model_sizes: List[str], bleu4_scores: List[str]) -> Tuple[List[str], List[float]]:
    """Parse model sizes and BLEU4 scores from command line arguments"""
    if len(model_sizes) != len(bleu4_scores):
        raise ValueError(f"Number of model sizes ({len(model_sizes)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    return model_sizes, [float(score) for score in bleu4_scores]

def parse_dataset_comparison_json(json_file: str) -> Tuple[List[str], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Parse dataset comparison data from JSON file
    
    Expected JSON format:
    {
        "datasets": [
            {
                "name": "Original",
                "bleu1": 0.2823,
                "bleu4": 0.0564,
                "rouge": 16.32
            },
            {
                "name": "+3w data",
                "bleu1": 0.2884,
                "bleu4": 0.0615,
                "rouge": 17.45
            }
        ]
    }
    
    Returns:
        (dataset_names, bleu1_original, bleu4_original, rouge_original, bleu1_enhanced, bleu4_enhanced, rouge_enhanced)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    dataset_names = []
    bleu1_scores = []
    bleu4_scores = []
    rouge_scores = []
    
    if 'datasets' in data:
        for dataset in data['datasets']:
            dataset_names.append(str(dataset['name']))
            bleu1_scores.append(float(dataset.get('bleu1', 0)))
            bleu4_scores.append(float(dataset.get('bleu4', 0)))
            rouge_scores.append(float(dataset.get('rouge', 0)))
    else:
        raise ValueError("JSON format must have 'datasets' key")
    
    if len(dataset_names) != 2:
        raise ValueError(f"Expected exactly 2 datasets, got {len(dataset_names)}")
    
    # Return as: original (first) and enhanced (second)
    return (dataset_names, 
            [bleu1_scores[0]], [bleu4_scores[0]], [rouge_scores[0]],  # Original
            [bleu1_scores[1]], [bleu4_scores[1]], [rouge_scores[1]])   # Enhanced

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
    bar_width: float = 0.12,
    show_values: bool = True,
    value_fontsize: int = 20,
    rotation: int = 0,
    ylim: Optional[Tuple[float, float]] = None,
    grid: bool = False,
    dpi: int = 300,
    edgecolor: Optional[str] = None,
    edgecolors: Optional[List[str]] = None,
    bleu1_scores: Optional[List[float]] = None,
    chart_type: str = 'bar',  # 'bar' or 'line'
    rouge_scores: Optional[List[float]] = None,  # ROUGE scores for dual Y-axis
    bleu1_right_scores: Optional[List[float]] = None  # BLEU1 scores for right Y-axis (shared with ROUGE)
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
    
    # Check if ROUGE scores are provided
    has_rouge = rouge_scores is not None and len(rouge_scores) > 0
    
    # Check if BLEU1 scores for right Y-axis are provided
    has_bleu1_right = bleu1_right_scores is not None and len(bleu1_right_scores) > 0
    
    # Validate inputs
    if len(model_sizes) != len(bleu4_scores):
        raise ValueError(f"Number of model sizes ({len(model_sizes)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    if is_grouped and len(bleu1_scores) != len(bleu4_scores):
        raise ValueError(f"Number of BLEU1 scores ({len(bleu1_scores)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    if has_rouge and len(rouge_scores) != len(bleu4_scores):
        raise ValueError(f"Number of ROUGE scores ({len(rouge_scores)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    if has_bleu1_right and len(bleu1_right_scores) != len(bleu4_scores):
        raise ValueError(f"Number of BLEU1 right scores ({len(bleu1_right_scores)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
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
        
        # Unified colors for all models
        # BLEU4 uses #7BAB62 for all models
        bleu4_color = '#7BAB62'
        
        for model_size in model_sizes:
            model_colors.append(bleu4_color)
            model_edge_colors.append(bleu4_color)
        
        # Override colors with model-specific colors
        colors = model_colors
        # Add hatch patterns for BLEU4 bars - no hatch (solid color) when ROUGE or BLEU1 right is present
        if has_rouge or has_bleu1_right:
            hatches = [None] * len(model_sizes)  # Solid color for BLEU4 bars (middle)
        else:
            hatches = [None] * len(model_sizes)  # No hatch patterns when no other metrics
        edge_colors = model_edge_colors
        
        # Create chart based on chart_type
        # Reduce spacing between model groups when ROUGE is present
        if has_rouge:
            x_pos = np.arange(len(model_sizes)) * 0.6  # Tighter spacing for dual bars
        else:
            x_pos = np.arange(len(model_sizes)) * 0.85  # Normal spacing for single bars
        
        if chart_type == 'line':
            # Create line chart with markers - connect all points with a single line
            markers = ['o', 's', '^', 'D', 'v', 'p']  # Different markers for different models
            
            # Draw the connecting line first - use blue color
            line = ax.plot(x_pos, bleu4_scores, linestyle='-', linewidth=2.5, 
                          color='#2196F3', alpha=0.65, zorder=1)  # Blue color
            
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
            # Adjust BLEU4 bar positions when BLEU1 and/or ROUGE are present
            gap = bar_width * 0.05
            if has_rouge and has_bleu1_right:
                # Three bars: BLEU1 (left), BLEU4 (middle), ROUGE (right)
                # BLEU4 should be in the middle
                bleu4_x_positions = [x + bar_width + gap for x in x_pos]
            elif has_bleu1_right:
                # Two bars: BLEU1 (left), BLEU4 (right)
                bleu4_x_positions = [x + bar_width + gap for x in x_pos]
            elif has_rouge:
                # Two bars: BLEU4 (left), ROUGE (right)
                bleu4_x_positions = x_pos
            else:
                # Single bar: BLEU4 only
                bleu4_x_positions = x_pos
            
            for i, (x, score, color, hatch, edge_col) in enumerate(zip(bleu4_x_positions, bleu4_scores, colors, hatches, edge_colors)):
                # Use lighter edge color for softer look
                edge_color_soft = edge_col if edge_col != '#666666' else '#999999'
                
                # Create bar with modern styling: no edge, better alpha
                bar = ax.bar(x, score, width=bar_width, color=color, 
                            edgecolor='none', linewidth=0, alpha=0.65, hatch=hatch,
                            zorder=2)
                # Set hatch color to specified color if hatch pattern is present
                if hatch:
                    # Grid pattern for BLEU4 - use same color as bar
                    hatch_color = '#7BAB62'
                    for patch in bar:
                        patch.set_hatch(hatch)
                        # Set hatch edge color
                        patch.set_edgecolor(hatch_color)
                        patch.set_linewidth(0.8)  # Thinner lines for hatch pattern
                
                # Add subtle shadow effect for depth (optional - can be disabled)
                # Shadow bar slightly offset
                shadow_offset = 0.02
                ax.bar(x + shadow_offset, score, width=bar_width, 
                      color='#000000', alpha=0.05, zorder=1)
                
                bars.append(bar[0])
        
        all_scores = bleu4_scores
    
    # Add ROUGE and BLEU1 bars on secondary Y-axis if provided
    rouge_bars = []
    bleu1_right_bars = []
    ax2 = None
    if (has_rouge or has_bleu1_right) and not is_grouped:
        # Create secondary Y-axis for ROUGE and BLEU1
        ax2 = ax.twinx()
        
        # Calculate positions for bars: BLEU1 (left), BLEU4 (middle), ROUGE (right)
        gap = bar_width * 0.05  # Minimal gap between bars
        if has_rouge and has_bleu1_right:
            # Three bars: BLEU1 (left), BLEU4 (middle), ROUGE (right)
            bleu1_right_x_pos = x_pos  # BLEU1 on the left
            bleu4_x_pos_for_right = x_pos + bar_width + gap  # BLEU4 in the middle (for ROUGE/BLEU1 calculation)
            rouge_x_pos = bleu4_x_pos_for_right + bar_width + gap  # ROUGE on the right
        elif has_rouge:
            # Two bars: BLEU4, ROUGE
            bleu4_x_pos_for_right = x_pos
            rouge_x_pos = x_pos + bar_width + gap
            bleu1_right_x_pos = None
        elif has_bleu1_right:
            # Two bars: BLEU1, BLEU4
            bleu1_right_x_pos = x_pos
            bleu4_x_pos_for_right = x_pos + bar_width + gap
            rouge_x_pos = None
        
        # Colors for ROUGE bars - unified color #fa8072 for all models
        if has_rouge:
            rouge_colors = ['#fa8072'] * len(model_sizes)
            rouge_edge_colors = ['#fa8072'] * len(model_sizes)
            
            # Draw ROUGE bars on secondary axis with backslash hatch pattern (opposite to BLEU1)
            for i, (x, score, color, edge_col) in enumerate(zip(rouge_x_pos, rouge_scores, rouge_colors, rouge_edge_colors)):
                # For PDF compatibility: hatch color is controlled by edgecolor, must be set explicitly
                # Use a slightly darker shade for better visibility in PDF
                hatch_edge_color = '#E85A4A'  # Darker red for hatch lines in PDF
                # Set edgecolor to hatch color, linewidth=0 to hide border but keep hatch visible
                bar = ax2.bar(x, score, width=bar_width, color=color,
                             edgecolor=hatch_edge_color, linewidth=0, alpha=0.60, hatch='\\',
                             zorder=2)
                # Ensure hatch pattern is set - hatch color controlled by edgecolor
                for patch in bar:
                    patch.set_hatch('\\')
                    # Set hatch line color (edgecolor controls hatch color in matplotlib)
                    patch.set_edgecolor(hatch_edge_color)
                    patch.set_linewidth(0.4)  # Thinner hatch lines
                rouge_bars.append(bar[0])
        
        # Colors for BLEU1 bars - unified color #A9B2C3 for all models
        if has_bleu1_right:
            bleu1_right_colors = ['#308BE4'] * len(model_sizes)
            bleu1_right_edge_colors = ['#308BE4'] * len(model_sizes)
            
            # Draw BLEU1 bars on secondary axis with hatch pattern (diagonal lines)
            # Use a visible color for hatch lines in PDF (not too light)
            hatch_color = '#1E6FC7'  # Medium blue for hatch lines - visible in PDF
            for i, (x, score, color, edge_col) in enumerate(zip(bleu1_right_x_pos, bleu1_right_scores, bleu1_right_colors, bleu1_right_edge_colors)):
                # For PDF compatibility: hatch color is controlled by edgecolor, must be set explicitly
                # Set edgecolor to hatch color, linewidth=0 to hide border but keep hatch visible
                bar = ax2.bar(x, score, width=bar_width, color=color,
                             edgecolor=hatch_color, linewidth=0, alpha=0.50, hatch='/',
                             zorder=2)
                # Ensure hatch pattern is set - hatch color controlled by edgecolor
                for patch in bar:
                    patch.set_hatch('/')
                    # Set hatch line color (edgecolor controls hatch color in matplotlib)
                    patch.set_edgecolor(hatch_color)
                    patch.set_linewidth(0.4)  # Thinner hatch lines
                bleu1_right_bars.append(bar[0])
        
        # Set Y-axis label and limits for right axis (ROUGE/BLEU1)
        # Determine the range based on what metrics are present
        if has_rouge and has_bleu1_right:
            # Combine ROUGE and BLEU1 scores to determine range
            all_right_scores = list(rouge_scores) + list(bleu1_right_scores)
            y_min = min(all_right_scores) - 1
            y_max = max(all_right_scores) + 1
            ax2.set_ylabel('ROUGE / BLEU1 Score', fontsize=20, fontweight='normal', color='black')
        elif has_rouge:
            ax2.set_ylabel('ROUGE Score', fontsize=20, fontweight='normal', color='black')
            y_min, y_max = 14, 29
        elif has_bleu1_right:
            ax2.set_ylabel('BLEU1 Score', fontsize=20, fontweight='normal', color='black')
            y_min = min(bleu1_right_scores) - 1
            y_max = max(bleu1_right_scores) + 1
        
        ax2.tick_params(axis='y', labelsize=14, labelcolor='black')
        ax2.set_ylim(y_min, y_max)
        
        # Color the right Y-axis ticks and spine
        ax2.spines['right'].set_color('black')
        ax2.spines['right'].set_linewidth(2)
        ax2.spines['top'].set_visible(False)
    
    # Add value labels on top of bars/points
    if show_values:
        if is_grouped:
            # Labels for grouped bars
            for bar, score in zip(bars_bleu1, bleu1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.1f}',
                       ha='center', va='bottom', fontsize=value_fontsize)
            for bar, score in zip(bars_bleu4, bleu4_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.1f}',
                       ha='center', va='bottom', fontsize=value_fontsize)
        else:
            # Labels for single bars or line points
            if chart_type == 'line':
                # For line chart, label above markers
                for x, score in zip(x_pos, bleu4_scores):
                    ax.text(x, score,
                           f'{score:.1f}',
                           ha='center', va='bottom', fontsize=value_fontsize,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
            else:
                # Labels for bars
                for bar, score in zip(bars, bleu4_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.1f}',
                           ha='center', va='bottom', fontsize=value_fontsize)
                
                # Labels for ROUGE bars if present
                if has_rouge and rouge_bars:
                    for bar, score in zip(rouge_bars, rouge_scores):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{score:.1f}',
                               ha='center', va='bottom', fontsize=value_fontsize,
                               color='black')
                
                # Labels for BLEU1 right bars if present
                if has_bleu1_right and bleu1_right_bars:
                    for bar, score in zip(bleu1_right_bars, bleu1_right_scores):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{score:.1f}',
                               ha='center', va='bottom', fontsize=value_fontsize,
                               color='black')
    
    # Customize axes
    # Remove x-axis label as requested
    ax.set_xlabel('', fontsize=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='normal', color='black')
    # Remove title as requested
    ax.set_title('', fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis ticks - x_pos is already defined in both branches
    # For grouped bars, x_pos represents the center between the two bars
    # For multiple bars (BLEU1 + BLEU4 + ROUGE), adjust tick position to center at BLEU4 (middle bar)
    if (has_rouge or has_bleu1_right) and not is_grouped:
        gap = bar_width * 0.05
        if has_rouge and has_bleu1_right:
            # Three bars: BLEU1 (left), BLEU4 (middle), ROUGE (right) - center at BLEU4
            center_positions = x_pos + bar_width + gap
        elif has_rouge:
            # Two bars: BLEU4, ROUGE - center between them
            center_positions = x_pos + bar_width / 2 + gap / 2
        elif has_bleu1_right:
            # Two bars: BLEU1, BLEU4 - center between them
            center_positions = x_pos + bar_width / 2 + gap / 2
        ax.set_xticks(center_positions)
    else:
        ax.set_xticks(x_pos)
    
    # Split model names into two lines: series name and size
    def split_model_name(name):
        """Split model name like 'QwenVL-3B' into ['QwenVL', '3B']"""
        if '-' in name:
            parts = name.split('-', 1)
            return f'{parts[0]}\n{parts[1]}'
        return name
    
    split_labels = [split_model_name(name) for name in model_sizes]
    ax.set_xticklabels(split_labels, rotation=rotation, fontsize=14, ha='center')
    # Increase y-axis tick label font size
    ax.tick_params(axis='y', labelsize=14, labelcolor='black')
    
    # Set x-axis limits to ensure all bars are visible
    # Add padding on both sides to prevent bars from being cut off
    if is_grouped:
        # For grouped bars, need more space
        x_padding_left = 0.85  # More padding on left to move first bar away from 0
        x_padding_right = 0.6  # More padding on right to ensure last bar is fully visible
    else:
        # For single bars, padding based on bar width
        # More padding on left to move first bar away from 0
        if has_rouge or has_bleu1_right:
            gap = bar_width * 0.05
            if has_rouge and has_bleu1_right:
                # Three bars: BLEU1 (left), BLEU4 (middle), ROUGE (right)
                x_padding_left = bar_width * 1.5
                x_padding_right = (bar_width * 3) + (gap * 2) + 0.08
            else:
                # Two bars side by side
                x_padding_left = bar_width * 1.5
                x_padding_right = (bar_width * 2) + gap + 0.08
        else:
            x_padding_left = bar_width * 1.5
            x_padding_right = bar_width * 0.8
    ax.set_xlim(x_pos[0] - x_padding_left, x_pos[-1] + x_padding_right)
    
    # Set y-axis limits if provided
    if ylim:
        ax.set_ylim(ylim)
    else:
        # Set y-axis to 2.5-8.5 for both chart types
        ax.set_ylim(2.5, 8.5)
    
    # Set background color to white
    ax.set_facecolor('white')  # White background
    
    # Enable minor ticks for denser grid
    ax.minorticks_on()
    
    # Add grid with light beige color - both horizontal and vertical, more dense
    # Always show grid, regardless of grid parameter
    # Lighter grid for PDF compatibility
    grid_color = '#E8D5B7'  # Lighter beige color for PDF
    grid_alpha = 0.5  # Lower alpha for lighter appearance in PDF
    ax.grid(True, alpha=grid_alpha, linestyle='--', axis='both', color=grid_color, linewidth=0.5, zorder=0, which='major')  # Major grid lines
    ax.grid(True, alpha=grid_alpha * 0.7, linestyle='--', axis='both', color=grid_color, linewidth=0.4, zorder=0, which='minor')  # Minor grid lines for denser grid
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove top and right borders, keep bottom and left axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set left Y-axis spine to match right Y-axis thickness
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    
    # Add legend in upper left corner when ROUGE or BLEU1 right is present
    if (has_rouge or has_bleu1_right) and not is_grouped:
        legend_handles = []
        
        # BLEU1 Score: with diagonal hatch pattern
        if has_bleu1_right:
            bleu1_patch = mpatches.Patch(facecolor='#308BE4', edgecolor='#1E6FC7', hatch='/', 
                                         linewidth=0.0, alpha=0.55, label='BLEU1 Score')
            legend_handles.append(bleu1_patch)
        
        # BLEU4 Score: solid color #7BAB62 (middle bar)
        bleu4_patch = mpatches.Patch(facecolor='#7BAB62', edgecolor='#7BAB62', 
                                     alpha=0.55, label='BLEU4 Score')
        legend_handles.append(bleu4_patch)
        
        # ROUGE Score: with backslash hatch pattern
        if has_rouge:
            rouge_patch = mpatches.Patch(facecolor='#fa8072', edgecolor='#E85A4A', hatch='\\',
                                         linewidth=0.0, alpha=0.65, label='ROUGE Score')
            legend_handles.append(rouge_patch)
        
        # Show legend with pattern distinction
        ax.legend(handles=legend_handles, 
                 loc='upper left', fontsize=12, frameon=True, framealpha=0.9)
    
    # Statistics text box removed as requested
    
    # Tight layout and save
    plt.tight_layout()
    
    # Determine output format from file extension
    output_path_obj = Path(output_path)
    file_ext = output_path_obj.suffix.lower()
    
    # Set format and DPI based on file extension
    if file_ext == '.pdf':
        # PDF format - DPI doesn't apply to vector formats
        # Set PDF backend parameters for better hatch rendering
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts
        matplotlib.rcParams['ps.fonttype'] = 42   # TrueType fonts
        plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📊 Saved bar chart (PDF) to: {output_path}")
    elif file_ext in ['.png', '.jpg', '.jpeg', '.svg', '.eps']:
        # Raster or vector formats that support DPI
        plt.savefig(output_path, format=file_ext[1:], dpi=dpi, bbox_inches='tight')
        print(f"📊 Saved bar chart ({file_ext[1:].upper()}) to: {output_path}")
    else:
        # Default to PNG if extension not recognized
        if not file_ext:
            output_path = str(output_path_obj) + '.png'
        plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
        print(f"📊 Saved bar chart (PNG) to: {output_path}")
    print(f"   Model sizes: {model_sizes}")
    if is_grouped:
        print(f"   BLEU1 scores: {[f'{s:.2f}' for s in bleu1_scores]}")
        print(f"   BLEU4 scores: {[f'{s:.2f}' for s in bleu4_scores]}")
    else:
        print(f"   BLEU4 scores: {[f'{s:.2f}' for s in bleu4_scores]}")
    if has_rouge:
        print(f"   ROUGE scores: {[f'{s:.2f}' for s in rouge_scores]}")
    if has_bleu1_right:
        print(f"   BLEU1 right scores: {[f'{s:.2f}' for s in bleu1_right_scores]}")
    
    plt.close()

def plot_dataset_comparison_bar_chart(
    dataset_names: List[str],
    bleu1_original: List[float],
    bleu4_original: List[float],
    rouge_original: List[float],
    bleu1_enhanced: List[float],
    bleu4_enhanced: List[float],
    rouge_enhanced: List[float],
    output_path: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "BLEU4 Score",
    figsize: Tuple[float, float] = (10, 6),
    bar_width: float = 0.06,
    show_values: bool = True,
    value_fontsize: int = 32,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
):
    """
    Create a grouped bar chart comparing metrics across two datasets
    
    Args:
        dataset_names: List of dataset names (should be 2: original and enhanced)
        bleu1_original: BLEU1 scores for original dataset
        bleu4_original: BLEU4 scores for original dataset
        rouge_original: ROUGE scores for original dataset
        bleu1_enhanced: BLEU1 scores for enhanced dataset
        bleu4_enhanced: BLEU4 scores for enhanced dataset
        rouge_enhanced: ROUGE scores for enhanced dataset
        output_path: Path to save the figure
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        bar_width: Width of bars (0-1)
        show_values: Whether to show values on top of bars
        value_fontsize: Font size for values on bars
        ylim: Y-axis limits (min, max)
        dpi: Resolution for saved figure
    """
    if len(dataset_names) != 2:
        raise ValueError(f"Expected exactly 2 datasets, got {len(dataset_names)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create secondary Y-axis for ROUGE (different scale)
    ax2 = ax.twinx()
    
    # X positions for the two datasets - reduce spacing between them significantly
    # Move first dataset to the right by adding an offset
    x_offset = 0.6  # Move first dataset to the right
    spacing = 0.35  # Much smaller spacing between datasets (was 0.6)
    x_pos = np.arange(len(dataset_names)) * spacing + x_offset  # Reduce spacing significantly
    
    # Calculate bar positions: BLEU1 (left), BLEU4 (middle), ROUGE (right)
    # Make bars tightly packed together (no gap or minimal gap)
    gap = bar_width * 0.01  # Very small gap, almost touching
    # Center the three bars around x_pos
    # BLEU4 center should be at x_pos, so BLEU4 left edge = x_pos - bar_width/2
    # BLEU1 right edge = BLEU4 left edge - gap = x_pos - bar_width/2 - gap
    # BLEU1 left edge = BLEU1 right edge - bar_width = x_pos - bar_width*1.5 - gap
    # ROUGE left edge = BLEU4 right edge + gap = x_pos + bar_width/2 + gap
    bleu1_x = x_pos - bar_width * 1.5 - gap
    bleu4_x = x_pos - bar_width / 2
    rouge_x = x_pos + bar_width / 2 + gap
    
    # Colors for each metric
    bleu1_color = '#308BE4'  # Blue
    bleu4_color = '#7BAB62'  # Green
    rouge_color = '#fa8072'   # Salmon
    
    # Hatch colors
    bleu1_hatch_color = '#1E6FC7'
    rouge_hatch_color = '#E85A4A'
    
    # Combine original and enhanced data
    bleu1_scores = [bleu1_original[0], bleu1_enhanced[0]]
    bleu4_scores = [bleu4_original[0], bleu4_enhanced[0]]
    rouge_scores = [rouge_original[0], rouge_enhanced[0]]
    
    # Draw BLEU1 bars (left) - on right Y-axis (same as ROUGE)
    # Match bleu4_chart: alpha=0.50 for BLEU1
    bars_bleu1 = ax2.bar(bleu1_x, bleu1_scores, width=bar_width, 
                       color=bleu1_color, edgecolor=bleu1_hatch_color, 
                       linewidth=0, alpha=0.50, hatch='/', zorder=2, label='BLEU1')
    for patch in bars_bleu1:
        patch.set_hatch('/')
        patch.set_edgecolor(bleu1_hatch_color)
        patch.set_linewidth(0.4)
    
    # Draw BLEU4 bars (middle) - on left Y-axis
    # Match bleu4_chart: alpha=0.65 for BLEU4
    bars_bleu4 = ax.bar(bleu4_x, bleu4_scores, width=bar_width,
                       color=bleu4_color, edgecolor='none', linewidth=0,
                       alpha=0.65, hatch=None, zorder=2, label='BLEU4')
    
    # Draw ROUGE bars (right) - on right Y-axis
    # Match bleu4_chart: alpha=0.60 for ROUGE
    bars_rouge = ax2.bar(rouge_x, rouge_scores, width=bar_width,
                         color=rouge_color, edgecolor=rouge_hatch_color, linewidth=0,
                         alpha=0.60, hatch='\\', zorder=2, label='ROUGE')
    for patch in bars_rouge:
        patch.set_hatch('\\')
        patch.set_edgecolor(rouge_hatch_color)
        patch.set_linewidth(0.4)
    
    # Add value labels
    if show_values:
        # BLEU1 labels - on right Y-axis
        for bar, score in zip(bars_bleu1, bleu1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}',
                   ha='center', va='bottom', fontsize=value_fontsize, color='black')
        
        # BLEU4 labels - on left Y-axis
        for bar, score in zip(bars_bleu4, bleu4_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}',
                   ha='center', va='bottom', fontsize=value_fontsize)
        
        # ROUGE labels - on right Y-axis
        for bar, score in zip(bars_rouge, rouge_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.1f}',
                    ha='center', va='bottom', fontsize=value_fontsize, color='black')
    
    # Customize axes - match bleu4_chart layout
    # Remove x-axis label as requested
    ax.set_xlabel('', fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='normal', color='black')
    # Remove title as requested
    ax.set_title('', fontsize=18, fontweight='bold', pad=20)
    
    # Set x-axis ticks - center at the middle of all three bars
    gap = bar_width * 0.05
    # Center position is x_pos (the original dataset position)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_names, rotation=0, fontsize=18, ha='center')
    ax.tick_params(axis='y', labelsize=14, labelcolor='black')
    
    # Set Y-axis limits for left axis (BLEU4 only)
    if ylim:
        ax.set_ylim(ylim)
    else:
        # Set to 6-10 as requested (BLEU4 only, not BLEU1)
        ax.set_ylim(6, 10)
    
    # Set Y-axis limits for right axis (BLEU1 and ROUGE)
    # Set to 25-40 as requested
    ax2.set_ylim(25, 40)
    ax2.set_ylabel('BLEU1 / ROUGE Score', fontsize=20, fontweight='normal', color='black')
    ax2.tick_params(axis='y', labelsize=14, labelcolor='black')
    
    # Set x-axis limits - add more padding to center the chart and move origin away from left edge
    gap = bar_width * 0.05
    # Calculate total width of three bars: bar_width * 3 + gap * 2
    total_bar_width = bar_width * 3 + gap * 2
    # Add padding on both sides, centered around x_pos
    x_padding = total_bar_width / 2 + 0.15
    ax.set_xlim(x_pos[0] - x_padding, x_pos[-1] + x_padding)
    
    # Set background color to white
    ax.set_facecolor('white')
    
    # Enable minor ticks for denser grid
    ax.minorticks_on()
    
    # Add grid with light beige color - both horizontal and vertical, more dense
    # Always show grid, regardless of grid parameter
    # Lighter grid for PDF compatibility
    grid_color = '#E8D5B7'  # Lighter beige color for PDF
    grid_alpha = 0.5  # Lower alpha for lighter appearance in PDF
    ax.grid(True, alpha=grid_alpha, linestyle='--', axis='both', color=grid_color, linewidth=0.5, zorder=0, which='major')  # Major grid lines
    ax.grid(True, alpha=grid_alpha * 0.7, linestyle='--', axis='both', color=grid_color, linewidth=0.4, zorder=0, which='minor')  # Minor grid lines for denser grid
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove top and right borders, keep bottom and left axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set left Y-axis spine to match right Y-axis thickness
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    # Color the right Y-axis ticks and spine
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_visible(False)
    
    # Add legend - match bleu4_chart legend alpha values
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(facecolor=bleu1_color, edgecolor=bleu1_hatch_color, hatch='/',
                      linewidth=0.0, alpha=0.55, label='BLEU1 Score'),
        mpatches.Patch(facecolor=bleu4_color, edgecolor=bleu4_color,
                      alpha=0.55, label='BLEU4 Score'),
        mpatches.Patch(facecolor=rouge_color, edgecolor=rouge_hatch_color, hatch='\\',
                      linewidth=0.0, alpha=0.65, label='ROUGE Score')
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
    
    # Tight layout with padding to avoid taking full page
    plt.tight_layout(pad=2.0)
    
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
        print(f"📊 Saved dataset comparison chart (PDF) to: {output_path}")
    elif file_ext in ['.png', '.jpg', '.jpeg', '.svg', '.eps']:
        plt.savefig(output_path, format=file_ext[1:], dpi=dpi, bbox_inches='tight')
        print(f"📊 Saved dataset comparison chart ({file_ext[1:].upper()}) to: {output_path}")
    else:
        if not file_ext:
            output_path = str(output_path_obj) + '.png'
        plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
        print(f"📊 Saved dataset comparison chart (PNG) to: {output_path}")
    
    print(f"   Datasets: {dataset_names}")
    print(f"   Original - BLEU1: {[f'{s:.2f}' for s in bleu1_original]}, BLEU4: {[f'{s:.2f}' for s in bleu4_original]}, ROUGE: {[f'{s:.2f}' for s in rouge_original]}")
    print(f"   Enhanced - BLEU1: {[f'{s:.2f}' for s in bleu1_enhanced]}, BLEU4: {[f'{s:.2f}' for s in bleu4_enhanced]}, ROUGE: {[f'{s:.2f}' for s in rouge_enhanced]}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Create bar chart comparing BLEU4 scores across different model sizes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dataset comparison mode (2 datasets, 3 metrics each):
  python plot_bleu4_bar_chart.py --dataset-comparison --json dataset_comparison.json --output comparison.pdf
  
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
    parser.add_argument('--dataset-comparison', action='store_true', 
                       help='Enable dataset comparison mode (2 datasets, 3 metrics each)')
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--json', type=str, help='JSON file with model sizes and BLEU4 scores')
    data_group.add_argument('--model-sizes', nargs='+', help='Model sizes (e.g., 1B 2B 4B)')
    
    parser.add_argument('--bleu4', nargs='+', help='BLEU4 scores (required if --model-sizes is used)')
    parser.add_argument('--bleu1', nargs='+', help='BLEU1 scores for grouped bar chart (optional)')
    parser.add_argument('--rouge', nargs='+', help='ROUGE scores for dual Y-axis chart (optional)')
    parser.add_argument('--bleu1-right', nargs='+', dest='bleu1_right', help='BLEU1 scores for right Y-axis (shared with ROUGE, optional)')
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
    parser.add_argument('--bar-width', type=float, default=0.12, help='Bar width (0-1, default: 0.12)')
    parser.add_argument('--no-values', action='store_true', help='Hide values on top of bars')
    parser.add_argument('--value-fontsize', type=int, default=10, help='Font size for values on bars')
    parser.add_argument('--rotation', type=int, default=0, help='Rotation angle for x-axis labels (degrees)')
    parser.add_argument('--ylim', type=float, nargs=2, metavar=('MIN', 'MAX'), help='Y-axis limits')
    parser.add_argument('--grid', action='store_true', help='Show grid lines (default: no grid)')
    parser.add_argument('--chart-type', type=str, default='bar', choices=['bar', 'line'],
                       help='Chart type: bar (default) or line')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution for saved figure (default: 300)')
    
    args = parser.parse_args()
    
    # Handle dataset comparison mode
    if args.dataset_comparison:
        if not args.json:
            parser.error("--json is required when using --dataset-comparison")
        
        # Parse dataset comparison data
        dataset_names, bleu1_original, bleu4_original, rouge_original, bleu1_enhanced, bleu4_enhanced, rouge_enhanced = parse_dataset_comparison_json(args.json)
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the dataset comparison plot
        # Use argparse default [10, 6] which matches function default (10, 6)
        figsize = tuple(args.figsize)
        plot_dataset_comparison_bar_chart(
            dataset_names=dataset_names,
            bleu1_original=bleu1_original,
            bleu4_original=bleu4_original,
            rouge_original=rouge_original,
            bleu1_enhanced=bleu1_enhanced,
            bleu4_enhanced=bleu4_enhanced,
            rouge_enhanced=rouge_enhanced,
            output_path=str(output_path),
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            figsize=figsize,
            show_values=not args.no_values,
            value_fontsize=args.value_fontsize if args.value_fontsize != 10 else 16,  # Use 32 if default (10) was used
            ylim=tuple(args.ylim) if args.ylim else None,
            dpi=args.dpi
        )
        
        print("\n✅ Visualization complete!")
        return
    
    # Parse data
    bleu1_scores = None
    rouge_scores = None
    bleu1_right_scores = None
    if args.json:
        # Try to parse as grouped first (with BLEU1)
        result = parse_data_from_json(args.json, grouped=True)
        # Check format: result is (model_sizes, bleu4_scores, bleu1_scores or None, rouge_scores or None, bleu1_right_scores or None)
        if len(result) == 5:
            # New format with ROUGE and BLEU1 right: (model_sizes, bleu4_scores, bleu1_scores or None, rouge_scores or None, bleu1_right_scores or None)
            if result[2] is not None:
                # Grouped format: (model_sizes, bleu1_scores, bleu4_scores, rouge_scores, bleu1_right_scores)
                model_sizes, bleu1_scores, bleu4_scores, rouge_scores, bleu1_right_scores = result
            else:
                # Single metric format: (model_sizes, bleu4_scores, None, rouge_scores, bleu1_right_scores)
                model_sizes, bleu4_scores, _, rouge_scores, bleu1_right_scores = result
        elif len(result) == 4:
            # Old format without BLEU1 right: (model_sizes, bleu4_scores, bleu1_scores or None, rouge_scores or None)
            if result[2] is not None:
                # Grouped format: (model_sizes, bleu1_scores, bleu4_scores, rouge_scores)
                model_sizes, bleu1_scores, bleu4_scores, rouge_scores = result
            else:
                # Single metric format: (model_sizes, bleu4_scores, None, rouge_scores)
                model_sizes, bleu4_scores, _, rouge_scores = result
        elif result[2] is not None:
            # Old grouped format: (model_sizes, bleu1_scores, bleu4_scores)
            model_sizes, bleu1_scores, bleu4_scores = result
        else:
            # Old single metric format: (model_sizes, bleu4_scores, None)
            model_sizes, bleu4_scores = result[0], result[1]
    elif args.model_sizes:
        if not args.bleu4:
            parser.error("--bleu4 is required when using --model-sizes")
        model_sizes, bleu4_scores = parse_data_from_args(args.model_sizes, args.bleu4)
        if args.bleu1:
            if len(args.bleu1) != len(bleu4_scores):
                parser.error(f"Number of BLEU1 scores ({len(args.bleu1)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
            bleu1_scores = [float(s) for s in args.bleu1]
        if args.rouge:
            if len(args.rouge) != len(bleu4_scores):
                parser.error(f"Number of ROUGE scores ({len(args.rouge)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
            rouge_scores = [float(s) for s in args.rouge]
        if args.bleu1_right:
            if len(args.bleu1_right) != len(bleu4_scores):
                parser.error(f"Number of BLEU1 right scores ({len(args.bleu1_right)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
            bleu1_right_scores = [float(s) for s in args.bleu1_right]
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
        chart_type=args.chart_type,
        rouge_scores=rouge_scores,
        bleu1_right_scores=bleu1_right_scores
    )
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main()

