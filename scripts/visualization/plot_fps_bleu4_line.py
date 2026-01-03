#!/usr/bin/env python3
"""
FPS vs BLEU4 Line Chart Visualization
Creates line charts comparing BLEU4 scores across different FPS values

Usage:
    # From JSON file:
    python scripts/visualization/plot_fps_bleu4_line.py \
        --json scripts/visualization/example_fps_data.json \
        --output outputs/visualizations/fps_bleu4_line.png
    
    # From command line:
    python scripts/visualization/plot_fps_bleu4_line.py \
        --fps 12 16 20 \
        --bleu4 0.0564 0.0615 0.0672 \
        --output outputs/visualizations/fps_bleu4_line.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

def parse_data_from_json(json_file: str, multi_series: bool = False) -> Union[List[Dict], Tuple[List[int], List[float]]]:
    """
    Parse FPS and BLEU4 scores from JSON file
    
    Expected JSON format (single series):
    {
        "data": [
            {"fps": 12, "bleu4": 0.0564},
            {"fps": 16, "bleu4": 0.0615},
            {"fps": 20, "bleu4": 0.0672}
        ]
    }
    OR
    {
        "12": 0.0564,
        "16": 0.0615,
        "20": 0.0672
    }
    
    Expected JSON format (multiple series):
    {
        "series": [
            {
                "name": "LoRA-1",
                "data": [
                    {"fps": 12, "bleu4": 0.0564},
                    {"fps": 16, "bleu4": 0.0615},
                    {"fps": 20, "bleu4": 0.0672}
                ]
            },
            {
                "name": "LoRA-2",
                "data": [
                    {"fps": 12, "bleu4": 0.0580},
                    {"fps": 16, "bleu4": 0.0620},
                    {"fps": 20, "bleu4": 0.0680}
                ]
            }
        ]
    }
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if multi_series and 'series' in data:
        # Multiple series format
        series_data = []
        for series in data['series']:
            name = series.get('name', 'Series')
            fps_list = []
            bleu4_scores = []
            for item in series['data']:
                fps_list.append(int(item['fps']))
                bleu4_scores.append(float(item['bleu4']))
            series_data.append({
                'name': name,
                'fps': fps_list,
                'bleu4': bleu4_scores
            })
        return series_data
    
    # Single series format (backward compatible)
    fps_list = []
    bleu4_scores = []
    
    if 'data' in data:
        # Format 1: {"data": [{"fps": 12, "bleu4": 0.0564}, ...]}
        for item in data['data']:
            fps_list.append(int(item['fps']))
            bleu4_scores.append(float(item['bleu4']))
    elif isinstance(data, dict):
        # Format 2: {"12": 0.0564, "16": 0.0615, ...}
        for fps_str, bleu4 in sorted(data.items(), key=lambda x: int(x[0])):
            fps_list.append(int(fps_str))
            bleu4_scores.append(float(bleu4))
    else:
        raise ValueError("Unsupported JSON format")
    
    return fps_list, bleu4_scores

def plot_fps_bleu4_line_multi(
    series_data: List[Dict],
    output_path: str,
    title: str = "BLEU4 Performance vs FPS",
    xlabel: str = "FPS",
    ylabel: str = "BLEU4 Score",
    figsize: Tuple[float, float] = (6, 4),
    show_values: bool = True,
    value_fontsize: int = 16,
    line_width: float = 2.5,
    marker_size: int = 10,
    grid: bool = True,
    dpi: int = 300,
    ylim: Optional[Tuple[float, float]] = None
):
    """
    Create a multi-line chart comparing BLEU4 scores across different FPS values for multiple series
    
    Args:
        series_data: List of dicts, each with 'name', 'fps', and 'bleu4' keys
        output_path: Path to save the figure
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        show_values: Whether to show values on markers
        value_fontsize: Font size for values on markers
        line_width: Width of the lines
        marker_size: Size of markers
        grid: Whether to show grid
        dpi: Resolution for saved figure
        ylim: Y-axis limits (min, max)
    """
    # Color palette for different series
    colors = ['#2196F3', '#7BAB62', '#fa8072', '#308BE4', '#FF9800', '#9C27B0', '#00BCD4', '#4CAF50']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all FPS values to determine x-axis range
    all_fps = set()
    for series in series_data:
        all_fps.update(series['fps'])
    
    # Plot each series
    for idx, series in enumerate(series_data):
        fps_values = series['fps']
        bleu4_scores = series['bleu4']
        series_name = series['name']
        
        # Sort data by FPS
        sorted_data = sorted(zip(fps_values, bleu4_scores))
        fps_sorted, bleu4_sorted = zip(*sorted_data)
        
        # Get color and marker for this series
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Plot line
        ax.plot(fps_sorted, bleu4_sorted, 
               linestyle='-', 
               linewidth=line_width, 
               color=color, 
               alpha=0.8, 
               zorder=1,
               label=series_name)
        
        # Plot markers
        ax.plot(fps_sorted, bleu4_sorted, 
               marker=marker, 
               color=color, 
               markersize=marker_size, 
               linewidth=0, 
               markeredgecolor='white',
               markeredgewidth=1.5,
               zorder=2)
        
        # Add value labels on markers
        if show_values:
            for fps, bleu4 in zip(fps_sorted, bleu4_sorted):
                ax.text(fps, bleu4,
                       f'{bleu4:.2f}',
                       ha='center', 
                       va='bottom', 
                       fontsize=value_fontsize,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Customize axes
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('', fontsize=16, fontweight='bold', pad=20)  # Remove title
    
    # Set x-axis ticks to all unique FPS values
    all_fps_sorted = sorted(all_fps)
    ax.set_xticks(all_fps_sorted)
    ax.set_xticklabels([str(fps) for fps in all_fps_sorted], fontsize=16)
    
    # Set x-axis limits with padding
    if len(all_fps_sorted) > 1:
        fps_min = min(all_fps_sorted)
        fps_max = max(all_fps_sorted)
        fps_range = fps_max - fps_min
        padding = fps_range * 0.10
        ax.set_xlim(fps_min - padding, fps_max + padding)
    
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
    grid_alpha = 0.7  # Higher alpha for darker appearance
    ax.grid(True, alpha=grid_alpha, linestyle='--', axis='both', color=grid_color, linewidth=0.5, zorder=0, which='major')  # Major grid lines
    ax.grid(True, alpha=grid_alpha * 0.8, linestyle='--', axis='both', color=grid_color, linewidth=0.4, zorder=0, which='minor')  # Minor grid lines for denser grid
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add legend
    ax.legend(loc='best', fontsize=12, frameon=True)
    
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
        print(f"📊 Saved multi-line chart (PDF) to: {output_path}")
    elif file_ext in ['.png', '.jpg', '.jpeg', '.svg', '.eps']:
        plt.savefig(output_path, format=file_ext[1:], dpi=dpi, bbox_inches='tight')
        print(f"📊 Saved multi-line chart ({file_ext[1:].upper()}) to: {output_path}")
    else:
        if not file_ext:
            output_path = str(output_path_obj) + '.png'
        plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
        print(f"📊 Saved multi-line chart (PNG) to: {output_path}")
    
    print(f"   Series: {[s['name'] for s in series_data]}")
    for series in series_data:
        print(f"   {series['name']}: FPS={series['fps']}, BLEU4={[f'{s:.4f}' for s in series['bleu4']]}")
    
    plt.close()

def plot_fps_bleu4_line(
    fps_values: List[int],
    bleu4_scores: List[float],
    output_path: str,
    title: str = "BLEU4 Performance vs FPS",
    xlabel: str = "FPS",
    ylabel: str = "BLEU4 Score",
    figsize: Tuple[float, float] = (6, 3),
    show_values: bool = True,
    value_fontsize: int = 16,
    line_color: str = '#2196F3',
    line_width: float = 2.5,
    marker: str = 'o',
    marker_size: int = 10,
    marker_color: str = '#1976D2',
    grid: bool = True,
    dpi: int = 300,
    ylim: Optional[Tuple[float, float]] = None
):
    """
    Create a line chart comparing BLEU4 scores across different FPS values
    
    Args:
        fps_values: List of FPS values (e.g., [12, 16, 20])
        bleu4_scores: List of BLEU4 scores
        output_path: Path to save the figure
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        show_values: Whether to show values on markers
        value_fontsize: Font size for values on markers
        line_color: Color of the line
        line_width: Width of the line
        marker: Marker style (e.g., 'o', 's', '^', 'D')
        marker_size: Size of markers
        marker_color: Color of markers
        grid: Whether to show grid
        dpi: Resolution for saved figure
        ylim: Y-axis limits (min, max)
    """
    # Validate inputs
    if len(fps_values) != len(bleu4_scores):
        raise ValueError(f"Number of FPS values ({len(fps_values)}) must match number of BLEU4 scores ({len(bleu4_scores)})")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort data by FPS to ensure proper line connection
    sorted_data = sorted(zip(fps_values, bleu4_scores))
    fps_sorted, bleu4_sorted = zip(*sorted_data)
    
    # Plot line
    line = ax.plot(fps_sorted, bleu4_sorted, 
                   linestyle='-', 
                   linewidth=line_width, 
                   color=line_color, 
                   alpha=0.8, 
                   zorder=1,
                   label='BLEU4 Score')
    
    # Plot markers
    markers = ax.plot(fps_sorted, bleu4_sorted, 
                     marker=marker, 
                     color=marker_color, 
                     markersize=marker_size, 
                     linewidth=0, 
                     markeredgecolor='white',
                     markeredgewidth=1.5,
                     zorder=2)
    
    # Add value labels on markers
    if show_values:
        for fps, bleu4 in zip(fps_sorted, bleu4_sorted):
            ax.text(fps, bleu4,
                   f'{bleu4:.2f}',
                   ha='center', 
                   va='bottom', 
                   fontsize=value_fontsize,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Customize axes
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('', fontsize=16, fontweight='bold', pad=20)  # Remove title
    
    # Set x-axis ticks to exact FPS values
    ax.set_xticks(fps_sorted)
    ax.set_xticklabels([str(fps) for fps in fps_sorted], fontsize=16)
    
    # Set x-axis limits with padding to center first and last points
    if len(fps_sorted) > 1:
        fps_min = min(fps_sorted)
        fps_max = max(fps_sorted)
        fps_range = fps_max - fps_min
        padding = fps_range * 0.10  # 15% padding on each side
        ax.set_xlim(fps_min - padding, fps_max + padding)
    else:
        # Single point: add padding based on the value
        padding = fps_sorted[0] * 0.1
        ax.set_xlim(fps_sorted[0] - padding, fps_sorted[0] + padding)
    
    # Set y-axis limits if provided
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
    grid_alpha = 0.7  # Higher alpha for darker appearance
    ax.grid(True, alpha=grid_alpha, linestyle='--', axis='both', color=grid_color, linewidth=0.5, zorder=0, which='major')  # Major grid lines
    ax.grid(True, alpha=grid_alpha * 0.8, linestyle='--', axis='both', color=grid_color, linewidth=0.4, zorder=0, which='minor')  # Minor grid lines for denser grid
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add legend
    ax.legend(loc='best', fontsize=12, frameon=True)
    
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
    
    print(f"   FPS values: {fps_sorted}")
    print(f"   BLEU4 scores: {[f'{s:.4f}' for s in bleu4_sorted]}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Create line chart comparing BLEU4 scores across different FPS values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON file:
  python plot_fps_bleu4_line.py --json data.json --output chart.png
  
  # From command line:
  python plot_fps_bleu4_line.py --fps 12 16 20 --bleu4 0.0564 0.0615 0.0672 --output chart.png
  
  # With custom styling:
  python plot_fps_bleu4_line.py --json data.json --output chart.png --line-color "#7BAB62" --marker-color "#5A8A4A"
        """
    )
    
    # Data input options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--json', type=str, help='JSON file with FPS and BLEU4 scores')
    data_group.add_argument('--fps', nargs='+', type=int, help='FPS values (e.g., 12 16 20)')
    
    parser.add_argument('--bleu4', nargs='+', type=float, help='BLEU4 scores (required if --fps is used)')
    parser.add_argument('--output', type=str, required=True, help='Output path for the plot')
    
    # Chart customization
    parser.add_argument('--title', type=str, default='BLEU4 Performance vs FPS', help='Chart title')
    parser.add_argument('--xlabel', type=str, default='FPS', help='X-axis label')
    parser.add_argument('--ylabel', type=str, default='BLEU4 Score', help='Y-axis label')
    parser.add_argument('--figsize', type=float, nargs=2, default=[6, 4], metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size (default: 6 4)')
    parser.add_argument('--no-values', action='store_true', help='Hide values on markers')
    parser.add_argument('--value-fontsize', type=int, default=16, help='Font size for values on markers')
    parser.add_argument('--line-color', type=str, default='#2196F3', help='Line color (hex code)')
    parser.add_argument('--line-width', type=float, default=2.5, help='Line width')
    parser.add_argument('--marker', type=str, default='o', help='Marker style (o, s, ^, D, v, p)')
    parser.add_argument('--marker-size', type=int, default=10, help='Marker size')
    parser.add_argument('--marker-color', type=str, default='#1976D2', help='Marker color (hex code)')
    parser.add_argument('--no-grid', action='store_true', help='Hide grid lines')
    parser.add_argument('--ylim', type=float, nargs=2, metavar=('MIN', 'MAX'), help='Y-axis limits')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution for saved figure (default: 300)')
    
    args = parser.parse_args()
    
    # Parse data
    if args.json:
        # Check if JSON contains multiple series
        import json
        with open(args.json, 'r') as f:
            json_data = json.load(f)
        
        if 'series' in json_data:
            # Multiple series mode
            series_data = parse_data_from_json(args.json, multi_series=True)
            # Create output directory if needed
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create multi-series plot
            plot_fps_bleu4_line_multi(
                series_data=series_data,
                output_path=str(output_path),
                title=args.title,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                figsize=tuple(args.figsize),
                show_values=not args.no_values,
                value_fontsize=args.value_fontsize,
                line_width=args.line_width,
                marker_size=args.marker_size,
                grid=not args.no_grid,
                dpi=args.dpi,
                ylim=tuple(args.ylim) if args.ylim else None
            )
        else:
            # Single series mode
            fps_values, bleu4_scores = parse_data_from_json(args.json)
            # Create output directory if needed
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create single-series plot
            plot_fps_bleu4_line(
                fps_values=fps_values,
                bleu4_scores=bleu4_scores,
                output_path=str(output_path),
                title=args.title,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                figsize=tuple(args.figsize),
                show_values=not args.no_values,
                value_fontsize=args.value_fontsize,
                line_color=args.line_color,
                line_width=args.line_width,
                marker=args.marker,
                marker_size=args.marker_size,
                marker_color=args.marker_color,
                grid=not args.no_grid,
                dpi=args.dpi,
                ylim=tuple(args.ylim) if args.ylim else None
            )
    elif args.fps:
        if not args.bleu4:
            parser.error("--bleu4 is required when using --fps")
        if len(args.fps) != len(args.bleu4):
            parser.error(f"Number of FPS values ({len(args.fps)}) must match number of BLEU4 scores ({len(args.bleu4)})")
        fps_values = args.fps
        bleu4_scores = args.bleu4
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create single-series plot
        plot_fps_bleu4_line(
            fps_values=fps_values,
            bleu4_scores=bleu4_scores,
            output_path=str(output_path),
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            figsize=tuple(args.figsize),
            show_values=not args.no_values,
            value_fontsize=args.value_fontsize,
            line_color=args.line_color,
            line_width=args.line_width,
            marker=args.marker,
            marker_size=args.marker_size,
            marker_color=args.marker_color,
            grid=not args.no_grid,
            dpi=args.dpi,
            ylim=tuple(args.ylim) if args.ylim else None
        )
    else:
        parser.error("Either --json or --fps must be provided")
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main()

