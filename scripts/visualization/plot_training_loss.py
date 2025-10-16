#!/usr/bin/env python3
"""
Training Loss Visualization Script
Parses training logs and creates loss diagrams
"""

import re
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import ast

def parse_log_file(log_file):
    """
    Parse training log file and extract loss values
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Dictionary containing parsed training metrics
    """
    steps = []
    losses = []
    learning_rates = []
    grad_norms = []
    epochs = []
    
    print(f"📖 Parsing log file: {log_file}")
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Try to find dictionary patterns like {'loss': 5.0826, ...}
            if 'loss' in line and 'learning_rate' in line:
                try:
                    # Extract the dictionary from the line
                    dict_match = re.search(r'\{[^}]+\}', line)
                    if dict_match:
                        metrics_str = dict_match.group(0)
                        # Use ast.literal_eval to safely parse the dictionary
                        metrics = ast.literal_eval(metrics_str)
                        
                        if 'loss' in metrics:
                            steps.append(len(steps) + 1)
                            losses.append(float(metrics['loss']))
                            learning_rates.append(float(metrics.get('learning_rate', 0)))
                            grad_norms.append(float(metrics.get('grad_norm', 0)))
                            epochs.append(float(metrics.get('epoch', 0)))
                except Exception as e:
                    # Skip lines that can't be parsed
                    continue
    
    print(f"✅ Parsed {len(steps)} training steps")
    
    return {
        'steps': steps,
        'losses': losses,
        'learning_rates': learning_rates,
        'grad_norms': grad_norms,
        'epochs': epochs
    }

def smooth_curve(values, weight=0.9):
    """
    Apply exponential moving average smoothing
    
    Args:
        values: List of values to smooth
        weight: Smoothing weight (0-1), higher = smoother
        
    Returns:
        Smoothed values
    """
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_loss_curves(data, output_path, smoothing_weight=0.9):
    """
    Create comprehensive loss visualization
    
    Args:
        data: Dictionary with training metrics
        output_path: Path to save the figure
        smoothing_weight: Smoothing factor for curves
    """
    steps = data['steps']
    losses = data['losses']
    learning_rates = data['learning_rates']
    grad_norms = data['grad_norms']
    epochs = data['epochs']
    
    # Create figure with 4 subplots (2x2)
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Training Loss (Raw + Smoothed)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(steps, losses, alpha=0.3, color='blue', label='Raw Loss', linewidth=0.5)
    smoothed_loss = smooth_curve(losses, smoothing_weight)
    ax1.plot(steps, smoothed_loss, color='blue', label=f'Smoothed Loss (w={smoothing_weight})', linewidth=2)
    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss vs Epoch
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(epochs, losses, alpha=0.3, s=1, color='green')
    # Group by epoch and calculate mean
    unique_epochs = sorted(set(epochs))
    epoch_mean_loss = []
    for epoch in unique_epochs:
        epoch_losses = [l for e, l in zip(epochs, losses) if e == epoch]
        if epoch_losses:
            epoch_mean_loss.append(np.mean(epoch_losses))
        else:
            epoch_mean_loss.append(0)
    ax2.plot(unique_epochs, epoch_mean_loss, color='darkgreen', linewidth=2, label='Mean Loss per Epoch')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Loss vs Epoch', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Gradient Norm
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(steps, grad_norms, alpha=0.3, color='red', label='Raw Grad Norm', linewidth=0.5)
    smoothed_grad = smooth_curve(grad_norms, smoothing_weight)
    ax3.plot(steps, smoothed_grad, color='red', label=f'Smoothed Grad Norm', linewidth=2)
    ax3.set_xlabel('Training Steps', fontsize=11)
    ax3.set_ylabel('Gradient Norm', fontsize=11)
    ax3.set_title('Gradient Norm Over Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss Distribution (Histogram)
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(losses, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(losses):.3f}')
    ax4.axvline(np.median(losses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(losses):.3f}')
    ax4.set_xlabel('Loss Value', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Loss Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Metrics Visualization\nTotal Steps: {len(steps)} | Final Loss: {losses[-1]:.4f} | Epochs: {epochs[-1]:.2f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved visualization to: {output_path}")
    
    # Also save a simple loss-only plot
    simple_output = output_path.replace('.png', '_simple.png')
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, alpha=0.3, color='blue', label='Raw Loss', linewidth=0.5)
    smoothed_loss = smooth_curve(losses, smoothing_weight)
    plt.plot(steps, smoothed_loss, color='blue', label=f'Smoothed Loss', linewidth=2)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(simple_output, dpi=300, bbox_inches='tight')
    print(f"📊 Saved simple plot to: {simple_output}")

def print_statistics(data):
    """Print training statistics"""
    losses = data['losses']
    grad_norms = data['grad_norms']
    
    print("\n" + "="*60)
    print("📈 Training Statistics")
    print("="*60)
    print(f"Total Training Steps: {len(losses)}")
    print(f"Final Epoch: {data['epochs'][-1]:.2f}")
    print(f"\nLoss Statistics:")
    print(f"  Initial Loss: {losses[0]:.4f}")
    print(f"  Final Loss: {losses[-1]:.4f}")
    print(f"  Min Loss: {min(losses):.4f} (Step {losses.index(min(losses)) + 1})")
    print(f"  Max Loss: {max(losses):.4f} (Step {losses.index(max(losses)) + 1})")
    print(f"  Mean Loss: {np.mean(losses):.4f}")
    print(f"  Median Loss: {np.median(losses):.4f}")
    print(f"  Std Dev: {np.std(losses):.4f}")
    print(f"\nGradient Norm Statistics:")
    print(f"  Mean: {np.mean(grad_norms):.2f}")
    print(f"  Max: {max(grad_norms):.2f}")
    print(f"  Min: {min(grad_norms):.2f}")
    print(f"\nLearning Rate:")
    print(f"  Initial: {data['learning_rates'][0]:.2e}")
    print(f"  Final: {data['learning_rates'][-1]:.2e}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Visualize training loss from log files')
    parser.add_argument('--log-file', type=str, required=True,
                       help='Path to training log file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the plot (default: auto-generated)')
    parser.add_argument('--smoothing', type=float, default=0.9,
                       help='Smoothing weight for curves (0-1, default: 0.9)')
    
    args = parser.parse_args()
    
    # Parse log file
    data = parse_log_file(args.log_file)
    
    if not data['steps']:
        print("❌ No training data found in log file!")
        return
    
    # Generate output path if not provided
    if args.output is None:
        log_path = Path(args.log_file)
        output_path = log_path.parent / f"{log_path.stem}_loss_plot.png"
    else:
        output_path = args.output
    
    # Print statistics
    print_statistics(data)
    
    # Create visualization
    plot_loss_curves(data, str(output_path), args.smoothing)
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main()

