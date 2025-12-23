"""
Visualization script for training metrics.
Supports comparing MDLM vs AR models on the same dataset.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob


def load_metrics(metrics_file):
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_training_curves(metrics_files, output_dir='./plots', model_names=None, title_suffix=""):
    """
    Plot training curves from one or more metrics files.
    
    Args:
        metrics_files: List of paths to metrics JSON files, or a directory containing metrics files
        output_dir: Directory to save plots
        model_names: Optional list of model names for legend (default: from metrics files)
        title_suffix: Optional suffix for plot titles
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle directory input
    if isinstance(metrics_files, str) and os.path.isdir(metrics_files):
        metrics_files = glob.glob(os.path.join(metrics_files, '*_metrics.json'))
    
    if not metrics_files:
        print(f"No metrics files found!")
        return
    
    # Load all metrics
    all_metrics = []
    for mf in metrics_files:
        try:
            metrics = load_metrics(mf)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Warning: Could not load {mf}: {e}")
            continue
    
    if not all_metrics:
        print("No valid metrics files found!")
        return
    
    # Use provided names or extract from metrics
    if model_names is None:
        model_names = [m.get('model_type', 'Unknown') for m in all_metrics]
    
    # Plot 1: Step-level loss (all steps)
    plt.figure(figsize=(12, 6))
    for metrics, name in zip(all_metrics, model_names):
        if 'steps' in metrics and 'step_losses' in metrics:
            steps = metrics['steps']
            losses = metrics['step_losses']
            # Downsample if too many points for performance
            if len(steps) > 10000:
                indices = np.linspace(0, len(steps)-1, 10000, dtype=int)
                steps = [steps[i] for i in indices]
                losses = [losses[i] for i in indices]
            plt.plot(steps, losses, label=name, alpha=0.7, linewidth=0.5)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss per Step{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_per_step.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'loss_per_step.png')}")
    plt.close()
    
    # Plot 2: Epoch-level average loss
    plt.figure(figsize=(10, 6))
    for metrics, name in zip(all_metrics, model_names):
        if 'epochs' in metrics and 'epoch_losses' in metrics:
            epochs = metrics['epochs']
            losses = metrics['epoch_losses']
            plt.plot(epochs, losses, label=name, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Training Loss per Epoch{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_per_epoch.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'loss_per_epoch.png')}")
    plt.close()
    
    # Plot 3: Smoothed step-level loss (moving average)
    plt.figure(figsize=(12, 6))
    window_size = 100
    for metrics, name in zip(all_metrics, model_names):
        if 'steps' in metrics and 'step_losses' in metrics:
            steps = metrics['steps']
            losses = metrics['step_losses']
            # Apply moving average
            if len(losses) > window_size:
                smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = steps[window_size-1:]
                # Downsample if too many points
                if len(smoothed_steps) > 5000:
                    indices = np.linspace(0, len(smoothed_steps)-1, 5000, dtype=int)
                    smoothed_steps = [smoothed_steps[i] for i in indices]
                    smoothed = [smoothed[i] for i in indices]
                plt.plot(smoothed_steps, smoothed, label=name, linewidth=1.5)
    plt.xlabel('Training Step')
    plt.ylabel('Smoothed Loss (100-step moving average)')
    plt.title(f'Smoothed Training Loss{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_smoothed.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'loss_smoothed.png')}")
    plt.close()
    
    # Plot 4: Comparison plot (epoch loss side-by-side)
    if len(all_metrics) > 1:
        plt.figure(figsize=(12, 6))
        for metrics, name in zip(all_metrics, model_names):
            if 'epochs' in metrics and 'epoch_losses' in metrics:
                epochs = metrics['epochs']
                losses = metrics['epoch_losses']
                plt.plot(epochs, losses, label=name, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title(f'Model Comparison: Training Loss{title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
        print(f"Saved: {os.path.join(output_dir, 'model_comparison.png')}")
        plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")


def print_metrics_summary(metrics_files):
    """Print a summary of training metrics."""
    if isinstance(metrics_files, str) and os.path.isdir(metrics_files):
        metrics_files = glob.glob(os.path.join(metrics_files, '*_metrics.json'))
    
    for mf in metrics_files:
        try:
            metrics = load_metrics(mf)
            print(f"\n{'='*60}")
            print(f"Experiment: {metrics.get('experiment_name', 'Unknown')}")
            print(f"Model Type: {metrics.get('model_type', 'Unknown')}")
            print(f"Start Time: {metrics.get('start_time', 'Unknown')}")
            print(f"End Time: {metrics.get('end_time', 'Unknown')}")
            
            if 'epoch_losses' in metrics and metrics['epoch_losses']:
                print(f"Epochs: {len(metrics['epoch_losses'])}")
                print(f"Initial Loss: {metrics['epoch_losses'][0]:.4f}")
                print(f"Final Loss: {metrics['epoch_losses'][-1]:.4f}")
                if len(metrics['epoch_losses']) > 1:
                    improvement = metrics['epoch_losses'][0] - metrics['epoch_losses'][-1]
                    print(f"Improvement: {improvement:.4f} ({improvement/metrics['epoch_losses'][0]*100:.2f}%)")
            
            if 'steps' in metrics:
                print(f"Total Steps: {len(metrics['steps'])}")
        except Exception as e:
            print(f"Error loading {mf}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('metrics_files', nargs='+', 
                        help='Path(s) to metrics JSON file(s), or directory containing metrics files')
    parser.add_argument('--output_dir', type=str, default='./plots',
                        help='Directory to save plots (default: ./plots)')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Custom names for models in legend (default: from metrics files)')
    parser.add_argument('--title', type=str, default='',
                        help='Suffix for plot titles')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of metrics instead of plotting')
    
    args = parser.parse_args()
    
    if args.summary:
        print_metrics_summary(args.metrics_files)
    else:
        plot_training_curves(args.metrics_files, args.output_dir, args.names, args.title)


if __name__ == '__main__':
    main()


