"""
Visualization script for training metrics.
Produces:
1. Training + validation loss graph for each full training run
2. Comparison graph with training + validation loss for both MDLM and AR for each epoch size
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict

# Set professional style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Color palette
MDLM_COLOR = '#1f77b4'  # Blue
AR_COLOR = '#ff7f0e'    # Orange


def load_metrics(metrics_file):
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def extract_epoch_count(metrics_file, metrics):
    """Extract epoch count from filename or metrics."""
    filename = os.path.basename(metrics_file)
    epoch_match = re.search(r'(\d+)\s*epoch', filename, re.IGNORECASE)
    if epoch_match:
        return int(epoch_match.group(1))
    
    exp_name = metrics.get('experiment_name', '')
    epoch_match = re.search(r'(\d+)\s*epoch', exp_name, re.IGNORECASE)
    if epoch_match:
        return int(epoch_match.group(1))
    
    if 'epoch_losses' in metrics:
        return len(metrics['epoch_losses'])
    
    return None


def extract_model_type(metrics_file, metrics):
    """Extract model type from metrics or filename."""
    model_type = metrics.get('model_type', '').upper()
    if model_type in ['MDLM', 'AR']:
        return model_type
    
    filename = os.path.basename(metrics_file)
    exp_name = metrics.get('experiment_name', '')
    combined = f"{filename} {exp_name}".lower()
    
    if 'mdlm' in combined or 'diffusion' in combined:
        return 'MDLM'
    elif 'ar' in combined or 'autoregressive' in combined:
        return 'AR'
    
    return 'UNKNOWN'


def plot_individual_runs(metrics_files, output_dir='./plots'):
    """
    Plot 1: Training loss + validation loss graph for each full training run.
    One plot per run showing all epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle directory input
    if isinstance(metrics_files, list) and len(metrics_files) == 1:
        metrics_files = metrics_files[0]
    if isinstance(metrics_files, str) and os.path.isdir(metrics_files):
        metrics_files = glob.glob(os.path.join(metrics_files, '*_metrics.json'))
    
    if not metrics_files:
        print("No metrics files found!")
        return
    
    # Create individual plots for each run
    for mf in metrics_files:
        try:
            metrics = load_metrics(mf)
            model_type = extract_model_type(mf, metrics)
            epoch_count = extract_epoch_count(mf, metrics)
            
            if not metrics.get('epoch_losses'):
                continue
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot training loss
            epochs = list(range(1, len(metrics['epoch_losses']) + 1))
            color = MDLM_COLOR if model_type == 'MDLM' else AR_COLOR
            ax.plot(epochs, metrics['epoch_losses'], 
                   color=color, linewidth=2.5, marker='o', 
                   markersize=5, alpha=0.8, label='Training Loss', linestyle='-')
            
            # Plot validation loss if available
            if 'validation_losses' in metrics and metrics['validation_losses']:
                val_epochs = list(range(1, len(metrics['validation_losses']) + 1))
                ax.plot(val_epochs, metrics['validation_losses'], 
                       color=color, linewidth=2, marker='s', 
                       markersize=4, alpha=0.7, label='Validation Loss', linestyle='--')
            
            # Formatting
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Loss', fontweight='bold')
            title = f'{model_type} Training: {epoch_count} Epochs'
            if metrics.get('experiment_name'):
                title = f"{metrics['experiment_name']}"
            ax.set_title(title, fontweight='bold', pad=15)
            ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            
            # Save plot
            exp_name = metrics.get('experiment_name', f'{model_type}_{epoch_count}epochs')
            safe_name = re.sub(r'[^\w\-_]', '_', exp_name)
            output_file = os.path.join(output_dir, f'{safe_name}_individual.png')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_file}")
            
        except Exception as e:
            print(f"Warning: Could not plot {mf}: {e}")
            continue


def plot_epoch_comparison(metrics_files, output_dir='./plots'):
    """
    Plot 1 comparison graph: All epoch counts in subplots.
    Each subplot shows MDLM train/val and AR train/val for that epoch count.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle directory input
    if isinstance(metrics_files, list) and len(metrics_files) == 1:
        metrics_files = metrics_files[0]
    if isinstance(metrics_files, str) and os.path.isdir(metrics_files):
        metrics_files = glob.glob(os.path.join(metrics_files, '*_metrics.json'))
    
    if not metrics_files:
        print("No metrics files found!")
        return
    
    # Load and group metrics by model type and epoch count
    metrics_by_model_epoch = defaultdict(lambda: defaultdict(list))
    for mf in metrics_files:
        try:
            metrics = load_metrics(mf)
            epoch_count = extract_epoch_count(mf, metrics)
            model_type = extract_model_type(mf, metrics)
            if epoch_count and model_type in ['MDLM', 'AR']:
                metrics_by_model_epoch[model_type][epoch_count].append((mf, metrics))
        except Exception as e:
            print(f"Warning: Could not load {mf}: {e}")
            continue
    
    if not metrics_by_model_epoch:
        print("No valid metrics files found!")
        return
    
    # Check if we have both MDLM and AR data
    if 'MDLM' not in metrics_by_model_epoch or 'AR' not in metrics_by_model_epoch:
        print("Need both MDLM and AR metrics for comparison!")
        return
    
    # Get common epoch counts
    common_epochs = sorted(set(metrics_by_model_epoch['MDLM'].keys()) & 
                          set(metrics_by_model_epoch['AR'].keys()))
    
    if not common_epochs:
        print("No common epoch counts found for both models!")
        return
    
    # Create one graph with subplots - one subplot per epoch count
    n_epochs = len(common_epochs)
    # Arrange subplots in a grid (2 columns)
    n_cols = 2
    n_rows = (n_epochs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    
    # Handle different subplot arrangements
    if n_epochs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, n_cols)
    
    for idx, epoch_count in enumerate(common_epochs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Get MDLM data for this epoch count
        mdlm_runs = metrics_by_model_epoch['MDLM'][epoch_count]
        for _, metrics in mdlm_runs:
            # Plot MDLM training loss
            if 'epoch_losses' in metrics and metrics['epoch_losses']:
                epochs = list(range(1, len(metrics['epoch_losses']) + 1))
                ax.plot(epochs, metrics['epoch_losses'], 
                       color=MDLM_COLOR, linewidth=2.5, marker='o', 
                       markersize=4, alpha=0.8, label='MDLM Training', linestyle='-')
            
            # Plot MDLM validation loss if available
            if 'validation_losses' in metrics and metrics['validation_losses']:
                val_epochs = list(range(1, len(metrics['validation_losses']) + 1))
                ax.plot(val_epochs, metrics['validation_losses'], 
                       color=MDLM_COLOR, linewidth=2, marker='o', 
                       markersize=3, alpha=0.6, label='MDLM Validation', linestyle=':')
            break  # Use first run if multiple
        
        # Get AR data for this epoch count
        ar_runs = metrics_by_model_epoch['AR'][epoch_count]
        for _, metrics in ar_runs:
            # Plot AR training loss
            if 'epoch_losses' in metrics and metrics['epoch_losses']:
                epochs = list(range(1, len(metrics['epoch_losses']) + 1))
                ax.plot(epochs, metrics['epoch_losses'], 
                       color=AR_COLOR, linewidth=2.5, marker='s', 
                       markersize=4, alpha=0.8, label='AR Training', linestyle='--')
            
            # Plot AR validation loss if available
            if 'validation_losses' in metrics and metrics['validation_losses']:
                val_epochs = list(range(1, len(metrics['validation_losses']) + 1))
                ax.plot(val_epochs, metrics['validation_losses'], 
                       color=AR_COLOR, linewidth=2, marker='s', 
                       markersize=3, alpha=0.6, label='AR Validation', linestyle='-.')
            break  # Use first run if multiple
        
        # Formatting for each subplot
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'{epoch_count} Epochs', fontweight='bold', pad=10)
        ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
    
    # Hide unused subplots
    for idx in range(n_epochs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')
    
    plt.suptitle('MDLM vs AR Comparison: Training and Validation Loss by Epoch Count', 
                fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('metrics_files', nargs='+', 
                        help='Path(s) to metrics JSON file(s), or directory containing metrics files')
    parser.add_argument('--output_dir', type=str, default='./plots',
                        help='Directory to save plots (default: ./plots)')
    
    args = parser.parse_args()
    
    # Generate both types of plots
    print("Generating individual run plots...")
    plot_individual_runs(args.metrics_files, args.output_dir)
    
    print("\nGenerating comparison plots...")
    plot_epoch_comparison(args.metrics_files, args.output_dir)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
