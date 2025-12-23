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
import re
from collections import defaultdict

# Try to use seaborn style, fallback to default if not available
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Professional color palette
COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]


def load_metrics(metrics_file):
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def extract_epoch_count(metrics_file, metrics):
    """Extract epoch count from filename or metrics."""
    # Try to extract from filename (e.g., "mdlm_wikitext2_8gpu_5epochs_metrics.json")
    filename = os.path.basename(metrics_file)
    epoch_match = re.search(r'(\d+)\s*epoch', filename, re.IGNORECASE)
    if epoch_match:
        return int(epoch_match.group(1))
    
    # Try to extract from experiment name
    exp_name = metrics.get('experiment_name', '')
    epoch_match = re.search(r'(\d+)\s*epoch', exp_name, re.IGNORECASE)
    if epoch_match:
        return int(epoch_match.group(1))
    
    # Use number of epochs from metrics
    if 'epoch_losses' in metrics:
        return len(metrics['epoch_losses'])
    
    return None


def extract_model_type(metrics_file, metrics):
    """Extract model type from metrics or filename."""
    # Check metrics first
    model_type = metrics.get('model_type', '').upper()
    if model_type in ['MDLM', 'AR']:
        return model_type
    
    # Try to extract from filename or experiment name
    filename = os.path.basename(metrics_file)
    exp_name = metrics.get('experiment_name', '')
    combined = f"{filename} {exp_name}".lower()
    
    if 'mdlm' in combined or 'diffusion' in combined:
        return 'MDLM'
    elif 'ar' in combined or 'autoregressive' in combined:
        return 'AR'
    
    return 'UNKNOWN'


def plot_epoch_comparison(metrics_files, output_dir='./plots', title_suffix=""):
    """
    Compare loss across different training runs with different epoch counts.
    Groups by model type (MDLM vs AR) and epoch count.
    Creates separate plots for each model type and a combined comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle directory input
    if isinstance(metrics_files, str) and os.path.isdir(metrics_files):
        metrics_files = glob.glob(os.path.join(metrics_files, '*_metrics.json'))
    
    if not metrics_files:
        print(f"No metrics files found!")
        return
    
    # Load and group metrics by model type and epoch count
    metrics_by_model_epoch = defaultdict(lambda: defaultdict(list))
    for mf in metrics_files:
        try:
            metrics = load_metrics(mf)
            epoch_count = extract_epoch_count(mf, metrics)
            model_type = extract_model_type(mf, metrics)
            if epoch_count:
                metrics_by_model_epoch[model_type][epoch_count].append((mf, metrics))
        except Exception as e:
            print(f"Warning: Could not load {mf}: {e}")
            continue
    
    if not metrics_by_model_epoch:
        print("No valid metrics files found!")
        return
    
    # Create separate plots for each model type
    for model_type in ['MDLM', 'AR']:
        if model_type not in metrics_by_model_epoch:
            continue
        
        model_metrics = metrics_by_model_epoch[model_type]
        sorted_epochs = sorted(model_metrics.keys())
        
        if not sorted_epochs:
            continue
        
        # Create comparison plot for this model type
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for idx, epoch_count in enumerate(sorted_epochs):
            runs = model_metrics[epoch_count]
            color = COLORS[idx % len(COLORS)]
            
            # Collect all epoch losses for this epoch count
            all_epoch_losses = []
            for _, metrics in runs:
                if 'epoch_losses' in metrics and metrics['epoch_losses']:
                    all_epoch_losses.append(metrics['epoch_losses'])
            
            if not all_epoch_losses:
                continue
            
            # Find max length
            max_len = max(len(losses) for losses in all_epoch_losses)
            
            # Normalize to same length (interpolate if needed)
            normalized_losses = []
            for losses in all_epoch_losses:
                if len(losses) == max_len:
                    normalized_losses.append(losses)
                else:
                    # Interpolate to max_len
                    x_old = np.linspace(0, 1, len(losses))
                    x_new = np.linspace(0, 1, max_len)
                    interpolated = np.interp(x_new, x_old, losses)
                    normalized_losses.append(interpolated.tolist())
            
            # Calculate mean and std
            normalized_losses = np.array(normalized_losses)
            mean_losses = np.mean(normalized_losses, axis=0)
            std_losses = np.std(normalized_losses, axis=0) if len(normalized_losses) > 1 else np.zeros_like(mean_losses)
            
            # Create epoch numbers
            epochs = np.arange(1, len(mean_losses) + 1)
            
            # Plot with confidence interval
            ax.plot(epochs, mean_losses, label=f'{epoch_count} epochs', 
                    color=color, linewidth=2.5, marker='o', markersize=4, alpha=0.9)
            
            # Add shaded confidence interval if multiple runs
            if len(normalized_losses) > 1:
                ax.fill_between(epochs, 
                              mean_losses - std_losses, 
                              mean_losses + std_losses,
                              color=color, alpha=0.2)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'{model_type} Training Loss Comparison Across Different Epoch Counts{title_suffix}', 
                     fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_type.lower()}_epoch_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, f'{model_type.lower()}_epoch_comparison.png')}")
        plt.close()
    
    # Create combined comparison plot (MDLM vs AR for each epoch count)
    if 'MDLM' in metrics_by_model_epoch and 'AR' in metrics_by_model_epoch:
        # Find common epoch counts
        mdlm_epochs = set(metrics_by_model_epoch['MDLM'].keys())
        ar_epochs = set(metrics_by_model_epoch['AR'].keys())
        common_epochs = sorted(mdlm_epochs & ar_epochs)
        
        if common_epochs:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot MDLM
            for epoch_count in common_epochs:
                runs = metrics_by_model_epoch['MDLM'][epoch_count]
                all_epoch_losses = []
                for _, metrics in runs:
                    if 'epoch_losses' in metrics and metrics['epoch_losses']:
                        all_epoch_losses.append(metrics['epoch_losses'])
                
                if all_epoch_losses:
                    max_len = max(len(losses) for losses in all_epoch_losses)
                    normalized_losses = []
                    for losses in all_epoch_losses:
                        if len(losses) == max_len:
                            normalized_losses.append(losses)
                        else:
                            x_old = np.linspace(0, 1, len(losses))
                            x_new = np.linspace(0, 1, max_len)
                            interpolated = np.interp(x_new, x_old, losses)
                            normalized_losses.append(interpolated.tolist())
                    
                    normalized_losses = np.array(normalized_losses)
                    mean_losses = np.mean(normalized_losses, axis=0)
                    epochs = np.arange(1, len(mean_losses) + 1)
                    
                    ax.plot(epochs, mean_losses, label=f'MDLM ({epoch_count} epochs)', 
                            color='#1f77b4', linewidth=2.5, marker='o', markersize=4, 
                            linestyle='-', alpha=0.9)
            
            # Plot AR
            for epoch_count in common_epochs:
                runs = metrics_by_model_epoch['AR'][epoch_count]
                all_epoch_losses = []
                for _, metrics in runs:
                    if 'epoch_losses' in metrics and metrics['epoch_losses']:
                        all_epoch_losses.append(metrics['epoch_losses'])
                
                if all_epoch_losses:
                    max_len = max(len(losses) for losses in all_epoch_losses)
                    normalized_losses = []
                    for losses in all_epoch_losses:
                        if len(losses) == max_len:
                            normalized_losses.append(losses)
                        else:
                            x_old = np.linspace(0, 1, len(losses))
                            x_new = np.linspace(0, 1, max_len)
                            interpolated = np.interp(x_new, x_old, losses)
                            normalized_losses.append(interpolated.tolist())
                    
                    normalized_losses = np.array(normalized_losses)
                    mean_losses = np.mean(normalized_losses, axis=0)
                    epochs = np.arange(1, len(mean_losses) + 1)
                    
                    ax.plot(epochs, mean_losses, label=f'AR ({epoch_count} epochs)', 
                            color='#ff7f0e', linewidth=2.5, marker='s', markersize=4, 
                            linestyle='--', alpha=0.9)
            
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Loss', fontweight='bold')
            ax.set_title(f'MDLM vs AR Training Loss Comparison{title_suffix}', 
                         fontweight='bold', pad=20)
            ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mdlm_vs_ar_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(output_dir, 'mdlm_vs_ar_comparison.png')}")
            plt.close()
    
    # Create final loss vs epoch count comparison (MDLM vs AR)
    if 'MDLM' in metrics_by_model_epoch and 'AR' in metrics_by_model_epoch:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Collect final losses for MDLM
        mdlm_epochs = []
        mdlm_final_losses = []
        for epoch_count in sorted(metrics_by_model_epoch['MDLM'].keys()):
            runs = metrics_by_model_epoch['MDLM'][epoch_count]
            for _, metrics in runs:
                if 'epoch_losses' in metrics and metrics['epoch_losses']:
                    mdlm_epochs.append(epoch_count)
                    mdlm_final_losses.append(metrics['epoch_losses'][-1])
        
        # Collect final losses for AR
        ar_epochs = []
        ar_final_losses = []
        for epoch_count in sorted(metrics_by_model_epoch['AR'].keys()):
            runs = metrics_by_model_epoch['AR'][epoch_count]
            for _, metrics in runs:
                if 'epoch_losses' in metrics and metrics['epoch_losses']:
                    ar_epochs.append(epoch_count)
                    ar_final_losses.append(metrics['epoch_losses'][-1])
        
        # Plot MDLM
        if mdlm_epochs:
            ax.scatter(mdlm_epochs, mdlm_final_losses, s=200, c='#1f77b4', 
                      alpha=0.7, edgecolors='black', linewidths=2, zorder=3, 
                      label='MDLM', marker='o')
            # Connect points with line
            sorted_mdlm = sorted(zip(mdlm_epochs, mdlm_final_losses))
            if len(sorted_mdlm) > 1:
                mdlm_x, mdlm_y = zip(*sorted_mdlm)
                ax.plot(mdlm_x, mdlm_y, '--', alpha=0.4, linewidth=2, color='#1f77b4', zorder=1)
        
        # Plot AR
        if ar_epochs:
            ax.scatter(ar_epochs, ar_final_losses, s=200, c='#ff7f0e', 
                      alpha=0.7, edgecolors='black', linewidths=2, zorder=3, 
                      label='AR', marker='s')
            # Connect points with line
            sorted_ar = sorted(zip(ar_epochs, ar_final_losses))
            if len(sorted_ar) > 1:
                ar_x, ar_y = zip(*sorted_ar)
                ax.plot(ar_x, ar_y, '--', alpha=0.4, linewidth=2, color='#ff7f0e', zorder=1)
        
        ax.set_xlabel('Number of Epochs', fontweight='bold')
        ax.set_ylabel('Final Loss', fontweight='bold')
        ax.set_title(f'Final Loss vs Epoch Count: MDLM vs AR{title_suffix}', 
                     fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_loss_mdlm_vs_ar.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'final_loss_mdlm_vs_ar.png')}")
        plt.close()


def plot_training_curves(metrics_files, output_dir='./plots', model_names=None, title_suffix=""):
    """
    Plot training curves from one or more metrics files with improved styling.
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
        model_names = []
        for m in all_metrics:
            exp_name = m.get('experiment_name', 'Unknown')
            epoch_count = extract_epoch_count('', m)
            if epoch_count:
                name = f"{exp_name} ({epoch_count} epochs)"
            else:
                name = exp_name
            model_names.append(name)
    
    # Plot 1: Epoch-level average loss (improved styling)
    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, (metrics, name) in enumerate(zip(all_metrics, model_names)):
        if 'epochs' in metrics and 'epoch_losses' in metrics:
            epochs = metrics['epochs']
            losses = metrics['epoch_losses']
            color = COLORS[idx % len(COLORS)]
            ax.plot(epochs, losses, label=name, marker='o', linewidth=2.5, 
                   markersize=6, color=color, alpha=0.9, markerfacecolor='white', 
                   markeredgewidth=1.5, markeredgecolor=color)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Average Loss', fontweight='bold')
    ax.set_title(f'Training Loss per Epoch{title_suffix}', fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_per_epoch.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'loss_per_epoch.png')}")
    plt.close()
    
    # Plot 2: Step-level loss (smoothed, improved styling)
    fig, ax = plt.subplots(figsize=(14, 7))
    window_size = 100
    for idx, (metrics, name) in enumerate(zip(all_metrics, model_names)):
        if 'steps' in metrics and 'step_losses' in metrics:
            steps = metrics['steps']
            losses = metrics['step_losses']
            color = COLORS[idx % len(COLORS)]
            
            # Apply Gaussian smoothing for better visualization
            if len(losses) > window_size:
                # Use Gaussian filter for smoother curve
                smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = steps[window_size-1:]
                
                # Downsample if too many points
                if len(smoothed_steps) > 5000:
                    indices = np.linspace(0, len(smoothed_steps)-1, 5000, dtype=int)
                    smoothed_steps = [smoothed_steps[i] for i in indices]
                    smoothed = [smoothed[i] for i in indices]
                
                ax.plot(smoothed_steps, smoothed, label=name, linewidth=2, color=color, alpha=0.85)
            else:
                ax.plot(steps, losses, label=name, linewidth=1.5, color=color, alpha=0.7)
    
    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Smoothed Loss (100-step window)', fontweight='bold')
    ax.set_title(f'Smoothed Training Loss{title_suffix}', fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_smoothed.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'loss_smoothed.png')}")
    plt.close()
    
    # Plot 3: Final loss vs epoch count (Pareto-style)
    if len(all_metrics) > 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        epoch_counts = []
        final_losses = []
        colors_list = []
        
        for idx, metrics in enumerate(all_metrics):
            if 'epoch_losses' in metrics and metrics['epoch_losses']:
                epoch_count = extract_epoch_count('', metrics)
                if epoch_count:
                    epoch_counts.append(epoch_count)
                    final_losses.append(metrics['epoch_losses'][-1])
                    colors_list.append(COLORS[idx % len(COLORS)])
        
        if epoch_counts:
            # Sort by epoch count
            sorted_data = sorted(zip(epoch_counts, final_losses, colors_list))
            epoch_counts, final_losses, colors_list = zip(*sorted_data)
            
            ax.scatter(epoch_counts, final_losses, s=150, c=colors_list, 
                      alpha=0.7, edgecolors='black', linewidths=1.5, zorder=3)
            ax.plot(epoch_counts, final_losses, '--', alpha=0.4, linewidth=1.5, color='gray', zorder=1)
            
            # Add labels
            for ec, fl, cl in zip(epoch_counts, final_losses, colors_list):
                ax.annotate(f'{ec} epochs\nLoss: {fl:.3f}', 
                           (ec, fl), xytext=(10, 10), textcoords='offset points',
                           fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=cl, alpha=0.3, edgecolor='black', linewidth=0.5))
        
        ax.set_xlabel('Number of Epochs', fontweight='bold')
        ax.set_ylabel('Final Loss', fontweight='bold')
        ax.set_title(f'Final Loss vs Training Epochs{title_suffix}', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_loss_vs_epochs.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'final_loss_vs_epochs.png')}")
        plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")


def print_metrics_summary(metrics_files):
    """Print a summary of training metrics."""
    if isinstance(metrics_files, str) and os.path.isdir(metrics_files):
        metrics_files = glob.glob(os.path.join(metrics_files, '*_metrics.json'))
    
    for mf in metrics_files:
        try:
            metrics = load_metrics(mf)
            epoch_count = extract_epoch_count(mf, metrics)
            print(f"\n{'='*60}")
            print(f"Experiment: {metrics.get('experiment_name', 'Unknown')}")
            if epoch_count:
                print(f"Epoch Count: {epoch_count}")
            print(f"Model Type: {metrics.get('model_type', 'Unknown')}")
            print(f"Start Time: {metrics.get('start_time', 'Unknown')}")
            print(f"End Time: {metrics.get('end_time', 'Unknown')}")
            
            if 'epoch_losses' in metrics and metrics['epoch_losses']:
                print(f"Epochs Completed: {len(metrics['epoch_losses'])}")
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
    parser = argparse.ArgumentParser(description='Visualize training metrics with improved styling')
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
    parser.add_argument('--compare_epochs', action='store_true',
                        help='Create epoch comparison plot (groups by epoch count)')
    
    args = parser.parse_args()
    
    if args.summary:
        print_metrics_summary(args.metrics_files)
    else:
        if args.compare_epochs:
            plot_epoch_comparison(args.metrics_files, args.output_dir, args.title)
            # Also plot regular curves for individual runs
            plot_training_curves(args.metrics_files, args.output_dir, args.names, args.title)
        else:
            plot_training_curves(args.metrics_files, args.output_dir, args.names, args.title)


if __name__ == '__main__':
    main()
