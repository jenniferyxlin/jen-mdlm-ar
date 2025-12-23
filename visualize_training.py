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
    Creates useful plots showing actual training progress without interpolation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle directory input
    if isinstance(metrics_files, list) and len(metrics_files) == 1:
        metrics_files = metrics_files[0]
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
    
    # Plot 1: Side-by-side MDLM vs AR for each epoch count (most useful!)
    if 'MDLM' in metrics_by_model_epoch and 'AR' in metrics_by_model_epoch:
        common_epochs = sorted(set(metrics_by_model_epoch['MDLM'].keys()) & 
                             set(metrics_by_model_epoch['AR'].keys()))
        
        if common_epochs:
            # Create subplots: one row per epoch count
            n_epochs = len(common_epochs)
            fig, axes = plt.subplots(n_epochs, 1, figsize=(14, 4 * n_epochs))
            if n_epochs == 1:
                axes = [axes]
            
            for idx, epoch_count in enumerate(common_epochs):
                ax = axes[idx]
                
                # Get MDLM data
                mdlm_runs = metrics_by_model_epoch['MDLM'][epoch_count]
                for _, metrics in mdlm_runs:
                    if 'epoch_losses' in metrics and metrics['epoch_losses']:
                        epochs = list(range(1, len(metrics['epoch_losses']) + 1))
                        ax.plot(epochs, metrics['epoch_losses'], 
                               color='#1f77b4', linewidth=2.5, marker='o', 
                               markersize=5, alpha=0.8, label='MDLM Train', linestyle='-')
                    # Plot validation loss if available
                    if 'validation_losses' in metrics and metrics['validation_losses']:
                        val_epochs = list(range(1, len(metrics['validation_losses']) + 1))
                        ax.plot(val_epochs, metrics['validation_losses'], 
                               color='#1f77b4', linewidth=2, marker='o', 
                               markersize=4, alpha=0.6, label='MDLM Val', linestyle=':')
                
                # Get AR data
                ar_runs = metrics_by_model_epoch['AR'][epoch_count]
                for _, metrics in ar_runs:
                    if 'epoch_losses' in metrics and metrics['epoch_losses']:
                        epochs = list(range(1, len(metrics['epoch_losses']) + 1))
                        ax.plot(epochs, metrics['epoch_losses'], 
                               color='#ff7f0e', linewidth=2.5, marker='s', 
                               markersize=5, alpha=0.8, label='AR Train', linestyle='--')
                    # Plot validation loss if available
                    if 'validation_losses' in metrics and metrics['validation_losses']:
                        val_epochs = list(range(1, len(metrics['validation_losses']) + 1))
                        ax.plot(val_epochs, metrics['validation_losses'], 
                               color='#ff7f0e', linewidth=2, marker='s', 
                               markersize=4, alpha=0.6, label='AR Val', linestyle='-.')
                
                ax.set_xlabel('Epoch', fontweight='bold')
                ax.set_ylabel('Loss', fontweight='bold')
                ax.set_title(f'MDLM vs AR: {epoch_count} Epochs (Train & Validation){title_suffix}', 
                            fontweight='bold', pad=10)
                ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, ncol=2)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mdlm_vs_ar_side_by_side.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(output_dir, 'mdlm_vs_ar_side_by_side.png')}")
            plt.close()
    
    # Plot 2: Loss at fixed epoch milestones (e.g., epoch 5, 10, 20, 50)
    if 'MDLM' in metrics_by_model_epoch and 'AR' in metrics_by_model_epoch:
        milestones = [5, 10, 20, 50, 100, 200, 500]
        available_milestones = []
        milestone_data = {'MDLM': {}, 'AR': {}}
        
        for milestone in milestones:
            mdlm_losses = []
            ar_losses = []
            
            # Get MDLM losses at this milestone
            for epoch_count in sorted(metrics_by_model_epoch['MDLM'].keys()):
                if epoch_count >= milestone:
                    runs = metrics_by_model_epoch['MDLM'][epoch_count]
                    for _, metrics in runs:
                        if 'epoch_losses' in metrics and len(metrics['epoch_losses']) >= milestone:
                            mdlm_losses.append(metrics['epoch_losses'][milestone - 1])
            
            # Get AR losses at this milestone
            for epoch_count in sorted(metrics_by_model_epoch['AR'].keys()):
                if epoch_count >= milestone:
                    runs = metrics_by_model_epoch['AR'][epoch_count]
                    for _, metrics in runs:
                        if 'epoch_losses' in metrics and len(metrics['epoch_losses']) >= milestone:
                            ar_losses.append(metrics['epoch_losses'][milestone - 1])
            
            if mdlm_losses or ar_losses:
                available_milestones.append(milestone)
                if mdlm_losses:
                    milestone_data['MDLM'][milestone] = np.mean(mdlm_losses)
                if ar_losses:
                    milestone_data['AR'][milestone] = np.mean(ar_losses)
        
        if available_milestones:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            mdlm_vals = [milestone_data['MDLM'].get(m, np.nan) for m in available_milestones]
            ar_vals = [milestone_data['AR'].get(m, np.nan) for m in available_milestones]
            
            x = np.arange(len(available_milestones))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, mdlm_vals, width, label='MDLM', 
                          color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x + width/2, ar_vals, width, label='AR', 
                          color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Epoch Milestone', fontweight='bold')
            ax.set_ylabel('Average Loss', fontweight='bold')
            ax.set_title(f'Loss at Fixed Epoch Milestones: MDLM vs AR{title_suffix}', 
                        fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([f'Epoch {m}' for m in available_milestones])
            ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'loss_at_milestones.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(output_dir, 'loss_at_milestones.png')}")
            plt.close()
    
    # Plot 3: Final loss vs epoch count (improved)
    if 'MDLM' in metrics_by_model_epoch and 'AR' in metrics_by_model_epoch:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Collect final losses (both train and validation)
        mdlm_train_data = []
        mdlm_val_data = []
        ar_train_data = []
        ar_val_data = []
        
        for epoch_count in sorted(metrics_by_model_epoch['MDLM'].keys()):
            runs = metrics_by_model_epoch['MDLM'][epoch_count]
            for _, metrics in runs:
                if 'epoch_losses' in metrics and metrics['epoch_losses']:
                    mdlm_train_data.append((epoch_count, metrics['epoch_losses'][-1]))
                if 'validation_losses' in metrics and metrics['validation_losses']:
                    mdlm_val_data.append((epoch_count, metrics['validation_losses'][-1]))
        
        for epoch_count in sorted(metrics_by_model_epoch['AR'].keys()):
            runs = metrics_by_model_epoch['AR'][epoch_count]
            for _, metrics in runs:
                if 'epoch_losses' in metrics and metrics['epoch_losses']:
                    ar_train_data.append((epoch_count, metrics['epoch_losses'][-1]))
                if 'validation_losses' in metrics and metrics['validation_losses']:
                    ar_val_data.append((epoch_count, metrics['validation_losses'][-1]))
        
        # Plot training losses
        if mdlm_train_data:
            mdlm_x, mdlm_y = zip(*sorted(mdlm_train_data))
            ax.plot(mdlm_x, mdlm_y, 'o-', color='#1f77b4', linewidth=3, 
                   markersize=10, label='MDLM Train', alpha=0.8, markeredgecolor='black', 
                   markeredgewidth=1.5)
        
        if ar_train_data:
            ar_x, ar_y = zip(*sorted(ar_train_data))
            ax.plot(ar_x, ar_y, 's--', color='#ff7f0e', linewidth=3, 
                   markersize=10, label='AR Train', alpha=0.8, markeredgecolor='black', 
                   markeredgewidth=1.5)
        
        # Plot validation losses if available
        if mdlm_val_data:
            mdlm_val_x, mdlm_val_y = zip(*sorted(mdlm_val_data))
            ax.plot(mdlm_val_x, mdlm_val_y, 'o:', color='#1f77b4', linewidth=2.5, 
                   markersize=8, label='MDLM Val', alpha=0.7, markeredgecolor='black', 
                   markeredgewidth=1.2)
        
        if ar_val_data:
            ar_val_x, ar_val_y = zip(*sorted(ar_val_data))
            ax.plot(ar_val_x, ar_val_y, 's-.', color='#ff7f0e', linewidth=2.5, 
                   markersize=8, label='AR Val', alpha=0.7, markeredgecolor='black', 
                   markeredgewidth=1.2)
        
        ax.set_xlabel('Number of Training Epochs', fontweight='bold')
        ax.set_ylabel('Final Loss', fontweight='bold')
        ax.set_title(f'Final Loss vs Training Duration: MDLM vs AR (Train & Validation){title_suffix}', 
                     fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_loss_mdlm_vs_ar.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'final_loss_mdlm_vs_ar.png')}")
        plt.close()
    
    # Plot 4: Overlay comparison (all runs on same plot, no interpolation)
    if 'MDLM' in metrics_by_model_epoch and 'AR' in metrics_by_model_epoch:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        common_epochs = sorted(set(metrics_by_model_epoch['MDLM'].keys()) & 
                             set(metrics_by_model_epoch['AR'].keys()))
        
        for epoch_count in common_epochs:
            # Plot MDLM
            mdlm_runs = metrics_by_model_epoch['MDLM'][epoch_count]
            for _, metrics in mdlm_runs:
                if 'epoch_losses' in metrics and metrics['epoch_losses']:
                    epochs = list(range(1, len(metrics['epoch_losses']) + 1))
                    ax.plot(epochs, metrics['epoch_losses'], 
                           color='#1f77b4', linewidth=2, marker='o', markersize=4, 
                           alpha=0.7, linestyle='-', 
                           label=f'MDLM ({epoch_count} epochs)' if epoch_count == common_epochs[0] else '')
            
            # Plot AR
            ar_runs = metrics_by_model_epoch['AR'][epoch_count]
            for _, metrics in ar_runs:
                if 'epoch_losses' in metrics and metrics['epoch_losses']:
                    epochs = list(range(1, len(metrics['epoch_losses']) + 1))
                    ax.plot(epochs, metrics['epoch_losses'], 
                           color='#ff7f0e', linewidth=2, marker='s', markersize=4, 
                           alpha=0.7, linestyle='--',
                           label=f'AR ({epoch_count} epochs)' if epoch_count == common_epochs[0] else '')
        
        # Add legend entries for all epoch counts
        from matplotlib.lines import Line2D
        legend_elements = []
        for epoch_count in common_epochs:
            legend_elements.append(Line2D([0], [0], color='#1f77b4', linewidth=2.5, 
                                         marker='o', linestyle='-', label=f'MDLM Train ({epoch_count} epochs)'))
            legend_elements.append(Line2D([0], [0], color='#1f77b4', linewidth=1.5, 
                                         marker='o', linestyle=':', alpha=0.6, label=f'MDLM Val ({epoch_count} epochs)'))
            legend_elements.append(Line2D([0], [0], color='#ff7f0e', linewidth=2.5, 
                                         marker='s', linestyle='--', 
                                         label=f'AR Train ({epoch_count} epochs)'))
            legend_elements.append(Line2D([0], [0], color='#ff7f0e', linewidth=1.5, 
                                         marker='s', linestyle='-.', alpha=0.6, 
                                         label=f'AR Val ({epoch_count} epochs)'))
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title(f'MDLM vs AR Loss Comparison - Train & Validation (All Runs){title_suffix}', 
                     fontweight='bold', pad=20)
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9, 
                 fancybox=True, shadow=True, ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mdlm_vs_ar_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'mdlm_vs_ar_comparison.png')}")
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
