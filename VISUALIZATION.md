# Training Visualization Guide

This guide explains how to visualize training metrics and compare MDLM vs AR models.

## Features

- **Step-level loss tracking**: Every training step is logged
- **Epoch-level metrics**: Average loss per epoch
- **Comparison plots**: Side-by-side comparison of multiple models
- **JSON export**: All metrics saved in structured format for analysis

## Usage

### During Training

Metrics are automatically saved when training. The training script will:
1. Log loss for every training step
2. Save metrics to JSON file in the checkpoint directory
3. File format: `{experiment_name}_metrics.json`

Example training command:
```bash
python train.py \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-103-v1" \
    --experiment_name "mdlm_wikitext_768d" \
    --epochs 5 \
    --save_dir ./checkpoints
```

This will create: `./checkpoints/mdlm_wikitext_768d_metrics.json`

### Visualizing Single Experiment

```bash
python visualize_training.py ./checkpoints/mdlm_wikitext_768d_metrics.json
```

This creates plots in `./plots/`:
- `loss_per_step.png` - Loss for every training step
- `loss_per_epoch.png` - Average loss per epoch
- `loss_smoothed.png` - Smoothed loss curve (moving average)

### Comparing MDLM vs AR Models

After training both models with different `--experiment_name`:

```bash
# Train MDLM
python train.py --experiment_name "mdlm_wikitext" --save_dir ./checkpoints ...

# Train AR (when you have AR training script)
python train_ar.py --experiment_name "ar_wikitext" --save_dir ./checkpoints ...

# Compare both
python visualize_training.py \
    ./checkpoints/mdlm_wikitext_metrics.json \
    ./checkpoints/ar_wikitext_metrics.json \
    --names "MDLM" "AR" \
    --output_dir ./comparison_plots
```

Or if all metrics files are in one directory:
```bash
python visualize_training.py ./checkpoints --output_dir ./plots
```

### Viewing Metrics Summary

Get a quick text summary of training metrics:
```bash
python visualize_training.py ./checkpoints --summary
```

## Metrics File Format

The JSON metrics file contains:
```json
{
  "model_type": "MDLM",
  "experiment_name": "mdlm_wikitext_20241222_120000",
  "start_time": "2024-12-22T12:00:00",
  "end_time": "2024-12-22T16:30:00",
  "epochs": [1, 2, 3, 4, 5],
  "epoch_losses": [3.45, 2.89, 2.34, 2.01, 1.87],
  "steps": [1, 2, 3, ...],
  "step_losses": [3.52, 3.48, 3.45, ...]
}
```

## Plot Types

1. **Loss per Step**: Raw loss values for every training step (may be downsampled for large runs)
2. **Loss per Epoch**: Average loss at the end of each epoch
3. **Smoothed Loss**: Moving average (100-step window) for easier trend visualization
4. **Model Comparison**: Side-by-side epoch loss comparison when multiple models are provided

## Tips for Comparison

1. **Use consistent experiment names**: Include model type, dataset, and config in the name
   - Example: `mdlm_wikitext_768d_12l` vs `ar_wikitext_768d_12l`

2. **Same dataset and splits**: Ensure both models train on the same data

3. **Similar hyperparameters**: For fair comparison, use similar:
   - Batch size
   - Learning rate
   - Model size (d_model, n_layers, n_heads)

4. **Multiple runs**: Consider running each model multiple times and averaging for statistical significance

## Example Workflow

```bash
# 1. Train MDLM
torchrun --nproc_per_node=8 train.py \
    --distributed \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-103-v1" \
    --experiment_name "mdlm_wikitext_h100" \
    --epochs 5 \
    --save_dir ./checkpoints

# 2. Train AR (when available)
torchrun --nproc_per_node=8 train_ar.py \
    --distributed \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-103-v1" \
    --experiment_name "ar_wikitext_h100" \
    --epochs 5 \
    --save_dir ./checkpoints

# 3. Visualize comparison
python visualize_training.py \
    ./checkpoints/mdlm_wikitext_h100_metrics.json \
    ./checkpoints/ar_wikitext_h100_metrics.json \
    --names "MDLM" "AR" \
    --title " - WikiText-103-v1" \
    --output_dir ./comparison_plots
```

