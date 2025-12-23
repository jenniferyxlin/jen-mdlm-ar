#!/bin/bash
# Training script for MDLM and AR models on WikiText-2-v1 dataset
# Automatically runs all epoch sizes (5, 10, 20, 50, 100, 200, 500) for both models
# Runs sequentially (one after another) using a single port

# Change to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Verify train.py exists
if [ ! -f "train.py" ]; then
    echo "Error: train.py not found in $SCRIPT_DIR"
    echo "Current directory: $(pwd)"
    echo "Please make sure you're in the project directory."
    exit 1
fi

echo "=========================================="
echo "WikiText-2 Training: All Epoch Sizes"
echo "=========================================="
echo "Running from: $(pwd)"
echo ""

# Epoch sizes to run
EPOCHS=(5 10 20 50 100 200 500)
MODEL_TYPES=("mdlm" "ar")

# Base configuration
MASTER_PORT=25419
MASTER_ADDR=localhost
BATCH_SIZE=16
D_MODEL=768
N_LAYERS=12
N_HEADS=12
MAX_SEQ_LEN=512
LR=1e-4
SAVE_DIR="./checkpoints"

# Memory optimization for CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Create logs directory
mkdir -p logs

# Function to run training and wait for completion
run_training() {
    local model_type=$1
    local epochs=$2
    local log_file="logs/${model_type}_${epochs}epochs.log"
    
    echo "----------------------------------------"
    echo "Starting: ${model_type^^} with ${epochs} epochs"
    echo "Log: ${log_file}"
    echo "----------------------------------------"
    
    # Run training in foreground (will wait for completion)
    torchrun --nproc_per_node=8 --master-port=$MASTER_PORT train.py \
        --distributed \
        --model_type $model_type \
        --hf_raw_repo "Salesforce/wikitext" \
        --hf_raw_split "wikitext-2-v1" \
        --hf_tokenizer "gpt2" \
        --hf_text_column "text" \
        --epochs $epochs \
        --batch_size $BATCH_SIZE \
        --d_model $D_MODEL \
        --n_layers $N_LAYERS \
        --n_heads $N_HEADS \
        --max_seq_len $MAX_SEQ_LEN \
        --use_rope \
        --lr $LR \
        --experiment_name "${model_type}_wikitext2_${epochs}epochs" \
        --save_dir $SAVE_DIR 2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ ${model_type^^} (${epochs} epochs) completed successfully"
        echo ""
    else
        echo "✗ ${model_type^^} (${epochs} epochs) failed with exit code $exit_code"
        echo "Check $log_file for details"
        echo ""
        return 1
    fi
    
    return 0
}

# Main execution loop (runs sequentially)
TOTAL_JOBS=$((${#EPOCHS[@]} * ${#MODEL_TYPES[@]}))
CURRENT_JOB=0

echo "Total jobs to run: $TOTAL_JOBS"
echo "Epoch sizes: ${EPOCHS[@]}"
echo "Model types: ${MODEL_TYPES[@]}"
echo ""
echo "Starting sequential execution..."
echo ""

START_TIME=$(date +%s)

for epochs in "${EPOCHS[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
        CURRENT_JOB=$((CURRENT_JOB + 1))
        
        echo "=========================================="
        echo "Job $CURRENT_JOB/$TOTAL_JOBS"
        echo "=========================================="
        
        # Run training (blocks until completion)
        if ! run_training $model_type $epochs; then
            echo "Error: Training failed. Stopping execution."
            exit 1
        fi
        
        # Small delay between jobs to ensure clean shutdown
        sleep 5
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "=========================================="
echo "All training runs completed!"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Generate combined comparison plots
echo "=========================================="
echo "Generating combined comparison plots..."
echo "=========================================="
PLOT_OUTPUT_DIR="$SAVE_DIR/plots"

if python visualize_training.py "$SAVE_DIR" --output_dir "$PLOT_OUTPUT_DIR" --compare_epochs; then
    echo ""
    echo "✓ Combined comparison plots generated successfully!"
    echo ""
    echo "Generated plots:"
    echo "  - mdlm_epoch_comparison.png (MDLM across all epoch counts)"
    echo "  - ar_epoch_comparison.png (AR across all epoch counts)"
    echo "  - mdlm_vs_ar_comparison.png (Side-by-side comparison)"
    echo "  - final_loss_mdlm_vs_ar.png (Final loss vs epoch count)"
    echo ""
    echo "Individual run plots are in subdirectories: $PLOT_OUTPUT_DIR/{experiment_name}/"
    echo "  Each run has its own folder with:"
    echo "    - loss_per_step.png"
    echo "    - loss_per_epoch.png"
    echo "    - loss_smoothed.png"
else
    echo ""
    echo "Warning: Failed to generate combined plots."
    echo "You can generate them manually with:"
    echo "  python visualize_training.py $SAVE_DIR --output_dir $PLOT_OUTPUT_DIR --compare_epochs"
    echo ""
fi

echo "Log files are in: ./logs/"
echo "Metrics are in: $SAVE_DIR/"
echo "Plots are in: $PLOT_OUTPUT_DIR/"
echo ""
