#!/bin/bash
# Training script for MDLM on WikiText-2-v1 dataset
# Make sure you're in the correct directory (e.g., ~/jen-mdlm-ar)

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

echo "Running from: $(pwd)"
echo "Using train.py: $(pwd)/train.py"

# Kill any processes using training ports (optional - uncomment if needed)
# ./kill_port.sh

# Distributed training with 8 GPUs on WikiText-2-v1
# This uses all 8 H100 GPUs for maximum performance

# Set MASTER_PORT environment variable to avoid port conflicts
export MASTER_PORT=25419
export MASTER_ADDR=localhost

# Memory optimization for CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if port is in use (optional - uncomment if needed)
# if lsof -Pi :$MASTER_PORT -sTCP:LISTEN -t >/dev/null ; then
#     echo "Warning: Port $MASTER_PORT is already in use!"
#     echo "You may need to kill the process or use a different port."
# fi

# Option 1: Run in background with nohup (recommended for long training)
# nohup torchrun --nproc_per_node=8 --master-port=25419 train.py \
#     --distributed \
#     --hf_raw_repo "Salesforce/wikitext" \
#     --hf_raw_split "wikitext-2-v1" \
#     --hf_tokenizer "gpt2" \
#     --hf_text_column "text" \
#     --epochs 5 \
#     --batch_size 64 \
#     --d_model 768 \
#     --n_layers 12 \
#     --n_heads 12 \
#     --max_seq_len 512 \
#     --use_rope \
#     --lr 1e-4 \
#     --experiment_name "mdlm_wikitext2_8gpu" \
#     --save_dir ./checkpoints > training.log 2>&1 &

# Option 2: Run in foreground (see output in real-time)
torchrun --nproc_per_node=8 --master-port=25419 train.py \
    --distributed \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-2-v1" \
    --hf_tokenizer "gpt2" \
    --hf_text_column "text" \
    --epochs 5 \
    --batch_size 16 \
    --d_model 768 \
    --n_layers 12 \
    --n_heads 12 \
    --max_seq_len 512 \
    --use_rope \
    --lr 1e-4 \
    --experiment_name "mdlm_wikitext2_8gpu" \
    --save_dir ./checkpoints

# Single GPU training (commented out - uncomment if needed for testing)
# python train.py \
#     --hf_raw_repo "Salesforce/wikitext" \
#     --hf_raw_split "wikitext-2-v1" \
#     --hf_tokenizer "gpt2" \
#     --hf_text_column "text" \
#     --epochs 5 \
#     --batch_size 64 \
#     --d_model 768 \
#     --n_layers 12 \
#     --n_heads 12 \
#     --max_seq_len 512 \
#     --use_rope \
#     --lr 1e-4 \
#     --experiment_name "mdlm_wikitext2" \
#     --save_dir ./checkpoints

