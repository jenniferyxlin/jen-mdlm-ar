#!/bin/bash
# Resume AR 1000 epochs training from epoch 729 with smaller batch size

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# Configuration
MASTER_PORT=25419
MASTER_ADDR=localhost
BATCH_SIZE=8  # Reduced from 16 to avoid OOM
D_MODEL=768
N_LAYERS=12
N_HEADS=12
MAX_SEQ_LEN=512
LR=1e-4
SAVE_DIR="./checkpoints"
EPOCHS=1000
MODEL_TYPE="ar"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Clean up port before training
cleanup_port() {
    local port=$1
    echo "Cleaning up port $port..."
    fuser -k "$port"/tcp 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Killed process(es) using port $port with fuser."
    else
        local pid=$(ss -tlnp 2>/dev/null | grep "$port" | awk '{print $6}' | cut -d',' -f2 | cut -d'=' -f2)
        if [ -z "$pid" ]; then
            pid=$(netstat -tlnp 2>/dev/null | grep "$port" | awk '{print $7}' | cut -d'/' -f1)
        fi
        if [ ! -z "$pid" ]; then
            echo "Found process $pid using port $port. Killing..."
            kill -9 "$pid" 2>/dev/null
        fi
    fi
    sleep 2
}

# Clean up port
cleanup_port $MASTER_PORT

# Clear GPU cache
echo "Clearing GPU cache..."
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || echo "Could not clear cache"

echo "=========================================="
echo "Resuming AR 1000 epochs training"
echo "Batch size: $BATCH_SIZE (reduced to avoid OOM)"
echo "Will resume from latest checkpoint (should be epoch 729)"
echo "=========================================="
echo ""

# Check if checkpoint exists
CHECKPOINT_PATTERN="${SAVE_DIR}/ar_wikitext2_1000epochs_checkpoint_epoch_*.pt"
CHECKPOINTS=($(ls -t $CHECKPOINT_PATTERN 2>/dev/null))
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "Warning: No checkpoint found matching pattern: $CHECKPOINT_PATTERN"
    echo "Will start from beginning if no checkpoint is found."
else
    LATEST_CHECKPOINT="${CHECKPOINTS[0]}"
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    echo ""
fi

# Run training (will automatically resume from checkpoint)
torchrun --nproc_per_node=8 --master-port=$MASTER_PORT train.py \
    --distributed \
    --model_type $MODEL_TYPE \
    --hf_raw_repo "Salesforce/wikitext" \
    --hf_raw_split "wikitext-2-v1" \
    --hf_tokenizer "gpt2" \
    --hf_text_column "text" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --max_seq_len $MAX_SEQ_LEN \
    --use_rope \
    --lr $LR \
    --experiment_name "${MODEL_TYPE}_wikitext2_${EPOCHS}epochs" \
    --save_dir $SAVE_DIR

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ AR (1000 epochs) completed successfully"
else
    echo ""
    echo "✗ AR (1000 epochs) failed with exit code $exit_code"
fi

exit $exit_code

