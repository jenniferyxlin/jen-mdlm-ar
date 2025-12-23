#!/bin/bash
# Quick script to monitor training progress

echo "=== Training Process Check ==="
ps aux | grep -E "(torchrun|train.py)" | grep -v grep

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -8

echo ""
echo "=== Recent Training Output (last 20 lines) ==="
if [ -f training.log ]; then
    tail -n 20 training.log
elif [ -f ~/jen-mdlm-ar/training.log ]; then
    tail -n 20 ~/jen-mdlm-ar/training.log
else
    echo "No training.log found. If running in foreground, output should be in terminal."
fi

echo ""
echo "=== Checkpoint Directory ==="
if [ -d ./checkpoints ]; then
    ls -lh ./checkpoints/*.pt 2>/dev/null | tail -5
    echo "Checkpoint count: $(ls ./checkpoints/*.pt 2>/dev/null | wc -l)"
elif [ -d ~/jen-mdlm-ar/checkpoints ]; then
    ls -lh ~/jen-mdlm-ar/checkpoints/*.pt 2>/dev/null | tail -5
    echo "Checkpoint count: $(ls ~/jen-mdlm-ar/checkpoints/*.pt 2>/dev/null | wc -l)"
else
    echo "No checkpoints directory found yet"
fi


