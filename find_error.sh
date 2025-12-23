#!/bin/bash
# Find the actual error message in training.log

cd ~/jen-mdlm-ar 2>/dev/null || cd /root/jen-mdlm-ar 2>/dev/null || pwd

if [ -f training.log ]; then
    echo "=== Looking for error messages ==="
    # Find lines with ERROR, Exception, or the actual error text
    grep -i -B 10 "error\|exception\|failed\|timeout\|12355\|29500" training.log | tail -50
    
    echo ""
    echo "=== Checking for DEBUG output (to see which port was used) ==="
    grep -i "DEBUG\|MASTER_PORT" training.log | tail -20
    
    echo ""
    echo "=== Last 30 lines ==="
    tail -30 training.log
else
    echo "training.log not found"
    echo "If running in foreground, errors should be in terminal output"
fi


