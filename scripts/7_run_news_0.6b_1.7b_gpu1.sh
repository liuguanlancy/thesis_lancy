#!/bin/bash
#
# Run News Articles 0.6B and 1.7B model experiments on GPU 1
# with 5-minute interval between experiments
#

set -e  # Exit on error

# Force GPU 1
export CUDA_VISIBLE_DEVICES=1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "News Articles 0.6B & 1.7B Experiments (GPU 1)"
echo "=========================================="
echo "Running on GPU: 1"
echo "Will run 2 experiments with different model sizes:"
echo "1. 0.6B model (batch_size=8, grad_accum=1)"
echo "2. 1.7B model (batch_size=4, grad_accum=2)"
echo ""
echo "5-minute interval between experiments"
echo "=========================================="
echo ""

# Record start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Starting experiments at: $START_TIME"
echo ""

# Experiment 1: 0.6B model
echo "[1/2] Starting News Articles 0.6B model experiment..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU: 1"
echo "----------------------------------------"
$SCRIPT_DIR/phase2b_rtx4090.sh --model 0.6b --experiments 7 \
    --batch-size 8 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 1

if [ $? -eq 0 ]; then
    echo "[1/2] News Articles 0.6B model experiment completed successfully!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
else
    echo "[1/2] ERROR: News Articles 0.6B model experiment failed!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
    exit 1
fi

# Wait 5 minutes before next experiment
echo ""
echo "=========================================="
echo "Waiting 5 minutes before next experiment..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Next experiment will start at: $(date -v +5M +"%Y-%m-%d %H:%M:%S")"
else
    echo "Next experiment will start at: $(date -d "+5 minutes" +"%Y-%m-%d %H:%M:%S")"
fi
echo "(Press Ctrl+C to cancel)"
echo "=========================================="
sleep 300  # 5 minutes = 300 seconds

# Experiment 2: 1.7B model
echo "[2/2] Starting News Articles 1.7B model experiment..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU: 1"
echo "----------------------------------------"
$SCRIPT_DIR/phase2b_rtx4090.sh --model 1.7b --experiments 7 \
    --batch-size 4 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 2

if [ $? -eq 0 ]; then
    echo "[2/2] News Articles 1.7B model experiment completed successfully!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
else
    echo "[2/2] ERROR: News Articles 1.7B model experiment failed!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
    exit 1
fi

# Final summary
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Start time:      $START_TIME"
echo "Completion time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU used: 1"
echo ""
echo "Summary:"
echo "  ✅ 0.6B model trained on News Articles"
echo "  ✅ 1.7B model trained on News Articles"
echo ""
echo "Check the following directories for results:"
echo "  - runs/*_qwen3_0.6b/news_articles/"
echo "  - runs/*_qwen3_1.7b/news_articles/"
echo "=========================================="