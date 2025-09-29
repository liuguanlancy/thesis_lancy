#!/bin/bash
#
# Run News Articles 4B model experiment on GPU 0
#

set -e  # Exit on error

# Force GPU 0
export CUDA_VISIBLE_DEVICES=0

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "News Articles 4B Model Experiment (GPU 0)"
echo "=========================================="
echo "Running on GPU: 0"
echo "Model: Qwen3-4B"
echo "Dataset: News Articles"
echo "=========================================="
echo ""

# Record start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Starting experiment at: $START_TIME"
echo ""

# Run 4B model experiment
echo "[News Articles] Starting 4B model experiment..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU: 0"
echo "----------------------------------------"
$SCRIPT_DIR/phase2b_rtx4090.sh --model 4b --experiments 7 \
    --batch-size 2 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 4

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ News Articles 4B model experiment completed successfully!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
else
    echo ""
    echo "❌ ERROR: News Articles 4B model experiment failed!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
    exit 1
fi

# Final summary
echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Start time:      $START_TIME"
echo "Completion time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU used: 0"
echo ""
echo "Summary:"
echo "  ✅ 4B model trained on News Articles"
echo ""
echo "Check the following directory for results:"
echo "  - runs/*_qwen3_4b/news_articles/"
echo "=========================================="