#!/bin/bash
#
# Run Mixed-Wiki experiments (7 financial + WikiText) on GPU 1
# for 3 different model sizes with 5-minute intervals
#

set -e  # Exit on error

# Force GPU 1
export CUDA_VISIBLE_DEVICES=1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Mixed-Wiki Datasets Experiments (GPU 1)"
echo "=========================================="
echo "Running on GPU: 1"
echo "Dataset configuration: Mixed-Wiki (7 financial + WikiText)"
echo "Will run 3 experiments with different model sizes:"
echo "1. 0.6B model (batch_size=8, grad_accum=1)"
echo "2. 4B model (batch_size=2, grad_accum=4)"
echo "3. 1.7B model (batch_size=4, grad_accum=2)"
echo ""
echo "5-minute interval between experiments"
echo "=========================================="
echo ""

# Experiment 1: Mixed-Wiki 0.6B model
echo "[1/3] Starting Mixed-Wiki 0.6B model experiment..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU: 1"
echo "----------------------------------------"
$SCRIPT_DIR/phase2b_rtx4090.sh --model 0.6b --experiments 10 \
    --batch-size 8 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 1

if [ $? -eq 0 ]; then
    echo "[1/3] Mixed-Wiki 0.6B model experiment completed successfully!"
else
    echo "[1/3] ERROR: Mixed-Wiki 0.6B model experiment failed!"
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

# Experiment 2: Mixed-Wiki 4B model
echo "[2/3] Starting Mixed-Wiki 4B model experiment..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU: 1"
echo "----------------------------------------"
$SCRIPT_DIR/phase2b_rtx4090.sh --model 4b --experiments 10 \
    --batch-size 2 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 4

if [ $? -eq 0 ]; then
    echo "[2/3] Mixed-Wiki 4B model experiment completed successfully!"
else
    echo "[2/3] ERROR: Mixed-Wiki 4B model experiment failed!"
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

# Experiment 3: Mixed-Wiki 1.7B model
echo "[3/3] Starting Mixed-Wiki 1.7B model experiment..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU: 1"
echo "----------------------------------------"
$SCRIPT_DIR/phase2b_rtx4090.sh --model 1.7b --experiments 10 \
    --batch-size 4 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 2

if [ $? -eq 0 ]; then
    echo "[3/3] Mixed-Wiki 1.7B model experiment completed successfully!"
else
    echo "[3/3] ERROR: Mixed-Wiki 1.7B model experiment failed!"
    exit 1
fi

# Final summary
echo ""
echo "=========================================="
echo "ALL MIXED-WIKI EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Completion time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "GPU used: 1"
echo ""
echo "Summary:"
echo "  ✓ 0.6B model trained on Mixed-Wiki datasets"
echo "  ✓ 4B model trained on Mixed-Wiki datasets"
echo "  ✓ 1.7B model trained on Mixed-Wiki datasets"
echo ""
echo "Check the following directories for results:"
echo "  - runs/*_qwen3_0.6b/mixed_wiki_financial/"
echo "  - runs/*_qwen3_4b/mixed_wiki_financial/"
echo "  - runs/*_qwen3_1.7b/mixed_wiki_financial/"
echo "=========================================="