#!/bin/bash
#
# Learning Rate Adjusted Twitter Financial experiments
# Only runs 1.7B and 4B models with reduced learning rates
# to address overfitting issues observed in original experiments
#
# LR Adjustments:
# - 1.7B: 1e-5 (50% reduction from original 2e-5)
# - 4B: 5e-6 (75% reduction from original 2e-5)
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Twitter Financial LR-Adjusted Experiments"
echo "=========================================="
echo "Will run 2 experiments with adjusted learning rates:"
echo "1. 1.7B model (LR=1e-5, batch_size=4, grad_accum=2)"
echo "2. 4B model (LR=5e-6, batch_size=2, grad_accum=4)"
echo ""
echo "Original LR was 2e-5 for all models"
echo "5-minute interval between experiments"
echo "=========================================="
echo ""

# Experiment 1: 1.7B model with reduced LR
echo "[1/2] Starting 1.7B model experiment (LR=1e-5)..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "----------------------------------------"
# Pass the learning rate to phase2b_financial_pretraining.sh via additional args
$SCRIPT_DIR/phase2b_financial_pretraining.sh --experiments 5 \
    --model Qwen/Qwen3-1.7B-Base --model-short qwen3_1.7b \
    --batch-size 4 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 2 \
    --max-steps 12207 --max-length 1024 \
    --learning-rate 1e-5 --precision bf16 --attn-implementation flash_attention_2 \
    --use-packing --eval-all-datasets

if [ $? -eq 0 ]; then
    echo "[1/2] 1.7B model experiment completed successfully!"
else
    echo "[1/2] ERROR: 1.7B model experiment failed!"
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

# Experiment 2: 4B model with heavily reduced LR
echo "[2/2] Starting 4B model experiment (LR=5e-6)..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "----------------------------------------"
$SCRIPT_DIR/phase2b_financial_pretraining.sh --experiments 5 \
    --model Qwen/Qwen3-4B-Base --model-short qwen3_4b \
    --batch-size 2 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 4 \
    --max-steps 12207 --max-length 1024 \
    --learning-rate 5e-6 --precision bf16 --attn-implementation flash_attention_2 \
    --use-packing --eval-all-datasets

if [ $? -eq 0 ]; then
    echo "[2/2] 4B model experiment completed successfully!"
else
    echo "[2/2] ERROR: 4B model experiment failed!"
    exit 1
fi

# Final summary
echo ""
echo "=========================================="
echo "LR-ADJUSTED EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Completion time: $(date +"%Y-%m-%d %H:%M:%S")"
echo ""
echo "Summary:"
echo "  ✓ 1.7B model trained with LR=1e-5 (50% reduction)"
echo "  ✓ 4B model trained with LR=5e-6 (75% reduction)"
echo ""
echo "Expected improvements:"
echo "  - 1.7B: Should maintain good performance (~2.5 loss)"
echo "  - 4B: Should improve from 2.891 to <2.5 loss"
echo ""
echo "Check the following directories for results:"
echo "  - runs/*_qwen3_1.7b/twitter/"
echo "  - runs/*_qwen3_4b/twitter/"
echo "=========================================="