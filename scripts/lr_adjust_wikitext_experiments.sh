#!/bin/bash
#
# Learning Rate Adjusted WikiText experiments
# Only runs 1.7B and 4B models with reduced learning rates
# to address catastrophic failure and instability issues
#
# LR Adjustments:
# - 1.7B: 5e-6 (75% reduction - more conservative due to infinity perplexity)
# - 4B: 3e-6 (85% reduction - very conservative)
#
# Note: Original 1.7B showed infinity perplexity, indicating severe instability
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "WikiText LR-Adjusted Experiments"
echo "=========================================="
echo "Will run 2 experiments with adjusted learning rates:"
echo "1. 1.7B model (LR=5e-6, batch_size=4, grad_accum=2)"
echo "2. 4B model (LR=3e-6, batch_size=2, grad_accum=4)"
echo ""
echo "CRITICAL: 1.7B had infinity perplexity in original run!"
echo "Using very conservative learning rates"
echo "5-minute interval between experiments"
echo "=========================================="
echo ""

# Experiment 1: 1.7B model with heavily reduced LR (priority fix)
echo "[1/2] Starting 1.7B model experiment (LR=5e-6)..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "WARNING: Original run showed infinity perplexity"
echo "----------------------------------------"
# Pass the learning rate to phase2b_financial_pretraining.sh via additional args
$SCRIPT_DIR/phase2b_financial_pretraining.sh --experiments 8 \
    --model Qwen/Qwen3-1.7B-Base --model-short qwen3_1.7b \
    --batch-size 4 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 2 \
    --max-steps 12207 --max-length 1024 \
    --learning-rate 5e-6 --precision bf16 --attn-implementation flash_attention_2 \
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

# Experiment 2: 4B model with very conservative LR
echo "[2/2] Starting 4B model experiment (LR=3e-6)..."
echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "----------------------------------------"
$SCRIPT_DIR/phase2b_financial_pretraining.sh --experiments 8 \
    --model Qwen/Qwen3-4B-Base --model-short qwen3_4b \
    --batch-size 2 --eval-steps 1000 --eval-batches 100 \
    --save-steps 1000 --save-total-limit 2 --gradient-accum 4 \
    --max-steps 12207 --max-length 1024 \
    --learning-rate 3e-6 --precision bf16 --attn-implementation flash_attention_2 \
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
echo "  ✓ 1.7B model trained with LR=5e-6 (75% reduction)"
echo "  ✓ 4B model trained with LR=3e-6 (85% reduction)"
echo ""
echo "Expected improvements:"
echo "  - 1.7B: Fix infinity perplexity, achieve <3.5 loss"
echo "  - 4B: Improve from 3.447 to <3.2 loss"
echo "  - Both: Stable training without gradient explosions"
echo ""
echo "Domain mismatch note:"
echo "  WikiText is general text, evaluating on financial datasets"
echo "  This explains poor transfer performance"
echo ""
echo "Check the following directories for results:"
echo "  - runs/*_qwen3_1.7b/wikitext/"
echo "  - runs/*_qwen3_4b/wikitext/"
echo "=========================================="