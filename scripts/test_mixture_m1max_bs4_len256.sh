#!/bin/bash
#
# Test script for mixture training on M1 Max
# Configuration: BS=4, SeqLen=256, Frequent evaluation
# Purpose: Quick test of dataset mixture with aggressive evaluation
#

set -e  # Exit on error

echo "========================================="
echo "Testing Dataset Mixture on M1 Max"
echo "========================================="
echo "Configuration:"
echo "  - Batch Size: 4"
echo "  - Sequence Length: 256"
echo "  - Warmup Steps: 10 (10% of max steps)"
echo "  - Evaluation: At start + every 10 steps"
echo "  - Save: Every 50 steps (will save at step 50)"
echo "  - Max Steps: 100 (quick test)"
echo "  - Strategy: 50cap (50% News cap)"
echo "========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the mixture experiment with specified settings
# Note: eval-on-start is enabled by default in the wrapper
"$SCRIPT_DIR/phase2b_m1max.sh" \
    --experiments mixed \
    --strategy 50cap \
    --batch-size 4 \
    --max-length 256 \
    --eval-steps 10 \
    --eval-batches 10 \
    --save-steps 50 \
    --max-steps 100 \
    --warmup-steps 10  # 10% of 100 steps

echo ""
echo "========================================="
echo "Test Complete"
echo "========================================="
echo "Token consumption:"
echo "  - Tokens per step: 4 × 256 = 1,024"
echo "  - Total steps: 100"
echo "  - Total tokens: ~102,400 (0.1% of 100M budget)"
echo ""
echo "This was a quick test. For full training, remove --max-steps 100"