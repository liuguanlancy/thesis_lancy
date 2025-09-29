#!/bin/bash

# Test script to verify all fixes using dryrun mode
# This quickly checks configuration without actual training

echo "================================================"
echo "Testing All Fixes with Dryrun Mode"
echo "================================================"
echo ""

# Test 1: Direct script with warmup calculation
echo "TEST 1: Direct script test with warmup calculation"
echo "Expected: warmup=10 (10% of 100 steps)"
echo "----------------------------------------"
cd scripts
bash phase2b_financial_pretraining.sh \
    mixed_financial \
    --max-steps 100 \
    --batch-size 4 \
    --max-length 128 \
    --warmup-steps 50 \
    --dry-run
cd ..

echo ""
echo "TEST 2: M1 Max wrapper with dynamic warmup"
echo "Expected: warmup=10 (calculated from 100 steps)"
echo "----------------------------------------"
bash scripts/phase2b_m1max.sh \
    --experiments mixed \
    --max-steps 100 \
    --batch-size 4 \
    --dry-run

echo ""
echo "TEST 3: Larger run to verify warmup calculation"
echo "Expected: warmup=100 (10% of 1000 steps)"
echo "----------------------------------------"
cd scripts
bash phase2b_financial_pretraining.sh \
    mixed_financial \
    --max-steps 1000 \
    --batch-size 4 \
    --warmup-steps 400 \
    --dry-run
cd ..

echo ""
echo "================================================"
echo "Dryrun Tests Complete"
echo "Check above for:"
echo "1. Warmup steps correctly calculated"
echo "2. Configuration properly displayed"
echo "3. No warmup >= max_steps warnings"
echo "================================================"