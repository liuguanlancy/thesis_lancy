#!/bin/bash

# Comprehensive dry run test for all experiments in both M1 Max and RTX 4090 scripts

echo "============================================================"
echo "DRY RUN TEST - ALL EXPERIMENTS"
echo "============================================================"
echo ""

# Function to extract key info from dry run output
extract_info() {
    local output="$1"
    local exp="$2"
    local device="$3"
    
    echo "  Experiment $exp ($device):"
    
    # Extract warmup steps
    warmup=$(echo "$output" | grep -o "warmup-steps [0-9]*" | grep -o "[0-9]*" | tail -1)
    max_steps=$(echo "$output" | grep -o "max-steps [0-9]*" | grep -o "[0-9]*" | tail -1)
    
    if [ -z "$warmup" ]; then
        warmup=$(echo "$output" | grep -o "Warmup Steps: [0-9]*" | grep -o "[0-9]*" | tail -1)
    fi
    if [ -z "$max_steps" ]; then
        max_steps=$(echo "$output" | grep -o "Max Steps: [0-9]*" | grep -o "[0-9]*" | tail -1)
    fi
    
    if [ -n "$warmup" ] && [ -n "$max_steps" ]; then
        expected_warmup=$((max_steps / 10))
        if [ "$expected_warmup" -lt 1 ]; then
            expected_warmup=1
        fi
        
        if [ "$warmup" -eq "$expected_warmup" ]; then
            echo "    ✓ Warmup: $warmup (correct: 10% of $max_steps)"
        else
            echo "    ✗ Warmup: $warmup (expected: $expected_warmup for $max_steps steps)"
        fi
    else
        echo "    ⚠ Could not extract warmup/max_steps"
    fi
    
    # Check for errors
    if echo "$output" | grep -q "Error:"; then
        echo "    ✗ ERROR found in output"
    fi
    
    if echo "$output" | grep -q "WARNING.*[Ww]armup.*>=.*max"; then
        echo "    ✗ WARNING: Warmup >= max steps issue"
    fi
    
    # Check if command would execute
    if echo "$output" | grep -q "Dry run - would execute:"; then
        echo "    ✓ Command generated successfully"
    else
        echo "    ✗ No command generated"
    fi
    
    echo ""
}

echo "================================"
echo "M1 MAX EXPERIMENTS"
echo "================================"
echo ""

# Test all M1 Max experiments
for exp in 1 2 3 4 5 6 7 mixed; do
    echo "Testing M1 Max Experiment $exp..."
    output=$(bash scripts/phase2b_m1max.sh --experiments $exp --dry-run 2>&1)
    extract_info "$output" "$exp" "M1 Max"
done

echo "================================"
echo "RTX 4090 EXPERIMENTS"
echo "================================"
echo ""

# Test all RTX 4090 experiments
for exp in 1 2 3 4 5 6 7 mixed; do
    echo "Testing RTX 4090 Experiment $exp..."
    output=$(bash scripts/phase2b_rtx4090.sh --experiments $exp --dry-run 2>&1)
    extract_info "$output" "$exp" "RTX 4090"
done

echo "================================"
echo "TESTING WITH CUSTOM PARAMETERS"
echo "================================"
echo ""

# Test with custom max-steps to verify dynamic warmup
echo "M1 Max with max-steps=1000:"
output=$(bash scripts/phase2b_m1max.sh --experiments 1 --max-steps 1000 --dry-run 2>&1)
extract_info "$output" "1 (1000 steps)" "M1 Max"

echo "RTX 4090 with max-steps=5000:"
output=$(bash scripts/phase2b_rtx4090.sh --experiments 2 --max-steps 5000 --dry-run 2>&1)
extract_info "$output" "2 (5000 steps)" "RTX 4090"

echo "============================================================"
echo "DRY RUN TEST COMPLETE"
echo "============================================================"
echo ""
echo "Summary:"
echo "- All experiments should show warmup = 10% of max_steps"
echo "- No errors or warnings should appear"
echo "- All should generate valid commands"
echo "============================================================"