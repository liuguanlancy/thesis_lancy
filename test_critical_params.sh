#!/bin/bash

# Test critical parameters: packing, 100M token budget, warmup steps

echo "============================================================"
echo "CRITICAL PARAMETERS VERIFICATION"
echo "============================================================"
echo ""

# Function to analyze experiment output
analyze_experiment() {
    local output="$1"
    local exp="$2"
    local device="$3"
    
    echo "Experiment $exp ($device):"
    
    # 1. Check Packing
    if echo "$output" | grep -q "use-packing"; then
        echo "  ✓ PACKING: Enabled (--use-packing found)"
    elif echo "$output" | grep -q "no-packing"; then
        echo "  ✗ PACKING: Disabled (--no-packing found)"
    else
        echo "  ⚠ PACKING: Unknown (no packing flag found)"
    fi
    
    # 2. Check Token Budget (should be ~100M)
    max_steps=$(echo "$output" | grep -o "Max Steps: [0-9]*" | grep -o "[0-9]*" | tail -1)
    batch_size=$(echo "$output" | grep -o "Batch Size: [0-9]*" | grep -o "[0-9]*" | tail -1)
    max_length=$(echo "$output" | grep -o "Max Length: [0-9]*" | grep -o "[0-9]*" | tail -1)
    
    if [ -n "$max_steps" ] && [ -n "$batch_size" ] && [ -n "$max_length" ]; then
        token_budget=$((max_steps * batch_size * max_length))
        token_budget_m=$((token_budget / 1000000))
        
        # Check if close to 100M (allow 95M-105M range)
        if [ "$token_budget_m" -ge 95 ] && [ "$token_budget_m" -le 105 ]; then
            echo "  ✓ TOKEN BUDGET: ${token_budget_m}M tokens (${max_steps} steps × ${batch_size} batch × ${max_length} length)"
        else
            echo "  ✗ TOKEN BUDGET: ${token_budget_m}M tokens (expected ~100M)"
            echo "    Formula: ${max_steps} steps × ${batch_size} batch × ${max_length} length = ${token_budget} tokens"
        fi
    else
        echo "  ⚠ TOKEN BUDGET: Could not calculate"
    fi
    
    # 3. Check Warmup Steps
    warmup=$(echo "$output" | grep -o "warmup-steps [0-9]*" | grep -o "[0-9]*" | tail -1)
    if [ -z "$warmup" ]; then
        warmup=$(echo "$output" | grep -o "Warmup Steps: [0-9]*" | grep -o "[0-9]*" | tail -1)
    fi
    
    if [ -n "$warmup" ] && [ -n "$max_steps" ]; then
        expected_warmup=$((max_steps / 10))
        if [ "$expected_warmup" -lt 1 ]; then
            expected_warmup=1
        fi
        
        if [ "$warmup" -eq "$expected_warmup" ]; then
            echo "  ✓ WARMUP: $warmup steps (correct: 10% of $max_steps)"
        else
            echo "  ✗ WARMUP: $warmup steps (expected: $expected_warmup for $max_steps steps)"
        fi
    else
        echo "  ⚠ WARMUP: Could not verify"
    fi
    
    # Show the actual command that would run
    cmd=$(echo "$output" | grep "Dry run - would execute:" -A1 | tail -1)
    if [ -n "$cmd" ]; then
        echo "  Command preview: ...$(echo "$cmd" | grep -o "\-\-use-packing\|\-\-no-packing\|\-\-warmup-steps [0-9]*\|\-\-max-steps [0-9]*" | tr '\n' ' ')"
    fi
    
    echo ""
}

echo "================================"
echo "M1 MAX EXPERIMENTS (100M Budget)"
echo "================================"
echo ""

for exp in 1 2 3 4 5 6 7 mixed; do
    output=$(bash scripts/phase2b_m1max.sh --experiments $exp --dry-run 2>&1)
    analyze_experiment "$output" "$exp" "M1 Max"
done

echo "================================"
echo "RTX 4090 EXPERIMENTS (100M Budget)"
echo "================================"
echo ""

for exp in 1 2 3 4 5 6 7 mixed; do
    output=$(bash scripts/phase2b_rtx4090.sh --experiments $exp --dry-run 2>&1)
    analyze_experiment "$output" "$exp" "RTX 4090"
done

echo "================================"
echo "TESTING CUSTOM CONFIGURATIONS"
echo "================================"
echo ""

# Test with --no-packing flag
echo "M1 Max Exp 1 with --no-packing:"
output=$(bash scripts/phase2b_m1max.sh --experiments 1 --no-packing --dry-run 2>&1)
analyze_experiment "$output" "1 (no-packing)" "M1 Max"

# Test with custom max-steps
echo "RTX 4090 Exp 2 with --max-steps 1000:"
output=$(bash scripts/phase2b_rtx4090.sh --experiments 2 --max-steps 1000 --dry-run 2>&1)
analyze_experiment "$output" "2 (1000 steps)" "RTX 4090"

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Expected for all default experiments:"
echo "1. PACKING: Should be ENABLED (--use-packing)"
echo "2. TOKEN BUDGET: Should be ~100M tokens"
echo "3. WARMUP: Should be 10% of max_steps"
echo "============================================================"