#!/bin/bash

echo "============================================================"
echo "LoRA CONFIGURATION CHECK"
echo "============================================================"
echo ""

# Function to extract LoRA info
check_lora() {
    local script="$1"
    local exp="$2"
    local desc="$3"
    
    echo "$desc - Experiment $exp:"
    
    # Run the script in dry run mode and capture output
    if [[ "$script" == *"phase2b_financial_pretraining.sh" ]]; then
        output=$(cd scripts && bash phase2b_financial_pretraining.sh --experiments $exp --dry-run 2>&1)
    else
        output=$(bash $script --experiments $exp --dry-run 2>&1)
    fi
    
    # Check if LoRA is mentioned in the command
    if echo "$output" | grep -q "use_lora"; then
        echo "  ✓ LoRA: ENABLED"
        
        # Extract rank and alpha
        rank=$(echo "$output" | grep -o "lora_r [0-9]*" | grep -o "[0-9]*" | head -1)
        alpha=$(echo "$output" | grep -o "lora_alpha [0-9]*" | grep -o "[0-9]*" | head -1)
        
        if [ -z "$rank" ]; then
            # Try alternative format
            rank=$(echo "$output" | grep "Rank:" | grep -o "[0-9]*" | head -1)
            alpha=$(echo "$output" | grep "Alpha:" | grep -o "[0-9]*" | head -1)
        fi
        
        if [ -n "$rank" ] && [ -n "$alpha" ]; then
            echo "  - Rank: $rank"
            echo "  - Alpha: $alpha (should be 2x rank = $((rank * 2)))"
            
            # Calculate parameter reduction
            # Qwen3-0.6B has ~600M params, LoRA reduces trainable params significantly
            # Rough estimate: rank 32 = ~2% of params trainable
            trainable_pct=$((rank * 100 / 1600))  # Rough approximation
            echo "  - Approx trainable params: ~${trainable_pct}% of model"
        else
            echo "  ⚠ Could not extract rank/alpha values"
        fi
    else
        echo "  ✗ LoRA: DISABLED (full fine-tuning)"
    fi
    
    # Show the actual command
    cmd=$(echo "$output" | grep "python.*train.py\|python.*src/main.py" | head -1)
    if echo "$cmd" | grep -q "use_lora"; then
        lora_params=$(echo "$cmd" | grep -o "\-\-use_lora\|\-\-lora_r [0-9]*\|\-\-lora_alpha [0-9]*" | tr '\n' ' ')
        echo "  Command: ...$lora_params"
    fi
    
    echo ""
}

echo "================================"
echo "MAIN SCRIPT DEFAULTS"
echo "================================"
echo ""

# Check the defaults in the main script
echo "Default configuration from phase2b_financial_pretraining.sh:"
grep "DEFAULT_USE_LORA\|DEFAULT_LORA_RANK\|DEFAULT_LORA_ALPHA" scripts/phase2b_financial_pretraining.sh | head -3
echo ""

echo "================================"
echo "M1 MAX EXPERIMENTS"
echo "================================"
echo ""

for exp in 1 2 mixed; do
    check_lora "scripts/phase2b_m1max.sh" "$exp" "M1 Max"
done

echo "================================"
echo "RTX 4090 EXPERIMENTS"
echo "================================"
echo ""

for exp in 1 2 mixed; do
    check_lora "scripts/phase2b_rtx4090.sh" "$exp" "RTX 4090"
done

echo "================================"
echo "DIRECT SCRIPT TEST"
echo "================================"
echo ""

check_lora "scripts/phase2b_financial_pretraining.sh" "1" "Direct"

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Expected LoRA Configuration:"
echo "- Status: ENABLED by default"
echo "- Rank: 32"
echo "- Alpha: 64 (2x rank)"
echo "- Target modules: q_proj k_proj v_proj o_proj"
echo ""
echo "Benefits:"
echo "- Memory usage: ~10% of full fine-tuning"
echo "- Trainable params: ~2% of model"
echo "- Training speed: 2-3x faster"
echo "============================================================"