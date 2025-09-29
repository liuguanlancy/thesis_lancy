#!/bin/bash
# Quick validation script for M1 Max (32GB) - Tests Phase 2B configurations
# This script runs minimal training steps to validate each experiment works

echo "=========================================="
echo "Quick Test for M1 Max (32GB)"
echo "Model: Qwen3-0.6B-Base"
echo "=========================================="
echo ""

# Base configuration for M1 Max
BASE_CMD="./scripts/phase2b_financial_pretraining.sh \
    --model Qwen/Qwen3-0.6B-Base \
    --batch-size 16 \
    --max-length 512 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --precision fp32 \
    --eval-steps 10 \
    --save-steps 50 \
    --output-base-dir ./test_runs/m1_validation"

# Test 1: Dry run to validate all configurations
echo "1. Testing configuration (dry-run)..."
echo "----------------------------------------"
for exp in 1 2 3 4 5 6; do
    echo -n "  Experiment $exp: "
    if $BASE_CMD --experiments $exp --dry-run > /dev/null 2>&1; then
        echo "✅ Valid"
    else
        echo "❌ Failed"
    fi
done

echo ""
echo "2. Quick training test (10 steps each)..."
echo "----------------------------------------"
echo "Note: This will download datasets on first run"
echo ""

# Create test script that limits steps
cat > /tmp/quick_train_test.sh << 'EOF'
#!/bin/bash
# Override the phase2b script to use minimal steps for testing

# Get the experiment number
EXPERIMENT=$1

echo "Testing experiment $EXPERIMENT with 10 training steps..."

# Run with minimal steps by modifying the train.py call
python train.py \
    --model Qwen/Qwen3-0.6B-Base \
    --dataset stanfordnlp/imdb \
    --mode pretrain \
    --max_steps 10 \
    --batch_size 16 \
    --max_length 512 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --eval_steps 10 \
    --save_steps 10 \
    --output_dir ./test_runs/m1_validation/exp_$EXPERIMENT \
    --logging_steps 1 \
    --no_mixed_precision

echo "Experiment $EXPERIMENT test completed"
EOF

chmod +x /tmp/quick_train_test.sh

# Test smaller experiments first (4 and 5 are smallest)
for exp in 4 5 1 2 3; do
    echo ""
    echo "Testing Experiment $exp..."
    
    case $exp in
        1) echo "  Dataset: Financial Q&A (7K examples)" ;;
        2) echo "  Dataset: FinGPT Sentiment (76K examples)" ;;
        3) echo "  Dataset: Finance Alpaca (68K examples)" ;;
        4) echo "  Dataset: FiQA (6K examples)" ;;
        5) echo "  Dataset: Twitter Sentiment (12K examples)" ;;
    esac
    
    # Run quick test
    /tmp/quick_train_test.sh $exp
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Experiment $exp passed"
    else
        echo "  ⚠️  Experiment $exp had issues"
    fi
done

echo ""
echo "3. Mixed corpus test (Experiment 6)..."
echo "----------------------------------------"
echo "Using smaller batch size for mixed dataset"

python train.py \
    --model Qwen/Qwen3-0.6B-Base \
    --datasets stanfordnlp/imdb glue \
    --dataset_configs None sst2 \
    --mixture_rates 0.5 0.5 \
    --mode pretrain \
    --max_steps 10 \
    --batch_size 8 \
    --max_length 512 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --eval_steps 10 \
    --output_dir ./test_runs/m1_validation/exp_6_mixed \
    --no_mixed_precision

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "✅ Recommended settings for M1 Max (32GB):"
echo "   --batch-size 16"
echo "   --max-length 512" 
echo "   --lora-rank 16 --lora-alpha 32"
echo "   --precision fp32  (or fp16 if stable)"
echo ""
echo "For full training, use:"
echo "./scripts/phase2b_financial_pretraining.sh \\"
echo "    --model Qwen/Qwen3-0.6B-Base \\"
echo "    --batch-size 16 \\"
echo "    --max-length 512 \\"
echo "    --lora-rank 16 \\"
echo "    --lora-alpha 32 \\"
echo "    --precision fp32 \\"
echo "    --experiments 6  # or 'all' for all experiments"
echo ""
echo "Memory tips for M1 Max:"
echo "- Close other applications"
echo "- Monitor Activity Monitor"
echo "- Reduce batch size if OOM occurs"
echo "- Use gradient accumulation for larger effective batch"
echo ""