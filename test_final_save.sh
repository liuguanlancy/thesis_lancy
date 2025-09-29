#!/bin/bash

# Test script to verify final checkpoint save and evaluation functionality

echo "Testing final checkpoint save and evaluation..."
echo "================================================"

# Test with a very small training run
echo "Running minimal training with 10 steps..."
python train.py \
    --model gpt2 \
    --dataset glue \
    --dataset_config sst2 \
    --mode sft \
    --max_steps 10 \
    --eval_steps 5 \
    --save_steps 5 \
    --batch_size 2 \
    --max_length 64 \
    --output_dir runs/test_final_save

# Check if final checkpoint was created
echo ""
echo "Checking for final checkpoint..."
if [ -d "runs/test_final_save/checkpoint-final" ]; then
    echo "✓ Final checkpoint directory found"
    
    # Check for model files
    if [ -f "runs/test_final_save/checkpoint-final/config.json" ] && \
       [ -f "runs/test_final_save/checkpoint-final/model.safetensors" ]; then
        echo "✓ Model files saved successfully"
    else
        echo "✗ Model files missing"
    fi
else
    echo "✗ Final checkpoint directory not found"
fi

# Check if final metrics were saved
echo ""
echo "Checking for final metrics..."
if [ -f "runs/test_final_save/final_metrics.json" ]; then
    echo "✓ Final metrics file found"
    echo "Final metrics content:"
    cat runs/test_final_save/final_metrics.json
else
    echo "✗ Final metrics file not found"
fi

echo ""
echo "Test completed!"