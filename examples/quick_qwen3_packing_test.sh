#!/bin/bash
# Quick test of Qwen3 + FlashAttention + Packing
# Uses WikiText-2 (small dataset) for fast experimentation

echo "======================================================"
echo "QUICK TEST: Qwen3 + FlashAttention + Packing"
echo "======================================================"
echo ""

# 1. Baseline: Standard training (no optimizations)
echo "1. BASELINE TEST (Standard Training)"
echo "-------------------------------------"
echo "Command:"
echo "python train.py \\"
echo "  --model Qwen/Qwen3-0.6B \\"
echo "  --dataset wikitext \\"
echo "  --dataset_config wikitext-2-raw-v1 \\"
echo "  --mode pretrain \\"
echo "  --max_steps 10 \\"
echo "  --batch_size 4 \\"
echo "  --max_length 512 \\"
echo "  --output_dir ./runs/qwen3_baseline"
echo ""
echo "Expected time: ~30-60 seconds"
echo ""

# Run baseline
python train.py \
  --model Qwen/Qwen3-0.6B \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --mode pretrain \
  --max_steps 10 \
  --batch_size 4 \
  --max_length 512 \
  --output_dir ./runs/qwen3_baseline

echo ""
echo "======================================================"
echo ""

# 2. With Packing only
echo "2. WITH PACKING (3-5x faster on short sequences)"
echo "-------------------------------------------------"
echo "Command:"
echo "python train.py \\"
echo "  --model Qwen/Qwen3-0.6B \\"
echo "  --dataset wikitext \\"
echo "  --dataset_config wikitext-2-raw-v1 \\"
echo "  --mode pretrain \\"
echo "  --max_steps 10 \\"
echo "  --batch_size 4 \\"
echo "  --max_length 2048 \\"
echo "  --use_packing \\"
echo "  --output_dir ./runs/qwen3_packing"
echo ""
echo "Expected time: ~15-30 seconds"
echo ""

# Run with packing
python train.py \
  --model Qwen/Qwen3-0.6B \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --mode pretrain \
  --max_steps 10 \
  --batch_size 4 \
  --max_length 2048 \
  --use_packing \
  --output_dir ./runs/qwen3_packing

echo ""
echo "======================================================"
echo ""

# 3. With FlashAttention only (CUDA only)
echo "3. WITH FLASHATTENTION (2-3x faster attention)"
echo "-----------------------------------------------"
echo "Command:"
echo "python train.py \\"
echo "  --model Qwen/Qwen3-0.6B \\"
echo "  --dataset wikitext \\"
echo "  --dataset_config wikitext-2-raw-v1 \\"
echo "  --mode pretrain \\"
echo "  --max_steps 10 \\"
echo "  --batch_size 4 \\"
echo "  --max_length 512 \\"
echo "  --use_flash_attention \\"
echo "  --output_dir ./runs/qwen3_flash"
echo ""
echo "Note: Requires CUDA GPU and flash-attn package"
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, running with FlashAttention..."
    python train.py \
      --model Qwen/Qwen3-0.6B \
      --dataset wikitext \
      --dataset_config wikitext-2-raw-v1 \
      --mode pretrain \
      --max_steps 10 \
      --batch_size 4 \
      --max_length 512 \
      --use_flash_attention \
      --output_dir ./runs/qwen3_flash
else
    echo "No CUDA detected, skipping FlashAttention test"
fi

echo ""
echo "======================================================"
echo ""

# 4. Full optimization: Packing + FlashAttention + LoRA
echo "4. FULLY OPTIMIZED (Packing + FlashAttention + LoRA)"
echo "-----------------------------------------------------"
echo "Command:"
echo "python train.py \\"
echo "  --model Qwen/Qwen3-0.6B \\"
echo "  --dataset wikitext \\"
echo "  --dataset_config wikitext-2-raw-v1 \\"
echo "  --mode pretrain \\"
echo "  --max_steps 10 \\"
echo "  --batch_size 8 \\"
echo "  --max_length 2048 \\"
echo "  --use_packing \\"
echo "  --use_flash_attention \\"
echo "  --use_lora --lora_r 8 --lora_alpha 16 \\"
echo "  --output_dir ./runs/qwen3_optimized"
echo ""
echo "Expected time: ~10-20 seconds"
echo ""

# Run fully optimized
python train.py \
  --model Qwen/Qwen3-0.6B \
  --dataset wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --mode pretrain \
  --max_steps 10 \
  --batch_size 8 \
  --max_length 2048 \
  --use_packing \
  --use_flash_attention \
  --use_lora --lora_r 8 --lora_alpha 16 \
  --output_dir ./runs/qwen3_optimized

echo ""
echo "======================================================"
echo "TEST COMPLETE!"
echo "======================================================"
echo ""
echo "Check the results in:"
echo "  ./runs/qwen3_baseline/   - Standard training"
echo "  ./runs/qwen3_packing/    - With packing"
echo "  ./runs/qwen3_flash/      - With FlashAttention"
echo "  ./runs/qwen3_optimized/  - Fully optimized"
echo ""
echo "Compare training times in the logs to see speedup!"