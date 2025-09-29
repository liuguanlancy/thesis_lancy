#!/bin/bash
# Example commands for BookCorpus pretraining with packing and FlashAttention

echo "BookCorpus Pretraining Examples"
echo "================================"

# 1. Basic BookCorpus pretraining (no optimizations)
echo ""
echo "1. Basic pretraining (baseline):"
echo "python train.py --model Qwen/Qwen3-1.7B --dataset bookcorpus/bookcorpus --mode pretrain --max_steps 10000 --batch_size 8 --max_length 512"

# 2. With sequence packing (3-5x speedup for short sequences)
echo ""
echo "2. With sequence packing (recommended for BookCorpus):"
echo "python train.py --model Qwen/Qwen3-1.7B --dataset bookcorpus/bookcorpus --mode pretrain --max_steps 10000 --batch_size 8 --max_length 2048 --use_packing"

# 3. With packing and larger context (better efficiency)
echo ""
echo "3. With packing and 4K context:"
echo "python train.py --model Qwen/Qwen3-1.7B --dataset bookcorpus/bookcorpus --mode pretrain --max_steps 10000 --batch_size 4 --max_length 4096 --use_packing --packing_max_length 4096"

# 4. With FlashAttention 2 (CUDA only)
echo ""
echo "4. With FlashAttention 2 (requires CUDA and flash-attn package):"
echo "python train.py --model Qwen/Qwen3-1.7B --dataset bookcorpus/bookcorpus --mode pretrain --max_steps 10000 --batch_size 8 --max_length 512 --use_flash_attention"

# 5. With both packing and FlashAttention (maximum performance)
echo ""
echo "5. With packing + FlashAttention (best performance on CUDA):"
echo "python train.py --model Qwen/Qwen3-1.7B --dataset bookcorpus/bookcorpus --mode pretrain --max_steps 10000 --batch_size 8 --max_length 2048 --use_packing --use_flash_attention"

# 6. With LoRA for memory efficiency
echo ""
echo "6. With packing + LoRA (memory efficient):"
echo "python train.py --model Qwen/Qwen3-1.7B --dataset bookcorpus/bookcorpus --mode pretrain --max_steps 10000 --batch_size 16 --max_length 2048 --use_packing --use_lora --lora_r 8 --lora_alpha 16"

# 7. Full optimized command for H100
echo ""
echo "7. Optimized for H100 GPU:"
echo "python train.py --model Qwen/Qwen3-1.7B --dataset bookcorpus/bookcorpus --mode pretrain --max_steps 50000 --batch_size 32 --gradient_accumulation_steps 2 --max_length 4096 --use_packing --use_flash_attention --use_lora --lora_r 16 --lora_alpha 32"

echo ""
echo "Performance Notes:"
echo "=================="
echo "- Packing provides 3-5x speedup on BookCorpus (short sentences)"
echo "- FlashAttention 2 adds 20-30% speedup on top of packing"
echo "- Larger context (2048-4096) improves packing efficiency"
echo "- LoRA reduces memory by 90% with minimal performance loss"
echo ""
echo "Expected speedups vs baseline:"
echo "- Packing only: 3-5x faster"
echo "- Packing + FlashAttention: 4-6x faster"
echo "- Packing + FlashAttention + optimal batch size: 5-8x faster"