#!/bin/bash
# Phase 2 Pretraining Experiments - Standalone Script
# Extracted from run_qwen_experiments.sh
# 28 total experiments across different pretraining strategies

# Configuration
MODEL="Qwen/Qwen3-0.6B-Base"  # Using 0.6B model for faster experimentation
PRETRAIN_LORA_RANK=16         # Higher rank for pretraining (more expressiveness needed)
PRETRAIN_LORA_ALPHA=32        # Standard 2x rank ratio
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj"  # Only target attention projections (space-separated)
BATCH_SIZE=256                 # Large batch for 40GB+ GPU
MAX_LENGTH=1024                # 1024 tokens for full context
BASE_DIR="./runs/phase2_pretraining"
PYTHON_CMD="/Users/mengzhao/miniconda3/envs/lancy/bin/python"  # Using conda environment

# Additional training stability parameters
WARMUP_STEPS=5000              # 5k warmup steps for all experiments
MAX_GRAD_NORM=1.0             # Gradient clipping to prevent NaN
EVAL_MAX_BATCHES=100          # Increased for better evaluation statistics
EVAL_STEPS=1000               # Evaluate every 1k steps
SAVE_STEPS=5000               # Save checkpoints every 5k steps

echo "=========================================="
echo "PHASE 2: PRETRAINING EXPERIMENTS"
echo "=========================================="
echo "Model: $MODEL"
echo "LoRA: rank=$PRETRAIN_LORA_RANK, alpha=$PRETRAIN_LORA_ALPHA"
echo "LoRA targets: $LORA_TARGET_MODULES"
echo "Batch size: $BATCH_SIZE (optimized for 40GB+ GPU)"
echo "Max length: $MAX_LENGTH tokens"
echo "Eval: every $EVAL_STEPS steps, $EVAL_MAX_BATCHES batches"
echo "Save: every $SAVE_STEPS steps"
echo "Warmup steps: $WARMUP_STEPS"
echo "Output directory: $BASE_DIR"
echo "Python: $PYTHON_CMD"
echo "=========================================="

# Create base directory
mkdir -p $BASE_DIR

# ===========================================
# PHASE 2A: GENERAL PRETRAINING (3 experiments, each saving checkpoints at 10k, 25k, 50k)
# Focus: WikiText, BookCorpus, and their mixtures
# Note: Each experiment saves checkpoints at multiple intervals for analysis
# ===========================================
echo ""
echo "PHASE 2A: GENERAL PRETRAINING"
echo "=============================="
echo "Focus: WikiText and BookCorpus"
echo "Note: Each run saves checkpoints every 5k steps, evaluates every 1k steps"

# --- WikiText only pretraining (saves at 10k, 25k, 50k) ---
echo ""
echo "WikiText Pretraining"
echo "--------------------"

echo "Experiment 1/3: WikiText pretraining (up to 200k steps)"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL --dataset wikitext --dataset_config wikitext-103-raw-v1 --mode pretrain \
    --max_steps 200000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --eval_on_start \
    --save_steps $SAVE_STEPS --save_total_limit 50 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/wikitext

# --- BookCorpus only pretraining (saves at 10k, 25k, 50k) ---
echo ""
echo "BookCorpus Pretraining"
echo "----------------------"

echo "Experiment 2/3: BookCorpus pretraining (up to 200k steps)"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL --dataset bookcorpus/bookcorpus --mode pretrain \
    --max_steps 200000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --eval_on_start \
    --save_steps $SAVE_STEPS --save_total_limit 50 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/bookcorpus

# --- WikiText + BookCorpus mixture (saves at 10k, 25k, 50k) ---
echo ""
echo "WikiText + BookCorpus Mixture"
echo "------------------------------"

echo "Experiment 3/3: WikiText+BookCorpus mixture (up to 200k steps)"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL --datasets wikitext bookcorpus/bookcorpus \
    --dataset_configs wikitext-103-raw-v1 None --mixture_rates 0.5 0.5 --mode pretrain \
    --max_steps 200000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --eval_on_start \
    --save_steps $SAVE_STEPS --save_total_limit 50 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/mixture

# ===========================================
# PHASE 2B: DOMAIN CONTINUED PRETRAINING (6 experiments, optimized from 12)
# Focus: Financial domain-specific pretraining
# Each experiment saves checkpoints at multiple intervals for analysis
# ===========================================
echo ""
echo "PHASE 2B: DOMAIN CONTINUED PRETRAINING"
echo "======================================="
echo "Focus: Financial domain texts"
echo "Note: Each run saves checkpoints every 5k steps, evaluates every 1k steps"

# --- Financial Q&A as pretraining corpus (saves at 5k, 10k, 20k) ---
echo ""
echo "Financial Q&A Pretraining"
echo "-------------------------"

echo "Experiment 4/16: Financial Q&A pretrain (up to 80k steps)"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL --dataset virattt/financial-qa-10K --mode pretrain \
    --max_steps 80000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 20 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/financial_qa

# --- FinGPT sentiment as pretraining corpus (saves at 5k, 10k, 20k) ---
echo ""
echo "FinGPT Sentiment Pretraining"
echo "-----------------------------"

echo "Experiment 5/16: FinGPT sentiment pretrain (up to 80k steps)"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL --dataset FinGPT/fingpt-sentiment-train --mode pretrain \
    --max_steps 80000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 20 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/fingpt

# --- Finance Alpaca as pretraining corpus (saves at 5k, 10k, 20k) ---
echo ""
echo "Finance Alpaca Pretraining"
echo "---------------------------"

echo "Experiment 6/16: Finance Alpaca pretrain (up to 80k steps)"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL --dataset gbharti/finance-alpaca --mode pretrain \
    --max_steps 80000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 20 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/alpaca

# --- Mixed financial corpus (saves at 5k, 10k, 20k) ---
echo ""
echo "Mixed Financial Corpus Pretraining"
echo "-----------------------------------"

echo "Experiment 7/16: Mixed financial corpus (up to 80k steps)"
echo "  Includes: Financial Q&A, FinGPT, Alpaca, FiQA, Twitter sentiment"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL \
    --datasets virattt/financial-qa-10K FinGPT/fingpt-sentiment-train gbharti/finance-alpaca LLukas22/fiqa zeroshot/twitter-financial-news-sentiment \
    --dataset_configs None None None None None --mixture_rates 0.25 0.20 0.20 0.20 0.15 --mode pretrain \
    --max_steps 80000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 20 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/mixed_financial

# --- FiQA as pretraining corpus (saves at 5k, 10k) ---
echo ""
echo "FiQA Pretraining"
echo "----------------"

echo "Experiment 8/16: FiQA pretrain (up to 40k steps)"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL --dataset LLukas22/fiqa --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 10 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/fiqa

# --- Twitter financial sentiment as pretraining (saves at 5k, 10k) ---
echo ""
echo "Twitter Financial Sentiment Pretraining"
echo "----------------------------------------"

echo "Experiment 9/16: Twitter financial sentiment pretrain (up to 40k steps)"
echo "  Checkpoints saved every 5k steps, evaluation every 1k steps"
$PYTHON_CMD train.py --model $MODEL --dataset zeroshot/twitter-financial-news-sentiment --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 10 \
    --save_strategy steps \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/twitter

# ===========================================
# PHASE 2C: CROSS-DOMAIN PRETRAINING (7 experiments)
# Focus: Math, code, and reasoning tasks
# ===========================================
echo ""
echo "PHASE 2C: CROSS-DOMAIN PRETRAINING"
echo "==================================="
echo "Focus: Math, code, and reasoning"

echo "Experiment 10/16: GSM8K math pretrain 40k steps"
$PYTHON_CMD train.py --model $MODEL --dataset openai/gsm8k --dataset_config main --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/gsm8k

echo "Experiment 11/16: DeepMind math pretrain 40k steps"
$PYTHON_CMD train.py --model $MODEL --dataset deepmind/math_dataset --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/deepmind

echo "Experiment 12/16: BigCodeBench pretrain 40k steps"
$PYTHON_CMD train.py --model $MODEL --dataset bigcode/bigcodebench --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/bigcode

echo "Experiment 13/16: GLUE MNLI pretrain 40k steps"
$PYTHON_CMD train.py --model $MODEL --dataset glue --dataset_config mnli --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/glue_mnli

echo "Experiment 14/16: MMLU-Pro pretrain 40k steps"
$PYTHON_CMD train.py --model $MODEL --dataset TIGER-Lab/MMLU-Pro --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/mmlu_pro

echo "Experiment 15/16: Math+Code mixture pretrain 40k steps"
$PYTHON_CMD train.py --model $MODEL --datasets openai/gsm8k bigcode/bigcodebench \
    --dataset_configs main None --mixture_rates 0.6 0.4 --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/math_code

echo "Experiment 16/16: OpenWebText general pretrain 40k steps"
$PYTHON_CMD train.py --model $MODEL --dataset Skylion007/openwebtext --mode pretrain \
    --max_steps 40000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --warmup_steps $WARMUP_STEPS \
    --output_dir $BASE_DIR/openwebtext

echo ""
echo "=========================================="
echo "PHASE 2 EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Total experiments: 16 (optimized from 28)"
echo "Results saved in: $BASE_DIR"
echo ""
echo "Experiment breakdown:"
echo "- Phase 2A: 3 experiments (WikiText, BookCorpus, mixtures) - 200k steps each"
echo "- Phase 2B: 6 experiments (Financial domain) - 80k/40k steps"
echo "- Phase 2C: 7 experiments (Cross-domain: math, code, reasoning)"
echo ""
echo "Key configurations:"
echo "- Model: Qwen3-0.6B-Base (600M parameters)"
echo "- LoRA rank: 16 (alpha 32) for pretraining"
echo "- Target modules: attention projections only"
echo "- Batch size: 256 (uniform across all tasks)"
echo "- Max length: 1024 tokens (uniform)"
echo "- Evaluation: every 1k steps, 100 batches"
echo "- Saves: every 5k steps"
echo "- Warmup: 5k steps (uniform)"
echo "- Training steps: 4x increased (200k/80k/40k)"
echo ""
echo "To analyze results:"
echo "1. Training logs: $BASE_DIR/*/logs/"
echo "2. TensorBoard: tensorboard --logdir $BASE_DIR"
echo "3. Checkpoints: $BASE_DIR/*/checkpoints/"
echo "=========================================="