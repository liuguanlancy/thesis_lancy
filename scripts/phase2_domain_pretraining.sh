#!/bin/bash
# Phase 2 Domain-Specific Pretraining Experiments
# Focused on financial domain and analytical capabilities
# Skips general pretraining (Phase 2A) as base model already has this
# Configuration
MODEL="Qwen/Qwen3-0.6B-Base"  # Using 0.6B model for faster experimentation
PRETRAIN_LORA_RANK=32         # Higher rank for pretraining (more expressiveness needed)
PRETRAIN_LORA_ALPHA=64        # Standard 2x rank ratio
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj"  # Only target attention projections (space-separated)
BATCH_SIZE=256                 # Large batch for 40GB+ GPU
MAX_LENGTH=1024                # 1024 tokens for full context
BASE_DIR="./runs/phase2_domain_pretraining"
PYTHON_CMD="/Users/mengzhao/miniconda3/envs/lancy/bin/python"  # Using conda environment
# Learning rate configuration
LEARNING_RATE=2e-5            # Lower LR for LoRA fine-tuning (was 5e-5 default)
LR_SCHEDULER="cosine"         # Cosine annealing for better convergence
WARMUP_STEPS=5000            # 5k warmup steps (5% of 100k typical training)
WEIGHT_DECAY=0.01            # Weight decay for regularization
# Additional training stability parameters
EVAL_MAX_BATCHES=100          # Increased for better evaluation statistics
EVAL_STEPS=1000               # Evaluate every 1k steps
SAVE_STEPS=5000               # Save checkpoints every 5k steps
# Multi-dataset evaluation settings (for mixture experiments)
SEPARATE_EVAL="--separate_mixture_eval"  # Evaluate each dataset in mixture separately
LOG_SPREAD="--log_eval_spread"          # Log spread metrics (min/max/std) for balance monitoring
echo "=========================================="
echo "PHASE 2: DOMAIN-SPECIFIC PRETRAINING"
echo "=========================================="
echo "Model: $MODEL"
echo "LoRA: rank=$PRETRAIN_LORA_RANK, alpha=$PRETRAIN_LORA_ALPHA"
echo "LoRA targets: $LORA_TARGET_MODULES"
echo "Batch size: $BATCH_SIZE (optimized for 40GB+ GPU)"
echo "Max length: $MAX_LENGTH tokens"
echo "Learning rate: $LEARNING_RATE with $LR_SCHEDULER scheduler"
echo "Warmup: $WARMUP_STEPS steps, Weight decay: $WEIGHT_DECAY"
echo "Eval: every $EVAL_STEPS steps, $EVAL_MAX_BATCHES batches"
echo "Save: every $SAVE_STEPS steps"
echo "Multi-dataset eval: ENABLED for mixtures (tracks individual dataset performance)"
echo "Output directory: $BASE_DIR"
echo "Python: $PYTHON_CMD"
echo "=========================================="
# Create base directory
mkdir -p $BASE_DIR
# ===========================================
# PHASE 2B: FINANCIAL DOMAIN PRETRAINING (6 experiments)
# Focus: Adapt model to financial language and concepts
# ===========================================
echo ""
echo "PHASE 2B: FINANCIAL DOMAIN PRETRAINING"
echo "======================================="
echo "Focus: Financial texts, Q&A, and sentiment data"
echo "Purpose: Adapt model to financial terminology and patterns"
echo ""
# --- Financial Q&A pretraining ---
echo "Experiment 1/13: Financial Q&A pretrain (100k steps)"
echo "  Dataset: 10-K filing Q&A pairs"
$PYTHON_CMD train.py --model $MODEL --dataset virattt/financial-qa-10K --mode pretrain \
    --max_steps 100000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 25 \
    --save_strategy steps \
    --output_dir $BASE_DIR/financial_qa
# --- FinGPT sentiment pretraining ---
echo ""
echo "Experiment 2/13: FinGPT sentiment pretrain (100k steps)"
echo "  Dataset: Financial sentiment analysis (76K examples)"
$PYTHON_CMD train.py --model $MODEL --dataset FinGPT/fingpt-sentiment-train --mode pretrain \
    --max_steps 100000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 25 \
    --save_strategy steps \
     \
    --output_dir $BASE_DIR/fingpt
# --- Finance Alpaca pretraining ---
echo ""
echo "Experiment 3/13: Finance Alpaca pretrain (100k steps)"
echo "  Dataset: Financial instruction following (68K examples)"
$PYTHON_CMD train.py --model $MODEL --dataset gbharti/finance-alpaca --mode pretrain \
    --max_steps 100000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 25 \
    --save_strategy steps \
     \
    --output_dir $BASE_DIR/alpaca
# --- FiQA pretraining ---
echo ""
echo "Experiment 4/13: FiQA pretrain (50k steps)"
echo "  Dataset: Financial Q&A from various sources"
$PYTHON_CMD train.py --model $MODEL --dataset LLukas22/fiqa --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 15 \
    --save_strategy steps \
     \
    --output_dir $BASE_DIR/fiqa
# --- Twitter financial sentiment pretraining ---
echo ""
echo "Experiment 5/13: Twitter financial sentiment pretrain (50k steps)"
echo "  Dataset: Real-world financial social media sentiment"
$PYTHON_CMD train.py --model $MODEL --dataset zeroshot/twitter-financial-news-sentiment --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 15 \
    --save_strategy steps \
     \
    --output_dir $BASE_DIR/twitter
# --- Mixed financial corpus pretraining (FINAL) ---
echo ""
echo "Experiment 6/13: Mixed financial corpus (150k steps)"
echo "  Combines all 5 financial datasets with proportional sampling"
echo "  Mixture rates based on dataset sizes (total ~169K examples):"
echo "    - Financial Q&A: 7K examples (4%)"
echo "    - FinGPT: 76K examples (45%)"
echo "    - Finance Alpaca: 68K examples (40%)"
echo "    - FiQA: 6K examples (4%)"
echo "    - Twitter: 12K examples (7%)"
echo "  Multi-dataset evaluation: ENABLED (tracks each dataset separately)"
$PYTHON_CMD train.py --model $MODEL \
    --datasets virattt/financial-qa-10K FinGPT/fingpt-sentiment-train gbharti/finance-alpaca LLukas22/fiqa zeroshot/twitter-financial-news-sentiment \
    --dataset_configs None None None None None --mixture_rates 0.04 0.45 0.40 0.04 0.07 --mode pretrain \
    --max_steps 150000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 35 \
    --save_strategy steps \
     \
    $SEPARATE_EVAL $LOG_SPREAD \
    --output_dir $BASE_DIR/mixed_financial
# ===========================================
# PHASE 2C: ANALYTICAL CAPABILITIES PRETRAINING (7 experiments)
# Focus: Math, code, and reasoning for financial analysis
# ===========================================
echo ""
echo "PHASE 2C: ANALYTICAL CAPABILITIES PRETRAINING"
echo "=============================================="
echo "Focus: Mathematical reasoning, code, and logical analysis"
echo "Purpose: Enhance analytical skills needed for financial modeling"
echo ""
# --- GSM8K math pretraining ---
echo "Experiment 7/13: GSM8K math pretrain (50k steps)"
echo "  Dataset: Grade school math word problems"
$PYTHON_CMD train.py --model $MODEL --dataset openai/gsm8k --dataset_config main --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS  \
    --output_dir $BASE_DIR/gsm8k
# --- DeepMind math pretraining ---
echo ""
echo "Experiment 8/13: DeepMind math pretrain (50k steps)"
echo "  Dataset: Diverse mathematical problems"
$PYTHON_CMD train.py --model $MODEL --dataset deepmind/math_dataset --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS  \
    --output_dir $BASE_DIR/deepmind
# --- BigCodeBench pretraining ---
echo ""
echo "Experiment 9/13: BigCodeBench pretrain (50k steps)"
echo "  Dataset: Code generation with function calls"
$PYTHON_CMD train.py --model $MODEL --dataset bigcode/bigcodebench --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS  \
    --output_dir $BASE_DIR/bigcode
# --- GLUE MNLI pretraining ---
echo ""
echo "Experiment 10/13: GLUE MNLI pretrain (50k steps)"
echo "  Dataset: Natural language inference"
$PYTHON_CMD train.py --model $MODEL --dataset glue --dataset_config mnli --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS  \
    --output_dir $BASE_DIR/glue_mnli
# --- MMLU-Pro pretraining ---
echo ""
echo "Experiment 11/13: MMLU-Pro pretrain (50k steps)"
echo "  Dataset: Advanced multi-choice reasoning"
$PYTHON_CMD train.py --model $MODEL --dataset TIGER-Lab/MMLU-Pro --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS  \
    --output_dir $BASE_DIR/mmlu_pro
# --- Math+Code mixture pretraining ---
echo ""
echo "Experiment 12/13: Math+Code mixture pretrain (75k steps)"
echo "  Dataset: Combined mathematical and coding problems"
echo "  Mixture: 60% GSM8K math, 40% BigCodeBench"
echo "  Multi-dataset evaluation: ENABLED (tracks math vs code performance)"
$PYTHON_CMD train.py --model $MODEL --datasets openai/gsm8k bigcode/bigcodebench \
    --dataset_configs main None --mixture_rates 0.6 0.4 --mode pretrain \
    --max_steps 75000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS  \
    $SEPARATE_EVAL $LOG_SPREAD \
    --output_dir $BASE_DIR/math_code
# --- Mixed analytical corpus ---
echo ""
echo "Experiment 13/13: Mixed analytical corpus (100k steps)"
echo "  Dataset: Math, code, and reasoning combined"
echo "  Mixture: 30% GSM8K, 30% DeepMind Math, 20% BigCode, 20% MMLU-Pro"
echo "  Multi-dataset evaluation: ENABLED (tracks performance across all 4 domains)"
$PYTHON_CMD train.py --model $MODEL \
    --datasets openai/gsm8k deepmind/math_dataset bigcode/bigcodebench TIGER-Lab/MMLU-Pro \
    --dataset_configs main None None None --mixture_rates 0.3 0.3 0.2 0.2 --mode pretrain \
    --max_steps 100000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 25 \
    --save_strategy steps \
     \
    $SEPARATE_EVAL $LOG_SPREAD \
    --output_dir $BASE_DIR/mixed_analytical
echo ""
echo "=========================================="
echo "PHASE 2 DOMAIN PRETRAINING COMPLETE!"
echo "=========================================="
echo "Total experiments: 13"
echo "Results saved in: $BASE_DIR"
echo ""
echo "Experiment breakdown:"
echo "- Phase 2B: 6 experiments (Financial domain)"
echo "  - 3 major datasets at 100k steps"
echo "  - 1 mixed corpus at 150k steps"
echo "  - 2 smaller datasets at 50k steps"
echo "- Phase 2C: 7 experiments (Analytical capabilities)"
echo "  - 5 individual datasets at 50k steps"
echo "  - 1 math+code mixture at 75k steps"
echo "  - 1 comprehensive mixture at 100k steps"
echo ""
echo "Key configurations:"
echo "- Model: Qwen3-0.6B-Base (600M parameters)"
echo "- LoRA rank: 16 (alpha 32) for pretraining"
echo "- Target modules: attention projections only"
echo "- Batch size: 256 (uniform)"
echo "- Max length: 1024 tokens (uniform)"
echo "- Learning rate: 2e-5 with cosine scheduler"
echo "- Warmup: 5k steps, Weight decay: 0.01"
echo "- Evaluation: every 1k steps, 100 batches"
echo "- Saves: every 5k steps"
echo ""
echo "Why no Phase 2A (WikiText/BookCorpus)?"
echo "- Base model already pretrained on these datasets"
echo "- Compute better spent on domain-specific data"
echo "- Focus on financial and analytical capabilities"
echo ""
echo "To analyze results:"
echo "1. Training logs: $BASE_DIR/*/logs/"
echo "2. TensorBoard: tensorboard --logdir $BASE_DIR"
echo "3. Checkpoints: $BASE_DIR/*/checkpoints/"
echo ""
echo "Multi-dataset evaluation metrics (for mixture experiments):"
echo "- Individual dataset performance: eval/{dataset_name}/loss and /perplexity"
echo "- Average metrics: eval/average/loss and /perplexity"
echo "- Spread metrics: eval/spread/max_min_diff, /std_dev, /relative_spread"
echo "- These help identify if any dataset is being neglected or if negative transfer occurs"
echo "=========================================="