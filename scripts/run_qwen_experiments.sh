#!/bin/bash
# Complete Qwen3-0.6B-Base Experimental Plan
# Updated with new evaluation options and optimized parameters
# Total: 72 experiments across 3 phases
# Estimated GPU hours: ~18-20 on RTX 4090 with optimized LoRA

# Configuration
MODEL="Qwen/Qwen3-0.6B-Base"  # Using 0.6B model for faster experimentation
PRETRAIN_LORA_RANK=16         # Higher rank for pretraining (more expressiveness needed)
PRETRAIN_LORA_ALPHA=32        # Standard 2x rank ratio
FINETUNE_LORA_RANK=4          # Lower rank for fine-tuning (task-specific adaptation)
FINETUNE_LORA_ALPHA=8         # Standard 2x rank ratio
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj"  # Only target attention projections (space-separated)
BATCH_SIZE=16                  # Larger batch size possible with smaller model
BASE_DIR="./runs/qwen_experiments"
PYTHON_CMD="/Users/mengzhao/miniconda3/envs/lancy/bin/python"  # Using conda environment

# Additional training stability parameters
WARMUP_STEPS=100               # Learning rate warmup for stability
MAX_GRAD_NORM=1.0             # Gradient clipping to prevent NaN

echo "=========================================="
echo "Starting Qwen3-0.6B-Base Experimental Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo "Pretrain LoRA: rank=$PRETRAIN_LORA_RANK, alpha=$PRETRAIN_LORA_ALPHA"
echo "Finetune LoRA: rank=$FINETUNE_LORA_RANK, alpha=$FINETUNE_LORA_ALPHA"
echo "LoRA targets: $LORA_TARGET_MODULES"
echo "Batch size: $BATCH_SIZE"
echo "Output directory: $BASE_DIR"
echo "Python: $PYTHON_CMD"
echo "=========================================="

# Create base directory
mkdir -p $BASE_DIR

# ===========================================
# PHASE 1: BASELINE ESTABLISHMENT (10 experiments)
# ===========================================
echo ""
echo "PHASE 1: BASELINE ESTABLISHMENT"
echo "================================"
echo "Establishing zero-shot baselines with limited evaluation"

# 1. Financial Sentiment (3-class)
echo "Experiment 1/72: Financial Sentiment Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset takala/financial_phrasebank --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 20 \
    --output_dir $BASE_DIR/phase1_baseline_financial_sentiment

# 2. Tweet Sentiment (3-class)
echo "Experiment 2/72: Tweet Sentiment Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset zeroshot/twitter-financial-news-sentiment --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 20 \
    --output_dir $BASE_DIR/phase1_baseline_tweet_sentiment

# 3. Topic Classification (20-class)
echo "Experiment 3/72: Topic Classification Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset zeroshot/twitter-financial-news-topic --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 20 \
    --output_dir $BASE_DIR/phase1_baseline_topic_classification

# 4. Financial Q&A
echo "Experiment 4/72: Financial Q&A Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset virattt/financial-qa-10K --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 10 --max_length 512 \
    --output_dir $BASE_DIR/phase1_baseline_financial_qa

# 5. Conversational Q&A
echo "Experiment 5/72: Conversational Q&A Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset AdaptLLM/finance-tasks --dataset_config ConvFinQA --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 10 \
    --output_dir $BASE_DIR/phase1_baseline_conv_qa

# 6. General Sentiment (IMDB)
echo "Experiment 6/72: IMDB Sentiment Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset stanfordnlp/imdb --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 20 \
    --output_dir $BASE_DIR/phase1_baseline_imdb

# 7. Math Reasoning
echo "Experiment 7/72: Math Reasoning Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset openai/gsm8k --dataset_config main --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 10 --max_length 512 \
    --output_dir $BASE_DIR/phase1_baseline_math

# 8. General Understanding (GLUE SST-2)
echo "Experiment 8/72: GLUE SST-2 Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset glue --dataset_config sst2 --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 20 \
    --output_dir $BASE_DIR/phase1_baseline_glue_sst2

# 9. Code Generation
echo "Experiment 9/72: Code Generation Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset bigcode/bigcodebench --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 5 --max_length 1024 \
    --output_dir $BASE_DIR/phase1_baseline_code

# 10. Named Entity Recognition
echo "Experiment 10/72: NER Baseline"
$PYTHON_CMD train.py --model $MODEL --dataset AdaptLLM/finance-tasks --dataset_config NER --mode sft \
    --max_steps 0 --evaluation_only --eval_max_batches 20 \
    --output_dir $BASE_DIR/phase1_baseline_ner

# ===========================================
# PHASE 2: PRETRAINING EXPERIMENTS (9 experiments - BookCorpus & WikiText only)
# ===========================================
echo ""
echo "PHASE 2: PRETRAINING EXPERIMENTS"
echo "================================="
echo "Focus: BookCorpus and WikiText only"

# --- WikiText only pretraining (3 scales) ---
echo ""
echo "Phase 2A: WikiText Pretraining"
echo "-------------------------------"

# 11-13: WikiText only (3 scales)
echo "Experiment 11/72: WikiText 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset wikitext --dataset_config wikitext-103-raw-v1 --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --eval_on_start \
    --save_steps 2000 --output_dir $BASE_DIR/phase2_wikitext_10k

echo "Experiment 12/72: WikiText 25k steps"
$PYTHON_CMD train.py --model $MODEL --dataset wikitext --dataset_config wikitext-103-raw-v1 --mode pretrain \
    --max_steps 25000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 2500 --eval_max_batches 10 \
    --save_steps 5000 --output_dir $BASE_DIR/phase2_wikitext_25k

echo "Experiment 13/72: WikiText 50k steps"
$PYTHON_CMD train.py --model $MODEL --dataset wikitext --dataset_config wikitext-103-raw-v1 --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 5000 --eval_max_batches 10 \
    --save_steps 10000 --output_dir $BASE_DIR/phase2_wikitext_50k

# --- BookCorpus only pretraining (3 scales) ---
echo ""
echo "Phase 2B: BookCorpus Pretraining"
echo "---------------------------------"

# 14-16: BookCorpus only (3 scales)
echo "Experiment 14/72: BookCorpus 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset bookcorpus/bookcorpus --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2_bookcorpus_10k

echo "Experiment 15/72: BookCorpus 25k steps"
$PYTHON_CMD train.py --model $MODEL --dataset bookcorpus/bookcorpus --mode pretrain \
    --max_steps 25000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 2500 --eval_max_batches 10 \
    --save_steps 5000 --output_dir $BASE_DIR/phase2_bookcorpus_25k

echo "Experiment 16/72: BookCorpus 50k steps"
$PYTHON_CMD train.py --model $MODEL --dataset bookcorpus/bookcorpus --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 5000 --eval_max_batches 10 \
    --save_steps 10000 --output_dir $BASE_DIR/phase2_bookcorpus_50k

# --- WikiText + BookCorpus mixture (3 scales) ---
echo ""
echo "Phase 2C: WikiText + BookCorpus Mixture"
echo "----------------------------------------"

# 17-19: WikiText + BookCorpus mixture (3 scales)
echo "Experiment 17/72: WikiText+BookCorpus mixture 10k steps"
$PYTHON_CMD train.py --model $MODEL --datasets wikitext bookcorpus/bookcorpus \
    --dataset_configs wikitext-103-raw-v1 None --mixture_rates 0.5 0.5 --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2_mixture_10k

echo "Experiment 18/72: WikiText+BookCorpus mixture 25k steps"
$PYTHON_CMD train.py --model $MODEL --datasets wikitext bookcorpus/bookcorpus \
    --dataset_configs wikitext-103-raw-v1 None --mixture_rates 0.5 0.5 --mode pretrain \
    --max_steps 25000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 2500 --eval_max_batches 10 \
    --save_steps 5000 --output_dir $BASE_DIR/phase2_mixture_25k

echo "Experiment 19/72: WikiText+BookCorpus mixture 50k steps"
$PYTHON_CMD train.py --model $MODEL --datasets wikitext bookcorpus/bookcorpus \
    --dataset_configs wikitext-103-raw-v1 None --mixture_rates 0.5 0.5 --mode pretrain \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 5000 --eval_max_batches 10 \
    --save_steps 10000 --output_dir $BASE_DIR/phase2_mixture_50k

# --- 2B: Domain Continued Pretraining (12 experiments) ---
echo ""
echo "Phase 2B: Domain Continued Pretraining"
echo "---------------------------------------"

# 20-21: Financial Q&A as pretraining corpus
echo "Experiment 20/72: Financial Q&A pretrain 5k steps"
$PYTHON_CMD train.py --model $MODEL --dataset virattt/financial-qa-10K --mode pretrain \
    --max_steps 5000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 5 \
    --save_steps 1000 --max_length 512 --output_dir $BASE_DIR/phase2b_financial_qa_5k

echo "Experiment 21/72: Financial Q&A pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset virattt/financial-qa-10K --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 5 \
    --save_steps 2000 --max_length 512 --output_dir $BASE_DIR/phase2b_financial_qa_10k

# 22-23: FinGPT sentiment as pretraining corpus
echo "Experiment 22/72: FinGPT sentiment pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset FinGPT/fingpt-sentiment-train --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2b_fingpt_10k

echo "Experiment 23/72: FinGPT sentiment pretrain 20k steps"
$PYTHON_CMD train.py --model $MODEL --dataset FinGPT/fingpt-sentiment-train --mode pretrain \
    --max_steps 20000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 2000 --eval_max_batches 10 \
    --save_steps 4000 --output_dir $BASE_DIR/phase2b_fingpt_20k

# 24-25: Finance Alpaca as pretraining corpus
echo "Experiment 24/72: Finance Alpaca pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset gbharti/finance-alpaca --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2b_alpaca_10k

echo "Experiment 25/72: Finance Alpaca pretrain 20k steps"
$PYTHON_CMD train.py --model $MODEL --dataset gbharti/finance-alpaca --mode pretrain \
    --max_steps 20000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 2000 --eval_max_batches 10 \
    --save_steps 4000 --output_dir $BASE_DIR/phase2b_alpaca_20k

# 26-27: Mixed financial corpus
echo "Experiment 26/72: Mixed financial corpus 10k steps"
$PYTHON_CMD train.py --model $MODEL \
    --datasets virattt/financial-qa-10K FinGPT/fingpt-sentiment-train gbharti/finance-alpaca \
    --dataset_configs None None None --mixture_rates 0.4 0.3 0.3 --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2b_mixed_financial_10k

echo "Experiment 27/72: Mixed financial corpus 20k steps"
$PYTHON_CMD train.py --model $MODEL \
    --datasets virattt/financial-qa-10K FinGPT/fingpt-sentiment-train gbharti/finance-alpaca \
    --dataset_configs None None None --mixture_rates 0.4 0.3 0.3 --mode pretrain \
    --max_steps 20000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 2000 --eval_max_batches 10 \
    --save_steps 4000 --output_dir $BASE_DIR/phase2b_mixed_financial_20k

# 28-29: FiQA as pretraining corpus
echo "Experiment 28/72: FiQA pretrain 5k steps"
$PYTHON_CMD train.py --model $MODEL --dataset LLukas22/fiqa --mode pretrain \
    --max_steps 5000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --output_dir $BASE_DIR/phase2b_fiqa_5k

echo "Experiment 29/72: FiQA pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset LLukas22/fiqa --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2b_fiqa_10k

# 30-31: Twitter financial sentiment as pretraining
echo "Experiment 30/72: Twitter financial sentiment pretrain 5k steps"
$PYTHON_CMD train.py --model $MODEL --dataset zeroshot/twitter-financial-news-sentiment --mode pretrain \
    --max_steps 5000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --output_dir $BASE_DIR/phase2b_twitter_5k

echo "Experiment 31/72: Twitter financial sentiment pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset zeroshot/twitter-financial-news-sentiment --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2b_twitter_10k

# --- 2C: Cross-Domain Pretraining (6 experiments) ---
echo ""
echo "Phase 2C: Cross-Domain Pretraining"
echo "-----------------------------------"

# 32: Math pretraining (GSM8K)
echo "Experiment 32/72: GSM8K math pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset openai/gsm8k --dataset_config main --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 5 \
    --save_steps 2000 --max_length 512 --output_dir $BASE_DIR/phase2c_gsm8k_10k

# 33: Math pretraining (DeepMind)
echo "Experiment 33/72: DeepMind math pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset deepmind/math_dataset --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2c_deepmind_10k

# 34: Code pretraining
echo "Experiment 34/72: BigCodeBench pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset bigcode/bigcodebench --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 1000 --eval_max_batches 3 \
    --save_steps 2000 --max_length 1024 --output_dir $BASE_DIR/phase2c_bigcode_10k

# 35: GLUE pretraining
echo "Experiment 35/72: GLUE MNLI pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset glue --dataset_config mnli --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2c_glue_mnli_10k

# 36: MMLU-Pro pretraining
echo "Experiment 36/72: MMLU-Pro pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --dataset TIGER-Lab/MMLU-Pro --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 5 \
    --save_steps 2000 --output_dir $BASE_DIR/phase2c_mmlu_pro_10k

# 37: Mixed math+code pretraining
echo "Experiment 37/72: Math+Code mixture pretrain 10k steps"
$PYTHON_CMD train.py --model $MODEL --datasets openai/gsm8k bigcode/bigcodebench \
    --dataset_configs main None --mixture_rates 0.6 0.4 --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 1000 --eval_max_batches 5 \
    --save_steps 2000 --max_length 768 --output_dir $BASE_DIR/phase2c_math_code_10k

# ===========================================
# PHASE 3: FINE-TUNING EXPERIMENTS (35 experiments)
# ===========================================
echo ""
echo "PHASE 3: FINE-TUNING EXPERIMENTS"
echo "================================="

# --- 3A: Direct Fine-tuning with LoRA (10 experiments) ---
echo ""
echo "Phase 3A: Direct Fine-tuning"
echo "-----------------------------"

# 38: Financial Sentiment
echo "Experiment 38/72: Financial Sentiment fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset takala/financial_phrasebank --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 250 --eval_max_batches 20 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3a_financial_sentiment

# 39: Tweet Sentiment
echo "Experiment 39/72: Tweet Sentiment fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset zeroshot/twitter-financial-news-sentiment --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 250 --eval_max_batches 20 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3a_tweet_sentiment

# 40: Topic Classification
echo "Experiment 40/72: Topic Classification fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset zeroshot/twitter-financial-news-topic --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 250 --eval_max_batches 20 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3a_topic_classification

# 41: Financial Q&A
echo "Experiment 41/72: Financial Q&A fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset virattt/financial-qa-10K --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --max_length 512 --output_dir $BASE_DIR/phase3a_financial_qa

# 42: Math Reasoning
echo "Experiment 42/72: Math Reasoning fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset openai/gsm8k --dataset_config main --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --max_length 512 --output_dir $BASE_DIR/phase3a_math_reasoning

# 43: IMDB Sentiment
echo "Experiment 43/72: IMDB Sentiment fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset stanfordnlp/imdb --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 250 --eval_max_batches 20 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3a_imdb

# 44: GLUE SST-2
echo "Experiment 44/72: GLUE SST-2 fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset glue --dataset_config sst2 --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 250 --eval_max_batches 20 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3a_glue_sst2

# 45: GLUE CoLA
echo "Experiment 45/72: GLUE CoLA fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset glue --dataset_config cola --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 250 --eval_max_batches 20 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3a_glue_cola

# 46: GLUE MRPC
echo "Experiment 46/72: GLUE MRPC fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset glue --dataset_config mrpc --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 250 --eval_max_batches 20 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3a_glue_mrpc

# 47: FiQA Q&A
echo "Experiment 47/72: FiQA Q&A fine-tuning"
$PYTHON_CMD train.py --model $MODEL --dataset LLukas22/fiqa --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3a_fiqa

# --- 3B: Multi-task Fine-tuning (10 experiments) ---
echo ""
echo "Phase 3B: Multi-task Fine-tuning"
echo "---------------------------------"

# 48: All sentiment tasks
echo "Experiment 48/72: All sentiment tasks"
$PYTHON_CMD train.py --model $MODEL \
    --datasets takala/financial_phrasebank zeroshot/twitter-financial-news-sentiment FinGPT/fingpt-sentiment-train \
    --dataset_configs None None None --mixture_rates 0.3 0.3 0.4 --mode sft \
    --max_steps 10000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 15 \
    --save_steps 2000 --output_dir $BASE_DIR/phase3b_all_sentiment

# 49: All Q&A tasks
echo "Experiment 49/72: All Q&A tasks"
$PYTHON_CMD train.py --model $MODEL \
    --datasets virattt/financial-qa-10K LLukas22/fiqa AdaptLLM/finance-tasks \
    --dataset_configs None None ConvFinQA --mixture_rates 0.4 0.3 0.3 --mode sft \
    --max_steps 10000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 500 --eval_max_batches 10 \
    --save_steps 2000 --max_length 512 --output_dir $BASE_DIR/phase3b_all_qa

# 50: Mixed classification tasks
echo "Experiment 50/72: Mixed classification tasks"
$PYTHON_CMD train.py --model $MODEL \
    --datasets zeroshot/twitter-financial-news-topic stanfordnlp/imdb glue \
    --dataset_configs None None sst2 --mixture_rates 0.4 0.3 0.3 --mode sft \
    --max_steps 10000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 15 \
    --save_steps 2000 --output_dir $BASE_DIR/phase3b_mixed_classification

# 51: Financial + General sentiment
echo "Experiment 51/72: Financial + General sentiment"
$PYTHON_CMD train.py --model $MODEL \
    --datasets takala/financial_phrasebank stanfordnlp/imdb \
    --dataset_configs None None --mixture_rates 0.6 0.4 --mode sft \
    --max_steps 10000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 20 \
    --save_steps 2000 --output_dir $BASE_DIR/phase3b_financial_general_sentiment

# 52: Math + Financial Q&A
echo "Experiment 52/72: Math + Financial Q&A"
$PYTHON_CMD train.py --model $MODEL \
    --datasets openai/gsm8k virattt/financial-qa-10K \
    --dataset_configs main None --mixture_rates 0.5 0.5 --mode sft \
    --max_steps 10000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 500 --eval_max_batches 10 \
    --save_steps 2000 --max_length 512 --output_dir $BASE_DIR/phase3b_math_financial_qa

# 53: GLUE multi-task
echo "Experiment 53/72: GLUE multi-task"
$PYTHON_CMD train.py --model $MODEL \
    --datasets glue glue glue \
    --dataset_configs sst2 cola mrpc --mixture_rates 0.4 0.3 0.3 --mode sft \
    --max_steps 10000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 20 \
    --save_steps 2000 --output_dir $BASE_DIR/phase3b_glue_multitask

# 54: Financial complete (all financial datasets)
echo "Experiment 54/72: Financial complete"
$PYTHON_CMD train.py --model $MODEL \
    --datasets takala/financial_phrasebank zeroshot/twitter-financial-news-sentiment zeroshot/twitter-financial-news-topic virattt/financial-qa-10K \
    --dataset_configs None None None None --mixture_rates 0.25 0.25 0.25 0.25 --mode sft \
    --max_steps 20000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 1000 --eval_max_batches 15 \
    --save_steps 4000 --output_dir $BASE_DIR/phase3b_financial_complete

# 55: Cross-domain mixture
echo "Experiment 55/72: Cross-domain mixture"
$PYTHON_CMD train.py --model $MODEL \
    --datasets stanfordnlp/imdb openai/gsm8k takala/financial_phrasebank \
    --dataset_configs None main None --mixture_rates 0.3 0.3 0.4 --mode sft \
    --max_steps 15000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 750 --eval_max_batches 15 \
    --save_steps 3000 --output_dir $BASE_DIR/phase3b_cross_domain

# 56: Instruction following mixture
echo "Experiment 56/72: Instruction following mixture"
$PYTHON_CMD train.py --model $MODEL \
    --datasets gbharti/finance-alpaca FinGPT/fingpt-sentiment-train \
    --dataset_configs None None --mixture_rates 0.5 0.5 --mode sft \
    --max_steps 10000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 15 \
    --save_steps 2000 --output_dir $BASE_DIR/phase3b_instruction_following

# 57: Advanced reasoning mixture
echo "Experiment 57/72: Advanced reasoning mixture"
$PYTHON_CMD train.py --model $MODEL \
    --datasets TIGER-Lab/MMLU-Pro openai/gsm8k \
    --dataset_configs None main --mixture_rates 0.5 0.5 --mode sft \
    --max_steps 15000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 750 --eval_max_batches 10 \
    --save_steps 3000 --max_length 512 --output_dir $BASE_DIR/phase3b_advanced_reasoning

# --- 3C: Sequential Fine-tuning (15 experiments across 6 sequences) ---
echo ""
echo "Phase 3C: Sequential Fine-tuning"
echo "---------------------------------"

# Sequence 1: General → Specific sentiment (3 experiments: 58-60)
echo "Experiment 58/72: Sequential - IMDB baseline"
$PYTHON_CMD train.py --model $MODEL --dataset stanfordnlp/imdb --mode sft \
    --max_steps 2000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 200 --eval_max_batches 20 \
    --save_steps 500 --output_dir $BASE_DIR/phase3c_seq1_imdb

echo "Experiment 59/72: Sequential - Financial sentiment from IMDB"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq1_imdb --dataset takala/financial_phrasebank --mode sft \
    --max_steps 2000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 200 --eval_max_batches 20 \
    --save_steps 500 --output_dir $BASE_DIR/phase3c_seq1_financial

echo "Experiment 60/72: Sequential - Twitter sentiment from Financial"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq1_financial --dataset zeroshot/twitter-financial-news-sentiment --mode sft \
    --max_steps 2000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 200 --eval_max_batches 20 \
    --save_steps 500 --output_dir $BASE_DIR/phase3c_seq1_twitter

# Sequence 2: Easy → Hard progression (4 experiments: 61-64)
echo "Experiment 61/72: Sequential - Easy binary sentiment"
$PYTHON_CMD train.py --model $MODEL --dataset stanfordnlp/imdb --mode sft \
    --max_steps 2000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 200 --eval_max_batches 20 \
    --save_steps 500 --output_dir $BASE_DIR/phase3c_seq2_easy

echo "Experiment 62/72: Sequential - Medium 3-way sentiment"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq2_easy --dataset takala/financial_phrasebank --mode sft \
    --max_steps 2000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 200 --eval_max_batches 20 \
    --save_steps 500 --output_dir $BASE_DIR/phase3c_seq2_medium

echo "Experiment 63/72: Sequential - Hard 20-way classification"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq2_medium --dataset zeroshot/twitter-financial-news-topic --mode sft \
    --max_steps 3000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 300 --eval_max_batches 20 \
    --save_steps 750 --output_dir $BASE_DIR/phase3c_seq2_hard

echo "Experiment 64/72: Sequential - Hardest Q&A"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq2_hard --dataset virattt/financial-qa-10K --mode sft \
    --max_steps 3000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 300 --eval_max_batches 10 \
    --save_steps 750 --max_length 512 --output_dir $BASE_DIR/phase3c_seq2_hardest

# Sequence 3: Pretrain → Fine-tune (2 experiments: 65-66)
echo "Experiment 65/72: Sequential - Financial pretrain"
$PYTHON_CMD train.py --model $MODEL --dataset gbharti/finance-alpaca --mode pretrain \
    --max_steps 10000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 1000 --eval_max_batches 10 \
    --save_steps 2000 --output_dir $BASE_DIR/phase3c_seq3_pretrain

echo "Experiment 66/72: Sequential - Sentiment fine-tune from pretrain"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq3_pretrain --dataset takala/financial_phrasebank --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 250 --eval_max_batches 20 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3c_seq3_finetune

# Sequence 4: Math → Financial (2 experiments: 67-68)
echo "Experiment 67/72: Sequential - Math pretraining"
$PYTHON_CMD train.py --model $MODEL --dataset openai/gsm8k --dataset_config main --mode pretrain \
    --max_steps 5000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --max_length 512 --output_dir $BASE_DIR/phase3c_seq4_math

echo "Experiment 68/72: Sequential - Financial Q&A from math"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq4_math --dataset virattt/financial-qa-10K --mode sft \
    --max_steps 5000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size 8 --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --max_length 512 --output_dir $BASE_DIR/phase3c_seq4_financial

# Sequence 5: Multi-domain progression (3 experiments: 69-71)
echo "Experiment 69/72: Sequential - General text pretrain"
$PYTHON_CMD train.py --model $MODEL --dataset Skylion007/openwebtext --mode pretrain \
    --max_steps 5000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3c_seq5_general

echo "Experiment 70/72: Sequential - Financial domain from general"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq5_general --dataset gbharti/finance-alpaca --mode pretrain \
    --max_steps 5000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 500 --eval_max_batches 10 \
    --save_steps 1000 --output_dir $BASE_DIR/phase3c_seq5_financial

echo "Experiment 71/72: Sequential - Task-specific from financial"
$PYTHON_CMD train.py --model $BASE_DIR/phase3c_seq5_financial --dataset takala/financial_phrasebank --mode sft \
    --max_steps 3000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 300 --eval_max_batches 20 \
    --save_steps 600 --output_dir $BASE_DIR/phase3c_seq5_task

# Sequence 6: GLUE progression (1 experiment: 72)
echo "Experiment 72/72: Sequential - GLUE progression (CoLA→SST2→MRPC)"
$PYTHON_CMD train.py --model $MODEL \
    --datasets glue glue glue \
    --dataset_configs cola sst2 mrpc --mixture_rates 0.3 0.3 0.4 --mode sft \
    --max_steps 6000 --use_lora --lora_r $FINETUNE_LORA_RANK --lora_alpha $FINETUNE_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --eval_steps 300 --eval_max_batches 20 \
    --save_steps 1200 --output_dir $BASE_DIR/phase3c_seq6_glue

echo ""
echo "=========================================="
echo "EXPERIMENTS COMPLETE!"
echo "=========================================="
echo "Total experiments: 72"
echo "Results saved in: $BASE_DIR"
echo ""
echo "Key configuration for Qwen3-0.6B-Base:"
echo "- Model: Qwen3-0.6B-Base (600M parameters)"
echo "- Pretraining: LoRA rank 16 (alpha 32) for maximum expressiveness"
echo "- Fine-tuning: LoRA rank 4 (alpha 8) for efficient task adaptation"
echo "- LoRA targets: q_proj, k_proj, v_proj, o_proj (attention only)"
echo "- Batch size 16 for most tasks (2x larger than 1.7B model)"
echo "- Batch size 8 for memory-intensive tasks (Q&A, long sequences)"
echo "- eval_steps for flexible evaluation frequency"
echo "- eval_max_batches to control evaluation cost"
echo ""
echo "To analyze results, check:"
echo "1. Training logs in each experiment's logs/ directory"
echo "2. TensorBoard logs in each experiment's tensorboard/ directory"
echo "3. Model checkpoints in each experiment's checkpoints/ directory"
echo ""
echo "Run tensorboard with:"
echo "tensorboard --logdir $BASE_DIR"
echo "==========================================
"