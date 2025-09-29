#!/bin/bash
# Phase 2C: Analytical Capabilities Pretraining for Qwen3-0.6B
# Run AFTER Phase 2B to build analytical skills on top of financial knowledge
# Expected runtime: ~7 hours for all 7 experiments on RTX 4090

# ============================================================
# CHECKPOINT CONFIGURATION (CRITICAL - READ THIS!)
# ============================================================
# Set this to a checkpoint from Phase 2B to continue training
# This ensures the model retains financial domain knowledge
# 
# Example checkpoints (replace {model_short} with your model, e.g., qwen3_0.6b):
#   "./runs/phase2b_financial_{model_short}/mixed_financial/checkpoints/checkpoint-150000"
#   "./runs/phase2b_financial_{model_short}/fingpt/checkpoints/checkpoint-100000"
#   "./runs/phase2b_financial_{model_short}/financial_qa/checkpoints/checkpoint-100000"
#
# To find available checkpoints:
#   ls ./runs/phase2b_financial_*/*/checkpoints/
#   # or for specific model:
#   ls ./runs/phase2b_financial_qwen3_0.6b/*/checkpoints/
#
# Leave empty to start from base model (NOT RECOMMENDED)
# ============================================================

CHECKPOINT_PATH=""  # <-- SET THIS TO YOUR PHASE 2B CHECKPOINT!

# ============================================================
# Verify checkpoint configuration
# ============================================================
if [ -n "$CHECKPOINT_PATH" ]; then
    if [ -d "$CHECKPOINT_PATH" ]; then
        echo "✓ Starting from Phase 2B checkpoint: $CHECKPOINT_PATH"
        CHECKPOINT_ARG="--resume_from_checkpoint $CHECKPOINT_PATH"
        # Extract checkpoint name for output directory
        CHECKPOINT_NAME=$(basename $(dirname "$CHECKPOINT_PATH"))_$(basename "$CHECKPOINT_PATH")
        OUTPUT_SUFFIX="_from_${CHECKPOINT_NAME}"
    else
        echo "✗ ERROR: Checkpoint path does not exist: $CHECKPOINT_PATH"
        echo "Please verify the path and try again."
        exit 1
    fi
else
    echo "⚠ WARNING: No Phase 2B checkpoint specified!"
    echo "Starting from base model will lose all financial domain knowledge."
    echo "This is NOT RECOMMENDED for Phase 2C."
    echo ""
    read -p "Are you sure you want to continue without a checkpoint? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please set CHECKPOINT_PATH to a Phase 2B checkpoint."
        exit 1
    fi
    CHECKPOINT_ARG=""
    OUTPUT_SUFFIX="_from_base"
fi

# ========================================
# MODEL CONFIGURATION
# ========================================
MODEL="Qwen/Qwen3-0.6B-Base"  # Same base model as Phase 2B
MODEL_SHORT="qwen3_0.6b"       # Used in output directory names

# ========================================
# LORA CONFIGURATION (MUST MATCH PHASE 2B!)
# ========================================
PRETRAIN_LORA_RANK=32          # Must match Phase 2B configuration
PRETRAIN_LORA_ALPHA=64         # Must match Phase 2B configuration
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj"  # Must match Phase 2B

# ========================================
# TRAINING CONFIGURATION
# ========================================
BATCH_SIZE=256                 # Same as Phase 2B for consistency
MAX_LENGTH=1024                # Same as Phase 2B
BASE_DIR="./runs/phase2c_analytical_${MODEL_SHORT}${OUTPUT_SUFFIX}"
PYTHON_CMD="/Users/mengzhao/miniconda3/envs/lancy/bin/python"

# Learning rate configuration (can be adjusted for continued training)
LEARNING_RATE=1e-5             # Slightly lower than Phase 2B for fine-tuning
LR_SCHEDULER="cosine"          # Cosine annealing
WARMUP_STEPS=2000             # Shorter warmup for continued training
WEIGHT_DECAY=0.01             # Weight decay for regularization

# Evaluation and saving
EVAL_MAX_BATCHES=100          # Batches for evaluation
EVAL_STEPS=1000               # Evaluate every 1k steps
SAVE_STEPS=5000               # Save checkpoints every 5k steps

# Multi-dataset evaluation settings
SEPARATE_EVAL="--separate_mixture_eval"  # For mixture experiments
LOG_SPREAD="--log_eval_spread"          # Log spread metrics

echo "=========================================="
echo "PHASE 2C: ANALYTICAL CAPABILITIES PRETRAINING"
echo "=========================================="
echo "Model: $MODEL ($MODEL_SHORT)"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Starting from: $CHECKPOINT_PATH"
else
    echo "Starting from: Base model (WARNING: Not recommended!)"
fi
echo "LoRA: rank=$PRETRAIN_LORA_RANK, alpha=$PRETRAIN_LORA_ALPHA"
echo "Target modules: $LORA_TARGET_MODULES"
echo "Batch size: $BATCH_SIZE"
echo "Max length: $MAX_LENGTH tokens"
echo "Learning rate: $LEARNING_RATE with $LR_SCHEDULER scheduler"
echo "Output directory: $BASE_DIR"
echo "=========================================="

# Create base directory
mkdir -p $BASE_DIR

# ===========================================
# EXPERIMENT 7: GSM8K Math Pretraining
# ===========================================
echo ""
echo "Experiment 7/13: GSM8K math pretrain (50k steps)"
echo "Dataset: openai/gsm8k (Grade school math word problems)"
echo "Purpose: Learn mathematical reasoning for financial calculations"

$PYTHON_CMD train.py --model $MODEL --dataset openai/gsm8k --dataset_config main --mode pretrain \
    $CHECKPOINT_ARG \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 15 \
    --save_strategy steps \
    --output_dir $BASE_DIR/gsm8k

# ===========================================
# EXPERIMENT 8: DeepMind Math Pretraining
# ===========================================
echo ""
echo "Experiment 8/13: DeepMind math pretrain (50k steps)"
echo "Dataset: deepmind/math_dataset (Diverse mathematical problems)"
echo "Purpose: Broaden mathematical capabilities beyond word problems"

$PYTHON_CMD train.py --model $MODEL --dataset deepmind/math_dataset --mode pretrain \
    $CHECKPOINT_ARG \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 15 \
    --save_strategy steps \
    --output_dir $BASE_DIR/deepmind

# ===========================================
# EXPERIMENT 9: BigCodeBench Pretraining
# ===========================================
echo ""
echo "Experiment 9/13: BigCodeBench pretrain (50k steps)"
echo "Dataset: bigcode/bigcodebench (Code generation with function calls)"
echo "Purpose: Learn coding for financial analysis and automation"

$PYTHON_CMD train.py --model $MODEL --dataset bigcode/bigcodebench --mode pretrain \
    $CHECKPOINT_ARG \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 15 \
    --save_strategy steps \
    --output_dir $BASE_DIR/bigcode

# ===========================================
# EXPERIMENT 10: GLUE MNLI Pretraining
# ===========================================
echo ""
echo "Experiment 10/13: GLUE MNLI pretrain (50k steps)"
echo "Dataset: glue/mnli (Natural language inference)"
echo "Purpose: Improve logical reasoning and entailment understanding"

$PYTHON_CMD train.py --model $MODEL --dataset glue --dataset_config mnli --mode pretrain \
    $CHECKPOINT_ARG \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 15 \
    --save_strategy steps \
    --output_dir $BASE_DIR/glue_mnli

# ===========================================
# EXPERIMENT 11: MMLU-Pro Pretraining
# ===========================================
echo ""
echo "Experiment 11/13: MMLU-Pro pretrain (50k steps)"
echo "Dataset: TIGER-Lab/MMLU-Pro (Advanced multi-choice reasoning)"
echo "Purpose: Complex reasoning across multiple domains"

$PYTHON_CMD train.py --model $MODEL --dataset TIGER-Lab/MMLU-Pro --mode pretrain \
    $CHECKPOINT_ARG \
    --max_steps 50000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 15 \
    --save_strategy steps \
    --output_dir $BASE_DIR/mmlu_pro

# ===========================================
# EXPERIMENT 12: Math+Code Mixture
# ===========================================
echo ""
echo "Experiment 12/13: Math+Code mixture pretrain (75k steps)"
echo "Datasets: GSM8K (60%) + BigCodeBench (40%)"
echo "Purpose: Combined mathematical and coding capabilities"
echo "Multi-dataset evaluation: ENABLED"

$PYTHON_CMD train.py --model $MODEL \
    --datasets openai/gsm8k bigcode/bigcodebench \
    --dataset_configs main None \
    --mixture_rates 0.6 0.4 \
    --mode pretrain \
    $CHECKPOINT_ARG \
    --max_steps 75000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 20 \
    --save_strategy steps \
    $SEPARATE_EVAL $LOG_SPREAD \
    --output_dir $BASE_DIR/math_code

# ===========================================
# EXPERIMENT 13: Mixed Analytical Corpus (FINAL)
# ===========================================
echo ""
echo "Experiment 13/13: Mixed analytical corpus (100k steps)"
echo "Combines 4 analytical datasets:"
echo "  - GSM8K: 30% (math word problems)"
echo "  - DeepMind Math: 30% (diverse math)"
echo "  - BigCodeBench: 20% (coding)"
echo "  - MMLU-Pro: 20% (reasoning)"
echo "Multi-dataset evaluation: ENABLED"

$PYTHON_CMD train.py --model $MODEL \
    --datasets openai/gsm8k deepmind/math_dataset bigcode/bigcodebench TIGER-Lab/MMLU-Pro \
    --dataset_configs main None None None \
    --mixture_rates 0.3 0.3 0.2 0.2 \
    --mode pretrain \
    $CHECKPOINT_ARG \
    --max_steps 100000 --use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
    --save_steps $SAVE_STEPS --save_total_limit 25 \
    --save_strategy steps \
    $SEPARATE_EVAL $LOG_SPREAD \
    --output_dir $BASE_DIR/mixed_analytical

echo ""
echo "=========================================="
echo "PHASE 2C ANALYTICAL PRETRAINING COMPLETE!"
echo "=========================================="
echo "Total experiments: 7"
echo "Results saved in: $BASE_DIR"
echo ""
echo "Experiment summary:"
echo "  7. GSM8K math: 50k steps"
echo "  8. DeepMind math: 50k steps"
echo "  9. BigCodeBench: 50k steps"
echo "  10. GLUE MNLI: 50k steps"
echo "  11. MMLU-Pro: 50k steps"
echo "  12. Math+Code mix: 75k steps"
echo "  13. Mixed analytical: 100k steps (FINAL MODEL)"
echo ""
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Started from Phase 2B checkpoint: $CHECKPOINT_PATH"
    echo "The model now has both financial domain knowledge AND analytical capabilities!"
else
    echo "Started from base model (not recommended)"
    echo "The model has analytical capabilities but lacks financial domain knowledge."
fi
echo ""
echo "Next steps:"
echo "1. Evaluate the final checkpoint on downstream tasks"
echo "2. Fine-tune for specific financial applications (Phase 3)"
echo ""
echo "Best checkpoint for downstream tasks:"
echo "  $BASE_DIR/mixed_analytical/checkpoints/checkpoint-100000"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir $BASE_DIR"
echo "=========================================="