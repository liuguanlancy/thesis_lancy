#!/bin/bash
# Phase 2B: Financial Domain Pretraining
# Supports any HuggingFace model with configurable LoRA or full fine-tuning
# Supports both online (HuggingFace) and local dataset loading
# Expected runtime: ~10 hours for all 6 experiments on RTX 4090 with default settings

# Set environment variable to trust remote code for HuggingFace datasets (required for SEC Reports)
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ========================================
# DEFAULT HYPERPARAMETERS
# ========================================
# System defaults
DEFAULT_PYTHON_CMD="/Users/mengzhao/miniconda3/envs/lancy/bin/python"

# Model defaults
DEFAULT_MODEL="Qwen/Qwen3-0.6B-Base"
DEFAULT_MODEL_SHORT=""  # Auto-generated if not specified

# LoRA defaults
DEFAULT_USE_LORA=true
DEFAULT_LORA_RANK=32
DEFAULT_LORA_ALPHA=64
DEFAULT_LORA_MODULES="q_proj k_proj v_proj o_proj"

# Training defaults
# Detect if running on Mac for memory-optimized defaults
if [[ "$(uname)" == "Darwin" ]]; then
    DEFAULT_BATCH_SIZE=8  # Optimized for 16GB Apple Silicon
else
    DEFAULT_BATCH_SIZE=256  # For GPUs with more memory
fi
DEFAULT_MAX_LENGTH=1024
DEFAULT_MAX_STEPS=4000  # Default training steps
DEFAULT_LEARNING_RATE="2e-5"
DEFAULT_LR_SCHEDULER="cosine"
DEFAULT_WARMUP_STEPS=400  # 10% of default max steps (4000)
DEFAULT_WEIGHT_DECAY=0.01

# Mixed precision defaults
DEFAULT_PRECISION="bf16"

# Evaluation defaults
DEFAULT_EVAL_MAX_BATCHES=100
DEFAULT_EVAL_STEPS=1000
DEFAULT_SAVE_STEPS=500
DEFAULT_SAVE_TOTAL_LIMIT=3  # Default to keeping 3 checkpoints
DEFAULT_EVAL_ON_START=false  # Evaluate at step 0 before training
DEFAULT_GRADIENT_ACCUM=1  # Gradient accumulation steps

# Packing defaults
DEFAULT_USE_PACKING=true  # Enable sequence packing for pretraining
DEFAULT_PACKING_MAX_LENGTH=""  # Empty means use MAX_LENGTH

# Output defaults
DEFAULT_OUTPUT_BASE_DIR=""  # Will be set based on MODEL_SHORT

# Attention implementation default
DEFAULT_ATTN_IMPLEMENTATION="auto"  # Auto-detect best implementation (flash_attention_2 on RTX/Ada, eager on MPS)

# Mixing strategy defaults
DEFAULT_MIXING_STRATEGY="50cap"  # Default to 50% News cap strategy

# Multi-dataset evaluation defaults
DEFAULT_EVAL_ALL_DATASETS=false  # Evaluate on all datasets even for single dataset training

# ========================================
# INITIALIZE VARIABLES WITH DEFAULTS
# ========================================
USE_LOCAL=false
DATASET_CACHE_DIR="./datasets/phase2b"
EXPERIMENTS="all"  # Default: run all experiments
DRY_RUN=false  # For testing configuration without running

# Initialize hyperparameters with defaults
PYTHON_CMD="$DEFAULT_PYTHON_CMD"
MODEL="$DEFAULT_MODEL"
MODEL_SHORT="$DEFAULT_MODEL_SHORT"
USE_LORA="$DEFAULT_USE_LORA"
PRETRAIN_LORA_RANK="$DEFAULT_LORA_RANK"
PRETRAIN_LORA_ALPHA="$DEFAULT_LORA_ALPHA"
LORA_TARGET_MODULES="$DEFAULT_LORA_MODULES"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
MAX_LENGTH="$DEFAULT_MAX_LENGTH"
MAX_STEPS="$DEFAULT_MAX_STEPS"
LEARNING_RATE="$DEFAULT_LEARNING_RATE"
LR_SCHEDULER="$DEFAULT_LR_SCHEDULER"
WARMUP_STEPS="$DEFAULT_WARMUP_STEPS"
WEIGHT_DECAY="$DEFAULT_WEIGHT_DECAY"
PRECISION="$DEFAULT_PRECISION"
EVAL_MAX_BATCHES="$DEFAULT_EVAL_MAX_BATCHES"
EVAL_STEPS="$DEFAULT_EVAL_STEPS"
SAVE_STEPS="$DEFAULT_SAVE_STEPS"
SAVE_TOTAL_LIMIT="$DEFAULT_SAVE_TOTAL_LIMIT"
EVAL_ON_START="$DEFAULT_EVAL_ON_START"
USE_PACKING="$DEFAULT_USE_PACKING"
PACKING_MAX_LENGTH="$DEFAULT_PACKING_MAX_LENGTH"
OUTPUT_BASE_DIR="$DEFAULT_OUTPUT_BASE_DIR"
ATTN_IMPLEMENTATION="$DEFAULT_ATTN_IMPLEMENTATION"
MIXING_STRATEGY="$DEFAULT_MIXING_STRATEGY"
CUSTOM_MIXTURE_RATES=""
EVAL_ALL_DATASETS="$DEFAULT_EVAL_ALL_DATASETS"
GRADIENT_ACCUM="$DEFAULT_GRADIENT_ACCUM"

# ========================================
# COMMAND LINE ARGUMENT PARSING
# ========================================
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Dataset options
        --local)
            USE_LOCAL=true
            shift
            ;;
        --cache-dir)
            DATASET_CACHE_DIR="$2"
            shift 2
            ;;
        --experiments)
            EXPERIMENTS="$2"
            shift 2
            ;;
        
        # System options
        --python-cmd)
            PYTHON_CMD="$2"
            shift 2
            ;;
        
        # Model options
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model-short)
            MODEL_SHORT="$2"
            shift 2
            ;;
        --attn-implementation)
            ATTN_IMPLEMENTATION="$2"
            shift 2
            ;;
        
        # LoRA options
        --no-lora)
            USE_LORA=false
            shift
            ;;
        --lora-rank)
            PRETRAIN_LORA_RANK="$2"
            shift 2
            ;;
        --lora-alpha)
            PRETRAIN_LORA_ALPHA="$2"
            shift 2
            ;;
        --lora-modules)
            LORA_TARGET_MODULES="$2"
            shift 2
            ;;
        
        # Training options
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --learning-rate|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lr-scheduler)
            LR_SCHEDULER="$2"
            shift 2
            ;;
        --warmup-steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --gradient-accum)
            GRADIENT_ACCUM="$2"
            shift 2
            ;;

        # Mixed precision options
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        
        # Evaluation options
        --eval-batches)
            EVAL_MAX_BATCHES="$2"
            shift 2
            ;;
        --eval-steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --eval-on-start)
            EVAL_ON_START=true
            shift
            ;;
        
        # Packing options
        --use-packing)
            USE_PACKING=true
            shift
            ;;
        --packing-max-length)
            PACKING_MAX_LENGTH="$2"
            shift 2
            ;;
        
        --save-steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        
        --save-total-limit)
            SAVE_TOTAL_LIMIT="$2"
            shift 2
            ;;
        
        # Output options
        --output-base-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        
        # Mixing strategy options
        --mixing-strategy)
            MIXING_STRATEGY="$2"
            shift 2
            ;;
        --custom-mixture-rates)
            CUSTOM_MIXTURE_RATES="$2"
            shift 2
            ;;
        
        # Multi-dataset evaluation options
        --eval-all-datasets)
            EVAL_ALL_DATASETS=true
            shift
            ;;
        --no-eval-all-datasets)
            EVAL_ALL_DATASETS=false
            shift
            ;;
        
        # Test options
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        
        # Information options
        --list-experiments)
            echo "Available experiments:"
            echo "  1 or financial_qa   - Financial Q&A (4k steps)"
            echo "  2 or fingpt         - FinGPT Sentiment (4k steps)"
            echo "  3 or alpaca         - Finance Alpaca (4k steps)"
            echo "  4 or fiqa           - FiQA (4k steps)"
            echo "  5 or twitter        - Twitter Sentiment (4k steps)"
            echo "  6 or sec_reports    - SEC Financial Reports (4k steps)"
            echo "  7 or news_articles  - Financial News Articles (4k steps)"
            echo "  8 or wikitext       - WikiText pretraining (4k steps)"
            echo "  9 or mixed          - Mixed Financial Corpus (7 datasets, 4k steps)"
            echo "  10 or mixed-wiki    - Mixed with WikiText (8 datasets, 4k steps)"
            echo ""
            echo "Usage examples:"
            echo "  --experiments all                    # Run all experiments (default)"
            echo "  --experiments 1,2,3                  # Run experiments 1, 2, and 3"
            echo "  --experiments financial_qa,fingpt    # Run by name"
            echo "  --experiments mixed                  # Run only mixed corpus (7 datasets)"
            echo "  --experiments mixed-wiki             # Run mixed with WikiText (8 datasets)"
            echo ""
            echo "Mixing strategies for experiments 9-10:"
            echo "  50cap         - 50% News cap with square root scaling (recommended)"
            echo "  sqrt          - Pure square root scaling"
            echo "  proportional  - Proportional to dataset sizes"
            echo "  uniform       - Equal weights for all datasets"
            echo "  custom        - Use --custom-mixture-rates"
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "DATASET OPTIONS:"
            echo "  --local                 Use locally cached datasets instead of downloading"
            echo "  --cache-dir PATH        Directory containing cached datasets (default: ./datasets/phase2b)"
            echo "  --experiments LIST      Run specific experiments (default: all)"
            echo "                          LIST can be: all, numbers (1-10), or names"
            echo "                          Examples: '1,2,3' or 'financial_qa,fingpt'"
            echo ""
            echo "SYSTEM OPTIONS:"
            echo "  --python-cmd PATH       Python command/path (default: $DEFAULT_PYTHON_CMD)"
            echo ""
            echo "MODEL OPTIONS:"
            echo "  --model NAME            Model name/path (default: $DEFAULT_MODEL)"
            echo "  --model-short NAME      Short name for directories (default: auto-generated)"
            echo "  --attn-implementation   Attention implementation: auto, eager, sdpa, flash_attention_2 (default: auto)"
            echo "                          Auto detects best: flash_attention_2 on RTX/Ada, eager on MPS, sdpa otherwise"
            echo ""
            echo "LORA OPTIONS:"
            echo "  --no-lora               Disable LoRA (full fine-tuning, high memory usage)"
            echo "  --lora-rank N           LoRA rank (default: $DEFAULT_LORA_RANK)"
            echo "  --lora-alpha N          LoRA alpha (default: $DEFAULT_LORA_ALPHA)"
            echo "  --lora-modules MODULES  Target modules (default: \"$DEFAULT_LORA_MODULES\")"
            echo ""
            echo "TRAINING OPTIONS:"
            echo "  --batch-size N          Batch size (default: $DEFAULT_BATCH_SIZE)"
            echo "  --max-length N          Max sequence length (default: $DEFAULT_MAX_LENGTH)"
            echo "  --max-steps N           Max training steps (default: $DEFAULT_MAX_STEPS)"
            echo "  --learning-rate LR      Learning rate (default: $DEFAULT_LEARNING_RATE)"
            echo "  --lr-scheduler TYPE     LR scheduler type (default: $DEFAULT_LR_SCHEDULER)"
            echo "  --warmup-steps N        Warmup steps (default: $DEFAULT_WARMUP_STEPS)"
            echo "  --weight-decay W        Weight decay (default: $DEFAULT_WEIGHT_DECAY)"
            echo ""
            echo "MIXED PRECISION OPTIONS:"
            echo "  --precision MODE        Mixed precision mode: bf16, fp16, or fp32 (default: $DEFAULT_PRECISION)"
            echo ""
            echo "EVALUATION OPTIONS:"
            echo "  --eval-batches N        Max eval batches (default: $DEFAULT_EVAL_MAX_BATCHES)"
            echo "  --eval-steps N          Evaluation frequency (default: $DEFAULT_EVAL_STEPS)"
            echo "  --eval-on-start         Evaluate at step 0 before training starts"
            echo "  --save-steps N          Save frequency (default: $DEFAULT_SAVE_STEPS)"
            echo "  --save-total-limit N    Max checkpoints to keep, -1 for all (default: $DEFAULT_SAVE_TOTAL_LIMIT)"
            echo ""
            echo "PACKING OPTIONS (for pretraining efficiency):"
            echo "  --use-packing           Enable sequence packing (~2.5x speedup for short sequences)"
            echo "  --packing-max-length N  Max length for packed sequences (default: uses --max-length)"
            echo ""
            echo "OUTPUT OPTIONS:"
            echo "  --output-base-dir PATH  Base output directory (default: ./runs/phase2b_financial_MODEL_SHORT)"
            echo ""
            echo "MIXING OPTIONS (for experiments 9-10):"
            echo "  --mixing-strategy NAME  Strategy for dataset mixing (default: 50cap)"
            echo "                          Options: 50cap, sqrt, proportional, uniform, custom"
            echo "  --custom-mixture-rates  Custom rates (7 space-separated floats summing to 1.0)"
            echo "                          Order: financial_qa fingpt alpaca fiqa twitter sec news"
            echo ""
            echo "EVALUATION OPTIONS:"
            echo "  --eval-all-datasets     Evaluate on all 8 datasets even when training on one (default: off)"
            echo "  --no-eval-all-datasets  Disable multi-dataset evaluation (default behavior)"
            echo ""
            echo "INFORMATION OPTIONS:"
            echo "  --list-experiments      Show available experiments and exit"
            echo "  --help                  Show this help message"
            echo ""
            echo "EXAMPLES:"
            echo "  $0                                              # Use all defaults"
            echo "  $0 --local                                      # Use local datasets"
            echo "  $0 --no-lora --batch-size 32                    # Full fine-tuning with smaller batch"
            echo "  $0 --lora-rank 16 --lora-alpha 32               # Custom LoRA configuration"
            echo "  $0 --model Qwen/Qwen3-1.7B-Base --batch-size 128  # Different model"
            echo "  $0 --precision fp16                             # Use FP16 instead of BF16"
            echo "  $0 --python-cmd python3                         # Use system python3"
            echo "  $0 --experiments 1,2 --eval-steps 100           # Quick testing setup"
            echo "  $0 --use-packing --packing-max-length 4096     # Enable packing for efficiency"
            echo "  $0 --experiments mixed --mixing-strategy sqrt   # Use square root scaling"
            echo "  $0 --experiments mixed --custom-mixture-rates '0.1 0.15 0.15 0.1 0.05 0.2 0.25'  # Custom rates"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ========================================
# POST-PROCESSING AND VALIDATION
# ========================================

# Auto-generate MODEL_SHORT if not specified
if [ -z "$MODEL_SHORT" ]; then
    # Extract model name from path and create short version
    MODEL_SHORT=$(echo "$MODEL" | sed -e 's|.*/||' -e 's/[-_]/ /g' | 
                  awk '{for(i=1;i<=NF;i++){$i=tolower($i)}; gsub(/ /,"_"); print}' | 
                  sed -e 's/base//' -e 's/__*/_/g' -e 's/_$//')
    # Handle specific patterns
    MODEL_SHORT=$(echo "$MODEL_SHORT" | sed \
        -e 's/qwen3/qwen3/g' \
        -e 's/llama_3\.2/llama3.2/g' \
        -e 's/gemma_2/gemma2/g')
fi

# Set output base directory if not specified
if [ -z "$OUTPUT_BASE_DIR" ]; then
    OUTPUT_BASE_DIR="./runs/phase2b_financial_${MODEL_SHORT}"
fi
BASE_DIR="$OUTPUT_BASE_DIR"

# Convert precision to mixed precision flag
case "$PRECISION" in
    bf16)
        MIXED_PRECISION="--bf16"
        ;;
    fp16)
        MIXED_PRECISION="--fp16"
        ;;
    fp32|none|"")
        MIXED_PRECISION="--no_mixed_precision"
        ;;
    *)
        echo "ERROR: Invalid precision mode: $PRECISION"
        echo "Valid options: bf16, fp16, fp32"
        exit 1
        ;;
esac

# Validate Python command
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "WARNING: Python command not found: $PYTHON_CMD"
    echo "Trying to use 'python' instead..."
    PYTHON_CMD="python"
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo "ERROR: No Python interpreter found"
        exit 1
    fi
fi

# Validate numeric parameters
if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -le 0 ]; then
    echo "ERROR: Batch size must be a positive integer: $BATCH_SIZE"
    exit 1
fi

if ! [[ "$MAX_LENGTH" =~ ^[0-9]+$ ]] || [ "$MAX_LENGTH" -le 0 ]; then
    echo "ERROR: Max length must be a positive integer: $MAX_LENGTH"
    exit 1
fi

if [ "$USE_LORA" = true ]; then
    if ! [[ "$PRETRAIN_LORA_RANK" =~ ^[0-9]+$ ]] || [ "$PRETRAIN_LORA_RANK" -le 0 ]; then
        echo "ERROR: LoRA rank must be a positive integer: $PRETRAIN_LORA_RANK"
        exit 1
    fi
    
    if ! [[ "$PRETRAIN_LORA_ALPHA" =~ ^[0-9]+$ ]] || [ "$PRETRAIN_LORA_ALPHA" -le 0 ]; then
        echo "ERROR: LoRA alpha must be a positive integer: $PRETRAIN_LORA_ALPHA"
        exit 1
    fi
    
    # Warning if alpha is not 2x rank (common convention)
    EXPECTED_ALPHA=$((PRETRAIN_LORA_RANK * 2))
    if [ "$PRETRAIN_LORA_ALPHA" -ne "$EXPECTED_ALPHA" ]; then
        echo "WARNING: LoRA alpha ($PRETRAIN_LORA_ALPHA) is not 2x rank ($PRETRAIN_LORA_RANK). Common convention is alpha = 2 * rank = $EXPECTED_ALPHA"
    fi
fi

# Build LoRA arguments conditionally
if [ "$USE_LORA" = true ]; then
    LORA_ARGS="--use_lora --lora_r $PRETRAIN_LORA_RANK --lora_alpha $PRETRAIN_LORA_ALPHA --lora_target_modules $LORA_TARGET_MODULES"
else
    LORA_ARGS=""
    echo "WARNING: LoRA is disabled. Full fine-tuning will use significantly more memory."
    echo "         Consider reducing batch size if you encounter OOM errors."
fi

# Calculate warmup steps dynamically based on max_steps
CALCULATED_WARMUP=$((MAX_STEPS / 10))  # 10% of max steps
if [ "$CALCULATED_WARMUP" -lt 1 ]; then
    CALCULATED_WARMUP=1  # Minimum 1 warmup step
fi

# Adjust warmup if it's larger than calculated value
if [ "$CALCULATED_WARMUP" -lt "$WARMUP_STEPS" ]; then
    echo "INFO: Adjusting warmup from $WARMUP_STEPS to $CALCULATED_WARMUP (10% of $MAX_STEPS steps)"
    WARMUP_STEPS=$CALCULATED_WARMUP
fi

# Ensure warmup doesn't exceed max steps
if [ "$WARMUP_STEPS" -ge "$MAX_STEPS" ]; then
    WARMUP_STEPS=$CALCULATED_WARMUP
    echo "WARNING: Warmup steps ($WARMUP_STEPS) was >= max steps ($MAX_STEPS)"
    echo "         Adjusted to $WARMUP_STEPS (10% of max steps)"
fi

# Build eval_on_start argument conditionally
if [ "$EVAL_ON_START" = true ]; then
    EVAL_START_ARG="--eval_on_start"
else
    EVAL_START_ARG=""
fi

# Build packing arguments conditionally
if [ "$USE_PACKING" = true ]; then
    PACKING_ARGS="--use_packing"
    if [ -n "$PACKING_MAX_LENGTH" ]; then
        PACKING_ARGS="$PACKING_ARGS --packing_max_length $PACKING_MAX_LENGTH"
    fi
else
    PACKING_ARGS=""
fi

# Validate experiment selection
if [ "$EXPERIMENTS" != "all" ]; then
    # Check for invalid experiment numbers/names
    IFS=',' read -ra EXP_ARRAY <<< "$EXPERIMENTS"
    for exp in "${EXP_ARRAY[@]}"; do
        case $exp in
            1|2|3|4|5|6|7|8|9|10|financial_qa|fingpt|alpaca|fiqa|twitter|sec_reports|news_articles|wikitext|mixed|mixed-wiki)
                # Valid experiment
                ;;
            *)
                echo "ERROR: Invalid experiment identifier: $exp"
                echo "Use --list-experiments to see available options"
                exit 1
                ;;
        esac
    done
fi

# Multi-dataset evaluation settings (for mixture experiments)
SEPARATE_EVAL="--separate_mixture_eval"  # Evaluate each dataset separately
LOG_SPREAD="--log_eval_spread"          # Log spread metrics for balance

# Function to run training with optional multi-dataset evaluation
run_training_with_eval() {
    local primary_dataset=$1
    local primary_config=$2
    local output_name=$3
    
    if [ "$EVAL_ALL_DATASETS" = true ]; then
        echo "Multi-dataset evaluation enabled for single dataset training"
        
        # Determine mixture rates based on primary dataset
        case $primary_dataset in
            "$DATASET_FINANCIAL_QA")
                EVAL_MIXTURE_RATES="1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
                ;;
            "$DATASET_FINGPT")
                EVAL_MIXTURE_RATES="0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0"
                ;;
            "$DATASET_ALPACA")
                EVAL_MIXTURE_RATES="0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0"
                ;;
            "$DATASET_FIQA")
                EVAL_MIXTURE_RATES="0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0"
                ;;
            "$DATASET_TWITTER")
                EVAL_MIXTURE_RATES="0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0"
                ;;
            "$DATASET_SEC")
                EVAL_MIXTURE_RATES="0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0"
                ;;
            "$DATASET_NEWS")
                EVAL_MIXTURE_RATES="0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0"
                ;;
            "$DATASET_WIKITEXT")
                EVAL_MIXTURE_RATES="0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0"
                ;;
            *)
                echo "Warning: Unknown dataset for multi-eval: $primary_dataset"
                EVAL_MIXTURE_RATES="0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125"
                ;;
        esac
        
        # Run with multi-dataset evaluation
        $PYTHON_CMD train.py --model $MODEL \
            --datasets "$DATASET_FINANCIAL_QA" "$DATASET_FINGPT" "$DATASET_ALPACA" "$DATASET_FIQA" "$DATASET_TWITTER" "$DATASET_SEC" "$DATASET_NEWS" "$DATASET_WIKITEXT" \
            --dataset_configs None None None None None small_lite None wikitext-103-v1 \
            --mixture_rates $EVAL_MIXTURE_RATES \
            $SEPARATE_EVAL $LOG_SPREAD \
            --mode pretrain \
            --attn_implementation $ATTN_IMPLEMENTATION \
            --max_steps $MAX_STEPS $LORA_ARGS \
            --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
            --gradient_accumulation_steps $GRADIENT_ACCUM \
            $PACKING_ARGS \
            --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
            --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
            $MIXED_PRECISION \
            --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
            $EVAL_START_ARG \
            --save_steps $SAVE_STEPS --save_total_limit $SAVE_TOTAL_LIMIT \
            --save_strategy steps \
            --output_dir $BASE_DIR/$output_name
    else
        # Single dataset training and evaluation (original behavior)
        if [ -n "$primary_config" ] && [ "$primary_config" != "None" ]; then
            $PYTHON_CMD train.py --model $MODEL \
                --dataset "$primary_dataset" --dataset_config $primary_config \
                --mode pretrain \
                --attn_implementation $ATTN_IMPLEMENTATION \
                --max_steps $MAX_STEPS $LORA_ARGS \
                --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
                --gradient_accumulation_steps $GRADIENT_ACCUM \
                $PACKING_ARGS \
                --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
                --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
                $MIXED_PRECISION \
                --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
                $EVAL_START_ARG \
                --save_steps $SAVE_STEPS --save_total_limit $SAVE_TOTAL_LIMIT \
                --save_strategy steps \
                --output_dir $BASE_DIR/$output_name
        else
            # Add dataset config if provided
            DATASET_CONFIG_ARG=""
            if [ "$primary_config" != "None" ] && [ -n "$primary_config" ]; then
                DATASET_CONFIG_ARG="--dataset_config $primary_config"
            fi

            # Debug output in dry-run mode
            if [ "$DRY_RUN" = true ]; then
                echo "Would execute single-dataset training command:"
                echo "$PYTHON_CMD train.py --model $MODEL \\"
                echo "    --dataset \"$primary_dataset\" $DATASET_CONFIG_ARG \\"
                echo "    --mode pretrain \\"
                echo "    --attn_implementation $ATTN_IMPLEMENTATION \\"
                echo "    --max_steps $MAX_STEPS $LORA_ARGS \\"
                echo "    --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \\"
                echo "    --gradient_accumulation_steps $GRADIENT_ACCUM \\"
                echo "    $PACKING_ARGS \\"
                echo "    --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \\"
                echo "    --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \\"
                echo "    $MIXED_PRECISION \\"
                echo "    --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \\"
                echo "    $EVAL_START_ARG \\"
                echo "    --save_steps $SAVE_STEPS --save_total_limit $SAVE_TOTAL_LIMIT \\"
                echo "    --save_strategy steps \\"
                echo "    --output_dir $BASE_DIR/$output_name"
                return 0
            fi

            $PYTHON_CMD train.py --model $MODEL \
                --dataset "$primary_dataset" $DATASET_CONFIG_ARG \
                --mode pretrain \
                --attn_implementation $ATTN_IMPLEMENTATION \
                --max_steps $MAX_STEPS $LORA_ARGS \
                --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
                --gradient_accumulation_steps $GRADIENT_ACCUM \
                $PACKING_ARGS \
                --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
                --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
                $MIXED_PRECISION \
                --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
                $EVAL_START_ARG \
                --save_steps $SAVE_STEPS --save_total_limit $SAVE_TOTAL_LIMIT \
                --save_strategy steps \
                --output_dir $BASE_DIR/$output_name
        fi
    fi
}

# ========================================
# HELPER FUNCTIONS
# ========================================
# Function to check if an experiment should run
should_run_experiment() {
    local exp_id=$1    # Can be number (1-6) or name
    
    # If running all experiments, always return true
    if [ "$EXPERIMENTS" = "all" ]; then
        return 0
    fi
    
    # Check if this experiment is in the list
    if echo ",$EXPERIMENTS," | grep -q ",$exp_id,"; then
        return 0
    fi
    
    # Also check by name mapping
    case $exp_id in
        1|financial_qa)
            if echo ",$EXPERIMENTS," | grep -q ",1,\|,financial_qa,"; then
                return 0
            fi
            ;;
        2|fingpt)
            if echo ",$EXPERIMENTS," | grep -q ",2,\|,fingpt,"; then
                return 0
            fi
            ;;
        3|alpaca)
            if echo ",$EXPERIMENTS," | grep -q ",3,\|,alpaca,"; then
                return 0
            fi
            ;;
        4|fiqa)
            if echo ",$EXPERIMENTS," | grep -q ",4,\|,fiqa,"; then
                return 0
            fi
            ;;
        5|twitter)
            if echo ",$EXPERIMENTS," | grep -q ",5,\|,twitter,"; then
                return 0
            fi
            ;;
        6|sec_reports)
            if echo ",$EXPERIMENTS," | grep -q ",6,\|,sec_reports,"; then
                return 0
            fi
            ;;
        7|news_articles)
            if echo ",$EXPERIMENTS," | grep -q ",7,\|,news_articles,"; then
                return 0
            fi
            ;;
        8|wikitext)
            if echo ",$EXPERIMENTS," | grep -q ",8,\|,wikitext,"; then
                return 0
            fi
            ;;
        9|mixed)
            if echo ",$EXPERIMENTS," | grep -q ",9,\|,mixed,"; then
                return 0
            fi
            ;;
        10|mixed-wiki)
            if echo ",$EXPERIMENTS," | grep -q ",10,\|,mixed-wiki,"; then
                return 0
            fi
            ;;
    esac
    
    return 1  # Don't run this experiment
}

# Display which experiments will run
display_selected_experiments() {
    echo ""
    echo "Selected experiments to run:"
    
    if [ "$EXPERIMENTS" = "all" ]; then
        echo "  ✓ All 10 experiments"
    else
        should_run_experiment "1" && echo "  ✓ Experiment 1: Financial Q&A"
        should_run_experiment "2" && echo "  ✓ Experiment 2: FinGPT Sentiment"
        should_run_experiment "3" && echo "  ✓ Experiment 3: Finance Alpaca"
        should_run_experiment "4" && echo "  ✓ Experiment 4: FiQA"
        should_run_experiment "5" && echo "  ✓ Experiment 5: Twitter Sentiment"
        should_run_experiment "6" && echo "  ✓ Experiment 6: SEC Reports"
        should_run_experiment "7" && echo "  ✓ Experiment 7: News Articles"
        should_run_experiment "8" && echo "  ✓ Experiment 8: WikiText"
        should_run_experiment "9" && echo "  ✓ Experiment 9: Mixed Corpus (7 datasets)"
        should_run_experiment "10" && echo "  ✓ Experiment 10: Mixed-Wiki Corpus (8 datasets)"
    fi
}

# Function to check dataset availability
check_dataset() {
    local dataset_path=$1
    local dataset_name=$2
    
    if [ "$USE_LOCAL" = true ]; then
        if [ ! -d "$dataset_path" ]; then
            echo "ERROR: Local dataset not found: $dataset_name"
            echo "Path: $dataset_path"
            echo "Please run: python scripts/download_phase2b_datasets.py"
            return 1
        fi
        echo "Found local dataset: $dataset_name"
    else
        echo "Will download from HuggingFace: $dataset_name"
    fi
    return 0
}

# Function to calculate mixture rates based on strategy
calculate_mixture_rates() {
    local strategy=$1
    local include_wiki=${2:-""}  # Optional second parameter for wiki inclusion

    if [ "$include_wiki" = "wiki" ]; then
        # 8 datasets including WikiText
        # Dataset tokens (M): 0.70, 4.14, 8.46, 3.60, 0.28, 8.12, 197.38, 103.00
        # WikiText-103 has ~103M tokens
        case $strategy in
            50cap|50_cap)
                # 50% cap with square root scaling (including WikiText)
                # sqrt values: 0.837, 2.035, 2.909, 1.897, 0.529, 2.850, 14.049, 10.149
                # News Articles (39.7%) is below 50% cap, so no capping needed
                # Adjusted to sum exactly to 1.0 (reduced News from 0.399 to 0.397)
                MIXTURE_RATES="0.024 0.058 0.083 0.054 0.015 0.081 0.397 0.288"
                STRATEGY_DESC="50% cap with square root scaling (including WikiText)"
                ;;
            sqrt|square_root)
                # Pure square root scaling for 8 datasets
                # Same as 50cap since News doesn't exceed 50%
                # Adjusted to sum exactly to 1.0 (reduced News from 0.399 to 0.397)
                MIXTURE_RATES="0.024 0.058 0.083 0.054 0.015 0.081 0.397 0.288"
                STRATEGY_DESC="Pure square root scaling (including WikiText)"
                ;;
            proportional)
                # Proportional to dataset sizes for 8 datasets
                # Total tokens: 325.68M
                MIXTURE_RATES="0.002 0.013 0.026 0.011 0.001 0.025 0.606 0.316"
                STRATEGY_DESC="Proportional to dataset sizes (including WikiText)"
                ;;
            uniform)
                # Equal weights for all 8 datasets
                MIXTURE_RATES="0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125"
                STRATEGY_DESC="Uniform (equal) weights (including WikiText)"
                ;;
            custom)
                if [ -z "$CUSTOM_MIXTURE_RATES" ]; then
                    echo "ERROR: --custom-mixture-rates required when using custom strategy"
                    exit 1
                fi
                MIXTURE_RATES="$CUSTOM_MIXTURE_RATES"
                STRATEGY_DESC="Custom mixture rates (including WikiText)"
                ;;
            *)
                echo "ERROR: Unknown mixing strategy: $strategy"
                exit 1
                ;;
        esac
    else
        # Original 7 datasets (no WikiText)
        case $strategy in
            50cap|50_cap)
                # 50% News cap with square root scaling for others (recommended)
                MIXTURE_RATES="0.04 0.09 0.13 0.085 0.025 0.13 0.50"
                STRATEGY_DESC="50% News cap with square root scaling"
                ;;
            sqrt|square_root)
                # Pure square root scaling
                # Dataset tokens (M): 0.70, 4.14, 8.46, 3.60, 0.28, 8.12, 197.38
                # sqrt: 0.84, 2.03, 2.91, 1.90, 0.53, 2.85, 14.05 (sum=25.11)
                # normalized: 0.033, 0.081, 0.116, 0.076, 0.021, 0.113, 0.560
                MIXTURE_RATES="0.033 0.081 0.116 0.076 0.021 0.113 0.560"
                STRATEGY_DESC="Pure square root scaling"
                ;;
            proportional)
                # Proportional to dataset sizes (extreme imbalance)
                # Dataset tokens (M): 0.70, 4.14, 8.46, 3.60, 0.28, 8.12, 197.38 (sum=222.68)
                # normalized: 0.003, 0.019, 0.038, 0.016, 0.001, 0.036, 0.887
                MIXTURE_RATES="0.003 0.019 0.038 0.016 0.001 0.036 0.887"
                STRATEGY_DESC="Proportional to dataset sizes"
                ;;
            uniform)
                # Equal weights for all datasets
                MIXTURE_RATES="0.143 0.143 0.143 0.143 0.143 0.143 0.142"
                STRATEGY_DESC="Uniform (equal) weights"
                ;;
            custom)
            if [ -z "$CUSTOM_MIXTURE_RATES" ]; then
                echo "ERROR: --custom-mixture-rates required when using custom strategy"
                exit 1
            fi
            MIXTURE_RATES="$CUSTOM_MIXTURE_RATES"
            STRATEGY_DESC="Custom user-provided rates"
            # Validate custom rates sum to ~1.0
            SUM=$(echo $MIXTURE_RATES | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
            if (( $(awk "BEGIN {print ($SUM < 0.99 || $SUM > 1.01)}") )); then
                echo "ERROR: Custom mixture rates must sum to 1.0 (got $SUM)"
                echo "Rates provided: $MIXTURE_RATES"
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unknown mixing strategy: $strategy"
            echo "Valid options: 50cap, sqrt, proportional, uniform, custom"
            exit 1
            ;;
    esac
    fi  # Close the if [ "$include_wiki" = "wiki" ] block

    echo "Mixing strategy: $STRATEGY_DESC"
    echo "Mixture rates: $MIXTURE_RATES"
}

# ========================================
# DATASET PATH CONFIGURATION
# ========================================
if [ "$USE_LOCAL" = true ]; then
    echo "=========================================="
    echo "Mode: Using LOCAL cached datasets"
    echo "Cache directory: $DATASET_CACHE_DIR"
    echo "=========================================="
    
    # Check if cache directory exists
    if [ ! -d "$DATASET_CACHE_DIR" ]; then
        echo "ERROR: Dataset cache directory not found: $DATASET_CACHE_DIR"
        echo ""
        echo "To download datasets locally, run:"
        echo "  python scripts/download_phase2b_datasets.py"
        echo ""
        echo "Or use online mode (default):"
        echo "  $0"
        exit 1
    fi
    
    # Load dataset paths if available
    if [ -f "$DATASET_CACHE_DIR/dataset_paths.sh" ]; then
        source "$DATASET_CACHE_DIR/dataset_paths.sh"
        echo "Loaded dataset paths from cache"
    else
        echo "WARNING: dataset_paths.sh not found. Setting paths manually..."
        # Manual paths (update these based on your setup)
        FINANCIAL_QA_10K_LOCAL="$DATASET_CACHE_DIR/virattt_financial-qa-10K"
        FINGPT_SENTIMENT_TRAIN_LOCAL="$DATASET_CACHE_DIR/FinGPT_fingpt-sentiment-train"
        FINANCE_ALPACA_LOCAL="$DATASET_CACHE_DIR/gbharti_finance-alpaca"
        FIQA_LOCAL="$DATASET_CACHE_DIR/LLukas22_fiqa"
        TWITTER_FINANCIAL_NEWS_SENTIMENT_LOCAL="$DATASET_CACHE_DIR/zeroshot_twitter-financial-news-sentiment"
        SEC_REPORTS_LOCAL="$DATASET_CACHE_DIR/JanosAudran_financial-reports-sec"
        NEWS_ARTICLES_LOCAL="$DATASET_CACHE_DIR/ashraq_financial-news-articles"
        WIKITEXT_LOCAL="$DATASET_CACHE_DIR/wikitext"
    fi
    
    # Set dataset variables for local mode
    DATASET_FINANCIAL_QA="${FINANCIAL_QA_10K_LOCAL}"
    DATASET_FINGPT="${FINGPT_SENTIMENT_TRAIN_LOCAL}"
    DATASET_ALPACA="${FINANCE_ALPACA_LOCAL}"
    DATASET_FIQA="${FIQA_LOCAL}"
    DATASET_TWITTER="${TWITTER_FINANCIAL_NEWS_SENTIMENT_LOCAL}"
    DATASET_SEC="${SEC_REPORTS_LOCAL}"
    DATASET_NEWS="${NEWS_ARTICLES_LOCAL}"
    DATASET_WIKITEXT="${WIKITEXT_LOCAL}"
    
    echo ""
    echo "Local dataset paths:"
    echo "  Financial Q&A: ${DATASET_FINANCIAL_QA}"
    echo "  FinGPT: ${DATASET_FINGPT}"
    echo "  Alpaca: ${DATASET_ALPACA}"
    echo "  FiQA: ${DATASET_FIQA}"
    echo "  Twitter: ${DATASET_TWITTER}"
    echo "  SEC Reports: ${DATASET_SEC}"
    echo "  News Articles: ${DATASET_NEWS}"
    echo "  WikiText: ${DATASET_WIKITEXT}"
else
    echo "=========================================="
    echo "Mode: Downloading from HUGGINGFACE"
    echo "=========================================="
    
    # Use HuggingFace dataset names
    DATASET_FINANCIAL_QA="virattt/financial-qa-10K"
    DATASET_FINGPT="FinGPT/fingpt-sentiment-train"
    DATASET_ALPACA="gbharti/finance-alpaca"
    DATASET_FIQA="LLukas22/fiqa"
    DATASET_TWITTER="zeroshot/twitter-financial-news-sentiment"
    DATASET_SEC="JanosAudran/financial-reports-sec"
    DATASET_NEWS="ashraq/financial-news-articles"
    DATASET_WIKITEXT="wikitext"
fi

# ========================================
# TRAINING CONFIGURATION DISPLAY
# ========================================
echo ""
echo "=========================================="
echo "PHASE 2B: FINANCIAL DOMAIN PRETRAINING"
echo "=========================================="
echo "System:"
echo "  Python command: $PYTHON_CMD"
echo ""
echo "Model:"
echo "  Model: $MODEL"
echo "  Short name: $MODEL_SHORT"
echo "  Attention: $ATTN_IMPLEMENTATION"
echo ""
if [ "$USE_LORA" = true ]; then
    echo "LoRA Configuration:"
    echo "  Status: ENABLED"
    echo "  Rank: $PRETRAIN_LORA_RANK"
    echo "  Alpha: $PRETRAIN_LORA_ALPHA"
    echo "  Target modules: $LORA_TARGET_MODULES"
else
    echo "LoRA Configuration:"
    echo "  Status: DISABLED (full fine-tuning)"
fi
echo ""
echo "Training:"
echo "  Batch size: $BATCH_SIZE"
echo "  Max length: $MAX_LENGTH tokens"
echo "  Max steps: $MAX_STEPS"
echo "  Learning rate: $LEARNING_RATE"
echo "  LR scheduler: $LR_SCHEDULER"
echo "  Warmup steps: $WARMUP_STEPS"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Mixed precision: $PRECISION"
echo ""
echo "Packing:"
echo "  Enabled: $USE_PACKING"
if [ "$USE_PACKING" = true ]; then
    if [ -n "$PACKING_MAX_LENGTH" ]; then
        echo "  Max packed length: $PACKING_MAX_LENGTH tokens"
    else
        echo "  Max packed length: $MAX_LENGTH tokens (default)"
    fi
fi
echo ""
echo "Evaluation:"
echo "  Eval steps: $EVAL_STEPS"
echo "  Eval at start: $EVAL_ON_START"
echo "  Eval max batches: $EVAL_MAX_BATCHES"
echo "  Save steps: $SAVE_STEPS"
echo "  Save total limit: $SAVE_TOTAL_LIMIT"
echo "  Eval all datasets: $EVAL_ALL_DATASETS"
echo ""
echo "Output:"
echo "  Base directory: $BASE_DIR"
display_selected_experiments
echo "=========================================="

# Exit if dry-run
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "DRY RUN MODE - Configuration validated, exiting without running experiments."
    
    # If mixed experiments are selected, show the mixing strategy details
    if should_run_experiment "9" || should_run_experiment "mixed" || \
       should_run_experiment "10" || should_run_experiment "mixed-wiki"; then
        echo ""
        echo "Would use mixing strategy: $MIXING_STRATEGY"
        calculate_mixture_rates "$MIXING_STRATEGY"
    fi
    
    exit 0
fi

# Create base directory
mkdir -p $BASE_DIR

# ===========================================
# EXPERIMENT 1: Financial Q&A Pretraining
# ===========================================
if should_run_experiment "1" || should_run_experiment "financial_qa"; then
    echo ""
    echo "Experiment 1/10: Financial Q&A pretrain (4k steps, ~1B tokens)"
    echo "Dataset: virattt/financial-qa-10K (7K Q&A pairs from 10-K filings)"
    echo "Purpose: Learn financial terminology and Q&A patterns"
    
    check_dataset "$DATASET_FINANCIAL_QA" "financial-qa-10K" && \
    run_training_with_eval "$DATASET_FINANCIAL_QA" "None" "financial_qa"
else
    echo "Skipping Experiment 1: Financial Q&A (not selected)"
fi

# ===========================================
# EXPERIMENT 2: FinGPT Sentiment Pretraining
# ===========================================
if should_run_experiment "2" || should_run_experiment "fingpt"; then
    echo ""
    echo "Experiment 2/10: FinGPT sentiment pretrain (4k steps, ~1B tokens)"
    echo "Dataset: FinGPT/fingpt-sentiment-train (76K financial sentiment examples)"
    echo "Purpose: Learn financial sentiment patterns and market language"
    
    check_dataset "$DATASET_FINGPT" "fingpt-sentiment-train" && \
    run_training_with_eval "$DATASET_FINGPT" "None" "fingpt"
else
    echo "Skipping Experiment 2: FinGPT Sentiment (not selected)"
fi

# ===========================================
# EXPERIMENT 3: Finance Alpaca Pretraining
# ===========================================
if should_run_experiment "3" || should_run_experiment "alpaca"; then
    echo ""
    echo "Experiment 3/10: Finance Alpaca pretrain (4k steps, ~1B tokens)"
    echo "Dataset: gbharti/finance-alpaca (68K instruction-following examples)"
    echo "Purpose: Learn financial instruction following and explanations"
    
    check_dataset "$DATASET_ALPACA" "finance-alpaca" && \
    run_training_with_eval "$DATASET_ALPACA" "None" "alpaca"
else
    echo "Skipping Experiment 3: Finance Alpaca (not selected)"
fi

# ===========================================
# EXPERIMENT 4: FiQA Pretraining
# ===========================================
if should_run_experiment "4" || should_run_experiment "fiqa"; then
    echo ""
    echo "Experiment 4/10: FiQA pretrain (4k steps, ~1B tokens)"
    echo "Dataset: LLukas22/fiqa (Financial Q&A from various sources)"
    echo "Purpose: Diverse financial Q&A understanding"
    
    check_dataset "$DATASET_FIQA" "fiqa" && \
    run_training_with_eval "$DATASET_FIQA" "None" "fiqa"
else
    echo "Skipping Experiment 4: FiQA (not selected)"
fi

# ===========================================
# EXPERIMENT 5: Twitter Financial Sentiment
# ===========================================
if should_run_experiment "5" || should_run_experiment "twitter"; then
    echo ""
    echo "Experiment 5/10: Twitter financial sentiment pretrain (4k steps, ~1B tokens)"
    echo "Dataset: zeroshot/twitter-financial-news-sentiment (12K real-world tweets)"
    echo "Purpose: Learn informal financial discourse and social media patterns"
    
    check_dataset "$DATASET_TWITTER" "twitter-financial-news-sentiment" && \
    run_training_with_eval "$DATASET_TWITTER" "None" "twitter"
else
    echo "Skipping Experiment 5: Twitter Sentiment (not selected)"
fi

# ===========================================
# EXPERIMENT 6: SEC Financial Reports
# ===========================================
if should_run_experiment "6" || should_run_experiment "sec_reports"; then
    echo ""
    echo "Experiment 6/10: SEC Financial Reports pretrain (4k steps, ~1B tokens)"
    echo "Dataset: JanosAudran/financial-reports-sec (200K SEC filing reports)"
    echo "Purpose: Learn formal financial reporting language and regulatory disclosures"
    
    check_dataset "$DATASET_SEC" "financial-reports-sec" && \
    run_training_with_eval "$DATASET_SEC" "small_lite" "sec_reports"
else
    echo "Skipping Experiment 6: SEC Reports (not selected)"
fi

# ===========================================
# EXPERIMENT 7: Financial News Articles
# ===========================================
if should_run_experiment "7" || should_run_experiment "news_articles"; then
    echo ""
    echo "Experiment 7/10: Financial News Articles pretrain (4k steps, ~1B tokens)"
    echo "Dataset: ashraq/financial-news-articles (306K news articles)"
    echo "Purpose: Learn diverse financial news language and market coverage"
    
    check_dataset "$DATASET_NEWS" "financial-news-articles" && \
    run_training_with_eval "$DATASET_NEWS" "None" "news_articles"
else
    echo "Skipping Experiment 7: News Articles (not selected)"
fi

# ===========================================
# EXPERIMENT 8: WikiText Pretraining
# ===========================================
if should_run_experiment "8" || should_run_experiment "wikitext"; then
    echo ""
    echo "Experiment 8/10: WikiText pretrain (4k steps, ~1B tokens)"
    echo "Dataset: wikitext (Wikipedia articles)"
    echo "Purpose: Learn general knowledge and encyclopedic language"

    check_dataset "$DATASET_WIKITEXT" "wikitext" && \
    run_training_with_eval "$DATASET_WIKITEXT" "wikitext-103-v1" "wikitext"
else
    echo "Skipping Experiment 8: WikiText (not selected)"
fi

# =========================================
# EXPERIMENT 9: Mixed Financial Corpus (Original 7 datasets)
# ===========================================
if should_run_experiment "9" || should_run_experiment "mixed"; then
    # Add warning if running mixed without individual experiments
    if [ "$EXPERIMENTS" != "all" ]; then
        echo ""
        echo "WARNING: Mixed corpus experiment selected without all individual experiments."
        echo "For best results, consider running all experiments to compare performance."
        echo ""
    fi
    
    echo ""
    echo "Experiment 9/10: Mixed financial corpus pretrain (4k steps, ~1B tokens)"
    
    # Calculate mixture rates based on strategy
    calculate_mixture_rates "$MIXING_STRATEGY"
    
    # Parse mixture rates into array for display
    IFS=' ' read -ra RATES_ARRAY <<< "$MIXTURE_RATES"
    
    echo "Combines all 7 financial datasets with $STRATEGY_DESC:"
    echo "  - Financial Q&A: 7K examples (${RATES_ARRAY[0]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[0]} * 100}")%)"
    echo "  - FinGPT: 76K examples (${RATES_ARRAY[1]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[1]} * 100}")%)"
    echo "  - Finance Alpaca: 68K examples (${RATES_ARRAY[2]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[2]} * 100}")%)"
    echo "  - FiQA: 15K examples (${RATES_ARRAY[3]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[3]} * 100}")%)"
    echo "  - Twitter: 10K examples (${RATES_ARRAY[4]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[4]} * 100}")%)"
    echo "  - SEC Reports: 200K examples (${RATES_ARRAY[5]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[5]} * 100}")%)"
    echo "  - News Articles: 306K examples (${RATES_ARRAY[6]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[6]} * 100}")%)"
    echo "Multi-dataset evaluation: ENABLED"
    
    # Check all datasets exist for mixed corpus (local mode only)
    if [ "$USE_LOCAL" = true ]; then
        ALL_DATASETS_EXIST=true
        for dataset_path in "$DATASET_FINANCIAL_QA" "$DATASET_FINGPT" "$DATASET_ALPACA" "$DATASET_FIQA" "$DATASET_TWITTER" "$DATASET_SEC" "$DATASET_NEWS"; do
            if [ ! -d "$dataset_path" ]; then
                ALL_DATASETS_EXIST=false
                echo "Missing dataset for mixed corpus: $dataset_path"
            fi
        done
        
        if [ "$ALL_DATASETS_EXIST" = false ]; then
            echo "ERROR: Cannot run mixed corpus experiment - not all datasets available locally"
            echo "Please run: python scripts/download_phase2b_datasets.py"
            exit 1
        fi
    fi
    
    echo "Running mixed financial corpus experiment..."
    $PYTHON_CMD train.py --model $MODEL \
        --attn_implementation $ATTN_IMPLEMENTATION \
        --datasets "$DATASET_FINANCIAL_QA" "$DATASET_FINGPT" "$DATASET_ALPACA" "$DATASET_FIQA" "$DATASET_TWITTER" "$DATASET_SEC" "$DATASET_NEWS" \
        --dataset_configs None None None None None small_lite None \
        --mixture_rates $MIXTURE_RATES \
        --mode pretrain \
        --max_steps $MAX_STEPS $LORA_ARGS \
        --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
        --gradient_accumulation_steps $GRADIENT_ACCUM \
        $PACKING_ARGS \
        --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
        --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
        $MIXED_PRECISION \
        --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
        $EVAL_START_ARG \
        --save_steps $SAVE_STEPS --save_total_limit $SAVE_TOTAL_LIMIT \
        --save_strategy steps \
        $SEPARATE_EVAL $LOG_SPREAD \
        --output_dir $BASE_DIR/mixed_financial
else
    echo "Skipping Experiment 9: Mixed Corpus (not selected)"
fi

# =========================================
# EXPERIMENT 10: Mixed-Wiki Financial Corpus (8 datasets including WikiText)
# ===========================================
if should_run_experiment "10" || should_run_experiment "mixed-wiki"; then
    # Add warning if running mixed without individual experiments
    if [ "$EXPERIMENTS" != "all" ]; then
        echo ""
        echo "WARNING: Mixed-wiki corpus experiment selected without all individual experiments."
        echo "For best results, consider running all experiments to compare performance."
        echo ""
    fi

    echo ""
    echo "Experiment 10/10: Mixed-wiki financial corpus pretrain (4k steps, ~1B tokens)"

    # Calculate mixture rates based on strategy for 8 datasets
    calculate_mixture_rates "$MIXING_STRATEGY" "wiki"

    # Parse mixture rates into array for display
    IFS=' ' read -ra RATES_ARRAY <<< "$MIXTURE_RATES"

    echo "Combines all 8 datasets (7 financial + WikiText) with $STRATEGY_DESC:"
    echo "  - Financial Q&A: 7K examples (${RATES_ARRAY[0]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[0]} * 100}")%)"
    echo "  - FinGPT: 76K examples (${RATES_ARRAY[1]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[1]} * 100}")%)"
    echo "  - Finance Alpaca: 68K examples (${RATES_ARRAY[2]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[2]} * 100}")%)"
    echo "  - FiQA: 15K examples (${RATES_ARRAY[3]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[3]} * 100}")%)"
    echo "  - Twitter: 10K examples (${RATES_ARRAY[4]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[4]} * 100}")%)"
    echo "  - SEC Reports: 200K examples (${RATES_ARRAY[5]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[5]} * 100}")%)"
    echo "  - News Articles: 306K examples (${RATES_ARRAY[6]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[6]} * 100}")%)"
    echo "  - WikiText: 103K examples (${RATES_ARRAY[7]} = $(awk "BEGIN {printf \"%.0f\", ${RATES_ARRAY[7]} * 100}")%)"
    echo "Multi-dataset evaluation: ENABLED"

    # Check all datasets exist for mixed corpus (local mode only)
    if [ "$USE_LOCAL" = true ]; then
        ALL_DATASETS_EXIST=true
        for dataset_path in "$DATASET_FINANCIAL_QA" "$DATASET_FINGPT" "$DATASET_ALPACA" "$DATASET_FIQA" "$DATASET_TWITTER" "$DATASET_SEC" "$DATASET_NEWS" "$DATASET_WIKITEXT"; do
            if [ ! -d "$dataset_path" ]; then
                ALL_DATASETS_EXIST=false
                echo "Missing dataset for mixed-wiki corpus: $dataset_path"
            fi
        done

        if [ "$ALL_DATASETS_EXIST" = false ]; then
            echo "ERROR: Cannot run mixed-wiki corpus experiment - not all datasets available locally"
            echo "Please run: python scripts/download_phase2b_datasets.py"
            exit 1
        fi
    fi

    echo "Running mixed-wiki financial corpus experiment..."
    $PYTHON_CMD train.py --model $MODEL \
        --attn_implementation $ATTN_IMPLEMENTATION \
        --datasets "$DATASET_FINANCIAL_QA" "$DATASET_FINGPT" "$DATASET_ALPACA" "$DATASET_FIQA" "$DATASET_TWITTER" "$DATASET_SEC" "$DATASET_NEWS" "$DATASET_WIKITEXT" \
        --dataset_configs None None None None None small_lite None wikitext-103-v1 \
        --mixture_rates $MIXTURE_RATES \
        --mode pretrain \
        --max_steps $MAX_STEPS $LORA_ARGS \
        --batch_size $BATCH_SIZE --max_length $MAX_LENGTH \
        --gradient_accumulation_steps $GRADIENT_ACCUM \
        $PACKING_ARGS \
        --learning_rate $LEARNING_RATE --lr_scheduler_type $LR_SCHEDULER \
        --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
        $MIXED_PRECISION \
        --eval_steps $EVAL_STEPS --eval_max_batches $EVAL_MAX_BATCHES \
        $EVAL_START_ARG \
        --save_steps $SAVE_STEPS --save_total_limit $SAVE_TOTAL_LIMIT \
        --save_strategy steps \
        $SEPARATE_EVAL $LOG_SPREAD \
        --output_dir $BASE_DIR/mixed_wiki_financial
else
    echo "Skipping Experiment 10: Mixed-Wiki Corpus (not selected)"
fi

echo ""
echo "=========================================="
echo "PHASE 2B TRAINING COMPLETE!"
echo "=========================================="
echo "Checkpoints saved in: $BASE_DIR"
echo ""
echo "To use the best checkpoint for Phase 2C:"
echo "1. Find the best checkpoint:"
echo "   ./list_checkpoints.sh"
echo ""
echo "2. Update Phase 2C script with the checkpoint path:"
echo "   CHECKPOINT_PATH=\"$BASE_DIR/mixed_financial/checkpoints/checkpoint-4000\""
echo ""
echo "3. Run Phase 2C analytical training:"
echo "   ./phase2c_analytical_qwen3_0.6b.sh"
echo "=========================================="
