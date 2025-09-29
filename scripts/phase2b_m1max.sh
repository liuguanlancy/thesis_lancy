#!/bin/bash
#
# M1 Max-optimized Phase 2B Financial Pretraining Script
# Wrapper for phase2b_financial_pretraining.sh with M1 Max-specific settings
#
# Features:
# - Supports both 0.6B and 1.7B Qwen3 models
# - 0.1B (100M) token budget PER EXPERIMENT
# - Optimized for M1 Max memory constraints
# - Flexible sequence length (256-1024)
# - Eval on start enabled
# - Eval every 500 steps
# - Save every 500 steps  
# - Multi-dataset evaluation enabled
# - Optimized for Apple Silicon (32GB RAM)
#

set -e  # Exit on error

# Set environment variable to trust remote code for HuggingFace datasets
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# TOKEN BUDGET CONFIGURATION (0.1B/100M tokens per experiment)
TOKEN_BUDGET=100000000

# Model configuration (default to 0.6B)
MODEL_SIZE="0.6b"  # Can be "0.6b" or "1.7b"

# M1 Max-specific defaults (optimized for memory efficiency)
M1MAX_BATCH_SIZE=8          # Conservative batch for M1 Max (0.6B model)
M1MAX_EVAL_STEPS=200        # Evaluate every 200 steps (better for 100M budget)
M1MAX_SAVE_STEPS=200        # Save every 200 steps (better for 100M budget)
M1MAX_SAVE_TOTAL_LIMIT=-1   # Keep all checkpoints by default
M1MAX_EVAL_MAX_BATCHES=50   # Reduced evaluation for faster cycles
M1MAX_MAX_LENGTH=512        # Default 512 for balance (can go 256-1024)
M1MAX_BF16=false            # M1 Max works better with FP32
M1MAX_USE_PACKING=true      # Enable packing for efficiency
M1MAX_DEVICE="mps"          # Use Metal Performance Shaders
M1MAX_ATTN_IMPLEMENTATION="eager"  # Use eager attention to avoid SDPA NaN bug on MPS

# Calculate max steps based on token budget
# Tokens per step = batch_size * max_length (regardless of packing)
# Packing improves efficiency by reducing padding waste, not by processing more tokens
calculate_max_steps() {
    local batch_size=$1
    local max_length=$2
    
    # Always batch_size * max_length tokens per step
    local tokens_per_step=$((batch_size * max_length))
    
    # Note: Packing doesn't change tokens/step, it just ensures all tokens are useful content
    # Without packing, many tokens would be padding (wasted)
    
    local max_steps=$((TOKEN_BUDGET / tokens_per_step))
    echo $max_steps
}

# Calculate optimal batch size for given sequence length on M1 Max
get_optimal_batch_config() {
    local seq_length=$1
    local exp=$2
    
    # For M1 Max 32GB with MPS
    if [ $seq_length -ge 1024 ]; then
        # Long sequences - small batch
        echo "4"  # batch_size
    elif [ $seq_length -ge 768 ]; then
        # Medium-long sequences
        echo "6"  # batch_size
    elif [ $seq_length -ge 512 ]; then
        # Medium sequences - balanced
        echo "8"   # batch_size
    elif [ $seq_length -ge 256 ]; then
        # Short sequences - larger batch
        echo "16"   # batch_size
    else
        # Very short sequences
        echo "32"  # batch_size
    fi
}

# Parse arguments
EXPERIMENT=""
STRATEGY=""
MAX_STEPS=""
CUSTOM_ARGS=""
DRY_RUN=false
LOCAL_DATA=false
NO_PACKING=false
CUSTOM_BATCH_SIZE=""
NO_EVAL_ON_START=false
CUSTOM_EVAL_STEPS=""
CUSTOM_EVAL_BATCHES=""
CUSTOM_SAVE_STEPS=""
CUSTOM_SAVE_TOTAL_LIMIT=""
CUSTOM_WARMUP_STEPS=""

print_usage() {
    echo "M1 Max Phase 2B Financial Pretraining Script"
    echo "100M Token Budget Per Experiment"
    echo ""
    echo "Usage: $0 --experiments <exp> [options]"
    echo ""
    echo "Required:"
    echo "  --experiments <exp>     Experiment(s) to run:"
    echo "                         Single: 1-8, mixed, mixed-wiki, or all"
    echo "                         Multiple: \"1 3 5\" or \"1 3 mixed\" (runs sequentially)"
    echo ""
    echo "Optional:"
    echo "  --model <size>         Model size: 0.6b, 1.7b, or 4b (default: 0.6b)"
    echo "  --strategy <name>       Mixing strategy for mixed experiments:"
    echo "                         50cap, sqrt, proportional, uniform, custom"
    echo "                         (default: 50cap)"
    echo "  --max-steps <n>        Override calculated max steps"
    echo "  --batch-size <n>       Override M1 Max default batch size (8 for 0.6b, 4 for 1.7b)"
    echo "  --max-length <n>       Override default max length (512, range: 256-1024)"
    echo "  --local                Use locally cached datasets"
    echo "  --no-packing           Disable sequence packing"
    echo "  --dry-run              Print command without executing"
    echo "  --no-eval-all          Disable multi-dataset evaluation"
    echo "  --no-eval-on-start     Disable evaluation before training"
    echo "  --eval-steps <n>       Evaluation frequency (default: 200)"
    echo "  --eval-batches <n>     Max batches per dataset eval (default: 50)"
    echo "  --save-steps <n>       Save checkpoint frequency (default: 200)"
    echo "  --save-total-limit <n> Max checkpoints to keep (-1 for all, default: -1)"
    echo "  --warmup-steps <n>     Warmup steps (default: auto-calculated as 10% of max)"
    echo ""
    echo "Token Budget: 100M tokens PER EXPERIMENT"
    echo ""
    echo "Default Configuration:"
    echo "  - Sequence length: 512 tokens (balanced for M1 Max)"
    echo "  - Batch size: 8 (conservative for stability)"
    echo "  - Tokens per step: 8 × 512 = 4,096"
    echo "  - Max steps: 100M ÷ 4,096 ≈ 24,414 steps"
    echo "  - With packing: All tokens are useful content (no padding waste)"
    echo ""
    echo "Alternative Configurations:"
    echo "  seq_len=256, BS=16 → 4,096 tok/step → 24,414 steps"
    echo "  seq_len=512, BS=8 → 4,096 tok/step → 24,414 steps"
    echo "  seq_len=768, BS=6 → 4,608 tok/step → 21,701 steps"
    echo "  seq_len=1024, BS=4 → 4,096 tok/step → 24,414 steps"
    echo ""
    echo "Note: With packing enabled, all tokens are actual content."
    echo "      Without packing, many tokens would be padding (wasted)."
    echo ""
    echo "Dataset Sizes (for context):"
    echo "  1. Financial Q&A:     7.1K samples"
    echo "  2. FinGPT Sentiment:  76.8K samples"
    echo "  3. Finance Alpaca:    68.9K samples"
    echo "  4. FiQA:             17.4K samples"
    echo "  5. Twitter:          1.1K samples (will repeat ~177x @ 512 seq)"
    echo "  6. SEC Reports:      54.3K samples (80M tokens)"
    echo "  7. News Articles:    300K samples (197M tokens)"
    echo "  8. WikiText:         103K samples (Wikipedia articles)"
    echo ""
    echo "Examples:"
    echo "  # Train single dataset with 100M tokens"
    echo "  $0 --experiments 1"
    echo ""
    echo "  # Mixed corpus training (7 datasets) with 100M tokens"
    echo "  $0 --experiments mixed --strategy 50cap"
    echo ""
    echo "  # Mixed corpus with WikiText (8 datasets) with 100M tokens"
    echo "  $0 --experiments mixed-wiki --strategy 50cap"
    echo ""
    echo "  # Run multiple experiments sequentially (5 min delay between)"
    echo "  $0 --experiments \"1 3 mixed\" --save-steps 500 --eval-steps 500"
    echo ""
    echo "  # Shorter sequences for faster training"
    echo "  $0 --experiments 2 --max-length 256"
    echo ""
    echo "  # Longer sequences for better context"
    echo "  $0 --experiments 6 --max-length 1024"
    echo ""
    echo "  # Quick test (uses less than 100M)"
    echo "  $0 --experiments 2 --max-steps 1000"
    echo ""
    echo "M1 Max Settings:"
    echo "  - Device: MPS (Metal Performance Shaders)"
    echo "  - Default sequence length: 512 tokens"
    echo "  - Batch size: 8 (optimized for M1 Max memory)"
    echo "  - Sequence packing: ENABLED (2.5x efficiency)"
    echo "  - Mixed precision: DISABLED (FP32 for stability)"
    echo "  - Attention: EAGER (avoids SDPA NaN bug on MPS)"
    echo "  - Eval on start: ENABLED"
    echo "  - Eval frequency: Every 200 steps"
    echo "  - Save frequency: Every 200 steps"
    echo "  - Multi-dataset eval: ENABLED by default"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --experiments|--experiment)
            # Support both single and multiple experiments
            shift
            EXPERIMENT=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                EXPERIMENT="$EXPERIMENT $1"
                shift
            done
            EXPERIMENT="${EXPERIMENT# }"  # Remove leading space
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            CUSTOM_BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            M1MAX_MAX_LENGTH="$2"
            if [ "$M1MAX_MAX_LENGTH" -gt 1024 ]; then
                echo "Warning: Maximum sequence length for M1 Max is 1024. Setting to 1024."
                M1MAX_MAX_LENGTH=1024
            elif [ "$M1MAX_MAX_LENGTH" -lt 256 ]; then
                echo "Warning: Minimum sequence length is 256. Setting to 256."
                M1MAX_MAX_LENGTH=256
            fi
            shift 2
            ;;
        --local)
            LOCAL_DATA=true
            shift
            ;;
        --no-packing)
            NO_PACKING=true
            M1MAX_USE_PACKING=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-eval-all)
            CUSTOM_ARGS="$CUSTOM_ARGS --no-eval-all-datasets"
            shift
            ;;
        --no-eval-on-start)
            NO_EVAL_ON_START=true
            shift
            ;;
        --eval-steps)
            CUSTOM_EVAL_STEPS="$2"
            shift 2
            ;;
        --eval-batches)
            CUSTOM_EVAL_BATCHES="$2"
            shift 2
            ;;
        --save-steps)
            CUSTOM_SAVE_STEPS="$2"
            shift 2
            ;;
        --save-total-limit)
            CUSTOM_SAVE_TOTAL_LIMIT="$2"
            shift 2
            ;;
        --warmup-steps)
            CUSTOM_WARMUP_STEPS="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$EXPERIMENT" ]; then
    echo "Error: --experiments is required"
    print_usage
    exit 1
fi

# Convert EXPERIMENT string to array for processing
EXPERIMENT_LIST=($EXPERIMENT)
NUM_EXPERIMENTS=${#EXPERIMENT_LIST[@]}

# Validate and set model-specific configuration
case $MODEL_SIZE in
    "0.6b"|"0.6B")
        MODEL_NAME="Qwen/Qwen3-0.6B-Base"
        MODEL_SHORT="qwen3_0.6b"
        # Default batch size for 0.6B model
        if [ -z "$CUSTOM_BATCH_SIZE" ]; then
            M1MAX_BATCH_SIZE=8
        fi
        ;;
    "1.7b"|"1.7B")
        MODEL_NAME="Qwen/Qwen3-1.7B-Base"
        MODEL_SHORT="qwen3_1.7b"
        # Reduce batch size for 1.7B model (uses more memory)
        if [ -z "$CUSTOM_BATCH_SIZE" ]; then
            M1MAX_BATCH_SIZE=4
        fi
        ;;
    "4b"|"4B")
        MODEL_NAME="Qwen/Qwen3-4B-Base"
        MODEL_SHORT="qwen3_4b"
        # Use same batch size as 0.6B model
        if [ -z "$CUSTOM_BATCH_SIZE" ]; then
            M1MAX_BATCH_SIZE=8
        fi
        ;;
    *)
        echo "Error: Invalid model size '$MODEL_SIZE'. Must be '0.6b', '1.7b', or '4b'"
        exit 1
        ;;
esac

# Check if running multiple experiments
if [ $NUM_EXPERIMENTS -gt 1 ]; then
    echo "=========================================="
    echo "Sequential Experiment Execution Plan"
    echo "=========================================="
    echo "Model: $MODEL_NAME"
    echo "Experiments to run: ${EXPERIMENT_LIST[@]}"
    echo "Total experiments: $NUM_EXPERIMENTS"
    echo "Delay between experiments: 5 minutes"
    echo ""
fi

# Function to run a single experiment
run_single_experiment() {
    local exp=$1
    local exp_num=$2
    local total=$3
    
    # Adjust batch size - custom overrides everything
    if [ ! -z "$CUSTOM_BATCH_SIZE" ]; then
        M1MAX_BATCH_SIZE=$CUSTOM_BATCH_SIZE
    fi
    # Note: Model-specific batch sizes are already set above in the model validation section

    # Override eval settings if provided
    if [ ! -z "$CUSTOM_EVAL_STEPS" ]; then
        M1MAX_EVAL_STEPS=$CUSTOM_EVAL_STEPS
    fi

    if [ ! -z "$CUSTOM_EVAL_BATCHES" ]; then
        M1MAX_EVAL_MAX_BATCHES=$CUSTOM_EVAL_BATCHES
    fi

    if [ ! -z "$CUSTOM_SAVE_STEPS" ]; then
        M1MAX_SAVE_STEPS=$CUSTOM_SAVE_STEPS
    fi
    
    if [ ! -z "$CUSTOM_SAVE_TOTAL_LIMIT" ]; then
        M1MAX_SAVE_TOTAL_LIMIT=$CUSTOM_SAVE_TOTAL_LIMIT
    fi

    # Calculate max steps if not provided
    if [ -z "$MAX_STEPS" ]; then
        if [ "$exp" = "all" ]; then
            echo "Note: Running 'all' experiments will use 700M tokens total (100M per experiment)"
            echo "Consider running experiments individually to monitor progress."
            # Don't exit, allow running all
        fi
        
        # Calculate based on 100M budget per experiment
        MAX_STEPS=$(calculate_max_steps $M1MAX_BATCH_SIZE $M1MAX_MAX_LENGTH)
    fi

    # Calculate token consumption
    TOKENS_PER_STEP=$((M1MAX_BATCH_SIZE * M1MAX_MAX_LENGTH))
    # Note: Packing doesn't change tokens per step, just makes them all useful content
    TOTAL_TOKENS=$((TOKENS_PER_STEP * MAX_STEPS))
    TOKEN_PERCENT=$((TOTAL_TOKENS * 100 / TOKEN_BUDGET))

    # Build command for phase2b_financial_pretraining.sh
    CMD="$SCRIPT_DIR/phase2b_financial_pretraining.sh"
    CMD="$CMD --experiments $exp"
    CMD="$CMD --model $MODEL_NAME"
    CMD="$CMD --model-short $MODEL_SHORT"

    # Add M1 Max-specific settings
    CMD="$CMD --batch-size $M1MAX_BATCH_SIZE"
    CMD="$CMD --eval-steps $M1MAX_EVAL_STEPS"
    CMD="$CMD --save-steps $M1MAX_SAVE_STEPS"
    CMD="$CMD --save-total-limit $M1MAX_SAVE_TOTAL_LIMIT"
    CMD="$CMD --eval-batches $M1MAX_EVAL_MAX_BATCHES"
    CMD="$CMD --max-length $M1MAX_MAX_LENGTH"
    CMD="$CMD --max-steps $MAX_STEPS"

    # Add warmup steps if specified or calculate automatically
    if [ ! -z "$CUSTOM_WARMUP_STEPS" ]; then
        CMD="$CMD --warmup-steps $CUSTOM_WARMUP_STEPS"
    else
        # Calculate 10% of max steps for warmup
        CALCULATED_WARMUP=$((MAX_STEPS / 10))
        if [ "$CALCULATED_WARMUP" -lt 1 ]; then
            CALCULATED_WARMUP=1
        fi
        CMD="$CMD --warmup-steps $CALCULATED_WARMUP"
    fi

    # Add eval-on-start unless disabled
    if [ "$NO_EVAL_ON_START" = false ]; then
        CMD="$CMD --eval-on-start"  # Evaluate before training starts
    fi

    # Always enable these features for M1 Max
    # Only add --eval-all-datasets if --no-eval-all was not specified
    if [[ "$CUSTOM_ARGS" != *"--no-eval-all-datasets"* ]]; then
        CMD="$CMD --eval-all-datasets"  # Multi-dataset evaluation
    fi
    CMD="$CMD --precision fp32"     # Use FP32 for M1 Max stability
    CMD="$CMD --attn-implementation $M1MAX_ATTN_IMPLEMENTATION"  # Eager attention for MPS

    # Add packing unless disabled
    if [ "$M1MAX_USE_PACKING" = true ]; then
        CMD="$CMD --use-packing"
    fi

    # Add optional arguments
    if [ ! -z "$STRATEGY" ] && [ "$exp" = "mixed" ]; then
        CMD="$CMD --mixing-strategy $STRATEGY"
    fi

    if [ "$LOCAL_DATA" = true ]; then
        CMD="$CMD --local"
    fi

    # Add any custom arguments
    if [ ! -z "$CUSTOM_ARGS" ]; then
        CMD="$CMD $CUSTOM_ARGS"
    fi

    # Print header for sequential runs
    if [ $total -gt 1 ]; then
        echo ""
        echo "=========================================="
        echo "[$exp_num/$total] Experiment: $exp"
        echo "=========================================="
    fi

    # Print configuration
    echo "=========================================="
    echo "M1 Max Phase 2B Financial Pretraining"
    echo "100M Token Budget Per Experiment"
    echo "=========================================="
    echo "Model: $MODEL_NAME"
    echo "Experiment: $exp"
    if [ ! -z "$STRATEGY" ] && [ "$exp" = "mixed" ]; then
        echo "Mixing Strategy: $STRATEGY"
    fi
    echo "Device: MPS (Metal Performance Shaders)"
    echo "Batch Size: $M1MAX_BATCH_SIZE"
    echo "Max Length: $M1MAX_MAX_LENGTH tokens"
    echo "Sequence Packing: $M1MAX_USE_PACKING"
    echo "Max Steps: $MAX_STEPS"
    echo "Tokens per Step: ~$TOKENS_PER_STEP"
    echo "Total Token Budget: ~${TOTAL_TOKENS} ($TOKEN_PERCENT% of 100M)"
    echo "Eval Steps: $M1MAX_EVAL_STEPS"
    echo "Save Steps: $M1MAX_SAVE_STEPS"
    echo "Mixed Precision: Disabled (FP32)"
    echo "Attention: Eager (MPS-safe)"
    if [ "$NO_EVAL_ON_START" = false ]; then
        echo "Eval on Start: Yes"
    else
        echo "Eval on Start: No"
    fi
    echo "Multi-Dataset Eval: Yes"
    if [ "$LOCAL_DATA" = true ]; then
        echo "Using Local Datasets: Yes"
    fi

    # Show dataset-specific information
    case $exp in
        1) echo "Dataset: Financial Q&A (7.1K samples, ~27.5 epochs @ 512)" ;;
        2) echo "Dataset: FinGPT Sentiment (76.8K samples, ~2.5 epochs @ 512)" ;;
        3) echo "Dataset: Finance Alpaca (68.9K samples, ~2.8 epochs @ 512)" ;;
        4) echo "Dataset: FiQA (17.4K samples, ~11.2 epochs @ 512)" ;;
        5) echo "Dataset: Twitter (1.1K samples, ~177 epochs @ 512)" ;;
        6) echo "Dataset: SEC Reports (54.3K samples, 80M tokens)" ;;
        7) echo "Dataset: News Articles (300K samples, 197M tokens)" ;;
        8) echo "Dataset: WikiText (103K samples, Wikipedia articles)" ;;
        mixed)
            echo "Dataset: Mixed corpus (7 financial datasets, 207M total tokens)"
            echo ""
            echo "Mixture Details (50cap strategy):"
            echo "  1. Financial Q&A:    ~3.4% (3.5M tokens)"
            echo "  2. FinGPT Sentiment: ~19.1% (19.1M tokens)"
            echo "  3. Finance Alpaca:   ~17.2% (17.2M tokens)"
            echo "  4. FiQA:             ~4.3% (4.3M tokens)"
            echo "  5. Twitter:          ~0.3% (0.3M tokens)"
            echo "  6. SEC Reports:      ~19.4% (19.4M tokens)"
            echo "  7. News Articles:    ~36.2% (36.2M tokens, capped at 50%)"
            ;;
        mixed-wiki)
            echo "Dataset: Mixed-wiki corpus (8 datasets including WikiText)"
            echo ""
            echo "Mixture Details (50cap strategy with WikiText):"
            echo "  1. Financial Q&A:    2.4% (2.4M tokens)"
            echo "  2. FinGPT Sentiment: 5.8% (5.8M tokens)"
            echo "  3. Finance Alpaca:   8.3% (8.3M tokens)"
            echo "  4. FiQA:             5.4% (5.4M tokens)"
            echo "  5. Twitter:          1.5% (1.5M tokens)"
            echo "  6. SEC Reports:      8.1% (8.1M tokens)"
            echo "  7. News Articles:    39.9% (39.9M tokens)"
            echo "  8. WikiText:         28.8% (28.8M tokens)"
            echo "  Note: News (39.9%) below 50% cap, no capping needed"
            ;;
    esac

    echo "=========================================="
    echo ""

    # Estimate training time and resource usage
    echo "Resource Requirements & Estimates:"
    echo "------------------------------------------"

    # Calculate estimated time based on device and configuration
    if [ "$M1MAX_MAX_LENGTH" -le 512 ]; then
        TOKENS_PER_SEC=450  # M1 Max typical throughput at 512 seq length
    else
        TOKENS_PER_SEC=300  # Slower for longer sequences
    fi

    ESTIMATED_TIME_SECONDS=$((TOTAL_TOKENS / TOKENS_PER_SEC))
    ESTIMATED_TIME_HOURS=$(echo "scale=1; $ESTIMATED_TIME_SECONDS / 3600" | bc 2>/dev/null || echo "N/A")

    # Memory estimation based on batch size and sequence length
    MEMORY_GB_ESTIMATE=$(echo "scale=1; ($M1MAX_BATCH_SIZE * $M1MAX_MAX_LENGTH) / 1000 + 8" | bc 2>/dev/null || echo "12-16")

    echo "Estimated Memory Usage: ~${MEMORY_GB_ESTIMATE}GB"
    echo "Estimated Training Time: ~${ESTIMATED_TIME_HOURS} hours"
    echo "Token Throughput: ~${TOKENS_PER_SEC} tokens/sec"
    echo "Checkpoint Size: ~2.5GB per checkpoint"
    echo "Total Checkpoints: ~$((MAX_STEPS / M1MAX_SAVE_STEPS + 1))"
    echo ""
    echo "=========================================="
    echo ""

    # Warn if configuration seems off
    if [ $TOKEN_PERCENT -gt 105 ]; then
        echo "WARNING: This configuration will exceed 100M token budget by ${TOKEN_PERCENT}%!"
        echo "Consider reducing max_steps, batch_size, or max_length."
        echo ""
    elif [ $TOKEN_PERCENT -lt 95 ]; then
        echo "Note: Using ${TOKEN_PERCENT}% of 100M budget. You can increase max_steps if needed."
        echo ""
    fi

    # M1 Max specific warnings
    if [ "$M1MAX_MAX_LENGTH" -ge 1024 ]; then
        echo "Note: Using max sequence length (1024). Monitor memory usage carefully."
        echo ""
    fi

    # Execute or print command
    if [ "$DRY_RUN" = true ]; then
        echo "Dry run - would execute:"
        echo "$CMD"
    else
        echo "Executing command..."
        echo ""
        $CMD
        
        # Check exit status
        if [ $? -ne 0 ]; then
            echo "ERROR: Experiment $exp failed!"
            exit 1
        fi
    fi
}

# Main execution loop
if [ $NUM_EXPERIMENTS -eq 1 ]; then
    # Single experiment - run directly
    run_single_experiment "${EXPERIMENT_LIST[0]}" 1 1
else
    # Multiple experiments - run sequentially with delays
    EXP_NUM=1
    for EXP in "${EXPERIMENT_LIST[@]}"; do
        run_single_experiment "$EXP" "$EXP_NUM" "$NUM_EXPERIMENTS"
        
        # Add delay between experiments (except after the last one)
        if [ $EXP_NUM -lt $NUM_EXPERIMENTS ]; then
            if [ "$DRY_RUN" = true ]; then
                echo ""
                echo "=========================================="
                echo "Would wait 5 minutes before next experiment"
                echo "=========================================="
            else
                echo ""
                echo "=========================================="
                echo "Waiting 5 minutes before next experiment..."
                # macOS compatible date command
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    echo "Next experiment will start at: $(date -v +5M +"%Y-%m-%d %H:%M:%S")"
                else
                    echo "Next experiment will start at: $(date -d "+5 minutes" +"%Y-%m-%d %H:%M:%S")"
                fi
                echo "(Press Ctrl+C to cancel)"
                echo "=========================================="
                sleep 300  # 5 minutes = 300 seconds
            fi
        fi
        
        EXP_NUM=$((EXP_NUM + 1))
    done
    
    echo ""
    echo "=========================================="
    echo "All experiments completed successfully!"
    echo "Total experiments run: $NUM_EXPERIMENTS"
    echo "=========================================="
fi