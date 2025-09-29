#!/usr/bin/env python3
"""
Test script for multi-dataset evaluation functionality.
Tests the separate evaluation of each dataset in a mixture.
"""

import sys
import os

# Test with a simple 2-dataset mixture
test_command = """
python src/main.py \
    --model gpt2 \
    --datasets stanfordnlp/imdb glue \
    --dataset_configs None sst2 \
    --mixture_rates 0.6 0.4 \
    --mode sft \
    --batch_size 4 \
    --max_length 128 \
    --max_steps 10 \
    --eval_steps 5 \
    --eval_max_batches 2 \
    --save_steps 10 \
    --output_dir ./tests/outputs/integration/multi_eval_test \
    --separate_mixture_eval \
    --log_eval_spread
"""

print("="*60)
print("TESTING MULTI-DATASET EVALUATION")
print("="*60)
print("\nThis test will:")
print("1. Load IMDB and GLUE SST-2 datasets")
print("2. Mix them with 60%/40% rates for training")
print("3. Keep evaluation sets separate")
print("4. Train for 10 steps with eval every 5 steps")
print("5. Show individual metrics for each dataset")
print("\nCommand:")
print(test_command)
print("\n" + "="*60)

# Execute the test
os.system(test_command)

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nCheck the following:")
print("1. Console output should show separate eval metrics for 'imdb' and 'glue_sst2'")
print("2. Logs in ./tests/outputs/integration/multi_eval_test/logs/training.log")
print("3. TensorBoard metrics: tensorboard --logdir ./tests/outputs/integration/multi_eval_test/tensorboard")
print("\nExpected metric structure in TensorBoard:")
print("  eval/imdb/loss")
print("  eval/imdb/perplexity")
print("  eval/glue_sst2/loss")
print("  eval/glue_sst2/perplexity")
print("  eval/average/loss")
print("  eval/average/perplexity")
print("  eval/spread/max_min_diff")
print("  eval/spread/std_dev")