#!/usr/bin/env python3
"""
Test script for cosine learning rate scheduler with TensorBoard logging.
"""

import os
import sys

# Test with cosine scheduler
test_command = f"""
{sys.executable} train.py \
    --model gpt2 \
    --dataset wikitext --dataset_config wikitext-2-raw-v1 \
    --mode pretrain \
    --max_steps 50 \
    --batch_size 4 \
    --max_length 128 \
    --eval_steps 20 \
    --eval_max_batches 2 \
    --save_steps 50 \
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 10 \
    --weight_decay 0.01 \
    --output_dir ./tests/outputs/integration/cosine_scheduler_test
"""

print("=" * 60)
print("TESTING COSINE LEARNING RATE SCHEDULER")
print("=" * 60)
print("\nThis test will:")
print("1. Train GPT-2 on WikiText for 50 steps")
print("2. Use cosine LR scheduler with:")
print("   - Initial LR: 5e-4")
print("   - Warmup: 10 steps (20% of training)")
print("   - Weight decay: 0.01")
print("3. Log learning rate to TensorBoard")
print("\nCommand:")
print(test_command)
print("\n" + "=" * 60)

# Execute the test
os.system(test_command)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nTo view the learning rate curve in TensorBoard:")
print("  tensorboard --logdir ./tests/outputs/integration/cosine_scheduler_test/tensorboard")
print("\nIn TensorBoard, look for:")
print("  - train/learning_rate - Shows LR at each step")
print("  - train/loss - Training loss")
print("  - eval/loss - Evaluation loss")
print("\nExpected behavior:")
print("  - LR increases linearly from 0 to 5e-4 over first 10 steps (warmup)")
print("  - LR decreases following cosine curve from step 10 to 50")
print("  - Final LR should be close to 0")
print("=" * 60)