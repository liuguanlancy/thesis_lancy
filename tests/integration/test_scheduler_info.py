#!/usr/bin/env python3
"""
Test script to check what learning rate scheduler is used by default
in the HuggingFace TrainingArguments.
"""

from transformers import TrainingArguments
import inspect

# Create a basic TrainingArguments instance
args = TrainingArguments(
    output_dir="./tests/outputs/integration/scheduler_info",
    per_device_train_batch_size=8,
)

print("=" * 60)
print("DEFAULT LEARNING RATE SCHEDULER CONFIGURATION")
print("=" * 60)
print()

# Check default values
print("Learning Rate Settings:")
print(f"  learning_rate: {args.learning_rate}")
print(f"  lr_scheduler_type: {args.lr_scheduler_type}")
print(f"  warmup_steps: {args.warmup_steps}")
print(f"  warmup_ratio: {args.warmup_ratio}")
print()

print("Other Scheduler Settings:")
print(f"  adam_beta1: {args.adam_beta1}")
print(f"  adam_beta2: {args.adam_beta2}")
print(f"  adam_epsilon: {args.adam_epsilon}")
print(f"  weight_decay: {args.weight_decay}")
print(f"  max_grad_norm: {args.max_grad_norm}")
print()

print("Available scheduler types in transformers.SchedulerType:")
from transformers import SchedulerType
scheduler_types = [s.value for s in SchedulerType]
for i, sched in enumerate(scheduler_types, 1):
    print(f"  {i:2d}. {sched}")
print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Default scheduler: {args.lr_scheduler_type}")
print(f"Default learning rate: {args.learning_rate}")
print(f"Default warmup: {args.warmup_steps} steps or {args.warmup_ratio} ratio")
print()

print("To customize the scheduler, you would need to add these arguments")
print("to the parse_arguments() function in src/config/args.py:")
print()
print("  parser.add_argument('--learning_rate', type=float, default=5e-5)")
print("  parser.add_argument('--lr_scheduler_type', type=str, default='linear')")
print("  parser.add_argument('--warmup_steps', type=int, default=0)")
print("  parser.add_argument('--warmup_ratio', type=float, default=0.0)")
print("  parser.add_argument('--weight_decay', type=float, default=0.0)")
print()
print("And pass them to create_training_arguments() in src/training/utils.py")
print("=" * 60)