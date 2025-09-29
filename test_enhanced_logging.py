#!/usr/bin/env python3
"""
Test script for enhanced logging feature on M1 Max.
Tests with quick evaluation intervals and multiple datasets.
"""

import os
import sys
import subprocess

# Test configuration
CONFIG = {
    "model": "gpt2",  # Small model for quick testing
    "datasets": [
        ("stanfordnlp/imdb", None),
        ("glue", "sst2"),
        ("glue", "cola"),
    ],
    "eval_steps": 5,  # Evaluate every 5 steps
    "eval_max_batches": 5,  # Only evaluate 5 batches for speed
    "max_steps": 20,  # Train for 20 steps total
    "batch_size": 2,  # Small batch size for M1 Max
    "max_length": 128,  # Short sequences for speed
}

def run_single_dataset_test(dataset_name, dataset_config=None):
    """Run test with a single dataset."""
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "--model", CONFIG["model"],
        "--dataset", dataset_name,
        "--mode", "sft",
        "--eval_strategy", "steps",
        "--eval_steps", str(CONFIG["eval_steps"]),
        "--eval_max_batches", str(CONFIG["eval_max_batches"]),
        "--max_steps", str(CONFIG["max_steps"]),
        "--batch_size", str(CONFIG["batch_size"]),
        "--max_length", str(CONFIG["max_length"]),
        "--eval_on_start",  # Evaluate before training
        "--output_dir", f"./test_runs/enhanced_logging_test/{dataset_name.replace('/', '_')}",
    ]
    
    if dataset_config:
        cmd.extend(["--dataset_config", dataset_config])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the training
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    return result.returncode == 0

def run_multi_dataset_test():
    """Run test with multiple datasets for multi-eval."""
    print(f"\n{'='*60}")
    print("Testing: Multi-dataset evaluation")
    print(f"{'='*60}")
    
    # Build command for mixed datasets
    datasets = []
    configs = []
    for dataset, config in CONFIG["datasets"]:
        datasets.append(dataset)
        configs.append(config if config else "None")
    
    # Equal mixture rates
    mixture_rates = [f"{1.0/len(datasets):.2f}" for _ in datasets]
    
    cmd = [
        sys.executable,
        "train.py",
        "--model", CONFIG["model"],
        "--datasets"] + datasets + [
        "--dataset_configs"] + configs + [
        "--mixture_rates"] + mixture_rates + [
        "--mode", "sft",
        "--eval_strategy", "steps",
        "--eval_steps", str(CONFIG["eval_steps"]),
        "--eval_max_batches", str(CONFIG["eval_max_batches"]),
        "--max_steps", str(CONFIG["max_steps"]),
        "--batch_size", str(CONFIG["batch_size"]),
        "--max_length", str(CONFIG["max_length"]),
        "--eval_on_start",
        "--separate_mixture_eval",  # Evaluate on all datasets separately
        "--output_dir", "./test_runs/enhanced_logging_test/multi_dataset",
    ]
    
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Configs: {', '.join(configs)}")
    print(f"Mixture rates: {', '.join(mixture_rates)}")
    
    # Run the training
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    return result.returncode == 0

def check_log_files(output_dir):
    """Check if enhanced log files were created."""
    logs_dir = os.path.join(output_dir, "logs")
    
    files_to_check = [
        "training.log",
        "training_enhanced.log",
        "config.json",
        "config.txt"
    ]
    
    print(f"\nChecking log files in {logs_dir}:")
    for file in files_to_check:
        file_path = os.path.join(logs_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({size:,} bytes)")
            
            # Show sample from enhanced log
            if file == "training_enhanced.log" and size > 0:
                print("\n  Sample from enhanced log:")
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # Show first few lines and last few lines
                    if len(lines) > 20:
                        for line in lines[:10]:
                            print(f"    {line.rstrip()}")
                        print("    ...")
                        for line in lines[-10:]:
                            print(f"    {line.rstrip()}")
                    else:
                        for line in lines:
                            print(f"    {line.rstrip()}")
        else:
            print(f"  ✗ {file} not found")

def main():
    """Main test function."""
    print("="*60)
    print("ENHANCED LOGGING TEST ON M1 MAX")
    print("="*60)
    print(f"Model: {CONFIG['model']}")
    print(f"Eval every: {CONFIG['eval_steps']} steps")
    print(f"Eval batches: {CONFIG['eval_max_batches']}")
    print(f"Total steps: {CONFIG['max_steps']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Max length: {CONFIG['max_length']}")
    
    # Test 1: Single dataset
    print("\n" + "="*60)
    print("TEST 1: Single Dataset")
    print("="*60)
    
    dataset, config = CONFIG["datasets"][0]
    success = run_single_dataset_test(dataset, config)
    
    if success:
        output_dir = f"./test_runs/enhanced_logging_test/{dataset.replace('/', '_')}"
        check_log_files(output_dir)
    
    # Test 2: Multi-dataset evaluation
    print("\n" + "="*60)
    print("TEST 2: Multi-Dataset Evaluation")
    print("="*60)
    
    success = run_multi_dataset_test()
    
    if success:
        check_log_files("./test_runs/enhanced_logging_test/multi_dataset")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nCheck the following directories for detailed logs:")
    print("  - ./test_runs/enhanced_logging_test/*/logs/training_enhanced.log")
    print("\nTo monitor logs in real-time:")
    print("  tail -f test_runs/enhanced_logging_test/*/logs/training_enhanced.log")

if __name__ == "__main__":
    main()