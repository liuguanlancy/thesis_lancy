#!/usr/bin/env python3
"""
Quick test script for Qwen3 + FlashAttention + Packing
Uses a tiny subset of WikiText-2 for immediate results
"""

import subprocess
import time
import sys

def run_training(description, command, output_dir):
    """Run a training command and measure time."""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(command)}")
    print("-"*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Check for success indicators
        if "Training completed" in result.stdout or "steps/s" in result.stdout:
            print(f"✓ SUCCESS in {elapsed_time:.1f} seconds")
        else:
            print(f"⚠ Completed in {elapsed_time:.1f} seconds")
            
        # Check for specific features
        if "SEQUENCE PACKING ENABLED" in result.stdout:
            print("  ✓ Packing activated")
        if "FlashAttention 2 enabled" in result.stdout:
            print("  ✓ FlashAttention activated")
        if "LoRA" in result.stdout and "trainable parameters" in result.stdout:
            print("  ✓ LoRA activated")
            
        return elapsed_time, True
        
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT after 120 seconds")
        return 120, False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return 0, False


def main():
    """Run quick tests of different configurations."""
    
    print("="*60)
    print("QUICK TEST: Qwen3 + Packing + FlashAttention")
    print("="*60)
    print("\nUsing Qwen3-0.6B (smallest model) for fast testing")
    print("Dataset: WikiText-2 (tiny subset)")
    print("Steps: 5 (just for testing)")
    
    # Base configuration
    base_config = [
        "python", "train.py",
        "--model", "Qwen/Qwen3-0.6B",
        "--dataset", "wikitext",
        "--dataset_config", "wikitext-2-raw-v1",
        "--mode", "pretrain",
        "--max_steps", "5",
        "--save_steps", "1000",  # Don't save checkpoints for quick test
    ]
    
    tests = [
        # Test 1: Baseline
        {
            "description": "Baseline (no optimizations)",
            "extra_args": [
                "--batch_size", "2",
                "--max_length", "256",
                "--output_dir", "./runs/quick_test_baseline"
            ]
        },
        
        # Test 2: With packing
        {
            "description": "With Packing (3-5x speedup expected)",
            "extra_args": [
                "--batch_size", "2",
                "--max_length", "1024",
                "--use_packing",
                "--output_dir", "./runs/quick_test_packing"
            ]
        },
        
        # Test 3: With FlashAttention (CUDA only)
        {
            "description": "With FlashAttention (CUDA only)",
            "extra_args": [
                "--batch_size", "2",
                "--max_length", "256",
                "--use_flash_attention",
                "--output_dir", "./runs/quick_test_flash"
            ]
        },
        
        # Test 4: Packing + FlashAttention
        {
            "description": "Packing + FlashAttention",
            "extra_args": [
                "--batch_size", "2",
                "--max_length", "1024",
                "--use_packing",
                "--use_flash_attention",
                "--output_dir", "./runs/quick_test_both"
            ]
        },
        
        # Test 5: Everything (Packing + FlashAttention + LoRA)
        {
            "description": "Fully Optimized (Packing + Flash + LoRA)",
            "extra_args": [
                "--batch_size", "4",
                "--max_length", "2048",
                "--use_packing",
                "--use_flash_attention",
                "--use_lora",
                "--lora_r", "4",
                "--lora_alpha", "8",
                "--output_dir", "./runs/quick_test_full"
            ]
        }
    ]
    
    results = []
    
    for test in tests:
        command = base_config + test["extra_args"]
        elapsed, success = run_training(
            test["description"],
            command,
            test["extra_args"][test["extra_args"].index("--output_dir") + 1]
        )
        results.append({
            "name": test["description"],
            "time": elapsed,
            "success": success
        })
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    baseline_time = results[0]["time"] if results[0]["success"] else None
    
    for result in results:
        if result["success"]:
            speedup = ""
            if baseline_time and result["time"] > 0:
                speedup_factor = baseline_time / result["time"]
                speedup = f" ({speedup_factor:.1f}x speedup)"
            print(f"✓ {result['name']:<40} {result['time']:>6.1f}s{speedup}")
        else:
            print(f"✗ {result['name']:<40} FAILED")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("• For WikiText/BookCorpus: Use --use_packing")
    print("• For CUDA GPUs: Add --use_flash_attention")
    print("• For memory efficiency: Add --use_lora")
    print("• For best performance: Combine all three!")
    print("\nFull command for BookCorpus:")
    print("python train.py --model Qwen/Qwen3-1.7B --dataset bookcorpus/bookcorpus \\")
    print("  --mode pretrain --max_steps 10000 --batch_size 8 --max_length 2048 \\")
    print("  --use_packing --use_flash_attention --use_lora --lora_r 16")


if __name__ == "__main__":
    main()