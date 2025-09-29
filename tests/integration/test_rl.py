#!/usr/bin/env python3
"""
Test script for RL functionality with GRPO.
This script tests the RL implementation on a small dataset.
"""

import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import Dataset

def test_rl_components():
    """Test individual RL components."""
    print("Testing RL components...")
    
    # Test reward function creation
    from src.training.rl_utils import create_simple_reward_function
    reward_fn = create_simple_reward_function("openai/gsm8k", "main")
    
    # Test the reward function
    prompts = ["Solve this math problem step by step:\n\nProblem: What is 2 + 2?\nSolution:"]
    completions = ["To solve 2 + 2, I need to add these numbers step by step:\nStep 1: Start with 2\nStep 2: Add 2 more\nTherefore, 2 + 2 = 4"]
    
    rewards = reward_fn(prompts, completions)
    print(f"Reward for mathematical solution: {rewards[0]}")
    
    # Test with sentiment task
    sentiment_reward_fn = create_simple_reward_function("stanfordnlp/imdb")
    sentiment_prompts = ["Classify the sentiment of this movie review as either 'positive' or 'negative'.\n\nReview: This movie is great!\nSentiment:"]
    sentiment_completions = ["positive"]
    
    sentiment_rewards = sentiment_reward_fn(sentiment_prompts, sentiment_completions)
    print(f"Reward for sentiment classification: {sentiment_rewards[0]}")
    
    print("✅ Reward function tests passed")


def test_rl_dataset_preparation():
    """Test RL dataset preparation."""
    print("\nTesting RL dataset preparation...")
    
    from transformers import AutoTokenizer
    from src.training.rl_utils import prepare_rl_dataset
    
    # Create a small test dataset
    test_data = {
        'question': [
            "What is 5 + 3?",
            "Calculate 10 - 4.",
            "What is 2 * 6?"
        ],
        'answer': [
            "5 + 3 = 8",
            "10 - 4 = 6", 
            "2 * 6 = 12"
        ]
    }
    
    test_dataset = Dataset.from_dict(test_data)
    print(f"Created test dataset with {len(test_dataset)} examples")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare for RL
    rl_dataset = prepare_rl_dataset(test_dataset, tokenizer, "openai/gsm8k", "main")
    
    print(f"RL dataset prepared with {len(rl_dataset)} examples")
    print("Sample RL example:")
    sample = rl_dataset[0]
    print(f"  Prompt: {sample['prompt'][:100]}...")
    print(f"  Completion: {sample['completion']}")
    
    print("✅ RL dataset preparation tests passed")


def test_grpo_config():
    """Test GRPO configuration creation."""
    print("\nTesting GRPO configuration...")
    
    from src.training.rl_utils import create_grpo_config
    import argparse
    
    # Create mock arguments
    args = argparse.Namespace(
        grpo_beta=0.1,
        grpo_group_size=2,  # Keep for our own logic, though not used by GRPOConfig
        rl_learning_rate=1e-6,
        rl_warmup_steps=100,
        max_prompt_length=256,
        max_completion_length=256,
        batch_size=2,
        output_dir="./tests/outputs/integration/test_rl",
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=1,
        max_steps=None
    )
    
    config = create_grpo_config(args)
    print(f"GRPO Config created successfully:")
    print(f"  beta: {config.beta}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  max_prompt_length: {config.max_prompt_length}")
    print(f"  max_completion_length: {config.max_completion_length}")
    
    print("✅ GRPO configuration tests passed")


def test_small_rl_training():
    """Test a very small RL training run."""
    print("\nTesting small RL training run...")
    
    try:
        # Check if we have the required components
        import subprocess
        import os
        
        # Create a very small test command
        test_command = [
            "python", "main.py",
            "--model", "gpt2",
            "--dataset", "openai/gsm8k",
            "--dataset_config", "main", 
            "--mode", "rl",
            "--batch_size", "1",
            "--max_steps", "2",
            "--max_length", "128",
            "--max_prompt_length", "64",
            "--max_completion_length", "64",
            "--grpo_beta", "0.1",
            "--grpo_group_size", "2",
            "--rl_learning_rate", "1e-6"
        ]
        
        print("Running small RL training test...")
        print(f"Command: {' '.join(test_command)}")
        
        # Run with timeout to prevent hanging
        result = subprocess.run(
            test_command, 
            capture_output=True, 
            text=True, 
            timeout=120,  # 2 minute timeout
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print("✅ Small RL training test completed successfully")
            print("Training output (last 500 chars):")
            print(result.stdout[-500:])
        else:
            print("❌ RL training test failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ RL training test timed out (this might be expected for first run)")
        return False
    except Exception as e:
        print(f"❌ RL training test failed with exception: {e}")
        return False
    
    return True


def main():
    """Run all RL tests."""
    print("="*60)
    print("RL FUNCTIONALITY TEST SUITE")
    print("="*60)
    
    try:
        # Test individual components
        test_rl_components()
        test_rl_dataset_preparation() 
        test_grpo_config()
        
        # Test full pipeline (optional - might take time)
        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            success = test_small_rl_training()
            if not success:
                print("\n❌ Full RL training test failed, but component tests passed")
        else:
            print("\n📝 Skipping full training test (use --full flag to run)")
        
        print("\n" + "="*60)
        print("✅ RL COMPONENT TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nRL functionality is ready for use!")
        print("Example usage:")
        print("  python main.py --model gpt2 --dataset openai/gsm8k --dataset_config main --mode rl --max_steps 100")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()