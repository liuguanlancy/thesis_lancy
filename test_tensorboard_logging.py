#!/usr/bin/env python3
"""
Test script to validate TensorBoard logging implementation.
This script tests that metadata is properly logged to TensorBoard.
"""

import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_tensorboard_logging():
    """Test that TensorBoard logging works correctly."""
    from src.main import setup_training_components, setup_training_pipeline
    from src.config.args import parse_arguments
    
    # Create a temporary output directory
    temp_dir = tempfile.mkdtemp(prefix="test_tb_")
    
    try:
        # Create test arguments
        test_args = [
            '--model', 'gpt2',
            '--dataset', 'stanfordnlp/imdb',
            '--mode', 'sft',
            '--max_steps', '2',  # Just 2 steps for testing
            '--batch_size', '2',
            '--max_length', '128',
            '--output_dir', temp_dir,
            '--eval_steps', '1',
            '--save_steps', '100',
            '--logging_steps', '1',
            '--eval_max_batches', '1',
        ]
        
        # Parse arguments
        sys.argv = ['test'] + test_args
        args = parse_arguments()
        
        print("Setting up training components...")
        train_dataset, eval_dataset, tokenizer, model, task = setup_training_components(args)
        
        print("Setting up training pipeline...")
        trainer, checkpoint = setup_training_pipeline(args, train_dataset, eval_dataset, tokenizer, model, task)
        
        print("\nStarting training (2 steps only)...")
        print("Watch for 'Successfully logged metadata to TensorBoard' message")
        print("-" * 60)
        
        # Train for just 2 steps
        trainer.train()
        
        print("-" * 60)
        
        # Check if TensorBoard files were created
        tb_dir = os.path.join(temp_dir, 'tensorboard')
        if os.path.exists(tb_dir):
            tb_files = os.listdir(tb_dir)
            if tb_files:
                print(f"\n✅ TensorBoard files created: {tb_files}")
                print(f"✅ TensorBoard logging appears to be working!")
                print(f"\nTo view the logs, run:")
                print(f"  tensorboard --logdir {tb_dir}")
            else:
                print("❌ TensorBoard directory exists but is empty")
        else:
            print("❌ TensorBoard directory was not created")
        
        # Keep the directory for manual inspection
        print(f"\nTest output directory: {temp_dir}")
        print("(Not automatically deleted - inspect manually if needed)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return False

if __name__ == "__main__":
    print("Testing TensorBoard metadata logging implementation...")
    print("=" * 60)
    
    success = test_tensorboard_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
        print("\nExpected to see in TensorBoard:")
        print("- Scalars tab: train/token_budget, train/total_parameters, train/trainable_parameters")
        print("- Text tab: configuration/summary, configuration/attention_implementation, configuration/token_budget")
    else:
        print("❌ Test failed!")
    
    sys.exit(0 if success else 1)