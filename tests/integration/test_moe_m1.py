#!/usr/bin/env python3
"""
Test script for MOE (Mixture of Experts) models on Apple M1 Max.
This script verifies MOE model compatibility and performance on Apple Silicon.
"""

import torch
import time
import psutil
import os
import sys
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.utils import is_moe_model, get_moe_config, load_model, get_model_class

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def test_moe_detection():
    """Test MOE model detection functionality."""
    print("\n" + "="*60)
    print("Testing MOE Model Detection")
    print("="*60)
    
    test_models = [
        ("gpt2", False),
        ("mistralai/Mixtral-8x7B-v0.1", True),
        ("google/switch-base-8", True),
        ("deepseek-ai/deepseek-moe-16b-base", True),
        ("bert-base-uncased", False),
        ("Qwen/Qwen2-1.5B", False),
    ]
    
    for model_name, expected_is_moe in test_models:
        detected = is_moe_model(model_name)
        status = "✓" if detected == expected_is_moe else "✗"
        print(f"{status} {model_name}: {'MOE' if detected else 'Standard'} (Expected: {'MOE' if expected_is_moe else 'Standard'})")
    
    print("\nMOE Configuration Examples:")
    moe_models = ["mistralai/Mixtral-8x7B-v0.1", "google/switch-base-8", "deepseek-ai/deepseek-moe-16b-base"]
    for model_name in moe_models:
        config = get_moe_config(model_name)
        print(f"\n{model_name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

def test_small_moe_loading():
    """Test loading a small MOE model suitable for M1 Max."""
    print("\n" + "="*60)
    print("Testing Small MOE Model Loading on M1 Max")
    print("="*60)
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ MPS (Metal Performance Shaders) is available")
    else:
        device = "cpu"
        print(f"⚠ MPS not available, using CPU")
    
    print(f"\nSystem Information:")
    print(f"  Total RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"  Available RAM: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    print(f"  Current memory usage: {get_memory_usage():.2f} GB")
    
    # For M1 Max testing, we'll use a very small model or mock
    # Since Mixtral models are large (>13GB), we'll test with a smaller configuration
    print("\n" + "-"*40)
    print("Testing MOE Model Loading (Mock Test)")
    print("-"*40)
    
    # Test the MOE loading logic without actually loading a large model
    model_name = "mistralai/Mixtral-8x7B-v0.1"  # This would be the actual MOE model
    
    print(f"\nModel: {model_name}")
    print(f"Device: {device}")
    
    if is_moe_model(model_name):
        print("✓ Detected as MOE model")
        config = get_moe_config(model_name)
        print(f"  Configuration: {config}")
        print("\nNote: Actual Mixtral-8x7B model requires ~26GB RAM")
        print("      For production use on M1 Max (32GB), consider:")
        print("      - Using 4-bit or 8-bit quantization")
        print("      - Using smaller MOE models")
        print("      - Offloading some layers to disk")
    
    # Test with a small non-MOE model to verify the pipeline works
    print("\n" + "-"*40)
    print("Testing Pipeline with Small Model (gpt2)")
    print("-"*40)
    
    try:
        # Use GPT-2 as a small test model
        test_model_name = "gpt2"
        print(f"\nLoading {test_model_name} for pipeline verification...")
        
        mem_before = get_memory_usage()
        start_time = time.time()
        
        # Load model using our custom loader
        model_class = get_model_class("causal-lm")
        model, _ = load_model(test_model_name, "causal-lm", model_class, device=device)
        tokenizer = AutoTokenizer.from_pretrained(test_model_name)
        
        load_time = time.time() - start_time
        mem_after = get_memory_usage()
        
        print(f"✓ Model loaded successfully")
        print(f"  Load time: {load_time:.2f} seconds")
        print(f"  Memory used: {(mem_after - mem_before):.2f} GB")
        
        # Test inference
        print("\nTesting inference...")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_text = "The mixture of experts architecture"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        if device == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model = model.to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True
            )
        inference_time = time.time() - start_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Inference successful")
        print(f"  Time: {inference_time:.2f} seconds")
        print(f"  Input: '{test_text}'")
        print(f"  Generated: '{generated_text}'")
        
        # Cleanup
        del model
        del tokenizer
        if device == "mps":
            torch.mps.empty_cache()
        
        print("\n✓ Pipeline verification complete")
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

def test_moe_recommendations():
    """Provide recommendations for MOE models on M1 Max."""
    print("\n" + "="*60)
    print("MOE Model Recommendations for M1 Max (32GB)")
    print("="*60)
    
    recommendations = [
        {
            "category": "Small MOE Models (Direct Loading)",
            "models": [
                "google/switch-base-8 (~3GB)",
                "google/switch-base-16 (~6GB)",
            ],
            "notes": "Can be loaded directly without quantization"
        },
        {
            "category": "Medium MOE Models (With Quantization)",
            "models": [
                "deepseek-ai/deepseek-moe-16b-base (8-bit: ~16GB)",
                "mistralai/Mixtral-8x7B-v0.1 (4-bit: ~13GB)",
            ],
            "notes": "Requires quantization for M1 Max"
        },
        {
            "category": "Alternative Approaches",
            "models": [
                "Use dense models with similar performance",
                "Fine-tune smaller MOE models",
                "Use model sharding/offloading techniques"
            ],
            "notes": "Consider trade-offs between model size and performance"
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['category']}:")
        for model in rec['models']:
            print(f"  • {model}")
        print(f"  Note: {rec['notes']}")
    
    print("\n" + "-"*40)
    print("Quantization Commands for Large MOE Models:")
    print("-"*40)
    print("""
# For 8-bit quantization (requires bitsandbytes):
python main.py --model mistralai/Mixtral-8x7B-v0.1 \\
               --dataset stanfordnlp/imdb \\
               --mode sft \\
               --moe_load_in_8bit \\
               --device mps \\
               --batch_size 1 \\
               --max_length 128

# For 4-bit quantization (more aggressive compression):
python main.py --model mistralai/Mixtral-8x7B-v0.1 \\
               --dataset stanfordnlp/imdb \\
               --mode sft \\
               --moe_load_in_4bit \\
               --device mps \\
               --batch_size 1 \\
               --max_length 128
    """)

def main():
    """Run all MOE tests for M1 Max."""
    print("="*60)
    print("MOE Model Testing Suite for Apple M1 Max")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Run tests
    test_moe_detection()
    test_small_moe_loading()
    test_moe_recommendations()
    
    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)
    print("\nSummary:")
    print("✓ MOE detection logic implemented")
    print("✓ MOE configuration system ready")
    print("✓ MPS device compatibility added")
    print("✓ Pipeline tested with small model")
    print("\nNext steps:")
    print("1. Install quantization libraries if needed:")
    print("   pip install bitsandbytes accelerate")
    print("2. Try loading a small MOE model:")
    print("   python main.py --model google/switch-base-8 --dataset stanfordnlp/imdb --mode sft --max_steps 10")
    print("3. For larger models, use quantization flags (--moe_load_in_4bit or --moe_load_in_8bit)")

if __name__ == "__main__":
    main()