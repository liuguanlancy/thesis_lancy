#!/usr/bin/env python3
"""
Test script to verify Flash Attention detection works correctly with device='auto'.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models.hardware_detection import get_optimal_attention_implementation

def test_device_detection():
    """Test that device detection correctly identifies CUDA and enables Flash Attention."""
    
    print("="*60)
    print("TESTING FLASH ATTENTION DETECTION")
    print("="*60)
    
    # Test device detection
    print("\n1. Device Detection:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   CUDA device name: {torch.cuda.get_device_name(0)}")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        print(f"   Compute capability: {major}.{minor}")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    
    # Test with device='auto'
    print("\n2. Testing with device='auto':")
    if torch.cuda.is_available():
        expected_device = 'cuda'
    elif torch.backends.mps.is_available():
        expected_device = 'mps'
    else:
        expected_device = 'cpu'
    
    print(f"   Expected device resolution: auto -> {expected_device}")
    
    # Test attention implementation detection
    print("\n3. Testing attention implementation detection:")
    optimal_attn = get_optimal_attention_implementation(device='auto', verbose=True)
    
    print(f"\n   Result: {optimal_attn}")
    
    # Verify expectations
    print("\n4. Verification:")
    if torch.cuda.is_available():
        # Check if Flash Attention package is installed
        try:
            import flash_attn
            flash_installed = True
            print("   ✓ flash-attn package is installed")
        except ImportError:
            flash_installed = False
            print("   ✗ flash-attn package NOT installed")
            print("     Install with: pip install flash-attn --no-build-isolation")
        
        # Check compute capability
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(device)
            if major >= 8:
                print(f"   ✓ GPU compute capability {major}.{minor} supports Flash Attention")
                if flash_installed:
                    if optimal_attn == 'flash_attention_2':
                        print("   ✓ Flash Attention 2 correctly selected!")
                    else:
                        print(f"   ⚠ Expected flash_attention_2 but got {optimal_attn}")
                else:
                    if optimal_attn == 'sdpa':
                        print("   ✓ SDPA correctly selected (flash-attn not installed)")
                    else:
                        print(f"   ⚠ Expected sdpa but got {optimal_attn}")
            else:
                print(f"   ℹ GPU compute capability {major}.{minor} doesn't support Flash Attention")
                if optimal_attn == 'sdpa':
                    print("   ✓ SDPA correctly selected for older GPU")
    elif torch.backends.mps.is_available():
        if optimal_attn == 'eager':
            print("   ✓ Eager attention correctly selected for MPS")
        else:
            print(f"   ⚠ Expected eager for MPS but got {optimal_attn}")
    else:
        if optimal_attn == 'eager':
            print("   ✓ Eager attention correctly selected for CPU")
        else:
            print(f"   ⚠ Expected eager for CPU but got {optimal_attn}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Test the actual model loading path
    print("\n5. Testing model loading with auto device:")
    from src.models.utils import load_model, get_model_class
    
    model_class = get_model_class('causal-lm')
    # Just test the device resolution without actually loading a model
    print("   This would load with the correct attention implementation.")
    
    return optimal_attn

if __name__ == "__main__":
    test_device_detection()