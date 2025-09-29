#!/usr/bin/env python3
"""
Test script for multi-GPU functionality.
This script tests various GPU configurations to ensure proper setup.
"""

import torch
import os
import sys
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.utils import check_device_availability

def test_single_gpu():
    """Test single GPU detection and setup."""
    print("=== Testing Single GPU ===")
    device_info = check_device_availability('auto', multi_gpu=False)
    print(f"Device info (single GPU): {device_info}")
    print()

def test_multi_gpu():
    """Test multi-GPU detection and setup."""
    print("=== Testing Multi-GPU ===")
    device_info = check_device_availability('auto', multi_gpu=True)
    print(f"Device info (multi-GPU): {device_info}")
    print()

def test_cuda_availability():
    """Test CUDA availability and device count."""
    print("=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print("CUDA not available")
    print()

def test_distributed_environment():
    """Test distributed training environment variables."""
    print("=== Distributed Environment ===")
    env_vars = ['LOCAL_RANK', 'WORLD_SIZE', 'RANK', 'MASTER_ADDR', 'MASTER_PORT']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    print()

def main():
    print("Multi-GPU Test Suite")
    print("=" * 50)
    
    test_cuda_availability()
    test_single_gpu()
    test_multi_gpu()
    test_distributed_environment()
    
    # Test with different device choices
    print("=== Testing Device Choice Override ===")
    for device in ['cuda', 'mps', 'cpu']:
        try:
            device_info = check_device_availability(device, multi_gpu=True)
            print(f"Device '{device}' result: {device_info}")
        except Exception as e:
            print(f"Device '{device}' failed: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()