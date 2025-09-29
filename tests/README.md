# Test Suite Documentation

## Overview

This directory contains the complete test suite for the Lancy Thesis training pipeline. All tests and their outputs are organized in a unified structure for easy management and execution.

## Directory Structure

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_modular_structure.py
│   └── check_device.py
│
├── integration/            # Integration tests (multi-component)
│   ├── test_multi_gpu.py
│   ├── test_moe_m1.py
│   ├── test_rl.py
│   ├── test_multi_eval.py
│   ├── test_cosine_scheduler.py
│   └── test_scheduler_info.py
│
├── functional/             # Functional tests (model-specific)
│   ├── test_llama32.py
│   └── test_qwen3.py
│
├── outputs/                # All test outputs go here
│   ├── unit/              # Unit test outputs
│   ├── integration/       # Integration test outputs
│   ├── functional/        # Functional test outputs
│   └── checkpoints/       # Test checkpoints
│
├── scripts/               # Test runner scripts
│   ├── run_all_tests.sh
│   ├── run_unit_tests.sh
│   ├── run_integration_tests.sh
│   └── run_functional_tests.sh
│
└── README.md             # This file
```

## Test Categories

### Unit Tests
Fast, isolated tests that verify individual components work correctly.

- **test_modular_structure.py**: Verifies all modules import correctly and basic functions work
- **check_device.py**: Checks available compute devices (CPU, CUDA, MPS)

**Run time**: < 10 seconds
**Dependencies**: None

### Integration Tests
Tests that verify multiple components work together correctly.

- **test_multi_gpu.py**: Tests multi-GPU training setup
- **test_moe_m1.py**: Tests Mixture of Experts models on Apple Silicon
- **test_rl.py**: Tests reinforcement learning (GRPO) functionality
- **test_multi_eval.py**: Tests multi-dataset evaluation with separate metrics
- **test_cosine_scheduler.py**: Tests cosine learning rate scheduler
- **test_scheduler_info.py**: Displays available scheduler options

**Run time**: 1-5 minutes per test
**Dependencies**: May require specific hardware (GPU, Apple Silicon)

### Functional Tests
End-to-end tests with specific models to ensure full pipeline functionality.

- **test_llama32.py**: Tests Llama 3.2 models (1B, 3B parameters)
- **test_qwen3.py**: Tests Qwen3 models (0.6B, 1.7B, 4B parameters)

**Run time**: 5-15 minutes per model
**Dependencies**: 
- HuggingFace authentication for gated models
- Model downloads (1-4GB per model)
- GPU recommended for performance

## Running Tests

### Quick Start

Run all tests:
```bash
./tests/scripts/run_all_tests.sh
```

### Running Specific Test Categories

**Unit tests only (fastest):**
```bash
./tests/scripts/run_unit_tests.sh
```

**Integration tests:**
```bash
# Run all integration tests
./tests/scripts/run_integration_tests.sh

# Run only quick integration tests
./tests/scripts/run_integration_tests.sh --quick
```

**Functional tests:**
```bash
# Run all functional tests
./tests/scripts/run_functional_tests.sh

# Run specific model tests
./tests/scripts/run_functional_tests.sh llama32
./tests/scripts/run_functional_tests.sh qwen3
```

### Running Individual Tests

You can run any test directly:
```bash
python tests/unit/test_modular_structure.py
python tests/integration/test_multi_eval.py
python tests/functional/test_qwen3.py
```

## Test Outputs

All test outputs are saved in `tests/outputs/` organized by test category:

```
tests/outputs/
├── unit/              # Minimal outputs, mostly logs
├── integration/       # Training logs, checkpoints, TensorBoard
│   ├── multi_eval_test/
│   ├── cosine_scheduler_test/
│   ├── test_rl/
│   └── ...
├── functional/        # Model-specific outputs
│   ├── llama32_test_*/
│   ├── qwen3_test_*/
│   └── ...
└── checkpoints/       # Reusable test checkpoints
```

### Viewing Results

**Console Output:**
Test results are displayed in the console with color coding:
- 🟢 Green: Test passed
- 🔴 Red: Test failed
- 🟡 Yellow: Test skipped

**TensorBoard:**
View training metrics for integration and functional tests:
```bash
# View all test metrics
tensorboard --logdir tests/outputs/

# View specific category
tensorboard --logdir tests/outputs/integration/
tensorboard --logdir tests/outputs/functional/
```

**Logs:**
Detailed logs are saved in each test's output directory:
```bash
# Example: View multi-eval test logs
cat tests/outputs/integration/multi_eval_test/logs/training.log
```

## Environment Requirements

### Python Dependencies
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets
- PEFT (for LoRA)
- Accelerate

### Hardware Requirements
- **Unit tests**: Any CPU
- **Integration tests**: CPU or GPU (some tests require specific hardware)
- **Functional tests**: GPU recommended (4GB+ VRAM)

### Authentication
Some models require HuggingFace authentication:
```bash
huggingface-cli login
```

## Writing New Tests

### Test Template
```python
#!/usr/bin/env python3
"""
Test description here.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_main_functionality():
    """Test the main functionality."""
    # Test implementation
    pass

def test_edge_cases():
    """Test edge cases."""
    # Test implementation
    pass

if __name__ == "__main__":
    test_main_functionality()
    test_edge_cases()
    print("All tests passed!")
```

### Output Path Convention
Always use the structured output paths:
```python
# For unit tests
output_dir = "./tests/outputs/unit/my_test"

# For integration tests
output_dir = "./tests/outputs/integration/my_integration_test"

# For functional tests
output_dir = "./tests/outputs/functional/my_model_test"
```

### Adding to Test Runners
Add your test to the appropriate runner script:
1. Edit the relevant script in `tests/scripts/`
2. Add a `run_test` or `run_*_test` call
3. Update this README

## Continuous Integration

For CI/CD pipelines, use the test runners with appropriate exit codes:

```yaml
# Example GitHub Actions workflow
- name: Run Unit Tests
  run: ./tests/scripts/run_unit_tests.sh
  
- name: Run Integration Tests (Quick)
  run: ./tests/scripts/run_integration_tests.sh --quick
  
- name: Run All Tests
  run: ./tests/scripts/run_all_tests.sh
  continue-on-error: false
```

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure you're running from the project root
- Check that `src/` is in the Python path

**Out of memory:**
- Reduce batch size in test configurations
- Use CPU instead of GPU for small tests
- Skip functional tests on limited hardware

**Model download failures:**
- Check internet connection
- Verify HuggingFace authentication
- Use smaller models for testing

**Test outputs not found:**
- Check that output directories exist
- Verify write permissions
- Look in `tests/outputs/` subdirectories

### Debug Mode

Run tests with verbose output:
```bash
PYTHONPATH=. python -v tests/unit/test_modular_structure.py
```

## Maintenance

### Cleaning Test Outputs

Remove all test outputs:
```bash
rm -rf tests/outputs/*/*
```

Remove specific category:
```bash
rm -rf tests/outputs/integration/*
```

### Updating Tests

When updating the main codebase:
1. Run unit tests first to catch breaking changes
2. Update integration tests if APIs change
3. Verify functional tests still work with new code
4. Update this README if test structure changes

## Test Coverage Goals

- **Unit Tests**: Cover all utility functions and basic operations
- **Integration Tests**: Cover all training modes and features
- **Functional Tests**: Cover all supported model architectures

Current coverage:
- ✅ Training modes: pretrain, sft, rl
- ✅ Features: LoRA, multi-GPU, dataset mixing, custom schedulers
- ✅ Models: GPT-2, Qwen3, Llama 3.2, MOE models
- 🔄 In progress: More comprehensive error handling tests

## Contact

For test-related issues:
1. Check this README first
2. Review test output logs
3. Check the main CLAUDE.md for project documentation
4. Create an issue with test logs attached