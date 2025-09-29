#!/usr/bin/env python3
"""
Test script to verify modular structure and functionality.
This script tests individual components without running full training.
"""

import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from src.config.args import parse_arguments, print_checkpointing_config
        from src.data.utils import load_and_prepare_dataset, create_tokenize_function, prepare_dataset_for_training, is_classification_task
        from src.models.utils import setup_tokenizer, determine_task, get_model_class, load_model, configure_model_padding, get_label_mapping, create_classification_prompt
        from src.training.utils import (
            check_device_availability, create_training_arguments, create_data_collator, 
            load_checkpoint_if_exists, create_trainer, setup_text_logging
        )
        print("✓ All imports successful")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_tokenizer_setup():
    """Test tokenizer setup functionality."""
    print("\nTesting tokenizer setup...")
    
    try:
        from src.models.utils import setup_tokenizer
        
        # Test with a small model that's likely to be available
        tokenizer = setup_tokenizer('distilbert-base-uncased')
        
        # Check that tokenizer has required attributes
        assert hasattr(tokenizer, 'pad_token')
        assert hasattr(tokenizer, 'padding_side')
        assert tokenizer.pad_token is not None
        
        print("✓ Tokenizer setup successful")
        return True
        
    except Exception as e:
        print(f"✗ Tokenizer setup failed: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\nTesting dataset loading...")
    
    try:
        from src.data.utils import load_and_prepare_dataset
        
        # Test with a small dataset - using a subset for speed
        dataset = load_and_prepare_dataset('imdb')
        
        # Check that dataset has expected structure
        assert hasattr(dataset, 'column_names')
        assert 'text' in dataset.column_names or 'sentence' in dataset.column_names
        
        print("✓ Dataset loading successful")
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_task_determination():
    """Test that task determination works correctly."""
    print("\nTesting task determination...")
    
    try:
        from src.models.utils import determine_task
        
        # Test pretrain mode
        task = determine_task('imdb', 'auto', 'pretrain')
        assert task == 'causal-lm'
        
        # Test SFT mode with IMDB (should now use generative approach)
        task = determine_task('imdb', 'auto', 'sft')
        assert task == 'causal-lm'
        
        # Test SFT mode with other dataset
        task = determine_task('squad', 'auto', 'sft')
        assert task == 'causal-lm'
        
        print("✓ Task determination successful")
        
    except Exception as e:
        print(f"✗ Task determination failed: {e}")
        return False
    
    return True


def test_classification_task_detection():
    """Test classification task detection for generative approach."""
    print("\nTesting classification task detection...")
    
    try:
        from src.data.utils import is_classification_task
        
        # Test IMDB
        assert is_classification_task('imdb') == True
        assert is_classification_task('stanfordnlp/imdb') == True
        
        # Test GLUE tasks
        assert is_classification_task('glue', 'sst2') == True
        assert is_classification_task('glue', 'cola') == True
        assert is_classification_task('glue', 'stsb') == False  # Regression task
        
        # Test non-classification tasks
        assert is_classification_task('squad') == False
        assert is_classification_task('wikitext') == False
        
        print("✓ Classification task detection successful")
        
    except Exception as e:
        print(f"✗ Classification task detection failed: {e}")
        return False
    
    return True


def test_label_mapping():
    """Test label mapping functionality."""
    print("\nTesting label mapping...")
    
    try:
        from src.models.utils import get_label_mapping
        
        # Test IMDB mapping
        imdb_mapping = get_label_mapping('imdb')
        assert imdb_mapping == {0: "negative", 1: "positive"}
        
        # Test GLUE mappings
        sst2_mapping = get_label_mapping('glue', 'sst2')
        assert sst2_mapping == {0: "negative", 1: "positive"}
        
        cola_mapping = get_label_mapping('glue', 'cola')
        assert cola_mapping == {0: "unacceptable", 1: "acceptable"}
        
        mnli_mapping = get_label_mapping('glue', 'mnli')
        assert mnli_mapping == {0: "entailment", 1: "neutral", 2: "contradiction"}
        
        # Test with None config
        default_mapping = get_label_mapping('glue', None)
        assert default_mapping == {0: "negative", 1: "positive"}
        
        print("✓ Label mapping successful")
        
    except Exception as e:
        print(f"✗ Label mapping failed: {e}")
        return False
    
    return True


def test_prompt_creation():
    """Test prompt creation for classification tasks."""
    print("\nTesting prompt creation...")
    
    try:
        from src.models.utils import create_classification_prompt
        
        # Test IMDB prompt
        imdb_prompt = create_classification_prompt("This movie is great!", dataset_name="imdb")
        assert "sentiment" in imdb_prompt.lower()
        assert "positive" in imdb_prompt and "negative" in imdb_prompt
        
        # Test GLUE SST2 prompt
        sst2_prompt = create_classification_prompt("Good film", dataset_name="glue", dataset_config="sst2")
        assert "sentiment" in sst2_prompt.lower()
        
        # Test GLUE MRPC prompt (sentence pair)
        mrpc_prompt = create_classification_prompt("Sentence 1", "Sentence 2", dataset_name="glue", dataset_config="mrpc")
        assert "equivalent" in mrpc_prompt.lower()
        
        print("✓ Prompt creation successful")
        
    except Exception as e:
        print(f"✗ Prompt creation failed: {e}")
        return False
    
    return True


def test_model_loading():
    """Test model loading functionality."""
    print("\nTesting model loading...")
    
    try:
        from src.models.utils import get_model_class, load_model
        
        # Test getting model class for generative classification
        model_class = get_model_class('sequence-classification')
        assert 'CausalLM' in model_class.__name__
        
        # Test loading a small model
        model, _ = load_model('distilgpt2', 'causal-lm', model_class)
        assert model is not None
        
        print("✓ Model loading successful")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Modular Structure (Generative Approach)")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_tokenizer_setup,
        test_task_determination,
        test_classification_task_detection,
        test_label_mapping,
        test_prompt_creation,
        test_model_loading,
        # Skip dataset and tokenization tests as they require more setup
        # test_dataset_loading,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print("=" * 50)
    
    if passed == total:
        print("✓ All tests passed! Modular structure is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 