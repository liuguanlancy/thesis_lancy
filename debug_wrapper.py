#!/usr/bin/env python
"""Debug wrapper that sets environment variables before importing anything."""

import os
import sys
import shutil

# Check if we're in debug mode
if os.environ.get('PYDEVD_LOAD_VALUES_ASYNC') or os.environ.get('PYDEVD_USE_FRAME_EVAL'):
    print("=" * 60)
    print("DEBUG MODE DETECTED - Setting up debug cache")
    print("=" * 60)
    
    # Set up a separate cache for debug mode BEFORE any imports
    debug_cache = os.path.expanduser("~/.cache/huggingface_debug")
    
    # Clear any existing debug cache
    if os.path.exists(debug_cache):
        print(f"Clearing existing debug cache: {debug_cache}")
        shutil.rmtree(debug_cache)
    
    os.makedirs(debug_cache, exist_ok=True)
    
    # Set ALL relevant environment variables
    os.environ['HF_HOME'] = debug_cache
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(debug_cache, 'transformers')
    os.environ['HF_DATASETS_CACHE'] = os.path.join(debug_cache, 'datasets')
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(debug_cache, 'hub')
    
    print(f"HF_HOME set to: {os.environ['HF_HOME']}")
    print(f"TRANSFORMERS_CACHE set to: {os.environ['TRANSFORMERS_CACHE']}")
    print("=" * 60)

# Now import and run the main training script
if __name__ == "__main__":
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    # Import main AFTER setting environment variables
    from main import main
    main()