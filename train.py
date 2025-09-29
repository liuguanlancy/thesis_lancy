#!/usr/bin/env python3
"""
Main entry point for training - redirects to src/main.py
This file allows users to run training from the project root.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main function
from main import main

if __name__ == "__main__":
    main()