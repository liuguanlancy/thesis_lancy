"""
Trainer wrapper that adds metadata logging to standard HuggingFace Trainer.
"""
from transformers import Trainer
from typing import Optional, Dict
from .metadata_mixin import MetadataLoggingMixin


class MetadataTrainer(MetadataLoggingMixin, Trainer):
    """
    Standard HuggingFace Trainer with added metadata logging capabilities.
    
    This is used for non-MPS, non-multi-eval training scenarios.
    """
    pass  # All functionality is provided by the mixin