"""
Mixin class for TensorBoard metadata logging functionality.

This mixin provides consistent metadata logging across all trainer types.
"""

import warnings
from typing import Dict, Optional, Any
from .utils import collect_training_metadata, format_metadata_as_markdown


class MetadataLoggingMixin:
    """
    Mixin to add metadata logging capabilities to any Trainer class.
    
    This mixin should be used with classes that inherit from transformers.Trainer
    and provides consistent metadata logging to TensorBoard across all trainer types.
    """
    
    def __init__(self, *args, training_args_dict: Optional[Dict] = None, use_packing: bool = False, **kwargs):
        """
        Initialize the mixin with metadata tracking.
        
        Args:
            training_args_dict: Original training arguments for metadata logging
            use_packing: Whether sequence packing is enabled (affects token counting)
        """
        
        # Store configuration needed for metadata
        self.training_args_dict = training_args_dict or {}
        self.use_packing = use_packing
        
        # Initialize metadata tracking
        self.training_metadata = None
        self.metadata_logged = False
        
        # Call parent init
        super().__init__(*args, **kwargs)
    
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """
        Override train method to log metadata at training start.
        """
        # Log metadata at training start
        self._log_training_metadata()
        
        # Call parent's train method
        result = super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, **kwargs)
        return result
    
    def _log_training_metadata(self):
        """
        Collect training metadata and store for later TensorBoard logging.
        """
        try:
            # Collect metadata
            metadata = collect_training_metadata(
                model=self.model,
                args=self.training_args_dict,  # Use the original args dict
                training_args=self.args,        # HuggingFace TrainingArguments
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset
            )
            
            # Store metadata for later TensorBoard logging
            self.training_metadata = metadata
            
            # Log to console (this works immediately)
            print("\n" + "="*80)
            print("TRAINING METADATA")
            print("="*80)
            for section, values in metadata.items():
                if isinstance(values, dict):
                    print(f"\n{section}:")
                    for key, value in values.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"{section}: {values}")
            print("="*80 + "\n")
            
        except Exception as e:
            warnings.warn(f"Could not collect training metadata: {e}")
    
    def _log_metadata_to_tensorboard(self):
        """
        Log stored metadata to TensorBoard when tb_writer is available.
        This is called at the right time when all components are initialized.
        """
        if not self.training_metadata or self.metadata_logged:
            return
        
        # Try to get tb_writer from TensorBoardCallback
        tb_writer = None
        if hasattr(self, 'callback_handler'):
            for callback in self.callback_handler.callbacks:
                if callback.__class__.__name__ == 'TensorBoardCallback':
                    if hasattr(callback, 'tb_writer'):
                        tb_writer = callback.tb_writer
                        break
        
        # Also check if it's directly on trainer (for compatibility)
        if tb_writer is None and hasattr(self, 'tb_writer'):
            tb_writer = self.tb_writer
        
        if tb_writer is None:
            return
        metadata = self.training_metadata
        
        try:
            # Log scalars at step 0 using tb_writer directly
            # Token budget
            if 'token_budget' in metadata:
                tb_writer.add_scalar('train/token_budget', metadata['token_budget'], 0)
            
            # Total parameters
            if 'total_parameters' in metadata:
                tb_writer.add_scalar('train/total_parameters', metadata['total_parameters'], 0)
            
            # Trainable parameters
            if 'trainable_parameters' in metadata:
                tb_writer.add_scalar('train/trainable_parameters', metadata['trainable_parameters'], 0)
            
            # Calculate and log trainable percentage as scalar
            if 'trainable_parameters' in metadata and 'total_parameters' in metadata:
                total = metadata['total_parameters']
                trainable = metadata['trainable_parameters']
                if total > 0:
                    percentage = (trainable / total) * 100
                    tb_writer.add_scalar('train/trainable_percentage', percentage, 0)
            
            # Log text summaries
            # Configuration summary with all details
            summary_text = f"""# Training Configuration Summary

## Compute Budget
- **Total Token Budget**: {metadata.get('token_budget_human', 'N/A')}
- **Raw Token Count**: {metadata.get('token_budget', 'N/A'):,}

## Model Configuration  
- **Attention Implementation**: {metadata.get('attention_implementation', 'auto')}
- **Model**: {metadata.get('model_name', 'unknown')}
- **Total Parameters**: {metadata.get('total_parameters_human', 'N/A')}
- **Total Parameters (raw)**: {metadata.get('total_parameters', 'N/A'):,}
- **Trainable Parameters**: {metadata.get('trainable_parameters', 'N/A'):,}
- **Trainable Percentage**: {metadata.get('trainable_percentage', 'N/A')}

## Training Settings
- **Mixed Precision**: {metadata.get('mixed_precision', 'fp32')}
- **Sequence Packing**: {metadata.get('sequence_packing', False)}
- **Max Sequence Length**: {metadata.get('max_sequence_length', 'N/A')}
- **Effective Batch Size**: {metadata.get('effective_batch_size', 'N/A')}
- **Learning Rate**: {metadata.get('learning_rate', 'N/A')}
- **Device**: {metadata.get('device', 'unknown')}
- **Device Type**: {metadata.get('device_type', 'unknown')}
- **Device Name**: {metadata.get('device_name', 'unknown')}
"""
            
            tb_writer.add_text('configuration/summary', summary_text, 0)
            tb_writer.add_text('configuration/attention_implementation', 
                                   f"Attention: {metadata.get('attention_implementation', 'auto')}", 0)
            tb_writer.add_text('configuration/token_budget', 
                                   f"Token Budget: {metadata.get('token_budget_human', 'N/A')} ({metadata.get('token_budget', 0):,} tokens)", 0)
            
            # Also log the full metadata as formatted markdown
            markdown_content = format_metadata_as_markdown(metadata)
            tb_writer.add_text('training_metadata', markdown_content, 0)
            
            # Mark as logged to avoid duplicate logging
            self.metadata_logged = True
            
            # Successfully logged
            
        except Exception as e:
            warnings.warn(f"Could not log metadata to TensorBoard: {e}")
    
    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """
        Override to ensure metadata is logged to TensorBoard at the first opportunity.
        
        Note: Uses *args and **kwargs to handle any number of arguments 
        that may be passed in different versions of transformers.
        """
        # Log metadata to TensorBoard on first call when tb_writer is available
        # Try on every call until successful
        if not self.metadata_logged:
            self._log_metadata_to_tensorboard()
        
        # Let parent handle the rest
        return super()._maybe_log_save_evaluate(*args, **kwargs)