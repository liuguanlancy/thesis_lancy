"""
Multi-dataset evaluation trainer for mixture training.

This trainer evaluates each dataset in a mixture separately, providing individual
metrics for each dataset. Metrics are logged to both console/files and TensorBoard.
"""

import math
import warnings
from typing import Dict, Optional, Any, Union, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction, denumpify_detensorize
from datasets import Dataset
from .utils import collect_training_metadata, format_metadata_as_markdown


class MultiEvalTrainer(Trainer):
    """
    Custom Trainer that evaluates multiple datasets separately during training.
    
    This is particularly useful for mixture training where we want to track
    performance on each individual dataset rather than just the aggregate.
    """
    
    def __init__(
        self,
        *args,
        eval_datasets_dict: Optional[Dict[str, Dataset]] = None,
        eval_dataset_names: Optional[List[str]] = None,
        eval_max_batches: int = -1,
        log_eval_spread: bool = True,
        use_packing: bool = False,
        training_args_dict: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize MultiEvalTrainer.
        
        Args:
            eval_datasets_dict: Dictionary mapping dataset names to eval datasets
            eval_dataset_names: Optional clean names for logging (defaults to dict keys)
            eval_max_batches: Maximum batches per dataset evaluation (-1 for full)
            log_eval_spread: Whether to log spread metrics for balance monitoring
            use_packing: Whether sequence packing is enabled (affects token counting)
            training_args_dict: Original training arguments for metadata logging
        """
        # Handle eval_dataset vs eval_datasets_dict
        if eval_datasets_dict is not None:
            self.eval_datasets_dict = eval_datasets_dict
            self.eval_dataset_names = eval_dataset_names or list(eval_datasets_dict.keys())
            # Set a placeholder eval_dataset for parent class
            # We'll override evaluate() so this won't be used directly
            if 'eval_dataset' not in kwargs:
                kwargs['eval_dataset'] = next(iter(eval_datasets_dict.values()))
        else:
            self.eval_datasets_dict = None
            self.eval_dataset_names = None
        
        self.eval_max_batches = eval_max_batches
        self.log_eval_spread = log_eval_spread
        self.use_packing = use_packing
        self.training_args_dict = training_args_dict or {}
        
        # Initialize token tracking
        self.total_tokens_trained = 0
        
        # Store metadata for later TensorBoard logging
        self.training_metadata = None
        self.metadata_logged = False  # Flag to avoid duplicate logging
        
        # Detect MPS device for safety measures
        self.is_mps = False
        if args and len(args) > 0 and hasattr(args[0], 'device'):
            model = args[0]
            try:
                device = next(model.parameters()).device
                self.is_mps = device.type == 'mps'
            except:
                pass
        
        if self.is_mps:
            warnings.warn(
                "MPS device detected in MultiEvalTrainer. Using MPS-safe evaluation.",
                UserWarning
            )
        
        super().__init__(*args, **kwargs)
    
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """
        Override train method to log metadata at training start.
        """
        # Log metadata at training start
        self._log_training_metadata()
        
        # Call parent's train method
        return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, **kwargs)
    
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
            
            # Trainable parameters (NEW)
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
            
        except Exception as e:
            warnings.warn(f"Could not log metadata to TensorBoard: {e}")
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step and track total tokens trained.
        
        This override counts tokens based on whether packing is enabled:
        - With packing: all positions contain real tokens
        - Without packing: only non-padded positions are real tokens
        
        Args:
            model: The model being trained
            inputs: The input batch
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        """
        # Count tokens in this batch
        if self.use_packing:
            # With packing, all positions are real tokens
            batch_tokens = inputs["input_ids"].numel()
        else:
            # Without packing, only count non-padded tokens
            if "attention_mask" in inputs:
                batch_tokens = inputs["attention_mask"].sum().item()
            else:
                # Fallback if no attention mask (shouldn't happen)
                batch_tokens = inputs["input_ids"].numel()
        
        # Accumulate total tokens
        self.total_tokens_trained += batch_tokens
        
        # Call parent training_step with appropriate arguments
        if num_items_in_batch is not None:
            # Newer transformers version with num_items_in_batch
            return super().training_step(model, inputs, num_items_in_batch)
        else:
            # Older transformers version
            return super().training_step(model, inputs)
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation on all datasets and return metrics.
        
        If eval_datasets_dict is provided, evaluates each dataset separately
        and returns individual + aggregate metrics.
        """
        # If we have multiple eval datasets, evaluate each separately
        if self.eval_datasets_dict is not None:
            return self._evaluate_multiple_datasets(ignore_keys, metric_key_prefix)
        else:
            # Fall back to standard evaluation for single dataset
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    def _evaluate_multiple_datasets(
        self,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Evaluate each dataset separately and aggregate metrics.
        """
        all_metrics = {}
        dataset_losses = []
        dataset_perplexities = []
        
        # Log start of multi-dataset evaluation
        if self.args.local_rank <= 0:  # Only log on main process
            print("\n" + "="*60)
            print(f"Evaluating {len(self.eval_datasets_dict)} datasets separately")
            print("="*60)
        
        # Evaluate each dataset
        for dataset_name in self.eval_dataset_names:
            if dataset_name not in self.eval_datasets_dict:
                warnings.warn(f"Dataset {dataset_name} not found in eval_datasets_dict")
                continue
            
            eval_dataset = self.eval_datasets_dict[dataset_name]
            
            # Create dataloader for this specific dataset
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
            # Apply batch limiting if specified
            if self.eval_max_batches > 0:
                eval_dataloader = self._limit_batches(eval_dataloader, self.eval_max_batches)
            
            # Run evaluation loop
            if self.args.local_rank <= 0:
                print(f"\nEvaluating {dataset_name}...")
            
            # Notify callbacks about dataset change
            # This allows EnhancedProgressCallback to reset step counters
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, '_current_eval_dataset'):
                    callback._current_eval_dataset = dataset_name
                    if dataset_name not in callback._eval_dataset_steps:
                        callback._eval_dataset_steps[dataset_name] = 0
            
            output = self.evaluation_loop(
                eval_dataloader,
                description=f"Evaluation - {dataset_name}",
                prediction_loss_only=True,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}/{dataset_name}"
            )
            
            # Extract metrics
            metrics = output.metrics
            
            # Get loss value (handle different metric key formats)
            loss_key = f"{metric_key_prefix}/{dataset_name}_loss"
            if loss_key in metrics:
                loss = metrics[loss_key]
            elif f"{metric_key_prefix}/{dataset_name}/loss" in metrics:
                loss = metrics[f"{metric_key_prefix}/{dataset_name}/loss"]
            else:
                # Try to find any loss metric for this dataset
                loss_keys = [k for k in metrics.keys() if 'loss' in k.lower() and dataset_name in k]
                if loss_keys:
                    loss = metrics[loss_keys[0]]
                else:
                    loss = None
            
            if loss is not None:
                # Calculate perplexity
                try:
                    perplexity = math.exp(loss) if loss < 10 else float('inf')
                except OverflowError:
                    perplexity = float('inf')
                
                # Store individual metrics with hierarchical structure
                # Using "/" creates folders in TensorBoard: eval/dataset_name/metric
                all_metrics[f"eval_{dataset_name}/loss"] = loss
                all_metrics[f"eval_{dataset_name}/perplexity"] = perplexity
                
                dataset_losses.append(loss)
                dataset_perplexities.append(perplexity if perplexity != float('inf') else 1000.0)
                
                # Log individual results
                if self.args.local_rank <= 0:
                    print(f"  {dataset_name:20s} - loss: {loss:.4f}, perplexity: {perplexity:.2f}")
        
        # Calculate aggregate metrics
        if dataset_losses:
            avg_loss = np.mean(dataset_losses)
            avg_perplexity = np.mean(dataset_perplexities)
            
            # Store aggregate metrics with hierarchical structure
            all_metrics["eval_average/loss"] = avg_loss
            all_metrics["eval_average/perplexity"] = avg_perplexity
            
            # Also add standard eval_loss for compatibility with metric_for_best_model
            all_metrics[f"{metric_key_prefix}_loss"] = avg_loss
            
            # Calculate spread metrics if requested
            if self.log_eval_spread and len(dataset_perplexities) > 1:
                all_metrics["eval_spread/max_min_diff"] = max(dataset_perplexities) - min(dataset_perplexities)
                all_metrics["eval_spread/std_dev"] = np.std(dataset_perplexities)
                all_metrics["eval_spread/relative_spread"] = (max(dataset_perplexities) - min(dataset_perplexities)) / avg_perplexity * 100
            
            # Log summary
            if self.args.local_rank <= 0:
                print("\n" + "-"*60)
                print(f"Average - loss: {avg_loss:.4f}, perplexity: {avg_perplexity:.2f}")
                
                if self.log_eval_spread and len(dataset_perplexities) > 1:
                    best_idx = np.argmin(dataset_perplexities)
                    worst_idx = np.argmax(dataset_perplexities)
                    print(f"Best: {self.eval_dataset_names[best_idx]} (perplexity: {dataset_perplexities[best_idx]:.2f})")
                    print(f"Worst: {self.eval_dataset_names[worst_idx]} (perplexity: {dataset_perplexities[worst_idx]:.2f})")
                    print(f"Spread: {max(dataset_perplexities) - min(dataset_perplexities):.2f} ({all_metrics['eval_spread/relative_spread']:.1f}% relative)")
                print("="*60 + "\n")
        
        # Log the metrics properly to ensure they appear in TensorBoard
        if len(all_metrics) > 0:
            # The base Trainer's evaluate() calls self.log() for metrics
            # Our metrics have "eval_" prefix to be recognized as eval metrics
            # even when logged during training (avoids "train/" prefix)
            self.log(all_metrics)
            
            # Print metrics to console with proper formatting
            self.log_metrics("eval", all_metrics)
            
            # Save metrics to JSON file (eval_results.json)
            self.save_metrics("eval", all_metrics)
            
            # Report to callbacks for TensorBoard logging
            # The on_evaluate callback should add "eval/" prefix to all metrics
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics=all_metrics
            )
        
        return all_metrics
    
    def _limit_batches(self, dataloader: DataLoader, max_batches: int) -> List:
        """
        Limit the number of batches from a dataloader.
        """
        limited_batches = []
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            limited_batches.append(batch)
        
        # Create a simple iterable wrapper
        class LimitedDataLoader:
            def __init__(self, batches):
                self.batches = batches
            
            def __iter__(self):
                return iter(self.batches)
            
            def __len__(self):
                return len(self.batches)
        
        return LimitedDataLoader(limited_batches)
    
    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """
        Override to ensure multi-dataset evaluation happens at the right steps.
        Also logs metadata to TensorBoard at the first opportunity.
        
        Note: Uses *args and **kwargs to handle any number of arguments 
        that may be passed in different versions of transformers.
        """
        # Log metadata to TensorBoard on first call when tb_writer is available
        # Try on every call until successful
        if not self.metadata_logged:
            self._log_metadata_to_tensorboard()
        
        # Let parent handle the decision of when to evaluate
        result = super()._maybe_log_save_evaluate(*args, **kwargs)
        
        # The parent's evaluate() will call our overridden version
        # which handles multiple datasets automatically
        
        return result
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log method to add token count when other training metrics are logged.
        
        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time (added in newer transformers versions)
        """
        # Check if this is a training metrics log (has loss or grad_norm)
        # These metrics appear without prefix during training
        if hasattr(self, 'total_tokens_trained'):
            # Check if this is a training log (not eval)
            has_training_metrics = any(key in logs for key in ['loss', 'grad_norm', 'learning_rate'])
            if has_training_metrics and not any('eval' in key for key in logs):
                # Add token count to the same log entry
                logs['train/total_tokens_trained'] = self.total_tokens_trained
        
        # Call parent's log method with appropriate arguments
        if start_time is not None:
            # Newer transformers version with start_time
            return super().log(logs, start_time)
        else:
            # Older transformers version
            return super().log(logs)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Override evaluation loop to handle MPS safety if needed.
        """
        if self.is_mps:
            # Apply MPS safety measures similar to MPSSafeTrainer
            # This helps prevent NaN values on Apple Silicon
            return self._mps_safe_evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            )
        else:
            # Use standard evaluation loop
            return super().evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            )
    
    def _mps_safe_evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        MPS-safe evaluation loop that prevents NaN values.
        
        Similar to MPSSafeTrainer but integrated into multi-eval flow.
        """
        # Store original settings
        original_eval_accumulation = self.args.eval_accumulation_steps
        
        # Force single-batch accumulation for MPS stability
        self.args.eval_accumulation_steps = 1
        
        try:
            # Run evaluation with safety measures
            output = super().evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            )
            
            # Check for NaN and handle gracefully
            if output.metrics:
                for key, value in output.metrics.items():
                    if 'loss' in key:
                        if math.isnan(value) or math.isinf(value):
                            warnings.warn(f"NaN/Inf detected in {key}, replacing with large value")
                            output.metrics[key] = 10.0  # Large but finite loss
            
            return output
            
        finally:
            # Restore original settings
            self.args.eval_accumulation_steps = original_eval_accumulation