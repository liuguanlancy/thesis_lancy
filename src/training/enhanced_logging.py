"""
Enhanced logging utilities for comprehensive training progress tracking.

This module provides advanced logging capabilities that capture all training metrics,
progress information, and evaluation results in a clean, readable format.
"""

import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import torch
import numpy as np


class EnhancedProgressCallback(TrainerCallback):
    """
    Enhanced callback for comprehensive logging of training progress.
    
    Features:
    - Prettified progress bars in logs (not raw tqdm)
    - Complete metrics tracking
    - Time estimates and throughput calculations
    - Memory usage monitoring
    - Detailed evaluation results
    """
    
    def __init__(self, log_file: Optional[str] = None, log_interval: int = 10, original_args=None):
        """
        Initialize the enhanced progress callback.

        Args:
            log_file: Path to log file. If None, uses standard logging
            log_interval: Log progress every N steps
            original_args: Original command-line arguments for metadata collection
        """
        self.log_file = log_file
        self.log_interval = log_interval
        self.original_args = original_args  # Store original args for metadata collection
        self.start_time = None
        self.step_times = []
        self.last_log_step = 0
        
        # Evaluation tracking per dataset
        self._current_eval_dataset = None
        self._eval_dataset_steps = {}
        
        # Set up file logger if specified
        if self.log_file:
            self.file_logger = self._setup_file_logger(log_file)
        else:
            self.file_logger = logging.getLogger(__name__)
    
    def _setup_file_logger(self, log_file: str) -> logging.Logger:
        """Set up a dedicated file logger."""
        logger = logging.getLogger(f"enhanced_training_{os.path.basename(log_file)}")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Create file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _format_tokens(self, num_tokens: int) -> str:
        """Format token count into human-readable format."""
        if num_tokens < 1000:
            return str(num_tokens)
        elif num_tokens < 1_000_000:
            return f"{num_tokens/1000:.1f}K"
        elif num_tokens < 1_000_000_000:
            return f"{num_tokens/1_000_000:.1f}M"
        else:
            return f"{num_tokens/1_000_000_000:.2f}B"
    
    def _get_memory_info(self) -> Dict[str, str]:
        """Get current memory usage information."""
        memory_info = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_info[f"GPU_{i}"] = f"{allocated:.1f}/{reserved:.1f}/{total:.1f}GB (alloc/reserved/total)"
        
        return memory_info
    
    def _create_progress_bar(self, current: int, total: int, width: int = 50) -> str:
        """Create a text-based progress bar."""
        percent = current / total if total > 0 else 0
        filled_length = int(width * percent)
        bar = '█' * filled_length + '░' * (width - filled_length)
        return f"[{bar}] {percent*100:.1f}%"
    
    def _log_message(self, message: str, also_print: bool = True):
        """Log a message to file and optionally print."""
        self.file_logger.info(message)
        if also_print:
            print(message)
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        self.total_steps = args.max_steps if args.max_steps > 0 else state.max_steps
        
        # Try to log comprehensive metadata if available
        try:
            # Try to import metadata collection utilities
            try:
                from .utils import collect_training_metadata
            except ImportError:
                from src.training.utils import collect_training_metadata
            
            # Get model and other objects from kwargs
            model = kwargs.get('model', None)
            tokenizer = kwargs.get('tokenizer', None)
            train_dataset = kwargs.get('train_dataset', None)
            
            if model is not None:
                # Collect comprehensive metadata
                metadata = collect_training_metadata(
                    model=model,
                    args=self.original_args if self.original_args else {},  # Use stored args if available
                    training_args=args,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset
                )
                
                # Log metadata sections
                self._log_message("="*80)
                self._log_message("TRAINING CONFIGURATION")
                self._log_message("="*80)
                
                # Model information
                if 'Model Architecture' in metadata:
                    model_info = metadata['Model Architecture']
                    self._log_message("\nModel:")
                    self._log_message(f"  Name: {model_info.get('Model Name', 'Unknown')}")
                    self._log_message(f"  Parameters: {model_info.get('Total Parameters', 'N/A')}")
                    self._log_message(f"  Trainable: {model_info.get('Trainable Parameters', 'N/A')}")
                    if model_info.get('Model Type'):
                        self._log_message(f"  Type: {model_info['Model Type']}")
                    if model_info.get('Attention Implementation'):
                        self._log_message(f"  Attention: {model_info['Attention Implementation']}")
                
                # Training configuration
                if 'Token Budget' in metadata:
                    budget_info = metadata['Token Budget']
                    self._log_message("\nTraining Budget:")
                    self._log_message(f"  Token Budget: {budget_info.get('Total Token Budget', 'N/A')}")
                    self._log_message(f"  Tokens per Step: {budget_info.get('Tokens per Step', 'N/A')}")
                
                # Hardware information  
                if 'Hardware' in metadata:
                    hw_info = metadata['Hardware']
                    self._log_message("\nHardware:")
                    self._log_message(f"  Device: {hw_info.get('Device', 'N/A')}")
                    if hw_info.get('Device Name'):
                        self._log_message(f"  Name: {hw_info['Device Name']}")
                    if hw_info.get('Device Count', 1) > 1:
                        self._log_message(f"  Count: {hw_info['Device Count']}")
                
                # Training settings
                self._log_message("\nTraining Settings:")
                self._log_message(f"  Total Steps: {self.total_steps}")
                self._log_message(f"  Batch Size: {args.per_device_train_batch_size}")
                self._log_message(f"  Gradient Accumulation: {args.gradient_accumulation_steps}")
                self._log_message(f"  Effective Batch Size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
                # Fix max_length access - look in metadata at top level
                max_length = getattr(args, 'max_length', None) or metadata.get('max_sequence_length', 'N/A')
                self._log_message(f"  Max Length: {max_length}")
                self._log_message(f"  Learning Rate: {args.learning_rate}")
                self._log_message(f"  Warmup Steps: {args.warmup_steps}")
                # Fix mixed precision access - look in metadata at top level
                mixed_precision = metadata.get('mixed_precision', 'fp32')
                self._log_message(f"  Mixed Precision: {mixed_precision}")
                # Fix packing access - check both locations
                packing_enabled = metadata.get('sequence_packing', False)
                if not packing_enabled and 'Training Configuration' in metadata:
                    packing_enabled = metadata['Training Configuration'].get('Sequence Packing', False)
                self._log_message(f"  Sequence Packing: {'Enabled' if packing_enabled else 'Disabled'}")
                
                # Evaluation settings
                self._log_message("\nEvaluation:")
                self._log_message(f"  Strategy: {args.eval_strategy if hasattr(args, 'eval_strategy') else 'steps'}")
                if args.eval_strategy == 'steps':
                    self._log_message(f"  Eval Steps: {args.eval_steps}")
                self._log_message(f"  Save Strategy: {args.save_strategy}")
                if args.save_strategy == 'steps':
                    self._log_message(f"  Save Steps: {args.save_steps}")
                
                self._log_message("="*80 + "\n")
            else:
                # Fallback to simple logging if metadata collection fails
                self._log_simple_info(args, kwargs)
        except Exception as e:
            # Fallback to simple logging if metadata collection fails
            self._log_simple_info(args, kwargs)
    
    def _log_simple_info(self, args, kwargs):
        """Log simple training information as fallback."""
        self._log_message("="*80)
        self._log_message("TRAINING STARTED")
        self._log_message("="*80)
        self._log_message(f"Model: {kwargs.get('model', 'Unknown')}")
        self._log_message(f"Total Steps: {self.total_steps}")
        self._log_message(f"Batch Size: {args.per_device_train_batch_size}")
        self._log_message(f"Max Length: {getattr(args, 'max_length', 'N/A')}")
        self._log_message(f"Learning Rate: {args.learning_rate}")
        self._log_message(f"Warmup Steps: {args.warmup_steps}")
        self._log_message(f"Evaluation Strategy: {getattr(args, 'eval_strategy', 'steps')}")
        self._log_message(f"Save Strategy: {getattr(args, 'save_strategy', 'steps')}")
        
        # Log memory info
        memory_info = self._get_memory_info()
        if memory_info:
            self._log_message("\nMemory Information:")
            for device, info in memory_info.items():
                self._log_message(f"  {device}: {info}")
        
        self._log_message("="*80 + "\n")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        current_step = state.global_step
        
        # Track step time
        current_time = time.time()
        if len(self.step_times) > 0:
            step_time = current_time - self.step_times[-1][1]
        else:
            step_time = current_time - self.start_time
        self.step_times.append((current_step, current_time))
        
        # Limit history to last 100 steps for memory efficiency
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
        
        # Log progress at intervals
        if current_step % self.log_interval == 0 or current_step == self.total_steps:
            self._log_training_progress(state, args)
    
    def _log_training_progress(self, state: TrainerState, args: TrainingArguments):
        """Log detailed training progress."""
        current_step = state.global_step
        
        # Calculate timing statistics
        elapsed_time = time.time() - self.start_time
        steps_per_second = current_step / elapsed_time if elapsed_time > 0 else 0
        
        # Estimate remaining time
        if current_step > 0:
            avg_step_time = elapsed_time / current_step
            remaining_steps = self.total_steps - current_step
            eta_seconds = remaining_steps * avg_step_time
            eta = self._format_time(eta_seconds)
        else:
            eta = "N/A"
        
        # Get current metrics - check for different loss keys
        current_loss = 'N/A'
        if state.log_history:
            # Look through recent entries for any loss value
            for i in range(min(5, len(state.log_history))):
                entry = state.log_history[-(i+1)]
                # Check multiple possible keys
                for key in ['loss', 'train_loss', 'train/loss', 'training_loss']:
                    if key in entry:
                        current_loss = entry[key]
                        break
                if current_loss != 'N/A':
                    break
        
        current_lr = state.log_history[-1].get('learning_rate', args.learning_rate) if state.log_history else args.learning_rate
        
        # Get total tokens trained from log history (search recent entries)
        total_tokens = 0
        if state.log_history:
            # Look for the most recent total_tokens_trained in the last few log entries
            for i in range(min(10, len(state.log_history))):
                entry = state.log_history[-(i+1)]
                if 'train/total_tokens_trained' in entry:
                    total_tokens = entry['train/total_tokens_trained']
                    break
        
        # Create progress bar
        progress_bar = self._create_progress_bar(current_step, self.total_steps)
        
        # Format the log message
        log_lines = [
            "",
            f"Step {current_step}/{self.total_steps} {progress_bar}",
            f"├─ Loss: {current_loss:.4f}" if isinstance(current_loss, float) else f"├─ Loss: {current_loss}",
            f"├─ Learning Rate: {current_lr:.2e}",
        ]
        
        # Add token count if available and we're at an evaluation step or significant interval
        # Only show tokens at reasonable intervals to avoid clutter
        show_tokens = total_tokens > 0 and (
            current_step % (self.log_interval * 5) == 0 or  # Every 5x log interval
            current_step == self.total_steps or  # At the end
            (hasattr(args, 'eval_steps') and current_step % args.eval_steps == 0)  # At eval steps
        )
        
        if show_tokens:
            log_lines.append(f"├─ Total Tokens: {self._format_tokens(total_tokens)}")
        
        log_lines.extend([
            f"├─ Speed: {steps_per_second:.2f} steps/s",
            f"├─ Elapsed: {self._format_time(elapsed_time)}",
            f"└─ ETA: {eta}"
        ])
        
        # Add memory info periodically
        if current_step % (self.log_interval * 10) == 0:
            memory_info = self._get_memory_info()
            if memory_info:
                log_lines.append("\nMemory Usage:")
                for device, info in memory_info.items():
                    log_lines.append(f"  {device}: {info}")
        
        # Log all lines
        for line in log_lines:
            self._log_message(line, also_print=(current_step % (self.log_interval * 5) == 0))
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after evaluation."""
        metrics = kwargs.get("metrics", {})
        
        # Reset evaluation dataset tracking after evaluation completes
        self._current_eval_dataset = None
        
        self._log_message("\n" + "="*60)
        self._log_message("EVALUATION RESULTS")
        self._log_message("="*60)
        self._log_message(f"Step: {state.global_step}")
        
        # Process and log metrics
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                # Handle special metrics
                if 'loss' in key:
                    self._log_message(f"  {key}: {value:.4f}")
                elif 'perplexity' in key:
                    # Handle infinite perplexity gracefully
                    if np.isinf(value):
                        self._log_message(f"  {key}: inf (model not trained yet)")
                    else:
                        self._log_message(f"  {key}: {value:.2f}")
                elif 'accuracy' in key or 'f1' in key or 'precision' in key or 'recall' in key:
                    self._log_message(f"  {key}: {value:.4f}")
                else:
                    self._log_message(f"  {key}: {value:.4f}")
            else:
                self._log_message(f"  {key}: {value}")
        
        # Calculate perplexity if not provided
        if 'eval_loss' in metrics and 'eval_perplexity' not in metrics:
            try:
                perplexity = torch.exp(torch.tensor(metrics['eval_loss'])).item()
                if np.isinf(perplexity):
                    self._log_message(f"  Calculated Perplexity: inf (model not trained yet)")
                else:
                    self._log_message(f"  Calculated Perplexity: {perplexity:.2f}")
            except:
                pass
        
        self._log_message("="*60 + "\n")
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when saving a checkpoint."""
        self._log_message(f"\n💾 Checkpoint saved at step {state.global_step}")
        self._log_message(f"   Location: {args.output_dir}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        
        self._log_message("\n" + "="*80)
        self._log_message("TRAINING COMPLETED")
        self._log_message("="*80)
        self._log_message(f"Total Steps: {state.global_step}")
        self._log_message(f"Total Time: {self._format_time(total_time)}")
        self._log_message(f"Average Speed: {state.global_step/total_time:.2f} steps/s")
        
        # Add total tokens trained if available
        if state.log_history:
            total_tokens = state.log_history[-1].get('train/total_tokens_trained', 0)
            if total_tokens > 0:
                self._log_message(f"Total Tokens Trained: {self._format_tokens(total_tokens)}")
        
        # Final metrics
        if state.log_history:
            final_metrics = state.log_history[-1]
            self._log_message("\nFinal Metrics:")
            for key, value in sorted(final_metrics.items()):
                if isinstance(value, float):
                    self._log_message(f"  {key}: {value:.4f}")
        
        self._log_message("="*80 + "\n")
    
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called during prediction/evaluation steps."""
        # Get current dataset name (set by MultiEvalTrainer) or use default
        current_dataset = self._current_eval_dataset or 'eval'
        
        # Increment step counter for current dataset
        if current_dataset not in self._eval_dataset_steps:
            self._eval_dataset_steps[current_dataset] = 0
        self._eval_dataset_steps[current_dataset] += 1
        
        # Log evaluation progress periodically
        current_step = self._eval_dataset_steps[current_dataset]
        if current_step % 10 == 0:
            if current_dataset != 'eval':
                self._log_message(f"  {current_dataset} - step {current_step}...", also_print=False)
            else:
                self._log_message(f"  Evaluation step {current_step}...", also_print=False)


def setup_enhanced_logging(output_dir: str, args=None) -> logging.Logger:
    """
    Set up enhanced logging with comprehensive progress tracking.
    
    Args:
        output_dir: Directory to save logs
        args: Training arguments to save
        
    Returns:
        Logger instance and callback
    """
    # Create logs directory
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up enhanced log file
    enhanced_log_file = os.path.join(logs_dir, 'training_enhanced.log')
    
    # Create callback with original args for metadata collection
    callback = EnhancedProgressCallback(
        log_file=enhanced_log_file,
        log_interval=10,  # Log every 10 steps
        original_args=args  # Pass args for metadata collection
    )
    
    # Save arguments if provided
    if args is not None:
        # Save as JSON
        args_file = os.path.join(logs_dir, 'config.json')
        args_dict = vars(args) if hasattr(args, '__dict__') else args
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "output_dir": output_dir,
            "config": args_dict
        }
        
        with open(args_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save human-readable version
        args_txt_file = os.path.join(logs_dir, 'config.txt')
        with open(args_txt_file, 'w') as f:
            f.write(f"Training Configuration\n")
            f.write(f"{'='*70}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            f.write(f"Arguments:\n")
            f.write(f"{'-'*70}\n")
            for key, value in sorted(args_dict.items()):
                f.write(f"{key:35s}: {value}\n")
        
        callback._log_message(f"Configuration saved to {logs_dir}")
    
    # Set up standard logging as well
    standard_log = os.path.join(logs_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(standard_log),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Enhanced logging initialized")
    logging.info(f"  Enhanced log: {enhanced_log_file}")
    logging.info(f"  Standard log: {standard_log}")
    
    return logging.getLogger(__name__), callback