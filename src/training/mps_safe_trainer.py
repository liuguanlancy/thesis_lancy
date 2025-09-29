"""
MPS-safe Trainer that handles numerical instabilities on Apple Silicon.
"""
from transformers import Trainer
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any, Union
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
import warnings
from .metadata_mixin import MetadataLoggingMixin


class MPSSafeTrainer(MetadataLoggingMixin, Trainer):
    """
    Custom Trainer that handles MPS (Apple Silicon) numerical instabilities.
    
    The main issue is that MPS can produce NaN values during evaluation with
    certain batch compositions, particularly with heavy padding and DataCollatorForLanguageModeling.
    
    This trainer detects MPS device and implements workarounds to prevent NaN losses.
    """
    
    def __init__(self, *args, eval_max_batches: int = -1, training_args_dict: Optional[Dict] = None, use_packing: bool = False, **kwargs):
        """
        Initialize MPS-safe trainer.
        
        Args:
            eval_max_batches: Maximum number of batches to use during evaluation.
                             Use -1 for full evaluation dataset.
            training_args_dict: Original training arguments for metadata logging
            use_packing: Whether sequence packing is enabled
        """
        # Initialize parent classes (mixin will handle metadata setup)
        super().__init__(*args, training_args_dict=training_args_dict, use_packing=use_packing, **kwargs)
        self.eval_max_batches = eval_max_batches
        
        # Detect if we're using MPS
        self.is_mps = False
        if hasattr(self.model, 'device'):
            self.is_mps = str(self.model.device).startswith('mps')
        elif hasattr(self.args, 'device'):
            self.is_mps = str(self.args.device).startswith('mps')
        else:
            # Check if model parameters are on MPS
            try:
                device = next(self.model.parameters()).device
                self.is_mps = device.type == 'mps'
            except:
                pass
        
        if self.is_mps:
            warnings.warn(
                "MPS device detected. Using MPS-safe evaluation to prevent NaN losses. "
                "This may slightly slow down evaluation but ensures numerical stability.",
                UserWarning
            )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to handle MPS numerical instabilities.
        
        During evaluation on MPS, we move the computation to CPU to avoid NaN.
        """
        # During evaluation on MPS, handle specially
        if self.is_mps and not model.training:
            # Save original device
            original_device = next(model.parameters()).device
            
            # Option 1: Move model and inputs to CPU for evaluation
            # This is the most reliable fix
            model_cpu = model.cpu()
            inputs_cpu = {k: v.cpu() if torch.is_tensor(v) else v 
                         for k, v in inputs.items()}
            
            # Compute loss on CPU
            loss = super().compute_loss(
                model_cpu, 
                inputs_cpu, 
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch
            )
            
            # Move model back to MPS
            model.to(original_device)
            
            # Move loss back to MPS if needed
            if return_outputs:
                loss, outputs = loss
                # Move loss tensor to original device
                if torch.is_tensor(loss):
                    loss = loss.to(original_device)
                return (loss, outputs) if return_outputs else loss
            else:
                if torch.is_tensor(loss):
                    loss = loss.to(original_device)
                return loss
        
        # For training or non-MPS devices, use standard computation
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Override evaluation loop to:
        1. Limit number of batches if specified
        2. Handle MPS device specially
        """
        # Apply batch limiting if specified
        if self.eval_max_batches > 0:
            limited_dataloader = []
            for i, batch in enumerate(dataloader):
                if i >= self.eval_max_batches:
                    break
                limited_dataloader.append(batch)
            
            if self.state.is_world_process_zero:
                total_batches = len(dataloader) if hasattr(dataloader, '__len__') else 'unknown'
                print(f"Evaluating on {self.eval_max_batches} batches out of {total_batches}")
            
            # Create a custom dataloader from limited batches
            class LimitedDataLoader:
                def __init__(self, batches):
                    self.batches = batches
                
                def __iter__(self):
                    return iter(self.batches)
                
                def __len__(self):
                    return len(self.batches)
            
            dataloader = LimitedDataLoader(limited_dataloader)
        
        # Call parent evaluation loop
        # The MPS handling is done in compute_loss
        return super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


class LightweightMPSTrainer(Trainer):
    """
    A more lightweight MPS-safe trainer that only moves loss computation to CPU,
    keeping the model on MPS for faster inference.
    """
    
    def __init__(self, *args, eval_max_batches: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_max_batches = eval_max_batches
        
        # Detect MPS
        try:
            device = next(self.model.parameters()).device
            self.is_mps = device.type == 'mps'
        except:
            self.is_mps = False
        
        if self.is_mps:
            print("MPS device detected. Using lightweight MPS-safe loss computation.")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with MPS-safe handling for evaluation only.
        
        Instead of moving the entire model, we only move the loss computation.
        """
        if self.is_mps and not model.training:
            # Get model outputs on MPS (fast)
            outputs = model(**inputs)
            
            # Check if loss is NaN
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                if torch.isnan(outputs.loss):
                    # Recompute loss on CPU
                    logits = outputs.logits
                    labels = inputs.get('labels')
                    
                    if labels is not None:
                        # Move to CPU for stable loss computation
                        logits_cpu = logits.cpu()
                        labels_cpu = labels.cpu()
                        
                        # Import here to avoid circular dependency
                        from torch.nn import CrossEntropyLoss
                        
                        # Shift for causal LM (assuming this is a causal LM task)
                        shift_logits = logits_cpu[..., :-1, :].contiguous()
                        shift_labels = labels_cpu[..., 1:].contiguous()
                        
                        # Compute loss on CPU
                        loss_fct = CrossEntropyLoss(ignore_index=-100)
                        loss_cpu = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Move back to MPS
                        outputs.loss = loss_cpu.to(outputs.logits.device)
                        
                        if not torch.isnan(outputs.loss):
                            print(f"Fixed NaN loss: {outputs.loss.item():.4f}")
            
            return (outputs.loss, outputs) if return_outputs else outputs.loss
        
        # Standard computation for training or non-MPS
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)