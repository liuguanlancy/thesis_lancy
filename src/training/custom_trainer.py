"""
Custom Trainer with evaluation batch limiting support.
"""
from transformers import Trainer
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Any
from transformers.trainer_utils import EvalPrediction
from .metadata_mixin import MetadataLoggingMixin


class LimitedEvalTrainer(MetadataLoggingMixin, Trainer):
    """
    Custom Trainer that supports limiting the number of batches during evaluation.
    """
    
    def __init__(self, *args, eval_max_batches: int = -1, training_args_dict: Optional[Dict] = None, use_packing: bool = False, **kwargs):
        """
        Initialize trainer with eval batch limiting.
        
        Args:
            eval_max_batches: Maximum number of batches to use during evaluation.
                             Use -1 for full evaluation dataset.
            training_args_dict: Original training arguments for metadata logging
            use_packing: Whether sequence packing is enabled
        """
        super().__init__(*args, training_args_dict=training_args_dict, use_packing=use_packing, **kwargs)
        self.eval_max_batches = eval_max_batches
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalPrediction:
        """
        Override evaluation loop to limit number of batches if specified.
        """
        if self.eval_max_batches > 0:
            # Create a limited dataloader
            limited_dataloader = []
            for i, batch in enumerate(dataloader):
                if i >= self.eval_max_batches:
                    break
                limited_dataloader.append(batch)
            
            # Log the limitation
            if self.state.is_world_process_zero:
                total_batches = len(dataloader) if hasattr(dataloader, '__len__') else 'unknown'
                print(f"Evaluating on {self.eval_max_batches} batches out of {total_batches}")
            
            # Create a custom dataloader from limited batches
            # We need to wrap this properly to work with the parent evaluation_loop
            class LimitedDataLoader:
                def __init__(self, batches):
                    self.batches = batches
                
                def __iter__(self):
                    return iter(self.batches)
                
                def __len__(self):
                    return len(self.batches)
            
            dataloader = LimitedDataLoader(limited_dataloader)
        
        # Call parent evaluation loop with potentially limited dataloader
        return super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )