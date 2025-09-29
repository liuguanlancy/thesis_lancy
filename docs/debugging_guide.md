# Debugging Guide

## Using VS Code Debugger with HuggingFace Models

### Problem
When using the VS Code debugger with HuggingFace models, you may encounter persistent cache corruption errors like:
```
FileNotFoundError: [Errno 2] No such file or directory: '../../blobs/...' -> '.../tokenizer_config.json'
```

This occurs because the debugger environment interferes with HuggingFace's symlink-based caching system.

### Solution
Use the `debug_wrapper.py` script which:
1. Sets up a separate debug cache directory (`~/.cache/huggingface_debug`)
2. Configures environment variables BEFORE importing any libraries
3. Clears the debug cache on each run to prevent corruption

### How to Use

#### VS Code Debug Configuration
The `.vscode/launch.json` configurations have been updated to use `debug_wrapper.py` instead of `train.py`:

```json
{
    "name": "Debug: GPT-2 WikiText Pretrain (MPS Fix Disabled)",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/debug_wrapper.py",
    "args": [
        "--model", "gpt2",
        "--dataset", "wikitext",
        // ... your arguments
    ]
}
```

#### Manual Debugging
If running the debugger manually:
```bash
python debug_wrapper.py --model gpt2 --dataset wikitext --mode pretrain ...
```

### Setting Breakpoints for NaN Investigation

To find where NaN first appears in loss computation:

1. **In GPT2 model** (`transformers/models/gpt2/modeling_gpt2.py`):
   - Line 1211: After `loss = loss_fct(...)` - Set conditional breakpoint: `torch.isnan(loss)`
   - Line 1207-1209: Check `shift_logits` and `shift_labels` before loss computation

2. **In Trainer** (`transformers/trainer.py`):
   - In `compute_loss()`: After `outputs = model(**inputs)`
   - In `prediction_step()`: After `loss = self.compute_loss(model, inputs)`

3. **In Custom Trainer** (`src/training/custom_trainer.py`):
   - Line 66: Before `super().evaluation_loop()` to inspect first batch

### Debugging Tips

1. **Check batch composition**: When NaN occurs, inspect:
   ```python
   shift_labels.unique()  # How many are -100 (padding)?
   (shift_labels != -100).sum()  # Count of valid tokens
   shift_logits.isnan().any()  # Check for NaN in logits
   ```

2. **MPS-specific issues**: NaN often occurs on MPS with heavily padded sequences. Use `--disable_mps_fix` to reproduce the issue for debugging.

3. **Cache locations**:
   - Normal mode: `~/.cache/huggingface/`
   - Debug mode: `~/.cache/huggingface_debug/`

### Troubleshooting

If you still encounter cache issues:
1. Clear both cache directories:
   ```bash
   rm -rf ~/.cache/huggingface ~/.cache/huggingface_debug
   ```
2. Run the script once without debugger to populate cache:
   ```bash
   python train.py --model gpt2 --dataset wikitext --max_steps 1
   ```
3. Then use the debugger with `debug_wrapper.py`