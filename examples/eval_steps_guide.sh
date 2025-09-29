#!/bin/bash
# Examples of using eval_steps for flexible evaluation scheduling

echo "Evaluation Steps Examples"
echo "============================="
echo ""
echo "The --eval_steps option provides a convenient way to control"
echo "how often evaluation runs during training, independent of save frequency."
echo ""

# Example 1: Frequent evaluation for monitoring
echo "# Example 1: Frequent evaluation for early stopping"
echo "# Evaluate every 50 steps to closely monitor performance"
echo "python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft \\"
echo "    --eval_steps 50 --save_steps 500 --max_steps 1000"
echo ""

# Example 2: Balanced evaluation and saving
echo "# Example 2: Balanced evaluation and checkpointing"
echo "# Evaluate every 100 steps, save every 500 steps"
echo "python train.py --model Qwen/Qwen3-0.6B-Base --dataset glue --dataset_config sst2 --mode sft \\"
echo "    --eval_steps 100 --save_steps 500 --use_lora --lora_r 8"
echo ""

# Example 3: Infrequent evaluation for long training
echo "# Example 3: Infrequent evaluation for long training runs"
echo "# Evaluate every 1000 steps to reduce overhead"
echo "python train.py --model Qwen/Qwen3-1.7B --dataset openai/gsm8k --dataset_config main --mode sft \\"
echo "    --eval_steps 1000 --save_steps 2000 --max_steps 10000"
echo ""

# Example 4: Fine-grained monitoring during critical training phase
echo "# Example 4: Fine-grained monitoring for critical phase"
echo "# Very frequent evaluation during first 500 steps"
echo "python train.py --model gpt2 --dataset takala/financial_phrasebank --mode sft \\"
echo "    --eval_steps 25 --save_steps 100 --max_steps 500"
echo ""

# Example 5: Combined with early stopping
echo "# Example 5: Early stopping with frequent evaluation"
echo "# Evaluate every 50 steps and stop if no improvement"
echo "python train.py --model gpt2 --dataset stanfordnlp/imdb --mode sft \\"
echo "    --eval_steps 50 --save_steps 200 \\"
echo "    --load_best_model_at_end --metric_for_best_model eval_loss"
echo ""

# Example 6: Memory-efficient evaluation
echo "# Example 6: Less frequent evaluation for large models"
echo "# Reduce evaluation frequency to save memory"
echo "python train.py --model Qwen/Qwen3-4B-Base --dataset virattt/financial-qa-10K --mode sft \\"
echo "    --eval_steps 500 --save_steps 1000 \\"
echo "    --use_lora --lora_r 16 --batch_size 2"
echo ""

echo "# === KEY POINTS ==="
echo ""
echo "# The eval_steps parameter:"
echo "# - Automatically sets eval_strategy='steps' when specified"
echo "# - Works independently of save_steps"
echo "# - Can be combined with eval_on_start for initial evaluation"
echo ""

echo "# === BENEFITS ==="
echo "1. Simple syntax - no need to specify eval_strategy"
echo "2. Intuitive - directly specify the interval"
echo "3. Flexible monitoring - decouple evaluation from checkpointing frequency"
echo "4. Automatic strategy setting - sets eval_strategy='steps' automatically"
echo ""

echo "# === USE CASES ==="
echo "- Quick experiments: eval_steps=10-50 for close monitoring"
echo "- Standard training: eval_steps=100-500 for balanced overhead"
echo "- Long training runs: eval_steps=1000+ to reduce compute overhead"
echo "- Early stopping: eval_steps=50-100 with load_best_model_at_end"