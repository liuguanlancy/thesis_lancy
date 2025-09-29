import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from typing import List, Dict, Optional, Callable
import re


def create_simple_reward_function(dataset_name: str, dataset_config: Optional[str] = None) -> Callable:
    """Create a simple reward function based on the dataset type."""
    
    def reward_function(prompts: List[str], completions: List[str], completion_ids: Optional[List[List[int]]] = None, **kwargs) -> List[float]:
        """Simple reward function that rewards correct and well-formed outputs.
        
        Args:
            prompts: List of prompts
            completions: List of completions
            completion_ids: List of completion token IDs (optional)
            **kwargs: Additional arguments
            
        Returns:
            List of reward scores
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            reward = 0.0
            
            # Basic quality rewards
            if len(completion.strip()) > 0:
                reward += 0.1  # Basic completion reward
            
            if len(completion.split()) >= 3:
                reward += 0.1  # Encourage longer, more informative responses
            
            # Dataset-specific rewards
            if 'gsm8k' in dataset_name.lower() or 'math' in dataset_name.lower():
                # Reward mathematical reasoning patterns
                if any(word in completion.lower() for word in ['step', 'therefore', 'because', 'calculate', 'solve']):
                    reward += 0.2
                
                # Look for numerical answers
                if re.search(r'\d+', completion):
                    reward += 0.1
                    
                # Reward step-by-step solutions
                if 'step' in completion.lower() and ('1' in completion or 'first' in completion.lower()):
                    reward += 0.3
                    
            elif 'imdb' in dataset_name.lower() or 'sentiment' in dataset_name.lower():
                # Reward sentiment classification
                sentiment_words = ['positive', 'negative', 'neutral', 'good', 'bad', 'excellent', 'terrible']
                if any(word in completion.lower() for word in sentiment_words):
                    reward += 0.3
                    
            elif 'glue' in dataset_name.lower():
                if dataset_config == 'cola':
                    # Grammar acceptability
                    if any(word in completion.lower() for word in ['acceptable', 'unacceptable', 'grammatical']):
                        reward += 0.3
                elif dataset_config in ['mrpc', 'qqp']:
                    # Paraphrase/equivalence detection
                    if any(word in completion.lower() for word in ['equivalent', 'not_equivalent', 'similar', 'different']):
                        reward += 0.3
                elif dataset_config in ['mnli', 'qnli', 'rte']:
                    # Natural language inference
                    if any(word in completion.lower() for word in ['entailment', 'contradiction', 'neutral']):
                        reward += 0.3
                        
            elif 'finance' in dataset_name.lower() or 'fiqa' in dataset_name.lower():
                # Financial tasks
                finance_terms = ['financial', 'market', 'investment', 'profit', 'loss', 'revenue', 'economic']
                if any(term in completion.lower() for term in finance_terms):
                    reward += 0.2
                    
            elif 'bigcode' in dataset_name.lower() or 'code' in dataset_name.lower():
                # Code generation
                code_patterns = ['def ', 'return ', 'import ', 'class ', 'if ', 'for ', 'while ']
                if any(pattern in completion for pattern in code_patterns):
                    reward += 0.3
            
            # Penalize very long or very short responses
            completion_length = len(completion.split())
            if completion_length < 2:
                reward -= 0.2
            elif completion_length > 200:
                reward -= 0.1
                
            # Penalize repetitive responses
            words = completion.split()
            if len(words) > 1 and len(set(words)) / len(words) < 0.5:
                reward -= 0.2
                
            # Clip reward to reasonable range
            rewards.append(max(-1.0, min(1.0, reward)))
        
        return rewards
    
    return reward_function


def create_grpo_config(args) -> GRPOConfig:
    """Create GRPO configuration from command line arguments."""
    
    # Calculate generation batch size to be compatible with num_generations
    # Default num_generations is 8, so generation_batch_size must be divisible by 8
    generation_batch_size = args.batch_size
    num_generations = 8  # Default in GRPOConfig
    if generation_batch_size < num_generations:
        generation_batch_size = num_generations
    elif generation_batch_size % num_generations != 0:
        generation_batch_size = ((generation_batch_size // num_generations) + 1) * num_generations
    
    config = GRPOConfig(
        # Core GRPO parameters
        beta=args.grpo_beta,
        
        # Learning parameters
        learning_rate=args.rl_learning_rate,
        warmup_steps=args.rl_warmup_steps,
        
        # Generation parameters
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        
        # Training parameters
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
        
        # Logging and checkpointing
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=50,
        
        # Training duration
        num_train_epochs=args.num_train_epochs if args.max_steps is None else 3.0,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        
        # Device and precision (conservative settings)
        fp16=False,  # Disable for compatibility
        bf16=False,  # Disable for compatibility
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        
        # GRPO specific settings
        remove_unused_columns=False,  # Keep all columns for RL
        do_train=True,
    )
    
    return config


def prepare_rl_dataset(dataset, tokenizer, dataset_name: str, dataset_config: Optional[str] = None):
    """Prepare dataset for RL training by creating prompt-completion pairs."""
    
    def create_rl_prompt_completion(examples):
        """Create prompt-completion pairs for RL training."""
        prompts = []
        completions = []
        
        # Determine the schema and extract relevant fields
        available_cols = examples.keys()
        
        for i in range(len(list(examples.values())[0])):  # Get batch size
            # Extract example data
            example_data = {col: examples[col][i] for col in available_cols}
            
            prompt = ""
            completion = ""
            
            # Handle different dataset types
            if 'gsm8k' in dataset_name.lower() or 'math' in dataset_name.lower():
                # Mathematical reasoning
                question = example_data.get('question', '')
                answer = example_data.get('answer', '')
                prompt = f"Solve this math problem step by step:\n\nProblem: {question}\nSolution:"
                completion = answer
                
            elif 'imdb' in dataset_name.lower():
                # Sentiment classification
                text = example_data.get('text', '')
                label = example_data.get('label', 0)
                label_text = "positive" if label == 1 else "negative"
                prompt = f"Classify the sentiment of this movie review as either 'positive' or 'negative'.\n\nReview: {text}\nSentiment:"
                completion = label_text
                
            elif dataset_name.lower() == 'glue':
                # GLUE tasks
                if dataset_config == 'sst2':
                    sentence = example_data.get('sentence', '')
                    label = example_data.get('label', 0)
                    label_text = "positive" if label == 1 else "negative"
                    prompt = f"Classify the sentiment of this sentence as either 'positive' or 'negative'.\n\nSentence: {sentence}\nSentiment:"
                    completion = label_text
                elif dataset_config == 'cola':
                    sentence = example_data.get('sentence', '')
                    label = example_data.get('label', 0)
                    label_text = "acceptable" if label == 1 else "unacceptable"
                    prompt = f"Determine if this sentence is grammatically acceptable or unacceptable.\n\nSentence: {sentence}\nGrammar:"
                    completion = label_text
                elif dataset_config == 'mrpc':
                    sentence1 = example_data.get('sentence1', '')
                    sentence2 = example_data.get('sentence2', '')
                    label = example_data.get('label', 0)
                    label_text = "equivalent" if label == 1 else "not_equivalent"
                    prompt = f"Determine if these two sentences are equivalent or not_equivalent in meaning.\n\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nRelation:"
                    completion = label_text
                    
            elif 'fiqa' in dataset_name.lower() and 'LLukas22' in dataset_name:
                # Financial Q&A
                question = example_data.get('question', '')
                answer = example_data.get('answer', '')
                prompt = f"Answer the following financial question based on your knowledge:\n\nQuestion: {question}\nAnswer:"
                completion = answer
                
            elif 'finance-alpaca' in dataset_name.lower():
                # Finance instruction following
                instruction = example_data.get('instruction', '')
                input_text = example_data.get('input', '')
                output_text = example_data.get('output', '')
                if input_text:
                    prompt = f"Below is an instruction that describes a financial task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
                else:
                    prompt = f"Below is an instruction that describes a financial task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
                completion = output_text
                
            # Fallback for other datasets
            if not prompt:
                # Use first text column as prompt and try to create a reasonable completion
                text_cols = [col for col in available_cols if 'text' in col.lower() or 'sentence' in col.lower()]
                if text_cols:
                    prompt = f"Complete this text:\n\n{example_data.get(text_cols[0], '')}"
                    completion = "This is a completion."  # Simple fallback
                else:
                    prompt = "Complete this task:"
                    completion = "Task completed."
            
            prompts.append(prompt)
            completions.append(completion)
        
        return {
            'prompt': prompts,
            'completion': completions
        }
    
    # Apply the transformation
    rl_dataset = dataset.map(
        create_rl_prompt_completion,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return rl_dataset


def create_grpo_trainer(model, tokenizer, rl_dataset, grpo_config, reward_function, device_info=None):
    """Create and configure the GRPO trainer."""
    
    # Create the trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        args=grpo_config,           # Use args instead of config
        train_dataset=rl_dataset,
        reward_funcs=reward_function,  # Use reward_funcs instead of reward_function
    )
    
    print(f"GRPO Trainer created with {len(rl_dataset)} training examples")
    print(f"Model device: {next(model.parameters()).device}")
    
    return trainer


def load_reward_model(reward_model_name: str, device: str = 'auto'):
    """Load a reward model for RL training."""
    try:
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_name)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        
        # Configure tokenizer
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        
        print(f"Loaded reward model: {reward_model_name}")
        return reward_model, reward_tokenizer
        
    except Exception as e:
        print(f"Failed to load reward model {reward_model_name}: {e}")
        print("Falling back to simple reward function")
        return None, None


def create_reward_function_from_model(reward_model, reward_tokenizer, device: str = 'auto'):
    """Create a reward function using a trained reward model."""
    
    def model_reward_function(prompt: str, completion: str) -> float:
        """Reward function using a pre-trained reward model."""
        try:
            # Combine prompt and completion
            full_text = f"{prompt} {completion}"
            
            # Tokenize
            inputs = reward_tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            if device == 'cuda' and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                reward_model.cuda()
            
            # Get reward (assuming the model outputs a scalar reward)
            with torch.no_grad():
                outputs = reward_model(**inputs)
                # For simplicity, use the last hidden state mean as reward
                # In practice, you'd have a proper reward head
                reward = outputs.logits.mean().item()
                
            # Normalize reward to [-1, 1] range
            return max(-1.0, min(1.0, reward / 10.0))
            
        except Exception as e:
            print(f"Error in model reward function: {e}")
            # Fallback to simple reward
            return 0.1 if len(completion.strip()) > 0 else -0.1
    
    return model_reward_function