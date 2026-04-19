"""
GRPO Training Script for Credit Assessment Environment
=======================================================

This script trains an LLM to make accurate loan underwriting decisions using
Group Relative Policy Optimization (GRPO) from HuggingFace TRL.

The agent learns to:
- Follow RBI guidelines (CIBIL score, FOIR, LTV ratios)
- Detect trap cases (e.g., perfect profile with one hidden flaw)
- Make correct decisions: approve, reject, request_docs, counter_offer

Usage (Local):
    pip install trl transformers datasets accelerate peft bitsandbytes
    python train_grpo.py

Usage (Colab with GPU):
    See the companion notebook: train_grpo_colab.ipynb

Environment Variables:
    HF_TOKEN: HuggingFace token for pushing to hub (optional)
    WANDB_API_KEY: Weights & Biases API key for logging (optional)
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Use standalone training utilities to avoid import issues
from train_utils import (
    generate_applicant,
    calculate_ground_truth,
    calculate_reward,
    build_profile_text,
    CreditAssessmentAction,
    LoanDecision,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Small model for fast iteration
    
    # Dataset
    num_train_samples: int = 500  # Number of training samples to generate
    num_eval_samples: int = 50   # Number of evaluation samples
    
    # GRPO settings
    num_generations: int = 4     # Completions per prompt for GRPO advantage calculation
    max_completion_length: int = 256
    max_prompt_length: int = 1024
    
    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    
    # LoRA (Parameter Efficient Fine-Tuning)
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Output
    output_dir: str = "./grpo_credit_assessment"
    push_to_hub: bool = False
    hub_model_id: str = None
    
    # Logging
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 100


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a senior loan officer at an Indian bank. You assess loan applications following RBI guidelines and standard banking norms.

You must respond with a JSON object containing:
- "decision": one of "approve", "reject", "request_docs", "counter_offer"
- "reasoning": a brief explanation for your decision
- "counter_offer_amount": (only if decision is "counter_offer") the reduced loan amount
- "docs_requested": (only if decision is "request_docs") what documents are needed

Key guidelines:
- CIBIL score below 700 → reject
- FOIR above 50% → reject
- Incomplete documents → request_docs
- For home loans: non-RERA property → reject regardless of other factors
- For vehicle loans: LTV above 85% → counter_offer with reduced amount
- For home loans: LTV limits are tiered by RBI:
  - Loan ≤ ₹30L → max LTV 90%
  - Loan ₹30-75L → max LTV 80%
  - Loan > ₹75L → max LTV 75%
- Employment: Personal/Vehicle loans need 1+ years, Home loans need 2+ years

IMPORTANT: Check EVERY criterion carefully. A single failing criterion means rejection, even if everything else is perfect.

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_dataset(num_samples: int, seed: int = 42) -> Dataset:
    """Generate a dataset of loan applications with ground truth decisions."""
    random.seed(seed)
    
    samples = []
    for i in range(num_samples):
        # Cycle through all 3 task types for balanced training
        task_id = (i % 3) + 1
        
        applicant = generate_applicant(task_id)
        ground_truth = calculate_ground_truth(applicant)
        profile_text = build_profile_text(applicant)
        
        # Format as chat messages
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": profile_text}
        ]
        
        samples.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "task_id": task_id,
            "loan_type": applicant["loan_type"],
            "applicant_data": json.dumps(applicant),
        })
    
    return Dataset.from_list(samples)


# =============================================================================
# Reward Function
# =============================================================================

def credit_assessment_reward(
    completions: list[list[dict]],
    ground_truth: list[str],
    applicant_data: list[str],
    **kwargs
) -> list[float]:
    """
    Custom reward function that evaluates loan decisions against ground truth.
    
    This function:
    1. Parses the model's JSON output
    2. Creates a CreditAssessmentAction
    3. Computes reward using the environment's reward function
    4. Returns normalized rewards for GRPO training
    
    Args:
        completions: List of model completions (conversational format)
        ground_truth: List of correct decisions
        applicant_data: JSON-serialized applicant dictionaries
        
    Returns:
        List of reward values in range [-1, 1]
    """
    rewards = []
    
    for completion, gt, applicant_json in zip(completions, ground_truth, applicant_data):
        try:
            # Extract content from completion
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0].get("content", "")
            else:
                content = str(completion)
            
            # Parse JSON from model output
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            parsed = json.loads(content.strip())
            
            # Extract decision
            decision_str = parsed.get("decision", "reject").lower()
            try:
                decision = LoanDecision(decision_str)
            except ValueError:
                decision = LoanDecision.REJECT
            
            # Create action
            action = CreditAssessmentAction(
                decision=decision,
                reasoning=parsed.get("reasoning", ""),
                counter_offer_amount=parsed.get("counter_offer_amount"),
                docs_requested=parsed.get("docs_requested"),
            )
            
            # Load applicant data
            applicant = json.loads(applicant_json)
            
            # Calculate raw reward using environment's reward function
            raw_reward = calculate_reward(action, applicant, gt)
            
            # Normalize reward from [-20, +10] to [-1, +1]
            normalized = (raw_reward - (-20.0)) / (10.0 - (-20.0)) * 2 - 1
            rewards.append(normalized)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Invalid JSON or missing fields → penalty
            rewards.append(-0.5)
        except Exception as e:
            # Unexpected error → moderate penalty
            rewards.append(-0.3)
    
    return rewards


def format_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Additional reward for proper JSON formatting.
    
    Rewards:
    - Valid JSON with all required fields: +0.2
    - Valid JSON but missing fields: +0.1
    - Invalid JSON: -0.2
    """
    rewards = []
    
    for completion in completions:
        try:
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0].get("content", "")
            else:
                content = str(completion)
            
            # Clean up potential markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            parsed = json.loads(content.strip())
            
            # Check required fields
            has_decision = "decision" in parsed
            has_reasoning = "reasoning" in parsed
            
            if has_decision and has_reasoning:
                rewards.append(0.2)
            elif has_decision:
                rewards.append(0.1)
            else:
                rewards.append(-0.1)
                
        except json.JSONDecodeError:
            rewards.append(-0.2)
        except Exception:
            rewards.append(-0.1)
    
    return rewards


# =============================================================================
# Training
# =============================================================================

def create_trainer(config: TrainConfig) -> GRPOTrainer:
    """Create and configure the GRPO trainer."""
    
    print(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate datasets
    print(f"Generating {config.num_train_samples} training samples...")
    train_dataset = generate_dataset(config.num_train_samples, seed=42)
    
    print(f"Generating {config.num_eval_samples} evaluation samples...")
    eval_dataset = generate_dataset(config.num_eval_samples, seed=123)
    
    # Configure GRPO
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        
        # GRPO specific
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        
        # Logging
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        
        # Memory optimization
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        
        # Push to hub
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        
        # Report to wandb if available
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    )
    
    # Configure LoRA if enabled
    peft_config = None
    if config.use_peft:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=config.model_name,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=[credit_assessment_reward, format_reward],
        reward_weights=[0.8, 0.2],  # 80% decision accuracy, 20% format
    )
    
    return trainer


def evaluate_model(trainer: GRPOTrainer, num_samples: int = 20) -> dict:
    """Evaluate the trained model on fresh samples."""
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Generate fresh test samples
    test_dataset = generate_dataset(num_samples, seed=999)
    
    correct = 0
    total = 0
    results_by_task = {1: {"correct": 0, "total": 0}, 
                       2: {"correct": 0, "total": 0}, 
                       3: {"correct": 0, "total": 0}}
    
    tokenizer = trainer.processing_class
    model = trainer.model
    
    for sample in test_dataset:
        prompt = tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            # Parse JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            parsed = json.loads(response.strip())
            decision = parsed.get("decision", "").lower()
            
            task_id = sample["task_id"]
            is_correct = (decision == sample["ground_truth"])
            
            if is_correct:
                correct += 1
                results_by_task[task_id]["correct"] += 1
            
            total += 1
            results_by_task[task_id]["total"] += 1
            
        except Exception:
            total += 1
            results_by_task[sample["task_id"]]["total"] += 1
    
    # Print results
    overall_acc = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {correct}/{total} ({overall_acc*100:.1f}%)")
    
    task_names = {1: "Personal Loan", 2: "Vehicle Loan", 3: "Home Loan"}
    for task_id, results in results_by_task.items():
        if results["total"] > 0:
            acc = results["correct"] / results["total"]
            print(f"  {task_names[task_id]}: {results['correct']}/{results['total']} ({acc*100:.1f}%)")
    
    return {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "by_task": results_by_task,
    }


def main():
    """Main training loop."""
    print("="*60)
    print("Credit Assessment Environment - GRPO Training")
    print("="*60)
    
    # Configuration
    config = TrainConfig()
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. Training will be slow.")
        config.per_device_train_batch_size = 1
        config.gradient_accumulation_steps = 8
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Print training info
    print(f"\nTraining configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Training samples: {config.num_train_samples}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Using LoRA: {config.use_peft}")
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model()
    
    # Evaluate
    eval_results = evaluate_model(trainer)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    return trainer, eval_results


if __name__ == "__main__":
    main()
