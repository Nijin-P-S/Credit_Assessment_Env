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
    generate_adversarial_case,
    AdversarialTracker,
    ADVERSARIAL_STRATEGIES,
    LLMAdversarialDesigner,
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
    
    # Curriculum Learning
    use_curriculum: bool = True  # Enable 3-phase curriculum learning (easy → medium → hard)
    samples_per_phase: int = 200  # Samples per curriculum phase (used if use_curriculum=True)
    
    # Adversarial Training
    use_adversarial: bool = True  # Enable adversarial training phase after curriculum
    adversarial_samples: int = 100  # Samples per adversarial round
    adversarial_rounds: int = 3  # Number of adversarial training rounds
    
    # LLM Adversarial Designer (requires API key)
    use_llm_adversarial: bool = False  # Use LLM to generate adversarial cases
    llm_provider: str = "anthropic"  # "anthropic" or "openai"
    llm_model: str = None  # Model name (default: claude-3-haiku or gpt-4o-mini)
    
    # GRPO settings
    num_generations: int = 4     # Completions per prompt for GRPO advantage calculation
    max_completion_length: int = 256
    
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

def generate_dataset(num_samples: int, seed: int = 42, difficulty: str = "all") -> Dataset:
    """
    Generate a dataset of loan applications with ground truth decisions.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        difficulty: "easy", "medium", "hard", or "all" for curriculum learning
    """
    random.seed(seed)
    
    samples = []
    for i in range(num_samples):
        # Cycle through all 3 task types for balanced training
        task_id = (i % 3) + 1
        
        applicant = generate_applicant(task_id, difficulty=difficulty)
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


def generate_adversarial_dataset(
    num_samples: int, 
    seed: int = 42, 
    tracker: AdversarialTracker = None,
    target_weakness: bool = True
) -> Dataset:
    """
    Generate a dataset of adversarial loan applications.
    
    These are specifically designed to test edge cases and exploit
    common LLM pattern-matching failures.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        tracker: Optional tracker to focus on weak areas
        target_weakness: If True and tracker provided, focus on weak areas
    """
    random.seed(seed)
    
    samples = []
    
    if tracker is not None and target_weakness:
        # Use tracker to generate targeted cases
        cases = tracker.generate_targeted_batch(num_samples, target_weakness=True)
        for case in cases:
            applicant = case["applicant"]
            strategy = case["strategy"]
            ground_truth = calculate_ground_truth(applicant)
            profile_text = build_profile_text(applicant)
            
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": profile_text}
            ]
            
            samples.append({
                "prompt": prompt,
                "ground_truth": ground_truth,
                "task_id": {"personal": 1, "vehicle": 2, "home": 3}[applicant["loan_type"]],
                "loan_type": applicant["loan_type"],
                "applicant_data": json.dumps(applicant),
                "adversarial_strategy": strategy,
            })
    else:
        # Generate random adversarial cases
        for i in range(num_samples):
            strategy = random.choice(ADVERSARIAL_STRATEGIES)
            applicant = generate_adversarial_case(strategy)
            ground_truth = calculate_ground_truth(applicant)
            profile_text = build_profile_text(applicant)
            
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": profile_text}
            ]
            
            samples.append({
                "prompt": prompt,
                "ground_truth": ground_truth,
                "task_id": {"personal": 1, "vehicle": 2, "home": 3}[applicant["loan_type"]],
                "loan_type": applicant["loan_type"],
                "applicant_data": json.dumps(applicant),
                "adversarial_strategy": strategy,
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


def format_reward_score(completion) -> float:
    """Calculate format reward for a single completion."""
    try:
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "")
        else:
            content = str(completion)
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        parsed = json.loads(content.strip())
        
        has_decision = "decision" in parsed
        has_reasoning = "reasoning" in parsed
        
        if has_decision and has_reasoning:
            return 0.2
        elif has_decision:
            return 0.1
        else:
            return -0.1
            
    except json.JSONDecodeError:
        return -0.2
    except Exception:
        return -0.1


def combined_reward(
    completions: list,
    ground_truth: list[str],
    applicant_data: list[str],
    **kwargs
) -> list[float]:
    """
    Combined reward function: 80% decision accuracy + 20% format quality.
    This is used instead of reward_weights for TRL compatibility.
    """
    decision_rewards = credit_assessment_reward(
        completions, ground_truth, applicant_data, **kwargs
    )
    format_rewards = [format_reward_score(c) for c in completions]
    
    return [0.8 * d + 0.2 * f for d, f in zip(decision_rewards, format_rewards)]


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
        reward_funcs=combined_reward,  # Combined: 80% decision + 20% format
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


def train_with_curriculum(config: TrainConfig):
    """
    Train using 3-phase curriculum learning.
    
    Phase 1 (Easy): Only obvious good/bad cases
    Phase 2 (Medium): Adds borderline cases
    Phase 3 (Hard): Adds all trap cases
    """
    print("\n" + "="*60)
    print("CURRICULUM LEARNING: 3-Phase Training")
    print("="*60)
    
    phases = [
        ("easy", "Phase 1: Learning Basics (Easy Cases)"),
        ("medium", "Phase 2: Refining (Medium Cases)"),
        ("hard", "Phase 3: Mastering (Hard Cases + Traps)"),
    ]
    
    trainer = None
    phase_results = []
    
    for phase_idx, (difficulty, phase_name) in enumerate(phases):
        print(f"\n{'='*60}")
        print(f"{phase_name}")
        print(f"{'='*60}")
        
        # Generate dataset for this phase
        train_dataset = generate_dataset(
            config.samples_per_phase, 
            seed=42 + phase_idx,
            difficulty=difficulty
        )
        eval_dataset = generate_dataset(
            config.num_eval_samples,
            seed=123,
            difficulty="all"  # Always evaluate on all difficulties
        )
        
        print(f"  Samples: {len(train_dataset)}")
        print(f"  Difficulty: {difficulty}")
        
        # Create or update trainer
        if trainer is None:
            # First phase: create new trainer
            trainer = create_trainer_with_datasets(config, train_dataset, eval_dataset)
        else:
            # Subsequent phases: update datasets, keep model
            trainer.train_dataset = train_dataset
            trainer.eval_dataset = eval_dataset
        
        # Train this phase
        print(f"\n  Training {phase_name}...")
        trainer.train()
        
        # Quick evaluation after each phase
        print(f"\n  Evaluating after {phase_name}...")
        phase_acc = quick_evaluate(trainer, eval_dataset, num_samples=10)
        phase_results.append((phase_name, phase_acc))
        print(f"  Accuracy: {phase_acc*100:.1f}%")
    
    # Print curriculum summary
    print("\n" + "="*60)
    print("CURRICULUM LEARNING RESULTS")
    print("="*60)
    for phase_name, acc in phase_results:
        print(f"  {phase_name}: {acc*100:.1f}%")
    
    return trainer, phase_results


def create_trainer_with_datasets(config: TrainConfig, train_dataset, eval_dataset):
    """Create trainer with provided datasets (used for curriculum learning)."""
    from transformers import AutoTokenizer
    from peft import LoraConfig as PeftLoraConfig
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,  # 1 epoch per phase for curriculum
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    
    peft_config = None
    if config.use_peft:
        peft_config = PeftLoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
    
    trainer = GRPOTrainer(
        model=config.model_name,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=combined_reward,
    )
    
    return trainer


def quick_evaluate(trainer, dataset, num_samples=10):
    """Quick evaluation for curriculum phase tracking."""
    correct = 0
    total = 0
    tokenizer = trainer.processing_class
    model = trainer.model
    
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        
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
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            parsed = json.loads(response.strip())
            decision = parsed.get("decision", "").lower()
            
            if decision == sample["ground_truth"]:
                correct += 1
            total += 1
        except:
            total += 1
    
    return correct / total if total > 0 else 0


def evaluate_adversarial(trainer, tracker: AdversarialTracker, num_samples: int = 30) -> dict:
    """
    Evaluate model on adversarial cases and update tracker.
    
    This identifies which strategies the model struggles with,
    enabling targeted training in the next round.
    """
    print("\n  Evaluating on adversarial cases...")
    
    tokenizer = trainer.processing_class
    model = trainer.model
    
    results_by_strategy = {s: {"correct": 0, "total": 0} for s in ADVERSARIAL_STRATEGIES}
    
    for strategy in ADVERSARIAL_STRATEGIES:
        # Generate a few samples per strategy
        samples_per_strategy = max(1, num_samples // len(ADVERSARIAL_STRATEGIES))
        
        for _ in range(samples_per_strategy):
            applicant = generate_adversarial_case(strategy)
            ground_truth = calculate_ground_truth(applicant)
            profile_text = build_profile_text(applicant)
            
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": profile_text}
            ]
            
            prompt_text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                
                parsed = json.loads(response.strip())
                decision = parsed.get("decision", "").lower()
                
                is_correct = (decision == ground_truth)
                tracker.record_result(strategy, is_correct)
                
                results_by_strategy[strategy]["total"] += 1
                if is_correct:
                    results_by_strategy[strategy]["correct"] += 1
                    
            except Exception:
                # Parse failure counts as incorrect
                tracker.record_result(strategy, False)
                results_by_strategy[strategy]["total"] += 1
    
    # Print summary
    print("  Adversarial accuracy by strategy:")
    for strategy, results in results_by_strategy.items():
        if results["total"] > 0:
            acc = results["correct"] / results["total"] * 100
            print(f"    {strategy}: {results['correct']}/{results['total']} ({acc:.0f}%)")
    
    return results_by_strategy


def train_with_adversarial(config: TrainConfig, trainer=None):
    """
    Adversarial self-play training loop.
    
    This is the key differentiator for Theme #4 (Self-Improvement):
    1. Evaluate model on adversarial cases
    2. Identify weaknesses
    3. Generate targeted training data
    4. Train on hard cases
    5. Repeat
    
    Supports two modes:
    - Rule-based (default): Uses predefined adversarial strategies
    - LLM-powered: Uses Claude/GPT to design novel trap cases
    
    Args:
        config: Training configuration
        trainer: Optional existing trainer (from curriculum learning)
    
    Returns:
        trainer: Updated trainer
        results: Dictionary of round-by-round results
    """
    print("\n" + "="*60)
    print("ADVERSARIAL SELF-PLAY TRAINING")
    print("="*60)
    print(f"  Rounds: {config.adversarial_rounds}")
    print(f"  Samples per round: {config.adversarial_samples}")
    
    # Initialize LLM adversarial designer if configured
    llm_designer = None
    if config.use_llm_adversarial:
        try:
            llm_designer = LLMAdversarialDesigner(
                provider=config.llm_provider,
                model=config.llm_model
            )
            print(f"  LLM Adversarial: {config.llm_provider} ({llm_designer.model})")
        except Exception as e:
            print(f"  Warning: LLM adversarial failed to initialize: {e}")
            print(f"  Falling back to rule-based adversarial")
    else:
        print(f"  Mode: Rule-based (10 predefined strategies)")
    
    # Initialize tracker
    tracker = AdversarialTracker()
    
    # Create trainer if not provided
    if trainer is None:
        print("\n  Creating initial trainer...")
        eval_dataset = generate_dataset(config.num_eval_samples, seed=123, difficulty="all")
        train_dataset = generate_adversarial_dataset(
            config.adversarial_samples, 
            seed=42, 
            tracker=None,  # First round: random adversarial
            target_weakness=False
        )
        trainer = create_trainer_with_datasets(config, train_dataset, eval_dataset)
    
    round_results = []
    
    for round_idx in range(config.adversarial_rounds):
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL ROUND {round_idx + 1}/{config.adversarial_rounds}")
        print(f"{'='*60}")
        
        # Step 1: Evaluate current model on adversarial cases
        eval_results = evaluate_adversarial(trainer, tracker, num_samples=30)
        
        # Step 2: Identify weakness
        weakness = tracker.get_weakness()
        weakness_rate = tracker.get_weakness_rate(weakness)
        print(f"\n  Identified weakness: {weakness} (failure rate: {weakness_rate*100:.0f}%)")
        
        # Step 3: Generate targeted adversarial dataset
        print(f"  Generating targeted training data...")
        
        if llm_designer is not None:
            # Use LLM to generate novel adversarial cases
            print(f"  Using LLM adversarial designer...")
            llm_cases = llm_designer.generate_batch(
                config.adversarial_samples // 2,  # Half from LLM
                tracker=tracker
            )
            
            # Convert LLM cases to dataset format
            llm_samples = []
            for case in llm_cases:
                gt = calculate_ground_truth(case)
                profile = build_profile_text(case)
                llm_samples.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": profile}
                    ],
                    "ground_truth": gt,
                    "task_id": {"personal": 1, "vehicle": 2, "home": 3}[case["loan_type"]],
                    "loan_type": case["loan_type"],
                    "applicant_data": json.dumps(case),
                    "adversarial_strategy": case.get("trap_type", "llm_generated"),
                })
            
            # Other half from rule-based
            rule_based_dataset = generate_adversarial_dataset(
                config.adversarial_samples // 2,
                seed=42 + round_idx + 100,
                tracker=tracker,
                target_weakness=True
            )
            
            adversarial_samples = llm_samples + list(rule_based_dataset)
            random.shuffle(adversarial_samples)
            adversarial_dataset = Dataset.from_list(adversarial_samples)
        else:
            # Use rule-based adversarial generation
            adversarial_dataset = generate_adversarial_dataset(
                config.adversarial_samples,
                seed=42 + round_idx + 100,
                tracker=tracker,
                target_weakness=True  # Focus on weak areas
            )
        
        # Also include some normal hard cases for balance
        normal_dataset = generate_dataset(
            config.adversarial_samples // 2,
            seed=42 + round_idx + 200,
            difficulty="hard"
        )
        
        # Combine datasets
        combined_samples = list(adversarial_dataset) + list(normal_dataset)
        random.shuffle(combined_samples)
        train_dataset = Dataset.from_list(combined_samples)
        
        # Step 4: Update trainer and train
        trainer.train_dataset = train_dataset
        print(f"\n  Training on {len(train_dataset)} samples (70% adversarial, 30% hard)...")
        trainer.train()
        
        # Step 5: Measure improvement
        print(f"\n  Measuring improvement...")
        post_eval = evaluate_adversarial(trainer, tracker, num_samples=20)
        
        # Calculate round accuracy
        total_correct = sum(r["correct"] for r in post_eval.values())
        total_samples = sum(r["total"] for r in post_eval.values())
        round_acc = total_correct / total_samples if total_samples > 0 else 0
        
        round_results.append({
            "round": round_idx + 1,
            "weakness_targeted": weakness,
            "accuracy": round_acc,
            "details": post_eval,
        })
        
        print(f"\n  Round {round_idx + 1} complete: {round_acc*100:.1f}% accuracy on adversarial cases")
    
    # Final summary
    print("\n" + "="*60)
    print("ADVERSARIAL TRAINING SUMMARY")
    print("="*60)
    print("\nProgress by round:")
    for result in round_results:
        print(f"  Round {result['round']}: {result['accuracy']*100:.1f}% (targeted: {result['weakness_targeted']})")
    
    print("\nFinal weakness analysis:")
    summary = tracker.get_summary()
    for strategy, data in sorted(summary.items(), key=lambda x: x[1]["accuracy"]):
        print(f"  {strategy}: {data['accuracy']*100:.0f}% ({data['failures']} failures)")
    
    return trainer, {
        "rounds": round_results,
        "final_summary": summary,
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
    
    # Print training info
    print(f"\nTraining configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Curriculum Learning: {config.use_curriculum}")
    if config.use_curriculum:
        print(f"  Samples per phase: {config.samples_per_phase}")
        print(f"  Total phases: 3 (easy → medium → hard)")
    else:
        print(f"  Training samples: {config.num_train_samples}")
    print(f"  Adversarial Training: {config.use_adversarial}")
    if config.use_adversarial:
        print(f"  Adversarial rounds: {config.adversarial_rounds}")
        print(f"  Adversarial samples per round: {config.adversarial_samples}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Using LoRA: {config.use_peft}")
    
    trainer = None
    eval_results = {}
    
    # Phase 1: Curriculum Learning (if enabled)
    if config.use_curriculum:
        trainer, phase_results = train_with_curriculum(config)
        eval_results["curriculum_phases"] = phase_results
    else:
        # Standard training (no curriculum)
        trainer = create_trainer(config)
        print("\nStarting training...")
        trainer.train()
        eval_results["standard_training"] = evaluate_model(trainer)
    
    # Phase 2: Adversarial Training (if enabled)
    if config.use_adversarial:
        trainer, adversarial_results = train_with_adversarial(config, trainer)
        eval_results["adversarial_training"] = adversarial_results
    
    # Save final model
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model()
    
    # Final evaluation on mixed test set
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    final_eval = evaluate_model(trainer, num_samples=30)
    eval_results["final_evaluation"] = final_eval
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    return trainer, eval_results


if __name__ == "__main__":
    main()
