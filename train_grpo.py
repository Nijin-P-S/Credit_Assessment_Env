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
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Dataset
    num_train_samples: int = 500  # Number of training samples to generate
    num_eval_samples: int = 50   # Number of evaluation samples
    
    # Curriculum Learning
    use_curriculum: bool = True  # Enable 3-phase curriculum learning (easy → medium → hard)
    samples_per_phase: int = 200  # Samples per curriculum phase (used if use_curriculum=True)
    phase_mastery_threshold: float = 0.65  # Min accuracy required to advance to next phase
    max_phase_retries: int = 2  # Max extra training attempts per phase before forced advance
    
    # Adversarial Training
    use_adversarial: bool = True  # Enable adversarial training phase after curriculum
    adversarial_samples: int = 100  # Samples per adversarial round
    adversarial_rounds: int = 3  # Number of adversarial training rounds
    use_self_generation: bool = True  # Model generates its own hard cases each round
    
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

SELF_GEN_PROMPT = """You are a loan underwriting trainer designing test cases for an AI loan officer.

Design ONE loan application that looks approvable at first glance but contains exactly one hidden flaw that should trigger rejection or a different decision.

The model currently struggles most with:
{weaknesses}

Output ONLY a JSON object — no explanation, no markdown:
{{
    "loan_type": "personal" | "vehicle" | "home",
    "credit_score": <integer 650-850>,
    "monthly_income": <integer 50000-1000000>,
    "foir": <float 0.15-0.60>,
    "employment_years": <integer 0-20>,
    "loan_amount": <integer 100000-20000000>,
    "documents_complete": true | false,
    "collateral_value": <integer, required for vehicle/home>,
    "ltv_ratio": <float 0.5-0.95, required for vehicle/home>,
    "rera_registered": true | false (required for home),
    "has_co_applicant": true | false,
    "trap_type": "<one line: what is the hidden flaw>"
}}"""


def self_generate_adversarial_cases(trainer, tracker, num_cases: int) -> list:
    """
    Prompt the model being trained to generate its own hard cases.

    The model uses its internalized knowledge of loan rules to design traps —
    cases it believes are tricky. These are then verified with calculate_ground_truth
    and fed back as training data in the next round, closing the self-improvement loop.
    """
    model = trainer.model
    tokenizer = trainer.processing_class

    weakness_summary = tracker.get_summary()
    top_weaknesses = sorted(weakness_summary.items(), key=lambda x: -x[1].get("failures", 0))[:3]
    weakness_str = "\n".join(
        f"- {s}: {d['failures']} failures" for s, d in top_weaknesses
    ) if top_weaknesses else "- No weakness data yet — generate varied trap cases"

    prompt_text = SELF_GEN_PROMPT.format(weaknesses=weakness_str)
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    samples = []
    attempts = 0
    max_attempts = num_cases * 4

    model.eval()
    with torch.no_grad():
        while len(samples) < num_cases and attempts < max_attempts:
            attempts += 1
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]

                case = json.loads(response.strip())

                required = ["loan_type", "credit_score", "monthly_income", "foir",
                            "employment_years", "loan_amount", "documents_complete"]
                if not all(k in case for k in required):
                    continue
                if case["loan_type"] not in ("personal", "vehicle", "home"):
                    continue

                if case["loan_type"] in ("vehicle", "home") and "collateral_value" not in case:
                    case["collateral_value"] = int(case["loan_amount"] * 1.3)
                    case["ltv_ratio"] = case["loan_amount"] / case["collateral_value"]
                if case["loan_type"] == "home":
                    case.setdefault("rera_registered", True)
                    case.setdefault("has_co_applicant", False)
                    case.setdefault("property_type", "apartment")
                if case["loan_type"] == "vehicle":
                    case.setdefault("vehicle_type", "sedan")

                gt = calculate_ground_truth(case)
                if gt is None:
                    continue

                profile = build_profile_text(case)
                samples.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": profile},
                    ],
                    "ground_truth": gt,
                    "task_id": {"personal": 1, "vehicle": 2, "home": 3}[case["loan_type"]],
                    "loan_type": case["loan_type"],
                    "applicant_data": json.dumps(case),
                    "adversarial_strategy": case.get("trap_type", "self_generated"),
                })
            except Exception:
                continue

    model.train()
    return samples


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
    Train using 3-phase curriculum learning with performance-gated advancement.

    Each phase repeats up to max_phase_retries times if accuracy stays below
    phase_mastery_threshold, ensuring the model earns advancement rather than
    advancing on a fixed timer. The final phase has no gate since there is
    nowhere left to advance.
    """
    print("\n" + "="*60)
    print("CURRICULUM LEARNING: Performance-Gated 3-Phase Training")
    print("="*60)
    print(f"  Mastery threshold: {config.phase_mastery_threshold*100:.0f}%")
    print(f"  Max retries per phase: {config.max_phase_retries}")

    # (difficulty, label, mastery_threshold)
    # Last phase has no gate — model trains through it regardless
    phases = [
        ("easy",   "Phase 1: Learning Basics (Easy Cases)",        config.phase_mastery_threshold),
        ("medium", "Phase 2: Refining (Medium Cases)",             config.phase_mastery_threshold),
        ("hard",   "Phase 3: Mastering (Hard Cases + Traps)",      0.0),
    ]

    trainer = None
    phase_results = []

    for phase_idx, (difficulty, phase_name, threshold) in enumerate(phases):
        print(f"\n{'='*60}")
        print(f"{phase_name}")
        print(f"{'='*60}")

        eval_dataset = generate_dataset(
            config.num_eval_samples,
            seed=123,
            difficulty="all"
        )

        phase_acc = 0.0
        for attempt in range(config.max_phase_retries + 1):
            if attempt > 0:
                print(f"\n  [Retry {attempt}/{config.max_phase_retries}] "
                      f"Accuracy {phase_acc*100:.1f}% < {threshold*100:.0f}% threshold — "
                      f"repeating phase with fresh samples...")

            train_dataset = generate_dataset(
                config.samples_per_phase,
                seed=42 + phase_idx * 100 + attempt,
                difficulty=difficulty
            )
            print(f"  Samples: {len(train_dataset)}  |  Difficulty: {difficulty}"
                  + (f"  |  Attempt: {attempt+1}/{config.max_phase_retries+1}" if config.max_phase_retries > 0 else ""))

            if trainer is None:
                trainer = create_trainer_with_datasets(config, train_dataset, eval_dataset)
            else:
                trainer.train_dataset = train_dataset
                trainer.eval_dataset = eval_dataset

            print(f"\n  Training...")
            trainer.train()

            print(f"\n  Evaluating...")
            phase_acc = quick_evaluate(trainer, eval_dataset, num_samples=10)
            print(f"  Accuracy: {phase_acc*100:.1f}%"
                  + (f" (threshold: {threshold*100:.0f}%)" if threshold > 0 else " (no gate on final phase)"))

            if threshold == 0.0 or phase_acc >= threshold:
                if threshold > 0:
                    print(f"  Mastery achieved — advancing to next phase.")
                break
        else:
            print(f"  Threshold not reached after {config.max_phase_retries+1} attempts — advancing anyway.")

        phase_results.append((phase_name, phase_acc))

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

    Each round:
    1. Evaluate model on adversarial cases to identify weaknesses
    2. Generate targeted rule-based cases focused on those weaknesses
    3. Optionally mix in self-generated cases from the previous round
    4. Train on the combined hard dataset
    5. Model generates its own hard cases for the next round

    The self-generation loop (step 5 → step 3) is what makes this recursive:
    the model's own failure patterns shape what it trains on next.
    """
    print("\n" + "="*60)
    print("ADVERSARIAL SELF-PLAY TRAINING")
    print("="*60)
    print(f"  Rounds: {config.adversarial_rounds}")
    print(f"  Samples per round: {config.adversarial_samples}")
    print(f"  Self-generation: {config.use_self_generation}")

    tracker = AdversarialTracker()

    if trainer is None:
        print("\n  Creating initial trainer...")
        eval_dataset = generate_dataset(config.num_eval_samples, seed=123, difficulty="all")
        train_dataset = generate_adversarial_dataset(
            config.adversarial_samples,
            seed=42,
            tracker=None,
            target_weakness=False
        )
        trainer = create_trainer_with_datasets(config, train_dataset, eval_dataset)

    round_results = []
    self_gen_carry = []  # Self-generated cases from previous round

    for round_idx in range(config.adversarial_rounds):
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL ROUND {round_idx + 1}/{config.adversarial_rounds}")
        print(f"{'='*60}")

        # Step 1: Evaluate and identify weakness
        evaluate_adversarial(trainer, tracker, num_samples=30)
        weakness = tracker.get_weakness()
        weakness_rate = tracker.get_weakness_rate(weakness)
        print(f"\n  Identified weakness: {weakness} (failure rate: {weakness_rate*100:.0f}%)")

        # Step 2: Rule-based targeted dataset
        print(f"  Generating targeted training data...")
        adversarial_dataset = generate_adversarial_dataset(
            config.adversarial_samples,
            seed=42 + round_idx + 100,
            tracker=tracker,
            target_weakness=True
        )
        adversarial_samples = list(adversarial_dataset)

        # Step 3: Mix in self-generated cases from previous round (capped at 30%)
        if self_gen_carry:
            max_carry = max(1, len(adversarial_samples) * 3 // 10)
            carry_to_use = self_gen_carry[:max_carry]
            adversarial_samples = carry_to_use + adversarial_samples
            print(f"  Mixed in {len(carry_to_use)} self-generated cases from previous round")

        normal_dataset = generate_dataset(
            config.adversarial_samples // 2,
            seed=42 + round_idx + 200,
            difficulty="hard"
        )

        combined_samples = adversarial_samples + list(normal_dataset)
        random.shuffle(combined_samples)
        train_dataset = Dataset.from_list(combined_samples)

        # Step 4: Train
        trainer.train_dataset = train_dataset
        print(f"\n  Training on {len(train_dataset)} samples...")
        trainer.train()

        # Step 5: Measure improvement
        print(f"\n  Measuring improvement...")
        post_eval = evaluate_adversarial(trainer, tracker, num_samples=20)

        total_correct = sum(r["correct"] for r in post_eval.values())
        total_samples = sum(r["total"] for r in post_eval.values())
        round_acc = total_correct / total_samples if total_samples > 0 else 0

        # Step 6: Model generates its own hard cases for the next round
        if config.use_self_generation:
            print(f"\n  Self-generating hard cases for next round...")
            self_gen_carry = self_generate_adversarial_cases(
                trainer, tracker, num_cases=config.adversarial_samples // 3
            )
            print(f"  Generated {len(self_gen_carry)} valid self-generated cases")

        round_results.append({
            "round": round_idx + 1,
            "weakness_targeted": weakness,
            "accuracy": round_acc,
            "self_generated": len(self_gen_carry) if config.use_self_generation else 0,
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
        print(f"  Mastery threshold: {config.phase_mastery_threshold*100:.0f}% (max retries: {config.max_phase_retries})")
    else:
        print(f"  Training samples: {config.num_train_samples}")
    print(f"  Adversarial Training: {config.use_adversarial}")
    if config.use_adversarial:
        print(f"  Adversarial rounds: {config.adversarial_rounds}")
        print(f"  Adversarial samples per round: {config.adversarial_samples}")
        print(f"  Self-generation: {config.use_self_generation}")
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
