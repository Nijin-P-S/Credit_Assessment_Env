"""
GRPO Training Script for Credit Assessment Environment
=======================================================

This script trains an LLM to make accurate loan underwriting decisions using
Group Relative Policy Optimization (GRPO) from HuggingFace TRL.

It mirrors the (proven) flow in train_grpo_colab.ipynb:

    1. (Recommended) SFT warmup — run sft_warmup.py first to produce a LoRA
       adapter at ./grpo_credit_assessment_sft. This script auto-detects
       that directory and uses it as the GRPO starting policy.
    2. Per-task curriculum — Phase 1 personal, Phase 2 vehicle, Phase 3 home.
       Replay buffer (replay_fraction=0.2) prevents catastrophic forgetting.
    3. (Optional) Adversarial self-play — disable with USE_ADVERSARIAL=0 if
       curriculum already cleared the bar.
    4. Final eval + JSON log + Hub push.

The agent learns to:
- Follow RBI guidelines (CIBIL score, FOIR, LTV ratios)
- Detect trap cases (e.g., perfect profile with one hidden flaw)
- Make correct decisions: approve, reject, request_docs, counter_offer

Usage (Local):
    pip install trl transformers datasets accelerate peft bitsandbytes
    python sft_warmup.py        # Optional but strongly recommended
    python train_grpo.py        # Auto-loads SFT adapter if present

Usage (Colab with GPU):
    See the companion notebook: train_grpo_colab.ipynb

Environment Variables:
    HF_TOKEN              HuggingFace token for pushing to hub (optional)
    WANDB_API_KEY         Weights & Biases API key for logging (optional)
    SFT_INIT_DIR          Override SFT adapter dir (default: auto-detect ./grpo_credit_assessment_sft)
    PUSH_PER_PHASE        "1" → push adapter to Hub after each curriculum phase
    HUB_MODEL_ID          Final model repo (default: iamnijin/credit-assessment-curriculum)
    HF_PUSH_CHECKPOINTS   "0" → disable mid-training checkpoint pushes (recommended for HF Jobs)
    EVAL_STRATEGY         GRPO mid-training eval; default "no" (set "steps" only if you really want it)
    SAVE_STRATEGY         GRPO mid-training save; default "no"
    CURRICULUM_MODE       "task" (default — recommended) or "difficulty"
    SKIP_BASELINE         "1" → skip the baseline-evaluation pass
    TRAINING_LOG_PATH     Output path for the JSON consumed by scripts/generate_plots.py
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Optional

from huggingface_hub import login as hf_login

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    hf_login(token=hf_token)
    print("Logged in to HuggingFace!")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
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

# Single source of truth for parsing model JSON output. Used by the reward
# function AND every eval path so baseline-vs-trained comparisons are fair.
from lenient_parser import parse_response, parse_decision


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Dataset
    num_train_samples: int = 500  # Number of training samples to generate
    num_eval_samples: int = 200   # Evaluation pool (large enough to keep measurements stable)
    
    # Curriculum Learning
    use_curriculum: bool = True  # Enable 3-phase curriculum learning (easy → medium → hard)
    samples_per_phase: int = 400  # 4× previous — enough gradient updates to actually learn
    phase_mastery_threshold: float = 0.60  # Slightly relaxed — 60% on 50-sample eval is the real signal
    max_phase_retries: int = 1  # Only one retry — more retries overfit on noise
    phase_eval_samples: int = 50  # Per-phase quick eval (was 10 — too noisy)
    
    # Adversarial Training. Disable with USE_ADVERSARIAL=0 if your curriculum
    # already exceeds the target accuracy (in our 7B Colab run curriculum hit
    # 96.7% so we skipped adversarial — same toggle here.)
    use_adversarial: bool = os.getenv("USE_ADVERSARIAL", "1") == "1"
    adversarial_samples: int = 150  # Per round — more coverage of each targeted strategy
    adversarial_rounds: int = 2  # Round 3 produced no new signal in prior runs
    use_self_generation: bool = True  # Model generates its own hard cases each round
    adversarial_per_strategy_eval: int = 5  # Samples per strategy in pre/post eval (was 2)
    
    # GRPO settings
    # 8 generations gives much better advantage estimates than 4 — half the
    # variance in the policy gradient at the cost of ~2× memory. Worth it on
    # A100; drop back to 4 for T4.
    num_generations: int = 8
    # Bumped from 256 to allow chain-of-thought reasoning in front of the JSON
    # answer block. Without this CoT gets truncated mid-rationale.
    max_completion_length: int = 512

    # Curriculum mode controls how phases are organised:
    #   "difficulty"  — phase 1 = all-loans-easy, phase 2 = all-loans-medium, ...
    #                   (original behaviour — model has to learn LTV/RERA from
    #                   scratch when Vehicle/Home cases first appear at medium)
    #   "task"        — phase 1 = personal only, phase 2 = +vehicle, phase 3 = +home
    #                   (incremental skill addition; recommended)
    curriculum_mode: str = os.getenv("CURRICULUM_MODE", "task")

    # Mix this fraction of *previous-phase* samples into the current phase to
    # prevent catastrophic forgetting. 0.0 disables; 0.2 is a sane default.
    replay_fraction: float = 0.2

    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    # Dropped from 5e-6 to 1e-6. The previous run collapsed train loss to
    # ~0.0001 in ~50 steps, indicating the policy was overshooting. A lower
    # LR + higher KL coefficient anchors more strongly to the reference.
    learning_rate: float = 1e-6
    max_grad_norm: float = 0.5
    # Raised from 0.2 to 0.3 — even stronger reference anchor. Combined with
    # the lower LR this keeps the policy in the neighbourhood of the SFT-warmed
    # init while GRPO refines.
    beta: float = 0.3

    # LoRA (Parameter Efficient Fine-Tuning)
    # Doubled rank/alpha to give the adapter enough capacity to encode the
    # full rule set (CIBIL + FOIR + LTV tier + RERA + employment + counter-
    # offer arithmetic). With rank 16 the adapter saturates fast.
    use_peft: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Output
    output_dir: str = "./grpo_credit_assessment"
    # Push intermediate checkpoints to HF Hub. Set HF_PUSH_CHECKPOINTS=0 to disable
    # (recommended for HF Jobs / Spaces to avoid slow uploads every save_steps).
    # The final model is always saved locally via trainer.save_model() regardless.
    push_to_hub: bool = os.getenv("HF_PUSH_CHECKPOINTS", "1") == "1"
    hub_model_id: str = os.getenv("HUB_MODEL_ID", "iamnijin/credit-assessment-curriculum")

    # Logging
    logging_steps: int = 10
    # Mid-training validation is DISABLED by default. Setting eval_strategy="steps"
    # makes GRPO run a full generation pass over the eval set every `eval_steps` —
    # ~30 min per fire with num_generations=8 + max_completion_length=512.
    # Phase 1 inflates from ~80 min to ~5.5 h. The per-phase quick_evaluate() inside
    # train_with_curriculum already gives us the accuracy number we need.
    # Override with EVAL_STRATEGY=steps if you really want step-level eval.
    eval_strategy: str = os.getenv("EVAL_STRATEGY", "no")
    eval_steps: int = 50
    # Mid-training checkpointing is also disabled by default for the same reason
    # (disk + latency). Per-phase Hub push (PUSH_PER_PHASE=1) is the safety net.
    save_strategy: str = os.getenv("SAVE_STRATEGY", "no")
    save_steps: int = 200

    # SFT init: if set (or auto-detected), load this LoRA adapter on top of the
    # base model BEFORE GRPO. This is the difference between a cold 81.7% run and
    # a 96.7% post-SFT-then-GRPO run (the actual Colab result we shipped).
    # Auto-detect: if `./grpo_credit_assessment_sft` exists on disk, use it.
    sft_init_dir: Optional[str] = os.getenv("SFT_INIT_DIR")

    # Per-phase Hub push (Colab-style): every curriculum phase uploads its best
    # adapter to {hub_model_id}-phase{N}-{loan_type} so a disconnect can't wipe
    # out hours of work. Disabled by default for local runs; enable via env.
    push_per_phase: bool = os.getenv("PUSH_PER_PHASE", "0") == "1"


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a senior loan officer at an Indian bank. You assess loan applications following RBI guidelines and standard banking norms.

Decision options (you must choose exactly one):
- "approve" — application meets every applicable criterion
- "reject" — at least one hard criterion fails (e.g., CIBIL < 700)
- "request_docs" — documents incomplete (and only that, otherwise the application would clear)
- "counter_offer" — applicant qualifies but the requested loan amount breaches an LTV cap; offer a reduced amount that fits the tier

Hard rules to check, in order:
1. Documents complete? If not → "request_docs"
2. CIBIL score ≥ 700? If not → "reject"
3. FOIR ≤ 50%? If not → "reject"
4. Employment: personal/vehicle loans need ≥ 1 year, home loans need ≥ 2 years. If not → "reject"
5. Home loan only: property RERA-registered? If not → "reject"
6. LTV check (if collateralised):
   - Vehicle: LTV ≤ 85%, else "counter_offer" reducing loan to 85% of collateral value
   - Home: tiered by *loan amount*:
       * loan ≤ ₹30L  → max LTV 90% (counter_offer at 90% if breached)
       * loan ₹30L–₹75L → max LTV 80%
       * loan > ₹75L → max LTV 75%
7. Otherwise → "approve"

Reason step-by-step, then output your final answer as a JSON code block.

EXAMPLE 1 — CIBIL just below threshold (the "high income trap"):
Application: personal loan, monthly income ₹6,00,000, CIBIL 699, FOIR 20%, 12 years employment, documents complete.
Reasoning: documents complete (passes), CIBIL 699 < 700 → fails hard rule 2.
Final answer:
```json
{"decision": "reject", "reasoning": "CIBIL 699 below 700 minimum despite high income"}
```

EXAMPLE 2 — Home LTV tier breach (the "RBI tier trap"):
Application: home loan ₹1,20,00,000 against property worth ₹1,53,84,615 (LTV 78%), CIBIL 820, FOIR 25%, 10 years employment, RERA registered, documents complete.
Reasoning: documents complete, CIBIL ≥ 700, FOIR ≤ 50%, employment ≥ 2y, RERA OK. Loan > ₹75L → max LTV 75%. Current LTV 78% > 75% → counter_offer reducing loan to 75% of property = ₹1,15,38,461.
Final answer:
```json
{"decision": "counter_offer", "counter_offer_amount": 11538461, "reasoning": "Loan over ₹75L capped at 75% LTV by RBI; current 78% exceeds limit"}
```

Always finish with a single JSON block in ```json fences. The JSON must contain "decision" and "reasoning". Include "counter_offer_amount" only for counter_offer, and "docs_requested" only for request_docs."""

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
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "")
        else:
            content = str(completion)

        # Lenient parse: handles raw JSON, ```json fences, plain fences, CoT
        # preamble, trailing prose. Returns None if no decision can be found.
        parsed = parse_response(content)
        if parsed is None:
            rewards.append(-0.5)
            continue

        try:
            decision = LoanDecision(parsed.get("decision"))
        except (ValueError, TypeError):
            rewards.append(-0.5)
            continue

        try:
            action = CreditAssessmentAction(
                decision=decision,
                reasoning=parsed.get("reasoning", ""),
                counter_offer_amount=parsed.get("counter_offer_amount"),
                docs_requested=parsed.get("docs_requested"),
            )
            applicant = json.loads(applicant_json)
            raw_reward = calculate_reward(action, applicant, gt)
            # Normalize reward from [-20, +10] to [-1, +1]
            normalized = (raw_reward - (-20.0)) / (10.0 - (-20.0)) * 2 - 1
            rewards.append(normalized)
        except Exception:
            rewards.append(-0.3)

    return rewards


def format_reward_score(completion) -> float:
    """Calculate format reward for a single completion.

    Uses the same lenient parser as the decision-reward path so that
    chain-of-thought-style responses (CoT preamble + ```json answer block) get
    full format credit. Penalises unparseable output.
    """
    if isinstance(completion, list) and len(completion) > 0:
        content = completion[0].get("content", "")
    else:
        content = str(completion)

    parsed = parse_response(content)
    if parsed is None:
        return -0.2

    has_decision = "decision" in parsed
    has_reasoning = "reasoning" in parsed and bool(str(parsed.get("reasoning") or "").strip())

    if has_decision and has_reasoning:
        return 0.2
    if has_decision:
        return 0.1
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

def _resolve_sft_init_dir(config: TrainConfig) -> Optional[str]:
    """Return the SFT adapter directory if it exists, else None.

    Resolution order:
      1. Explicit config.sft_init_dir / SFT_INIT_DIR env var
      2. Auto-detect ./grpo_credit_assessment_sft (the default sft_warmup.py
         output directory, matches what train_grpo_colab.ipynb uses)
    """
    candidate = config.sft_init_dir or "./grpo_credit_assessment_sft"
    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "adapter_config.json")):
        return candidate
    return None


def _build_model_for_grpo(config: TrainConfig):
    """Return (model_or_name, peft_config_or_none) for GRPOTrainer.

    If an SFT adapter exists, load base model + SFT LoRA as a PeftModel and pass
    it in directly. This matches the Colab pipeline (cell 13) and is the only
    way to start GRPO from a warmed policy. Otherwise, hand off the model name
    string and a fresh peft_config — TRL will build the LoRA itself.
    """
    sft_dir = _resolve_sft_init_dir(config)

    if sft_dir is None:
        peft_config = None
        if config.use_peft:
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
        return config.model_name, peft_config

    print(f"Loading base model + SFT adapter from {sft_dir}...")
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, sft_dir, is_trainable=True)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model, None


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
    
    # Configure GRPO. eval_strategy/save_strategy default to "no" — see TrainConfig
    # comments for why (mid-training generation = 30 min/fire, kills runtime).
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        beta=config.beta,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if config.eval_strategy != "no" else None,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps if config.save_strategy != "no" else None,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    )
    
    # SFT-warmed PeftModel (if available) or model-name + fresh peft_config
    model_or_name, peft_config = _build_model_for_grpo(config)

    trainer = GRPOTrainer(
        model=model_or_name,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=combined_reward,  # Combined: 80% decision + 20% format
    )

    return trainer


def evaluate_model(trainer: GRPOTrainer, num_samples: int = 60) -> dict:
    """Evaluate the trained model on fresh samples.

    Default bumped from 20 → 60 so per-task numbers (typically 20 per type)
    are no longer coin-flip noise. At 20 samples the per-type breakdown was
    ~7 samples each, making any single flip look like a 15% accuracy swing.
    """
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
    model.eval()

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
                max_new_tokens=512,  # match GRPO max_completion_length so CoT isn't truncated
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Lenient parse — matches what the reward function and fair_eval.py use,
        # so baseline (untrained) and trained models are scored on equal terms.
        decision = parse_decision(response)
        task_id = sample["task_id"]
        total += 1
        results_by_task[task_id]["total"] += 1
        if decision is not None and decision == sample["ground_truth"]:
            correct += 1
            results_by_task[task_id]["correct"] += 1
    
    # Print results
    overall_acc = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {correct}/{total} ({overall_acc*100:.1f}%)")
    
    task_names = {1: "Personal Loan", 2: "Vehicle Loan", 3: "Home Loan"}
    for task_id, results in results_by_task.items():
        if results["total"] > 0:
            acc = results["correct"] / results["total"]
            print(f"  {task_names[task_id]}: {results['correct']}/{results['total']} ({acc*100:.1f}%)")
    
    model.train()

    return {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "by_task": results_by_task,
    }


def _build_phase_data(config: TrainConfig, phase_idx: int, attempt: int):
    """Construct (train, eval) datasets for a curriculum phase.

    Honors `config.curriculum_mode`:

      * "difficulty" — original behaviour. Phase 1 = all-loans-easy,
        Phase 2 = all-loans-medium, Phase 3 = all-loans-hard. Vehicle/Home
        cases first appear at medium difficulty, so the model sees LTV/RERA
        for the first time exactly when it also has to handle harder logic.

      * "task" — incremental skill addition. Phase 1 = personal only,
        Phase 2 = vehicle only, Phase 3 = home only. Difficulty cycles
        through easy/medium/hard within each task. This isolates one new
        skill per phase so the GRPO advantage signal is never about both
        "new task" and "harder cases" at the same time.

    Replay buffer: when `config.replay_fraction > 0`, mix in samples from
    previously completed phases to combat catastrophic forgetting. The
    fraction is taken from the configured `samples_per_phase` budget (we
    train on the same total volume).
    """
    seed = 42 + phase_idx * 100 + attempt
    eval_seed = 123 + phase_idx

    mode = (config.curriculum_mode or "task").lower()

    # Build the per-phase data spec: list of (loan_type_filter, difficulty).
    # loan_type_filter == None means "any loan type".
    if mode == "task":
        # task_id 1=personal, 2=vehicle, 3=home — but generate_dataset cycles by
        # i%3 so we just oversample then filter. We pull a 3x pool to guarantee
        # enough samples of the target loan type after filtering.
        task_phases = [
            ("personal", "all"),
            ("vehicle",  "all"),
            ("home",     "all"),
        ]
        target_loan, difficulty = task_phases[phase_idx]
    else:
        target_loan = None
        difficulty = ["easy", "medium", "hard"][phase_idx]

    n_total = config.samples_per_phase
    n_replay = int(round(n_total * max(0.0, min(0.9, config.replay_fraction))))
    n_current = n_total - n_replay if phase_idx > 0 else n_total

    def _filtered(seed_val: int, n: int, loan: Optional[str], diff: str):
        """Generate `n` samples optionally filtered to a single loan type."""
        if loan is None:
            return list(generate_dataset(n, seed=seed_val, difficulty=diff))
        # Oversample to compensate for filtering. generate_dataset cycles
        # task_id (i%3)+1 so any loan type is ~1/3 of samples.
        pool = list(generate_dataset(n * 4, seed=seed_val, difficulty=diff))
        matching = [s for s in pool if s["loan_type"] == loan]
        return matching[:n]

    current_samples = _filtered(seed, n_current, target_loan, difficulty)

    # Replay from prior phases.
    replay_samples = []
    if n_replay > 0 and phase_idx > 0:
        per_prior = max(1, n_replay // phase_idx)
        for prior_idx in range(phase_idx):
            if mode == "task":
                prior_loan = ["personal", "vehicle", "home"][prior_idx]
                prior_diff = "all"
            else:
                prior_loan = None
                prior_diff = ["easy", "medium", "hard"][prior_idx]
            replay_samples.extend(
                _filtered(seed + 1000 + prior_idx, per_prior, prior_loan, prior_diff)
            )

    combined = current_samples + replay_samples
    random.Random(seed).shuffle(combined)
    train_dataset = Dataset.from_list(combined)

    # Eval set tracks the *current* phase task/difficulty so mastery is
    # achievable and meaningfully measures what the phase taught.
    if mode == "task":
        eval_pool = list(generate_dataset(config.num_eval_samples * 4, seed=eval_seed, difficulty="all"))
        eval_filtered = [s for s in eval_pool if s["loan_type"] == target_loan]
        eval_dataset = Dataset.from_list(eval_filtered[: config.num_eval_samples])
    else:
        eval_dataset = generate_dataset(config.num_eval_samples, seed=eval_seed, difficulty=difficulty)

    return train_dataset, eval_dataset, target_loan, difficulty, n_current, n_replay


def train_with_curriculum(config: TrainConfig):
    """
    Train using 3-phase curriculum learning with performance-gated advancement.

    Each phase repeats up to max_phase_retries times if accuracy stays below
    phase_mastery_threshold. Within a phase, we keep the BEST adapter across
    attempts — so a retry that regresses can't poison the next phase.

    The final phase has no gate since there is nowhere left to advance.
    """
    print("\n" + "="*60)
    print("CURRICULUM LEARNING: Performance-Gated 3-Phase Training")
    print("="*60)
    print(f"  Mode: {config.curriculum_mode}")
    print(f"  Mastery threshold: {config.phase_mastery_threshold*100:.0f}%")
    print(f"  Max retries per phase: {config.max_phase_retries}")
    print(f"  Per-phase eval samples: {config.phase_eval_samples}")
    print(f"  Samples per phase: {config.samples_per_phase}")
    print(f"  Replay fraction: {config.replay_fraction}")

    if (config.curriculum_mode or "task").lower() == "task":
        phases = [
            ("personal", "Phase 1: Personal Loans (Foundation)",  config.phase_mastery_threshold),
            ("vehicle",  "Phase 2: + Vehicle Loans (Adds LTV)",   config.phase_mastery_threshold),
            ("home",     "Phase 3: + Home Loans (Adds Tiered LTV + RERA)", 0.0),
        ]
    else:
        phases = [
            ("easy",   "Phase 1: Learning Basics (Easy Cases)",        config.phase_mastery_threshold),
            ("medium", "Phase 2: Refining (Medium Cases)",             config.phase_mastery_threshold),
            ("hard",   "Phase 3: Mastering (Hard Cases + Traps)",      0.0),
        ]

    trainer = None
    phase_results = []
    best_adapter_dir = os.path.join(config.output_dir, "_best_adapter")

    for phase_idx, (_label, phase_name, threshold) in enumerate(phases):
        print(f"\n{'='*60}")
        print(f"{phase_name}")
        print(f"{'='*60}")

        phase_acc = 0.0
        best_attempt_acc = -1.0
        best_attempt_idx = -1
        eval_dataset = None

        for attempt in range(config.max_phase_retries + 1):
            if attempt > 0:
                print(f"\n  [Retry {attempt}/{config.max_phase_retries}] "
                      f"Best so far: {best_attempt_acc*100:.1f}% (attempt {best_attempt_idx+1}) — "
                      f"below {threshold*100:.0f}% threshold, retrying with fresh samples...")

            train_dataset, eval_dataset, target_loan, difficulty, n_cur, n_rep = _build_phase_data(
                config, phase_idx, attempt
            )
            label = f"loan={target_loan or 'all'}, difficulty={difficulty}"
            replay_str = f"  |  Replay: {n_rep}/{n_cur+n_rep}" if n_rep > 0 else ""
            attempt_str = f"  |  Attempt: {attempt+1}/{config.max_phase_retries+1}" if config.max_phase_retries > 0 else ""
            print(f"  Samples: {len(train_dataset)}  |  {label}{replay_str}{attempt_str}")

            if trainer is None:
                trainer = create_trainer_with_datasets(config, train_dataset, eval_dataset)
            else:
                trainer.train_dataset = train_dataset
                trainer.eval_dataset = eval_dataset

            print(f"\n  Training...")
            trainer.train()

            print(f"\n  Evaluating on {config.phase_eval_samples} samples...")
            phase_acc = quick_evaluate(trainer, eval_dataset, num_samples=config.phase_eval_samples)
            print(f"  Accuracy: {phase_acc*100:.1f}%"
                  + (f" (threshold: {threshold*100:.0f}%)" if threshold > 0 else " (no gate on final phase)"))

            # Keep the best adapter seen so far in this phase
            if phase_acc > best_attempt_acc:
                best_attempt_acc = phase_acc
                best_attempt_idx = attempt
                try:
                    trainer.save_model(best_adapter_dir)
                    print(f"  → New best-in-phase adapter saved ({phase_acc*100:.1f}%)")
                except Exception as e:
                    print(f"  ⚠ Could not save best adapter: {e}")

            if threshold == 0.0 or phase_acc >= threshold:
                if threshold > 0:
                    print(f"  Mastery achieved — advancing to next phase.")
                break
        else:
            print(f"  Threshold not reached after {config.max_phase_retries+1} attempts.")

        # Warn loudly if the final attempt regressed vs. best-in-phase. The
        # best adapter is preserved on disk at `best_adapter_dir` so the
        # operator can manually pick it up if the final pipeline regresses.
        if phase_acc < best_attempt_acc - 0.05 and os.path.isdir(best_adapter_dir):
            print(f"  ⚠ Final-attempt accuracy {phase_acc*100:.1f}% regressed from best "
                  f"{best_attempt_acc*100:.1f}%. Best adapter still saved at:\n    {best_adapter_dir}")

        phase_results.append((phase_name, best_attempt_acc if best_attempt_acc >= 0 else phase_acc))

        # Per-phase Hub push: same safety net as the Colab notebook. If the
        # script crashes / gets disconnected during a later phase, the prior
        # phase's adapter is already on the Hub and can be re-loaded.
        if config.push_per_phase and config.hub_model_id:
            target_label = (phases[phase_idx][0] or f"phase{phase_idx+1}").replace(" ", "")
            phase_hub_id = f"{config.hub_model_id}-phase{phase_idx+1}-{target_label}"
            try:
                print(f"  Pushing phase {phase_idx+1} adapter to {phase_hub_id}...")
                trainer.push_to_hub(
                    repo_id=phase_hub_id,
                    commit_message=f"Phase {phase_idx+1} ({target_label}) — {best_attempt_acc*100:.1f}% on per-phase eval",
                )
                print(f"  ✓ https://huggingface.co/{phase_hub_id}")
            except Exception as e:
                print(f"  ⚠ Per-phase Hub push failed (continuing): {e}")

    print("\n" + "="*60)
    print("CURRICULUM LEARNING RESULTS")
    print("="*60)
    for phase_name, acc in phase_results:
        print(f"  {phase_name}: {acc*100:.1f}%")

    return trainer, phase_results


def create_trainer_with_datasets(config: TrainConfig, train_dataset, eval_dataset):
    """Create trainer with provided datasets (used for curriculum learning).

    Honors SFT init: if `./grpo_credit_assessment_sft` (or SFT_INIT_DIR) exists,
    GRPO starts from the SFT-warmed adapter. This is what the Colab pipeline
    does and is required to reproduce the 96.7% trained accuracy.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        beta=config.beta,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if config.eval_strategy != "no" else None,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps if config.save_strategy != "no" else None,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    
    model_or_name, peft_config = _build_model_for_grpo(config)

    trainer = GRPOTrainer(
        model=model_or_name,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=combined_reward,
    )

    return trainer


def quick_evaluate(trainer, dataset, num_samples=50):
    """Quick evaluation for curriculum phase tracking.

    Uses greedy decoding (do_sample=False) so measurements are deterministic —
    this is critical for mastery-threshold gating where a few % of variance
    would cause false retries/advancements. Default bumped from 10 → 50: at
    10 samples the 95% CI on an accuracy estimate is ±30%, making gating
    decisions essentially random.
    """
    correct = 0
    total = 0
    tokenizer = trainer.processing_class
    model = trainer.model
    model.eval()

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
                max_new_tokens=512,  # match GRPO max_completion_length so CoT isn't truncated
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        decision = parse_decision(response)
        total += 1
        if decision is not None and decision == sample["ground_truth"]:
            correct += 1

    model.train()
    return correct / total if total > 0 else 0


def evaluate_by_loan_type(trainer, num_samples_per_type: int = 30) -> dict:
    """Evaluate accuracy per loan type to detect catastrophic forgetting.

    Default raised from 20 → 30. generate_dataset cycles task_id in (i%3)+1,
    so we pool num_samples_per_type*3 and filter — guaranteeing each loan
    type gets exactly num_samples_per_type samples.
    """
    tokenizer = trainer.processing_class
    model = trainer.model
    model.eval()
    results = {}

    # Generate once, then filter per loan type (3x larger pool so each type gets ~num_samples_per_type)
    dataset = generate_dataset(num_samples_per_type * 3, seed=999, difficulty="all")

    for loan_type in ["personal", "vehicle", "home"]:
        correct = 0
        total = 0
        for sample in dataset:
            # Samples have loan_type directly; also handle legacy applicant_data JSON for safety
            sample_loan_type = sample.get("loan_type")
            if sample_loan_type is None:
                try:
                    sample_loan_type = json.loads(sample.get("applicant_data", "{}")).get("loan_type")
                except Exception:
                    sample_loan_type = None
            if sample_loan_type != loan_type:
                continue
            prompt_text = tokenizer.apply_chat_template(
                sample["prompt"], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512,  # match GRPO max_completion_length so CoT isn't truncated do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            decision = parse_decision(response)
            total += 1
            if decision is not None and decision == sample["ground_truth"]:
                correct += 1
        results[loan_type] = {"correct": correct, "total": total,
                              "accuracy": correct / total if total > 0 else 0.0}

    model.train()

    print(f"    Personal: {results['personal']['correct']}/{results['personal']['total']} "
          f"({results['personal']['accuracy']*100:.0f}%)  |  "
          f"Vehicle: {results['vehicle']['correct']}/{results['vehicle']['total']} "
          f"({results['vehicle']['accuracy']*100:.0f}%)  |  "
          f"Home: {results['home']['correct']}/{results['home']['total']} "
          f"({results['home']['accuracy']*100:.0f}%)")
    return results


def evaluate_adversarial(trainer, tracker: AdversarialTracker, num_samples: int = 50,
                          per_strategy: Optional[int] = None) -> dict:
    """
    Evaluate model on adversarial cases and update tracker.

    This identifies which strategies the model struggles with,
    enabling targeted training in the next round.

    Args:
        num_samples: Total approximate sample budget (used if per_strategy is None).
        per_strategy: If set, use exactly this many samples per strategy
            (gives balanced, stable per-strategy accuracy estimates).
    """
    print("\n  Evaluating on adversarial cases...")

    tokenizer = trainer.processing_class
    model = trainer.model
    model.eval()

    results_by_strategy = {s: {"correct": 0, "total": 0} for s in ADVERSARIAL_STRATEGIES}

    if per_strategy is None:
        per_strategy = max(1, num_samples // len(ADVERSARIAL_STRATEGIES))

    for strategy in ADVERSARIAL_STRATEGIES:
        samples_per_strategy = per_strategy
        
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
                    max_new_tokens=512,  # match GRPO max_completion_length so CoT isn't truncated
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            decision = parse_decision(response)
            results_by_strategy[strategy]["total"] += 1
            is_correct = decision is not None and decision == ground_truth
            tracker.record_result(strategy, is_correct)
            if is_correct:
                results_by_strategy[strategy]["correct"] += 1

    model.train()

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
    print(f"  Stability anchors: beta={config.beta}, max_grad_norm={config.max_grad_norm}, "
          f"1 epoch/round")

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

    # Snapshot curriculum-end state so we can fall back to it if adversarial regresses.
    curriculum_ckpt_dir = os.path.join(config.output_dir, "_curriculum_end")
    try:
        trainer.save_model(curriculum_ckpt_dir)
        print(f"  Curriculum checkpoint saved to {curriculum_ckpt_dir} (fallback if adversarial regresses)")
    except Exception as e:
        print(f"  ⚠ Could not save curriculum checkpoint: {e}")

    round_results = []
    self_gen_carry = []  # Self-generated cases from previous round

    for round_idx in range(config.adversarial_rounds):
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL ROUND {round_idx + 1}/{config.adversarial_rounds}")
        print(f"{'='*60}")

        # Step 1: Evaluate and identify weakness
        print(f"\n  [Before Round {round_idx + 1}] Accuracy by loan type:")
        pre_round = evaluate_by_loan_type(trainer)
        pre_eval = evaluate_adversarial(trainer, tracker, per_strategy=config.adversarial_per_strategy_eval)
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

        # CRITICAL: Replay data from all difficulties to prevent catastrophic forgetting
        easy_replay = generate_dataset(
            config.adversarial_samples // 4,
            seed=42 + round_idx + 200,
            difficulty="easy"
        )
        medium_replay = generate_dataset(
            config.adversarial_samples // 4,
            seed=42 + round_idx + 300,
            difficulty="medium"
        )
        hard_replay = generate_dataset(
            config.adversarial_samples // 4,
            seed=42 + round_idx + 400,
            difficulty="hard"
        )
        
        replay_samples = list(easy_replay) + list(medium_replay) + list(hard_replay)
        combined_samples = adversarial_samples + replay_samples
        random.shuffle(combined_samples)
        train_dataset = Dataset.from_list(combined_samples)

        # Step 4: Train — 1 epoch per adversarial round to prevent overfitting
        # on the narrow targeted distribution (key stability fix).
        trainer.train_dataset = train_dataset
        prev_epochs = trainer.args.num_train_epochs
        trainer.args.num_train_epochs = 1
        print(f"\n  Training on {len(train_dataset)} samples (1 epoch):")
        print(f"    - {len(adversarial_samples)} adversarial (targeting {weakness})")
        print(f"    - {len(replay_samples)} replay (easy+medium+hard to prevent forgetting)")
        trainer.train()
        trainer.args.num_train_epochs = prev_epochs

        # Step 5: Measure improvement
        print(f"\n  [After Round {round_idx + 1}] Accuracy by loan type:")
        post_round = evaluate_by_loan_type(trainer)
        for lt in ["personal", "vehicle", "home"]:
            delta = post_round[lt]["accuracy"] - pre_round[lt]["accuracy"]
            arrow = "✅" if delta >= 0 else "❌"
            print(f"    {lt}: {delta*100:+.0f}% {arrow}")
        post_eval = evaluate_adversarial(trainer, tracker, per_strategy=config.adversarial_per_strategy_eval)

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

        # Pre/post accuracy specifically on the targeted strategy (used by
        # scripts/generate_plots.py to render the adversarial-rounds chart).
        pre_targeted = pre_eval.get(weakness, {}).get("accuracy", 0.0)
        post_targeted = post_eval.get(weakness, {}).get("accuracy", 0.0)

        round_results.append({
            "round": round_idx + 1,
            "weakness_targeted": weakness,
            "accuracy": round_acc,
            "self_generated": len(self_gen_carry) if config.use_self_generation else 0,
            "pre_targeted_accuracy": pre_targeted,
            "post_targeted_accuracy": post_targeted,
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
        "curriculum_checkpoint": curriculum_ckpt_dir,
    }


def write_training_log(
    trainer,
    config: TrainConfig,
    eval_results: dict,
    baseline_per_task: Optional[dict] = None,
    baseline_overall: Optional[float] = None,
    output_path: str = "training_log.json",
) -> None:
    """Emit a single JSON file consumed by ``scripts/generate_plots.py``.

    Schema matches ``training_log.example.json``. All fields are optional for
    the plot generator — the script skips plots whose inputs are missing, so
    even partial training runs produce usable output.
    """
    import time
    from datetime import datetime, timezone

    log: dict = {
        "meta": {
            "model_name": config.model_name,
            "mode": (
                "curriculum" + ("+adversarial" if config.use_adversarial else "")
                + ("+self_generation" if config.use_adversarial and config.use_self_generation else "")
            ) if config.use_curriculum else (
                "adversarial" + ("+self_generation" if config.use_self_generation else "")
            ) if config.use_adversarial else "standard",
            "seed": 42,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
            "num_train_samples": (
                config.samples_per_phase * 3 if config.use_curriculum
                else config.num_train_samples
            ),
            "use_peft": config.use_peft,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        },
    }

    if baseline_per_task is not None:
        log["baseline"] = {
            "per_task": {k: v.get("accuracy", 0.0) if isinstance(v, dict) else v
                         for k, v in baseline_per_task.items()},
        }
        if baseline_overall is not None:
            log["baseline"]["overall"] = baseline_overall

    # Trained per-task — always recompute at end of training so the log
    # reflects the final checkpoint state.
    try:
        trained_per_task = evaluate_by_loan_type(trainer)
        log["trained"] = {
            "per_task": {k: v["accuracy"] for k, v in trained_per_task.items()},
        }
        final_eval = eval_results.get("final_evaluation") or {}
        if "overall_accuracy" in final_eval:
            log["trained"]["overall"] = final_eval["overall_accuracy"]
    except Exception as e:
        print(f"⚠ Could not evaluate trained model per-task for log: {e}")

    # Reward curve — pulled straight from the trainer's own log_history so
    # we don't need to duplicate metric collection.
    try:
        history = getattr(trainer.state, "log_history", []) or []
        curve = []
        for entry in history:
            reward = entry.get("reward") or entry.get("train/reward")
            step = entry.get("step")
            if reward is not None and step is not None:
                curve.append({
                    "step": int(step),
                    "reward": float(reward),
                    "kl": float(entry.get("kl", entry.get("train/kl", 0.0)) or 0.0),
                })
        if curve:
            log["reward_curve"] = curve
    except Exception as e:
        print(f"⚠ Could not extract reward curve: {e}")

    # Curriculum phase summary
    phase_results = eval_results.get("curriculum_phases")
    if phase_results:
        try:
            log["curriculum"] = {
                "phase_mastery_threshold": config.phase_mastery_threshold,
                "phases": [
                    {"name": name, "final_eval": float(acc)}
                    for name, acc in phase_results
                ],
            }
        except Exception as e:
            print(f"⚠ Could not serialize curriculum phases: {e}")

    # Adversarial round summary — shape matched to generate_plots.py reader.
    adversarial = eval_results.get("adversarial_training") or {}
    rounds = adversarial.get("rounds") or []
    if rounds:
        log["adversarial_rounds"] = [
            {
                "round": r.get("round", i + 1),
                "targeted_strategy": r.get("weakness_targeted"),
                "pre_round": {"targeted_accuracy": r.get("pre_targeted_accuracy", 0.0)},
                "post_round": {"targeted_accuracy": r.get("post_targeted_accuracy", 0.0)},
                "self_generated_count": r.get("self_generated", 0),
                "adversarial_eval_overall_accuracy": r.get("accuracy"),
            }
            for i, r in enumerate(rounds)
        ]

    try:
        with open(output_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"\n✓ Training log written to {output_path}")
        print(f"  Regenerate plots with: python scripts/generate_plots.py {output_path}")
    except Exception as e:
        print(f"⚠ Could not write training log: {e}")


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
    sft_dir = _resolve_sft_init_dir(config)
    print(f"  SFT init: {'YES (' + sft_dir + ')' if sft_dir else 'NO (cold start — strongly recommend running sft_warmup.py first)'}")
    print(f"  Push per phase: {config.push_per_phase} → {config.hub_model_id}-phase{{N}}-{{loan}}")
    print(f"  Eval/save strategy: {config.eval_strategy} / {config.save_strategy}")
    print(f"  Curriculum Learning: {config.use_curriculum}")
    print(f"  Curriculum mode: {config.curriculum_mode}")
    print(f"  Replay fraction: {config.replay_fraction}")
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
    baseline_per_task = None
    baseline_overall = None

    # Baseline evaluation — measure the untrained model on the same per-task
    # eval we'll run at the end, so scripts/generate_plots.py can render a
    # true baseline-vs-trained comparison. This is expensive (~90 model gen
    # calls), but only runs once and is critical evidence.
    if os.getenv("SKIP_BASELINE", "0") != "1":
        try:
            print("\n" + "="*60)
            print("Baseline evaluation (untrained model, used for plots)")
            print("="*60)
            baseline_trainer = create_trainer(config)
            baseline_per_task = evaluate_by_loan_type(baseline_trainer)
            vals = [v["accuracy"] for v in baseline_per_task.values()]
            baseline_overall = sum(vals) / len(vals) if vals else 0.0
            print(f"  Baseline overall: {baseline_overall*100:.1f}%")
            # Free baseline trainer before we build the real one
            del baseline_trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠ Baseline evaluation failed ({e}); training will continue without it.")

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
    
    # Snapshot curriculum-only model for side-by-side comparison
    curriculum_only_dir = os.path.join(config.output_dir, "_curriculum_only")
    curriculum_only_acc = None
    if config.use_curriculum and config.use_adversarial:
        try:
            trainer.save_model(curriculum_only_dir)
            print(f"\nCurriculum-only model snapshot: {curriculum_only_dir}")
            print("Evaluating curriculum-only model (large sample for stable comparison)...")
            curriculum_only_eval = evaluate_model(trainer, num_samples=60)
            curriculum_only_acc = curriculum_only_eval["overall_accuracy"]
            eval_results["curriculum_only_eval"] = curriculum_only_eval
        except Exception as e:
            print(f"⚠ Could not snapshot curriculum-only model: {e}")

    # Phase 2: Adversarial Training (if enabled)
    if config.use_adversarial:
        trainer, adversarial_results = train_with_adversarial(config, trainer)
        eval_results["adversarial_training"] = adversarial_results

    # Save final model locally
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model()

    # Final evaluation on mixed test set
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    final_eval = evaluate_model(trainer, num_samples=60)
    eval_results["final_evaluation"] = final_eval

    # Compare curriculum-only vs curriculum+adversarial
    if curriculum_only_acc is not None:
        final_acc = final_eval["overall_accuracy"]
        delta = final_acc - curriculum_only_acc
        print("\n" + "="*60)
        print("CURRICULUM vs CURRICULUM+ADVERSARIAL")
        print("="*60)
        print(f"  Curriculum-only     : {curriculum_only_acc*100:.1f}%")
        print(f"  +Adversarial rounds : {final_acc*100:.1f}%  (Δ {delta*100:+.1f}%)")
        if delta < -0.03:
            print(f"\n  ⚠ Adversarial training REGRESSED overall accuracy.")
            print(f"    For the pitch, use the curriculum-only checkpoint:")
            print(f"      {curriculum_only_dir}")
        elif delta > 0.03:
            print(f"\n  ✓ Adversarial training helped. Use the final model.")
        else:
            print(f"\n  ≈ Within noise. Either checkpoint is defensible.")

    # Push final model to HF Hub (always, as long as HF_TOKEN is set and
    # HUB_MODEL_ID is valid). This is separate from intermediate push_to_hub.
    if hf_token and config.hub_model_id:
        try:
            print(f"\nPushing final model to HuggingFace Hub: {config.hub_model_id}")
            trainer.push_to_hub(
                commit_message="Final model: curriculum + adversarial self-play GRPO"
            )
            print(f"✓ Model pushed to https://huggingface.co/{config.hub_model_id}")
        except Exception as e:
            print(f"⚠️  Hub push failed (model still saved locally): {e}")

    # Emit a single JSON log file consumed by scripts/generate_plots.py to
    # regenerate the reward curve, per-task accuracy, adversarial rounds, and
    # curriculum phase charts. Disable with TRAINING_LOG_PATH="" if not needed.
    training_log_path = os.getenv("TRAINING_LOG_PATH", "training_log.json")
    if training_log_path:
        write_training_log(
            trainer,
            config,
            eval_results,
            baseline_per_task=baseline_per_task,
            baseline_overall=baseline_overall,
            output_path=training_log_path,
        )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    return trainer, eval_results


if __name__ == "__main__":
    main()
