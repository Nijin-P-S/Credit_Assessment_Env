"""SFT warmup before GRPO.

Why this script exists
----------------------
GRPO struggles when the policy starts far from optimal. With sparse 4-class
rewards on a multi-criterion task, most rollouts in early training get the
same (negative) reward, the advantage signal collapses to ~zero, and the
policy never finds the high-reward modes for the harder loan types.

We saw this concretely in our previous run: even after 2 rounds of targeted
adversarial training, the model stayed at 0% on `employment_trap_home`. The
decision was simply not in its rollout distribution. GRPO can refine a
working policy but it cannot teach a brand-new behaviour from sparse rewards.

This script does standard supervised fine-tuning on programmatically-
generated (applicant, gold-response) pairs to give GRPO a better starting
policy. The gold response is a chain-of-thought walkthrough of the RBI rule
checks ending in a fenced JSON answer block — exactly the format the GRPO
reward function and lenient parser expect.

Pipeline
--------
1. SFT (this script) — teaches the FORMAT (CoT + ```json) and the RULE-CHECK
   STYLE (step-by-step elimination). Runs ~30 min on A100, ~6 credits.
2. GRPO (train_grpo.py) — refines the SFT policy with reward signal. The
   GRPO LoRA picks up where SFT left off because we use the same r/alpha.

Output
------
Saves a LoRA adapter to `--output-dir` (default `./grpo_credit_assessment_sft`)
that can be loaded by `train_grpo.py` as a starting point. Set
`SFT_INIT_DIR=...` before running `train_grpo.py` and the GRPO trainer will
attach to that adapter instead of starting fresh.

Usage
-----
    python sft_warmup.py \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --num-samples 600 \\
        --num-epochs 2 \\
        --output-dir ./grpo_credit_assessment_sft

On A100 with Qwen 7B + LoRA r=32, ~600 samples x 2 epochs ~= 25-35 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Gold response generator
# ---------------------------------------------------------------------------
# This is the heart of the script. For each applicant we build a deterministic
# chain-of-thought response that:
#   1. Walks through the RBI rules in the SAME order as the system prompt
#   2. Names which rule fires the decision (or notes that all rules pass)
#   3. Ends in a fenced JSON answer block matching the lenient parser format
# The model learns FORMAT and RULE-CHECK STYLE from this; GRPO then learns
# to apply the style correctly on cases it has never seen.


def _ltv_cap_for_home_loan(loan_amount: int) -> float:
    """RBI tiered LTV: <=30L -> 0.90, 30L-75L -> 0.80, >75L -> 0.75."""
    if loan_amount <= 30_00_000:
        return 0.90
    if loan_amount <= 75_00_000:
        return 0.80
    return 0.75


def _counter_offer_amount(applicant: dict) -> int:
    """Reduce the loan to fit the applicable LTV cap."""
    collateral = applicant.get("collateral_value")
    if not collateral:
        return applicant.get("loan_amount", 0)
    if applicant["loan_type"] == "vehicle":
        cap = 0.85
    else:  # home
        cap = _ltv_cap_for_home_loan(applicant["loan_amount"])
    return int(collateral * cap)


def _gold_reasoning_for(applicant: dict, ground_truth: str) -> tuple[str, dict]:
    """Build a step-by-step rationale string + the JSON payload for the answer.

    The rationale follows the system prompt's rule order exactly so SFT
    teaches the model the same procedural style the prompt asks for.
    """
    cibil = applicant["credit_score"]
    foir = applicant["foir"]
    employment = applicant["employment_years"]
    docs_complete = applicant.get("documents_complete", True)
    loan_type = applicant["loan_type"]
    loan_amount = applicant["loan_amount"]
    ltv = applicant.get("ltv_ratio")
    collateral = applicant.get("collateral_value")
    rera = applicant.get("rera_registered")
    min_employment = 2 if loan_type == "home" else 1

    steps = []
    answer: dict = {"reasoning": ""}

    if not docs_complete:
        steps.append("Documents incomplete -> request_docs.")
        answer["decision"] = "request_docs"
        answer["docs_requested"] = "Please submit pending documents (income proof, KYC)."
        answer["reasoning"] = "Required documents are not complete; cannot make a final decision yet."
        return "\n".join(steps), answer
    steps.append(f"1. Documents complete: yes.")

    if cibil < 700:
        steps.append(f"2. CIBIL {cibil} < 700 -> reject. (No need to evaluate further criteria.)")
        answer["decision"] = "reject"
        answer["reasoning"] = f"CIBIL score {cibil} is below the 700 minimum required for any loan."
        return "\n".join(steps), answer
    steps.append(f"2. CIBIL {cibil} >= 700: pass.")

    if foir > 0.50:
        steps.append(f"3. FOIR {foir*100:.0f}% > 50% -> reject.")
        answer["decision"] = "reject"
        answer["reasoning"] = f"FOIR {foir*100:.0f}% exceeds the 50% cap; debt service load too high."
        return "\n".join(steps), answer
    steps.append(f"3. FOIR {foir*100:.0f}% <= 50%: pass.")

    if employment < min_employment:
        steps.append(
            f"4. Employment {employment}y < {min_employment}y required for {loan_type} loan -> reject."
        )
        answer["decision"] = "reject"
        answer["reasoning"] = (
            f"Employment tenure {employment} year(s) is below the {min_employment}-year "
            f"minimum required for a {loan_type} loan."
        )
        return "\n".join(steps), answer
    steps.append(f"4. Employment {employment}y >= {min_employment}y: pass.")

    if loan_type == "home":
        if not rera:
            steps.append("5. Property NOT RERA-registered -> reject (mandatory for home loans).")
            answer["decision"] = "reject"
            answer["reasoning"] = "Property is not RERA-registered; mandatory for any home loan."
            return "\n".join(steps), answer
        steps.append("5. Property RERA-registered: pass.")

    if loan_type == "vehicle" and ltv is not None:
        if ltv > 0.85:
            counter = _counter_offer_amount(applicant)
            steps.append(
                f"6. Vehicle LTV {ltv*100:.0f}% > 85% cap -> counter_offer reducing loan to "
                f"INR {counter:,} (85% of collateral)."
            )
            answer["decision"] = "counter_offer"
            answer["counter_offer_amount"] = counter
            answer["reasoning"] = (
                f"LTV {ltv*100:.0f}% exceeds the 85% cap for vehicle loans; counter-offering at "
                f"85% of the collateral value."
            )
            return "\n".join(steps), answer
        steps.append(f"6. Vehicle LTV {ltv*100:.0f}% <= 85%: pass.")

    if loan_type == "home" and ltv is not None:
        cap = _ltv_cap_for_home_loan(loan_amount)
        if ltv > cap + 1e-9:
            counter = _counter_offer_amount(applicant)
            tier = (
                "<= INR 30L (90% cap)" if loan_amount <= 30_00_000
                else "INR 30L-75L (80% cap)" if loan_amount <= 75_00_000
                else "> INR 75L (75% cap)"
            )
            steps.append(
                f"6. Home loan amount INR {loan_amount:,} falls in tier {tier}. "
                f"LTV {ltv*100:.0f}% > {cap*100:.0f}% cap -> counter_offer reducing to INR {counter:,}."
            )
            answer["decision"] = "counter_offer"
            answer["counter_offer_amount"] = counter
            answer["reasoning"] = (
                f"Loan amount INR {loan_amount:,} caps LTV at {cap*100:.0f}% per RBI tiered rules; "
                f"current LTV {ltv*100:.0f}% exceeds it. Counter-offering at the cap."
            )
            return "\n".join(steps), answer
        steps.append(f"6. Home LTV {ltv*100:.0f}% within {cap*100:.0f}% tier cap: pass.")

    steps.append("All criteria pass -> approve.")
    answer["decision"] = "approve"
    answer["reasoning"] = (
        f"Strong profile: CIBIL {cibil}, FOIR {foir*100:.0f}%, {employment}y employment"
        + (f", LTV {ltv*100:.0f}% within cap" if ltv is not None else "")
        + (", RERA registered" if loan_type == "home" else "")
        + ". All RBI criteria satisfied."
    )
    return "\n".join(steps), answer


def gold_response_text(applicant: dict, ground_truth: str) -> str:
    """Build the assistant turn that SFT trains on: rationale + JSON block."""
    rationale, answer = _gold_reasoning_for(applicant, ground_truth)
    # Sanity check: our rule-walk should match the env's calculate_ground_truth.
    # If it diverges, prefer the env's ground truth and keep only the JSON
    # answer (rationale would be misleading). This guards against future
    # changes to env rules that don't make it into this script.
    if answer["decision"] != ground_truth:
        answer = {
            "decision": ground_truth,
            "reasoning": "Per RBI guidelines applied to this applicant.",
        }
        rationale = "Evaluating each RBI criterion in order..."
    payload_keys_order = ["decision", "reasoning", "counter_offer_amount", "docs_requested"]
    ordered = {k: answer[k] for k in payload_keys_order if k in answer}
    json_block = json.dumps(ordered, indent=2, ensure_ascii=False)
    return f"{rationale}\n\nFinal answer:\n```json\n{json_block}\n```"


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_sft_dataset(num_samples: int, seed: int = 7):
    """Mix of all loan types and difficulties so SFT covers the rule space."""
    from datasets import Dataset
    from train_utils import generate_applicant, calculate_ground_truth, build_profile_text
    from train_grpo import SYSTEM_PROMPT  # use the *same* prompt as GRPO

    random.seed(seed)
    rows = []
    # 50% standard, 50% adversarial-flavoured for trap-style coverage.
    for i in range(num_samples):
        task_id = (i % 3) + 1
        difficulty = random.choice(["easy", "medium", "hard"])
        applicant = generate_applicant(task_id, difficulty=difficulty)
        ground_truth = calculate_ground_truth(applicant)
        profile_text = build_profile_text(applicant)
        gold = gold_response_text(applicant, ground_truth)
        rows.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": profile_text},
                {"role": "assistant", "content": gold},
            ],
            "ground_truth": ground_truth,
            "loan_type": applicant["loan_type"],
        })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class SFTArgs:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    num_samples: int = 600
    num_epochs: int = 2
    learning_rate: float = 2e-5
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_seq_length: int = 1024
    output_dir: str = "./grpo_credit_assessment_sft"
    seed: int = 7


def run_sft(args: SFTArgs) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.model_name}...")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )

    print(f"Building SFT dataset: {args.num_samples} samples...")
    dataset = build_sft_dataset(args.num_samples, seed=args.seed)
    print("  Loan-type distribution:",
          {lt: sum(1 for r in dataset if r["loan_type"] == lt) for lt in ["personal", "vehicle", "home"]})
    print("  Decision distribution:",
          {d: sum(1 for r in dataset if r["ground_truth"] == d)
           for d in ["approve", "reject", "request_docs", "counter_offer"]})

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_length,
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\nStarting SFT warmup...")
    trainer.train()

    print(f"\nSaving SFT-warmed adapter to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print()
    print(f"SFT warmup complete. To use this checkpoint as the GRPO init:")
    print(f"  export SFT_INIT_DIR={args.output_dir}")
    print(f"  python train_grpo.py")


def main():
    parser = argparse.ArgumentParser(description="SFT warmup before GRPO.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--output-dir", default="./grpo_credit_assessment_sft")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sft_args = SFTArgs(
        model_name=args.model_name,
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    run_sft(sft_args)


if __name__ == "__main__":
    main()
