"""
Baseline inference script for the Credit Assessment Environment.

Runs three agents across all 3 tasks and reports reproducible scores:
  1. Random Agent     — picks a random action (lower bound)
  2. Rule-Based Agent — uses simple heuristics matching bank criteria (upper bound)
  3. LLM Agent        — uses OpenAI API to reason about loan applications

Usage:
    # Run random + rule-based baselines (no API key needed)
    python -m credit_assessment_env.baseline

    # Run all agents including LLM (requires OPENAI_API_KEY)
    OPENAI_API_KEY=sk-... python -m credit_assessment_env.baseline --llm

    # Customize episodes and seed
    python -m credit_assessment_env.baseline --llm --episodes 20 --seed 42 --model gpt-4o-mini
"""

import argparse
import json
import os
import random
from typing import Callable

from .loan_decision import LoanDecision
from .models import CreditAssessmentAction, CreditAssessmentObservation
from .server.credit_assessment_env_environment import CreditAssessmentEnvironment


def random_agent(obs: CreditAssessmentObservation) -> CreditAssessmentAction:
    """Baseline: picks a random decision every time."""
    decision = random.choice(list(LoanDecision))
    return CreditAssessmentAction(
        decision=decision,
        reasoning="Random baseline",
        counter_offer_amount=obs.loan_amount * 0.8 if decision == LoanDecision.COUNTER_OFFER else None,
        docs_requested="All documents" if decision == LoanDecision.REQUEST_DOCS else None,
    )


def rule_based_agent(obs: CreditAssessmentObservation) -> CreditAssessmentAction:
    """Heuristic agent that mirrors the ground truth logic."""

    if not obs.documents_complete:
        return CreditAssessmentAction(
            decision=LoanDecision.REQUEST_DOCS,
            reasoning="Documents are incomplete",
            docs_requested="All pending documents",
        )

    if obs.rera_registered is False:
        return CreditAssessmentAction(
            decision=LoanDecision.REJECT,
            reasoning="Property is not RERA registered — compliance risk",
        )

    if obs.credit_score < 700:
        return CreditAssessmentAction(
            decision=LoanDecision.REJECT,
            reasoning=f"Credit score {obs.credit_score} is below 700 minimum",
        )

    if obs.foir > 0.50:
        return CreditAssessmentAction(
            decision=LoanDecision.REJECT,
            reasoning=f"FOIR {obs.foir:.0%} exceeds 50% limit",
        )

    if obs.employment_years < 1:
        return CreditAssessmentAction(
            decision=LoanDecision.REJECT,
            reasoning=f"Only {obs.employment_years} year(s) employment — too risky",
        )

    if obs.loan_type == "home" and obs.employment_years < 2:
        return CreditAssessmentAction(
            decision=LoanDecision.REJECT,
            reasoning=f"Home loan requires 2+ years employment, applicant has {obs.employment_years}",
        )

    if obs.ltv_ratio:
        ltv_limit = _get_ltv_limit(obs)
        if obs.ltv_ratio > ltv_limit:
            reduced = obs.loan_amount * ltv_limit / obs.ltv_ratio
            return CreditAssessmentAction(
                decision=LoanDecision.COUNTER_OFFER,
                reasoning=f"LTV {obs.ltv_ratio:.0%} exceeds {ltv_limit:.0%} limit",
                counter_offer_amount=round(reduced, -3),
            )

    return CreditAssessmentAction(
        decision=LoanDecision.APPROVE,
        reasoning="All criteria met — credit score, FOIR, employment, and LTV within limits",
    )


def _get_ltv_limit(obs: CreditAssessmentObservation) -> float:
    """Return the applicable LTV limit based on loan type and amount."""
    if obs.loan_type == "vehicle":
        return 0.85
    if obs.loan_amount <= 3_000_000:
        return 0.90
    elif obs.loan_amount <= 7_500_000:
        return 0.80
    return 0.75


SYSTEM_PROMPT = """You are a senior loan officer at an Indian bank. You assess loan applications 
following RBI guidelines and standard banking norms.

You must respond with a JSON object containing:
- "decision": one of "approve", "reject", "request_docs", "counter_offer"
- "reasoning": a brief explanation for your decision
- "counter_offer_amount": (only if decision is "counter_offer") the reduced loan amount
- "docs_requested": (only if decision is "request_docs") what documents are needed

Key guidelines:
- CIBIL score below 700 → reject
- FOIR above 50% → reject
- Incomplete documents → request_docs
- For home loans: non-RERA property → reject
- For vehicle loans: LTV above 85% → counter_offer
- For home loans: LTV limits are tiered by RBI (≤30L: 90%, 30-75L: 80%, >75L: 75%)

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""


def create_llm_agent(model: str = "gpt-4o-mini"):
    """Create an LLM agent using the OpenAI API.

    Supports both standard OpenAI and Azure OpenAI endpoints.
    Set OPENAI_BASE_URL in .env to use a custom endpoint (e.g. Azure),
    otherwise falls back to the default OpenAI API.
    For Azure, also set OPENAI_API_VERSION (defaults to 2024-12-01-preview).
    """
    from openai import AzureOpenAI, OpenAI

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for LLM agent. "
            "Set it in .env or with: export OPENAI_API_KEY=sk-..."
        )

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_version = os.environ.get("OPENAI_API_VERSION", "2024-08-01-preview")

    if base_url:
        client = AzureOpenAI(
            azure_endpoint=base_url,
            api_key=api_key,
            api_version=api_version,
        )
    else:
        client = OpenAI(api_key=api_key)

    def llm_agent(obs: CreditAssessmentObservation) -> CreditAssessmentAction:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs.applicant_profile},
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        decision_str = parsed.get("decision", "reject")
        try:
            decision = LoanDecision(decision_str)
        except ValueError:
            decision = LoanDecision.REJECT

        docs = parsed.get("docs_requested")
        if isinstance(docs, list):
            docs = ", ".join(str(d) for d in docs)

        return CreditAssessmentAction(
            decision=decision,
            reasoning=parsed.get("reasoning", "LLM decision"),
            counter_offer_amount=parsed.get("counter_offer_amount"),
            docs_requested=docs,
        )

    return llm_agent


def run_evaluation(
    agent_fn: Callable,
    agent_name: str,
    episodes_per_task: int,
    seed: int,
) -> dict:
    """Run an agent across all tasks and collect scores."""

    random.seed(seed)
    env = CreditAssessmentEnvironment()

    results = {}

    for task_id in [1, 2, 3]:
        task_name = env.TASKS[task_id]["name"]
        rewards = []
        grades = []

        for ep in range(episodes_per_task):
            obs = env.reset(seed=seed + ep, task_id=task_id)
            done = False

            while not done:
                action = agent_fn(obs)
                obs = env.step(action)
                done = obs.done

            rewards.append(env._total_reward)
            grades.append(env.grade())

        avg_reward = sum(rewards) / len(rewards)
        avg_grade = sum(grades) / len(grades)
        perfect = sum(1 for g in grades if g == 1.0)

        results[task_id] = {
            "task_name": task_name,
            "avg_reward": avg_reward,
            "avg_grade": avg_grade,
            "perfect_rate": perfect / len(grades),
            "episodes": len(grades),
        }

        print(f"  Task {task_id} ({task_name:15s}) | "
              f"Avg Reward: {avg_reward:+6.2f} | "
              f"Avg Grade: {avg_grade:.3f} | "
              f"Perfect: {perfect}/{len(grades)} ({perfect/len(grades)*100:.0f}%)")

    overall_grade = sum(r["avg_grade"] for r in results.values()) / len(results)
    print(f"  {'':29s} | Overall Grade: {overall_grade:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for Credit Assessment Environment")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per task (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--llm", action="store_true", help="Include LLM agent (requires OPENAI_API_KEY)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for LLM agent (default: gpt-4o-mini)")
    args = parser.parse_args()

    print(f"\nSeed: {args.seed} | Episodes per task: {args.episodes}")
    print("=" * 80)

    print(f"\n--- Random Agent ---")
    random_results = run_evaluation(random_agent, "Random", args.episodes, args.seed)

    print(f"\n--- Rule-Based Agent ---")
    rule_results = run_evaluation(rule_based_agent, "Rule-Based", args.episodes, args.seed)

    llm_results = None
    if args.llm:
        print(f"\n--- LLM Agent ({args.model}) ---")
        try:
            llm_fn = create_llm_agent(model=args.model)
            llm_results = run_evaluation(llm_fn, f"LLM ({args.model})", args.episodes, args.seed)
        except ValueError as e:
            print(f"  Skipped: {e}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    header = f"  {'Task':33s} | {'Random':>8s} | {'Rule-Based':>10s}"
    if llm_results:
        header += f" | {'LLM':>8s}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for task_id in [1, 2, 3]:
        task = random_results[task_id]["task_name"]
        rg = random_results[task_id]["avg_grade"]
        rbg = rule_results[task_id]["avg_grade"]
        row = f"  Task {task_id} ({task:15s})       | {rg:8.3f} | {rbg:10.3f}"
        if llm_results:
            lg = llm_results[task_id]["avg_grade"]
            row += f" | {lg:8.3f}"
        print(row)

    random_overall = sum(r["avg_grade"] for r in random_results.values()) / 3
    rule_overall = sum(r["avg_grade"] for r in rule_results.values()) / 3
    row = f"  {'Overall':33s} | {random_overall:8.3f} | {rule_overall:10.3f}"
    if llm_results:
        llm_overall = sum(r["avg_grade"] for r in llm_results.values()) / 3
        row += f" | {llm_overall:8.3f}"
    print(row)


if __name__ == "__main__":
    main()
