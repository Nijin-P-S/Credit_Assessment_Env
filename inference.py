"""
Inference Script for Credit Assessment Environment
===================================================
MANDATORY environment variables (set before running):
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)
    HF_TOKEN       Your Hugging Face / API key

Uses the OpenAI Client for all LLM calls as required.

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export HF_TOKEN="hf_..."
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    uv run python inference.py

The script runs an LLM agent across all 3 tasks (Personal, Vehicle, Home Loan)
for 10 episodes each and prints per-task and overall grades.
"""

import json
import os
import random
import sys
import textwrap

from openai import OpenAI

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from credit_assessment_env.loan_decision import LoanDecision
from credit_assessment_env.models import CreditAssessmentAction, CreditAssessmentObservation
from credit_assessment_env.server.credit_assessment_env_environment import CreditAssessmentEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

EPISODES_PER_TASK = 10
SEED = 42

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior loan officer at an Indian bank. You assess loan applications
    following RBI guidelines and standard banking norms.

    You must respond with a JSON object containing:
    - "decision": one of "approve", "reject", "request_docs", "counter_offer"
    - "reasoning": a brief explanation for your decision
    - "counter_offer_amount": (only if decision is "counter_offer") the reduced loan amount
    - "docs_requested": (only if decision is "request_docs") what documents are needed

    Key guidelines:
    - CIBIL score below 700 -> reject
    - FOIR above 50% -> reject
    - Incomplete documents -> request_docs
    - For home loans: non-RERA property -> reject
    - For vehicle loans: LTV above 85% -> counter_offer
    - For home loans: LTV limits are tiered by RBI (<=30L: 90%, 30-75L: 80%, >75L: 75%)

    Respond ONLY with valid JSON. No markdown, no explanation outside the JSON.""")


def llm_agent(client: OpenAI, obs: CreditAssessmentObservation) -> CreditAssessmentAction:
    """Use LLM to assess a loan application and return an action."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs.applicant_profile},
            ],
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except Exception as exc:
        print(f"  LLM request failed ({exc}). Defaulting to reject.")
        return CreditAssessmentAction(
            decision=LoanDecision.REJECT,
            reasoning=f"LLM error fallback: {exc}",
        )

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


def main() -> None:
    if not API_KEY:
        print("ERROR: No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.")
        sys.exit(1)

    print(f"API Base URL : {API_BASE_URL}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Episodes/task: {EPISODES_PER_TASK}")
    print(f"Seed         : {SEED}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CreditAssessmentEnvironment()

    results = {}

    for task_id in [1, 2, 3]:
        task_meta = env.TASKS[task_id]
        task_name = task_meta["name"]
        print(f"\n{'=' * 60}")
        print(f"Task {task_id}: {task_name} ({task_meta['difficulty']})")
        print("=" * 60)

        grades = []
        rewards = []

        for ep in range(EPISODES_PER_TASK):
            random.seed(SEED + ep)
            obs = env.reset(seed=SEED + ep, task_id=task_id)
            done = False

            while not done:
                action = llm_agent(client, obs)
                obs = env.step(action)
                done = obs.done

            grade = env.grade()
            reward = env._total_reward
            grades.append(grade)
            rewards.append(reward)
            print(f"  Episode {ep + 1:3d}: grade={grade:.2f}  reward={reward:+.2f}  "
                  f"decision={action.decision.value}")

        avg_grade = sum(grades) / len(grades)
        avg_reward = sum(rewards) / len(rewards)
        perfect = sum(1 for g in grades if g >= 1.0)
        results[task_id] = {
            "task_name": task_name,
            "avg_grade": avg_grade,
            "avg_reward": avg_reward,
        }
        print(f"  >> Avg Grade: {avg_grade:.3f} | Avg Reward: {avg_reward:+.2f} | "
              f"Perfect: {perfect}/{EPISODES_PER_TASK}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Task':33s} | {'Grade':>8s}")
    print("  " + "-" * 46)
    for task_id in [1, 2, 3]:
        r = results[task_id]
        print(f"  Task {task_id} ({r['task_name']:15s})       | {r['avg_grade']:8.3f}")

    overall = sum(r["avg_grade"] for r in results.values()) / 3
    print(f"  {'Overall':33s} | {overall:8.3f}")
    print()


if __name__ == "__main__":
    main()
