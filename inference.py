"""
Inference Script for Credit Assessment Environment
===================================================
MANDATORY environment variables (set before running):
    API_BASE_URL       The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME         The model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)
    HF_TOKEN           Your Hugging Face / API key
    LOCAL_IMAGE_NAME   (optional) local image name – not used for this env

STDOUT FORMAT (emitted per episode):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export HF_TOKEN="hf_..."
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    uv run python inference.py
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

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
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "credit_assessment_env-env:latest")
TASK_NAME = os.getenv("TASK_NAME", "all")
BENCHMARK = os.getenv("BENCHMARK", "credit-assessment")

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


# ---------------------------------------------------------------------------
# Structured stdout logging (mandatory hackathon format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

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

    raw_amount = parsed.get("counter_offer_amount")
    try:
        counter_offer_amount = float(raw_amount) if raw_amount is not None else None
    except (ValueError, TypeError):
        counter_offer_amount = None

    return CreditAssessmentAction(
        decision=decision,
        reasoning=parsed.get("reasoning") or "LLM decision",
        counter_offer_amount=counter_offer_amount,
        docs_requested=docs,
    )


def action_to_str(action: CreditAssessmentAction) -> str:
    """Format action as a compact string for [STEP] lines."""
    val = action.decision.value
    if action.decision == LoanDecision.COUNTER_OFFER and action.counter_offer_amount is not None:
        return f"counter_offer({action.counter_offer_amount:.0f})"
    if action.decision == LoanDecision.REQUEST_DOCS and action.docs_requested:
        docs = action.docs_requested.replace(" ", "_")
        return f"request_docs({docs})"
    return val


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: CreditAssessmentEnvironment,
    client: OpenAI,
    task_id: int,
    task_label: str,
    seed: int,
) -> None:
    """Run one episode and emit [START] / [STEP]* / [END] to stdout."""
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_label, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(seed=seed, task_id=task_id)

        step = 0
        while not obs.done:
            step += 1
            action = llm_agent(client, obs)
            obs = env.step(action)

            reward = obs.reward
            done = obs.done
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                success = reward > 0.0
                break

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CreditAssessmentEnvironment()

    # Determine which tasks to run
    task_map = {str(v["name"]).lower().replace(" ", "-"): k for k, v in env.TASKS.items()}
    task_map.update({str(k): k for k in env.TASKS})  # also allow "1", "2", "3"

    if TASK_NAME == "all":
        task_ids = list(env.TASKS.keys())
    else:
        task_id = task_map.get(TASK_NAME.lower())
        if task_id is None:
            print(f"ERROR: Unknown TASK_NAME '{TASK_NAME}'. Use 'all', '1', '2', '3', "
                  "'personal-loan', 'vehicle-loan', or 'home-loan'.")
            sys.exit(1)
        task_ids = [task_id]

    for task_id in task_ids:
        task_label = env.TASKS[task_id]["name"].lower().replace(" ", "-")
        for ep in range(EPISODES_PER_TASK):
            run_episode(
                env=env,
                client=client,
                task_id=task_id,
                task_label=task_label,
                seed=SEED + ep,
            )


if __name__ == "__main__":
    main()
