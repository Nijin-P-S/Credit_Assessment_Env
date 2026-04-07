"""
Inference Script for Credit Assessment Environment
===================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=personal-loan env=credit-assessment model=meta-llama/Llama-3.1-8B-Instruct
    [STEP] step=1 action=approve reward=10.00 done=true error=null
    [END] success=true steps=1 rewards=10.00

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export HF_TOKEN="hf_..."
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    uv run python inference.py
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from openenv.core.containers.runtime.providers import LocalDockerProvider


def rebuild_docker_image(image_name: str) -> None:
    """Remove existing image and rebuild from current Dockerfile."""
    import subprocess
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    # Remove existing image (ignore errors if it doesn't exist)
    subprocess.run(["docker", "rmi", "-f", image_name], capture_output=True)
    print(f"[DEBUG] Building Docker image {image_name} from {repo_dir}", flush=True)
    result = subprocess.run(
        ["docker", "build", "-t", image_name, repo_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[DEBUG] Docker build failed:\n{result.stderr}", flush=True)
        raise RuntimeError(f"Docker build failed: {result.stderr}")
    print(f"[DEBUG] Docker image built successfully.", flush=True)


class SlowStartProvider(LocalDockerProvider):
    """Maps to container port 7860 (Dockerfile CMD) and uses a longer ready-timeout."""

    def start_container(self, image: str, port=None, env_vars=None, **kwargs):
        import subprocess, time
        if port is None:
            port = self._find_available_port()
        self._container_name = self._generate_container_name(image)
        cmd = ["docker", "run", "-d", "--name", self._container_name, "-p", f"{port}:7860"]
        if env_vars:
            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])
        cmd.append(image)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self._container_id = result.stdout.strip()
        time.sleep(1)
        return f"http://localhost:{port}"

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        super().wait_for_ready(base_url, timeout_s=timeout_s)


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from credit_assessment_env import CreditAssessmentAction, CreditAssessmentEnv
from credit_assessment_env.loan_decision import LoanDecision

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or "credit_assessment_env-env:latest"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
TASK_NAME = os.getenv("TASK_NAME", "all")
BENCHMARK = os.getenv("BENCHMARK", "credit-assessment")

EPISODES_PER_TASK = 10
SEED = 42
MAX_STEPS = 3

TASKS = {
    1: "personal-loan",
    2: "vehicle-loan",
    3: "home-loan",
}

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


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

async def llm_agent(client: OpenAI, applicant_profile: str) -> CreditAssessmentAction:
    """Use LLM to assess a loan application and return an action."""
    try:
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": applicant_profile},
                ],
                response_format={"type": "json_object"},
            ),
        )
        raw = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
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

async def run_episode(
    env: CreditAssessmentEnv,
    llm_client: OpenAI,
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
        result = await env.reset(seed=seed, task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = await llm_agent(llm_client, result.observation.applicant_profile)
            result = await env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_to_str(action), reward=reward, done=done, error=error)

            if done:
                break

        success = any(r > 0 for r in rewards)

    finally:
        avg_reward = sum(rewards) / max(len(rewards), 1)
        score = max(0.01, min(0.99, avg_reward))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        print("ERROR: No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.")
        sys.exit(1)

    rebuild_docker_image(LOCAL_IMAGE_NAME)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await CreditAssessmentEnv.from_docker_image(LOCAL_IMAGE_NAME, provider=SlowStartProvider())

    try:
        task_name_map = {v: k for k, v in TASKS.items()}
        task_name_map.update({str(k): k for k in TASKS})

        if TASK_NAME == "all":
            task_ids = list(TASKS.keys())
        else:
            task_id = task_name_map.get(TASK_NAME.lower())
            if task_id is None:
                print(
                    f"ERROR: Unknown TASK_NAME '{TASK_NAME}'. "
                    "Use 'all', '1', '2', '3', 'personal-loan', 'vehicle-loan', or 'home-loan'."
                )
                sys.exit(1)
            task_ids = [task_id]

        for task_id in task_ids:
            task_label = TASKS[task_id]
            for ep in range(EPISODES_PER_TASK):
                await run_episode(
                    env=env,
                    llm_client=llm_client,
                    task_id=task_id,
                    task_label=task_label,
                    seed=SEED + ep,
                )

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
