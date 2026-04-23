"""Fair head-to-head evaluation: base model vs trained adapter.

Why this script exists
----------------------
The original baseline-vs-trained comparison in train_grpo.py uses two different
evaluation paths for the two models:

  * Baseline goes through evaluate_by_loan_type() (strict json.loads, no markdown
    stripping, 30 samples per type via filtering a pool of 90).
  * Trained goes through evaluate_model() (lenient parser that strips ```json
    fences, exactly 60 samples).

Because Qwen-2.5-Instruct loves wrapping JSON in markdown code fences while the
GRPO-trained policy was rewarded for emitting raw JSON, the strict baseline
parser silently scores correct-but-fenced baseline responses as wrong. The two
paths also generate *different applicants* (different num_samples to the same
seed cycles different cases), so per-task numbers are not directly comparable.

This script fixes both issues. It:

  1. Generates a single fixed test set (same seed, same difficulty mix).
  2. Runs the SAME 60-120 applicants through the base model and the
     base+adapter model.
  3. Uses ONE lenient parser for both (handles ```json fences, ``` fences,
     leading/trailing prose, common Qwen formatting tics).
  4. Reports per-task accuracy AND a 95% Wilson confidence interval, so you
     know which deltas are statistically meaningful versus noise.
  5. Writes fair_eval_results.json + a labelled chart to assets/.

Usage
-----
Run from the repo root, on a machine with the trained adapter available:

    # Adapter on disk
    python scripts/fair_eval.py \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --adapter-path /content/credit-assessment-env/grpo_curriculum_end_snapshot \\
        --num-samples 120 \\
        --output-dir assets/

    # Adapter on the Hugging Face Hub
    python scripts/fair_eval.py \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --adapter-repo iamnijin/credit-assessment-curriculum \\
        --num-samples 120

On A100, 120 samples through both models takes ~10 minutes.

Sample size guidance
--------------------
At 40 samples per loan type (num_samples=120) the 95% Wilson CI on a single
proportion is roughly +/-15pp. At 20 samples per type it is +/-22pp, which is
why the original eval cannot distinguish a real effect from sampling noise.
Bump num_samples to 240 if you want CIs of about +/-10pp per task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# train_grpo provides the canonical SYSTEM_PROMPT and applicant generator,
# so the test set is identical to what the model was trained against.
from train_grpo import SYSTEM_PROMPT, generate_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Lenient response parser used identically for base and trained models
# ---------------------------------------------------------------------------

VALID_DECISIONS = {"approve", "reject", "request_docs", "counter_offer"}


def parse_decision(response: str) -> Optional[str]:
    """Extract a normalised decision from a model response.

    Handles all the formats the base Qwen-Instruct model emits in practice:
      * raw JSON                                  -> {"decision": "approve", ...}
      * fenced JSON                               -> ```json\n{...}\n```
      * unlabelled fences                         -> ```\n{...}\n```
      * JSON preceded or followed by prose        -> "Sure, here is my answer:\n{...}"
      * decision spelled with capitals/whitespace -> "Approve "

    Returns the lowercased decision string if parsing succeeds, else None.
    """
    if not response:
        return None

    text = response.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    if "```json" in text:
        try:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        except IndexError:
            pass
    elif "```" in text:
        try:
            text = text.split("```", 1)[1].split("```", 1)[0]
        except IndexError:
            pass

    text = text.strip()

    # Try direct JSON parse first.
    decision = _try_json(text)
    if decision is not None:
        return decision

    # Fallback: find the first {...} block and try parsing that.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        decision = _try_json(text[start : end + 1])
        if decision is not None:
            return decision

    return None


def _try_json(text: str) -> Optional[str]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    decision = parsed.get("decision")
    if not isinstance(decision, str):
        return None
    decision = decision.strip().lower()
    return decision if decision in VALID_DECISIONS else None


# ---------------------------------------------------------------------------
# Statistics: Wilson score interval (better than normal approx at small n)
# ---------------------------------------------------------------------------


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion.

    Preferred over the normal approximation when n is small (<30) or the
    proportion is near 0 or 1. Returns (low, high) bounds in [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    correct: int = 0
    total: int = 0
    parse_failures: int = 0
    decisions: list[str] = field(default_factory=list)
    ground_truths: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.total)


def run_evaluation(
    model,
    tokenizer,
    samples,
    label: str,
    max_new_tokens: int = 256,
) -> dict[str, TaskResult]:
    """Score `model` on `samples`, returning per-loan-type results.

    `samples` should be the iterable produced by generate_dataset(). The same
    samples are passed to both base and trained models so the comparison is
    apples-to-apples.
    """
    import torch  # local import keeps script importable without GPU stack

    results: dict[str, TaskResult] = {
        "personal": TaskResult(),
        "vehicle": TaskResult(),
        "home": TaskResult(),
    }

    model.eval()
    print(f"\n[{label}] scoring {len(samples)} samples...")

    for idx, sample in enumerate(samples):
        loan_type = sample["loan_type"]
        ground_truth = sample["ground_truth"]

        prompt_text = tokenizer.apply_chat_template(
            sample["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        decision = parse_decision(response)

        bucket = results[loan_type]
        bucket.total += 1
        bucket.ground_truths.append(ground_truth)
        if decision is None:
            bucket.parse_failures += 1
            bucket.decisions.append("<unparseable>")
        else:
            bucket.decisions.append(decision)
            if decision == ground_truth:
                bucket.correct += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(
                f"  [{label}] {idx + 1}/{len(samples)} "
                f"(running personal={results['personal'].accuracy*100:.0f}% "
                f"vehicle={results['vehicle'].accuracy*100:.0f}% "
                f"home={results['home'].accuracy*100:.0f}%)"
            )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def overall_stats(per_task: dict[str, TaskResult]) -> tuple[int, int, float, tuple[float, float]]:
    correct = sum(r.correct for r in per_task.values())
    total = sum(r.total for r in per_task.values())
    acc = correct / total if total else 0.0
    return correct, total, acc, wilson_interval(correct, total)


def format_table(baseline: dict[str, TaskResult], trained: dict[str, TaskResult]) -> str:
    lines = []
    lines.append("=" * 86)
    lines.append("FAIR EVALUATION: same applicants, same parser, both models")
    lines.append("=" * 86)
    header = f"{'Task':<15}{'Baseline':>22}{'Trained':>22}{'Δ (pp)':>12}{'Verdict':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for task in ["personal", "vehicle", "home"]:
        b = baseline[task]
        t = trained[task]
        b_lo, b_hi = b.ci
        t_lo, t_hi = t.ci
        delta_pp = (t.accuracy - b.accuracy) * 100
        # CIs overlap iff baseline_high >= trained_low and baseline_low <= trained_high
        overlap = (b_hi >= t_lo) and (b_lo <= t_hi)
        verdict = "noise" if overlap else ("better" if delta_pp > 0 else "worse")
        lines.append(
            f"{task.capitalize():<15}"
            f"{b.accuracy*100:>6.1f}% [{b_lo*100:>4.0f}-{b_hi*100:>3.0f}]   "
            f"{t.accuracy*100:>6.1f}% [{t_lo*100:>4.0f}-{t_hi*100:>3.0f}]   "
            f"{delta_pp:>+8.1f}    "
            f"{verdict:>10}"
        )
    lines.append("-" * len(header))

    b_correct, b_total, b_acc, b_ci = overall_stats(baseline)
    t_correct, t_total, t_acc, t_ci = overall_stats(trained)
    overlap_overall = (b_ci[1] >= t_ci[0]) and (b_ci[0] <= t_ci[1])
    verdict = "noise" if overlap_overall else ("better" if t_acc > b_acc else "worse")
    lines.append(
        f"{'Overall':<15}"
        f"{b_acc*100:>6.1f}% [{b_ci[0]*100:>4.0f}-{b_ci[1]*100:>3.0f}]   "
        f"{t_acc*100:>6.1f}% [{t_ci[0]*100:>4.0f}-{t_ci[1]*100:>3.0f}]   "
        f"{(t_acc - b_acc)*100:>+8.1f}    "
        f"{verdict:>10}"
    )
    lines.append("=" * len(header))
    lines.append(
        "Confidence intervals are 95% Wilson score intervals. "
        "Verdict 'noise' means the CIs overlap."
    )
    return "\n".join(lines)


def write_results_json(
    out_path: Path,
    baseline: dict[str, TaskResult],
    trained: dict[str, TaskResult],
    meta: dict,
) -> None:
    def serialize(per_task):
        return {
            task: {
                "correct": r.correct,
                "total": r.total,
                "parse_failures": r.parse_failures,
                "accuracy": r.accuracy,
                "ci_low": r.ci[0],
                "ci_high": r.ci[1],
            }
            for task, r in per_task.items()
        }

    b_correct, b_total, b_acc, b_ci = overall_stats(baseline)
    t_correct, t_total, t_acc, t_ci = overall_stats(trained)

    payload = {
        "meta": meta,
        "baseline": {
            "per_task": serialize(baseline),
            "overall": {
                "correct": b_correct,
                "total": b_total,
                "accuracy": b_acc,
                "ci_low": b_ci[0],
                "ci_high": b_ci[1],
            },
        },
        "trained": {
            "per_task": serialize(trained),
            "overall": {
                "correct": t_correct,
                "total": t_total,
                "accuracy": t_acc,
                "ci_low": t_ci[0],
                "ci_high": t_ci[1],
            },
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))


def write_chart(
    out_path: Path,
    baseline: dict[str, TaskResult],
    trained: dict[str, TaskResult],
    base_label: str,
    trained_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping chart")
        return

    tasks = ["personal", "vehicle", "home"]
    b_acc = [baseline[t].accuracy * 100 for t in tasks]
    t_acc = [trained[t].accuracy * 100 for t in tasks]
    b_err = [
        [b_acc[i] - baseline[tasks[i]].ci[0] * 100 for i in range(3)],
        [baseline[tasks[i]].ci[1] * 100 - b_acc[i] for i in range(3)],
    ]
    t_err = [
        [t_acc[i] - trained[tasks[i]].ci[0] * 100 for i in range(3)],
        [trained[tasks[i]].ci[1] * 100 - t_acc[i] for i in range(3)],
    ]

    x = list(range(len(tasks)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [xi - width / 2 for xi in x], b_acc, width,
        label=base_label, color="#cf6e6e", yerr=b_err, capsize=4,
    )
    ax.bar(
        [xi + width / 2 for xi in x], t_acc, width,
        label=trained_label, color="#4f9d6c", yerr=t_err, capsize=4,
    )

    for xi, val in zip(x, b_acc):
        ax.text(xi - width / 2, val + 1, f"{val:.0f}%", ha="center", fontsize=9)
    for xi, val in zip(x, t_acc):
        ax.text(xi + width / 2, val + 1, f"{val:.0f}%", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Personal\n(Easy)", "Vehicle\n(Medium)", "Home\n(Hard)"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Fair head-to-head: same applicants, same parser, 95% Wilson CIs")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_models(base_model: str, adapter_source: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {base_model}...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, device_map="auto"
    )

    print(f"Loading adapter from {adapter_source}...")
    from peft import PeftModel
    trained = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, device_map="auto"
    )
    trained = PeftModel.from_pretrained(trained, adapter_source)

    return tokenizer, base, trained


def main():
    parser = argparse.ArgumentParser(description="Fair head-to-head model evaluation.")
    parser.add_argument("--base-model", required=True, help="HF Hub id of the base model.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter-path", help="Local path to a saved PEFT adapter directory.")
    src.add_argument("--adapter-repo", help="HF Hub repo id for a published adapter.")
    parser.add_argument("--num-samples", type=int, default=120,
                        help="Total samples (split evenly across 3 loan types). 120 -> ~40 per type.")
    parser.add_argument("--seed", type=int, default=999, help="Dataset seed for reproducibility.")
    parser.add_argument("--difficulty", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--output-dir", default="assets", help="Where to save fair_eval outputs.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_source = args.adapter_path or args.adapter_repo

    print(f"Generating fair test set: {args.num_samples} samples, "
          f"seed={args.seed}, difficulty={args.difficulty}")
    samples = list(generate_dataset(args.num_samples, seed=args.seed, difficulty=args.difficulty))

    by_type = {"personal": 0, "vehicle": 0, "home": 0}
    for s in samples:
        by_type[s["loan_type"]] += 1
    print(f"  Distribution: personal={by_type['personal']} "
          f"vehicle={by_type['vehicle']} home={by_type['home']}")

    tokenizer, base_model, trained_model = load_models(
        args.base_model, adapter_source, dtype=args.dtype
    )

    baseline_results = run_evaluation(
        base_model, tokenizer, samples, label="baseline",
        max_new_tokens=args.max_new_tokens,
    )

    # Free baseline weights before scoring trained model if memory is tight.
    del base_model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    trained_results = run_evaluation(
        trained_model, tokenizer, samples, label="trained",
        max_new_tokens=args.max_new_tokens,
    )

    print()
    print(format_table(baseline_results, trained_results))

    json_path = output_dir / "fair_eval_results.json"
    chart_path = output_dir / "fair_eval_chart.png"
    write_results_json(
        json_path, baseline_results, trained_results,
        meta={
            "base_model": args.base_model,
            "adapter_source": adapter_source,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "difficulty": args.difficulty,
            "samples_per_type": by_type,
            "parser": "lenient (markdown-fence stripping)",
        },
    )
    write_chart(
        chart_path, baseline_results, trained_results,
        base_label=f"Baseline ({os.path.basename(args.base_model)})",
        trained_label="Trained (curriculum + adversarial)",
    )

    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {chart_path}")
    print()
    print("Use the per-task verdict column when updating the slide deck and README.")
    print("If a delta is marked 'noise', do not claim improvement or regression on that task.")


if __name__ == "__main__":
    main()
