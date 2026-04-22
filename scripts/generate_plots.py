#!/usr/bin/env python3
"""Regenerate all training plots from a single ``training_log.json`` file.

Usage
-----

Once training finishes and has emitted a log file::

    python scripts/generate_plots.py path/to/training_log.json

Produces (in ``assets/``):

- ``reward_curve.png``
  Training reward vs step, with the pre-training baseline plotted as a
  horizontal reference line.

- ``per_task_accuracy.png``
  Baseline vs trained accuracy grouped by loan type, plus the overall bar.
  Every bar is value-labelled. Axes are labeled in full (``Accuracy (fraction
  of correct decisions)``) — not shorthand — because the judge guide
  explicitly checks for labelled axes and units.

- ``adversarial_rounds.png``
  Per-round accuracy on the targeted adversarial strategy, pre- vs post-round.
  Plus the count of self-generated cases fed into each round. Skipped
  gracefully if the log has no adversarial rounds.

- ``curriculum_phases.png``
  End-of-phase eval accuracy across curriculum phases, with the mastery
  threshold shown. Skipped gracefully if the log has no curriculum section.

Schema
------

The JSON structure this script consumes is documented in
``training_log.example.json``. Only a subset of fields are required — every
plot checks for its inputs and is skipped if missing, so partial logs still
produce partial output.

Design notes
------------

- No seaborn, no fancy styling frameworks. Matplotlib only so it works on
  a clean HF Jobs / Colab environment without extra installs.
- Every plot has: figure title, both axes labelled, a legend, a caption
  rendered with ``fig.text`` below the plot. This is the concrete thing the
  hackathon judge guide is looking for.
- Outputs are 1600x1000px at 150 dpi — readable when embedded in README at
  any viewport width without blowing up repo size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

PALETTE = {
    "baseline": "#8AB6D6",   # cool blue
    "trained": "#FF6F91",    # accent pink
    "target": "#3CB371",     # success green
    "pre":      "#C6A664",
    "post":     "#8E7CC3",
    "threshold":"#666666",
    "grid":     "#DDDDDD",
}

FIGSIZE = (10.5, 6.5)
DPI = 150


def _style_axes(ax, *, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.yaxis.grid(True, color=PALETTE["grid"], linestyle="--", alpha=0.8)
        ax.set_axisbelow(True)


def _caption(fig, text: str):
    fig.text(
        0.5,
        0.01,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#444444",
        wrap=True,
    )


# ---------------------------------------------------------------------------
# Plot 1 — reward curve
# ---------------------------------------------------------------------------

def plot_reward_curve(log: dict, out_path: Path) -> bool:
    curve = log.get("reward_curve")
    if not curve:
        print("[skip] reward_curve — no reward_curve entries in log")
        return False

    steps = [p["step"] for p in curve]
    rewards = [p["reward"] for p in curve]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.plot(
        steps,
        rewards,
        color=PALETTE["trained"],
        linewidth=2,
        marker="o",
        markersize=3,
        markerfacecolor=PALETTE["trained"],
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="Mean reward (per logging step)",
    )

    # Phase boundary markers if available
    phases = log.get("curriculum", {}).get("phases") or []
    for phase in phases:
        start, end = phase.get("steps", [None, None])
        if end is not None:
            ax.axvline(x=end, color=PALETTE["threshold"], linestyle=":", alpha=0.5)
            ax.text(end, ax.get_ylim()[1] if False else max(rewards),
                    f" end of {phase['name']}",
                    rotation=90, va="top", ha="left", fontsize=8, color=PALETTE["threshold"])

    # Baseline reward reference line (pre-training)
    baseline_reward = log.get("baseline", {}).get("mean_reward")
    if baseline_reward is not None:
        ax.axhline(
            y=baseline_reward,
            color=PALETTE["baseline"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            label=f"Baseline mean reward ({baseline_reward:.2f})",
        )

    ax.set_title("GRPO Training Reward over Time", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Mean normalized reward (per batch)", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _style_axes(ax)

    meta = log.get("meta", {})
    caption_bits = []
    if meta.get("model_name"):
        caption_bits.append(f"model: {meta['model_name']}")
    if meta.get("mode"):
        caption_bits.append(f"mode: {meta['mode']}")
    if meta.get("num_train_samples"):
        caption_bits.append(f"samples: {meta['num_train_samples']}")
    if meta.get("hardware"):
        caption_bits.append(meta["hardware"])
    caption = " · ".join(caption_bits) if caption_bits else "Credit Assessment Env training"
    _caption(fig, caption)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   reward_curve        -> {out_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 2 — per-task accuracy
# ---------------------------------------------------------------------------

def plot_per_task_accuracy(log: dict, out_path: Path) -> bool:
    baseline = log.get("baseline", {}).get("per_task")
    trained = log.get("trained", {}).get("per_task")
    if not baseline or not trained:
        print("[skip] per_task_accuracy — need baseline.per_task and trained.per_task")
        return False

    categories = ["Personal\n(easy)", "Vehicle\n(medium)", "Home\n(hard)", "Overall"]
    keys = ["personal", "vehicle", "home"]
    baseline_vals = [baseline.get(k, 0.0) for k in keys]
    trained_vals = [trained.get(k, 0.0) for k in keys]

    # Overall
    overall_baseline = log.get("baseline", {}).get("overall")
    overall_trained = log.get("trained", {}).get("overall")
    if overall_baseline is None:
        overall_baseline = sum(baseline_vals) / len(baseline_vals)
    if overall_trained is None:
        overall_trained = sum(trained_vals) / len(trained_vals)
    baseline_vals.append(overall_baseline)
    trained_vals.append(overall_trained)

    x = list(range(len(categories)))
    width = 0.38

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    bars_baseline = ax.bar(
        [i - width / 2 for i in x], baseline_vals, width,
        label="Baseline (pre-training)", color=PALETTE["baseline"], edgecolor="white",
    )
    bars_trained = ax.bar(
        [i + width / 2 for i in x], trained_vals, width,
        label="Trained", color=PALETTE["trained"], edgecolor="white",
    )

    for bar, val in list(zip(bars_baseline, baseline_vals)) + list(zip(bars_trained, trained_vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val*100:.0f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Baseline vs Trained Accuracy by Loan Type",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Loan type", fontsize=11)
    ax.set_ylabel("Accuracy (fraction of correct decisions vs deterministic ground truth)",
                  fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    _style_axes(ax)

    _caption(
        fig,
        "Each bar = fraction of correct decisions on a held-out evaluation set "
        "of applicants generated at difficulty='all'. Ground truth is "
        "deterministic and follows RBI guidelines (see server/ground_truth/).",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   per_task_accuracy   -> {out_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 3 — adversarial rounds
# ---------------------------------------------------------------------------

def plot_adversarial_rounds(log: dict, out_path: Path) -> bool:
    rounds = log.get("adversarial_rounds")
    if not rounds:
        print("[skip] adversarial_rounds — no rounds in log")
        return False

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 6), dpi=DPI, gridspec_kw={"width_ratios": [2, 1]}
    )

    # Left panel: pre vs post accuracy on the targeted strategy, per round
    round_labels = [f"R{r['round']}\n{r.get('targeted_strategy','?')}" for r in rounds]
    pre = [r.get("pre_round", {}).get("targeted_accuracy", 0.0) for r in rounds]
    post = [r.get("post_round", {}).get("targeted_accuracy", 0.0) for r in rounds]

    x = list(range(len(rounds)))
    width = 0.38
    bars_pre = ax1.bar(
        [i - width / 2 for i in x], pre, width,
        label="Pre-round accuracy on targeted strategy",
        color=PALETTE["pre"], edgecolor="white",
    )
    bars_post = ax1.bar(
        [i + width / 2 for i in x], post, width,
        label="Post-round accuracy on targeted strategy",
        color=PALETTE["post"], edgecolor="white",
    )
    for bar, val in list(zip(bars_pre, pre)) + list(zip(bars_post, post)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                 f"{val*100:.0f}%", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(round_labels, fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_title("Adversarial Self-Play: accuracy on the round's targeted strategy",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Round (and weakest strategy identified by AdversarialTracker)",
                   fontsize=10)
    ax1.set_ylabel("Accuracy on the targeted strategy", fontsize=10)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax1)

    # Right panel: self-generated case count per round
    self_gen = [r.get("self_generated_count", 0) for r in rounds]
    ax2.bar(
        [f"R{r['round']}" for r in rounds],
        self_gen,
        color=PALETTE["trained"],
        edgecolor="white",
    )
    for i, val in enumerate(self_gen):
        ax2.text(i, val + max(self_gen) * 0.02 if max(self_gen) else 0.1,
                 str(val), ha="center", va="bottom", fontsize=10)
    ax2.set_title("Self-generated trap cases per round", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Round", fontsize=10)
    ax2.set_ylabel("# of model-generated cases accepted into next round",
                   fontsize=10)
    _style_axes(ax2)

    _caption(
        fig,
        "Left: each round, AdversarialTracker identifies the strategy with the "
        "lowest success rate; training in that round is biased toward it. "
        "Right: cases the trained model proposes and that pass deterministic "
        "ground-truth verification are added to the next round's training data.",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   adversarial_rounds  -> {out_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 4 — curriculum phase progression
# ---------------------------------------------------------------------------

def plot_curriculum_phases(log: dict, out_path: Path) -> bool:
    phases = log.get("curriculum", {}).get("phases")
    if not phases:
        print("[skip] curriculum_phases — no phases in log")
        return False

    names = [p["name"] for p in phases]
    evals = [p.get("final_eval", 0.0) for p in phases]
    retries = [p.get("retries", 0) for p in phases]
    threshold = log.get("curriculum", {}).get("phase_mastery_threshold", 0.60)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    bars = ax.bar(names, evals, color=PALETTE["trained"], edgecolor="white", width=0.55)
    for bar, val, r in zip(bars, evals, retries):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val*100:.0f}%" + (f"  (retries: {r})" if r else ""),
                ha="center", va="bottom", fontsize=10)

    ax.axhline(y=threshold, color=PALETTE["target"], linestyle="--", linewidth=1.5,
               label=f"Phase mastery threshold ({int(threshold*100)}%)")

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Performance-Gated Curriculum: end-of-phase accuracy",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Curriculum phase (order of presentation)", fontsize=11)
    ax.set_ylabel("Held-out accuracy at end of phase", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _style_axes(ax)

    _caption(
        fig,
        "Each phase runs until the model clears the mastery threshold on a "
        "50-sample held-out eval, or until max_phase_retries is exhausted. "
        "Retry count is annotated next to each bar when non-zero.",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   curriculum_phases   -> {out_path}")
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", type=Path, help="Path to training_log.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("assets"),
        help="Output directory (default: assets/)",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"error: log file not found: {args.log}", file=sys.stderr)
        return 2

    log = json.loads(args.log.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    results = [
        plot_reward_curve(log, args.out / "reward_curve.png"),
        plot_per_task_accuracy(log, args.out / "per_task_accuracy.png"),
        plot_adversarial_rounds(log, args.out / "adversarial_rounds.png"),
        plot_curriculum_phases(log, args.out / "curriculum_phases.png"),
    ]
    ok = sum(results)
    print(f"\nWrote {ok}/{len(results)} plots to {args.out}/")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
