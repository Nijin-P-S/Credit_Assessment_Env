#!/usr/bin/env python3
"""Generate all slide-deck visual assets in one shot.

Outputs land in ``assets/slides/`` so they don't clutter the README plot
folder. Re-run any time to regenerate.

Produces:
- reward_bar.png          (slide 4)
- self_improvement_loop.png (slide 5)
- applicant_profile.txt   (slide 3 — text snippet)
- validator_screenshot.png (slide 8)
- file_tree.png           (slide 9)
- qr_hf_space.png         (slide 10)
- qr_colab.png            (slide 10)
- qr_github.png           (slide 10)

All PNGs use the deck palette documented in docs/slide_deck.md:
- Background: #0F0F12 (near-black)
- Text:       #F4E9D8 (warm cream)
- Pink:       #FF6F91 (primary accent)
- Blue:       #8AB6D6 (secondary accent)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Import path for train_utils (sample profile)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Palette
BG = "#0F0F12"
TEXT = "#F4E9D8"
PINK = "#FF6F91"
BLUE = "#8AB6D6"
DIM = "#6B6B75"

OUT = ROOT / "assets" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "savefig.facecolor": BG,
    "savefig.edgecolor": BG,
})


# -------------------------------------------------------------------
# 1. Reward asymmetry bar chart  (slide 4)
# -------------------------------------------------------------------
def build_reward_bar():
    scenarios = [
        ("Correct decision", 10, "#6FCF97"),
        ("Reject a good applicant", -5, BLUE),
        ("Approve a bad loan", -15, "#F2994A"),
        ("Approve non-RERA home loan", -20, PINK),
    ]
    labels = [s[0] for s in scenarios]
    values = [s[1] for s in scenarios]
    colors = [s[2] for s in scenarios]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.barh(labels, values, color=colors, edgecolor=TEXT, linewidth=1.2)

    for bar, val in zip(bars, values):
        offset = 0.8 if val >= 0 else -0.8
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+d}", ha=ha, va="center",
            fontsize=18, fontweight="bold", color=TEXT,
        )

    ax.axvline(0, color=DIM, linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlim(-24, 14)
    ax.set_xlabel("Raw reward", fontsize=14, color=TEXT)
    ax.set_title(
        "Asymmetric reward — approving a bad loan is 3× worse than rejecting a good one",
        fontsize=14, color=TEXT, pad=14,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(DIM)
    ax.spines["bottom"].set_color(DIM)
    ax.invert_yaxis()  # best scenario on top

    plt.tight_layout()
    out = OUT / "reward_bar.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   {out.name}")


# -------------------------------------------------------------------
# 2. Self-improvement loop diagram  (slide 5)
# -------------------------------------------------------------------
def build_loop_diagram():
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis("off")

    # 5 nodes on a circle
    steps = [
        "Train on\nadversarial cases",
        "Tracker finds\nweakest strategy",
        "Bias next batch\ntoward weakness",
        "Model designs\nits own traps",
        "Rule engine\nverifies cases",
    ]
    n = len(steps)
    radius = 3.4
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    for idx, ((x, y), label) in enumerate(zip(positions, steps)):
        is_hero = idx == 3  # Step 4: self-generation is the ⭐
        face = PINK if is_hero else BG
        edge = PINK if is_hero else BLUE
        tc = BG if is_hero else TEXT

        box = FancyBboxPatch(
            (x - 1.1, y - 0.55), 2.2, 1.1,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            linewidth=2.4, edgecolor=edge, facecolor=face, alpha=1.0,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.22, f"Step {idx + 1}", ha="center", va="center",
                fontsize=10, color=tc, fontweight="bold")
        ax.text(x, y - 0.15, label, ha="center", va="center",
                fontsize=11, color=tc)

    # Arrows between consecutive nodes (clockwise)
    for i in range(n):
        x1, y1 = positions[i]
        x2, y2 = positions[(i + 1) % n]
        # shrink so arrow ends outside the box
        dx, dy = x2 - x1, y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5
        shrink = 1.2
        sx = x1 + (dx / length) * shrink
        sy = y1 + (dy / length) * shrink
        ex = x2 - (dx / length) * shrink
        ey = y2 - (dy / length) * shrink
        arrow = FancyArrowPatch(
            (sx, sy), (ex, ey),
            arrowstyle="-|>", mutation_scale=22,
            color=BLUE, linewidth=2.0,
        )
        ax.add_patch(arrow)

    ax.text(0, 0, "The environment\nimproves its own\ntraining data",
            ha="center", va="center", fontsize=18, color=PINK, fontweight="bold")

    out = OUT / "self_improvement_loop.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   {out.name}")


# -------------------------------------------------------------------
# 3. Sample applicant profile  (slide 3 — text)
# -------------------------------------------------------------------
def build_applicant_sample():
    import random
    try:
        from train_utils import generate_applicant, build_profile_text, calculate_ground_truth
    except ImportError as e:
        print(f"[skip] applicant_profile.txt — {e}")
        return

    random.seed(777)
    applicant = generate_applicant(task_id=3, difficulty="hard")  # home loan trap
    profile = build_profile_text(applicant)
    gt = calculate_ground_truth(applicant)

    text = (
        "Sample Loan Application (task 3 · home loan · hard)\n"
        "-----------------------------------------------------\n"
        f"{profile}\n"
        "-----------------------------------------------------\n"
        f"Correct decision: {gt}\n"
    )
    out = OUT / "applicant_profile.txt"
    out.write_text(text)
    print(f"[ok]   {out.name}")


# -------------------------------------------------------------------
# 4. Validator screenshot  (slide 8)
# -------------------------------------------------------------------
def build_validator_screenshot():
    src = ROOT / "assets" / "validation_output.txt"
    if not src.exists():
        print(f"[skip] validator_screenshot.png — {src.name} missing")
        return

    lines = src.read_text().splitlines()
    n = len(lines)
    # Height scales with line count. Width stays constant for consistent font.
    fig_w, fig_h = 13, 0.32 * n + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#1a1a20")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 2)
    ax.axis("off")

    # Title bar — full width, simple label only (no circle dots since
    # matplotlib circles distort under non-equal aspect ratios).
    ax.add_patch(plt.Rectangle((0, n + 1), 1, 1,
                                facecolor="#2a2a30", edgecolor="none",
                                transform=ax.transData))
    ax.text(0.5, n + 1.5, "validate-submission.sh",
            ha="center", va="center", color=DIM, fontsize=12,
            fontfamily="monospace", fontweight="bold")

    for i, line in enumerate(lines):
        y = n - i + 0.2
        color = TEXT
        if "PASSED" in line or "All" in line or "ready to submit" in line:
            color = "#6FCF97"
        elif "========" in line:
            color = DIM
        elif "Step" in line and "[" in line:
            color = BLUE
        ax.text(0.02, y, line, ha="left", va="center",
                color=color, fontsize=12, fontfamily="monospace")

    out = OUT / "validator_screenshot.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#1a1a20")
    plt.close(fig)
    print(f"[ok]   {out.name}")


# -------------------------------------------------------------------
# 5. File-tree diagram  (slide 9)
# -------------------------------------------------------------------
def build_file_tree():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Tree structure - each item: (x, y, label, color, bold)
    nodes = [
        (0.5, 9.0, "credit_assessment_env/", PINK, True),
        (1.0, 8.1, "├── openenv.yaml", TEXT, False),
        (1.0, 7.4, "├── server/", BLUE, True),
        (1.6, 6.7, "├── generators/", TEXT, True),
        (2.2, 6.0, "│   ├── personal_loan.py", TEXT, False),
        (2.2, 5.3, "│   ├── vehicle_loan.py", TEXT, False),
        (2.2, 4.6, "│   ├── home_loan.py", TEXT, False),
        (2.2, 3.9, "│   └── business_loan.py", DIM, False),
        (1.6, 3.2, "├── ground_truth/  ← add rule fn", TEXT, True),
        (1.6, 2.5, "├── rewards/       ← add reward fn", TEXT, True),
        (1.6, 1.8, "└── router.py       ← register route", TEXT, True),
    ]
    for (x, y, label, color, bold) in nodes:
        weight = "bold" if bold else "normal"
        ax.text(x, y, label, color=color, fontsize=14,
                fontfamily="monospace", fontweight=weight)

    # Highlight box around the 4 files
    ax.add_patch(FancyBboxPatch(
        (0.4, 1.5), 7.5, 5.6,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        linewidth=0, facecolor=PINK, alpha=0.06,
    ))

    # Caption
    ax.text(5, 0.7,
            "Adding a new loan type = 4 file edits. No core changes.",
            ha="center", va="center", color=PINK,
            fontsize=14, fontstyle="italic")

    out = OUT / "file_tree.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[ok]   {out.name}")


# -------------------------------------------------------------------
# 6. QR codes  (slide 10)
# -------------------------------------------------------------------
def build_qr_codes():
    try:
        import qrcode
    except ImportError:
        print("[skip] QR codes — qrcode library not installed")
        return

    targets = {
        "qr_hf_space.png": "https://huggingface.co/spaces/iamnijin/credit-assessment-env",
        "qr_colab.png": "https://colab.research.google.com/github/Nijin-P-S/Credit_Assessment_Env/blob/main/train_grpo_colab.ipynb",
        "qr_github.png": "https://github.com/Nijin-P-S/Credit_Assessment_Env",
    }
    for filename, url in targets.items():
        qr = qrcode.QRCode(
            version=4, error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=12, border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color=TEXT, back_color=BG).convert("RGB")
        img.save(OUT / filename)
        print(f"[ok]   {filename}  ({url})")


def main():
    print(f"Generating slide assets → {OUT}\n")
    build_reward_bar()
    build_loop_diagram()
    build_applicant_sample()
    build_validator_screenshot()
    build_file_tree()
    build_qr_codes()
    print(f"\nDone. Open {OUT} to preview.")


if __name__ == "__main__":
    main()
