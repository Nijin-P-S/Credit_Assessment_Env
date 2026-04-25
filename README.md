---
title: Credit Assessment Environment
emoji: 🏦
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Credit Assessment Environment

### Can an LLM learn to be a loan officer — without ever seeing a real loan?

We hand a 7B language model a stack of synthetic loan applications, the RBI underwriting rulebook, and a reward signal. No banking pre-training. No supervised labels for the rules. Just raw applicant profiles and an environment that scores the decision.

After SFT warmup → per-task curriculum GRPO → one adversarial round, accuracy moves from **80.8% → 94.2%** (+13.3pp, n=120 with non-overlapping Wilson 95% CIs), with **Vehicle +30pp** (the Wilson-CI-clean win), **Personal +5pp** (ceiling), and **Home +5pp**. The trained model and the environment that taught it are both public.

Built on [OpenEnv](https://github.com/facebookresearch/openenv) · trained with [HF TRL](https://github.com/huggingface/trl) · runnable in Colab on a free T4.

---

## Demo & Materials

| Resource | Link |
|---|---|
| 🌐 **Live environment (HF Space)** | [iamnijin/credit-assessment-env](https://huggingface.co/spaces/iamnijin/credit-assessment-env) |
| 🏆 **Trained adapter (headline — curriculum + adversarial)** | [iamnijin/credit-assessment-adversarial](https://huggingface.co/iamnijin/credit-assessment-adversarial) |
| 🔁 **Onsite HF Jobs reproduction (curriculum + adversarial, 2 rounds)** | [iamnijin/credit-assessment-onsite-adversarial](https://huggingface.co/iamnijin/credit-assessment-onsite-adversarial) |
| 📂 **Onsite training logs + plots + fair-eval (run-20260425-105001)** | [HF Dataset · run-20260425-105001](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/tree/main/run-20260425-105001) |
| 📊 **Training log JSON (step-level rewards + curriculum + adversarial)** | [`training_log.json`](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/blob/main/run-20260425-105001/training_log.json) |
| 📈 **Cold-vs-trained chart with Wilson CIs (use this for slides)** | [`fair_eval_chart.png`](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/blob/main/run-20260425-105001/fair_eval_chart.png) · [`fair_eval_results.json`](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/blob/main/run-20260425-105001/fair_eval_results.json) |
| 🤖 Curriculum-only checkpoint (intermediate) | [iamnijin/credit-assessment-curriculum](https://huggingface.co/iamnijin/credit-assessment-curriculum) |
| 🤖 Phase 1 adapter (Personal) | [iamnijin/credit-assessment-curriculum-phase1-personal](https://huggingface.co/iamnijin/credit-assessment-curriculum-phase1-personal) |
| 🤖 Phase 2 adapter (Vehicle) | [iamnijin/credit-assessment-curriculum-phase2-vehicle](https://huggingface.co/iamnijin/credit-assessment-curriculum-phase2-vehicle) |
| 🤖 Phase 3 adapter (Home) | [iamnijin/credit-assessment-curriculum-phase3-home](https://huggingface.co/iamnijin/credit-assessment-curriculum-phase3-home) |
| 🤖 Onsite phase adapters (run-20260425) | [phase1-personal](https://huggingface.co/iamnijin/credit-assessment-onsite-phase1-personal) · [phase2-vehicle](https://huggingface.co/iamnijin/credit-assessment-onsite-phase2-vehicle) · [phase3-home](https://huggingface.co/iamnijin/credit-assessment-onsite-phase3-home) |
| ▶️ **Train it yourself in Colab** | [`train_grpo_colab.ipynb`](train_grpo_colab.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nijin-P-S/Credit_Assessment_Env/blob/main/train_grpo_colab.ipynb) |
| 📺 **Demo video (<2 min)** | [YouTube](https://www.youtube.com/watch?v=d4feqxbc87o) |
| 📝 **Project writeup / blog** | [`docs/blog.md`](docs/blog.md) — "Teaching a Language Model to Be a Loan Officer" |
| 📊 **Slide deck** | [Google Slides](https://docs.google.com/presentation/d/1J6wZbd4EyOqlfikty9F1kk4qLuRmRMtNY7Mk3Qf2ls0/edit?usp=sharing) |
| 🛠 **Colab runbook** | [`docs/colab_runbook.md`](docs/colab_runbook.md) |
| ✅ **Submission validator output** | [`assets/validation_output.txt`](assets/validation_output.txt) (3/3 checks pass) |

---

## Headline Result

**Qwen2.5-7B-Instruct + LoRA, fair head-to-head on n=120 (40 per task) with Wilson 95% CIs:**

![Baseline vs Trained (curriculum + adversarial), n=120, 95% Wilson CIs](assets/hackathon_results.png)

| Loan Type | Baseline | Trained | Δ | 95% CIs overlap? |
|---|---|---|---|---|
| Personal (easy) | 95.0% [83.5, 98.6] | **100%** [91.2, 100] | **+5.0pp** | Just barely (ceiling) |
| Vehicle (medium) | 62.5% [47.0, 75.8] | **92.5%** [80.1, 97.4] | **+30.0pp** ✅ | **No — significant** |
| Home (hard) | 85.0% [70.9, 92.9] | **90.0%** [76.9, 96.0] | **+5.0pp** | Yes (overlap) |
| **Overall (n=120)** | **80.8%** [72.9, 86.9] | **94.2%** [88.4, 97.1] | **+13.3pp** ✅ | **No — significant at p<0.05** |

The base model already reads CIBIL/FOIR well on Personal Loans (95%) and handles the easy "either RERA-yes or clearly broken" Home Loan cases (85%). **Vehicle Loans were the gap**: the base model only managed 62.5% — it couldn't reliably distinguish "approve" from "counter_offer" when LTV was just over 85%, and the LTV-tier traps fooled it consistently. Training closed that gap to 92.5% — a 30-point swing whose Wilson CIs don't overlap with baseline, so the gain is statistically real, not a sampling artifact.

The trained model **strictly beats baseline on every task** (no regression) and the overall +13.3pp delta clears the bar for a publishable result. The result you see is from the **curriculum + adversarial** adapter (`iamnijin/credit-assessment-adversarial`); a pure-curriculum checkpoint (`iamnijin/credit-assessment-curriculum`, 93.3% overall) is also published for ablation.

### 🔁 Independent reproduction on HF Jobs (onsite, 2 adversarial rounds)

The same pipeline was rerun end-to-end on **Hugging Face Jobs** (L40S × 1, ~5h, ~$11) with adversarial rounds bumped from 1 → 2. The fair-eval (n=120, same seed=999, same parser) reproduced the headline within noise:

| Loan Type | Baseline | Trained (onsite, 2 adv rounds) | Δ | Verdict |
|---|---|---|---|---|
| Personal | 95.0% [83-99] | **100%** [91-100] | +5.0pp | noise (ceiling) |
| Vehicle | 67.5% [52-80] | **92.5%** [80-97] | **+25.0pp ✅** | **better** |
| Home | 82.5% [68-91] | **92.5%** [80-97] | +10.0pp | noise |
| **Overall (n=120)** | **81.7%** [74-88] | **95.0%** [90-98] | **+13.3pp ✅** | **better** |

**The +13.3pp overall delta matches the Colab result exactly** — independent reproduction on different hardware, different random seeds during training, with one extra adversarial round, lands on the same statistically-significant gain. Adapter: [`iamnijin/credit-assessment-onsite-adversarial`](https://huggingface.co/iamnijin/credit-assessment-onsite-adversarial). Full training logs, the cold-vs-trained fair-eval chart with Wilson CIs, the fair-eval JSON, and the complete stdout transcript are committed to a public dataset for judge audit: [run-20260425-105001](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/tree/main/run-20260425-105001).

> **Which file to use for the headline:** the cold-Qwen-vs-trained comparison is in **[`fair_eval_chart.png`](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/blob/main/run-20260425-105001/fair_eval_chart.png)** and **[`fair_eval_results.json`](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/blob/main/run-20260425-105001/fair_eval_results.json)**. The other `per_task_accuracy.png` in that same folder is an **internal training-time chart** whose "baseline" is the post-SFT, post-curriculum-but-pre-adversarial model — useful for ablation, not the cold-vs-trained headline. Use `fair_eval_chart.png` for slides, model card, and any judge-facing artifact.

---

## Themes Addressed

This submission addresses **both** themes; the evidence shipped is strongest for #3.1, with the self-improvement loop (#4) demonstrated by the per-phase mastery curve.

### Primary: Theme #3.1 — World Modeling / Professional Tasks
- Real RBI guidelines (tiered LTV, FOIR caps, RERA compliance, employment thresholds) — not toy rules
- Asymmetric reward that mirrors actual NPA economics (approving a bad loan costs 3× more than rejecting a good one; non-RERA breach is 4×)
- Multi-step workflows (`request_docs → re-evaluate`, `counter_offer → recalculate`)
- 10 hand-crafted trap profiles ([`train_utils.py`](train_utils.py)) targeting the specific failure modes RBI rules create (CIBIL boundaries, LTV tiers, RERA blocks, FOIR co-applicant mirage)

### Secondary: Theme #4 — Self-Improvement
- **Performance-gated curriculum** — Personal → Vehicle → Home, gated by per-phase accuracy (60% mastery threshold), not a fixed step count. Each phase produces a checkpoint that becomes the starting policy for the next — that's the self-improvement chain you can see climbing in [`assets/curriculum_phases.png`](assets/curriculum_phases.png)
- **Replay buffer** — past-phase samples are mixed into later phases (default `replay_fraction=0.2`) to prevent catastrophic forgetting
- **Adversarial self-play, dynamically re-targeted** — after the curriculum cleared every mastery gate, we ran adversarial rounds (LR=5e-7, β=0.4, KL anchor to the curriculum reference) trained exclusively on the 10 trap profiles. The `AdversarialTracker` in [`train_utils.py`](train_utils.py) records which of the 10 trap profiles the model fails most each round and re-weights training data toward the worst trap automatically. The Colab run shipped 1 round → [`iamnijin/credit-assessment-adversarial`](https://huggingface.co/iamnijin/credit-assessment-adversarial) (Home 87.5 → 90% on n=40, zero regression elsewhere). The onsite HF Jobs run shipped 2 rounds → [`iamnijin/credit-assessment-onsite-adversarial`](https://huggingface.co/iamnijin/credit-assessment-onsite-adversarial), where Round 1 transferred to lift `perfect_but_rera` 80 → 100% and Round 2 specifically targeted `perfect_but_ltv_tier` and lifted it 0 → 40% — concrete evidence that the tracker-driven self-improvement loop actually moves the needle on previously-unseen trap classes between rounds.

---

## Quick Start

```bash
# Clone the live environment from HF Spaces
git clone https://huggingface.co/spaces/iamnijin/credit-assessment-env
cd credit-assessment-env
uv pip install .
```

```python
from credit_assessment_env import CreditAssessmentAction, CreditAssessmentEnv
from credit_assessment_env.loan_decision import LoanDecision

env = CreditAssessmentEnv(base_url="https://iamnijin-credit-assessment-env.hf.space").sync()

with env:
    result = env.reset()
    print(result.observation.applicant_profile)

    result = env.step(CreditAssessmentAction(
        decision=LoanDecision.APPROVE,
        reasoning="CIBIL 788, FOIR 32%, 5 yrs employment — all green",
    ))
    print(f"Reward: {result.reward}, Done: {result.done}")
```

---

## The Environment

Three loan types of escalating difficulty. Each `reset()` produces a fresh applicant — good, bad, borderline, or one of 10 explicitly-designed **trap profiles** (full list in [§ Themes Addressed](#themes-addressed)).

| Task | Loan Type | Difficulty | What the agent must reason about |
|------|-----------|------------|----------------------------------|
| 1 | Personal | Easy | CIBIL ≥ 700, FOIR ≤ 50%, 1y employment, docs complete |
| 2 | Vehicle | Medium | All of the above + LTV ≤ 85% (counter-offer if exceeded) |
| 3 | Home | Hard | All of the above + **RBI tiered LTV** (90/80/75% by loan size) + **RERA compliance** + 2y employment |

### Action space

```python
class CreditAssessmentAction:
    decision: LoanDecision           # approve | reject | request_docs | counter_offer
    reasoning: str                   # natural-language explanation
    counter_offer_amount: float      # required if decision is counter_offer
    docs_requested: str              # required if decision is request_docs
```

### Observation space

Each observation contains the applicant profile as **narrative text** (so the LLM reads it like a real loan file) plus structured fields:

- Core: `credit_score`, `monthly_income`, `foir`, `employment_years`, `loan_type`, `loan_amount`, `documents_complete`
- Secured loans: `collateral_value`, `ltv_ratio`
- Home loans: `rera_registered`, `has_co_applicant`
- Episode bookkeeping: `task_id`, `reward`, `done`, `available_actions`

### Episode flow

![Episode flow](assets/episode_flow.png)

Multi-step episodes when the agent picks a procedural action:
- `request_docs` → applicant returns with documents complete; agent must re-decide
- `counter_offer` → applicant returns with the reduced loan amount; LTV recomputed

Episodes terminate on a final `approve` or `reject`, or after 3 steps (cap).

### Trap profiles — what makes this hard

The environment ships **10 trap profiles** (defined as `ADVERSARIAL_STRATEGIES` in [`train_utils.py`](train_utils.py): `threshold_credit`, `threshold_foir`, `perfect_but_rera`, `perfect_but_ltv_tier`, `coapplicant_trap`, `high_income_low_cibil`, `employment_trap_home`, `vehicle_ltv_trap`, `docs_incomplete_good`, `borderline_multiple`). The most painful ones for LLMs:

- **Threshold credit (CIBIL 699)** — a single point below the cutoff with everything else perfect. Pattern-matching says approve; rules say reject.
- **Perfect-but-RERA** — 830 CIBIL, ₹2L income, 20% FOIR, **RERA = No**. The −20 reward is the single worst outcome in the environment.
- **LTV tier mismatch** — ₹1.2Cr home loan with LTV 78%. Looks safe under an 80% cap; RBI caps loans > ₹75L at **75%**, so it's a counter-offer.
- **Vehicle LTV trap** — perfect profile, LTV 86% (1 point over the 85% cap). Counter-offer, not approve.
- **Co-applicant mirage** — co-applicant present, but FOIR is computed off the *primary* applicant. Adding a name doesn't reduce risk.

---

## Reward Structure

The reward function reflects the asymmetric cost structure of real banking. Raw rewards are returned per step; the final **grade (0.01–0.99)** is computed by normalising the average reward over `[-20, +10]` using `(avg + 20) / 30`.

| Scenario | Raw Reward | Grade | Why |
|----------|-----------|-------|------|
| Correct decision | **+10.0** | 0.99 | Ideal |
| Request docs when docs missing | +2.0 | 0.73 | Correct procedural step |
| Counter-offer catches high LTV (vehicle) | +5.0 | 0.83 | Smart risk mitigation |
| Reject a good applicant | −5.0 | 0.50 | Lost revenue, recoverable |
| Approve/reject when docs missing | −8.0 | 0.40 | Skipped a required step |
| Approve/reject when counter needed | −8.0 | 0.40 | Ignored LTV that required restructuring |
| **Approve a bad applicant** | **−15.0** | 0.17 | NPA risk — 3× worse than rejecting good |
| **Approve non-RERA home loan** | **−20.0** | 0.01 | Regulatory liability — worst case |
| Counter-offer without specifying amount | −3.0 | 0.57 | Invalid action |
| Other wrong decisions | −2.0 | 0.60 | Wrong but not catastrophic |

**Why this can't be gamed:** rejecting everything tops out around −5 average (lost revenue on every good case). Approving everything blows up on RERA cases at −20. The only reward-maximising strategy is to actually read the profile and apply the rules.

---

## Training Pipeline

The default mode is the full pipeline: **SFT warmup → per-task curriculum (with replay buffer) → optional adversarial round**.

### Configuration

| Parameter | Default | Note |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-7B-Instruct` | Base policy |
| `lora_r` / `lora_alpha` | 32 / 64 | Adapter capacity tuned for stability |
| `learning_rate` | 1e-6 | Conservative (KL anchor handles drift) |
| `beta` (KL coef) | 0.3 | Strong policy-drift control |
| `num_generations` | 8 | More samples per prompt → stable advantages |
| `max_completion_length` | 512 | Room for chain-of-thought reasoning |
| `samples_per_phase` | 400 | Per curriculum phase |
| `replay_fraction` | 0.2 | Past-phase samples mixed into later phases |
| `phase_mastery_threshold` | 0.60 | Required to advance to the next phase |

### Stages

1. **SFT warmup** ([`sft_warmup.py`](sft_warmup.py)) — 600 supervised examples, 2 epochs. Anchors the model on the desired output format (chain-of-thought + JSON) before GRPO starts. *In our run, a quick 30-sample post-SFT spot check (separate from the n=120 fair-eval) returned 90% accuracy, which we used purely as a "format is anchored, safe to launch curriculum GRPO" gate — not as a benchmark number.*
2. **Per-task curriculum** ([`train_grpo.py`](train_grpo.py)) — 3 phases (Personal → Vehicle → Home), 400 samples each, with 20% replay from earlier phases. Produces `iamnijin/credit-assessment-curriculum` (93.3% on n=120).
3. **Adversarial round** ([Section 15 of the Colab notebook](train_grpo_colab.ipynb)) — 50 GRPO steps trained exclusively on the 10 trap profiles, starting from the curriculum adapter, LR=5e-7, β=0.4 to anchor against drift. Produces `iamnijin/credit-assessment-adversarial` (94.2% on n=120 — the Colab headline). The `AdversarialTracker` records per-strategy failure rates so subsequent rounds can re-weight toward the worst trap automatically. The HF Jobs onsite rerun ran **2 rounds** (R1 lifted `perfect_but_rera` 80→100% via transfer; R2 targeted `perfect_but_ltv_tier` and lifted it 0→40%) and produced [`iamnijin/credit-assessment-onsite-adversarial`](https://huggingface.co/iamnijin/credit-assessment-onsite-adversarial) at **95.0% on n=120**.

### Why this combination beats vanilla GRPO

Earlier we tried vanilla GRPO on a mixed-difficulty batch with Qwen2.5-1.5B — overall improved by +10pp but **Vehicle Loans regressed by 14pp** (the model over-fit the high-density CIBIL/RERA patterns and forgot LTV reasoning). Per-task curriculum + replay buffer fixes that exact failure: each phase masters one rule cluster before the next is introduced, and replay keeps earlier rules from being overwritten. **The adversarial round on top adds another +0.83pp overall (and +2.5pp specifically on Home Loans) by surgically targeting the trap profiles the curriculum policy still missed**, while strictly not regressing any task.

---

## Results

### Reward improvement across the curriculum (the headline "rewards going up" plot)

![Per-phase reward — final eval accuracy as the policy advances through the curriculum](assets/curriculum_phases.png)

This is the **reward-improvement** chart for the rubric: each phase's per-task evaluation score (which on this binary-correct task IS the reward signal averaged) climbs across the curriculum chain — Personal 100% → Vehicle 98% → Home 92% on 50 held-out samples per phase during training. All three phases cleared the 60% mastery gate on the first attempt, no retries needed. The y-axis is a direct proxy for mean episode reward because correct = +10 and incorrect outcomes are bounded in [−20, +5]. The adversarial round on top of this curriculum policy then took the held-out n=120 Home Loan accuracy from 87.5% to 90% with no regression elsewhere.

### GRPO training-loss trajectory (auxiliary, 3 phases)

![GRPO training loss across 3 curriculum phases](assets/grpo_loss.png)

*This is a **GRPO loss** trajectory, not a reward chart — included for transparency on training stability, not as the rubric's reward-improvement evidence.* x-axis: training step (0 → 1200 across 3 phases). Vertical dashed lines mark phase boundaries. The loss stays close to zero throughout — that's the SFT warmup paying off (the policy is already in the right region of output space; GRPO is doing fine-grained shaping rather than coarse correction). Spikes correspond to phase transitions when a new loan type's training samples enter the buffer. **For the actual reward-going-up evidence, see the per-phase mastery chart immediately above.**

### Per-task accuracy: baseline vs trained (n=120 with Wilson 95% CIs)

![Per-task accuracy comparison](assets/per_task_accuracy.png)

*This local chart is the **Colab** fair-eval (cold Qwen vs `iamnijin/credit-assessment-adversarial`). The independently-rerun **onsite** version with 2 adversarial rounds is at [`fair_eval_chart.png` in the dataset run folder](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/blob/main/run-20260425-105001/fair_eval_chart.png).*

The Vehicle Loan jump (62.5% → 92.5%) is the headline — non-overlapping CIs so the +30pp gain is statistically real, not a sampling artifact. Personal Loan hits the 40/40 ceiling. Home Loan moves from 85% → 90% after the adversarial round; CIs overlap on this 40-sample slice but the absolute trained number is the highest the project has produced.

### Training-log JSON

The exact baseline / trained / per-task numbers (n=120, Wilson CIs, both adapter checkpoints) are committed in [`training_log.json`](training_log.json) so judges can verify without re-running. Raw per-task tallies for both adapters are in [`assets/fair_eval_results_adversarial_n120.json`](assets/fair_eval_results_adversarial_n120.json) and [`assets/fair_eval_results_curriculum_n120.json`](assets/fair_eval_results_curriculum_n120.json).

```json
{
  "baseline": {"overall": 0.808, "per_task": {"personal": 0.95, "vehicle": 0.625, "home": 0.85}},
  "trained":  {"overall": 0.942, "per_task": {"personal": 1.00, "vehicle": 0.925, "home": 0.90}},
  "delta":    {"overall": "+13.3pp (CIs do not overlap)"},
  "adversarial": {"rounds_run": 1, "delta_vs_curriculum_only": {"home": "+2.5pp", "overall": "+0.83pp"}}
}
```

---

## A Note on Evaluation Methodology

We evaluate the **same applicant pool** for baseline and trained models, with the **same lenient JSON parser** ([`lenient_parser.py`](lenient_parser.py)) that strips markdown fences, handles trailing prose, and falls back to regex extraction. This matters because Qwen-Instruct base models default to wrapping JSON in ```` ```json ```` fences while GRPO-trained models emit raw JSON — a strict parser would silently mark correct base-model answers as wrong and inflate the training delta.

For an even more rigorous head-to-head, [`scripts/fair_eval.py`](scripts/fair_eval.py) loads the base model and the trained adapter sequentially, runs both through identical applicants, and reports per-task accuracy with **95% Wilson confidence intervals**:

```bash
# Reproduce the Colab headline (curriculum + 1 adversarial round):
python scripts/fair_eval.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-repo iamnijin/credit-assessment-adversarial \
  --num-samples 120

# Reproduce the onsite HF Jobs result (curriculum + 2 adversarial rounds, 95.0% overall):
python scripts/fair_eval.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-repo iamnijin/credit-assessment-onsite-adversarial \
  --num-samples 120

# Or evaluate the curriculum-only checkpoint for the ablation:
python scripts/fair_eval.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-repo iamnijin/credit-assessment-curriculum \
  --num-samples 120
```

Output is written to `assets/fair_eval_results.json` and `assets/fair_eval_chart.png`. Both runs reuse the same `seed=999` applicant pool, so any difference between the two adapters is purely the model, not the data.

---

## Reproducibility

### Option A: Open in Colab (free T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nijin-P-S/Credit_Assessment_Env/blob/main/train_grpo_colab.ipynb)

Step-by-step instructions including timing checkpoints and credit budget are in [`docs/colab_runbook.md`](docs/colab_runbook.md). The notebook will pull the latest repo, run the SFT warmup + curriculum, push per-phase adapters to your HF account, and emit `training_log.json` plus all four charts above.

### Option B: Hugging Face Jobs (one-shot, ~$11 / ~5h on L40S)

The whole pipeline (SFT → curriculum → 2 adversarial rounds → fair_eval → upload artifacts) runs in a **single** HF Job. The adapters get pushed to date-stamped repos (so they can never overwrite the published Colab adapters), and every artifact (`training_log.json`, all 4 plots, `fair_eval_results.json`, full stdout transcript) lands in a versioned dataset folder under `iamnijin/credit-assessment-training-logs/run-<timestamp>/` for judge audit.

**Pre-flight (one-time):**

```bash
pip install -U "huggingface_hub[cli]"
hf auth login                    # paste a write-scoped HF token
hf jobs ps                       # confirms HF Jobs is enabled on your account
```

> **Note on access:** `hf jobs run` requires a HF Pro / Team / Enterprise subscription (separate from credits). Free-tier accounts will get 401/403. Hackathons typically grant temporary Pro alongside the credit grant — check yours.

**One-command run** (mirrors what produced the [run-20260425-105001 artifacts](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs/tree/main/run-20260425-105001)):

```bash
hf jobs run \
  --flavor l40sx1 \
  --secrets HF_TOKEN \
  --timeout 12h \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -c '
    set -e
    apt-get update -qq && apt-get install -qq -y git
    cd /tmp && git clone https://github.com/Nijin-P-S/Credit_Assessment_Env repo
    cd repo
    pip install -q -r requirements-train.txt huggingface_hub
    bash scripts/run_onsite_pipeline.sh
  '
```

**Useful overrides** (insert before the `pytorch/pytorch:...` image line):

| Flag | Purpose |
|---|---|
| `-e HUB_REPO_PREFIX=yourname/credit-assessment-onsite` | Use your own HF namespace for the per-phase adapter repos |
| `-e HUB_MODEL_ID=yourname/credit-assessment-onsite-adversarial` | Final adapter repo name |
| `-e USE_ADVERSARIAL=0` | Skip adversarial training (curriculum-only, ~$5, ~3h) |
| `-e GRPO_NUM_ADVERSARIAL_ROUNDS=2` | Default is 2; bump to 3+ for more self-improvement loops |
| `-e LOG_DATASET_REPO=yourname/credit-assessment-training-logs` | Push artifacts to your own dataset repo |
| `--flavor a100-large` | Faster (~3h) but ~3× costlier than L40S; use if L40S is queueing |

**Monitor:**

```bash
hf jobs ps                                # list running jobs
hf jobs logs <JOB_ID> -f                  # follow live stdout
```

The pipeline has a **hard guard** that refuses to start if `HUB_MODEL_ID` matches any of the published Colab repos — so the existing headline adapters can't be accidentally overwritten. Full runbook with troubleshooting + crash recovery in [`docs/hf_jobs_runbook.md`](docs/hf_jobs_runbook.md).

### Option C: Local

```bash
pip install -r requirements-train.txt
python sft_warmup.py --num-samples 600 --num-epochs 2
python train_grpo.py  # uses ./grpo_credit_assessment_sft as warm start
```

Hardware: any GPU with ≥40GB VRAM (A100 / H100 / L40S / RTX 6000 Ada). On L40S × 1 (~$1.80/h on HF Jobs), the full SFT + 3-phase curriculum + 2-round adversarial pipeline takes ~5 hours.

---

## Underwriting Criteria — RBI References

The ground-truth rules are taken from primary sources, not vibes. Each rule below has a citation.

| Rule | Threshold | Reference |
|---|---|---|
| CIBIL minimum | 700 (all loan types) | [HDFC Bank](https://www.hdfcbank.com/personal/products/loans/personal-loans/eligbilty-criteria) · [ICICI](https://www.icici.bank.in/blogs/car-loan/minimum-credit-score-required-to-get-car-loan) · [Bajaj Markets](https://www.bajajfinservmarkets.in/cibil-score/cibil-score-for-home-loan) |
| FOIR cap | 50% (all loan types) | [TechFinServ](https://www.techfinserv.com/blogs/foir/) · [MoneyControl](https://www.moneycontrol.com/news/business/personal-finance/foir-calculation-impact-of-foir-on-personal-loan-eligibility-and-tips-to-improve-it-12970645.html) |
| Vehicle LTV cap | 85% | [Axis Bank](https://axisbank.com/progress-with-us-articles/money-matters/borrow/what-is-a-good-car-loan-ltv-ratio) |
| Home LTV (≤ ₹30L) | 90% | [RBI Master Circular on Housing Finance](https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=6161&Mode=0) |
| Home LTV (₹30–75L) | 80% | RBI Master Circular |
| Home LTV (> ₹75L) | 75% | RBI Master Circular |
| RERA registration | Mandatory for home loan | [Enterslice — No loan to non-RERA builders](https://enterslice.com/learning/no-loan-builders-not-registered-rera/) |
| Employment minimum (personal/vehicle) | 1 year at current employer | [HDFC](https://www.hdfcbank.com/personal/products/loans/personal-loans/eligbilty-criteria) · [Axis](https://www.creditmantri.com/axis-bank-personal-loan-eligibility/) |
| Employment minimum (home) | 2 years | RBI / HDFC home loan eligibility |

---

## Project Structure

```
Credit_Assessment_Env/
├── client.py                  # CreditAssessmentEnv client (Gym-style API)
├── models.py                  # Action / Observation pydantic schemas
├── loan_decision.py           # LoanDecision enum
├── inference.py               # OpenAI-compatible LLM inference entry point
├── baseline.py                # Random / Rule-Based / LLM baselines
├── train_grpo.py              # GRPO + curriculum + adversarial pipeline
├── sft_warmup.py              # Supervised fine-tuning warm start
├── lenient_parser.py          # Robust JSON parser (shared by training + eval)
├── train_grpo_colab.ipynb     # One-click Colab notebook
├── openenv.yaml               # OpenEnv manifest (3 tasks)
├── Dockerfile                 # Environment server
├── server/
│   ├── app.py                                  # FastAPI server
│   ├── credit_assessment_env_environment.py    # Main env (orchestrator)
│   ├── generators/{personal,vehicle,home}_loan.py   # Applicant generators
│   ├── ground_truth/{personal,vehicle,home}_loan.py # Decision rules
│   ├── rewards/{personal,vehicle,home}_loan.py      # Reward shaping
│   └── helpers/profile_builder.py                   # Builds LLM-readable narratives
├── train_utils.py             # AdversarialTracker + 10 trap-profile generators
├── scripts/
│   ├── fair_eval.py           # Apples-to-apples baseline-vs-trained with Wilson CIs
│   └── generate_plots.py      # Re-render all charts from training_log.json
├── tests/                     # 63 pytest cases (ground truth + rewards + adversarial)
├── docs/                      # colab_runbook.md
└── assets/                    # Charts + slide assets
```

### Adding a new loan type

The modular structure makes adding (e.g.) a business loan a 4-file change:

1. `server/generators/business_loan.py` — applicant generator (good/bad/borderline/trap)
2. `server/ground_truth/business_loan.py` — underwriting rules
3. `server/rewards/business_loan.py` — reward shaping
4. Register in `__init__.py` routers + add a new task entry in `TASKS` inside `credit_assessment_env_environment.py`

No changes needed to `models.py`, `client.py`, or the Dockerfile.

---

## Engineering Hygiene

| Check | Status | How |
|---|---|---|
| OpenEnv compliance | ✅ | `openenv validate` passes — see `assets/validation_output.txt` |
| HF Space live | ✅ | `/reset` responds — validator pings it |
| Docker build | ✅ | Validator runs `docker build` |
| Tests | ✅ | 63 pytest cases (`pytest`) — ground truth, rewards, adversarial strategies |
| Client/server separation | ✅ | `client.py` never imports from `server/` |
| Gym API | ✅ | `reset()`, `step()`, `state()` |
| Reproducible eval | ✅ | `training_log.json` committed; `scripts/generate_plots.py` re-renders charts deterministically |

Run the full validator locally:

```bash
./validate-submission.sh
# 3/3 checks passed → assets/validation_output.txt
```

---

## All Agents — Side by Side

The full comparison across deterministic baselines and our LLM agents:

| Agent | Personal | Vehicle | Home | Overall | Methodology |
|---|---|---|---|---|---|
| Random | 0.467 | 0.350 | 0.400 | 0.406 | `baseline.py`, 100 eps/task, seed 42 |
| Qwen2.5-7B baseline (ours) | 0.950 | 0.625 | 0.850 | 0.808 | n=120 (40/task), lenient parser, seed 999 |
| Qwen2.5-7B trained, curriculum-only | 1.000 | 0.925 | 0.875 | 0.933 | Same n=120 head-to-head |
| **Qwen2.5-7B trained, curriculum + adversarial (Colab headline)** | **1.000** | **0.925** | **0.900** | **0.942** | Same n=120 head-to-head |
| **Qwen2.5-7B trained, onsite HF Jobs (curriculum + 2 adv rounds)** | **1.000** | **0.925** | **0.925** | **0.950** | Same n=120 head-to-head, independent rerun |
| Rule-Based (oracle) | 1.000 | 1.000 | 1.000 | 1.000 | `baseline.py`, mirrors `calculate_ground_truth` |

**Reading this table:**
- **Random ≈ 40%** confirms the environment isn't trivially solvable by guessing.
- **Rule-Based = 100%** by construction — confirms rules are internally consistent and the env grading is correct (it's a sanity ceiling, not a competing approach: it sees pre-parsed JSON fields, not raw narratives).
- **Qwen trained (curriculum + adversarial) beats baseline by +13.3pp overall**, with the biggest swing on Vehicle Loans (+30pp) where LTV nuance matters most. The trained model also strictly does not regress on any task. See [Headline Result](#headline-result) for the full table with Wilson 95% CIs.

**Why methodologies differ:** Random and Rule-Based are deterministic — running them on more samples just confirms what they'd do anyway. The two LLM agents share an identical n=120 (40-per-task) head-to-head slice, identical prompt, and identical lenient parser, which is what makes the +13.3pp number defensible — the Wilson 95% CIs for baseline overall ([72.9, 86.9]) and trained overall ([88.4, 97.1]) **do not overlap**, so the gain is statistically significant at p<0.05, not a sampling artifact. `scripts/fair_eval.py` reproduces that exact comparison.

---

## Roadmap

- 🧪 Extend to business / education / gold loans using the 4-file pattern documented above
- 📊 Run `scripts/fair_eval.py` against frontier APIs (GPT-4o-mini, Claude, Gemini) on the same n=120 slice with the matched CoT prompt for a strict head-to-head
- 🔄 Run multiple adversarial rounds with the `AdversarialTracker` actively re-weighting toward the worst-performing trap each round (one round shipped; next ones target the residual 10% Home Loan errors at LTV-tier boundaries)
- 🧠 Self-generated challenges — prompt the trained model to design new trap cases, verify each against deterministic ground truth, then train on them

---

## References

- [RBI Master Circular on Housing Finance](https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=6161&Mode=0)
- [RBI Circular on Consumer Credit Risk Weights (Nov 2023)](https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12567&Mode=0)
- [RERA Act and Bank Lending](https://enterslice.com/learning/no-loan-builders-not-registered-rera/)
- [FOIR Calculation and Impact](https://www.techfinserv.com/blogs/foir/)
- [CIBIL Score Requirements by Loan Type](https://paytm.com/blog/credit-score/the-ultimate-guide-to-cibil-score-requirements-for-different-loan-types-home-car-personal-2/)
- [OpenEnv — Meta](https://github.com/facebookresearch/openenv)
- [HuggingFace TRL — GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
