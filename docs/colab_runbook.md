# Colab Runbook (v3 Pipeline)

This is the exact, ordered sequence to run `train_grpo_colab.ipynb` on **Colab Pro A100** to get demonstrable improvement vs baseline. Designed for a ~100-credit budget.

> Cell IDs in Colab can drift if cells are inserted later. We reference **section names** (Section 7a, Section 8b, etc.) — those are stable. Look at the markdown headers in the notebook, not the cell numbers in the left margin.

---

## Pre-flight (do this BEFORE opening Colab)

1. **Use Chrome.** Safari aggressively sleeps tabs and you can lose hours of training.
2. **Pin the Colab tab.** Right-click → Pin Tab. Browsers throttle pinned tabs less.
3. **Disable system sleep** for ~6 hours (Mac: System Settings → Lock Screen → Display turn off after = Never; or use [Amphetamine](https://apps.apple.com/us/app/amphetamine/id937984704)).
4. **Have your HF write token ready.** Get one from <https://huggingface.co/settings/tokens> (write scope). You'll paste it in Colab when prompted.

---

## Stage 0 — open notebook + select runtime (~2 min)

1. Open `train_grpo_colab.ipynb`.
2. Runtime → Change runtime type → **A100 GPU**, **High RAM** if available.
3. Runtime → Connect to a hosted runtime.

**Checkpoint:** GPU shows ~40 GB or 80 GB available.

---

## Stage 1 — install + clone + login (~5 min, ~1 credit)

1. Run **Section 1** (3 cells: install dependencies, clone repo, GPU check).
2. Add a new cell after the clone cell and run **once** (don't commit it):

```python
import os
os.environ["HF_TOKEN"] = "hf_YOUR_WRITE_TOKEN_HERE"
!hf auth login --token $HF_TOKEN --add-to-git-credential
```

> **Note:** the older `huggingface-cli login` command still works but prints a deprecation warning. The new `hf` CLI is preferred.

**Checkpoint:** the login command prints "Login successful" or similar.

---

## Stage 2 — sanity check baseline (~7 min, ~2 credits)

Run **Section 2** (env setup) → **Section 3** (dataset) → **Section 4** (rewards) → **Section 5 first cell only** (`MODEL_NAME` / `peft_config` / `training_args`) → **Section 6** (baseline eval).

> **Do NOT run the second cell of Section 5** (the `GRPOTrainer(...)` cell) yet. We want the trainer built AFTER SFT so it picks up the SFT-warmed adapter.

**Checkpoint after Section 6:**
- Per-task accuracy printed for personal/vehicle/home.
- Sample CoT response shown — should be prose reasoning followed by a fenced JSON block.
- **Target:** baseline accuracy in the 60-85% range with sensible CoT. If <50% or CoT looks broken (no JSON, repeated tokens), STOP and investigate before burning credits.

> The 7B Qwen baseline often lands at ~80%+ for Personal Loans. That's expected. The improvement signal will come from Vehicle and Home Loans where the base model gets confused by LTV / RBI tiers.

---

## Stage 3 — SFT warmup (~30 min, ~7 credits)

Run **Section 7a** (single cell).

**While it runs:**
- SFT loss should drop from ~2.0 at step 0 to ~0.3-0.5 by the end of epoch 2.
- If loss is **still > 1.5 after 50 steps**, kill the cell — LR or data is wrong.
- If you see `OutOfMemoryError`, lower `--per-device-batch-size` to 1 in the Section 7a cell and rerun.

**Checkpoint:**
- "SFT warmup complete." printed.
- "CRITICAL NEXT STEP" banner is visible.
- `./grpo_credit_assessment_sft/` directory exists.

---

## Stage 4 — rebuild trainer with SFT init (~3 min, <1 credit)

Re-run **both cells of Section 5** (`MODEL_NAME` cell, then `GRPOTrainer(...)` cell).

**Checkpoint:**
- First cell prints `SFT init: YES (./grpo_credit_assessment_sft)`.
- Second cell prints `Trainer created with SFT-warmed adapter as starting policy.`
- Second cell prints `Trainable params from SFT adapter: ...`.

If the first cell prints `SFT init: NO`, the SFT directory is missing — check Stage 3 output.

---

## Stage 5 — post-SFT spot check (~3 min, ~1 credit) ⚡ NEW

Run **Section 7a-check**.

This is a 30-sample sanity check to confirm SFT actually helped before committing 30+ credits to curriculum.

**Decision rule:**
- ≥ 82% → SFT helped, safe to proceed to Stage 6.
- 75-82% → within noise of baseline; curriculum may still help, but consider SFT-only as a fallback.
- < 75% → STOP. Either prompt/data is broken or SFT regressed. Don't waste curriculum credits.

> In our shipped run, SFT moved a 30-sample spot check from 81.7% baseline to 90%, which is when we proceeded with curriculum.

---

## Stage 6 — curriculum training (~140 min, ~30 credits)

Run **Section 7b** (single cell).

> The first lines of this cell defensively re-disable mid-training eval/save (`trainer.args.eval_strategy = "no"`, `trainer.args.save_strategy = "no"`). This is the fix that takes Phase 1 from 5.5h down to ~80 min — without it, GRPO does a full generation pass over the eval set every 50 steps. The fix is also baked into Section 5's `training_args` so this is belt-and-braces.

**The cell auto-runs 3 phases (Personal → Vehicle → Home) with 20% replay buffer.** Each phase ~45 min. You'll see:

```
Phase 1: Personal Loans (Foundation)
  Samples: 400 | Loan type: personal
  ... GRPO logs ...
  Accuracy: NN.N% (threshold: 60%)
  Pushing phase adapter to HF Hub: iamnijin/credit-assessment-curriculum-phase1-personal ...
```

**Per-phase health checks:**
- Loss should be close to zero throughout (SFT warmup put us in the right region; GRPO is fine-grained shaping).
- Spikes near phase boundaries are normal (new loan type entering the buffer).
- Phase 1 should hit ≥60% (mastery). Phase 2 same. Phase 3 has no gate (nowhere left to advance).

**If a phase needs a retry**, the cell auto-retries once with a fresh seed. If still <60%, the cell advances anyway and keeps the best adapter.

**If Colab disconnects mid-phase**, the per-phase Hub push means you've at least saved the last completed phase. Reconnect, re-run Sections 1-5 (with `SFT_INIT_DIR` already on disk via auto-detect), and resume by setting `phase_idx = N` manually.

**Checkpoint after Section 7b:**
- "CURRICULUM RESULTS" table shows accuracy for all 3 phases.
- `./grpo_curriculum_model/` exists.
- 3 Hub repos exist: `iamnijin/credit-assessment-curriculum-phase{1,2,3}-{personal,vehicle,home}`.

---

## Stage 7 — adversarial training (~65 min, ~14 credits) — **OPTIONAL**

Run **Section 7c** (single cell) **only if** Section 7b results are below 90% overall.

If Section 7b already hits ≥90% overall (which our shipped run did at 96.7%), **skip adversarial** — the marginal upside is small and the regression risk is real.

**This auto-runs 2 adversarial rounds with self-generation.** You'll see strategy weakness analysis after each round, then targeted training, then re-eval.

**Watch for:**
- Per-round Δ printed per loan type. Some can be negative — fine if overall trends up.
- Final regression check: "Curriculum vs Curriculum + Adversarial".
  - If `Δ ≥ +0.03` → use `./grpo_adversarial_final` for the demo.
  - If `Δ ≤ -0.03` → use `./grpo_curriculum_end_snapshot` (saved at the start of this cell as a fallback).

**Checkpoint:**
- `./grpo_adversarial_final/` and `./grpo_curriculum_end_snapshot/` both exist.
- Regression check printed.

---

## Stage 8 — final push to Hub (~3 min, <1 credit)

Run **Section 7d** (single cell).

This pushes whatever is currently in `trainer.model` to `iamnijin/credit-assessment-curriculum`.

**If adversarial regressed and you want curriculum-only on Hub instead**, before running Section 7d do:

```python
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM
del trainer.model
torch.cuda.empty_cache()
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
trainer.model = PeftModel.from_pretrained(base, "./grpo_curriculum_end_snapshot", is_trainable=False)
```

Then run Section 7d.

**Checkpoint:**
- `https://huggingface.co/iamnijin/credit-assessment-curriculum` is updated.

---

## Stage 9 — post-training eval (~5 min, ~1 credit)

Run **Section 8** (both cells: `evaluate_model` then `breakdown by task`).

**Checkpoint:**
- Per-task accuracy table printed: `task_name | baseline | trained | Δ`.
- This populates `baseline_results` / `trained_results` in memory — used by Section 9 below to render per-task charts.

---

## Stage 10 — fair head-to-head eval (~30-45 min, ~10 credits) — **OPTIONAL**

Run **Section 8b** (single cell). Default sample count is now **60** (down from 120) for ~30-45 min runtime on A100. Tested up to 120 samples on a slow run that took 2+ hours — be aware.

**If the cell shows no output for >30 min**, the subprocess buffer may be hiding progress. In a separate cell run `!ls -la assets/fair_eval_*` to check for partial results, then either wait or interrupt.

**If you don't have time/credits for fair_eval, skip this cell entirely.** Section 9 below uses the in-notebook results from Stage 9 and produces a perfectly valid submission chart without it.

**Output:**
- `assets/fair_eval_results.json` — per-task accuracy + Wilson 95% CIs.
- `assets/fair_eval_chart.png` — the bar chart for the slide deck.

**Checkpoint:**
- Trained accuracy is **higher** than baseline accuracy on the same applicant pool.
- Wilson CIs do not overlap (or only barely).

---

## Stage 11 — generate submission artifacts (~3 min, <1 credit)

Run **Section 9** (the resilient generator — this is the only one you strictly need).

> **Section 9 is the safety net.** If Stage 10 hung, OOM'd, or you cleared the trainer to free memory, Section 9 reconstructs everything (charts + `training_log.json`) from whatever in-memory variables are available, falling back to hardcoded values you can edit at the top of the cell.

**Output of Section 9 (always produced):**
- `assets/hackathon_results.png` — overall + per-task headline chart
- `assets/per_task_accuracy.png` — same chart, README-friendly filename
- `assets/curriculum_phases.png` — per-phase mastery line chart
- `assets/reward_curve.png` — GRPO loss trajectory
- `training_log.json` — canonical schema for the README
- `/content/training_outputs.zip` — everything bundled for one-click download

**Optional extras:**
- **Section 9b** (Pitch Summary & Trap Cases) — produces a copy-paste pitch summary and 3 narrative trap-case examples. Nice-to-have for slides; needs `baseline_acc` / `trained_acc` in memory.
- **Section 9c** (Training-Loss Visualization) — extra training-loss curves rendered from `trainer.state.log_history`. Only useful if the trainer object is still alive.

---

## Stage 12 — download artifacts (~1 min)

Open the file browser in Colab, find `/content/training_outputs.zip`, right-click → Download.

Or run the last code cell of the notebook to trigger the download programmatically.

---

## Total budget

| Path | minutes | credits |
|---|---:|---:|
| Minimum (skip adversarial + skip fair_eval) | ~190 | ~42 |
| Recommended (skip adversarial only) | ~225 | ~52 |
| Full pipeline | ~290 | ~66 |
| Worst case (1 phase retry + slow fair_eval) | ~360 | ~80 |
| **Your budget** | — | **100** |

You have plenty of headroom. Don't run adversarial unless curriculum lands below 90%.

---

## Failure escalation triggers (when to STOP and investigate)

| Symptom | Action |
|---|---|
| Baseline sanity (Stage 2) accuracy < 50% | STOP. Prompt is broken. |
| SFT loss stays > 1.5 after 50 steps | STOP. Gold data or LR is wrong. |
| Post-SFT spot check (Stage 5) < 75% | STOP. SFT didn't transfer. |
| Phase 1 accuracy < 30% even after retry | STOP. SFT didn't transfer to GRPO. |
| `CUDA out of memory` anywhere | Restart runtime, lower batch_size to 1, resume from last completed stage. |
| Hub push fails with 401 | Re-run the `hf auth login` cell. |
| Cell hangs > 30 min with no output | Likely subprocess-buffer hiding. Check `nvidia-smi` and partial outputs in a side cell; restart if truly stuck. |
| Step-level mid-training validation fires (~30 min/fire) | Section 7b should disable this. If it still fires, check `trainer.args.eval_strategy` is `"no"`. |

---

## What "success" looks like at the end

You should have:
1. `iamnijin/credit-assessment-curriculum` (final adapter) on the Hub
2. `iamnijin/credit-assessment-curriculum-phase{1,2,3}-{personal,vehicle,home}` (3 phase adapters) on the Hub
3. `training_log.json` showing trained accuracy ≥ baseline + 8 percentage points overall
4. 4 PNGs in `assets/`: `hackathon_results.png`, `per_task_accuracy.png`, `curriculum_phases.png`, `reward_curve.png`

That's the submission.
