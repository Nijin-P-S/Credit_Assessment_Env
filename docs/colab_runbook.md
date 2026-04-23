# Colab Runbook (v3 Pipeline)

This is the exact, ordered sequence to run in Google Colab tomorrow on **Colab Pro A100** to get **demonstrable improvement** vs baseline. Designed for the 100-credit budget.

---

## Pre-flight (do this BEFORE opening Colab)

1. **Use Chrome.** Safari put your tab to sleep last time and you lost 2 hours of training.
2. **Pin the Colab tab.** Right-click → Pin Tab. Browsers throttle background tabs less aggressively when pinned.
3. **Disable system sleep** for ~6 hours (System Settings → Lock Screen → Display turn off after = Never; or use [Amphetamine](https://apps.apple.com/us/app/amphetamine/id937984704)).
4. **Have your HF write token ready.** Get it from <https://huggingface.co/settings/tokens> (write scope). You'll paste it in Colab when prompted.

---

## Stage 0 — open notebook + select runtime (~2 min)

1. Open the notebook from your repo: `train_grpo_colab.ipynb`.
2. Runtime → Change runtime type → **A100 GPU**, **High RAM** if available.
3. Runtime → Connect to a hosted runtime (top-right corner).

**Checkpoint:** GPU shows ~40 GB or 80 GB available.

---

## Stage 1 — install + clone + set HF token (~5 min, ~1 credit)

Run cells **1, 2, 3** in order.

After cell 3 (the clone cell), open a new code cell and run **once**:

```python
import os
os.environ["HF_TOKEN"] = "hf_YOUR_WRITE_TOKEN_HERE"
!huggingface-cli login --token $HF_TOKEN --add-to-git-credential
```

(Don't commit this cell. It's just for this session.)

**Checkpoint:** `huggingface-cli login` prints "Login successful".

---

## Stage 2 — sanity check baseline (~7 min, ~2 credits)

Run cells **4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15** (skip cell 13 — we'll create the trainer after SFT).

> **Do NOT run cell 13 yet.** It builds the GRPO trainer. We want to build it AFTER SFT so it picks up the SFT-warmed adapter.

**Checkpoint after cell 15:**
- Per-task accuracy printed for personal/vehicle/home.
- Sample CoT response shown — should look like prose reasoning followed by a fenced JSON block.
- **Target:** baseline accuracy in the 50-65% range with sensible-looking CoT. If it's <40% or the CoT looks broken (no JSON, repeated tokens), STOP and ping for help. The new prompt isn't doing what we expect.

---

## Stage 3 — SFT warmup (~30 min, ~7 credits)

Run cell **20** (Section 7a).

**While it runs:**
- Watch the loss in the streaming output. SFT loss should drop from ~2.0 at step 0 to ~0.3-0.5 by the end of epoch 2.
- If loss is **still > 1.5 after 50 steps**, kill the cell and report — the LR or data is wrong.
- If you see "OutOfMemoryError", lower `--per-device-batch-size` to 1 in cell 20 and rerun.

**Checkpoint after cell 20:**
- "SFT warmup complete." printed.
- The big "CRITICAL NEXT STEP" banner is visible.
- `./grpo_credit_assessment_sft/` directory exists (LoRA adapter saved).

---

## Stage 4 — rebuild trainer with SFT init (~3 min, <1 credit)

Re-run cell **12** then cell **13**.

**Checkpoint:**
- Cell 12 prints `SFT init: YES (./grpo_credit_assessment_sft)`.
- Cell 13 prints `Trainer created with SFT-warmed adapter as starting policy.`
- Cell 13 prints `Trainable params from SFT adapter: trainable params: ~XXX | all params: ~7.6B | trainable%: ~0.X%`.

If cell 12 prints `SFT init: NO`, the SFT directory is missing — check stage 3 output.

---

## Stage 5 — curriculum training (~140 min, ~30 credits)

Run cell **22** (Section 7b).

**The cell auto-runs 3 phases (Personal → Vehicle → Home) with replay.** Each phase ~45 min. You'll see:

```
Phase 1: Personal Loans (Foundation)
  Samples: 400 | Loan type: personal
  ... GRPO logs ...
  Accuracy: NN.N% (threshold: 60%)
  Pushing phase adapter to HF Hub: iamnijin/credit-assessment-curriculum-phase1-personal ...
```

**Per-phase health checks:**
- Reward should be **trending up** within a phase. Look at the `loss` and `reward_mean` in the trainer logs every 10 steps.
- If reward stays flat or drops by >10% across the phase, kill and report.
- Phase 1 should hit ≥60% (mastery threshold). Phase 2 same. Phase 3 has no gate.

**If phase 1 needs a retry**, the cell auto-retries once with a fresh random seed. If still <60%, the cell advances anyway.

**If Colab disconnects mid-phase**, the per-phase Hub push means you've at least saved the last completed phase. Reconnect, re-run cells 1-13 (with `SFT_INIT_DIR` already set from disk via auto-detect), and resume from the next phase by setting `phase_idx = N` manually.

**Checkpoint after cell 22:**
- "CURRICULUM RESULTS" table shows accuracy for all 3 phases.
- `./grpo_curriculum_model/` exists.
- 3 Hub repos exist: `iamnijin/credit-assessment-curriculum-phase{1,2,3}-{personal,vehicle,home}`.

---

## Stage 6 — adversarial training (~65 min, ~14 credits)

Run cell **24** (Section 7c).

**This auto-runs 2 adversarial rounds with self-generation.** You'll see strategy weakness analysis after each round, then targeted training, then re-eval.

**Watch for:**
- After round 1, "Δ: +X%" per loan type. Some can be negative; that's fine.
- After round 2, the regression check at the end: "Curriculum vs Curriculum + Adversarial".
  - If `Δ ≥ +0.03` → use `./grpo_adversarial_final` for the demo.
  - If `Δ ≤ -0.03` → use `./grpo_curriculum_end_snapshot` (it was saved at the start of this cell as a fallback).

**Checkpoint after cell 24:**
- `./grpo_adversarial_final/` exists.
- `./grpo_curriculum_end_snapshot/` exists.
- Regression check printed.

---

## Stage 7 — final push to Hub (~3 min, <1 credit)

Run cell **26** (Section 7d).

This pushes whatever is currently in `trainer.model` to `iamnijin/credit-assessment-curriculum`.

**If adversarial regressed and you want curriculum-only on Hub instead**, before running cell 26 do:

```python
# Reload the snapshot into trainer.model
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM
del trainer.model
torch.cuda.empty_cache()
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
trainer.model = PeftModel.from_pretrained(base, "./grpo_curriculum_end_snapshot", is_trainable=False)
```

Then run cell 26.

**Checkpoint:**
- `https://huggingface.co/iamnijin/credit-assessment-curriculum` is updated.

---

## Stage 8 — fair head-to-head eval (~25 min, ~5 credits) — **THIS IS THE PROOF**

Run cell **31** (Section 8b).

The cell auto-detects the best on-disk adapter (prefers `./grpo_adversarial_final`, falls back to `./grpo_curriculum_model`). To override:

```python
import os
os.environ["FAIR_EVAL_ADAPTER"] = "./grpo_curriculum_model"  # explicit override
```

**Output:**
- `assets/fair_eval_results.json` — exact numbers + Wilson 95% CIs
- `assets/fair_eval_chart.png` — the bar chart you'll show in the slide deck
- Inline display of the chart at the end

**Checkpoint:**
- Trained-model accuracy is **higher** than baseline accuracy on the same 120 samples.
- Wilson CIs do NOT overlap (or only barely overlap).
- This is the screenshot for slide 6.

---

## Stage 9 — collect outputs + download (~3 min, <1 credit)

Run cells **33** (results narrative), **34, 35** (training curves), **41, 42** (zip + download).

You'll get a `training_outputs.zip` with everything: results JSON, plots, training log, fair_eval outputs.

---

## Total budget

| | minutes | credits |
|---|---:|---:|
| Best case (no retries) | 282 | ~62 |
| Worst case (2 phase retries + fair-eval retry) | 401 | ~87 |
| **Your budget** | — | **100** |
| Margin | — | **~13** |

You have headroom for one full re-run of any single phase if needed.

---

## Failure escalation triggers (when to STOP and report)

| Symptom | Action |
|---|---|
| Baseline sanity (stage 2) accuracy < 40% | STOP. Prompt is broken. |
| SFT loss stays > 1.5 after 50 steps | STOP. Gold data or LR is wrong. |
| Phase 1 accuracy < 30% even after retry | STOP. SFT didn't transfer. |
| `cuda out of memory` anywhere | Restart runtime, lower batch_size to 1, retry from last completed stage. |
| Hub push fails with 401 | Re-run the `huggingface-cli login` cell. |
| Notebook hangs > 10 min with no output | Kernel is stuck. Restart runtime, resume from last completed stage. |

---

## What "success" looks like at the end

The fair-eval chart shows **trained accuracy ≥ baseline + 8 percentage points** with non-overlapping Wilson CIs on at least 2 of 3 loan types. That's the proof.
