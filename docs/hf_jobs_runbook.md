# HF Jobs Runbook — Onsite Training

The playbook for tomorrow's onsite training run on Hugging Face Jobs.
Goal: train the curriculum-GRPO adapter live in front of judges, capture the
full training log + reward curve, and push every artifact to a public HF
Dataset repo so judges can verify training happened (not just talked about).

---

## Budget (vs the $30 hackathon credit)

| What | Hardware | $/hr | Time | Cost |
|---|---|---|---|---|
| Smoketest | `l4x1` | $0.80 | ~5 min | **$0.07** |
| Full pipeline (curriculum + adversarial + n=120 eval) | `a100-large` | $2.50 | ~2.5 hr | **$6.25** |
| **Total expected spend** | | | | **~$6.32** |
| **Headroom for re-runs / debugging** | | | | **~$23.68** |

So you can run the smoketest 2-3 times and the full pipeline twice and
still have $15+ left. The $30 budget is **not** a constraint here.

---

## Hard prerequisite — must verify TONIGHT

> **HF Jobs requires Pro/Team/Enterprise**, separate from the credits.
> The hackathon's $30 credit grant explicitly says "subscription fees
> excluded" — it covers the per-minute compute consumption but does NOT
> pay for the Pro subscription that gates `hf jobs run` access. Most
> hackathons grant temporary Pro access *alongside* the credits, but you
> must confirm yours actually got that. A free-tier account will fail
> with 401/403 the moment you try `hf jobs run`.

**Tonight, run these three checks on your laptop:**

```bash
# 1. Confirm `hf` CLI is installed and you're logged in as the right user.
hf auth whoami
#   → should print "iamnijin" (or whatever profile owns the credit grant).
#   If it says a different user, run `hf auth logout && hf auth login`.

# 2. Confirm Jobs is reachable for your account (cheapest possible call).
hf jobs ps
#   → should print "JOB ID  STATUS  ..." (empty list is fine).
#   If it prints "Jobs is only available on Pro / Team / Enterprise" → STOP.
#   The credits won't work without a Pro/org plan. Resolve this with the
#   hackathon organisers before tomorrow.

# 3. Confirm hardware list shows a100-large.
hf jobs hardware
#   → should print a table including "a100-large" with $/hr.
```

If all three pass, you're good. Skip ahead to **TL;DR**.

---

## TL;DR — copy-paste path

```bash
# 1. Smoketest (~$0.30, ~3-4 min, runs on L4 — proves plumbing)
hf jobs run \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --timeout 30m \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -c '
    set -e
    apt-get update -qq && apt-get install -qq -y git
    cd /tmp && git clone https://github.com/Nijin-P-S/Credit_Assessment_Env repo
    cd repo
    pip install -q -r requirements-train.txt huggingface_hub
    bash scripts/run_onsite_smoketest.sh
  '

# 2. Real pipeline (~$6.25 of your $30, ~2.5 hr, runs on A100)
#    SFT (25 min) → curriculum GRPO (75 min) → adversarial (~30 min)
#                 → n=120 fair-eval (10 min) → upload everything
hf jobs run \
  --flavor a100-large \
  --secrets HF_TOKEN \
  --timeout 4h \
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

**Important syntax notes:**
- `image` (`pytorch/pytorch:...`) is a **positional** argument that comes
  AFTER the flags but BEFORE the `bash -c '...'` command. Don't try to use
  `--image`; that flag doesn't exist.
- `--secrets HF_TOKEN` (with no `=value`) tells the CLI to pull `HF_TOKEN`
  from your local environment / cached login and pass it into the job.
- `--timeout 4h` defaults to 30 minutes if omitted — your 2-hour pipeline
  WILL get killed if you forget this.
- The shell scripts (`run_onsite_*.sh`) live in the repo and do all the
  heavy lifting. The `bash -c '...'` block is just the bootstrap.

After the pipeline finishes, two things land on the Hub:

**1. The trained adapters** at date-stamped repos so they can never collide
with the already-published Colab adapters:

| Repo | Contents | Created when |
|---|---|---|
| `iamnijin/credit-assessment-onsite-YYYYMMDD-phase1-personal` | SFT + GRPO Phase 1 | After Phase 1 finishes |
| `iamnijin/credit-assessment-onsite-YYYYMMDD-phase2-vehicle` | + GRPO Phase 2 | After Phase 2 finishes |
| `iamnijin/credit-assessment-onsite-YYYYMMDD-phase3-home` | + GRPO Phase 3 (≡ **curriculum-only final**) | After Phase 3 finishes |
| `iamnijin/credit-assessment-onsite-YYYYMMDD` | + Adversarial round (≡ **curriculum + adversarial final**, fed into `fair_eval`) | End of pipeline |

So judges can compare the two adapters side-by-side, mirroring the published
Colab pattern (`iamnijin/credit-assessment-curriculum` vs
`iamnijin/credit-assessment-adversarial`):
- "Curriculum only" = the `...-phase3-home` repo
- "Curriculum + adversarial" = the date-stamped main repo

The `YYYYMMDD` is computed at run time (UTC). If you re-run the same day
and want a separate repo, override with:
`-e HUB_MODEL_ID=iamnijin/credit-assessment-onsite-attempt2` in the
`hf jobs run` invocation.

The pipeline has a hard guard that **refuses to start** if `HUB_MODEL_ID`
matches any of the known published Colab repos
(`iamnijin/credit-assessment-curriculum`, `...-adversarial`, or any of the
three `...-phaseN-...` variants). So your published headline adapters are
safe even if someone fat-fingers an env var.

**2. The training artifacts** under a timestamped subdir of the dataset
repo [`iamnijin/credit-assessment-training-logs`](https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs):

| File | What it proves |
|---|---|
| `run-<ts>/training_log.json` | Full run metadata + per-step reward curve (every 10 steps) |
| `run-<ts>/stdout.log` | Full unbuffered console output of the entire run |
| `run-<ts>/fair_eval_results.json` | n=120 head-to-head with Wilson CIs |
| `run-<ts>/fair_eval_chart.png` | Per-task accuracy chart |

---

## What this pipeline does

`scripts/run_onsite_pipeline.sh` runs in **one** HF Job (single container)
because HF Job containers are ephemeral — splitting SFT and GRPO into
separate jobs would require pushing the SFT adapter to the Hub between
them. Keeping everything in one job means SFT writes to local disk and
GRPO reads it directly. Saves a code change to `sft_warmup.py` and one
class of failure mode.

```
┌────────────────────────────────────────────────────────────┐
│  Single A100-large HF Job (~2.5 hours, ~$6.25)             │
│                                                            │
│  Step 1: sft_warmup.py                                     │
│    600 examples · 2 epochs · LoRA r=32                     │
│    → ./grpo_credit_assessment_sft/  (local disk)           │
│                                                            │
│  Step 2: train_grpo.py                                     │
│    Auto-detects SFT adapter on disk.                       │
│    Curriculum: personal → vehicle → home                   │
│    400 samples/phase · 8 generations · KL β=0.05           │
│    PUSH_PER_PHASE=1 → adapter pushed to Hub after each     │
│    phase (mid-run crash leaves a usable artifact).         │
│    HF_PUSH_CHECKPOINTS=0 → no slow per-save uploads.       │
│    USE_ADVERSARIAL=1 → adds ~30min adversarial round       │
│      after curriculum, mirroring published headline.       │
│    Final: writes training_log.json (incl. reward_curve)    │
│                                                            │
│  Step 3: fair_eval.py                                      │
│    Pulls adapter from the Hub (round-trip proof).          │
│    n=120 head-to-head vs Qwen-2.5-7B-Instruct base.        │
│    Writes assets/fair_eval_results.json + chart.           │
│                                                            │
│  EXIT trap: uploads everything to                          │
│    iamnijin/credit-assessment-training-logs/run-<ts>/      │
│    Runs even on partial failure — partial logs > no logs.  │
└────────────────────────────────────────────────────────────┘
```

To **disable** adversarial (curriculum-only ~$5, ~2 hr), insert
`-e USE_ADVERSARIAL=0` into the `hf jobs run` command. The default keeps
it ON because at $2.50/hr A100 and a $30 budget, the extra $1.25 is
worth a richer demo and matching the published headline pattern.

---

## Tomorrow morning, in order

1. **Final identity check** — `hf auth whoami` on your laptop.
2. **Smoketest** (Step 1 above). The CLI will stream logs; expect
   `✓ All probes passed` in 3-4 min. Note the URL it prints — that's the
   live job page on huggingface.co.
3. If smoketest passes → **kick off the real pipeline** (Step 2 above).
4. **Note the job ID** that prints right after launch. You can stream
   logs in another terminal:
   ```bash
   hf jobs logs <job-id> -f
   ```
5. While it runs, do the demo. The pipeline doesn't need babysitting:
   - The EXIT trap guarantees partial uploads on failure.
   - `PUSH_PER_PHASE=1` puts each phase's adapter on the Hub the moment
     it finishes.

---

## What to show judges (live)

While the pipeline runs:

1. **Stream the logs in a terminal**:
   ```bash
   hf jobs logs <job-id> -f
   ```
   This is your "training is happening live, not pre-baked" proof.

2. **Refresh the model repo page** (it gets a new commit per phase). The
   exact URL depends on today's date — check the pipeline's first echo
   block, but it'll look like:
   `https://huggingface.co/iamnijin/credit-assessment-onsite-YYYYMMDD`

3. **After it finishes, point to the dataset repo**:
   <https://huggingface.co/datasets/iamnijin/credit-assessment-training-logs>

   Open `run-<ts>/training_log.json` in the browser — show the
   `reward_curve` array of `{step, reward, kl}` entries. That's
   ~30-60 measurements depending on dataset size. Then open `stdout.log`
   for the full console trace.

---

## Failure modes and recovery

| Symptom | Cause | Recovery |
|---|---|---|
| `hf jobs run` prints "Pro / Team / Enterprise required" | Free-tier account | Resolve with hackathon organisers BEFORE tomorrow |
| `HF_TOKEN env var not set` | Forgot `--secrets HF_TOKEN` | Re-run with the flag |
| `hf upload` fails with 401/403 | Token is read-only | Regenerate write-scoped token at https://huggingface.co/settings/tokens |
| `git: command not found` inside container | pytorch image lacks git | The bootstrap `apt-get install -y git` line should fix this; if not, switch image to `huggingface/transformers-pytorch-gpu` |
| OOM during SFT | Wrong flavor (L4 instead of A100) | Re-run with `--flavor a100-large` |
| OOM during GRPO | Container memory pressure | Re-run; `PUSH_PER_PHASE=1` saved earlier phases |
| Job hits `--timeout 4h` | Pipeline ran longer than expected | Logs and any pushed phase adapters survive; rerun fair_eval separately against the partial adapter on Hub |
| `git clone` fails | GitHub temporary issue | Retry; cheap |
| Pipeline crashes mid-way | Various | The EXIT trap uploads partial training_log.json + stdout.log to the dataset repo — open it to debug |

If the pipeline crashes mid-curriculum:
- The phase that finished is on the Hub at
  `iamnijin/credit-assessment-onsite-YYYYMMDD-phase{N}-{loan_type}`
  (substitute today's UTC date and the highest-numbered phase that
  completed — check the pipeline's stdout log for the exact URL printed
  after each phase push).
- Run fair_eval against that phase to still get *some* number to show
  (replace `YYYYMMDD` with today's date):
  ```bash
  hf jobs run --flavor l4x1 --secrets HF_TOKEN --timeout 30m \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    bash -c '
      set -e
      apt-get update -qq && apt-get install -qq -y git
      cd /tmp && git clone https://github.com/Nijin-P-S/Credit_Assessment_Env repo
      cd repo
      pip install -q -r requirements-train.txt huggingface_hub
      python scripts/fair_eval.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --adapter-repo iamnijin/credit-assessment-onsite-YYYYMMDD-phase2-vehicle \
        --num-samples 60 --output-dir /tmp/eval/
    '
  ```

---

## Worst case: live training breaks entirely

Fall back to the **already-published** results:
- Adapter:  <https://huggingface.co/iamnijin/credit-assessment-curriculum>
- Adversarial: <https://huggingface.co/iamnijin/credit-assessment-adversarial>
- Numbers: see `assets/fair_eval_results_curriculum_n120.json` (committed to repo)
- Pitch: "We trained this on Colab to validate the pipeline — the onsite
  run is a reproducibility demonstration. The headline numbers come from
  the n=120 evaluation already in the README."

This is a **legitimate fallback** — the published numbers are real,
reproducible, and statistically sound. The onsite run is bonus credibility,
not the only proof of work.
