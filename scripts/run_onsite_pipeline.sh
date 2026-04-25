#!/usr/bin/env bash
# Onsite training pipeline for HF Jobs.
#
# What this script does (in one HF Job invocation):
#   1. Run SFT warmup        (~25 min on A100, writes ./grpo_credit_assessment_sft/)
#   2. Run curriculum GRPO   (~75 min on A100 — adversarial DISABLED here;
#                             pushes adapter to Hub per phase via PUSH_PER_PHASE=1)
#   3. Run n=120 fair-eval   (~10 min on A100, pulls adapter from Hub)
#   4. Push every artifact   (training_log.json, fair_eval_results.json,
#                             stdout log, plots) to a HF Dataset repo so
#                             judges can verify training happened live.
#
# Why one job, not multiple: HF Job containers are ephemeral. Splitting SFT
# from GRPO would require pushing/pulling the SFT adapter via Hub, which
# adds a code change to sft_warmup.py and another failure mode. One job
# keeps state on the local disk for free.
#
# Failure handling: an EXIT trap uploads whatever artifacts exist, even if
# any step crashes. Per-phase Hub push (PUSH_PER_PHASE=1) means partial
# curriculum results survive a mid-training crash.
#
# Required env vars (set in the HF Job invocation):
#   HF_TOKEN                — write-scoped token (push adapter + dataset)
#
# Optional env vars (sensible defaults baked in):
#   HUB_REPO_PREFIX         — base for per-phase repos
#                             (default: iamnijin/credit-assessment-onsite,
#                              produces -phase1-personal, -phase2-vehicle, -phase3-home)
#   HUB_MODEL_ID            — final adapter repo
#                             (default: ${HUB_REPO_PREFIX}-adversarial,
#                              i.e. iamnijin/credit-assessment-onsite-adversarial)
#                             Mirrors the published Colab adapters with an
#                             "-onsite-" marker so judges can compare side-by-side.
#   LOG_DATASET_REPO        — log artifacts (default: iamnijin/credit-assessment-training-logs)
#   USE_ADVERSARIAL         — "0" → skip adversarial round (~$1.25 / 30 min less);
#                             default "1" matches the published headline pattern
#                             (curriculum + adversarial). At $2.50/hr A100, the full
#                             pipeline is ~$6 of the $30 hackathon credit.
#   SKIP_SFT                — "1" → reuse adapter from a prior run (debugging only)

set -e
set -o pipefail

# ---------------------------------------------------------------------------
# Configuration — these become env vars consumed by train_grpo.py
# ---------------------------------------------------------------------------
# Naming scheme (matches the published Colab adapters with an "-onsite-"
# marker so judges can compare the two runs side-by-side):
#
#   Final (curriculum + adversarial)  iamnijin/credit-assessment-onsite-adversarial
#   Per-phase (P1)                     iamnijin/credit-assessment-onsite-phase1-personal
#   Per-phase (P2)                     iamnijin/credit-assessment-onsite-phase2-vehicle
#   Per-phase (P3)                     iamnijin/credit-assessment-onsite-phase3-home
#
# HUB_REPO_PREFIX is used by train_grpo.py for the per-phase repo names;
# HUB_MODEL_ID is the final adapter repo. Override either env var on the
# command line if you need to re-run without overwriting these.
export HUB_REPO_PREFIX="${HUB_REPO_PREFIX:-iamnijin/credit-assessment-onsite}"
export HUB_MODEL_ID="${HUB_MODEL_ID:-${HUB_REPO_PREFIX}-adversarial}"
export LOG_DATASET_REPO="${LOG_DATASET_REPO:-iamnijin/credit-assessment-training-logs}"
export PUSH_PER_PHASE="${PUSH_PER_PHASE:-1}"
# Disable mid-training checkpoint Hub pushes (slow uploads every save_steps).
# Per-phase push (above) is the resilience net we actually want.
export HF_PUSH_CHECKPOINTS="${HF_PUSH_CHECKPOINTS:-0}"
# Run adversarial by default — matches the published headline pattern
# (curriculum + adversarial → 100% on personal traps). At $2.50/hr A100 the
# extra ~30 min costs ~$1.25, well within the $30 hackathon budget. Disable
# with USE_ADVERSARIAL=0 if you only want curriculum.
export USE_ADVERSARIAL="${USE_ADVERSARIAL:-1}"

RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="run-${RUN_ID}"
STDOUT_LOG="/tmp/onsite_stdout_${RUN_ID}.log"

echo "============================================================"
echo "Onsite GRPO pipeline"
echo "  Run ID:           ${RUN_ID}"
echo "  Final adapter:    ${HUB_MODEL_ID}"
echo "  Per-phase repos:  ${HUB_REPO_PREFIX}-phase{N}-{loan_type}"
echo "  Logs dataset:     ${LOG_DATASET_REPO}/run-${RUN_ID}/"
echo "  Adversarial:      ${USE_ADVERSARIAL} (0 = curriculum only)"
echo "  Per-phase push:   ${PUSH_PER_PHASE}"
echo "  Stdout log file:  ${STDOUT_LOG}"
echo "============================================================"

# Defensive: never overwrite the published Colab adapters. If anyone sets
# HUB_MODEL_ID to one of the known-published names, abort.
case "${HUB_MODEL_ID}" in
    iamnijin/credit-assessment-curriculum|\
    iamnijin/credit-assessment-adversarial|\
    iamnijin/credit-assessment-curriculum-phase1-personal|\
    iamnijin/credit-assessment-curriculum-phase2-vehicle|\
    iamnijin/credit-assessment-curriculum-phase3-home)
        echo ""
        echo "ERROR: HUB_MODEL_ID='${HUB_MODEL_ID}' would overwrite a"
        echo "       published Colab adapter. Pick a different name, e.g."
        echo "       HUB_MODEL_ID=iamnijin/credit-assessment-onsite-adversarial"
        exit 3
        ;;
esac
case "${HUB_REPO_PREFIX}" in
    iamnijin/credit-assessment-curriculum|\
    iamnijin/credit-assessment-adversarial)
        echo ""
        echo "ERROR: HUB_REPO_PREFIX='${HUB_REPO_PREFIX}' would cause per-phase"
        echo "       pushes to overwrite published Colab adapters. Pick e.g."
        echo "       HUB_REPO_PREFIX=iamnijin/credit-assessment-onsite"
        exit 3
        ;;
esac

# Authenticate. `hf` CLI is the modern entry point (huggingface-cli is the
# legacy alias for the same code). Use `hf` to be future-proof.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env var not set. Pass --secrets HF_TOKEN to hf jobs run."
    exit 2
fi
hf auth login --token "$HF_TOKEN" --add-to-git-credential || true

# Idempotent dataset-repo create via Python (more robust than CLI flag
# variations across huggingface_hub versions).
python -c "
from huggingface_hub import HfApi
HfApi().create_repo('${LOG_DATASET_REPO}', repo_type='dataset', exist_ok=True, private=False)
print('Dataset repo ready: https://huggingface.co/datasets/${LOG_DATASET_REPO}')
"

# ---------------------------------------------------------------------------
# Upload helper — used by EXIT trap and the final explicit upload pass.
# Uploads everything that exists; missing files are skipped silently.
# ---------------------------------------------------------------------------
upload_artifacts() {
    echo ""
    echo "── Uploading artifacts to ${LOG_DATASET_REPO}/${RUN_DIR}/ ──"

    # Each artifact is uploaded individually so a single failure doesn't
    # abort the whole batch. `|| true` keeps the trap path safe.
    for src in training_log.json \
               assets/fair_eval_results.json \
               assets/fair_eval_chart.png \
               assets/grpo_loss.png \
               assets/per_task_accuracy.png \
               assets/adversarial_rounds.png \
               assets/curriculum_phases.png; do
        if [ -f "$src" ]; then
            dest="${RUN_DIR}/$(basename "$src")"
            echo "  → $src → $dest"
            hf upload "${LOG_DATASET_REPO}" "$src" "$dest" \
                --repo-type dataset \
                --commit-message "onsite ${RUN_ID}: $(basename "$src")" \
                || echo "  ⚠ upload failed for $src (continuing)"
        fi
    done

    if [ -f "$STDOUT_LOG" ]; then
        echo "  → $STDOUT_LOG → ${RUN_DIR}/stdout.log"
        hf upload "${LOG_DATASET_REPO}" "$STDOUT_LOG" "${RUN_DIR}/stdout.log" \
            --repo-type dataset \
            --commit-message "onsite ${RUN_ID}: stdout" \
            || echo "  ⚠ upload failed for stdout (continuing)"
    fi

    echo "── Upload pass complete ──"
}

# Always attempt to upload on exit, even on failure. Idempotent.
trap 'upload_artifacts || true' EXIT

# ---------------------------------------------------------------------------
# Pipe everything below to the stdout log file (and the live console).
# ---------------------------------------------------------------------------
{
    echo ""
    echo "─── Step 1/4: SFT warmup ───"
    if [ "${SKIP_SFT:-0}" = "1" ] && [ -d "./grpo_credit_assessment_sft" ]; then
        echo "SKIP_SFT=1 and ./grpo_credit_assessment_sft/ exists — skipping SFT."
    else
        python -u sft_warmup.py \
            --num-samples 600 \
            --num-epochs 2 \
            --learning-rate 2e-5 \
            --output-dir ./grpo_credit_assessment_sft
    fi

    echo ""
    echo "─── Step 2/4: Curriculum + adversarial GRPO ───"
    # train_grpo.py auto-detects ./grpo_credit_assessment_sft/ and uses it as
    # the GRPO init policy. PUSH_PER_PHASE=1 pushes per phase so a mid-run
    # crash leaves a usable adapter on the Hub.
    python -u train_grpo.py

    echo ""
    echo "─── Step 3/4: Generate plots from training_log.json ───"
    # Plots are generated BEFORE fair_eval so a fair_eval failure (e.g. the
    # final adapter push didn't land on Hub) doesn't lose us the plots —
    # set -e + pipefail would otherwise abort before we get here. Plots only
    # need training_log.json which train_grpo.py writes at end of Step 2.
    if [ -f training_log.json ]; then
        python -u scripts/generate_plots.py training_log.json --out assets/ \
            || echo "⚠ plot generation failed (continuing — JSON log is the source of truth)"
    else
        echo "⚠ training_log.json missing — skipping plots"
    fi

    echo ""
    echo "─── Step 4/4: n=120 fair-eval ───"
    # Pull the final adapter from the Hub (proves the push round-trip and
    # produces the headline cold-Qwen vs trained metric in fair_eval_results.json).
    python -u scripts/fair_eval.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --adapter-repo "${HUB_MODEL_ID}" \
        --num-samples 120 \
        --output-dir assets/

    echo ""
    echo "─── Pipeline complete ───"
} 2>&1 | tee "$STDOUT_LOG"

# Explicit final upload (in addition to the EXIT trap) so the success path
# gets a clean final commit message on the dataset repo.
upload_artifacts

echo ""
echo "============================================================"
echo "Done. Artifacts at:"
echo "  https://huggingface.co/datasets/${LOG_DATASET_REPO}/tree/main/${RUN_DIR}"
echo "============================================================"
