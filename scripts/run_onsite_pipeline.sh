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
#   HUB_MODEL_ID            — adapter repo  (default: iamnijin/credit-assessment-onsite-YYYYMMDD;
#                             date-stamped so re-runs on different days don't collide,
#                             AND so we never overwrite the published Colab adapters
#                             at iamnijin/credit-assessment-curriculum* )
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
# Default adapter repo: date-stamped so it's obvious this came from the
# onsite run AND so a re-run on a different day creates a fresh repo
# (no overwrite of yesterday's evidence). Pass HUB_MODEL_ID=... to override
# if you need to re-run the same day or want a different name.
export HUB_MODEL_ID="${HUB_MODEL_ID:-iamnijin/credit-assessment-onsite-$(date -u +%Y%m%d)}"
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
echo "  Adapter repo:     ${HUB_MODEL_ID}"
echo "    (per-phase:     ${HUB_MODEL_ID}-phase{N}-{loan_type})"
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
        echo "       HUB_MODEL_ID=iamnijin/credit-assessment-onsite-\$(date -u +%Y%m%d)"
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
               assets/reward_curve.png \
               assets/grpo_loss.png \
               assets/per_task_accuracy.png; do
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
    echo "─── Step 1/3: SFT warmup ───"
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
    echo "─── Step 2/3: Curriculum GRPO ───"
    # train_grpo.py auto-detects ./grpo_credit_assessment_sft/ and uses it as
    # the GRPO init policy. PUSH_PER_PHASE=1 pushes per phase so a mid-run
    # crash leaves a usable adapter on the Hub.
    python -u train_grpo.py

    echo ""
    echo "─── Step 3/3: n=120 fair-eval ───"
    # Pull the adapter we just pushed from the Hub (proves the round-trip).
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
