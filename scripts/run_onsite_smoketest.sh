#!/usr/bin/env bash
# 5-minute smoketest for the onsite HF Job pipeline.
#
# Goal: prove plumbing works (HF auth, GPU available, model loads from Hub,
# fair_eval runs end-to-end, dataset repo writable) WITHOUT actually
# training. We deliberately do NOT run GRPO here — train_grpo.py uses
# hardcoded hyperparams and would take ~75 min. The smoketest validates
# everything else so a bad token / wrong flavor / network issue is caught
# for cents instead of $20.
#
# Cost: ~$0.30 on L4 (~3-4 min wall clock).
# Run this BEFORE the real pipeline.
#
# Required env vars (set in `hf jobs run`):
#   HF_TOKEN — write-scoped token

set -e
set -o pipefail

LOG_DATASET_REPO="${LOG_DATASET_REPO:-iamnijin/credit-assessment-training-logs}"
RUN_ID="smoketest-$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="run-${RUN_ID}"

echo "============================================================"
echo "Onsite pipeline smoketest"
echo "  Run ID:        ${RUN_ID}"
echo "  Logs dataset:  ${LOG_DATASET_REPO}"
echo "============================================================"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env var not set."
    exit 2
fi
hf auth login --token "$HF_TOKEN" --add-to-git-credential || true

echo ""
echo "── Probe 1: Python + PyTorch + GPU ──"
python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA — wrong flavor?'
print('CUDA OK. Device:', torch.cuda.get_device_name(0))
print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
"

echo ""
echo "── Probe 2: HF identity (which user is this token for?) ──"
hf auth whoami

echo ""
echo "── Probe 3: Dataset repo write access ──"
python -c "
from huggingface_hub import HfApi
HfApi().create_repo('${LOG_DATASET_REPO}', repo_type='dataset', exist_ok=True, private=False)
print('Dataset repo ready: https://huggingface.co/datasets/${LOG_DATASET_REPO}')
"

echo ""
echo "── Probe 4: Tiny n=10 fair-eval against the published curriculum adapter ──"
# Proves: base model downloads from Hub, adapter downloads from Hub,
# fair_eval.py runs end-to-end, writes JSON.
python -u scripts/fair_eval.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter-repo iamnijin/credit-assessment-curriculum \
    --num-samples 10 \
    --output-dir /tmp/smoketest_eval/

echo ""
echo "── Probe 5: Upload marker + tiny eval output to dataset repo ──"
cat > /tmp/smoketest_marker.json <<EOF
{
  "run_id": "${RUN_ID}",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "smoketest_passed",
  "probes_passed": ["gpu", "hf_auth", "dataset_write", "fair_eval_e2e"],
  "next_step": "run scripts/run_onsite_pipeline.sh on a100-large"
}
EOF
hf upload "${LOG_DATASET_REPO}" /tmp/smoketest_marker.json "${RUN_DIR}/smoketest_marker.json" \
    --repo-type dataset --commit-message "smoketest ${RUN_ID}: marker"

if [ -f /tmp/smoketest_eval/fair_eval_results.json ]; then
    hf upload "${LOG_DATASET_REPO}" /tmp/smoketest_eval/fair_eval_results.json "${RUN_DIR}/tiny_fair_eval_results.json" \
        --repo-type dataset --commit-message "smoketest ${RUN_ID}: tiny eval"
fi

echo ""
echo "============================================================"
echo "✓ All probes passed. Plumbing is healthy."
echo "  Artifacts: https://huggingface.co/datasets/${LOG_DATASET_REPO}/tree/main/${RUN_DIR}"
echo ""
echo "  Next: launch the real pipeline (see docs/hf_jobs_runbook.md)"
echo "============================================================"
