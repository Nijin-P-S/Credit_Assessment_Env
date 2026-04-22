"""Contract tests for adversarial case generators.

Every strategy in ``ADVERSARIAL_STRATEGIES`` is supposed to produce a specific
type of trap. If the generator drifts so the case no longer triggers the
expected ground-truth decision, adversarial training is silently undermined
(the model is being trained on cases that don't match their label).

These tests sample each strategy many times and assert that the *expected*
ground truth decision is always produced.
"""

import random

import pytest

from train_utils import (
    ADVERSARIAL_STRATEGIES,
    calculate_ground_truth,
    generate_adversarial_case,
)


# Maps each strategy to the ground-truth decision it must produce.
# If a new strategy is added to ADVERSARIAL_STRATEGIES and not listed here,
# the test_strategy_coverage test below will flag it.
EXPECTED_GT = {
    "threshold_credit":      "reject",          # CIBIL 699
    "threshold_foir":        "reject",          # FOIR 0.51+
    "perfect_but_rera":      "reject",          # RERA=False
    "perfect_but_ltv_tier":  "counter_offer",   # >75L loan, LTV 76-79%
    "coapplicant_trap":      "reject",          # FOIR 0.52-0.58
    "high_income_low_cibil": "reject",          # CIBIL < 700
    "employment_trap_home":  "reject",          # home, employment=1 (needs 2)
    "vehicle_ltv_trap":      "counter_offer",   # vehicle LTV 0.86-0.90
    "docs_incomplete_good":  "request_docs",    # docs_complete=False
    "borderline_multiple":   "approve",         # all fields EXACTLY at threshold
}


def test_strategy_coverage():
    """Every registered strategy must have an expected ground truth."""
    missing = set(ADVERSARIAL_STRATEGIES) - set(EXPECTED_GT)
    assert not missing, f"Adversarial strategies missing expected ground truth: {missing}"


@pytest.mark.parametrize("strategy", list(EXPECTED_GT.keys()))
def test_strategy_produces_expected_ground_truth(strategy):
    """Sample the strategy 25 times; every generated case must agree with
    the expected decision. This catches drift between generator and rule
    engine (e.g., if an RBI threshold moves in one place but not the other)."""
    rng = random.Random(strategy)  # deterministic per-strategy seed
    random.seed(rng.randint(0, 1_000_000))

    expected = EXPECTED_GT[strategy]
    mismatches = []
    for _ in range(25):
        case = generate_adversarial_case(strategy)
        gt = calculate_ground_truth(case)
        if gt != expected:
            mismatches.append((case, gt))

    assert not mismatches, (
        f"Strategy '{strategy}' produced {len(mismatches)}/25 cases that "
        f"did not yield expected ground truth '{expected}'. "
        f"First mismatch: {mismatches[0]}"
    )
