"""Reward function regression tests.

The reward schedule is the spine of training. These tests lock in:
- The +10 reward on correct decisions
- The asymmetric penalties (approve-bad == 3x worse than reject-good)
- The RERA-violation super-penalty (-20)
- The procedural penalties (-8) for skipping a required step
"""

import pytest

from train_utils import (
    CreditAssessmentAction,
    LoanDecision,
    calculate_ground_truth,
    calculate_reward,
)


def _act(decision, amount=None, docs=None):
    return CreditAssessmentAction(
        decision=decision,
        reasoning="test",
        counter_offer_amount=amount,
        docs_requested=docs,
    )


def _personal(**overrides):
    base = dict(
        loan_type="personal",
        credit_score=750,
        monthly_income=100_000,
        foir=0.30,
        employment_years=5,
        loan_amount=500_000,
        documents_complete=True,
    )
    base.update(overrides)
    return base


def _vehicle(**overrides):
    base = dict(
        loan_type="vehicle",
        credit_score=750,
        monthly_income=100_000,
        foir=0.30,
        employment_years=5,
        loan_amount=800_000,
        documents_complete=True,
        collateral_value=1_000_000,
        ltv_ratio=0.80,
    )
    base.update(overrides)
    return base


def _home(**overrides):
    base = dict(
        loan_type="home",
        credit_score=750,
        monthly_income=200_000,
        foir=0.30,
        employment_years=5,
        loan_amount=5_000_000,
        documents_complete=True,
        collateral_value=7_000_000,
        ltv_ratio=0.71,
        rera_registered=True,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Correct decision always yields +10
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "applicant, correct",
    [
        (_personal(), LoanDecision.APPROVE),
        (_personal(credit_score=650), LoanDecision.REJECT),
        (_personal(documents_complete=False), LoanDecision.REQUEST_DOCS),
        (_vehicle(), LoanDecision.APPROVE),
        (_vehicle(ltv_ratio=0.90), LoanDecision.COUNTER_OFFER),
        (_home(), LoanDecision.APPROVE),
        (_home(rera_registered=False), LoanDecision.REJECT),
        (_home(loan_amount=10_000_000, ltv_ratio=0.78), LoanDecision.COUNTER_OFFER),
    ],
)
def test_correct_decision_scores_10(applicant, correct):
    gt = calculate_ground_truth(applicant)
    amount = 500_000 if correct == LoanDecision.COUNTER_OFFER else None
    docs = "all" if correct == LoanDecision.REQUEST_DOCS else None
    reward = calculate_reward(_act(correct, amount=amount, docs=docs), applicant, gt)
    assert reward == 10.0


# ---------------------------------------------------------------------------
# Asymmetric penalty: approving a bad loan is 3x rejecting a good one
# ---------------------------------------------------------------------------

def test_approve_bad_personal_penalty_is_minus_15():
    bad = _personal(credit_score=620)
    gt = calculate_ground_truth(bad)
    assert gt == "reject"
    assert calculate_reward(_act(LoanDecision.APPROVE), bad, gt) == -15.0


def test_approve_bad_vehicle_penalty_is_minus_15():
    bad = _vehicle(credit_score=620)
    gt = calculate_ground_truth(bad)
    assert gt == "reject"
    assert calculate_reward(_act(LoanDecision.APPROVE), bad, gt) == -15.0


def test_reject_good_personal_penalty_is_minus_5():
    good = _personal()
    gt = calculate_ground_truth(good)
    assert gt == "approve"
    assert calculate_reward(_act(LoanDecision.REJECT), good, gt) == -5.0


def test_asymmetry_approve_bad_is_worse_than_reject_good():
    bad = _personal(credit_score=620)
    good = _personal()
    gt_bad = calculate_ground_truth(bad)
    gt_good = calculate_ground_truth(good)
    approve_bad = calculate_reward(_act(LoanDecision.APPROVE), bad, gt_bad)
    reject_good = calculate_reward(_act(LoanDecision.REJECT), good, gt_good)
    assert approve_bad < reject_good
    assert approve_bad == reject_good * 3  # -15 == -5 * 3


# ---------------------------------------------------------------------------
# RERA super-penalty: approving a non-RERA home loan is -20 (worst case)
# ---------------------------------------------------------------------------

def test_approve_non_rera_home_penalty_is_minus_20():
    non_rera = _home(rera_registered=False)
    gt = calculate_ground_truth(non_rera)
    assert gt == "reject"
    reward = calculate_reward(_act(LoanDecision.APPROVE), non_rera, gt)
    assert reward == -20.0


def test_rera_penalty_exceeds_regular_bad_approve():
    """Approving a non-RERA home loan must be strictly worse than any other
    approve-bad mistake, because RERA breach is a regulatory liability."""
    non_rera = _home(rera_registered=False)
    plain_bad = _home(credit_score=620)
    gt_a = calculate_ground_truth(non_rera)
    gt_b = calculate_ground_truth(plain_bad)
    r_a = calculate_reward(_act(LoanDecision.APPROVE), non_rera, gt_a)
    r_b = calculate_reward(_act(LoanDecision.APPROVE), plain_bad, gt_b)
    assert r_a < r_b


# ---------------------------------------------------------------------------
# Procedural penalty: approve/reject when GT says request_docs = -8
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [_personal, _vehicle, _home])
def test_approve_when_docs_missing_penalty_is_minus_8(builder):
    applicant = builder(documents_complete=False)
    gt = calculate_ground_truth(applicant)
    assert gt == "request_docs"
    assert calculate_reward(_act(LoanDecision.APPROVE), applicant, gt) == -8.0


# ---------------------------------------------------------------------------
# Counter-offer specific rewards
# ---------------------------------------------------------------------------

def test_vehicle_counter_offer_without_amount_when_gt_is_approve_penalized():
    """KNOWN BEHAVIOR: the ``counter_offer`` without amount penalty (-3) only
    fires when the ground truth is NOT ``counter_offer``. When GT matches,
    the correct-decision reward (+10) wins the dispatch ordering. Changing
    that ordering would change the training signal, so we lock the current
    behavior in."""
    v = _vehicle()  # clean profile, GT = approve
    gt = calculate_ground_truth(v)
    assert gt == "approve"
    reward = calculate_reward(_act(LoanDecision.COUNTER_OFFER, amount=None), v, gt)
    assert reward == -3.0


def test_vehicle_counter_offer_with_amount_gets_full_reward():
    v = _vehicle(ltv_ratio=0.90)
    gt = calculate_ground_truth(v)
    reward = calculate_reward(_act(LoanDecision.COUNTER_OFFER, amount=700_000), v, gt)
    assert reward == 10.0
