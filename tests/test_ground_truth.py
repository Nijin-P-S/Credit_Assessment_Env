"""Ground truth regression tests.

Locks in the exact decision boundaries used by the environment's rule engine.
Any drift between these tests and the generators/trap strategies indicates a
behaviour change that could silently break training.
"""

import pytest

from train_utils import calculate_ground_truth


def _personal(**overrides):
    base = {
        "loan_type": "personal",
        "credit_score": 750,
        "monthly_income": 100_000,
        "foir": 0.30,
        "employment_years": 5,
        "loan_amount": 500_000,
        "documents_complete": True,
    }
    base.update(overrides)
    return base


def _vehicle(**overrides):
    base = {
        "loan_type": "vehicle",
        "credit_score": 750,
        "monthly_income": 100_000,
        "foir": 0.30,
        "employment_years": 5,
        "loan_amount": 800_000,
        "documents_complete": True,
        "collateral_value": 1_000_000,
        "ltv_ratio": 0.80,
    }
    base.update(overrides)
    return base


def _home(**overrides):
    base = {
        "loan_type": "home",
        "credit_score": 750,
        "monthly_income": 200_000,
        "foir": 0.30,
        "employment_years": 5,
        "loan_amount": 5_000_000,
        "documents_complete": True,
        "collateral_value": 7_000_000,
        "ltv_ratio": 0.71,
        "rera_registered": True,
        "has_co_applicant": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Documents-incomplete rule fires before all other checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [_personal, _vehicle, _home])
def test_incomplete_docs_always_requests_docs(builder):
    applicant = builder(documents_complete=False)
    assert calculate_ground_truth(applicant) == "request_docs"


# ---------------------------------------------------------------------------
# CIBIL boundary: <700 rejects, >=700 passes the CIBIL check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [_personal, _vehicle, _home])
def test_cibil_699_rejects(builder):
    assert calculate_ground_truth(builder(credit_score=699)) == "reject"


@pytest.mark.parametrize("builder", [_personal, _vehicle, _home])
def test_cibil_700_does_not_reject_on_cibil(builder):
    decision = calculate_ground_truth(builder(credit_score=700))
    assert decision != "reject"


# ---------------------------------------------------------------------------
# FOIR boundary: >0.50 rejects, ==0.50 passes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [_personal, _vehicle, _home])
def test_foir_exactly_50_passes(builder):
    decision = calculate_ground_truth(builder(foir=0.50))
    assert decision != "reject"


@pytest.mark.parametrize("builder", [_personal, _vehicle, _home])
def test_foir_just_above_50_rejects(builder):
    assert calculate_ground_truth(builder(foir=0.51)) == "reject"


# ---------------------------------------------------------------------------
# Employment years: personal/vehicle need 1+, home needs 2+
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [_personal, _vehicle])
def test_personal_and_vehicle_accept_1_year_employment(builder):
    decision = calculate_ground_truth(builder(employment_years=1))
    assert decision != "reject"


@pytest.mark.parametrize("builder", [_personal, _vehicle])
def test_personal_and_vehicle_reject_0_year_employment(builder):
    assert calculate_ground_truth(builder(employment_years=0)) == "reject"


def test_home_rejects_1_year_employment():
    assert calculate_ground_truth(_home(employment_years=1)) == "reject"


def test_home_accepts_2_year_employment():
    assert calculate_ground_truth(_home(employment_years=2)) != "reject"


# ---------------------------------------------------------------------------
# RERA: home loan only; False forces reject regardless of other factors
# ---------------------------------------------------------------------------

def test_non_rera_home_rejects_even_with_perfect_profile():
    applicant = _home(
        credit_score=820,
        foir=0.20,
        employment_years=10,
        rera_registered=False,
    )
    assert calculate_ground_truth(applicant) == "reject"


def test_rera_registered_home_with_clean_profile_approves():
    assert calculate_ground_truth(_home()) == "approve"


# ---------------------------------------------------------------------------
# Vehicle LTV boundary: >0.85 → counter_offer, <=0.85 → approve
# ---------------------------------------------------------------------------

def test_vehicle_ltv_85_approves():
    assert calculate_ground_truth(_vehicle(ltv_ratio=0.85)) == "approve"


def test_vehicle_ltv_86_counter_offer():
    assert calculate_ground_truth(_vehicle(ltv_ratio=0.86)) == "counter_offer"


# ---------------------------------------------------------------------------
# RBI tiered LTV for home loans
# Tier 1: loan <= 30L  -> max LTV 90%
# Tier 2: 30L < loan <= 75L -> max LTV 80%
# Tier 3: loan > 75L -> max LTV 75%
# ---------------------------------------------------------------------------

def test_home_tier1_ltv_90_approves():
    assert calculate_ground_truth(_home(loan_amount=2_500_000, ltv_ratio=0.90)) == "approve"


def test_home_tier1_ltv_91_counter_offer():
    assert calculate_ground_truth(_home(loan_amount=2_500_000, ltv_ratio=0.91)) == "counter_offer"


def test_home_tier2_ltv_80_approves():
    assert calculate_ground_truth(_home(loan_amount=5_000_000, ltv_ratio=0.80)) == "approve"


def test_home_tier2_ltv_81_counter_offer():
    assert calculate_ground_truth(_home(loan_amount=5_000_000, ltv_ratio=0.81)) == "counter_offer"


def test_home_tier3_ltv_75_approves():
    assert calculate_ground_truth(_home(loan_amount=10_000_000, ltv_ratio=0.75)) == "approve"


def test_home_tier3_ltv_76_counter_offer():
    """The infamous '>75L loan with LTV 76%' trap — looks fine under the 80%
    assumption but violates the RBI tier cap."""
    assert calculate_ground_truth(_home(loan_amount=10_000_000, ltv_ratio=0.76)) == "counter_offer"


# ---------------------------------------------------------------------------
# Tier boundary exactly at 30L and 75L
# ---------------------------------------------------------------------------

def test_home_loan_exactly_30L_uses_tier1():
    # At exactly 30L, tier 1 (cap 90%) applies — 85% LTV is fine
    assert calculate_ground_truth(_home(loan_amount=3_000_000, ltv_ratio=0.85)) == "approve"
    # One rupee over crosses into tier 2 (cap 80%) — 85% LTV now violates
    assert calculate_ground_truth(_home(loan_amount=3_000_001, ltv_ratio=0.85)) == "counter_offer"


def test_home_loan_exactly_75L_uses_tier2():
    # At exactly 75L, tier 2 limit (80%) should apply
    assert calculate_ground_truth(_home(loan_amount=7_500_000, ltv_ratio=0.80)) == "approve"
    # One rupee over crosses into tier 3 (75%)
    assert calculate_ground_truth(_home(loan_amount=7_500_001, ltv_ratio=0.80)) == "counter_offer"
