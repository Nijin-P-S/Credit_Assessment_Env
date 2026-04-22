"""
Standalone training utilities for Credit Assessment Environment.

This module provides self-contained versions of the environment components
for use in training scripts without the full OpenEnv dependency chain.
"""

import random
from enum import Enum
from typing import Optional
from pydantic import BaseModel


# =============================================================================
# LoanDecision Enum
# =============================================================================

class LoanDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_DOCS = "request_docs"
    COUNTER_OFFER = "counter_offer"


# =============================================================================
# Action Model
# =============================================================================

class CreditAssessmentAction(BaseModel):
    decision: LoanDecision
    reasoning: str = ""
    counter_offer_amount: Optional[float] = None
    docs_requested: Optional[str] = None


# =============================================================================
# Profile Builder
# =============================================================================

def build_profile_text(a: dict) -> str:
    """Build human-readable applicant profile for LLM reasoning."""
    loan_type = a["loan_type"]
    
    lines = [
        f"Loan Type: {loan_type.title()} Loan",
        f"Requested Amount: ₹{a['loan_amount']:,.0f}",
        "",
        "--- Applicant Profile ---",
        f"CIBIL Score: {a['credit_score']}",
        f"Monthly Income: ₹{a['monthly_income']:,.0f}",
        f"FOIR (Fixed Obligation to Income Ratio): {a['foir']:.0%}",
        f"Employment: {a['employment_years']} years at current employer",
        f"Documents: {'Complete' if a['documents_complete'] else 'Incomplete'}",
    ]
    
    if loan_type in ("vehicle", "home"):
        lines.append("")
        lines.append("--- Collateral Details ---")
        lines.append(f"Collateral Value: ₹{a.get('collateral_value', 0):,.0f}")
        if a.get("ltv_ratio"):
            lines.append(f"LTV Ratio: {a['ltv_ratio']:.0%}")
    
    if loan_type == "home":
        lines.append("")
        lines.append("--- Property Details ---")
        lines.append(f"RERA Registered: {'Yes' if a.get('rera_registered') else 'No'}")
        lines.append(f"Co-applicant: {'Yes' if a.get('has_co_applicant') else 'No'}")
        if a.get("property_type"):
            lines.append(f"Property Type: {a['property_type']}")
    
    return "\n".join(lines)


# =============================================================================
# Curriculum Difficulty Levels
# =============================================================================

DIFFICULTY_PROFILES = {
    "personal": {
        "easy": ["good", "bad"],
        "medium": ["good", "bad", "borderline"],
        "hard": ["good", "bad", "borderline", "trap_credit", "trap_foir"],
        "all": ["good", "bad", "borderline", "trap_credit", "trap_foir"],
    },
    "vehicle": {
        "easy": ["good", "bad"],
        "medium": ["good", "bad", "borderline"],
        "hard": ["good", "bad", "borderline", "trap_ltv", "trap_credit_ltv"],
        "all": ["good", "bad", "borderline", "trap_ltv", "trap_credit_ltv"],
    },
    "home": {
        "easy": ["good", "bad"],
        "medium": ["good", "bad", "borderline"],
        "hard": ["good", "bad", "borderline", "trap_rera_perfect", "trap_ltv_tier", 
                 "trap_employment", "trap_foir_coapplicant", "trap_all_green_one_red"],
        "all": ["good", "bad", "borderline", "trap_rera_perfect", "trap_ltv_tier",
                "trap_employment", "trap_foir_coapplicant", "trap_all_green_one_red"],
    },
}


# =============================================================================
# Generators
# =============================================================================

def generate_personal_loan(difficulty: str = "all") -> dict:
    """
    Personal loan with straightforward approve/reject logic.
    
    Args:
        difficulty: "easy", "medium", "hard", or "all"
            - easy: only good/bad (obvious cases)
            - medium: adds borderline cases
            - hard/all: adds trap cases
    """
    profiles = DIFFICULTY_PROFILES["personal"].get(difficulty, DIFFICULTY_PROFILES["personal"]["all"])
    profile = random.choice(profiles)
    
    if profile == "good":
        credit_score = random.randint(720, 850)
        monthly_income = random.randint(40000, 150000)
        foir = round(random.uniform(0.20, 0.40), 2)
        employment_years = random.randint(2, 15)
        documents_complete = True
    elif profile == "bad":
        credit_score = random.randint(300, 650)
        monthly_income = random.randint(15000, 40000)
        foir = round(random.uniform(0.55, 0.80), 2)
        employment_years = random.randint(0, 1)
        documents_complete = random.choice([True, False])
    elif profile == "trap_credit":
        credit_score = random.randint(680, 699)
        monthly_income = random.randint(80000, 200000)
        foir = round(random.uniform(0.15, 0.30), 2)
        employment_years = random.randint(5, 15)
        documents_complete = True
    elif profile == "trap_foir":
        credit_score = random.randint(750, 850)
        monthly_income = random.randint(50000, 100000)
        foir = round(random.uniform(0.51, 0.60), 2)
        employment_years = random.randint(3, 10)
        documents_complete = True
    else:
        credit_score = random.randint(680, 720)
        monthly_income = random.randint(35000, 60000)
        foir = round(random.uniform(0.42, 0.52), 2)
        employment_years = random.randint(1, 3)
        documents_complete = random.choice([True, False])
    
    loan_amount = monthly_income * random.randint(12, 36)
    
    return {
        "loan_type": "personal",
        "credit_score": credit_score,
        "monthly_income": monthly_income,
        "foir": foir,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "documents_complete": documents_complete,
    }


def generate_vehicle_loan(difficulty: str = "all") -> dict:
    """
    Vehicle loan adds LTV ratio complexity.
    
    Args:
        difficulty: "easy", "medium", "hard", or "all"
    """
    profiles = DIFFICULTY_PROFILES["vehicle"].get(difficulty, DIFFICULTY_PROFILES["vehicle"]["all"])
    profile = random.choice(profiles)
    
    if profile == "good":
        credit_score = random.randint(720, 850)
        monthly_income = random.randint(40000, 150000)
        foir = round(random.uniform(0.20, 0.40), 2)
        employment_years = random.randint(2, 15)
        vehicle_value = random.randint(500000, 2000000)
        down_payment = vehicle_value * random.uniform(0.20, 0.35)
        documents_complete = True
    elif profile == "bad":
        credit_score = random.randint(300, 650)
        monthly_income = random.randint(20000, 40000)
        foir = round(random.uniform(0.55, 0.80), 2)
        employment_years = random.randint(0, 1)
        vehicle_value = random.randint(300000, 1000000)
        down_payment = vehicle_value * random.uniform(0.05, 0.12)
        documents_complete = random.choice([True, False])
    elif profile == "trap_ltv":
        credit_score = random.randint(750, 850)
        monthly_income = random.randint(60000, 150000)
        foir = round(random.uniform(0.20, 0.35), 2)
        employment_years = random.randint(3, 10)
        vehicle_value = random.randint(800000, 1500000)
        down_payment = vehicle_value * random.uniform(0.10, 0.14)
        documents_complete = True
    elif profile == "trap_credit_ltv":
        credit_score = random.randint(680, 699)
        monthly_income = random.randint(80000, 200000)
        foir = round(random.uniform(0.20, 0.30), 2)
        employment_years = random.randint(5, 12)
        vehicle_value = random.randint(600000, 1200000)
        down_payment = vehicle_value * random.uniform(0.20, 0.30)
        documents_complete = True
    else:
        credit_score = random.randint(690, 720)
        monthly_income = random.randint(40000, 80000)
        foir = round(random.uniform(0.40, 0.52), 2)
        employment_years = random.randint(1, 3)
        vehicle_value = random.randint(400000, 900000)
        down_payment = vehicle_value * random.uniform(0.12, 0.18)
        documents_complete = random.choice([True, False])
    
    loan_amount = vehicle_value - down_payment
    ltv_ratio = round(loan_amount / vehicle_value, 2)
    
    return {
        "loan_type": "vehicle",
        "vehicle_type": random.choice(["sedan", "SUV", "hatchback", "two-wheeler"]),
        "credit_score": credit_score,
        "monthly_income": monthly_income,
        "foir": foir,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "documents_complete": documents_complete,
        "collateral_value": vehicle_value,
        "ltv_ratio": ltv_ratio,
    }


def generate_home_loan(difficulty: str = "all") -> dict:
    """
    Home loan with tiered LTV, RERA compliance, and trap profiles.
    
    Args:
        difficulty: "easy", "medium", "hard", or "all"
    """
    profiles = DIFFICULTY_PROFILES["home"].get(difficulty, DIFFICULTY_PROFILES["home"]["all"])
    profile = random.choice(profiles)
    
    if profile == "good":
        credit_score = random.randint(720, 850)
        monthly_income = random.randint(60000, 200000)
        foir = round(random.uniform(0.20, 0.40), 2)
        employment_years = random.randint(3, 15)
        property_value = random.randint(3000000, 10000000)
        down_payment = property_value * random.uniform(0.25, 0.35)
        rera_registered = True
        has_co_applicant = random.choice([True, False])
        documents_complete = True
    elif profile == "bad":
        credit_score = random.randint(300, 660)
        monthly_income = random.randint(20000, 50000)
        foir = round(random.uniform(0.52, 0.80), 2)
        employment_years = random.randint(0, 2)
        property_value = random.randint(2000000, 8000000)
        down_payment = property_value * random.uniform(0.05, 0.15)
        rera_registered = random.choice([True, False])
        has_co_applicant = False
        documents_complete = random.choice([True, False])
    elif profile == "trap_rera_perfect":
        credit_score = random.randint(780, 850)
        monthly_income = random.randint(100000, 250000)
        foir = round(random.uniform(0.15, 0.30), 2)
        employment_years = random.randint(5, 15)
        property_value = random.randint(4000000, 8000000)
        down_payment = property_value * random.uniform(0.30, 0.40)
        rera_registered = False
        has_co_applicant = True
        documents_complete = True
    elif profile == "trap_ltv_tier":
        credit_score = random.randint(740, 830)
        monthly_income = random.randint(120000, 300000)
        foir = round(random.uniform(0.20, 0.35), 2)
        employment_years = random.randint(4, 12)
        property_value = random.randint(10000000, 15000000)
        down_payment = property_value * random.uniform(0.20, 0.24)
        rera_registered = True
        has_co_applicant = random.choice([True, False])
        documents_complete = True
    elif profile == "trap_employment":
        credit_score = random.randint(750, 830)
        monthly_income = random.randint(80000, 180000)
        foir = round(random.uniform(0.20, 0.35), 2)
        employment_years = 1
        property_value = random.randint(3000000, 7000000)
        down_payment = property_value * random.uniform(0.25, 0.35)
        rera_registered = True
        has_co_applicant = random.choice([True, False])
        documents_complete = True
    elif profile == "trap_foir_coapplicant":
        credit_score = random.randint(720, 790)
        monthly_income = random.randint(50000, 100000)
        foir = round(random.uniform(0.51, 0.58), 2)
        employment_years = random.randint(3, 8)
        property_value = random.randint(3000000, 6000000)
        down_payment = property_value * random.uniform(0.25, 0.35)
        rera_registered = True
        has_co_applicant = True
        documents_complete = True
    elif profile == "trap_all_green_one_red":
        credit_score = random.randint(750, 850)
        monthly_income = random.randint(80000, 200000)
        foir = round(random.uniform(0.20, 0.35), 2)
        employment_years = random.randint(3, 10)
        property_value = random.randint(4000000, 8000000)
        down_payment = property_value * random.uniform(0.25, 0.35)
        rera_registered = True
        has_co_applicant = random.choice([True, False])
        documents_complete = True
        
        trap = random.choice(["credit", "foir", "rera", "employment"])
        if trap == "credit":
            credit_score = random.randint(650, 699)
        elif trap == "foir":
            foir = round(random.uniform(0.52, 0.62), 2)
        elif trap == "rera":
            rera_registered = False
        elif trap == "employment":
            employment_years = random.randint(0, 1)
    else:
        credit_score = random.randint(680, 730)
        monthly_income = random.randint(50000, 100000)
        foir = round(random.uniform(0.44, 0.54), 2)
        employment_years = random.randint(1, 4)
        property_value = random.randint(3000000, 7000000)
        down_payment = property_value * random.uniform(0.18, 0.22)
        rera_registered = random.choice([True, False])
        has_co_applicant = random.choice([True, False])
        documents_complete = random.choice([True, False])
    
    loan_amount = property_value - down_payment
    ltv_ratio = round(loan_amount / property_value, 2)
    
    return {
        "loan_type": "home",
        "property_type": random.choice([
            "2BHK apartment", "3BHK apartment", "independent villa",
            "under-construction flat", "ready-to-move flat", "builder floor",
        ]),
        "credit_score": credit_score,
        "monthly_income": monthly_income,
        "foir": foir,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "documents_complete": documents_complete,
        "collateral_value": property_value,
        "ltv_ratio": ltv_ratio,
        "rera_registered": rera_registered,
        "has_co_applicant": has_co_applicant,
    }


def generate_applicant(task_id: int, difficulty: str = "all") -> dict:
    """
    Generate an applicant based on task ID and difficulty.
    
    Args:
        task_id: 1 (personal), 2 (vehicle), 3 (home)
        difficulty: "easy", "medium", "hard", or "all"
            - easy: obvious good/bad cases only
            - medium: adds borderline cases
            - hard/all: adds trap cases
    """
    generators = {
        1: generate_personal_loan,
        2: generate_vehicle_loan,
        3: generate_home_loan,
    }
    return generators[task_id](difficulty=difficulty)


# =============================================================================
# Ground Truth
# =============================================================================

def ground_truth_personal(a: dict) -> str:
    """Ground truth for personal loans."""
    if not a["documents_complete"]:
        return "request_docs"
    if a["credit_score"] < 700:
        return "reject"
    if a["foir"] > 0.50:
        return "reject"
    if a["employment_years"] < 1:
        return "reject"
    return "approve"


def ground_truth_vehicle(a: dict) -> str:
    """Ground truth for vehicle loans."""
    if not a["documents_complete"]:
        return "request_docs"
    if a["credit_score"] < 700:
        return "reject"
    if a["foir"] > 0.50:
        return "reject"
    if a["employment_years"] < 1:
        return "reject"
    if a.get("ltv_ratio", 0) > 0.85:
        return "counter_offer"
    return "approve"


def ground_truth_home(a: dict) -> str:
    """Ground truth for home loans based on RBI guidelines."""
    if not a["documents_complete"]:
        return "request_docs"
    if a["rera_registered"] is False:
        return "reject"
    if a["credit_score"] < 700:
        return "reject"
    if a["foir"] > 0.50:
        return "reject"
    if a["employment_years"] < 2:
        return "reject"
    
    # RBI tiered LTV — tiers are MUTUALLY EXCLUSIVE, not cumulative.
    # See note in server/ground_truth/home_loan.py for details.
    if a.get("ltv_ratio"):
        loan_amount = a["loan_amount"]
        ltv = a["ltv_ratio"]
        if loan_amount <= 3000000:
            if ltv > 0.90:
                return "counter_offer"
        elif loan_amount <= 7500000:
            if ltv > 0.80:
                return "counter_offer"
        else:
            if ltv > 0.75:
                return "counter_offer"
    
    return "approve"


def calculate_ground_truth(applicant: dict) -> str:
    """Calculate ground truth decision for an applicant."""
    loan_type = applicant["loan_type"]
    calculators = {
        "personal": ground_truth_personal,
        "vehicle": ground_truth_vehicle,
        "home": ground_truth_home,
    }
    return calculators[loan_type](applicant)


# =============================================================================
# Rewards
# =============================================================================

def reward_personal(action: CreditAssessmentAction, applicant: dict, ground_truth: str) -> float:
    """Reward for personal loan decisions."""
    decision = action.decision.value
    
    if decision == ground_truth:
        return 10.0
    if decision == "request_docs" and not applicant["documents_complete"]:
        return 2.0
    if decision in ("approve", "reject") and ground_truth == "request_docs":
        return -8.0
    if decision == "approve" and ground_truth == "reject":
        return -15.0
    if decision == "reject" and ground_truth == "approve":
        return -5.0
    if decision == "counter_offer" and not action.counter_offer_amount:
        return -3.0
    return -2.0


def reward_vehicle(action: CreditAssessmentAction, applicant: dict, ground_truth: str) -> float:
    """Reward for vehicle loan decisions."""
    decision = action.decision.value
    
    if decision == ground_truth:
        return 10.0
    if decision == "request_docs" and not applicant["documents_complete"]:
        return 2.0
    if decision == "counter_offer" and ground_truth == "counter_offer":
        return 5.0
    if decision in ("approve", "reject") and ground_truth == "request_docs":
        return -8.0
    if decision in ("approve", "reject") and ground_truth == "counter_offer":
        return -8.0
    if decision == "approve" and ground_truth == "reject":
        return -15.0
    if decision == "reject" and ground_truth == "approve":
        return -5.0
    if decision == "counter_offer" and not action.counter_offer_amount:
        return -3.0
    return -2.0


def reward_home(action: CreditAssessmentAction, applicant: dict, ground_truth: str) -> float:
    """Reward for home loan decisions."""
    decision = action.decision.value
    
    if decision == ground_truth:
        return 10.0
    if decision == "request_docs" and not applicant["documents_complete"]:
        return 2.0
    
    # RERA violation is the worst possible outcome
    if decision == "approve" and applicant.get("rera_registered") is False:
        return -20.0
    
    if decision == "counter_offer" and ground_truth == "counter_offer":
        return 5.0
    if decision in ("approve", "reject") and ground_truth == "request_docs":
        return -8.0
    if decision in ("approve", "reject") and ground_truth == "counter_offer":
        return -8.0
    if decision == "approve" and ground_truth == "reject":
        return -15.0
    if decision == "reject" and ground_truth == "approve":
        return -5.0
    if decision == "counter_offer" and not action.counter_offer_amount:
        return -3.0
    return -2.0


def calculate_reward(action: CreditAssessmentAction, applicant: dict, ground_truth: str) -> float:
    """Calculate reward for an action."""
    loan_type = applicant["loan_type"]
    calculators = {
        "personal": reward_personal,
        "vehicle": reward_vehicle,
        "home": reward_home,
    }
    return calculators[loan_type](action, applicant, ground_truth)

# =============================================================================
# Adversarial Case Generator (Rule-Based)
# =============================================================================

ADVERSARIAL_STRATEGIES = [
    "threshold_credit",       # CIBIL exactly 699 (just below 700)
    "threshold_foir",         # FOIR exactly 51% (just above 50%)
    "perfect_but_rera",       # Everything perfect, RERA=No
    "perfect_but_ltv_tier",   # >75L loan with LTV 76% (limit is 75%)
    "coapplicant_trap",       # Has co-applicant but FOIR>50%
    "high_income_low_cibil",  # Very high income but CIBIL 695
    "employment_trap_home",   # Perfect home loan, only 1 year employment
    "vehicle_ltv_trap",       # Perfect vehicle loan, LTV 86%
    "docs_incomplete_good",   # Good profile but docs missing
    "borderline_multiple",    # Multiple metrics right at threshold
]


def generate_adversarial_case(strategy: str = None) -> dict:
    """
    Generate adversarial loan application targeting specific weaknesses.
    
    These cases are designed to trick LLMs that rely on pattern matching
    rather than precise rule-following.
    
    Args:
        strategy: Specific strategy to use, or None for random
    
    Returns:
        Adversarial loan application dict
    """
    if strategy is None:
        strategy = random.choice(ADVERSARIAL_STRATEGIES)
    
    if strategy == "threshold_credit":
        # CIBIL 699 - just 1 point below threshold
        # High income and good metrics make it tempting to approve
        return {
            "loan_type": "personal",
            "credit_score": 699,  # Just below 700!
            "monthly_income": random.randint(150000, 300000),
            "foir": round(random.uniform(0.20, 0.30), 2),
            "employment_years": random.randint(8, 15),
            "loan_amount": random.randint(500000, 1000000),
            "documents_complete": True,
        }
    
    elif strategy == "threshold_foir":
        # FOIR 51% - just 1% above threshold
        # Excellent credit score makes it tempting to approve
        return {
            "loan_type": "personal",
            "credit_score": random.randint(780, 850),
            "monthly_income": random.randint(80000, 150000),
            "foir": round(random.uniform(0.51, 0.53), 2),  # Just above 50%!
            "employment_years": random.randint(5, 12),
            "loan_amount": random.randint(400000, 800000),
            "documents_complete": True,
        }
    
    elif strategy == "perfect_but_rera":
        # Everything is perfect EXCEPT RERA registration
        # This is the classic trap - must reject despite dream financials
        return {
            "loan_type": "home",
            "property_type": random.choice(["3BHK apartment", "independent villa"]),
            "credit_score": random.randint(800, 850),
            "monthly_income": random.randint(200000, 400000),
            "foir": round(random.uniform(0.20, 0.30), 2),
            "employment_years": random.randint(8, 15),
            "loan_amount": random.randint(5000000, 8000000),
            "collateral_value": random.randint(7000000, 12000000),
            "ltv_ratio": round(random.uniform(0.65, 0.72), 2),
            "documents_complete": True,
            "rera_registered": False,  # THE TRAP!
            "has_co_applicant": True,
        }
    
    elif strategy == "perfect_but_ltv_tier":
        # Loan >75L with LTV between 75-80%
        # Looks fine if you assume 80% limit, but RBI says 75% for >75L
        property_value = random.randint(12000000, 18000000)
        ltv = round(random.uniform(0.76, 0.79), 2)  # Above 75% limit!
        loan_amount = int(property_value * ltv)
        return {
            "loan_type": "home",
            "property_type": "independent villa",
            "credit_score": random.randint(780, 840),
            "monthly_income": random.randint(250000, 500000),
            "foir": round(random.uniform(0.25, 0.35), 2),
            "employment_years": random.randint(6, 12),
            "loan_amount": loan_amount,
            "collateral_value": property_value,
            "ltv_ratio": ltv,
            "documents_complete": True,
            "rera_registered": True,
            "has_co_applicant": random.choice([True, False]),
        }
    
    elif strategy == "coapplicant_trap":
        # Has co-applicant (looks reassuring) but FOIR > 50%
        # LLMs might think co-applicant compensates for high FOIR
        return {
            "loan_type": "home",
            "property_type": "2BHK apartment",
            "credit_score": random.randint(740, 800),
            "monthly_income": random.randint(70000, 120000),
            "foir": round(random.uniform(0.52, 0.58), 2),  # Above 50%!
            "employment_years": random.randint(4, 8),
            "loan_amount": random.randint(4000000, 6000000),
            "collateral_value": random.randint(5500000, 8000000),
            "ltv_ratio": round(random.uniform(0.70, 0.78), 2),
            "documents_complete": True,
            "rera_registered": True,
            "has_co_applicant": True,  # Misleading safety signal!
        }
    
    elif strategy == "high_income_low_cibil":
        # Extremely high income but CIBIL just below threshold
        # Tests if model over-weights income vs mandatory threshold
        return {
            "loan_type": "personal",
            "credit_score": random.randint(680, 699),  # Below 700
            "monthly_income": random.randint(400000, 800000),  # Very high!
            "foir": round(random.uniform(0.10, 0.20), 2),  # Very low!
            "employment_years": random.randint(10, 20),
            "loan_amount": random.randint(1000000, 2000000),
            "documents_complete": True,
        }
    
    elif strategy == "employment_trap_home":
        # Perfect home loan profile but only 1 year employment
        # Home loans require 2+ years (unlike personal/vehicle)
        return {
            "loan_type": "home",
            "property_type": "3BHK apartment",
            "credit_score": random.randint(780, 840),
            "monthly_income": random.randint(150000, 250000),
            "foir": round(random.uniform(0.25, 0.35), 2),
            "employment_years": 1,  # Home loan needs 2+!
            "loan_amount": random.randint(5000000, 7000000),
            "collateral_value": random.randint(7000000, 10000000),
            "ltv_ratio": round(random.uniform(0.68, 0.75), 2),
            "documents_complete": True,
            "rera_registered": True,
            "has_co_applicant": True,
        }
    
    elif strategy == "vehicle_ltv_trap":
        # Perfect vehicle loan but LTV just above 85%
        vehicle_value = random.randint(800000, 1500000)
        ltv = round(random.uniform(0.86, 0.90), 2)  # Above 85%!
        loan_amount = int(vehicle_value * ltv)
        return {
            "loan_type": "vehicle",
            "vehicle_type": random.choice(["sedan", "SUV"]),
            "credit_score": random.randint(760, 830),
            "monthly_income": random.randint(80000, 150000),
            "foir": round(random.uniform(0.25, 0.35), 2),
            "employment_years": random.randint(4, 10),
            "loan_amount": loan_amount,
            "collateral_value": vehicle_value,
            "ltv_ratio": ltv,
            "documents_complete": True,
        }
    
    elif strategy == "docs_incomplete_good":
        # Excellent profile but documents incomplete
        # Tests if model checks docs FIRST before other criteria
        loan_type = random.choice(["personal", "vehicle", "home"])
        base = {
            "loan_type": loan_type,
            "credit_score": random.randint(780, 850),
            "monthly_income": random.randint(150000, 300000),
            "foir": round(random.uniform(0.20, 0.30), 2),
            "employment_years": random.randint(5, 12),
            "loan_amount": random.randint(500000, 2000000),
            "documents_complete": False,  # Must request docs!
        }
        if loan_type in ("vehicle", "home"):
            base["collateral_value"] = base["loan_amount"] * 1.4
            base["ltv_ratio"] = round(base["loan_amount"] / base["collateral_value"], 2)
        if loan_type == "home":
            base["rera_registered"] = True
            base["has_co_applicant"] = True
            base["property_type"] = "3BHK apartment"
        if loan_type == "vehicle":
            base["vehicle_type"] = "sedan"
        return base
    
    elif strategy == "borderline_multiple":
        # Multiple metrics right at the borderline
        # CIBIL exactly 700, FOIR exactly 50%, employment exactly 1 year
        return {
            "loan_type": "personal",
            "credit_score": 700,  # Exactly at threshold
            "monthly_income": random.randint(60000, 100000),
            "foir": 0.50,  # Exactly at threshold
            "employment_years": 1,  # Exactly at threshold
            "loan_amount": random.randint(300000, 600000),
            "documents_complete": True,
        }
    
    # Fallback to hard case from regular generator
    return generate_applicant(random.choice([1, 2, 3]), difficulty="hard")


class AdversarialTracker:
    """
    Tracks agent failures and generates targeted adversarial cases.
    
    This enables self-improvement by focusing training on weak areas.
    """
    
    def __init__(self):
        self.failure_counts = {strategy: 0 for strategy in ADVERSARIAL_STRATEGIES}
        self.success_counts = {strategy: 0 for strategy in ADVERSARIAL_STRATEGIES}
        self.recent_failures = []
    
    def record_result(self, strategy: str, was_correct: bool):
        """Record whether agent handled this strategy correctly."""
        if was_correct:
            self.success_counts[strategy] = self.success_counts.get(strategy, 0) + 1
        else:
            self.failure_counts[strategy] = self.failure_counts.get(strategy, 0) + 1
            self.recent_failures.append(strategy)
            if len(self.recent_failures) > 20:
                self.recent_failures.pop(0)
    
    def get_weakness(self) -> str:
        """Identify the strategy the agent fails at most."""
        if not any(self.failure_counts.values()):
            return random.choice(ADVERSARIAL_STRATEGIES)
        return max(self.failure_counts, key=self.failure_counts.get)
    
    def get_weakness_rate(self, strategy: str) -> float:
        """Get failure rate for a specific strategy."""
        total = self.success_counts.get(strategy, 0) + self.failure_counts.get(strategy, 0)
        if total == 0:
            return 0.5  # Unknown
        return self.failure_counts.get(strategy, 0) / total
    
    def generate_targeted_batch(self, batch_size: int, target_weakness: bool = True) -> list:
        """
        Generate batch of adversarial cases.
        
        Args:
            batch_size: Number of cases to generate
            target_weakness: If True, focus on agent's weak areas
        
        Returns:
            List of adversarial applicant dicts
        """
        cases = []
        weakness = self.get_weakness() if target_weakness else None
        
        for i in range(batch_size):
            if target_weakness and random.random() < 0.7:
                # 70% target weakness, 30% random adversarial
                strategy = weakness
            else:
                strategy = random.choice(ADVERSARIAL_STRATEGIES)
            
            cases.append({
                "applicant": generate_adversarial_case(strategy),
                "strategy": strategy,
            })
        
        return cases
    
    def get_summary(self) -> dict:
        """Get summary of agent performance on adversarial cases."""
        summary = {}
        for strategy in ADVERSARIAL_STRATEGIES:
            successes = self.success_counts.get(strategy, 0)
            failures = self.failure_counts.get(strategy, 0)
            total = successes + failures
            if total > 0:
                summary[strategy] = {
                    "accuracy": successes / total,
                    "total": total,
                    "failures": failures,
                }
        return summary
