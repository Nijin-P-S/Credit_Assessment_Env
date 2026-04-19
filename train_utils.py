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
# Generators
# =============================================================================

def generate_personal_loan() -> dict:
    """Easy: Personal loan with straightforward approve/reject logic."""
    profile = random.choice(["good", "bad", "borderline", "trap_credit", "trap_foir"])
    
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


def generate_vehicle_loan() -> dict:
    """Medium: Vehicle loan adds LTV ratio complexity."""
    profile = random.choice(["good", "bad", "borderline", "trap_ltv", "trap_credit_ltv"])
    
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


def generate_home_loan() -> dict:
    """Hard: Home loan with tiered LTV, RERA compliance, and trap profiles."""
    profile = random.choice([
        "good", "bad", "borderline",
        "trap_rera_perfect", "trap_ltv_tier", "trap_employment",
        "trap_foir_coapplicant", "trap_all_green_one_red",
    ])
    
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


def generate_applicant(task_id: int) -> dict:
    """Generate an applicant based on task ID."""
    generators = {
        1: generate_personal_loan,
        2: generate_vehicle_loan,
        3: generate_home_loan,
    }
    return generators[task_id]()


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
    
    # RBI tiered LTV
    if a.get("ltv_ratio"):
        loan_amount = a["loan_amount"]
        if loan_amount <= 3000000 and a["ltv_ratio"] > 0.90:
            return "counter_offer"
        elif loan_amount <= 7500000 and a["ltv_ratio"] > 0.80:
            return "counter_offer"
        elif loan_amount > 7500000 and a["ltv_ratio"] > 0.75:
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
