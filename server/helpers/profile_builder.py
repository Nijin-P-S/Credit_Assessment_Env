import random

_PERSONAL_PURPOSES = [
    "home renovation", "medical expenses", "wedding",
    "debt consolidation", "travel", "education fees",
]

_VEHICLE_TYPES = [
    "new sedan", "new SUV", "new hatchback",
    "used sedan (2 years old)", "used SUV (3 years old)",
]

_PROPERTY_TYPES = [
    "2BHK apartment", "3BHK apartment", "independent villa",
    "under-construction flat", "ready-to-move flat", "builder floor",
]

_CONSTRUCTION_STAGES = [
    "Foundation complete", "Superstructure in progress",
    "Ready for possession", "Finishing stage",
]


def build_profile_text(a: dict) -> str:
    """Build a realistic, narrative-style loan application for LLM reasoning."""

    loan_type = a["loan_type"]

    if loan_type == "personal":
        return _build_personal_profile(a)
    elif loan_type == "vehicle":
        return _build_vehicle_profile(a)
    elif loan_type == "home":
        return _build_home_profile(a)
    return _build_generic_profile(a)


def _build_personal_profile(a: dict) -> str:
    purpose = random.choice(_PERSONAL_PURPOSES)

    text = f"""LOAN APPLICATION — PERSONAL LOAN
{'=' * 55}
Purpose: {purpose.title()}

APPLICANT DETAILS
  Monthly Income:     ₹{a['monthly_income']:,.0f}
  Employment:         {a['employment_years']} year(s) at current employer
  CIBIL Score:        {a['credit_score']}

FINANCIAL OBLIGATIONS
  Current FOIR:       {a['foir'] * 100:.1f}%
  Requested Amount:   ₹{a['loan_amount']:,.0f}

DOCUMENTATION
  Status: {'All documents submitted and verified' if a['documents_complete'] else 'Pending — some documents not yet submitted'}

{'=' * 55}
Please assess this application.
ACTIONS: approve | reject | request_docs | counter_offer"""

    return text.strip()


def _build_vehicle_profile(a: dict) -> str:
    vehicle = random.choice(_VEHICLE_TYPES)

    text = f"""LOAN APPLICATION — VEHICLE LOAN
{'=' * 55}
Vehicle: {vehicle.title()}

APPLICANT DETAILS
  Monthly Income:     ₹{a['monthly_income']:,.0f}
  Employment:         {a['employment_years']} year(s) at current employer
  CIBIL Score:        {a['credit_score']}

FINANCIAL OBLIGATIONS
  Current FOIR:       {a['foir'] * 100:.1f}%

LOAN DETAILS
  Vehicle Value:      ₹{a['collateral_value']:,.0f}
  Loan Requested:     ₹{a['loan_amount']:,.0f}
  LTV Ratio:          {a['ltv_ratio'] * 100:.1f}%

DOCUMENTATION
  Status: {'All documents submitted and verified' if a['documents_complete'] else 'Pending — some documents not yet submitted'}

{'=' * 55}
Please assess this application.
ACTIONS: approve | reject | request_docs | counter_offer"""

    return text.strip()


def _build_home_profile(a: dict) -> str:
    prop_type = random.choice(_PROPERTY_TYPES)
    construction = random.choice(_CONSTRUCTION_STAGES)

    co_app_text = ""
    if a.get("has_co_applicant"):
        co_app_text = "\n  Co-applicant:      Yes (spouse)"
    else:
        co_app_text = "\n  Co-applicant:      No"

    text = f"""LOAN APPLICATION — HOME LOAN
{'=' * 55}
Property: {prop_type.title()}
Construction Stage: {construction}

APPLICANT DETAILS
  Monthly Income:     ₹{a['monthly_income']:,.0f}
  Employment:         {a['employment_years']} year(s) at current employer
  CIBIL Score:        {a['credit_score']}{co_app_text}

FINANCIAL OBLIGATIONS
  Current FOIR:       {a['foir'] * 100:.1f}%

PROPERTY & LOAN DETAILS
  Property Value:     ₹{a['collateral_value']:,.0f}
  Loan Requested:     ₹{a['loan_amount']:,.0f}
  RERA Registered:    {'Yes' if a['rera_registered'] else 'No'}

DOCUMENTATION
  Status: {'All documents submitted and verified' if a['documents_complete'] else 'Pending — some documents not yet submitted'}

{'=' * 55}
Please assess this application.
Note: Verify all applicable regulatory norms before approval.
ACTIONS: approve | reject | request_docs | counter_offer"""

    return text.strip()


def _build_generic_profile(a: dict) -> str:
    text = f"""LOAN APPLICATION
{'=' * 55}
Loan Type: {a['loan_type'].upper()}
Credit Score: {a['credit_score']}
Monthly Income: ₹{a['monthly_income']:,.0f}
FOIR: {a['foir'] * 100:.1f}%
Employment: {a['employment_years']} years
Loan Amount: ₹{a['loan_amount']:,.0f}
Documents: {'Complete' if a['documents_complete'] else 'INCOMPLETE'}

{'=' * 55}
ACTIONS: approve | reject | request_docs | counter_offer"""

    return text.strip()
