import random


def generate_personal_loan() -> dict:
    """Easy: Personal loan — mostly clear signals, occasional borderline."""

    profile = random.choice(["good", "bad", "borderline", "trap_credit", "trap_employment"])

    if profile == "good":
        credit_score = random.randint(720, 850)
        monthly_income = random.randint(40000, 150000)
        foir = round(random.uniform(0.20, 0.40), 2)
        employment_years = random.randint(2, 10)
        documents_complete = True

    elif profile == "bad":
        credit_score = random.randint(300, 640)
        monthly_income = random.randint(10000, 30000)
        foir = round(random.uniform(0.55, 0.80), 2)
        employment_years = random.randint(0, 1)
        documents_complete = random.choice([True, False])

    elif profile == "trap_credit":
        # Looks great on paper — high income, low FOIR — but credit is just below cutoff
        credit_score = random.randint(660, 699)
        monthly_income = random.randint(80000, 200000)
        foir = round(random.uniform(0.15, 0.30), 2)
        employment_years = random.randint(5, 15)
        documents_complete = True

    elif profile == "trap_employment":
        # Excellent credit and income, but brand new to the job
        credit_score = random.randint(750, 850)
        monthly_income = random.randint(60000, 150000)
        foir = round(random.uniform(0.20, 0.35), 2)
        employment_years = 0
        documents_complete = True

    else:
        credit_score = random.randint(650, 720)
        monthly_income = random.randint(25000, 50000)
        foir = round(random.uniform(0.42, 0.52), 2)
        employment_years = random.randint(1, 3)
        documents_complete = random.choice([True, False])

    loan_amount = monthly_income * random.randint(8, 15)

    return {
        "loan_type": "personal",
        "purpose": random.choice([
            "home renovation", "medical expenses", "wedding",
            "debt consolidation", "travel", "education fees",
        ]),
        "credit_score": credit_score,
        "monthly_income": monthly_income,
        "foir": foir,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "documents_complete": documents_complete,
        "collateral_value": None,
        "ltv_ratio": None,
        "rera_registered": None,
        "has_co_applicant": None,
    }
