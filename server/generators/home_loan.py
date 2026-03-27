import random


def generate_home_loan() -> dict:
    """Hard: Home loan — conflicting signals, tiered LTV, RERA traps."""

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
        # Every financial metric is excellent — BUT property is not RERA registered
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
        # Loan > 75L, everything great, BUT LTV is between 75-80% — exceeds
        # the 75% limit for >75L loans. Agent must know the RBI tiered rule.
        credit_score = random.randint(740, 830)
        monthly_income = random.randint(120000, 300000)
        foir = round(random.uniform(0.20, 0.35), 2)
        employment_years = random.randint(4, 12)
        property_value = random.randint(10000000, 15000000)
        # Down payment that puts LTV between 76-80% — looks OK if you assume 80% limit
        down_payment = property_value * random.uniform(0.20, 0.24)
        rera_registered = True
        has_co_applicant = random.choice([True, False])
        documents_complete = True

    elif profile == "trap_employment":
        # Excellent financials but only 1 year employment (home loan needs 2+)
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
        # Has co-applicant (looks reassuring) BUT FOIR is just over 50%
        # LLM might think co-applicant compensates for high FOIR — it doesn't
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
        # Randomly pick ONE disqualifying factor, everything else is green
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
