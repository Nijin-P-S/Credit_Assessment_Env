import random


def generate_vehicle_loan() -> dict:
    """Medium: Vehicle loan — adds LTV complexity and traps."""

    profile = random.choice([
        "good", "bad", "borderline",
        "trap_ltv", "trap_credit_rich", "trap_foir_hidden",
    ])

    if profile == "good":
        credit_score = random.randint(700, 820)
        monthly_income = random.randint(35000, 120000)
        foir = round(random.uniform(0.20, 0.42), 2)
        employment_years = random.randint(2, 8)
        vehicle_value = random.randint(500000, 1500000)
        down_payment = vehicle_value * random.uniform(0.20, 0.30)
        documents_complete = True

    elif profile == "bad":
        credit_score = random.randint(300, 650)
        monthly_income = random.randint(15000, 35000)
        foir = round(random.uniform(0.52, 0.75), 2)
        employment_years = random.randint(0, 1)
        vehicle_value = random.randint(800000, 2000000)
        down_payment = vehicle_value * random.uniform(0.05, 0.10)
        documents_complete = random.choice([True, False])

    elif profile == "trap_ltv":
        # Everything looks great BUT down payment is too low — LTV exceeds 85%
        credit_score = random.randint(740, 820)
        monthly_income = random.randint(60000, 150000)
        foir = round(random.uniform(0.20, 0.35), 2)
        employment_years = random.randint(3, 10)
        vehicle_value = random.randint(800000, 1500000)
        down_payment = vehicle_value * random.uniform(0.05, 0.12)
        documents_complete = True

    elif profile == "trap_credit_rich":
        # Very high income but credit score just under 700 — must reject
        credit_score = random.randint(660, 699)
        monthly_income = random.randint(100000, 300000)
        foir = round(random.uniform(0.10, 0.25), 2)
        employment_years = random.randint(5, 15)
        vehicle_value = random.randint(500000, 1000000)
        down_payment = vehicle_value * random.uniform(0.25, 0.40)
        documents_complete = True

    elif profile == "trap_foir_hidden":
        # Great credit, good income, but FOIR just over the edge
        credit_score = random.randint(720, 800)
        monthly_income = random.randint(40000, 80000)
        foir = round(random.uniform(0.51, 0.58), 2)
        employment_years = random.randint(3, 8)
        vehicle_value = random.randint(600000, 1200000)
        down_payment = vehicle_value * random.uniform(0.20, 0.30)
        documents_complete = True

    else:
        credit_score = random.randint(660, 710)
        monthly_income = random.randint(30000, 60000)
        foir = round(random.uniform(0.43, 0.52), 2)
        employment_years = random.randint(1, 3)
        vehicle_value = random.randint(600000, 1200000)
        down_payment = vehicle_value * random.uniform(0.12, 0.18)
        documents_complete = random.choice([True, False])

    loan_amount = vehicle_value - down_payment
    ltv_ratio = round(loan_amount / vehicle_value, 2)

    return {
        "loan_type": "vehicle",
        "vehicle_type": random.choice([
            "new sedan", "new SUV", "new hatchback",
            "used sedan (2 years old)", "used SUV (3 years old)",
        ]),
        "credit_score": credit_score,
        "monthly_income": monthly_income,
        "foir": foir,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "documents_complete": documents_complete,
        "collateral_value": vehicle_value,
        "ltv_ratio": ltv_ratio,
        "rera_registered": None,
        "has_co_applicant": None,
    }
