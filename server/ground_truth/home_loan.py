def ground_truth_home(a: dict) -> str:
    """Ground truth for home loans based on RBI Master Circular on Housing Finance."""
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

    # RBI tiered LTV: ≤30L → 90%, 30-75L → 80%, >75L → 75%
    # Tiers are MUTUALLY EXCLUSIVE — a 25L loan with 85% LTV is tier 1
    # (cap 90%) and must be approved, not pushed into tier 2's 80% cap.
    if a["ltv_ratio"]:
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
