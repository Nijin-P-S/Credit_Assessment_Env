def ground_truth_vehicle(a: dict) -> str:
    """Ground truth for vehicle loans based on Indian bank underwriting norms."""
    if not a["documents_complete"]:
        return "request_docs"
    if a["credit_score"] < 700:
        return "reject"
    if a["foir"] > 0.50:
        return "reject"
    if a["employment_years"] < 1:
        return "reject"
    if a["ltv_ratio"] and a["ltv_ratio"] > 0.85:
        return "counter_offer"
    return "approve"
