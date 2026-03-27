def ground_truth_personal(a: dict) -> str:
    """Ground truth for personal loans based on Indian bank underwriting norms."""
    if not a["documents_complete"]:
        return "request_docs"
    if a["credit_score"] < 700:
        return "reject"
    if a["foir"] > 0.50:
        return "reject"
    if a["employment_years"] < 1:
        return "reject"
    return "approve"
