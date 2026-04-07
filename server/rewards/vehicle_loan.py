try:
    from ...models import CreditAssessmentAction
except ImportError:
    from models import CreditAssessmentAction


def reward_vehicle(action: CreditAssessmentAction, applicant: dict, ground_truth: str) -> float:
    """Reward for vehicle loan decisions."""
    decision = action.decision.value

    if decision == ground_truth:
        return 10.0

    if decision == "request_docs" and not applicant["documents_complete"]:
        return 2.0

    if decision == "counter_offer" and applicant["ltv_ratio"] and applicant["ltv_ratio"] > 0.85:
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
