try:
    from ...models import CreditAssessmentAction
except ImportError:
    from models import CreditAssessmentAction


def reward_home(action: CreditAssessmentAction, applicant: dict, ground_truth: str) -> float:
    """Reward for home loan decisions."""
    decision = action.decision.value

    if decision == ground_truth:
        return 0.9

    if decision == "request_docs" and not applicant["documents_complete"]:
        return 0.5

    if decision == "approve" and applicant["rera_registered"] is False:
        return 0.1

    if decision == "approve" and ground_truth == "reject":
        return 0.1

    if decision == "reject" and ground_truth == "approve":
        return 0.2

    if decision == "counter_offer" and not action.counter_offer_amount:
        return 0.15

    return 0.2
