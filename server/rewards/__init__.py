from .personal_loan import reward_personal
from .vehicle_loan import reward_vehicle
from .home_loan import reward_home

try:
    from ...models import CreditAssessmentAction
except ImportError:
    from models import CreditAssessmentAction


def calculate_reward(action: CreditAssessmentAction, applicant: dict, ground_truth: str) -> float:
    """Route reward calculation by loan type."""
    loan_type = applicant["loan_type"]
    calculators = {
        "personal": reward_personal,
        "vehicle": reward_vehicle,
        "home": reward_home,
    }
    calculator = calculators.get(loan_type)
    if not calculator:
        raise ValueError(f"Unknown loan_type: {loan_type}")
    return calculator(action, applicant, ground_truth)
