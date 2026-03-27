from .personal_loan import ground_truth_personal
from .vehicle_loan import ground_truth_vehicle
from .home_loan import ground_truth_home


def calculate_ground_truth(applicant: dict) -> str:
    """Route ground truth calculation by loan type."""
    loan_type = applicant["loan_type"]
    calculators = {
        "personal": ground_truth_personal,
        "vehicle": ground_truth_vehicle,
        "home": ground_truth_home,
    }
    calculator = calculators.get(loan_type)
    if not calculator:
        raise ValueError(f"Unknown loan_type: {loan_type}")
    return calculator(applicant)
