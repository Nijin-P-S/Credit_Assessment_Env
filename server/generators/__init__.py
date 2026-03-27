from .personal_loan import generate_personal_loan
from .vehicle_loan import generate_vehicle_loan
from .home_loan import generate_home_loan


def generate_applicant(task_id: int) -> dict:
    """Route applicant generation by task/loan type."""
    generators = {
        1: generate_personal_loan,
        2: generate_vehicle_loan,
        3: generate_home_loan,
    }
    generator = generators.get(task_id)
    if not generator:
        raise ValueError(f"Unknown task_id: {task_id}")
    return generator()
