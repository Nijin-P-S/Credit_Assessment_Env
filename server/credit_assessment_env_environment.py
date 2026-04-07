"""
Credit Assessment Environment.

Simulates Indian loan underwriting across three difficulty levels:
  Task 1 (Easy):   Personal Loan
  Task 2 (Medium): Vehicle Loan
  Task 3 (Hard):   Home Loan
"""

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import CreditAssessmentAction, CreditAssessmentObservation, LoanDecision
except ImportError:
    from models import CreditAssessmentAction, CreditAssessmentObservation, LoanDecision

from .generators import generate_applicant
from .ground_truth import calculate_ground_truth
from .rewards import calculate_reward
from .helpers import build_profile_text


class CreditAssessmentEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASKS = {
        1: {"name": "Personal Loan", "loan_type": "personal", "difficulty": "easy"},
        2: {"name": "Vehicle Loan", "loan_type": "vehicle", "difficulty": "medium"},
        3: {"name": "Home Loan", "loan_type": "home", "difficulty": "hard"},
    }

    MAX_STEPS_PER_EPISODE = 3

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._current_applicant = None
        self._current_task_id = 1
        self._ground_truth = None
        self._last_reward = 0.0
        self._total_reward = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CreditAssessmentObservation:
        if seed is not None:
            random.seed(seed)

        ep_id = episode_id or str(uuid4())
        self._state = State(episode_id=ep_id, step_count=0)
        self._reset_count += 1
        self._last_reward = 0.0
        self._total_reward = 0.0

        task_id = kwargs.get("task_id")
        if task_id:
            self._current_task_id = task_id
        else:
            self._current_task_id = random.choice([1, 2, 3])

        self._current_applicant = generate_applicant(self._current_task_id)
        self._ground_truth = calculate_ground_truth(self._current_applicant)

        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: CreditAssessmentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CreditAssessmentObservation:
        self._state.step_count += 1

        reward = calculate_reward(action, self._current_applicant, self._ground_truth)
        self._last_reward = reward
        self._total_reward += reward

        done = self._is_done(action)

        if not done:
            self._evolve_applicant(action)
            self._ground_truth = calculate_ground_truth(self._current_applicant)

        return self._build_observation(reward=reward, done=done)

    def _evolve_applicant(self, action: CreditAssessmentAction) -> None:
        """Simulate the applicant responding to an intermediate action.

        request_docs  -> applicant submits missing documents
        counter_offer -> applicant accepts reduced loan amount
        """
        if action.decision == LoanDecision.REQUEST_DOCS:
            self._current_applicant["documents_complete"] = True

        elif action.decision == LoanDecision.COUNTER_OFFER and action.counter_offer_amount:
            old_amount = self._current_applicant["loan_amount"]
            new_amount = min(action.counter_offer_amount, old_amount)
            self._current_applicant["loan_amount"] = new_amount
            if self._current_applicant.get("collateral_value"):
                collateral = self._current_applicant["collateral_value"]
                self._current_applicant["ltv_ratio"] = round(new_amount / collateral, 2)

    def grade(self) -> float:
        """Normalised score (0.0–1.0) based on cumulative episode reward."""
        r = self._total_reward
        if r >= 10.0:
            return 0.99
        elif r >= 7.0:
            return 0.8
        elif r >= 3.0:
            return 0.5
        elif r >= 0:
            return 0.2
        return 0.01

    def _is_done(self, action: CreditAssessmentAction) -> bool:
        if self._state.step_count >= self.MAX_STEPS_PER_EPISODE:
            return True
        if action.decision in [LoanDecision.APPROVE, LoanDecision.REJECT]:
            return True
        return False

    def _normalize_reward(self, reward: float) -> float:
        """Normalize a raw reward to strictly (0.0, 1.0) using grade thresholds."""
        if reward >= 10.0:
            return 0.99
        elif reward >= 7.0:
            return 0.8
        elif reward >= 3.0:
            return 0.5
        elif reward >= 0:
            return 0.2
        return 0.01

    def _build_observation(self, reward: float, done: bool) -> CreditAssessmentObservation:
        a = self._current_applicant

        return CreditAssessmentObservation(
            applicant_profile=build_profile_text(a),
            credit_score=a["credit_score"],
            monthly_income=a["monthly_income"],
            foir=a["foir"],
            employment_years=a["employment_years"],
            loan_type=a["loan_type"],
            loan_amount=a["loan_amount"],
            documents_complete=a["documents_complete"],
            collateral_value=a.get("collateral_value"),
            ltv_ratio=a.get("ltv_ratio"),
            rera_registered=a.get("rera_registered"),
            has_co_applicant=a.get("has_co_applicant"),
            task_id=self._current_task_id,
            reward=self._normalize_reward(reward),
            done=done,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        readme = ""
        try:
            import os
            from pathlib import Path
            candidates = [
                os.environ.get("ENV_README_PATH"),
                Path(__file__).resolve().parent.parent / "README.md",
                Path("/app/env/README.md"),
            ]
            for candidate in candidates:
                if candidate and Path(candidate).exists():
                    readme = Path(candidate).read_text(encoding="utf-8")
                    break
        except Exception:
            pass

        return EnvironmentMetadata(
            name="Credit Assessment Environment",
            description=(
                "Simulates Indian loan underwriting across personal, vehicle, and home loans. "
                "Agents must apply RBI-compliant rules including CIBIL thresholds, FOIR limits, "
                "LTV caps, and RERA compliance to approve or reject applications."
            ),
            readme_content=readme or None,
            version="0.1.0",
            author="Nijin",
            documentation_url="https://huggingface.co/spaces/iamnijin/credit-assessment-env",
        )


    @property
    def state(self) -> State:
        return self._state
