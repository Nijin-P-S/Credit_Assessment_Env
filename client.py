"""Client for the Credit Assessment Environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CreditAssessmentAction, CreditAssessmentObservation


class CreditAssessmentEnv(
    EnvClient[CreditAssessmentAction, CreditAssessmentObservation, State]
):
    """
    WebSocket client for interacting with the Credit Assessment server.

    Example:
        >>> with CreditAssessmentEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.applicant_profile)
        ...     result = client.step(CreditAssessmentAction(
        ...         decision=LoanDecision.APPROVE,
        ...         reasoning="Strong credit profile"
        ...     ))
    """

    def _step_payload(self, action: CreditAssessmentAction) -> Dict:
        payload = {
            "decision": action.decision.value,
            "reasoning": action.reasoning,
        }
        if action.counter_offer_amount is not None:
            payload["counter_offer_amount"] = action.counter_offer_amount
        if action.docs_requested is not None:
            payload["docs_requested"] = action.docs_requested
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[CreditAssessmentObservation]:
        obs_data = payload.get("observation", {})
        observation = CreditAssessmentObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
