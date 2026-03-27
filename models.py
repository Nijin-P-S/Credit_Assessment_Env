# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Action and Observation schemas for the Credit Assessment Environment."""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional
from .loan_decision import LoanDecision


class CreditAssessmentAction(Action):
    decision: LoanDecision = Field(
        ...,
        description="approve/reject/request_docs/counter_offer"
    )
    reasoning: str = Field(
        ...,
        description="Why agent made this decision"
    )
    counter_offer_amount: Optional[float] = Field(
        default=None,
        description="Only required if decision is counter_offer"
    )
    docs_requested: Optional[str] = Field(
        default=None,
        description="Only required if decision is request_docs"
    )


class CreditAssessmentObservation(Observation):
    """Loan application data returned to the agent. Optional fields are None
    when not applicable to the loan type (e.g., rera_registered is None for
    personal loans)."""

    applicant_profile: str = Field(
        ...,
        description="Full text description of applicant for LLM reasoning"
    )

    credit_score: int = Field(
        ...,
        description="CIBIL credit score (300-900)"
    )
    monthly_income: float = Field(
        ...,
        description="Net monthly income in INR"
    )
    foir: float = Field(
        ...,
        description="Fixed Obligation to Income Ratio (0.0-1.0). Above 0.5 is high risk."
    )
    employment_years: int = Field(
        ...,
        description="Years at current employer. Below 1 year is high risk."
    )
    loan_type: str = Field(
        ...,
        description="Type of loan: personal/vehicle/home"
    )
    loan_amount: float = Field(
        ...,
        description="Requested loan amount in INR"
    )
    documents_complete: bool = Field(
        ...,
        description="Whether all required documents are submitted and verified"
    )

    collateral_value: Optional[float] = Field(
        default=None,
        description="Value of collateral in INR. Present for vehicle and home loans only."
    )
    ltv_ratio: Optional[float] = Field(
        default=None,
        description="Loan to Value ratio (0.0-1.0). Max 0.85 for vehicle, 0.80 for home."
    )

    rera_registered: Optional[bool] = Field(
        default=None,
        description="Whether property is RERA registered. Mandatory for home loans in India."
    )
    has_co_applicant: Optional[bool] = Field(
        default=None,
        description="Whether application has a co-applicant. Strengthens home loan applications."
    )

    task_id: int = Field(
        ...,
        description="Task identifier: 1=personal(easy), 2=vehicle(medium), 3=home(hard)"
    )
    available_actions: str = Field(
        default="approve, reject, request_docs, counter_offer",
        description="Actions available to the agent at this step"
    )
    reward: float = Field(
        default=0.0,
        description="Reward received for the previous action"
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended"
    )