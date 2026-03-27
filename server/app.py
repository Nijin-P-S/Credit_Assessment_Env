# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server for the Credit Assessment Environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import CreditAssessmentAction, CreditAssessmentObservation
    from .credit_assessment_env_environment import CreditAssessmentEnvironment
except ModuleNotFoundError:
    from models import CreditAssessmentAction, CreditAssessmentObservation
    from server.credit_assessment_env_environment import CreditAssessmentEnvironment


app = create_app(
    CreditAssessmentEnvironment,
    CreditAssessmentAction,
    CreditAssessmentObservation,
    env_name="credit_assessment_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the server directly: uv run --project . server"""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
