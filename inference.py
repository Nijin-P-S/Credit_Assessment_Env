"""
Inference script for Credit Assessment Environment.

Runs LLM agent against all 3 tasks and reports scores.

Usage:
    python inference.py

Requirements:
    OPENAI_API_KEY environment variable must be set.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_assessment_env.baseline import main

if __name__ == "__main__":
    if "--llm" not in sys.argv:
        sys.argv.append("--llm")
    main()