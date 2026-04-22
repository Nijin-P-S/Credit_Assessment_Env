"""Shared pytest setup for the Credit Assessment Environment test suite.

Adds the repo root to ``sys.path`` so tests can import ``train_utils`` directly
without needing the package installed. Mirrors how ``train_grpo.py`` imports
from ``train_utils`` at the top level.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
