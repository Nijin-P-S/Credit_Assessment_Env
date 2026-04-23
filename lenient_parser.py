"""Shared lenient parser for loan-decision JSON responses.

This module is the single source of truth for extracting a structured loan
decision from a model response. It is used by:

  * The GRPO reward function (train_grpo.py)
  * The phase eval, baseline eval, final eval (train_grpo.py)
  * The standalone fair_eval.py
  * The Colab notebook

By centralising parsing here, the baseline-vs-trained comparison no longer
suffers from the original bug where two different evaluators used two
different parsers — silently giving the trained model an unfair advantage.

The parser handles every response format observed from instruction-tuned
LLMs in practice:

  * raw JSON                                   -> {"decision": "approve", ...}
  * ```json fenced JSON                        -> ```json\\n{...}\\n```
  * unlabelled fenced JSON                     -> ```\\n{...}\\n```
  * JSON preceded by chain-of-thought prose    -> "Step 1: CIBIL...\\n{...}"
  * JSON followed by trailing commentary       -> "{...}\\nThat is my answer."
  * Multiple JSON blocks (last wins)           -> "{...}\\n\\nFinal: {...}"
  * Decision spelled with capitals/whitespace  -> "Approve "

Design choices:

  * The parser intentionally returns the *parsed dict*, not just the decision
    string, so callers can pull counter_offer_amount, docs_requested, and
    reasoning when needed.
  * Returns None on failure rather than raising — the caller decides what a
    parse failure should cost in their reward / accuracy calculation.
  * Stays in pure-Python with stdlib json only — no torch / transformers
    dependency, so it is cheap to import and easy to unit test.
"""

from __future__ import annotations

import json
import re
from typing import Optional

# Decisions defined by LoanDecision in train_utils. Listed here so the parser
# module has zero dependency on the env package and can be imported from
# anywhere (including the Colab notebook before train_utils is on the path).
VALID_DECISIONS: frozenset[str] = frozenset(
    {"approve", "reject", "request_docs", "counter_offer"}
)

# Pattern that matches a JSON-like {...} block with balanced (non-nested)
# braces. Real applicant responses sometimes contain numerical fields that
# include other braces (rare), so we additionally try ALL matched blocks and
# return the last one that parses successfully — typically the model's final
# answer at the end of a CoT trace.
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_response(response: str) -> Optional[dict]:
    """Extract a loan-decision dict from a model response.

    Returns the parsed dict if it contains a recognisable ``decision`` field
    with a value in ``VALID_DECISIONS``. Returns ``None`` if no usable JSON
    block can be recovered.
    """
    if not response:
        return None

    text = response.strip()

    # Step 1: try the obvious — a fenced JSON block.
    fenced = _extract_fenced_block(text)
    if fenced is not None:
        parsed = _try_parse_dict(fenced)
        if parsed is not None and _has_valid_decision(parsed):
            return parsed

    # Step 2: try to parse the whole text as JSON (raw JSON response).
    parsed = _try_parse_dict(text)
    if parsed is not None and _has_valid_decision(parsed):
        return parsed

    # Step 3: scan for any {...} blocks and return the last one that parses
    # with a valid decision. This handles CoT responses that end in a JSON
    # answer block, and "multiple JSON" responses where the last block is the
    # final answer.
    candidates = _JSON_BLOCK_RE.findall(text)
    for block in reversed(candidates):
        parsed = _try_parse_dict(block)
        if parsed is not None and _has_valid_decision(parsed):
            return parsed

    # Step 4: last-ditch — find the outermost {...} span by character index.
    # This catches responses where braces are nested or the JSON spans a long
    # multi-line block that the simple regex above missed.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        parsed = _try_parse_dict(text[start : end + 1])
        if parsed is not None and _has_valid_decision(parsed):
            return parsed

    return None


def parse_decision(response: str) -> Optional[str]:
    """Return just the normalised decision string, or None on failure.

    Convenience wrapper for callers that only need the decision (e.g. accuracy
    scoring). Use parse_response when you also need counter_offer_amount,
    docs_requested, or reasoning text.
    """
    parsed = parse_response(response)
    if parsed is None:
        return None
    return _normalise_decision(parsed.get("decision"))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_fenced_block(text: str) -> Optional[str]:
    """Return the contents of the first ```json/``` fence, if present.

    Tolerates lower/upper case ``json`` tag and missing closing fence.
    """
    # Prefer ```json over plain ``` — instruction-tuned models almost always
    # use the labelled form when emitting JSON specifically.
    for marker in ("```json", "```JSON", "```Json"):
        if marker in text:
            after = text.split(marker, 1)[1]
            if "```" in after:
                return after.split("```", 1)[0].strip()
            # No closing fence — assume the rest of the response is the block.
            return after.strip()

    if "```" in text:
        # Plain triple-backtick block. Could be language-tagged with something
        # other than json (e.g. ```python) so we still attempt to parse it.
        after = text.split("```", 1)[1]
        # Drop a leading single-line tag like ``` followed by a language name.
        first_newline = after.find("\n")
        if first_newline != -1:
            tag = after[:first_newline].strip().lower()
            if tag and " " not in tag and len(tag) <= 16:
                after = after[first_newline + 1 :]
        if "```" in after:
            return after.split("```", 1)[0].strip()
        return after.strip()

    return None


def _try_parse_dict(text: str) -> Optional[dict]:
    """Best-effort JSON-decode; returns the dict or None."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _has_valid_decision(parsed: dict) -> bool:
    return _normalise_decision(parsed.get("decision")) is not None


def _normalise_decision(value) -> Optional[str]:
    """Lowercase, strip, and validate against VALID_DECISIONS."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if candidate in VALID_DECISIONS else None
