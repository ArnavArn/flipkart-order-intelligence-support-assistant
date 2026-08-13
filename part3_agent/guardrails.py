"""Input-side prompt-injection filter and output-side groundedness check."""
from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"ignore\s+all\s+rules",
    r"disregard\s+(the\s+)?(above|previous|system)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"you\s+are\s+now\s+",
    r"(reveal|show|print|repeat)\s+(your\s+)?(system\s+)?(prompt|instructions|policies)",
    r"act\s+as\s+(a|an|the)\s+",
    r"developer\s+mode|jailbreak|bypass\s+(your\s+)?(rules|filters|guardrails)",
    r"(new|updated)\s+(system|admin)\s+(prompt|instruction)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_input(text: str) -> tuple[bool, str | None]:
    """Case-insensitive scan against INJECTION_PATTERNS -> (blocked, matched_pattern)."""
    for pattern, compiled in zip(INJECTION_PATTERNS, _COMPILED):
        if compiled.search(text):
            return True, pattern
    return False, None


def check_groundedness(top_score: float, threshold: float) -> tuple[bool, str]:
    """Output-side check for policy answers: is the top retrieved chunk similar enough to the
    query to trust the answer, or should verify_output overwrite it with a refusal?
    """
    grounded = top_score >= threshold
    msg = (f"top retrieved chunk similarity = {top_score:.4f} "
           f"{'>=' if grounded else '<'} threshold {threshold:.4f}")
    return grounded, msg
