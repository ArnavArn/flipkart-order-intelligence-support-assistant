"""AgentState -- the LangGraph state schema shared by all 6 nodes in graph.py."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    turn_index: int
    history: list[dict]              # [{role, content}] within THIS conversation

    intent: str                      # policy | return_risk | product_category | unknown
    matched_fewshot: str             # which few-shot example the router matched, for transcripts

    injection_blocked: bool
    injection_pattern: str | None
    coref_unresolved: bool          # pronoun ("its"/"that order") with no last_order_id in state
    coref_resolved: bool            # pronoun resolved to last_order_id from this thread's state

    retrieved: list[dict]
    top_score: float

    tool_name: str | None
    tool_result: dict | None

    # state carried across turns in a multi-turn conversation
    last_order_id: str | None
    last_order_features: dict | None
    last_image_path: str | None

    grounded: bool
    groundedness_msg: str            # "top retrieved chunk similarity = X.XXXX >= threshold Y.YYYY"
    final: dict                      # {answer, source, confidence}
