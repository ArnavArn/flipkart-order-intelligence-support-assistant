"""Deterministic template composer -- MOCK_LLM mode. Zero network calls, zero API keys.

compose(state) implements, by hand, what RESPONSE_PROMPT (prompts.py) would ask a live chat
model to do: read the context (retrieved chunks or a tool's structured result) and emit exactly
one {"answer", "source", "confidence"} JSON object. Every branch below is a plain string
template plus arithmetic -- there is nothing here that talks to a network.
"""
from __future__ import annotations

from part3_agent.tools.return_risk_tool import T_STAR

BLOCKED_ANSWER = (
    "I can't follow instructions that try to override my configuration. I can help with "
    "return policies, return-risk checks, or product-image categorisation."
)

UNGROUNDED_REFUSAL_TEMPLATE = (
    "I don't have a policy document covering that, so I won't guess. Please contact a human "
    "support agent. ({groundedness_msg})"
)

NO_ORDER_REFERENCED_ANSWER = "There is no order referenced earlier in this conversation."


def _risk_confidence(p: float, t_star: float) -> float:
    """Deterministic, documented confidence function for return_risk answers: how far `p` sits
    from the Low/Medium boundary (t_star), normalised by the largest possible distance to
    either side of that boundary (max(t_star, 1 - t_star)). A p right at the boundary yields
    confidence 0; a p at the extreme (0 or 1) yields confidence close to 1.
    """
    span = max(t_star, 1 - t_star, 1e-6)
    dist = abs(p - t_star)
    return round(min(1.0, dist / span), 4)


def _compose_blocked(state: dict) -> dict:
    return {"answer": BLOCKED_ANSWER, "source": "policy_kb", "confidence": 1.0}


def _compose_policy(state: dict) -> dict:
    retrieved = state.get("retrieved") or []
    top_score = state.get("top_score", 0.0)
    if not retrieved:
        answer = "I don't have a policy document covering that."
    else:
        sentences = [c["text"] for c in retrieved[:2]]
        answer = "Based on Flipkart's policy documents: " + " ".join(sentences)
        if state.get("coref_resolved") and state.get("last_order_id"):
            # The user's pronoun ("its"/"that order") was resolved to an order carried in
            # THIS thread's state -- make that resolution explicit in the answer text.
            answer = f"For order {state['last_order_id']}: " + answer
    return {"answer": answer, "source": "policy_kb", "confidence": round(float(top_score), 3)}


def _compose_return_risk(state: dict) -> dict:
    tool_result = state.get("tool_result") or {}
    p = tool_result.get("return_probability", 0.0)
    bucket = tool_result.get("risk_bucket", "Unknown")
    t_star = tool_result.get("t_star_rf", T_STAR)
    cut_points = tool_result.get("cut_points", {})
    order_id = state.get("last_order_id")
    order_ref = f"Order {order_id}" if order_id else "This order"
    answer = (
        f"{order_ref} is flagged as {bucket} return risk: predicted return probability "
        f"{p:.2%}, against a model threshold t*_rf={t_star} "
        f"(Low < {cut_points.get('low_max', t_star)}, "
        f"High >= {cut_points.get('high_min', round(t_star + 0.15, 4))})."
    )
    confidence = _risk_confidence(p, t_star)
    return {"answer": answer, "source": "return_risk_tool", "confidence": confidence}


def _compose_product_category(state: dict) -> dict:
    tool_result = state.get("tool_result") or {}
    category = tool_result.get("category")
    confidence = tool_result.get("confidence", 0.0)
    if category is None:
        answer = tool_result.get("error", "I could not classify that image.")
        confidence = 0.0
    else:
        answer = f"This product image is classified as '{category}' with confidence {confidence:.2%}."
    return {"answer": answer, "source": "image_classifier_tool", "confidence": round(float(confidence), 4)}


def _compose_coref_unresolved(state: dict) -> dict:
    return {"answer": NO_ORDER_REFERENCED_ANSWER, "source": "policy_kb", "confidence": 0.0}


def compose(state: dict) -> dict:
    """Rule-based/template composition of the final structured answer."""
    if state.get("injection_blocked"):
        return _compose_blocked(state)

    if state.get("coref_unresolved"):
        # A pronoun ("its"/"that order") was used but this thread's state has no last_order_id
        # -- there is nothing to resolve it to, so we say so directly instead of guessing.
        return _compose_coref_unresolved(state)

    intent = state.get("intent", "unknown")
    if intent == "return_risk":
        return _compose_return_risk(state)
    if intent == "product_category":
        return _compose_product_category(state)
    # "policy" and "unknown" both go through retrieval and share the same composition template;
    # verify_output is what may later overwrite this with an ungrounded refusal.
    return _compose_policy(state)
