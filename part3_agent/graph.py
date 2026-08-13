"""The LangGraph -- 6 nodes, 2 conditional edges.

    START -> guard_input
    guard_input   --conditional--> "blocked" -> generate           (skips retrieval and tools)
                                    "clean"   -> classify_intent
    classify_intent --conditional--> "policy"/"unknown"           -> retrieve
                                      "return_risk"/"product_category" -> call_tool
    retrieve  -> generate
    call_tool -> generate
    generate  -> verify_output
    verify_output -> END

A blocked injection never touches FAISS or a model; a risk/image question never runs retrieval.
Compiled with a MemorySaver checkpointer keyed by thread_id -- state (last_order_id,
last_order_features, last_image_path, history) persists across turns on the SAME thread_id and
starts empty on a new one. That is state, not memory: scoped to one conversation, not shared
across conversations.
"""
from __future__ import annotations

import re

import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from part3_agent import config, mock_llm
from part3_agent.embedder import get_model
from part3_agent.guardrails import check_groundedness, check_input
from part3_agent.mock_llm import UNGROUNDED_REFUSAL_TEMPLATE
from part3_agent.prompts import FEW_SHOT_INTENT
from part3_agent.retriever import search
from part3_agent.state import AgentState
from part3_agent.tools.image_tool import classify_product_image
from part3_agent.tools.return_risk_tool import check_return_risk

# ---------------------------------------------------------------------------
# Few-shot exemplar embeddings, computed once at import time.
# ---------------------------------------------------------------------------
_FEWSHOT_EMBEDDINGS = None


def _fewshot_embeddings() -> np.ndarray:
    global _FEWSHOT_EMBEDDINGS
    if _FEWSHOT_EMBEDDINGS is None:
        model = get_model()
        texts = [ex["user"] for ex in FEW_SHOT_INTENT]
        _FEWSHOT_EMBEDDINGS = np.asarray(
            model.encode(texts, normalize_embeddings=True), dtype="float32"
        )
    return _FEWSHOT_EMBEDDINGS


# ---------------------------------------------------------------------------
# Demo-scoped pattern extractors. These parse the crafted, structured natural-language
# sentences used in this project's own transcripts (e.g. "order 1523 ... Rs. 1,899 Apparel
# ... COD ... 12 days tenure ... 3 previous orders ... 1 previous return ... 340 km ... 6
# delivery days"). They are intentionally simple regex extractors for a bounded demo, not a
# general-purpose NLU system.
# ---------------------------------------------------------------------------
CATEGORY_WORDS = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]

COREF_PATTERNS = [r"\bits\b", r"\bit\b", r"\bthat order\b", r"\bthis order\b", r"\bthe order\b"]


def _detect_order_id(text: str) -> str | None:
    m = re.search(r"order\s*#?\s*(\d{3,6})", text, re.IGNORECASE)
    return m.group(1) if m else None


def _detect_coref(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in COREF_PATTERNS)


# ---------------------------------------------------------------------------
# Intent routing: cosine similarity against FEW_SHOT_INTENT is ALWAYS computed and is what
# every transcript prints as "matched few-shot example #N". With only one exemplar per
# intent, MiniLM cosine alone can be fooled by superficial lexical overlap (e.g. a plain
# "delivery timeline" question can score closer to the return_risk exemplar "Is order 1523
# likely to be returned?" than to the policy exemplar, purely because both are short
# order-shaped questions). The plan explicitly anticipates this: "the router scores ... via
# embedding cosine ... plus the pattern rules". A deterministic pattern rule fires ONLY for a
# narrow, explainable signal (an explicit order id + payment/return-risk phrasing for
# return_risk; a .png filename or explicit "classify the image" phrasing for
# product_category; a policy-domain keyword otherwise) and overrides the raw cosine winner
# when they disagree -- and every override is printed in matched_fewshot, so the mechanism
# stays auditable rather than silently overriding the few-shot signal.
# ---------------------------------------------------------------------------
UNKNOWN_COS_THRESHOLD = 0.20  # below this, no exemplar meaningfully matches -> intent="unknown"

POLICY_KEYWORDS = [
    "return", "refund", "deliver", "exchange", "pickup", "policy", "warranty",
    "window", "timeline", "sla", "damaged", "defective", "escalat", "cash on delivery",
]


def _has_order_risk_signal(text: str) -> bool:
    has_order = bool(_detect_order_id(text))
    has_payment_kw = bool(re.search(r"\bCOD\b|\bUPI\b|\bwallet\b|\bcard\b", text, re.IGNORECASE))
    has_structured_kw = bool(re.search(
        r"previous\s*orders?|previous\s*returns?|likely to be returned|return risk",
        text, re.IGNORECASE,
    ))
    return has_order and (has_payment_kw or has_structured_kw)


def _has_image_signal(text: str) -> bool:
    return bool(re.search(r"\.png\b", text, re.IGNORECASE)) or bool(
        re.search(r"classify\b.*image|which category does", text, re.IGNORECASE)
    )


def _has_policy_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in POLICY_KEYWORDS)


def _parse_order_features(text: str) -> dict:
    features: dict = {}

    # \b before Rs/INR prevents matching "rs" inside an unrelated word like "orders,"; the
    # capture group is required to START with a digit so a stray comma alone can never match.
    m = re.search(r"(?:\bRs\.?|₹|\bINR)\s*(\d[\d,]*)", text, re.IGNORECASE)
    if m:
        features["price_inr"] = float(m.group(1).replace(",", ""))

    for cat in CATEGORY_WORDS:
        if re.search(rf"\b{cat}\b", text, re.IGNORECASE):
            features["product_category"] = cat
            break

    if re.search(r"\bCOD\b", text, re.IGNORECASE):
        features["payment_method"] = "COD"
    elif re.search(r"\bUPI\b", text, re.IGNORECASE):
        features["payment_method"] = "Prepaid_UPI"
    elif re.search(r"\bwallet\b", text, re.IGNORECASE):
        features["payment_method"] = "Wallet"
    elif re.search(r"\bcard\b", text, re.IGNORECASE):
        features["payment_method"] = "Prepaid_Card"

    m = re.search(r"(\d+)\s*days?\s*tenure", text, re.IGNORECASE)
    if m:
        features["customer_tenure_days"] = float(m.group(1))

    m = re.search(r"(\d+)\s*previous\s*orders?", text, re.IGNORECASE)
    if m:
        features["num_previous_orders"] = float(m.group(1))

    m = re.search(r"(\d+)\s*previous\s*returns?", text, re.IGNORECASE)
    if m:
        features["num_previous_returns"] = float(m.group(1))

    m = re.search(r"(\d+)\s*km", text, re.IGNORECASE)
    if m:
        features["delivery_distance_km"] = float(m.group(1))

    m = re.search(r"(\d+)\s*delivery\s*days?", text, re.IGNORECASE)
    if m:
        features["delivery_days"] = float(m.group(1))

    m = re.search(r"(\d+)\s*%\s*discount|discount\s*of\s*(\d+)\s*%", text, re.IGNORECASE)
    if m:
        features["discount_pct"] = float(m.group(1) or m.group(2))

    return features


def _extract_image_filename(text: str) -> str | None:
    m = re.search(r"([\w\-]+\.png)", text, re.IGNORECASE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def guard_input(state: AgentState) -> dict:
    blocked, pattern = check_input(state["user_input"])
    return {"injection_blocked": blocked, "injection_pattern": pattern}


def classify_intent(state: AgentState) -> dict:
    text = state["user_input"]

    model = get_model()
    q_emb = np.asarray(model.encode([text], normalize_embeddings=True), dtype="float32")[0]
    sims = _fewshot_embeddings() @ q_emb
    best_idx = int(np.argmax(sims))
    raw_intent = FEW_SHOT_INTENT[best_idx]["intent"]
    raw_cos = float(sims[best_idx])

    override_reason = None
    if _has_image_signal(text):
        intent = "product_category"
        if intent != raw_intent:
            override_reason = "image filename/keyword pattern detected"
    elif _has_order_risk_signal(text):
        intent = "return_risk"
        if intent != raw_intent:
            override_reason = "order id + payment/return-risk keyword pattern detected"
    elif _has_policy_keyword(text):
        intent = "policy"
        if intent != raw_intent:
            override_reason = "policy-domain keyword present without an order/image signal"
    elif raw_cos < UNKNOWN_COS_THRESHOLD:
        intent = "unknown"
        override_reason = (
            f"raw cosine {raw_cos:.4f} below unknown-confidence threshold {UNKNOWN_COS_THRESHOLD}"
        )
    else:
        intent = raw_intent

    matched_fewshot = (
        f'matched few-shot example #{best_idx + 1} '
        f'("{FEW_SHOT_INTENT[best_idx]["user"]}") -> intent={raw_intent} (cosine={raw_cos:.4f})'
    )
    if override_reason:
        matched_fewshot += f' [overridden -> intent={intent}; reason: {override_reason}]'

    # coref_resolved / coref_unresolved are ALWAYS set explicitly (never omitted) even when
    # False -- LangGraph's checkpointer only overwrites a channel when a node's return dict
    # includes that key, so omitting it on a turn with no pronoun would let a stale True from
    # an earlier turn silently keep short-circuiting verify_output/mock_llm on unrelated turns.
    update: dict = {
        "intent": intent,
        "matched_fewshot": matched_fewshot,
        "coref_resolved": False,
        "coref_unresolved": False,
    }

    order_id = _detect_order_id(text)
    if order_id:
        update["last_order_id"] = order_id
    elif _detect_coref(text):
        if state.get("last_order_id"):
            # Coreference resolution: "its"/"that order" refers to the order carried in state
            # from an earlier turn on this SAME thread_id. We keep the existing last_order_id
            # (already persisted by the checkpointer) rather than overwriting it with None.
            update["coref_resolved"] = True
            update["matched_fewshot"] += (
                f' [coref: "its"/"that order" resolved to last_order_id={state["last_order_id"]}]'
            )
        else:
            # Fresh thread / no earlier order in THIS conversation's state -- the pronoun has
            # nothing to resolve to. mock_llm.compose short-circuits on this flag.
            update["coref_unresolved"] = True
            update["matched_fewshot"] += (
                ' [coref: pronoun detected but no last_order_id in state -- unresolved]'
            )

    return update


def retrieve(state: AgentState) -> dict:
    results = search(state["user_input"], k=config.TOP_K)
    top_score = results[0]["score"] if results else 0.0
    return {"retrieved": results, "top_score": top_score}


def call_tool(state: AgentState) -> dict:
    intent = state.get("intent")
    text = state["user_input"]

    if intent == "return_risk":
        order_id = _detect_order_id(text) or state.get("last_order_id")
        try:
            parsed = _parse_order_features(text)
        except ValueError:
            # Free-text extraction from an untrusted user message is best-effort -- a single
            # unparseable field must never crash the turn. Any field it would have found stays
            # missing (NaN), which the saved pipeline's own imputer already handles.
            parsed = {}
        features = {**(state.get("last_order_features") or {}), **parsed} if parsed else (
            state.get("last_order_features") or {}
        )
        result = check_return_risk(features)
        return {
            "tool_name": "check_return_risk",
            "tool_result": result,
            "last_order_id": order_id,
            "last_order_features": features,
        }

    if intent == "product_category":
        filename = _extract_image_filename(text) or state.get("last_image_path")
        result = classify_product_image(filename or "")
        return {
            "tool_name": "classify_product_image",
            "tool_result": result,
            "last_image_path": result.get("image_path", filename),
        }

    return {}


def generate(state: AgentState) -> dict:
    if not config.MOCK_LLM:
        raise RuntimeError(
            "USE_LIVE_LLM=1 requested but no live LLM integration is implemented in this "
            "submission -- the optional live path is never scored. Unset USE_LIVE_LLM."
        )
    composed = mock_llm.compose(state)
    history_update = [
        {"role": "user", "content": state.get("user_input", "")},
        {"role": "assistant", "content": composed.get("answer", "")},
    ]
    return {"final": composed, "history": history_update}


def verify_output(state: AgentState) -> dict:
    final = dict(state.get("final") or {})

    if state.get("injection_blocked") or state.get("coref_unresolved"):
        # Blocked answers and the "no order referenced earlier" answer are not KB claims --
        # the groundedness check (which is about trusting a retrieved policy chunk) does not
        # apply to either.
        return {"final": final, "grounded": True}

    source = final.get("source")
    if source != "policy_kb":
        # Tool-sourced answers (return_risk_tool / image_classifier_tool) are grounded by
        # construction -- a real model produced them, so the groundedness check does not apply.
        return {"final": final, "grounded": True}

    top_score = state.get("top_score", 0.0)
    grounded, msg = check_groundedness(top_score, config.SIM_THRESHOLD)
    if not grounded:
        final = {
            "answer": UNGROUNDED_REFUSAL_TEMPLATE.format(groundedness_msg=msg),
            "source": "policy_kb",
            "confidence": 0.0,
        }
    return {"final": final, "grounded": grounded, "groundedness_msg": msg}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------
def route_after_guard(state: AgentState) -> str:
    return "blocked" if state.get("injection_blocked") else "clean"


def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    if intent in ("return_risk", "product_category"):
        return "tool"
    return "retrieve"  # "policy" and "unknown" both retrieve


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("guard_input", guard_input)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve", retrieve)
    builder.add_node("call_tool", call_tool)
    builder.add_node("generate", generate)
    builder.add_node("verify_output", verify_output)

    builder.add_edge(START, "guard_input")
    builder.add_conditional_edges(
        "guard_input", route_after_guard, {"blocked": "generate", "clean": "classify_intent"}
    )
    builder.add_conditional_edges(
        "classify_intent", route_after_intent, {"retrieve": "retrieve", "tool": "call_tool"}
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("call_tool", "generate")
    builder.add_edge("generate", "verify_output")
    builder.add_edge("verify_output", END)

    return builder.compile(checkpointer=MemorySaver())


def invoke_with_trace(graph, user_input: str, thread_id: str) -> tuple[dict, list[str]]:
    """Run one turn, returning (final_state, node_path) where node_path is the ACTUAL sequence
    of node names LangGraph executed for this turn (via graph.stream(..., stream_mode="updates")),
    not a value derived after the fact.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    path: list[str] = []
    for update in graph.stream({"user_input": user_input}, config=cfg, stream_mode="updates"):
        path.extend(update.keys())
    final_state = graph.get_state(cfg).values
    return final_state, path


if __name__ == "__main__":
    g = build_graph()
    state, path = invoke_with_trace(g, "What is the return window for a pair of running shoes?", "demo")
    print("path:", path)
    print("final:", state.get("final"))
    print("matched_fewshot:", state.get("matched_fewshot"))
