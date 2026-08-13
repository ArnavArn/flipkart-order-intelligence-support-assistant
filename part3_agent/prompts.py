"""Prompt engineering for Part 3.

Two SEPARATE prompts -- that separation is itself an instance of the SINGLE principle applied
one level up: INTENT_PROMPT does classification only, RESPONSE_PROMPT does composition only.
Neither prompt is ever sent over the network in MOCK_LLM mode (config.MOCK_LLM is True by
default) -- they exist as documented, annotated specifications that mock_llm.py's deterministic
template logic implements by hand. The optional USE_LIVE_LLM=1 path (not implemented/graded)
would send RESPONSE_PROMPT.format(...) to a real chat model; MOCK_LLM mode never does.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# INTENT_PROMPT -- classification only.
# ---------------------------------------------------------------------------
INTENT_PROMPT = """
You are an intent router for Flipkart customer support.            # ROLE PROMPTING
Classify the user's message into exactly one of four intents:      # [SPECIFIC] closed label set, no ambiguity
policy, return_risk, product_category, unknown.
Use the few-shot examples below as your calibration anchors —      # [SPECIFIC] the mechanism named, not left implicit
pick whichever example the message is most similar to.
If the message resolves a pronoun like "its" or "that order" to    # [SPECIFIC] the failure mode (dangling reference) named
an order mentioned earlier, keep the SAME intent that order's
topic implies unless the new message clearly asks something else.

Reply with exactly one lowercase word: the intent label.           # [SINGLE] one task, one output shape
"""                                                                 # [SHORT] under 120 words, no filler

# ---------------------------------------------------------------------------
# RESPONSE_PROMPT -- composition only.
# ---------------------------------------------------------------------------
RESPONSE_PROMPT = """
You are Flipkart's customer-support assistant.          # ROLE PROMPTING
Answer only from the context supplied below.            # [SPECIFIC] scope is explicit
You have exactly three capabilities: policy lookup,      # [SPECIFIC] enumerated, no ambiguity
return-risk scoring, product-image categorisation.
Never invent a policy that is not in the context.        # [SPECIFIC] the failure mode, named

<<<CONTEXT>>>                                            # [SURROUND] delimited context block
{context}
<<<END CONTEXT>>>

Reply with one JSON object and nothing else:             # [SINGLE] one task, one output shape
{{"answer": str, "source": "policy_kb"|"return_risk_tool"|"image_classifier_tool",
  "confidence": float}}
"""                                                       # [SHORT] under 120 words, no filler

# Kept for backward-compatible naming with the plan doc; SYSTEM_PROMPT is RESPONSE_PROMPT.
SYSTEM_PROMPT = RESPONSE_PROMPT

# ---------------------------------------------------------------------------
# Few-shot intent examples (>=2 required; we ship 3).
#
# In MOCK_LLM mode these are NOT text pasted into a live prompt -- graph.py's classify_intent
# node embeds the user's input with the same MiniLM model used for retrieval, scores it against
# the embedding of each exemplar's "user" text via cosine similarity, and routes to the winning
# exemplar's intent. state["matched_fewshot"] records which exemplar won, so every transcript
# can print e.g.:
#   [intent] matched few-shot example #2 ("Is order 1523 likely to be returned?") -> intent=return_risk
# ---------------------------------------------------------------------------
FEW_SHOT_INTENT = [
    {"user": "What is the return window for a pair of running shoes?",
     "intent": "policy"},
    {"user": "Is order 1523 likely to be returned?",
     "intent": "return_risk"},
    {"user": "Which category does data/sample_images/07_sneaker.png belong to?",
     "intent": "product_category"},
]
