"""Single shared MiniLM SentenceTransformer instance -- used by both retriever.py (KB search)
and graph.py's classify_intent node (few-shot exemplar cosine matching), so the same embedding
space backs retrieval and intent routing.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from part3_agent import config

_MODEL = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _MODEL
