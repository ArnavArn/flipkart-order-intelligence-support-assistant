"""search(query, k) -> top-k chunks with cosine similarity scores, backed by the persisted
FAISS IndexFlatIP + chunks.json built by index_build.py.
"""
from __future__ import annotations

import json

import faiss
import numpy as np

from part3_agent import config
from part3_agent.embedder import get_model

_INDEX = None
_CHUNKS = None


def _load():
    global _INDEX, _CHUNKS
    if _INDEX is None:
        if not config.FAISS_INDEX_PATH.exists() or not config.CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run `python -m part3_agent.index_build` first."
            )
        _INDEX = faiss.read_index(str(config.FAISS_INDEX_PATH))
        _CHUNKS = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    return _INDEX, _CHUNKS, get_model()


def search(query: str, k: int = config.TOP_K) -> list[dict]:
    """Return top-k chunks for `query`, sorted by cosine score descending
    -> [{chunk_id, doc_id, doc_title, text, score}]."""
    index, chunks, model = _load()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    k = min(k, len(chunks))
    scores, idxs = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "doc_title": chunk["doc_title"],
            "text": chunk["text"],
            "score": float(score),
        })
    return results


if __name__ == "__main__":
    for r in search("How many days do I have to return a kurta I bought?"):
        print(r)
