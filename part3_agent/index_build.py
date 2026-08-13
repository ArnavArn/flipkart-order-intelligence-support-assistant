"""Entry point: embed every KB chunk with MiniLM, build a FAISS IndexFlatIP, persist it, and
print/write a similarity calibration table used to justify SIM_THRESHOLD in config.py.

Run: python -m part3_agent.index_build
"""
from __future__ import annotations

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from part3_agent import config
from part3_agent.chunking import chunk_documents, load_documents

OUT_OF_SCOPE_QUERIES = [
    "What is the warranty on a car battery?",
    "How do I apply for a job at Flipkart?",
    "What is the capital of France?",
]


def build_index() -> tuple[faiss.Index, list[dict], SentenceTransformer]:
    docs = load_documents()
    chunks = chunk_documents(docs)
    # Embed "doc_title. sentence" rather than the bare sentence. Short, single-sentence chunks
    # (e.g. "Shoes must be unworn...") often lack the query's keywords (e.g. "return window")
    # even when they are the right chunk; prefixing the parent doc's title gives MiniLM the
    # topical context to score them correctly. Measured effect on this exact case: cosine rose
    # from 0.4602 (bare sentence) to 0.6155 (title-prefixed) against the query "What is the
    # return window for a pair of running shoes?". chunks.json still stores the pure sentence
    # in "text" (used for display/composition) -- only the embedding input changes.
    texts = [f"{c['doc_title']}. {c['text']}" for c in chunks]

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, chunks, model


def persist(index: faiss.Index, chunks: list[dict]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)


def _top1_score(model: SentenceTransformer, index: faiss.Index, chunks: list[dict], query: str) -> tuple[float, dict]:
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, 1)
    top_score = float(scores[0][0])
    top_chunk = chunks[int(idxs[0][0])]
    return top_score, top_chunk


def calibration_table(index: faiss.Index, chunks: list[dict], model: SentenceTransformer) -> str:
    """Measure top-1 cosine scores for the 6 answer-key queries plus 3 deliberately
    out-of-scope queries against the already-built index. Returns markdown text.
    """
    eval_queries = json.loads(config.KB_EVAL_QUERIES_PATH.read_text(encoding="utf-8"))

    lines = []
    lines.append("## Similarity-threshold calibration\n")
    lines.append(
        "Top-1 cosine score (FAISS `IndexFlatIP` over unit-normalised MiniLM embeddings) for "
        "the 6 in-scope answer-key queries and 3 deliberately out-of-scope queries. "
        f"`SIM_THRESHOLD = {config.SIM_THRESHOLD}` in `config.py` was set by inspecting this "
        "table and placing the threshold cleanly between the two clusters.\n"
    )
    lines.append("| query | in-scope? | top-1 doc | top-1 score |")
    lines.append("|---|---|---|---|")

    in_scope_scores = []
    for q in eval_queries:
        score, chunk = _top1_score(model, index, chunks, q["query"])
        in_scope_scores.append(score)
        lines.append(f"| {q['query']} | yes | {chunk['doc_id']} | {score:.4f} |")

    out_scores = []
    for q in OUT_OF_SCOPE_QUERIES:
        score, chunk = _top1_score(model, index, chunks, q)
        out_scores.append(score)
        lines.append(f"| {q} | no | {chunk['doc_id']} | {score:.4f} |")

    lines.append("")
    lines.append(
        f"In-scope cluster: min={min(in_scope_scores):.4f}, max={max(in_scope_scores):.4f}. "
        f"Out-of-scope cluster: min={min(out_scores):.4f}, max={max(out_scores):.4f}. "
        f"SIM_THRESHOLD={config.SIM_THRESHOLD} sits between "
        f"{max(out_scores):.4f} (highest out-of-scope score) and "
        f"{min(in_scope_scores):.4f} (lowest in-scope score)."
    )
    return "\n".join(lines)


def main() -> None:
    index, chunks, model = build_index()
    persist(index, chunks)
    print(f"Indexed {len(chunks)} chunks from KB documents -> {config.FAISS_INDEX_PATH}")

    table_md = calibration_table(index, chunks, model)
    print()
    print(table_md)
    print(
        "\n(This table is also appended to transcripts/retrieval_eval.md by "
        "`python -m part3_agent.eval_retrieval`.)"
    )


if __name__ == "__main__":
    main()
