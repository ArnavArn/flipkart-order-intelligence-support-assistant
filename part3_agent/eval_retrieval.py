"""Document-level, deduplicated Precision@3 / Recall@3 over the 6 answer-key queries, written to transcripts/retrieval_eval.md.
Run: python -m part3_agent.eval_retrieval
"""
from __future__ import annotations

import json

from part3_agent import config
from part3_agent.index_build import build_index, calibration_table
from part3_agent.retriever import search


def _dedup_docs(chunks: list[dict]) -> list[str]:
    """Dedup chunk doc_ids while preserving first-seen order."""
    seen = []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.append(c["doc_id"])
    return seen


def evaluate() -> tuple[str, float, float]:
    eval_queries = json.loads(config.KB_EVAL_QUERIES_PATH.read_text(encoding="utf-8"))

    lines = []
    lines.append("# Retrieval evaluation -- Precision@3 / Recall@3 (document level, deduped)\n")
    lines.append(
        "Scoring is at the **document** level: each query's top-3 chunks are mapped to their "
        "parent `doc_id` and deduplicated, so the Precision@3 denominator is the number of "
        "*unique documents* retrieved (<= 3), not a fixed 3. Recall@3's denominator is the "
        "number of documents in the answer key for that query.\n"
    )

    precisions = []
    recalls = []

    for q in eval_queries:
        qid = q["qid"]
        query = q["query"]
        relevant = set(q["relevant_doc_ids"])

        chunks = search(query, k=config.TOP_K)
        ret_docs_ordered = _dedup_docs(chunks)
        ret_docs = set(ret_docs_ordered)
        hits = ret_docs & relevant

        precision = len(hits) / len(ret_docs) if ret_docs else 0.0
        recall = len(hits) / len(relevant) if relevant else 0.0
        precisions.append(precision)
        recalls.append(recall)

        chunk_str = ", ".join(f"{c['chunk_id']} ({c['score']:.2f})" for c in chunks)

        lines.append(f"### {qid} -- \"{query}\"")
        lines.append(f"- relevant docs (answer key): {{{', '.join(sorted(relevant))}}}")
        lines.append(f"- top-3 chunks: {chunk_str}")
        lines.append(
            f"- retrieved docs after dedup: {{{', '.join(ret_docs_ordered)}}}  "
            f"({len(chunks)} chunks -> {len(ret_docs)} unique documents)"
        )
        lines.append(f"- hits: {{{', '.join(sorted(hits)) if hits else ''}}} -> {len(hits)}")
        lines.append(f"- Precision@3 = {len(hits)} / {len(ret_docs)} = {precision:.3f}")
        lines.append(f"- Recall@3    = {len(hits)} / {len(relevant)} = {recall:.3f}")
        lines.append("")

    mean_p = sum(precisions) / len(precisions)
    mean_r = sum(recalls) / len(recalls)

    lines.append("## Averages across all 6 queries")
    lines.append(
        f"- mean Precision@3 = ({' + '.join(f'{p:.3f}' for p in precisions)}) / {len(precisions)} "
        f"= {mean_p:.3f}"
    )
    lines.append(
        f"- mean Recall@3    = ({' + '.join(f'{r:.3f}' for r in recalls)}) / {len(recalls)} "
        f"= {mean_r:.3f}"
    )
    lines.append("")

    report_md = "\n".join(lines)
    return report_md, mean_p, mean_r


def main() -> None:
    report_md, mean_p, mean_r = evaluate()
    print(report_md)
    print(f"mean Precision@3 = {mean_p:.3f}, mean Recall@3 = {mean_r:.3f}")

    # Rebuilt in-memory since this is a read-only eval script -- never overwrites the committed index files.
    index, chunks, model = build_index()
    calib_md = calibration_table(index, chunks, model)

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.TRANSCRIPTS_DIR / "retrieval_eval.md"
    out_path.write_text(report_md + "\n---\n\n" + calib_md + "\n", encoding="utf-8")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
