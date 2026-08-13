"""Sentence-wise chunking of the KB markdown docs.

The brief names sentence-wise splitting as the production-appropriate strategy over fixed-size
or overlapping windows for short, hand-authored policy text. A regex split on
`(?<=[.!?])\\s+` is sufficient and fully deterministic for text we authored ourselves — no NLTK
dependency needed.

Each doc has a YAML-ish frontmatter block (---\\ndoc_id: ...\\ntitle: ...\\ncategory: ...\\n---)
followed by the body. We parse the frontmatter with simple line splitting (no external YAML
library needed for this trivial 3-key format) and split the body into sentences.
"""
from __future__ import annotations

import re
from pathlib import Path

from part3_agent import config

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_doc(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError(f"{path} is missing the expected YAML frontmatter block")
    front_raw, body = m.group(1), m.group(2)

    meta = {}
    for line in front_raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()

    body = body.strip()
    return {
        "doc_id": meta["doc_id"],
        "title": meta["title"],
        "category": meta.get("category", ""),
        "body": body,
    }


def load_documents(documents_dir: Path = config.KB_DOCUMENTS_DIR) -> list[dict]:
    """Read every POL-*.md file, sorted by doc_id, parsed into {doc_id, title, category, body}."""
    paths = sorted(documents_dir.glob("*.md"))
    return [_parse_doc(p) for p in paths]


def split_sentences(text: str) -> list[str]:
    """Split body text into sentences on `(?<=[.!?])\\s+`, dropping empties.

    Line-wrapped source markdown can embed newlines inside a sentence; collapse all
    whitespace runs to a single space first so chunk text reads as one clean sentence.
    """
    normalised = re.sub(r"\s+", " ", text.strip())
    parts = SENTENCE_SPLIT_RE.split(normalised)
    return [p.strip() for p in parts if p.strip()]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Sentence-wise chunk every document, carrying doc_id/title for traceback.

    Returns a flat list of:
        {"chunk_id": "POL-01#2", "doc_id": "POL-01", "doc_title": "...",
         "sentence_index": 2, "text": "..."}
    """
    chunks = []
    for doc in documents:
        sentences = split_sentences(doc["body"])
        for i, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#{i}",
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "sentence_index": i,
                "text": sentence,
            })
    return chunks


if __name__ == "__main__":
    docs = load_documents()
    all_chunks = chunk_documents(docs)
    print(f"{len(docs)} documents -> {len(all_chunks)} sentence-wise chunks")
    for c in all_chunks[:5]:
        print(c)
