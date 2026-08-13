"""Part 3 -- paths, MOCK_LLM flag, retrieval knobs.
All paths resolve relative to the repo root, never hardcoded absolute.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PART3_DIR = REPO_ROOT / "part3_agent"
KB_DIR = PART3_DIR / "kb"
KB_DOCUMENTS_DIR = KB_DIR / "documents"
KB_EVAL_QUERIES_PATH = KB_DIR / "eval_queries.json"

INDEX_DIR = PART3_DIR / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

MODELS_DIR = REPO_ROOT / "models"
RETURN_RISK_MODEL_PATH = MODELS_DIR / "return_risk_model.pkl"
RETURN_RISK_META_PATH = MODELS_DIR / "return_risk_meta.json"

SAMPLE_IMAGES_DIR = REPO_ROOT / "data" / "sample_images"

TRANSCRIPTS_DIR = REPO_ROOT / "transcripts"

# embedding / retrieval config
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Calibrated from index_build.py's table (transcripts/retrieval_eval.md): in-scope top-1 scores
# ranged [0.4996, 0.7875], out-of-scope [0.0881, 0.4550]; 0.48 sits cleanly between the two.
SIM_THRESHOLD = 0.48

# MOCK_LLM is the default and the only graded mode
MOCK_LLM = os.getenv("USE_LIVE_LLM") != "1"

SEED = 42
