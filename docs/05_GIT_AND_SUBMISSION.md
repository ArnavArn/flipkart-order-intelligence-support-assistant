# Phase 4–5 — README, Git History, and Submission

---

## The README is graded

It is the one document tying all three Parts together. Required sections, in this order:

1. **Title + one-paragraph overview** — the connected system, with the ASCII flow diagram from
   `PLAN.md`.
2. **Setup** — Python 3.12 venv, `pip install -r requirements.txt`, one-time download note
   (Fashion-MNIST, ResNet-18 weights, MiniLM — all free, no keys).
3. **How to regenerate Part 1's dataset and model** — the exact commands.
4. **How to run Part 2's training/evaluation** — exact commands + runtime expectation.
5. **How to run Part 3's agent in default mock mode** — exact commands, plus the explicit
   statement that `USE_LIVE_LLM` is unset by default and no API key is needed anywhere.
6. **Results summary tables** — headline numbers per Part, each with a link to the generated
   report that produced it. Copy numbers *from the reports*; never type one from memory.
7. **The `t*_rf` sentence** — value, cut points, and that it was computed on the RF's own
   `predict_proba`.
8. **The annotated system prompt** — reproduced with the 4S + role-prompting annotations visible.
9. **At least one full example transcript inlined** (use transcript 05, the multi-turn one — it
   shows the most) plus a linked index of all 9 in `transcripts/`.
10. **Repo map** — condensed from `docs/01_REPO_MAP.md`.
11. **Optional live-LLM note** — marked optional, and a statement that removing it leaves every
    acceptance criterion satisfied via MOCK_LLM.

Write it **last**, after all reports exist, so every number is copied from a real file.

---

## Git history requirement

> "The repository's overall commit history must show at least one feature branch created,
> committed to at least twice, and merged into main."

Checked **once across the whole repo**, not per Part. This is a real criterion — do not push a
single flat commit.

### The exact sequence

```bash
cd /Users/arnav.garg/Developer/projects/capstone
git init -b main
git add .gitignore requirements.txt PLAN.md docs/
git commit -m "chore: project scaffolding, environment pins, and build plan"

# ── Part 1 ──
git add generate_orders.py orders_dataset.csv part1_return_risk/ models/return_risk_model.pkl models/return_risk_meta.json
git commit -m "feat(part1): seeded order dataset, leak-free pipeline, tuned RF, t*_rf artifact"

# ── Part 2 ──
git add part2_image_classifier/ models/product_classifier.pt data/sample_images/
git commit -m "feat(part2): ResNet-18 transfer learning classifier, evaluation, sample PNG export"

# ── Part 3 on a FEATURE BRANCH (this is the graded bit) ──
git checkout -b feature/part3-support-agent

git add part3_agent/kb/ part3_agent/chunking.py part3_agent/index_build.py part3_agent/retriever.py part3_agent/index/
git commit -m "feat(part3): policy knowledge base, sentence-wise chunking, FAISS index"        # commit 1

git add part3_agent/tools/ part3_agent/guardrails.py part3_agent/prompts.py part3_agent/mock_llm.py part3_agent/state.py part3_agent/graph.py part3_agent/run_agent.py
git commit -m "feat(part3): LangGraph agent, real Part 1/2 tool calls, guardrails, MOCK_LLM"    # commit 2

git add part3_agent/run_transcripts.py part3_agent/eval_retrieval.py transcripts/
git commit -m "feat(part3): 9 mock-mode transcripts and document-level retrieval evaluation"    # commit 3

# ── merge back with a REAL merge commit ──
git checkout main
git merge --no-ff feature/part3-support-agent -m "merge: Flipkart support agent (Part 3) into main"

# ── README last ──
git add README.md
git commit -m "docs: root README tying Parts 1-3 into one support-assistant demo"
```

`--no-ff` is **mandatory**. Without it Git fast-forwards and there is no merge commit in the
graph — the criterion becomes unverifiable.

### Verify before pushing

```bash
git log --graph --all --oneline --decorate
```

You must be able to see: a branch diverging from `main`, three commits on it, and a merge commit
joining it back. If the graph is a straight line, the criterion is not met — redo the merge.

---

## Pushing to GitHub

**This must go to your personal GitHub account (github.com/ArnavArn), never to a company
account or org.** This machine's local git identity for this repo is already set to
`ArnavArn <gargarnav2982@gmail.com>` (local override, does not touch your global/company git
config). Claude Code will not run `git push` or `gh repo create` on your behalf — you do this
step yourself, by design.

`gh` CLI is not installed on this machine, so create the repo through the web UI:

1. Go to **github.com/new** while logged into **github.com/ArnavArn** (your personal account —
   double check the account switcher in the top-right before creating).
2. Repository name: `flipkart-order-intelligence-support-assistant`
3. Visibility: **Public** ← required; a private repo cannot be graded.
4. **Do not** initialise with a README, .gitignore, or licence — the local repo already has them
   and an initialised remote forces an awkward merge.
5. Create, then run these yourself from the repo root:

```bash
git remote add origin https://github.com/ArnavArn/flipkart-order-intelligence-support-assistant.git
git push -u origin main
git push origin feature/part3-support-agent      # push the branch too, so the history is visible
```

If `git push` prompts for credentials, authenticate as your **personal** GitHub account — not
whatever account/token your work VS Code or company Claude Code setup may have cached.

### Post-push verification (do all five)

- [ ] Open the repo in a logged-out browser window → it loads (proves it is public).
- [ ] **Insights → Network** (or the commits graph) shows the branch and merge.
- [ ] `models/return_risk_model.pkl` and `models/product_classifier.pt` are present and are real
      files, not Git LFS pointers.
- [ ] `data/sample_images/` shows 10 PNGs that render as thumbnails in the GitHub UI.
- [ ] README renders correctly — tables, the ASCII diagram, and the inlined transcript.

Then submit **only** that repository URL. No files, no PDFs, no screenshots.

---

## Fresh-clone smoke test (strongly recommended)

The single highest-value check before submitting — it catches every absolute path, every
uncommitted artifact, and every undeclared dependency:

```bash
cd /tmp && rm -rf smoke && git clone <your-repo-url> smoke && cd smoke
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m part3_agent.run_transcripts     # must work off committed artifacts alone
python -m part3_agent.eval_retrieval
```

Part 3 must run **without** re-running Parts 1 or 2, because `return_risk_model.pkl`,
`return_risk_meta.json`, `product_classifier.pt`, the sample PNGs, and the FAISS index are all
committed. If this fails, something that should be committed is gitignored.
