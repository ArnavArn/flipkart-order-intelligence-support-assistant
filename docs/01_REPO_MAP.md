# Repo Map — every file and what it is for

Legend: **[C]** committed artifact · **[G]** generated, gitignored · **[R]** generated report,
committed

```
capstone/
├── README.md                              [C] THE deliverable doc. Written last (Phase 4).
├── PLAN.md                                [C] master build plan
├── requirements.txt                       [C]
├── .gitignore                             [C]
│
├── generate_orders.py                     [C] Part 1 Task 1 — VERBATIM from the brief. Never edit.
├── orders_dataset.csv                     [C] 6000 × 13, produced by the above
│
├── models/
│   ├── return_risk_model.pkl              [C] tuned RF Pipeline (preproc + model), joblib
│   ├── return_risk_meta.json              [C] t_star_rf, bucket cut points, feature order,
│   │                                          metric snapshot. Part 3 reads t*_rf from HERE.
│   └── product_classifier.pt              [C] torch state dict + config
│
├── data/
│   ├── fashion_mnist/                     [G] raw IDX, auto-downloaded
│   └── sample_images/                     [C] ≥5 (we do 10) real test-split PNGs
│       ├── 00_tshirt_top.png ... 09_ankle_boot.png
│       └── labels.json                    [C] filename → true label + test index
│
├── part1_return_risk/
│   ├── __init__.py
│   ├── config.py                          paths, SEED=42, column lists, split params
│   ├── data_checks.py                     Task 2 — row count, return rate, missingness,
│   │                                        MAR evidence, breakdown tables
│   ├── pipeline.py                        Task 3 — build_preprocessor() → ColumnTransformer
│   ├── models.py                          Task 4/5/6 — dummy, logreg, RF grid
│   ├── thresholds.py                      shared sweep_threshold() used by BOTH Task 5 and
│   │                                        Task 9 (same procedure, different model)
│   ├── explain.py                         Task 7 — impurity vs permutation importance
│   ├── subgroups.py                       Task 8 — recall/precision by category & payment
│   ├── train.py                           ENTRY POINT — runs Tasks 2–9 end to end,
│   │                                        writes all reports, saves both artifacts
│   └── reports/                           [R]
│       ├── 01_data_checks.md
│       ├── 02_baseline_and_logreg.md
│       ├── 03_threshold_sweep_logreg.md   + .csv
│       ├── 04_random_forest_gridsearch.md
│       ├── 05_feature_importance.md
│       ├── 06_subgroup_analysis.md
│       └── 07_final_artifact.md           t*_rf, cut points, saved-model verification
│
├── part2_image_classifier/
│   ├── __init__.py
│   ├── config.py                          paths, SEED=42, INPUT_SIZE=224, class names,
│   │                                        batch size / lr / epochs
│   ├── data.py                            download + stratified 55k/5k/10k splits + transforms
│   ├── features.py                        frozen-backbone feature extraction → cache/*.npy
│   ├── train.py                           ENTRY POINT — head training, then conditional
│   │                                        fine-tune if val < 80%
│   ├── evaluate.py                        test accuracy, 10×10 confusion matrix,
│   │                                        per-class precision/recall
│   ├── export_samples.py                  Task 8 — write real PNGs to data/sample_images/
│   ├── model_io.py                        ★ load_model() + predict_image(path)
│   │                                        THE documented snippet. Part 3 imports this.
│   ├── cache/                             [G] resnet18 features .npy (~150 MB)
│   └── reports/                           [R]
│       ├── 01_splits_and_setup.md
│       ├── 02_training_log.md             feature-extraction vs fine-tune val accuracy
│       ├── 03_test_evaluation.md          accuracy + per-class P/R
│       ├── 04_confusion_matrix.md         the 10×10 table (+ .csv)
│       └── 05_confusion_analysis.md       ≥2 pairs, one paragraph each
│
├── part3_agent/
│   ├── __init__.py
│   ├── config.py                          MOCK_LLM default, SIM_THRESHOLD, TOP_K=3, paths
│   ├── kb/
│   │   ├── documents/POL-01.md … POL-15.md   [C] 15 policy docs, 2–4 sentences each
│   │   └── eval_queries.json              [C] 6 queries → relevant doc_ids (answer key)
│   ├── chunking.py                        sentence-wise split + chunk→parent-doc map
│   ├── index_build.py                     ENTRY POINT — embed + FAISS, persists index
│   ├── index/                             [C] faiss.index + chunks.json (so graders can
│   │                                        run without rebuilding)
│   ├── retriever.py                       search(query, k) → chunks + cosine scores
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── return_risk_tool.py            check_return_risk(order_features: dict) -> dict
│   │   └── image_tool.py                  classify_product_image(image_path: str) -> dict
│   ├── guardrails.py                      input injection filter + output groundedness check
│   ├── prompts.py                         4S-annotated system prompt + 2 few-shot examples
│   ├── mock_llm.py                        deterministic template composer → structured JSON
│   ├── state.py                           AgentState TypedDict
│   ├── graph.py                           ★ the LangGraph — 6 nodes, 2 conditional edges
│   ├── run_agent.py                       interactive/one-shot CLI
│   ├── run_transcripts.py                 ENTRY POINT — regenerates all 9 transcripts
│   └── eval_retrieval.py                  ENTRY POINT — P@3 / R@3 with per-query arithmetic
│
├── transcripts/                           [R] all generated by run_transcripts.py
│   ├── 01_policy_apparel_return_window.md      (a) policy via RAG #1
│   ├── 02_policy_cod_refund_timeline.md        (a) policy via RAG #2
│   ├── 03_return_risk_tool_call.md             (b) check_return_risk
│   ├── 04_image_classification_tool_call.md    (c) classify_product_image
│   ├── 05_multiturn_state_carried.md           (d) state carried across turns
│   ├── 06_fresh_conversation_state_absent.md   (d) same question, new thread → no state
│   ├── 07_prompt_injection_blocked.md          (e) input guardrail
│   ├── 08_ungrounded_refusal.md                (f) output guardrail + printed sim score
│   ├── 09_intent_routing_fewshot.md            few-shot examples visibly driving routing
│   └── retrieval_eval.md                       P@3 / R@3, per-query arithmetic
│
└── docs/                                  [C] these planning documents
```

## The three entry points a grader runs

```bash
python generate_orders.py                    # Part 1 data
python -m part1_return_risk.train            # Part 1 model + all reports
python -m part2_image_classifier.train       # Part 2 model
python -m part2_image_classifier.export_samples
python -m part3_agent.index_build            # build FAISS index (already committed)
python -m part3_agent.run_transcripts        # regenerate all 9 transcripts
python -m part3_agent.eval_retrieval         # P@3 / R@3
python -m part3_agent.run_agent              # interactive demo
```

These exact commands go in the README.
