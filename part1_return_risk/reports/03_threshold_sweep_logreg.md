# 03 — Threshold Sweep (Logistic Regression)

- **[PASS] Chosen t* recall >= default recall + 15pp** — recall 0.5 -> t*: 0.5788 -> 0.7582 (+17.9 pp)

Full 41-row sweep (threshold 0.10 -> 0.90, step 0.02): `03_threshold_sweep_logreg.csv`
Plot: `03_threshold_sweep_logreg.png`

## F1-maximising point from the sweep

- threshold = **0.44**, precision = 0.2801, recall = 0.7582, f1 = 0.4091
- recall gap vs default (0.5): +17.9 pp

The F1-maximising threshold already clears the +15pp recall gate, so it is adopted directly as the **chosen threshold**.

## Chosen threshold

- **Chosen t\* = 0.44** (mode: `f1_max`)
- precision = 0.2801, recall = 0.7582, f1 = 0.4091
- recall 0.5 -> t\*: 0.5788 -> 0.7582 (+17.9 pp)
- precision 0.5 -> t\*: 0.2964 -> 0.2801 (-1.6 pp)

## Trade-off paragraph

Lowering the threshold flags more orders as likely returns, which trades **precision for recall**. The expensive error here is the **false negative**: a return we failed to flag, so no proactive intervention (packaging check, refund pre-authorization, courier instructions) happens, and the reverse-pickup cost lands anyway. The cheap error is the **false positive**: support time spent double-checking an order that was never going to be returned. Because a missed return is far costlier than an unnecessary check, moving the threshold down to trade some precision for a meaningful recall gain is the right business call — 0.5 was never a business decision, it is just the modelling default.
