# 05 — Confusion Analysis

The pairs below were found **programmatically** from the real confusion matrix in report 04 (largest off-diagonal cells, deduplicated by unordered class pair) — they were not guessed in advance.

## Largest confused pairs (from the actual matrix)

| True class | Predicted class | Count (true->pred) | Count (pred->true) |
|---|---|---|---|
| Shirt | T-shirt/top | 155 | 52 |
| Shirt | Coat | 82 | 45 |
| Shirt | Pullover | 71 | 35 |

## Why these pairs are visually plausible at 28x28

### Shirt <-> T-shirt/top

Shirt and T-shirt/top are both short-sleeved torso garments with the same rectangular body outline and the same shoulder-to-hem aspect ratio. At 28x28 grayscale the only cues that would distinguish them — a collar edge or a button placket — are a handful of pixels wide in the original image; they do not survive the downsampling to 28x28, nor does upsampling to 224x224 recover detail that was never captured. The model is left with two near-identical silhouettes and has to guess from faint shading cues.

### Shirt <-> Coat

Shirt sits between the sleeved-torso classes and picks up confusion from both directions. Against Coat, the difference is again the coat's open-front seam versus the shirt's closed body outline — a cue that is only a pixel or two wide and is easily lost at 28x28, so errors between the two are visually plausible rather than random.

### Shirt <-> Pullover

Shirt and Pullover share the same long-torso, set-in-sleeve silhouette; a pullover has no front opening at all and a shirt's placket is a thin, easily-lost line at this resolution, so the two classes overlap heavily in pixel space even though they are visually distinct garments at full resolution.

As the trainer's line goes: an error is not a wrong answer — it is the model telling you where the visual signal genuinely is ambiguous.
