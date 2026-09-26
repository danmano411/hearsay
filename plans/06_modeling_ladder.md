# Phase 6 — Modeling ladder (ML → DL → fusion)

Each rung: train on `train`, tune on `val`, report on `val`, LOGO folds, `val_testlike`. Append to `reports/leaderboard.md`.
Stop when `val_testlike` minDCF ≤ 0.05 (both interpretations) or 3 consecutive rungs improve < 0.005.

| Rung | Model | Features | Why |
|---|---|---|---|
| R0 | Constant / trivial-cue tree | duration, silence, level, rolloff | shortcut baseline; must be beaten by a lot |
| R1 | Logistic regression, LightGBM | MFCC/LFCC/CQCC stats + Phase-4 bio features | interpretable classic ML |
| R2 | Small CNN / ResNet (CPU) | log-mel / LFCC 4 s crops, SpecAugment + codec aug | first DL rung |
| R3 | Pretrained ASVspoof5 AASIST (given weights) | raw wave | off-the-shelf anti-spoofing prior (score as feature) |
| R4 | Frozen SSL (wav2vec2-base / WavLM-base+ / XLS-R-300M) layer-wise mean-pool → LR/MLP | SSL embeddings | best known generalization to unseen generators |
| R5 | Fusion (LR on rung scores) + calibration + default strategy | rung scores | final system |

CPU budget: SSL embeddings extracted once, cached to `data/features/<model>.npy` (layer means), capped per source so
extraction stays < ~6 h. Fine-tuning SSL end-to-end is out of scope on CPU (ponytail: revisit if a GPU appears).

Final deliverable: `submission/<team>_scores.tsv` — same rows/order as the template, `filename<TAB>cm-score`, 0 = real, 1 = fake.
