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

## Execution details (added after phases 1-5 landed)

Data: 180,726 clips (56,962 real / 123,764 fake) across 14 sources, 126+ generators.

- **Preprocessing** (`src/hearsay/preprocess.py`, every rung, train *and* test): DC removal → silence trim → **7 kHz low-pass**
  → RMS normalization. The low-pass exists because the organizers' LJ reals keep 7–8 kHz energy that properly
  resampled clips lack; since we can't inspect test audio, we remove the band so it can't decide anything.
  Class-symmetric augmentation (`augment()`: mp3 round trip, noise 15–40 dB SNR, resampler round trip) during training.
- **Crops**: test clips are 3.0–13.6 s (median 3.4 s) → train on random 4 s crops, score full clips (or center 4 s for fixed-size nets).
- **Evaluation** (`src/hearsay/evaluate.py`): `report(name, scores_df)` → leaderboard rows for `val`, `val_testlike`,
  `test_internal_testlike` (headline, never tuned on) in both minDCF readings + combined; caches scores to
  `data/scores/<name>.parquet` for fusion. `per_source()` gives the error breakdown.
- **Model selection metric**: `combined` minDCF on `val_testlike` until organizers clarify the cost reading.
- **HGT test**: each final rung also scores `data/raw/hgt_test/*.wav` (inference only) → `data/scores/<name>__hgt.parquet`.

Stages: **A** (parallel) sim completion · R0/R1 classic ML · R3 AASIST · R4 SSL embeddings + heads →
**B** R2 CNN + best-rung improvements → **C** R5 fusion, default-value strategy, submission, docs.

## Final-submission rule (pre-registered 2026-09-26 06:15, before any R4ft headline score existed)

1. Candidates: every rung with scores on all `val_testlike` + `test_internal_testlike` rows **and** all 1,671 HGT clips,
   plus LR fusions (`scripts/fuse.py`) of R4ft with R1 and/or R3.
2. **Selection metric: `val_testlike` combined minDCF.** Fusions are compared with their *group-cross-fitted*
   `val_testlike` number (in-sample fusion scores would be optimistic).
3. A fusion replaces the best single model only if it beats it on `val_testlike` by ≥ 0.002 (the same margin
   G5 uses for a new best); otherwise the simpler model wins.
4. The chosen candidate's `test_internal_testlike` number is the headline. It is reported, never used to switch the choice.
5. Submission = `scripts/make_submission.py --model <chosen>` (`--sigmoid` for margin outputs), validated with
   `scripts/score.py validate`. Every HGT row must be scored (no defaults).
