# HEARSAY — notes for Claude

HackGT 2026 NSA challenge: score 1,671 test clips 0.0 (real) → 1.0 (synthetic); graded 60% minDCF, 20% creativity,
20% GitHub docs. **Start by reading `docs/GPU_HANDOFF.md`**: current results, what's paused, and the ordered to-do list.
Two machines work on this repo at once: **read `plans/07_two_node_protocol.md`** for roles, channels and rules.

## Owner decisions (don't re-ask)
- **Test audio (`data/raw/hgt_test`) is inference only**: no fitting, normalization stats, calibration, thresholds,
  pseudo-labels, or model selection from it or from its score distribution.
- **Stop** when `test_internal_testlike` combined minDCF ≤ 0.05, or after 3 consecutive rungs improve < 0.005.
- Report **both** minDCF readings (`official_as_written`, `brief_as_written`) + `combined`; select models on
  `val_testlike` combined. Why two readings: `docs/scoring.md` §2 (the organizers' scorer and the brief disagree).
- TTS sim uses open-source models only (no ElevenLabs API).

## Invariants
- Every clip, train **and** test, goes through `hearsay.preprocess.prep()` (7 kHz low-pass, trim, DC, RMS). The
  organizers' real clips keep 7–8 kHz energy that resampled clips lack; removing the band stops it deciding anything.
- Train only on `split == "train"`; tune on `val`/`val_testlike`; `test_internal_testlike` is the headline, never tuned on.
- Report every model through `hearsay.evaluate.report(name, df[path, score])` (higher = more fake); score HGT with
  final models into `data/scores/<name>__hgt.parquet` (`filename`, `score`).
- Submission: template order, `filename<TAB>cm-score`; validate with `python scripts/score.py validate`. Default
  for any unscored clip = 0.3, never the template's 0.006 (`docs/scoring.md` §4).

## Conventions
- Run with `PYTHONPATH=src`; tests `python -m pytest -q tests`. Torch scripts take `--device auto|cpu|cuda`.
- `data/`, `third_party/`, venvs are gitignored: never commit audio, parquet, npy, or weights.
- Write a plan in `plans/` before a new phase; branch + PR per phase; append results to `reports/leaderboard.md`;
  write findings for judges in `reports/`/`docs/`. Commits end with a `Co-Authored-By: Claude ...` line.
