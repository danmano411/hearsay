# Phase 1 — Data cleaning + canonical dataset

## Inputs (`data/raw/`)
- `diffssd/` — 70,000 synthetic clips, 10 generators (wav + ElevenLabs mp3), mixed sample rates.
- `lj_real/` — 242 real LJSpeech clips, already 16 kHz.
- `hgt_test/` — 1,671 unlabeled test clips + `HGT_Hearsay_score_template.csv` (TSV). **Read-only, inference only.**

## Steps
1. **Audit** every file: decodable?, sample rate, channels, bit depth, duration, peak, RMS, % clipped samples,
   leading/trailing silence length (|x| < 1e-4), DC offset, effective bandwidth (95% spectral rolloff).
   Write `reports/data_audit.md` with per-source tables. For `hgt_test` run a *format* check
   (format checks only: sample rate, channels, duration range — no level/spectral stats, per the inference-only rule).
2. **Clean**: drop undecodable/empty files and exact duplicates (hash of PCM); mono-mix; resample to 16 kHz
   (`soxr`-quality via librosa/torchaudio); write PCM16 WAV to `data/processed/<source>/<generator>/...`.
   Do **not** trim or loudness-normalize at storage time (keeps cues auditable; normalization happens at model input).
3. **Filter**: keep clips ≥ 3.0 s (test clips are > 3 s); record dropped counts.
4. **Manifest** `data/processed/manifest.parquet`: `path, label (bonafide|spoof), source, generator, speaker,
   text_id, duration, sr_orig, codec_orig, split`.
5. **Splits** (grouped, no leakage of speaker+text across splits):
   - `train` / `val` / `test_internal` by group (generator × speaker × sentence id).
   - LOGO folds: each generator held out once.
   - `val_testlike`: 70% bonafide / 30% spoof, spoof from held-out generators where possible.
6. **Shortcut audit**: train a depth-3 tree on the trivial audit columns only (duration, silence, peak, rolloff).
   If it beats minDCF 0.5 on val, those cues are leaking → neutralize (random crop, silence trim both classes, gain aug).

## Check
`scripts/check_manifest.py` asserts: every path exists, all 16 kHz mono, no group in two splits, class counts printed.
