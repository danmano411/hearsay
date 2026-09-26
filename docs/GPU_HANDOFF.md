# GPU handoff — where things stand (paused 2026-09-26, 01:45 EDT)

Work ran CPU-only on a laptop (Ryzen 7840U, no CUDA) and was paused at a clean point so it can move to a GPU machine.
All code, docs and results are on `main`. **Data, features, scores and weights are not in git** (see "Data" below).

## Results so far

Headline set `test_internal_testlike` (9,747 clips, 70/30 real/fake, many unseen generators; never tuned on).
minDCF: 0 = perfect, 1 = trivial. Full table: [`reports/leaderboard.md`](../reports/leaderboard.md).

| Rung | Model | official | brief | **combined** | status |
|---|---|---|---|---|---|
| R0 | trivial cues, raw audio (LightGBM) | 0.634 | 0.510 | 0.572 | done — shows the dataset leaks via trivial cues |
| R0 | trivial cues after `prep()` | 0.939 | 0.972 | 0.955 | done — `prep()` neutralizes them |
| R1 | LightGBM, 228 LFCC/MFCC/spectral + bio feats | 0.293 | 0.316 | **0.305** | done, HGT scored (`data/scores/R1_lgbm_all__hgt.parquet`) |
| R1 | LightGBM, bio feats only | 0.703 | 0.723 | 0.713 | done |
| R3 | AASIST fine-tune (CPU) | — | — | 0.692 (val_testlike, step 200, still falling) | **paused** |
| R3 | AASIST zero-shot | — | — | — | **paused** mid-scoring |
| R4 | WavLM-base+ layer-3 mean + LR (partial: 14.8k clips) | — | — | 0.389 (val_testlike) | **paused** |
| R4 | XLS-R-300M | — | — | — | not started |

## Remaining work, in order

1. **Add GPU support** (none of the scripts take a device yet): add `--device` (default `cuda` if available) to
   `scripts/extract_ssl.py` + `src/hearsay/features/ssl.py`, `scripts/run_aasist.py`, `scripts/finetune_aasist.py`
   (+ `src/hearsay/models/aasist_wrap.py`), and `scripts/run_sim.py`; move model and batches `.to(device)`; use
   `torch.autocast` for SSL extraction. Raise batch sizes.
2. **Finish the sim** (`scripts/run_sim.py`, resumable): vits_ljs ~1,220/1,500, speecht5 ~420/420, mms ~545/1,000; then
   rebuild `data/processed/manifests/sim.parquet` and rerun `scripts/make_splits.py` (existing rows keep their split).
3. **R4 SSL** (biggest expected gain): finish WavLM extraction (resumes from the last completed shard), then XLS-R-300M
   (`--model xls_r_300m`, drop `--max-blocks` on GPU), with a larger train subset (`--train-n 40000+`). Then
   `scripts/train_ssl_heads.py sweep|heads|plot --model <m> --report --hgt`.
4. **R3 AASIST**: finish zero-shot scoring (`run_aasist.py --name R3_aasist_zeroshot_prep`, and `--raw` for comparison);
   resume fine-tuning from `data/models/r3_aasist_ft_last.pth` with more data (`--per-class 30000`) and hours.
5. **GPU-only rungs** now feasible: fine-tune the SSL front-end end-to-end (e.g. XLS-R + AASIST back-end, the
   ASVspoof5 top-system recipe) with `prep()` + `augment()` on random 4 s crops; R2 CNN (LFCC/log-mel ResNet).
6. **R5 fusion**: logistic regression on cached `data/scores/*.parquet`, fit on `val_testlike`, reported on
   `test_internal_testlike`; default value for any unscored HGT clip = **0.3** (`docs/scoring.md` §4).
7. **Submission**: write `submission/<team>_scores.tsv` in template order; validate with `python scripts/score.py validate`.
8. **Stop criterion** (owner decision): `test_internal_testlike` combined minDCF ≤ 0.05, or 3 consecutive rungs
   improving < 0.005.

## Rules to keep (owner decisions)

- HGT test audio is **inference only** — no fitting, normalization, calibration or selection on it.
- Every clip (train and test) goes through `hearsay.preprocess.prep()` (7 kHz low-pass, trim, DC, RMS). Reason: the
  organizers' real LJ clips keep 7–8 kHz energy that properly resampled clips lack (`reports/data_audit.md`).
- Report both minDCF readings (`docs/scoring.md` §2) until the organizers say which error costs 4×.

## Data (not in git)

Everything lives under `data/` (gitignored). To set up a GPU machine, either copy the whole `data/` folder (fastest), or rebuild:

1. Challenge data from the organizers' Drive folder → `data/raw/{diffssd,lj_real,hgt_test}` and the ASVspoof5 repo →
   `third_party/asvspoof5` (layout in `README.md`).
2. `python scripts/clean_given.py` → DiffSSD + LJ canonical clips and manifests.
3. External corpora: `scripts/ingest_*.py` (see `docs/external_data.md`; ~27 GB, several hours of downloading).
4. `python scripts/run_sim.py`, then `python scripts/make_splits.py`.

Paused artifacts worth copying so nothing is redone: `data/features/wavlm_base_plus/` (~31 shards + index),
`data/features/classic*.parquet`, `data/scores/`, `data/models/r3_aasist_ft*.pth`, `data/processed/sim/`.
