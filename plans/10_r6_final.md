# Plan 10: R6, one shot at a better final submission

Written 2026-09-26 ~15:45 EDT, **before any R6 training**. Everything in the "Rule" section is fixed now.

## Why
The official HGT score is **minDCF 0.0584 / EER 2.5 %** (Pspoof 0.3, Cfa 4, higher = real), against 0.0138 on our
own held-out set. The gap says the test differs from our data. The organizers keep the **best of the initial and
final** submission, so a new final can only help the score. Our own rule forbids choosing anything from the HGT audio
or from the organizers' feedback, so the choice below uses only our own data.

## What changes
- **More diverse training data** (the lever our analysis keeps pointing to): the public benchmark sets from plan 09
  that R6 may train on: **ASVspoof 2021 DF** (two disjoint samples: `asvspoof2021_df` + `asvspoof2021_df_b`; 100+
  vocoders, m4a/ogg codecs), **CD-ADD** (zero-shot TTS 2023-24, TED reals), **DECRO-en** (commercial TTS).
  They enter the manifest as ordinary sources (`manifests/bench_*.parquet`), and `make_splits.py` puts about 80 %
  of each in train. The frozen eval subsets don't change (the script refuses otherwise).
- **Held out for selection:** **DeepVoice** (voice conversion, the documented weak spot) is never trained on. The
  `keyguard_*` sets are **never** used: they are derived from `test_internal` audio.
- **Model:** the exact R4ft recipe (config.json args of `R4ft_xlsr_light`), **seed 1**, name `R6_xlsr_light`, same
  4 h budget and early stopping on `val_testlike`, window choice on `val_testlike`. GPU node.

## Candidates
| id | what | how it's formed |
|---|---|---|
| R5 | current submission (R4ft + R1 LR fusion) | frozen |
| R6 | the new model alone | as trained |
| E | R4ft + R6 ensemble | **fixed equal weights** on margins standardized by their `val_testlike` mean/std; no fitting |
| E5 | E + R1 | `scripts/fuse.py` LR fusion, group cross-fitted on `val_testlike` (as R5 was) |

## Rule (pre-registered)
- **Metric:** the organizers' official minDCF (Pspoof 0.3, Cmiss 1, Cfa 4, ASVspoof direction).
- **Selection score** = mean of the official minDCF on **`val_testlike`** and on **DeepVoice** (unseen corpus and
  generator family). Fusion candidates use their cross-fitted `val_testlike` scores.
- **Switch** the final submission from R5 to the best candidate only if its selection score is **≥ 10 % lower** than
  R5's. Otherwise the final stays R5. Either way it is written with `make_submission.py --higher-is-real`.
- `test_internal_testlike` and the other benchmark sets are scored **once, after** the choice, report-only.
- No HGT feedback is used at any step. The HGT audio is scored only by the chosen model.

## Division of work
- **cpu:** ingest DF batch B; write the bench manifests + `make_splits.py`; publish the manifest on the hub; after
  training, score DeepVoice, compute the selection table, apply the rule, build the final file, write
  `reports/r6_final.md`.
- **gpu:** pull the code, manifest and `data/bench/` (DeepVoice is already there from plan 09); train R6; `score`
  with `--hgt`; score DeepVoice with `bench_score.py r4ft --name R6_xlsr_light --device cuda --sets deepvoice`; upload
  the parquets with `hubctl put`. Inference-only rules as always.

## Time box
Deadline Sep 27, 12:00 EDT. If R6 has not finished by 08:00 EDT, the final stays R5.
