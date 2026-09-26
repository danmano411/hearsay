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
- **Held out as a guardrail:** **DeepVoice** (voice conversion, fully unseen corpus) is never trained on. It
  checks non-inferiority on unfamiliar audio; improving it is not a goal (see Rule). The
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

## Rule (pre-registered; revised 2026-09-26 ~16:10, before any R6 training)
**Goal: the lowest official minDCF on the HGT test.** Nothing else (not DeepVoice, not our headline) is a goal.

**Why the final is never R5 again.** The organizers keep the best of the initial and final submission, and R5's
flipped file (0.0584) is already scored. Resubmitting R5 adds nothing. The final should therefore be the candidate
most likely to *beat* 0.0584. A worse final costs nothing, because 0.0584 still counts.

**Why selection can't just be "lowest validation score".** `val_testlike` is in-domain and nearly saturated
(R5 0.0088 official), while the HGT gap (0.014 held-out → 0.058 official) says the test is out-of-domain for us.
Validation cannot see out-of-domain gains. The candidates are therefore ranked by a **prior fixed now**: more
training diversity and more independently trained models generalize better to unseen data (the most consistent
result in anti-spoofing and ensembling). Validation and DeepVoice act as **guardrails** that catch a broken or
degraded run; they don't drive the choice.

1. **Metric everywhere:** the organizers' official minDCF (Pspoof 0.3, Cmiss 1, Cfa 4, ASVspoof direction).
2. **Guardrails** (a candidate must pass both; the baseline is R5, with fusions on cross-fitted `val_testlike`
   scores):
   - `val_testlike`: at most **R5 + 0.003** (a noise allowance on a set where R5 sits at 0.0088; catches a worse
     model);
   - DeepVoice (unseen corpus, never trained on): at most **R5 + 0.02**. This is a non-inferiority check on
     unfamiliar audio, not a target.
3. **Preference order among the candidates that pass: E5 > E > R6.** E5 = R4ft + R6 + R1 (three models, two
   independent fine-tunes, most diversity); E = R4ft + R6 at fixed equal weights; R6 alone last. One exception,
   fixed now: if E5's cross-fitted `val_testlike` is worse than E's by more than 0.002 (the old fusion rule), take
   E over E5.
4. **If no candidate passes:** the final is R5 (already counted, harmless) and the failure is reported.
5. The final is written with `make_submission.py --higher-is-real`. `test_internal_testlike` and the remaining
   benchmark sets are scored once, after the choice, report-only. No HGT feedback or HGT score distribution is used at
   any step; the HGT audio is only ever scored by the models.

## Division of work
- **cpu:** ingest DF batch B; write the bench manifests + `make_splits.py`; publish the manifest on the hub; after
  training, score DeepVoice, compute the selection table, apply the rule, build the final file, write
  `reports/r6_final.md`.
- **gpu:** pull the code, manifest and `data/bench/` (DeepVoice is already there from plan 09); train R6; `score`
  with `--hgt`; score DeepVoice with `bench_score.py r4ft --name R6_xlsr_light --device cuda --sets deepvoice`; upload
  the parquets with `hubctl put`. Inference-only rules as always.

## Time box
Deadline Sep 27, 12:00 EDT. If R6 has not finished by 08:00 EDT, the final stays R5.
