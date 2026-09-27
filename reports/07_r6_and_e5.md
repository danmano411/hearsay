# R6 and E5

**E5 = R4ft + R6 + R1 (LR fusion)** was the final submission until E7 replaced it ([report 08](08_r7_rawboost_and_final.md)). It was chosen by the rule written before any R6 result existed
([development log](../docs/00_development_log.md), stage 10). File: `submission/HearsayScoreKey4GeorgiaMellon_FINAL.tsv`. It is written in the scorer's
direction (**higher = real**), with 1,671 rows in template order, all distinct, and `scripts/score.py validate`
returns OK. Its rank correlation with the already-scored R5 file is 0.989. The organizers keep the better of the
initial (R5: minDCF 0.0584, EER 2.5 %) and this final.

## R6: what was trained
- **Recipe:** the exact R4ft recipe (`reports/04_r4ft_xlsr.md`), with seed 1.
- **Extra training data:** 6,155 clips from ASVspoof 2021 DF (two disjoint samples), CD-ADD and DECRO-en
  (`scripts/ingest_bench.py`, `scripts/bench_to_manifest.py`). The frozen evaluation sets and all 185,915 existing
  rows are unchanged.

Validation curve (val_testlike combined, centre window; R4ft at the same step in brackets):

| step (epoch) | R6 | R4ft |
|---|---|---|
| 300 | 0.143 | 0.193 |
| 1,024 (1) | 0.069 | 0.065 |
| 2,048 (2) | 0.036 | 0.041 |
| 3,072 (3) | 0.035 | 0.035 |
| **4,096 (4)** | **0.0267 (best, kept)** | 0.030 |
| 5,120 (5) | 0.0295 | 0.019 |
| 6,144 (6) | 0.0290 | 0.0166 (R4ft's best) |

- **Inference mode:** chosen on val_testlike only. win3 scored 0.0206 against the centre window's 0.0267.
- **Checkpoint:** `best.pth` sha256 `db5053dd…`, step 4,096.

### Deviation from the plan (recorded as it happened)
The plan says to train until early stopping. That didn't happen, for three reasons:
- The GPU was shared with another project's training runs that evening.
- The GPU laptop's low-memory safeguard stopped R6 twice (19:32 and 20:20), both times at step
  ~7,150, while early stopping stood at 2 of 4 evaluations without improvement.
- The owner chose to score the best checkpoint (step 4,096) without the last two evaluations.

For those evaluations to change the checkpoint, one of them would have needed ≤ 0.0247. The two evaluations after the
best were 0.0295 and 0.0290. Earlier in the run R6 lost about 150 steps to an external process kill, and paused twice
for the other project's jobs; each time it resumed from its checkpoint.

## The rule, applied
Metric: the organizers' official minDCF (Pspoof 0.3, Cmiss 1, Cfa 4). The guardrails are relative to R5:
val_testlike ≤ R5 + 0.003 (fusions cross-fitted), and DeepVoice ≤ R5 + 0.02. The preference order is E5 > E > R6.

| candidate | val_testlike | DeepVoice | passes |
|---|---|---|---|
| R5 (baseline, already scored) | 0.0088 | 0.150 | n/a |
| R6 alone | 0.0148 | 0.182 | no (both) |
| E: R4ft + R6, equal weights | 0.0096 | 0.172 | no (DeepVoice by 0.002) |
| **E5: R4ft + R6 + R1, LR fusion** | **0.0099** | **0.152** | **yes → selected** |

- **E5 fusion weights** (standardized inputs, fit on val_testlike): R4ft 6.00, R6 3.93, R1 1.69.
- **R1 inside E5** is the refit booster (`reports/06_generalization.md`), used for every split.
- **Sigmoid temperature:** 7, sized from the validation margins (max 35.3) like the R5 file.
- Selection table: `reports/07_r6_selection.csv`; code: `scripts/r6_select.py`.

## Report-only, scored once after the choice
`test_internal_testlike`, official settings:

| model | minDCF | EER |
|---|---|---|
| R5 | 0.0138 | 0.54 % |
| **E5 (final)** | **0.0154** | **0.69 %** |
| R4ft | 0.0192 | 0.76 % |
| R6 | 0.0236 | 0.89 % |

E5 is slightly behind R5 on this in-domain set. Two things to weigh with that:
- R5's number is optimistic, because this set informed the switch to R5 ([development log](../docs/00_development_log.md), stage 7).
- R6 gave up some in-domain accuracy for more varied training data.

The rule was written knowing that validation cannot see gains on unfamiliar audio, which is what the HGT test
measured (0.0584 against our 0.0138). Whether E5 beats 0.0584 is known only when the organizers score the final.

## Why it can't lose
Best-of-two: if E5 scores worse than 0.0584 on the HGT test, R5's score still counts.
