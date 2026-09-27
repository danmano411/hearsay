# R7: RawBoost XLS-R and the final submission

**Final submission: E7** = LR fusion of R4ft + R6 + R7a + R1, chosen by the Stage 11 rule
([development log](../docs/00_development_log.md)), file `HearsayScoreKey4GeorgiaMellon_R7_FINAL.tsv`
(`--higher-is-real`, sigmoid T = 7, 1,671 distinct scores, validated).

## R7a

XLS-R-300M (12 blocks) + light back end, as R4ft/R6, plus RawBoost channel augmentation (Tak et al., ICASSP 2022) on
60 % of training clips, seed 2. Training ran in three legs: 1.25 h to step 5,164; an owner-approved extension killed
by the laptop's memory safeguard at step ~6,250 (checkpoint at 6,144 intact); an owner-approved resume with 2 loader
workers (same batches). Early stop at step 14,336; best checkpoint step 10,240. R7b (WavLM-Large) was cancelled so the
GPU time went to one fully trained model.

| | val_testlike combined (training eval) | test_internal_testlike combined (report-only) |
|---|---|---|
| R4ft | 0.0121 | 0.0282 |
| R7a @6,144 (interim fallback) | 0.0127 | 0.0258 |
| **R7a @10,240** | **0.0119** | **0.0207** |

## Selection (official settings; val cross-fitted)

| candidate | val_testlike | DeepVoice | passes |
|---|---|---|---|
| E5 (previous final) | 0.0099 | 0.1515 | — |
| **E7** (R4ft + R6 + R7a + R1) | **0.0088** | **0.1330** | ✅ picked |
| E7s (without R1) | 0.0083 | 0.1461 | ✅ (beats E7 by < 0.002) |

Full table: [`08_r7_selection.csv`](08_r7_selection.csv). Fusion weights (E7): R4ft 3.93, R6 2.07, R7a 4.74, R1 1.47.
R7a is the largest weight.

**Disclosed:** with the step-6,144 checkpoint E7 measured 0.0084 / 0.1262, slightly better than with the final
checkpoint. The rule uses each model's best checkpoint by its own training recipe, so the final one is used; the two
files differ on 4 of 1,671 clips at 0.5 (Spearman 0.993). E7 vs E5: 10 clips switch sides, Spearman 0.990.
Nothing was chosen from the HGT audio or the organizers' feedback.

## Report-only, scored once after the choice
`test_internal_testlike`, official settings (Pspoof 0.3, Cfa 4):

| model | minDCF | EER |
|---|---|---|
| **E7 (final)** | **0.0132** | **0.54 %** |
| R5 | 0.0138 | 0.54 % |
| R7a | 0.0149 | 0.61 % |
| E5 | 0.0154 | 0.69 % |
| R4ft | 0.0192 | 0.76 % |
| R6 | 0.0236 | 0.89 % |

E7 is the best system on this set too, and R7a is the best single model. The HGT test has so far been about 4× harder
than this set (R5: 0.0138 here, 0.0584 official), so expect E7's official score to be well above 0.0132.
