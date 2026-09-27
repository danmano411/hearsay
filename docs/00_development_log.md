# Development log

This file replaces the phase plans that guided the project. It records, stage by stage, what question we asked, which rules we fixed before the results were known, and what happened.
Amendments and deviations are recorded where they occurred, with the evidence behind them.
All times are EDT, 25 to 27 September 2026. Numbers are quoted from the linked reports and docs.

## Stage 0: scope and owner decisions

- **When:** 25 Sep, 21:49 (project skeleton and phase plans).
- **Question:** Under what constraints do we score 1,671 test clips from 0 (real) to 1 (synthetic)?
- **Decisions fixed in advance:**

| topic | decision |
|---|---|
| Compute | Local CPU only at the start (Ryzen 7840U, 16 threads, 60 GB RAM). Deep learning limited to frozen pretrained features and light heads. SSL fine-tuning deferred until a GPU was available. |
| Synthetic speech | Open-source local TTS and vocoders only. ElevenLabs studied from public docs, not the API. |
| Stop criterion | Held-out combined minDCF ≤ 0.05, or 3 consecutive model rungs improving by < 0.005. |
| Test audio | Inference only: no fitting, normalization statistics, calibration, thresholds, pseudo-labels or model selection from the HGT clips or their score distribution. |

- **Outcome:** The TTS, stop and test-audio rules held for the whole project. The CPU-only rule was relaxed when a GPU laptop joined on 26 Sep (Stage 6), which reopened SSL fine-tuning.
- **Details:** [docs/01_challenge_and_scoring.md](01_challenge_and_scoring.md).

## Stage 1: data cleaning and audit

- **When:** 25 Sep, 22:07 (audit, cleaning, splits); split fixes at 22:24.
- **Question:** What is in the given data, and which non-speech cues could a model cheat with?
- **Rules fixed in advance:** Keep clips ≥ 3.0 s (test clips are longer than 3 s). Split by group (generator × speaker × sentence) so no group spans two splits. Build a `val_testlike` set at 70 % real / 30 % fake with held-out generators. Shortcut audit: if a depth-3 tree on trivial cues (duration, silence, peak, rolloff) beats minDCF 0.5 on validation, those cues must be neutralized. HGT gets format checks only.
- **Outcome** ([reports/01_data_audit.md](../reports/01_data_audit.md)): all 70,242 given files decode; 550 DiffSSD clips under 3.0 s were dropped. The given reals are 242 clips of one speaker. The depth-3 tree scored minDCF 0.000 on the given data (band edge and trailing silence suffice) and still 0.39 after adding external corpora, so the rule fired. The organizers' real clips keep energy up to 8.0 kHz while resampled clips stop near 7.8 kHz. The fix became a shared `prep()` step for every clip, train and test: DC removal, silence trim, 7 kHz low-pass, RMS normalization. A LightGBM on trivial cues scored 0.572 combined minDCF on raw clips and 0.955 (near chance) after `prep()`.
- **Details:** [docs/02_dataset.md](02_dataset.md), [reports/02_r1_classic.md](../reports/02_r1_classic.md).

## Stage 2: external data

- **When:** 25 Sep, 22:07 to 22:56.
- **Question:** How do we stop a model from learning "LJ's voice = real"?
- **Rules fixed in advance:** Disk budget ≤ 40 GB. Priority: diverse real speech first, then fakes from generators DiffSSD lacks (especially LJ-voice fakes). Every external clip goes through the same cleaning path and manifest, with its license recorded.
- **Outcome:** 11 sources ingested: 56,720 real and 54,299 fake clips, 233 hours, 26.9 GB. Real clips went from 0.3 % to 31 % of the data ([reports/05_r5_submission_and_official_score.md](../reports/05_r5_submission_and_official_score.md)).
- **Details:** [docs/03_external_data.md](03_external_data.md).

## Stage 3: TTS simulator

- **When:** 25 Sep, 22:05 (first generators) to 26 Sep, 03:40 (simulation finished).
- **Question:** How is fake speech made, and can we generate hard negatives locally?
- **Rules fixed in advance:** Open-source models only. Size cap about 5k clips. Priority on copy-synthesis of real LJ, where only the vocoder differs from the real clip.
- **Outcome:** 5,204 simulated fakes, 3,916 of them in every model's training data. HiFi-GAN copy-synthesis of LibriSpeech speakers is the hardest fake type in validation: R1 0.917, R4ft 0.13 ([reports/05_r5_submission_and_official_score.md](../reports/05_r5_submission_and_official_score.md)). No no-sim ablation was run, so its training effect is unmeasured.
- **Details:** [docs/04_synthetic_speech.md](04_synthetic_speech.md).

## Stage 4: speech biology

- **When:** 25 Sep, 22:01 to 22:32.
- **Question:** Which physiological properties of real speech can be measured, and do synthesizers get them wrong?
- **Rule fixed in advance:** Every feature must tie a physiological process to a measurable signal and a stated hypothesis about synthesis.
- **Outcome:** 48 biology features. Alone they score 0.600 and they add to spectral features in R1. Their largest contribution was diagnostic: they exposed the 8 kHz channel shortcut in the given reals ([reports/05_r5_submission_and_official_score.md](../reports/05_r5_submission_and_official_score.md)).
- **Details:** [docs/05_speech_biology.md](05_speech_biology.md).

## Stage 5: scoring analysis

- **When:** 25 Sep, 21:59 to 22:06.
- **Question:** What does the organizers' scorer actually reward?
- **Rules fixed in advance:** The organizers' code reads higher scores as real and weights a missed fake 4×; the brief says 1.0 = synthetic and weights a flagged real 4×. Until the organizers answered, every model was reported under both readings (`official_as_written`, `brief_as_written`) and selected on their mean (`combined`). Any clip left unscored gets 0.3, never the template's 0.006.
- **Outcome:** A local scorer that matches the organizers' script ([tests/test_metrics.py](../tests/test_metrics.py)). minDCF is rank-only, and any constant file scores 1.0. In the end every clip was scored, so no default was used.
- **Details:** [docs/01_challenge_and_scoring.md](01_challenge_and_scoring.md) §1 to §4.

## Stage 6: modeling ladder R0 to R3, and frozen SSL (R4)

- **When:** 25 Sep, 23:08 (execution plan) to 26 Sep, 06:10.
- **Question:** How far do trivial cues, classic features and off-the-shelf models go?
- **Rules fixed in advance:** Train on `split == "train"` only. Every choice is made on `val_testlike` combined minDCF. `test_internal_testlike` (9,747 clips, 70/30, with 3 generators never seen in training) is the headline and is never tuned on.
- **Outcome** (headline combined minDCF):

| rung | model | val_testlike | headline |
|---|---|---|---|
| R0 | trivial cues after `prep()` | 0.933 | 0.921 |
| R1 | LightGBM, 228 spectral + biology features | 0.228 | 0.253 |
| R3 | organizers' AASIST, zero-shot | 0.831 | 0.823 |
| R4 | frozen WavLM-base+ layer 3 + LR (partial, 14.8k train clips) | 0.389 | not scored |

R1 improved from 0.305 to 0.253 when trained on the full training set (26 Sep, 04:05). Frozen SSL heads did not beat R1. ([reports/02_r1_classic.md](../reports/02_r1_classic.md), [reports/05_r5_submission_and_official_score.md](../reports/05_r5_submission_and_official_score.md)).
- **Also fixed from validation only:** the R1 error analysis showed asvspoof5 reals as the worst slice (0.501 on `val_testlike`). A first draft also proposed a ×2 weight for LibriSeVoc reals taken from the headline set; review removed it as leakage ([reports/03_error_analysis.md](../reports/03_error_analysis.md)).

## Stage 7: fine-tuned XLS-R (R4ft), selection rule and the R5 amendment

- **When:** plan 26 Sep, 04:34 (sampling weight fixed 04:38); run 05:45 to 08:22; selection rule 06:15; amendment about 09:30.
- **Question:** Does end-to-end fine-tuning of an SSL front end close the gap to R1?
- **Rules fixed in advance (training):** XLS-R-300M cut to 12 blocks, light back end. Sampling weight: asvspoof5 reals ×2, taken from `val_testlike` slices only. Save a new best only when `val_testlike` improves by ≥ 0.002; stop after 4 evaluations without that, or at 16 virtual epochs. At most 3 runs. VRAM config: fastest with peak reserved ≤ 85 % of the cap. Write `config.json` (checkpoint and inference mode) before the headline is scored.
- **Rule fixed in advance (final submission, 06:15, before any R4ft headline existed):** compare candidates on `val_testlike` combined, fusions by their group-cross-fitted number. A fusion replaces the best single model only if it wins by ≥ 0.002. The headline is reported, never used to switch.
- **Outcome:** One run of 144 minutes, early-stopped at step 10,240 with the best at step 6,144. The 3-window inference mode was chosen on `val_testlike` (0.0121 against 0.0166 for the centre window). Headline 0.0282 (official 0.0301, brief 0.0262, EER 0.76 %), about 9× better than R1; the stop criterion was met. The rule picked R4ft alone (0.0121 against 0.0128 for the cross-fitted R4ft + R1 fusion).
- **Recorded deviations:** the VRAM rule picked no config (all reserved more than 6.20 GiB), so config (a) was chosen by hand. The run trained with asvspoof5 ×2 while `args.json` recorded an empty list; the bookkeeping bug was fixed afterwards.
- **Owner amendment (made after headline scores were seen):** the submission switched to the R4ft + R1 fusion (R5). Test-free checks tied (`val_testlike` 0.0121 vs 0.0128; 9,595 unselected validation fakes 0.0329 vs 0.0327), and the HGT-like proxy favoured the fusion (0.0007 vs 0.0000). Because the headline also informed the switch, R5's 0.0218 is labelled optimistic and R4ft's 0.0282 remains the unbiased estimate.
- **Details:** [reports/04_r4ft_xlsr.md](../reports/04_r4ft_xlsr.md), [reports/05_r5_submission_and_official_score.md](../reports/05_r5_submission_and_official_score.md).

## Stage 8: generalization benchmarks and call-audio stress tests

- **When:** plan 26 Sep, 12:04; report 12:55; revised after review 16:11.
- **Question:** Does the frozen submission hold on whole corpora it never saw, and on real call conditions?
- **Rules fixed in advance:** Nothing is trained and the submission is not touched. No threshold, calibration or model choice from these sets; one scoring pass per model. Thresholds are fixed on `val_testlike` and applied unchanged. Leakage caveats are printed next to each number.
- **Outcome** (R5, combined minDCF): ASVspoof 2021 DF 0.061, DECRO-en 0.031, CD-ADD 0.000, DeepVoice 0.231. On DeepVoice (voice conversion), 45 % of fakes pass as real at the fixed threshold. Typing as loud as the voice: at most 1.7 % of real speakers flagged. As run, R3 and the ODSS set were dropped (time; ODSS clips are mostly under 3 s).
- **Details:** [reports/06_generalization.md](../reports/06_generalization.md).

## Stage 9: official result

- **When:** 26 Sep, afternoon (recorded 16:09).
- **Question:** How did the organizers score the submission?
- **Outcome:** The organizers confirmed the scorer: Pspoof 0.3, Cmiss 1, Cfa 4, and **higher score = real**, the opposite of the brief. Our first file followed the brief and scored 1.0. The flipped file scored **minDCF 0.0584, EER 2.5 %**. Under the same settings our held-out estimate was 0.0138 (R5) and 0.0192 (R4ft). The better of the initial and final submission counts. Nothing was tuned on this feedback.
- **Details:** [docs/01_challenge_and_scoring.md](01_challenge_and_scoring.md) §7, [reports/05_r5_submission_and_official_score.md](../reports/05_r5_submission_and_official_score.md).

## Stage 10: R6 and the final submission (E5)

- **When:** plan about 15:45; rule revised about 16:10, before any R6 training; R6 selected 21:18. Submission deadline 27 Sep, 08:00; R5 would stay the final if R6 had not finished by then.
- **Question:** Can more diverse training data beat 0.0584, using only our own data to choose?
- **Rule fixed in advance (revised version):** R6 = the exact R4ft recipe with seed 1, plus ASVspoof 2021 DF, CD-ADD and DECRO-en as training sources. DeepVoice is never trained on. Metric: official minDCF (Pspoof 0.3, Cmiss 1, Cfa 4). Guardrails relative to R5: `val_testlike` ≤ R5 + 0.003 (fusions cross-fitted) and DeepVoice ≤ R5 + 0.02. Preference among passing candidates: E5 > E > R6, except E if E5's `val_testlike` is worse than E's by more than 0.002. If none passes, the final stays R5.
- **Outcome** ([reports/07_r6_and_final_submission.md](../reports/07_r6_and_final_submission.md)):

| candidate | val_testlike | DeepVoice | passes |
|---|---|---|---|
| R5 (baseline) | 0.0088 | 0.150 | n/a |
| R6 alone | 0.0148 | 0.182 | no |
| E: R4ft + R6, equal weights | 0.0096 | 0.172 | no (DeepVoice by 0.002) |
| **E5: R4ft + R6 + R1** | **0.0099** | **0.152** | **yes, selected** |

Report-only, scored once after the choice (`test_internal_testlike`, official): E5 0.0154, R5 0.0138, R4ft 0.0192, R6 0.0236.
- **Recorded deviation:** R6 did not train to early stopping. The GPU was shared with another project, and a low-memory safeguard stopped R6 twice (19:32 and 20:20) at about step 7,150, with early stopping at 2 of 4. The owner chose to score the best checkpoint (step 4,096, 0.0267). The two evaluations after it were 0.0295 and 0.0290; a later one would have needed ≤ 0.0247 to change the checkpoint.
- **Details:** selection table [reports/07_r6_selection.csv](../reports/07_r6_selection.csv).

## How the work was run

Two laptops shared the work. A CPU machine acted as coordinator and was the only writer of the manifest, leaderboard, canonical scores and submission files. A GPU machine (RTX 5050, 8 GB) ran the fine-tuning and uploaded its scores. They exchanged messages and large files through a small token-authenticated HTTP hub on the local network, and tracked tasks as GitHub issues. Every change went through a pull request that the other machine reviewed before merge. Reviews caught a submission writer that silently defaulted every row, leaky cross-validation folds in the fusion, and a training weight derived from the headline set ([reports/05_r5_submission_and_official_score.md](../reports/05_r5_submission_and_official_score.md)).

## Stage 11: R7, RawBoost and a second backbone

- **When:** rule written 27 Sep, 02:00, before any R7 training. Submission deadline 08:00.
- **Question:** Team 7 leads at minDCF 0.0317 (EER 1.44 %); our best official score is 0.0584. The gap between our
  held-out set (0.0138) and the test (0.0584) points to unfamiliar recording channels and generators. Can channel
  augmentation plus a second pretrained backbone close it?
- **What changes:**
  - **RawBoost** (Tak et al., ICASSP 2022): with probability 0.6 per training clip, one of the paper's channel
    simulations (convolutive linear/non-linear noise, impulsive signal-dependent noise, stationary coloured noise, or
    their series combinations) runs before our usual augmentation, identically for real and fake clips.
  - **R7a:** XLS-R-300M, the R4ft recipe, seed 2, RawBoost 0.6, the current training set (including R6's extra data).
  - **R7b:** WavLM-Large (a different self-supervised family, same size), same recipe, seed 3, RawBoost 0.6.
- **Rule fixed in advance:**
  - Metric: the organizers' official minDCF (Pspoof 0.3, Cmiss 1, Cfa 4).
  - Candidates: **E7** = LR fusion of every finished XLS-R/WavLM model (R4ft, R6, R7a, R7b) + R1; **E7s** = the same
    without R1. Fusions are fit on `val_testlike` and judged on their group-cross-fitted scores, as before.
  - Guardrails relative to E5 (the current final): `val_testlike` ≤ E5 + 0.003 and DeepVoice ≤ E5 + 0.02.
  - Preference: E7 > E7s. Exception: E7s if its `val_testlike` beats E7's by more than 0.002. If neither passes, E5
    stays the final.
  - Time box: a model not scored and uploaded by 05:45 is left out; the final file is built by 06:30.
  - Nothing is chosen from the organizers' feedback or the HGT audio.
