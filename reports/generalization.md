# Generalization: public benchmarks and call-audio stress tests

Plan: [`plans/09`](../plans/09_generalization_and_keyguard.md). **Inference only.** The models are the frozen submission
models; nothing was trained, calibrated, thresholded or selected on any set below. Each model's threshold was fixed
on `val_testlike` (brief reading: a real voice flagged as fake costs 4×) and applied unchanged, which shows what a
deployed detector would do. minDCF, by contrast, picks the best threshold per set after the fact.

Models: **R4ft** (fine-tuned XLS-R, the core of the submission), **R5** (R4ft + R1 fusion = the submitted system),
**R1** (classic features + biology, LightGBM). R1 had to be refit because its model object was never saved. The refit
is not bit-identical (Spearman 0.998 with the saved scores; `val_testlike` 0.2286 vs 0.228). Thresholds come from the
exact objects scored here: the refit R1's `val_testlike` scores, R4ft's frozen scores, and the frozen full-fit R5
fusion applied to those two (`val_testlike` combined 0.0125).

R4ft was scored on the GPU node (bf16) and spot-checked against CPU fp32 re-scoring: 50 clips, correlation 0.9999,
max logit gap 1.1 on a ±40 range. Reproduce: `scripts/ingest_bench.py`, `scripts/make_keyguard_sets.py`,
`scripts/bench_score.py r1|r4ft`, then `scripts/bench_score.py report --sets deepvoice cd_add decro_en
asvspoof2021_df keyguard_clean keyguard_type20 keyguard_type10 keyguard_type5 keyguard_type0 keyguard_shield_clean
keyguard_shield_type10` (the default sets leave out `keyguard_*`). Full table: `data/bench/generalization.csv`.

Cells are minDCF **official / brief / combined** (`docs/scoring.md` §2; 0 = perfect, 1 = no better than always
answering the same). The last column is R5 at its fixed `val_testlike` threshold.

## 1. Public benchmarks never used in training

| set (English, clips ≥ 3 s) | real / fake | R1 | R4ft | **R5 (submitted)** | R5 EER | R5: reals flagged / fakes missed |
|---|---|---|---|---|---|---|
| ASVspoof 2021 DF | 645 / 668 | 0.798 / 0.631 / 0.714 | 0.092 / 0.047 / 0.069 | **0.087 / 0.035 / 0.061** | 1.8 % | 0.2 % / 4.8 % |
| DECRO (English half) | 892 / 1,087 | 0.218 / 0.187 / 0.202 | 0.084 / 0.045 / 0.064 | **0.040 / 0.023 / 0.031** | 1.0 % | 0.0 % / 8.9 % |
| CD-ADD (zero-shot TTS 2023-24) | 1,377 / 1,499 | 0.523 / 0.485 / 0.504 | 0.000 / 0.000 / 0.000 | **0.000 / 0.000 / 0.000** | 0.0 % | 0.1 % / 0.0 % |
| DeepVoice (celebrity RVC voice conversion, YouTube) | 625 / 1,489 | 0.771 / 0.907 / 0.839 | 0.226 / 0.217 / 0.221 | **0.231 / 0.232 / 0.231** | 5.8 % | 0.5 % / 45.4 % |

**How clean each test is** (read the numbers with this; R5 combined unless noted):
- **ASVspoof 2021 DF.** By source: fakes built on the VCC 2018 / 2020 corpora (unseen) are caught almost perfectly
  (0.023 / 0.008; EER 0.8 % / 0.2 %). The ASVspoof 2019 LA-sourced part is harder (0.078, EER 3.0 %, 22 % of its 99
  fakes missed at the fixed threshold), although that corpus is in training: its attacks are A07-A19, all of which we
  trained on, and 431 of the 645 reals come from LA eval, where our train split holds 2,941 of the ~7,355 bona fide
  recordings (~40 %). DF IDs can't be mapped back, but ~170 DF reals are likely codec'd copies of training clips,
  which flatters "reals flagged 0.2 %". Codecs: R4ft trained with an MP3 round trip (25 % of clips), and 409 of the
  1,313 clips are mp3 / mp3m4a; m4a and ogg are unseen, and on the clips without mp3 R5 scores 0.060, the same as the
  full set. Not comparable to published full-DF EERs: this is a 1,313-clip sample, and half of the sampled clips are
  under 3 s and were dropped.
- **DECRO-en: mostly familiar data.** All 892 reals are ASVspoof 2019 LA eval recordings (`LA_E_*`) that are in our
  manifest, 597 (67 %) in train. 573 of the 1,087 fakes (53 %) are hifigan / mbmelgan / pwg vocodings of LJSpeech:
  all 573 LJ IDs are in our manifest, 453 of them in train (same speaker, same texts), and WaveFake gives us ~1.1k
  training clips of each of those vocoder families. R4ft misses 0 % of the pwg and mbmelgan fakes and 7.8 % of the
  hifigan ones. **Without the LJ-vocoder fakes** the numbers roughly double: R5 0.060 / 0.039 / **0.049**, R4ft
  0.120 / 0.086 / **0.103** (vs 0.064 full). Weakest spots at the fixed threshold: Tacotron (90 % of 51 missed),
  Baidu TTS (23 % of 141).
- **CD-ADD: easy, partly familiar.** Reals: 886 are LibriTTS test-clean (14 speakers in this sample, all of whom are
  in our train via 1,775 LibriSpeech test-clean clips) and 491 (36 %) are TED talks. Fakes: 3 of the 5 "zero-shot"
  generators are in training, 845 of 1,499 fakes (56 %): OpenVoice (openvoicev2, 20,181 train clips, plus 47
  mlaad_OpenVoiceV2), WhisperSpeech (mlaad, 76), VALL-E (sonar_VALLE, 58). YourTTS and Seamless are unseen. R4ft and
  R5 are 0.000 on every subset: unseen generators only, and TED reals only (no familiar real speakers). So the
  unseen part is also separated, but this is still "these generators are easy for it", not "0.000 in the wild".
- **DeepVoice: the main unseen test, and the weak spot.** Real celebrity speech from YouTube against RVC voice
  conversion of the same people. Ranking is decent (EER 5.8 %), but at the threshold tuned on our validation set
  **45 % of the conversions pass as real** (R4ft alone: 35 %). The fakes keep the real prosody and breath of the
  source speaker and only swap timbre, and YouTube's codec chain hides vocoder traces. Our training has little voice
  conversion (83 mlaad_RVC clips). It is the clearest next data gap. Real voices are still almost never flagged
  (0.5 %). Overlap: two DeepVoice identities are In-the-Wild speakers we hold, Trump in `val` (973 real / 66 fake)
  and Obama in `test_internal` (1,175 / 105), so the real side is not wholly new.

**Takeaway.** The fine-tuned SSL front end transfers; R1 (classic features) does not. Against the in-domain headline
(0.022 / 0.028), R5 stays within 2× on two of four benchmarks (DECRO 0.031, CD-ADD 0.000), and both share data with
our training. Where the data is new, the picture is: unseen VCC-sourced DF fakes are near-perfect, DECRO without the
familiar LJ vocodings is 0.049, and voice conversion (DeepVoice, 0.231) is the real failure. The brief's expensive
error, a real voice flagged, stays ≤ 0.5 % everywhere at the fixed threshold; the miss rate is what moves.

## 2. Call-audio stress tests (Hearsay × Keyguard, `docs/keyguard_fusion.md`)

600 held-out clips (300 real / 300 fake, `test_internal_testlike`). Keystrokes: Keyguard's recorded presses, 5
keyboards, 3 presses/s, level vs speech power as in Keyguard's `synth.mix`. Shield: Keyguard's DSP shield
(dashboard settings).

| condition | R1 | R4ft | **R5** | R5: reals flagged / fakes missed |
|---|---|---|---|---|
| clean | 0.280 / 0.187 / 0.233 | 0.033 / 0.033 / 0.033 | **0.037 / 0.037 / 0.037** | 0.7 % / 2.0 % |
| typing, 20 dB below speech | 0.460 / 0.380 / 0.420 | 0.067 / 0.043 / 0.055 | **0.047 / 0.030 / 0.038** | 0.3 % / 4.7 % |
| typing, 10 dB | 0.660 / 0.543 / 0.602 | 0.080 / 0.047 / 0.063 | **0.057 / 0.050 / 0.053** | 0.0 % / 5.3 % |
| typing, 5 dB | 0.680 / 0.600 / 0.640 | 0.103 / 0.073 / 0.088 | **0.077 / 0.077 / 0.077** | 0.7 % / 5.7 % |
| typing, 0 dB (as loud as the voice) | 0.717 / 0.657 / 0.687 | 0.117 / 0.073 / 0.095 | **0.073 / 0.093 / 0.083** | 1.7 % / 5.0 % |
| Keyguard shield on clean speech | 0.470 / 0.383 / 0.427 | 0.060 / 0.047 / 0.053 | **0.057 / 0.043 / 0.050** | 1.3 % / 2.0 % |
| typing 10 dB + Keyguard shield | 0.677 / 0.500 / 0.588 | 0.087 / 0.057 / 0.072 | **0.043 / 0.063 / 0.053** | 1.3 % / 2.7 % |

- **Typing while talking:** the submitted system degrades gracefully. Even with keystrokes as loud as the voice, it
  flags ≤ 1.7 % of real speakers; misses rise from 2 % to ~5 %. R1 falls apart: 5-8 % of reals flagged at any
  typing level.
- **Keyguard's shield:** for R1 it is a real problem (17 % of real voices flagged, from 1 %). For the submitted
  system the effect is small: 2 more real clips flagged out of 300 on clean speech (0.7 % → 1.3 %), within noise at
  this sample size. So the "authenticity-preserving shield" constraint matters most for weaker detectors, and a
  detector-aware shield should still check it: the other side of a call may run a weaker detector than ours.
- Sample size: 300 per class, so one clip = 0.33 %. Differences under about 1 % are not meaningful.
