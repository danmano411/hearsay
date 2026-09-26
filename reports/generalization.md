# Generalization: public benchmarks and call-audio stress tests

Plan: [`plans/09`](../plans/09_generalization_and_keyguard.md). **Inference only.** The models are the frozen submission
models; nothing was trained, calibrated, thresholded or selected on any set below. Each model's threshold was fixed
on `val_testlike` (brief reading: a real voice flagged as fake costs 4×) and applied unchanged, which shows what a
deployed detector would do. minDCF, by contrast, picks the best threshold per set after the fact.

Models: **R4ft** (fine-tuned XLS-R, the core of the submission), **R5** (R4ft + R1 fusion = the submitted system),
**R1** (classic features + biology, LightGBM). R1 had to be refit because its model object was never saved. The refit
is not bit-identical (Spearman 0.998 with the saved scores; `val_testlike` 0.2286 vs 0.228).

R4ft was scored on the GPU node (bf16) and spot-checked against CPU fp32 re-scoring: 50 clips, correlation 0.9999,
max logit gap 1.1 on a ±40 range. Reproduce: `scripts/ingest_bench.py`, `scripts/make_keyguard_sets.py`,
`scripts/bench_score.py`.

## 1. Public benchmarks never used in training

| set (English, clips ≥ 3 s) | real / fake | R1 | R4ft | **R5 (submitted)** | R5 EER | R5 at the fixed threshold: reals flagged / fakes missed |
|---|---|---|---|---|---|---|
| ASVspoof 2021 DF | 645 / 668 | 0.714 | 0.069 | **0.061** | 1.8 % | 0.2 % / 4.5 % |
| DECRO (English half) | 892 / 1,087 | 0.202 | 0.064 | **0.031** | 1.0 % | 0.0 % / 7.8 % |
| CD-ADD (zero-shot TTS 2023-24) | 1,377 / 1,499 | 0.504 | 0.000 | **0.000** | 0.0 % | 0.1 % / 0.0 % |
| DeepVoice (celebrity RVC voice conversion, YouTube) | 625 / 1,489 | 0.839 | 0.222 | **0.231** | 5.8 % | 0.5 % / 41.6 % |

minDCF is the combined reading (0 = perfect, 1 = no better than always answering the same).

**How clean each test is** (read the numbers with this):
- **ASVspoof 2021 DF.** By source: fakes made from the VCC 2018 / 2020 corpora, which are unseen, are caught almost
  perfectly (EER 0.8 % / 0.2 %). Fakes built on ASVspoof 2019 LA material are harder (EER 3.0 %; 22 % missed at the
  fixed threshold), although that is the corpus we trained on. DF adds compression codecs we never trained with. Not
  comparable to published full-DF EERs: this is a 1,313-clip sample, and half of the sampled clips are under 3 s and
  were dropped.
- **DECRO-en.** Its real clips are ASVspoof 2019 LA eval recordings (`LA_E_*`), which *are* in our training data, and
  many fakes are in the LJ voice. This measures new generators, not new real speech. Weakest spot: the Tacotron
  fakes (84 % missed at the fixed threshold).
- **CD-ADD.** All five zero-shot generators (VALL-E, WhisperSpeech, Seamless, YourTTS, OpenVoice) are separated
  perfectly. But its real clips are LibriTTS, cut from LibriSpeech speakers whose recordings are in our training
  data, so the real side is familiar. Read this as "these generators are easy for it", not as "0.000 in the wild".
- **DeepVoice: the one fully unseen test, and the weak spot.** Real celebrity speech from YouTube against RVC voice
  conversion of the same people. Ranking is decent (EER 5.8 %), but at the threshold tuned on our validation set
  **42 % of the conversions pass as real**. The fakes keep the real prosody and breath of the source speaker and
  only swap timbre, and YouTube's codec chain hides vocoder traces. Our training has little voice conversion and
  no RVC. It is the clearest next data gap. Real voices are still almost never flagged (0.5 %).

**Takeaway.** The fine-tuned SSL front end transfers. On three of four unseen benchmarks the submitted system stays
within 2× of its in-domain headline (0.022 / 0.028), and R1 (classic features) does not transfer at all. The
brief's expensive error, a real voice flagged, stays ≤ 0.5 % everywhere at the fixed threshold. The miss rate is what
moves when the generator family is new (voice conversion).

## 2. Call-audio stress tests (Hearsay × Keyguard, `docs/keyguard_fusion.md`)

600 held-out clips (300 real / 300 fake, `test_internal_testlike`). Keystrokes: Keyguard's recorded presses, 5
keyboards, 3 presses/s, level vs speech power as in Keyguard's `synth.mix`. Shield: Keyguard's DSP shield
(dashboard settings).

| condition | R1 | R4ft | **R5** | R5: reals flagged / fakes missed |
|---|---|---|---|---|
| clean | 0.233 | 0.033 | **0.037** | 0.7 % / 1.7 % |
| typing, 20 dB below speech | 0.420 | 0.055 | **0.038** | 0.3 % / 3.3 % |
| typing, 10 dB | 0.602 | 0.063 | **0.053** | 0.3 % / 4.7 % |
| typing, 5 dB | 0.640 | 0.088 | **0.077** | 1.0 % / 4.7 % |
| typing, 0 dB (as loud as the voice) | 0.687 | 0.095 | **0.083** | 1.7 % / 3.7 % |
| Keyguard shield on clean speech | 0.427 | 0.053 | **0.050** | 1.7 % / 2.0 % |
| typing 10 dB + Keyguard shield | 0.588 | 0.072 | **0.053** | 1.3 % / 1.3 % |

- **Typing while talking:** the submitted system degrades gracefully. Even with keystrokes as loud as the voice, it
  flags ≤ 1.7 % of real speakers. R1 falls apart: 7-11 % of reals flagged at any typing level.
- **Keyguard's shield:** for R1 it is a real problem (18 % of real voices flagged, from 1 %). For the submitted
  system the effect is small: 2 more real clips flagged out of 300 on clean speech, within noise at this sample size.
  So the "authenticity-preserving shield" constraint matters most for weaker detectors, and a detector-aware shield
  should still check it: the other side of a call may run a weaker detector than ours.
- Sample size: 300 per class, so one clip = 0.33 %. Differences under about 1 % are not meaningful.
