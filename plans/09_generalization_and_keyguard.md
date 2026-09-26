# Plan 09: does it generalize, and what do we build on it?

Written 2026-09-26, after the submission was frozen (`submission/HearsayScoreKey4GeorgiaMellon.tsv`, R5_r4ft_r1).
**Nothing in this phase trains a new model or touches the submission.** The GPU node is busy elsewhere; everything
here is CPU inference with the frozen models.

## Part A: public benchmarks the model never saw

**Question.** Our headline (0.0282) is measured on held-out clips from corpora that *were* in training (other
speakers, other utterances, 3 held-out generators). Does the method hold on whole corpora it never saw?

**Sets** (HuggingFace `SpeechAntiSpoofingBenchmarks` repacks, 16 kHz FLAC, English only, none ingested before;
see `docs/external_data.md` §2):

| set | why | leakage caveat |
|---|---|---|
| ASVspoof 2021 DF | the standard deepfake benchmark; 100+ vocoders, compression codecs | part of its sources = ASVspoof 2019 LA eval utterances, which we trained on; report the VCC sources separately if the metadata allows |
| CD-ADD | zero-shot TTS 2023-24 (OpenVoice, VALL-E-style, …) on LibriTTS | LibriTTS reals come from LibriSpeech, some of whose utterances are in our training reals |
| DECRO (English half) | cross-lingual benchmark, different TTS stack | none known |
| DeepVoice | celebrity voice conversion (RVC) from YouTube, real-world channel | none known |
| ODSS (English) | VITS / FastPitch on Hi-Fi TTS voices | none known |

Per set: all reals up to 1,500, fakes sampled to 1,500 (seeded), clips ≥ 3 s after decode, `prep()` as always.
Written to `data/bench/<set>/` with an index parquet; the training manifest is **not** touched.

**Models:** R1 (refit deterministically on the same features: the model object was never saved), R3 (organizers'
AASIST, zero-shot, as the reference point), R4ft (frozen checkpoint + frozen win3), R5 fusion (frozen transform
`data/scores/R5_r4ft_r1.json`).

**Metrics:** EER, both minDCF readings + combined (rank-only). Plus the number minDCF hides: **at the threshold fixed
on `val_testlike`**, the % of reals flagged and % of fakes missed. A deployed detector has one threshold; this is
the honest "would it work out of the box" number.

**Honesty rules:** no threshold, calibration or model choice from these sets; one scoring pass per model; per-set
leakage caveats printed next to the numbers. Output: `reports/generalization.md`.

**As run (2026-09-26):** two items were dropped. R3 (AASIST zero-shot) was skipped for time; `bench_score.py r3`
exists but was not run, and `report` skips R3 when its scores are missing. ODSS was not ingested: almost all of its
clips are under 3 s, so the ≥ 3 s filter leaves too few.

## Part B: Hearsay × Keyguard

Keyguard (teammate's repo, `LordKarV/keyboard-acoustic-shield`) = keystroke eavesdropping through call audio + an
adversarial shield that hides keystrokes while keeping speech intact (PESQ/STOI/Whisper). Both projects defend a
**live call**: one asks "is this voice a real person?", the other "is my typing leaking?".

No-training experiments (CPU):
- **E1 typing while talking.** Real speech from our held-out set + Keyguard keystroke recordings at several SNRs.
  Does Hearsay flag real people who type as fakes? (The brief makes that error cost 4x.)
- **E2 shield side effects.** Run Keyguard's shields on real speech. Does the shield make a real voice look
  synthetic? If it does, Hearsay becomes a fourth constraint in Keyguard's min-max (next to PESQ, STOI, Whisper).

Ideas and pitch: `docs/keyguard_fusion.md`. Code stays in this repo under `integrations/` or as a thin scoring API
Keyguard can call; no changes are pushed to the teammate's repo without them.
