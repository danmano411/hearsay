# Data audit (stage 1)

What is in the training data we were given, what we changed, and which trivial cues a model could cheat with.
Reproduce: `scripts/audit_raw.py` → `scripts/clean_given.py` → `scripts/make_splits.py` → `scripts/audit_shortcuts.py`
(details in [docs/02_dataset.md](../docs/02_dataset.md)).

## TL;DR

1. **All 70,242 given files decode.** No empty files, no exact PCM duplicates. 550 DiffSSD clips (0.8%) are
   shorter than 3.0 s and were dropped, mostly from the four LJ-voice generators.
2. **The given data is fully separable by non-speech cues.** On DiffSSD + lj_real alone, a depth-3 decision tree
   on eight "cheap" statistics (duration, level, silence, DC, bandwidth) scores **minDCF = 0.000** on our held-out
   split. Two cues are enough: *spectral band edge* and *trailing digital silence*.
3. **One of those cues comes from our own pipeline.** The organizers' 242 real LJ clips arrive at 16 kHz with energy
   up to 8.0 kHz. Everything we resample from 22.05/24/44.1 kHz with soxr stops at about 7.8 kHz. So "energy at
   7.9–8 kHz" means "was already 16 kHz", which here means "real".
4. **Adding the external corpora (stage 2) dilutes the shortcuts but does not remove them.** With 133k clips from
   10 sources, the same depth-3 tree still gets **minDCF 0.39** (DC offset and trailing silence carry most of it).
   A model that is not stopped from using these cues will learn them.
5. **Test clips are much shorter than training clips.** HGT test median is 3.41 s; DiffSSD median is 6–9 s.
6. **The given real class is tiny.** It is 242 clips of one speaker (LJ). External real speech (stage 2) raises
   bonafide to 30.7k clips.

## 1. Raw formats (native, before cleaning)

`scripts/audit_raw.py` covers all 70,242 files, runs 8 worker processes, and writes
`data/processed/audit/raw_audit.parquet`.

| generator | files | speakers | native sr (Hz) | codec | voice |
|---|---:|---:|---:|---|---|
| lj_real (bonafide) | 242 | 1 | 16000 | WAV PCM16 | LJ (real) |
| diffgan_tts | 5000 | 1 | 22050 | WAV PCM16 | LJ, pre-trained |
| grad_tts | 5000 | 1 | 22050 | WAV PCM16 | LJ, pre-trained |
| pro_diff | 5000 | 1 | 22050 | WAV PCM16 | LJ, pre-trained |
| wavegrad2 | 5000 | 1 | 22050 | WAV PCM16 | LJ, pre-trained |
| elevenlabs | 5000 | 10 | 44100 | **MP3** | zero-shot clone |
| playht | 5000 | 10 | 24000 | **MP3** | zero-shot clone |
| openvoicev2 | 25000 | 10 | 22050 | WAV PCM16 | zero-shot clone, 5 accents |
| unit_speech | 5000 | 10 | 22050 | WAV PCM16 | zero-shot clone |
| xtts_v2 | 5000 | 10 | 24000 | WAV PCM16 | zero-shot clone |
| your_tts | 5000 | 10 | 16000 | WAV PCM16 | zero-shot clone |

All files are mono.

### Signal statistics at native rate (median [p5, p95])

`lead_sil` and `trail_sil` count leading and trailing samples with |x| < 1e-4. `bw60` is the highest frequency
whose mean power is within 60 dB of the spectral peak, i.e. the low-pass edge. `rolloff95` is the frequency below
which 95% of the energy lies.

| generator | duration s | peak | RMS dBFS | lead_sil s | trail_sil s | bw60 Hz | DC |
|---|---|---|---|---|---|---|---|
| lj_real | 7.1 [4.35, 9.49] | 0.54 | -24.0 | 0 | 0 | 8000 | 8.9e-06 |
| diffgan_tts | 5.86 [3.41, 9.9] | 0.78 | -19.8 | 0 | 0 | 11025 | 5.9e-06 |
| grad_tts | 6.08 [3.41, 10.6] | 0.54 | -23.9 | 0 | 0 | 11025 | 3.1e-05 |
| pro_diff | 6.14 [3.38, 10.4] | 0.47 | -24.9 | 0 | 0 | 11025 | -2.1e-04 |
| wavegrad2 | 6.31 [3.71, 11.1] | 0.83 | -21.0 | 0 | 0 | 11025 | -1.3e-05 |
| elevenlabs | 7.35 [4.96, 12.7] | 0.55 | -24.9 | **0.025** | 0.013 | 15100 | 1.7e-05 |
| playht | 7.33 [4.92, 12.7] | 0.65 | -24.3 | 0 | **0.101** | 8840 | -1.8e-05 |
| openvoicev2 | 7.52 [5.15, 12.3] | 0.44 | -25.5 | 0 | 0 | 7580 | **-5.0e-04** |
| unit_speech | 7.73 [5.21, 13.8] | 0.38 | -26.7 | 0 | 0 | 10400 | -6.5e-05 |
| xtts_v2 | 7.94 [5.29, 13.3] | **1.00** | -17.2 | 0 | **0.417 [0.417, 0.582]** | 12000 | -8.2e-05 |
| your_tts | 8.88 [5.71, 14.6] | **1.00** | -18.6 | 0 | **0.625 [0.625, 0.625]** | 8000 | 2.3e-04 |

Signatures a model could memorize:

- **Peak normalization.** xtts_v2 and your_tts are normalized to exactly full scale (peak 1.0, with a few clipped
  samples).
- **Fixed-length trailing silence.** your_tts pads every clip with exactly 0.625 s of digital zeros, xtts_v2 with at
  least 0.417 s, and playht with about 0.1 s. MP3 encoder delay adds about 25 ms of leading silence to elevenlabs.
- **DC offset.** openvoicev2 sits at about -5e-4 and pro_diff at about -2e-4. Real LJ is near 1e-5.
- **Band edge.** Each native rate and codec leaves its own low-pass edge. MP3 cuts off between 8.8 and 15 kHz, and
  openvoicev2's vocoder rolls off near 7.6 kHz.

### HGT test format check (format only, per the inference-only rule)

| n | sample rate | channels | codec | duration min / median / max |
|---:|---|---|---|---|
| 1671 | 16000 | 1 | PCM16 | 3.02 / **3.41** / 13.58 s |

The test set matches our canonical format: 16 kHz mono PCM16, and every clip is ≥ 3 s. Test clips are about half
as long as training clips, so durations are mismatched.

## 2. Cleaning (`scripts/clean_given.py`)

Each clip is decoded, mixed to mono, resampled to 16 kHz with soxr_hq, and written as a PCM16 WAV. We did no trimming
and no loudness normalization. That keeps the cues above auditable; neutralizing them is a model-input decision (§4).

| generator | kept | dropped < 3 s | undecodable | exact duplicate |
|---|---:|---:|---:|---:|
| lj_real (bonafide) | 242 | 0 | 0 | 0 |
| diffgan_tts | 4865 | 135 | 0 | 0 |
| grad_tts | 4862 | 138 | 0 | 0 |
| pro_diff | 4851 | 149 | 0 | 0 |
| wavegrad2 | 4904 | 96 | 0 | 0 |
| elevenlabs | 4995 | 5 | 0 | 0 |
| playht | 4999 | 1 | 0 | 0 |
| openvoicev2 | 24974 | 26 | 0 | 0 |
| unit_speech | 5000 | 0 | 0 | 0 |
| xtts_v2 | 5000 | 0 | 0 | 0 |
| your_tts | 5000 | 0 | 0 | 0 |
| **total** | **69,692** | **550** | 0 | 0 |

Duplicates were checked with a SHA-1 hash of the int16 PCM. The five openvoicev2 accent renderings of a sentence are
different audio, so all of them are kept.

## 3. Shortcut audit (`scripts/audit_shortcuts.py`)

**Method.** The features are the eight statistics above, computed on the canonical 16 kHz clip that a model sees. We
fit `DecisionTreeClassifier(max_depth=3, class_weight="balanced")` on `split=train` and score on `val ∪
test_internal`, pooled because the real class is small. The metric is ASVspoof5 Track-1 normalized minDCF
(π_spoof = 0.05, C_miss = 1, C_fa = 10). A score of 1.0 means the cues carry no information; the plan's alarm
threshold is **< 0.5**.

| subset | eval bona / spoof | all 8 cues | best single cue |
|---|---:|---:|---|
| **given only** (DiffSSD + lj_real) | 47 / 13,411 | **0.000** | bw60 **0.072**, DC 0.31, trail_sil 0.44 |
| LJ voice only (lj_real + full LJSpeech vs LJ-voice fakes) | 2,380 / 3,826 | **0.237** | DC 0.43, trail_sil 1.00 |
| all sources (stage 1 + stage 2 + stage 3 manifests) | 7,866 / 19,970 | **0.393** | DC 0.49, trail_sil 0.74, bw60 0.82 |

Every row is below 0.5, so trivial cues leak in every configuration.

Tree learned on the given data (verbatim from `export_text`; the root split is the band edge):

```
bw60 <= 7953 Hz                      -> spoof     (anything we resampled)
bw60 >  7953 Hz, trail_sil <= 0.31 s -> bonafide  (native 16 kHz, no padding = lj_real)
bw60 >  7953 Hz, trail_sil >  0.31 s -> spoof     (your_tts: native 16 kHz but 0.625 s padding)
```

The band-edge cue is an artifact of the pipeline, not of the generators. Medians of `bw60` after canonicalization:

| clips | bw60 median |
|---|---:|
| your_tts fakes, natively 16 kHz | 8000 Hz |
| lj_real (organizers' 16 kHz real) | **8000** Hz (p5 = 8000) |
| full LJSpeech, resampled by us from 22.05 kHz (stage 2) | **7859** Hz |
| LJ-voice DiffSSD fakes, resampled by us from 22.05 kHz | 7828–7859 Hz |
| external corpora shipped at 16 kHz (ASVspoof, In-the-Wild, LibriSeVoc, …) | 7484–8000 Hz, varies by corpus and class |

The same speaker, recorded the same way, lands on different sides of the split depending only on who resampled
it. The organizers' resampler keeps energy up to Nyquist, and soxr_hq's anti-alias filter does not. The HGT test clips
come at 16 kHz, most likely through the organizers' pipeline.

## 4. Recommendations for modeling (stage 6)

Apply every item to both classes, at model input, in training and inference alike.

1. **Low-pass everything at 7.5 kHz** (or randomize the cutoff between 7 and 8 kHz during training). This removes
   the band-edge cue, which is the single most dangerous one because it comes from our own pipeline and the test
   set will differ.
2. **Remove DC and high-pass at about 20–50 Hz.** This kills the per-generator DC signature (the best cue on all
   sources).
3. **Trim leading and trailing digital silence (|x| < 1e-4), then take random 3–4 s crops.** This neutralizes the
   padding and duration cues and matches the test duration distribution (median 3.41 s).
4. **Apply random gain and peak normalization.** This neutralizes the full-scale normalization of xtts_v2 and
   your_tts.
5. **Rerun `scripts/audit_shortcuts.py` on the model's actual input transform** and keep it as a regression check.
   After steps 1–4 the tree should sit near minDCF 1.0.
6. **Validate on `val_testlike` and the LOGO folds, not on plain `val`.** Plain `val` shares generators with train,
   so it overstates how well the model generalizes.

## Caveats

- On the given data alone the real class is 242 clips from one speaker. The given-only row has just 47 bonafide
  eval clips, so a minDCF of exactly 0.000 is noisy. The finding (cues separate the classes) is robust; the exact
  number is not.
- The sonar manifest listed 3,831 paths whose files were missing when this audit ran (stage 2 was still in
  progress), plus 8 duplicate rows. The shortcut audit skips missing files; `make_splits.py` drops duplicate paths
  with a warning.
