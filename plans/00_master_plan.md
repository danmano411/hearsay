# HEARSAY — Master Plan

Goal: score 1,671 test clips (16 kHz, >3 s, ~70% real) with a real-vs-synthetic score in [0,1]
(0 = real, 1 = synthetic) that minimizes **minDCF** as computed by the organizers' ASVspoof5 Track-1 scorer.

Grading: 60% minDCF · 20% creativity/depth · 20% GitHub documentation.

## Decisions (agreed with owner, 2026-09-25)

| Topic | Decision |
|---|---|
| Compute | Local CPU only (Ryzen 7840U, 16 threads, 60 GB RAM, no CUDA). DL = frozen pretrained SSL features + light heads; small CNNs only. |
| TTS sim | Open-source local TTS / vocoders only. ElevenLabs studied from public docs, not the API. |
| Stopping criterion | Held-out **minDCF ≤ 0.05** on the *test-like* validation set (70/30 real/fake, includes leave-out generators), **or** 3 consecutive model iterations improving < 0.005. |
| Test audio | **Inference only.** No normalization stats, pseudo-labels, or thresholds derived from HGT test audio. |
| Repo | Private GitHub repo `danmano411/hearsay`. Data, audio, and model weights are never committed. |

## Phases

| # | Phase | Plan | Output |
|---|---|---|---|
| 0 | Setup: layout, venv, repo, plans | this file | repo skeleton |
| 1 | Data cleaning + canonical dataset | [01_data_cleaning.md](01_data_cleaning.md) | `data/processed/`, `manifest.parquet` |
| 2 | External data scouting + download | [02_external_data.md](02_external_data.md) | `docs/external_data.md`, extra corpora |
| 3 | Synthetic speech sim (how TTS/cloning works) | [03_tts_sim.md](03_tts_sim.md) | `docs/synthetic_speech.md`, sim-generated fakes |
| 4 | Biology of real speech → measurable features | [04_speech_biology.md](04_speech_biology.md) | `docs/speech_biology.md`, `src/hearsay/features/bio.py` |
| 5 | Scoring analysis + local metric | [05_scoring.md](05_scoring.md) | `src/hearsay/metrics.py`, `docs/scoring.md` |
| 6 | Modeling ladder ML → DL → fusion | [06_modeling_ladder.md](06_modeling_ladder.md) | `reports/leaderboard.md`, submission TSV |

Phases 2, 3, 4, 5 run **in parallel** (independent). Phase 1 runs in parallel with them too; phase 6 waits for 1 + 5
and folds in 2/3/4 outputs as they land.

## Biggest risk: shortcut learning

The given real data is 242 clips of a **single speaker** (LJ), while fakes cover many speakers, generators, and codecs.
A naive model learns "sounds like Linda Johnson / LJ's microphone = real". Mitigations, in priority order:

1. More, diverse real speech (full LJSpeech, LibriSpeech speakers that DiffSSD clones, VCTK, etc.). See phase 2.
2. Channel-matched fakes: fakes of the *same* LJ voice and text (DiffSSD's LJ-trained models, sim vocoder resynthesis of LJ).
3. Symmetric augmentation: codec (mp3/opus), resampling, noise, gain are applied to **both** classes.
4. Validation that punishes shortcuts: leave-one-generator-out, and a test-like 70/30 mix.
5. Audit trivial cues (leading/trailing digital silence, duration, peak level, bandwidth) and neutralize them.

## Engineering conventions

- `src/hearsay/` package; `scripts/` CLIs; everything reproducible from `data/raw` via scripts.
- Canonical audio: 16 kHz, mono, PCM16 WAV. Manifest = one parquet row per clip.
- Branch per phase (`phase/1-data`, …), PR into `main`, merge after the phase's check passes.
- Each experiment appends one row to `reports/leaderboard.md` (model, features, val sets, minDCF both conventions, EER).
- Deliberate shortcuts are marked with `ponytail:` comments.
