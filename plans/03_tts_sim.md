# Phase 3 — Synthetic speech sim ("how is fake speech actually made?")

Owner decision: open-source local only; ElevenLabs is studied from public docs/papers, not the API.

## Understand (→ `docs/synthetic_speech.md`)
- The modern pipeline: text normalization → G2P/phonemes → acoustic model (Tacotron/FastSpeech/VITS/diffusion/
  flow-matching/codec-LM) → mel or codec tokens → vocoder (Griffin-Lim, WaveNet, HiFi-GAN, BigVGAN, WaveGrad/DiffWave)
  → post-processing (loudness norm, codec).
- Voice cloning: speaker encoders (d-vector/ECAPA), zero-shot prompting (XTTS, YourTTS, OpenVoice), codec LMs (VALL-E style).
- What ElevenLabs publicly discloses (model families, latency tiers, output formats: mp3 44.1 kHz, pcm) and what can be inferred.
- Map each DiffSSD generator to its architecture family.
- **Artifact catalogue**: for each pipeline stage, which fingerprints it leaves (vocoder upsampling periodicity/checkerboard,
  phase incoherence, over-smooth high band, missing breath/micro-prosody, band-limit at native SR, codec traces).

## Build (`src/hearsay/sim/`, `scripts/run_sim.py`)
1. **Vocoder resynthesis of real LJ** (copy-synthesis): real wav → mel → {Griffin-Lim, a HiFi-GAN/BigVGAN checkpoint}.
   Same speaker, same text, same channel → the *only* difference is the vocoder. Best possible hard negatives.
2. **Full TTS** of LJ-style text with ≥ 2 open models runnable on CPU (e.g. Piper, Kokoro, MMS-TTS / SpeechT5).
3. Ablation knobs recorded in manifest (`generator=sim_<model>_<vocoder>`): vocoder, sample rate, codec.
4. Figures: spectrogram / phase / modulation-spectrum comparisons real vs each sim stage → `docs/figures/`.

Size cap: ~5k sim clips (CPU time). All output goes through the Phase-1 cleaning path.
