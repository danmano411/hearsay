# Phase 4 — Biology of real speech → measurable features

## Understand (→ `docs/speech_biology.md`)
Source–filter model and the physiology behind it, each item tied to a *measurable* signal property and a hypothesis about
how synthesis gets it wrong:

| Process | Physiology | Measurable | Synthetic-speech hypothesis |
|---|---|---|---|
| Respiration | lungs, subglottal pressure, breath groups | breath noises, amplitude declination over a phrase | breaths absent or pasted; flat declination |
| Phonation | vocal-fold vibration (myoelastic-aerodynamic), glottal pulse shape | F0, jitter, shimmer, HNR, CPP, H1–H2, glottal open quotient | too-regular or random jitter; wrong H1–H2 / spectral tilt |
| Turbulence | aspiration, fricatives | high-band noise structure (4–8 kHz) | over-smooth / noisy high band from vocoder |
| Articulation | tongue/jaw/lips, coarticulation, finite muscle speed | formant trajectories + velocities, VOT | physically implausible formant jumps |
| Resonance | vocal-tract length, nasal coupling | formant spacing ↔ VTL consistency | VTL inconsistent across utterance |
| Neuro-motor | ~4–8 Hz syllabic rhythm, micro-prosody | amplitude-modulation spectrum, F0 micro-perturbations after obstruents | missing micro-prosody |
| Radiation/room | lip radiation (+6 dB/oct), room acoustics | phase coherence, reverberation tail | phase artifacts from vocoders; dry/mismatched room |

## Build (`src/hearsay/features/bio.py`)
Per-clip feature vector via Praat (parselmouth) + librosa: F0 stats, jitter (local, rap), shimmer (local, apq3),
HNR, CPPS, H1–H2, spectral tilt, formant F1–F3 means/velocity stats, modulation-spectrum bands, breath/pause stats,
phase features (group delay stats). Cached to `data/features/bio.parquet`. Unit check on a synthetic vowel with known jitter.

Result feeds the Phase-6 classic-ML rung and an interpretability section (which biology cues separate real/fake).
