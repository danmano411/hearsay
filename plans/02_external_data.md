# Phase 2 — External data (HuggingFace + elsewhere)

Challenge rules allow any dataset we are permitted to use. Disk budget: **≤ 40 GB** extra (109 GB free).

## Priorities
1. **Real speech diversity** (our weakest class): full LJSpeech (13,100 clips; same speaker as given reals),
   LibriSpeech dev/test-clean (includes speakers DiffSSD cloned → matched real/fake pairs), VCTK, Common Voice EN subset.
2. **Fakes from generators not in DiffSSD**, esp. LJ-voice fakes: WaveFake (LJ vocoder copies), ASVspoof 2019 LA,
   ASVspoof5 subset, MLAAD (English), In-the-Wild, Fake-or-Real (FoR), LibriSeVoc.
3. Noise / RIR corpora for augmentation (MUSAN, RIRs) — only if small.

## Method
- Research agent builds `docs/external_data.md`: name, URL, license, size, #real/#fake, generators, sample rate, why useful, verdict.
- Download only verdict=USE sets, capped by budget; subsample big ones (e.g. ≤ 20k clips each).
- Each dataset gets a loader in `scripts/ingest_<name>.py` that writes into the Phase-1 canonical format + manifest rows
  (`source=<name>`), so external data goes through the same cleaning/audit.
- Record provenance + license in the manifest and in the doc. Nothing with a no-redistribution or unclear-license flag
  gets mirrored in the repo (we never commit audio anyway).
