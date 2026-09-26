# Error analysis — where the models fail

Source: `hearsay.evaluate.per_source()` on `test_internal_testlike` (frozen, 6,823 real / 2,924 fake). For a *source*,
the row scores that source's real clips against all fakes; for a *generator*, that generator's fakes against all reals.
Higher minDCF = that slice is where the errors come from.

> **Which set drives decisions.** Anything that changes training (sampling weights, augmentation, model choice) is
> derived from **`val_testlike`** only. The `test_internal_testlike` tables further down are report-only, for judges.
> Deriving training choices from the headline set would leak into the headline. The first draft of this report did
> that (a ×2 weight for LibriSeVoc reals), and the `val_testlike` table below shows it was an artifact.

## Decision table: R1_lgbm_all_full on `val_testlike` (6,274 real / 2,689 fake)

| real source (vs all fakes) | n real | combined | EER % |
|---|---|---|---|
| **asvspoof5** | 266 | **0.501** | 14.4 |
| asvspoof2019_la | 559 | 0.278 | 8.0 |
| librisevoc | 251 | 0.245 | 6.8 |
| cvoicefake_en | 422 | 0.238 | 6.2 |
| lj_real | 18 | 0.233 | 5.6 |
| librispeech | 1,621 | 0.227 | 6.1 |
| ljspeech | 1,167 | 0.187 | 5.1 |
| sonar | 222 | 0.154 | 4.4 |
| in_the_wild | 1,179 | 0.136 | 4.2 |
| mlaad_tiny | 526 | 0.095 | 2.7 |

| fake generator (vs all reals), n ≥ 100 | n fake | combined |
|---|---|---|
| playht (held out of train) | 470 | **0.302** |
| unit_speech (held out) | 470 | 0.220 |
| openvoicev2 | 281 | 0.004 |
| diffgan_tts (held out) | 500 | 0.000 |

Decisions taken from it: upweight **asvspoof5 reals ×2** in G5 sampling (the only clear outlier); no special weight for
LibriSeVoc; success = moving playht / unit_speech and the asvspoof5 real slice.

## Report-only: R1_lgbm_all_full on `test_internal_testlike` (headline 0.253)

**Real clips that look fake** (combined minDCF, source's reals vs all fakes):

| source (real clips) | n real | combined | EER % |
|---|---|---|---|
| asvspoof5 | 234 | **0.550** | 15.8 |
| librisevoc | 276 | **0.398** | 12.7 |
| cvoicefake_en | 425 | 0.301 | 8.0 |
| asvspoof2019_la | 829 | 0.242 | 6.5 |
| librispeech | 1,669 | 0.235 | 6.2 |
| ljspeech | 1,166 | 0.215 | 5.4 |
| in_the_wild | 1,342 | 0.175 | 4.8 |
| mlaad_tiny | 568 | 0.110 | 3.2 |

**Fakes that pass as real** (≥ 100 clips, or notable):

| generator | n fake | combined | note |
|---|---|---|---|
| playht (DiffSSD, held out of training) | 489 | **0.346** | commercial, unseen |
| unit_speech (DiffSSD, held out) | 490 | 0.263 | diffusion + unit-based, unseen |
| librisevoc_wavernn | 31 | 0.635 | autoregressive neural vocoder |
| in_the_wild_unknown | 49 | 0.315 | real-world deepfakes, unknown tools |
| mlaad codec-LM TTS (MegaTTS3, VibeVoice, MiniMax, Dia) | 1–3 each | 0.5–0.9 | tiny n, directionally worrying |

## What it tells us

1. **Channel, not speech.** The worst real slice on both sets is ASVspoof 5 bona fide, whose *recording chain* differs
   (heavily processed audiobook speech). Classic spectral features partly learn the
   corpus, so reals from an unusual chain look fake. Expect the same risk on the HGT test if its reals come from a chain
   we lack. Hence the class-symmetric codec/noise/resampler augmentation, and why SSL front ends, which model speech
   rather than the channel, should help most here.
2. **Unseen generators are the real test.** The two largest error sources are DiffSSD generators held out of training
   (PlayHT, UnitSpeech): they are exactly what the challenge brief warns about. Held-out generators stay held out; this is
   the number to move.
3. **Newest TTS (codec language models) is the frontier.** Few clips, but consistently the hardest. More of them in
   training (full MLAAD is gated; see `docs/external_data.md`) would be the next data investment.

Next: the same table for each GPU rung, to see which model fixes which slice; that decides fusion weights more
honestly than the aggregate number.
