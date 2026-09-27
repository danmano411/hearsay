# External data (Stage 2)

**Problem this phase attacks.** The organizer training data is badly lopsided: **242 real clips from one
speaker (LJ Speech)** against **70,000 fakes** from 10 generators and ~11 voices (DiffSSD). The test set is
~70 % real. A model trained only on that learns *"sounds like LJ's voice/microphone ⇒ real, everything else ⇒ fake"*,
which is the worst possible prior for a real-heavy test set. External data has to (1) make the real class
diverse, (2) break the voice shortcut by adding *fakes of the LJ voice* and *reals of the voices DiffSSD clones*,
and (3) add generators DiffSSD lacks.

All ingests are reproducible scripts (`scripts/ingest_*.py`) that write canonical clips
(16 kHz mono PCM16 WAV, ≥ 3.0 s, via `src/hearsay/audio.py`) to `data/processed/<source>/` plus
`data/processed/manifests/<source>.parquet` with the shared `MANIFEST_COLUMNS` (incl. `license`).
Nothing under `data/` is committed.

## 1. What we ingested

| source | bonafide | spoof | hours | spoof generators | speakers | GB on disk |
|---|---:|---:|---:|---:|---:|---:|
| ljspeech | 11,945 | 0 | 22.9 | 0 | 1 (LJ) | 2.64 |
| librispeech | 16,365 | 0 | 52.4 | 0 | 1,001 | 6.03 |
| wavefake | 0 | 11,256 | 20.6 | 8 | 1 (LJ) | 2.38 |
| in_the_wild | 6,030 | 5,179 | 19.3 | 1 (unlabelled) | 54 | 2.23 |
| asvspoof2019_la | 4,329 | 3,460 | 9.1 | 13 | 67 | 1.05 |
| librisevoc | 2,641 | 10,000 | 33.4 | 6 | 207 | 3.85 |
| asvspoof5 | 2,777 | 10,773 | 26.6 | 16 | 730 | 3.07 |
| sonar | 2,274 | 1,565 | 8.0 | 8 | n/a | 0.92 |
| dfadd | 458 | 2,373 | 3.5 | 5 | n/a | 0.41 |
| mlaad_tiny | 5,714 | 5,796 | 24.4 | 64 | n/a | 2.82 |
| cvoicefake_en | 4,187 | 3,897 | 13.1 | 5 | n/a | 1.51 |
| **total external** | **56,720** | **54,299** | **233** | **126** | | **26.9** |

**Class balance.** The given data is 242 real vs 70,000 fake, so 0.3 % real. With the external data it is
**56,962 real vs 124,299 fake (31 % real)**. The external part alone is balanced (51 % real). Stage 6 can move the
ratio toward the test's ~70 % by subsampling DiffSSD fakes or weighting classes. It no longer has to up-weight 242 clips of one voice.

Key counts behind the shortcut fix:
- **LJ voice**: 12,187 real (11,945 new + 242 given) vs 11,256 WaveFake + 20,000 DiffSSD LJ-model fakes.
  The LJ voice is no longer a real-only cue.
- **DiffSSD-cloned voices**: 1,128 real LibriSpeech utterances (103–129 per speaker for all 10 speakers) vs
  their DiffSSD clones.

Disk: 26.9 GB of canonical audio. Raw leftovers are about 28 MB (protocol files, logs), because every raw shard and
download was deleted after its ingest (logged in `data/external/logs/`). The 40 GB budget is not exceeded.

Each row can be re-checked with `python scripts/ingest_summary.py`. It asserts the schema, the labels,
≥ 3 s durations, a filled license, unique paths that exist on disk, and zero overlap with `data/raw/lj_real`.

### How each set counters the shortcut

| Need | Sets | Why it matters here |
|---|---|---|
| Same voice, real | `ljspeech` (every LJSpeech clip except the 242 given ids) | Grows the only real voice we have ~50×, so "LJ" stops being rare. It is also the real counterpart of DiffSSD's LJ-trained generators (Grad-TTS, DiffGAN-TTS, ProDiff, WaveGrad 2). |
| Same voice, fake | `wavefake` (7 vocoders + 1 TTS, all in the LJ voice, same LJ sentence ids) | Matched pairs: identical speaker, text and recording chain, differing only in the synthesis. This directly removes "LJ timbre ⇒ real". |
| Cloned voices, real | `librispeech`: **all** utterances of the 10 speakers DiffSSD clones (100, 1487, 2061, 3654, 4490, 5448, 6167, 6575, 7995, 8848) | The real side of DiffSSD's voice-cloning generators (XTTS v2, YourTTS, OpenVoice v2, UnitSpeech, PlayHT, ElevenLabs). Without it those 10 voices are *only ever fake*. |
| Real speaker/channel diversity | `librispeech` (dev/test-clean + 12 utts from each train-clean-360 speaker), `in_the_wild` (celebrity speech from the web), `asvspoof2019_la` (VCTK speakers), `asvspoof5` (MLS, crowdsourced, with codecs), `cvoicefake_en` (Common Voice, consumer mics), `sonar` (LibriTTS), `mlaad_tiny` (M-AILABS audiobooks), `librisevoc` (LibriTTS), `dfadd` (VCTK) | Test reals probably are not all LJ. Many voices, microphones and codecs on the real side force the model to learn artifacts, not identity. |
| New generator families | ASVspoof 2019 A07–A19, ASVspoof 5 A17–A32, SONAR (OpenAI TTS, VALL-E, VoiceBox, NaturalSpeech 3, …), MLAAD-tiny (64 English systems from 2023–25), DFADD (diffusion / flow matching), LibriSeVoc + CVoiceFake (vocoder resynthesis), In-the-Wild (unknown in-the-wild fakes) | Unseen-generator robustness. Leave-one-generator-out validation (Stage 1/6) needs many generators to hold out. |

## 2. Candidate survey

Sizes are for the full upstream release. "Real/fake" counts are upstream counts, not what we kept.

| Dataset | URL | License | Size | Real / fake | Generators | SR | Access | Verdict |
|---|---|---|---|---|---|---|---|---|
| LJSpeech 1.1 | [keithito.com](https://keithito.com/LJ-Speech-Dataset/) | Public domain | 2.6 GB | 13,100 / 0 | – | 22.05 kHz | open | **USE**: the given real voice |
| LibriSpeech | [openslr.org/12](https://www.openslr.org/12) | CC BY 4.0 | 60 GB (all) | 292k / 0 | – | 16 kHz | open | **USE**: the 10 DiffSSD-cloned speakers + diversity |
| WaveFake | [zenodo 5642694](https://zenodo.org/records/5642694) | CC BY-SA 4.0 | 28.9 GB zip | 0 / 117,985 | MelGAN, MelGAN-L, FB-MelGAN, MB-MelGAN, PWG, WaveGlow, HiFi-GAN, Conformer+PWG TTS (+2 JSUT) | 22.05 kHz | open | **USE**: LJ-voice fakes |
| In-the-Wild | [paper](https://arxiv.org/abs/2203.16263), [HF repack](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/InTheWild) | CC BY-SA 4.0 upstream ([HF](https://huggingface.co/datasets/mueller91/In-The-Wild)); Apache-2.0 repack | 8.2 GB / 2.3 GB | 19,963 / 11,816 | unknown (web deepfakes) | 16 kHz | open | **USE**: most realistic channel diversity |
| ASVspoof 2019 LA | [datashare 10283/3336](https://datashare.ed.ac.uk/handle/10283/3336), [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA) | ODC-By 1.0 | 7.6 GB | eval 7,355 / 63,882 | A07–A19 (TTS, VC, vocoders) | 16 kHz | open | **USE** (eval split, attack ids from official protocol) |
| ASVspoof 5 | [zenodo 14498691](https://zenodo.org/records/14498691), [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5) | ODC-By 1.0 | 142 GB | eval 138,688 / 542,086 | A17–A32 + 11 codec conditions | 16 kHz | open | **USE (4 of 200 shards)**: our metric's home corpus |
| LibriSeVoc | [github](https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts), [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/LibriSeVoc) | CC BY-SA 4.0 | 3.2 GB (test) | 2,641 / 15,846 | DiffWave, MelGAN, PWG, WaveGrad, WaveNet, WaveRNN | 24→16 kHz | open | **USE**: matched real/vocoded pairs |
| SONAR | [github](https://github.com/Jessegator/SONAR), [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/SONAR) | CC BY-NC 4.0 | 0.5 GB | 2,274 / 1,674 | OpenAI TTS, xTTS, FlashSpeech, VoiceBox, AudioGen, VALL-E, NaturalSpeech 3, PromptTTS 2 | 16 kHz | open | **USE**: newest commercial/SOTA systems |
| DFADD | [paper](https://arxiv.org/abs/2409.08731), [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/DFADD) | MIT | 0.5 GB (test) | VCTK / 3,000 | Grad-TTS, Matcha-TTS, NaturalSpeech 2, StyleTTS 2, P-Flow | 16 kHz | open | **USE**: diffusion/flow TTS like DiffSSD's, on other voices |
| MLAAD-tiny | [HF](https://huggingface.co/datasets/mueller91/MLAAD-tiny) | CC BY-NC 4.0 | 3.5 GB | ~6k / ~6.4k (all langs) | ~78 English TTS systems (64 survive the 3 s filter) | mixed | open | **USE (English only)** |
| CVoiceFake (small) | [paper](https://arxiv.org/abs/2409.09272), [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CVoiceFake_small) | CC BY 4.0 | 4.6 GB | 23,544 / 114,592 (en 4,315 / 21,438) | Griffin-Lim, WORLD, PWG, MB-MelGAN, Style-MelGAN | 16 kHz mp3 | open | **USE (English)**: gives Common Voice reals without the gate |
| MLAAD (full) | [HF](https://huggingface.co/datasets/mueller91/MLAAD) | CC BY-NC 4.0 | 35 GB | M-AILABS / ~100k+ | 100+ TTS, 40 langs | mixed | HF click-through gate | MAYBE: owner can accept the gate; tiny covers the same systems |
| Common Voice (EN) | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/) | CC0 | ~80 GB (en) | millions / 0 | – | 48 kHz mp3 | account / gated | MAYBE: covered by CVoiceFake-en reals |
| VCTK 0.92 | [datashare 10283/3443](https://datashare.ed.ac.uk/handle/10283/3443) | CC BY 4.0 | 10.9 GB | 44k / 0 | – | 48 kHz | open | SKIP: most clips < 3 s; VCTK reals already arrive via ASVspoof 2019 + DFADD |
| Fake-or-Real (FoR) | [York BIL](https://bil.eecs.yorku.ca/datasets/), [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) | not stated | ~16 GB (norm) | ~111k / ~87k | DeepVoice 3, Google/Azure/Polly/Baidu TTS (33 voices) | mixed | Kaggle login | MAYBE: no explicit license; owner decision |
| CodecFake | [HF](https://huggingface.co/datasets/rogertseng/CodecFake) | CC BY 4.0 | 102 GB | VCTK / 700k | codec resynthesis (EnCodec, DAC, SpeechTokenizer, …) | 16 kHz | open | MAYBE: over budget; one shard would add codec-LM artifacts if Stage 6 finds ElevenLabs/codec fakes hard |
| ASVspoof 2021 DF | [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF) | ODbL | 34 GB | 14,869 / 519,059 | 2019 attacks + 100+ vocoders, compression codecs | 16 kHz | open | MAYBE: largely redundant with 2019 LA + ASVspoof 5 |
| EmoSpoof-TTS | [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/EmoSpoofTTS) | CC BY 4.0 | 3.4 GB | 0 / 36,000 | StyleTTS 2, F5-TTS, CosyVoice (emotional) | 16 kHz | open | MAYBE: spoof-only; adds to the imbalance |
| ODSS | [HF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ODSS) | CC BY-SA 4.0 | 2.6 GB | Hi-Fi TTS etc. / VITS, FastPitch | VITS, FastPitch+HiFi-GAN | 16 kHz | open | MAYBE: 2 generators, 2/3 non-English |
| MUSAN / RIRs | [openslr 17](https://www.openslr.org/17), [28](https://www.openslr.org/28) | CC BY 4.0 / Apache | 11 GB / 1.3 GB | noise/RIR | – | 16 kHz | open | MAYBE: augmentation is Stage 6's call; the RIR set is small enough to add then |

## 3. Method details and deviations

- **Streaming, no raw copies where possible.** LJSpeech (tar.bz2) and LibriSpeech (tar.gz) are decoded straight from
  the HTTP stream into canonical clips. WaveFake's 28.9 GB zip is never downloaded: we read its central directory
  over HTTP Range requests and fetch only the sampled members (`HttpRangeFile`, `zip_member_bytes` in
  `scripts/ingest_common.py`, tested in `tests/test_ingest.py`). HF parquet shards are downloaded to
  `data/external/<name>/`, converted, then **deleted** (each deletion is logged in `data/external/logs/hf_sasb.log`).
- **Deviation: LibriSpeech subset.** The task brief said the DiffSSD speakers are in train-clean-100.
  LibriSpeech's `SPEAKERS.TXT` puts **all ten in train-clean-360**, so we stream that 23 GB tarball and keep every
  one of their utterances. Other speakers are capped at 12 utterances each (the first 12 in archive order).
- **Deviation: WaveFake sampling is by archive block.** Zenodo returned HTTP 429 after a few hundred single-file
  requests, so each generator takes one contiguous block of members. Member order inside a folder is unrelated to
  LJ ids, so a block is still a scattered id sample. It is fetched in ~36 MB ranged reads. Requests drop from ~12k to ~110.
- **Caps (seeded, `random.Random(0)`) are applied before the ≥ 3 s filter.** The filter then removes short clips.
  This matters most for In-the-Wild and ASVspoof 2019, whose clips are often 2–4 s (see "kept" vs "selected" in the logs).
- **ASVspoof 5**: shards 0, 66, 133, 199 of the 200-shard eval repack. Rows are ordered by utterance id, which is not
  tied to attack, so this is a random ~2 % sample. The ASVspoof 5 codec condition (C01–C11) is kept in `codec_orig`
  (e.g. `flac;asv5_C08`) so Stage 6 can study codec robustness.
- **Generator naming**: `bonafide` for all reals. Fakes are `<source>_<upstream attack/system id>`, e.g.
  `asvspoof5_A19`, `wavefake_ljspeech_hifiGAN`, `mlaad_f5-tts`. In-the-Wild fakes have no generator label (`in_the_wild_unknown`).
- **Speaker ids** are upstream ids (LibriSpeech numeric ids like `100` match DiffSSD's `speaker_100` folders;
  LJ is `LJ`). `text_id` is the upstream utterance id. For WaveFake/LibriSeVoc it is the id of the *real* utterance
  that was resynthesised, so matched pairs can be grouped into the same split.
- **Bug caught by the summary check**: SONAR file stems contain dots (`..._valle.0`), and the first version of
  `out_path` used `with_suffix`, so 8 VALL-E pairs collided on one output file. `ingest_summary.py` asserts unique
  paths and caught it. `out_path` now appends `.wav`, which has a regression test, and SONAR was re-ingested cleanly.
- **Clips < 3 s dropped** (the test clips are > 3 s): LJSpeech 913 of 12,858. In-the-Wild kept 11,209 of 20,000 selected.
  ASVspoof 2019 kept 7,789 of 15,355. WaveFake kept 11,256 of 12,500.
- `sr_orig` for the HF repacks is the repack's rate (16 kHz), not necessarily the original recording rate.

## 4. Recommendations for Phases 1 and 6

1. **Split by `text_id` + `speaker`**, not by clip. WaveFake/LJSpeech and LibriSeVoc share utterances across
   classes, and the pairs must stay in the same split or the model can memorise sentences.
2. **Keep a few sets out of training as an external OOD check.** Our choice: **SONAR** (newest commercial
   systems) and **In-the-Wild** (real-world channels). They were built as evaluation benchmarks, and a held-out score on them
   is the best proxy we have for unknown test generators.
3. **Balance by source when sampling batches.** Per-source weights keep one corpus (e.g. LibriSpeech reals, DiffSSD fakes) from defining a class.
4. **Licenses**: SONAR and MLAAD-tiny are CC BY-NC. That is fine for this non-commercial challenge entry, but any
   model trained on them inherits the non-commercial restriction. Everything else is PD / CC BY / CC BY-SA / ODC-By / MIT / Apache.
   No audio is redistributed by this repo.

## 5. Reproduce

```bash
PY=.venv/Scripts/python
$PY scripts/ingest_ljspeech.py          # ~40 min, streams 2.6 GB
$PY scripts/ingest_librispeech.py       # streams dev/test-clean + 23 GB train-clean-360
$PY scripts/ingest_wavefake.py          # ~4 GB of ranged reads from Zenodo
$PY scripts/ingest_hf_sasb.py sonar dfadd in_the_wild asvspoof2019_la librisevoc asvspoof5 cvoicefake_en
$PY scripts/ingest_mlaad_tiny.py
$PY scripts/ingest_summary.py           # tables above + schema/path/overlap assertions
PYTHONPATH=src $PY -m pytest -q tests/test_ingest.py
```

All scripts resume: an existing canonical clip is reused and only its header is read. Every script uses ≤ 4 worker processes.
