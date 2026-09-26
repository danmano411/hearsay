# Dataset: canonical clips, manifest, and splits

This page covers how raw audio becomes `data/processed/manifest.parquet`, and how that manifest is split so that
validation numbers mean something. The audit findings (formats, dropped clips, shortcut cues) are in
[reports/data_audit.md](../reports/data_audit.md).

## Pipeline

```
data/raw/diffssd, data/raw/lj_real ─ audit_raw.py ──► data/processed/audit/raw_audit.parquet   (native-rate stats)
                                   └ clean_given.py ─► data/processed/{diffssd,lj_real}/**.wav   (16 kHz mono PCM16)
                                                      data/processed/manifests/{diffssd,lj_real}.parquet
other phases ───────────────────────────────────────► data/processed/manifests/<source>.parquet
data/processed/manifests/*.parquet ─ make_splits.py ─► data/processed/manifest.parquet  (+ split columns)
                                   ─ check_manifest.py  (assertions)
                                   ─ audit_shortcuts.py (trivial-cue tree → reports/data_audit.md)
```

```bash
export PYTHONPATH=src   # run from the repo root with the project venv
python scripts/audit_raw.py        # minutes on 8 workers
python scripts/clean_given.py      # minutes on 8 workers; writes ~17 GB of WAV (69,692 clips, ~151 h)
python scripts/make_splits.py      # seconds; rerun whenever any source manifest changes
python scripts/check_manifest.py   # opens every clip; --sources diffssd lj_real to check a subset
python scripts/audit_shortcuts.py
pytest tests/test_data_splits.py
```

## What the given data actually is

We read the DiffSSD paper and its dataset card rather than guessing
(Bhagtani et al., *DiffSSD: A Diffusion-Based Dataset for Speech Forensics*, arXiv
[2409.13049](https://arxiv.org/abs/2409.13049); data at
[huggingface.co/datasets/purdueviperlab/diffssd](https://huggingface.co/datasets/purdueviperlab/diffssd),
CC BY-NC-ND 4.0). The metadata CSVs from that repository are stored in `data/raw/diffssd_meta/`.

| question | answer | evidence |
|---|---|---|
| Which generators speak in the LJ voice? | `diffgan_tts`, `grad_tts`, `pro_diff`, `wavegrad2`: pre-trained on LJSpeech, flat folder, `sentence_0..4999` | paper §3; folder layout |
| Which generators clone voices? | `elevenlabs`, `openvoicev2`, `playht`, `unit_speech`, `xtts_v2`, `your_tts`: zero-shot cloning of 10 LibriSpeech speakers (5 F, 5 M); `speaker_NNN` is the LibriSpeech speaker id; 500 sentences per speaker | paper §3; folder layout |
| Is `sentence_N` an LJSpeech transcript? | **No.** The 5,000 lines were written with ChatGPT-3.5 (conversation, quotes, weather, news, …). Real LJ clips therefore never share text with DiffSSD fakes | paper §3; `text_script_0_4999.csv` |
| Is `sentence_N` the same text for every generator and speaker? | **Yes.** `text_script_0_499.csv` (zero-shot) equals the first 500 rows of `text_script_0_4999.csv` (pre-trained), checked row by row | `diffssd_meta/*.csv` |
| What are openvoicev2's 5 files per sentence? | The same text and target speaker, rendered with five accent styles (`en-au`, `en-br`, `en-default`, `en-india`, `en-us`); none are exact duplicates | file names; PCM hash check |
| Which generators does DiffSSD treat as unseen? | `diffgan_tts`, `playht`, `unit_speech` appear only in the official *test* set | `train_val_test_splits.csv` |
| What is `lj_real`? | 242 LJSpeech 1.1 utterances (ids `LJ001-0001` …), delivered at 16 kHz | file names; audit |

Native rates and codecs per generator are listed in the audit report.

## Canonical clip and manifest

Each clip is stored as a 16 kHz, mono, PCM16 WAV at `data/processed/<source>/<original relative path>.wav`. It is
at least 3.0 s long, is **not** trimmed, and is **not** loudness-normalized. Neutralizing cues is left to the model
input transform on purpose, because the audit shows those cues are strong.

Every source manifest (`data/processed/manifests/<source>.parquet`) has exactly `hearsay.audio.MANIFEST_COLUMNS`:

| column | given-data convention |
|---|---|
| `path` | relative to the repo root, forward slashes |
| `label` | `bonafide` or `spoof` |
| `source` | `diffssd` or `lj_real` |
| `generator` | DiffSSD folder name, or `bonafide` |
| `speaker` | `ljspeech:LJ` (real LJ **and** the 4 LJ-voice generators) or `librispeech:<id>` |
| `text_id` | `diffssd:<N>` (shared across all generators and speakers) or `ljspeech:<LJ id>` |
| `duration`, `sr_orig`, `codec_orig`, `license` | measured values; `codec_orig` is e.g. `wav/pcm_16` or `mp3/mpeg_layer_iii` |

`make_splits.py` concatenates every source manifest and adds the columns below.

## Split scheme (`src/hearsay/data/splits.py`)

**Leakage unit (`group`).** Rows are grouped by `text_id`. All renderings of one sentence therefore stay in the same
split: every generator, every cloned speaker, and every accent. This guarantees that no `(speaker, text_id)` pair
appears in two splits. It also keeps a real LJ recording and any LJ-voice fake of the same sentence together.
LJSpeech ids are normalized first, so `LJ001-0001` (phase 3's sim fakes) and `ljspeech:LJ001-0001` (our real
clips) fall in one group.

Some external corpora use a per-file utterance id as `text_id`: `in_the_wild`, `asvspoof2019_la`, `asvspoof5` and
`librisevoc`. For these the group is the **speaker**, so one voice and its recording channel never sit on both
sides. Rows with neither a `text_id` nor a speaker are grouped by file.

| column | rule |
|---|---|
| `split` | `sha1(group) mod 100`: 0–79 → `train`, 80–89 → `val`, 90–99 → `test_internal` |
| | Rows from the **held-out generators** (`diffgan_tts`, `playht`, `unit_speech`, as in DiffSSD's own protocol) that hash to train become `excluded`. They are never trained on, and their groups are not split. This costs 11,990 fakes, so 3 of 10 DiffSSD generators stay truly unseen. |
| `logo_fold` | spoof: its own `generator`. bonafide: spread over the generator names by `sha1("logo"+group)`. Fold *g* = train on `logo_fold != g`, evaluate on `logo_fold == g`, so each fold evaluates one unseen generator plus a disjoint slice of real clips. Use it within `split != test_internal`. |
| `val_testlike`, `test_internal_testlike` | All bonafide rows of that split, plus spoof rows so that spoof is **30%** (the HGT test is ~70% real). Spoof rows are taken from held-out generators first (ordered by `sha1(path)`), then from the rest. |

**Why hashes and not a seeded shuffle.** A row's split depends only on its own group string. Rerunning the script is
byte-identical, and when another phase adds a source, existing rows never move between splits.

**Caveat on LOGO.** LOGO folds share sentences across folds, because every generator speaks the same 5,000 lines.
That is text overlap, not voice or channel overlap, and text content is not a real-vs-fake cue. If it ever matters,
restrict LOGO evaluation to one `split`.

### Current contents (2026-09-25, all sources present at the time)

| split | bonafide | spoof | total |
|---|---:|---:|---:|
| train | 22,830 | 70,855 | 93,685 |
| val | 3,705 | 9,600 | 13,305 |
| test_internal | 4,161 | 10,370 | 14,531 |
| excluded | 0 | 11,990 | 11,990 |
| val_testlike | 3,627 | 1,554 | 5,181 |

On the given data alone (DiffSSD + lj_real), bonafide is 195 / 18 / 29 across train / val / test_internal. That
leaves `val_testlike` with only 18 real clips, which is why the external real corpora matter.

### Frozen eval subsets (2026-09-26)

`split` is stable when sources are added, but the test-like subsets are re-sampled: adding the 5,189 finished sim
clips moved 65 rows in or out of `val_testlike` and 86 of `test_internal_testlike`, which would make new scores
incomparable with the leaderboard. Since then `make_splits.py` reads `data/processed/eval_subsets_frozen.parquet`
(`path, val_testlike, test_internal_testlike`, snapshot of the 180,726-row manifest every leaderboard entry was scored
on) and keeps those two columns exactly; rows added later are never in them. `val_testlike` = 8,963 rows
(6,274 / 2,689), `test_internal_testlike` = 9,747 rows (6,823 / 2,924). `logo_fold` is not frozen (not used for
headline numbers).

## Checks

`scripts/check_manifest.py` asserts each of the following, then prints class counts per split and per generator:

- every path exists and is 16 kHz / mono / PCM16 / ≥ 3.0 s
- labels are valid and paths are unique
- no `group` and no `(speaker, text_id)` spans splits (`excluded` counts as train's bucket)
- held-out generators never appear in `train`
- the test-like subsets are 30 ± 5% spoof

On 2026-09-25 it passed on `--sources diffssd lj_real`: 69,692 clips and 5,150 groups, every file opened. On all
sources it failed, correctly, because 3,831 sonar paths were not yet on disk (see the audit caveats).
`tests/test_data_splits.py` checks the split invariants on a synthetic manifest, including the cross-source LJ id
case and speaker grouping.

## Adding a source

1. Write `data/processed/manifests/<source>.parquet` with `MANIFEST_COLUMNS`.
2. Fill `text_id` when the spoken sentence is known. For LJSpeech sentences, use the LJ id so real and fake copies
   group together.
3. If the corpus is multi-speaker with per-file ids, add it to `SPEAKER_GROUPED_SOURCES`.
4. Run `make_splits.py`, then `check_manifest.py`.
