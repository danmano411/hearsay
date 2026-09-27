# Scripts, in the order the pipeline runs

Run from the repo root with `PYTHONPATH=src` (see the main README). Stage numbers match
[`docs/00_development_log.md`](../docs/00_development_log.md).

## Stage 1: clean and audit the given data
| script | what it does |
|---|---|
| `audit_raw.py` | Audits the raw files at their native rate and format: decoding, duration, level, silence, band edges. |
| `clean_given.py` | Converts DiffSSD and the given LJ reals into canonical 16 kHz clips plus a manifest. |
| `audit_shortcuts.py` | Shortcut audit: can a depth-3 tree on trivial cues separate real from fake? |
| `check_manifest.py` | Asserts the combined manifest is sound. |

## Stage 2: external data
| script | what it does |
|---|---|
| `ingest_ljspeech.py`, `ingest_librispeech.py`, `ingest_wavefake.py`, `ingest_mlaad_tiny.py` | One corpus each → canonical clips + manifest. |
| `ingest_hf_sasb.py` | Corpora from the HuggingFace `SpeechAntiSpoofingBenchmarks` repacks. |
| `ingest_common.py` | Shared helpers for the ingest scripts. |
| `ingest_summary.py` | Summary tables and sanity checks for `docs/03_external_data.md`. |
| `make_splits.py` | Unions all manifests, assigns grouped splits, and applies the frozen evaluation subsets. |

## Stage 3: synthetic-speech simulator
| script | what it does |
|---|---|
| `run_sim.py` | Generates the 5,204 simulated fakes (copy-synthesis vocoders and open-source TTS). |
| `sim_figures.py` | Figures comparing real speech with each simulator stage. |

## Stage 4: speech biology
| script | what it does |
|---|---|
| `extract_bio.py` | The 48 biology features for a manifest. |
| `bio_figures.py` | Real-vs-fake effect sizes and figures for `docs/05_speech_biology.md`. |

## Stage 5: scoring
| script | what it does |
|---|---|
| `score.py` | Local copy of the organizers' minDCF, both readings; validates submission files. |

## Stage 6: R0, R1, R3 and frozen SSL (R4)
| script | what it does |
|---|---|
| `extract_classic.py` | R0/R1 features (trivial stats, spectral, biology) for train/eval rows or the test clips. |
| `train_classic.py` | Trains R0 and R1 and writes their scores. |
| `run_aasist.py` | R3: scores clips with the organizers' pretrained AASIST. |
| `extract_ssl.py`, `train_ssl_heads.py` | R4: frozen self-supervised embeddings plus simple heads. |

## Stage 7: fine-tuned XLS-R (R4ft) and fusion (R5)
| script | what it does |
|---|---|
| `vram_probe.py` | Picks a memory configuration that fits the GPU before training. |
| `finetune_ssl.py` | Trains and scores the fine-tuned XLS-R model (R4ft; also R6). |
| `plot_r4ft_curve.py` | Training-curve figure for `reports/04_r4ft_xlsr.md`. |
| `fuse.py` | Logistic-regression fusion of several models' scores (R5). |
| `make_submission.py` | Writes a submission file from a model's test scores (`--higher-is-real` for the organizers' scorer). |

## Stage 8: generalization and call-audio tests
| script | what it does |
|---|---|
| `ingest_bench.py` | Public benchmark sets never used in training. |
| `make_keyguard_sets.py` | Typing-noise and keystroke-shield versions of held-out clips. |
| `bench_score.py` | Scores the frozen models on those sets; writes the generalization table. |

## Stage 10: R6 and the final submission
| script | what it does |
|---|---|
| `bench_to_manifest.py` | Registers three benchmark sets as R6 training sources (never DeepVoice or the Keyguard sets). |
| `r6_select.py` | Applies the pre-registered rule to R6 and its ensembles and writes the final submission (E5). |

## Stage 11: R7 (RawBoost) and the final submission
| script | what it does |
|---|---|
| `r7_select.py` | Applies the Stage 11 rule to R7a and its ensembles and writes the final submission (E7). |
