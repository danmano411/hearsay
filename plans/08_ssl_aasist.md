# Phase 8 — End-to-end SSL fine-tune: XLS-R-300M + AASIST-style back end (G5, issue #15)

The main bet for the GPU node. Fine-tune a self-supervised speech front end end to end with a small spoofing back
end on raw 16 kHz audio. This is the recipe common to many of the strongest ASVspoof5 systems: an SSL front end (wav2vec 2.0 /
XLS-R / WavLM) plus an AASIST-type graph back end or a light pooling head, trained with codec/noise augmentation
(Tak et al., Odyssey 2022, "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data
augmentation"; ASVspoof 5 overview, arXiv 2502.08857). It is rung **R4ft** of
`plans/06_modeling_ladder.md`, which deferred SSL fine-tuning until a GPU was available.
Status: PLAN (step 1 of #15). Code, probe and runs follow on this branch (`gpu/15-plan08`).

Where we start: best headline = **0.253** combined minDCF on `test_internal_testlike` (R1 LightGBM, 0.228 on
`val_testlike`). The best frozen-SSL result is WavLM-base+ layer-3 mean + LR at **0.389** on `val_testlike`
(partial, 14.8k train clips). Frozen SSL heads have not beaten hand-made features yet. Fine-tuning is how SSL systems
usually close that gap, but 8 GB of VRAM makes memory the binding constraint, so §3 is the core of this plan.

## 1. Hard rules (from `CLAUDE.md`, restated because this plan touches all of them)

| Rule | How this plan honours it |
|---|---|
| HGT test audio (`data/raw/hgt_test`) is inference only | HGT is scored once, by the frozen final checkpoint, in `eval()` mode (BatchNorm uses train running stats, no test-time adaptation). No fitting, normalization stats, calibration, thresholds, pseudo-labels or checkpoint/window/variant choice uses HGT audio or its score distribution. The only per-clip normalization is the SSL feature extractor's zero-mean/unit-variance, computed from that clip alone. |
| Train only on `split == "train"` | The sampler draws only from the 135,436 `train` rows. `excluded` rows (11,990 held-out-generator fakes: playht, unit_speech, diffgan_tts) and all `val`/`test_internal` rows never reach the optimizer. The loader asserts this on every row it reads. |
| Tune and early-stop on `val_testlike` | Every choice (step, back end, depth, LR, sampling weights, inference windows) uses `val_testlike` combined minDCF (8,963 clips: 6,274 real / 2,689 fake). |
| `test_internal_testlike` is the headline, never tuned on | It is scored once, after the final choice is written to `config.json` (checkpoint sha256 + inference mode). Variants that lose on `val_testlike` are still uploaded, but the rung's result is fixed in advance as the `val_testlike` winner. |
| `prep()` on every clip, train and test | Train: 6 s pre-crop → `augment()` → `prep()` → random 4 s `crop()`. Eval/HGT: `prep()` on the full clip → windows (§6). |
| Report both minDCF readings + combined | Every eval logs `official_as_written`, `brief_as_written`, `combined` and EER (`hearsay.metrics`). |
| The gpu node never calls `report()` | Self-check with `hearsay.evaluate.evaluate()`. Upload `data/scores/<name>.parquet` (`path, score`) and `data/scores/<name>__hgt.parquet` (`filename, score`) with `python tools/hubctl.py put`. The cpu node runs `report()` and owns the leaderboard. |

## 2. Model

| Part | Choice | Why |
|---|---|---|
| Front end | `facebook/wav2vec2-xls-r-300m` (24 pre-LN blocks, d = 1024, FFN 4096, 16 heads; 7-layer conv feature encoder, 20 ms hop → **199 frames per 4 s**) | 128-language pretraining. It is a common front end among strong ASVspoof5 systems. |
| Depth | **Truncate to the first K = 12 blocks** by default (same truncation as `SSLEmbedder(max_blocks=)`, including dropping the final pre-LN `layer_norm`). K is fixed from the G3 frozen layer sweep on `val_testlike`: K = min(24, best layer + 4), floored at 8. | Spoofing cues sit in lower/middle SSL layers (our WavLM sweep: layer 3 best). Halving depth halves compute and VRAM (§3). |
| Frozen | Conv feature encoder always frozen, run under `no_grad` (standard wav2vec2 fine-tuning). Optionally also the lowest N transformer blocks (fallback ladder, §4). | Its activations on raw audio dominate memory (§3). |
| Back end A, light (run first) | Learned softmax weights over the K+1 hidden states → Linear 1024→256 → attentive statistics pooling → Linear → 1 logit | Tiny, stable, uses every layer ("sensitive layer selection" style). |
| Back end B, AASIST | Tak-2022 SSL-AASIST: Linear 1024→128 → (1, 128, 199) "spectrogram" → max-pool → AASIST residual encoder + spectral/temporal graph attention + HS-GAL. Graph modules imported in place from `third_party/asvspoof5/Baseline-AASIST/models/AASIST.py` (MIT, not copied), like `aasist_wrap.py`. | The ASVspoof5 top-system back end. Its sinc front end is replaced by the SSL features. |
| Loss / output | BCE on one logit (1 = spoof). Score = spoof logit (higher = fake), as the harness expects. | minDCF is rank-only (`docs/scoring.md` §1), so no calibration is needed here. The cpu node's submission builder applies its fixed sigmoid. |
| Regularization | `layerdrop = 0` (it fights checkpointing and makes step time erratic). HF time masking `mask_time_prob = 0.05`, length 10 (mild SpecAugment on hidden states). Dropout at the config default of 0.1. | |

Code (step 2): `src/hearsay/models/ssl_e2e.py` (front end wrapper, both back ends, freezing/truncation, param
groups) and `scripts/finetune_ssl.py probe|train|score` (DataLoader, resume, eval). CPU tests in
`tests/test_ssl_e2e.py` use a tiny random-init `Wav2Vec2Config`: output shapes, truncation equals the full model's
block K, frozen params get no grad, the sampler never yields non-train rows, and its batches are class-balanced.

## 3. Hardware and the VRAM budget (the key constraint)

**Machine:** one NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB GDDR (≈ 7.0 GB free with the desktop running), compute
capability 12.0 (Blackwell: bf16 tensor cores supported), Windows WDDM, torch 2.14 + cu130, transformers 5.17.
**WDDM trap:** when the driver over-allocates, it silently spills to shared system memory and training slows by an
order of magnitude instead of failing. So every run calls `torch.cuda.set_per_process_memory_fraction(0.92)`, which the
G1 PR adds to `hearsay.device.resolve()` (the script sets it itself if G1 has not merged). Over-allocation then raises
a clean OOM. Usable budget for tensors: ≈ 7.0 GB free − ~0.4 GB CUDA context/cuBLAS workspace → **plan for ≤ 6.0 GB
peak reserved**, keeping ~0.5 GB for allocator fragmentation.
**Evidence it matters:** WavLM-base+ *inference* at batch 64 × 4 s (fp32) already exceeded 8 GB. The conv feature
encoder's activations on raw audio dominate. Its first conv layer outputs 512 × 12,799 values per 4 s clip (26 MB fp32),
and LayerNorm/GELU make copies, so batch 64 needs several GB just in transients.

### 3.1 Parameters (XLS-R-300M ≈ 315 M)

| Block | Params | Notes |
|---|---|---|
| Conv feature encoder | 4.2 M | frozen |
| Feature projection + positional conv (k = 128, 16 groups) | 0.5 M + 8.4 M | trainable |
| Transformer block (×24) | 12.6 M each → 302 M | K = 12 keeps 151 M |
| Back end A / B | ~0.3 M / ~0.45 M | trainable, LR ×50 |

### 3.2 Memory per component (estimates; the probe in §4 replaces them)

Byte counts: fp32 = 4, bf16 = 2. AdamW keeps 2 fp32 states per trainable param. We keep **fp32 master weights** and
run the forward under `torch.autocast("cuda", dtype=torch.bfloat16)`. bf16 needs no GradScaler. Pure-bf16 weights are
rejected: updates at LR 1e-5 underflow bf16's 8-bit mantissa.

| Item | Formula | K = 12, all blocks trainable (**default**) | K = 24, lower 12 frozen | K = 24, all trainable |
|---|---|---|---|---|
| Weights (fp32) | 4 B × all params | 164 M → **0.66 GB** | 315 M → 1.26 GB | 1.26 GB |
| Gradients (fp32) | 4 B × trainable | 160 M → **0.64 GB** | 151 M → 0.60 GB | 311 M → 1.24 GB |
| AdamW states, fp32 | 8 B × trainable | **1.28 GB** | 1.21 GB | 2.49 GB |
| AdamW states, 8-bit (optional) | 2 B × trainable | 0.32 GB | 0.30 GB | 0.62 GB |
| **Static total (fp32 AdamW)** | | **2.6 GB** | 3.1 GB | 5.0 GB |
| Transformer activations, no checkpointing | ≈ 14 MB / block / clip (≈ 18·T·d + 3·H·T² values, T = 199, mixed bf16/fp32) | 168 MB / clip | 168 MB / clip (upper 12 only) | 336 MB / clip |
| Transformer activations, checkpointing | block input only (0.8 MB fp32 / block / clip) + one block's full activations during recompute | ~10 MB / clip + 14 MB × micro-batch | ~10 MB / clip + 14 MB × mb | ~20 MB / clip + 14 MB × mb |
| Frozen conv encoder, `no_grad`, bf16 | transient peak ≈ 3–4 copies of 512 × 12,799 | ~50 MB / clip (freed before backward) | same | same |
| Same encoder if it were trainable | all 7 layers' outputs + norm/GELU copies kept | 100–200 MB / clip, never used | | |
| Back end A / B activations | | < 2 MB / ~15 MB per clip | | |
| **Peak, micro-batch 8, checkpointing** | static + 8 × (10 + 14 + 50 + 15) MB + 0.5 GB slack | **≈ 3.8 GB** ✅ | ≈ 4.3 GB ✅ | ≈ 6.3 GB ❌ (mb 4: ≈ 5.9 GB ⚠️) |
| **Peak, micro-batch 8, no checkpointing** | static + 8 × (168 + 50 + 15) MB + 0.5 GB | ≈ 4.9 GB ✅ (faster; probe decides) | ≈ 5.4 GB ⚠️ | ≈ 8.7 GB ❌ |
| Peak, micro-batch 4, checkpointing, 8-bit AdamW | | — | — | ≈ 4.0 GB (only if bitsandbytes works, see below) |

### 3.3 Derived config (default run)

| Setting | Value |
|---|---|
| Precision | bf16 autocast, fp32 master weights and fp32 AdamW |
| Front end | XLS-R-300M truncated to K = 12 blocks, conv encoder frozen (`no_grad`), `attn_implementation="sdpa"` if the probe confirms HF supports it for wav2vec2 (removes most of the 3·H·T² term) |
| Gradient checkpointing | on the transformer blocks, `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`. Non-reentrant is required: with frozen inputs, reentrant checkpointing silently drops gradients. Switched off if the probe shows the no-checkpoint peak ≤ 5.2 GB (≈ 25 % faster). |
| Micro-batch × accumulation | 8 × 4 = **effective 32** (4 real + 4 fake per micro-batch). 64 = 8 × 8 is the alternative if the loss is noisy. |
| Optimizer | AdamW, betas (0.9, 0.98), weight decay 0.01 (none on norms/biases). Front-end top LR **2e-5**, **layer-wise LR decay 0.85** per block downward (block 1 gets 2e-5 × 0.85¹¹ ≈ 3.3e-6; projection/pos-conv get the block-1 LR). Back end 1e-3. Grad-norm clip 1.0. |
| Schedule | Back-end warm-up: front end frozen for the first 300 optimizer steps. Then linear warm-up of front-end LRs over 500 steps, then cosine decay to 5 % over the planned budget (§7). |
| 8-bit AdamW | Optional and **unverified on Windows + sm_120** (bitsandbytes needs a build with Blackwell kernels). Tried only if the default must go to full depth. Never the default. |
| Not used | `torch.compile` (Triton on Windows is not reliable for us), DeepSpeed/FSDP (single GPU), fp16 (bf16 is available and needs no loss scaling). |

## 4. Step 1 of the run: VRAM probe (before any training)

`python scripts/finetune_ssl.py probe --device cuda` runs **before** the first training job and writes
`data/models/r4ft_xlsr/probe.json`. The results table goes into the PR and `reports/r4ft_xlsr.md`, replacing the
estimates in §3.2 and §5.

1. **Memory:** for each config below, run 12 optimizer steps on random 4 s waveforms (memory does not depend on
   content) and record `max_memory_allocated`, `max_memory_reserved`, step time (median of the last 8), and OOM or
   not. A step time that jumps > 3× between steps is treated as WDDM spill even without an OOM.
   Order: (a) K12 ckpt mb8 → (b) K12 no-ckpt mb8 → (c) K12 ckpt mb16 → (d) K24 lower-12-frozen ckpt mb8 →
   (e) K24 all-trainable ckpt mb4 → (f) (e) with 8-bit AdamW, only if bitsandbytes imports and a 1-step sanity check
   matches fp32 AdamW. Each config runs in its own subprocess, so an OOM cannot fragment the next one.
2. **Throughput:** the chosen config runs 50 steps with the real DataLoader, reporting clips/s for GPU and loader
   separately (a loader-only loop over 2,000 clips). If the loader is slower than the GPU, add workers or pre-crop
   long clips on disk.
3. **Pick:** the fastest config whose peak reserved is ≤ 85 % of the capped budget (~5.5 GB). Log it in
   `config.json`.

**Fallback ladder if (a) OOMs** (apply in order, re-probe after each):
1. Freeze more blocks: lowest 4, then lowest 8 of the K.
2. Shorter crops: 4 s → 3 s (test minimum is 3.0 s, median 3.4 s, so 3 s still covers the test). Eval stays 4 s.
3. Smaller micro-batch: 8 → 4 → 2, with accumulation raised to keep effective 32.
4. Smaller front end: WavLM-base+ or wav2vec2-base (95 M params, 12 × 768), full fine-tune, same recipe. The static
   footprint is ≈ 1.5 GB, so it fits comfortably.

## 5. Wall-clock estimates (assumptions stated; replaced by the probe)

Assumptions: RTX 5050 Laptop ≈ 2,560 CUDA cores; bf16 tensor throughput assumed ~40–50 TFLOPS peak; **~12 TFLOPS
effective** (≈ 25–30 % utilization at micro-batch 8 under WDDM); training ≈ 3× forward FLOPs, + 1 forward for
checkpointed blocks.

| Config | FLOPs / clip / train step | Est. clips/s | 32,768-clip virtual epoch | Full pass over 135k train |
|---|---|---|---|---|
| Conv encoder forward (frozen, every config) | ≈ 20 GFLOP | — | — | — |
| K12, checkpointing (default) | 4 × 60 + 20 ≈ 260 GFLOP | ~45 | ~12 min | ~50 min |
| K12, no checkpointing | 3 × 60 + 20 ≈ 200 GFLOP | ~60 | ~9 min | ~38 min |
| K24, lower 12 frozen | 60 + 4 × 60 + 20 ≈ 320 GFLOP | ~37 | ~15 min | ~61 min |
| K24, all trainable | 4 × 120 + 20 ≈ 500 GFLOP | ~24 | ~23 min | ~94 min |

Data side: CPU cost per training clip is ≈ 30 ms (read + 6 s pre-crop, then `augment()`: mp3 round trip on 25 % of
clips ≈ 40 ms each, resampling on 15 %, then `prep()` with trim and a 10th-order `sosfiltfilt`). With 6 DataLoader
workers (Windows spawn, `persistent_workers=True`), that is ≈ 200 clips/s, above the GPU rate. The probe verifies this.
Eval of the full `val_testlike` set from the RAM/disk cache is forward only: ~80 GFLOP/clip → ~150 clips/s → **~1 min**.
Final scoring of val + test_internal + HGT (40,160 clips, ≤ 3 windows each) takes ~15–25 min, mostly decoding + `prep()`.
**Budget per run:** ≤ 16 virtual epochs ≈ 3.5–4 h for the default, less with early stopping.

## 6. Data

**Rows.** `data/processed/manifest.parquet` has 185,915 rows. Columns: `path, label, source, generator, speaker,
text_id, duration, sr_orig, codec_orig, license, group, split, logo_fold, val_testlike, test_internal_testlike`.
Train = `split == "train"`: 135,436 rows (43,865 bonafide / 91,571 spoof, 11 real sources, 11 fake sources,
140 fake generators, durations 3.0–58.5 s, median 7.1 s). Re-pull the manifest if a `SYNC` arrives, but only between
runs, never mid-run.

**Class-balanced, source-diverse sampling** (includes the cpu node's error-analysis input). Every micro-batch has
4 bonafide + 4 spoof. Within a label, cells are **(source, label)**, drawn with probability ∝ √n. The two real sources
R1 gets most wrong on `test_internal_testlike` get ×2: **asvspoof5 reals** (R1 combined 0.55) and **librisevoc reals**
(0.40). Both look fake to R1, which suggests a channel/corpus effect. Within a fake cell, generators are also drawn
∝ √n, then a clip uniformly at random. Resulting shares (from the current manifest):

| Bonafide cell | n | natural % | sampled % | | Spoof cell | n | natural % | sampled % |
|---|---|---|---|---|---|---|---|---|
| librispeech | 13,075 | 29.8 | 16.0 | | diffssd | 44,049 | 48.1 | 24.1 |
| ljspeech | 9,612 | 21.9 | 13.7 | | wavefake | 9,022 | 9.9 | 10.9 |
| **asvspoof5** | 2,277 | 5.2 | **13.4** | | asvspoof5 | 8,789 | 9.6 | 10.8 |
| **librisevoc** | 2,114 | 4.8 | **12.9** | | librisevoc | 7,988 | 8.7 | 10.3 |
| mlaad_tiny | 4,620 | 10.5 | 9.5 | | mlaad_tiny | 4,688 | 5.1 | 7.9 |
| in_the_wild | 3,509 | 8.0 | 8.3 | | in_the_wild | 4,563 | 5.0 | 7.8 |
| cvoicefake_en | 3,340 | 7.6 | 8.1 | | sim | 3,929 | 4.3 | 7.2 |
| asvspoof2019_la | 2,941 | 6.7 | 7.6 | | cvoicefake_en | 3,090 | 3.4 | 6.4 |
| sonar | 1,812 | 4.1 | 6.0 | | asvspoof2019_la | 2,297 | 2.5 | 5.5 |
| dfadd | 370 | 0.8 | 2.7 | | dfadd | 1,877 | 2.0 | 5.0 |
| lj_real (organizers') | 195 | 0.4 | 2.0 | | sonar | 1,279 | 1.4 | 4.1 |

lj_real is drawn ≈ 1.7 times per clip per virtual epoch, which is enough to learn the organizers' channel without
memorizing a single speaker. The sampler is a seeded generator whose state is saved in every checkpoint (§8). The ×2
weights and the √ exponent may be retuned between runs based on the `val_testlike` slices (§7). That tunes on
validation, which is allowed. It never uses test slices.

**Augmentation.** `hearsay.preprocess.augment()` is class-symmetric and applied to real and fake alike, before
`prep()`. It picks exactly one of: MP3 round trip (25 %), additive Gaussian noise at 15–40 dB SNR (20 %), or a
resampler round trip through 8 / 11.025 / 22.05 / 24 kHz (15 %). The remaining 40 % pass through unchanged. `prep()`
then removes DC, trims silence, low-passes at 7 kHz and RMS-normalizes, so the augmentations cannot reintroduce the
7–8 kHz band cue. On top of that, we use only the SSL-internal time masking (§2). RawBoost and extra codecs are left
for a later run if slices show channel errors persisting.

**Crops and windows.** Train uses a random 4 s crop (`crop()` tiles clips shorter than 4 s; train minimum is 3.0 s).
Eval during training uses the centre 4 s window of `prep(full clip)`. The `val_testlike` windows are prepared once and
cached as a float32 memmap (8,963 × 64,000 ≈ 2.3 GB, `data/models/r4ft_xlsr/cache/`, gitignored). Final scoring uses
the mean logit over up to 3 evenly spaced 4 s windows, as in `aasist_wrap.windows()`. Centre vs 3 windows is chosen
once, on `val_testlike` only.

## 7. Evaluation cadence, slices, stopping

- **Cadence:** full `val_testlike` eval every **1,024 optimizer steps** (= one 32,768-clip virtual epoch; ~12 min of
  training + ~1 min eval, est.), plus one eval of the pretrained-front-end + warmed-up-back-end state at step 300.
- **Logged per eval** (`log.jsonl`): step, clips seen, train loss, LRs, `official_as_written`, `brief_as_written`,
  `combined`, EER, `max_memory_reserved`, clips/s.
- **Slices on `val_testlike`** (diagnostic, every eval, via `hearsay.evaluate.per_source(scores, "val_testlike")`):
  each real source's minDCF vs all fakes, each fake source and generator vs all reals. Watch list, from R1's errors:
  asvspoof5 reals, librisevoc reals, **playht** (470 fakes in `val_testlike`, held out of train) and
  **unit_speech** (470, held out), plus codec-LM TTS generators (tiny n, noted but not acted on). Success means these
  move, not just the aggregate. Slices guide the *next* run's sampling weights. Checkpoint selection uses the
  aggregate only, to avoid picking the luckiest of many slices.
- **Early stop:** save `best.pth` when `val_testlike` combined improves by ≥ 0.002. Stop after **4 evals without such
  an improvement**, or at 16 virtual epochs, or if train loss becomes NaN (restore last good checkpoint, halve LRs, one
  retry).
- **Run budget (all selection on `val_testlike`):** run 1 = default (K12, back end A). Run 2 = back end B (AASIST) on
  the same config. Run 3 = the better back end with one change, chosen from run-1/2 slices (e.g. K24 lower-12-frozen,
  or retuned sampling weights). Three runs at most (~10–12 h of GPU), then write up.

**Stopping criterion (`CLAUDE.md`).** This is one rung. Its headline is the single `val_testlike` winner's
`test_internal_testlike` combined minDCF, as reported by the cpu node. If it is ≤ 0.05, the project criterion is met.
If it improves on 0.253 by ≥ 0.005, the "3 rungs < 0.005" counter resets. Otherwise it counts as one of the three.
Losing variants do not count as extra rungs. The rung's value to R5 fusion (its scores on `val_testlike` next to R1's)
is reported separately by the cpu node.

## 8. Checkpointing and resume (the laptop may sleep)

- Every **250 optimizer steps** (~3 min est.), plus at every eval: atomic write (`last.tmp` → `os.replace`) of
  `last.pth`. It holds model state (the truncated front end + back end, ~0.7 GB), AdamW state (~1.3 GB), LR scheduler,
  step / clips seen / best / patience counter, the sampler generator state, and the torch/numpy/python RNG states.
  Only the latest `last.pth` is kept. `best.pth` holds weights only. Disk: ~3 GB per run under
  `data/models/r4ft_xlsr/<run>/`.
- `train` resumes automatically from `last.pth` (as `finetune_aasist.py` does). A resume mid-accumulation
  restarts that accumulation group. A resumed run is bitwise-continuous up to DataLoader worker timing.
- During training the script calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` (ctypes), so Windows
  does not sleep, and clears it on exit. No system power settings are changed. If the laptop sleeps anyway (lid,
  battery), resume from the last save loses at most ~3 min.
- Heartbeat: `python tools/hubctl.py job --me gpu "G5 run1 ep 5/16 vt 0.21"` after each eval. `PROGRESS` message to
  the cpu node on every new best.

## 9. Deliverables

| What | Where | Who |
|---|---|---|
| Plan (this file) | `plans/08_ssl_aasist.md` | gpu, this commit |
| Model + script + CPU tests | `src/hearsay/models/ssl_e2e.py`, `scripts/finetune_ssl.py`, `tests/test_ssl_e2e.py` | gpu, PR on `gpu/15-plan08` (or a follow-up branch) |
| Probe results | `data/models/r4ft_xlsr/probe.json` → table in the PR and the report | gpu |
| Scores, all `val` + `test_internal` rows | `data/scores/R4ft_xlsr_<backend>.parquet` (`path, score`, higher = fake) → `hubctl put` | gpu |
| HGT scores | `data/scores/R4ft_xlsr_<backend>__hgt.parquet` (`filename, score`, all 1,671 clips, none defaulted) → `hubctl put` | gpu |
| Self-check | `hearsay.evaluate.evaluate()` on the uploaded frame; numbers in the `DONE` message and the PR | gpu |
| Leaderboard rows, fusion | `reports/leaderboard.md` via `report()`, R5 | cpu |
| Judge write-up | `reports/r4ft_xlsr.md`: recipe, VRAM probe vs estimate, training curves, `val_testlike` slices per eval, final per-source table on `test_internal_testlike` (**report only, computed after the choice is frozen, never used for selection**), what failed | gpu |

Weights, caches and scores are never committed (`data/` is gitignored). The best weights may be put on the hub under
`data/models/` if the cpu node asks for them.

## 10. Risks

| Risk | Mitigation |
|---|---|
| WDDM spill instead of OOM | memory fraction 0.92 + step-time spike detector in the probe and in training logs |
| Estimates in §3/§5 off by 2× | the probe runs first; the fallback ladder is ordered and cheap |
| Fine-tuning learns corpus/speaker identity (LJ = real) instead of spoofing cues | (source, label) sampling, symmetric `augment()`, `prep()`, `val_testlike` slices, held-out generators in val |
| Catastrophic forgetting / collapse early in training | back-end warm-up with a frozen front end, low LR with layer-wise decay, grad clip, eval at step 300 |
| Overfitting `val_testlike` through repeated selection | ≤ 3 runs, a fixed selection metric, headline scored once after the choice is frozen |
| Loader-bound on the laptop CPU | loader-only throughput in the probe. Fixes: more workers, or on-disk pre-crops of train clips to 6 s |
| bitsandbytes / SDPA / checkpointing API differences in transformers 5.17 | each is a probe item with a fallback (fp32 AdamW, eager attention, smaller micro-batch) |
| GPU shared with other jobs (e.g. G2/G3 extraction) | G5 runs alone. It starts after the current GPU job, and extraction jobs queue behind it or wait for its evals |
| Laptop sleeps / power loss | resumable every 250 steps; keep-awake flag; charger on |
| Gains don't beat R1 | still valuable for R5 fusion: scores uploaded either way, and slices show where SSL and R1 disagree |
