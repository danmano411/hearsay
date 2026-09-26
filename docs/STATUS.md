# HEARSAY — live status

_Updated by the cpu node (coordinator) on every task completion. Last update: 2026-09-26 06:15 EDT._

## ⚠ Needs the owner (on the GPU laptop)

1. ~~Approve stopping the stuck CUDA benchmark.~~ Resolved 05:39: Claude Code's low-memory reaper killed it. Its notice
   says not to restart that benchmark without you, so the G1 throughput numbers wait for you (optional).
2. *(Optional)* The GPU laptop's permission layer blocks its Claude from merging PRs. Not blocking: the cpu node merges
   GPU PRs after reviewing them, which is what the protocol requires.

## Best so far

Headline = combined minDCF on `test_internal_testlike` (9,747 clips, 70/30, frozen; never tuned on). Lower is better.

| rank | model | headline | val_testlike | HGT scored |
|---|---|---|---|---|
| 1 | R1_lgbm_all_full (LightGBM, 228 spectral + bio feats, full train set) | **0.253** | 0.228 | ✅ |
| 2 | R1_lgbm_all (same, 30k-fake subset) | 0.305 | 0.282 | ✅ |

In progress (validation only, headline pending): **R4ft run 1** (XLS-R-300M fine-tuned, K12) after 2 virtual epochs:
`val_testlike` **0.0414** (EER 1.15 %; R1: 0.228), already below the 0.05 target *on validation*. Slices (combined):
playht 0.027 (R1 0.302), unit_speech 0.037 (0.220), asvspoof2019_la reals 0.023 (0.278), asvspoof5 reals 0.221 (0.501).
The stopping criterion is judged on the headline, scored once after early stopping and a frozen config.json.

Target (stopping criterion): ≤ 0.05, or 3 consecutive rungs improving < 0.005.

## Nodes

| node | now | next |
|---|---|---|
| cpu (coordinator) | merged #22 #23 #24 #25 #26 after cross-review; hub restarted on hardened code | C4 heads as gpu embeddings arrive; C5 fusion |
| gpu (RTX 5050 Laptop, 8 GB, CUDA 13) | GPU blocked (see above). Code done + merged: G1 memory cap (#32), G4 DataLoader (#31), **G5 XLS-R fine-tune (#34, 65 tests)** | **running: G5 VRAM probe → run 1 (XLS-R K12, back end A)** |

## Board

GitHub issues #10–#21 (`gh issue list`). Every PR is reviewed by the other node before merge; reviews so far caught a silent
all-defaults submission bug (#24), a listener that could die on a hub restart (#25), and an over-broad temp-file filter (#26).

## Log

- 06:12 R4ft run 1, 2 epochs: val_testlike 0.0414.
- 06:10 AASIST zero-shot: 0.823 headline (doesn't transfer); R1+AASIST fusion 0.247.
- 05:58 R4ft run 1, 1 epoch of front-end fine-tuning: val_testlike 0.0647 (3.5× better than R1).
- 05:47 R4ft run 1 step 300 (frozen XLS-R + trained back end): val_testlike 0.193, already below R1 (0.228).
- 05:45 VRAM probe: XLS-R-300M K12 fits in 8 GB at 4.9 GiB allocated, 35 clips/s. Run 1 started.
- 05:39 GPU free; G5 VRAM probe started on CUDA.
- 05:35 G5 end-to-end fine-tune code merged (#34) after two review rounds (resume/early-stop parity, OOM-safe eval, frozen config.json). README + committed frozen eval subsets (#30).
- 04:45 #27 fusion, #28 error analysis, #29 G5 plan, #31 AASIST DataLoader merged after cross-review. GPU queue reordered: G5 first.
- 04:30 Fallback submission from R1_lgbm_all_full validated (1,671 rows, 1,671 distinct scores).
- 04:25 Cross-reviews: #24 #25 #26 merged; hub restarted.
- 04:09 gpu node online (RTX 5050 8 GB); data verified (187,561 files).
- 04:05 C3: R1 on the full training set, 0.305 → 0.253.
- 03:40 C2: sim finished (5,204 clips); eval subsets frozen after re-sampling would have moved 151 rows.
- 03:18 Hub online; two-node protocol adopted (`plans/07_two_node_protocol.md`).
- 02:50 Data stream to the gpu laptop started (hotspot, ~11–13 MB/s).
