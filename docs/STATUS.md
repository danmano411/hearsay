# HEARSAY — live status

_Updated by the cpu node (coordinator) on every task completion. Last update: 2026-09-26 04:30 EDT._

## Best so far

Headline = combined minDCF on `test_internal_testlike` (9,747 clips, 70/30, frozen; never tuned on). Lower is better.

| rank | model | headline | val_testlike | HGT scored |
|---|---|---|---|---|
| 1 | R1_lgbm_all_full (LightGBM, 228 spectral + bio feats, full train set) | **0.253** | 0.228 | ✅ |
| 2 | R1_lgbm_all (same, 30k-fake subset) | 0.305 | 0.282 | ✅ |

Target (stopping criterion): ≤ 0.05, or 3 consecutive rungs improving < 0.005.

## Nodes

| node | now | next |
|---|---|---|
| cpu (coordinator) | merged #22 #23 #24 #25 #26 after cross-review; hub restarted on hardened code | C4 heads as gpu embeddings arrive; C5 fusion |
| gpu (RTX 5050 Laptop, 8 GB, CUDA 13) | G1 CUDA throughput bench; G0 done (hub hardening #25) | G3 XLS-R + G2 WavLM embeddings |

## Board

GitHub issues #10–#21 (`gh issue list`). Every PR is reviewed by the other node before merge; reviews so far caught a silent
all-defaults submission bug (#24), a listener that could die on a hub restart (#25), and an over-broad temp-file filter (#26).

## Log

- 04:30 Fallback submission from R1_lgbm_all_full validated (1,671 rows, 1,671 distinct scores).
- 04:25 Cross-reviews: #24 #25 #26 merged; hub restarted.
- 04:09 gpu node online (RTX 5050 8 GB); data verified (187,561 files).
- 04:05 C3: R1 on the full training set, 0.305 → 0.253.
- 03:40 C2: sim finished (5,204 clips); eval subsets frozen after re-sampling would have moved 151 rows.
- 03:18 Hub online; two-node protocol adopted (`plans/07_two_node_protocol.md`).
- 02:50 Data stream to the gpu laptop started (hotspot, ~11–13 MB/s).
