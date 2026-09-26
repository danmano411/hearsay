# HEARSAY — live status

_Updated by the cpu node (coordinator) on every task completion. Last update: 2026-09-26 04:10 EDT._

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
| cpu (coordinator) | hub up; C2 done (PR #22), C3 done (PR #23); streaming data to gpu | C4 heads as gpu embeddings arrive; C5 fusion |
| gpu | receiving data over the hotspot (~47 GB) | G1 CUDA setup + smoke test, G0 hub review, then G3/G2 embeddings |

## Board

GitHub issues #10–#21 (`gh issue list`). Open PRs awaiting cross-review: #22 (C2 sim + frozen eval subsets), #23 (C3 R1 full).

## Log

- 04:05 C3: R1 on the full training set, 0.305 → 0.253.
- 03:40 C2: sim finished (5,204 clips); eval subsets frozen after re-sampling would have moved 151 rows.
- 03:18 Hub online; two-node protocol adopted (`plans/07_two_node_protocol.md`).
- 02:50 Data stream to the gpu laptop started (hotspot, ~11–13 MB/s).
