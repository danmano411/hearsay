# Phase 7 — Two-node protocol (CPU laptop + GPU laptop)

Two Claude Code sessions (same account) work in parallel on one repo. The **GPU node** runs everything that needs
CUDA; the **CPU node** (16 threads, 60 GB RAM, hosts the hotspot) does everything else and coordinates.
Status: ADOPTED 2026-09-26 (owner: CPU coordinates, cross-review before merge, GitHub Issues board). Hub: `tools/hub.py` + `tools/hubctl.py`.

## 1. Nodes and roles

| | CPU node `cpu` (this laptop) | GPU node `gpu` |
|---|---|---|
| Address | `192.168.137.1` (hotspot host, fixed) | `192.168.137.x` (DHCP) |
| Role | **Coordinator**: owns the task board, the hub, the manifest, the leaderboard, fusion, the submission | **Worker**: all torch training/inference on CUDA |
| Owns (single writer) | `data/processed/manifest.parquet`, `reports/leaderboard.md`, `data/scores/` (canonical), `submission/`, `docs/STATUS.md` | its model scripts and its checkpoints in `data/models/` |

Single-writer files remove the two ways parallel work usually breaks: git conflicts on shared files, and training on
a stale manifest.

## 2. Three channels, each for one job

| Channel | Carries | Why |
|---|---|---|
| **GitHub repo** (PRs) | code, plans, docs, reports | durable, reviewable, and it *is* the 20% documentation grade |
| **GitHub Issues** (task board) | one issue per task: owner, status, result | durable, visible to the owner from a phone, shows the judges the process |
| **LAN hub** (HTTP on `192.168.137.1:8770`) | fast messages + artifacts too big for git (scores, embeddings, weights, manifest) | ~30 MB/s, no internet round trip, no git noise |

Fallback: if the hub is down, messages go to comments on the relevant issue. Large artifacts wait until it's back.

## 3. LAN hub (runs on the CPU node)

Small stdlib-Python HTTP server (`tools/hub.py`, not part of the model code). Every request carries
`X-Hearsay-Token` (shared secret, never committed). Bound to the hotspot interface only.

| Endpoint | Purpose |
|---|---|
| `POST /msg` `{from, to, type, task, body}` → `{id}` | send a message; appended to `data/hub/messages.jsonl` (append-only log) |
| `GET /msg?to=<node>&after=<id>` | fetch new messages (polled, see §5) |
| `PUT /files/<path>` / `GET /files/<path>` | upload/download an artifact; only under `data/scores/`, `data/features/`, `data/models/`, `data/processed/manifest.parquet`, `data/processed/sim/`; response includes sha256, verified by the receiver |
| `GET /files/<dir>/?list` | list a directory with sizes + sha256 (for sync) |
| `POST /heartbeat` `{node, job, progress, eta, gpu_util}` · `GET /status` | liveness + what each node is doing |

## 4. Message types

`ASSIGN` (coordinator → worker: issue #, spec) · `CLAIM` · `PROGRESS` · `ARTIFACT` (path, sha256, what it is) ·
`DONE` (issue #, PR, metrics, artifacts) · `REVIEW_REQUEST` / `REVIEW` (PR #, verdict, must-fix) ·
`QUESTION` / `ANSWER` · `BLOCKED` (reason) · `SYNC` (manifest or rules changed: re-pull before the next run) ·
`STOP` (owner asked to pause).
Every message names its GitHub issue so the durable record and the fast channel stay linked.

## 5. How each session listens

Neither session can be pushed to, so each runs a **Monitor** that polls the hub every 20 s and prints only new
messages addressed to it. Each new message wakes that Claude. Heartbeat every 5 min. A node silent for > 15 min
gets one `QUESTION`; if still silent, the other node tells the owner.

## 6. Task lifecycle

1. The coordinator opens an issue (`node:gpu`/`node:cpu`, `status:todo`) and sends `ASSIGN`.
2. The worker replies `CLAIM`, sets `status:doing`, and branches `gpu/<issue#>-<slug>` (or `cpu/...`).
3. The worker uploads score files to the hub (`data/scores/<name>.parquet`, `__hgt.parquet`), then sends `DONE`
   with the PR.
4. The **other node reviews the PR** (same checks as the Stage-A verifiers: leakage, prep() everywhere, HGT
   inference-only, recomputed metrics), then replies `REVIEW`. Merge only after `REVIEW: pass`.
5. The coordinator runs `hearsay.evaluate.report()` on uploaded scores, so the leaderboard has a single writer, and
   closes the issue with the numbers.

## 7. Git rules

- Branch per task; rebase on `origin/main` before opening a PR; never push to `main` directly (except coordinator
  doc-only commits to `docs/STATUS.md`).
- Neither node edits the other's owned files (§1). Shared modules (`src/hearsay/{preprocess,evaluate,metrics,audio}.py`)
  change only through a PR reviewed by the other node, followed by a `SYNC` message.
- The rules in `CLAUDE.md` apply to both nodes unchanged (HGT inference-only, `prep()` everywhere, both minDCF readings).

## 8. Initial work split

**GPU node** (in order):
- G1: CUDA torch setup, tests, `--device cuda` smoke run, throughput numbers.
- G2: finish WavLM extraction with a bigger train subset (`--train-n 40000`), `--batch 64`.
- G3: XLS-R-300M full-depth extraction.
- G4: finish AASIST zero-shot scoring; move the fine-tune to a DataLoader with workers and resume it with more data.
- G5: end-to-end fine-tune of XLS-R + AASIST-style back end (`prep()` + `augment()`, 4 s crops) — the main bet.
- G6: R2 CNN on LFCC/log-mel.

Each GPU task uploads embeddings or scores to the hub rather than training heads itself.

**CPU node** (in order):
- C1: build the hub and the listeners (§3, §5).
- C2: finish the sim on CPU, rerun `make_splits.py`, then send `SYNC` with the new manifest.
- C3: full bio + classic features on all rows, and retrain R1.
- C4: train heads on every embedding shard set the GPU uploads (`train_ssl_heads.py`, CPU is plenty).
- C5: R5 fusion, default-value strategy, submission builder and validator.
- C6: error analysis, `docs/STATUS.md`, judge-facing write-ups, and reviews of GPU PRs.

Timing: the GPU never waits on the CPU, because every GPU task only needs the manifest (already transferred). The
CPU picks up each GPU artifact as it lands.

## 9. Owner visibility

- Issues board = what's queued, in progress, and done.
- `docs/STATUS.md` (coordinator, updated at every `DONE`) = current best score, what each node is running, ETA.
- `data/hub/messages.jsonl` = full conversation log between the nodes, summarized into `reports/collaboration.md` at the end.

## 10. Open decisions for the owner

1. Coordinator = CPU node (proposed). The alternative is the GPU node, but it would then be both busy and in charge.
2. Should the other node's PR review be required before merge (proposed), or should authors self-merge for speed?
3. Should GitHub Issues be the task board (proposed), or a `tasks.md` file on the hub (less visible)?
