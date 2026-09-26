# HEARSAY — real vs. synthetic speech detection (HackGT 2026, NSA challenge)

Score each test clip from 0.0 (confident real) to 1.0 (confident synthetic), optimizing the ASVspoof5 Track-1 **minDCF**.

- **Plans:** [`plans/`](plans/00_master_plan.md) — written before each phase.
- **Docs:** [`docs/`](docs/) — data audit, external data, how synthetic speech is made, biology of real speech, scoring.
- **Results:** [`reports/leaderboard.md`](reports/leaderboard.md)
- **Status / GPU handoff:** [`docs/GPU_HANDOFF.md`](docs/GPU_HANDOFF.md) — what is done, what is paused, how to resume.

## Layout
```
plans/      phase plans (written first)
docs/       research write-ups + figures
src/hearsay package: data, features, sim, models, metrics
scripts/    reproducible CLIs (ingest, clean, extract, train, predict)
reports/    audits, leaderboard, experiment notes
data/       (gitignored) raw + processed audio, features
third_party/(gitignored) organizers' ASVspoof5 scorer
```

## Setup
```
py -3.12 -m venv .venv && .venv\Scripts\pip install -r requirements.txt
```
Place the challenge data under `data/raw/` (`diffssd/`, `lj_real/`, `hgt_test/`) and the organizers' `asvspoof5` repo under `third_party/`.
