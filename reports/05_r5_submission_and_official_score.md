# Final results

**Stopping criterion met** (owner's rule: headline combined minDCF ≤ 0.05). Submission = **R5_r4ft_r1** (fusion of the
fine-tuned XLS-R model with the classic-feature model), chosen by an owner amendment. Unbiased headline estimate:
**0.0282** (R4ft alone); the fusion measured 0.0218 on the same set, but that number helped choose it, so it is optimistic.

## Official result (organizers, 2026-09-26)

The organizers scored our submission on the 1,671-clip HGT test set: **minDCF 0.0584, EER 2.5 %**.

| | held-out estimate (`test_internal_testlike`, same settings) | **HGT test (official)** |
|---|---|---|
| minDCF, Pspoof 0.3 / Cfa 4, higher = real | 0.0138 (R5) · 0.0192 (R4ft alone) | **0.0584** |
| EER | 0.54 % (R5) | **2.5 %** |

- **Scorer, now confirmed:** `calculate_metrics.py` from the ASVspoof5 evaluation package with **Pspoof = 0.3,
  Cmiss = 1, Cfa = 4**, reading a **higher score as bonafide (real)**. The brief says the opposite (1.0 = synthetic).
  Our first file followed the brief and scored exactly 1.0: every clip ranked backwards. The flipped copy
  (`1 − score`) is the one scored above (`docs/01_challenge_and_scoring.md` §7).
- **Best of two:** the organizers score the initial and the final submission and keep the lower minDCF. Any file we
  send from now on must use the flipped direction (`make_submission.py --higher-is-real`).
- **The HGT test is harder than our held-out set** (0.058 vs 0.014), as expected: its sources and generators differ
  from anything we held out. The gap was already visible from validation to headline (0.012 → 0.028). Nothing was
  tuned on the HGT audio or on this feedback.
- In plain terms: at the equal-error point about 1 in 40 real clips and 1 in 40 fakes are misjudged, so roughly 97-98 %
  of test clips are classified correctly.

Headline = `test_internal_testlike`: 9,747 held-out clips, 70/30 real/fake, 1,434 of its 2,924 fakes from three
generators never seen in training. It was scored once per model, after that model's choices were frozen, and never used
to select anything. minDCF: 0 = perfect, 1 = trivial.

## The ladder

| rung | model | val_testlike (combined) | **headline** official / brief / **combined** | EER |
|---|---|---|---|---|
| R0 | trivial cues after `prep()` (LightGBM) | 0.933 | 0.915 / 0.928 / 0.921 | — |
| R3 | organizers' pretrained AASIST, zero-shot | 0.831 | 0.990 / 0.655 / 0.823 | 32.5 % |
| R1 | LightGBM, 228 spectral + biology features | 0.228 | 0.262 / 0.243 / 0.253 | 6.5 % |
| **R4ft** | **XLS-R-300M (12 blocks) fine-tuned end to end + light back end** | **0.0121** | **0.0301 / 0.0262 / 0.0282** | **0.76 %** |
| **R5** | **LR fusion R4ft + R1 (submitted, owner amendment below)** | 0.0128 (cross-fit) | 0.0228 / 0.0207 / 0.0218 | 0.54 % |
| R5 | LR fusion R4ft + R1 + R3 (not selected) | 0.0127 (cross-fit) | 0.0230 / 0.0214 / 0.0222 | 0.58 % |

R4ft is **9× better than the best classic model**, and both cost readings agree: 0.030 official, 0.026 brief.

## Where the gain came from (val_testlike slices, combined)

| slice | R1 | R4ft (step 6,144) |
|---|---|---|
| playht fakes (held out of training) | 0.302 | 0.014 |
| unit_speech fakes (held out) | 0.220 | 0.011 |
| asvspoof2019_la reals | 0.278 | 0.004 |
| asvspoof5 reals (unusual recording chain) | 0.501 | 0.073 |

The fine-tuned SSL front end fixed exactly the two failure modes the error analysis identified: unseen generators, and
real speech from unusual recording chains.

## Selection (pre-registered before any R4ft headline existed; [development log](../docs/00_development_log.md), stage 7)

Candidates were compared on `val_testlike`: fusions by their group-cross-fitted number, and a fusion had to win by
≥ 0.002. R4ft alone (0.0121) beat both fusions (0.0128, 0.0127), so **R4ft is the submission**. The R4ft + R1 fusion
scores better on the headline (0.0218 vs 0.0282), but switching after seeing that would be selecting on the test set.

**Known bias in the rule (lesson, not acted on):** R4ft's `val_testlike` number is optimistic, because its checkpoint and
window mode were both chosen on that set, while the fusions' numbers are cross-fitted. The rule therefore leans toward
the single model. A fairer comparison would use `val` rows outside `val_testlike`, which neither candidate touched. We
record this instead of re-running selection after seeing headline numbers.

**Val → headline gap:** 0.0121 → 0.0282. Part is the selection optimism above; part is real, since the headline set's
held-out fakes and real corpora differ from validation. It is a reminder that the HGT score will likely be higher again.

## Owner amendment: switching to the fusion

The rule above picked R4ft. After the headline scores were in, the owner switched to the fusion (full record:
[development log](../docs/00_development_log.md), stage 7). Evidence that does not use the headline set:

| check (no selection on it) | R4ft | R4ft + R1 fusion |
|---|---|---|
| `val_testlike` (fusion group-cross-fitted) | 0.0121 | 0.0128 |
| fair check: 9,595 validation fakes nobody selected on + validation reals | 0.0329 | **0.0327** |
| HGT-like proxy: 1,185 LJ reals vs 6,703 DiffSSD fakes (validation) | 0.0007 | **0.0000** |

Report-only: on the headline set the fusion gains most on ElevenLabs (0.031 → 0.006), In-the-Wild reals (0.036 → 0.017),
PlayHT and UnitSpeech fakes, and loses on ASVspoof 5 reals (0.073 → 0.085). Tuning the fusion further on validation
(regularization C = 0.1 / 0.01, spectral + biology or biology-only inputs) made it worse, not better. Because the headline
informed this switch, we report **0.0282 as the honest estimate** and 0.0218 as optimistic.

## What the numbers mean

minDCF is **not** an error rate. It is the organizers' weighted cost at the model's best threshold, divided by the cost
of a system that always gives the same answer (= 1.0). For R4ft on the headline set (9,747 clips):

| reading | reals flagged as fake | fakes missed | cost |
|---|---|---|---|
| brief (real flagged = 4×) | **0.22 %** (15 of 6,823) | 1.74 % (51 of 2,924) | 4 × 0.22 + 1.74 = 2.62 % → 0.0262 |
| official code (missed fake = 4×) | 1.10 % (75) | **0.48 %** (14) | 1.10 + 4 × 0.48 = 3.01 % → 0.0301 |

Equal-error rate 0.76 %; best plain accuracy **99.40 %** (58 errors in 9,747). R1 for comparison: 94.2 % (565 errors).

## How each part of the project fed the result

- **Simulation** (`docs/04_synthetic_speech.md`): 5,204 own fakes, 3,916 of them in every model's training data. The
  HiFi-GAN copy-synthesis of LibriSpeech speakers is the **hardest fake type in all of validation**: R1 is at chance
  (0.917) and R4ft at 0.13. VITS in the LJ voice is second (R4ft 0.045). The sim's artifact catalogue also motivated
  the 7 kHz / high-band handling. We did not run a no-sim ablation, so its effect on training is not measured.
- **Biology** (`docs/05_speech_biology.md`): 48 physiology features. Alone 0.600, and they add to spectral features in R1.
  Their biggest contribution was diagnostic: they exposed the 8 kHz channel shortcut in the given reals. In the final
  fusion they add nothing beyond R1 (R4ft + biology-only: 0.0141 validation).
- **External data**: took reals from 0.3 % to 31 % of the data. Without it the model would learn "LJ voice = real".
- **Scoring analysis** (`docs/01_challenge_and_scoring.md`): exact local scorer, both cost readings tracked everywhere, and the
  default-value game theory (no clip left unscored in the end).

## Challenge criteria (from the brief)

| criterion | status |
|---|---|
| 1,671 × 16 kHz English clips > 3 s, score 0.0 real → 1.0 synthetic | ✅ template order, all rows, 0..1 |
| Answer key format (TSV, keep every row, no 0.006 defaults) | ✅ validated; 0 unscored rows |
| ~70 % real, analyst scenario, false alarms on real cost 4× | ✅ eval sets are 70/30; both cost readings reported; at the brief's operating point only 0.22 % of reals are flagged |
| ASVspoof 5 Track-1 minDCF (organizers' code) | ✅ our scorer matches theirs exactly (`tests/test_metrics.py`) |
| Default-answer strategy | ✅ analysed (0.3); moot since every clip is scored |
| GitHub documentation | ✅ plans, docs, reports, README, status page |

## Submission

`submission/<team>_scores.tsv` via `scripts/make_submission.py --model R5_r4ft_r1 --sigmoid --temperature 8`:
1,671 rows in template order, all scored, **1,671 distinct values**, validated by `scripts/score.py validate`.
The temperature is a fixed, rank-preserving map sized from validation margins (max 38 for the fusion, 61 for R4ft). At T = 1 the sigmoid saturated to
exactly 0/1 at 10 digits and tied 28 % of eval scores. Nothing was fit on, normalized with, or chosen from HGT audio or
its scores.

## Process

- **Two machines:** a CPU coordinator and a GPU worker, exchanging messages and files over a small LAN service.
- **Cross-review:** 20+ PRs, each reviewed by the other machine. Reviews caught a submission writer that silently
  defaulted every row, leaky CV folds, a training weight derived from the headline set, resume-dependent early stopping,
  an overwritable frozen model choice, and a GiB/GB mix-up in the memory budget.
- **Compute:** R4ft trained for 144 min on an RTX 5050 Laptop (8 GB): 4.9 GiB allocated, bf16, gradient checkpointing,
  early-stopped at step 10,240 with the best at step 6,144.
