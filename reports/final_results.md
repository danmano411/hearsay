# Final results

**Stopping criterion met** (owner's rule: headline combined minDCF ≤ 0.05). Submission = **R4ft_xlsr_light**.

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
| R5 | LR fusion R4ft + R1 (not selected, see below) | 0.0128 (cross-fit) | 0.0228 / 0.0207 / 0.0218 | 0.54 % |
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

## Selection (pre-registered before any R4ft headline existed; `plans/06_modeling_ladder.md`)

Candidates were compared on `val_testlike`: fusions by their group-cross-fitted number, and a fusion had to win by
≥ 0.002. R4ft alone (0.0121) beat both fusions (0.0128, 0.0127), so **R4ft is the submission**. The R4ft + R1 fusion
scores better on the headline (0.0218 vs 0.0282), but switching after seeing that would be selecting on the test set.

**Known bias in the rule (lesson, not acted on):** R4ft's `val_testlike` number is optimistic, because its checkpoint and
window mode were both chosen on that set, while the fusions' numbers are cross-fitted. The rule therefore leans toward
the single model. A fairer comparison would use `val` rows outside `val_testlike`, which neither candidate touched. We
record this instead of re-running selection after seeing headline numbers.

**Val → headline gap:** 0.0121 → 0.0282. Part is the selection optimism above; part is real, since the headline set's
held-out fakes and real corpora differ from validation. It is a reminder that the HGT score will likely be higher again.

## Submission

`submission/<team>_scores.tsv` via `scripts/make_submission.py --model R4ft_xlsr_light --sigmoid --temperature 8`:
1,671 rows in template order, all scored, **1,671 distinct values**, validated by `scripts/score.py validate`.
The temperature is a fixed, rank-preserving map sized from validation logits (max 61). At T = 1 the sigmoid saturated to
exactly 0/1 at 10 digits and tied 28 % of eval scores. Nothing was fit on, normalized with, or chosen from HGT audio or
its scores.

## Process

- **Two machines:** a CPU coordinator and a GPU worker, talking through a LAN hub ([`plans/07`](../plans/07_two_node_protocol.md)).
- **Cross-review:** 20+ PRs, each reviewed by the other machine. Reviews caught a submission writer that silently
  defaulted every row, leaky CV folds, a training weight derived from the headline set, resume-dependent early stopping,
  an overwritable frozen model choice, and a GiB/GB mix-up in the memory budget.
- **Compute:** R4ft trained for 144 min on an RTX 5050 Laptop (8 GB): 4.9 GiB allocated, bf16, gradient checkpointing,
  early-stopped at step 10,240 with the best at step 6,144.
