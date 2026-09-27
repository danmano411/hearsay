"""Scoring CLI (see docs/01_challenge_and_scoring.md).

  python scripts/score.py metrics  SCORES.tsv KEY.tsv [--higher-is-bonafide]
  python scripts/score.py validate SUBMISSION.tsv [--template TEMPLATE]
  python scripts/score.py simulate [--out docs/figures] [--reps 200]

SCORES.tsv: 'filename<TAB>cm-score' (our direction: higher = fake unless --higher-is-bonafide).
KEY.tsv:    'filename<TAB>cm-label' with labels bonafide/spoof (organizers' key format).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.audio import DATA
from hearsay.metrics import INTERPRETATIONS, all_metrics, min_dcf

TEMPLATE = DATA / "raw/hgt_test/HGT_Hearsay_score_template.csv"
HEADER = "filename\tcm-score"
TEMPLATE_DEFAULT = 0.006


def cmd_metrics(a):
    s = pd.read_csv(a.scores, sep="\t").set_index("filename")
    k = pd.read_csv(a.key, sep="\t").set_index("filename")
    if set(s.index) != set(k.index) or s.index.duplicated().any():
        sys.exit(f"score/key filenames differ or duplicate ({len(s)} scores, {len(k)} keys)")
    df = s.join(k)
    m = all_metrics(df["cm-score"].to_numpy(), df["cm-label"].to_numpy(), higher_is_fake=not a.higher_is_bonafide)
    n_b, n_s = (df["cm-label"] == "bonafide").sum(), (df["cm-label"] == "spoof").sum()
    print(f"trials: {len(df)}  bonafide: {n_b}  spoof: {n_s}")
    for name, v in m.items():
        print(f"{name:28s} {v:.6f}" if "eer" not in name else f"{name:28s} {100 * v:.3f} %")


def validate(path, template=TEMPLATE):
    """-> (errors, warnings). Errors = the organizers' scorer would fail or mis-join; warnings = likely mistakes."""
    errors, warnings = [], []
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("UTF-8 BOM at start of file (header would read '\\ufefffilename')")
        raw = raw[3:]
    if b"\r" in raw:
        warnings.append("CRLF line endings (template uses LF)")
    lines = raw.decode("utf-8").replace("\r\n", "\n").rstrip("\n").split("\n")
    tmpl = Path(template).read_text().rstrip("\n").split("\n")
    if lines[0] != HEADER:
        errors.append(f"header is {lines[0]!r}, expected {HEADER!r}")
    rows = [ln.split("\t") for ln in lines[1:]]
    bad = [i + 2 for i, r in enumerate(rows) if len(r) != 2]
    if bad:
        errors.append(f"{len(bad)} rows are not exactly 2 TAB-separated fields (first at line {bad[0]})")
    names = [r[0] for r in rows]
    want = [ln.split("\t")[0] for ln in tmpl[1:]]
    if len(names) != len(want):
        errors.append(f"{len(names)} rows, template has {len(want)}")
    if names != want:
        missing, extra = set(want) - set(names), set(names) - set(want)
        first = next((i for i, (x, y) in enumerate(zip(names, want)) if x != y), None)
        errors.append(f"filenames differ from template: {len(missing)} missing, {len(extra)} extra, "
                      f"{len(names) - len(set(names))} duplicates, first order mismatch at data row {first}")
    vals = []
    for i, r in enumerate(rows):
        try:
            v = float(r[1])
        except (ValueError, IndexError):
            errors.append(f"line {i + 2}: score {r[1:]!r} is not a number")
            continue
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            errors.append(f"line {i + 2}: score {v} not in [0, 1]")
        vals.append(v)
    vals = np.array(vals)
    if vals.size:
        n_def = int(np.sum(vals == TEMPLATE_DEFAULT))
        if n_def:
            warnings.append(f"{n_def} rows still hold the template default {TEMPLATE_DEFAULT}")
        if np.unique(vals).size < 3:
            warnings.append(f"only {np.unique(vals).size} distinct scores: minDCF is 1.0 for a constant file")
    return errors, warnings


def cmd_validate(a):
    errors, warnings = validate(a.submission, a.template)
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"FAILED ({len(errors)} errors)")
    sys.exit(1 if errors else 0)


# ---------------------------------------------------------------- simulate (figures + tables for docs/01_challenge_and_scoring.md)
N, P_FAKE = 1671, 0.3  # test set size and ~70/30 real/fake (brief)


def calibrated_scores(rng, n, d, p_fake=P_FAKE):
    """Equal-variance Gaussian system with separation d', scores = exact posterior P(fake | x) at prior p_fake."""
    y = rng.random(n) < p_fake
    x = rng.normal(size=n) + d * y
    llr = d * x - d * d / 2
    return 1 / (1 + (1 - p_fake) / p_fake * np.exp(-llr)), y


def brute_min_dcf(s, y, w_fa_real, w_miss_fake):
    """Derivation check: min over thresholds t (flag fake iff s >= t) of w1 P(real flagged) + w2 P(fake passed)."""
    ts = np.concatenate([np.unique(s), [np.inf]])
    return min(w_fa_real * np.mean(s[~y] >= t) + w_miss_fake * np.mean(s[y] < t) for t in ts)


def cmd_simulate(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # 1) derivation check: closed forms == organizers-equivalent min_dcf
    for _ in range(20):
        s, y = calibrated_scores(rng, 300, rng.uniform(0, 3))
        s = np.round(s, 2)  # include ties
        assert abs(brute_min_dcf(s, y, 1, 4) - min_dcf(s, y, "official_as_written")) < 1e-12
        assert abs(brute_min_dcf(s, y, 4, 1) - min_dcf(s, y, "brief_as_written")) < 1e-12
    print("derivation check passed: official = min_t P(real flagged) + 4 P(fake passed); brief = 4x / 1x")

    # 2) iso-cost figure on a ROC (x = real flagged rate, y = fake caught rate)
    s, y = calibrated_scores(rng, 200_000, 2.0)
    ts = np.quantile(s, np.linspace(0, 1, 2001))
    fpr = np.array([np.mean(s[~y] >= t) for t in ts])
    tpr = np.array([np.mean(s[y] >= t) for t in ts])
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot(fpr, tpr, color="#333", lw=2, label="example system (d' = 2)")
    xs = np.linspace(0, 1, 50)
    styles = {"official_as_written": ("#1f77b4", 1, 4), "brief_as_written": ("#d62728", 4, 1)}
    rows = []
    for name, (col, wf, wm) in styles.items():
        cost = wf * fpr + wm * (1 - tpr)
        i = int(np.argmin(cost))
        c = cost[i]
        # iso-cost line through the optimum: wf*x + wm*(1-y) = c  ->  y = 1 - (c - wf*x)/wm
        ax.plot(xs, 1 - (c - wf * xs) / wm, "--", color=col, lw=1.2,
                label=f"{name}: iso-cost {wf}·FPR + {wm}·FNR = {c:.3f}")
        ax.plot(xs, 1 - (1 - wf * xs) / wm, ":", color=col, lw=1, label=f"{name}: minDCF = 1 (trivial) line")
        ax.plot(fpr[i], tpr[i], "o", color=col, ms=8)
        rows.append((name, fpr[i], tpr[i], c, ts[i]))
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="real clips flagged as fake (FPR)",
           ylabel="fake clips caught (TPR)", title="Where each reading of the cost puts the operating point")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "scoring_iso_cost.png", dpi=150)
    plt.close(fig)
    print("\noperating points (d'=2, calibrated posterior at prior 0.3):")
    print("| reading | FPR | TPR | minDCF | threshold on P(fake) |")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} |")

    # 3) default-value game: K of N clips get one constant c; where should c sit?
    cs = np.concatenate([[0.0, 0.006], np.linspace(0.02, 0.98, 49), [0.999, 1.0]])
    named = {"0.006 (template)": 0.006, "0.097 (official Bayes thr)": 0.0968, "0.3 (prior)": 0.3,
             "0.632 (brief Bayes thr)": 0.6316, "1.0": 1.0}
    grid = sorted(set(cs) | set(named.values()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    table = []
    for ax, d in zip(axes, (2.0, 3.0)):
        for K, ls in ((84, ":"), (167, "--"), (418, "-")):
            res = {k: np.zeros(len(grid)) for k in [*INTERPRETATIONS, "combined"]}
            base = {k: 0.0 for k in res}
            for _ in range(a.reps):
                s, y = calibrated_scores(rng, N, d)
                for k in res:
                    base[k] += min_dcf(s, y, k) / a.reps
                miss = rng.choice(N, K, replace=False)
                for j, c in enumerate(grid):
                    s2 = s.copy()
                    s2[miss] = c
                    for k in res:
                        res[k][j] += min_dcf(s2, y, k) / a.reps
            for k, col in (("official_as_written", "#1f77b4"), ("brief_as_written", "#d62728"),
                           ("combined", "#2ca02c")):
                ax.plot(grid, res[k], ls, color=col, lw=1.4, label=f"{k}, K={K}" if d == 2.0 else None)
            for lab, c in named.items():
                j = grid.index(c)
                table.append((d, K, lab, *(res[k][j] for k in res), *(base[k] for k in res)))
        for c in (0.0968, 0.3, 0.6316):
            ax.axvline(c, color="#999", lw=0.8)
        ax.set(title=f"d' = {d}  (K unscored of N = {N}, 70/30)", xlabel="constant given to unscored clips, c")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"mean minDCF over {a.reps} draws")
    axes[0].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out / "scoring_default_value.png", dpi=150)
    plt.close(fig)
    print("\n| d' | K | default c | official | brief | combined | (no defaults: official / brief / combined) |")
    for r in table:
        print(f"| {r[0]:.0f} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} / {r[7]:.3f} / {r[8]:.3f} |")

    # 4) trivial submissions
    s, y = calibrated_scores(rng, N, 2.0)
    print("\ntrivial all-constant file:", {k: min_dcf(np.full(N, 0.006), y, k) for k in [*INTERPRETATIONS, "combined"]})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("metrics", help="all metrics, both cost readings")
    m.add_argument("scores")
    m.add_argument("key")
    m.add_argument("--higher-is-bonafide", action="store_true", help="scores are ASVspoof direction")
    m.set_defaults(fn=cmd_metrics)
    v = sub.add_parser("validate", help="check a submission file against the template")
    v.add_argument("submission")
    v.add_argument("--template", default=TEMPLATE)
    v.set_defaults(fn=cmd_validate)
    s = sub.add_parser("simulate", help="regenerate docs/figures/scoring_* and the doc's tables")
    s.add_argument("--out", default=Path(__file__).resolve().parents[1] / "docs/figures")
    s.add_argument("--reps", type=int, default=200)
    s.set_defaults(fn=cmd_simulate)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
