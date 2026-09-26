"""One evaluation harness for every rung: fixed eval sets, both minDCF readings, leaderboard + cached scores.

Usage from a training script:
    from hearsay.evaluate import report
    report("R1_lgbm_lfcc", scores_df)   # scores_df: columns path, score (higher = more likely fake)
Scores for every manifest row the model scored are cached to data/scores/<name>.parquet for fusion (R5).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.audio import DATA, PROCESSED, ROOT
from hearsay.metrics import eer, min_dcf

SCORES = DATA / "scores"
LEADERBOARD = ROOT / "reports" / "leaderboard.md"
HEADER = ("| model | set | n real/fake | minDCF official | minDCF brief | minDCF combined | EER % | notes |\n"
          "|---|---|---|---|---|---|---|---|\n")
# Headline set = test_internal_testlike (70/30, never used for tuning). Tune on val / val_testlike only.
EVAL_SETS = ["val", "val_testlike", "test_internal_testlike"]


def manifest():
    return pd.read_parquet(PROCESSED / "manifest.parquet")


def eval_masks(m):
    return {"val": m.split == "val", "val_testlike": m.val_testlike, "test_internal_testlike": m.test_internal_testlike}


def metrics_for(scores, labels):
    y = (np.asarray(labels) == "spoof").astype(int)
    s = np.asarray(scores, dtype=float)
    return {
        "n": f"{(y == 0).sum()}/{y.sum()}",
        "official": min_dcf(s, y, "official_as_written"),
        "brief": min_dcf(s, y, "brief_as_written"),
        "combined": min_dcf(s, y, "combined"),
        "eer": 100 * eer(s, y),
    }


def evaluate(scores_df, m=None):
    """-> DataFrame, one row per eval set (only rows the model scored)."""
    m = manifest() if m is None else m
    df = m.merge(scores_df[["path", "score"]], on="path", how="inner")
    rows = []
    for name, mask in eval_masks(df).items():
        sub = df[mask]
        if sub.label.nunique() == 2:
            rows.append({"set": name, **metrics_for(sub.score, sub.label)})
    return pd.DataFrame(rows)


def per_source(scores_df, split_mask="test_internal_testlike", m=None):
    """Breakdown for error analysis: minDCF of each source's reals vs all fakes, and each generator's fakes vs all reals."""
    m = manifest() if m is None else m
    df = m.merge(scores_df[["path", "score"]], on="path")
    df = df[df[split_mask]] if split_mask in df else df[df.split == split_mask]
    real, fake = df[df.label == "bonafide"], df[df.label == "spoof"]
    out = []
    for key, grp, other in [("source", real, fake), ("generator", fake, real)]:
        for v, g in grp.groupby(key):
            both = pd.concat([g, other])
            out.append({"by": key, "value": v, "n": len(g), **metrics_for(both.score, both.label)})
    return pd.DataFrame(out).sort_values("combined", ascending=False)


def report(name, scores_df, notes="", m=None):
    """Evaluate, cache scores, append rows to reports/leaderboard.md. Returns the results DataFrame."""
    SCORES.mkdir(parents=True, exist_ok=True)
    scores_df[["path", "score"]].to_parquet(SCORES / f"{name}.parquet", index=False)
    res = evaluate(scores_df, m)
    if not LEADERBOARD.exists() or LEADERBOARD.stat().st_size == 0:
        LEADERBOARD.write_text("# Leaderboard\n\nLower minDCF is better (0 = perfect, 1 = trivial). "
                               "Headline = `test_internal_testlike`.\n\n" + HEADER, encoding="utf-8")
    with open(LEADERBOARD, "a", encoding="utf-8") as f:
        for r in res.itertuples():
            f.write(f"| {name} | {r.set} | {r.n} | {r.official:.4f} | {r.brief:.4f} | {r.combined:.4f} | "
                    f"{r.eer:.2f} | {notes} |\n")
    print(res.to_string(index=False))
    return res
