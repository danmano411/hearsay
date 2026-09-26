"""Write the challenge submission from a model's HGT scores, then validate it against the template.

    python scripts/make_submission.py --model R1_lgbm_all_full --team <team>
    python scripts/make_submission.py --model R4_xlsr_300m_lr --team <team> --sigmoid   # margin/logit models
Reads data/scores/<model>__hgt.parquet (filename, score; higher = more fake), writes submission/<team>_scores.tsv
in template order ('filename<TAB>cm-score', LF, no BOM), and runs scripts/score.py's validator.
- Every score filename must be a template filename and every template row must be scored, unless
  --max-unscored allows some; unscored rows get DEFAULT = 0.3 (docs/scoring.md §4), never the template's 0.006.
- Probability models write their scores as-is. Margin/logit models need --sigmoid, a fixed monotone map
  (minDCF unchanged). The choice is per model type, never decided from HGT scores.
- Scores are written with 10 significant digits so rounding doesn't create ties (ties never break in our favour).
  For large logits pass --temperature T (sigmoid(x / T)) so the sigmoid itself doesn't saturate into ties.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import TEMPLATE, validate  # noqa: E402

from hearsay.audio import DATA, ROOT  # noqa: E402

DEFAULT = 0.3


def build(scores, template=TEMPLATE, sigmoid=False, max_unscored=0, temperature=1.0):
    """scores: Series filename -> score. -> (template filenames, values, n_unscored). Raises ValueError."""
    names = [ln.split("\t")[0] for ln in Path(template).read_text().rstrip("\n").split("\n")[1:]]
    if not temperature > 0:  # 0 divides by zero; a negative T would silently reverse the ranking
        raise ValueError(f"--temperature must be > 0, got {temperature}")
    if scores.index.duplicated().any() or not np.isfinite(scores).all():
        raise ValueError("duplicate filenames or non-finite scores")
    extra = set(scores.index) - set(names)
    if extra:
        raise ValueError(f"{len(extra)} score filenames are not in the template, e.g. {sorted(extra)[:3]}")
    if sigmoid:
        scores = 1 / (1 + np.exp(-scores / temperature))
    elif scores.min() < 0 or scores.max() > 1:
        raise ValueError("scores outside [0, 1]: this is a margin/logit model, pass --sigmoid")
    vals = scores.reindex(names)
    n_unscored = int(vals.isna().sum())
    if n_unscored > max_unscored:
        raise ValueError(f"{n_unscored} template rows unscored (allowed: {max_unscored})")
    return names, vals.fillna(DEFAULT).to_numpy(), n_unscored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="reads data/scores/<model>__hgt.parquet")
    ap.add_argument("--team", help="output name submission/<team>_scores.tsv (or give --out)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigmoid", action="store_true", help="model outputs margins/logits, not probabilities")
    ap.add_argument("--max-unscored", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="with --sigmoid: sigmoid(x / T). Large logits saturate to exactly 0/1 at 10 digits and tie; size T "
                         "from the model's VALIDATION logit range (never from HGT scores). Rank-preserving, minDCF-neutral.")
    a = ap.parse_args()
    if not (a.team or a.out):
        ap.error("give --team or --out")

    s = pd.read_parquet(DATA / "scores" / f"{a.model}__hgt.parquet").set_index("filename").score
    try:
        names, vals, n_unscored = build(s, sigmoid=a.sigmoid, max_unscored=a.max_unscored, temperature=a.temperature)
    except ValueError as e:
        sys.exit(f"ERROR {e}")
    if n_unscored and a.sigmoid:
        print(f"WARN  {n_unscored} rows get {DEFAULT}, which is only 'no evidence' for probabilities calibrated to "
              f"the 0.3 prior; a sigmoid of a margin is not calibrated")
    out = Path(a.out) if a.out else ROOT / "submission" / f"{a.team}_scores.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["filename\tcm-score"] + [f"{n}\t{v:.10g}" for n, v in zip(names, vals)]
    out.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

    errors, warnings = validate(out)
    print(f"{out}: {len(names)} rows from {a.model}, {n_unscored} unscored -> {DEFAULT}, "
          f"{len(np.unique(vals))} distinct scores")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("VALID" if not errors else "INVALID")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
