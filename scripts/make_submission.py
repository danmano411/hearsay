"""Write the challenge submission from a model's HGT scores, then validate it against the template.

    python scripts/make_submission.py --model R1_lgbm_all_full --team <team>
Reads data/scores/<model>__hgt.parquet (filename, score; higher = more fake), writes submission/<team>_scores.tsv
in template order ('filename<TAB>cm-score', LF, no BOM), and runs scripts/score.py's validator.
- Scores outside [0, 1] (logits/margins) go through a fixed sigmoid: monotone, so minDCF is unchanged, and nothing
  is fit on HGT audio or its score distribution.
- Clips the model did not score get DEFAULT = 0.3, the "no evidence" value at the 70/30 prior (docs/scoring.md §4),
  never the template's 0.006.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="reads data/scores/<model>__hgt.parquet")
    ap.add_argument("--team", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    s = pd.read_parquet(DATA / "scores" / f"{a.model}__hgt.parquet").set_index("filename").score
    if s.index.duplicated().any() or not np.isfinite(s).all():
        sys.exit("duplicate filenames or non-finite scores in the model's HGT file")
    if s.min() < 0 or s.max() > 1:
        s = 1 / (1 + np.exp(-s))
        print("scores outside [0,1]: applied a fixed sigmoid (monotone, minDCF unchanged)")
    names = [ln.split("\t")[0] for ln in Path(TEMPLATE).read_text().rstrip("\n").split("\n")[1:]]
    vals = s.reindex(names)
    n_missing = int(vals.isna().sum())
    vals = vals.fillna(DEFAULT)
    out = Path(a.out) if a.out else ROOT / "submission" / f"{a.team}_scores.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["filename\tcm-score"] + [f"{n}\t{v:.6f}" for n, v in zip(names, vals)]
    out.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

    errors, warnings = validate(out)
    print(f"{out}: {len(names)} rows from {a.model}, {n_missing} unscored -> {DEFAULT}")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("VALID" if not errors else "INVALID")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
