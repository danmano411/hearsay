"""R5: score-level fusion. Logistic regression over several models' cached scores.

    python scripts/fuse.py --name R5_fuse_a --models R1_lgbm_all_full R4_xlsr_300m_lr R3_aasist_ft
- Fit ONLY on val_testlike rows; the headline (test_internal_testlike) is never touched by fitting.
- Per-model input transform is decided ONCE from that model's val_testlike scores (all finite and inside [0,1] ->
  probability -> logit(clip(p)); otherwise a margin, used as-is), saved to data/scores/<name>.json, and applied
  unchanged to test_internal_testlike, val and HGT. Nothing is chosen from HGT scores.
- The val_testlike number is 5-fold cross-fitted with StratifiedGroupKFold over manifest `group` (clips of one sentence
  or speaker never straddle folds), with standardization fit inside each fold. Base models were themselves tuned or
  early-stopped on val_testlike, so that number is still mildly optimistic; test_internal_testlike is the honest one.
- HGT: every model must score every HGT clip (fusion needs aligned inputs); output is a margin, so use
  make_submission --sigmoid.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from hearsay.evaluate import SCORES, manifest, report
from hearsay.metrics import min_dcf


def decide_transform(s):
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).all():
        raise SystemExit("non-finite scores in val_testlike")
    return "logit" if s.min() >= 0 and s.max() <= 1 else "margin"


def apply_transform(s, kind):
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).all():
        raise SystemExit("non-finite scores")
    if kind == "logit":
        s = np.clip(s, 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s))
    return s


def load(models, hgt=False):
    frames = []
    for mdl in models:
        d = pd.read_parquet(SCORES / (f"{mdl}__hgt.parquet" if hgt else f"{mdl}.parquet"))
        frames.append(d.set_index("filename" if hgt else "path")["score"].rename(mdl))
    return pd.concat(frames, axis=1, join="outer")


def fit(Z, y, C):
    mu, sd = Z.mean(), Z.std() + 1e-9
    return LogisticRegression(C=C, class_weight="balanced").fit((Z - mu) / sd, y), mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()

    m = manifest().set_index("path")
    X = load(a.models)
    need = m.index[m.val_testlike | m.test_internal_testlike]
    lacking = need.difference(X.dropna().index)
    if len(lacking):
        raise SystemExit(f"{len(lacking)} val_testlike/test_internal_testlike rows lack a score from some model")
    X = X.dropna().join(m[["label", "group", "val_testlike"]], how="inner")
    vt = X[X.val_testlike]
    kinds = {mdl: decide_transform(vt[mdl]) for mdl in a.models}
    Z = pd.DataFrame({mdl: apply_transform(X[mdl], kinds[mdl]) for mdl in a.models}, index=X.index)
    Zv, y = Z.loc[vt.index], (vt.label == "spoof").astype(int).to_numpy()

    cv = np.zeros(len(vt))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Zv, y, vt.group):
        clf, mu, sd = fit(Zv.iloc[tr], y[tr], a.C)
        cv[te] = clf.decision_function((Zv.iloc[te] - mu) / sd)
    print(f"val_testlike group-cross-fitted combined minDCF: {min_dcf(cv, y, 'combined'):.4f}")

    clf, mu, sd = fit(Zv, y, a.C)
    w = {k: float(v) for k, v in zip(a.models, clf.coef_[0])}
    print("transforms:", kinds, "\nweights (standardized):", {k: round(v, 4) for k, v in w.items()})
    (SCORES / f"{a.name}.json").write_text(json.dumps(
        {"models": a.models, "transforms": kinds, "weights": w, "intercept": float(clf.intercept_[0]),
         "mu": mu.to_dict(), "sd": sd.to_dict(), "C": a.C}, indent=1))

    s = pd.Series(clf.decision_function((Z - mu) / sd), index=Z.index)
    s.loc[vt.index] = cv  # rows the weights were fit on get their cross-fitted score
    report(a.name, pd.DataFrame({"path": s.index, "score": s.values}),
           notes=f"LR fusion of {'+'.join(a.models)}; fit on val_testlike (group 5-fold cross-fit shown there; "
                 f"'val' mixes cross-fit and full-fit rows; base models early-stopped on val_testlike)")

    try:
        H = load(a.models, hgt=True)
    except FileNotFoundError as e:
        print(f"no HGT fusion (a model has no HGT scores): {e.filename}")
        return
    if H.isna().any().any():
        raise SystemExit(f"{int(H.isna().any(axis=1).sum())} HGT clips lack a score from some model")
    Hz = pd.DataFrame({mdl: apply_transform(H[mdl], kinds[mdl]) for mdl in a.models}, index=H.index)
    out = pd.DataFrame({"filename": Hz.index, "score": clf.decision_function((Hz - mu) / sd)})
    out.to_parquet(SCORES / f"{a.name}__hgt.parquet", index=False)
    print(f"HGT fused: {len(out)} clips -> {a.name}__hgt.parquet (margins: make_submission --sigmoid)")


if __name__ == "__main__":
    main()
