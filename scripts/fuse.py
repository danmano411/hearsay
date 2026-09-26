"""R5: score-level fusion. Logistic regression over several models' cached scores.

    python scripts/fuse.py --name R5_fuse_a --models R1_lgbm_all_full R4_xlsr_300m_lr R3_aasist_ft
Fit ONLY on val_testlike rows (the same rows every model was tuned on); the headline stays test_internal_testlike,
which nothing here touches. Per-model input transform is fixed by type, never fit on eval or HGT scores:
probabilities in [0,1] -> logit(clip(p)); anything else is taken as a margin. Inputs are standardized with
val_testlike statistics, then the fitted weights go to test_internal_testlike, val and HGT unchanged.
Reports val_testlike via 5-fold cross-fitting (an in-sample number would be optimistic) and writes
data/scores/<name>.parquet (+ __hgt.parquet when every model has HGT scores).
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from hearsay.evaluate import SCORES, manifest, report
from hearsay.metrics import min_dcf


def to_logit(s):
    s = np.asarray(s, dtype=float)
    if s.min() >= 0 and s.max() <= 1:
        s = np.clip(s, 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s))
    return s


def load(models, hgt=False):
    frames = []
    for mdl in models:
        f = SCORES / (f"{mdl}__hgt.parquet" if hgt else f"{mdl}.parquet")
        d = pd.read_parquet(f).rename(columns={"score": mdl})
        frames.append(d.set_index("filename" if hgt else "path")[mdl])
    return pd.concat(frames, axis=1, join="inner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()

    m = manifest().set_index("path")
    X = load(a.models)
    X = X.join(m[["label", "val_testlike", "test_internal_testlike", "split"]], how="inner")
    vt = X[X.val_testlike]
    missing = m.index[m.val_testlike | m.test_internal_testlike].difference(X.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} val_testlike/test_internal_testlike rows lack scores from some model")
    Z = pd.DataFrame({mdl: to_logit(X[mdl]) for mdl in a.models}, index=X.index)
    mu, sd = Z.loc[vt.index].mean(), Z.loc[vt.index].std() + 1e-9
    Zs = (Z - mu) / sd
    y = (vt.label == "spoof").astype(int).to_numpy()

    # honest val_testlike number: 5-fold cross-fitted scores
    cv = np.zeros(len(vt))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(vt, y):
        clf = LogisticRegression(C=a.C, class_weight="balanced").fit(Zs.loc[vt.index].iloc[tr], y[tr])
        cv[te] = clf.decision_function(Zs.loc[vt.index].iloc[te])
    print(f"val_testlike cross-fitted combined minDCF: {min_dcf(cv, y, 'combined'):.4f}")

    clf = LogisticRegression(C=a.C, class_weight="balanced").fit(Zs.loc[vt.index], y)
    w = dict(zip(a.models, np.round(clf.coef_[0], 4)))
    print("weights (standardized logits):", w)
    s = pd.Series(clf.decision_function(Zs), index=Zs.index)
    s.loc[vt.index] = cv  # report the cross-fitted scores on the rows the weights were fit on
    res = report(a.name, pd.DataFrame({"path": s.index, "score": s.values}),
                 notes=f"LR fusion of {'+'.join(a.models)}; fit on val_testlike (5-fold cross-fit shown there)")

    try:
        H = load(a.models, hgt=True)
    except FileNotFoundError as e:
        print(f"no HGT fusion: {e}")
        return
    Hz = (pd.DataFrame({mdl: to_logit(H[mdl]) for mdl in a.models}, index=H.index) - mu) / sd
    out = pd.DataFrame({"filename": Hz.index, "score": clf.decision_function(Hz)})
    out.to_parquet(SCORES / f"{a.name}__hgt.parquet", index=False)
    (SCORES / f"{a.name}.json").write_text(json.dumps({"models": a.models, "weights": w, "mu": mu.to_dict(),
                                                       "sd": sd.to_dict(), "C": a.C}, indent=1))
    print(f"HGT fused: {len(out)} clips -> {a.name}__hgt.parquet (margins: use make_submission --sigmoid)")


if __name__ == "__main__":
    main()
