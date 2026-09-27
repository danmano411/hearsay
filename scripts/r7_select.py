"""R7 / final submission: apply the pre-registered Stage 11 rule (docs/00_development_log.md) and write the final
submission. Never reads HGT labels or feedback.

    python scripts/r7_select.py [--models R4ft_xlsr_light R6_xlsr_light R7a_xlsr_rb R7b_wavlm_rb] [--write]

Candidates: E7 = LR fusion of every listed SSL model that has scores + R1 (refit); E7s = the same without R1.
Baseline: E5 (R4ft + R6 + R1), recomputed here the same way. Guardrails relative to E5: val_testlike <= E5 + 0.003
(group-cross-fitted) and DeepVoice <= E5 + 0.02. Preference E7 > E7s (E7s if its val_testlike beats E7's by > 0.002);
if neither passes, E5 stays the final.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuse import apply_transform, decide_transform, fit  # noqa: E402
from r6_select import BENCH, load, official, r1_refit  # noqa: E402

from hearsay.audio import ROOT  # noqa: E402
from hearsay.evaluate import SCORES, manifest  # noqa: E402

VAL_TOL, DV_TOL, SMALLER_BY = 0.003, 0.02, 0.002  # the rule, fixed before any R7 result


def fusion(models, S, vt, y, group):
    """fuse.py recipe: transforms decided on val_testlike, LR (C=1, balanced) on standardized inputs, 5-fold group
    cross-fit for the val_testlike number, full fit for DeepVoice and HGT."""
    V = pd.concat([S[k]["eval"].reindex(vt.index).rename(k) for k in models], axis=1)
    kinds = {k: decide_transform(V[k]) for k in models}
    Z = lambda X: pd.DataFrame({k: apply_transform(X[k], kinds[k]) for k in models}, index=X.index)  # noqa: E731
    Zv = Z(V)
    cv = np.zeros(len(vt))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Zv, y, group):
        clf, mu, sd = fit(Zv.iloc[tr], y[tr], 1.0)
        cv[te] = clf.decision_function((Zv.iloc[te] - mu) / sd)
    clf, mu, sd = fit(Zv, y, 1.0)
    out = {"val": pd.Series(cv, vt.index), "weights": dict(zip(models, np.round(clf.coef_[0], 3)))}
    for kind in ("dv", "hgt"):
        X = pd.concat([S[k][kind].rename(k) for k in models], axis=1, join="inner")
        out[kind] = pd.Series(clf.decision_function((Z(X) - mu) / sd), index=X.index)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["R4ft_xlsr_light", "R6_xlsr_light", "R7a_xlsr_rb", "R7b_wavlm_rb"])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    m = manifest().set_index("path")
    vt = m[m.val_testlike]
    y = (vt.label == "spoof").astype(int).to_numpy()
    dv = pd.read_parquet(BENCH / "deepvoice.parquet").set_index("path")

    r1_eval, r1_dv, r1_hgt = r1_refit()
    S = {"R1": {"eval": r1_eval, "dv": r1_dv, "hgt": r1_hgt}}
    ssl = []
    for k in a.models:
        try:
            S[k] = {kind: load(k, kind) for kind in ("eval", "dv", "hgt")}
            ssl.append(k)
        except FileNotFoundError as e:
            print(f"{k}: not available ({Path(e.filename).name}), left out per the time box")
    cands = {"E5 (baseline)": ["R4ft_xlsr_light", "R6_xlsr_light", "R1"],
             "E7": ssl + ["R1"], "E7s": ssl}
    rows, fused = [], {}
    for name, models in cands.items():
        f = fusion(models, S, vt, y, vt.group)
        fused[name] = f
        d = f["dv"].reindex(dv.index).dropna()
        rows.append(dict(cand=name, models="+".join(models), val=official(f["val"], vt.label),
                         dv=official(d, dv.label.reindex(d.index)), weights=f["weights"]))
    t = pd.DataFrame(rows)
    base = t.iloc[0]
    t["passes"] = [None] + [bool(r.val <= base.val + VAL_TOL and r.dv <= base.dv + DV_TOL) for r in t.iloc[1:].itertuples()]
    print(t[["cand", "models", "val", "dv", "passes"]].round(4).to_string(index=False))
    for r in t.itertuples():
        print(f"  {r.cand} weights: {r.weights}")

    ok = {r.cand: r for r in t.iloc[1:].itertuples() if r.passes}
    pick = "E7" if "E7" in ok else ("E7s" if "E7s" in ok else None)
    if pick == "E7" and "E7s" in ok and ok["E7s"].val < ok["E7"].val - SMALLER_BY:
        pick = "E7s"
    if not [k for k in ssl if k not in ("R4ft_xlsr_light", "R6_xlsr_light")]:
        pick = None  # no R7 model finished inside the time box: nothing new to submit
    print(f"\nDECISION (pre-registered Stage 11 rule): {pick or 'none passes -> E5 stays the final'}")
    t["picked"] = t.cand == pick
    t.drop(columns="weights").to_csv(ROOT / "reports" / "08_r7_selection.csv", index=False)

    if pick:
        f = fused[pick]
        name = f"R7_final_{pick}"
        h = f["hgt"]
        pd.DataFrame({"filename": h.index, "score": h.values}).to_parquet(SCORES / f"{name}__hgt.parquet", index=False)
        temp = max(1.0, round(float(np.abs(f["val"]).max()) / 5))  # same sizing as the earlier files
        print(f"{name}: {len(h)} HGT scores; sigmoid temperature {temp}")
        if a.write:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--model", name,
                            "--sigmoid", "--temperature", str(temp), "--higher-is-real",
                            "--out", str(ROOT / "submission" / "HearsayScoreKey4GeorgiaMellon_R7_FINAL.tsv")], check=True)


if __name__ == "__main__":
    main()
