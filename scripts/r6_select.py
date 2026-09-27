"""R6 / final submission: apply the pre-registered rule (docs/00_development_log.md, stage 10) and write the final
submission. Never reads HGT labels or feedback.

    python scripts/r6_select.py            # table + decision -> reports/07_r6_selection.csv, data/scores/<pick>__hgt.parquet
    python scripts/r6_select.py --write    # also builds submission/HearsayScoreKey4GeorgiaMellon_FINAL.tsv (higher = real)

Inputs: data/scores/{R4ft_xlsr_light,R6_xlsr_light}{,__hgt}.parquet, data/bench/scores/*__deepvoice.parquet, the R1
refit booster (data/models/r1_lgbm_all_full.txt) with data/features/classic{,_hgt}.parquet and
data/bench/features/deepvoice.parquet. R1 inside every new candidate is the refit, so val, DeepVoice and HGT all use
one model. R5 (the baseline) keeps its frozen cross-fitted val scores (data/scores/R5_r4ft_r1.parquet).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_score import fused  # noqa: E402
from fuse import apply_transform, decide_transform, fit  # noqa: E402

from hearsay.audio import DATA, ROOT  # noqa: E402
from hearsay.evaluate import SCORES, manifest  # noqa: E402
from hearsay.features.bio import FEATURES as BIO  # noqa: E402
from hearsay.features.spectral import FEATURES as SPEC  # noqa: E402
from hearsay.metrics import min_dcf  # noqa: E402

BENCH = DATA / "bench"
VAL_TOL, DV_TOL, E5_VS_E = 0.003, 0.02, 0.002  # the rule, fixed before R6 existed


def official(s, y):
    """Organizers' scorer: Pspoof 0.3, Cmiss 1, Cfa 4, ASVspoof direction (our min_dcf flips s internally)."""
    return min_dcf(np.asarray(s, float), np.asarray(y), "official_as_written", pspoof=0.3)


def r1_refit():
    """R1 refit scores on val_testlike + test_internal_testlike (by path), DeepVoice (by path) and HGT (by filename)."""
    b = lgb.Booster(model_file=str(DATA / "models" / "r1_lgbm_all_full.txt"))
    f = pd.read_parquet(DATA / "features" / "classic.parquet").query("err == ''").set_index("path")
    d = pd.read_parquet(BENCH / "features" / "deepvoice.parquet").query("err == ''").set_index("path")
    h = pd.read_parquet(DATA / "features" / "classic_hgt.parquet").set_index("filename")
    return (pd.Series(b.predict(f[SPEC + BIO]), f.index), pd.Series(b.predict(d[SPEC + BIO]), d.index),
            pd.Series(b.predict(h[SPEC + BIO]), h.index))


def load(name, kind):
    f = {"eval": SCORES / f"{name}.parquet", "hgt": SCORES / f"{name}__hgt.parquet",
         "dv": BENCH / "scores" / f"{name}__deepvoice.parquet"}[kind]
    return pd.read_parquet(f).set_index("filename" if kind == "hgt" else "path").score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--r6", default="R6_xlsr_light", help="R6 score name (a stand-in only for dry runs)")
    a = ap.parse_args()
    m = manifest().set_index("path")
    vt = m[m.val_testlike]
    y_vt = (vt.label == "spoof").astype(int).to_numpy()
    dv = pd.read_parquet(BENCH / "deepvoice.parquet").set_index("path")

    r1_eval, r1_dv, r1_hgt = r1_refit()
    S = {k: {"eval": load(k, "eval"), "dv": load(k, "dv"), "hgt": load(k, "hgt")}
         for k in ("R4ft_xlsr_light", a.r6)}
    S["R6_xlsr_light"] = S.pop(a.r6)
    S["R1"] = {"eval": r1_eval, "dv": r1_dv, "hgt": r1_hgt}

    def aligned(models, kind, idx):
        return pd.concat([S[k][kind].reindex(idx).rename(k) for k in models], axis=1)

    cands, rows = {}, []
    # R6 alone
    cands["R6"] = {k: S["R6_xlsr_light"][k] for k in ("eval", "dv", "hgt")}
    # E: fixed equal weights on val_testlike-standardized margins
    pair = ["R4ft_xlsr_light", "R6_xlsr_light"]
    V = aligned(pair, "eval", vt.index)
    mu, sd = V.mean(), V.std()
    cands["E"] = {kind: ((aligned(pair, kind, S[pair[0]][kind].index) - mu) / sd).mean(axis=1)
                  for kind in ("eval", "dv", "hgt")}
    # E5: LR fusion of R4ft + R6 + R1 (fuse.py recipe: transforms decided on val_testlike, group 5-fold cross-fit)
    tri = pair + ["R1"]
    Vt = aligned(tri, "eval", vt.index)
    kinds = {k: decide_transform(Vt[k]) for k in tri}
    Z = lambda X: pd.DataFrame({k: apply_transform(X[k], kinds[k]) for k in tri}, index=X.index)  # noqa: E731
    Zv = Z(Vt)
    cv = np.zeros(len(vt))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Zv, y_vt, vt.group):
        clf, mu3, sd3 = fit(Zv.iloc[tr], y_vt[tr], 1.0)
        cv[te] = clf.decision_function((Zv.iloc[te] - mu3) / sd3)
    clf, mu3, sd3 = fit(Zv, y_vt, 1.0)
    e5 = {kind: pd.Series(clf.decision_function((Z(aligned(tri, kind, S[pair[0]][kind].index).dropna()) - mu3) / sd3),
                          index=aligned(tri, kind, S[pair[0]][kind].index).dropna().index)
          for kind in ("dv", "hgt")}
    e5["eval_vt_crossfit"] = pd.Series(cv, vt.index)
    cands["E5"] = e5
    print("E5 weights (standardized):", dict(zip(tri, np.round(clf.coef_[0], 3))), "transforms:", kinds)

    # baseline R5 (frozen): cross-fitted val scores; DeepVoice via the frozen fusion weights on R4ft + refit R1
    r5_vt = pd.read_parquet(SCORES / "R5_r4ft_r1.parquet").set_index("path").score.reindex(vt.index)
    r5_dv = fused({"R4ft_xlsr_light": S["R4ft_xlsr_light"]["dv"], "R1_lgbm_all_full": r1_dv})
    base = {"val": official(r5_vt, vt.label), "dv": official(r5_dv, dv.label.reindex(r5_dv.index))}
    rows.append(dict(cand="R5 (baseline)", val=base["val"], dv=base["dv"], passes=None))

    for name, c in cands.items():
        v = c["eval_vt_crossfit"] if "eval_vt_crossfit" in c else c["eval"]
        val = official(v.reindex(vt.index), vt.label)
        d = c["dv"].reindex(dv.index).dropna()
        dvs = official(d, dv.label.reindex(d.index))
        ok = val <= base["val"] + VAL_TOL and dvs <= base["dv"] + DV_TOL
        rows.append(dict(cand=name, val=val, dv=dvs, passes=ok))
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))

    ok = {r.cand: r for r in t.itertuples() if r.passes}
    pick = next((c for c in ("E5", "E", "R6") if c in ok), None)
    if pick == "E5" and "E" in ok and ok["E5"].val > ok["E"].val + E5_VS_E:
        pick = "E"
    print(f"\nDECISION (pre-registered rule): {pick or 'none passes -> final stays R5'}")
    t["picked"] = t.cand == pick
    out = ROOT / "reports" / "07_r6_selection.csv"
    t.to_csv(out, index=False)

    if pick:
        h = cands[pick]["hgt"]
        name = f"R6_final_{pick}"
        pd.DataFrame({"filename": h.index, "score": h.values}).to_parquet(SCORES / f"{name}__hgt.parquet", index=False)
        c = cands[pick]
        vmax = float(np.abs((c["eval_vt_crossfit"] if "eval_vt_crossfit" in c else c["eval"].reindex(vt.index)).dropna()).max())
        temp = max(1.0, round(vmax / 5))  # same sizing as the R5 file (max 38 -> T 8): no sigmoid saturation ties
        print(f"{name}: {len(h)} HGT scores; sigmoid temperature {temp} (from val margins, max {vmax:.1f})")
        if a.write:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "make_submission.py"), "--model", name,
                            "--sigmoid", "--temperature", str(temp), "--higher-is-real",
                            "--out", str(ROOT / "submission" / "HearsayScoreKey4GeorgiaMellon_FINAL.tsv")], check=True)


if __name__ == "__main__":
    main()
