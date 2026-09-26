"""R0 (trivial cues) and R1 (classic ML) on data/features/classic.parquet (from scripts/extract_classic.py).

Train on split=='train' rows only; pick hyper-parameters by combined minDCF on val_testlike; report every model via
hearsay.evaluate.report (val / val_testlike / test_internal_testlike). The best R1 model also scores the HGT test
features (inference only) -> data/scores/<name>__hgt.parquet. Analysis tables -> data/scores/r1_analysis/,
figure -> docs/figures/r1_summary.png.

  python scripts/train_classic.py [--threads 4] [--no-svm]
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from hearsay.audio import DATA
from hearsay.data.stats import STAT_COLUMNS
from hearsay.evaluate import SCORES, manifest, metrics_for, per_source, report
from hearsay.features.bio import FEATURES as BIO
from hearsay.features.spectral import FEATURES as SPEC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEAT = DATA / "features"
OUT = SCORES / "r1_analysis"
FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"  # this checkout (figures are committed)
R0_RAW = [f"raw_{c}" for c in STAT_COLUMNS]
R0_PREP = [f"prep_{c}" for c in STAT_COLUMNS]


def family(f):
    if f.startswith(("lfcc", "mfcc")):
        return f[:4] + ("_delta" if "_d_" in f else "")
    return "bio" if f in BIO else "spectral"


def vt_dcf(model_scores, d):
    s = model_scores[d.val_testlike.values]
    return metrics_for(s, d.label[d.val_testlike])["combined"]


def fit_lgbm(tr, ev, feats, threads, grid):
    """Light grid; early stopping + selection on val_testlike only."""
    y = (tr.label == "spoof").astype(int)
    vt = ev[ev.val_testlike]
    best = None
    for params in grid:
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, class_weight="balanced", n_jobs=threads, verbose=-1,
                               random_state=0, **params)
        m.fit(tr[feats], y, eval_set=[(vt[feats], (vt.label == "spoof").astype(int))],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        s = m.predict_proba(ev[feats])[:, 1]
        dcf = vt_dcf(s, ev)
        print(f"   lgbm {params} iters={m.best_iteration_} vt_combined={dcf:.4f}", flush=True)
        if best is None or dcf < best[0]:
            best = (dcf, m, s, params)
    return best


def fit_sk(make, tr, ev, feats, grid, score_fn):
    y = (tr.label == "spoof").astype(int)
    best = None
    for p in grid:
        m = make(p).fit(tr[feats], y)
        s = score_fn(m, ev[feats])
        dcf = vt_dcf(s, ev)
        print(f"   {p} vt_combined={dcf:.4f}", flush=True)
        if best is None or dcf < best[0]:
            best = (dcf, m, s, p)
    return best


SUF = ""  # --suffix: distinguishes reruns (e.g. "_full") so earlier leaderboard rows/score files stay intact


def emit(name, ev, s, notes, results):
    name += SUF
    res = report(name, pd.DataFrame({"path": ev.path.values, "score": s}), notes=notes)
    results[name] = res.set_index("set")
    per_source(pd.DataFrame({"path": ev.path.values, "score": s})).to_csv(OUT / f"{name}_per_source.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-svm", action="store_true")
    ap.add_argument("--features", default=str(FEAT / "classic.parquet"))
    ap.add_argument("--suffix", default="", help="appended to every model name and output file")
    args = ap.parse_args()
    global SUF
    SUF = args.suffix
    OUT.mkdir(parents=True, exist_ok=True)

    m = manifest()
    d = pd.read_parquet(args.features)
    d = m.merge(d[d.err == ""], on="path")
    tr = d[d.split == "train"].reset_index(drop=True)
    ev = d[(d.split == "val") | d.val_testlike | d.test_internal_testlike].reset_index(drop=True)
    print(f"train {len(tr)} ({(tr.label == 'bonafide').sum()} real), eval {len(ev)}", flush=True)
    results = {}
    grid = [dict(num_leaves=15, min_child_samples=100), dict(num_leaves=63, min_child_samples=50),
            dict(num_leaves=255, min_child_samples=20)]

    # ---- R0: trivial cues, raw vs after prep()
    for tag, feats in (("raw", R0_RAW), ("prep", R0_PREP)):
        tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
        tree.fit(tr[feats], tr.label == "spoof")
        (OUT / f"R0_{tag}_tree.txt").write_text(export_text(tree, feature_names=feats))
        emit(f"R0_{tag}_tree", ev, tree.predict_proba(ev[feats])[:, 1], f"depth-3 tree, 9 trivial stats, {tag}",
             results)
        _, gbm, s, p = fit_lgbm(tr, ev, feats, args.threads, grid[:2])
        emit(f"R0_{tag}_lgbm", ev, s, f"LightGBM {p}, 9 trivial stats, {tag}", results)
        pd.Series(gbm.booster_.feature_importance("gain"), feats).sort_values().to_csv(OUT / f"R0_{tag}_gain.csv")

    # ---- R1: classic features after prep()
    # nolow: ablation without the 0-200 Hz contrast band (sub-200 Hz = rumble / recording-chain high-pass, a channel cue)
    sets = {"all": SPEC + BIO, "spec": SPEC, "bio": BIO, "nolow": [f for f in SPEC + BIO if not f.startswith("contrast0")]}
    models = {}
    for tag, feats in sets.items():
        print(f"R1 lgbm {tag} ({len(feats)} feats)", flush=True)
        dcf, gbm, s, p = fit_lgbm(tr, ev, feats, args.threads, grid)
        models[f"R1_lgbm_{tag}"] = (dcf, gbm, feats)
        emit(f"R1_lgbm_{tag}", ev, s, f"LightGBM {p} it={gbm.best_iteration_}, {len(feats)} feats", results)

    print("R1 logreg all", flush=True)
    feats = sets["all"]
    lr = lambda C: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),  # noqa: E731
                                 LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
    dcf, lrm, s, C = fit_sk(lr, tr, ev, feats, [0.01, 0.1, 1.0], lambda mm, X: mm.decision_function(X))
    models["R1_logreg_all"] = (dcf, lrm, feats)
    emit("R1_logreg_all", ev, s, f"standardized LR C={C}, {len(feats)} feats", results)

    if not args.no_svm:
        print("R1 svm-rbf all (15k train subsample)", flush=True)
        sub = pd.concat([g.sample(min(len(g), 7500), random_state=0) for _, g in tr.groupby("label")])
        svm = lambda C: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),  # noqa: E731
                                      SVC(C=C, gamma="scale", class_weight="balanced", cache_size=2000))
        dcf, svm_m, s, C = fit_sk(svm, sub, ev, feats, [1.0, 10.0], lambda mm, X: mm.decision_function(X))
        models["R1_svm_all"] = (dcf, svm_m, feats)
        # ponytail: 15k-clip subsample (O(n^2) kernel); all-train SVM only if it ever looks competitive
        emit("R1_svm_all", ev, s, f"SVM-RBF C={C}, 15k balanced train subsample", results)

    # ---- analysis on the full-feature LightGBM
    gbm = models["R1_lgbm_all"][1]
    feats = sets["all"]
    vt = ev[ev.val_testlike]
    contrib = gbm.predict(vt[feats], pred_contrib=True)[:, :-1]
    imp = pd.DataFrame({"feature": feats, "gain": gbm.booster_.feature_importance("gain"),
                        "mean_abs_shap": np.abs(contrib).mean(0),
                        # sign: mean contribution on real vs fake clips (positive = pushes toward fake)
                        "shap_real": contrib[(vt.label == "bonafide").values].mean(0),
                        "shap_fake": contrib[(vt.label == "spoof").values].mean(0)})
    imp["family"] = imp.feature.map(family)
    imp["gain"] /= imp.gain.sum()
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT / f"R1_lgbm_all{SUF}_importance.csv", index=False)
    print("\nfamily share of mean|SHAP|:\n", (imp.groupby("family").mean_abs_shap.sum() / imp.mean_abs_shap.sum())
          .sort_values(ascending=False).round(3).to_string())
    print("\ntop 25:\n", imp.head(25).round(4).to_string(index=False))
    print("\ntop bio:\n", imp[imp.family == "bio"].head(15).round(4).to_string(index=False))
    figure(results, imp)

    # ---- summary + HGT inference with the best R1 model (by val_testlike combined)
    summ = pd.DataFrame({n: {f"{s}_{k}": r.loc[s, k] for s in r.index for k in ("official", "brief", "combined")}
                         for n, r in results.items()}).T
    summ.to_csv(OUT / f"summary{SUF}.csv")
    print("\n", summ.round(4).to_string())
    best = min(models, key=lambda k: models[k][0])
    _, mdl, feats = models[best]
    best += SUF
    print(f"\nbest R1 by val_testlike combined: {best}")
    h = pd.read_parquet(FEAT / "classic_hgt.parquet")
    assert (h.err == "").all(), h[h.err != ""].head()
    if "lgbm" in best:
        s = mdl.predict_proba(h[feats])[:, 1]
    else:  # LR/SVM margin -> [0,1] by a fixed sigmoid (monotone; minDCF is rank-only, nothing fit on HGT)
        s = 1 / (1 + np.exp(-mdl.decision_function(h[feats])))
    pd.DataFrame({"filename": h.filename, "score": s}).to_parquet(SCORES / f"{best}__hgt.parquet", index=False)
    print(f"HGT: {len(h)} clips -> {SCORES / (best + '__hgt.parquet')}")


def figure(results, imp):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    names = list(results)
    x = np.arange(len(names))
    for i, (st, c) in enumerate([("val_testlike", "#8da0cb"), ("test_internal_testlike", "#1b3a6b")]):
        a.bar(x + (i - 0.5) * 0.38, [results[n].loc[st, "combined"] for n in names], 0.38, color=c, label=st)
    a.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
    a.set_ylabel("minDCF (combined reading, lower = better)")
    a.set_title("R0 trivial cues vs R1 classic features")
    a.axhline(1.0, color="grey", lw=0.8, ls="--")
    a.legend(fontsize=8)
    top = imp.head(25)[::-1]
    colors = {"bio": "#d95f02", "spectral": "#1b9e77", "lfcc": "#7570b3", "lfcc_delta": "#b3b0e0",
              "mfcc": "#e7298a", "mfcc_delta": "#f4a6cc"}
    b.barh(top.feature, top.mean_abs_shap, color=[colors[f] for f in top.family])
    b.tick_params(axis="y", labelsize=7)
    b.set_xlabel("mean |SHAP| on val_testlike (log-odds)")
    b.set_title("R1_lgbm_all: top 25 features")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], labels=list(colors),
             fontsize=7, loc="lower right")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"r1_summary{SUF}.png", dpi=90)
    print(f"figure -> {FIG / f'r1_summary{SUF}.png'}")


if __name__ == "__main__":
    main()
