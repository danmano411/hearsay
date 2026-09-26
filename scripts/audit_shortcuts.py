"""Shortcut audit (plan 01 step 6): can a depth-3 tree on trivial cues separate real from fake?

Features = hearsay.data.stats.STAT_COLUMNS on the canonical 16 kHz clip (duration, level, silence, bandwidth...).
Fit on split=train, score on val + test_internal (pooled: the real class is tiny). minDCF well below 0.5 = those
cues leak and must be neutralized before modeling. Stats for manifest rows not yet in
data/processed/audit/clip_stats.parquet (e.g. new sources) are computed and cached.

  python scripts/audit_shortcuts.py [--workers 8]
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.tree import DecisionTreeClassifier, export_text

from hearsay.audio import PROCESSED, ROOT
from hearsay.data.stats import STAT_COLUMNS, clip_stats

STATS = PROCESSED / "audit" / "clip_stats.parquet"


def min_dcf(score_spoof, is_spoof, p_spoof=0.05, c_miss=1.0, c_fa=10.0):
    """ASVspoof5 Track-1 normalized minDCF. miss = bonafide rejected, fa = spoof accepted.
    ponytail: local copy for this audit only; hearsay.metrics (phase 5) is the reference implementation."""
    s, y = np.asarray(score_spoof, float), np.asarray(is_spoof, bool)
    thr = np.concatenate([[-np.inf], np.unique(s), [np.inf]])
    p_miss = np.array([(s[~y] >= t).mean() for t in thr])  # bonafide called spoof
    p_fa = np.array([(s[y] < t).mean() for t in thr])  # spoof called bonafide
    dcf = c_miss * (1 - p_spoof) * p_miss + c_fa * p_spoof * p_fa
    return float(dcf.min() / min(c_miss * (1 - p_spoof), c_fa * p_spoof))


def _stats(path):
    y, sr = sf.read(str(ROOT / path), dtype="float64")
    return {"path": path, **clip_stats(y, sr)}


def load_stats(paths, workers):
    cache = pd.read_parquet(STATS) if STATS.exists() else pd.DataFrame(columns=["path"] + STAT_COLUMNS)
    todo = sorted(set(paths) - set(cache["path"]))
    if todo:
        print(f"computing stats for {len(todo)} new clips")
        with Pool(workers) as pool:
            cache = pd.concat([cache, pd.DataFrame(pool.map(_stats, todo, chunksize=64))], ignore_index=True)
        cache.to_parquet(STATS, index=False)
    return cache


def run(df, name, feats):
    tr, ev = df[df.split == "train"], df[df.split.isin(["val", "test_internal"])]
    if tr.label.nunique() < 2 or ev.label.nunique() < 2:
        return None
    clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    clf.fit(tr[feats], tr.label == "spoof")
    score = clf.predict_proba(ev[feats])[:, 1]
    return {"subset": name, "features": ",".join(feats) if len(feats) < 3 else "all", "n_train": len(tr),
            "n_eval_bona": int((ev.label == "bonafide").sum()), "n_eval_spoof": int((ev.label == "spoof").sum()),
            "minDCF": round(min_dcf(score, ev.label == "spoof"), 4)}, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    df = pd.read_parquet(PROCESSED / "manifest.parquet")
    missing = ~df["path"].map(lambda p: (ROOT / p).exists())
    if missing.any():
        print(f"WARNING: skipping {missing.sum()} manifest rows whose file is missing:",
              df.loc[missing, "source"].value_counts().to_dict())
        df = df[~missing]
    df = df.merge(load_stats(df["path"], args.workers), on="path")

    subsets = {"all": df, "lj_voice_only": df[df.speaker == "ljspeech:LJ"]}
    rows = []
    for name, sub in subsets.items():
        r = run(sub, name, STAT_COLUMNS)
        if r:
            rows.append(r[0])
            print(f"\n--- depth-3 tree, {name} ---\n" + export_text(r[1], feature_names=STAT_COLUMNS))
        for f in STAT_COLUMNS:
            r = run(sub, name, [f])
            if r:
                rows.append(r[0])
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\nper-class medians (all data):\n", df.groupby("label")[STAT_COLUMNS].median().T.to_string())


if __name__ == "__main__":
    main()
