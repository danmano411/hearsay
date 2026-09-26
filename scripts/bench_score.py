"""Score the frozen models on the public benchmark sets (plan 09, Part A). Inference only: nothing here is fit,
calibrated or selected on benchmark data.

    python scripts/bench_score.py r1   [--sets ...]   # refits R1_lgbm_all_full exactly (once, saved), then scores
    python scripts/bench_score.py r3   [--sets ...]   # organizers' AASIST, zero-shot
    python scripts/bench_score.py r4ft [--sets ...]   # needs data/models/r4ft_xlsr/R4ft_xlsr_light/{best.pth,config.json}
    python scripts/bench_score.py report [--sets ...] # -> data/bench/generalization.csv (+ printed table)
                                                      # default sets exclude keyguard_*: pass them with --sets

Scores -> data/bench/scores/<model>__<set>.parquet (path, score; higher = more fake). The fusion R5_r4ft_r1 is
computed in `report` from the frozen weights in data/scores/R5_r4ft_r1.json.

Operating point: minDCF picks its own best threshold per set, which a deployed detector cannot do. So `report` also
fixes each model's threshold on val_testlike (brief reading: a real flagged as fake costs 4x) and applies it unchanged.
The val_testlike scores come from the exact objects scored here: the refit R1 booster, the frozen R4ft scores, and
fused() on those two for R5 (not the 5-fold cross-fitted R5 val scores in data/scores/R5_r4ft_r1.parquet).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hearsay.audio import DATA, ROOT  # noqa: E402
from hearsay.evaluate import SCORES, manifest, metrics_for  # noqa: E402
from hearsay.metrics import eer  # noqa: E402

BENCH = DATA / "bench"
OUT = BENCH / "scores"
SETS = ["asvspoof2021_df", "cd_add", "decro_en", "deepvoice", "odss_en"]
NAMES = {"r1": "R1_lgbm_all_full", "r3": "R3_aasist_zeroshot_prep", "r4ft": "R4ft_xlsr_light"}
R1_MODEL = DATA / "models" / "r1_lgbm_all_full.txt"


def bench(name):
    return pd.read_parquet(BENCH / f"{name}.parquet")


def save(model, name, paths, s):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": paths, "score": np.asarray(s, float)}).to_parquet(OUT / f"{model}__{name}.parquet",
                                                                            index=False)
    print(f"{model} {name}: {len(paths)} scored", flush=True)


# ---------------------------------------------------------------------------------------------------------- R1
def r1_model():
    """The R1 booster was never saved; refit it with train_classic's exact recipe and check it reproduces."""
    import lightgbm as lgb
    if R1_MODEL.exists():
        return lgb.Booster(model_file=str(R1_MODEL))
    from train_classic import FEAT, fit_lgbm, vt_dcf
    from hearsay.features.bio import FEATURES as BIO
    from hearsay.features.spectral import FEATURES as SPEC
    feats = SPEC + BIO
    d = manifest().merge(pd.read_parquet(FEAT / "classic.parquet").query("err == ''"), on="path")
    tr = d[d.split == "train"].reset_index(drop=True)
    ev = d[(d.split == "val") | d.val_testlike | d.test_internal_testlike].reset_index(drop=True)
    grid = [dict(num_leaves=15, min_child_samples=100), dict(num_leaves=63, min_child_samples=50),
            dict(num_leaves=255, min_child_samples=20)]
    _, gbm, s, _ = fit_lgbm(tr, ev, feats, 4, grid)  # 4 = train_classic default; LightGBM is thread-count sensitive
    ref = pd.read_parquet(SCORES / "R1_lgbm_all_full.parquet").set_index("path").score.reindex(ev.path)
    diff = float(np.nanmax(np.abs(ref.values - s)))
    print(f"R1 refit vs saved scores: max |diff| = {diff:.2e} over {ref.notna().sum()} clips", flush=True)
    rho = pd.Series(s).corr(pd.Series(ref.values), method="spearman")
    print(f"R1 refit: spearman {rho:.4f} vs saved scores; val_testlike combined {vt_dcf(s, ev):.4f} (saved: 0.228)")
    if diff > 1e-3:  # classic.parquet changed after the _full run; the refit is a stand-in, reported as such
        print("WARNING: refit is not bit-identical to R1_lgbm_all_full; bench numbers use the refit")
    gbm.booster_.save_model(str(R1_MODEL))
    return gbm.booster_


def r1(sets, workers):
    import multiprocessing as mp
    from extract_classic import work
    from hearsay.features.bio import FEATURES as BIO
    from hearsay.features.spectral import FEATURES as SPEC
    booster = r1_model()
    for name in sets:
        b = bench(name)
        fp = BENCH / "features" / f"{name}.parquet"
        if not fp.exists():
            with mp.Pool(workers) as pool:
                rows = pool.map(work, [(p, ROOT / p, False) for p in b.path], chunksize=16)
            fp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).rename(columns={"key": "path"}).to_parquet(fp, index=False)
        f = pd.read_parquet(fp).set_index("path").reindex(b.path)
        bad = f.err != ""
        if bad.any():
            print(f"  {int(bad.sum())} clips failed featurization, dropped: {f[bad].err.iloc[0]}")
        f = f[~bad]
        save("R1_lgbm_all_full", name, f.index, booster.predict(f[SPEC + BIO]))


# ---------------------------------------------------------------------------------------------------------- R3
def r3(sets):
    import soundfile as sf
    import torch
    from hearsay.models import aasist_wrap
    from hearsay.preprocess import prep
    model = aasist_wrap.load(device="cpu")
    for name in sets:
        paths, out = bench(name).path.tolist(), []
        with torch.no_grad():
            for i in range(0, len(paths), 16):
                clips = [prep(sf.read(str(ROOT / p), dtype="float32")[0]) for p in paths[i:i + 16]]
                out += list(aasist_wrap.score_batch(model, clips, 1))  # R3 was scored with 1 window
        save(NAMES["r3"], name, paths, out)


# -------------------------------------------------------------------------------------------------------- R4ft
def r4ft(sets, threads, device="cpu"):
    import torch
    import finetune_ssl as fs
    a = fs.parse(["score", "--device", device, "--name", NAMES["r4ft"], "--workers", "4"])
    run = Path(a.out) / a.name
    cfg = json.loads((run / "config.json").read_text())
    for k in fs.ARCH:
        setattr(a, k, cfg["arch"][k])  # config.json froze the architecture with the checkpoint
    if fs.sha256(run / "best.pth") != cfg["sha256"]:
        raise SystemExit("best.pth does not match the frozen config.json sha256")
    torch.set_num_threads(threads)
    from hearsay import device as hdev
    dev = hdev.resolve(device)
    model = fs.make_model(a, grad_ckpt=False).to(dev)
    model.load_state_dict(torch.load(run / "best.pth", map_location="cpu"))
    model.eval()
    for name in sets:
        paths = bench(name).path.tolist()
        cache = OUT / f"cache_{name}_{cfg['inference']}_{cfg['sha256'][:12]}.csv"
        OUT.mkdir(parents=True, exist_ok=True)
        save(NAMES["r4ft"], name, paths, fs.score_paths(model, paths, cfg["max_windows"], dev, a, cache))


# ------------------------------------------------------------------------------------------------------ report
def fused(frames):
    """R5_r4ft_r1 on benchmark scores, with the weights frozen in data/scores/R5_r4ft_r1.json."""
    from fuse import apply_transform
    cfg = json.loads((SCORES / "R5_r4ft_r1.json").read_text())
    X = pd.concat([frames[m].rename(m) for m in cfg["models"]], axis=1, join="inner")
    z = sum(cfg["weights"][m] * (apply_transform(X[m], cfg["transforms"][m]) - cfg["mu"][m]) / cfg["sd"][m]
            for m in cfg["models"])
    return pd.Series(np.asarray(z) + cfg["intercept"], index=X.index)


def threshold(s, y, w_real=4.0, w_fake=1.0):
    """Threshold minimizing w_real * P(real flagged) + w_fake * P(fake passed); flag = score >= t."""
    s, fake = np.asarray(s, float), np.asarray(y) == "spoof"
    t = np.unique(s)
    real_flag = 1 - np.searchsorted(np.sort(s[~fake]), t, "left") / (~fake).sum()
    fake_pass = np.searchsorted(np.sort(s[fake]), t, "left") / fake.sum()
    return t[np.argmin(w_real * real_flag + w_fake * fake_pass)]


def op_point(s, y, t):
    s, fake = np.asarray(s, float), np.asarray(y) == "spoof"
    return 100 * np.mean(s[~fake] >= t), 100 * np.mean(s[fake] < t)


def val_scores():
    """val_testlike scores of the frozen objects used on the benchmarks (R3 only if its scores exist)."""
    from hearsay.features.bio import FEATURES as BIO
    from hearsay.features.spectral import FEATURES as SPEC
    m = manifest().set_index("path")
    vt = m[m.val_testlike]
    f = pd.read_parquet(DATA / "features" / "classic.parquet").query("err == ''").set_index("path")
    f = f[f.index.isin(vt.index)]
    v = {"R1_lgbm_all_full": pd.Series(r1_model().predict(f[SPEC + BIO]), index=f.index)}
    for mdl in ["R3_aasist_zeroshot_prep", "R4ft_xlsr_light"]:
        p = SCORES / f"{mdl}.parquet"
        if p.exists():
            s = pd.read_parquet(p).set_index("path").score
            v[mdl] = s[s.index.isin(vt.index)]
    v["R5_r4ft_r1"] = fused(v)
    return v, vt.label


def report(sets):
    v, lab = val_scores()
    thr = {mdl: threshold(s.values, lab.reindex(s.index).values) for mdl, s in v.items()}
    for mdl, s in v.items():
        print(f"{mdl}: val_testlike n={len(s)} combined {metrics_for(s.values, lab.reindex(s.index).values)['combined']:.4f}"
              f" threshold {thr[mdl]:.4f}")
    models = ["R1_lgbm_all_full", "R3_aasist_zeroshot_prep", "R4ft_xlsr_light", "R5_r4ft_r1"]
    rows = []
    for name in sets:
        b = bench(name).set_index("path")
        fr = {}
        for mdl in models[:3]:
            f = OUT / f"{mdl}__{name}.parquet"
            if f.exists() and mdl in thr:
                fr[mdl] = pd.read_parquet(f).set_index("path").score
        if {"R4ft_xlsr_light", "R1_lgbm_all_full"} <= fr.keys():
            fr["R5_r4ft_r1"] = fused(fr)
        for mdl, s in fr.items():
            y = b.label.reindex(s.index)
            r = metrics_for(s.values, y.values)
            rf, fm = op_point(s.values, y.values, thr[mdl])
            rows.append(dict(set=name, model=mdl, n_real=int((y == "bonafide").sum()), n_fake=int((y == "spoof").sum()),
                             official=r["official"], brief=r["brief"], combined=r["combined"],
                             eer=100 * eer(s.values, y.values), real_flagged_pct=rf, fake_missed_pct=fm))
    t = pd.DataFrame(rows)
    t.to_csv(BENCH / "generalization.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["r1", "r3", "r4ft", "report"])
    ap.add_argument("--sets", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cpu", help="r4ft: cpu | cuda | auto")
    a = ap.parse_args()
    sets = a.sets or [s for s in SETS if (BENCH / f"{s}.parquet").exists()]
    {"r1": lambda: r1(sets, a.workers), "r3": lambda: r3(sets), "r4ft": lambda: r4ft(sets, a.threads, a.device),
     "report": lambda: report(sets)}[a.what]()


if __name__ == "__main__":
    main()
