"""Per-feature real-vs-fake effect sizes for the bio features + figures for docs/05_speech_biology.md.

  python scripts/bio_figures.py [--feats data/features/bio_sample.parquet] [--no-audit]

Pooled real-vs-fake comparisons mix "is it fake?" with "which corpus/microphone/speaker is it?". So every number is
also computed WITHIN a matched pair (bonafide and spoof from the same corpus -> same recording pipeline): every
corpus that ships both classes, plus "lj" (LJ real vs DiffSSD generators trained on LJ's voice).

Writes docs/figures/bio_effect_sizes.{png,csv}, bio_pair_auc.png, bio_distributions.png; prints markdown tables,
cross-validated and leave-one-pair-out LightGBM AUCs, and a near-Nyquist bandwidth audit.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from hearsay.audio import DATA, ROOT
from hearsay.features.bio import FEATURES

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
LJ_VOICE = {"diffgan_tts", "grad_tts", "pro_diff", "wavegrad2"}  # DiffSSD generators trained on LJSpeech
LJ_REAL = {"lj_real", "ljspeech"}
GROUPS = [  # (name, key prefixes) in categorical slot order
    ("phonation", ("f0_", "voiced", "jitter", "shimmer", "hnr", "cpps", "h1h2", "tilt", "alpha")),
    ("articulation", ("f1_", "f2_", "f3_", "fvel", "fjump")),
    ("resonance", ("vtl",)),
    ("rhythm", ("am_",)),
    ("respiration", ("pause", "breath", "energy_decl")),
    ("turbulence", ("hf_",)),
    ("phase", ("gd_",)),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def group_of(f):
    return next(i for i, (_, pre) in enumerate(GROUPS) if f.startswith(pre))


def pair_of(r, paired):
    if r.source in LJ_REAL or r.generator in LJ_VOICE:
        return "lj"
    if r.source in paired:
        return r.source
    return None  # fakes without matched real speech (other DiffSSD voices, sim): pooled analysis only


def effect(x, y):
    """Cohen's d (spoof - bonafide, pooled SD) and AUC with spoof as the positive class; NaNs dropped."""
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if len(y) < 10 or y.min() == y.max():
        return np.nan, np.nan
    a, b = x[y == 1], x[y == 0]
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sd, roc_auc_score(y, x)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def bandwidth_audit(df, n=30):
    """Long-term spectrum near Nyquist per source/class (dB re 1 kHz): exposes resampling / band-limit shortcuts."""
    import librosa
    from hearsay.audio import load
    freqs = np.fft.rfftfreq(1024, 1 / 16000)
    marks = (4000, 6000, 7000, 7500, 7900)
    print("\n| source | label | orig sr | " + " | ".join(f"{f / 1000:g} kHz" for f in marks) + " |\n|---|---|---|"
          + "---:|" * 5)
    for (s, lab), sub in df.groupby(["source", "label"]):
        L = []
        srs = {int(v) for v in sub.sr_orig.dropna()} if "sr_orig" in sub else set()
        for p in sub.path.head(n):
            y, _ = load(ROOT / p)
            P = np.abs(librosa.stft(y, n_fft=1024, hop_length=160)) ** 2 + 1e-12
            e = P.sum(0)
            L.append(10 * np.log10(P[:, e > e.max() * 10 ** -3.5].mean(1)))
        L = np.mean(L, 0)
        at = lambda f: L[np.argmin(abs(freqs - f))]
        print(f"| {s} | {lab} | {'/'.join(map(str, sorted(srs)))} | "
              + " | ".join(f"{at(f) - at(1000):.0f}" for f in marks) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=str(DATA / "features" / "bio_sample.parquet"))
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()
    df = pd.read_parquet(args.feats)
    df = df[df.bio_error == ""].reset_index(drop=True)
    paired = {s for s, g in df.groupby("source") if g.label.nunique() == 2}  # corpora shipping both classes
    df["pair"] = [pair_of(r, paired) for r in df.itertuples()]
    y = (df.label == "spoof").to_numpy(int)
    pairs = sorted(p for p in df.pair.dropna().unique() if df[df.pair == p].label.nunique() == 2)
    print(f"{len(df)} clips: {(y == 0).sum()} bonafide ({df[y == 0].source.nunique()} sources), {(y == 1).sum()} spoof "
          f"({df[y == 1].generator.nunique()} generators)")
    print(df.groupby(["pair", "label"]).size().unstack().to_string())

    rows = []
    for f in FEATURES:
        x = df[f].to_numpy(float)
        d, auc = effect(x, y)
        r = dict(feature=f, group=GROUPS[group_of(f)][0], d=d, auc=auc, nan_frac=np.isnan(x).mean())
        for p in pairs:
            m = (df.pair == p).to_numpy()
            r[f"auc_{p}"] = effect(x[m], y[m])[1]
        r["auc_within"] = np.nanmean([r[f"auc_{p}"] for p in pairs])
        # sign consistency: in how many pairs does the fake shift go the same way as the pooled one
        r["same_sign"] = sum(np.sign(r[f"auc_{p}"] - 0.5) == np.sign(r["auc_within"] - 0.5) for p in pairs)
        rows.append(r)
    es = pd.DataFrame(rows)
    es = es.iloc[(es.auc_within - 0.5).abs().argsort()[::-1]].reset_index(drop=True)
    FIG.mkdir(parents=True, exist_ok=True)
    es.round(3).to_csv(FIG / "bio_effect_sizes.csv", index=False)
    print(f"\n| feature | group | Cohen's d (pooled) | AUC pooled | AUC within-pair (mean) | "
          f"same direction in k/{len(pairs)} pairs | NaN % |\n|---|---|---:|---:|---:|---:|---:|")
    for r in es.itertuples():
        print(f"| `{r.feature}` | {r.group} | {r.d:+.2f} | {r.auc:.3f} | {r.auc_within:.3f} | {r.same_sign} | "
              f"{100 * r.nan_frac:.0f} |")

    # --- fig 1: AUC per feature, pooled vs mean within-pair; bars grow from the 0.5 chance line
    fig, axes = plt.subplots(1, 2, figsize=(10, 11), sharey=True)
    order = es.feature.tolist()[::-1]
    col = [COLORS[group_of(f)] for f in order]
    for ax, key, title in [(axes[0], "auc", "pooled (all real vs all fake)"),
                           (axes[1], "auc_within", f"within corpus (mean of {len(pairs)} matched pairs)")]:
        v = es.set_index("feature").loc[order, key].to_numpy() - 0.5
        ax.barh(order, v, left=0.5, color=col, height=0.72, edgecolor="white", linewidth=1)
        ax.axvline(0.5, color=MUTED, lw=1)
        ax.set_xlim(0.1, 0.9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("AUC (fake = positive); 0.5 = no separation; right = higher in fakes", fontsize=8, color=MUTED)
        style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, [g for g, _ in GROUPS], loc="lower center", ncol=7, frameon=False, fontsize=8)
    fig.suptitle("Per-feature real-vs-fake separation (preliminary, 3k-clip sample)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(FIG / "bio_effect_sizes.png", dpi=100)
    plt.close(fig)

    # --- fig 2: within-pair AUC heatmap, 20 most separating features (diverging around 0.5)
    top = es.feature.head(20).tolist()
    M = es.set_index("feature").loc[top, [f"auc_{p}" for p in pairs]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pairs)), pairs, rotation=30, ha="right")
    ax.set_yticks(range(len(top)), top)
    for i in range(len(top)):
        for j in range(len(pairs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j] - 0.5) > 0.3 else INK)
    style(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("AUC (red: higher in fakes, blue: lower in fakes)", fontsize=8)
    ax.set_title("Within-corpus AUC, top-20 features", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "bio_pair_auc.png", dpi=100)
    plt.close(fig)

    # --- fig 3: distributions of the 6 best within-pair features, real vs fake, one line per matched pair
    best = top[:6]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, f in zip(axes.flat, best):
        x = df[f].to_numpy(float)
        lo, hi = np.nanpercentile(x, [1, 99])
        for lab, c, name in [("bonafide", COLORS[0], "real"), ("spoof", COLORS[1], "fake")]:
            m = ((df.label == lab) & df.pair.notna()).to_numpy() & np.isfinite(x)
            ax.hist(x[m], bins=np.linspace(lo, hi, 40), density=True, histtype="step", lw=2, color=c, label=name)
        ax.set_title(f, fontsize=9, color=INK)
        ax.set_yticks([])
        style(ax)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Six most separating features (matched-pair clips): normalized histograms", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "bio_distributions.png", dpi=100)
    plt.close(fig)

    # --- multivariate: LightGBM on bio features only. 5-fold pooled, then leave-one-pair-out (true transfer)
    import lightgbm as lgb
    X = df[FEATURES].to_numpy(float)
    mk = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=4)
    p = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"\n5-fold LightGBM AUC on bio features (pooled): {roc_auc_score(y, p):.3f}")
    print("\n| held-out corpus | n real | n fake | AUC (trained on all other clips) |\n|---|---:|---:|---:|")
    for pr in pairs:
        te = (df.pair == pr).to_numpy()
        clf = mk().fit(X[~te], y[~te])
        print(f"| {pr} | {(y[te] == 0).sum()} | {(y[te] == 1).sum()} | "
              f"{roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]):.3f} |")
    if not args.no_audit:
        bandwidth_audit(df)


if __name__ == "__main__":
    main()
