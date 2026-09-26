"""R4 heads on frozen SSL layer embeddings (from scripts/extract_ssl.py).

    PYTHONPATH=src python scripts/train_ssl_heads.py sweep --model wavlm_base_plus      # per-layer LR -> figure + csv
    PYTHONPATH=src python scripts/train_ssl_heads.py heads --model wavlm_base_plus      # LR/SVM/MLP/layer-weighted
    PYTHONPATH=src python scripts/train_ssl_heads.py heads --model wavlm_base_plus --report --hgt   # final

Fit on the stratified `train` subset only. Every choice (layer, C, epochs, head) is made on val_testlike combined
minDCF; test_internal_testlike is only reported. HGT rows are scored by the chosen head, never fitted or normalized on.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from hearsay.audio import DATA
from hearsay.evaluate import SCORES, manifest, per_source, report
from hearsay.metrics import min_dcf

FIG = Path(__file__).parents[1] / "docs" / "figures"


def load(model):
    """-> index rows that have features (with label), X (n, L, 2D) float16."""
    d = DATA / "features" / model
    idx = pd.read_parquet(d / "index.parquet")
    have = [s for s in sorted(idx.shard.unique()) if (d / f"shard_{s:04d}.npy").exists()]
    idx = idx[idx.shard.isin(have)].reset_index(drop=True)
    X = np.concatenate([np.load(d / f"shard_{s:04d}.npy") for s in have])
    lab = manifest().set_index("path").label
    idx["y"] = (idx.path.map(lab) == "spoof").astype(float)
    idx.loc[idx.set == "hgt", "y"] = np.nan
    return idx, X


def dcf(s, y):
    return min_dcf(s, y.astype(int), "combined")


def feats(X, layers, part):
    D = X.shape[2] // 2
    sl = {"mean": slice(0, D), "std": slice(D, 2 * D), "both": slice(0, 2 * D)}[part]
    return X[:, layers, sl].reshape(len(X), -1).astype(np.float32)


def fit_lr(Xtr, ytr, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
    return lambda Z: clf.decision_function(sc.transform(Z))


def fit_svm(Xtr, ytr, C):
    sc = StandardScaler().fit(Xtr)
    clf = LinearSVC(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(Xtr), ytr)
    return lambda Z: clf.decision_function(sc.transform(Z))


class LayerMix(torch.nn.Module):
    """Softmax-weighted sum over layers (per-layer standardized) -> MLP (hidden=0: linear)."""

    def __init__(self, L, F, hidden):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(L))
        self.head = (torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(F, hidden), torch.nn.GELU(),
                                         torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1))
                     if hidden else torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(F, 1)))

    def forward(self, x):
        return self.head((torch.softmax(self.a, 0)[None, :, None] * x).sum(1)).squeeze(-1)


def fit_mix(X, tr, va, y, layers, hidden, epochs=30, seed=0):
    """Mini-batch Adam; keeps the epoch with the best val_testlike minDCF (early stopping on val only)."""
    torch.manual_seed(seed)
    Xs = X[:, layers].astype(np.float32)
    mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-5
    Xs = torch.from_numpy((Xs - mu) / sd)
    net = LayerMix(len(layers), Xs.shape[2], hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    yt = torch.from_numpy(y[tr]).float()
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((yt == 0).sum() / (yt == 1).sum()))
    Xtr, best = Xs[tr], (9.0, None, -1)
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(tr)).split(256):
            opt.zero_grad()
            lossf(net(Xtr[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            d = dcf(net(Xs[va]).numpy(), y[va])
        if d < best[0]:
            best = (d, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    net.eval()

    def score(idx_rows):
        with torch.no_grad():
            return net(Xs[idx_rows]).numpy()
    w = torch.softmax(net.a, 0).detach().numpy()
    return score, best[0], best[2], w


def sweep(a, idx, X):
    tr, va = (idx.set == "train").values, (idx.set == "val_testlike").values
    y = idx.y.values
    rows = []
    for l in range(X.shape[1]):
        for part in ["mean", "both"]:
            f = fit_lr(feats(X[tr], [l], part), y[tr], a.C)
            rows.append({"layer": l, "part": part, "val_testlike_combined": dcf(f(feats(X[va], [l], part)), y[va])})
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    r.to_csv(FIG / f"r4_layers_{a.model}.csv", index=False)
    plot_layers()


def plot_layers():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for csv in sorted(FIG.glob("r4_layers_*.csv")):
        r = pd.read_csv(csv)
        name = csv.stem.replace("r4_layers_", "")
        for part, ls in [("mean", "-"), ("both", "--")]:
            q = r[r.part == part]
            x = q.layer / q.layer.max()  # relative depth, so models with 12 and 24 blocks share an axis
            ax.plot(x, q.val_testlike_combined, ls, marker="o", ms=3, label=f"{name} ({part})")
    ax.set_xlabel("relative depth (0 = CNN output, 1 = last transformer block)")
    ax.set_ylabel("val_testlike minDCF (combined)")
    ax.set_title("R4: one logistic regression per SSL layer")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "r4_layers.png", dpi=150)


def heads(a, idx, X):
    tr, va = np.where(idx.set == "train")[0], np.where(idx.set == "val_testlike")[0]
    y = idx.y.values
    lay = pd.read_csv(FIG / f"r4_layers_{a.model}.csv")
    top = lay[lay.part == "both"].nsmallest(3, "val_testlike_combined").layer.tolist()
    best_l = top[0]
    print("top layers (mean+std):", top, flush=True)
    cands = {}  # name -> (val dcf, scorer over row indices, info)

    def add(name, f, info):
        d = dcf(f(va), y[va])
        cands[name] = (d, f, info)
        print(f"{name}: val_testlike combined {d:.4f} {info}", flush=True)

    for C in [0.001, 0.01, 0.1]:
        f = fit_lr(feats(X[tr], [best_l], "both"), y[tr], C)
        add(f"lr_L{best_l}_C{C}", lambda r, f=f: f(feats(X[r], [best_l], "both")), {"layer": best_l, "C": C})
    for C in [0.001, 0.01]:
        f = fit_lr(feats(X[tr], top, "both"), y[tr], C)
        add(f"lr_top3_C{C}", lambda r, f=f: f(feats(X[r], top, "both")), {"layers": top, "C": C})
    for C in [1e-4, 1e-3]:
        f = fit_svm(feats(X[tr], [best_l], "both"), y[tr], C)
        add(f"svm_L{best_l}_C{C}", lambda r, f=f: f(feats(X[r], [best_l], "both")), {"layer": best_l, "C": C})
    alll = list(range(X.shape[1]))
    for hidden in [0, 256]:
        f, d, ep, w = fit_mix(X, tr, va, y, alll, hidden)
        add(f"mix_h{hidden}", f, {"hidden": hidden, "epoch": ep, "layer_w": np.round(w, 3).tolist()})
    f, d, ep, w = fit_mix(X, tr, va, y, [best_l], 256)
    add(f"mlp_L{best_l}", f, {"hidden": 256, "epoch": ep})

    # one reported model per head family: the best config of that family on val_testlike
    fam = {}
    for n, (d, f, info) in cands.items():
        k = n.split("_")[0]
        if k not in fam or d < fam[k][1]:
            fam[k] = (n, d, f, info)
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"r4_heads_{a.model}.json").write_text(json.dumps(
        {n: {"val_testlike_combined": d, **i} for n, (d, _, i) in cands.items()}, indent=1))
    if not a.report:
        return
    ev = np.where(idx.set.isin(["val_testlike", "test_internal_testlike", "val_rest"]))[0]
    best = min(fam.values(), key=lambda t: t[1])
    for k, (n, d, f, info) in fam.items():
        name = f"R4_{a.model}_{k}"
        report(name, pd.DataFrame({"path": idx.path.values[ev], "score": f(ev)}), notes=f"{n}; 4 s center crop")
        ps = per_source(pd.DataFrame({"path": idx.path.values[ev], "score": f(ev)}))
        ps.to_csv(FIG / f"r4_per_source_{name}.csv", index=False)
    if a.hgt:
        n, d, f, info = best
        h = np.where(idx.set == "hgt")[0]
        name = f"R4_{a.model}_{n.split('_')[0]}"
        out = pd.DataFrame({"filename": [Path(p).name for p in idx.path.values[h]], "score": f(h)})
        assert len(out) == 1671, len(out)
        out.to_parquet(SCORES / f"{name}__hgt.parquet", index=False)
        print("HGT scored with", name, n, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "heads", "plot"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--hgt", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.cmd == "plot":
        return plot_layers()
    idx, X = load(a.model)
    print(idx.groupby("set").y.agg(["size", "sum"]).to_string(), flush=True)
    {"sweep": sweep, "heads": heads}[a.cmd](a, idx, X)


if __name__ == "__main__":
    main()
