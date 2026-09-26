"""Extract frozen SSL layer embeddings (mean+std pool of every hidden layer) for R4.

    PYTHONPATH=src python scripts/extract_ssl.py --model wavlm_base_plus --threads 5 --train-n 16000
    PYTHONPATH=src python scripts/extract_ssl.py --model xlsr_300m --threads 5 --train-n 6000 --eval testlike         --train-from wavlm_base_plus --max-blocks 12
    GPU: --device auto (default) picks cuda when available; raise --batch (e.g. 64) and drop --max-blocks.

Rows in priority order (val_testlike + train first so heads can be developed while the rest extracts):
val_testlike, a stratified train subset, test_internal_testlike, HGT test (inference only), rest of `val` (--eval full).
Output: data/features/<model>/index.parquet (path, set, shard, row) + shard_XXXX.npy float16 (n, layers, 2*dim).
Resumable: the index is frozen on first run; finished shards are skipped.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf

from hearsay import device
from hearsay.audio import DATA, ROOT
from hearsay.evaluate import manifest
from hearsay.features.ssl import SSLEmbedder, center
from hearsay.preprocess import prep

SHARD = 512


def stratified(df, n, seed=0):
    """~n rows, half bonafide half spoof; within a label, (source, generator) groups get sqrt-proportional quotas
    (capped at the group size, water-filled), so small sources/generators stay represented."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, d in df.groupby("label"):
        sizes = d.groupby(["source", "generator"]).size()
        lo, hi = 0.0, float(sizes.max())
        for _ in range(60):  # binary search the sqrt multiplier so sum(min(n_g, c*sqrt(n_g))) == n/2
            c = (lo + hi) / 2
            lo, hi = (c, hi) if np.minimum(sizes, c * np.sqrt(sizes)).sum() < n / 2 else (lo, c)
        quota = np.minimum(sizes, np.round(hi * np.sqrt(sizes))).astype(int)
        for key, g in d.groupby(["source", "generator"]):
            parts.append(g.iloc[rng.permutation(len(g))[: quota[key]]])
    return pd.concat(parts)


def build_index(model, train_n, eval_mode, base):
    m = manifest()
    vt, ti = m[m.val_testlike].assign(set="val_testlike"), m[m.test_internal_testlike].assign(set="test_internal_testlike")
    hgt = sorted((DATA / "raw" / "hgt_test").glob("*.wav"))
    hg = pd.DataFrame({"path": [p.relative_to(ROOT).as_posix() for p in hgt], "set": "hgt",
                       "duration": [sf.info(p).duration for p in hgt]})
    if base is not None:  # nested subset of another model's train rows -> comparable heads
        pool = m[m.path.isin(base.loc[base.set == "train", "path"])]
    else:
        pool = m[m.split == "train"]
    tr = stratified(pool, train_n).assign(set="train")
    parts = [vt, tr, ti, hg]
    if eval_mode == "full":
        parts.append(m[(m.split == "val") & ~m.val_testlike].assign(set="val_rest"))
    idx = []
    for p in parts:  # sort by (cropped) length within each priority block -> little padding per batch
        p = p[["path", "set", "duration"]].copy()
        p["dur_c"] = p.duration.clip(upper=4.0)
        idx.append(p.sort_values(["dur_c", "path"]))
    idx = pd.concat(idx, ignore_index=True)
    idx["shard"] = np.arange(len(idx)) // SHARD
    idx["row"] = np.arange(len(idx)) % SHARD
    return idx[["path", "set", "duration", "shard", "row"]]


def load_clip(rel):
    y, sr = sf.read(str(ROOT / rel), dtype="float32", always_2d=True)
    assert sr == 16000, rel
    return center(prep(y.mean(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--train-n", type=int, default=16000)
    ap.add_argument("--eval", choices=["full", "testlike"], default="full")
    ap.add_argument("--train-from", default=None, help="take the train subset from this model's index (nested)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=None, help="truncate the transformer (cost ~ blocks)")
    device.add_argument(ap)
    a = ap.parse_args()

    out = DATA / "features" / a.model
    out.mkdir(parents=True, exist_ok=True)
    ip = out / "index.parquet"
    if ip.exists():
        idx = pd.read_parquet(ip)
    else:
        base = pd.read_parquet(DATA / "features" / a.train_from / "index.parquet") if a.train_from else None
        idx = build_index(a.model, a.train_n, a.eval, base)
        idx.to_parquet(ip, index=False)
    print(idx.groupby("set").size().to_string(), flush=True)

    dev = device.resolve(a.device)
    print(f"device {dev}", flush=True)
    emb = SSLEmbedder(a.model, threads=a.threads, max_blocks=a.max_blocks, device=dev)
    pool = ThreadPoolExecutor(2)  # decode + prep overlaps with the forward pass
    todo = [s for s in sorted(idx.shard.unique()) if not (out / f"shard_{s:04d}.npy").exists()]
    t0, done = time.time(), 0
    for s in todo:
        paths = idx.loc[idx.shard == s, "path"].tolist()
        waves = list(pool.map(load_clip, paths))
        order = np.argsort([len(w) for w in waves], kind="stable")  # post-trim lengths -> batches of near-equal length
        feats = np.empty((len(waves),) + emb([waves[0][:16000]]).shape[1:], np.float32)
        for i in range(0, len(order), a.batch):
            feats[order[i:i + a.batch]] = emb([waves[j] for j in order[i:i + a.batch]])
        tmp = out / f"shard_{s:04d}.tmp.npy"
        np.save(tmp, feats.astype(np.float16))
        tmp.replace(out / f"shard_{s:04d}.npy")  # atomic: a killed run never leaves a half shard
        done += len(paths)
        el = time.time() - t0
        print(f"shard {s} done | {done} clips {el / done:.3f} s/clip | "
              f"eta {(len(todo) * SHARD - done) * el / done / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
