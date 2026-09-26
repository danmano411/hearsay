"""R0/R1 feature extraction -> data/features/classic.parquet (keyed by path; resumable via part files).

Per clip (canonical 16 kHz WAV):
  raw_*   hearsay.data.stats.clip_stats on the RAW full clip (R0: how much do trivial cues leak?)
  then    train only: augment() (class-symmetric)  ->  prep()  ->  crop (train: random 3-4 s, else center <= 4 s)
  prep_*  the same trivial stats on that model input (R0 after prep: are the cues neutralized?)
  R1      hearsay.features.spectral (LFCC/MFCC/spectral, 0-7 kHz) + hearsay.features.bio (48 bio features)

Rows: split=='train' capped (all reals + --n-fake fakes spread evenly over source x generator), plus every row of
val / val_testlike / test_internal_testlike. --hgt instead featurizes data/raw/hgt_test (inference only; nothing is
ever fit on it) -> data/features/classic_hgt.parquet.

  python scripts/extract_classic.py --workers 4
  python scripts/extract_classic.py --hgt --workers 4
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # 1 BLAS thread per worker process; parallelism = --workers

import argparse
import hashlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from hearsay.audio import DATA, PROCESSED, ROOT
from hearsay.data.stats import clip_stats
from hearsay.features.bio import FEATURES as BIO, extract_bio
from hearsay.features.spectral import FEATURES as SPEC, extract_spectral
from hearsay.preprocess import SR, augment, prep

sys.path.insert(0, str(Path(__file__).parent))
from extract_bio import even  # noqa: E402  (water-filling stratified sampler from phase 4)

FEAT = DATA / "features"
EVAL_CROP = 4.0


def select(m, n_fake, seed=0):
    tr = m[m.split == "train"]
    real = tr[tr.label == "bonafide"]
    fake = tr[tr.label == "spoof"].assign(sg=lambda d: d.source + "/" + d.generator)
    fake = even(fake, "sg", min(n_fake, len(fake)), seed).drop(columns="sg")
    ev = m[(m.split == "val") | m.val_testlike | m.test_internal_testlike]
    return pd.concat([real, fake, ev], ignore_index=True)


def features(y, train, rng):
    if train:
        y = augment(y, rng)
    y = prep(y)
    n = int((rng.uniform(3.0, 4.0) if train else EVAL_CROP) * SR)
    if len(y) > n:
        s = int(rng.integers(0, len(y) - n + 1)) if train else (len(y) - n) // 2
        y = y[s:s + n]
    return {**{f"prep_{k}": v for k, v in clip_stats(y, SR).items()}, **extract_spectral(y), **extract_bio(y)}


def work(job):
    key, path, train = job
    t0 = time.time()
    rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:12], 16))  # reproducible per clip
    try:
        y, sr = sf.read(str(path), dtype="float32")
        assert sr == SR and y.ndim == 1, (sr, y.shape)
        f = {**{f"raw_{k}": v for k, v in clip_stats(y, sr).items()}, **features(y, train, rng)}
        err = ""
    except Exception as e:  # NaN row with the error; retried on the next run
        f, err = {}, repr(e)[:200]
    return {"key": key, **f, "err": err, "sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-fake", type=int, default=30000, help="train fakes to keep (all train reals are kept)")
    ap.add_argument("--hgt", action="store_true", help="featurize data/raw/hgt_test instead (inference only)")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N clips")
    args = ap.parse_args()

    if args.hgt:
        files = sorted((DATA / "raw" / "hgt_test").glob("*.wav"))
        jobs = [(p.name, p, False) for p in files]
        name = "classic_hgt"
    else:
        m = pd.read_parquet(PROCESSED / "manifest.parquet")
        sel = select(m, args.n_fake)
        jobs = [(p, ROOT / p, s == "train") for p, s in zip(sel.path, sel.split)]
        name = "classic"
    if args.limit:
        jobs = jobs[:args.limit]
    parts = FEAT / f"{name}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in parts.glob("*.parquet"):
        d = pd.read_parquet(p, columns=["key", "err"])
        done |= set(d.key[d.err == ""])
    todo = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} clips, {len(done)} done, {len(todo)} to go -> {parts}", flush=True)

    rows, t0, k = [], time.time(), len(list(parts.glob("*.parquet")))
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, todo, chunksize=4), 1):
            rows.append(r)
            if i % args.chunk == 0 or i == len(todo):
                pd.DataFrame(rows).to_parquet(parts / f"part_{k:05d}.parquet", index=False)
                rows, k = [], k + 1
                print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.3f} s/clip wall", flush=True)

    # consolidate: last successful row per key, for the requested jobs only
    df = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))], ignore_index=True)
    df = df.sort_values("err").drop_duplicates("key")  # "" sorts first -> keep a success when there is one
    df = df[df.key.isin({j[0] for j in jobs})]
    df = df.rename(columns={"key": "filename" if args.hgt else "path"})
    cols = [c for c in df.columns if c.startswith(("raw_", "prep_"))]
    assert set(SPEC + BIO) <= set(df.columns) or df.err.ne("").all()
    df.to_parquet(FEAT / f"{name}.parquet", index=False)
    print(f"wrote {len(df)} rows ({(df.err != '').sum()} errors, {len(cols)} stat cols) -> {FEAT / name}.parquet")


if __name__ == "__main__":
    main()
