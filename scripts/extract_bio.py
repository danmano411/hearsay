"""Extract biology features (hearsay.features.bio) for every clip in a manifest -> data/features/bio.parquet.

Manifest resolution: --manifest PATH > data/processed/manifest.parquet > data/processed/manifests/*.parquet
> (--raw only) a manifest built on the fly from data/raw/lj_real + data/raw/diffssd. data/raw/hgt_test is never read.

Resumable: paths already in the output parquet are skipped; output is rewritten every --chunk clips.

  python scripts/extract_bio.py --sample 3000 --out data/features/bio_sample.parquet   # stage-4 validation sample
  python scripts/extract_bio.py --workers 12            # full corpus
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd

from hearsay.audio import DATA, MANIFESTS, PROCESSED, ROOT, load
from hearsay.features.bio import FEATURES, extract_bio

OUT = DATA / "features" / "bio.parquet"
KEEP = ["path", "label", "source", "generator", "speaker", "sr_orig", "codec_orig"]
LJ_VOICE = {"diffgan_tts", "grad_tts", "pro_diff", "wavegrad2"}  # DiffSSD generators trained on LJSpeech


def raw_manifest():
    rows = [dict(path=p.relative_to(ROOT).as_posix(), label="bonafide", source="lj_real", generator="bonafide",
                 speaker="LJ") for p in sorted((DATA / "raw" / "lj_real").glob("*.wav"))]
    for p in sorted((DATA / "raw" / "diffssd").rglob("*")):
        if p.suffix.lower() in (".wav", ".mp3", ".flac"):
            gen = p.relative_to(DATA / "raw" / "diffssd").parts[0]
            spk = "LJ" if gen in LJ_VOICE else p.parent.name
            rows.append(dict(path=p.relative_to(ROOT).as_posix(), label="spoof", source="diffssd", generator=gen,
                             speaker=spk))
    return pd.DataFrame(rows)


def resolve_manifest(args):
    if args.manifest:
        return pd.read_parquet(args.manifest)
    if (PROCESSED / "manifest.parquet").exists():
        return pd.read_parquet(PROCESSED / "manifest.parquet")
    parts = sorted(MANIFESTS.glob("*.parquet"))
    if parts:
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if args.raw:
        return raw_manifest()
    raise SystemExit("no manifest found; pass --manifest or --raw")


def even(df, key, budget, seed):
    """Sample `budget` rows split as evenly as possible over groups of `key`; small groups donate leftovers."""
    sizes = df.groupby(key).size().sort_values()
    parts, left = [], budget
    for k, (g, s) in enumerate(sizes.items()):
        t = min(s, left // (len(sizes) - k))
        parts.append(df[df[key] == g].sample(t, random_state=seed))
        left -= t
    return pd.concat(parts)


def stratified(df, n, seed):
    """n/3 bonafide spread evenly over sources (speakers/channels), 2n/3 spoof spread evenly over generators."""
    real = even(df[df.label == "bonafide"], "source", n // 3, seed)
    fake = even(df[df.label == "spoof"], "generator", n - len(real), seed)
    return pd.concat([real, fake], ignore_index=True)


def work(row):
    t0 = time.time()
    try:
        y, _ = load(ROOT / row["path"])
        f = extract_bio(y)
        err = ""
    except Exception as e:  # unreadable/missing file: NaN row with the error; retried on the next run
        f, err = dict.fromkeys(FEATURES, float("nan")), repr(e)[:200]
    return {**{k: row.get(k) for k in KEEP}, **f, "bio_error": err, "bio_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--raw", action="store_true", help="fall back to data/raw (lj_real + diffssd) if no manifests")
    ap.add_argument("--sample", type=int, default=0, help="stratified sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    df = resolve_manifest(args)
    assert not df.path.str.contains("hgt_test").any(), "test audio is inference-only"
    if args.sample:
        df = stratified(df, args.sample, args.seed)
    out = ROOT / args.out  # relative paths are taken from the project root; absolute ones pass through
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    if len(done):
        done = done[done.bio_error == ""]  # retry failures (e.g. files another ingest had not written yet)
    todo = df[~df.path.isin(done.path)] if len(done) else df
    print(f"{len(df)} clips in manifest, {len(done)} already done, {len(todo)} to go -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    records = [r._asdict() for r in todo[[c for c in KEEP if c in todo]].itertuples(index=False)]
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, records, chunksize=8), 1):
            rows.append(r)
            if i % args.chunk == 0 or i == len(records):
                # full rewrite per chunk is O(n) each time; switch to part files if the corpus passes ~500k
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_parquet(out, index=False)
                rows = []
                print(f"{i}/{len(records)}  {(time.time() - t0) / i * args.workers:.3f} s/clip/worker", flush=True)
    if len(done):
        print(f"errors: {(done.bio_error != '').sum()}  median s/clip: {done.bio_sec.median():.3f}")


if __name__ == "__main__":
    main()
