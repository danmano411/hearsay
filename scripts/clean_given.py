"""Clean the challenge-provided data (DiffSSD + lj_real) into canonical clips.

decode -> mono -> 16 kHz (soxr_hq) -> PCM16 WAV under data/processed/<source>/...; no trimming, no loudness norm.
Drops: undecodable/empty, < 3.0 s, exact PCM duplicates (first path in sorted order is kept).
Writes data/processed/manifests/{diffssd,lj_real}.parquet (MANIFEST_COLUMNS) and
data/processed/audit/clip_stats.parquet (trivial-cue stats on the canonical 16 kHz signal, for the shortcut audit).

  python scripts/clean_given.py [--workers 8]
"""
import argparse
from multiprocessing import Pool

import pandas as pd
import soundfile as sf

from hearsay.audio import MANIFEST_COLUMNS, MANIFESTS, MIN_DURATION, PROCESSED, ROOT, SR, load, pcm_hash, rel, save
from hearsay.data.given import list_given, processed_path
from hearsay.data.stats import STAT_COLUMNS, clip_stats

STATS = PROCESSED / "audit" / "clip_stats.parquet"


def _codec(path):
    i = sf.info(str(path))
    return f"{i.format}/{i.subtype}".lower()  # e.g. wav/pcm_16, mp3/mpeg_layer_iii


def clean_one(item):
    raw, meta = item
    out = processed_path(raw, meta)
    row = {**meta, "raw_path": rel(raw), "path": None}
    try:
        y, sr_orig = load(raw)
    except Exception as e:
        return {**row, "drop": f"undecodable: {e!r}"[:200]}
    row.update(sr_orig=sr_orig, codec_orig=_codec(raw), **clip_stats(y, SR))
    if len(y) == 0:
        return {**row, "drop": "empty"}
    if row["duration"] < MIN_DURATION:
        return {**row, "drop": "short"}
    save(out, y)
    return {**row, "path": rel(out), "hash": pcm_hash(y), "drop": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with Pool(args.workers) as pool:
        df = pd.DataFrame(pool.map(clean_one, list_given(), chunksize=64))

    kept = df["drop"].isna()
    dup = kept & df.duplicated("hash", keep="first")  # list_given() is sorted -> deterministic survivor
    for p in df.loc[dup, "path"]:
        (ROOT / p).unlink()
    df.loc[dup, "drop"] = "duplicate"

    print("Dropped per generator:\n", df.fillna({"drop": "kept"}).groupby("generator")["drop"].value_counts().unstack(fill_value=0), "\n")
    print("Duplicate groups:", df.loc[dup, ["raw_path"]].head(20).to_string() if dup.any() else "none")

    keep = df[df["drop"].isna()]
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    for source, part in keep.groupby("source"):
        part[MANIFEST_COLUMNS].to_parquet(MANIFESTS / f"{source}.parquet", index=False)
        print(f"{source}: {len(part)} clips -> {MANIFESTS / (source + '.parquet')}")
    STATS.parent.mkdir(parents=True, exist_ok=True)
    keep[["path"] + STAT_COLUMNS].to_parquet(STATS, index=False)
    df.drop(columns=["hash"]).to_parquet(PROCESSED / "audit" / "clean_log.parquet", index=False)


if __name__ == "__main__":
    main()
