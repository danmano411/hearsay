"""Audit raw audio at its NATIVE rate/format (plan 01 step 1).

Writes data/processed/audit/raw_audit.parquet (one row per file) and prints markdown tables for reports/data_audit.md.
hgt_test gets a format-only check (sr, channels, duration) — no level/spectral stats (inference-only rule).

  python scripts/audit_raw.py [--workers 8]
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf

from hearsay.audio import PROCESSED, rel
from hearsay.data.given import RAW, list_given
from hearsay.data.stats import clip_stats

OUT = PROCESSED / "audit" / "raw_audit.parquet"


def audit_one(item):
    path, meta = item
    row = {"raw_path": rel(path), **{k: meta[k] for k in ("source", "generator", "speaker", "accent")}}
    try:
        info = sf.info(str(path))
        y, sr = sf.read(str(path), dtype="float64", always_2d=True)
        row.update(ok=len(y) > 0, sr=sr, channels=y.shape[1], codec=f"{info.format}/{info.subtype}",
                   **clip_stats(y.mean(axis=1), sr))
    except Exception as e:  # undecodable -> recorded, dropped at clean time
        row.update(ok=False, error=repr(e)[:200])
    return row


def format_one(path):
    info = sf.info(str(path))
    return {"sr": info.samplerate, "channels": info.channels, "duration": info.duration, "codec": info.subtype}


def md(df):
    """Minimal markdown table (tabulate is not installed)."""
    df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with Pool(args.workers) as pool:
        rows = pool.map(audit_one, list_given(), chunksize=64)
        test = pd.DataFrame(pool.map(format_one, sorted((RAW / "hgt_test").glob("*.wav"))))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"## Raw audit ({len(df)} files, {int((~df.ok).sum())} undecodable/empty)\n")
    g = df[df.ok].groupby("generator")
    print("### Format per generator\n")
    fmt = g.agg(n=("raw_path", "size"), speakers=("speaker", "nunique"),
                sr=("sr", lambda s: ",".join(map(str, sorted(s.unique())))),
                channels=("channels", lambda s: ",".join(map(str, sorted(s.unique())))),
                codec=("codec", lambda s: ",".join(sorted(s.unique()))))
    print(md(fmt), "\n")
    print("### Signal stats per generator (median [p5, p95])\n")
    cols = ["duration", "peak", "rms_db", "lead_sil", "trail_sil", "rolloff95", "bw60", "clip_frac", "dc"]
    q = lambda s: f"{s.median():.3g} [{s.quantile(.05):.3g}, {s.quantile(.95):.3g}]"
    print(md(g[cols].agg(q)), "\n")
    print("### Clips < 3.0 s per generator\n")
    print(md(g.duration.apply(lambda s: int((s < 3.0).sum())).to_frame("n_lt_3s")), "\n")
    print("### hgt_test format check\n")
    print(f"n={len(test)}; sr={sorted(test.sr.unique().tolist())}; channels={sorted(test.channels.unique().tolist())}; "
          f"codec={sorted(test.codec.unique())}; duration min/median/max = "
          f"{test.duration.min():.2f}/{test.duration.median():.2f}/{test.duration.max():.2f} s")


if __name__ == "__main__":
    main()
