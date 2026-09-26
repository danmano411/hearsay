"""Phase-1 check: assert the combined manifest is sound.

- every path exists and is 16 kHz mono PCM16 WAV, >= 3.0 s
- labels in {bonafide, spoof}; no duplicate paths
- no leakage group and no (speaker, text_id) pair spans two splits ("excluded" counts as train's bucket)
- held-out generators never in train; test-like subsets are ~70/30
Prints class counts per split / generator.

  python scripts/check_manifest.py [--workers 8] [--skip-audio] [--sources diffssd lj_real]
"""
import argparse
from multiprocessing import Pool

import pandas as pd
import soundfile as sf

from hearsay.audio import MIN_DURATION, PROCESSED, ROOT, SR
from hearsay.data.splits import HELDOUT_GENERATORS


def probe(path):
    p = ROOT / path
    if not p.exists():
        return (path, "missing")
    i = sf.info(str(p))
    if (i.samplerate, i.channels, i.subtype) != (SR, 1, "PCM_16"):
        return (path, f"format {i.samplerate} Hz / {i.channels} ch / {i.subtype}")
    if i.frames / i.samplerate < MIN_DURATION:
        return (path, f"short {i.frames / i.samplerate:.2f} s")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-audio", action="store_true", help="skip opening every file (fast schema-only check)")
    ap.add_argument("--sources", nargs="*", help="only check these sources (default: all)")
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / "manifest.parquet")
    if args.sources:
        df = df[df["source"].isin(args.sources)]
    assert df["label"].isin(["bonafide", "spoof"]).all(), "bad label"
    assert not df["path"].duplicated().any(), "duplicate paths"
    assert (df["duration"] >= MIN_DURATION).all(), "manifest duration < 3 s"

    bucket = df["split"].replace({"excluded": "train"})
    spans = bucket.groupby(df["group"]).nunique()
    assert (spans == 1).all(), f"groups spanning splits: {spans[spans > 1].index[:5].tolist()}"
    st = df.dropna(subset=["speaker", "text_id"])
    spans = bucket[st.index].groupby([st["speaker"], st["text_id"]]).nunique()
    assert (spans == 1).all(), f"(speaker, text_id) spanning splits: {spans[spans > 1].index[:5].tolist()}"
    assert not (df["generator"].isin(HELDOUT_GENERATORS) & (df["split"] == "train")).any(), "held-out gen in train"
    for s in ("val", "test_internal"):
        t = df[df[f"{s}_testlike"]]
        assert (df.loc[t.index, "split"] == s).all()
        if len(t) and not args.sources:  # ratio is defined over all sources
            frac = (t["label"] == "spoof").mean()
            assert abs(frac - 0.3) < 0.05, f"{s}_testlike spoof fraction {frac:.2f}"

    if not args.skip_audio:
        with Pool(args.workers) as pool:
            bad = [b for b in pool.map(probe, df["path"], chunksize=256) if b]
        if bad:
            print(pd.Series([df.set_index("path").loc[p, "source"] for p, _ in bad]).value_counts().to_string())
        assert not bad, f"{len(bad)} bad files, e.g. {bad[:5]}"

    print(pd.crosstab(df["split"], df["label"], margins=True).to_string(), "\n")
    print(pd.crosstab(df["generator"], df["split"], margins=True).to_string(), "\n")
    for s in ("val", "test_internal"):
        print(f"{s}_testlike:", df.loc[df[f"{s}_testlike"], "label"].value_counts().to_dict())
    print(f"\nOK: {len(df)} clips, {df['group'].nunique()} groups, audio checked: {not args.skip_audio}")


if __name__ == "__main__":
    main()
