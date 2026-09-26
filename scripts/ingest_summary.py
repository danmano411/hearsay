"""Summarise Phase-2 manifests (markdown tables for docs/external_data.md) and sanity-check them.

Checks: exact MANIFEST_COLUMNS, labels in {bonafide, spoof}, durations >= 3 s, license filled,
every path exists, and no clip from data/raw/lj_real re-ingested.
"""
import sys

import pandas as pd

from ingest_common import DATA, MANIFEST_COLUMNS, MANIFESTS, MIN_DURATION, PROCESSED, ROOT

SOURCES = ["ljspeech", "librispeech", "wavefake", "in_the_wild", "asvspoof2019_la", "librisevoc", "asvspoof5",
           "sonar", "dfadd", "mlaad_tiny", "cvoicefake_en"]


def main(check_files=True):
    lines = ["| source | bonafide | spoof | hours | generators | speakers | GB on disk |", "|---|---:|---:|---:|---:|---:|---:|"]
    tot = {"bonafide": 0, "spoof": 0}
    given_lj = {p.stem for p in (DATA / "raw" / "lj_real").glob("*.wav")}
    for s in SOURCES:
        f = MANIFESTS / f"{s}.parquet"
        if not f.exists():
            lines.append(f"| {s} | - | - | - | - | - | not ingested |")
            continue
        df = pd.read_parquet(f)
        assert list(df.columns) == MANIFEST_COLUMNS, s
        assert set(df["label"]) <= {"bonafide", "spoof"}, s
        assert (df["duration"] >= MIN_DURATION).all() and df["license"].notna().all(), s
        assert df["path"].is_unique, s
        if check_files:
            missing = [p for p in df["path"] if not (ROOT / p).exists()]
            assert not missing, (s, missing[:3])
        if s == "ljspeech":
            assert not given_lj & set(df["text_id"]), "overlap with data/raw/lj_real"
        c = df["label"].value_counts()
        gb = sum(p.stat().st_size for p in (PROCESSED / s).rglob("*.wav")) / 1e9
        spk = df["speaker"].replace("", pd.NA).nunique()
        lines.append(f"| {s} | {c.get('bonafide', 0):,} | {c.get('spoof', 0):,} | {df['duration'].sum() / 3600:.1f} | "
                     f"{df.loc[df.label == 'spoof', 'generator'].nunique()} | {spk or '-'} | {gb:.2f} |")
        for k in tot:
            tot[k] += int(c.get(k, 0))
    lines.append(f"| **total external** | **{tot['bonafide']:,}** | **{tot['spoof']:,}** | | | | |")
    print("\n".join(lines))
    print(f"\nwith given data (242 LJ real, 70,000 DiffSSD fakes): bonafide {tot['bonafide'] + 242:,} / "
          f"spoof {tot['spoof'] + 70000:,}")


if __name__ == "__main__":
    main(check_files="--fast" not in sys.argv)
