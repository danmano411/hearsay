"""Union every data/processed/manifests/*.parquet -> data/processed/manifest.parquet with split columns.

Deterministic and re-runnable: run again whenever a phase adds a source manifest.
Adds: group, split (train|val|test_internal|excluded), logo_fold, val_testlike, test_internal_testlike.

  python scripts/make_splits.py
"""
import pandas as pd

from hearsay.audio import MANIFEST_COLUMNS, MANIFESTS, PROCESSED
from hearsay.data.splits import assign_splits

OUT = PROCESSED / "manifest.parquet"


def main():
    parts = []
    for f in sorted(MANIFESTS.glob("*.parquet")):
        m = pd.read_parquet(f)
        missing = set(MANIFEST_COLUMNS) - set(m.columns)
        if missing:
            raise SystemExit(f"{f.name}: missing columns {missing}")
        parts.append(m[MANIFEST_COLUMNS])
        print(f"{f.name}: {len(m)} rows")
    df = pd.concat(parts, ignore_index=True)
    bad = ~df["label"].isin(["bonafide", "spoof"])
    if bad.any():
        raise SystemExit(f"bad labels: {df.loc[bad, 'label'].unique()}")
    dups = df["path"].duplicated()
    if dups.any():
        print(f"WARNING: dropping {dups.sum()} rows with a path listed twice")
        df = df[~dups]
    df = assign_splits(df.sort_values("path", ignore_index=True))
    df.to_parquet(OUT, index=False)

    print(f"\n{OUT}: {len(df)} rows")
    print(pd.crosstab([df["split"]], df["label"], margins=True).to_string())
    for s in ("val", "test_internal"):
        t = df[df[f"{s}_testlike"]]
        print(f"{s}_testlike: {len(t)} rows, {t['label'].value_counts().to_dict()}, "
              f"spoof generators {t.loc[t.label == 'spoof', 'generator'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
