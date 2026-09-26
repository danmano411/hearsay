"""Plan 10: register public benchmark sets as ordinary training sources -> data/processed/manifests/bench_<set>.parquet.

    python scripts/bench_to_manifest.py asvspoof2021_df asvspoof2021_df_b cd_add decro_en
    python scripts/make_splits.py

Never pass deepvoice (held out for R6 selection) or keyguard_* (derived from test_internal audio): both are refused.
"""
import sys

import pandas as pd

from hearsay.audio import DATA, MANIFEST_COLUMNS, MANIFESTS

REFUSED = ("deepvoice", "keyguard")


def main(names):
    for name in names:
        if name.startswith(REFUSED):
            raise SystemExit(f"{name}: held out / derived from test_internal, never a training source (plan 10)")
        b = pd.read_parquet(DATA / "bench" / f"{name}.parquet")
        b["source"] = f"bench_{name}"
        b["generator"] = b.generator.where(b.label == "bonafide", f"bench_{name}_" + b.generator.astype(str))
        b["sr_orig"], b["codec_orig"] = 16000, "flac"
        b["license"] = f"see huggingface.co/datasets/SpeechAntiSpoofingBenchmarks ({name})"
        out = MANIFESTS / f"bench_{name}.parquet"
        b[MANIFEST_COLUMNS].to_parquet(out, index=False)
        print(f"{out.name}: {len(b)} rows {b.label.value_counts().to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1:])
