"""Public benchmark sets for the generalization check (plan 09, Part A). Never training data.

usage: python scripts/ingest_bench.py <name> [<name> ...]   names: see CONFIGS

Same SpeechAntiSpoofingBenchmarks parquet repacks as scripts/ingest_hf_sasb.py, but clips go to data/bench/<name>/
and the index to data/bench/<name>.parquet. data/processed/manifests is NOT touched, so these rows can never reach
make_splits.py or any training script. Per set: seeded sample of up to CAP reals and CAP fakes, clips >= 3 s.
Raw shards are deleted after conversion.
"""
import itertools
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from ingest_common import DATA, EXTERNAL, WORKERS, convert

ORG = "SpeechAntiSpoofingBenchmarks"
BENCH = DATA / "bench"
CAP = 1500
LABELS = {0: "bonafide", 1: "spoof"}


def _en(notes, path):
    lang = notes.get("language") or notes.get("lang")
    return lang in (None, "en", "english", "English", "EN")


CONFIGS = {
    # name: repo, n_shards (evenly spaced; None = all), keep(notes, path) filter
    "asvspoof2021_df": dict(repo="ASVspoof2021_DF", n_shards=4),
    # second, disjoint DF sample (plan 10, R6 training data); the first sample used shards 0, 26, 53, 79 of 80
    "asvspoof2021_df_b": dict(repo="ASVspoof2021_DF", n_shards=None, shards=[5, 12, 19, 33, 40, 47, 60, 70]),
    "cd_add": dict(repo="CD-ADD", n_shards=3),
    "decro_en": dict(repo="DECRO", n_shards=None,
                     keep=lambda n, p: p.rsplit("/", 1)[-1].startswith("en")),
    "deepvoice": dict(repo="DeepVoice", n_shards=None),
    "odss_en": dict(repo="ODSS", n_shards=None, keep=_en),
}


def generator(notes, path, label):
    if label == "bonafide":
        return "bonafide"
    for k in ("attack_id", "vocoder", "system", "generator", "system_id", "model", "tts", "method", "source_stem"):
        if notes.get(k):
            return str(notes[k])
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0].rsplit("_", 1)[-1][:40]  # e.g. CD-ADD ".../openvoice"


def ingest(name):
    cfg = CONFIGS[name]
    repo, local = f"{ORG}/{cfg['repo']}", EXTERNAL / f"bench_{name}"
    shards = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("data/test-"))
    if cfg.get("shards"):
        shards = [shards[i] for i in cfg["shards"]]
    elif cfg["n_shards"]:
        k = cfg["n_shards"]
        shards = [shards[round(i * (len(shards) - 1) / max(k - 1, 1))] for i in range(k)]
    files = [hf_hub_download(repo, f, repo_type="dataset", local_dir=local) for f in shards]

    idx = pd.concat([pq.read_table(f, columns=["path", "label", "notes"]).to_pandas() for f in files])
    idx["label"] = idx["label"].map(LABELS)
    print(f"[{name}] notes example: {idx.notes.iloc[0][:300]}", flush=True)
    keep = cfg.get("keep", lambda n, p: True)
    idx = idx[[keep(json.loads(n or "{}"), p) for n, p in zip(idx.notes, idx.path)]]
    rng, chosen = random.Random(0), set()
    for lab, grp in idx.groupby("label"):
        paths = sorted(grp.path)
        chosen |= set(rng.sample(paths, min(CAP, len(paths))))
    print(f"[{name}] {len(idx)} rows after filter {idx.label.value_counts().to_dict()}, {len(chosen)} selected",
          flush=True)

    def jobs():
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    if r["path"] not in chosen:
                        continue
                    notes, label = json.loads(r["notes"] or "{}"), LABELS[r["label"]]
                    gen = generator(notes, r["path"], label)
                    stem = r["path"].replace("/", "__").rsplit(".", 1)[0]
                    yield dict(src=r["audio"]["bytes"], out=str(BENCH / name / label / (stem + ".wav")),
                               label=label, source=name, generator=gen,
                               speaker=str(notes.get("speaker_id") or notes.get("speaker") or ""),
                               text_id=r["path"], license=f"see huggingface.co/datasets/{repo}",
                               notes=json.dumps(notes)[:500])

    rows = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for chunk in itertools.batched(jobs(), 256):
            for job, row in zip(chunk, pool.map(convert, chunk, chunksize=8)):
                if row:
                    rows.append(dict(row, generator=job["generator"], notes=job["notes"]))
            print(f"[{name}] {len(rows)} kept", flush=True)
    df = pd.DataFrame(rows)[["path", "label", "source", "generator", "speaker", "text_id", "duration", "notes"]]
    BENCH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BENCH / f"{name}.parquet", index=False)
    print(f"[{name}] index: {len(df)} rows {df.label.value_counts().to_dict()}; "
          f"generators {df.generator.value_counts().head(12).to_dict()}", flush=True)
    for f in files:
        os.remove(f)
    return df


if __name__ == "__main__":
    for n in sys.argv[1:]:
        ingest(n)
