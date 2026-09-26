"""Anti-spoofing corpora from the HuggingFace org SpeechAntiSpoofingBenchmarks (uniform 16 kHz FLAC parquet repacks).

usage: python scripts/ingest_hf_sasb.py <name> [<name> ...]   names: see CONFIGS

Downloads the parquet shards to data/external/<name>/, picks rows (seeded random caps per label),
writes canonical clips + manifest, then deletes the shards (logged) to stay inside the disk budget.
"""
import json
import os
import random
import sys

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from ingest_common import EXTERNAL, out_path, run

ORG = "SpeechAntiSpoofingBenchmarks"
CONFIGS = {
    # name: repo, license, caps {label: max rows}, shards (indices; None = all), keep(notes) filter
    "in_the_wild": dict(repo="InTheWild", license="CC-BY-SA-4.0 (orig. Mueller et al.); Apache-2.0 repack",
                        caps={"bonafide": 12000, "spoof": 8000}),
    "asvspoof2019_la": dict(repo="ASVspoof2019_LA", license="ODC-By-1.0", caps={"spoof": 8000}),
    "librisevoc": dict(repo="LibriSeVoc", license="CC-BY-SA-4.0", caps={"spoof": 10000}),
    "asvspoof5": dict(repo="ASVspoof5", license="ODC-By-1.0", shards=[0, 66, 133, 199]),
    "sonar": dict(repo="SONAR", license="CC-BY-NC-4.0 (research only)",
                  keep=lambda n: n.get("language", "en") == "en"),
    "dfadd": dict(repo="DFADD", license="MIT"),
}
LABELS = {0: "bonafide", 1: "spoof"}


def asv19_attacks(local_dir):
    """utterance_id -> attack id (A07..A19) from the official eval protocol shipped in the repo."""
    p = hf_hub_download(f"{ORG}/ASVspoof2019_LA", "protocols/ASVspoof2019.LA.cm.eval.trl.txt",
                        repo_type="dataset", local_dir=local_dir)
    return {line.split()[1]: line.split()[3] for line in open(p) if line.strip()}


def meta(name, notes, label, attacks):
    if label == "bonafide":
        gen = "bonafide"
    else:
        g = (notes.get("attack_id") or notes.get("vocoder") or notes.get("system") or notes.get("generator")
             or attacks.get(notes.get("utterance_id")) or "unknown")
        gen = f"{name}_{g}"
    codec = notes.get("codec")
    return dict(generator=gen, speaker=str(notes.get("speaker_id") or notes.get("speaker") or ""),
                text_id=notes.get("base_id") or notes.get("utterance_id"),
                codec_orig="flac" + (f";asv5_{codec}" if codec and codec != "-" else ""))


def ingest(name):
    cfg = CONFIGS[name]
    repo, local = f"{ORG}/{cfg['repo']}", EXTERNAL / name
    shards = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("data/test-"))
    if cfg.get("shards") is not None:
        shards = [shards[i] for i in cfg["shards"]]
    files = [hf_hub_download(repo, f, repo_type="dataset", local_dir=local) for f in shards]
    attacks = asv19_attacks(local) if name == "asvspoof2019_la" else {}

    # pass 1: metadata only -> seeded selection
    idx = pd.concat([pq.read_table(f, columns=["path", "label", "notes"]).to_pandas() for f in files])
    idx["label"] = idx["label"].map(LABELS)
    idx = idx[[cfg.get("keep", lambda n: True)(json.loads(n)) for n in idx["notes"]]]
    rng, chosen = random.Random(0), set()
    for lab, grp in idx.groupby("label"):
        paths = sorted(grp["path"])
        cap = cfg.get("caps", {}).get(lab, len(paths))
        chosen |= set(rng.sample(paths, min(cap, len(paths))))
    print(f"[{name}] {len(idx)} rows available, {len(chosen)} selected", flush=True)

    def jobs():
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    if r["path"] not in chosen:
                        continue
                    notes, label = json.loads(r["notes"]), LABELS[r["label"]]
                    m = meta(name, notes, label, attacks)
                    stem = r["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
                    yield dict(src=r["audio"]["bytes"], out=str(out_path(name, m["generator"], stem)),
                               label=label, source=name, license=cfg["license"], **m)

    df = run(jobs(), name)
    for f in files:  # budget: raw shards are re-downloadable
        print(f"[{name}] deleting raw shard {f}", flush=True)
        os.remove(f)
    return df


if __name__ == "__main__":
    for n in sys.argv[1:]:
        ingest(n)
