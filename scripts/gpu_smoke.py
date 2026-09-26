"""G1: --device cuda smoke test + throughput vs CPU for the two GPU workloads (SSL extraction, AASIST scoring).

    PYTHONPATH=src python scripts/gpu_smoke.py            # -> JSON on stdout; findings in reports/gpu_smoke.md

Read-only on shared data: no manifest, score, feature or leaderboard writes (the gpu node never calls report()).
- SSL: one shard of an existing extraction index (default wavlm_base_plus shard 0, 512 clips), extracted on cuda at
  several batch sizes; batch 8 is compared with the CPU-extracted shard on disk (parity check: GPU shards must
  match CPU ones, extraction stays fp32). CPU throughput is timed on the first --cpu-n clips.
- AASIST: the first --aasist-n val_testlike clips, pretrained weights, scored on cpu and cuda; scores compared.
Timings are model time only (clips are decoded + prep()'d once up front) plus the decode/prep cost per clip.
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from hearsay import device
from hearsay.audio import DATA, ROOT, load
from hearsay.evaluate import eval_masks, manifest
from hearsay.features.ssl import SSLEmbedder, center
from hearsay.models import aasist_wrap
from hearsay.preprocess import prep


def log(msg):  # progress on stderr as each measurement lands; the JSON summary stays on stdout
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def ssl_clip(rel):  # same as extract_ssl.load_clip
    y, sr = sf.read(str(ROOT / rel), dtype="float32", always_2d=True)
    assert sr == 16000, rel
    return center(prep(y.mean(1)))


def extract(emb, waves, batch):
    """extract_ssl's shard loop: sort by length, batch, scatter back."""
    order = np.argsort([len(w) for w in waves], kind="stable")
    feats = None
    for i in range(0, len(order), batch):
        f = emb([waves[j] for j in order[i:i + batch]])
        if feats is None:
            feats = np.empty((len(waves),) + f.shape[1:], np.float32)
        feats[order[i:i + batch]] = f
    return feats


def timed(fn, dev):
    sync(dev)
    t = time.perf_counter()
    out = fn()
    sync(dev)
    return out, time.perf_counter() - t


def ssl_bench(a, cuda, res):
    out = DATA / "features" / a.ssl_model
    idx = pd.read_parquet(out / "index.parquet")
    paths = idx.loc[idx.shard == a.shard].sort_values("row").path.tolist()
    t = time.perf_counter()
    waves = [ssl_clip(p) for p in paths]
    r = {"clips": len(waves), "decode_prep_s_per_clip": (time.perf_counter() - t) / len(waves),
         "mean_seconds": float(np.mean([len(w) for w in waves]) / 16000)}

    emb = SSLEmbedder(a.ssl_model, device=cuda)
    extract(emb, waves[:16], 8)  # warm-up (cuDNN autotune, allocator)
    ref = np.load(out / f"shard_{a.shard:04d}.npy").astype(np.float32) if (out / f"shard_{a.shard:04d}.npy").exists() else None
    for b in a.ssl_batches:
        torch.cuda.reset_peak_memory_stats(cuda)
        try:
            f, s = timed(lambda: extract(emb, waves, b), cuda)
        except torch.OutOfMemoryError:
            r[f"cuda_batch{b}"] = "OOM"
            torch.cuda.empty_cache()
            log(f"ssl cuda batch {b}: OOM")
            continue
        rb = {"clips_per_s": len(waves) / s, "peak_vram_gb": torch.cuda.max_memory_allocated(cuda) / 1e9}
        if ref is not None and b == 8:  # parity vs the CPU-extracted shard (stored float16)
            rel = np.abs(f - ref).max(axis=(1, 2)) / (np.abs(ref).max(axis=(1, 2)) + 1e-6)
            rb["vs_cpu_shard"] = {"median_rel_maxdiff": float(np.median(rel)), "max_rel_maxdiff": float(rel.max()),
                                  "cos_min": float(min(np.dot(x.ravel(), y.ravel()) / np.linalg.norm(x) / np.linalg.norm(y)
                                                       for x, y in zip(f, ref)))}
        r[f"cuda_batch{b}"] = rb
        log(f"ssl cuda batch {b}: {rb}")
    del emb
    torch.cuda.empty_cache()

    if a.cpu_n:
        cpu_emb = SSLEmbedder(a.ssl_model, device="cpu")
        sub = waves[:a.cpu_n]
        fc, s = timed(lambda: extract(cpu_emb, sub, 8), torch.device("cpu"))
        r["cpu_batch8"] = {"clips": len(sub), "clips_per_s": len(sub) / s, "threads": torch.get_num_threads()}
        fg = extract(SSLEmbedder(a.ssl_model, device=cuda), sub, 8)
        r["cpu_vs_cuda_same_run_max_rel_diff"] = float((np.abs(fc - fg).max(axis=(1, 2)) /
                                                        (np.abs(fc).max(axis=(1, 2)) + 1e-6)).max())
        log(f"ssl cpu: {r['cpu_batch8']}, cpu vs cuda max rel diff {r['cpu_vs_cuda_same_run_max_rel_diff']:.2e}")
    res["ssl"] = r


def aasist_bench(a, cuda, res):
    m = manifest()
    paths = m.loc[eval_masks(m)["val_testlike"], "path"].sort_values().tolist()[:a.aasist_n]
    t = time.perf_counter()
    clips = [prep(load(ROOT / p)[0]) for p in paths]
    r = {"clips": len(clips), "decode_prep_s_per_clip": (time.perf_counter() - t) / len(clips)}

    def run(model, dev, batch):
        return np.concatenate([aasist_wrap.score_batch(model, clips[i:i + batch]) for i in range(0, len(clips), batch)])

    model = aasist_wrap.load(device=cuda).to(memory_format=torch.channels_last)  # as run_aasist.py
    run(model, cuda, 16)  # warm-up
    scores = {}
    for b in a.aasist_batches:
        torch.cuda.reset_peak_memory_stats(cuda)
        try:
            scores[b], s = timed(lambda: run(model, cuda, b), cuda)
        except torch.OutOfMemoryError:
            r[f"cuda_batch{b}"] = "OOM"
            torch.cuda.empty_cache()
            log(f"aasist cuda batch {b}: OOM")
            continue
        r[f"cuda_batch{b}"] = {"clips_per_s": len(clips) / s, "peak_vram_gb": torch.cuda.max_memory_allocated(cuda) / 1e9}
        log(f"aasist cuda batch {b}: {r[f'cuda_batch{b}']}")
    cpu_model = aasist_wrap.load(device="cpu").to(memory_format=torch.channels_last)
    sc, s = timed(lambda: run(cpu_model, torch.device("cpu"), 16), torch.device("cpu"))
    r["cpu_batch16"] = {"clips_per_s": len(clips) / s, "threads": torch.get_num_threads()}
    log(f"aasist cpu: {r['cpu_batch16']}")
    g = scores[min(scores)]
    r["cpu_vs_cuda_score_max_abs_diff"] = float(np.abs(sc - g).max())
    r["score_range"] = [float(sc.min()), float(sc.max())]
    res["aasist"] = r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl-model", default="wavlm_base_plus")
    ap.add_argument("--shard", type=int, default=0)
    # 64 x 4 s WavLM clips needs > 8 GB (CNN feature-encoder activations on raw audio dominate); 16-32 fits
    ap.add_argument("--ssl-batches", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--cpu-n", type=int, default=64)
    ap.add_argument("--aasist-n", type=int, default=200)
    ap.add_argument("--aasist-batches", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--skip", choices=["ssl", "aasist"], nargs="*", default=[])
    device.add_argument(ap)
    a = ap.parse_args()
    cuda = device.resolve("cuda" if a.device == "auto" else a.device)
    p = torch.cuda.get_device_properties(cuda)
    res = {"gpu": p.name, "vram_gb": p.total_memory / 1e9, "capability": f"{p.major}.{p.minor}",
           "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if "ssl" not in a.skip:
        ssl_bench(a, cuda, res)
    if "aasist" not in a.skip:
        aasist_bench(a, cuda, res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
