"""R4ft VRAM probe (plans/08_ssl_aasist.md §4): run BEFORE the first training job.

    python scripts/vram_probe.py --device cuda                 # -> data/models/r4ft_xlsr/probe.json + probe.md
    python scripts/vram_probe.py --device cpu --dry-run --out <tmp>   # tiny random model, CPU only (tests)
    (also: python scripts/finetune_ssl.py probe --device cuda)

1. Memory: each config below runs in its own subprocess (an OOM cannot fragment the next one): --steps optimizer
   steps on random 4 s waveforms (memory does not depend on content), bf16 autocast, fp32 AdamW. Records
   max_memory_allocated / max_memory_reserved, step time (median of the last 8), OOM, and a WDDM-spill flag (a step
   > 3x the median of the steps before it, even without an OOM).
   (a) K12 ckpt mb8  (b) K12 no-ckpt mb8  (c) K12 ckpt mb16  (d) K24 lower-12-frozen ckpt mb8
   (e) K24 all-trainable ckpt mb4  (f) (e) + 8-bit AdamW, only if bitsandbytes imports and a 1-step sanity check
   matches fp32 AdamW.
2. Pick: the fastest config (clips/s) whose peak reserved <= 85 % of the capped budget (0.92 x VRAM, G1).
3. Throughput: a loader-only loop (--loader-clips clips through the real train DataLoader: load, augment, prep, crop)
   and the picked config for --real-steps steps fed by the real DataLoader, clips/s for GPU and loader separately.
--dry-run: tiny random-init wav2vec2 (24 blocks x 32 dims, same 199-frame geometry), synthetic wavs instead of the
manifest, 3 steps per config: exercises every code path on CPU without a download or a GPU.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

CONFIGS = {
    "a": {"desc": "K12 ckpt mb8", "max_blocks": 12, "freeze_blocks": 0, "ckpt": True, "mb": 8},
    "b": {"desc": "K12 no-ckpt mb8", "max_blocks": 12, "freeze_blocks": 0, "ckpt": False, "mb": 8},
    "c": {"desc": "K12 ckpt mb16", "max_blocks": 12, "freeze_blocks": 0, "ckpt": True, "mb": 16},
    "d": {"desc": "K24 lower-12-frozen ckpt mb8", "max_blocks": 24, "freeze_blocks": 12, "ckpt": True, "mb": 8},
    "e": {"desc": "K24 all-trainable ckpt mb4", "max_blocks": 24, "freeze_blocks": 0, "ckpt": True, "mb": 4},
    "f": {"desc": "K24 all-trainable ckpt mb4 + 8-bit AdamW", "max_blocks": 24, "freeze_blocks": 0, "ckpt": True,
          "mb": 4, "adam8": True},
}
BUDGET_FRAC = 0.85


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="directory for probe.json / probe.md (default data/models/r4ft_xlsr)")
    p.add_argument("--model", default="xlsr_300m")
    p.add_argument("--backend", default="light")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--configs", default="abcdef")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--eval-batch", type=int, default=32,
                   help="windows per eval forward to probe (= finetune_ssl.py --eval-batch)")
    p.add_argument("--dry-run", action="store_true", help="tiny random model + synthetic wavs (CPU test mode)")
    p.add_argument("--loader-clips", type=int, default=2000)
    p.add_argument("--real-steps", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--timeout", type=float, default=1800, help="seconds per config subprocess")
    p.add_argument("--one", default=None, help=argparse.SUPPRESS)  # internal: run one config in this process
    p.add_argument("--synthetic", default=None, help=argparse.SUPPRESS)  # internal: dry-run data dir
    p.add_argument("--real", action="store_true", help=argparse.SUPPRESS)  # internal: feed from the real DataLoader
    a = p.parse_args(argv)
    if a.dry_run:
        a.model = "tiny"
        a.steps = min(a.steps, 3)
        a.loader_clips = min(a.loader_clips, 16)
        a.real_steps = min(a.real_steps, 2)
        a.workers = min(a.workers, 0)
    return a


def synthetic_data(d):
    """Dry run: 16 white-noise wavs (3-8 s) + a train 'manifest' -> (DataFrame, root)."""
    import pandas as pd
    import soundfile as sf
    d = Path(d)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(16):
        lab = ("bonafide", "spoof")[i % 2]
        f = d / f"s{i}.wav"
        if not f.exists():
            sf.write(f, (rng.standard_normal(int(rng.uniform(3, 8) * 16000)) * 0.1).astype(np.float32), 16000,
                     subtype="PCM_16")
        rows.append({"path": f.name, "label": lab, "source": f"src{i % 4}", "split": "train",
                     "generator": "bonafide" if lab == "bonafide" else f"g{i % 3}"})
    return pd.DataFrame(rows), d


def train_data(a):
    if a.synthetic:
        return synthetic_data(a.synthetic)
    from hearsay.audio import ROOT
    from hearsay.evaluate import manifest
    m = manifest()
    return m[m.split == "train"].reset_index(drop=True), ROOT


def adam8_sanity():
    """-> (ok, note). One AdamW8bit step must match fp32 AdamW on a 64x128 tensor within 2 % of the update size."""
    try:
        import bitsandbytes as bnb
        import torch
    except Exception as e:  # noqa: BLE001 - any import failure means "not available"
        return False, f"bitsandbytes unavailable: {type(e).__name__}: {e}"
    try:
        torch.manual_seed(0)
        w0 = torch.randn(64, 128, device="cuda")
        g = torch.randn_like(w0)
        out = []
        for Opt in (torch.optim.AdamW, bnb.optim.AdamW8bit):
            w = torch.nn.Parameter(w0.clone())
            opt = Opt([w], lr=1e-3)
            w.grad = g.clone()
            opt.step()
            out.append(w.detach() - w0)
        err = float((out[0] - out[1]).abs().max() / out[0].abs().max())
        return err < 0.02, f"1-step relative update error {err:.4f}"
    except Exception as e:  # noqa: BLE001
        return False, f"bitsandbytes sanity step failed: {type(e).__name__}: {e}"


def spill_flag(times):
    return any(t > 3 * float(np.median(times[1:i])) for i, t in enumerate(times) if i >= 3)


def run_one(a):
    """One config in this process -> result dict (printed as a PROBE_RESULT line by main)."""
    import torch
    import torch.nn.functional as F

    from hearsay import device
    from hearsay.models.ssl_e2e import N_SAMP, build_model, param_groups
    cfg = CONFIGS[a.one]
    res = {"key": a.one, **cfg, "model": a.model, "backend": a.backend, "oom": False, "error": None}
    dev = device.resolve(a.device)  # CUDA: caps the allocator at 0.92 x VRAM (G1)
    cuda = dev.type == "cuda"
    if cuda:
        total = torch.cuda.get_device_properties(dev).total_memory
        res.update(gpu=torch.cuda.get_device_name(dev), total_gb=total / 2**30,
                   cap_gb=total * device.CUDA_MEMORY_FRACTION / 2**30)
    if cfg.get("adam8"):
        ok, note = adam8_sanity() if cuda else (False, "8-bit AdamW needs CUDA")
        res["adam8_check"] = note
        if not ok:
            res["skipped"] = note
            return res
    torch.manual_seed(0)
    mb = cfg["mb"]
    try:
        model = build_model(a.model, a.backend, cfg["max_blocks"], cfg["freeze_blocks"], cfg["ckpt"],
                            attn=a.attn).to(dev).train()
        res["attn"] = model.frontend.attn
        groups = param_groups(model)
        params = [p for g in groups for p in g["params"]]
        res["trainable_m"] = sum(p.numel() for p in params) / 1e6
        if cfg.get("adam8"):
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.98))
        else:
            opt = torch.optim.AdamW(groups, betas=(0.9, 0.98))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"]  # post-warm-up regime: the front end gets gradients
        amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda)
        times, data_times = [], []
        loader = None
        if a.real:
            from hearsay.models.ssl_e2e_data import BalancedBatches, TrainSet, train_loader
            df, root = train_data(a)
            loader = iter(train_loader(TrainSet(df, root), BalancedBatches(df, mb // 2), a.workers, cuda))
        x = torch.randn(mb, N_SAMP, device=dev) * 0.05
        y = torch.tensor([0.0, 1.0] * (mb // 2), device=dev)
        n = a.real_steps if loader is not None else a.steps
        for _ in range(n):
            t = time.time()
            if loader is not None:
                xb, yb = next(loader)
                x, y = xb.to(dev), yb.to(dev)
            td = time.time()
            with amp:
                logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.synchronize(dev)
            times.append(time.time() - t)
            data_times.append(td - t)
        last = times[-8:] if len(times) > 1 else times
        med = float(np.median(last[1:] if len(last) > 1 else last))
        res.update(step_s=med, clips_per_s=mb / med, spill=spill_flag(times), step_times=times)
        if loader is None:  # eval forward at --eval-batch windows, AdamW states still resident (as at step 300)
            res.update(eval_batch=a.eval_batch, **eval_forward(model, dev, a.eval_batch, amp))
        if loader is not None:
            compute = [t - d for t, d in zip(times, data_times)]
            res.update(real_steps=n, e2e_clips_per_s=mb * n / sum(times),
                       gpu_clips_per_s=mb * n / sum(compute), data_wait_s=sum(data_times))
    except torch.OutOfMemoryError as e:
        res.update(oom=True, error=str(e).splitlines()[0])
    except Exception as e:  # noqa: BLE001 - reported per config, the probe goes on
        res["error"] = f"{type(e).__name__}: {e}"
    if cuda:
        res.update(peak_alloc_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                   peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2**30)
    else:
        res.update(peak_alloc_gb=None, peak_reserved_gb=None)
    return res


def eval_forward(model, dev, batch, amp):
    """One no_grad eval() forward of `batch` random 4 s windows (2 timed calls) -> eval_ok / eval_s / eval_oom."""
    import torch

    from hearsay.models.ssl_e2e import N_SAMP
    model.eval()
    try:
        with torch.no_grad(), amp:
            x = torch.randn(batch, N_SAMP, device=dev) * 0.05
            ts = []
            for _ in range(2):
                t = time.time()
                model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                ts.append(time.time() - t)
        return {"eval_ok": True, "eval_oom": False, "eval_s": ts[-1]}
    except torch.OutOfMemoryError as e:
        return {"eval_ok": False, "eval_oom": True, "eval_error": str(e).splitlines()[0]}
    finally:
        model.train()


def child(a, key, real=False):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", key, "--device", a.device, "--model", a.model,
           "--backend", a.backend, "--attn", a.attn, "--steps", str(a.steps), "--workers", str(a.workers),
           "--real-steps", str(a.real_steps if real else 0), "--eval-batch", str(a.eval_batch)]
    if a.synthetic:
        cmd += ["--synthetic", a.synthetic]
    if a.dry_run:
        cmd.append("--dry-run")
    if real:
        cmd.append("--real")
    try:
        src = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([src] + [q for q in [os.environ.get("PYTHONPATH")] if q]))
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"key": key, **CONFIGS[key], "error": f"timeout after {a.timeout} s (spill?)", "spill": True}
    for line in p.stdout.splitlines()[::-1]:
        if line.startswith("PROBE_RESULT "):
            return json.loads(line[len("PROBE_RESULT "):])
    return {"key": key, **CONFIGS[key], "error": f"exit {p.returncode}: {p.stderr.strip()[-800:]}"}


def pick(results):
    ok = [r for r in results if not r.get("oom") and not r.get("error") and not r.get("skipped")
          and not r.get("spill") and r.get("clips_per_s") and r.get("eval_ok", True)]
    limit = None
    caps = [r["cap_gb"] for r in results if r.get("cap_gb")]
    if caps:
        limit = BUDGET_FRAC * caps[0]
        ok = [r for r in ok if r["peak_reserved_gb"] is not None and r["peak_reserved_gb"] <= limit]
    best = max(ok, key=lambda r: r["clips_per_s"]) if ok else None
    return (best["key"] if best else None), limit


def loader_throughput(a):
    """Loader-only loop over --loader-clips clips of the real train pipeline -> clips/s (no model)."""
    from hearsay.models.ssl_e2e_data import BalancedBatches, TrainSet, train_loader
    df, root = train_data(a)
    it = iter(train_loader(TrainSet(df, root), BalancedBatches(df, 4), a.workers))
    next(it)  # worker start-up is not throughput
    n, t = 0, time.time()
    while n < a.loader_clips:
        n += len(next(it)[0])
    return {"clips": n, "workers": a.workers, "clips_per_s": n / (time.time() - t)}


def markdown(results, pick_key, limit, thr):
    head = ("| cfg | config | trainable M | peak alloc GB | peak reserved GB | s/step | clips/s | OOM | spill | "
            "eval fwd | note |\n|---|---|---|---|---|---|---|---|---|---|---|\n")

    def ev(r):
        if "eval_batch" not in r:
            return "n/a"
        return f"OOM at {r['eval_batch']}" if r.get("eval_oom") else f"{r['eval_s']:.3f} s @ {r['eval_batch']}"

    def f(v, fmt="{:.2f}"):
        return "n/a" if v is None else fmt.format(v)
    rows = "".join(
        f"| {r['key']}{' **pick**' if r['key'] == pick_key else ''} | {r['desc']} | {f(r.get('trainable_m'), '{:.1f}')}"
        f" | {f(r.get('peak_alloc_gb'))} | {f(r.get('peak_reserved_gb'))} | {f(r.get('step_s'), '{:.3f}')} | "
        f"{f(r.get('clips_per_s'), '{:.1f}')} | {'yes' if r.get('oom') else 'no'} | "
        f"{'yes' if r.get('spill') else 'no'} | {ev(r)} | {r.get('skipped') or r.get('error') or ''} |\n"
        for r in results)
    lim = f"{limit:.2f} GB (85 % of the capped budget)" if limit else "n/a (CPU)"
    tail = f"\nPick: **{pick_key}** (fastest with peak reserved <= {lim}).\n"
    if thr:
        tail += (f"Loader only: {thr['loader']['clips_per_s']:.1f} clips/s ({thr['loader']['workers']} workers). "
                 f"Picked config on the real DataLoader: {f(thr['real'].get('e2e_clips_per_s'), '{:.1f}')} clips/s "
                 f"end to end, {f(thr['real'].get('gpu_clips_per_s'), '{:.1f}')} clips/s compute only.\n")
    return head + rows + tail


def main(argv=None):
    a = parse(argv)
    if a.one:
        res = run_one(a)
        print("PROBE_RESULT " + json.dumps(res, default=float), flush=True)
        return res
    if a.out is None:
        from hearsay.audio import DATA
        a.out = str(DATA / "models" / "r4ft_xlsr")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory() if a.dry_run else None
    if tmp:
        a.synthetic = tmp.name
        synthetic_data(a.synthetic)
    try:
        results = []
        for key in a.configs:
            if CONFIGS[key].get("adam8") and a.device == "cpu":  # don't spawn a process just to learn that
                r = {"key": key, **CONFIGS[key], "skipped": "8-bit AdamW needs CUDA"}
            else:
                r = child(a, key)
            print(f"({key}) {r['desc']}: " + ", ".join(f"{k} {r.get(k)}" for k in (
                "peak_reserved_gb", "step_s", "clips_per_s", "oom", "spill", "error", "skipped") if r.get(k) is not None),
                flush=True)
            results.append(r)
        key, limit = pick(results)
        thr = None
        if key:
            thr = {"loader": loader_throughput(a), "real": child(a, key, real=True)}
            print(f"pick ({key}); loader {thr['loader']['clips_per_s']:.1f} clips/s; real DataLoader "
                  f"{thr['real'].get('e2e_clips_per_s')} clips/s e2e", flush=True)
            if thr["loader"]["clips_per_s"] < (thr["real"].get("gpu_clips_per_s") or 0):
                print("loader is slower than the GPU: add workers or pre-crop long clips on disk", flush=True)
    finally:
        if tmp:
            tmp.cleanup()
    rep = {"pick": key, "pick_desc": CONFIGS[key]["desc"] if key else None, "limit_gb": limit, "dry_run": a.dry_run,
           "model": a.model, "backend": a.backend, "results": results, "throughput": thr,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "probe.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "probe.md").write_text(markdown(results, key, limit, thr), encoding="utf-8")
    print(markdown(results, key, limit, thr), flush=True)
    return rep


if __name__ == "__main__":
    main()
