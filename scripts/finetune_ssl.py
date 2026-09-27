"""R4ft / R6: end-to-end fine-tune of XLS-R-300M + a spoofing back end on raw 16 kHz audio.

    python scripts/finetune_ssl.py probe --device cuda                       # VRAM probe first (scripts/vram_probe.py)
    python scripts/finetune_ssl.py train --backend light                     # run 1: K = 12, back end A
    python scripts/finetune_ssl.py train --backend aasist                    # run 2: back end B
    python scripts/finetune_ssl.py score --backend light --hgt               # (or --score) final scoring
    python scripts/finetune_ssl.py train --tiny --device cpu --max-steps 5 --out <tmp>   # CPU smoke, random tiny model

Train: only split == "train" rows (asserted per row in the loader), micro-batches of 4 bonafide + 4 spoof
(ssl_e2e_data.BalancedBatches), bf16 autocast on CUDA with fp32 master weights and fp32 AdamW, gradient accumulation,
layer-wise LR decay on the front end, back-end-only warm-up (front end frozen), then linear warm-up + cosine to 5 %.
Eval on val_testlike (prep() + centre 4 s window, cached memmap) every --eval-every optimizer steps and at the end of
the back-end warm-up: official/brief/combined minDCF + EER, per-source slices (diagnostic only). best.pth (weights
only) on a >= --min-delta combined improvement; early stop after --patience evals without one, or at --max-epochs
virtual epochs; NaN loss -> restore last.pth, halve LRs, one retry. last.pth (atomic, every --save-every steps and at
every eval) holds model/optimizer/schedule/sampler/RNG state, and `train` resumes from it automatically.
Outputs under <out>/<name>/: args.json, log.jsonl, last.pth, best.pth, config.json; cache under <out>/cache/.

Score: loads best.pth, chooses centre vs 3 windows on val_testlike only, writes config.json (checkpoint sha256 +
inference mode) BEFORE scoring test_internal_testlike, then data/scores/<name>.parquet (path, score) for
val / val_testlike / test_internal_testlike rows and, with --hgt, data/scores/<name>__hgt.parquet (filename, score) —
HGT strictly inference-only in eval() mode, after config.json exists. Self-check with evaluate(); the leaderboard
(reports/leaderboard.md) is written separately with hearsay.evaluate.report().
"""
import argparse
import contextlib
import csv
import ctypes
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hearsay import device
from hearsay.audio import DATA, ROOT
from hearsay.evaluate import SCORES, evaluate, manifest, metrics_for, per_source
from hearsay.models import ssl_e2e
from hearsay.models.ssl_e2e_data import (DEFAULT_UP, BalancedBatches, TrainSet, center_cache, eval_loader, parse_up,
                                        train_loader)

HGT_DIR = DATA / "raw" / "hgt_test"
TEMPLATE_NAME = "HGT_Hearsay_score_template.csv"  # scripts/score.py TEMPLATE
N_HGT = 1671
OUT = DATA / "models" / "r4ft_xlsr"
# args that define the model/optimizer layout: a resume or a score with different values would be silently wrong
ARCH = ("model", "backend", "max_blocks", "freeze_blocks", "micro_batch", "accum")
WATCH = ("asvspoof5", "asvspoof2019_la", "playht", "unit_speech")  # plan §7 watch list (val_testlike slices)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", nargs="?", default="train", choices=["train", "score", "probe"])
    p.add_argument("--score", action="store_true", help="same as mode 'score'")
    p.add_argument("--model", default="xlsr_300m", help="xlsr_300m | wavlm_base_plus | w2v2_base | HF id | tiny")
    p.add_argument("--tiny", action="store_true", help="debug: random-init tiny wav2vec2 (no download)")
    p.add_argument("--backend", default="light", help="light (A) | aasist (B)")
    p.add_argument("--max-blocks", type=int, default=12, help="K: keep the first K transformer blocks")
    p.add_argument("--freeze-blocks", type=int, default=0, help="also freeze the lowest N blocks (+ everything below)")
    p.add_argument("--no-ckpt", action="store_true", help="disable gradient checkpointing")
    p.add_argument("--attn", default="sdpa", help="sdpa | eager (falls back to eager if unsupported)")
    p.add_argument("--mask-time-prob", type=float, default=0.05)
    p.add_argument("--mask-time-length", type=int, default=10)
    p.add_argument("--micro-batch", type=int, default=8, help="clips per forward (half bonafide, half spoof)")
    p.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step (effective 32)")
    p.add_argument("--lr", type=float, default=2e-5, help="front-end LR of the top block")
    p.add_argument("--lr-decay", type=float, default=0.85, help="layer-wise LR decay per block downward")
    p.add_argument("--backend-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    p.add_argument("--warmup-backend", type=int, default=300, help="optimizer steps with the front end frozen")
    p.add_argument("--warmup-front", type=int, default=500, help="linear warm-up of front-end LRs after that")
    p.add_argument("--min-lr-frac", type=float, default=0.05, help="cosine floor")
    p.add_argument("--epoch-steps", type=int, default=1024, help="optimizer steps per virtual epoch (32,768 clips)")
    p.add_argument("--max-epochs", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=1024, help="optimizer steps")
    p.add_argument("--patience", type=int, default=4, help="evals without a >= min-delta improvement")
    p.add_argument("--min-delta", type=float, default=0.002)
    p.add_argument("--save-every", type=int, default=250, help="optimizer steps between last.pth saves")
    p.add_argument("--log-every", type=int, default=50, help="optimizer steps between train log lines")
    p.add_argument("--max-steps", type=int, default=0, help="stop at this optimizer step (0 = no limit; smoke tests)")
    p.add_argument("--hours", type=float, default=0, help="wall-clock budget for this invocation (0 = none)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--up", action="append", default=None,
                   help="cell up-weight source:label=w (repeatable; default asvspoof5:bonafide=2)")
    p.add_argument("--power", type=float, default=0.5, help="cell/generator weight = n^power")
    p.add_argument("--val-n", type=int, default=0, help="subsample val_testlike (0 = all 8,963; smoke tests only)")
    p.add_argument("--eval-batch", type=int, default=32, help="4 s windows per eval forward (halved on OOM)")
    p.add_argument("--workers", type=int, default=6, help="DataLoader worker processes (0 = main process)")
    p.add_argument("--prefetch", type=int, default=4)
    p.add_argument("--threads", type=int, default=0, help="torch threads in the main process (0 = torch default)")
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "off"], help="auto = bf16 autocast on CUDA only")
    p.add_argument("--out", default=str(OUT), help="run dirs go to <out>/<name>/, eval cache to <out>/cache/")
    p.add_argument("--name", default=None, help="run/score name (default R4ft_xlsr_<backend>)")
    p.add_argument("--scores-dir", default=str(SCORES))
    p.add_argument("--windows", default="auto", choices=["auto", "center", "win3"],
                   help="score: inference windows (auto = choose on val_testlike)")
    p.add_argument("--hgt", action="store_true", help="score: also score HGT test audio (inference only)")
    p.add_argument("--force", action="store_true",
                   help="score: overwrite an existing config.json with another checkpoint/inference mode (recorded)")
    device.add_argument(p)
    a, rest = p.parse_known_args(argv)
    if a.mode != "probe" and rest:
        p.error(f"unrecognized arguments: {rest}")
    a.probe_args = rest
    if a.score:
        a.mode = "score"
    if a.tiny:
        a.model = "tiny"
    a.backend = ssl_e2e.backend_name(a.backend)
    if a.micro_batch % 2:
        p.error("--micro-batch must be even (half bonafide, half spoof)")
    if a.name is None:
        short = {"xlsr_300m": "xlsr"}.get(a.model, a.model.replace("/", "_"))
        a.name = f"R4ft_{short}_{a.backend}"
    # resolve the default here so args.json / config.json record the weights the sampler actually applies
    a.up = parse_up(a.up) if a.up else dict(DEFAULT_UP)
    return a


# ---------------------------------------------------------------------------------------------------------------- #
# keep-awake (Windows): during training only, restored on exit; no power settings are changed

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def _kernel32():
    if sys.platform != "win32":
        return None
    k = ctypes.windll.kernel32
    k.SetThreadExecutionState.restype = ctypes.c_uint32
    k.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    return k


@contextlib.contextmanager
def keep_awake():
    k = _kernel32()
    if k is None:
        yield
        return
    k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        k.SetThreadExecutionState(ES_CONTINUOUS)  # clear the system-required flag


# ---------------------------------------------------------------------------------------------------------------- #

def make_model(a, grad_ckpt=None):
    return ssl_e2e.build_model(a.model, a.backend, a.max_blocks, a.freeze_blocks,
                               (not a.no_ckpt) if grad_ckpt is None else grad_ckpt,
                               a.mask_time_prob, a.mask_time_length, a.attn)


def lr_mult(kind, step, a):
    """Front end: 0 during the back-end warm-up, then linear warm-up, then cosine to min_lr_frac at the budget end.
    Back end: full LR until the front-end warm-up ends, then the same cosine."""
    total = a.max_epochs * a.epoch_steps
    w0, w1 = a.warmup_backend, a.warmup_backend + a.warmup_front
    if kind == "front":
        if step < w0:
            return 0.0
        if step < w1:
            return (step - w0 + 1) / a.warmup_front
    elif step < w1:
        return 1.0
    frac = min(1.0, (step - w1) / max(1, total - w1))
    return a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac))


def set_lrs(opt, step, a, scale):
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * lr_mult(g["kind"], step, a) * scale


def rng_state():
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_initialized():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if "cuda" in s and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    tmp = Path(path).with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def autocast(dev, a):
    on = a.amp == "bf16" or (a.amp == "auto" and dev.type == "cuda")
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=on)


def log_line(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")


@torch.no_grad()
def forward_windows(model, x, dev, a, log=print):
    """Logits of an (n, 64000) window array in chunks of a.eval_batch WINDOWS (not clips). On a CUDA OOM: empty the
    cache, halve a.eval_batch (kept for later calls), log it and retry the same chunk; re-raises at batch 1."""
    out, i = [], 0
    while i < len(x):
        chunk = x[i:i + a.eval_batch]
        try:
            with autocast(dev, a):
                o = model(torch.from_numpy(np.array(chunk, np.float32)).to(dev)).float().cpu()
        except torch.OutOfMemoryError:
            if a.eval_batch <= 1:
                raise
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            a.eval_batch = max(1, a.eval_batch // 2)
            log(f"eval OOM: eval batch halved to {a.eval_batch} windows")
            continue
        out.append(o)
        i += len(chunk)
    return torch.cat(out).numpy().astype(np.float64) if out else np.zeros(0)


def logits_of(model, x, dev, a, log=print):
    was = model.training
    model.eval()
    # HF's encoder draws torch.rand([]) per layer (layerdrop) even in eval mode: fork the RNG so an eval does not
    # shift the training stream (a run is then the same whether or not/when it evaluated or resumed)
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        s = forward_windows(model, x, dev, a, log)
    model.train(was)
    return s


def val_testlike(m, a):
    vt = m[m.val_testlike]
    if a.val_n and a.val_n < len(vt):
        vt = vt.sample(a.val_n, random_state=0)
    return vt.sort_values("path").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------- #

def train(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = device.resolve(a.device)
    pin = dev.type == "cuda"
    run = Path(a.out) / a.name
    run.mkdir(parents=True, exist_ok=True)
    LAST, BEST, LOG = run / "last.pth", run / "best.pth", run / "log.jsonl"
    print(f"device {dev}, run {run}, workers {a.workers}", flush=True)

    m = manifest()
    tr = m[m.split == "train"].reset_index(drop=True)
    train_sha1 = hashlib.sha1("\n".join(tr.path).encode()).hexdigest()
    vt = val_testlike(m, a)
    val_y = vt.label.to_numpy()
    val_x = center_cache(vt.path.tolist(), ROOT, Path(a.out) / "cache", workers=a.workers)
    print(f"train rows {len(tr)}, val_testlike {len(vt)} ({(val_y == 'spoof').sum()} spoof)", flush=True)

    torch.manual_seed(a.seed)  # model init, dropout
    np.random.seed(a.seed)  # HF time masking draws from numpy's global RNG
    random.seed(a.seed)
    model = make_model(a).to(dev).train()
    groups = ssl_e2e.param_groups(model, a.lr, a.lr_decay, a.backend_lr, a.weight_decay)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
    print(f"front end {a.model} K={model.frontend.n_blocks} attn={model.frontend.attn} frozen-lowest="
          f"{a.freeze_blocks} ckpt={not a.no_ckpt}; trainable {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    state = {"step": 0, "clips": 0, "best": None, "bad": 0, "lr_scale": 1.0, "retries": 0, "evals": 0,
             "stop": None, "since_eval": []}  # since_eval: train losses since the last eval (survives a pause)

    def save_last():
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "state": dict(state),
                     "sched": {"step": state["step"], "lr_scale": state["lr_scale"]},
                     "sampler": sampler_state(), "rng": rng_state(), "args": arch(a), "train_sha1": train_sha1},
                    LAST)

    def load_last():
        ck = torch.load(LAST, map_location="cpu", weights_only=False)
        if ck["args"] != arch(a):
            raise SystemExit(f"{LAST} was trained with {ck['args']}, not {arch(a)}: use another --name")
        if ck.get("train_sha1") != train_sha1:
            raise SystemExit(f"{LAST} was trained on a different train split (train-path sha1 {ck.get('train_sha1')} "
                             f"!= {train_sha1}): the manifest changed (SYNC?) between legs. Sampler row indices would "
                             f"point at other clips, so refusing to resume; start a new --name.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state.update(ck["state"])
        set_rng_state(ck["rng"])
        return ck

    def sampler_state():
        return sampler.state_dict(state["step"] * a.accum)

    sampler = BalancedBatches(tr, a.micro_batch // 2, a.seed, 0, a.up, a.power)
    if LAST.exists():
        ck = load_last()
        print(f"resumed at step {state['step']} (best {state['best']}, bad {state['bad']})", flush=True)
        sampler = BalancedBatches.from_state(tr, ck["sampler"])
    else:
        (run / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "up"} |
                                                  {"up": [[s, lab, w] for (s, lab), w in a.up.items()]},
                                                  indent=1))
        save_last()  # step 0: the "last good checkpoint" a NaN restore can fall back to
    if state["stop"]:
        print(f"run already finished ({state['stop']}); delete {LAST} to retrain", flush=True)
        return state
    ds = TrainSet(tr, ROOT, a.seed)

    def batches():
        sampler.start = state["step"] * a.accum  # a resume mid-accumulation restarts that group
        return iter(train_loader(ds, sampler, a.workers, pin, a.prefetch))

    def oom_log(msg):
        print(msg, flush=True)
        log_line(LOG, {"type": "oom", "step": state["step"], "where": "eval", "eval_batch": a.eval_batch})

    def do_eval(train_loss):
        t = time.time()
        s = logits_of(model, val_x, dev, a, log=oom_log)
        r = metrics_for(s, val_y)
        state["evals"] += 1
        better = state["best"] is None or r["combined"] <= state["best"] - a.min_delta
        if better:
            state["best"], state["bad"] = r["combined"], 0
            atomic_save(model.state_dict(), BEST)
        else:
            state["bad"] += 1
        sl = per_source(pd.DataFrame({"path": vt.path, "score": s}), "val_testlike", m)
        rec = {"type": "eval", "step": state["step"], "clips": state["clips"], "train_loss": train_loss,
               "lr_front_top": max((g["lr"] for g in opt.param_groups if g["kind"] == "front"), default=0.0),
               "lr_back": next(g["lr"] for g in opt.param_groups if g["kind"] == "back"),
               "official_as_written": r["official"], "brief_as_written": r["brief"], "combined": r["combined"],
               "eer": r["eer"], "best": state["best"], "bad": state["bad"], "new_best": better,
               "max_memory_reserved": torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None,
               "clips_per_s": tput(), "eval_s": time.time() - t,
               "slices": sl[["by", "value", "n", "official", "brief", "combined"]].to_dict("records")}
        log_line(LOG, rec)
        watch = sl[sl.value.isin(WATCH)]
        flag = "*best*" if better else f"no improvement ({state['bad']}/{a.patience})"
        print(f"eval step {state['step']}: official {r['official']:.4f} brief {r['brief']:.4f} combined "
              f"{r['combined']:.4f} EER {r['eer']:.2f}%  {flag}", flush=True)
        if len(watch):
            print("  watch slices (combined): " + ", ".join(f"{b}={v} {c:.3f}" for b, v, c in
                                                            zip(watch.by, watch.value, watch.combined)), flush=True)
        save_last()

    t_win = [time.time(), 0]

    def tput():
        now = time.time()
        v = t_win[1] / max(now - t_win[0], 1e-9)
        t_win[0], t_win[1] = now, 0
        return v

    total = a.max_epochs * a.epoch_steps
    t0 = time.time()
    losses, step_times = [], []
    it = batches()
    last_eval_step = None
    with keep_awake():
        while True:
            if state["bad"] >= a.patience:
                state["stop"] = f"early stop: {a.patience} evals without a {a.min_delta} improvement"
            elif state["step"] >= total:
                state["stop"] = f"budget: {a.max_epochs} virtual epochs"
            if state["stop"] or (a.max_steps and state["step"] >= a.max_steps) or \
                    (a.hours and time.time() - t0 > a.hours * 3600):
                break
            ts = time.time()
            set_lrs(opt, state["step"], a, state["lr_scale"])
            front = state["step"] >= a.warmup_backend
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for _ in range(a.accum):
                x, y = next(it)
                x, y = x.to(dev, non_blocking=pin), y.to(dev, non_blocking=pin)
                with autocast(dev, a):
                    logit = model(x, frontend_grad=front)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y) / a.accum
                loss.backward()
                tot += loss.item()
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
            if not (math.isfinite(tot) and math.isfinite(float(gn))):
                if state["retries"] >= 1:
                    msg = f"non-finite loss/grad at step {state['step']} after one retry"
                    log_line(LOG, {"type": "nan", "step": state["step"], "action": "stop"})
                    load_last()  # keep the last good weights in last.pth, marked as stopped
                    state["stop"] = msg
                    save_last()
                    break
                scale, retries = state["lr_scale"] * 0.5, state["retries"] + 1
                bad_step = state["step"]
                load_last()
                state["lr_scale"], state["retries"] = scale, retries
                log_line(LOG, {"type": "nan", "step": bad_step, "action": "restore", "to_step": state["step"],
                               "lr_scale": scale})
                print(f"non-finite loss at step {bad_step}: restored step {state['step']}, LRs x{scale}", flush=True)
                it = batches()
                continue
            opt.step()
            state["step"] += 1
            state["clips"] += a.micro_batch * a.accum
            t_win[1] += a.micro_batch * a.accum
            losses.append(tot)
            state["since_eval"].append(tot)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            dt = time.time() - ts
            spill = len(step_times) >= 8 and dt > 3 * float(np.median(step_times[-50:]))
            step_times.append(dt)
            s = state["step"]
            if s % a.log_every == 0 or spill:
                rec = {"type": "train", "step": s, "clips": state["clips"], "loss": float(np.mean(losses[-a.log_every:])),
                       "step_loss": tot, "step_s": dt, "spill_suspect": spill, "grad_norm": float(gn),
                       "lr_front_top": max((g["lr"] for g in opt.param_groups if g["kind"] == "front"), default=0.0),
                       "lr_back": next(g["lr"] for g in opt.param_groups if g["kind"] == "back")}
                log_line(LOG, rec)
                print(f"step {s} loss {rec['loss']:.4f} {dt:.2f} s/step{' SPILL?' if spill else ''} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if (a.warmup_backend and s == a.warmup_backend) or s % a.eval_every == 0:
                do_eval(float(np.mean(state["since_eval"])))
                state["since_eval"], last_eval_step = [], s
            elif s % a.save_every == 0:
                save_last()
        stop = state["stop"] or ""
        if stop.startswith("budget") and losses and last_eval_step != state["step"]:
            do_eval(float(np.mean(state["since_eval"])) if state["since_eval"] else None)  # final scheduled eval
        elif not stop.startswith("non-finite"):
            # a pause (--max-steps/--hours) or early stop only saves: an unscheduled eval would move best.pth and the
            # patience counter, so a paused + resumed run would stop differently from an uninterrupted one
            save_last()
    print(f"done at step {state['step']}: {state['stop'] or 'paused (resume with the same command)'}; "
          f"best val_testlike combined {state['best']}", flush=True)
    return state


def arch(a):
    return {k: getattr(a, k) for k in ARCH}


# ---------------------------------------------------------------------------------------------------------------- #

def template_names():
    """Filenames of the organizers' score template (same parsing as scripts/make_submission.py: tab-separated,
    header 'filename<TAB>cm-score'), located like scripts/score.py (HGT_DIR / HGT_Hearsay_score_template.csv)."""
    t = HGT_DIR / TEMPLATE_NAME
    if not t.exists():
        raise SystemExit(f"score template {t} missing")
    return [ln.split("\t")[0] for ln in t.read_text().rstrip("\n").split("\n")[1:]]


def hgt_paths():
    """HGT test audio (inference only). Only called after config.json is written."""
    return sorted(p.relative_to(ROOT).as_posix() for p in HGT_DIR.glob("*.wav"))


def score_paths(model, paths, max_windows, dev, a, cache_csv=None):
    """-> np.array of mean window logits, in `paths` order. Resumable through an optional per-clip CSV cache."""
    done = {}
    if cache_csv is not None and Path(cache_csv).exists():
        done = dict(pd.read_csv(cache_csv).itertuples(index=False, name=None))
    todo = [p for p in paths if p not in done]
    if todo:
        f = open(cache_csv, "a", newline="") if cache_csv is not None else None
        w = csv.writer(f) if f else None
        if f and f.tell() == 0:
            w.writerow(["path", "score"])
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            clips = max(1, a.eval_batch // max_windows)  # <= eval_batch windows per loader batch
            for n, (x, ids, k) in enumerate(eval_loader(todo, ROOT, max_windows, clips, a.workers,
                                                         dev.type == "cuda")):
                s = forward_windows(model, x.numpy(), dev, a)
                bounds = np.cumsum([0] + k.tolist())
                for i, lo, hi in zip(ids.tolist(), bounds[:-1], bounds[1:]):
                    done[todo[i]] = float(s[lo:hi].mean())
                    if w:
                        w.writerow([todo[i], done[todo[i]]])
                if f:
                    f.flush()
                if n % 50 == 0:
                    print(f"  scored {min((n + 1) * clips, len(todo))}/{len(todo)} "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if f:
            f.close()
    return np.array([done[p] for p in paths])


def score(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = device.resolve(a.device)
    run = Path(a.out) / a.name
    best = run / "best.pth"
    if not best.exists():
        raise SystemExit(f"{best} missing: train first")
    saved = json.loads((run / "args.json").read_text())
    for k in ARCH:  # the architecture comes from the run, not from the command line
        setattr(a, k, saved[k])
    digest = sha256(best)
    model = make_model(a, grad_ckpt=False).to(dev)
    model.load_state_dict(torch.load(best, map_location="cpu"))
    model.eval()  # BatchNorm uses train running stats; no test-time adaptation
    cache = run / "score_cache"
    cache.mkdir(exist_ok=True)
    tag = digest[:12]

    # 1) inference mode, chosen on val_testlike only
    m = manifest()
    vt = m[m.val_testlike]
    modes = {"center": 1, "win3": 3}
    vt_scores, vt_res = {}, {}
    for mode, k in modes.items():
        if a.windows not in ("auto", mode):
            continue
        vt_scores[mode] = score_paths(model, vt.path.tolist(), k, dev, a, cache / f"val_testlike_{mode}_{tag}.csv")
        vt_res[mode] = metrics_for(vt_scores[mode], vt.label)
        print(f"val_testlike {mode}: {vt_res[mode]}", flush=True)
    mode = min(vt_res, key=lambda md: (vt_res[md]["combined"], md != "center"))  # tie -> centre (cheaper)
    manual = a.windows != "auto"
    cfg_path = run / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if (old.get("sha256"), old.get("inference")) != (digest, mode) and not a.force:
            raise SystemExit(f"{cfg_path} already fixes checkpoint {str(old.get('sha256'))[:16]}... / "
                             f"{old.get('inference')}; this run would be {digest[:16]}... / {mode}. The headline is "
                             f"scored once after the choice is frozen: refusing to overwrite (pass --force to override, "
                             f"it is recorded).")
    probe = Path(a.out) / "probe.json"
    config = {"name": a.name, "checkpoint": str(best), "sha256": digest, "inference": mode,
              "max_windows": modes[mode], "val_testlike": vt_res,
              "selected_on": "manual (--windows)" if manual else "val_testlike combined minDCF",
              "forced": manual, "overwrite_forced": bool(a.force and cfg_path.exists()),
              "arch": arch(a), "args": saved,
              "probe_pick": json.loads(probe.read_text()).get("pick") if probe.exists() else None,
              "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = run / "config.json.tmp"
    tmp.write_text(json.dumps(config, indent=1, default=str))
    os.replace(tmp, cfg_path)
    print(f"config.json written: {mode} ({modes[mode]} window(s)), sha256 {digest[:16]}...", flush=True)

    # 2) the rest of val + test_internal_testlike, with the frozen choice
    rest = m[(m.split == "val") | m.test_internal_testlike]
    rest = rest[~rest.path.isin(vt.path)]
    s_rest = score_paths(model, rest.path.tolist(), modes[mode], dev, a, cache / f"rest_{mode}_{tag}.csv")
    df = pd.DataFrame({"path": list(vt.path) + list(rest.path),
                       "score": np.concatenate([vt_scores[mode], s_rest])})
    if not np.isfinite(df.score).all():
        raise SystemExit("non-finite scores on val/test_internal")
    out_dir = Path(a.scores_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.name}.parquet"
    df.to_parquet(out, index=False)
    res = evaluate(df, m)  # self-check; report() writes the leaderboard separately
    print(res.to_string(index=False), flush=True)
    puts = [out]

    # 3) HGT: inference only, frozen checkpoint + frozen window choice, after config.json exists
    if a.hgt:
        if not cfg_path.exists():
            raise SystemExit("config.json must exist before HGT is read")
        paths = hgt_paths()
        names = [p.rsplit("/", 1)[-1] for p in paths]
        want = template_names()
        if len(paths) != N_HGT or len(want) != N_HGT or sorted(names) != sorted(want):
            raise SystemExit(f"HGT audio ({len(paths)} files) does not match the organizers' template "
                             f"({len(want)} rows, expected {N_HGT}); missing {sorted(set(want) - set(names))[:3]}, "
                             f"extra {sorted(set(names) - set(want))[:3]}")
        s = score_paths(model, paths, modes[mode], dev, a, cache / f"hgt_{mode}_{tag}.csv")
        hg = pd.DataFrame({"filename": names, "score": s})
        if len(hg) != N_HGT or not np.isfinite(hg.score).all():
            raise SystemExit("HGT scoring incomplete or non-finite")
        hout = out_dir / f"{a.name}__hgt.parquet"
        hg.to_parquet(hout, index=False)
        print(f"HGT: {len(hg)} clips scored -> {hout}", flush=True)
        puts.append(hout)
    print("wrote:", flush=True)
    for p in puts:
        print(f"  {p}", flush=True)
    return res, config


def main(argv=None):
    a = parse(argv)
    if a.mode == "probe":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import vram_probe
        return vram_probe.main(a.probe_args + ["--device", a.device, "--out", a.out])
    return score(a) if a.mode == "score" else train(a)


if __name__ == "__main__":
    main()
