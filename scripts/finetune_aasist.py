"""R3: fine-tune the organizers' ASVspoof5 AASIST on our train split (wall-clock budgeted, CPU or CUDA).

    python scripts/finetune_aasist.py --hours 3 --threads 2
    python scripts/finetune_aasist.py --device cuda --batch 64 --workers 6 --per-class 30000 --hours 4
Start: Baseline-AASIST best.pth. Data: balanced subset of split=='train', random 4 s crops,
augment() (class-symmetric) -> prep() -> crop, done in DataLoader workers (hearsay.models.aasist_data).
Early stopping: combined minDCF on a fixed val_testlike subset (prep() + centre 4 s window, same as run_aasist.py).
Best weights -> <out>/<tag>.pth; resumable from <out>/<tag>_last.pth; log <out>/<tag>_log.jsonl
(default out = data/models, tag = r3_aasist_ft).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hearsay import device
from hearsay.audio import DATA, ROOT
from hearsay.evaluate import manifest
from hearsay.metrics import min_dcf
from hearsay.models import aasist_wrap
from hearsay.models.aasist_data import EvalClips, TrainClips, eval_loader, train_loader


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this global step (0 = no limit; smoke tests)")
    p.add_argument("--threads", type=int, default=2, help="torch threads in the main process")
    p.add_argument("--workers", type=int, default=4, help="DataLoader worker processes (0 = load in the main process)")
    p.add_argument("--prefetch", type=int, default=4, help="batches prefetched per worker")
    p.add_argument("--per-class", type=int, default=12000)
    p.add_argument("--val-n", type=int, default=1200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--eval-every", type=int, default=150, help="steps")
    p.add_argument("--patience", type=int, default=4, help="evals without improvement")
    p.add_argument("--seed", type=int, default=0, help="augmentation/crop seed (per sample index)")
    p.add_argument("--out", default=str(DATA / "models"), help="directory for checkpoint + log")
    p.add_argument("--tag", default="r3_aasist_ft", help="output file stem under --out")
    device.add_argument(p)
    return p.parse_args(argv)


def train_subset(m, per_class):
    tr = m[m.split == "train"]
    # ponytail: per-class sample weighted 1/sqrt(source size) so small sources are seen; no per-generator balancing
    parts = []
    for lab, g in tr.groupby("label"):
        w = 1 / np.sqrt(g.groupby("source").path.transform("size").to_numpy())
        pick = np.random.default_rng(0).choice(len(g), min(per_class, len(g)), replace=False, p=w / w.sum())
        parts.append(g.iloc[pick])
    return pd.concat(parts).sample(frac=1, random_state=1).reset_index(drop=True)


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(a.threads)
    dev = device.resolve(a.device)
    pin = dev.type == "cuda"
    print(f"device {dev}, workers {a.workers}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    BEST, LAST, LOG = out / f"{a.tag}.pth", out / f"{a.tag}_last.pth", out / f"{a.tag}_log.jsonl"

    m = manifest()
    train = train_subset(m, a.per_class)
    print(train.groupby(["label", "source"]).size().to_string(), flush=True)

    vt = m[m.val_testlike].sample(a.val_n, random_state=0)
    val_y = (vt.label == "spoof").astype(int).to_numpy()
    val_x = torch.cat(list(eval_loader(EvalClips(vt.path.tolist(), ROOT), workers=a.workers)))
    print(f"val subset {len(vt)} ({val_y.sum()} spoof), train {len(train)}", flush=True)

    @torch.no_grad()
    def evaluate(model):
        model.eval()
        s = np.concatenate([(lambda o: (o[:, 0] - o[:, 1]).cpu().numpy())(model(val_x[i:i + 32].to(dev))[1])
                            for i in range(0, len(val_x), 32)])
        model.train()
        return {k: min_dcf(s, val_y, k) for k in ("official_as_written", "brief_as_written", "combined")}

    model = aasist_wrap.load(device=dev).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    state = {"step": 0, "best": None, "bad": 0}
    if LAST.exists():
        ck = torch.load(LAST, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
        print(f"resumed at step {state['step']}", flush=True)
    else:
        r = evaluate(model)
        state["best"] = r["combined"]
        torch.save(model.state_dict(), BEST)
        with open(LOG, "a") as f:
            f.write(json.dumps({"step": 0, "loss": None, **r}) + "\n")
        print("pretrained", r, flush=True)

    losses = []

    def checkpoint():
        r = evaluate(model)
        better = r["combined"] < state["best"]
        if better:
            state["best"], state["bad"] = r["combined"], 0
            torch.save(model.state_dict(), BEST)
        else:
            state["bad"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, LAST)
        with open(LOG, "a") as f:
            f.write(json.dumps({"step": state["step"], "loss": float(np.mean(losses[-a.eval_every:])), **r}) + "\n")
        print(f"eval step {state['step']}: {r} {'*best*' if better else ''}", flush=True)

    ds = TrainClips(train.path.tolist(), train.label.to_numpy(), ROOT, seed=a.seed)
    batches = iter(train_loader(ds, a.batch, state["step"], a.workers, pin, a.prefetch))
    model.train()
    t0 = time.time()
    tw, nw = t0, 0  # throughput window
    while time.time() - t0 < a.hours * 3600 and state["bad"] < a.patience \
            and not (a.max_steps and state["step"] >= a.max_steps):
        x, y = next(batches)
        x, y = x.to(dev, non_blocking=pin), y.to(dev, non_blocking=pin)
        _, o = model(x)
        loss = F.cross_entropy(o, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item()); state["step"] += 1; nw += len(y)
        if state["step"] % 10 == 0:
            now = time.time()
            print(f"step {state['step']} loss {np.mean(losses[-10:]):.4f} {(now - t0) / 60:.1f} min "
                  f"{nw / (now - tw):.1f} samples/s", flush=True)
            tw, nw = now, 0
        if state["step"] % a.eval_every == 0:
            checkpoint()
    if losses and state["step"] % a.eval_every:  # budget ran out between evals
        checkpoint()
    print(f"done {state}  {len(losses) * a.batch / (time.time() - t0):.1f} samples/s overall", flush=True)


if __name__ == "__main__":
    main()
