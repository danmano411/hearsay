"""R3: fine-tune the organizers' ASVspoof5 AASIST on our train split (CPU, wall-clock budgeted).

    python scripts/finetune_aasist.py --hours 3 --threads 2
Start: Baseline-AASIST best.pth. Data: balanced subset of split=='train', random 4 s crops,
augment() (class-symmetric) -> prep() -> crop. Early stopping: combined minDCF on a fixed val_testlike subset
(centre 4 s window, same as run_aasist.py). Best weights -> data/models/r3_aasist_ft.pth; resumable from
data/models/r3_aasist_ft_last.pth.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hearsay.audio import DATA, ROOT, SR, load
from hearsay.evaluate import manifest
from hearsay.metrics import min_dcf
from hearsay.models import aasist_wrap
from hearsay.preprocess import augment, crop, prep

p = argparse.ArgumentParser()
p.add_argument("--hours", type=float, default=3.0)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--per-class", type=int, default=12000)
p.add_argument("--val-n", type=int, default=1200)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--eval-every", type=int, default=150, help="steps")
p.add_argument("--patience", type=int, default=4, help="evals without improvement")
p.add_argument("--tag", default="r3_aasist_ft", help="output file stem under data/models")
a = p.parse_args()
torch.set_num_threads(a.threads)
OUT = DATA / "models"
BEST, LAST, LOG = OUT / f"{a.tag}.pth", OUT / f"{a.tag}_last.pth", OUT / f"{a.tag}_log.jsonl"
WIN_S = aasist_wrap.NB_SAMP / SR  # 4.0375 s

m = manifest()
tr = m[m.split == "train"]
# ponytail: per-class sample weighted 1/sqrt(source size) so small sources are seen; no per-generator balancing
parts = []
for lab, g in tr.groupby("label"):
    w = 1 / np.sqrt(g.groupby("source").path.transform("size"))
    parts.append(g.sample(min(a.per_class, len(g)), weights=w, random_state=0))
train = pd.concat(parts).sample(frac=1, random_state=1).reset_index(drop=True)
print(train.groupby(["label", "source"]).size().to_string(), flush=True)

vt = m[m.val_testlike].sample(a.val_n, random_state=0)
val_y = (vt.label == "spoof").astype(int).to_numpy()
val_x = torch.from_numpy(np.concatenate([aasist_wrap.windows(prep(load(ROOT / q)[0])) for q in vt.path]))
print(f"val subset {len(vt)} ({val_y.sum()} spoof), train {len(train)}", flush=True)


def train_clip(path, rng):
    y = load(ROOT / path)[0]
    y = crop(y, min(len(y) / SR, 6.0), rng)  # cheap pre-crop so augment/prep don't process 30 s clips
    y = prep(augment(y, rng))
    return crop(y, WIN_S, rng)


@torch.no_grad()
def evaluate(model):
    model.eval()
    s = np.concatenate([(lambda o: (o[:, 0] - o[:, 1]).numpy())(model(val_x[i:i + 32].contiguous())[1])
                        for i in range(0, len(val_x), 32)])
    model.train()
    return {k: min_dcf(s, val_y, k) for k in ("official_as_written", "brief_as_written", "combined")}


model = aasist_wrap.load().to(memory_format=torch.channels_last)
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


model.train()
rng = np.random.default_rng(state["step"])
t0, losses = time.time(), []
while time.time() - t0 < a.hours * 3600 and state["bad"] < a.patience:
    idx = [(state["step"] * a.batch + j) % len(train) for j in range(a.batch)]
    rows = train.iloc[idx]
    x = torch.from_numpy(np.stack([train_clip(q, rng) for q in rows.path]))
    y = torch.from_numpy((rows.label == "bonafide").astype(int).to_numpy())  # AASIST convention: 1 = bonafide
    _, out = model(x)
    loss = F.cross_entropy(out, y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item()); state["step"] += 1
    if state["step"] % 10 == 0:
        print(f"step {state['step']} loss {np.mean(losses[-10:]):.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
    if state["step"] % a.eval_every == 0:
        checkpoint()
if state["step"] % a.eval_every:  # time budget ran out between evals
    checkpoint()
print("done", state, flush=True)
