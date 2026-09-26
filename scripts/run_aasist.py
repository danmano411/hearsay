"""R3: score eval sets (val, val_testlike, test_internal_testlike) or HGT test with AASIST.

    python scripts/run_aasist.py --name R3_aasist_zeroshot_prep
    python scripts/run_aasist.py --name R3_aasist_zeroshot_raw --raw
    python scripts/run_aasist.py --name R3_aasist_ft --weights data/models/r3_aasist_ft.pth
    python scripts/run_aasist.py --name R3_aasist_ft --weights ... --hgt     # inference only
Resumable: per-clip scores are appended to data/scores/cache/<name>[__hgt].csv and skipped on restart.
"""
import argparse
import csv
import time

import numpy as np
import pandas as pd
import torch

from hearsay import device
from hearsay.audio import DATA, ROOT, load
from hearsay.evaluate import SCORES, eval_masks, manifest, report
from hearsay.models import aasist_wrap
from hearsay.preprocess import prep

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--weights", default=str(aasist_wrap.PRETRAINED))
p.add_argument("--raw", action="store_true", help="skip prep() (comparison only)")
p.add_argument("--hgt", action="store_true", help="score data/raw/hgt_test instead of the eval sets")
p.add_argument("--max-windows", type=int, default=1,
               help="4 s windows averaged per clip; 1 = centre window (HGT test median is 3.4 s, so one window is test-like)")
p.add_argument("--testlike-only", action="store_true", help="skip val rows outside val_testlike (cheaper comparisons)")
p.add_argument("--threads", type=int, default=3)
p.add_argument("--batch", type=int, default=16)
device.add_argument(p)
a = p.parse_args()
torch.set_num_threads(a.threads)
dev = device.resolve(a.device)
print(f"device {dev}", flush=True)

if a.hgt:
    paths = sorted(p.relative_to(ROOT).as_posix() for p in (DATA / "raw" / "hgt_test").glob("*.wav"))
else:
    m = manifest()
    masks = eval_masks(m)
    keep = masks["val_testlike"] | masks["test_internal_testlike"]
    paths = m.loc[keep if a.testlike_only else keep | masks["val"], "path"].tolist()

cache = SCORES / "cache" / f"{a.name}{'__hgt' if a.hgt else ''}.csv"
cache.parent.mkdir(parents=True, exist_ok=True)
done = set(pd.read_csv(cache).path) if cache.exists() else set()
todo = [q for q in paths if q not in done]
print(f"{len(paths)} clips, {len(todo)} to score", flush=True)

model = aasist_wrap.load(a.weights, dev).to(memory_format=torch.channels_last)
new = not cache.exists()
t0 = time.time()
with open(cache, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["path", "score"])
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        clips = [load(ROOT / q)[0] for q in chunk]
        clips = clips if a.raw else [prep(y) for y in clips]
        s = aasist_wrap.score_batch(model, clips, a.max_windows)
        w.writerows(zip(chunk, s))
        f.flush()
        if (i // a.batch) % 50 == 0:
            el = time.time() - t0
            print(f"{i + len(chunk)}/{len(todo)}  {el / 60:.1f} min  eta {el / (i + len(chunk)) * (len(todo) - i - len(chunk)) / 60:.0f} min",
                  flush=True)

df = pd.read_csv(cache).drop_duplicates("path")
if a.hgt:
    out = pd.DataFrame({"filename": df.path.str.rsplit("/", n=1).str[-1], "score": df.score})
    out.to_parquet(SCORES / f"{a.name}__hgt.parquet", index=False)
    print(out.score.describe())
else:
    notes = f"AASIST {'ft' if a.weights != str(aasist_wrap.PRETRAINED) else 'pretrained'}, " \
            f"{'raw' if a.raw else 'prep'}, <= {a.max_windows} x 4 s windows"
    report(a.name, df, notes=notes)
