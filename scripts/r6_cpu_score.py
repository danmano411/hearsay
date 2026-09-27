"""Plan 10 CPU backup: score R6's best.pth on CPU when the GPU is unavailable (owner-approved, 2026-09-26).

    python scripts/r6_cpu_score.py [--threads 14] [--name R6_xlsr_light] [--out-name R6_xlsr_light_cpu]

Scores only what the plan-10 rule needs: val_testlike, DeepVoice and the HGT test (inference only). To fit the time,
the window mode is fixed to R4ft's frozen choice (win3) instead of being re-chosen on val_testlike: a documented
deviation. Outputs use a distinct name (default R6_xlsr_light_cpu) so a later GPU upload can't be confused with them:
  data/scores/<out>.parquet (val_testlike rows), data/scores/<out>__hgt.parquet, data/bench/scores/<out>__deepvoice.parquet
Resumable through per-set CSV caches under data/models/r4ft_xlsr/<name>/score_cache_cpu/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_ssl as fs  # noqa: E402

from hearsay.audio import DATA  # noqa: E402
from hearsay.evaluate import SCORES, manifest  # noqa: E402

WIN = 3  # R4ft's frozen inference choice (win3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--name", default="R6_xlsr_light")
    ap.add_argument("--out-name", default="R6_xlsr_light_cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fa = fs.parse(["score", "--device", "cpu", "--name", a.name, "--workers", "4"])
    run = Path(fa.out) / a.name
    saved = json.loads((run / "args.json").read_text())
    for k in fs.ARCH:
        setattr(fa, k, saved[k])
    digest = fs.sha256(run / "best.pth")
    print(f"{a.name} best.pth sha256 {digest[:16]}..., arch {fs.arch(fa)}", flush=True)
    dev = torch.device("cpu")
    model = fs.make_model(fa, grad_ckpt=False)
    model.load_state_dict(torch.load(run / "best.pth", map_location="cpu"))
    model.eval()
    cache = run / "score_cache_cpu"
    cache.mkdir(exist_ok=True)

    m = manifest()
    vt = m[m.val_testlike].path.tolist()
    dv = pd.read_parquet(DATA / "bench" / "deepvoice.parquet").path.tolist()
    hgt = fs.hgt_paths()
    # order: the small sets first so the rule's inputs arrive as early as possible
    for tag, paths in (("deepvoice", dv), ("hgt", hgt), ("val_testlike", vt)):
        s = fs.score_paths(model, paths, WIN, dev, fa, cache / f"{tag}_win3_{digest[:12]}.csv")
        if not np.isfinite(s).all():
            raise SystemExit(f"non-finite scores in {tag}")
        if tag == "deepvoice":
            out = DATA / "bench" / "scores" / f"{a.out_name}__deepvoice.parquet"
            pd.DataFrame({"path": paths, "score": s}).to_parquet(out, index=False)
        elif tag == "hgt":
            out = SCORES / f"{a.out_name}__hgt.parquet"
            pd.DataFrame({"filename": [p.rsplit("/", 1)[-1] for p in paths], "score": s}).to_parquet(out, index=False)
        else:
            out = SCORES / f"{a.out_name}.parquet"
            pd.DataFrame({"path": paths, "score": s}).to_parquet(out, index=False)
        print(f"{tag}: {len(paths)} scored -> {out}", flush=True)
    (run / f"{a.out_name}.json").write_text(json.dumps(
        {"checkpoint_sha256": digest, "windows": "win3 (fixed to R4ft's choice; deviation)", "device": "cpu"}, indent=1))


if __name__ == "__main__":
    main()
