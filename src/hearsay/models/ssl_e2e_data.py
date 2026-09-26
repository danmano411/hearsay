"""R4ft data (plans/08_ssl_aasist.md §6): class-balanced, source-diverse train batches; val_testlike eval windows + cache.

Train: BalancedBatches yields micro-batches of `per_class` bonafide + `per_class` spoof rows of split == "train".
Within a label, a (source, label) cell is drawn with probability ∝ n_cell^power (power 0.5 = √n) times an optional
up-weight (default: asvspoof5 bonafide ×2, from R1's val_testlike slices); within a spoof cell a generator is drawn
∝ n_gen^power; then a clip uniformly at random. Micro-batch b is drawn from np.random.default_rng([seed, b]), so the
sampler "state" is just (seed, next micro-batch) and a resumed run continues the same stream whatever --workers is.
Each item's crop/augmentation RNG is seeded from (seed, global sample id) (same pattern as aasist_data.TrainClips).
Pipeline per item: load -> 6 s pre-crop -> augment() -> prep() -> random 4 s crop (64,000 samples).

Eval: prep(full clip) -> centre 4 s window (crop() tiles short clips), or up to 3 evenly spaced 4 s windows (mean
logit) for final scoring. The val_testlike centre windows are cached once as a float32 memmap.

Top-level classes/functions only: DataLoader workers on Windows are spawned and must import them.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from hearsay.audio import load
from hearsay.models.aasist_data import _loader_kw, train_clip
from hearsay.preprocess import SR, crop, prep

# defined here (re-exported by ssl_e2e) so spawned DataLoader workers need not import transformers
CROP_S = 4.0
N_SAMP = int(CROP_S * SR)  # 64,000 samples -> 199 frames at the 20 ms hop

DEFAULT_UP = {("asvspoof5", "bonafide"): 2.0}
LABELS = ("bonafide", "spoof")


def parse_up(items):
    """['asvspoof5:bonafide=2', ...] -> {(source, label): weight}."""
    out = {}
    for it in items or []:
        key, w = it.split("=")
        src, lab = key.split(":")
        assert lab in LABELS, it
        out[(src, lab)] = float(w)
    return out


def cell_table(df, up=None, power=0.5):
    """-> DataFrame label, source, n, weight, share (sampling share of the cell within its label)."""
    up = DEFAULT_UP if up is None else up
    t = df.groupby(["label", "source"]).size().rename("n").reset_index()
    t["weight"] = t.n.astype(float) ** power * [up.get((s, lab), 1.0) for s, lab in zip(t.source, t.label)]
    t["share"] = t.weight / t.groupby("label").weight.transform("sum")
    return t


class BalancedBatches(Sampler):
    """Endless batch sampler: each batch = list of (row, sample_id) with per_class bonafide then per_class spoof rows.
    `row` indexes df (positional); only split == "train" rows are accepted."""

    def __init__(self, df, per_class=4, seed=0, start=0, up=None, power=0.5):
        if not (df.split.to_numpy() == "train").all():
            raise ValueError("BalancedBatches: every row must have split == 'train'")
        self.per_class, self.seed, self.start, self.power = int(per_class), int(seed), int(start), float(power)
        self.up = DEFAULT_UP if up is None else dict(up)
        self.table = cell_table(df, self.up, power)
        lab, src, gen = df.label.to_numpy(), df.source.to_numpy(), df.generator.to_numpy()
        self.cells = {}  # label -> (probs, [ (gen_probs, [row arrays per generator]) per cell ])
        for label in LABELS:
            t = self.table[self.table.label == label]
            cells = []
            for s in t.source:
                rows = np.flatnonzero((lab == label) & (src == s))
                if label == "spoof":
                    gens, inv = np.unique(gen[rows], return_inverse=True)
                    parts = [rows[inv == g] for g in range(len(gens))]
                    gp = np.array([len(p) for p in parts], float) ** power
                    cells.append((gp / gp.sum(), parts))
                else:
                    cells.append((np.ones(1), [rows]))
            if not cells:
                raise ValueError(f"no train rows with label {label}")
            self.cells[label] = (t.share.to_numpy(), cells)

    def batch(self, b):
        rng = np.random.default_rng([self.seed, int(b)])
        out = []
        for label in LABELS:
            probs, cells = self.cells[label]
            for c in rng.choice(len(cells), size=self.per_class, p=probs):
                gp, parts = cells[c]
                rows = parts[rng.choice(len(parts), p=gp)] if len(parts) > 1 else parts[0]
                out.append(int(rows[rng.integers(len(rows))]))
        n = 2 * self.per_class
        return [(r, int(b) * n + j) for j, r in enumerate(out)]

    def __iter__(self):
        b = self.start
        while True:
            yield self.batch(b)
            b += 1

    def state_dict(self, next_batch):
        """next_batch = micro-batches the trainer has consumed (not what DataLoader workers prefetched)."""
        return {"seed": self.seed, "next_batch": int(next_batch), "per_class": self.per_class, "power": self.power,
                "up": [[s, lab, w] for (s, lab), w in self.up.items()]}

    @classmethod
    def from_state(cls, df, state):
        return cls(df, state["per_class"], state["seed"], state["next_batch"],
                   {(s, lab): w for s, lab, w in state["up"]}, state["power"])


class TrainSet(Dataset):
    """item (row, sample_id) -> (float32 [64000] prep()'d random 4 s crop, float32 label with 1 = spoof)."""

    def __init__(self, df, root, seed=0):
        self.paths = [str(Path(root) / p) for p in df.path]
        self.split = df.split.to_numpy().astype(str)
        self.y = (df.label.to_numpy() == "spoof").astype(np.float32)
        self.seed = int(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        row, sid = item
        # hard rule: nothing outside split == "train" may reach the optimizer
        if self.split[row] != "train":  # a raise, not an assert: asserts vanish under python -O
            raise ValueError(f"non-train row {row} ({self.split[row]}) reached the train loader")
        rng = np.random.default_rng([self.seed, int(sid)])
        x = train_clip(self.paths[row], rng, seconds=CROP_S)
        if len(x) != N_SAMP:
            raise ValueError(f"train crop has {len(x)} samples, expected {N_SAMP}")
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.tensor(self.y[row])


def eval_windows(y, max_windows=1):
    """prep()'d clip -> (k, 64000). max_windows=1: centre window (short clips tile, as crop()). Else up to max_windows
    evenly spaced windows (like aasist_wrap.windows): k = min(max_windows, ceil(len / 64000))."""
    if max_windows == 1 or len(y) <= N_SAMP:
        return crop(y, CROP_S)[None]
    k = min(max_windows, int(np.ceil(len(y) / N_SAMP)))
    starts = np.linspace(0, len(y) - N_SAMP, k).astype(int)
    return np.stack([y[s:s + N_SAMP] for s in starts])


class EvalSet(Dataset):
    """i -> (float32 (k, 64000) windows of prep(full clip), i)."""

    def __init__(self, paths, root, max_windows=1):
        self.paths = [str(Path(root) / p) for p in paths]
        self.max_windows = max_windows

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(eval_windows(prep(load(self.paths[i])[0]), self.max_windows).astype(np.float32)), i


def collate_windows(items):
    """-> (windows (sum k, 64000), clip ids (n,), windows per clip (n,))."""
    return (torch.cat([w for w, _ in items]), torch.tensor([i for _, i in items]),
            torch.tensor([len(w) for w, _ in items]))


def train_loader(ds, sampler, workers=6, pin=False, prefetch=4):
    return DataLoader(ds, batch_sampler=sampler, **_loader_kw(workers, pin, prefetch, ds.seed + sampler.start))


def eval_loader(paths, root, max_windows=1, batch=32, workers=4, pin=False):
    kw = _loader_kw(workers, pin, 2)
    kw.pop("persistent_workers", None)  # one pass
    return DataLoader(EvalSet(paths, root, max_windows), batch_size=batch, shuffle=False, collate_fn=collate_windows,
                      **kw)


def center_cache(paths, root, cache_dir, tag="val_testlike", workers=4):
    """float32 memmap (n, 64000) of prep(full clip) centre windows, built once (resumable across crashes by
    rebuilding), keyed by a hash of the path list so a manifest SYNC invalidates it. -> read-only np.memmap."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1("\n".join(paths).encode()).hexdigest()[:12]
    data, meta = cache_dir / f"{tag}_center_{h}.f32", cache_dir / f"{tag}_center_{h}.json"
    if meta.exists() and data.exists():
        return np.memmap(data, np.float32, "r", shape=(len(paths), N_SAMP))
    tmp = data.with_suffix(".tmp")
    mm = np.memmap(tmp, np.float32, "w+", shape=(len(paths), N_SAMP))
    for x, ids, _ in eval_loader(paths, root, 1, workers=workers):
        mm[ids.numpy()] = x.numpy()
    mm.flush()
    del mm
    os.replace(tmp, data)
    meta.write_text(json.dumps({"n": len(paths), "samples": N_SAMP, "paths_sha1": h, "tag": tag}))
    return np.memmap(data, np.float32, "r", shape=(len(paths), N_SAMP))
