"""R3 data pipeline for AASIST fine-tuning: torch Datasets + DataLoaders so load/augment/prep run in worker processes.

Lives in the package (not in scripts/) because DataLoader workers on Windows are *spawned*: the Dataset, sampler and
worker_init must be importable top-level objects, not lambdas or closures in a __main__ script.

Reproducibility: each training item's augmentation RNG is seeded from (seed, global sample index), so a run gives the
same crops/augmentations whatever --workers is, and a resumed run continues the same stream. worker_init additionally
seeds the global numpy/random state per worker from torch.initial_seed() for any library code that uses them.
"""
import itertools
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from hearsay.audio import SR, load
from hearsay.models.aasist_wrap import NB_SAMP, windows
from hearsay.preprocess import augment, crop, prep

# 4.0375 s. The +0.5 sample matters: crop() does int(seconds * SR) and int(64600 / 16000 * 16000) == 64599, so the
# serial loop before this module trained on 64599-sample crops (AASIST accepted them); now train = eval = NB_SAMP.
WIN_S = (NB_SAMP + 0.5) / SR


def train_clip(path, rng, seconds=WIN_S, rawboost_p=0.0):
    """load -> cheap 6 s pre-crop -> augment() (class-symmetric) -> prep() -> random `seconds` crop (also ssl_e2e)."""
    y = load(path)[0]
    y = crop(y, min(len(y) / SR, 6.0), rng)  # cheap pre-crop so augment/prep don't process 30 s clips
    y = prep(augment(y, rng, rawboost_p))
    return crop(y, seconds, rng)


class TrainClips(Dataset):
    """Item k = row k % len(paths); returns (float32 [NB_SAMP] prep()'d crop, int64 label with 1 = bonafide)."""

    def __init__(self, paths, labels, root, seed=0):
        self.paths = [str(Path(root) / p) for p in paths]
        self.y = (np.asarray(labels) == "bonafide").astype(np.int64)  # AASIST: 1 = bonafide
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        i = k % len(self.paths)
        rng = np.random.default_rng([self.seed, k])
        return torch.from_numpy(np.ascontiguousarray(train_clip(self.paths[i], rng))), torch.tensor(self.y[i])


class EvalClips(Dataset):
    """Deterministic eval input: prep() then the centre 4 s window (same as run_aasist.py --max-windows 1)."""

    def __init__(self, paths, root):
        self.paths = [str(Path(root) / p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(windows(prep(load(self.paths[i])[0]))[0].astype(np.float32))


class CountFrom(Sampler):
    """Endless sample indices start, start+1, ... (batch at step s = indices s*batch .. s*batch+batch-1, as before)."""

    def __init__(self, start=0):
        self.start = start

    def __iter__(self):
        return itertools.count(self.start)


def worker_init(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
    torch.set_num_threads(1)  # workers do numpy/librosa work; don't oversubscribe the CPU


def _loader_kw(workers, pin, prefetch, seed=0):
    kw = {"num_workers": workers, "pin_memory": pin, "generator": torch.Generator().manual_seed(seed)}
    if workers > 0:
        kw.update(worker_init_fn=worker_init, prefetch_factor=prefetch, persistent_workers=True)
    return kw


def train_loader(ds, batch, start_step=0, workers=4, pin=False, prefetch=4):
    """Endless batches continuing from start_step (resume = same sample indices as the serial loop)."""
    return DataLoader(ds, batch_size=batch, sampler=CountFrom(start_step * batch), drop_last=True,
                      **_loader_kw(workers, pin, prefetch, ds.seed + start_step))


def eval_loader(ds, batch=32, workers=4, pin=False):
    kw = _loader_kw(workers, pin, 2)
    kw.pop("persistent_workers", None)  # one pass at startup
    return DataLoader(ds, batch_size=batch, shuffle=False, **kw)
