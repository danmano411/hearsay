"""R3 DataLoader (finetune_aasist.py) and run_aasist.py's gpu-safe path. CPU only, synthetic wavs, a few seconds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from hearsay import evaluate as E
from hearsay.models import aasist_data as D
from hearsay.models.aasist_wrap import NB_SAMP, windows
from hearsay.preprocess import SR, prep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """White noise wavs of 2, 5 and 8 s (broadband, so the 7 kHz low-pass in prep() is visible)."""
    root = tmp_path_factory.mktemp("wavs")
    rng = np.random.default_rng(0)
    paths, labels = [], []
    for i, (sec, lab) in enumerate([(2, "bonafide"), (5, "spoof"), (8, "bonafide"), (3, "spoof")]):
        sf.write(root / f"c{i}.wav", (rng.standard_normal(int(sec * SR)) * 0.1).astype(np.float32), SR,
                 subtype="PCM_16")
        paths.append(f"c{i}.wav"); labels.append(lab)
    return root, paths, labels


def hf_fraction(x):
    """Share of spectral energy above 7.5 kHz (white noise: ~6%; after prep()'s 7 kHz low-pass: ~0)."""
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return p[f > 7500].sum() / p.sum()


def test_train_item_shape_label_and_prep(clips):
    root, paths, labels = clips
    ds = D.TrainClips(paths, labels, root, seed=3)
    for k in range(6):  # wraps past len(ds)
        x, y = ds[k]
        assert x.dtype == torch.float32 and x.shape == (NB_SAMP,)
        assert y.dtype == torch.int64 and int(y) == (labels[k % len(paths)] == "bonafide")  # 1 = bonafide
        assert hf_fraction(x.numpy()) < 1e-3  # went through prep()
        assert 0.01 < float(x.pow(2).mean().sqrt()) < 0.2  # RMS-normalized (0.05) before the crop
    assert torch.equal(ds[1][0], ds[1][0])  # deterministic per sample index
    assert not torch.equal(ds[1][0], ds[1 + len(paths)][0])  # same clip, next epoch: new crop/augmentation


def test_eval_item_is_prep_centre_window(clips):
    root, paths, _ = clips
    ds = D.EvalClips(paths, root)
    y = sf.read(root / paths[2], dtype="float32")[0]
    np.testing.assert_allclose(ds[2].numpy(), windows(prep(y))[0], atol=1e-4)
    assert ds[0].shape == (NB_SAMP,)  # 2 s clip tiled


def _batches(ds, workers, start_step=0, n=3):
    it = iter(D.train_loader(ds, batch=2, start_step=start_step, workers=workers, prefetch=2))
    return [next(it) for _ in range(n)]


def test_loader_workers_0_and_2_match(clips):
    """Spawned workers (Windows) work, and the batches don't depend on the worker count or on resuming."""
    root, paths, labels = clips
    ds = D.TrainClips(paths, labels, root, seed=0)
    b0 = _batches(ds, 0)
    b2 = _batches(ds, 2)
    for (x0, y0), (x2, y2) in zip(b0, b2):
        assert x0.shape == (2, NB_SAMP) and y0.tolist() == y2.tolist()
        torch.testing.assert_close(x0, x2)
    # resume at step 1 = continue the same sample indices (s*batch ...), like the old serial loop
    (xr, yr), = _batches(ds, 0, start_step=1, n=1)
    torch.testing.assert_close(xr, b0[1][0])
    assert yr.tolist() == [int(labels[i] == "bonafide") for i in (2, 3)]
    ev = torch.cat(list(D.eval_loader(D.EvalClips(paths, root), batch=3, workers=2)))
    assert ev.shape == (len(paths), NB_SAMP)


# --- run_aasist.py: the gpu node must never call report() (cpu node = single leaderboard writer) ---------------

def _fake_run(monkeypatch, tmp_path):
    import run_aasist as R
    rng = np.random.default_rng(0)
    n = 40
    lab = np.where(np.arange(n) % 3 == 0, "spoof", "bonafide")
    m = pd.DataFrame({"path": [f"x/{i}.wav" for i in range(n)], "label": lab,
                      "split": np.where(np.arange(n) < 20, "val", "test"),
                      "val_testlike": np.arange(n) < 10, "test_internal_testlike": np.arange(n) >= 20})
    calls = []

    class Dummy:
        def to(self, *a, **k):
            return self

    monkeypatch.setattr(R, "manifest", lambda: m)
    monkeypatch.setattr(R, "SCORES", tmp_path)
    monkeypatch.setattr(R, "load", lambda p: (np.zeros(SR, np.float32), SR))
    monkeypatch.setattr(R.aasist_wrap, "load", lambda *a, **k: Dummy())
    monkeypatch.setattr(R.aasist_wrap, "score_batch", lambda model, c, w: rng.standard_normal(len(c)))
    monkeypatch.setattr(R, "report", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(E, "LEADERBOARD", tmp_path / "leaderboard.md")
    return R, m, calls


def test_run_aasist_default_is_gpu_safe(monkeypatch, tmp_path, capsys):
    R, m, calls = _fake_run(monkeypatch, tmp_path)
    R.main(["--name", "T", "--device", "cpu", "--threads", str(torch.get_num_threads())])
    assert calls == [] and not (tmp_path / "leaderboard.md").exists()
    out = pd.read_parquet(tmp_path / "T.parquet")
    assert list(out.columns) == ["path", "score"] and set(out.path) == set(m.path)  # what report() would write
    printed = capsys.readouterr().out
    for s in ("official", "brief", "combined", "val_testlike", "test_internal_testlike"):
        assert s in printed


def test_run_aasist_report_flag_calls_report(monkeypatch, tmp_path):
    R, _, calls = _fake_run(monkeypatch, tmp_path)
    R.main(["--name", "T", "--device", "cpu", "--threads", str(torch.get_num_threads()), "--report"])
    assert len(calls) == 1 and calls[0][0] == "T"
