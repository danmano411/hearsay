"""R3: the organizers' pretrained ASVspoof5 AASIST (Baseline-AASIST, MIT license, Tak & Jung / NAVER + EURECOM).

The model code is imported in place from third_party/asvspoof5/Baseline-AASIST (not copied, not modified);
see its LICENSE/NOTICE. Facts this wrapper relies on (checked in main.py / data_utils.py of that repo):
- input: raw 16 kHz waveform, exactly 64600 samples (~4.04 s); `pos_S` is sized for that length.
  Short clips are tiled (data_utils.pad), long clips cropped.
- output: `(last_hidden, logits)`, 2 logits; training labels are 1 = bonafide, 0 = spoof and the
  official CM score is `logits[:, 1]` (higher = bonafide). We return spoof log-odds `logits[:,0] - logits[:,1]`
  (higher = fake), which is what our harness wants.
"""
import importlib.util
import json

import numpy as np
import torch

from hearsay.audio import ROOT

AASIST_DIR = ROOT / "third_party" / "asvspoof5" / "Baseline-AASIST"
PRETRAINED = AASIST_DIR / "models" / "weights" / "AASIST" / "best.pth"
NB_SAMP = 64600


def _model_class():
    spec = importlib.util.spec_from_file_location("asvspoof5_aasist", AASIST_DIR / "models" / "AASIST.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def load(weights=PRETRAINED, device="cpu"):
    conf = json.loads((AASIST_DIR / "config" / "AASIST_ASVspoof5.conf").read_text())
    model = _model_class()(conf["model_config"])
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval().to(device)  # AASIST.py moves its sinc filter bank to x.device itself


def windows(y, max_windows=1):
    """-> (k, 64600). Short clips tile like data_utils.pad; long clips get up to max_windows evenly spaced windows."""
    n = NB_SAMP
    if len(y) <= n:
        return np.tile(y, n // len(y) + 1)[None, :n]
    k = min(max_windows, int(np.ceil(len(y) / n)))
    starts = np.linspace(0, len(y) - n, k).astype(int)
    return np.stack([y[s:s + n] for s in starts])


@torch.no_grad()
def score_batch(model, clips, max_windows=1):
    """clips: list of 1-D float arrays (already prep()'d or raw) -> spoof log-odds per clip (window mean)."""
    wins = [windows(y, max_windows) for y in clips]
    x = torch.from_numpy(np.concatenate(wins).astype(np.float32)).to(next(model.parameters()).device)
    _, logits = model(x)
    s = (logits[:, 0] - logits[:, 1]).cpu().numpy()
    bounds = np.cumsum([0] + [len(w) for w in wins])
    return np.array([s[a:b].mean() for a, b in zip(bounds[:-1], bounds[1:])])
