"""Synthetic vowel with known F0/jitter -> bio features recover it; degenerate inputs never crash."""
import numpy as np
from scipy.signal import lfilter

from hearsay.features.bio import FEATURES, extract_bio

SR = 16000


def vowel(f0=120.0, jitter=0.0, dur=3.0, seed=0):
    """Glottal pulse train (period jitter = relative std) -> 2-pole glottal LPF -> /a/ formant resonators."""
    rng = np.random.default_rng(seed)
    src = np.zeros(int(dur * SR))
    t = 0.0
    while t < len(src) - 1:
        src[int(t)] = 1.0
        t += SR / f0 * (1 + jitter * rng.standard_normal())
    src = lfilter([1], [1, -1.9, 0.9025], src)  # ~-12 dB/oct glottal source spectrum
    for fc, bw in [(700, 80), (1220, 90), (2600, 120), (3300, 150)]:
        r = np.exp(-np.pi * bw / SR)
        src = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / SR), r * r], src)
    src = 0.5 * src / np.abs(src).max() + 1e-4 * rng.standard_normal(len(src))
    return src.astype(np.float32)


def test_f0_and_jitter_ordering():
    lo, hi = extract_bio(vowel(jitter=0.002)), extract_bio(vowel(jitter=0.02))
    f0 = 100 * 2 ** (lo["f0_mean_st"] / 12)
    assert abs(f0 - 120) / 120 < 0.02, f0
    assert hi["jitter_local"] > 2 * lo["jitter_local"], (lo["jitter_local"], hi["jitter_local"])
    assert hi["jitter_rap"] > lo["jitter_rap"]
    assert lo["hnr_db"] > hi["hnr_db"]
    assert lo["voiced_frac"] > 0.9
    # /a/ F1 = 700 Hz; formant tracker should land close
    assert abs(lo["f1_mean"] - 700) < 100, lo["f1_mean"]


def test_degenerate_inputs_do_not_crash():
    rng = np.random.default_rng(0)
    for y in [np.zeros(SR * 3, np.float32), 0.1 * rng.standard_normal(SR * 3).astype(np.float32),
              np.zeros(10, np.float32), np.full(SR, np.nan, np.float32)]:
        f = extract_bio(y)
        assert list(f) == FEATURES
        assert all(isinstance(v, float) for v in f.values())
