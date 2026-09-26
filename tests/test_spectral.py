import numpy as np

from hearsay.features.spectral import FEATURES, extract_spectral
from hearsay.preprocess import SR, prep


def test_fixed_keys_finite_and_blind_above_7khz():
    rng = np.random.default_rng(0)
    t = np.arange(3 * SR) / SR
    y = (0.1 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    a = extract_spectral(prep(y))
    assert list(a) == FEATURES and np.isfinite(list(a.values())).all()
    # energy in the 7.5-8 kHz shortcut band must not move the features (pre-prep, like a native-16k real clip)
    b = extract_spectral(prep(y + 0.05 * np.sin(2 * np.pi * 7900 * t).astype(np.float32)))
    rel = np.abs(np.array(list(a.values())) - np.array(list(b.values()))) / (np.abs(list(a.values())) + 1e-3)
    assert np.median(rel) < 0.01 and rel.max() < 0.2, rel.max()
    assert np.isfinite(list(extract_spectral(np.zeros(SR, np.float32)).values())).all()
