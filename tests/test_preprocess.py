import numpy as np
from scipy.signal import welch

from hearsay.preprocess import SR, augment, crop, prep


def test_prep_removes_high_band_dc_and_normalizes():
    rng = np.random.default_rng(0)
    y = (0.3 * rng.standard_normal(3 * SR) + 0.1).astype(np.float32)  # white noise + DC
    out = prep(y)
    f, p = welch(out, SR, nperseg=1024)
    assert p[f > 7600].mean() < 1e-4 * p[(f > 1000) & (f < 6000)].mean()
    assert abs(out.mean()) < 1e-3 and abs(np.sqrt((out**2).mean()) - 0.05) < 1e-3


def test_crop_and_augment_lengths():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(2 * SR).astype(np.float32)
    assert len(crop(y, 4.0)) == 4 * SR and len(crop(y, 1.0, rng)) == SR
    for _ in range(20):
        assert len(augment(y, rng)) == len(y)
