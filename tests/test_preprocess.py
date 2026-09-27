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


def test_rawboost_keeps_shape_and_changes_signal():
    from hearsay.preprocess import augment, rawboost
    rng = np.random.default_rng(0)
    y = (0.1 * np.sin(2 * np.pi * 220 * np.arange(48000) / 16000)).astype(np.float32)
    for seed in range(10):  # every algorithm gets exercised
        z = rawboost(y, np.random.default_rng(seed))
        assert z.shape == y.shape and z.dtype == np.float32 and np.isfinite(z).all()
        assert not np.allclose(z / (np.abs(z).max() + 1e-9), y / np.abs(y).max())
    a = augment(y, np.random.default_rng(3), 1.0)
    b = augment(y, np.random.default_rng(3), 1.0)
    assert np.array_equal(a, b)  # seeded: reproducible per clip


def test_rawboost_off_is_the_original_recipe():
    from hearsay.preprocess import augment
    y = np.random.default_rng(1).normal(0, 0.1, 32000).astype(np.float32)
    for seed in range(8):
        assert np.array_equal(augment(y, np.random.default_rng(seed)), augment(y, np.random.default_rng(seed), 0.0))
