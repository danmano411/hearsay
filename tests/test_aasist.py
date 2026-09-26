import numpy as np
import pytest

from hearsay.models import aasist_wrap as A


def test_windows_tile_short_and_split_long():
    short = np.arange(50000, dtype=np.float32)
    w = A.windows(short)
    assert w.shape == (1, A.NB_SAMP) and w[0, 50000] == 0  # tiled from the start, like data_utils.pad
    long = np.arange(3 * A.NB_SAMP, dtype=np.float32)
    w = A.windows(long, max_windows=3)
    assert w.shape == (3, A.NB_SAMP) and w[0, 0] == 0 and w[-1, -1] == len(long) - 1
    assert A.windows(long).shape == (1, A.NB_SAMP)  # default: centre window only


@pytest.mark.skipif(not A.PRETRAINED.exists(), reason="organizers' weights not present")
def test_pretrained_scores_are_finite_per_clip():
    model = A.load()
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(n).astype(np.float32) * 0.05 for n in (48000, 64600, 150000)]
    s = A.score_batch(model, clips, max_windows=2)
    assert s.shape == (3,) and np.isfinite(s).all()
