import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hearsay.features.ssl import center

spec = importlib.util.spec_from_file_location("extract_ssl", Path(__file__).parents[1] / "scripts" / "extract_ssl.py")
extract_ssl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_ssl)


def test_center_crop():
    assert len(center(np.zeros(10 * 16000))) == 4 * 16000
    assert len(center(np.zeros(3 * 16000))) == 3 * 16000


def test_stratified_balances_labels_and_keeps_small_groups():
    df = pd.DataFrame({"label": ["bonafide"] * 1000 + ["spoof"] * 1010,
                       "source": ["a"] * 900 + ["b"] * 100 + ["c"] * 1000 + ["d"] * 10,
                       "generator": ["g"] * 2010})
    s = extract_ssl.stratified(df, 200)
    n = s.groupby("source").size()
    assert abs(len(s) - 200) <= 4 and not s.index.duplicated().any()
    assert n["d"] >= 9 and n["b"] > 100 * 100 / 1000  # sqrt allocation favours the small groups


def test_batching_does_not_change_embeddings():
    pytest.importorskip("transformers")
    from hearsay.features.ssl import SSLEmbedder
    try:
        e = SSLEmbedder("wavlm_base_plus", threads=2)
    except OSError:
        pytest.skip("model not cached")
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal(n).astype(np.float32) * 0.05 for n in (48000, 48016))
    alone, batched = e([a]), e([a, b])  # b is 1 ms longer -> cropped, a must come out identical
    assert alone.shape == (1, 13, 1536)
    assert np.allclose(alone[0], batched[0], atol=1e-3 * np.abs(alone).max())
