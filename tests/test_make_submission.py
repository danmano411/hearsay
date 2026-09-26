import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_submission import DEFAULT, build  # noqa: E402


@pytest.fixture
def template(tmp_path):
    t = tmp_path / "template.csv"
    t.write_text("filename\tcm-score\n" + "".join(f"HGT{i}.wav\t0.006\n" for i in range(5)))
    return t


def test_template_order_and_values(template):
    s = pd.Series([0.9, 0.1, 0.5, 0.2, 0.7], index=[f"HGT{i}.wav" for i in (4, 3, 2, 1, 0)])
    names, vals, n = build(s, template)
    assert names == [f"HGT{i}.wav" for i in range(5)] and n == 0
    assert list(vals) == [0.7, 0.2, 0.5, 0.1, 0.9]


def test_filename_mismatch_is_an_error_not_all_defaults(template):
    s = pd.Series(np.linspace(0, 1, 5), index=[f"HGT{i}" for i in range(5)])  # no .wav
    with pytest.raises(ValueError, match="not in the template"):
        build(s, template)


def test_unscored_rows_and_sigmoid(template):
    s = pd.Series([0.9, 0.1], index=["HGT0.wav", "HGT1.wav"])
    with pytest.raises(ValueError, match="3 template rows unscored"):
        build(s, template)
    _, vals, n = build(s, template, max_unscored=3)
    assert n == 3 and list(vals[2:]) == [DEFAULT] * 3
    logits = pd.Series([2.0, -3.0], index=["HGT0.wav", "HGT1.wav"])
    with pytest.raises(ValueError, match="--sigmoid"):
        build(logits, template, max_unscored=3)
    _, vals, _ = build(logits, template, sigmoid=True, max_unscored=3)
    assert vals[0] > 0.5 > vals[1]
    big = pd.Series([40.0, 45.0], index=["HGT0.wav", "HGT1.wav"])  # saturates to 1.0 at T=1 -> a tie
    _, v1, _ = build(big, template, sigmoid=True, max_unscored=3)
    _, v8, _ = build(big, template, sigmoid=True, max_unscored=3, temperature=8)
    assert float(f"{v1[0]:.10g}") == float(f"{v1[1]:.10g}") and float(f"{v8[0]:.10g}") < float(f"{v8[1]:.10g}")
