import pytest
import torch

from hearsay.device import resolve


def test_resolve():
    assert resolve("cpu") == torch.device("cpu")
    assert resolve("auto") == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit):
            resolve("cuda")
