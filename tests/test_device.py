import pytest
import torch

from hearsay.device import resolve


def test_resolve():
    assert resolve("cpu") == torch.device("cpu")
    assert resolve("auto") == torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit):
            resolve("cuda")


def test_resolve_caps_memory_with_an_indexed_device(monkeypatch):
    """--device cuda / auto give torch.device('cuda') (no index); the cap needs an index. No real CUDA call here."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: calls.append((f, d)))
    assert resolve("cuda") == torch.device("cuda")
    assert resolve("cuda:1") == torch.device("cuda:1")
    assert calls == [(0.92, 0), (0.92, 1)]
