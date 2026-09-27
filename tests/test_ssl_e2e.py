"""R4ft: model, sampler, data, training/resume, scoring rules. CPU only, tiny random-init
front ends (no downloads), synthetic wavs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from hearsay import evaluate as E
from hearsay.models import ssl_e2e as S
from hearsay.models import ssl_e2e_data as D
from hearsay.preprocess import SR, crop, prep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

X = torch.randn(4, S.N_SAMP, generator=torch.Generator().manual_seed(0)) * 0.05


def tiny(backend="light", K=3, freeze=0, ckpt=True, mask=0.05, name="tiny", seed=0):
    torch.manual_seed(seed)
    return S.build_model(name, backend, max_blocks=K, freeze_blocks=freeze, grad_ckpt=ckpt, mask_time_prob=mask)


# --- model ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tiny", "tiny_wavlm", "tiny_postln", "tiny_wavlm_postln"])
def test_truncation_equals_full_model_blocks(name):
    from transformers import AutoModel
    torch.manual_seed(0)
    kind, pre_ln = S.TINY[name]
    full = AutoModel.from_config(S.tiny_config(kind, pre_ln=pre_ln), attn_implementation="eager").eval()
    fe = S.SSLFrontEnd(name, max_blocks=3, grad_ckpt=False, mask_time_prob=0.0, attn="eager")
    fe.ssl.load_state_dict(full.state_dict(), strict=False)
    fe.eval()
    x = X[:2]
    xn = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-7)
    with torch.no_grad():
        ref = full(xn, output_hidden_states=True).hidden_states
        hs = fe(x)
    assert fe.n_blocks == 3 and len(hs) == 4 and len(ref) == 25
    for a, b in zip(hs, ref[:4]):
        torch.testing.assert_close(a, b)
    if pre_ln:  # pre-LN: the final layer_norm (applied after the last block) is dropped
        assert isinstance(fe.ssl.encoder.layer_norm, torch.nn.Identity)
    else:  # post-LN: encoder.layer_norm normalizes the INPUT of block 1 and must be kept, at block 1's LR
        assert isinstance(fe.ssl.encoder.layer_norm, torch.nn.LayerNorm)
        assert fe.block_of("encoder.layer_norm.weight") == 0


def test_post_ln_front_end_trains_with_freezing():
    model = tiny("light", K=3, freeze=1, name="tiny_postln").train()
    model(X[:2]).sum().backward()
    ln = model.frontend.ssl.encoder.layer_norm.weight
    assert not ln.requires_grad and ln.grad is None  # below the frozen block 1
    model = tiny("aasist", K=3, freeze=0, name="tiny_postln").train()
    model(X[:2]).sum().backward()
    ln = model.frontend.ssl.encoder.layer_norm.weight
    assert ln.grad is not None
    g = next(g for g in S.param_groups(model, lr=1.0, decay=0.5) if any(p is ln for p in g["params"]))
    assert g["lr"] == pytest.approx(0.5 ** 2) and g["weight_decay"] == 0  # block 1 of K = 3, no decay on norms


@pytest.mark.parametrize("backend", ["light", "aasist"])
def test_forward_backward_shapes_and_freezing(backend):
    model = tiny(backend, K=3, freeze=1).train()
    out = model(X)
    assert out.shape == (4,) and out.dtype == torch.float32
    F.binary_cross_entropy_with_logits(out, torch.tensor([0.0, 1, 0, 1])).backward()
    fe = model.frontend
    for n, p in fe.ssl.named_parameters():
        b = fe.block_of(n)
        if b is None:  # conv feature encoder: frozen, run under no_grad
            assert not p.requires_grad and p.grad is None, n
        elif b <= 1:  # lowest block + everything below it frozen
            assert not p.requires_grad and p.grad is None, n
        else:
            assert p.requires_grad, n
    assert all(p.grad is not None for n, p in fe.ssl.named_parameters() if p.requires_grad
               and "masked_spec_embed" not in n)
    # upstream AASIST Residual_block computes bn1(x) but convolves x (its output is discarded): no grad, by design
    assert all(p.grad is not None for n, p in model.backend.named_parameters() if ".bn1." not in n)
    groups = S.param_groups(model, lr=2e-5, decay=0.85, backend_lr=1e-3)
    grouped = {id(p) for g in groups for p in g["params"]}
    assert grouped == {id(p) for p in model.parameters() if p.requires_grad}  # frozen params excluded
    lrs = {g["name"]: g["lr"] for g in groups}
    assert lrs["front3"] == pytest.approx(2e-5) and lrs["front2"] == pytest.approx(2e-5 * 0.85)
    assert lrs["back"] == 1e-3 and all(g["weight_decay"] == 0 for g in groups if g["name"].endswith("_nd"))


def test_bottom_params_get_block1_lr_and_conv_has_no_grad():
    model = tiny(K=4, freeze=0)
    model(X[:2]).sum().backward()
    assert all(p.grad is None for p in model.frontend.ssl.feature_extractor.parameters())
    groups = S.param_groups(model, lr=1.0, decay=0.5)
    proj = model.frontend.ssl.feature_projection.projection.weight
    g = next(g for g in groups if any(p is proj for p in g["params"]))
    assert g["lr"] == pytest.approx(0.5 ** 3)  # block 1 of K = 4


def test_backend_warmup_leaves_frontend_without_grad():
    model = tiny("light", K=2)
    model(X[:2], frontend_grad=False).sum().backward()
    assert all(p.grad is None for p in model.frontend.parameters())
    assert all(p.grad is not None for p in model.backend.parameters())


@pytest.mark.parametrize("backend", ["light", "aasist"])
def test_gradient_checkpointing_matches_plain(backend):
    res = []
    for ckpt in (False, True):
        model = tiny(backend, K=3, ckpt=ckpt, mask=0.0).train()
        torch.manual_seed(5)  # same dropout masks (non-reentrant checkpointing replays the RNG state)
        loss = F.binary_cross_entropy_with_logits(model(X), torch.tensor([0.0, 1, 0, 1]))
        loss.backward()
        res.append((loss.item(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    assert res[0][0] == pytest.approx(res[1][0], rel=1e-5)
    assert res[0][1].keys() == res[1][1].keys() and len(res[0][1]) > 20
    for n, g in res[0][1].items():
        torch.testing.assert_close(res[1][1][n], g, rtol=1e-4, atol=1e-6, msg=n)


# --- sampler -------------------------------------------------------------------------------------------------------

def fake_train_df():
    rows = []
    for src, n in [("big", 400), ("asvspoof5", 100), ("small", 25)]:
        rows += [{"path": f"{src}/r{i}.wav", "label": "bonafide", "source": src, "generator": "bonafide"}
                 for i in range(n)]
    for src, gens in [("diffssd", {"g1": 900, "g2": 100}), ("asvspoof5", {"a1": 64, "a2": 16}), ("tiny", {"t": 9})]:
        for g, n in gens.items():
            rows += [{"path": f"{src}/{g}_{i}.wav", "label": "spoof", "source": src, "generator": g} for i in range(n)]
    df = pd.DataFrame(rows)
    df["split"] = "train"
    return df


def test_sampler_proportions_follow_plan_formula():
    df = fake_train_df()
    t = D.cell_table(df).set_index(["label", "source"])
    w_b = {"big": 20.0, "asvspoof5": 2 * 10.0, "small": 5.0}  # sqrt(n), asvspoof5 bonafide x2
    w_s = {"diffssd": np.sqrt(1000), "asvspoof5": np.sqrt(80), "tiny": 3.0}
    for lab, w in (("bonafide", w_b), ("spoof", w_s)):
        tot = sum(w.values())
        for s, v in w.items():
            assert t.loc[(lab, s), "share"] == pytest.approx(v / tot)
    smp = D.BalancedBatches(df, per_class=4, seed=1)
    rows = [r for b in range(4000) for r, _ in smp.batch(b)]
    drawn = df.iloc[rows]
    assert (drawn.label.to_numpy().reshape(-1, 8)[:, :4] == "bonafide").all()  # 4 real + 4 fake per micro-batch
    assert (drawn.label.to_numpy().reshape(-1, 8)[:, 4:] == "spoof").all()
    for lab, w in (("bonafide", w_b), ("spoof", w_s)):
        got = drawn[drawn.label == lab].source.value_counts(normalize=True)
        tot = sum(w.values())
        for s, v in w.items():
            assert got[s] == pytest.approx(v / tot, abs=0.015)
    gens = drawn[drawn.source == "diffssd"].generator.value_counts(normalize=True)
    assert gens["g2"] == pytest.approx(np.sqrt(100) / (np.sqrt(900) + np.sqrt(100)), abs=0.02)  # generators ∝ √n


def test_sampler_never_yields_non_train_rows():
    df = fake_train_df()
    df.loc[df.index[::7], "split"] = "val"
    with pytest.raises(ValueError):
        D.BalancedBatches(df)
    tr = df[df.split == "train"].reset_index(drop=True)
    smp = D.BalancedBatches(tr, seed=3)
    rows = {r for b in range(500) for r, _ in smp.batch(b)}
    assert (tr.iloc[sorted(rows)].split == "train").all()
    ds = D.TrainSet(df, "/nonexistent")  # the per-row assertion in the Dataset is the last line of defence
    with pytest.raises(ValueError, match="non-train"):
        ds[(0, 0)]


def test_sampler_state_round_trips():
    df = fake_train_df()
    a = D.BalancedBatches(df, per_class=4, seed=7, up={("small", "bonafide"): 3.0}, power=0.4)
    it = iter(a)
    first = [next(it) for _ in range(6)]
    st = json.loads(json.dumps(a.state_dict(next_batch=3)))  # survives serialization into a checkpoint
    b = D.BalancedBatches.from_state(df, st)
    it2 = iter(b)
    assert [next(it2) for _ in range(3)] == first[3:]
    ids = [sid for batch in first for _, sid in batch]
    assert ids == list(range(48))  # unique per-sample augmentation ids


# --- data ----------------------------------------------------------------------------------------------------------

def hf_fraction(x):
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return p[f > 7500].sum() / p.sum()


def write_wavs(root, specs, seed=0):
    rng = np.random.default_rng(seed)
    for rel, sec in specs:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(root / rel, (rng.standard_normal(int(sec * SR)) * 0.1).astype(np.float32), SR, subtype="PCM_16")


def test_dataset_items_are_prepped(tmp_path):
    write_wavs(tmp_path, [("a.wav", 3.2), ("b.wav", 9.5)])
    df = pd.DataFrame({"path": ["a.wav", "b.wav"], "label": ["bonafide", "spoof"], "split": "train"})
    ds = D.TrainSet(df, tmp_path, seed=1)
    for row in (0, 1):
        x, y = ds[(row, 10 + row)]
        assert x.shape == (S.N_SAMP,) and x.dtype == torch.float32 and float(y) == row  # 1 = spoof
        assert hf_fraction(x.numpy()) < 1e-3 and 0.01 < float(x.pow(2).mean().sqrt()) < 0.2
    torch.testing.assert_close(ds[(1, 5)][0], ds[(1, 5)][0])
    ev = D.EvalSet(["a.wav", "b.wav"], tmp_path, max_windows=1)
    yb = sf.read(tmp_path / "b.wav", dtype="float32")[0]
    np.testing.assert_allclose(ev[1][0].numpy()[0], crop(prep(yb), 4.0), atol=1e-5)  # prep(full clip) -> centre 4 s
    assert ev[0][0].shape == (1, S.N_SAMP)  # 3.2 s clip tiled
    w3 = D.EvalSet(["a.wav", "b.wav"], tmp_path, max_windows=3)
    assert w3[0][0].shape == (1, S.N_SAMP) and w3[1][0].shape == (3, S.N_SAMP)
    mm = D.center_cache(["a.wav", "b.wav"], tmp_path, tmp_path / "cache", workers=0)
    np.testing.assert_allclose(mm[1], ev[1][0].numpy()[0])
    assert D.center_cache(["a.wav", "b.wav"], tmp_path, tmp_path / "cache", workers=0).filename == mm.filename


# --- training script -----------------------------------------------------------------------------------------------

@pytest.fixture()
def world(tmp_path, monkeypatch):
    """Synthetic repo root: train / val / val_testlike / test_internal_testlike rows + 3 HGT clips."""
    import finetune_ssl as FT
    rows, specs = [], []
    rng = np.random.default_rng(0)
    for i in range(24):
        lab = ("bonafide", "spoof")[i % 2]
        split = "train" if i < 12 else ("val" if i < 18 else "test_internal")
        rel = f"data/processed/x/c{i}.wav"
        specs.append((rel, float(rng.uniform(3, 9))))
        rows.append({"path": rel, "label": lab, "source": f"s{i % 3}", "generator": "bonafide" if lab == "bonafide"
                     else f"g{i % 4}", "split": split, "val_testlike": 12 <= i < 16,
                     "test_internal_testlike": i >= 18})
    write_wavs(tmp_path, specs)
    write_wavs(tmp_path, [(f"data/raw/hgt_test/HGT{i}.wav", 3.5) for i in range(3)], seed=1)
    (tmp_path / "data/raw/hgt_test" / FT.TEMPLATE_NAME).write_text(
        "filename\tcm-score\n" + "".join(f"HGT{i}.wav\t0.006\n" for i in (2, 0, 1)))
    monkeypatch.setattr(FT, "N_HGT", 3)
    m = pd.DataFrame(rows)
    monkeypatch.setattr(FT, "manifest", lambda: m)
    monkeypatch.setattr(FT, "ROOT", tmp_path)
    monkeypatch.setattr(FT, "HGT_DIR", tmp_path / "data" / "raw" / "hgt_test")
    calls = []

    class K32:
        def SetThreadExecutionState(self, flags):
            calls.append(flags)
            return 0

    monkeypatch.setattr(FT, "_kernel32", lambda: K32())

    def no_report(*a, **k):
        raise AssertionError("scoring must never call report()")
    monkeypatch.setattr(E, "report", no_report)
    monkeypatch.setattr(E, "LEADERBOARD", tmp_path / "leaderboard.md")
    return FT, m, tmp_path, calls


def train_args(out, name, steps, *extra):
    return ["train", "--tiny", "--device", "cpu", "--out", str(out), "--name", name, "--max-steps", str(steps),
            "--micro-batch", "2", "--accum", "2", "--max-blocks", "2", "--warmup-backend", "1", "--warmup-front", "1",
            "--eval-every", "1000", "--log-every", "1", "--workers", "0", "--threads", str(torch.get_num_threads()),
            *extra]


def step_losses(run):
    recs = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    return {r["step"]: r["step_loss"] for r in recs if r["type"] == "train"}, recs


def test_train_resume_reproduces_uninterrupted_run(world):
    FT, m, root, calls = world
    out = root / "runs"
    FT.main(train_args(out, "full", 3, "--eval-every", "3"))
    FT.main(train_args(out, "split", 2, "--eval-every", "3"))  # pause at step 2: saves, no unscheduled eval
    FT.main(train_args(out, "split", 3, "--eval-every", "3"))  # resumes from last.pth at step 2
    full, recs = step_losses(out / "full")
    split, recs2 = step_losses(out / "split")
    assert sorted(full) == [1, 2, 3] and sorted(split) == [1, 2, 3]
    for s in (1, 2, 3):
        assert split[s] == pytest.approx(full[s], rel=1e-6, abs=1e-7), s
    ev = [r for r in recs2 if r["type"] == "eval"]
    assert [r["step"] for r in ev] == [1, 3]  # warm-up end + scheduled; the pause at step 2 did not evaluate
    assert [r["step"] for r in recs if r["type"] == "eval"] == [1, 3]
    # paused + resumed == uninterrupted: same best.pth, same best / patience state, same eval numbers
    ck_f = torch.load(out / "full" / "last.pth", map_location="cpu", weights_only=False)
    ck_s = torch.load(out / "split" / "last.pth", map_location="cpu", weights_only=False)
    for k in ("best", "bad", "evals", "step", "clips"):
        assert ck_s["state"][k] == ck_f["state"][k], k
    bf, bs = (torch.load(out / r / "best.pth", map_location="cpu") for r in ("full", "split"))
    assert bf.keys() == bs.keys()
    for k in bf:
        torch.testing.assert_close(bs[k], bf[k], msg=k)
    ev_f = [r for r in recs if r["type"] == "eval"]
    for a_, b_ in zip(ev, ev_f):
        assert a_["combined"] == pytest.approx(b_["combined"]) and a_["train_loss"] == pytest.approx(b_["train_loss"])
    for k in ("official_as_written", "brief_as_written", "combined", "eer", "clips", "train_loss", "slices"):
        assert k in ev[0]
    ck = torch.load(out / "split" / "last.pth", map_location="cpu", weights_only=False)
    assert ck["state"]["step"] == 3 and ck["sampler"]["next_batch"] == 6 and {"torch", "numpy", "python"} <= set(ck["rng"])
    assert (out / "split" / "best.pth").exists() and (out / "cache").is_dir()
    # keep-awake: set for each training call, always cleared again (ES_CONTINUOUS alone)
    assert calls == [FT.ES_CONTINUOUS | FT.ES_SYSTEM_REQUIRED, FT.ES_CONTINUOUS] * 3


def test_resume_refuses_a_changed_train_split(world, monkeypatch):
    FT, m, root, _ = world
    out = root / "runs"
    FT.main(train_args(out, "sync", 1))
    m2 = m.drop(index=3)  # a manifest SYNC between legs removed a train row
    monkeypatch.setattr(FT, "manifest", lambda: m2)
    with pytest.raises(SystemExit, match="different train split"):
        FT.main(train_args(out, "sync", 2))


class FakeOOMModel(torch.nn.Module):
    """Raises CUDA-style OOM once for chunks above `limit` windows; logit = window mean."""

    def __init__(self, limit):
        super().__init__()
        self.limit, self.sizes, self.raised = limit, [], 0

    def forward(self, x):
        self.sizes.append(len(x))
        if len(x) > self.limit and not self.raised:
            self.raised += 1
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return x.mean(1)


def test_eval_oom_halves_batch_and_retries():
    import types

    import finetune_ssl as FT
    a = types.SimpleNamespace(eval_batch=8, amp="off")
    x = np.random.default_rng(0).standard_normal((20, 50)).astype(np.float32)
    logs = []
    model = FakeOOMModel(limit=4)
    s = FT.logits_of(model, x, torch.device("cpu"), a, log=logs.append)
    np.testing.assert_allclose(s, x.mean(1), rtol=1e-6)  # every window scored once, in order
    assert model.sizes[0] == 8 and model.sizes[1] == 4 and max(model.sizes[1:]) == 4
    assert a.eval_batch == 4 and len(logs) == 1 and "halved to 4" in logs[0]
    a.eval_batch = 1
    with pytest.raises(torch.OutOfMemoryError):  # nothing left to halve
        FT.forward_windows(FakeOOMModel(limit=0), x, torch.device("cpu"), a, log=logs.append)


def test_score_batches_by_windows_not_clips(world):
    import types
    FT, m, root, _ = world
    paths = m.path.tolist()[:10]  # 3-9 s clips -> up to 3 windows each
    a = types.SimpleNamespace(eval_batch=5, amp="off", workers=0)
    model = FakeOOMModel(limit=10**9)
    s = FT.score_paths(model, paths, 3, torch.device("cpu"), a)
    assert len(s) == 10 and np.isfinite(s).all() and max(model.sizes) <= 5


def test_keep_awake_restored_on_error(monkeypatch):
    import finetune_ssl as FT
    calls = []

    class K32:
        def SetThreadExecutionState(self, flags):
            calls.append(flags)
            return 0
    monkeypatch.setattr(FT, "_kernel32", lambda: K32())
    with pytest.raises(RuntimeError):
        with FT.keep_awake():
            assert calls == [FT.ES_CONTINUOUS | FT.ES_SYSTEM_REQUIRED]
            raise RuntimeError("boom")
    assert calls == [FT.ES_CONTINUOUS | FT.ES_SYSTEM_REQUIRED, FT.ES_CONTINUOUS]


def test_nan_loss_restores_and_halves_lr_then_stops(world, monkeypatch):
    FT, m, root, _ = world
    out = root / "runs"
    real = F.binary_cross_entropy_with_logits
    monkeypatch.setattr(FT.F, "binary_cross_entropy_with_logits", lambda *a, **k: real(*a, **k) * float("nan"))
    st = FT.main(train_args(out, "nan", 3))
    recs = [json.loads(line) for line in (out / "nan" / "log.jsonl").read_text().splitlines()]
    nan = [r for r in recs if r["type"] == "nan"]
    assert nan[0]["action"] == "restore" and nan[0]["lr_scale"] == 0.5 and nan[1]["action"] == "stop"
    assert st["stop"].startswith("non-finite")


def test_score_writes_config_first_and_never_reports(world, monkeypatch, capsys):
    FT, m, root, _ = world
    out, scores = root / "runs", root / "scores"
    FT.main(train_args(out, "sc", 2))
    run = out / "sc"
    order = []
    real_score_paths = FT.score_paths

    def spy(model, paths, k, *a, **kw):
        order.append((paths[0], (run / "config.json").exists()))
        assert not model.training  # inference in eval() mode
        return real_score_paths(model, paths, k, *a, **kw)

    def hgt():
        assert (run / "config.json").exists(), "HGT read before config.json"
        return sorted(p.relative_to(root).as_posix() for p in (root / "data/raw/hgt_test").glob("*.wav"))

    monkeypatch.setattr(FT, "score_paths", spy)
    monkeypatch.setattr(FT, "hgt_paths", hgt)
    assert not hasattr(FT, "report")
    res, cfg = FT.main(["--score", "--tiny", "--device", "cpu", "--out", str(out), "--name", "sc", "--scores-dir",
                        str(scores), "--hgt", "--workers", "0"])
    vt = set(m[m.val_testlike].path)
    # val_testlike (both inference modes) scored before config.json; everything else after it
    assert [c for p, c in order if p in vt] == [False, False]
    assert [c for p, c in order if p not in vt] == [True, True]
    assert cfg["sha256"] == FT.sha256(run / "best.pth") and cfg["inference"] in ("center", "win3")
    assert json.loads((run / "config.json").read_text())["sha256"] == cfg["sha256"]
    df = pd.read_parquet(scores / "sc.parquet")
    want = m[(m.split == "val") | m.val_testlike | m.test_internal_testlike].path
    assert list(df.columns) == ["path", "score"] and set(df.path) == set(want) and np.isfinite(df.score).all()
    hg = pd.read_parquet(scores / "sc__hgt.parquet")
    assert list(hg.columns) == ["filename", "score"] and sorted(hg.filename) == ["HGT0.wav", "HGT1.wav", "HGT2.wav"]
    assert not (root / "leaderboard.md").exists()
    printed = capsys.readouterr().out
    assert "wrote:" in printed and "val_testlike" in printed and "combined" in printed
    assert cfg["selected_on"] == "val_testlike combined minDCF" and cfg["forced"] is False


def score_args(out, scores, name, *extra):
    return ["--score", "--tiny", "--device", "cpu", "--out", str(out), "--name", name, "--scores-dir", str(scores),
            "--workers", "0", *extra]


def test_score_refuses_to_overwrite_config_and_records_manual_choice(world):
    FT, m, root, _ = world
    out, scores = root / "runs", root / "scores"
    FT.main(train_args(out, "cf", 2))
    _, cfg = FT.main(score_args(out, scores, "cf"))
    FT.main(score_args(out, scores, "cf"))  # same checkpoint + same choice: allowed (idempotent)
    other = {"center": "win3", "win3": "center"}[cfg["inference"]]
    before = (out / "cf" / "config.json").read_text()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        FT.main(score_args(out, scores, "cf", "--windows", other))
    assert (out / "cf" / "config.json").read_text() == before
    torch.save({k: v + 1e-3 for k, v in torch.load(out / "cf" / "best.pth").items()}, out / "cf" / "best.pth")
    with pytest.raises(SystemExit, match="refusing to overwrite"):  # a different checkpoint, same mode
        FT.main(score_args(out, scores, "cf", "--windows", cfg["inference"]))
    _, cfg2 = FT.main(score_args(out, scores, "cf", "--windows", other, "--force"))
    saved = json.loads((out / "cf" / "config.json").read_text())
    assert saved["inference"] == other and saved["forced"] is True and saved["selected_on"].startswith("manual")
    assert saved["overwrite_forced"] is True and list(saved["val_testlike"]) == [other]


def test_manual_windows_on_a_fresh_run_is_recorded(world):
    FT, m, root, _ = world
    out, scores = root / "runs", root / "scores"
    FT.main(train_args(out, "mw", 2))
    _, cfg = FT.main(score_args(out, scores, "mw", "--windows", "win3"))
    assert cfg["inference"] == "win3" and cfg["forced"] is True and cfg["selected_on"].startswith("manual")
    assert cfg["overwrite_forced"] is False


def test_default_up_weights_are_recorded(world):
    """Without --up the sampler applies DEFAULT_UP; args.json and config.json must record those effective weights."""
    FT, m, root, _ = world
    out, scores = root / "runs", root / "scores"
    assert FT.parse(train_args(out, "up", 1)).up == {("asvspoof5", "bonafide"): 2.0}
    FT.main(train_args(out, "up", 1))
    want = [["asvspoof5", "bonafide", 2.0]]
    assert json.loads((out / "up" / "args.json").read_text())["up"] == want
    _, cfg = FT.main(score_args(out, scores, "up"))
    assert cfg["args"]["up"] == want
    assert json.loads((out / "up" / "config.json").read_text())["args"]["up"] == want
    # an explicit --up is still recorded as given
    FT.main(train_args(out, "up2", 1, "--up", "asvspoof2019_la:bonafide=1.5"))
    assert json.loads((out / "up2" / "args.json").read_text())["up"] == [["asvspoof2019_la", "bonafide", 1.5]]


def test_hgt_must_match_the_template(world):
    FT, m, root, _ = world
    out, scores = root / "runs", root / "scores"
    FT.main(train_args(out, "ht", 2))
    (root / "data/raw/hgt_test/HGT1.wav").unlink()
    with pytest.raises(SystemExit, match="does not match the organizers' template"):
        FT.main(score_args(out, scores, "ht", "--hgt"))
    assert (out / "ht" / "config.json").exists() and not (scores / "ht__hgt.parquet").exists()


# --- probe ---------------------------------------------------------------------------------------------------------

def test_vram_probe_cpu_dry_run(tmp_path):
    import vram_probe as VP
    rep = VP.main(["--device", "cpu", "--dry-run", "--configs", "af", "--out", str(tmp_path), "--eval-batch", "6"])
    by = {r["key"]: r for r in rep["results"]}
    assert by["a"]["clips_per_s"] > 0 and by["a"]["ckpt"] is True and not by["a"]["oom"]
    assert by["a"]["eval_batch"] == 6 and by["a"]["eval_ok"] and by["a"]["eval_s"] > 0  # eval forward, states resident
    assert "skipped" in by["f"]  # 8-bit AdamW needs CUDA + bitsandbytes
    assert rep["pick"] == "a" and rep["throughput"]["loader"]["clips_per_s"] > 0
    assert rep["throughput"]["real"]["e2e_clips_per_s"] > 0
    assert json.loads((tmp_path / "probe.json").read_text())["pick"] == rep["pick"]
    assert "| a" in (tmp_path / "probe.md").read_text(encoding="utf-8")


def test_vram_probe_pick_respects_budget():
    import vram_probe as VP
    cap = 7.30  # GiB: the RTX 5050's first real probe
    res = [{"key": "a", "clips_per_s": 35, "peak_alloc_gb": 4.92, "peak_reserved_gb": 6.42, "cap_gb": cap},
           # the real probe's faster config: reserved 94.5 % of the cap is allocator slack, allocated only 67 % -> ok
           {"key": "b", "clips_per_s": 42, "peak_alloc_gb": 4.91, "peak_reserved_gb": 6.90, "cap_gb": cap},
           {"key": "c", "clips_per_s": 60, "peak_alloc_gb": 6.5, "peak_reserved_gb": 6.6, "cap_gb": cap},  # alloc > 85 %
           {"key": "g", "clips_per_s": 70, "peak_alloc_gb": 4.0, "peak_reserved_gb": 7.0, "cap_gb": cap},  # reserved > 95 %
           {"key": "h", "clips_per_s": 90, "peak_alloc_gb": 3.0, "peak_reserved_gb": 3.0, "cap_gb": cap, "spill": True},
           {"key": "d", "clips_per_s": 99, "oom": True, "peak_alloc_gb": 7.3, "peak_reserved_gb": 7.3, "cap_gb": cap},
           {"key": "e", "clips_per_s": 95, "peak_alloc_gb": 3.0, "peak_reserved_gb": 3.0, "cap_gb": cap,
            "eval_ok": False}]  # eval OOM
    key, limit = VP.pick(res)
    assert key == "b" and limit == pytest.approx(0.85 * cap)
    assert VP.pick([dict(r, peak_alloc_gb=None) for r in res])[0] is None  # no allocated number -> never picked
    assert VP.spill_flag([1.0, 0.5, 0.5, 0.5, 2.0]) and not VP.spill_flag([1.0, 0.5, 0.5, 0.6, 0.55])
