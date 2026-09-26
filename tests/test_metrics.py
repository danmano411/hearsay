"""hearsay.metrics must equal the organizers' evaluation.py (third_party/asvspoof5/evaluation-package)."""
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hearsay.audio import ROOT
from hearsay.metrics import INTERPRETATIONS, act_dcf, all_metrics, cllr, eer, min_dcf

EVAL_DIR = ROOT / "third_party/asvspoof5/evaluation-package"  # third_party is untracked; lives in the main checkout
needs_scorer = pytest.mark.skipif(not EVAL_DIR.exists(), reason="organizers' scorer not found")
TOL = 1e-6


def run_official(score_file, key_file, cwd):
    """Run the unmodified organizers' script; parse the full-precision track1_result.txt it writes to cwd."""
    subprocess.run([sys.executable, str(EVAL_DIR / "evaluation.py"), "--m", "t1", "--cm", str(score_file),
                    "--cm_keys", str(key_file)], cwd=cwd, check=True, capture_output=True)
    txt = (Path(cwd) / "track1_result.txt").read_text()
    num = lambda name: float(re.search(name + r"\s*=\s*([-\d.eE+]+)", txt).group(1))
    return {"min_dcf": num("min DCF"), "eer": num("EER") / 100, "cllr": num("CLLR"), "act_dcf": num("actDCF")}


def official_brief(bona, spoof):
    """Organizers' own compute_det_curve/compute_mindcf with the brief's swapped costs (Cmiss=4, Cfa=1)."""
    sys.path.insert(0, str(EVAL_DIR))
    try:
        import calculate_modules as cm
    finally:
        sys.path.remove(str(EVAL_DIR))
    frr, far, thr = cm.compute_det_curve(bona, spoof)
    return cm.compute_mindcf(frr, far, thr, 0.5, 4, 1)[0]


def check(ours_fake_scores, labels, bona_scores_written, tmp_path, higher_is_fake=True):
    names = [f"E_{i:06d}" for i in range(len(labels))]
    sf, kf = tmp_path / "scores.tsv", tmp_path / "key.tsv"
    pd.DataFrame({"filename": names, "cm-score": bona_scores_written}).to_csv(sf, sep="\t", index=False,
                                                                             float_format="%.17g")
    pd.DataFrame({"filename": names, "cm-label": labels}).to_csv(kf, sep="\t", index=False)
    ref = run_official(sf, kf, tmp_path)
    kw = {"higher_is_fake": higher_is_fake}
    assert abs(min_dcf(ours_fake_scores, labels, **kw) - ref["min_dcf"]) < TOL
    assert abs(eer(ours_fake_scores, labels, **kw) - ref["eer"]) < TOL
    assert abs(cllr(ours_fake_scores, labels, **kw) - ref["cllr"]) < TOL
    assert abs(act_dcf(ours_fake_scores, labels, **kw) - ref["act_dcf"]) < TOL
    lab = np.asarray(labels)
    bona_w = np.asarray(bona_scores_written, dtype=float)
    ref_brief = official_brief(bona_w[lab == "bonafide"], bona_w[lab == "spoof"])
    assert abs(min_dcf(ours_fake_scores, labels, "brief_as_written", **kw) - ref_brief) < TOL


@needs_scorer
def test_bundled_files(tmp_path):
    for f in ("test_cm_file", "test_cm_file2"):
        s = pd.read_csv(EVAL_DIR / f, sep="\t").set_index("filename")
        k = pd.read_csv(EVAL_DIR / "test_cm_key", sep="\t").set_index("filename")
        df = s.join(k)  # bundled files are in ASVspoof direction already
        check(df["cm-score"].to_numpy(), df["cm-label"].to_numpy(), df["cm-score"].to_numpy(), tmp_path,
              higher_is_fake=False)


@needs_scorer
@pytest.mark.parametrize("case", ["gauss_70_30", "heavy_ties", "uniform_small", "separable"])
def test_random_sets(tmp_path, case):
    rng = np.random.default_rng(sum(map(ord, case)))
    n = {"gauss_70_30": 1671, "heavy_ties": 400, "uniform_small": 37, "separable": 200}[case]
    is_fake = rng.random(n) < 0.3
    is_fake[:2] = [True, False]
    if case == "gauss_70_30":
        s = 1 / (1 + np.exp(-(rng.normal(size=n) + 1.5 * is_fake)))
    elif case == "heavy_ties":
        s = np.round(np.clip(rng.normal(0.4, 0.2, n) + 0.2 * is_fake, 0, 1), 1)
    elif case == "uniform_small":
        s = rng.random(n)
    else:
        s = np.where(is_fake, 0.6, 0.1) + 0.3 * rng.random(n)
    labels = np.where(is_fake, "spoof", "bonafide")
    check(s, labels, 1.0 - s, tmp_path)  # official_as_written = organizers score 1 - s


def test_monotone_invariance_and_trivial():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.3
    s = 1 / (1 + np.exp(-(rng.normal(size=5000) + 2 * y)))
    for k in [*INTERPRETATIONS, "combined"]:
        base = min_dcf(s, y, k)
        for t in (s ** 3, 0.2 + 0.5 * s, np.sqrt(s)):
            assert abs(min_dcf(t, y, k) - base) < 1e-12
        assert min_dcf(np.full(5000, 0.006), y, k) == 1.0  # all-constant submission
        assert min_dcf(y.astype(float), y, k) == 0.0  # perfect
    assert min_dcf(1 - y.astype(float), y) == 1.0  # perfectly inverted is as bad as constant


def test_fast_100k():
    rng = np.random.default_rng(1)
    y = rng.random(100_000) < 0.3
    t0 = time.perf_counter()
    all_metrics(rng.random(100_000), y)
    assert time.perf_counter() - t0 < 2.0


def test_submission_validator(tmp_path):
    spec = importlib.util.spec_from_file_location("score", Path(__file__).resolve().parents[1] / "scripts/score.py")
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    tmpl, sub = tmp_path / "t.tsv", tmp_path / "s.tsv"
    tmpl.write_bytes(b"filename\tcm-score\na.wav\t0.006\nb.wav\t0.006\nc.wav\t0.006\n")
    sub.write_bytes(b"filename\tcm-score\na.wav\t0.1\nb.wav\t0.9\nc.wav\t0.5\n")
    assert score.validate(sub, tmpl) == ([], [])
    errors, warnings = score.validate(tmpl, tmpl)
    assert errors == [] and len(warnings) == 2  # untouched defaults + constant file
    for bad in (b"filename,cm-score\na.wav\t0.1\nb.wav\t0.9\nc.wav\t0.5\n",  # header
                b"filename\tcm-score\nb.wav\t0.1\na.wav\t0.9\nc.wav\t0.5\n",  # order
                b"filename\tcm-score\na.wav\t0.1\nb.wav\t0.9\n",  # count
                b"filename\tcm-score\na.wav\t0.1\nb.wav\tnan\nc.wav\t0.5\n",  # NaN
                b"filename\tcm-score\na.wav\t0.1\nb.wav\t1.2\nc.wav\t0.5\n",  # range
                b"filename\tcm-score\na.wav 0.1\nb.wav\t0.9\nc.wav\t0.5\n",  # not TAB
                b"\xef\xbb\xbffilename\tcm-score\na.wav\t0.1\nb.wav\t0.9\nc.wav\t0.5\n"):  # UTF-8 BOM
        sub.write_bytes(bad)
        assert score.validate(sub, tmpl)[0], bad
