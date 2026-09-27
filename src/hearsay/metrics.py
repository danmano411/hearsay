"""Local re-implementation of the organizers' ASVspoof5 Track-1 scorer (see docs/01_challenge_and_scoring.md).

Scores are in OUR submission direction: higher = more likely fake (0 = real, 1 = synthetic).
Labels: 'bonafide'/'spoof' strings, or ints/bools with 1 = spoof (fake), 0 = bonafide (real).

The organizers' scorer (third_party/asvspoof5/evaluation-package, calculate_metrics.py) expects the
ASVspoof direction (higher = bonafide). We therefore feed it b = 1 - s, which is what we assume the
organizers do with our file. Every function reproduces the scorer bit-for-bit on b, including its tie
handling (stable sort, bonafide before spoof within a tie).

Two readings of the cost weighting (Pspoof = 0.5 in both, normalizer min(Cmiss(1-P), Cfa P) = 0.5):
  official_as_written  Cmiss=1, Cfa=4  -> minDCF = min_t [ P(real flagged)   + 4 P(fake passed) ]
  brief_as_written     Cmiss=4, Cfa=1  -> minDCF = min_t [ 4 P(real flagged) +   P(fake passed) ]
  combined             mean of the two (our model-selection metric until organizers confirm).
"""
from __future__ import annotations

import numpy as np

INTERPRETATIONS = {
    "official_as_written": {"pspoof": 0.5, "cmiss": 1.0, "cfa": 4.0},
    "brief_as_written": {"pspoof": 0.5, "cmiss": 4.0, "cfa": 1.0},
}


def _split(scores, labels, higher_is_fake=True):
    """-> (bonafide scores, spoof scores) in ASVspoof direction (higher = bonafide)."""
    s = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels)
    is_spoof = lab == "spoof" if lab.dtype.kind in "USO" else lab.astype(bool)
    if lab.dtype.kind in "USO" and not np.isin(lab, ["bonafide", "spoof"]).all():
        raise ValueError("labels must be 'bonafide' or 'spoof'")
    if s.shape != is_spoof.shape or not np.isfinite(s).all():
        raise ValueError("scores/labels shape mismatch or non-finite scores")
    b = 1.0 - s if higher_is_fake else s
    bona, spoof = b[~is_spoof], b[is_spoof]
    if bona.size == 0 or spoof.size == 0:
        raise ValueError("need at least one bonafide and one spoof")
    return bona, spoof


def _det(bona, spoof):
    """Scorer's compute_det_curve: frr[k] = bona among k lowest / Nb, far[k] = spoof above them / Ns."""
    all_s = np.concatenate([bona, spoof])
    is_bona = np.concatenate([np.ones(bona.size), np.zeros(spoof.size)])[np.argsort(all_s, kind="mergesort")]
    cb = np.cumsum(is_bona)
    frr = np.concatenate([[0.0], cb / bona.size])
    far = np.concatenate([[1.0], (spoof.size - (np.arange(1, all_s.size + 1) - cb)) / spoof.size])
    return frr, far


def _costs(interpretation, pspoof, cmiss, cfa):
    if interpretation == "combined":
        raise ValueError("'combined' is only valid for min_dcf/act_dcf")
    c = dict(INTERPRETATIONS[interpretation])
    for k, v in (("pspoof", pspoof), ("cmiss", cmiss), ("cfa", cfa)):
        if v is not None:
            c[k] = float(v)
    return c["pspoof"], c["cmiss"], c["cfa"]


def min_dcf(scores, labels, interpretation="official_as_written", *, pspoof=None, cmiss=None, cfa=None,
            higher_is_fake=True):
    """Normalized minimum DCF. pspoof/cmiss/cfa override the named interpretation's values."""
    if interpretation == "combined":
        return float(np.mean([min_dcf(scores, labels, k, higher_is_fake=higher_is_fake) for k in INTERPRETATIONS]))
    p, cm, cf = _costs(interpretation, pspoof, cmiss, cfa)
    frr, far = _det(*_split(scores, labels, higher_is_fake))
    return float(np.min(cm * frr * (1 - p) + cf * far * p) / min(cm * (1 - p), cf * p))


def act_dcf(scores, labels, interpretation="official_as_written", *, pspoof=None, cmiss=None, cfa=None,
            higher_is_fake=True):
    """Scorer's actual DCF: treats b = 1 - s as an LLR, threshold -log(beta). For [0,1] scores this is 1.0."""
    if interpretation == "combined":
        return float(np.mean([act_dcf(scores, labels, k, higher_is_fake=higher_is_fake) for k in INTERPRETATIONS]))
    p, cm, cf = _costs(interpretation, pspoof, cmiss, cfa)
    bona, spoof = _split(scores, labels, higher_is_fake)
    thr = -np.log(cm * (1 - p) / (cf * p))
    dcf = cm * (1 - p) * np.mean(bona < thr) + cf * p * np.mean(spoof >= thr)
    return float(dcf / min(cf * p, cm * (1 - p)))


def eer(scores, labels, higher_is_fake=True):
    """Scorer's EER (as a fraction): mean of FRR/FAR at the point where |FRR - FAR| is smallest."""
    frr, far = _det(*_split(scores, labels, higher_is_fake))
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2)


def cllr(scores, labels, higher_is_fake=True):
    """Scorer's CLLR in bits, treating b = 1 - s as a bonafide-vs-spoof LLR (uncalibrated for [0,1] scores)."""
    bona, spoof = _split(scores, labels, higher_is_fake)
    return float(0.5 * (np.mean(np.log1p(np.exp(-bona))) + np.mean(np.log1p(np.exp(spoof)))) / np.log(2))


def all_metrics(scores, labels, higher_is_fake=True) -> dict:
    kw = {"higher_is_fake": higher_is_fake}
    out = {f"min_dcf_{k}": min_dcf(scores, labels, k, **kw) for k in INTERPRETATIONS}
    out["min_dcf_combined"] = (out["min_dcf_official_as_written"] + out["min_dcf_brief_as_written"]) / 2
    out |= {f"act_dcf_{k}": act_dcf(scores, labels, k, **kw) for k in INTERPRETATIONS}
    out["eer"] = eer(scores, labels, **kw)
    out["cllr"] = cllr(scores, labels, **kw)
    return out
