"""Classic spectral features (R1) on a prep()'d 16 kHz clip -> dict with a FIXED key set (FEATURES).

LFCC (linear triangular filterbank cepstra, the ASVspoof baseline front end) and MFCC, 20 coeffs + deltas, each
summarized by mean/std over frames; plus spectral centroid/bandwidth/rolloff/flatness/contrast stats.

Everything is computed on the 0-7 kHz band only. prep() low-passes at 7 kHz, but a 10th-order Butterworth still
leaves ~-20 dB of the 7-8 kHz band, which is exactly where the resampler shortcut lives (reports/data_audit.md).
ponytail: CQCC skipped (CQT is ~10x the cost of an STFT here); LFCC covers the same "linear HF resolution" idea.
"""
import librosa
import numpy as np
from scipy.fft import dct

SR = 16000
NFFT, WIN, HOP = 512, 400, 160   # 25 ms window, 10 ms hop
FMAX = 7000.0
NCEP, NFILT = 20, 40
_FREQS = np.fft.rfftfreq(NFFT, 1 / SR)
_KEEP = _FREQS <= FMAX


def _tri_bank(edges):
    """Triangular filters with the given (NFILT + 2) edge frequencies, on the kept STFT bins."""
    f = _FREQS[_KEEP]
    lo, c, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)))


_LIN = _tri_bank(np.linspace(0, FMAX, NFILT + 2))
_MEL = _tri_bank(librosa.mel_frequencies(NFILT + 2, fmin=0, fmax=FMAX))

CEPS = [f"{kind}{d}_{i}_{stat}" for kind in ("lfcc", "mfcc") for d in ("", "_d") for i in range(NCEP)
        for stat in ("mean", "std")]
SPEC = [f"{n}_{stat}" for n in ("centroid", "bandwidth", "rolloff85", "flatness", *[f"contrast{b}" for b in range(6)])
        for stat in ("mean", "std")]
FEATURES = CEPS + SPEC


def _cep(P, bank):
    c = dct(np.log(bank @ P + 1e-10), type=2, axis=0, norm="ortho")[:NCEP]
    return c, librosa.feature.delta(c, width=5, mode="nearest")


def extract_spectral(y):
    y = np.asarray(y, dtype=np.float32)
    P = np.abs(librosa.stft(y, n_fft=NFFT, win_length=WIN, hop_length=HOP)) ** 2
    P = P[_KEEP]  # 0-7 kHz only
    out = {}
    for kind, bank in (("lfcc", _LIN), ("mfcc", _MEL)):
        for d, c in zip(("", "_d"), _cep(P, bank)):
            for i in range(NCEP):
                out[f"{kind}{d}_{i}_mean"], out[f"{kind}{d}_{i}_std"] = c[i].mean(), c[i].std()
    f = _FREQS[_KEEP][:, None]
    tot = P.sum(0) + 1e-12
    cen = (f * P).sum(0) / tot
    bands = {
        "centroid": cen,
        "bandwidth": np.sqrt(((f - cen) ** 2 * P).sum(0) / tot),
        "rolloff85": f[np.argmax(np.cumsum(P, 0) >= 0.85 * tot, axis=0), 0],
        "flatness": np.exp(np.log(P + 1e-12).mean(0)) / (P.mean(0) + 1e-12),
    }
    # octave-band contrast (peak vs valley, 20% quantiles) over 0-200-400-800-1600-3200-7000 Hz
    edges = [0, 200, 400, 800, 1600, 3200, FMAX]
    for b in range(6):
        sub = np.sort(np.log(P[(f[:, 0] >= edges[b]) & (f[:, 0] < edges[b + 1])] + 1e-10), axis=0)
        k = max(1, int(0.2 * len(sub)))
        bands[f"contrast{b}"] = sub[-k:].mean(0) - sub[:k].mean(0)
    for n, v in bands.items():
        out[f"{n}_mean"], out[f"{n}_std"] = float(np.mean(v)), float(np.std(v))
    return {k: float(out[k]) for k in FEATURES}
