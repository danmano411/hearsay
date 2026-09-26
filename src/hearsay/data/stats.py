"""Trivial per-clip signal statistics used by the raw audit and the shortcut audit.

These are exactly the "cheap cues" a classifier could exploit instead of learning real-vs-synthetic.
"""
import numpy as np

SIL_THR = 1e-4  # |x| below this counts as digital silence
STAT_COLUMNS = ["duration", "peak", "rms_db", "clip_frac", "dc", "lead_sil", "trail_sil", "rolloff95", "bw60"]


def clip_stats(y, sr):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    loud = np.flatnonzero(np.abs(y) >= SIL_THR)
    lead = loud[0] if len(loud) else n
    trail = n - 1 - loud[-1] if len(loud) else n
    rms = np.sqrt(np.mean(y ** 2)) if n else 0.0
    return {
        "duration": n / sr,
        "peak": float(np.max(np.abs(y))) if n else 0.0,
        "rms_db": float(20 * np.log10(rms + 1e-10)),
        "clip_frac": float(np.mean(np.abs(y) >= 0.999)) if n else 0.0,
        "dc": float(np.mean(y)) if n else 0.0,
        "lead_sil": float(lead / sr),
        "trail_sil": float(trail / sr),
        **spectral(y, sr),
    }


def spectral(y, sr, n_fft=1024):
    """rolloff95: Hz below which 95% of spectral energy lies.
    bw60: highest Hz whose mean power is within 60 dB of the spectral peak (effective bandwidth / low-pass edge)."""
    if len(y) < n_fft:
        return {"rolloff95": 0.0, "bw60": 0.0}
    hop = n_fft // 2
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop] * np.hanning(n_fft)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    if power.sum() <= 0:
        return {"rolloff95": 0.0, "bw60": 0.0}
    hz = sr / n_fft
    cum = np.cumsum(power)
    above = np.flatnonzero(power > power.max() * 1e-6)
    return {"rolloff95": float(np.searchsorted(cum, 0.95 * cum[-1]) * hz), "bw60": float(above[-1] * hz)}
