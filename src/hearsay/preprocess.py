"""Model-input preprocessing shared by every rung (train AND test go through prep()).

Neutralizes cues the audits showed are artifacts of how each corpus was prepared, not of real vs fake:
- 7-8 kHz band: the organizers' LJ reals keep energy to 8 kHz (no anti-alias roll-off), properly resampled
  clips don't -> low-pass at 7 kHz (docs/05_speech_biology.md, reports/01_data_audit.md).
- DC offset (openvoicev2), leading/trailing digital silence (your_tts/xtts_v2/playht padding), peak normalization
  (xtts_v2/your_tts at full scale) -> remove DC, trim, RMS-normalize.
"""
import io

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, firwin, freqz, oaconvolve, sosfiltfilt

SR = 16000
LOWPASS_HZ = 7000
TARGET_RMS = 0.05
_SOS = butter(10, LOWPASS_HZ, btype="low", fs=SR, output="sos")


def prep(y, sr=SR):
    """Deterministic cleanup: DC removal, silence trim, 7 kHz low-pass, RMS normalization."""
    assert sr == SR, "canonical clips are 16 kHz"
    y = y - np.mean(y)
    yt, _ = librosa.effects.trim(y, top_db=40)
    if len(yt) >= SR:  # keep untrimmed if trimming leaves < 1 s (near-silent clips)
        y = yt
    y = sosfiltfilt(_SOS, y)
    rms = np.sqrt(np.mean(y**2)) + 1e-8
    return (y * (TARGET_RMS / rms)).astype(np.float32)


def crop(y, seconds, rng=None):
    """Fixed-length window: random offset when rng is given (train), centered otherwise. Short clips tile."""
    n = int(seconds * SR)
    if len(y) < n:
        y = np.tile(y, int(np.ceil(n / len(y))))
    start = rng.integers(0, len(y) - n + 1) if rng is not None else (len(y) - n) // 2
    return y[start:start + n]


def _notch_fir(rng, n_bands=5, min_f=20, max_f=8000, min_bw=100, max_bw=1000, min_c=10, max_c=100, min_g=0, max_g=0):
    """RawBoost's random multi-band FIR filter (Tak et al., ICASSP 2022): cascade of band-pass FIRs, peak gain G dB."""
    b = np.ones(1)
    for _ in range(n_bands):
        fc, bw = rng.uniform(min_f, max_f), rng.uniform(min_bw, max_bw)
        c = int(rng.integers(min_c, max_c))
        c += 1 - c % 2  # odd number of taps
        lo, hi = max(fc - bw / 2, 1.0), min(fc + bw / 2, SR / 2 - 1)
        if hi > lo:
            b = np.convolve(firwin(c, [lo, hi], window="hamming", fs=SR), b)
    _, h = freqz(b, 1, fs=SR)
    return 10 ** (rng.uniform(min(min_g, max_g), max(min_g, max_g)) / 20) * b / (np.max(np.abs(h)) + 1e-12)


def _norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def _lnl(x, rng, n_f=5, bias_lo=5, bias_hi=20):
    """Linear and non-linear convolutive noise: sum_i FIR_i(x ** i), higher powers attenuated."""
    y = np.zeros_like(x)
    lo, hi = 0.0, 0.0
    for i in range(n_f):
        if i == 1:
            lo, hi = lo - bias_lo, hi - bias_hi
        y = y + oaconvolve(x ** (i + 1), _notch_fir(rng, min_g=lo, max_g=hi), mode="same")
    return _norm(y - y.mean())


def _isd(x, rng, p=10, g=2):
    """Impulsive signal-dependent noise on a random P % of samples."""
    n = int(len(x) * rng.uniform(0, p) / 100)
    idx = rng.permutation(len(x))[:n]
    y = x.copy()
    y[idx] = x[idx] + g * x[idx] * (2 * rng.random(n) - 1) * (2 * rng.random(n) - 1)
    return _norm(y)


def _ssi(x, rng, snr_lo=10, snr_hi=40):
    """Stationary signal-independent coloured noise at a random SNR."""
    noise = _norm(oaconvolve(rng.normal(0, 1, len(x)), _notch_fir(rng), mode="same"))
    snr = rng.uniform(snr_lo, snr_hi)
    return x + noise / (np.linalg.norm(noise) + 1e-12) * np.linalg.norm(x) / 10 ** (0.05 * snr)


RAWBOOST = ((_lnl,), (_isd,), (_ssi,), (_lnl, _isd), (_lnl, _isd, _ssi))  # the paper's algorithms 1, 2, 3, 5, 4


def rawboost(y, rng):
    """One RawBoost algorithm chosen at random, in series where the paper does. Length and dtype preserved."""
    x = y.astype(np.float64)
    for f in RAWBOOST[int(rng.integers(len(RAWBOOST)))]:
        x = f(x, rng)
    return x.astype(np.float32)


def augment(y, rng, rawboost_p=0.0):
    """Class-symmetric channel augmentation (apply to real AND fake, before prep()). With rawboost_p > 0, RawBoost runs
    first with that probability; at 0 the random stream and the output are exactly the original recipe's."""
    if rawboost_p and rng.random() < rawboost_p:
        y = rawboost(y, rng)
    r = rng.random()
    if r < 0.25:  # lossy codec round trip
        buf = io.BytesIO()
        sf.write(buf, y, SR, format="MP3")
        buf.seek(0)
        y2, _ = sf.read(buf, dtype="float32")
        y = y2[: len(y)] if len(y2) >= len(y) else np.pad(y2, (0, len(y) - len(y2)))
    elif r < 0.45:  # additive noise at 15-40 dB SNR
        snr = rng.uniform(15, 40)
        p = np.mean(y**2) + 1e-10
        y = y + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(y)).astype(np.float32)
    elif r < 0.6:  # resampler round trip through 8-24 kHz (different anti-alias behaviour)
        mid = int(rng.choice([8000, 11025, 22050, 24000]))
        y = librosa.resample(librosa.resample(y, orig_sr=SR, target_sr=mid), orig_sr=mid, target_sr=SR)
    return y.astype(np.float32)
