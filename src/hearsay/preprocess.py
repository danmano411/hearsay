"""Model-input preprocessing shared by every rung (train AND test go through prep()).

Neutralizes cues the audits showed are artifacts of how each corpus was prepared, not of real vs fake:
- 7-8 kHz band: the organizers' LJ reals keep energy to 8 kHz (no anti-alias roll-off), properly resampled
  clips don't -> low-pass at 7 kHz (docs/speech_biology.md, reports/data_audit.md).
- DC offset (openvoicev2), leading/trailing digital silence (your_tts/xtts_v2/playht padding), peak normalization
  (xtts_v2/your_tts at full scale) -> remove DC, trim, RMS-normalize.
"""
import io

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SR = 16000
LOWPASS_HZ = 7000
TARGET_RMS = 0.05
_SOS = butter(10, LOWPASS_HZ, btype="low", fs=SR, output="sos")


def prep(y, sr=SR):
    """Deterministic cleanup: DC removal, silence trim, 7 kHz low-pass, RMS normalization."""
    assert sr == SR, "canonical clips are 16 kHz"
    y = y - np.mean(y)
    yt, _ = librosa.effects.trim(y, top_db=40)
    if len(yt) >= SR:  # ponytail: keep untrimmed if trimming leaves < 1 s (near-silent clips)
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


def augment(y, rng):
    """Class-symmetric channel augmentation (apply to real AND fake, before prep())."""
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
