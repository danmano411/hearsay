"""Canonical audio I/O + manifest schema shared by every ingest path.

Canonical clip: 16 kHz, mono, PCM16 WAV under data/processed/<source>/...
Each ingest writes data/processed/manifests/<source>.parquet with MANIFEST_COLUMNS.
"""
import hashlib
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(os.environ.get("HEARSAY_ROOT", r"C:\Users\danma\Documents\Dan\Projects\Hearsay"))
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MANIFESTS = PROCESSED / "manifests"
SR = 16000
MIN_DURATION = 3.0

# path is relative to ROOT, forward slashes. label is "bonafide" or "spoof" (ASVspoof vocabulary).
MANIFEST_COLUMNS = [
    "path", "label", "source", "generator", "speaker", "text_id",
    "duration", "sr_orig", "codec_orig", "license",
]


def load(path, sr=SR):
    """Decode any format librosa/soundfile can read -> float32 mono at `sr`, plus original sample rate."""
    sr_orig = librosa.get_samplerate(str(path))
    y, _ = librosa.load(str(path), sr=sr, mono=True, res_type="soxr_hq")
    return y.astype(np.float32), sr_orig


def save(path, y, sr=SR):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")


def pcm_hash(y):
    """Content hash for exact-duplicate detection (on the 16-bit quantized signal)."""
    return hashlib.sha1((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()
