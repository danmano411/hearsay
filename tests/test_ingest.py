"""Checks for the Phase-2 ingest helpers: canonical conversion + ranged zip member extraction."""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ingest_common as ic  # noqa: E402


def wav_bytes(seconds, sr=22050, channels=2):
    t = np.arange(int(seconds * sr)) / sr
    y = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_convert_canonical_and_min_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "rel", lambda p: Path(p).name)
    meta = dict(label="spoof", source="t", generator="g", speaker="s", text_id="x", codec_orig="wav", license="L")
    row = ic.convert(dict(src=wav_bytes(4.0), out=str(tmp_path / "a.wav"), **meta))
    assert list(row) == ic.MANIFEST_COLUMNS
    assert row["sr_orig"] == 22050 and abs(row["duration"] - 4.0) < 0.01
    info = sf.info(str(tmp_path / "a.wav"))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert ic.convert(dict(src=wav_bytes(2.0), out=str(tmp_path / "b.wav"), **meta)) is None
    assert not (tmp_path / "b.wav").exists()
    # resume path: existing output is reused
    assert ic.convert(dict(src=None, out=str(tmp_path / "a.wav"), **meta))["duration"] == row["duration"]


def test_zip_member_bytes_ranged():
    buf = io.BytesIO()
    payloads = {"d/stored.wav": wav_bytes(0.1), "d/deflated.wav": wav_bytes(0.2)}
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("d/stored.wav", payloads["d/stored.wav"], compress_type=zipfile.ZIP_STORED)
        z.writestr("d/deflated.wav", payloads["d/deflated.wav"], compress_type=zipfile.ZIP_DEFLATED)
    blob = buf.getvalue()
    for info in zipfile.ZipFile(io.BytesIO(blob)).infolist():
        assert ic.zip_member_bytes(lambda o, n: blob[o:o + n], info) == payloads[info.filename]
