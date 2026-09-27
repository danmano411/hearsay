"""Shared helpers for Stage-2 external-data ingest scripts.

Every ingest turns (audio bytes or path) + metadata into a canonical clip via hearsay.audio
and writes data/processed/manifests/<source>.parquet with exactly MANIFEST_COLUMNS.
"""
import io
import itertools
import struct
import time
import sys
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hearsay.audio import DATA, MANIFEST_COLUMNS, MANIFESTS, MIN_DURATION, PROCESSED, ROOT, SR, load, rel, save  # noqa: E402,F401

EXTERNAL = DATA / "external"
WORKERS = 4  # shared 16-thread box, 5 agents


def decode_bytes(b, sr=SR):
    """Same contract as hearsay.audio.load, for in-memory wav/flac bytes."""
    y, sr_orig = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr_orig != sr:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr, res_type="soxr_hq")
    return y.astype(np.float32), sr_orig


def convert(job):
    """job = dict(src=<path|bytes>, out=<abs .wav path>, **manifest fields). Returns a row, or None if < MIN_DURATION.

    Resumable: an existing output is re-used (only its header is read).
    """
    out = Path(job["out"])
    if out.exists():
        dur, sr_orig = sf.info(str(out)).duration, job.get("sr_orig")
    else:
        try:
            src = job["src"]
            y, sr_orig = decode_bytes(src) if isinstance(src, bytes) else load(src)
        except Exception as e:  # corrupt upstream file: skip, never crash the whole ingest
            print(f"[skip] {out.name}: {e}", flush=True)
            return None
        dur = len(y) / SR
        if dur < MIN_DURATION:
            return None
        save(out, y)
    if dur < MIN_DURATION:
        return None
    row = {k: job.get(k) for k in MANIFEST_COLUMNS}
    row.update(path=rel(out), duration=round(float(dur), 3), sr_orig=sr_orig)
    return row


def run(jobs, source, batch=256):
    """Convert an (possibly lazy / streaming) iterable of jobs with WORKERS processes and write the manifest.

    Jobs are consumed in batches so a streamed archive never sits in RAM whole.
    """
    rows, seen = [], 0
    with ProcessPoolExecutor(WORKERS) as pool:
        for chunk in itertools.batched(jobs, batch):
            rows += [r for r in pool.map(convert, chunk, chunksize=8) if r]
            seen += len(chunk)
            print(f"[{source}] {seen} seen, {len(rows)} kept", flush=True)
    return write_manifest(rows, source)


def write_manifest(rows, source):
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df["sr_orig"] = df["sr_orig"].astype("Int64")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANIFESTS / f"{source}.parquet", index=False)
    print(f"[{source}] manifest: {len(df)} rows", df["label"].value_counts().to_dict(),
          f"{df['duration'].sum() / 3600:.1f} h", flush=True)
    return df


def out_path(source, *parts):
    """parts[-1] is a stem; appended (not with_suffix) so stems containing dots stay unique."""
    return PROCESSED.joinpath(source, *parts[:-1], parts[-1] + ".wav")


# ---------------------------------------------------------------- remote zip (HTTP range) access
class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP Range requests (enough for zipfile's central directory)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def read_range(self, off, n):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={off}-{off + n - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception as e:
                if attempt == 4:
                    raise
                wait = int(getattr(e, "headers", {}).get("Retry-After", 0) or 0) or 10 * 2 ** attempt
                print(f"[http] {e}; retry in {wait}s", flush=True)
                time.sleep(wait)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        b = self.read_range(self.pos, n)
        self.pos += len(b)
        return b


def zip_member_bytes(read_range, info):
    """Fetch + decompress one zip member with a single ranged read (thread-safe, no shared file position).

    read_range(offset, n) -> bytes; info is a zipfile.ZipInfo from the central directory.
    """
    # Local header is 30 bytes + name + extra; its extra field can differ from the central one, so over-fetch 1 KiB.
    blob = read_range(info.header_offset, 30 + len(info.orig_filename.encode()) + 1024 + info.compress_size)
    assert blob[:4] == b"PK\x03\x04", "bad local header"
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    data = blob[start:start + info.compress_size]
    if info.compress_type == 0:
        return data
    if info.compress_type == 8:
        return zlib.decompressobj(-15).decompress(data)
    raise ValueError(f"unsupported compression {info.compress_type}")
