"""Enumerate the challenge-provided training data (DiffSSD fakes + LJSpeech reals) with metadata parsed from paths.

DiffSSD layout (Bhagtani et al. 2024, arXiv 2409.13049; HF purdueviperlab/diffssd):
  <gen>/sentence_N.wav                         pre-trained on LJSpeech -> LJ voice, N in 0..4999
  <gen>/speaker_S/sentence_N[_accent].{wav,mp3} zero-shot clone of LibriSpeech speaker S, N in 0..499
sentence_N is the SAME text line N for every generator and speaker (text_script_0_499.csv is the first 500 rows
of text_script_0_4999.csv; verified row-by-row), so text_id = "diffssd:N" is global.
"""
import re
from pathlib import Path

from hearsay.audio import DATA

RAW = DATA / "raw"
LJ_GENERATORS = {"diffgan_tts", "grad_tts", "pro_diff", "wavegrad2"}  # single-speaker, trained on LJSpeech
LICENSES = {"diffssd": "CC BY-NC-ND 4.0", "lj_real": "Public domain (LJSpeech 1.1)"}
_SENT = re.compile(r"sentence_(\d+)(?:_(en-[a-z]+))?$")


def diffssd_meta(path):
    p = Path(path)
    rel_parts = p.relative_to(RAW / "diffssd").parts
    gen = rel_parts[0]
    m = _SENT.match(p.stem)
    if m is None:
        raise ValueError(f"unexpected DiffSSD file name: {path}")
    speaker = "ljspeech:LJ" if len(rel_parts) == 2 else "librispeech:" + rel_parts[1].split("_")[1]
    return {"label": "spoof", "source": "diffssd", "generator": gen, "speaker": speaker,
            "text_id": f"diffssd:{int(m.group(1))}", "accent": m.group(2), "license": LICENSES["diffssd"]}


def lj_meta(path):
    return {"label": "bonafide", "source": "lj_real", "generator": "bonafide", "speaker": "ljspeech:LJ",
            "text_id": f"ljspeech:{Path(path).stem}", "accent": None, "license": LICENSES["lj_real"]}


def list_given():
    """[(raw_path, meta)] for every DiffSSD + lj_real file, sorted for determinism."""
    out = [(p, diffssd_meta(p)) for p in sorted((RAW / "diffssd").rglob("*")) if p.suffix in (".wav", ".mp3")]
    out += [(p, lj_meta(p)) for p in sorted((RAW / "lj_real").glob("*.wav"))]
    return out


def processed_path(raw_path, meta):
    """data/processed/<source>/<same relative layout>.wav"""
    raw_path = Path(raw_path)
    sub = raw_path.relative_to(RAW / meta["source"])
    return DATA / "processed" / meta["source"] / sub.with_suffix(".wav")
