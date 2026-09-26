"""Generate the Phase-3 synthetic set and its manifest.

Run with the overlay venv (shared .venv + phonemizer/espeakng-loader/sentencepiece):
  PYTHONPATH=src .venv-sim/Scripts/python scripts/run_sim.py            # all generators, then manifest
  PYTHONPATH=src .venv-sim/Scripts/python scripts/run_sim.py --only sim_griffinlim --n 20

Resumable: existing wavs are kept, so a crashed run just restarts. Outputs:
  data/processed/sim/<generator>/<text_id>[_<speaker>].wav   (16 kHz mono PCM16)
  data/processed/manifests/sim.parquet                        (audio.MANIFEST_COLUMNS)
"""
import argparse
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import torch

from hearsay import audio
from hearsay.sim import copysyn, tts

LJ_DIR = audio.DATA / "raw" / "lj_real"
OUT = audio.PROCESSED / "sim"
TEXT_DIR = OUT / "_text"
# LJSpeech transcripts as distributed in NVIDIA's Tacotron 2 filelists (LJSpeech itself is public domain).
FILELIST_URL = ("https://raw.githubuser" "content.com/NVIDIA/tacotron2/master/filelists/"
                "ljs_audio_text_{}_filelist.txt")
CLEAN = re.compile(r"^[A-Za-z ,.;:'?!-]+$")  # no digits/abbrev symbols: MMS drops unknown chars silently

LIC_LJ = "LJSpeech public domain"
# generator -> (n clips, native sr, license of the model + source)
PLAN = {
    "sim_griffinlim": (None, 16000, LIC_LJ + "; librosa ISC"),
    "sim_hifigan_copysyn": (None, 16000, LIC_LJ + "; microsoft/speecht5_hifigan MIT"),
    "sim_vits_ljs": (1500, 22050, "kakao-enterprise/vits-ljs MIT; text " + LIC_LJ),
    "sim_mms_tts_eng": (1000, 16000, "facebook/mms-tts-eng CC-BY-NC-4.0; text " + LIC_LJ),
    "sim_speecht5": (420, 16000, "microsoft/speecht5_tts MIT; CMU ARCTIC x-vectors; text " + LIC_LJ),
}


def texts():
    """{LJ id: transcript} for clean sentences, ordered: ids in lj_real first (paired text), then shuffled."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {}
    for split in ("train", "val", "test"):
        f = TEXT_DIR / f"ljs_{split}.txt"
        if not f.exists():
            urllib.request.urlretrieve(FILELIST_URL.format(split), f)
        for line in f.read_text(encoding="utf8").splitlines():
            wav, text = line.split("|", 1)
            rows[Path(wav).stem] = text.strip()
    clean = {k: v for k, v in rows.items() if CLEAN.match(v) and 60 <= len(v) <= 220}
    paired = sorted(p.stem for p in LJ_DIR.glob("*.wav") if p.stem in clean)
    rest = sorted(set(clean) - set(paired))
    random.Random(0).shuffle(rest)
    return {k: clean[k] for k in paired + rest}


def jobs(gen, n, text):
    """Yield (text_id, speaker, out_path, fn) where fn() -> (wav at 16 kHz, native sr)."""
    d = OUT / gen
    if gen in ("sim_griffinlim", "sim_hifigan_copysyn"):
        f = copysyn.griffin_lim if gen == "sim_griffinlim" else copysyn.hifigan
        for p in sorted(LJ_DIR.glob("*.wav"))[:n]:
            yield p.stem, "LJ", d / p.name, lambda p=p: (f(audio.load(p)[0]), 16000)
        return
    ids = list(text)[:n]
    for i, tid in enumerate(ids):
        if gen == "sim_speecht5":
            spk = tts.speecht5_speakers()[i % 7]
            yield tid, f"arctic_{spk}", d / f"{tid}_{spk}.wav", lambda t=text[tid], s=spk, i=i: tts.speecht5(t, s, i)
        else:
            repo = "kakao-enterprise/vits-ljs" if gen == "sim_vits_ljs" else "facebook/mms-tts-eng"
            yield tid, ("LJ" if gen == "sim_vits_ljs" else "mms_eng"), d / f"{tid}.wav", \
                lambda t=text[tid], r=repo, i=i: tts.vits(t, r, seed=i)


def run(gen, n, text):
    n_default, sr_orig, lic = PLAN[gen]
    rows, t0 = [], time.time()
    for k, (tid, spk, out, fn) in enumerate(jobs(gen, n or n_default, text)):
        if not out.exists():
            y, sr = fn()
            if sr != audio.SR:  # same resampler as audio.load so all sources share one path
                y = librosa.resample(y, orig_sr=sr, target_sr=audio.SR, res_type="soxr_hq")
            if len(y) / audio.SR < audio.MIN_DURATION:
                continue
            audio.save(out, y)
        rows.append(dict(path=audio.rel(out), label="spoof", source="sim", generator=gen, speaker=spk,
                         text_id=tid, duration=sf.info(str(out)).duration, sr_orig=sr_orig,
                         codec_orig="none", license=lic))
        if k % 50 == 0:
            print(f"{gen} {k} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows, columns=audio.MANIFEST_COLUMNS).to_parquet(OUT / gen / "_rows.parquet", index=False)
    print(f"{gen}: {len(rows)} clips in {time.time() - t0:.0f}s", flush=True)


def write_manifest():
    parts = [pd.read_parquet(p) for p in sorted(OUT.glob("*/_rows.parquet"))]
    m = pd.concat(parts, ignore_index=True)[audio.MANIFEST_COLUMNS]
    assert (m.duration >= audio.MIN_DURATION).all() and m.path.is_unique
    audio.MANIFESTS.mkdir(parents=True, exist_ok=True)
    m.to_parquet(audio.MANIFESTS / "sim.parquet", index=False)
    print(m.groupby("generator").duration.agg(["count", "mean", "sum"]).round(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(PLAN))
    ap.add_argument("--n", type=int, default=None, help="override clips per generator (smoke tests)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    text = texts()
    print(f"{len(text)} clean LJ sentences", flush=True)
    for g in a.only:
        run(g, a.n, text)
    write_manifest()


if __name__ == "__main__":
    sys.exit(main())
