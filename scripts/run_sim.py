"""Generate the Phase-3 synthetic set and its manifest.

Run with the overlay venv (shared .venv + phonemizer/espeakng-loader/sentencepiece):
  PYTHONPATH=src .venv-sim/Scripts/python scripts/run_sim.py            # all generators, then manifest
  PYTHONPATH=src .venv-sim/Scripts/python scripts/run_sim.py --only sim_griffinlim --n 20
  PYTHONPATH=src .venv-sim/Scripts/python scripts/run_sim.py --only     # manifest only

Resumable: existing wavs are kept (written atomically), clips too short to use are recorded in
<generator>/_skipped.txt and not re-synthesized. The manifest is always rebuilt from the wavs on disk for every
generator at its default size, so a partial or --only run can't leave it stale. Outputs:
  data/processed/sim/<generator>/<text_id>[_<speaker>].wav   (16 kHz mono PCM16)
  data/processed/sim/_copysyn_sources.parquet                 (frozen sample of real clips to copy-synthesize)
  data/processed/manifests/sim.parquet                        (audio.MANIFEST_COLUMNS)
"""
import argparse
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from hearsay import audio, device
from hearsay.sim import copysyn, tts

LJ_DIR = audio.DATA / "raw" / "lj_real"
OUT = audio.PROCESSED / "sim"
TEXT_DIR = OUT / "_text"
SOURCES = OUT / "_copysyn_sources.parquet"
# LJSpeech transcripts as distributed in NVIDIA's Tacotron 2 filelists (LJSpeech itself is public domain).
FILELIST_URL = ("https://raw.githubuser" "content.com/NVIDIA/tacotron2/master/filelists/"
                "ljs_audio_text_{}_filelist.txt")
CLEAN = re.compile(r"^[A-Za-z ,.;:'?!-]+$")  # no digits/abbrev symbols: MMS drops unknown chars silently
ARCTIC = ["awb", "bdl", "clb", "jmk", "ksp", "rms", "slt"]  # == sorted tts.speecht5_speakers()
TTS_RMS = 0.063  # median RMS of the 242 real LJ clips: TTS output is level-matched to real speech
# Real clips copy-synthesized per split (2/3 LibriSpeech, 1/3 LJSpeech), on top of the 242 lj_real clips.
N_SOURCES = {"train": 600, "val": 150, "test_internal": 150}

LIC_LJ = "LJSpeech public domain"
LIC_LIBRI = "LibriSpeech CC-BY-4.0"
# generator -> (n clips, native sr, license of the model + source)
PLAN = {
    "sim_griffinlim": (None, 16000, LIC_LJ + "; librosa ISC"),
    "sim_hifigan_copysyn": (None, 16000, LIC_LJ + "; microsoft/speecht5_hifigan MIT"),
    "sim_griffinlim_copysyn_libri": (None, 16000, LIC_LIBRI + "; librosa ISC"),
    "sim_hifigan_copysyn_libri": (None, 16000, LIC_LIBRI + "; microsoft/speecht5_hifigan MIT"),
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


def copysyn_sources():
    """Real LibriSpeech/LJSpeech clips to copy-synthesize, per split. Frozen to disk on first use so a later
    make_splits run can't reshuffle it. speaker/text_id are copied verbatim from the source row, so each vocoded
    clip lands in the same split group (text_id) as its real original."""
    if SOURCES.exists():
        return pd.read_parquet(SOURCES)
    m = pd.read_parquet(audio.PROCESSED / "manifest.parquet")
    lj_real = {p.stem for p in LJ_DIR.glob("*.wav")}  # already copy-synthesized; also avoids file-name clashes
    m = m[(m.label == "bonafide") & m.source.isin(["librispeech", "ljspeech"])
          & ~m.path.map(lambda p: Path(p).stem).isin(lj_real)]
    parts = []
    for split, n in N_SOURCES.items():
        s = m[m.split == split]
        n_libri = n * 2 // 3
        parts += [s[s.source == "librispeech"].sample(n_libri, random_state=0),
                  s[s.source == "ljspeech"].sample(n - n_libri, random_state=0)]
    src = pd.concat(parts)[["path", "source", "speaker", "text_id", "split"]].sort_values("path", ignore_index=True)
    src.to_parquet(SOURCES, index=False)
    return src


def jobs(gen, n, text):
    """Yield (text_id, speaker, out_path, license, fn) where fn() -> (wav, native sr)."""
    n_default, _, lic = PLAN[gen]
    d = OUT / gen
    if "copysyn" in gen or gen == "sim_griffinlim":
        f = copysyn.griffin_lim if "griffinlim" in gen else copysyn.hifigan
        if gen.endswith("_libri"):
            src = [(r.text_id, r.speaker, audio.ROOT / r.path, lic)
                   for r in copysyn_sources().itertuples() if r.source == "librispeech"]
        else:  # LJ voice: the 242 lj_real clips, then the sampled LJSpeech clips
            src = [(p.stem, "LJ", p, lic) for p in sorted(LJ_DIR.glob("*.wav"))]
            src += [(r.text_id, r.speaker, audio.ROOT / r.path, lic)
                    for r in copysyn_sources().itertuples() if r.source == "ljspeech"]
        for tid, spk, p, li in src[:n]:
            yield tid, spk, d / Path(p).name, li, lambda p=p: (f(audio.load(p)[0]), 16000)
        return
    for i, tid in enumerate(list(text)[:n or n_default]):
        if gen == "sim_speecht5":
            spk = ARCTIC[i % 7]
            yield tid, f"arctic_{spk}", d / f"{tid}_{spk}.wav", lic, \
                lambda t=text[tid], s=spk, i=i: tts.speecht5(t, s, i)
        else:
            repo = "kakao-enterprise/vits-ljs" if gen == "sim_vits_ljs" else "facebook/mms-tts-eng"
            yield tid, ("LJ" if gen == "sim_vits_ljs" else "mms_eng"), d / f"{tid}.wav", lic, \
                lambda t=text[tid], r=repo, i=i: tts.vits(t, r, seed=i)


def level(y):
    """TTS has no paired real clip: scale to the median real-LJ RMS so level is not a class cue."""
    return (y * TTS_RMS / (np.sqrt(np.mean(y**2)) + 1e-9)).astype(np.float32)


def run(gen, n, text, reverse=False):
    d, t0, made = OUT / gen, time.time(), 0
    skip_file = d / "_skipped.txt"
    skipped = set(skip_file.read_text().split()) if skip_file.exists() else set()
    todo = list(jobs(gen, n, text))
    for k, (tid, spk, out, lic, fn) in enumerate(todo[::-1] if reverse else todo):
        if out.exists() or out.name in skipped:
            continue
        y, sr = fn()
        if sr != audio.SR:  # same resampler as audio.load so all sources share one path
            y = librosa.resample(y, orig_sr=sr, target_sr=audio.SR, res_type="soxr_hq")
        if len(y) / audio.SR < audio.MIN_DURATION:
            d.mkdir(parents=True, exist_ok=True)
            with open(skip_file, "a") as fh:
                fh.write(out.name + "\n")
            continue
        if "copysyn" not in gen and gen != "sim_griffinlim":
            y = level(y)  # copy-synthesis is already RMS-matched to its source clip (copysyn._match)
        tmp = d / "_tmp" / out.name
        audio.save(tmp, y)
        os.replace(tmp, out)  # atomic: a killed run never leaves a truncated wav behind
        made += 1
        if made % 50 == 0:
            print(f"{gen} {k} made={made} {time.time() - t0:.0f}s", flush=True)
    print(f"{gen}: made {made} in {time.time() - t0:.0f}s", flush=True)


def write_manifest(text):
    """Every generator at its default size, from the wavs that exist on disk."""
    rows = []
    for gen, (_, sr_orig, _) in PLAN.items():
        for tid, spk, out, lic, _ in jobs(gen, None, text):
            if out.exists():
                rows.append(dict(path=audio.rel(out), label="spoof", source="sim", generator=gen, speaker=spk,
                                 text_id=tid, duration=sf.info(str(out)).duration, sr_orig=sr_orig,
                                 codec_orig="none", license=lic))
    m = pd.DataFrame(rows, columns=audio.MANIFEST_COLUMNS)
    assert (m.duration >= audio.MIN_DURATION).all() and m.path.is_unique
    on_disk = {audio.rel(p) for p in OUT.glob("sim_*/*.wav")}
    if on_disk - set(m.path):
        print(f"WARNING: {len(on_disk - set(m.path))} wavs on disk are not in the plan (not in manifest)")
    audio.MANIFESTS.mkdir(parents=True, exist_ok=True)
    m.to_parquet(audio.MANIFESTS / "sim.parquet", index=False)
    print(m.groupby("generator").duration.agg(["count", "mean", "sum"]).round(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(PLAN))
    ap.add_argument("--n", type=int, default=None, help="override clips per generator (smoke tests)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reverse", action="store_true", help="work from the end: a 2nd process on one generator")
    device.add_argument(ap)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tts.DEVICE = copysyn.DEVICE = device.resolve(a.device)
    print(f"device {tts.DEVICE}", flush=True)
    text = texts()
    print(f"{len(text)} clean LJ sentences", flush=True)
    for g in a.only:
        run(g, a.n, text, a.reverse)
    write_manifest(text)


if __name__ == "__main__":
    sys.exit(main())
