"""Real vs each sim stage (and the DiffSSD generators) -> docs/figures/sim_*.png + a printed stats table.

  PYTHONPATH=src .venv/Scripts/python scripts/sim_figures.py [--n 60]

Figures: one utterance through every stage (spectrogram, group delay) and per-generator averages over
--n clips (long-term spectrum / high band, modulation spectrum). Uses only real LJ/LibriSpeech, sim and DiffSSD audio
(never HGT test audio).
"""
import argparse
import random
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hearsay import audio  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
SIM = audio.PROCESSED / "sim"
DIFFSSD = audio.DATA / "raw" / "diffssd"
N_FFT, HOP = 512, 128
PAIR_ID = "LJ001-0101"  # a clean-text real clip, so every sim stage has the same sentence
STAGES = [  # (label, path of the paired clip)
    ("real LJ", audio.DATA / "raw" / "lj_real" / f"{PAIR_ID}.wav"),
    ("Griffin-Lim copy-syn", SIM / "sim_griffinlim" / f"{PAIR_ID}.wav"),
    ("HiFi-GAN copy-syn", SIM / "sim_hifigan_copysyn" / f"{PAIR_ID}.wav"),
    ("VITS-LJS (TTS)", SIM / "sim_vits_ljs" / f"{PAIR_ID}.wav"),
    ("MMS-TTS eng (TTS)", SIM / "sim_mms_tts_eng" / f"{PAIR_ID}.wav"),
    ("SpeechT5 (TTS)", next(iter(sorted((SIM / "sim_speecht5").glob(f"{PAIR_ID}_*.wav"))), None)),
]


def groups(n):
    """label -> list of up to n file paths, for real, each sim generator and each DiffSSD generator."""
    rng = random.Random(0)
    pick = lambda fs: rng.sample(sorted(fs), min(n, len(fs)))  # noqa: E731
    g = {"real LJ": pick(list((audio.DATA / "raw" / "lj_real").glob("*.wav")))}
    src = pd.read_parquet(SIM / "_copysyn_sources.parquet")  # the real LibriSpeech clips that were copy-synthesized
    g["real LibriSpeech"] = pick([audio.ROOT / q for q in src.loc[src.source == "librispeech", "path"]])
    for d in sorted(SIM.glob("sim_*")):
        g[d.name] = pick(list(d.glob("*.wav")))
    for d in sorted(DIFFSSD.iterdir()):
        g["diffssd/" + d.name] = pick([p for p in d.rglob("*") if p.suffix in (".wav", ".mp3")])
    return {k: v for k, v in g.items() if v}


def stft(y):
    return librosa.stft(y, n_fft=N_FFT, hop_length=HOP)


def phase_coherence(S, axis):
    """Magnitude-weighted resultant length of unit phase increments, averaged over frames/bins (1 = coherent).
    axis=0: along frequency (group-delay consistency); axis=1: along time, after removing each bin's expected
    advance (instantaneous-frequency consistency)."""
    W = np.abs(S)
    if axis == 0:
        d, w = S[1:] * np.conj(S[:-1]), W[1:] * W[:-1]
    else:
        k = np.arange(S.shape[0])[:, None]
        d = S[:, 1:] * np.conj(S[:, :-1]) * np.exp(-2j * np.pi * k * HOP / N_FFT)
        w = W[:, 1:] * W[:, :-1]
    u = d / (np.abs(d) + 1e-12)
    return float(np.mean(np.abs((u * w).sum(axis=axis)) / (w.sum(axis=axis) + 1e-12)))


def ltas_db(y):
    return 10 * np.log10(np.mean(np.abs(stft(y)) ** 2, axis=1) + 1e-12)


def modulation(y, sr=audio.SR):
    """Temporal modulation spectrum: FFT over time of mean-removed log-mel band envelopes, band-averaged."""
    env = np.log(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=160, n_mels=32) + 1e-8)
    env = env - env.mean(axis=1, keepdims=True)
    m = np.abs(np.fft.rfft(env, n=1024, axis=1)).mean(axis=0)
    return np.fft.rfftfreq(1024, d=160 / sr), m / m.sum()


def fig_stages():
    stages = [(k, p) for k, p in STAGES if p is not None and Path(p).exists()]
    fig, ax = plt.subplots(1, len(stages), figsize=(2.6 * len(stages), 3.0), sharey=True)
    for i, (label, p) in enumerate(stages):
        S = stft(audio.load(p)[0][: 2 * audio.SR])
        t = np.arange(S.shape[1]) * HOP / audio.SR
        f = np.fft.rfftfreq(N_FFT, 1 / audio.SR) / 1000
        ax[i].pcolormesh(t, f, librosa.amplitude_to_db(np.abs(S), ref=np.max, top_db=80), cmap="magma",
                         rasterized=True, shading="auto")
        ax[i].set(title=label, xlabel="time (s)")
    ax[0].set_ylabel("frequency (kHz)")
    fig.suptitle(f"Same sentence ({PAIR_ID}), first 2 s, through each sim stage", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "sim_stages.png", dpi=72)


def fig_averages(n):
    g = groups(n)
    f = np.fft.rfftfreq(N_FFT, 1 / audio.SR)
    stats, lt, mod, ph = [], {}, {}, {}
    for label, files in g.items():
        L, M, gdc, ifc = [], [], [], []
        for p in files:
            y = audio.load(p)[0]
            L.append(ltas_db(y))
            mf, m = modulation(y)
            M.append(m)
            S = stft(y)
            gdc.append(phase_coherence(S, 0))
            ifc.append(phase_coherence(S, 1))
        lt[label], mod[label], ph[label] = np.mean(L, 0), np.mean(M, 0), (gdc, ifc)
        p = 10 ** (lt[label] / 10)
        stats.append(dict(group=label, n=len(files),
                          hf_4_8k_db=10 * np.log10(p[f >= 4000].sum() / p.sum()),
                          hf_7_8k_db=10 * np.log10(p[f >= 7000].sum() / p.sum()),
                          mod_4hz_share=mod[label][(mf >= 2) & (mf <= 8)].sum(),
                          gd_coherence=np.mean(gdc), if_coherence=np.mean(ifc)))
    st = pd.DataFrame(stats).round(3)
    print(st.to_string(index=False))

    sims = [k for k in lt if not k.startswith("diffssd/")]
    diffs = [k for k in lt if k.startswith("diffssd/")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, keys, title in [(ax[0], sims, "real LJ / LibriSpeech vs sim generators"), (ax[1], ["real LJ"] + diffs, "real LJ vs DiffSSD")]:
        for k in keys:
            a.plot(f / 1000, lt[k] - lt[k][(f > 300) & (f < 1000)].mean(), lw=2.2 if k.startswith("real") else 1,
                   color={"real LJ": "k", "real LibriSpeech": "0.5"}.get(k), label=k.replace("diffssd/", ""))
        a.axvspan(4, 8, color="0.9", zorder=-1)
        a.set(title=title, xlabel="frequency (kHz)", xlim=(0, 8), ylim=(-75, 10))
        a.legend(fontsize=7, ncol=2)
    ax[0].set_ylabel("long-term spectrum (dB, 0 = 300-1000 Hz mean)")
    fig.suptitle(f"High band (shaded 4-8 kHz): average spectrum over up to {n} clips per generator, all at 16 kHz")
    fig.tight_layout()
    fig.savefig(FIG / "sim_highband.png", dpi=80)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    ref = mod["real LJ"]
    for a, keys, title in [(ax[0], sims[1:], "sim generators"), (ax[1], diffs, "DiffSSD generators")]:
        for k in keys:
            a.plot(mf, 10 * np.log10(mod[k] / ref), lw=1, label=k.replace("diffssd/", ""))
        a.axhline(0, color="k", lw=2)
        a.axvspan(2, 8, color="0.9", zorder=-1)
        a.set(title=title + " relative to real LJ", xlabel="modulation frequency (Hz)", xlim=(0.5, 30),
              xscale="log", ylim=(-6, 6))
        a.legend(fontsize=7, ncol=2)
    ax[0].set_ylabel("modulation energy vs real LJ (dB)")
    fig.suptitle("Temporal modulation spectrum of log-mel envelopes, relative to real LJ (shaded: 2-8 Hz syllable band)")
    fig.tight_layout()
    fig.savefig(FIG / "sim_modulation.png", dpi=80)

    keys = list(ph)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for a, j, title in [(ax[0], 0, "along frequency (group-delay consistency)"),
                        (ax[1], 1, "along time (instantaneous-frequency consistency)")]:
        a.boxplot([ph[k][j] for k in keys], vert=False, widths=0.6, showfliers=False)
        a.axvline(np.median(ph["real LJ"][j]), color="k", ls="--", lw=1)
        a.set(title=title, xlabel="phase coherence (1 = perfectly coherent)")
    ax[0].set_yticks(range(1, len(keys) + 1), [k.replace("diffssd/", "DiffSSD ") for k in keys], fontsize=8)
    fig.suptitle("Phase coherence per clip (magnitude-weighted resultant of STFT phase increments); dashed = real LJ median")
    fig.tight_layout()
    fig.savefig(FIG / "sim_phase.png", dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_stages()
    fig_averages(a.n)


if __name__ == "__main__":
    main()
