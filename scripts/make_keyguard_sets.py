"""Hearsay x Keyguard probes (plan 09, Part B): does typing noise, or Keyguard's keystroke shield, make real
speech look fake (or fakes look real)?

    python scripts/make_keyguard_sets.py --keyguard <path to a keyguard clone> [--n 300]

Speech: n real + n fake clips from test_internal_testlike (seeded; the models never selected anything on it).
Keystrokes: Keyguard's committed press banks data/pool/*.npz (wins: (n, 4800) 16 kHz, onset ~20 ms in).
Mixing follows keyguard/synth.py: each press is scaled so key power / whole-clip speech power = SNR, placed at
random positions; here at a typing rate of RATE presses per second instead of a fixed 5 per clip.
Shield: keyguard.shield.shield.Shield (DSP: STFT inpainting + residue randomization + decoys), dashboard settings.

Writes data/bench/keyguard/<cond>/... and data/bench/keyguard_<cond>.parquet; score them with
    python scripts/bench_score.py r1|r4ft --sets keyguard_clean keyguard_type20 ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from hearsay.audio import DATA, ROOT, rel, save
from hearsay.evaluate import manifest

BENCH = DATA / "bench"
RATE = 3.0  # presses/s: steady typing while talking
SNRS = (20, 10, 5, 0)
DOMAINS = ("harrison", "mka_mac", "kad_spatam_mechanical", "hf_sakuzas_keyboard", "live")


def presses(kg):
    return {d: np.load(Path(kg) / "data" / "pool" / f"{d}.npz")["wins"] for d in DOMAINS}


def add_typing(y, bank, snr, rng):
    """-> (mix, onsets). Keyguard's synth.mix scaling, RATE presses/s, one keyboard domain per clip."""
    wins = bank[rng.choice(list(bank))]
    n, w = max(1, int(RATE * len(y) / 16000)), wins.shape[1]
    starts = np.sort(rng.integers(0, max(1, len(y) - w), n))
    p_speech, out = np.mean(y ** 2), y.copy()
    for s in starts:
        k = wins[rng.integers(len(wins))]
        out[s:s + w] += k * np.sqrt(p_speech / 10 ** (snr / 10) / (np.mean(k ** 2) + 1e-12))
    return out, starts + 320  # synth.py: onset = start + PRE_S * SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyguard", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    sys.path.insert(0, a.keyguard)
    from keyguard.shield.shield import Shield, ShieldConfig
    shield = lambda y, on, seed: Shield(ShieldConfig(strength=1.0, randomize=1.0, decoys=6, key_frames=6),  # noqa: E731
                                        seed=seed).apply(y.astype(np.float32), on)  # server.py:219 settings

    m = manifest()
    t = m[m.test_internal_testlike]
    pick = pd.concat([g.sample(a.n, random_state=0) for _, g in t.groupby("label")])
    bank = presses(a.keyguard)
    rows = {c: [] for c in ["clean", *[f"type{s}" for s in SNRS], "shield_clean", "shield_type10"]}
    for i, r in enumerate(pick.itertuples()):
        y, sr = sf.read(str(ROOT / r.path), dtype="float32")
        assert sr == 16000
        rng = np.random.default_rng(i)
        out = {"clean": y, "shield_clean": shield(y, None, i)}  # no keys: shield inpaints detected onsets
        for s in SNRS:
            mix, on = add_typing(y, bank, s, rng)
            out[f"type{s}"] = mix
            if s == 10:
                out["shield_type10"] = shield(mix, on, i)  # victim side: true key timestamps, as in Keyguard
        for c, z in out.items():
            p = BENCH / "keyguard" / c / f"{i:04d}_{r.label}.wav"
            peak = np.abs(z).max()
            save(p, z / peak * 0.99 if peak > 1 else z)  # avoid clipping; prep() RMS-normalizes anyway
            rows[c].append(dict(path=rel(p), label=r.label, source=r.source, generator=r.generator, orig=r.path))
        if i % 100 == 0:
            print(f"{i}/{len(pick)}", flush=True)
    for c, rr in rows.items():
        pd.DataFrame(rr).to_parquet(BENCH / f"keyguard_{c}.parquet", index=False)
    print({c: len(rr) for c, rr in rows.items()})


if __name__ == "__main__":
    main()
