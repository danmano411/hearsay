"""Physiology-motivated acoustic features (see docs/speech_biology.md for the why).

extract_bio(y, sr=16000) -> dict[str, float] with a FIXED key set (FEATURES). Every group is computed
independently and falls back to NaN, so silent / unvoiced / tiny clips never crash.

Groups (doc section in brackets):
  f0_*        phonation: F0 level, range, declination, micro-perturbation          [Phonation, Neuromotor]
  jitter_* shimmer_* hnr cpps h1h2                                                  [Phonation]
  tilt_* alpha_ratio                                                                [Phonation / Radiation]
  f{1,2,3}_* fvel_*                                                                 [Articulation]
  vtl_*                                                                             [Resonance]
  am_*                                                                              [Neuromotor timing]
  pause_* breath_* energy_decl                                                      [Respiration]
  hf_*        4-7 kHz noise structure (not 7-8 kHz: resampler anti-alias band = shortcut)     [Turbulence]
  gd_*                                                                              [Radiation / phase]
"""
import warnings

import numpy as np
import parselmouth
from parselmouth.praat import call

SR = 16000
HOP = 160          # 10 ms
NFFT = 1024        # 64 ms window: resolves harmonics down to F0 ~ 60 Hz
C_CM_S = 35000.0   # speed of sound in warm humid air (cm/s)
F0_MIN, F0_MAX = 60.0, 500.0
AM_BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 16), (16, 32)]

FEATURES = [
    # phonation / F0
    "f0_mean_st", "f0_std_st", "f0_range_st", "f0_decl_st_s", "f0_delta_std_st", "f0_onset_dev_st",
    "voiced_frac", "voiced_segs_per_s",
    "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3", "hnr_db", "cpps_db", "h1h2_db",
    "tilt_db_oct", "alpha_ratio_db",
    # articulation / resonance
    "f1_mean", "f2_mean", "f3_mean", "f1_std", "f2_std", "f3_std",
    "fvel_f1_med", "fvel_f2_med", "fvel_f2_p95", "fvel_f3_p95", "fjump_frac",
    "vtl_cm", "vtl_cv",
    # neuromotor rhythm
    *[f"am_{lo:g}_{hi:g}" for lo, hi in AM_BANDS], "am_peak_hz",
    # respiration / pauses
    "pause_frac", "pauses_per_s", "pause_mean_s", "pause_floor_db", "breath_per_min", "energy_decl_db_s",
    # turbulence
    "hf_flatness", "hf_flatness_unv", "hf_ratio_db",
    # phase
    "gd_std", "gd_absmean", "gd_voiced_std",
]


def _nan(x):
    return float(x) if x is not None and np.isfinite(x) else float("nan")


def _st(f0):
    """Hz -> semitones re 100 Hz (log scale matches perception and removes speaker level from spreads)."""
    return 12.0 * np.log2(f0 / 100.0)


def _pitch(snd):
    pitch = snd.to_pitch_ac(time_step=HOP / SR, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0 = pitch.selected_array["frequency"]  # 0 = unvoiced
    return pitch, f0, pitch.xs()


def _runs(mask):
    """(start, end) index pairs of True runs, end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _f0_feats(f0, t, dur):
    out = {}
    v = f0 > 0
    out["voiced_frac"] = v.mean() if len(v) else np.nan
    runs = _runs(v)
    out["voiced_segs_per_s"] = len(runs) / dur
    if v.sum() < 5:
        return out
    st = _st(f0[v])
    out["f0_mean_st"] = st.mean()
    out["f0_std_st"] = st.std()
    out["f0_range_st"] = np.percentile(st, 95) - np.percentile(st, 5)
    out["f0_decl_st_s"] = np.polyfit(t[v], st, 1)[0]
    # frame-to-frame F0 change inside voiced runs: micro-prosody / smoothness of the laryngeal control loop
    deltas = [np.diff(_st(f0[a:b])) for a, b in runs if b - a >= 3]
    if deltas:
        out["f0_delta_std_st"] = np.concatenate(deltas).std()
    # F0 excursion in the first 50 ms after each voicing onset (obstruent-induced micro-prosody, Lehiste/Silverman)
    onset = [abs(_st(f0[a + 4]) - _st(f0[a])) for a, b in runs if b - a >= 8 and a > 0]
    if onset:
        out["f0_onset_dev_st"] = float(np.mean(onset))
    return out


def _perturbation(snd, pitch):
    pp = call([snd, pitch], "To PointProcess (cc)")
    if call(pp, "Get number of points") < 10:
        return {}
    # Praat defaults (Boersma & Weenink): period floor 0.1 ms, ceiling 20 ms, max period factor 1.3, max amp factor 1.6
    out = {
        "jitter_local": call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
        "jitter_rap": call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
        "shimmer_local": call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        "shimmer_apq3": call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
    }
    hnr = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0, silence_threshold=0.1, periods_per_window=1.0)
    hv = hnr.values[0]
    hv = hv[hv > -199]  # Praat marks unvoiced frames with -200
    if len(hv):
        out["hnr_db"] = hv.mean()
    return out


def _stft(y):
    import librosa
    win = np.hanning(NFFT)
    X = librosa.stft(y, n_fft=NFFT, hop_length=HOP, window=win, center=True)
    # time-weighted STFT for group delay: tau(w) = Re(X_n X*) / |X|^2  (Yegnanarayana & Murthy 1992)
    Xn = librosa.stft(y, n_fft=NFFT, hop_length=HOP, window=win * np.arange(NFFT), center=True)
    return X, Xn


def _spectral_feats(X, Xn, f0_frames, speech):
    out = {}
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    P = np.abs(X) ** 2 + 1e-12
    n = min(P.shape[1], len(f0_frames), len(speech))
    P, X, Xn, f0_frames, speech = P[:, :n], X[:, :n], Xn[:, :n], f0_frames[:n], speech[:n]
    voiced = (f0_frames > 0) & speech
    unv = (~(f0_frames > 0)) & speech
    logP = 10 * np.log10(P)

    # long-term average spectrum tilt over speech frames, dB/octave in 100-5000 Hz
    if speech.sum() >= 5:
        ltas = 10 * np.log10(P[:, speech].mean(1))
        band = (freqs >= 100) & (freqs <= 5000)
        out["tilt_db_oct"] = np.polyfit(np.log2(freqs[band]), ltas[band], 1)[0]
        lo = P[(freqs >= 50) & (freqs < 1000)][:, speech].sum()
        hi = P[(freqs >= 1000) & (freqs < 5000)][:, speech].sum()
        out["alpha_ratio_db"] = 10 * np.log10(hi / lo)
        # turbulence: flatness of the 4-7 kHz band (1 = white noise, ->0 = tonal/over-smoothed).
        # Stops at 7 kHz on purpose: above that, the resampler's anti-alias transition band (fakes resampled from
        # 22.05/24/44.1 kHz) vs none (the given 16 kHz LJ files) dominates -> a pipeline shortcut, not turbulence.
        hb = (freqs >= 4000) & (freqs < 7000)
        lb = (freqs >= 50) & (freqs < 4000)
        flat = np.exp(np.log(P[hb]).mean(0)) / P[hb].mean(0)
        out["hf_flatness"] = flat[speech].mean()
        if unv.sum() >= 3:
            out["hf_flatness_unv"] = flat[unv].mean()
        out["hf_ratio_db"] = 10 * np.log10(P[hb][:, speech].sum() / P[lb][:, speech].sum())
        # group delay (in samples) over 100-7000 Hz, energy-gated to avoid the 1/|X|^2 blow-up near zeros
        gb = (freqs >= 100) & (freqs <= 7000)
        Xs, Xns = X[gb][:, speech], Xn[gb][:, speech]
        mag2 = np.abs(Xs) ** 2
        gd = np.real(Xns * np.conj(Xs)) / (mag2 + 1e-12) - NFFT / 2  # centre of window -> 0
        keep = mag2 > 1e-3 * mag2.max(0, keepdims=True)
        gd = np.where(keep, np.clip(gd, -NFFT / 2, NFFT / 2), np.nan)
        out["gd_std"] = np.nanmean(np.nanstd(gd, 0))
        out["gd_absmean"] = np.nanmean(np.abs(gd))
        if voiced.sum() >= 3:
            vs = voiced[speech]
            out["gd_voiced_std"] = np.nanmean(np.nanstd(gd[:, vs], 0))

    if voiced.sum() >= 3:
        Pv, f0v = logP[:, voiced], f0_frames[voiced]
        # H1-H2 (uncorrected for formants): peak dB near F0 minus peak near 2F0
        df = freqs[1]
        h = []
        for j in range(Pv.shape[1]):
            k1, k2 = int(round(f0v[j] / df)), int(round(2 * f0v[j] / df))
            w = max(1, int(0.1 * f0v[j] / df))
            if k2 + w < len(freqs):
                h.append(Pv[k1 - w:k1 + w + 1, j].max() - Pv[k2 - w:k2 + w + 1, j].max())
        if h:
            out["h1h2_db"] = float(np.median(h))
        # CPPS (Hillenbrand 1994; smoothed variant, Hillenbrand & Houde 1996): cepstral peak in the F0 quefrency range
        # above a linear regression baseline, in dB of the power cepstrum (Praat convention); log spectra smoothed
        # over 3 voiced frames first. NOT on Praat's absolute "Get CPPS" scale
        # (different window/baseline range) but ordering agrees on test vowels and it is ~10x cheaper; use relatively.
        from scipy.ndimage import uniform_filter1d
        Pv_s = uniform_filter1d(Pv, 3, axis=1)
        ceps = np.fft.irfft(Pv_s, axis=0)[: NFFT // 2]
        q = np.arange(NFFT // 2) / SR
        qb = (q >= 1 / F0_MAX) & (q <= 1 / F0_MIN)
        cq = 20 * np.log10(np.abs(ceps[qb]) + 1e-12)
        idx = cq.argmax(0)
        A = np.vstack([q[qb], np.ones(qb.sum())]).T
        coef, *_ = np.linalg.lstsq(A, cq, rcond=None)
        base = coef[0] * q[qb][idx] + coef[1]
        out["cpps_db"] = float(np.median(cq[idx, np.arange(cq.shape[1])] - base))
    return out


def _energy(y):
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y, (200, 200)), 400)[::HOP]
    return 10 * np.log10((frames ** 2).mean(1) + 1e-10)


def _respiration(e_db, f0_frames, X):
    """Speech/pause segmentation from frame energy; breath = unvoiced, non-silent, pause-internal noise."""
    out = {}
    n = min(len(e_db), len(f0_frames), X.shape[1])
    e_db, f0_frames = e_db[:n], f0_frames[:n]
    peak = np.percentile(e_db, 99)
    speech = e_db > peak - 35
    if speech.sum() < 10:
        return out, speech
    first, last = np.flatnonzero(speech)[[0, -1]]
    # ignore leading/trailing silence: it is an editing artefact (trivial shortcut), not physiology
    inner = np.zeros(n, bool)
    inner[first:last + 1] = True
    pause = inner & ~speech
    runs = [(a, b) for a, b in _runs(pause) if b - a >= 15]  # >= 150 ms
    span = (last - first + 1) * HOP / SR
    out["pause_frac"] = sum(b - a for a, b in runs) * HOP / SR / span
    out["pauses_per_s"] = len(runs) / span
    out["pause_mean_s"] = np.mean([b - a for a, b in runs]) * HOP / SR if runs else 0.0
    # energy of the quietest part of pauses re speech peak: digital zero vs room/breath noise
    out["pause_floor_db"] = (np.percentile(e_db[pause], 10) - peak) if pause.sum() >= 5 else np.nan
    # breath: an unvoiced stretch >= 200 ms (longer than most fricatives) that is clearly above the noise floor
    # but well below speech level (inhalation noise is typically 20-40 dB under vowels)
    floor = np.percentile(e_db[inner], 5)
    unv_runs = [(a, b) for a, b in _runs(inner & (f0_frames <= 0)) if b - a >= 20]
    breaths = sum(1 for a, b in unv_runs if floor + 10 < np.median(e_db[a:b]) < peak - 20)
    out["breath_per_min"] = breaths / span * 60
    t = np.arange(n)[speech] * HOP / SR
    out["energy_decl_db_s"] = np.polyfit(t, e_db[speech], 1)[0]
    return out, speech


def _am_spectrum(e_db, speech):
    """Modulation spectrum of the (linear) energy envelope at 100 Hz frame rate (Houtgast & Steeneken; Greenberg)."""
    out = {}
    if speech.sum() < 100:  # need >= ~1 s for a 0.5 Hz band to mean anything
        return out
    first, last = np.flatnonzero(speech)[[0, -1]]
    env = 10 ** (e_db[first:last + 1] / 20)
    env = (env - env.mean()) * np.hanning(len(env))
    S = np.abs(np.fft.rfft(env, n=max(1024, len(env)))) ** 2
    fm = np.fft.rfftfreq(max(1024, len(env)), HOP / SR)
    tot = S[(fm >= 0.5) & (fm < 32)].sum() + 1e-20
    for lo, hi in AM_BANDS:
        out[f"am_{lo:g}_{hi:g}"] = S[(fm >= lo) & (fm < hi)].sum() / tot
    sy = (fm >= 2) & (fm <= 10)
    out["am_peak_hz"] = fm[sy][S[sy].argmax()]
    return out


def _formants(snd, f0, t_pitch):
    out = {}
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) < 10:
        return out
    # Praat convention: ceiling 5000 Hz for adult male, 5500 for adult female; pick by median F0
    ceiling = 5000.0 if np.median(voiced_f0) < 160 else 5500.0
    fm = snd.to_formant_burg(time_step=HOP / SR, max_number_of_formants=5, maximum_formant=ceiling)
    ft = np.array(fm.xs())
    # only voiced frames: formants of noise/silence are meaningless
    vt = np.interp(ft, t_pitch, (f0 > 0).astype(float), left=0, right=0) > 0.5
    F = np.full((4, len(ft)), np.nan)
    for i, tt in enumerate(ft):
        if vt[i]:
            for k in range(4):
                F[k, i] = fm.get_value_at_time(k + 1, tt)
    ok = np.isfinite(F[:3]).all(0)
    if ok.sum() < 10:
        return out
    for k in range(3):
        out[f"f{k + 1}_mean"] = np.nanmean(F[k, ok])
        out[f"f{k + 1}_std"] = np.nanstd(F[k, ok])
    # velocity in Hz/s between consecutive voiced frames (articulators have finite speed)
    both = ok[1:] & ok[:-1]
    if both.sum() >= 5:
        vel = np.abs(np.diff(F[:3], axis=1))[:, both] / (HOP / SR)
        out["fvel_f1_med"] = np.median(vel[0])
        out["fvel_f2_med"] = np.median(vel[1])
        out["fvel_f2_p95"] = np.percentile(vel[1], 95)
        out["fvel_f3_p95"] = np.percentile(vel[2], 95)
        # >= 400 Hz change in 10 ms on any formant is faster than articulators move (~ tracking jump or splice)
        out["fjump_frac"] = (np.abs(np.diff(F[:3], axis=1))[:, both] > 400).any(0).mean()
    # apparent VTL per frame from formant spacing (Reby & McComb 2003): F_n = (2n-1) dF/2, VTL = c / (2 dF)
    ok4 = np.isfinite(F).all(0)
    if ok4.sum() >= 10:
        x = (2 * np.arange(1, 5) - 1) / 2.0
        dF = (x @ F[:, ok4]) / (x @ x)
        vtl = C_CM_S / (2 * dF)
        out["vtl_cm"] = np.median(vtl)
        out["vtl_cv"] = np.std(vtl) / np.mean(vtl)
    return out


def extract_bio(y, sr=SR):
    """Per-clip biology feature dict (fixed keys = FEATURES, NaN where undefined). Never raises on bad audio."""
    feats = dict.fromkeys(FEATURES, float("nan"))
    y = np.asarray(y, dtype=np.float64).ravel()
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = np.nan_to_num(y)
    if len(y) < NFFT or not np.any(y):
        return feats
    dur = len(y) / SR

    def run(fn, *a):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = fn(*a)
            feats.update({k: _nan(v) for k, v in r.items()})
        except Exception:  # ponytail: a failing group only NaNs its own keys; log if silent failures become a problem
            pass

    snd = parselmouth.Sound(y, sampling_frequency=SR)
    try:
        pitch, f0, tp = _pitch(snd)
    except Exception:
        return feats
    # align Praat pitch frames (centred at xs) to our 10 ms STFT grid (frame i centred at i*HOP/SR)
    e_db = _energy(y)
    grid = np.arange(len(e_db)) * HOP / SR
    f0_grid = np.interp(grid, tp, f0, left=0, right=0) if len(tp) else np.zeros(len(grid))
    f0_grid[np.interp(grid, tp, (f0 > 0).astype(float), left=0, right=0) < 0.5] = 0

    run(_f0_feats, f0, tp, dur)
    run(_perturbation, snd, pitch)
    X, Xn = _stft(y.astype(np.float32))
    speech = np.zeros(len(e_db), bool)
    try:
        r, speech = _respiration(e_db, f0_grid, X)
        feats.update({k: _nan(v) for k, v in r.items()})
    except Exception:
        pass
    run(_spectral_feats, X, Xn, f0_grid, speech)
    run(_am_spectrum, e_db, speech)
    run(_formants, snd, f0, tp)
    return feats
