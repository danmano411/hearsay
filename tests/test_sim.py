import numpy as np

from hearsay.sim import copysyn


def test_griffin_lim_copysyn_keeps_length_level_and_pitch():
    sr = 16000
    t = np.arange(3 * sr) / sr
    y = (0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    out = copysyn.griffin_lim(y, sr, n_iter=8)
    assert out.shape == y.shape and out.dtype == np.float32
    assert np.isclose(np.sqrt(np.mean(out**2)), np.sqrt(np.mean(y**2)), rtol=1e-3)  # level is not a class cue
    spec = np.abs(np.fft.rfft(out))
    assert abs(np.argmax(spec) * sr / len(out) - 200) < 40  # 80-bin mel smears, but f0 survives
    assert not np.allclose(out, y, atol=1e-2)  # phase is re-estimated: waveform differs
