"""Copy-synthesis: real wav -> mel -> vocoder. Same speaker, text and channel as the real clip, so the
vocoder is the only thing that differs (hardest possible negatives for a detector).
"""
from functools import lru_cache

import librosa
import numpy as np
import torch

from hearsay.audio import SR

DEVICE = torch.device("cpu")  # set by scripts/run_sim.py --device before the first model load (Griffin-Lim stays CPU)
# Mel settings for Griffin-Lim, chosen to match a typical 16 kHz TTS front end (80 bins, 1024/256).
N_FFT, HOP, N_MELS = 1024, 256, 80


def mel(y, sr=SR):
    return librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=1.0)


def griffin_lim(y, sr=SR, n_iter=32):
    """wav -> 80-bin magnitude mel -> pseudo-inverse -> Griffin-Lim phase estimate (the Tacotron-1 vocoder)."""
    out = librosa.feature.inverse.mel_to_audio(mel(y, sr), sr=sr, n_fft=N_FFT, hop_length=HOP, power=1.0,
                                               n_iter=n_iter)
    return _match(out, y)


@lru_cache(maxsize=1)
def _speecht5():
    from transformers import SpeechT5FeatureExtractor, SpeechT5HifiGan
    fe = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")
    voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval().to(DEVICE)
    return fe, voc


@torch.inference_mode()
def hifigan(y, sr=SR):
    """wav -> SpeechT5 log-mel (80 bins, 80-7600 Hz, 1024/256 @16 kHz) -> SpeechT5 HiFi-GAN V1 vocoder."""
    assert sr == 16000, "SpeechT5 HiFi-GAN is a 16 kHz vocoder"
    fe, voc = _speecht5()
    m = fe(audio_target=y, sampling_rate=sr, return_tensors="pt")["input_values"][0]  # (frames, 80)
    return _match(voc(m.to(DEVICE)).cpu().numpy(), y)


def _match(out, ref):
    """Trim/pad to the reference length and match its RMS, so duration and level are not class cues."""
    out = np.pad(out, (0, max(0, len(ref) - len(out))))[: len(ref)]
    rms_o, rms_r = np.sqrt(np.mean(out**2)) + 1e-9, np.sqrt(np.mean(ref**2))
    return (out * rms_r / rms_o).astype(np.float32)
