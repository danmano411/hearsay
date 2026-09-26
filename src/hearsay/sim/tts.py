"""Full text-to-speech with open checkpoints that run on CPU. Each returns (wav float32, native sr)."""
import io
import zipfile
from functools import lru_cache

import numpy as np
import torch


def _espeak():
    """VITS-LJS phonemizes with espeak-ng; espeakng-loader ships the DLL + data (needs .venv-sim)."""
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


@lru_cache(maxsize=None)
def _vits(repo):
    from transformers import AutoTokenizer, VitsModel
    if repo == "kakao-enterprise/vits-ljs":
        _espeak()
    return AutoTokenizer.from_pretrained(repo), VitsModel.from_pretrained(repo).eval()


@torch.inference_mode()
def vits(text, repo, seed=0):
    """VITS: conditional VAE + normalizing flow + stochastic duration predictor + built-in HiFi-GAN decoder."""
    tok, model = _vits(repo)
    torch.manual_seed(seed)  # VITS samples the latent and the durations; seed makes each clip reproducible
    wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
    return wav.astype(np.float32), model.config.sampling_rate


@lru_cache(maxsize=1)
def _speecht5():
    from huggingface_hub import hf_hub_download
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
    proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").eval()
    voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
    # CMU ARCTIC x-vectors (512-d, speechbrain spkrec-xvect-voxceleb), the embeddings SpeechT5 was fine-tuned with.
    z = zipfile.ZipFile(hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset"))
    xv = {}
    for n in sorted(z.namelist()):
        if n.endswith(".npy"):
            xv.setdefault(n.split("/")[-1].split("-")[0].replace("cmu_us_", "").replace("_arctic", ""), []).append(n)
    return proc, model, voc, z, xv


def speecht5_speakers():
    return sorted(_speecht5()[4])


@torch.inference_mode()
def speecht5(text, speaker, seed=0):
    """SpeechT5: autoregressive transformer -> 80-bin log-mel -> HiFi-GAN; voice set by an x-vector."""
    proc, model, voc, z, xv = _speecht5()
    names = xv[speaker]
    emb = torch.from_numpy(np.load(io.BytesIO(z.read(names[seed % len(names)])))).float()[None]
    torch.manual_seed(seed)  # dropout in the prenet is kept on at inference (Tacotron-style), so seed it
    wav = model.generate_speech(proc(text=text, return_tensors="pt")["input_ids"], emb, vocoder=voc)
    return wav.numpy().astype(np.float32), 16000
