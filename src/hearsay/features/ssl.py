"""Frozen self-supervised speech embeddings (rung R4).

SSLEmbedder(model_id)(list of prep()'d 16 kHz waves) -> float32 array (batch, n_layers, 2 * dim):
per-layer [mean, std] over time of every hidden state (layer 0 = CNN/projection output, then each transformer block).
A batch is cropped to its shortest wave (no padding, see __call__).
"""
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

MODELS = {  # short name -> HF id
    "wavlm_base_plus": "microsoft/wavlm-base-plus",
    "xlsr_300m": "facebook/wav2vec2-xls-r-300m",
    "w2v2_base": "facebook/wav2vec2-base",
}
# ponytail: 4 s, not 6 s -- CPU cost is linear in length (0.28 vs 0.59 s/clip for WavLM) and the test median is
# 3.4 s, so 4 s is closer to what the heads will see at test time. Longer context only if a GPU appears.
MAX_SECONDS = 4.0
SR = 16000


def center(y, seconds=MAX_SECONDS):
    """Center crop to at most `seconds`; shorter clips are kept whole (no tiling)."""
    n = int(seconds * SR)
    if len(y) <= n:
        return y
    s = (len(y) - n) // 2
    return y[s:s + n]


class SSLEmbedder:
    def __init__(self, name, threads=None, max_blocks=None):
        if threads:
            torch.set_num_threads(threads)
        hf = MODELS.get(name, name)
        self.fe = AutoFeatureExtractor.from_pretrained(hf)
        self.model = AutoModel.from_pretrained(hf).eval()
        if max_blocks:  # keep only the first transformer blocks (hidden_states then has max_blocks + 1 entries)
            enc = self.model.encoder
            enc.layers = enc.layers[:max_blocks]
            # pre-LN (XLS-R "stable layer norm") applies encoder.layer_norm only to the LAST hidden state; drop it so
            # the truncated model's last entry equals block max_blocks of the full model, like every other entry
            if hasattr(enc, "layer_norm") and self.model.config.do_stable_layer_norm:
                enc.layer_norm = torch.nn.Identity()

    @torch.inference_mode()
    def __call__(self, waves):
        """No padding: the group-norm CNN front-end normalizes over padded zeros too, which shifted pooled features by
        ~10% in tests. Instead every wave is center-cropped to the batch's shortest; callers batch by similar length
        so this drops only milliseconds."""
        n = min(len(w) for w in waves)
        waves = [center(w, n / SR) for w in waves]
        x = self.fe(waves, sampling_rate=SR, return_tensors="pt")
        hs = torch.stack(self.model(x.input_values, output_hidden_states=True).hidden_states, 1)  # (B, L, T, D)
        return torch.cat([hs.mean(2), hs.std(2)], -1).numpy()
