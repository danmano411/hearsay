"""R4ft: end-to-end fine-tuned SSL front end + spoofing back end (plans/08_ssl_aasist.md §2, §3.3).

    model = build_model("xlsr_300m", backend="light", max_blocks=12)
    logit = model(wave_batch)            # (B,) spoof logit, higher = fake; waves are prep()'d 16 kHz, 64,000 samples

Front end (SSLFrontEnd): a HF AutoModel (XLS-R-300M by default) truncated to its first K transformer blocks exactly
like features.ssl.SSLEmbedder(max_blocks=K) (for pre-LN "stable layer norm" models the final encoder.layer_norm is
dropped so hidden state K is block K's raw output). The conv feature encoder is frozen and runs under no_grad;
optionally the lowest N blocks (and everything below them) are frozen too. Non-reentrant gradient checkpointing on the
transformer blocks, SpecAugment-style time masking (mask_time_prob 0.05, length 10), layerdrop 0.
The only input normalization is the SSL feature extractor's per-clip zero-mean/unit-variance (from that clip alone).

Back ends:
- "light" (A): softmax weights over the K+1 hidden states -> Linear(D, 256) -> attentive statistics pooling -> 1 logit.
- "aasist" (B): Tak et al. 2022 SSL-AASIST: Linear(D, 128) -> (1, 128, T) "spectrogram" -> max-pool -> residual
  encoder -> spectral/temporal graph attention -> HS-GAL -> 1 logit. The graph layers and residual block are imported
  in place from third_party/asvspoof5/Baseline-AASIST/models/AASIST.py (MIT, not copied), via aasist_wrap.module().
Back ends run in fp32 (autocast disabled) — they are tiny, and BatchNorm/variance pooling are safer in fp32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2Config, WavLMConfig

from hearsay.features.ssl import MODELS
from hearsay.models import aasist_wrap
from hearsay.models.ssl_e2e_data import CROP_S, N_SAMP, SR  # noqa: F401 (re-exported)

BACKENDS = ("light", "aasist")
_BACKEND_ALIASES = {"a": "light", "b": "aasist"}


def backend_name(name):
    name = _BACKEND_ALIASES.get(name.lower(), name.lower())
    if name not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS} (or A/B), got {name!r}")
    return name


def tiny_config(kind="wav2vec2", layers=24, dim=32):
    """Random-init debug front end (no download): same conv strides as XLS-R (20 ms hop, 199 frames per 4 s),
    pre-LN like XLS-R, 24 small blocks so K = 12 / 24 configs have the same shape as the real model."""
    Cfg = {"wav2vec2": Wav2Vec2Config, "wavlm": WavLMConfig}[kind]
    return Cfg(hidden_size=dim, num_hidden_layers=layers, num_attention_heads=2, intermediate_size=2 * dim,
               conv_dim=(16,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
               do_stable_layer_norm=True, feat_extract_norm="layer")


TINY = {"tiny": "wav2vec2", "tiny_wavlm": "wavlm"}


def _load_ssl(name, attn, overrides):
    """-> (HF model, do_normalize). Falls back to eager attention when the architecture has no SDPA path (WavLM)."""
    if name in TINY:
        cfg = tiny_config(TINY[name])
        for k, v in overrides.items():
            setattr(cfg, k, v)
        try:
            return AutoModel.from_config(cfg, attn_implementation=attn), True
        except ValueError:
            return AutoModel.from_config(cfg, attn_implementation="eager"), True
    hf = MODELS.get(name, name)
    norm = bool(getattr(AutoFeatureExtractor.from_pretrained(hf), "do_normalize", True))
    try:
        return AutoModel.from_pretrained(hf, attn_implementation=attn, **overrides), norm
    except ValueError:
        return AutoModel.from_pretrained(hf, attn_implementation="eager", **overrides), norm


def truncate(model, max_blocks):
    """Keep the first max_blocks transformer blocks (same as SSLEmbedder(max_blocks=)): hidden_states then has
    max_blocks + 1 entries, and for pre-LN models the last one is block max_blocks' raw output (layer_norm dropped)."""
    enc = model.encoder
    if max_blocks and max_blocks < len(enc.layers):
        enc.layers = enc.layers[:max_blocks]
        model.config.num_hidden_layers = max_blocks
    if hasattr(enc, "layer_norm") and model.config.do_stable_layer_norm:
        enc.layer_norm = nn.Identity()
    return model


class SSLFrontEnd(nn.Module):
    def __init__(self, name="xlsr_300m", max_blocks=12, freeze_blocks=0, grad_ckpt=True, mask_time_prob=0.05,
                 mask_time_length=10, attn="sdpa"):
        super().__init__()
        overrides = {"layerdrop": 0.0, "mask_time_prob": mask_time_prob, "mask_time_length": mask_time_length,
                     # HF's default mask_time_min_masks=2 would force >= 2 spans (20 frames = 10 % of 199) whatever
                     # the prob; 0 makes mask_time_prob=0.05 mean ~5 % of frames (one 10-frame span per 4 s clip)
                     "mask_time_min_masks": 0, "mask_feature_prob": 0.0}
        ssl, self.normalize = _load_ssl(name, attn, overrides)
        self.ssl = truncate(ssl, max_blocks)
        self.n_blocks = len(self.ssl.encoder.layers)
        self.dim = self.ssl.config.hidden_size
        self.attn = self.ssl.config._attn_implementation
        if grad_ckpt:  # non-reentrant: with frozen inputs, reentrant checkpointing silently drops gradients
            self.ssl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # conv feature encoder: frozen, no_grad (its activations on raw audio dominate memory), never checkpointed
        fe = self.ssl.feature_extractor
        self.ssl.freeze_feature_encoder()
        for mod in fe.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = False
        fe.forward = torch.no_grad()(fe.forward)
        self.freeze_blocks = int(freeze_blocks)
        if self.freeze_blocks:
            for p in self.bottom_params():
                p.requires_grad_(False)
            for layer in self.ssl.encoder.layers[:self.freeze_blocks]:
                layer.requires_grad_(False)

    def bottom_params(self):
        """Trainable-by-default params below block 1: feature projection, positional conv, mask embedding and (post-LN
        models) the encoder's input layer_norm. They get block 1's learning rate."""
        return [p for n, p in self.ssl.named_parameters()
                if not n.startswith("feature_extractor.") and not n.startswith("encoder.layers.")]

    def block_of(self, param_name):
        """1..K for params in transformer block i, 0 for bottom params, None for the frozen conv encoder."""
        if param_name.startswith("feature_extractor."):
            return None
        if param_name.startswith("encoder.layers."):
            return int(param_name.split(".")[2]) + 1
        return 0

    def forward(self, x):
        """x: (B, samples) float -> tuple of K+1 hidden states (B, T, D)."""
        if self.normalize:  # Wav2Vec2FeatureExtractor's zero_mean_unit_var_norm, per clip
            x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-7)
        return self.ssl(x, output_hidden_states=True).hidden_states


class LightBackEnd(nn.Module):
    """A: softmax layer weights over K+1 hidden states -> Linear(D, 256) -> attentive statistics pooling -> 1 logit."""

    def __init__(self, n_states, dim, hid=256, att=128):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_states))
        self.proj = nn.Linear(dim, hid)
        self.att = nn.Sequential(nn.Linear(hid, att), nn.Tanh(), nn.Linear(att, 1))
        self.out = nn.Linear(2 * hid, 1)

    def forward(self, hs):
        h = torch.stack([t.float() for t in hs], 0)  # (L, B, T, D)
        h = (torch.softmax(self.layer_w, 0).view(-1, 1, 1, 1) * h).sum(0)
        h = self.proj(h)
        a = torch.softmax(self.att(h), dim=1)  # (B, T, 1)
        mu = (a * h).sum(1)
        sd = ((a * h * h).sum(1) - mu * mu).clamp(min=1e-5).sqrt()
        return self.out(torch.cat([mu, sd], -1)).squeeze(-1)


class AASISTBackEnd(nn.Module):
    """B: Tak-2022 SSL-AASIST on the last hidden state. Graph/residual modules imported from the organizers' AASIST.py;
    gat_dims / pool_ratios / temperatures from its AASIST_ASVspoof5.conf. As in Tak's SSL variant, the residual blocks'
    per-block (1, 3) time max-pool is disabled (199 frames -> 66 after the stem; six /3 pools would reach 0), and the
    spectral/temporal node features come from an attention-weighted sum instead of AASIST's max."""

    def __init__(self, dim, lin=128):
        super().__init__()
        A = aasist_wrap.module()
        conf = aasist_wrap.model_config()
        gat, pr, temp = conf["gat_dims"], conf["pool_ratios"], conf["temperatures"]
        self.LL = nn.Linear(dim, lin)
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        filts = [[1, 32], [32, 32], [32, 64], [64, 64], [64, 64], [64, 64]]
        self.encoder = nn.Sequential(*[nn.Sequential(A.Residual_block(nb_filts=f, first=i == 0))
                                       for i, f in enumerate(filts)])
        for blk in self.encoder:
            blk[0].mp = nn.Identity()
        c = filts[-1][-1]
        self.attention = nn.Sequential(nn.Conv2d(c, 128, (1, 1)), nn.SELU(inplace=True), nn.BatchNorm2d(128),
                                       nn.Conv2d(128, c, (1, 1)))
        self.pos_S = nn.Parameter(torch.randn(1, lin // 3, c))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat[0]))
        self.GAT_layer_S = A.GraphAttentionLayer(c, gat[0], temperature=temp[0])
        self.GAT_layer_T = A.GraphAttentionLayer(c, gat[0], temperature=temp[1])
        self.HtrgGAT_layer_ST11 = A.HtrgGraphAttentionLayer(gat[0], gat[1], temperature=temp[2])
        self.HtrgGAT_layer_ST12 = A.HtrgGraphAttentionLayer(gat[1], gat[1], temperature=temp[2])
        self.HtrgGAT_layer_ST21 = A.HtrgGraphAttentionLayer(gat[0], gat[1], temperature=temp[2])
        self.HtrgGAT_layer_ST22 = A.HtrgGraphAttentionLayer(gat[1], gat[1], temperature=temp[2])
        self.pool_S = A.GraphPool(pr[0], gat[0], 0.3)
        self.pool_T = A.GraphPool(pr[1], gat[0], 0.3)
        self.pool_hS1 = A.GraphPool(pr[2], gat[1], 0.3)
        self.pool_hT1 = A.GraphPool(pr[2], gat[1], 0.3)
        self.pool_hS2 = A.GraphPool(pr[2], gat[1], 0.3)
        self.pool_hT2 = A.GraphPool(pr[2], gat[1], 0.3)
        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)
        self.out_layer = nn.Linear(5 * gat[1], 1)

    def _hs_gal(self, out_T, out_S, master, l1, l2, pool_S, pool_T):
        out_T1, out_S1, master1 = l1(out_T, out_S, master=master)
        out_S1, out_T1 = pool_S(out_S1), pool_T(out_T1)
        t_aug, s_aug, m_aug = l2(out_T1, out_S1, master=master1)
        return out_T1 + t_aug, out_S1 + s_aug, master1 + m_aug

    def forward(self, hs):
        x = self.LL(hs[-1].float())  # (B, T, 128)
        x = F.max_pool2d(x.transpose(1, 2).unsqueeze(1), (3, 3))  # (B, 1, 42, T/3)
        e = self.encoder(self.selu(self.first_bn(x)))  # (B, C, 42, T/3)
        w = self.attention(e)
        e_S = (e * torch.softmax(w, -1)).sum(-1).transpose(1, 2) + self.pos_S  # spectral nodes (B, 42, C)
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        e_T = (e * torch.softmax(w, -2)).sum(-2).transpose(1, 2)  # temporal nodes (B, T/3, C)
        out_T = self.pool_T(self.GAT_layer_T(e_T))
        T1, S1, M1 = self._hs_gal(out_T, out_S, self.master1, self.HtrgGAT_layer_ST11, self.HtrgGAT_layer_ST12,
                                  self.pool_hS1, self.pool_hT1)
        T2, S2, M2 = self._hs_gal(out_T, out_S, self.master2, self.HtrgGAT_layer_ST21, self.HtrgGAT_layer_ST22,
                                  self.pool_hS2, self.pool_hT2)
        out_T = torch.max(self.drop_way(T1), self.drop_way(T2))
        out_S = torch.max(self.drop_way(S1), self.drop_way(S2))
        master = torch.max(self.drop_way(M1), self.drop_way(M2))
        h = torch.cat([out_T.abs().max(1)[0], out_T.mean(1), out_S.abs().max(1)[0], out_S.mean(1),
                       master.squeeze(1)], 1)
        return self.out_layer(self.drop(h)).squeeze(-1)


class SSLSpoofModel(nn.Module):
    def __init__(self, frontend, backend="light"):
        super().__init__()
        self.backend_name = backend_name(backend)
        self.frontend = frontend
        if self.backend_name == "light":
            self.backend = LightBackEnd(frontend.n_blocks + 1, frontend.dim)
        else:
            self.backend = AASISTBackEnd(frontend.dim)

    def forward(self, x, frontend_grad=True):
        """-> (B,) spoof logit (higher = fake). frontend_grad=False runs the front end under no_grad (back-end warm-up)."""
        with torch.set_grad_enabled(frontend_grad and torch.is_grad_enabled()):
            hs = self.frontend(x)
        with torch.autocast(x.device.type, enabled=False):
            return self.backend(hs)


def build_model(name="xlsr_300m", backend="light", max_blocks=12, freeze_blocks=0, grad_ckpt=True,
                mask_time_prob=0.05, mask_time_length=10, attn="sdpa"):
    fe = SSLFrontEnd(name, max_blocks, freeze_blocks, grad_ckpt, mask_time_prob, mask_time_length, attn)
    return SSLSpoofModel(fe, backend)


def param_groups(model, lr=2e-5, decay=0.85, backend_lr=1e-3, weight_decay=0.01):
    """AdamW groups: one per front-end depth with layer-wise LR decay (block K gets lr, block i gets lr * decay^(K-i),
    bottom params get block 1's LR), then the back end at backend_lr. No weight decay on biases/norms (ndim <= 1).
    Each group carries 'base_lr' and 'kind' ('front'/'back') for the scheduler. Frozen params are left out."""
    fe = model.frontend
    K = fe.n_blocks
    buckets = {}
    for n, p in fe.ssl.named_parameters():
        b = fe.block_of(n)
        if b is None or not p.requires_grad:
            continue
        buckets.setdefault(("front", max(b, 1)), []).append((n, p))
    buckets[("back", 0)] = [(n, p) for n, p in model.backend.named_parameters() if p.requires_grad]
    groups = []
    for (kind, depth), named in sorted(buckets.items(), key=lambda kv: (kv[0][0] == "back", kv[0][1])):
        base = backend_lr if kind == "back" else lr * decay ** (K - depth)
        for wd, sel in ((weight_decay, [p for n, p in named if p.ndim > 1]),
                        (0.0, [p for n, p in named if p.ndim <= 1])):
            if sel:
                groups.append({"params": sel, "lr": base, "base_lr": base, "weight_decay": wd, "kind": kind,
                               "name": f"{kind}{depth if kind == 'front' else ''}{'' if wd else '_nd'}"})
    return groups
