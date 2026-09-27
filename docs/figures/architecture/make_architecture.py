"""Architecture flowcharts for every rung (docs/06_architecture.md). Regenerate: python docs/figures/architecture/make_architecture.py

Hand-laid SVG, no dependencies. Colour key (same in every chart): grey = data, blue = preprocessing / features,
teal = trained model part, dashed = frozen pretrained part, amber = fusion / decision, dark = output.
Numbers come from the code and reports cited in docs/06_architecture.md.
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
W = 900
STYLE = {
    "data": ("#eef1f4", "#6b7784", ""),
    "pre": ("#e3edf8", "#3d6fa3", ""),
    "model": ("#dff1ef", "#0c6b6b", ""),
    "frozen": ("#f4f8f8", "#0c6b6b", "6 4"),
    "fuse": ("#fbf0dc", "#a8620f", ""),
    "out": ("#1b2430", "#1b2430", ""),
    "note": ("#ffffff", "#c5ccd3", "3 3"),
}


class Chart:
    def __init__(self, title, subtitle, height):
        self.h, self.parts, self.boxes = height, [], {}
        self.parts.append(f'<text x="24" y="34" class="t">{escape(title)}</text>')
        self.parts.append(f'<text x="24" y="56" class="s">{escape(subtitle)}</text>')

    def box(self, key, x, y, w, h, title, lines=(), kind="model"):
        fill, stroke, dash = STYLE[kind]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="1.6"{d}/>')
        tc = "#ffffff" if kind == "out" else "#1b2430"
        self.parts.append(f'<text x="{x + 12}" y="{y + 21}" class="bt" fill="{tc}">{escape(title)}</text>')
        for i, ln in enumerate(lines):
            self.parts.append(f'<text x="{x + 12}" y="{y + 40 + 16 * i}" class="bl" '
                              f'fill="{"#dfe6ec" if kind == "out" else "#3a4652"}">{escape(ln)}</text>')
        self.boxes[key] = (x, y, w, h)

    def arrow(self, a, b, label=None, dx_a=0, dx_b=0):
        ax, ay, aw, ah = self.boxes[a]
        bx, by, bw, bh = self.boxes[b]
        x1, y1 = ax + aw / 2 + dx_a, ay + ah
        x2, y2 = bx + bw / 2 + dx_b, by
        if abs(x1 - x2) < 1:
            path = f"M{x1},{y1} L{x2},{y2 - 2}"
        else:
            ym = (y1 + y2) / 2
            path = f"M{x1},{y1} L{x1},{ym} L{x2},{ym} L{x2},{y2 - 2}"
        self.parts.append(f'<path d="{path}" fill="none" stroke="#5d6975" stroke-width="1.6" marker-end="url(#ah)"/>')
        if label:
            lx = (x1 + x2) / 2 + 8 if abs(x1 - x2) < 1 else x2 + 8
            self.parts.append(f'<text x="{lx}" y="{(y1 + y2) / 2 + 4}" class="al">{escape(label)}</text>')

    def text(self, x, y, s, cls="bl"):
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}">{escape(s)}</text>')

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" width="{W}" height="{self.h}" '
                f'font-family="Segoe UI, Helvetica, Arial, sans-serif">'
                '<style>.t{font-size:20px;font-weight:700;fill:#1b2430}.s{font-size:13px;fill:#5d6975}'
                '.bt{font-size:14px;font-weight:600}.bl{font-size:12px}.al{font-size:11.5px;fill:#5d6975;font-style:italic}'
                '.k{font-size:11px;fill:#5d6975}</style>'
                '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5d6975"/></marker></defs>'
                f'<rect width="{W}" height="{self.h}" fill="#ffffff"/>')
        key = [("data", "data"), ("pre", "preprocessing / features"), ("model", "trained"),
               ("frozen", "frozen, pretrained"), ("fuse", "fusion / decision"), ("out", "output")]
        x = 24
        legend = []
        for kind, lab in key:
            fill, stroke, dash = STYLE[kind]
            d = f' stroke-dasharray="{dash}"' if dash else ""
            legend.append(f'<rect x="{x}" y="{self.h - 26}" width="14" height="12" rx="2" fill="{fill}" '
                          f'stroke="{stroke}"{d}/><text x="{x + 20}" y="{self.h - 16}" class="k">{lab}</text>')
            x += 30 + 6.6 * len(lab)
        (OUT / f"{name}.svg").write_text(head + "".join(self.parts) + "".join(legend) + "</svg>", encoding="utf-8")


def prep_box(c, key, x, y, w=420):
    c.box(key, x, y, w, 76, "prep()  (every clip, train and test)",
          ["DC removal → trim silence (40 dB) → 7 kHz low-pass", "(10th-order Butterworth) → RMS normalize to 0.05"], "pre")


def r0():
    c = Chart("R0 · trivial cues (a leak detector, not a real model)",
              "If 9 non-speech statistics can separate real from fake, the data leaks. Run twice: raw audio, then after prep().", 560)
    c.box("in", 240, 80, 420, 50, "Audio clip, 16 kHz mono", [], "data")
    c.box("raw", 60, 170, 330, 60, "Path A: raw audio", ["no preprocessing"], "data")
    prep_box(c, "prep", 470, 162, 370)
    c.box("st", 170, 280, 560, 92, "9 trivial statistics per clip",
          ["duration · peak level · RMS · clipping fraction · DC offset", "leading silence · trailing silence · 95 % roll-off · bandwidth (bw60)"], "pre")
    c.box("clf", 250, 410, 400, 60, "Depth-3 decision tree  or  LightGBM", ["trained on the train split"], "model")
    c.box("o", 300, 496, 300, 34, "p(fake)", [], "out")
    for a, b in (("in", "raw"), ("in", "prep"), ("raw", "st"), ("prep", "st"), ("st", "clf"), ("clf", "o")):
        c.arrow(a, b)
    c.text(660, 438, "raw: 0.572 (leaks)", "al")
    c.text(660, 454, "after prep(): 0.921 (≈ chance: good)", "al")
    c.save("r0_trivial_cues")


def r1():
    c = Chart("R1 · classic features + LightGBM",
              "228 hand-made features per clip, gradient-boosted trees. Headline 0.253 (EER 6.6 %).", 780)
    c.box("in", 240, 76, 420, 46, "Audio clip, 16 kHz mono", [], "data")
    c.box("aug", 240, 150, 420, 60, "augment()  (training only, same for both classes)",
          ["MP3 round trip 25 % · noise 15–40 dB SNR 20 % · resample 15 %"], "pre")
    prep_box(c, "prep", 240, 234, 420)
    c.box("crop", 240, 334, 420, 46, "Crop: random 3–4 s (train) · centre 4 s (eval)", [], "pre")
    c.box("sp", 60, 410, 380, 108, "Spectral features (180), 0–7 kHz",
          ["LFCC 20 + deltas (mean, std)", "MFCC 20 + deltas (mean, std)", "centroid · bandwidth · roll-off · flatness",
           "6-band spectral contrast"], "pre")
    c.box("bio", 460, 410, 380, 108, "Speech-biology features (48)",
          ["phonation / pitch 17 (jitter, shimmer, HNR, tilt…)", "articulation 13 (formants, their speed, VTL)",
           "rhythm 6 · respiration 6 · turbulence 3 (4–7 kHz)", "phase 3 (group delay)"], "pre")
    c.box("cat", 300, 550, 300, 40, "228-dim feature vector", [], "pre")
    c.box("gbm", 240, 616, 420, 62, "LightGBM (gradient-boosted decision trees)",
          ["class-balanced · leaves chosen on val_testlike · early stopping"], "model")
    c.box("o", 330, 700, 240, 34, "p(fake)", [], "out")
    for a, b in (("in", "aug"), ("aug", "prep"), ("prep", "crop"), ("crop", "sp"), ("crop", "bio"),
                 ("sp", "cat"), ("bio", "cat"), ("cat", "gbm"), ("gbm", "o")):
        c.arrow(a, b)
    c.save("r1_classic_lightgbm")


def r3():
    c = Chart("R3 · AASIST (the organizers' baseline), used zero-shot",
              "Graph attention over spectral and temporal nodes. Pretrained on ASVspoof 5; on our data 0.823 (does not transfer).", 1010)
    c.box("in", 230, 76, 440, 50, "Raw waveform, 64,600 samples (≈ 4.04 s)", ["short clips tiled, long clips cropped"], "data")
    c.box("sinc", 230, 150, 440, 62, "Sinc convolution: 70 learnable band-pass filters",
          ["each filter = a sinc function with learned cut-offs (kernel 128)"], "frozen")
    c.box("mp", 230, 236, 440, 46, "|x| → max-pool 3×3 → batch norm → SELU", [], "frozen")
    c.box("enc", 230, 306, 440, 62, "Residual CNN encoder: 6 blocks",
          ["channels 32 · 32 · 64 · 64 · 64 · 64  →  feature map (channels × freq × time)"], "frozen")
    c.box("S", 60, 400, 370, 78, "Spectral graph",
          ["nodes = frequency bands (max over time)", "+ position embedding → graph attention → pool"], "frozen")
    c.box("T", 470, 400, 370, 78, "Temporal graph",
          ["nodes = time steps (max over frequency)", "graph attention → pool"], "frozen")
    c.box("h1", 60, 512, 370, 94, "HS-GAL branch 1",
          ["heterogeneous graph: spectral + temporal", "nodes together, plus a learned 'master' node", "that attends to both · 2 layers + pooling"], "frozen")
    c.box("h2", 470, 512, 370, 94, "HS-GAL branch 2",
          ["same structure, its own weights and", "master node", "2 layers + pooling"], "frozen")
    c.box("mx", 230, 640, 440, 46, "Element-wise max of the two branches", [], "frozen")
    c.box("ro", 230, 710, 440, 78, "Readout → 160 values",
          ["max and mean of temporal nodes · max and mean of", "spectral nodes · master node (5 × 32)"], "frozen")
    c.box("lin", 230, 812, 440, 46, "Linear 160 → 2 logits (spoof, bonafide)", [], "frozen")
    c.box("o", 230, 884, 440, 50, "Our score = logit(spoof) − logit(bonafide)", ["higher = more fake"], "out")
    for a, b in (("in", "sinc"), ("sinc", "mp"), ("mp", "enc"), ("enc", "S"), ("enc", "T"),
                 ("S", "h1"), ("T", "h1"), ("S", "h2"), ("T", "h2"), ("h1", "mx"), ("h2", "mx"), ("mx", "ro"),
                 ("ro", "lin"), ("lin", "o")):
        c.arrow(a, b)
    c.text(24, 960, "Every block is frozen: we used the organizers' weights as-is, after prep(), one 4 s window per clip.", "al")
    c.save("r3_aasist")


def r4ft():
    c = Chart("R4ft · XLS-R-300M, fine-tuned end to end (the core model)",
              "A self-supervised speech network pretrained on 436k hours in 128 languages, cut to 12 of 24 blocks. Headline 0.0282.", 1010)
    c.box("in", 230, 76, 440, 46, "Audio clip, 16 kHz mono", [], "data")
    prep_box(c, "prep", 230, 146, 440)
    c.box("win", 230, 246, 440, 60, "4 s window = 64,000 samples",
          ["inference 'win3': up to 3 evenly spaced windows, logits averaged"], "pre")
    c.box("cnn", 150, 336, 600, 62, "XLS-R feature encoder: 7 convolution layers (frozen)",
          ["raw samples → one 512-dim vector every 20 ms → 199 frames"], "frozen")
    c.box("proj", 150, 422, 600, 46, "Feature projection 512 → 1024  +  positional convolution", [], "model")
    c.box("tr", 150, 492, 600, 94, "Transformer blocks 1–12 of 24 (fine-tuned; 13–24 removed)",
          ["each block: 16-head self-attention + feed-forward (1024 → 4096 → 1024)", "every frame can look at every other frame in the 4 s",
           "outputs 13 hidden states (projection + 12 blocks), each 199 × 1024"], "model")
    c.box("lw", 150, 620, 600, 46, "Learned softmax weights over the 13 layers → one 199 × 1024 sequence", [], "model")
    c.box("lin", 150, 690, 600, 46, "Linear 1024 → 256 per frame", [], "model")
    c.box("asp", 150, 760, 600, 78, "Attentive statistics pooling",
          ["a small net (256 → 128 → tanh → 1) scores each frame; softmax over frames", "weighted mean (256) + weighted std (256) → 512"], "model")
    c.box("out1", 150, 862, 600, 46, "Linear 512 → 1  →  logit (higher = more fake)", [], "model")
    c.box("o", 300, 932, 300, 34, "score (margin)", [], "out")
    for a, b in (("in", "prep"), ("prep", "win"), ("win", "cnn"), ("cnn", "proj"), ("proj", "tr"), ("tr", "lw"),
                 ("lw", "lin"), ("lin", "asp"), ("asp", "out1"), ("out1", "o")):
        c.arrow(a, b)
    c.text(770, 360, "pretrained", "al")
    c.text(770, 520, "pretrained,", "al")
    c.text(770, 536, "then fine-tuned", "al")
    c.text(770, 700, "new 'light'", "al")
    c.text(770, 716, "back end,", "al")
    c.text(770, 732, "trained from", "al")
    c.text(770, 748, "scratch", "al")
    c.save("r4ft_xlsr")


def r6():
    c = Chart("R6 · same network as R4ft, how it is trained (and what R6 changed)",
              "The training loop both XLS-R models used. R6 = seed 1 + 6,155 more clips; checkpoint kept at step 4,096.", 880)
    c.box("data", 60, 76, 380, 94, "Training split (135k clips + R6's 6,155)",
          ["R6 adds: ASVspoof 2021 DF (2 samples),", "CD-ADD, DECRO-en", "never: DeepVoice, test sets"], "data")
    c.box("samp", 460, 76, 380, 94, "Balanced sampler",
          ["groups = (source, label), drawn ∝ √size", "each batch: 4 real + 4 fake", "ASVspoof 5 reals × 2"], "pre")
    c.box("aug", 230, 200, 440, 62, "augment() → prep() → random 4 s crop",
          ["MP3 / noise / resample, same for both classes"], "pre")
    c.box("net", 230, 292, 440, 62, "XLS-R (12 blocks) + light back end",
          ["exactly the R4ft architecture (see r4ft_xlsr.svg)"], "model")
    c.box("loss", 230, 384, 440, 46, "Binary cross-entropy on the logit", [], "model")
    c.box("opt", 150, 460, 600, 94, "AdamW, 4 micro-batches accumulated = 32 clips per step",
          ["learning rate 2e-5 at block 12, × 0.85 per block downward; back end 1e-3",
           "steps 0–300: only the back end learns · warm-up 500 · cosine decay",
           "bf16 mixed precision + gradient checkpointing: 4.9 GB on an 8 GB GPU"], "model")
    c.box("ev", 150, 584, 600, 62, "Every 1,024 steps: score val_testlike (8,963 clips)",
          ["keep best.pth if combined minDCF improves by ≥ 0.002; stop after 4 evals without"], "fuse")
    c.box("ck", 230, 676, 440, 78, "R6 result: best checkpoint = step 4,096",
          ["val 0.0267 (centre) / 0.0206 (win3) · R4ft: 0.0166 / 0.0121", "stopped at step ~7,150 (memory safeguard; recorded deviation)"], "out")
    c.arrow("data", "aug", dx_b=-110)
    c.arrow("samp", "aug", dx_b=110)
    for a, b in (("aug", "net"), ("net", "loss"), ("loss", "opt"), ("opt", "ev"), ("ev", "ck")):
        c.arrow(a, b)
    c.parts.append('<path d="M150,515 L110,515 L110,323 L228,323" fill="none" stroke="#5d6975" stroke-width="1.4" '
                   'stroke-dasharray="4 3" marker-end="url(#ah)"/>')
    c.text(40, 420, "update", "al")
    c.text(40, 436, "weights", "al")
    c.save("r6_training")


def fusion(name, title, subtitle, inputs, weights_line, intercept, temp, h=740):
    c = Chart(title, subtitle, h)
    n = len(inputs)
    bw, gap = (260, 30) if n == 3 else (360, 60)
    x0 = (W - (n * bw + (n - 1) * gap)) / 2
    for i, (key, t, lines, tr) in enumerate(inputs):
        x = x0 + i * (bw + gap)
        c.box(key, x, 76, bw, 80, t, lines, "model")
        c.box(key + "_t", x, 186, bw, 62, "transform", [tr], "pre")
        c.box(key + "_z", x, 278, bw, 62, "standardize", ["z = (x − μ) / σ, from val_testlike"], "pre")
        c.arrow(key, key + "_t")
        c.arrow(key + "_t", key + "_z")
    c.box("lr", 150, 380, 600, 94, "Logistic regression (the fusion)",
          [weights_line, f"intercept {intercept}; fit on val_testlike only", "(its honest score comes from 5-fold group cross-fitting)"], "fuse")
    for key, *_ in inputs:
        c.arrow(key + "_z", "lr")
    c.box("m", 250, 504, 400, 46, "fused margin (higher = more fake)", [], "fuse")
    c.box("sig", 150, 580, 600, 62, f"sigmoid(margin / {temp}) → p(fake), then write 1 − p",
          [f"temperature {temp}: stops ties at 10 digits · 1 − p: the scorer reads higher = real"], "pre")
    c.box("o", 250, 664, 400, 34, "submission .tsv (1,671 rows)", [], "out")
    c.arrow("lr", "m")
    c.arrow("m", "sig")
    c.arrow("sig", "o")
    c.save(name)


def main():
    r0(); r1(); r3(); r4ft(); r6()
    fusion("r5_fusion", "R5 · fusion of R4ft and R1 (first submission: HGT minDCF 0.0584)",
           "Each model gives a score; a logistic regression learns how much to trust each one.",
           [("a", "R4ft: XLS-R, win3", ["logit, higher = fake", "(r4ft_xlsr.svg)"], "margin: used as-is"),
            ("b", "R1: LightGBM", ["p(fake) in [0, 1]", "(r1_classic_lightgbm.svg)"], "logit: log(p / (1 − p))")],
           "8.82 × z(R4ft) + 1.99 × z(R1)   (μ, σ: R4ft −4.83, 16.73 · R1 −0.46, 6.13)", "−1.02", 8)
    fusion("e5_fusion", "E5 · fusion of R4ft, R6 and R1 (final submission)",
           "Two independently trained XLS-R models plus the classic model; chosen by the pre-registered plan-10 rule.",
           [("a", "R4ft: XLS-R, win3", ["logit, higher = fake", "seed 0, original data"], "margin: as-is"),
            ("b", "R6: XLS-R, win3", ["logit, higher = fake", "seed 1, + 6,155 clips"], "margin: as-is"),
            ("c", "R1: LightGBM (refit)", ["p(fake) in [0, 1]", "228 features"], "logit: log(p / (1 − p))")],
           "6.00 × z(R4ft) + 3.92 × z(R6) + 1.69 × z(R1)", "−1.31", 7)
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
