"""Figure for reports/04_r4ft_xlsr.md: R4ft run 1 val_testlike combined minDCF per eval.

    PYTHONPATH=src python scripts/plot_r4ft_curve.py    # reads data/models/r4ft_xlsr/R4ft_xlsr_light/log.jsonl
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

from hearsay.audio import DATA  # noqa: E402

log = DATA / "models" / "r4ft_xlsr" / "R4ft_xlsr_light" / "log.jsonl"
out = Path(__file__).resolve().parents[1] / "reports" / "figures" / "r4ft_xlsr_curve.png"
ev = [r for r in map(json.loads, open(log)) if r["type"] == "eval"]
x = [r["step"] for r in ev]; y = [r["combined"] for r in ev]
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150, facecolor=BG)
ax.set_facecolor(BG)
ax.set_yscale("log")
ax.axhline(0.228, color=INK2, lw=1.5, ls="--")
ax.text(9950, 0.228 * 1.07, "R1 LightGBM (best classic): 0.228", ha="right", va="bottom", color=INK2)
ax.axhline(0.05, color="#a8a7a1", lw=1.2, ls=":")
ax.text(9950, 0.05 * 1.07, "0.05 (owner's stop target, on the headline set)", ha="right", va="bottom", color=INK2,
        fontsize=8.5)
ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=6, mec=BG, mew=1.5, zorder=3, label="R4ft val_testlike, centre window")
kept = ev[[r["step"] for r in ev].index(6144)]
ax.plot([6144], [kept["combined"]], marker="o", ms=13, mfc="none", mec=ORANGE, mew=2, zorder=4)
ax.annotate(f"kept checkpoint, step 6144: {kept['combined']:.4f}\n(last gain >= 0.002)", (6144, kept["combined"]),
            xytext=(4700, 0.0085), color=INK, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.plot([6144], [0.0121], marker="D", ms=7, color=ORANGE, mec=BG, mew=1.2, zorder=4)
ax.annotate("same checkpoint, 3 windows: 0.0121\n(inference choice, val_testlike)", (6144, 0.0121),
            xytext=(8300, 0.0085), color=INK, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.axvline(10240, color=INK2, lw=1, ls="-.")
ax.text(10150, 0.12, "early stop, step 10240\n(4 evals without a\n>= 0.002 gain)", ha="right", va="top", color=INK2,
        fontsize=8.5)
ax.annotate("back-end warm-up only\n(front end frozen)", (300, y[0]), xytext=(1400, 0.15), color=INK2, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.set_xlim(-200, 10700); ax.set_ylim(0.0065, 0.33)
ticks = [0.01, 0.02, 0.05, 0.1, 0.2]
ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xticks(range(0, 10241, 1024))
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_xlabel("optimizer step (1,024 steps = 32,768 clips = one virtual epoch)")
ax.set_ylabel("combined minDCF (log scale, lower is better)")
ax.set_title("R4ft XLS-R fine-tune: val_testlike combined minDCF per eval", loc="left", color=INK, fontsize=11.5)
fig.tight_layout(); fig.savefig(out, facecolor=BG); print(out)
