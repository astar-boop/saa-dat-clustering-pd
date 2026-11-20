"""Figure S6 (corrected): reliability of the common kinetic parameters and the
specification curve of the MoCA coefficient. Values from the project outputs
26_fig14_drawn.csv and 24_multiverse.csv (run_sporadici/outputs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from matplotlib.ticker import FixedLocator

PALETTE = {"blue_main": "#0F4D92", "red_strong": "#B64342", "neutral": "#CFCECE",
           "gray_mid": "#767676", "gray_dark": "#4D4D4D"}

@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2
    use_tex: bool = False
    font_family: tuple = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")

def apply_publication_style(style=None):
    s = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(s.font_family),
        "font.size": s.font_size, "axes.linewidth": s.axes_linewidth,
        "xtick.major.width": s.axes_linewidth, "ytick.major.width": s.axes_linewidth,
        "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "text.usetex": s.use_tex, "figure.facecolor": "white", "savefig.facecolor": "white",
    })

def finalize_figure(fig, out_path, formats, dpi=300, pad=1):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    for f in formats:
        fig.savefig(out_path.with_suffix("." + f), dpi=dpi)
    plt.close(fig)

# ---- data: read from the outputs of 26_review_figures.py and 24_multiverse.py
import os, pandas as pd
BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
params = ["Fmax", "TTT", "AUC", "SLOPE"]
_icc = pd.read_csv(os.path.join(OUT, "26_fig14_drawn.csv")).set_index(['protocol', 'parameter']).ICC_1_3_drawn
icc24 = [float(_icc[("24h", k)]) for k in params]
icc150 = [float(_icc[("150h", k)]) for k in params]
p = np.sort(pd.read_csv(os.path.join(OUT, "24_multiverse.csv")).p.to_numpy())
assert len(p) == 54 and (p < 0.05).sum() == 37

apply_publication_style(FigureStyle(16, 2))
fig, (a, b) = plt.subplots(1, 2, figsize=(18, 6))

x = np.arange(4); w = 0.36
b24 = a.bar(x - w / 2, icc24, w, color=PALETTE["blue_main"], edgecolor="black", linewidth=1.5, label="24 h protocol")
b150 = a.bar(x + w / 2, icc150, w, color=PALETTE["red_strong"], edgecolor="black", linewidth=1.5, hatch="//", label="150 h protocol")
for bars in (b24, b150):
    for r in bars:
        a.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 4),
                   textcoords="offset points", ha="center", va="bottom", fontsize=14)
a.axhline(0.75, color=PALETTE["gray_dark"], ls="--", lw=1.5)
a.set_xlim(-0.6, 4.55)
a.text(3.62, 0.765, "0.75 = acceptable\nreliability", ha="left", va="bottom", fontsize=12, color=PALETTE["gray_dark"])
a.set_xticks(x, params)
a.set_ylim(0, 1.2); a.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
a.set_ylabel("ICC(1,3) from triplicates")
a.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
a.set_title("Reliability of the four parameters common to both protocols\n"
            "(11,487 well measurements; TSmax and T50 exist in one protocol only, see Table S11)", fontsize=14)

i = np.arange(54); sig = p < 0.05
b.scatter(i[sig], p[sig], s=60, color=PALETTE["blue_main"], edgecolor="black", linewidth=0.6, label="p < 0.05 (37)")
b.scatter(i[~sig], p[~sig], s=60, color=PALETTE["neutral"], edgecolor="black", linewidth=0.6, label="p ≥ 0.05 (17)")
b.axhline(0.05, color=PALETTE["red_strong"], ls="--", lw=1.5)
b.text(53.5, 0.056, "p = 0.05", color=PALETTE["red_strong"], ha="right", va="bottom", fontsize=13)
b.set_yscale("log"); b.set_ylim(2e-4, 1)
b.set_xticks([0, 10, 20, 30, 40, 50])
b.set_xlabel("specifications, ordered by p"); b.set_ylabel("uncorrected p (log scale)")
b.legend(loc="lower right")
b.set_title("Specification curve for the MoCA coefficient\n37/54 specifications with p < 0.05, all in the expected direction", fontsize=14)

finalize_figure(fig, os.path.join(BASE, "figures", "FigureS6_reliability_specification"), ["png", "pdf"], dpi=300, pad=1)
