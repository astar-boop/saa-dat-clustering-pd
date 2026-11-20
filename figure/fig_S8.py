"""Figure S8: kinetic parameters by assay run date.
Input: FigureS8_values_by_participant.csv (one row per participant: cohort, run date of the
sample that gave the kinetic values, enrolment phase, natural log of four kinetic parameters),
written by estrai_valori_S8.py and checked there against 126_by_run.csv.
Usage: python3 fig_S8.py [csv] [output basename]"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from matplotlib.lines import Line2D

PALETTE = {"blue_main": "#0F4D92", "red_strong": "#B64342", "gray_dark": "#4D4D4D", "neutral": "#CFCECE"}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 15
    axes_linewidth: float = 2
    font_family: tuple = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(s=FigureStyle()):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(s.font_family),
        "font.size": s.font_size, "axes.linewidth": s.axes_linewidth,
        "xtick.major.width": s.axes_linewidth, "ytick.major.width": s.axes_linewidth,
        "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def finalize_figure(fig, out_path, formats, dpi=300, pad=1):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    for f in formats:
        fig.savefig(out_path.with_suffix("." + f), dpi=dpi)
    plt.close(fig)


BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "outputs", "126_values_by_participant.csv")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "figures", "FigureS8_kinetics_by_run")
d = pd.read_csv(src)
PARAMS = {"24h": ["log Fmax", "log TTT", "log AUC", "log SLOPEmax"], "150h": ["log Fmax", "log T50", "log AUC", "log SLOPE"]}
ROW = {"24h": "24 h cohort", "150h": "150 h cohort"}
COL = {1: PALETTE["red_strong"], 2: PALETTE["blue_main"]}
MRK = {1: "s", 2: "o"}

apply_publication_style()
fig = plt.figure(figsize=(26, 13))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.42])
rng = np.random.default_rng(20261116)
for r, tag in enumerate(("24h", "150h")):
    x = d[d.cohort == tag].reset_index(drop=True)
    runs = sorted(x.run.unique())
    pos = x.run.map({c: i for i, c in enumerate(runs)}).to_numpy()
    jit = rng.uniform(-0.27, 0.27, len(x))
    n_run = x.groupby('run').size()
    for c, nm in enumerate(PARAMS[tag]):
        ax = fig.add_subplot(gs[r, c])
        y = x[nm].to_numpy()
        for ph in (1, 2):
            m = (x.phase == ph).to_numpy()
            ax.scatter(pos[m] + jit[m], y[m], s=16, marker=MRK[ph], color=COL[ph], alpha=0.45, linewidths=0, zorder=2)
        med = pd.Series(y).groupby(pos).median()
        ax.hlines(med.values, med.index - 0.38, med.index + 0.38, color="black", lw=2.6, zorder=3)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([f"{c_} ({n_run[c_]})" for c_ in runs], rotation=90, fontsize=10.5)
        ax.set_xlim(-0.8, len(runs) - 0.2)
        ax.set_title(f"{ROW[tag]}: {nm}", fontsize=16)
        if c == 0:
            ax.set_ylabel("natural log of the parameter")
    lg = fig.add_subplot(gs[r, 4]); lg.set_axis_off()
    h = [Line2D([], [], ls="", marker=MRK[ph], color=COL[ph], alpha=0.8, markersize=9,
                label=f'PPMI phase {ph}\n(n = {int((x.phase == ph).sum())})') for ph in (1, 2)]
    h.append(Line2D([], [], color="black", lw=2.6, label="run median"))
    lg.legend(handles=h, loc="center left", title=ROW[tag], alignment="left", labelspacing=1.0)
    lg.text(0.0, 0.08, "x axis: run date\n(participants in the run)", transform=lg.transAxes, fontsize=12, color=PALETTE["gray_dark"], va="bottom")
finalize_figure(fig, dst, ["png", "pdf"], dpi=300, pad=1)
print("ok", len(d))
