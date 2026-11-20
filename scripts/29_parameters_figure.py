"""
29 - Explanatory figure of the kinetic parameters of the assay.

Panel A: idealised aggregation curve with the parameters marked.
Panel B: the three real replicates of the sample cited in §2.3, which shows
         why the median is preferable to the mean as an aggregator.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paths import DATA

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(BASE, "figures")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY, GREEN = "#1f5f8b", "#c44536", "#9aa0a6", "#2e6b4f"
# The palette GRIGIO measures 2.64:1 on white: below the 4.5:1 threshold for
# text and below 3:1 for graphical elements that carry meaning. A darker
# variant (6.05:1) is needed in here, to be used for the Fmax label,
# for the TSmax label and for the lines that accompany them. The palette
# constant stays intact because it is shared with the other scripts.
GREY_DARK = "#5f6368"

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

# ---------------------------------------------------------------- panel A
ax = axes[0]
t = np.linspace(0, 24, 600)
Fmax, t50, k = 100.0, 11.0, 0.55
y = Fmax / (1 + np.exp(-k * (t - t50)))          # sigmoid
threshold = 20.0
ttt = t50 - np.log(Fmax / threshold - 1) / k
slope = k * y * (1 - y / Fmax)
i_max = int(np.argmax(slope))
tsmax, smax = t[i_max], slope[i_max]

ax.fill_between(t, 0, y, color=BLUE, alpha=.10)
ax.text(17.5, 26, "AUC\narea under the curve", color=BLUE, fontsize=8.5,
        ha="center", style="italic")
ax.plot(t, y, color=BLUE, lw=2.4)

ax.axhline(Fmax, ls="--", color=GREY_DARK, lw=1)
ax.text(23.6, Fmax + 2.5, "Fmax — maximum fluorescence", ha="right", fontsize=8.5,
        color=GREY_DARK)
ax.axhline(threshold, ls=":", color=RED, lw=1.2)
ax.text(0.3, threshold + 2.5, "positivity threshold", fontsize=8, color=RED)

# The curve and the tangent passed over two of these labels. Instead of
# covering them with an opaque box — which interrupts the curve and makes it
# look broken — the two that collided are moved to the left of their own
# line, in the top-left region, which is empty.
for x, lab, col, dy, ha in [(ttt, "TTT\ntime to threshold", RED, 62, "left"),
                            (t50, "T50\ntime to half-aggregation", GREEN, 84,
                             "right"),
                            (tsmax, "TSmax", GREY_DARK, 44, "left")]:
    ax.plot([x, x], [0, np.interp(x, t, y)], ls="--", color=col, lw=1.1)
    ax.plot(x, np.interp(x, t, y), "o", color=col, ms=6)
    ax.text(x + (.35 if ha == "left" else -.35), dy, lab, ha=ha,
            fontsize=8.5, color=col, linespacing=1.4)

# tangent at the point of maximum slope
xx = np.linspace(tsmax - 3.2, tsmax + 3.2, 10)
ax.plot(xx, y[i_max] + smax * (xx - tsmax), color="k", lw=1.6)
# The label of the tangent sits at top left, the only empty area of the
# panel, and a thin leader line connects it to the tangent: without the
# leader it would remain ambiguous which element it refers to.
# The leader anchored at tsmax+1.9 ran down too steeply and cut through
# the T50 label, moved there to free the curve. Re-anchoring it higher
# up on the tangent (tsmax+2.9), the line stays above the green block for its
# whole length, without moving either of the two labels.
ax.annotate("SLOPEmax\nmaximum slope",
            xy=(tsmax + 2.9, y[i_max] + smax * 2.9), xycoords="data",
            xytext=(tsmax - 4.6, y[i_max] + smax * 3.4 + 14),
            ha="right", fontsize=8.5, linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color="0.45", lw=.8,
                            shrinkA=2, shrinkB=3))

ax.set_xlabel("Incubation time (hours)")
ax.set_ylabel("Thioflavin T fluorescence (arbitrary units)")
ax.set_xlim(0, 24); ax.set_ylim(0, 118)
ax.set_title("A · The parameters describe two distinct properties of the curve:\n"
             "WHEN aggregation starts (TTT, T50, TSmax) and HOW MUCH occurs "
             "(Fmax, SLOPEmax, AUC)", fontsize=9.5)

# ---------------------------------------------------------------- panel B
ax = axes[1]
sa = pd.read_csv(os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"))
sel = sa[(sa.PROJECTID == 155) & (sa.COHORT == "PD") &
         (sa.CLINICAL_EVENT == "BL") & (sa.SAA_Status == "Positive")]
cols = [f"SLOPE_150h_Rep{i}" for i in (1, 2, 3)]
v = sel[cols].apply(pd.to_numeric, errors="coerce")
i = v.max(axis=1).idxmax()
rep = v.loc[i].to_numpy(float)

x = np.arange(3)
ax.bar(x, rep, color=[RED, BLUE, BLUE], width=.55)
for xi, val in zip(x, rep):
    ax.text(xi, val * 1.35, f"{val:,.1f}", ha="center", fontsize=9)
ax.axhline(np.mean(rep), color=RED, ls="--", lw=1.6)
ax.text(2.62, np.mean(rep) * 1.15, f"mean = {np.mean(rep):,.0f}",
        color=RED, fontsize=9, ha="right", fontweight="bold")
ax.axhline(np.median(rep), color=GREEN, ls="-", lw=1.6)
# the label of the median goes in the empty space between the first and the
# second bar: on the right it overlapped the third
ax.text(0.5, np.median(rep) * 1.25, f"median = {np.median(rep):,.0f}",
        color=GREEN, fontsize=9, ha="center", fontweight="bold")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(["replicate 1", "replicate 2", "replicate 3"])
ax.set_ylabel("Maximum slope (log scale)")
ax.set_ylim(5, 40000)
ax.set_title("B · A single aberrant well moves the mean by 65-fold,\n"
             "and the median not at all", fontsize=9.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig17_parameters.png"), bbox_inches="tight")
print('saved fig17_parameters.png |', rep.round(1).tolist(),
      'mean', round(float(np.mean(rep))), 'median', round(float(np.median(rep))))
