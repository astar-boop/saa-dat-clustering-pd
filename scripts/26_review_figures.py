"""
26 - Figures for the analyses of the review.

fig12  power curves of the battery (P1.1)
fig13  Monte Carlo null distributions with observed values (P1.3)
fig14  reliability of the measures and specification curve (P1.8, P1.6)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT, FIG = os.path.join(BASE, "outputs"), os.path.join(BASE, "figures")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY, GREEN = "#1f5f8b", "#c44536", "#9aa0a6", "#2e6b4f"

# ===========================================================================
# Fig 12 — power curves
# ===========================================================================
pw = pd.read_csv(os.path.join(OUT, "22_power.csv"))
dz = pd.read_csv(os.path.join(OUT, "34_delta_zero.csv"))

# At Delta = 0 the injected matrix coincides with the observed data: the value of
# the curve at that point is the false-positive rate of the criterion, not power.
# Without this anchoring the Jaccard curve, which starts already at saturation,
# would appear the most sensitive of the battery.
DIREC_EN = {"PC1": "PC1", "PC2": "PC2", 'random': "random"}

# Each criterion also carries a marker of its own: red and green have the same
# lightness and are confused under deuteranopia/protanopia, so colour
# alone is not enough to tell them apart. The minority proportion pi stays
# encoded by the dash pattern alone, so as not to spend the same channel twice.
CRIT = [('power_PS', "PS", BLUE, "prediction strength", "o"),
        ('power_dip', "dip", RED, "dip on projections", "s"),
        ('power_jaccard', "jaccard", GREEN, "Jaccard", "^")]

# Semi-opaque box for the in-axis legends: without a frame the stroke of the
# curves passed through the labels of the top-right panel.
LEG_BOX = dict(frameon=True, facecolor="white", framealpha=.92,
               edgecolor="0.8", fancybox=False, borderpad=.6)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
for i, cohort in enumerate(pw.cohort.unique()):
    z = dz[dz.cohort == cohort].set_index('criterion').false_positives
    for j, direc in enumerate(["PC1", "PC2", 'random']):
        ax = axes[i, j]
        s = pw[(pw.cohort == cohort) & (pw.direction == direc)]
        for pi, ls in [(0.50, "-"), (0.30, "--"), (0.15, ":")]:
            t = s[s.pi == pi].sort_values("delta")
            for col_pw, key, col, _, mk in CRIT:
                ax.plot(np.r_[0.0, t.delta], np.r_[z[key], t[col_pw]], ls,
                        marker=mk, ms=3.5, color=col, alpha=.85,
                        label=f"{key} · π={pi:.2f}")
        ax.axhline(.8, color="k", ls="--", lw=1)
        # Keep the Δ = 0 observations in the white plotting field.  Ending the
        # reference band at (or just to the right of) zero made its sharp edge
        # run through the three ringed markers and look like an axis line.
        ax.axvspan(-.35, -.10, color=GREY, alpha=.13, zorder=0)
        for _, key, col, _, mk in CRIT:
            ax.plot(0, z[key], mk, color=col, ms=7.5, mec="k", mew=.8, zorder=6)
        ax.set_ylim(-.05, 1.14); ax.set_xlim(-.35, 3.2)
        # Place the label above the plotting field.  A small opaque backing
        # keeps both lines clear of the shaded-band boundary and of the
        # ringed markers at x = 0.
        ax.annotate("Δ = 0\nobserved matrix", xy=(0, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, 6),
                    textcoords="offset points", fontsize=6.2, color="k",
                    ha="center", va="bottom", style="italic",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=.95),
                    annotation_clip=False)
        ax.set_xlabel("Δ (Mahalanobis SD)")
        if j == 0:
            ax.set_ylabel(f'{cohort}\nthreshold-crossing probability')
        ax.set_title(f'direction: {DIREC_EN.get(direc, direc)}', fontsize=9.5)
        # The two legends sit in the "random" panels, where the curves rise
        # late: the central band on the left is empty in both rows
        # (at Delta <= 1.5 the Jaccard is above 0.80 and PS and dip below 0.35).
        # The semi-opaque box is kept anyway as a safety net.
        if i == 0 and j == 2:
            h, l = ax.get_legend_handles_labels()
            keep = [hh for hh, ll in zip(h, l) if "π=0.50" in ll]
            ax.legend(keep, [c[3] for c in CRIT], fontsize=7.5,
                      loc="center left", title="criterion", title_fontsize=7.5,
                      **LEG_BOX)
        if i == 1 and j == 2:
            style = [plt.Line2D([], [], color="k", ls=ls, lw=1.4)
                     for ls in ["-", "--", ":"]]
            ax.legend(style, ["π = 0.50", "π = 0.30", "π = 0.15"], fontsize=7.5,
                      loc="center left",
                      title="minority proportion", title_fontsize=7.5,
                      **LEG_BOX)
fig.suptitle("Threshold-crossing probability under injected mean shifts.\n"
             "The ringed point at Δ = 0 is the crossing rate on the observed "
             "matrix itself: a curve that does not start at zero is not a "
             "power curve.",
             fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig12_power.png"), bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# Fig 13 — Monte Carlo nulls
# ===========================================================================
# SINGLE SOURCE: the unified table of the nulls of script 41, which is the same
# one from which Table 4 of the manuscript is built. Reading 25_nulls.csv here
# produced a figure with the p-values of the previous version for prediction
# strength and Jaccard, and it is the defect flagged in the second round of review.
nl = pd.read_csv(os.path.join(OUT, "41_unified_nulls.csv"))
NICE = {"hopkins": "Hopkins", "silhouette_k2": "Silhouette (k = 2)",
        'n_projections_multimodal': "Multimodal projections",
        "dip_pmin_bonf": "Minimum dip (Bonferroni)", "PS_k2": "Prediction strength",
        "jaccard_k2": "Jaccard (k = 2)", 'ARI_algorithms_k2': "Between-algorithm ARI"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, cohort in zip(axes, nl.cohort.unique()):
    s = nl[nl.cohort == cohort].set_index('statistic').loc[list(NICE)]
    y = np.arange(len(s))[::-1]
    for k, (name, r) in enumerate(s.iterrows()):
        span = r["nullo_IC97.5"] - r["nullo_IC2.5"]
        # normalisation: position of the observed value relative to the CI of the null
        if span > 0:
            lo, hi = 0, 1
            obs = (r.observed - r["nullo_IC2.5"]) / span
            med = (r['null_median'] - r["nullo_IC2.5"]) / span
        else:
            lo, hi, obs, med = 0, 1, 1.5 if r.observed > r['null_median'] else -0.5, .5
        yy = y[k]
        ax.plot([lo, hi], [yy, yy], color=GREY, lw=6, solid_capstyle="butt",
                alpha=.55)
        ax.plot(med, yy, "|", color="k", ms=10, mew=1.5)
        sig = r.p_montecarlo < 0.05
        # Red and blue have the same lightness (112 and 90 out of 255): in
        # greyscale the two points are the same tone. The fill is the
        # non-chromatic channel: filled = significant, empty (edge only) = not
        # significant. The bold of the p-value stays as a third redundancy.
        col = RED if sig else BLUE
        ax.plot(obs, yy, "o", ms=8, zorder=5, mec=col, mew=1.8,
                mfc=col if sig else "white")
        ax.text(1.9, yy, f"p = {r.p_montecarlo:.3f}", va="center", fontsize=8,
                color=RED if sig else GREY,
                fontweight="bold" if sig else "normal")
    ax.set_yticks(y); ax.set_yticklabels([NICE[i] for i in s.index], fontsize=8.5)
    ax.set_xlim(-.6, 2.6); ax.set_xticks([0, .5, 1])
    ax.set_xticklabels(["2.5th pct", "median", "97.5th pct"], fontsize=8)
    ax.set_title(f'{cohort}', fontsize=10)
    ax.axvspan(0, 1, color=GREY, alpha=.12, zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig13_nulls_montecarlo.png"), bbox_inches="tight")
plt.close(fig)
# Verification sidecar: the values actually drawn, row by row. Script
# 74 compares this file with the canonical source, so that a figure
# can no longer diverge from the table without a check failing.
nl[['cohort', 'statistic', 'observed', 'null_median', "nullo_IC2.5",
    "nullo_IC97.5", "p_montecarlo", "B"]].to_csv(
    os.path.join(OUT, "26_fig13_drawn.csv"), index=False)

# ===========================================================================
# Fig 14 — reliability and specification curve
# ===========================================================================
icc = pd.read_csv(os.path.join(OUT, "21_reliability_kinetics.csv"))
mv = pd.read_csv(os.path.join(OUT, "24_multiverse.csv"))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

ax = axes[0]
BARS_ICC = []
params = ["Fmax", "TTT", "AUC", "SLOPE"]
w = .36
x = np.arange(len(params))
# The hatching doubles the colour: the two series stay distinct even when printed
# in greyscale. "//" is sparse enough not to saturate the bar.
for k, (prot, col, htc) in enumerate([("24h", BLUE, ""), ("150h", RED, "//")]):
    vals = []
    for p in params:
        r = icc[(icc.protocol == prot) & (icc.parameter.isin([p, "SLOPEMax"]))]
        r = r[r.parameter.str.startswith(p[:4])]
        vals.append(float(r.ICC_1_3.iloc[0]) if len(r) else np.nan)
    ax.bar(x + (k - .5) * w, vals, w, color=col, hatch=htc, edgecolor="white",
           label=f"{prot} protocol")
    BARS_ICC.extend({'protocol': prot, 'parameter': pn, 'ICC_1_3_drawn': vv}
                     for pn, vv in zip(params, vals))
    for xi, v in zip(x + (k - .5) * w, vals):
        if not np.isnan(v):
            ax.text(xi, v + .015, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(.75, color="k", ls="--", lw=1)
# The threshold label goes in the free space, not on the bars: right-aligned
# in the box it ended up printed in black over the 24h SLOPE bar, dark
# blue, where it measured 3.07:1. Above the 150h of the AUC (0.56) the column is
# empty up to the top edge, so the text stays resting on the dashed
# line but on a white background. The 24h bars occupy [x-0.36, x] and the 150h
# ones [x, x+0.36]: the free interval at this height is [2.00, 2.64].
ax.text(2.32, .765, "0.75\n= acceptable\nreliability", ha="center", va="bottom",
        fontsize=7.5, linespacing=1.3)
ax.set_xticks(x); ax.set_xticklabels(params); ax.set_ylim(0, 1.05)
ax.set_ylabel("ICC(1,3) from triplicates")
ax.legend(fontsize=8, frameon=False)
# the count must concern only the parameters drawn, not all ten
_n_wl = int(sum(
    icc[(icc.protocol == prot) & (icc.parameter.str.startswith(p[:4]))].n_subjects.iloc[0]
    for prot in ("24h", "150h") for p in params) * 3)
ax.set_title(f'Reliability of the four parameters common to both protocols\n({_n_wl:,} well '
             f'measurements; TSmax and T50 exist in one protocol only, see Table 10)', fontsize=9.5)

ax = axes[1]
srt = mv.sort_values("p").reset_index(drop=True)
col = [BLUE if p < .05 else GREY for p in srt.p]
ax.scatter(range(len(srt)), srt.p, c=col, s=22)
ax.axhline(.05, color=RED, ls="--", lw=1.2)
ax.text(len(srt) - 1, .054, "p = 0.05", ha="right", fontsize=8, color=RED)
ax.set_yscale("log")
ax.set_xlabel("specifications, ordered by p")
ax.set_ylabel("uncorrected p (log scale)")
ax.set_title(f"Specification curve for the MoCA coefficient\n"
             f"{(srt.p<.05).sum()}/{len(srt)} specifications with p < 0.05, "
             "all in the expected direction", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig14_reliability_multiverse.png"), bbox_inches="tight")
plt.close(fig)
pd.DataFrame(BARS_ICC).to_csv(os.path.join(OUT, "26_fig14_drawn.csv"),
                               index=False)

print('saved fig12, fig13, fig14')
