"""
05 - Figures.

fig1_distributions   distributions of the 7 features (unimodality)
fig2_pca_continuum   PCA space: density and k=2/k=3 partitions
fig4_k_selection     gap, silhouette, prediction strength, PAC, Jaccard, ARI
fig5_correlations    correlation matrix of the features
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.decomposition import PCA

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
FIG = os.path.join(BASE, "figures")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

d = pd.read_csv(os.path.join(OUT, "analysis_with_clusters.csv"))
npz = np.load(os.path.join(OUT, "features.npz"))
names = json.load(open(os.path.join(OUT, "feature_names.json")))
X = npz["F_primary"]
F = names["primary"]
LAB = {"lg_Fmax": "log Fmax, standardised",
       "lg_TTT": "log TTT, standardised",
       "lg_AUC": "log AUC, standardised",
       "lg_SLOPE": "log maximum slope, standardised",
       "dat_putamen_low": "DAT lower putamen (SBR), standardised",
       "dat_caudate_mean": "DAT mean caudate (SBR), standardised",
       "age_bl": "Age at baseline, standardised"}
# Matrix labels aligned with the terminology of the main text.
# The caption already states that all the variables are standardised, so
# there is no need to repeat the qualifier on each axis.
LAB_CORR = {"lg_Fmax": "log Fmax",
            "lg_TTT": "log TTT",
            "lg_AUC": "log AUC",
            "lg_SLOPE": "log SLOPEmax",
            "dat_putamen_low": "Lowest-uptake putamen",
            "dat_caudate_mean": "Mean caudate",
            "age_bl": "Age"}
RAW = {"lg_Fmax": "saa_Fmax", "lg_TTT": "saa_TTT", "lg_AUC": "saa_AUC",
       "lg_SLOPE": "saa_SLOPEMax", "dat_putamen_low": "dat_putamen_low",
       "dat_caudate_mean": "dat_caudate_mean", "age_bl": "age_bl"}
C = {"obs": "#1f5f8b", "null": "#c44536", "grey": "#9aa0a6"}

# ---------------------------------------------------------------------------
# Fig 1 - distributions
# ---------------------------------------------------------------------------
dip = pd.read_csv(os.path.join(OUT, "02_diptest.csv"))
dipf = dip[dip.space == "feature"].set_index("var")

fig, axes = plt.subplots(2, 4, figsize=(13, 6))
# The distributions must be drawn on the ANALYSIS scale: the dip test in
# the title is computed on the logarithm, and drawing the raw variable under
# a "log" title is the discrepancy flagged in review. X is the matrix
# already transformed and standardised (features.npz).
for j, (ax, f) in enumerate(zip(axes.ravel(), F)):
    v = pd.Series(X[:, j])
    ax.hist(v, bins=40, color=C["obs"], alpha=.55, density=True, edgecolor="white", lw=.3)
    xs = np.linspace(v.min(), v.max(), 300)
    ax.plot(xs, stats.gaussian_kde(v)(xs), color="#0b3954", lw=1.8)
    ax.set_title(f"{LAB[f]}\ndip p = {dipf.loc[f,'p']:.3f} (Bonferroni {dipf.loc[f,'p_bonf']:.2f})",
                 fontsize=8.5)
    ax.set_yticks([])
axes.ravel()[-1].axis("off")
axes.ravel()[-1].text(.02, .5,
    "Hartigan dip test:\nH₀ = unimodal distribution.\n\n"
    "No variable rejects\nunimodality after correction\nfor multiple comparisons.",
    va="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_distributions.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2 - PCA space
# ---------------------------------------------------------------------------
pca = PCA().fit(X)
P = pca.transform(X)
ev = pca.explained_variance_ratio_

fig = plt.figure(figsize=(13, 4.2))
gs = GridSpec(1, 3, figure=fig, wspace=.28)

ax = fig.add_subplot(gs[0])
hb = ax.hexbin(P[:, 0], P[:, 1], gridsize=28, cmap="Blues", mincnt=1)
plt.colorbar(hb, ax=ax, label="participants")
ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
ax.set_title("Density in PCA space\n(a single cloud, no separation)", fontsize=9.5)


# Explicit colour from the project palette plus a marker per group: without
# this the groups fell on the matplotlib default cycle (for k = 3
# orange against green, indistinguishable in colour blindness) and were encoded
# by colour alone. "^" and "s" read smaller than "o", so the area is
# compensated.
COL_K = ["#1f5f8b", "#c44536", "#2e6b4f"]
MK_K = ["o", "^", "s"]
SZ_K = [9, 13, 10]

# Legend with an OPAQUE box: without a frame, in the k = 3 panel a square of
# group 3 fell inside the legend area, at the end of the "group 1" row, and
# read as a fourth key. The box closes the perimeter and removes the
# ambiguity. Since it is opaque, however, it hides what lies beneath it, so the
# corner is not fixed by hand: the points covered in the four corners are
# counted and the minimum is chosen (in a tie, the order below decides).
CORNERS = ("upper right", "upper left", "lower right", "lower left")
LEG_BOX = dict(fontsize=7, frameon=True, framealpha=1, facecolor="white",
               edgecolor="0.5", fancybox=False, borderpad=.5)


def _legend_opaque(ax, pts):
    chosen, covered = CORNERS[0], None
    for loc in CORNERS:
        leg = ax.legend(loc=loc, **LEG_BOX)
        ax.figure.canvas.draw()
        bb = leg.get_window_extent().transformed(ax.transData.inverted())
        n = int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                 (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum())
        if covered is None or n < covered:
            chosen, covered = loc, n
    ax.legend(loc=chosen, **LEG_BOX)
    return chosen, covered


for j, k in enumerate((2, 3)):
    ax = fig.add_subplot(gs[j + 1])
    lab = d[f"cluster_k{k}"].to_numpy()
    for c in np.unique(lab):
        m = lab == c
        ax.scatter(P[m, 0], P[m, 1], s=SZ_K[c], alpha=.65, color=COL_K[c],
                   marker=MK_K[c], linewidths=0,
                   label=f"group {c+1} (n = {m.sum()})")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.set_title(f"k-means partition, k = {k}\n(a cut through a continuum)", fontsize=9.5)
    loc, cop = _legend_opaque(ax, P[:, :2])
    print(f'  fig2 k={k}: legend in {loc}, points covered = {cop}')
fig.savefig(os.path.join(FIG, "fig2_pca_continuum.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3 - REMOVED. The figure fig3_tendency.png is written by
# 40_tendency_figure.py, which uses the copula null and three panels and is
# the one that the caption of the manuscript describes. The block that used
# to be here wrote an outdated version of it (Gaussian null of equal
# covariance, two panels): as long as it remained, rerunning this script
# on its own overwrote the good figure with the wrong one, and the
# caption described an image that was no longer there. One file name,
# one script only that writes it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fig 4 - selection of k
# ---------------------------------------------------------------------------
gap_o = pd.read_csv(os.path.join(OUT, "03_gap_observed.csv"))
gap_n = pd.read_csv(os.path.join(OUT, "03_gap_null.csv"))
idx = pd.read_csv(os.path.join(OUT, "03_internal_indices.csv"))
ps = pd.read_csv(os.path.join(OUT, "03_prediction_strength.csv"))
jac = pd.read_csv(os.path.join(OUT, "03_bootstrap_jaccard.csv"))
pac = pd.read_csv(os.path.join(OUT, "03_pac.csv"))
ari = pd.read_csv(os.path.join(OUT, "03_algo_agreement.csv"))

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))

ax = axes[0, 0]
ax.errorbar(gap_o.k, gap_o.gap, yerr=gap_o.s_k, marker="o", ms=4, color=C["obs"],
            capsize=2, label="observed")
ax.errorbar(gap_n.k, gap_n.gap, yerr=gap_n.s_k, marker="s", ms=4, color=C["null"],
            capsize=2, alpha=.8, label="Gaussian null (single realisation)")
ax.set_xlabel("k"); ax.set_ylabel("Gap"); ax.legend(fontsize=7.5, frameon=False)
ax.set_title("Gap statistic", fontsize=9.5)

# key in the data, visible label, colour, marker
SERIES = [('observed', "observed", C["obs"], "o"),
         ('nullref', "Gaussian null (single realisation)", C["null"], "s")]

ax = axes[0, 1]
for key, lab, col, mk in SERIES:
    s = idx[(idx.algorithm == "kmeans") & (idx.data == key)]
    ax.plot(s.k, s.silhouette, marker=mk, ms=4, color=col, label=lab)
ax.set_xlabel("k"); ax.set_ylabel("Silhouette (k-means)")
ax.legend(fontsize=7.5, frameon=False); ax.set_title("Silhouette", fontsize=9.5)

ax = axes[0, 2]
ax.plot(ps.k, ps.PS_mean_obsv, marker="o", ms=4, color=C["obs"], label="observed")
ax.plot(ps.k, ps.PS_mean_null, marker="s", ms=4, color=C["null"], label="Gaussian null")
ax.axhline(.8, ls="--", color="k", lw=1)
ax.text(ps.k.max(), .81, "0.80 threshold", ha="right", fontsize=7.5)
ax.set_xlabel("k"); ax.set_ylabel("Prediction strength"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, frameon=False); ax.set_title("Prediction strength", fontsize=9.5)

ax = axes[1, 0]
for key, lab, col, mk in SERIES:
    s = jac[jac.data == key]
    ax.plot(s.k, s.jaccard_min, marker=mk, ms=4, color=col, label=f"{lab} (minimum)")
    ax.plot(s.k, s.jaccard_avg, marker=mk, ms=3, ls=":", color=col, alpha=.6,
            label=f"{lab} (mean)")
ax.axhline(.75, ls="--", color="k", lw=1)
ax.text(jac.k.max(), .76, "0.75 threshold", ha="right", fontsize=7.5)
# Hennig's threshold does not discriminate on correlated data: the null exceeds
# it in 100% of the replicates (script 32). The readable comparison is between
# the two curves.
o2 = float(jac[(jac.data == 'observed') & (jac.k == 2)].jaccard_avg.iloc[0])
n2 = float(jac[(jac.data == 'nullref') & (jac.k == 2)].jaccard_avg.iloc[0])
ax.annotate("at k = 2 the single Gaussian null ({})\nexceeds the observed value ({})".format(f"{n2:.2f}", f"{o2:.2f}"),
            xy=(2.08, n2), xytext=(4.6, 1.0), fontsize=7.5, va="center",
            arrowprops=dict(arrowstyle="->", lw=.9, color="k",
                            connectionstyle="arc3,rad=.18"))
ax.set_xlabel("k"); ax.set_ylabel("bootstrap Jaccard"); ax.set_ylim(0, 1.12)
ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
ax.set_title("Bootstrap stability", fontsize=9.5)

ax = axes[1, 1]
for key, lab, col, mk in SERIES:
    s = pac[pac.data == key]
    ax.plot(s.k, s.PAC, marker=mk, ms=4, color=col, label=lab)
ax.set_xlabel("k"); ax.set_ylabel("PAC")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("Consensus ambiguity (PAC)", fontsize=9.5)

ax = axes[1, 2]
piv = ari.pivot_table(index="k", columns='data', values="ARI")
ax.plot(piv.index, piv['observed'], marker="o", ms=4, color=C["obs"], label="observed")
ax.plot(piv.index, piv['nullref'], marker="s", ms=4, color=C["null"],
        label="Gaussian null (single realisation)")
ax.set_xlabel("k"); ax.set_ylabel("mean between-algorithm ARI"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("Between-algorithm agreement", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_k_selection.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5 - correlations
# ---------------------------------------------------------------------------
corr = pd.read_csv(os.path.join(OUT, "02_corr_primary.csv"), index_col=0)
# RdBu_r is symmetric in luminance around zero: in monochrome rho = -1 and
# rho = +1 both render as grey 48, and the sign cannot be recovered from the
# colour. The sign therefore receives a non-chromatic channel (diagonal hatching
# on the negative cells); the printed value remains the authoritative channel.
matplotlib.rcParams["hatch.linewidth"] = .55
CMAP = plt.get_cmap("RdBu_r")
NORM = matplotlib.colors.Normalize(vmin=-1, vmax=1)


def _luminance(rgb):
    """WCAG relative luminance of the cell tint."""
    c = np.asarray(rgb[:3], float)
    c = np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)
    return float(c @ np.array([.2126, .7152, .0722]))


def _contrast(l1, l2):
    lo, hi = min(l1, l2), max(l1, l2)
    return (hi + .05) / (lo + .05)


def _colour_stroke(v):
    """White or black: the one with more contrast on the cell tint is chosen.

    A fixed threshold on |rho| does not follow the luminance of RdBu_r, which is
    not monotonic in |rho|: at rho = -0.58 the cell is a medium light blue
    (#4996C5) and white measured 3.26:1 on it, below the minimum of 4.5:1 for
    text. The point-by-point choice between white and black instead guarantees
    a theoretical minimum of 4.58:1 on any tint, because the two contrasts
    cross at luminance 0.179. The criterion applies both to the number and to
    the hatching that passes behind it, so the two never come apart.
    """
    y = _luminance(CMAP(NORM(v)))
    return "white" if _contrast(y, 1.0) > _contrast(y, 0.0) else "black"


# Point at which, on the negative half, the stroke switches from black to white:
# the colour bar needs it to carry the same criterion as the cells.
_g = np.linspace(-1, 0, 2001)
_whites = [float(x) for x in _g if _colour_stroke(x) == "white"]
THRESHOLD_STROKE = max(_whites) if _whites else -1.0

fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(corr.values, cmap=CMAP, norm=NORM)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels([LAB_CORR[c] for c in corr.columns], rotation=40,
                   ha="right", fontsize=8)
ax.set_yticklabels([LAB_CORR[c] for c in corr.index], fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.values[i, j]
        if v < 0:
            # Rectangle without a border: the hatching is the only graphic mark,
            # and it passes under the number, which is drawn afterwards.
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   linewidth=0, hatch="///",
                                   edgecolor=_colour_stroke(v)))
        # On the hatched cells the number sits on a patch of the colour of the
        # cell itself: it interrupts the hatching without altering the tint.
        bb = (dict(boxstyle="square,pad=.15", ec="none",
                   fc=im.cmap(im.norm(v))) if v < 0 else None)
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color=_colour_stroke(v), bbox=bb)
cb = plt.colorbar(im, ax=ax, label="Spearman ρ  (hatched half: ρ < 0)",
                  shrink=.85)
# The negative half of the bar carries the same hatching as the cells, so
# the bar declares its own symmetry instead of suggesting a monotonic
# scale: in monochrome the tint gives |rho|, the hatching gives the sign.
cb.ax.add_patch(Rectangle((0, -1), 1, THRESHOLD_STROKE + 1, fill=False,
                          linewidth=0, hatch="///", edgecolor="white",
                          transform=cb.ax.transData, clip_on=True))
cb.ax.add_patch(Rectangle((0, THRESHOLD_STROKE), 1, -THRESHOLD_STROKE, fill=False,
                          linewidth=0, hatch="///", edgecolor="black",
                          transform=cb.ax.transData, clip_on=True))
ax.set_title("Kinetic block and DAT block are weakly rank-correlated", fontsize=10)
fig.tight_layout()
fig.text(.5, .005,
         "Shade encodes |ρ| only in monochrome: negative coefficients carry a "
         "diagonal hatch.\nThe printed coefficient in each cell is the "
         "authoritative value.",
         ha="center", va="top", fontsize=7.2)
fig.savefig(os.path.join(FIG, "fig5_correlations.png"), bbox_inches="tight")
plt.close(fig)

print('figures written to', FIG)
for f in sorted(os.listdir(FIG)):
    print(" -", f)
