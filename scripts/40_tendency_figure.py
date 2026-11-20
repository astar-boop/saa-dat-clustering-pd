"""
40 - Regeneration of the figure on clustering tendency.

The previous version compared the Hopkins statistic with a single
value of a covariance-matched Gaussian null, and read the multimodal
projections against the nominal 5%. Both references have been replaced
in the final analysis by the Gaussian copula null with empirical marginals and by
its Monte Carlo distribution (§3.13), so that the figure was no longer consistent
with the text: the title of the left panel stated «osservato ≈ nullo» while
Hopkins exceeds the null with p = 0.002. The values shown come from the
unified table of script 41, so that figure and text read the same
numbers from a single source.

The figure is here rebuilt entirely from the outputs of script 25, so
that every value shown coincides with those tabulated in §3.13.

Output: figures/fig3_tendency.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT, FIG = os.path.join(BASE, "outputs"), os.path.join(BASE, "figures")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY = "#1f5f8b", "#c44536", "#9aa0a6"


def it(x, dec=4):
    """Italian format: decimal comma."""
    return f"{x:.{dec}f}".replace(".", ",")

# unified table: a single observed value per statistic (script 41)
nl = pd.read_csv(os.path.join(OUT, "41_unified_nulls.csv"))
N = pd.read_csv(os.path.join(OUT, "25_null_distribution_24h.csv"))
pv = np.load(os.path.join(OUT, "25_dip_pvals_observed_24h.npy"))
s = nl[nl.cohort.str.startswith("24h")].set_index('statistic')

fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))

# --- A) Hopkins against the null distribution ------------------------------
ax = axes[0]
r = s.loc["hopkins"]
ax.hist(N.hopkins, bins=30, color=GREY, alpha=.55, edgecolor="white", lw=.3,
        label=f"copula null (B = {len(N)})")
ax.axvline(r.observed, color=RED, lw=2, label=f'observed {r.observed:.4f}')
ax.axvline(r.null_median, color="k", lw=1, ls="--",
           label=f"null median {r['null_median']:.4f}")
ax.set_xlabel("Hopkins statistic"); ax.set_ylabel("null realisations")
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title(f"Hopkins exceeds the null median by {r.observed - r['null_median']:.4f}\nMonte Carlo p "
             f"= {r.p_montecarlo:.4f}", fontsize=9.5)

# --- B) distribution of the dip p values on the observed projections --------
ax = axes[1]
ax.hist(pv, bins=25, color=BLUE, alpha=.75, edgecolor="white", lw=.3)
ax.axvline(.05, color=RED, ls="--", lw=1.4)
ax.text(.06, ax.get_ylim()[1] * .95, "p = 0.05", fontsize=7.5, color=RED,
        va="top")
ax.set_xlabel("dip-test p value of the projection")
ax.set_ylabel("projections")
ax.set_title(f"Dip test over {len(pv)} random projections\n"
             f"{int((pv < .05).sum())} with p < 0.05; the mass piles up "
             f"near 1", fontsize=9.5)

# --- C) number of multimodal projections against the null ------------------
ax = axes[2]
r = s.loc['n_projections_multimodal']
vals = N.n_projections_multimodal.astype(int)
bins = np.arange(-0.5, max(vals.max(), int(r.observed)) + 1.5)
ax.hist(vals, bins=bins, color=GREY, alpha=.65, edgecolor="white", lw=.4,
        label=f"copula null (B = {len(N)})")
ax.axvline(r.observed, color=RED, lw=2,
           label=f'observed {int(r.observed)} / {len(pv)}')
hi = max(int(vals.max()), int(r.observed))
ax.set_xlim(-0.6, hi + 0.6)
ax.set_xticks(range(0, hi + 1))
ax.set_xlabel(f"projections with dip p < 0.05 (of {len(pv)})")
ax.set_ylabel("null realisations")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title(f"The reference is not the nominal 5%\nnull median {r['null_median']:.0f}; Monte Carlo "
             f"p = {r.p_montecarlo:.4f}", fontsize=9.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_tendency.png"), bbox_inches="tight")
print('saved fig3_tendency.png')
print(f"  Hopkins  observed {s.loc['hopkins'].observed:.4f}  null median "
      f"{s.loc['hopkins']['null_median']:.4f}  p {s.loc['hopkins'].p_montecarlo:.4f}")
print(f"  observed projections {int(s.loc['n_projections_multimodal'].observed)} / {len(pv)}  null "
      f"median {s.loc['n_projections_multimodal']['null_median']:.0f}  p "
      f"{s.loc['n_projections_multimodal'].p_montecarlo:.4f}")
