"""
14 - Figure 8: the longitudinal signal on MoCA and its specificity.
     Figure 9: the CSF concentration factor.
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
W = os.path.join(BASE, "outputs", "cache") + "/"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"fast": "#c44536", "mid": "#9aa0a6", "slow": "#1f5f8b"}
# Channels redundant with colour, identical to those of 16_figure_trajectories.py
# so that the two figures teach the same map: the line style identifies the
# level of T50 (faster = dashed, medium = dotted, slower =
# solid) and the marker identifies the observed series. In black and white BLU,
# VERDE and ROSSO have the same luminance, so colour alone is not enough.
LS_FAST, LS_MID, LS_SLOW = (0, (5.5, 2.2)), (0, (1.4, 1.8)), "solid"
MK_FAST, MK_SLOW = "o", "s"

cur = pd.read_parquet(W + "curated.parquet")
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete].reset_index(drop=True)
d["z_T50"] = np.log(d.k_T50)
d["z_T50"] = (d.z_T50 - d.z_T50.mean()) / d.z_T50.std()

L = cur[cur.PATNO.isin(d.PATNO)][["PATNO", "YEAR", "moca"]].dropna()
L = L.merge(d[["PATNO", "z_T50"]], on="PATNO")
q = d.z_T50.quantile([1/3, 2/3]).values
L['tertile'] = np.where(L.z_T50 <= q[0], "fast",
                        np.where(L.z_T50 >= q[1], "slow", "intermediate"))

# =====================================================================
# Fig 8
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# --- MoCA trajectories predicted by the mixed model ---
# The raw means by year are biased by selective attrition (those who remain in
# follow-up at 10+ years are healthier): the trajectories predicted by the
# model are used, which is what the interaction coefficient represents.
ax = axes[0]
import statsmodels.formula.api as smf
covbl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                   "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
M = L.merge(covbl, on="PATNO", how="left").rename(columns={"YEAR": "years"})
M = M[["PATNO", "years", "moca", "z_T50", "age_bl", "SEX", "EDUCYRS", "dur_bl"]].dropna()
fitm = smf.mixedlm("moca ~ z_T50*years + age_bl + SEX + EDUCYRS + dur_bl",
                   M, groups=M.PATNO, re_formula="~years").fit(method="lbfgs",
                                                               maxiter=300)
b_int = fitm.params.get("z_T50:years", fitm.params.get("years:z_T50"))
b_yr = fitm.params["years"]
b_k = fitm.params["z_T50"]
icept = (fitm.params["Intercept"] + fitm.params["age_bl"] * M.age_bl.mean()
         + fitm.params["SEX"] * M.SEX.mean() + fitm.params["EDUCYRS"] * M.EDUCYRS.mean()
         + fitm.params["dur_bl"] * M.dur_bl.mean())
yrs = np.linspace(0, 10, 50)
COLM = {-1: C["fast"], 0: C["mid"], 1: C["slow"]}
LSM = {-1: LS_FAST, 0: LS_MID, 1: LS_SLOW}
LWM = {-1: 2.1, 0: 1.4, 1: 2.1}
# The entries stay short because the legend box sits in the free triangle
# at the bottom left: if they were lengthened it would end up under the −1 SD curve.
# That the lines are the model and the markers the observed data is stated by the title.
LABM = {-1: "T50 −1 SD — faster seeding (dashed)",
        0: "T50 at the mean (dotted)",
        1: "T50 +1 SD — slower seeding (solid)"}
handA = []
for z in (-1, 0, 1):
    ax.plot(yrs, icept + b_k * z + (b_yr + b_int * z) * yrs,
            color=COLM[z], lw=LWM[z], ls=LSM[z])
    handA.append(plt.Line2D([], [], color=COLM[z], lw=LWM[z], ls=LSM[z],
                            label=LABM[z]))
# Raw observations in transparency, as context: markers only, one
# of its own per tertile. Without a connecting line the line style remains a
# code exclusive to the model and the two observed series are distinguished by
# shape, not by colour.
for t, z, mk, name in [
        ("fast", -1, MK_FAST, "observed: lowest-T50 tertile (circles)"),
        ("slow", 1, MK_SLOW, "observed: highest-T50 tertile (squares)")]:
    s = L[(L.tertile == t) & (L.YEAR <= 10)]
    g = s.groupby("YEAR").moca.agg(["mean", "size"])
    g = g[g["size"] >= 20]
    ax.plot(g.index, g["mean"], ls="none", marker=mk, ms=4.6,
            color=COLM[z], alpha=.55)
    handA.append(plt.Line2D([], [], color=COLM[z], ls="none", marker=mk,
                            ms=4.6, alpha=.75, label=name))
ax.set_xlabel("Years from baseline"); ax.set_ylabel("MoCA")
ax.legend(handles=handA, fontsize=7, frameon=False, loc="lower left",
          handlelength=3.4, borderaxespad=.3)
ax.set_title("Mixed-model predicted trajectories\n"
             "(lines: model; markers: observed tertile means)", fontsize=9.5)

# --- specificity: forest plot ---
ax = axes[1]
sp = pd.read_csv(os.path.join(OUT, "13_specificity.csv"))
NAMES = {"moca": "MoCA", "updrs3_score": "UPDRS-III", "updrs_totscore": "UPDRS total",
         "updrs2_score": "UPDRS-II", "MIA_STRIATUM_BILAT": "DAT striatum",
         "scopa": "SCOPA-AUT"}
sp = sp.sort_values("beta_relativo_%")
y = np.arange(len(sp))
cols = [C["slow"] if p < .05 else C["mid"] for p in sp.p_BH]
ax.barh(y, sp["beta_relativo_%"], color=cols, height=.6)
ax.axvline(0, color="k", lw=.9)
ax.set_yticks(y); ax.set_yticklabels([NAMES.get(e, e) for e in sp.outcome], fontsize=8.5)
ax.set_xlabel("Modulation of the annual slope per 1 SD of T50 (%)")
# The annotation always starts to the right of the rightmost end of the bar:
# for negative bars that end is zero, so the text falls in the
# white space to the right of the axis and cannot end up over the tick labels,
# whatever the value of the bar.
for i, (v, p) in enumerate(zip(sp["beta_relativo_%"], sp.p_BH)):
    ax.text(max(v, 0) + 1.5, i, f"p={p:.3f}",
            va="center", ha="left", fontsize=7.5)
ax.set_xlim(-15, 52)
ax.set_title("Same specification across six outcomes",
             fontsize=9.5)

# --- robustness to the specifications ---
ax = axes[2]
cf = pd.read_csv(os.path.join(OUT, "13_moca_confounders.csv"))
y = np.arange(len(cf))[::-1]
ax.errorbar(cf.beta_T50_x_years, y, xerr=1.96 * cf.se, fmt="o", ms=5,
            color=C["slow"], capsize=3, lw=1.3)
ax.axvline(0, color="k", lw=.9, ls="--")
# The labels arrive in Italian from the CSV: they are translated here, as for NAMES
# in the central panel. Only the displayed string changes, not the order or the
# values: the map is applied row by row on the original column.
SPEC = {'base model': "base model",
        '+ age at onset x years': "+ age at onset × years",
        '+ baseline age x years': "+ baseline age × years",
        '+ education x years': "+ education × years",
        '+ disease duration x years': "+ disease duration × years",
        '+ sex x years': "+ sex × years"}
ax.set_yticks(y)
ax.set_yticklabels([SPEC.get(s, s) for s in cf.specification], fontsize=8)
ax.set_xlabel("β for T50 × years (95% CI)")
ax.set_title("Sensitivity of β to the covariate set", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig8_longitudinal.png"), bbox_inches="tight")
plt.close(fig)

# =====================================================================
# Fig 9 — concentration factor
# =====================================================================
ANALYTES = ["abeta", "tau", "ptau", "asyn", "IU_pTau181_CSF",
            "IU_ABeta40_CSF", "IU_ABeta42_CSF"]
NICE = ["Aβ42", "t-tau", "p-tau", "α-syn total", "p-tau181", "Aβ40", "Aβ42 (IU)"]
bl = cur[cur.EVENT_ID == "BL"]
b = bl[bl.PATNO.isin(d.PATNO)][ANALYTES].dropna()

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
cm = b.corr(method="spearman")
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(NICE))); ax.set_yticks(range(len(NICE)))
ax.set_xticklabels(NICE, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(NICE, fontsize=8)
for i in range(len(NICE)):
    for j in range(len(NICE)):
        v = cm.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                color="white" if abs(v) > .6 else "black")
plt.colorbar(im, ax=ax, shrink=.8, label="Spearman ρ")
ax.set_title(f"All CSF analytes correlate positively (n = {len(b)})",
             fontsize=9.5)

ax = axes[1]
R = np.column_stack([stats.rankdata(b[c]) for c in ANALYTES])
R = (R - R.mean(0)) / R.std(0)
p = PCA().fit(R)
ax.bar(range(1, 8), p.explained_variance_ratio_, color=C["slow"], width=.6)
ax.set_xlabel("Component"); ax.set_ylabel("Share of variance")
ax.set_title(f"A single dimension dominates\nPC1 = {p.explained_variance_ratio_[0]:.1%}, "
             "all loadings positive", fontsize=9.5)
for i, v in enumerate(p.explained_variance_ratio_[:3]):
    ax.text(i + 1, v + .012, f"{v:.0%}", ha="center", fontsize=8)

ax = axes[2]
cc = pd.read_csv(os.path.join(OUT, "12_kinetics_biomarkers_adjusted.csv"))
labels, l2, l3 = [], [], []
for tag in ["150h", "24h"]:
    s = cc[cc.cohort == tag]
    labels.append(f"{tag.replace('h', ' h')}\n(n = {len(s)} pairs)")
    # L2 = age and sex; L3 = age, sex AND shared component. The like-for-like
    # comparison with L3 is L2, not the raw L1: L3 is cumulative.
    l2.append(int((s.pBH_L2 < .05).sum()))
    l3.append(int((s.pBH_L3 < .05).sum()))
x = np.arange(2)
ax.bar(x - .19, l2, .38, color=C["fast"], label="adjusted for age and sex")
# Channel redundant with colour: the second series carries a dashed style, so
# that the two series remain distinguishable even in black-and-white print.
ax.bar(x + .19, l3, .38, color=C["slow"], hatch="//", edgecolor="white",
       label="+ shared factor")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Significant associations (BH-FDR < 0.05)")
ax.legend(fontsize=7.5, frameon=False)
for i, (a, bb) in enumerate(zip(l2, l3)):
    ax.text(i - .19, a + .4, str(a), ha="center", fontsize=10, fontweight="bold")
    ax.text(i + .19, bb + .4, str(bb), ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(l2) + 3)
ax.set_title("After adjustment for the shared factor,\nno association remains", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig9_concentration.png"), bbox_inches="tight")
print('saved fig8_longitudinal.png and fig9_concentration.png')
