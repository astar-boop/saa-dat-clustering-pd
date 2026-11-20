"""
07 - Figure 6: the contrast between the extended population (PD SAA+ and SAA-) and
the SAA-positive population. Shows that the only real discrete structure is the
positive/negative dichotomy of the assay, not a kinetic subgroup.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from _paths import DATA, curated_xlsx

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT, FIG = os.path.join(BASE, "outputs"), os.path.join(BASE, "figures")
SEED = 20260804
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

# rebuild the extended population (as in script 06, section D)
saa = pd.read_csv(os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"))
sel = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")].copy()
for f in ["Fmax", "AUC", "SLOPEMax"]:
    cols = [f"{f}_24h_Rep{i}" for i in (1, 2, 3)]
    sel[f"saa_{f}"] = sel[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
sel = sel.groupby("PATNO", as_index=False).agg(
    {"saa_Fmax": "mean", "saa_AUC": "mean", "saa_SLOPEMax": "mean", "SAA_Status": "first"})
full = pd.read_excel(curated_xlsx(BASE),
                     sheet_name="20260511")
fbl = full[full.EVENT_ID == "BL"][["PATNO", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
                                   "MIA_CAUDATE_BILAT", "age_at_visit",
                                   "subgroup"]]
ext = sel.merge(fbl, on="PATNO", how="inner")

# same population as the primary one (cf. 06_sensitivity.py, section D)
if os.environ.get('ONLY_SPORADIC', "1") == "1":
    n_pre = len(ext)
    ext = ext[ext.subgroup == "Sporadic PD"].reset_index(drop=True)
    print(f'ONLY_SPORADIC=1: extended population {n_pre} -> {len(ext)}')
ext["dat_putamen_low"] = ext[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
FE = ["saa_Fmax", "saa_AUC", "saa_SLOPEMax", "dat_putamen_low", "MIA_CAUDATE_BILAT",
      "age_at_visit"]
ext = ext.dropna(subset=FE).reset_index(drop=True)
Xe = ext[FE].copy()
for c in ["saa_Fmax", "saa_AUC", "saa_SLOPEMax"]:
    Xe[c] = np.log(Xe[c].clip(lower=1e-9))
Xe = StandardScaler().fit_transform(Xe.to_numpy(float))
Pe = PCA().fit_transform(Xe)
lab2 = KMeans(2, n_init=50, random_state=SEED).fit_predict(Xe)
ari = adjusted_rand_score(lab2, (ext.SAA_Status == "Positive").astype(int))

# SAA-positive population (primary analysis)
npz = np.load(os.path.join(OUT, "features.npz"))
X = npz["F_primary"]
P = PCA().fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
COL = {"Positive": "#1f5f8b", "Negative": "#c44536", "Inconclusive": "#9aa0a6"}
# channel redundant with colour (review requirement: no series
# distinguishable by colour alone). "^" and "x" read smaller than "o" at
# equal s, so the area is compensated; "x" needs a line width.
MK = {"Positive": "o", "Negative": "^", "Inconclusive": "x"}
SZ = {"Positive": 11, "Negative": 15, "Inconclusive": 18}

ax = axes[0]
for st in ["Positive", "Negative", "Inconclusive"]:
    m = (ext.SAA_Status == st).to_numpy()
    if m.sum():
        ax.scatter(Pe[m, 0], Pe[m, 1], s=SZ[st], alpha=.7, color=COL[st],
                   marker=MK[st], linewidths=.9 if st == "Inconclusive" else 0,
                   label=f"{st} (n={m.sum()})")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title(f"Sporadic PD with a 24 h assay, complete DAT-SPECT\nand age (n = {len(ext)}), coloured by assay outcome",
             fontsize=9.5)

ax = axes[1]
# explicit project palette + marker per group: without this the two
# groups fell on matplotlib's default cycle, distinguishable by colour
# alone.
COL_K = ["#1f5f8b", "#c44536"]
MK_K = ["o", "^"]
SZ_K = [11, 15]
for c in np.unique(lab2):
    m = lab2 == c
    ax.scatter(Pe[m, 0], Pe[m, 1], s=SZ_K[c], alpha=.7, color=COL_K[c],
               marker=MK_K[c], linewidths=0,
               label=f"group {c+1} (n = {m.sum()})")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title(f"k-means, k = 2, same population\nARI against assay outcome = {ari:.2f}",
             fontsize=9.5)

ax = axes[2]
hb = ax.hexbin(P[:, 0], P[:, 1], gridsize=26, cmap="Blues", mincnt=1)
plt.colorbar(hb, ax=ax, label="participants")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title(f"Type 1 assay-positive only (n = {len(X)})\nno residual structure",
             fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_extended_population.png"), bbox_inches="tight")
# Verifiable sidecar. This figure depends on ONLY_SPORADIC: without the
# variable it draws the whole population (815) instead of the 678 sporadic
# participants, and the caption — which reads its own numbers from
# 06_sensitivity.txt — would describe another population. It has happened.
# Script 74 compares the caption with this file, so that a wrong regeneration
# makes the verification fail.
import pandas as _pd
_pd.DataFrame([{'only_sporadic': os.environ.get('ONLY_SPORADIC', "1") == "1",
                'n_plotted': int(len(ext)),
                "ari": float(ari),
                'n_positives': int((ext.SAA_Status == "Positive").sum())}]).to_csv(
    os.path.join(OUT, "07_fig6_drawn.csv"), index=False)

print('saved fig6_extended_population.png | ARI =', round(ari, 3), "| n =", len(ext))
