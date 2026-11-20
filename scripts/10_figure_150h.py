"""
10 - Figure 7: replication on the 150h protocol and the T50-biomarker artefact.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT, FIG = os.path.join(BASE, "outputs"), os.path.join(BASE, "figures")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"obs": "#1f5f8b", "null": "#c44536", "grey": "#9aa0a6"}

gap = pd.read_csv(os.path.join(OUT, "08_gap_150h.csv"))
ps = pd.read_csv(os.path.join(OUT, "08_predstrength_150h.csv"))
st = pd.read_csv(os.path.join(OUT, "08_stability_150h.csv"))
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete]

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))

# --- gap ---
ax = axes[0]
ax.errorbar(gap.k, gap.gap, yerr=gap.s_k, marker="o", ms=4, color=C["obs"], capsize=2)
ax.set_xlabel("k"); ax.set_ylabel("Gap")
ax.set_title("Gap statistic — 150 h\n1-SE rule: k = 1 (the rise at high k is\ntypical of continuous data)", fontsize=9.5)

# --- prediction strength ---
ax = axes[1]
ax.plot(ps.k, ps.PS_obsv, marker="o", ms=4, color=C["obs"], label="observed")
ax.plot(ps.k, ps.PS_null, marker="s", ms=4, color=C["null"], label="unimodal null")
ax.axhline(.8, ls="--", color="k", lw=1)
ax.text(ps.k.max(), .82, "0.80 threshold", ha="right", fontsize=7.5)
ax.set_xlabel("k"); ax.set_ylabel("Prediction strength"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("Prediction strength — 150 h\nnever above threshold", fontsize=9.5)

# --- jaccard ---
ax = axes[2]
for key, lab, col, mk in [('observed', "observed", C["obs"], "o"),
                          ('nullref', "Gaussian null", C["null"], "s")]:
    s = st[st.data == key]
    ax.plot(s.k, s.jaccard_avg, marker=mk, ms=4, color=col, label=lab)
ax.axhline(.75, ls="--", color="k", lw=1)
ax.text(st.k.max(), .77, "0.75", ha="right", fontsize=7.5)
ax.axhline(.60, ls=":", color=C["grey"], lw=1)
ax.text(st.k.max(), .61, "0.60", ha="right", fontsize=7.5, color=C["grey"])
ax.set_xlabel("k"); ax.set_ylabel("bootstrap Jaccard"); ax.set_ylim(0, 1.08)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
# The Hennig thresholds do not discriminate on correlated data (script 32): the
# readable comparison is between the two curves, not between curve and threshold.
o2 = float(st[(st.data == 'observed') & (st.k == 2)].jaccard_avg.iloc[0])
n2 = float(st[(st.data == 'nullref') & (st.k == 2)].jaccard_avg.iloc[0])
ax.set_title("Bootstrap stability — 150 h\nsingle Gaussian null ({}) vs observed ({}) at k = 2"
             .format(f"{n2:.2f}", f"{o2:.2f}"),
             fontsize=9.5)

# --- biomarker artefact ---
ax = axes[3]
m = d[["lg_T50", "abeta", "tau", "ptau"]].dropna()
R = np.column_stack([stats.rankdata(m[c]) for c in ["abeta", "tau", "ptau"]])
R = StandardScaler().fit_transform(R)
common = PCA().fit_transform(R)[:, 0]
labels = ["Aβ42", "t-tau", "p-tau", "shared\nfactor"]
raw = [stats.spearmanr(m.lg_T50, m[c]).statistic for c in ["abeta", "tau", "ptau"]]
raw.append(stats.spearmanr(m.lg_T50, common).statistic)
import statsmodels.api as sm
adj = []
for c in ["abeta", "tau", "ptau"]:
    Z = sm.add_constant(common)
    rk = sm.OLS(stats.rankdata(m.lg_T50), Z).fit().resid
    rt = sm.OLS(stats.rankdata(m[c]), Z).fit().resid
    adj.append(stats.pearsonr(rk, rt).statistic)
adj.append(np.nan)
x = np.arange(4)
ax.bar(x - .19, raw, .38, color=C["obs"], label="raw")
# Channel redundant with colour: the adjusted series carries a hatch. It is
# needed above all for the fourth group, where the "adjusted" bar is NaN and a
# single bar remains: the hatch identifies it as raw without resorting to colour.
ax.bar(x[:3] + .19, adj[:3], .38, color=C["null"], hatch="//", edgecolor="white",
       label="adjusted for the\nshared factor")
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Spearman ρ with log T50"); ax.set_ylim(-.1, .42)
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title("T50 vs CSF biomarkers\nthe association is not analyte-specific", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig7_replication_150h.png"), bbox_inches="tight")
print('saved fig7_replication_150h.png')
