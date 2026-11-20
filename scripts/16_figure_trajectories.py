"""
16 - The trajectories, visually.

Fig. 10 — grid of the 12 primary outcomes: trajectories predicted by the mixed
          model for T50 at -1 SD, mean and +1 SD.
Fig. 11 — the three predicted outcomes, in detail: model trajectories (lines)
          and observed change from each participant's own baseline (points
          with SE).

On the observed change: the WITHIN-participant change from each participant's
own baseline is used, not the raw mean per year. The raw mean rises again after
the eighth year because of selective attrition (those who remain in follow-up
are healthier); the within-participant change removes the differences in level
between those who remain and those who leave, even though it does not entirely
cancel differential selection.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT, FIG = os.path.join(BASE, "outputs"), os.path.join(BASE, "figures")
W = os.path.join(BASE, "outputs", "cache") + "/"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
# MID was #9aa0a6: the dotted line of the middle stratum, drawn at 75%
# opacity, composited on white to (178, 182, 187) and measured 2.04:1, below
# the minimum of 3:1 for a graphical element that carries meaning. The grey has
# been darkened just enough to exceed the threshold once composited,
# while staying lighter than the other two strata.
FAST, MID, SLOW = "#c44536", "#7d8288", "#1f5f8b"
# Opacity of the non-significant panels. At 0.75 no grey light enough to
# stay subordinate to the other two strata reached 3:1: at that opacity even
# the red drops to 3.26:1 and leaves no margin above the threshold.
# At 0.90 the dimming remains visible and the three strata are all above the
# minimum. Significance is in any case encoded elsewhere (title in
# bold, axes coloured and thickened, p in bold and coloured).
ALPHA_NS = .90

# Channels redundant with colour: each stratum also carries its own dash
# pattern (and, where there are points, its own marker), so the series remain
# distinguishable in black and white too. Dashes long enough to be read
# as such at print size.
LS_FAST, LS_MID, LS_SLOW = (0, (5.5, 2.2)), (0, (1.4, 1.8)), "solid"
MK_FAST, MK_SLOW = "o", "s"

cur = pd.read_parquet(W + "curated.parquet")
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete].reset_index(drop=True)
d["T50"] = np.log(d.k_T50)
d["T50"] = (d.T50 - d.T50.mean()) / d.T50.std()
covbl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                   "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base = d[["PATNO", "T50"]].merge(covbl, on="PATNO", how="left")

SPEC = [
    ("moca", "MoCA", "Global cognition", True),
    ("SDMTOTAL", "Symbol Digit", "Executive/attention", True),
    ("rem", "RBDSQ", "REM sleep", False),
    ("hvlt_immediaterecall", "HVLT recall", "Memory", True),
    ("bjlot", "BJLOT", "Visuospatial", True),
    ("ess", "Epworth", "Sleepiness", False),
    ("updrs1_score", "MDS-UPDRS I", "Non-motor total", False),
    ("scopa", "SCOPA-AUT", "Autonomic", False),
    ("stai", "STAI", "Anxiety", False),
    ("gds", "GDS-15", "Depression", False),
    ("VLTANIM", "Semantic fluency", "Language", True),
    ("quip", "QUIP", "Impulse control", False),
]
prim = pd.read_csv(os.path.join(OUT, "15_primary_nonmotor.csv")).set_index('outcome')

L = cur[cur.PATNO.isin(base.PATNO)][["PATNO", "YEAR"] + [s[0] for s in SPEC]].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")

q = d.T50.quantile([1/3, 2/3]).values


def model_lines(var, educ, tmax=10):
    cov = ["age_bl", "SEX", "dur_bl"] + (["EDUCYRS"] if educ else [])
    m = L[["PATNO", "years", var, "T50"] + cov].dropna()
    r = smf.mixedlm(f"{var} ~ T50*years + " + " + ".join(cov), m,
                    groups=m.PATNO, re_formula="~years").fit(method="lbfgs",
                                                             maxiter=300)
    t = "T50:years" if "T50:years" in r.params.index else "years:T50"
    icept = r.params["Intercept"] + sum(r.params[c] * m[c].mean() for c in cov)
    yrs = np.linspace(0, tmax, 60)
    out = {}
    for z in (-1, 0, 1):
        out[z] = icept + r.params["T50"] * z + (r.params["years"] + r.params[t] * z) * yrs
    return yrs, out, m


def observed_change(var, tmax=10, min_n=25):
    """Mean within-participant change from own baseline, by tertile."""
    s = L[["PATNO", "years", var, "T50"]].dropna()
    b0 = s[s.years == 0].set_index("PATNO")[var]
    s = s[s.PATNO.isin(b0.index) & (s.years <= tmax)].copy()
    s["delta"] = s[var] - s.PATNO.map(b0)
    s['tertile'] = np.where(s.T50 <= q[0], -1, np.where(s.T50 >= q[1], 1, 0))
    res = {}
    for z in (-1, 1):
        g = s[s.tertile == z].groupby("years").delta.agg(["mean", "sem", "size"])
        res[z] = g[g["size"] >= min_n]
    return res, float(b0.mean())


# ===========================================================================
# Fig 10 — grid of the 12 outcomes
# ===========================================================================
fig, axes = plt.subplots(3, 4, figsize=(14.5, 9))
for ax, (var, lab, dom, educ) in zip(axes.ravel(), SPEC):
    yrs, lines, _ = model_lines(var, educ)
    sig = prim.loc[var, "p_BH"] < 0.05
    for z, col, lw, ls in [(-1, FAST, 2.1, LS_FAST), (0, MID, 1.4, LS_MID),
                           (1, SLOW, 2.1, LS_SLOW)]:
        ax.plot(yrs, lines[z], color=col, lw=lw, ls=ls,
                alpha=1.0 if sig else ALPHA_NS)
    ax.fill_between(yrs, lines[-1], lines[1],
                    color=SLOW if sig else MID, alpha=.12 if sig else .06, lw=0)
    pbh = prim.loc[var, "p_BH"]
    eff = prim.loc[var, "effetto_protettivo_%"]
    title = f"{lab}  ·  {dom}"
    ax.set_title(title, fontsize=9.5, fontweight="bold" if sig else "normal")
    # The annotation goes in the corner that the curves do not reach. The curves
    # rise in some panels and fall in others, so the corner cannot be
    # fixed by hand: the free space above and below the three lines is
    # measured in the left half of the frame, where the text sits, and the side
    # with more margin is chosen. This way no dashed line crosses the text.
    fin = yrs <= yrs.max() * .55
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    lo = min(float(lines[z][fin].min()) for z in (-1, 0, 1))
    hi = max(float(lines[z][fin].max()) for z in (-1, 0, 1))
    lower = (lo - ylo) >= (yhi - hi)
    ax.text(.03, .035 if lower else .965,
            f"p$_{{BH}}$ = {pbh:.3f}" + (f"   {eff:+.0f}%" if sig else ""),
            transform=ax.transAxes, fontsize=8,
            va="bottom" if lower else "top",
            color=SLOW if sig else "#6b7075",   # 5:1 on white: stays dimmed but legible
            fontweight="bold" if sig else "normal")
    ax.set_xlabel("Years", fontsize=8.5)
    ax.tick_params(labelsize=8)
    if sig:
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(SLOW)
            ax.spines[sp].set_linewidth(1.6)

# The legend samples carry the same opacity as the lines drawn: if
# no panel is significant all the curves are dimmed, and a key at
# full opacity would show a tone that does not appear in the plot.
A_LEG = 1.0 if any(prim.loc[v, "p_BH"] < 0.05 for v, _, _, _ in SPEC) else ALPHA_NS
hand = [plt.Line2D([], [], color=FAST, lw=2.2, ls=LS_FAST, alpha=A_LEG,
                   label="T50 −1 SD — faster seeding (dashed)"),
        plt.Line2D([], [], color=MID, lw=1.5, ls=LS_MID, alpha=A_LEG,
                   label="T50 at the mean (dotted)"),
        plt.Line2D([], [], color=SLOW, lw=2.2, ls=LS_SLOW, alpha=A_LEG,
                   label="T50 +1 SD — slower seeding (solid)")]
fig.legend(handles=hand, loc="lower center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(.5, -.015))
fig.tight_layout(rect=[0, .035, 1, 1])
fig.savefig(os.path.join(FIG, "fig10_trajectories_grid.png"), bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# Fig 11 — the three predicted outcomes, with the observed data
# ===========================================================================
THREE = [("moca", "MoCA", True, "points"),
         ("SDMTOTAL", "Symbol Digit Modalities", True, "points"),
         ("rem", "RBDSQ", False, "points")]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, (var, lab, educ, unit) in zip(axes, THREE):
    yrs, lines, m = model_lines(var, educ)
    obs, b0mean = observed_change(var)
    hand3 = []
    for z, col, ls, mk, name in [
            (-1, FAST, LS_FAST, MK_FAST,
             "T50 −1 SD (points: lowest T50 tertile)"),
            (1, SLOW, LS_SLOW, MK_SLOW,
             "T50 +1 SD (points: highest tertile)")]:
        ax.plot(yrs, lines[z], color=col, lw=2.3, ls=ls)
        g = obs[z]
        ax.errorbar(g.index, b0mean + g["mean"], yerr=g["sem"], fmt=mk, ms=4.5,
                    color=col, alpha=.75, capsize=2.5, lw=1.1, ls="none")
        # a single legend entry per stratum, showing both the dash pattern
        # of the line and the marker of the points
        hand3.append(plt.Line2D([], [], color=col, lw=2.3, ls=ls, marker=mk,
                                ms=4.5, label=f"model · {name}"))
    # Full opacity: here there is nothing to dim, and the middle stratum is the
    # lightest line of the three. At 0.8 it stayed below 3:1 on white.
    ax.plot(yrs, lines[0], color=MID, lw=1.2, ls=LS_MID)
    # The lines drawn are three: the median goes in the legend too, with the
    # same wording as in fig. 10, otherwise the reader sees three lines and
    # two keys. It sits in the middle, as in the order of fig. 10.
    hand3.insert(1, plt.Line2D([], [], color=MID, lw=1.2, ls=LS_MID,
                               label="model · T50 at the mean (dotted)"))
    e = prim.loc[var, "effetto_protettivo_%"]
    p = prim.loc[var, "p_BH"]
    ax.set_title(f"{lab}\nmodulation {e:+.0f}%  ·  p$_{{BH}}$ = {p:.3f}", fontsize=10)
    ax.set_xlabel("Years from baseline"); ax.set_ylabel(lab)
    # wide handlelength and markers at the ends: without this the central
    # symbol would cover the only gap in the dash pattern and the legend
    # would show two lines both solid
    # With three entries the legend is taller: in the panel that rises it must be
    # kept at top left, the only corner that neither the lines nor the error bars
    # reach; in those that fall it stays at bottom left.
    ax.legend(handles=hand3, fontsize=7, frameon=False, handlelength=4.6,
              numpoints=2,
              loc="upper left" if var == "rem" else "lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig11_three_outcomes.png"), bbox_inches="tight")
print('saved fig10_trajectories_grid.png and fig11_three_outcomes.png')
