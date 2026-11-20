"""
88 - How correlated the two dopaminergic outcomes are (bilateral striatum and
     putamen) and their individual slopes.

Why. The review of the fourth round observes that a significant regional
test and a non-significant one do not by themselves demonstrate a difference
between regions, all the more so if the two outcomes are strongly correlated.
Here that correlation is measured: on the repeated observations, on the levels
at baseline and on the individual OLS slopes, in the same 24-hour cohort
(595 participants) used by scripts 77 and 87.

Output: outputs/88_dat_correlation.csv, .txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
OUTCOMES = ["MIA_STRIATUM_BILAT", "MIA_PUTAMEN_BILAT"]

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
cc = cc[cc.complete_primary]

L = cur[cur.PATNO.isin(set(cc.PATNO))][
    ["PATNO", "EVENT_ID", "YEAR"] + OUTCOMES].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()]
m = L[["PATNO", "years"] + OUTCOMES].dropna()

log("=" * 78)
log('88 — CORRELATION BETWEEN THE TWO DOPAMINERGIC OUTCOMES')
log("=" * 78)
log(f'\n24 h cohort: {m.PATNO.nunique()} subjects, {len(m)} observations')

rows = []

r, p = stats.pearsonr(m[OUTCOMES[0]], m[OUTCOMES[1]])
rs, ps = stats.spearmanr(m[OUTCOMES[0]], m[OUTCOMES[1]])
log(f'\nall repeated observations (n = {len(m)}):')
log(f'  Pearson  r = {r:.4f}  (p = {p:.3g})')
log(f'  Spearman r = {rs:.4f}  (p = {ps:.3g})')
rows.append({'level': 'repeated observations', "n": len(m),
              "pearson_r": r, "pearson_p": p,
              "spearman_r": rs, "spearman_p": ps})

b = m.sort_values("years").groupby("PATNO").first().reset_index()
r, p = stats.pearsonr(b[OUTCOMES[0]], b[OUTCOMES[1]])
rs, ps = stats.spearmanr(b[OUTCOMES[0]], b[OUTCOMES[1]])
log(f'\nfirst observation of each subject (n = {len(b)}):')
log(f'  Pearson  r = {r:.4f}  (p = {p:.3g})')
log(f'  Spearman r = {rs:.4f}  (p = {ps:.3g})')
rows.append({'level': 'first observation', "n": len(b),
              "pearson_r": r, "pearson_p": p,
              "spearman_r": rs, "spearman_p": ps})

# individual OLS slopes, same criterion as script 77 (>=3 observations)
pen = []
for pid, g in m.groupby("PATNO"):
    if len(g) >= 3 and g.years.nunique() >= 2:
        x = g.years.to_numpy(float)
        pen.append({"PATNO": pid,
                    "s_str": np.polyfit(x, g[OUTCOMES[0]].to_numpy(float), 1)[0],
                    "s_put": np.polyfit(x, g[OUTCOMES[1]].to_numpy(float), 1)[0]})
pen = pd.DataFrame(pen)
r, p = stats.pearsonr(pen.s_str, pen.s_put)
rs, ps = stats.spearmanr(pen.s_str, pen.s_put)
log(f'\nindividual OLS slopes, subjects with >=3 observations (n = {len(pen)}):')
log(f'  Pearson  r = {r:.4f}  (p = {p:.3g})')
log(f'  Spearman r = {rs:.4f}  (p = {ps:.3g})')
rows.append({'level': 'individual slopes', "n": len(pen),
              "pearson_r": r, "pearson_p": p,
              "spearman_r": rs, "spearman_p": ps})

log("\n" + "=" * 78)
log('READING')
log("=" * 78)
log("The two outcomes are not independent: the putamen is a part of the striatum,")
log('and the individual slopes share most of the variation. A significant')
log("test on one region and a non-significant one on the other do not")
log("therefore constitute evidence of a difference between regions; the direct")
log("comparison between the two associations has not been carried out.")

pd.DataFrame(rows).to_csv(
    os.path.join(OUT, "88_dat_correlation.csv"), index=False)
with open(os.path.join(OUT, "88_dat_correlation.txt"), "w") as f:
    f.write("\n".join(rep) + "\n")
log('\nwritten outputs/88_dat_correlation.csv, .txt')
