"""
100 - The albumin controls extended to the primary cohort.

In the table of the albumin-kinetic parameter associations three columns are
empty for the 24-hour cohort: confidence interval, adjustment for the
haemoglobin indicator, replacement with Qalb. The data are not missing — CSF
and plasma albumin and the haemoglobin indicator are per participant, not per
protocol — the calculation is missing: script 23 ran them on the 150-hour
cohort only.

Here they are also run on the primary cohort, with the same method to the
letter: the function `partial` is copied from 23_review_P17_P19.py, partial
rank correlation via OLS residuals on the ranks, with a Fisher interval that
takes into account the degrees of freedom spent on the covariates.

TSmax is added, which the 24-hour protocol reports and which the original
analysis had omitted: the 150-hour cohort contributed five parameters and the
primary cohort four, an asymmetry with no stated reason. With TSmax the
correction family goes from nine to ten tests and must be recomputed.

To be read with the caveat that matters: the primary cohort has 55 participants
with CSF albumin, and the subsets of the two controls are smaller
still. The intervals here are wide, and that is why they must be printed.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 scripts/100_albumin_primary_cohort.py
Output: outputs/100_albumin_primary_cohort.csv, .txt
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _alb import path_alb                                    # noqa: E402

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def partial(y, x, Z):
    """Copied to the letter from 23_review_P17_P19.py."""
    Zc = sm.add_constant(Z)
    ry = sm.OLS(stats.rankdata(y), Zc).fit().resid
    rx = sm.OLS(stats.rankdata(x), Zc).fit().resid
    r, p = stats.pearsonr(ry, rx)
    n = len(y)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - Zc.shape[1] - 2)
    return r, p, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), n


# ------------------------------------------------------------------ data
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO")

a = pd.read_csv(path_alb(OUT))
a = a[a.CLINICAL_EVENT.isin(["BL", "SC"])].copy()
a["v"] = pd.to_numeric(a.TESTVALUE, errors="coerce")
alb = a[a.TESTNAME == "CSF Albumin"].groupby("PATNO").v.mean().rename("alb_csf")
plq = a[a.TESTNAME == "Plasma Albumin"].groupby("PATNO").v.mean().rename("alb_pl")
A = pd.concat([alb, plq], axis=1).reset_index()
A["qalb"] = A.alb_csf / A.alb_pl

KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_TSmax": "TSmax", "saa_SLOPEMax": "SLOPE"}
d = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
if "complete_primary" in d.columns:
    d = d[d.complete_primary]
D = d[["PATNO"] + list(KIN24)].rename(columns=KIN24)
for c in KIN24.values():
    D[c] = np.log(D[c])
D = D.merge(A, on="PATNO", how="left").merge(
    bl[["PATNO", "age_at_visit", "SEX", "hemohi"]].rename(
        columns={"age_at_visit": "age_bl"}), on="PATNO", how="left")

log("=" * 78)
log('100 — ALBUMIN AND KINETIC PARAMETERS IN THE PRIMARY COHORT: THE MISSING CHECKS')
log("=" * 78)
log(f'primary cohort: {len(D)} subjects')
log(f'  with CSF albumin              : {D.alb_csf.notna().sum()}')
log(f'  with plasma albumin (Qalb)    : {D.qalb.notna().sum()}')
log(f"  with haemoglobin indicator    : {D[['alb_csf', 'hemohi']].dropna().shape[0]}")
log("")

rows = []
for k in KIN24.values():
    base = D[[k, "alb_csf", "age_bl", "SEX"]].dropna()
    r0, p0, lo0, hi0, n0 = partial(base.alb_csf, base[k],
                                   base[["age_bl", "SEX"]].to_numpy(float))
    m = D[[k, "alb_csf", "age_bl", "SEX", "hemohi"]].dropna()
    if len(m) > 5:
        r1, p1, lo1, hi1, n1 = partial(
            m.alb_csf, m[k], m[["age_bl", "SEX", "hemohi"]].to_numpy(float))
    else:
        r1 = p1 = lo1 = hi1 = np.nan
        n1 = len(m)
    q = D[[k, "qalb", "age_bl", "SEX"]].dropna()
    if len(q) > 5:
        r2, p2, lo2, hi2, n2 = partial(q.qalb, q[k],
                                       q[["age_bl", "SEX"]].to_numpy(float))
    else:
        r2 = p2 = lo2 = hi2 = np.nan
        n2 = len(q)
    rows.append({'cohort': "24h", 'kinetic': k, "n_alb": n0,
                  'rho_albumin': r0, 'CI_inf': lo0, 'CI_upp': hi0, "p": p0,
                  'rho_agg_haemoglobin': r1, "p_agg": p1,
                  'CI_agg_inf': lo1, 'CI_agg_upp': hi1, "n_agg": n1,
                  "rho_Qalb": r2, "p_Qalb": p2,
                  'CI_Qalb_inf': lo2, 'CI_Qalb_upp': hi2, "n_Qalb": n2})

P = pd.DataFrame(rows)

log("=" * 78)
log('THE PRIMARY COHORT, WITH THE INTERVALS AND THE TWO CHECKS')
log("=" * 78)
log(f"{'par.':7s} {'n':>3s} {'rho':>7s} {'95% CI':>18s} {'p':>9s} | {'n':>3s} {'rho|Hb':>7s} "
    f"{'p|Hb':>9s} | {'n':>3s} {'rho Q':>7s} {'p Q':>9s}")
for _, r in P.iterrows():
    log(f'{r.kinetic:7s} {int(r.n_alb):3d} {r.rho_albumin:+7.3f} [{r.CI_inf:+.3f}, {r.CI_upp:+.3f}] '
        f'{r.p:9.5f} | {int(r.n_agg):3d} {r.rho_agg_haemoglobin:+7.3f} {r.p_agg:9.5f} | '
        f'{int(r.n_Qalb):3d} {r.rho_Qalb:+7.3f} {r.p_Qalb:9.5f}')

# ------------------------------------- the BH family recomputed at ten tests
log("")
log("=" * 78)
log('THE CORRECTION FAMILY CHANGES: FROM NINE TESTS TO TEN')
log("=" * 78)
v = pd.read_csv(os.path.join(OUT, "20_kinetics_vs_albumin.csv"))
r150 = v[v.cohort == "150h"][['cohort', 'kinetic', "n", 'rho_partial', "p"]]
r24 = P[['cohort', 'kinetic', "n_alb", 'rho_albumin', "p"]].rename(
    columns={"n_alb": "n", 'rho_albumin': 'rho_partial'})
everyone = pd.concat([r150, r24], ignore_index=True)
_, bh, _, _ = multipletests(everyone.p.values, alpha=0.05, method="fdr_bh")
everyone['p_BH_ten'] = bh

prior = v.set_index(['cohort', 'kinetic']).p_BH
log(f"{'cohort':7s} {'par.':7s} {'p':>10s} {'BH at 9':>9s} {'BH at 10':>9s}  note")
for _, r in everyone.iterrows():
    old = prior.get((r.cohort, r.kinetic), np.nan)
    remark = 'TSmax, new' if r.kinetic == "TSmax" else ""
    if not np.isnan(old) and (old < 0.05) != (r.p_BH_ten < 0.05):
        remark = 'CHANGES STATUS'
    log(f"{r.cohort:7s} {r.kinetic:7s} {r.p:10.6f} "
        f"{(old if not np.isnan(old) else float('nan')):9.6f} {r.p_BH_ten:9.6f}  {remark}")

above_9 = int((v.p_BH < 0.05).sum())
above_10 = int((everyone.p_BH_ten < 0.05).sum())
log("")
log(f'associations that pass the correction: {above_9} out of 9 before, {above_10} out of 10 now')

everyone.to_csv(os.path.join(OUT, "100_albumin_primary_cohort.csv"), index=False)
P.to_csv(os.path.join(OUT, "100_checks_cohort_primary.csv"), index=False)

log("")
log("=" * 78)
log('HOW TO READ IT')
log("=" * 78)
log('The intervals of the primary cohort are wide: with 55 subjects the')
log("Fisher half-width is about 0.27 on the z scale. A coefficient of 0.5")
log("remains far from zero, but its precision is not comparable with")
log("that of the replication, and this is why the intervals should be")
log('printed rather than summarised in a p value.')
log("The two checks run on even smaller subsets, indicated")
log("row by row: where n falls, the absence of significance is not")
log('informative.')

with open(os.path.join(OUT, "100_albumin_primary_cohort.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/100_albumin_primary_cohort.csv, 100_checks_cohort_primary.csv and .txt')
