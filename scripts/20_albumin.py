"""
20 - Is the "concentration factor" really a concentration effect?

In the manuscript the factor is estimated indirectly, as the first principal
component of the ranks of the CSF analytes: its interpretation as a
scale effect of the sample is an inference from the covariance structure,
not a measurement.

CSF albumin allows the direct test. It is the most abundant protein of the
CSF and tracks its overall protein content; the ratio to
plasma albumin (Qalb) is moreover the standard index of permeability of the
blood-brain barrier. If the shared component were an effect of
concentration of the sample, it should correlate with CSF albumin; if
instead it were permeability of the barrier, it should correlate with Qalb.

Output: outputs/20_albumin.txt, outputs/20_*.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
from _alb import path_alb
ALB = path_alb(OUT)   # rebuilt from the PPMI biospecimen file
rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"]

# ---------------------------------------------------------------------------
# Albumin at the baseline visit
# ---------------------------------------------------------------------------
a = pd.read_csv(ALB)
a = a[a.CLINICAL_EVENT.isin(["BL", "SC"])].copy()
a["v"] = pd.to_numeric(a.TESTVALUE, errors="coerce")
alb = (a[a.TESTNAME == "CSF Albumin"].groupby("PATNO").v.mean()
       .rename("alb_csf").reset_index())
plq = (a[a.TESTNAME == "Plasma Albumin"].groupby("PATNO").v.mean()
       .rename("alb_plasma").reset_index())
alb = alb.merge(plq, on="PATNO", how="outer")
alb["qalb"] = alb.alb_csf / alb.alb_plasma
log(f'CSF albumin at baseline: {alb.alb_csf.notna().sum()} subjects')

ANALYTES = ["abeta", "tau", "ptau", "asyn", "IU_pTau181_CSF",
            "IU_ABeta40_CSF", "IU_ABeta42_CSF"]
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}
KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}


def prep(fn, kin_map, tag):
    d = pd.read_csv(os.path.join(OUT, fn))
    d = d[d.complete] if "complete" in d.columns else d[d.complete_primary]
    k = d[["PATNO"] + list(kin_map)].rename(columns=kin_map)
    for c in kin_map.values():
        k[c] = np.log(k[c])
    cols = ["PATNO", "age_at_visit", "SEX"] + [c for c in ANALYTES if c in bl.columns]
    m = k.merge(bl[cols].rename(columns={"age_at_visit": "age_bl"}),
                on="PATNO", how="left").merge(alb, on="PATNO", how="left")
    # concentration factor, as in script 12
    av = [c for c in ANALYTES if c in m.columns and m[c].notna().sum() >= 150]
    sub = m[av].dropna()
    R = np.column_stack([stats.rankdata(sub[c]) for c in av])
    R = (R - R.mean(0)) / R.std(0)
    f = PCA(n_components=1).fit_transform(R).ravel()
    if stats.spearmanr(f, sub[av[0]]).statistic < 0:
        f = -f
    m["conc"] = np.nan
    m.loc[sub.index, "conc"] = f
    log(f"\n[{tag}] n = {len(m)} | with factor {m.conc.notna().sum()} | with CSF albumin "
        f"{m.alb_csf.notna().sum()} | with both {m[['conc', 'alb_csf']].dropna().shape[0]}")
    return m, av


d150, AV150 = prep("08_dataset_150h.csv", KIN150, "150h")
d24, AV24 = prep("analysis_complete_cases.csv", KIN24, "24h")


def partial(y, x, Z):
    Zc = sm.add_constant(Z)
    ry = sm.OLS(stats.rankdata(y), Zc).fit().resid
    rx = sm.OLS(stats.rankdata(x), Zc).fit().resid
    return stats.pearsonr(ry, rx)


# ---------------------------------------------------------------------------
# 1. Does the shared component correspond to CSF albumin?
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("1. DOES THE COMMON FACTOR CORRESPOND TO CSF ALBUMIN?")
log("=" * 74)
rows = []
for d, tag in [(d150, "150h"), (d24, "24h")]:
    m = d[["conc", "alb_csf", "alb_plasma", "qalb"]].dropna(subset=["conc", "alb_csf"])
    if len(m) < 30:
        log(f'  [{tag}] insufficient overlap (n = {len(m)})')
        continue
    r_alb, p_alb = stats.spearmanr(m.conc, m.alb_csf)
    mq = m.dropna(subset=["qalb"])
    r_q, p_q = stats.spearmanr(mq.conc, mq.qalb)
    r_pl, p_pl = stats.spearmanr(mq.conc, mq.alb_plasma)
    rows.append({'cohort': tag, "n": len(m),
                 'rho_factor_vs_albumin_CSF': r_alb, "p_alb": p_alb,
                 'rho_factor_vs_Qalb': r_q, "p_Qalb": p_q,
                 'rho_factor_vs_albumin_plasma': r_pl, "p_plasma": p_pl})
res1 = pd.DataFrame(rows)
log(res1.round(4).to_string(index=False))
res1.to_csv(os.path.join(OUT, "20_component_vs_albumin.csv"), index=False)

# ---------------------------------------------------------------------------
# 2. Albumin and individual analytes
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("2. DOES CSF ALBUMIN CORRELATE WITH ALL THE ANALYTES?")
log("=" * 74)
rows = []
for d, tag, av in [(d150, "150h", AV150), (d24, "24h", AV24)]:
    for c in av:
        m = d[[c, "alb_csf"]].dropna()
        if len(m) < 40:
            continue
        r, p = stats.spearmanr(m[c], m.alb_csf)
        rows.append({'cohort': tag, 'analyte': c, "n": len(m), "rho": r, "p": p})
res2 = pd.DataFrame(rows)
if len(res2):
    res2["p_BH"] = multipletests(res2.p, method="fdr_bh")[1]
    log(res2.round(4).to_string(index=False))
    res2.to_csv(os.path.join(OUT, "20_albumin_vs_analytes.csv"), index=False)

# ---------------------------------------------------------------------------
# 3. Does adjusting for albumin replicate the effect of the factor?
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("3. DOES ADJUSTING FOR ALBUMIN ABOLISH THE KINETIC PARAMETER-BIOMARKER ASSOCIATIONS?")
log("=" * 74)
log('Comparison of three adjustments on the SAME subset of subjects.')
rows = []
for d, tag, kins, av in [(d150, "150h", list(KIN150.values()), AV150),
                         (d24, "24h", list(KIN24.values()), AV24)]:
    for k in kins:
        for b in av:
            m = d[[k, b, "age_bl", "SEX", "conc", "alb_csf"]].dropna()
            if len(m) < 60:
                continue
            Z2 = m[["age_bl", "SEX"]].to_numpy(float)
            r2, p2 = partial(m[b], m[k], Z2)
            r3, p3 = partial(m[b], m[k],
                             np.column_stack([Z2, m.conc.to_numpy(float)]))
            r4, p4 = partial(m[b], m[k],
                             np.column_stack([Z2, stats.rankdata(m.alb_csf)]))
            rows.append({'cohort': tag, 'kinetic': k, 'biomarker': b, "n": len(m),
                         'rho_age_sex': r2, 'p_age_sex': p2,
                         'rho_more_factor': r3, 'p_factor': p3,
                         'rho_more_albumin': r4, 'p_albumin': p4})
res3 = pd.DataFrame(rows)
if len(res3):
    for col, lab in [('p_age_sex', "L2"), ('p_factor', "L3"), ('p_albumin', "L4")]:
        res3[f"pBH_{lab}"] = np.nan
        for tag in res3.cohort.unique():
            msk = res3.cohort == tag
            res3.loc[msk, f"pBH_{lab}"] = multipletests(res3.loc[msk, col],
                                                        method="fdr_bh")[1]
    for tag in res3.cohort.unique():
        s = res3[res3.cohort == tag]
        log(f'\n[{tag}] out of {len(s)} pairs, significant after BH-FDR:')
        log(f'  adjusting for age and sex              : {int((s.pBH_L2 < 0.05).sum())}')
        log(f'  + concentration factor (PC1)           : {int((s.pBH_L3 < 0.05).sum())}')
        log(f'  + CSF albumin (direct measurement)     : {int((s.pBH_L4 < 0.05).sum())}')
        log(f'  maximum |rho|: {s.rho_age_sex.abs().max():.3f} -> '
            f'{s.rho_more_factor.abs().max():.3f} (PC1) / {s.rho_more_albumin.abs().max():.3f} '
            f'(albumin)')
    res3.to_csv(os.path.join(OUT, "20_adjustment_albumin.csv"), index=False)

# ---------------------------------------------------------------------------
# 4. Do the kinetic parameters correlate directly with albumin?
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("4. DO THE KINETIC PARAMETERS CORRELATE DIRECTLY WITH CSF ALBUMIN?")
log("=" * 74)
rows = []
for d, tag, kins in [(d150, "150h", list(KIN150.values())),
                     (d24, "24h", list(KIN24.values()))]:
    for k in kins:
        m = d[[k, "alb_csf", "age_bl", "SEX"]].dropna()
        if len(m) < 40:
            continue
        r, p = partial(m.alb_csf, m[k], m[["age_bl", "SEX"]].to_numpy(float))
        rows.append({'cohort': tag, 'kinetic': k, "n": len(m),
                     'rho_partial': r, "p": p})
res4 = pd.DataFrame(rows)
if len(res4):
    res4["p_BH"] = multipletests(res4.p, method="fdr_bh")[1]
    log(res4.round(4).to_string(index=False))
    res4.to_csv(os.path.join(OUT, "20_kinetics_vs_albumin.csv"), index=False)

with open(os.path.join(OUT, "20_albumin.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/20_albumin.txt')
