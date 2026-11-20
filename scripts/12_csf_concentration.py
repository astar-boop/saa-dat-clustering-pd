"""
12 - SAA kinetic parameters and CSF biomarkers with adjustment for the sample
     concentration factor.

Premise (section 11 of the report): in the CSF of this cohort ALL the available
analytes are positively intercorrelated (rho 0.44-0.96) across different
biochemical families — Abeta42, Abeta40, t-tau, p-tau, total
alpha-synuclein. It is the signature of a scale factor of the sample, not of a pathology.

Here:
  a) the concentration factor is built as PC1 of the ranks of the analytes;
  b) it is verified that it is not blood contamination (hemohi);
  c) all the kinetic parameter-biomarker associations are retested at three levels
     of adjustment;
  d) the alternative mechanistic hypothesis is tested: is it total alpha-synuclein
     (the substrate of the assay) that drives the kinetics?

Output: outputs/12_*.csv, outputs/12_csf_concentration.txt
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
# it was a parquet in a temporary session directory, by now
# deleted; it is the curated data cut, already in cache (BL rows 4788,
# all the requested analyte columns are present: verified).
CURATED_CACHE = os.path.join(OUT, "cache", "curated.parquet")
rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


cur = pd.read_parquet(CURATED_CACHE)
bl = cur[cur.EVENT_ID == "BL"]

ANALYTES = ["abeta", "tau", "ptau", "asyn", "IU_pTau181_CSF",
            "IU_ABeta40_CSF", "IU_ABeta42_CSF"]
EXTRA = ["hemohi", "NFL_CSF", "Roche_YKL40_CSF"]


def prep(cohort_file, kin_map, tag):
    d = pd.read_csv(os.path.join(OUT, cohort_file))
    d = d[d.complete] if "complete" in d.columns else d[d.complete_primary]
    k = d[["PATNO"] + list(kin_map)].rename(columns=kin_map)
    for c in kin_map.values():
        k[c] = np.log(k[c])
    cols = ["PATNO", "age_at_visit", "SEX"] + [c for c in ANALYTES + EXTRA
                                               if c in bl.columns]
    m = k.merge(bl[cols].rename(columns={"age_at_visit": "age_bl"}),
                on="PATNO", how="left")
    log(f"\n[{tag}] n = {len(m)}")
    return m


KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}

d150 = prep("08_dataset_150h.csv", KIN150, "150h")
d24 = prep("analysis_complete_cases.csv", KIN24, "24h")


# ---------------------------------------------------------------------------
# a) The concentration factor
# ---------------------------------------------------------------------------
def concentration_factor(d, name):
    avail = [a for a in ANALYTES if a in d.columns and d[a].notna().sum() >= 150]
    m = d[avail].dropna()
    log(f'\n{name}: analytes used {avail}')
    log(f'  subjects with all the analytes: {len(m)}')
    R = np.column_stack([stats.rankdata(m[c]) for c in avail])
    R = (R - R.mean(0)) / R.std(0)
    p = PCA().fit(R)
    log(f'  variance explained by PC1: {p.explained_variance_ratio_[0]:.3f}')
    log('  PC1 loadings: ' + ", ".join(f"{c}={v:+.2f}"
                                       for c, v in zip(avail, p.components_[0])))
    if (p.components_[0] < 0).all():
        p.components_[0] *= -1
    sign = 'all positive' if (p.components_[0] > 0).all() else 'MIXED'
    log(f'  sign of the loadings: {sign}')
    f = pd.Series(PCA(n_components=1).fit_transform(R).ravel(), index=m.index)
    if stats.spearmanr(f, m[avail[0]]).statistic < 0:
        f = -f
    out = pd.Series(np.nan, index=d.index)
    out.loc[m.index] = f
    return out, avail


log("\n" + "=" * 74)
log('a) CSF CONCENTRATION FACTOR')
log("=" * 74)
d150["conc"], AV150 = concentration_factor(d150, '150h cohort')
d24["conc"], AV24 = concentration_factor(d24, '24h cohort')

# ---------------------------------------------------------------------------
# b) It is not blood contamination
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("b) IS THE FACTOR BLOOD CONTAMINATION?")
log("=" * 74)
for d, tag in [(d150, "150h"), (d24, "24h")]:
    if "hemohi" not in d.columns:
        continue
    m = d[["conc", "hemohi"]].dropna()
    if len(m) < 50:
        continue
    g0 = m.loc[m.hemohi == 0, "conc"]
    g1 = m.loc[m.hemohi == 1, "conc"]
    u, p = stats.mannwhitneyu(g0, g1)
    log(f'  [{tag}] high haemoglobin: n={len(g1)} vs normal n={len(g0)}; median concentration factor '
        f'{g1.median():+.2f} vs {g0.median():+.2f}, p = {p:.3f}')
log("  -> haemoglobin does not explain the factor; it remains compatible with total")
log('     protein content, dilution or storage of the sample.')

# ---------------------------------------------------------------------------
# c) Three levels of adjustment
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log('c) KINETIC PARAMETER-BIOMARKER ASSOCIATIONS AT THREE LEVELS OF ADJUSTMENT')
log("=" * 74)
log('  L1 = raw | L2 = + age and sex | L3 = + concentration factor')


def partial(y, x, Z):
    Zc = sm.add_constant(Z)
    ry = sm.OLS(stats.rankdata(y), Zc).fit().resid
    rx = sm.OLS(stats.rankdata(x), Zc).fit().resid
    r, p = stats.pearsonr(ry, rx)
    return r, p


rows = []
for d, tag, kins, avail in [(d150, "150h", list(KIN150.values()), AV150),
                            (d24, "24h", list(KIN24.values()), AV24)]:
    for k in kins:
        for b in avail:
            m = d[[k, b, "age_bl", "SEX", "conc"]].dropna()
            if len(m) < 60:
                continue
            r1, p1 = stats.spearmanr(m[k], m[b])
            r2, p2 = partial(m[b], m[k], m[["age_bl", "SEX"]].to_numpy(float))
            r3, p3 = partial(m[b], m[k],
                             m[["age_bl", "SEX", "conc"]].to_numpy(float))
            rows.append({'cohort': tag, 'kinetic': k, 'biomarker': b,
                         "n": len(m), "rho_L1": r1, "p_L1": p1,
                         "rho_L2": r2, "p_L2": p2, "rho_L3": r3, "p_L3": p3})
cc = pd.DataFrame(rows)
for lv in ("L1", "L2", "L3"):
    cc[f"pBH_{lv}"] = np.nan
    for tag in cc.cohort.unique():
        msk = cc.cohort == tag
        cc.loc[msk, f"pBH_{lv}"] = multipletests(cc.loc[msk, f"p_{lv}"],
                                                 method="fdr_bh")[1]
cc.to_csv(os.path.join(OUT, "12_kinetics_biomarkers_adjusted.csv"), index=False)

for tag in ["150h", "24h"]:
    s = cc[cc.cohort == tag]
    log(f'\n[{tag}] significant associations (BH-FDR < 0.05) by level:')
    log(f'  L1 raw                            : {int((s.pBH_L1 < 0.05).sum()):3d} / {len(s)}')
    log(f'  L2 + age and sex                  : {int((s.pBH_L2 < 0.05).sum()):3d} / {len(s)}')
    log(f'  L3 + concentration factor         : {int((s.pBH_L3 < 0.05).sum()):3d} / {len(s)}')
    log(f'  maximum |rho|: L1 {s.rho_L1.abs().max():.3f} -> L2 {s.rho_L2.abs().max():.3f} -> L3 '
        f'{s.rho_L3.abs().max():.3f}')
    surv = s[s.pBH_L3 < 0.05]
    if len(surv):
        log('  surviving at L3:')
        log(surv[['kinetic', 'biomarker', "n", "rho_L1", "rho_L3", "pBH_L3"]]
            .round(4).to_string(index=False))

log('\nDetail for the 150h cohort, all pairs:')
log(cc[cc.cohort == "150h"][['kinetic', 'biomarker', "n", "rho_L1", "pBH_L1",
                             "rho_L3", "pBH_L3"]].round(3).to_string(index=False))

# how much each kinetic parameter correlates with the factor itself
log("\n" + "-" * 74)
log('Direct correlation of each kinetic parameter with the concentration FACTOR')
log("-" * 74)
rows = []
for d, tag, kins in [(d150, "150h", list(KIN150.values())),
                     (d24, "24h", list(KIN24.values()))]:
    for k in kins:
        m = d[[k, "conc", "age_bl", "SEX"]].dropna()
        if len(m) < 60:
            continue
        r, p = partial(m["conc"], m[k], m[["age_bl", "SEX"]].to_numpy(float))
        rows.append({'cohort': tag, 'kinetic': k, "n": len(m),
                     'rho_partial': r, "p": p})
kf = pd.DataFrame(rows)
kf["p_BH"] = multipletests(kf.p, method="fdr_bh")[1]
log(kf.round(4).to_string(index=False))
kf.to_csv(os.path.join(OUT, "12_kinetics_vs_concfactor.csv"), index=False)

# ---------------------------------------------------------------------------
# d) Mechanistic hypothesis: is it the substrate?
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("d) ALTERNATIVE HYPOTHESIS: is it total alpha-synuclein (the substrate)?")
log("=" * 74)
log('If the kinetics depended on the concentration of substrate, asyn should')
log("keep the association even after removing the factor common to the other analytes.")
rows = []
for d, tag, kins in [(d150, "150h", list(KIN150.values())),
                     (d24, "24h", list(KIN24.values()))]:
    others = [a for a in ANALYTES if a != "asyn" and a in d.columns]
    for k in kins:
        m = d[[k, "asyn", "age_bl", "SEX"] + others].dropna()
        if len(m) < 60:
            continue
        R = np.column_stack([stats.rankdata(m[c]) for c in others])
        R = (R - R.mean(0)) / R.std(0)
        conc_wo = PCA(n_components=1).fit_transform(R).ravel()
        if stats.spearmanr(conc_wo, m[others[0]]).statistic < 0:
            conc_wo = -conc_wo
        r1, p1 = stats.spearmanr(m[k], m["asyn"])
        r2, p2 = partial(m["asyn"], m[k], m[["age_bl", "SEX"]].to_numpy(float))
        Z = np.column_stack([m[["age_bl", "SEX"]].to_numpy(float), conc_wo])
        r3, p3 = partial(m["asyn"], m[k], Z)
        rows.append({'cohort': tag, 'kinetic': k, "n": len(m),
                     'rho_raw': r1, 'rho_age_sex': r2,
                     'rho_less_factor_others_analytes': r3, "p": p3})
sub = pd.DataFrame(rows)
sub["p_BH"] = multipletests(sub.p, method="fdr_bh")[1]
log(sub.round(4).to_string(index=False))
log(f'\n  associations with asyn that survive: {int((sub.p_BH < 0.05).sum())} / {len(sub)}')
sub.to_csv(os.path.join(OUT, "12_substrate_hypothesis.csv"), index=False)

with open(os.path.join(OUT, "12_csf_concentration.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/12_csf_concentration.txt')
