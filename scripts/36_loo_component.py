"""
36 - «Leave-one-analyte-out» shared component.

The review observes that adjusting the association between a kinetic parameter
and an analyte for a principal component that contains that same analyte can
attenuate the association mechanically: part of the variance of the analyte is
already inside the regressor.

Here the component is rebuilt, for each kinetic parameter-analyte pair,
excluding the analyte under examination from the calculation of the principal
component. The comparison between the two adjustments says whether the
cancellation of the associations reported in §3.16 is substantive or circular.

Output: outputs/36_loo_component.csv, outputs/36_loo_component.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


ANALYTES = ["abeta", "tau", "ptau", "asyn", "IU_pTau181_CSF",
            "IU_ABeta40_CSF", "IU_ABeta42_CSF"]
NICE = {"abeta": "Aβ42 (Elecsys)", "tau": "t-tau (Elecsys)",
        "ptau": "p-tau181 (Elecsys)", "asyn": 'α-syn total (ELISA)',
        "IU_pTau181_CSF": "p-tau181 (Indiana)",
        "IU_ABeta40_CSF": "Aβ40 (Indiana)",
        "IU_ABeta42_CSF": "Aβ42 (Indiana)"}


def factor(M, cols):
    """PC1 of the ranks, positively oriented."""
    R = np.column_stack([stats.rankdata(M[c]) for c in cols])
    R = (R - R.mean(0)) / R.std(0)
    f = PCA(n_components=1).fit_transform(R).ravel()
    if np.mean([stats.spearmanr(f, M[c]).statistic for c in cols]) < 0:
        f = -f
    return f


def partial(x, y, Z):
    """Partial correlation on the ranks, adjusted for the columns of Z."""
    Zc = sm.add_constant(np.asarray(Z, float))
    rx = sm.OLS(stats.rankdata(x), Zc).fit().resid
    ry = sm.OLS(stats.rankdata(y), Zc).fit().resid
    r, p = stats.pearsonr(rx, ry)
    return r, p


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"]

KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC",
          "k_T50": "T50", "k_SLOPE": "SLOPE"}

d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete]
k = d[["PATNO"] + list(KIN150)].rename(columns=KIN150)
for c in KIN150.values():
    k[c] = np.log(k[c])
cols = ["PATNO", "age_at_visit", "SEX"] + ANALYTES
M = k.merge(bl[cols].rename(columns={"age_at_visit": "age_bl"}),
            on="PATNO", how="left").dropna(subset=ANALYTES + ["age_bl", "SEX"])

log("=" * 78)
log('LEAVE-ONE-ANALYTE-OUT SHARED COMPONENT — 150 h cohort')
log("=" * 78)
log(f'n = {len(M)} subjects with all seven analytes')
log(f'pairs examined: {len(KIN150)} kinetic parameters × {len(ANALYTES)} analytes = '
    f'{len(KIN150) * len(ANALYTES)}')
log("")

f_full = factor(M, ANALYTES)
var_full = PCA(n_components=1).fit(
    np.column_stack([stats.zscore(stats.rankdata(M[c])) for c in ANALYTES])
).explained_variance_ratio_[0]
log(f'Seven-analyte component: variance explained by PC1 = {var_full:.3f}')

rows = []
for a in ANALYTES:
    others = [x for x in ANALYTES if x != a]
    f_loo = factor(M, others)
    rho_ff = stats.spearmanr(f_full, f_loo).statistic
    for kn in KIN150.values():
        gr, gp = stats.spearmanr(M[kn], M[a])
        ar, ap = partial(M[kn], M[a], M[["age_bl", "SEX"]])
        pr, pp = partial(M[kn], M[a],
                          np.column_stack([M[["age_bl", "SEX"]], f_full]))
        lr, lp = partial(M[kn], M[a],
                          np.column_stack([M[["age_bl", "SEX"]], f_loo]))
        rows.append({'kinetic': kn, 'analyte': NICE[a], "n": len(M),
                     'rho_raw': gr, 'p_raw': gp,
                     'rho_age_sex': ar, 'p_age_sex': ap,
                     'rho_factor_full': pr, 'p_factor_full': pp,
                     'rho_factor_loo': lr, 'p_factor_loo': lp,
                     'rho_factors': rho_ff})

res = pd.DataFrame(rows)
for c in ['p_age_sex', 'p_factor_full', 'p_factor_loo']:
    res[c.replace("p_", "pBH_")] = multipletests(res[c], method="fdr_bh")[1]
res.to_csv(os.path.join(OUT, "36_loo_component.csv"), index=False)

log('\nCorrelation between full component and leave-one-out component, by excluded analyte:')
for a in ANALYTES:
    r = res[res.analyte == NICE[a]].rho_factors.iloc[0]
    log(f'  excluded {NICE[a]:22s} ρ = {r:.4f}')

log("\n" + "=" * 78)
log('SIGNIFICANT ASSOCIATIONS (BH-FDR < 0.05) BY LEVEL OF ADJUSTMENT')
log("=" * 78)
tot = len(res)
for lbl, col in [('age and sex', 'pBH_age_sex'),
                 ('+ full factor (contains the analyte)', 'pBH_factor_full'),
                 ('+ leave-one-out factor', 'pBH_factor_loo')]:
    n = int((res[col] < 0.05).sum())
    log(f"  {lbl:38s} {n:2d} / {tot}")

log('\nMaximum |ρ| by level:')
for lbl, col in [('age and sex', 'rho_age_sex'),
                 ('+ full factor', 'rho_factor_full'),
                 ('+ leave-one-out factor', 'rho_factor_loo')]:
    log(f"  {lbl:26s} {res[col].abs().max():.3f}")

log("\n" + "=" * 78)
log('THE THREE ASSOCIATIONS OF §3.16 (T50 with the Elecsys biomarkers)')
log("=" * 78)
s = res[(res.kinetic == "T50") &
        (res.analyte.isin(["t-tau (Elecsys)", "p-tau181 (Elecsys)",
                           "Aβ42 (Elecsys)"]))]
log(s[['analyte', 'rho_age_sex', 'rho_factor_full', 'rho_factor_loo',
       'p_factor_loo']].round(4).to_string(index=False))

log("\n" + "=" * 78)
log('DIFFERENCE BETWEEN THE TWO ADJUSTMENTS')
log("=" * 78)
dif = (res.rho_factor_loo.abs() - res.rho_factor_full.abs())
log(f'  |ρ| leave-one-out − |ρ| full component: median {dif.median():+.4f}, range [{dif.min():+.4f}; '
    f'{dif.max():+.4f}]')
log("  If the cancellation were an artefact of circularity, the")
log('  leave-one-out component should return appreciably larger associations.')

with open(os.path.join(OUT, "36_loo_component.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/36_loo_component.csv')
