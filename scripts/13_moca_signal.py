"""
13 - The signal on MoCA: characterisation and check for confounding.

In the longitudinal models (script 11) the only consistent association is
between the TEMPORAL parameters of the assay (TTT, T50, AUC) and the rate of
cognitive decline. Direction: faster kinetics -> faster decline.

Here the following are checked:
  a) the effect in interpretable units (MoCA points at 5 and 10 years);
  b) whether it survives adjustment for the CSF CONCENTRATION FACTOR
     (script 12), given that T50 correlates with it at rho = 0.22;
  c) whether it is explained by age at onset, education or disease duration;
  d) specificity: the same model on the other outcomes.

Output: outputs/13_*.csv, outputs/13_moca_signal.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from _fit import fit_mixed
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
W = os.path.join(BASE, "outputs", "cache") + "/"
rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


cur = pd.read_parquet(W + "curated.parquet")
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete].reset_index(drop=True)

KIN = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
       "k_SLOPE": "SLOPE"}
base = d[["PATNO"] + list(KIN)].rename(columns=KIN)
for c in KIN.values():
    base[c] = np.log(base[c])
    base[c] = (base[c] - base[c].mean()) / base[c].std()

# CSF concentration factor (as in script 12)
ANALYTES = ["abeta", "tau", "ptau", "asyn", "IU_pTau181_CSF",
            "IU_ABeta40_CSF", "IU_ABeta42_CSF"]
bl = cur[cur.EVENT_ID == "BL"]
cf = bl[["PATNO"] + ANALYTES].copy()
cf = cf[cf.PATNO.isin(base.PATNO)]
m = cf[ANALYTES].dropna()
R = np.column_stack([stats.rankdata(m[c]) for c in ANALYTES])
R = (R - R.mean(0)) / R.std(0)
f = PCA(n_components=1).fit_transform(R).ravel()
if stats.spearmanr(f, m[ANALYTES[0]]).statistic < 0:
    f = -f
cf["conc"] = np.nan
cf.loc[m.index, "conc"] = (f - f.mean()) / f.std()
base = base.merge(cf[["PATNO", "conc"]], on="PATNO", how="left")
log(f'Subjects: {len(base)} | with concentration factor: {base.conc.notna().sum()}')

covbl = bl[["PATNO", "age_at_visit", "SEX", "EDUCYRS", "duration_yrs",
            "ageonset"]].rename(columns={"age_at_visit": "age_bl",
                                         "duration_yrs": "dur_bl"})
base = base.merge(covbl, on="PATNO", how="left")

OUTS = ["moca", "updrs3_score", "updrs_totscore", "updrs2_score",
        "MIA_STRIATUM_BILAT", "scopa"]
L = cur[cur.PATNO.isin(base.PATNO)][["PATNO", "YEAR"] + OUTS].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")


def fit(formula, data):
    r, _method, _ = fit_mixed(formula, data, data.PATNO)
    return r


def term(r, k):
    for t in (f"{k}:years", f"years:{k}"):
        if t in r.params.index:
            return t
    return None


# ---------------------------------------------------------------------------
# a) effect in interpretable units
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("a) EFFECT SIZE ON MoCA")
log("=" * 74)
mm = L[["PATNO", "years", "moca", "T50", "age_bl", "SEX", "EDUCYRS", "dur_bl"]].dropna()
r = fit("moca ~ T50*years + age_bl + SEX + EDUCYRS + dur_bl", mm)
t = term(r, "T50")
b, se, p = r.params[t], r.bse[t], r.pvalues[t]
slope = r.params["years"]
log(f'n = {mm.PATNO.nunique()} subjects, {len(mm)} observations')
log(f'mean MoCA slope               : {slope:+.3f} points/year')
log(f'T50 x years interaction       : {b:+.4f} (SE {se:.4f}, p = {p:.4f})')
log(f"IC 95%                        : [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}]")
log("\nExpected decline by level of T50 (1 SD = slower kinetics):")
for sd, lab in [(-1, 'T50 -1 SD (faster seeding)'), (0, 'mean T50'),
                (1, 'T50 +1 SD (slower seeding)')]:
    s = slope + b * sd
    log(f'  {lab:34s} {s:+.3f} points/year  |  5 years {5 * s:+.2f}  |  10 years {10 * s:+.2f}')
log(f'\nDifference between extremes (-1 SD vs +1 SD) at 10 years: {10 * (-2 * b):+.2f} MoCA points')

# ---------------------------------------------------------------------------
# b) confounding with the concentration factor
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log('b) DOES THE SIGNAL SURVIVE THE CSF CONCENTRATION FACTOR?')
log("=" * 74)
log('The factor enters both as a main effect and in interaction with time.')
rows = []
for k in ["TTT", "T50", "AUC", "Fmax", "SLOPE"]:
    m1 = L[["PATNO", "years", "moca", k, "age_bl", "SEX", "EDUCYRS", "dur_bl"]].dropna()
    r1 = fit(f"moca ~ {k}*years + age_bl + SEX + EDUCYRS + dur_bl", m1)
    t1 = term(r1, k)
    m2 = L[["PATNO", "years", "moca", k, "conc", "age_bl", "SEX",
            "EDUCYRS", "dur_bl"]].dropna()
    r2 = fit(f"moca ~ {k}*years + conc*years + age_bl + SEX + EDUCYRS + dur_bl", m2)
    t2 = term(r2, k)
    tc = term(r2, "conc")
    rows.append({'kinetic': k,
                 'n_subj_base': m1.PATNO.nunique(), "beta_base": r1.params[t1],
                 "p_base": r1.pvalues[t1],
                 'n_subj_agg': m2.PATNO.nunique(), "beta_agg": r2.params[t2],
                 "p_agg": r2.pvalues[t2],
                 'beta_conc_x_years': r2.params[tc], "p_conc": r2.pvalues[tc]})
cb = pd.DataFrame(rows)
cb["pBH_base"] = multipletests(cb.p_base, method="fdr_bh")[1]
cb["pBH_agg"] = multipletests(cb.p_agg, method="fdr_bh")[1]
log(cb.round(4).to_string(index=False))
cb.to_csv(os.path.join(OUT, "13_moca_conc_adjusted.csv"), index=False)
log("\n  Note: the subset with the concentration factor is smaller,")
log('  so the comparison beta_base vs beta_agg must be read on the coefficients,')
log('  not on the p-values alone.')
# same subset, without the factor: fair comparison
log('\nFair comparison, same subset (with and without the factor):')
rows = []
for k in ["TTT", "T50", "AUC"]:
    m2 = L[["PATNO", "years", "moca", k, "conc", "age_bl", "SEX",
            "EDUCYRS", "dur_bl"]].dropna()
    ra = fit(f"moca ~ {k}*years + age_bl + SEX + EDUCYRS + dur_bl", m2)
    rb = fit(f"moca ~ {k}*years + conc*years + age_bl + SEX + EDUCYRS + dur_bl", m2)
    rows.append({'kinetic': k, 'n_subj': m2.PATNO.nunique(),
                 'beta_without_factor': ra.params[term(ra, k)],
                 'p_without': ra.pvalues[term(ra, k)],
                 'beta_having_factor': rb.params[term(rb, k)],
                 'p_having': rb.pvalues[term(rb, k)]})
eq = pd.DataFrame(rows)
log(eq.round(4).to_string(index=False))
eq.to_csv(os.path.join(OUT, "13_moca_same_subset.csv"), index=False)

# ---------------------------------------------------------------------------
# c) other confounders
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log('c) OTHER POSSIBLE CONFOUNDERS')
log("=" * 74)
specs = {
    'base model': "moca ~ T50*years + age_bl + SEX + EDUCYRS + dur_bl",
    '+ age at onset x years': "moca ~ T50*years + ageonset*years + age_bl + SEX + EDUCYRS + dur_bl",
    '+ baseline age x years': "moca ~ T50*years + age_bl*years + SEX + EDUCYRS + dur_bl",
    '+ education x years': "moca ~ T50*years + EDUCYRS*years + age_bl + SEX + dur_bl",
    '+ disease duration x years': "moca ~ T50*years + dur_bl*years + age_bl + SEX + EDUCYRS",
    '+ sex x years': "moca ~ T50*years + SEX*years + age_bl + EDUCYRS + dur_bl",
}
rows = []
for name, f in specs.items():
    need = ["PATNO", "years", "moca", "T50", "age_bl", "SEX", "EDUCYRS", "dur_bl"]
    if "ageonset" in f:
        need.append("ageonset")
    mm2 = L[need].dropna()
    try:
        rr = fit(f, mm2)
        t = term(rr, "T50")
        rows.append({'specification': name, 'n_subj': mm2.PATNO.nunique(),
                     'beta_T50_x_years': rr.params[t], "se": rr.bse[t],
                     "p": rr.pvalues[t]})
    except Exception as e:
        rows.append({'specification': name, 'n_subj': mm2.PATNO.nunique(),
                     'beta_T50_x_years': np.nan, "se": np.nan, "p": np.nan})
cf2 = pd.DataFrame(rows)
log(cf2.round(4).to_string(index=False))
cf2.to_csv(os.path.join(OUT, "13_moca_confounders.csv"), index=False)

# ---------------------------------------------------------------------------
# d) specificity with respect to the other outcomes
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("d) SPECIFICITY: same specification on the other outcomes")
log("=" * 74)
rows = []
for out in OUTS:
    mm3 = L[["PATNO", "years", out, "T50", "age_bl", "SEX", "EDUCYRS", "dur_bl"]].dropna()
    if len(mm3) < 200:
        continue
    rr = fit(f"{out} ~ T50*years + age_bl + SEX + EDUCYRS + dur_bl", mm3)
    t = term(rr, "T50")
    sl = rr.params["years"]
    rows.append({'outcome': out, 'n_subj': mm3.PATNO.nunique(), 'n_obsv': len(mm3),
                 'slope_mean': sl, 'beta_T50_x_years': rr.params[t],
                 "beta_relativo_%": 100 * rr.params[t] / abs(sl) if sl else np.nan,
                 "p": rr.pvalues[t]})
sp = pd.DataFrame(rows)
sp["p_BH"] = multipletests(sp.p, method="fdr_bh")[1]
log(sp.round(4).to_string(index=False))
sp.to_csv(os.path.join(OUT, "13_specificity.csv"), index=False)

with open(os.path.join(OUT, "13_moca_signal.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/13_moca_signal.txt')
