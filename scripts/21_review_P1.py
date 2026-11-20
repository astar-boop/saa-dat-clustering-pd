"""
21 - Response to the blocking remarks of the review: P1.5, P1.8, P1.10.

P1.5  Genetic confounder. Carrier status does not appear in the manuscript,
      although reference [7] reports faster kinetics in GBA1-PD and the
      GBA1 carrier status is among the best-established predictors of accelerated
      cognitive decline, with a profile that involves precisely processing speed.
P1.8  Measurement uncertainty of the kinetic parameters: within-triplet CV, ICC(1,1)
      and ICC(1,3), repeatability coefficient, disattenuated coefficients.
P1.10 Reconciliation of the sample accounting.

Output: outputs/21_*.csv, outputs/21_review_P1.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from _paths import DATA

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"]
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete].reset_index(drop=True)
d24 = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))

# ===========================================================================
# P1.10 — sample accounting
# ===========================================================================
log("=" * 78)
log('P1.10 — RECONCILIATION OF THE SAMPLE ACCOUNTING')
log("=" * 78)
sa = pd.read_csv(os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"))

p24 = sa[(sa.COHORT == "PD") & (sa.PROJECTID == 237)]
p24bl = p24[p24.CLINICAL_EVENT == "BL"]
log('\n24 h protocol (Project 237), PD cohort')
log(f'  samples at any visit                     : {len(p24)}  ({p24.PATNO.nunique()} subjects)')
log(f'  samples at the baseline visit            : {len(p24bl)}  ({p24bl.PATNO.nunique()} subjects)')
for st in ["Positive", "Negative", "Inconclusive"]:
    s = p24bl[p24bl.SAA_Status == st]
    log(f'    of which {st:13s}: {len(s):4d} samples, {s.PATNO.nunique():4d} subjects')
t1 = p24bl[(p24bl.SAA_Status == "Positive") & (p24bl.SAA_Type == "Type1")]
log(f'  type 1 positives at baseline             : {len(t1)} samples, {t1.PATNO.nunique()} subjects')
log(f'    (the {len(t1) - t1.PATNO.nunique()} excess samples are repeat collections from the same '
    f'subject, averaged)')
log(f'  in the curated data cut at baseline      : {bl.PATNO.isin(t1.PATNO).sum()}')
log(f'  with complete data on the 7 variables    : {len(d24)}')
log('\n  NOTE on the 244 negative samples cited in §2.2: they refer to ALL the')
log('  visits of the 24 h protocol, not to baseline alone. At baseline the negatives')
log(f"  are {p24bl[p24bl.SAA_Status == 'Negative'].PATNO.nunique()}, consistent with the 107 of the "
    f"extended population after")
log("  the merge with DAT and age.")

# enlarged population
ext_ids = p24bl.PATNO.unique()
fb = bl[bl.PATNO.isin(ext_ids)][["PATNO", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
                                 "MIA_CAUDATE_BILAT", "age_at_visit"]].dropna()
ext = p24bl[p24bl.PATNO.isin(fb.PATNO)].drop_duplicates("PATNO")
log(f'\nExtended population (all assay results, with DAT and age): {len(ext)} subjects')
log(f"  {ext.SAA_Status.value_counts().to_dict()}")
pos_ext = ext[(ext.SAA_Status == "Positive")]
log(f'  positives in the extended population     : {len(pos_ext)}')
log(f'  type 1 positives with complete data      : {len(d24)}')
log(f"  difference {len(pos_ext) - len(d24)}: positives NOT of type 1 "
    f"({(pos_ext.SAA_Type != 'Type1').sum()} between Type2 and indeterminate)")

p150 = sa[(sa.PROJECTID == 155) & (sa.COHORT == "PD") & (sa.CLINICAL_EVENT == "BL")]
log(f'\n150 h protocol (Project 155), PD at baseline: {len(p150)} samples, {p150.PATNO.nunique()} '
    f'subjects')
log(f"  {p150.SAA_Status.value_counts().to_dict()}")
log(f'  positives with complete data             : {len(d150)}')

flow = pd.DataFrame([
    {'step': 'PD with 24h assay (all visits)', 'samples': len(p24), 'subjects': p24.PATNO.nunique()},
    {'step': 'PD with 24h assay at baseline', 'samples': len(p24bl), 'subjects': p24bl.PATNO.nunique()},
    {'step': 'type 1 positive at baseline', 'samples': len(t1), 'subjects': t1.PATNO.nunique()},
    {'step': 'present in the curated data cut at baseline', 'samples': np.nan, 'subjects': int(bl.PATNO.isin(t1.PATNO).sum())},
    {'step': 'with complete DAT, age and kinetic parameters', 'samples': np.nan, 'subjects': len(d24)},
    {'step': 'enlarged population (positive control)', 'samples': np.nan, 'subjects': len(ext)},
    {'step': 'PD with 150h assay at baseline', 'samples': len(p150), 'subjects': p150.PATNO.nunique()},
    {'step': '150h positives with complete data', 'samples': np.nan, 'subjects': len(d150)},
])
flow.to_csv(os.path.join(OUT, "21_flow_samples.csv"), index=False)

# ===========================================================================
# P1.8 — reliability of the kinetic parameters from the technical triplicates
# ===========================================================================
log("\n" + "=" * 78)
log('P1.8 — MEASUREMENT UNCERTAINTY OF THE KINETIC PARAMETERS')
log("=" * 78)
log('ICC(1,1) = reliability of a single replicate; ICC(1,3) = of the mean of')
log('three replicates (Spearman-Brown). Estimated by one-way ANOVA on subjects.')


def icc_from_reps(mat):
    """mat: n subjects x k replicates. Returns ICC(1,1), ICC(1,3), CV%, CR."""
    m = mat[~np.isnan(mat).any(axis=1)]
    n, k = m.shape
    gm = m.mean()
    msb = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msw = ((m - m.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    icc3 = (msb - msw) / msb
    cv = np.nanmedian(m.std(axis=1, ddof=1) / m.mean(axis=1)) * 100
    cv_iqr = np.nanpercentile(m.std(axis=1, ddof=1) / m.mean(axis=1), [25, 75]) * 100
    cr = 1.96 * np.sqrt(2 * msw)                      # repeatability coefficient
    return n, icc1, icc3, cv, cv_iqr, cr


rows = []
SPECS = [("24h", 237, ["Fmax", "TTT", "AUC", "TSmax", "SLOPEMax"], "_24h_Rep"),
         ("150h", 155, ["Fmax", "TTT", "AUC", "T50", "SLOPE"], "_150h_Rep")]
for tag, pid, params, suffix in SPECS:
    sel = sa[(sa.PROJECTID == pid) & (sa.COHORT == "PD") &
             (sa.CLINICAL_EVENT == "BL") & (sa.SAA_Status == "Positive")]
    # same population as the primary one (cf. 38_affidabilita_mediana.py)
    if os.environ.get('ONLY_SPORADIC', "1") == "1":
        _sp = set(bl.loc[bl.subgroup == "Sporadic PD", "PATNO"])
        n_pre = sel.PATNO.nunique()
        sel = sel[sel.PATNO.isin(_sp)]
        log(f"  [{pid}] reliability on sporadic only: {n_pre} -> {sel.PATNO.nunique()} subjects")
    for p in params:
        cols = [f"{p}{suffix}{i}" for i in (1, 2, 3)]
        if not all(c in sel.columns for c in cols):
            continue
        v = sel[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        v = np.log(np.where(v > 0, v, np.nan))        # analysis scale
        n, i1, i3, cv, cviqr, cr = icc_from_reps(v)
        rows.append({'protocol': tag, 'parameter': p, 'n_subjects': n,
                     "CV%_mediana": cv, "CV%_Q1": cviqr[0], "CV%_Q3": cviqr[1],
                     "ICC_1_1": i1, "ICC_1_3": i3, "rip_coef_log": cr})
icc = pd.DataFrame(rows)
log('\nCV computed on the raw values; ICC on the logarithmic scale of analysis.')
log(icc.round(3).to_string(index=False))
icc.to_csv(os.path.join(OUT, "21_reliability_kinetics.csv"), index=False)
log(f'\nWells characterised overall: {int(icc.n_subjects.sum() * 3)}')

# coefficients disattenuated for reliability
log("\nDisattenuation of the MoCA-T50 coefficient for the reliability of the measurement:")
icc3_t50 = float(icc[(icc.protocol == "150h") & (icc.parameter == "T50")].ICC_1_3.iloc[0])
# the beta must be read from the actual result, not hard-coded: it changes
# with the population under analysis.
_pn = os.path.join(OUT, "15_primary_nonmotor.csv")
beta = (float(pd.read_csv(_pn).set_index('outcome').loc["moca", "beta"])
        if os.path.exists(_pn) else 0.0761)
log(f'  ICC(1,3) of T50            : {icc3_t50:.3f}')
log(f'  observed beta              : {beta:.4f}')
log(f"  disattenuated beta         : {beta / np.sqrt(icc3_t50):.4f}  (beta / sqrt(ICC): measurement "
    f"error attenuates towards zero)")

# ===========================================================================
# P1.5 — genetic confounder
# ===========================================================================
log("\n" + "=" * 78)
log('P1.5 — GENETIC CONFOUNDER (GBA1, LRRK2)')
log("=" * 78)

for tag, dd in [("150h", d150), ("24h", d24)]:
    sg = bl[bl.PATNO.isin(dd.PATNO)][["PATNO", "subgroup"]]
    log(f'\nGenetic composition, {tag} cohort (n = {len(dd)}):')
    vc = sg.subgroup.value_counts(dropna=False)
    for k, v in vc.items():
        log(f"  {str(k):18s} {v:4d}  ({100*v/len(sg):.1f}%)")

sg = bl[["PATNO", "subgroup"]].copy()
sg["gba"] = sg.subgroup.fillna("").str.contains("GBA").astype(int)
sg["lrrk2"] = sg.subgroup.fillna("").str.contains("LRRK2").astype(int)
sg['sporadic'] = (sg.subgroup == "Sporadic PD").astype(int)

ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
if ONLY:
    log('\nONLY_SPORADIC=1 — the primary cohort contains no carriers: the genetic confounder is '
        'eliminated BY DESIGN, not by adjustment.')
    log("The sections that cross-tabulate the genotypes are therefore structurally empty and are "
        "omitted. Only the base model remains, which in this version coincides with the «sporadic "
        "patients only» specification of the full-cohort version.")
    log("Warning: elimination by design is not free. It moves the question from confounding to "
        "generalisability (the result no longer speaks of the carriers) and costs power. See script "
        "61.")

# --- do the kinetic parameters differ by genotype? ---
if not ONLY:
    log('\nBaseline kinetic parameters by genotype (150h cohort):')
KIN = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
       "k_SLOPE": "SLOPE"}
g150 = d150.merge(sg, on="PATNO", how="left")
rows = []
for c, name in ({} if ONLY else KIN).items():
    a = np.log(g150.loc[g150.gba == 1, c].dropna())
    b = np.log(g150.loc[g150.sporadic == 1, c].dropna())
    l = np.log(g150.loc[g150.lrrk2 == 1, c].dropna())
    u1, p1 = stats.mannwhitneyu(a, b)
    u2, p2 = stats.mannwhitneyu(l, b)
    rows.append({'parameter': name, "n_GBA": len(a), 'n_sporadic': len(b),
                 "n_LRRK2": len(l),
                 'median_GBA': np.exp(a.median()), 'median_sporadic': np.exp(b.median()),
                 'p_GBA_vs_sporadic': p1,
                 'median_LRRK2': np.exp(l.median()), 'p_LRRK2_vs_sporadic': p2})
kg = pd.DataFrame(rows)
if not kg.empty:
    kg["pBH_GBA"] = multipletests(kg.p_GBA_vs_sporadic, method="fdr_bh")[1]
    kg["pBH_LRRK2"] = multipletests(kg.p_LRRK2_vs_sporadic, method="fdr_bh")[1]
    log(kg.round(4).to_string(index=False))
    kg.to_csv(os.path.join(OUT, "21_kinetics_by_genotype.csv"), index=False)

# --- does the MoCA signal survive the genetic adjustment? ---
log("\n" + "-" * 78)
log('Does the MoCA signal survive genotype?')
log("-" * 78)
base = d150[["PATNO"]].copy()
base["T50"] = np.log(d150.k_T50)
base["T50"] = (base.T50 - base.T50.mean()) / base.T50.std()
covbl = bl[["PATNO", "age_at_visit", "SEX", "EDUCYRS", "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base = base.merge(covbl, on="PATNO", how="left").merge(
    sg[["PATNO", "gba", "lrrk2", 'sporadic']], on="PATNO", how="left")
L = cur[cur.PATNO.isin(base.PATNO)][["PATNO", "YEAR", "moca", "SDMTOTAL", "rem"]].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")


def fit(f, data):
    return smf.mixedlm(f, data, groups=data.PATNO,
                       re_formula="~years").fit(method="lbfgs", maxiter=300)


def term(r, k="T50"):
    for t in (f"{k}:years", f"years:{k}"):
        if t in r.params.index:
            return t


COV = "age_bl + SEX + EDUCYRS + dur_bl"
rows = []
for outcome in ["moca", "SDMTOTAL", "rem"]:
    cov = COV if outcome != "rem" else "age_bl + SEX + dur_bl"
    specs = {'model of the manuscript': f'{outcome} ~ T50*years + {cov}'}
    if not ONLY:
        # with sporadic only, gba and lrrk2 are constant at zero: the two
        # adjusted specifications are singular, not conservative.
        specs.update({
            '+ GBA and LRRK2 as covariates':
                f'{outcome} ~ T50*years + gba + lrrk2 + {cov}',
            '+ GBA and LRRK2 in interaction with time':
                f'{outcome} ~ T50*years + gba*years + lrrk2*years + {cov}',
        })
    for nm, f in specs.items():
        need = ["PATNO", "years", outcome, "T50", "gba", "lrrk2",
                "age_bl", "SEX", "dur_bl"] + (["EDUCYRS"] if outcome != "rem" else [])
        m = L[need].dropna()
        r = fit(f, m)
        t = term(r)
        rows.append({'outcome': outcome, 'specification': nm,
                     'n_subj': m.PATNO.nunique(), "beta": r.params[t],
                     "se": r.bse[t], "p": r.pvalues[t]})
    # sporadic only — under ONLY_SPORADIC it coincides with the base model
    if ONLY:
        continue
    need = ["PATNO", "years", outcome, "T50", 'sporadic', "age_bl", "SEX",
            "dur_bl"] + (["EDUCYRS"] if outcome != "rem" else [])
    m = L[need].dropna()
    m = m[m.sporadic == 1]
    r = fit(f'{outcome} ~ T50*years + {cov}', m)
    t = term(r)
    rows.append({'outcome': outcome, 'specification': 'sporadic patients only',
                 'n_subj': m.PATNO.nunique(), "beta": r.params[t],
                 "se": r.bse[t], "p": r.pvalues[t]})
gen = pd.DataFrame(rows)
gen['CI95_inf'] = gen.beta - 1.96 * gen.se
gen['CI95_upp'] = gen.beta + 1.96 * gen.se
log(gen.round(4).to_string(index=False))
gen.to_csv(os.path.join(OUT, "21_moca_genotype.csv"), index=False)

with open(os.path.join(OUT, "21_review_P1.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/21_review_P1.txt')
