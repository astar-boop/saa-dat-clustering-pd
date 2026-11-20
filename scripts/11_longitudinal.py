"""
11 - The SAA kinetic parameters as continuous predictors of the trajectories.

Primary cohort: 150h protocol (n = 473). It is the right choice because 96%
of the subjects belong to phase 1 of PPMI: median follow-up 7 years, 131
subjects beyond 10 years. In the 24h, 80% are PPMI 2.0 with a median follow-up
of 2 years, so it enters as a secondary replication, under-powered on the slopes.

Model: linear mixed model
    outcome ~ z(kinetic) * years + age_BL + sex [+ education] [+ duration_BL]
    random effects: intercept + slope per subject
The coefficient of interest is the kinetic x years interaction: the effect on
the RATE of progression.

Methodological constraint (from CORRECTION_AUDIT_2026-07-18): the baseline value
of the outcome does NOT enter as a covariate in the all-visits models, because
it would be at once the response at t = 0 and a predictor. The ANCOVA
conditional on baseline is reported separately, restricted to years > 0.

Output: outputs/11_*.csv, outputs/11_longitudinal.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

OUTCOMES = {
    "updrs3_score": ('MDS-UPDRS III (motor)', ["age_bl", "SEX", "dur_bl"]),
    "updrs_totscore": ('MDS-UPDRS total', ["age_bl", "SEX", "dur_bl"]),
    "updrs2_score": ("MDS-UPDRS II (ADL)", ["age_bl", "SEX", "dur_bl"]),
    "moca": ('MoCA (cognitive)', ["age_bl", "SEX", "dur_bl", "EDUCYRS"]),
    "MIA_STRIATUM_BILAT": ('DAT bilateral striatum', ["age_bl", "SEX", "dur_bl"]),
    "MIA_PUTAMEN_BILAT": ('DAT bilateral putamen', ["age_bl", "SEX", "dur_bl"]),
    "scopa": ('SCOPA-AUT (dysautonomia)', ["age_bl", "SEX", "dur_bl"]),
}


def build(cohort_file, kin_map, tag):
    d = pd.read_csv(os.path.join(OUT, cohort_file))
    if "complete" in d.columns:
        d = d[d.complete]
    elif "complete_primary" in d.columns:
        d = d[d.complete_primary]
    base = d[["PATNO"] + list(kin_map)].rename(columns=kin_map)
    # Log and standardisation at SUBJECT level, before the merge with the
    # repeated visits. Standardising after the merge would weight each patient
    # by their own number of visits, making the scale of the coefficient
    # dependent on the follow-up design.
    for k in kin_map.values():
        assert (base[k].dropna() > 0).all(), f'{k}: non-positive values'
        base[k] = np.log(base[k])
        base[k] = (base[k] - base[k].mean()) / base[k].std()
    bl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                    "duration_yrs"]].rename(
        columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
    base = base.merge(bl, on="PATNO", how="left")
    L = cur[cur.PATNO.isin(base.PATNO)][
        ["PATNO", "EVENT_ID", "YEAR", "LEDD"] + list(OUTCOMES)].copy()
    L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
    L = L[L.years.notna()]
    L = L.merge(base, on="PATNO", how="left")
    # the kinetic parameters stay RAW here: the log transformation and the
    # standardisation happen once only, in the caller.
    log(f'\n[{tag}] subjects {L.PATNO.nunique()}, rows {len(L)}, years 0-{L.years.max():.0f}')
    return L


KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}


def fit_all(L, kins, tag):
    rows = []
    for out, (label, cov) in OUTCOMES.items():
        for k in kins:
            cols = ["PATNO", "years", out, k] + cov
            m = L[cols].dropna()
            nsub = m.PATNO.nunique()
            # subjects with at least two observations are needed to estimate slopes
            multi = m.groupby("PATNO").size()
            n_multi = int((multi >= 2).sum())
            if len(m) < 120 or n_multi < 40:
                continue
            f = f"{out} ~ {k}*years + " + " + ".join(cov)
            try:
                md = smf.mixedlm(f, m, groups=m.PATNO, re_formula="~years")
                r = md.fit(method="lbfgs", maxiter=200)
                conv = r.converged
            except Exception:
                try:
                    md = smf.mixedlm(f, m, groups=m.PATNO)
                    r = md.fit(method="lbfgs", maxiter=200)
                    conv = r.converged
                except Exception:
                    continue
            term = f"{k}:years"
            if term not in r.params.index:
                term = f"years:{k}"
            if term not in r.params.index:
                continue
            rows.append({
                'cohort': tag, 'outcome': out, 'label': label, 'kinetic': k,
                'n_obsv': len(m), 'n_subj': nsub, 'n_subj_multivisit': n_multi,
                'beta_interaction': r.params[term], "se": r.bse[term],
                "p": r.pvalues[term],
                'beta_years': r.params.get("years", np.nan),
                'convergence': conv})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_BH"] = multipletests(df.p, method="fdr_bh")[1]
    return df


log("=" * 74)
log('MIXED MODELS — baseline SAA kinetic parameters as predictors of progression')
log("=" * 74)

# ---- 150h cohort (primary) ----
L150 = build("08_dataset_150h.csv", KIN150, "150h")
res150 = fit_all(L150, list(KIN150.values()), "150h")

# ---- 24h cohort (secondary replication) ----
L24 = build("analysis_complete_cases.csv", KIN24, "24h")
res24 = fit_all(L24, list(KIN24.values()), "24h")

res = pd.concat([res150, res24], ignore_index=True)
res.to_csv(os.path.join(OUT, "11_mixed_models.csv"), index=False)

log("\n" + "=" * 74)
log('PRIMARY COHORT 150h — kinetic parameter x years interaction')
log('(beta = change in the annual slope per 1 SD of the kinetic parameter)')
log("=" * 74)
for out in OUTCOMES:
    s = res150[res150.outcome == out]
    if not len(s):
        continue
    log(f'\n{OUTCOMES[out][0]}  (n = {s.n_subj.iloc[0]} subj, {s.n_obsv.iloc[0]} observations, mean '
        f'slope {s.beta_years.iloc[0]:+.3f}/year)')
    log(s[['kinetic', 'beta_interaction', "se", "p", "p_BH", 'convergence']]
        .round(4).to_string(index=False))

nsig = int((res150.p_BH < 0.05).sum())
log(f'\n>>> 150h: {nsig} significant interactions out of {len(res150)} after BH-FDR')
if nsig:
    log(res150[res150.p_BH < 0.05][
        ['outcome', 'kinetic', 'beta_interaction', "se", "p", "p_BH"]]
        .round(4).to_string(index=False))

log("\n" + "=" * 74)
log('SECONDARY COHORT 24h (short follow-up, underpowered on the slopes)')
log("=" * 74)
nsig24 = int((res24.p_BH < 0.05).sum())
log(f'{nsig24} significant interactions out of {len(res24)} after BH-FDR')
if nsig24:
    log(res24[res24.p_BH < 0.05][
        ['outcome', 'kinetic', 'beta_interaction', "se", "p", "p_BH"]]
        .round(4).to_string(index=False))
else:
    log('(none)')

# ------------------------------------------------------------------
# Sensitivity 1: adjustment for LEDD (UPDRS outcomes)
# ------------------------------------------------------------------
log("\n" + "=" * 74)
log("SENSITIVITY 1 — UPDRS adjusted for time-varying LEDD")
log("=" * 74)
rows = []
for out in ["updrs3_score", "updrs_totscore", "updrs2_score"]:
    for k in KIN150.values():
        m = L150[["PATNO", "years", out, k, "age_bl", "SEX", "dur_bl", "LEDD"]].dropna()
        if len(m) < 120:
            continue
        try:
            r = smf.mixedlm(f"{out} ~ {k}*years + age_bl + SEX + dur_bl + LEDD",
                            m, groups=m.PATNO, re_formula="~years").fit(
                method="lbfgs", maxiter=200)
        except Exception:
            continue
        t = f"{k}:years" if f"{k}:years" in r.params.index else f"years:{k}"
        rows.append({'outcome': out, 'kinetic': k, 'n_obsv': len(m),
                     "beta": r.params[t], "p": r.pvalues[t]})
sled = pd.DataFrame(rows)
if len(sled):
    sled["p_BH"] = multipletests(sled.p, method="fdr_bh")[1]
    log(sled.round(4).to_string(index=False))
    log(f'  significant after BH: {int((sled.p_BH < 0.05).sum())} / {len(sled)}')
    sled.to_csv(os.path.join(OUT, "11_ledd_adjusted.csv"), index=False)

# ------------------------------------------------------------------
# Sensitivity 2: two-stage approach (individual slopes)
# ------------------------------------------------------------------
log("\n" + "=" * 74)
log("SENSITIVITY 2 — individual slopes estimated per subject, then correlated")
log("=" * 74)
log('Only subjects with >= 3 observations. Spearman between kinetic parameter and OLS slope.')
rows = []
for out in OUTCOMES:
    sub = L150[["PATNO", "years", out]].dropna()
    cnt = sub.groupby("PATNO").size()
    keep = cnt[cnt >= 3].index
    sub = sub[sub.PATNO.isin(keep)]
    if sub.PATNO.nunique() < 60:
        continue
    slopes = (sub.groupby("PATNO")
              .apply(lambda g: np.polyfit(g.years, g[out], 1)[0])
              .rename("slope").reset_index())
    kk = L150.groupby("PATNO")[list(KIN150.values())].first().reset_index()
    mm = slopes.merge(kk, on="PATNO")
    for k in KIN150.values():
        rho, p = stats.spearmanr(mm[k], mm.slope)
        rows.append({'outcome': out, 'kinetic': k, 'n_subj': len(mm),
                     "rho": rho, "p": p})
two = pd.DataFrame(rows)
two["p_BH"] = multipletests(two.p, method="fdr_bh")[1]
log(two.round(4).to_string(index=False))
log(f'\n  significant after BH: {int((two.p_BH < 0.05).sum())} / {len(two)}; maximum |rho| '
    f'{two.rho.abs().max():.3f}')
two.to_csv(os.path.join(OUT, "11_two_stage.csv"), index=False)

# ------------------------------------------------------------------
# Sensitivity 3: ANCOVA conditional on baseline, years > 0
# ------------------------------------------------------------------
log("\n" + "=" * 74)
log("SENSITIVITY 3 — ANCOVA conditional on baseline, restricted to years > 0")
log("=" * 74)
rows = []
for out in ["updrs3_score", "moca", "MIA_STRIATUM_BILAT"]:
    b0 = (L150[L150.years == 0][["PATNO", out]]
          .rename(columns={out: "base_out"}).dropna())
    post = L150[L150.years > 0].merge(b0, on="PATNO")
    for k in KIN150.values():
        m = post[["PATNO", "years", out, k, "age_bl", "SEX", "base_out"]].dropna()
        if len(m) < 120:
            continue
        try:
            r = smf.mixedlm(f"{out} ~ {k}*years + age_bl + SEX + base_out",
                            m, groups=m.PATNO, re_formula="~years").fit(
                method="lbfgs", maxiter=200)
        except Exception:
            continue
        t = f"{k}:years" if f"{k}:years" in r.params.index else f"years:{k}"
        rows.append({'outcome': out, 'kinetic': k, 'n_obsv': len(m),
                     'timeval_min': m.years.min(), "beta": r.params[t],
                     "p": r.pvalues[t]})
anc = pd.DataFrame(rows)
if len(anc):
    anc["p_BH"] = multipletests(anc.p, method="fdr_bh")[1]
    log(anc.round(4).to_string(index=False))
    log(f'  minimum time in the ANCOVA models: {anc.timeval_min.min():.2f} years (no row at t = 0, '
        f'as required)')
    log(f'  significant after BH: {int((anc.p_BH < 0.05).sum())} / {len(anc)}')
    anc.to_csv(os.path.join(OUT, "11_ancova.csv"), index=False)

with open(os.path.join(OUT, "11_longitudinal.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/11_longitudinal.txt')
