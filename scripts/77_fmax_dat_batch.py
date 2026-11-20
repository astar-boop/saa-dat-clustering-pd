"""
77 - The Fmax x years result on dopaminergic binding, re-examined for the assay
     run, the instrument and the centre.

The only result that survives the correction for multiple comparisons in the
24 h cohort is the interaction between baseline Fmax and time on dopaminergic
binding (bilateral striatum and bilateral putamen, outputs/72_mixed_models.csv).
Fmax is, however, the parameter with the largest technical component: the share
of variance attributable to the assay run is eta^2 = 0.201 (variance component
0.158) in this cohort (outputs/69_run_variance_components.csv). If the runs were
distributed unevenly across the centres and the centres differed in the
reconstruction of the DAT-SPECT images, the association could be technical.

The two models are refitted along a specification ladder:

  M0  published model: outcome ~ z(Fmax)*years + age_bl + SEX + dur_bl
  M1  M0 + assay run (RUNDATE) as a fixed effect
  M2  M0 + instrument (InstrumentRep1) as a fixed effect
  M3  M0 + centre (SITE of the curated data cut at BL) as a fixed effect
  M4  M0 + run + centre
  M5  M4 + the interactions with time: run x years and centre x years
  M6  M0 on log Fmax residualised for run and restandardised

All the estimates go through the optimiser cascade of _fit.py, so that
convergence is guaranteed and recorded.

Added: (a) the direct test of the confounding path, that is, whether the
run is associated with the DAT outcome and with its slope; (b) the run x centre
contingency table with Cramér's V, restricted to the 595, to make
the degree of aliasing explicit.

Output: outputs/77_fmax_dat.csv, outputs/77_fmax_dat.txt,
        outputs/77_run_x_centre.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import fit_mixed, OPTIMISERS
from _paths import DATA
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")


def fit_ml(formula, data, groups, re_formula="~years"):
    """Among the optimisers that converge, the one with the maximum likelihood.

    `fit_mixed` returns the FIRST optimiser that converges. On these data that
    is not enough: in the striatum model `lbfgs` declares convergence at a
    point of lower likelihood than that reached by `powell` and `nm`
    (1010.365 against 1015.342 and 1015.345), and the estimate of the
    interaction is affected. Here all of them are tried and the maximum is
    kept, as a check on the specification ladder.
    """
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, it in OPTIMISERS:
            try:
                r = smf.mixedlm(formula, data, groups=groups,
                                re_formula=re_formula).fit(method=method,
                                                           maxiter=it)
            except Exception:
                continue
            if not r.converged:
                continue
            if best is None or r.llf > best[0].llf:
                best = (r, method)
    return best if best else (None, None)
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA_FILE = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
N_MIN = 5           # merging threshold for runs and centres
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# ---------------------------------------------------------------------------
# construction of the dataset, identical to that of 72_longitudinal_refit.py
# ---------------------------------------------------------------------------
# The caudate enters here too: with only two compartments, one of which contains
# the other (rho = 0.94), the adjustment could say nothing about regional
# specificity. The caudate is the relatively spared compartment in PD,
# so it is the benchmark that makes the gradient observable.
OUTCOMES = {"MIA_STRIATUM_BILAT": 'DAT striatum, bilateral',
            "MIA_PUTAMEN_BILAT": 'DAT putamen, bilateral',
            "MIA_CAUDATE_BILAT": 'DAT caudate, bilateral'}
COV = ["age_bl", "SEX", "dur_bl"]

cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
cc = cc[cc.complete_primary]

# z(Fmax) as in script 72: log, then standardisation on the cohort
base = cc[["PATNO", "saa_Fmax"]].rename(columns={"saa_Fmax": "Fmax"}).copy()
base["logFmax"] = np.log(base.Fmax)
base["Fmax"] = (base.logFmax - base.logFmax.mean()) / base.logFmax.std()

bl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base = base.merge(bl, on="PATNO", how="left")

# run and instrument from the raw SAA file (PROJECTID 237, CLINICAL_EVENT == "BL")
saa = pd.read_csv(SAA_FILE)
s = saa[(saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")
        & (saa.PATNO.isin(set(base.PATNO)))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
s = s[["PATNO", "RUNDATE", "InstrumentRep1"]].rename(
    columns={"RUNDATE": "run_raw", "InstrumentRep1": 'instr'})
base = base.merge(s, on="PATNO", how="left")

# centre from the curated data cut at BL
site_bl = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO")[["PATNO", "SITE"]]
base = base.merge(site_bl, on="PATNO", how="left")

log("=" * 78)
log('77 — Fmax x YEARS ON DOPAMINERGIC BINDING, RE-EXAMINATION BY RUN AND CENTRE')
log("=" * 78)
log(f'\n24 h cohort: {len(base)} subjects')
log(f'  with assay run:     {int(base.run_raw.notna().sum())}')
log(f'  with instrument:    {int(base.instr.notna().sum())}')
log(f'  with centre:        {int(base.SITE.notna().sum())}')

# merging of the rare categories, on the 595 subjects
vr = base.run_raw.astype(str).value_counts()
base["run"] = base.run_raw.astype(str).where(
    base.run_raw.astype(str).map(vr) >= N_MIN, 'other')
vs = base.SITE.astype(str).value_counts()
base["site"] = base.SITE.astype(str).where(
    base.SITE.astype(str).map(vs) >= N_MIN, 'other')
base['instr'] = base.instr.astype(str)

log(f"\ndistinct runs: {int(vr.size)}; with at least {N_MIN} samples: {int((vr >= N_MIN).sum())} "
    f"(covering {int(vr[vr >= N_MIN].sum())} subjects); pooled into 'altro': "
    f"{int((vr < N_MIN).sum())} runs, {int(vr[vr < N_MIN].sum())} subjects -> {base.run.nunique()} "
    f"categories")
log(f"distinct centres: {int(vs.size)}; with at least {N_MIN} subjects: {int((vs >= N_MIN).sum())} "
    f"(covering {int(vs[vs >= N_MIN].sum())} subjects); pooled into 'altro': "
    f"{int((vs < N_MIN).sum())} centres, {int(vs[vs < N_MIN].sum())} subjects -> "
    f"{base.site.nunique()} categories")
log(f'distinct instruments: {base.instr.nunique()} ({base.instr.value_counts().to_dict()})')

# Fmax residualised for run (M6): log Fmax is centred on the mean of its own
# run, brought back to the overall mean, then restandardised
med = base.groupby("run").logFmax.transform("mean")
resid = base.logFmax - med + base.logFmax.mean()
base["Fmax_res"] = (resid - resid.mean()) / resid.std()
log(f"\nshare of variance of log Fmax between runs: "
    f"{smf.ols('logFmax ~ C(run)', base).fit().rsquared:.4f} -> "
    f"{smf.ols('Fmax_res ~ C(run)', base).fit().rsquared:.4f} after residualisation")
log(f'correlation between z(Fmax) and residualised z(Fmax): {base.Fmax.corr(base.Fmax_res):.4f}')

# longitudinal dataset
L = cur[cur.PATNO.isin(base.PATNO)][
    ["PATNO", "EVENT_ID", "YEAR"] + list(OUTCOMES)].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()]
L = L.merge(base[["PATNO", "Fmax", "Fmax_res", "run", "site", 'instr']
                 + COV], on="PATNO", how="left")

# ---------------------------------------------------------------------------
# the specification ladder
# ---------------------------------------------------------------------------
SPEC = [
    ("M0", 'published: z(Fmax)*years + age + sex + duration', "Fmax", ""),
    ("M1", 'M0 + assay run', "Fmax", " + C(run)"),
    ("M2", 'M0 + instrument', "Fmax", ' + C(instr)'),
    ("M3", 'M0 + centre', "Fmax", " + C(site)"),
    ("M4", 'M0 + run + centre', "Fmax", " + C(run) + C(site)"),
    ("M5", 'M4 + run x years + centre x years', "Fmax",
     " + C(run)*years + C(site)*years"),
    ("M6", 'M0 on Fmax residualised for run', "Fmax_res", ""),
    # decomposition of M5, to attribute the attenuation to the run or to the centre
    ("M5a", 'M0 + run x years (run only)', "Fmax", " + C(run)*years"),
    ("M5b", 'M0 + centre x years (centre only)', "Fmax", " + C(site)*years"),
]

rows = []
for out, label in OUTCOMES.items():
    log("\n" + "=" * 78)
    log(f"{label}  ({out})")
    log("=" * 78)
    for abbrev, descr, kin, extra in SPEC:
        cols = ["PATNO", "years", out, kin] + COV
        for c in ("run", "site", 'instr'):
            if f"C({c})" in extra:
                cols.append(c)
        m = L[list(dict.fromkeys(cols))].dropna()
        f = f"{out} ~ {kin}*years + " + " + ".join(COV) + extra
        r, method, fellback = fit_mixed(f, m, m.PATNO)
        if r is None:
            log(f'  {abbrev}: no optimiser converges')
            rows.append({"outcome": out, "label": label, "spec": abbrev,
                          'description': descr, "formula": f,
                          "n_subjects": int(m.PATNO.nunique()),
                          "n_obs": int(len(m)), "beta": np.nan, "se": np.nan,
                          "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                          "converged": False, "optimiser": ""})
            continue
        t = (f"{kin}:years" if f"{kin}:years" in r.params.index
             else f"years:{kin}")
        b, se = float(r.params[t]), float(r.bse[t])
        rml, metml = fit_ml(f, m, m.PATNO)
        rows.append({"outcome": out, "label": label, "spec": abbrev,
                      'description': descr, "formula": f,
                      "n_subjects": int(m.PATNO.nunique()),
                      "n_obs": int(len(m)), 'n_params_fixed': int(len(r.fe_params)),
                      "beta": b, "se": se, "ci_lo": b - 1.96 * se,
                      "ci_hi": b + 1.96 * se, "p": float(r.pvalues[t]),
                      "converged": bool(r.converged), "optimiser": method,
                      'fallback_intercept': bool(fellback),
                      "llf": float(r.llf),
                      "beta_vmax": float(rml.params[t]) if rml is not None else np.nan,
                      "se_vmax": float(rml.bse[t]) if rml is not None else np.nan,
                      "p_vmax": float(rml.pvalues[t]) if rml is not None else np.nan,
                      "llf_vmax": float(rml.llf) if rml is not None else np.nan,
                      "optimiser_vmax": metml})
        log(f'  {abbrev:3s} {descr:42s} beta = {b:+.5f}  SE = {se:.5f}  95% CI '
            f'[{b - 1.96 * se:+.5f}, {b + 1.96 * se:+.5f}]  p = {r.pvalues[t]:.5f}  n = '
            f'{m.PATNO.nunique()}/{len(m)}  conv = {r.converged} ({method})')
        if rml is not None:
            log(f'      maximum likelihood ({metml}): beta = {rml.params[t]:+.5f}  SE = '
                f'{rml.bse[t]:.5f}  95% CI [{rml.params[t] - 1.96 * rml.bse[t]:+.5f}, '
                f'{rml.params[t] + 1.96 * rml.bse[t]:+.5f}]  p = {rml.pvalues[t]:.5f}  (logL '
                f'{r.llf:.3f} -> {rml.llf:.3f})')

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "77_fmax_dat.csv"), index=False)

# change with respect to M0
log("\n" + "=" * 78)
log('CHANGE IN THE ESTIMATE RELATIVE TO M0')
log("=" * 78)
for out, label in OUTCOMES.items():
    dd = d[d.outcome == out].set_index("spec")
    b0 = dd.loc["M0", "beta"]
    log(f'\n{label}: M0 beta = {b0:+.5f}')
    for abbrev in [s[0] for s in SPEC[1:]]:
        b = dd.loc[abbrev, "beta"]
        log(f"  {abbrev}: beta = {b:+.5f}  ({100 * (b - b0) / abs(b0):+.1f}% relative to M0)  p = "
            f"{dd.loc[abbrev, 'p']:.5f}")
    b0m = dd.loc["M0", "beta_vmax"]
    log(f'  (by maximum likelihood: M0 beta = {b0m:+.5f}; ' + "; ".join(
        f"{s[0]} {dd.loc[s[0], 'beta_vmax']:+.5f} "
        f"[{100 * (dd.loc[s[0], 'beta_vmax'] - b0m) / abs(b0m):+.0f}%] "
        f"p = {dd.loc[s[0], 'p_vmax']:.4f}" for s in SPEC[1:]) + ")")

# ---------------------------------------------------------------------------
# what remains of the correction for multiple comparisons
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log('CONSEQUENCE FOR THE BENJAMINI-HOCHBERG CORRECTION (28 tests, 24h cohort)')
log("=" * 78)
log("With BH over m = 28 tests, a result of rank r is significant at 5%")
log("only if its raw p is < 0.05 * r / 28. The necessary condition is")
log("therefore raw p < 0.05; the minimum rank required is ceil(p * 28 / 0.05).")
M_TEST = 28
for _, r_ in d.iterrows():
    for lbl, pv in [('cascade', r_.p), ('max. likelihood', r_.p_vmax)]:
        if not np.isfinite(pv):
            continue
        rank = int(np.ceil(pv * M_TEST / 0.05))
        outcome = (f'minimum rank required {rank} out of {M_TEST}'
                 if rank <= M_TEST
                 else 'cannot pass BH at any rank')
        log(f'  {r_.label[:22]:22s} {r_.spec:3s} ({lbl:16s}) p = {pv:.5f} -> {outcome}')

# ---------------------------------------------------------------------------
# M7 — where the signal lies: between the runs or within the runs?
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log('M7 — DECOMPOSITION OF z(Fmax) INTO BETWEEN-RUN AND WITHIN-RUN COMPONENTS')
log("=" * 78)
log('M6 removes the between-run variation from z(Fmax) and leaves only the within-run one.')
log("The decomposition puts the two components in the same model, so that")
log('the two coefficients are comparable and their difference testable.')
base['Fmax_between'] = base.groupby("run").Fmax.transform("mean")
base['Fmax_within'] = base.Fmax - base.Fmax_between
log(f'\nSD of the between-run component: {base.Fmax_between.std():.4f}; within runs: '
    f'{base.Fmax_within.std():.4f} (z(Fmax) has SD {base.Fmax.std():.4f})')
L = L.merge(base[["PATNO", 'Fmax_between', 'Fmax_within']], on="PATNO", how="left")

rows7 = []
for out, label in OUTCOMES.items():
    m = L[["PATNO", "years", out, 'Fmax_between', 'Fmax_within'] + COV].dropna()
    f = (f'{out} ~ Fmax_between*years + Fmax_within*years + ' + " + ".join(COV))
    r, method, _ = fit_mixed(f, m, m.PATNO)
    rml, metml = fit_ml(f, m, m.PATNO)
    log(f"\n--- {label} ---")
    for name, res, met in [('_fit.py cascade', r, method),
                           ('maximum likelihood', rml, metml)]:
        if res is None:
            log(f'  {name}: no convergence')
            continue
        tf = ('Fmax_between:years' if 'Fmax_between:years' in res.params.index
              else 'years:Fmax_between')
        te = ('Fmax_within:years' if 'Fmax_within:years' in res.params.index
              else 'years:Fmax_within')
        bf, sf = float(res.params[tf]), float(res.bse[tf])
        be, se_ = float(res.params[te]), float(res.bse[te])
        # test of the equality of the two coefficients
        cov = res.cov_params()
        if isinstance(cov, pd.DataFrame):
            vff, vee, vfe = cov.loc[tf, tf], cov.loc[te, te], cov.loc[tf, te]
        else:
            i = list(res.fe_params.index).index(tf)
            j = list(res.fe_params.index).index(te)
            A = np.asarray(cov)
            vff, vee, vfe = A[i, i], A[j, j], A[i, j]
        sd = float(np.sqrt(vff + vee - 2 * vfe))
        z = (bf - be) / sd
        pdiff = float(2 * stats.norm.sf(abs(z)))
        log(f'  {name} ({met}, logL {res.llf:.3f}):')
        log(f'    between runs x years: beta = {bf:+.5f}  SE = {sf:.5f}  95% CI '
            f'[{bf - 1.96 * sf:+.5f}, {bf + 1.96 * sf:+.5f}]  p = {res.pvalues[tf]:.5f}')
        log(f'    within runs  x years: beta = {be:+.5f}  SE = {se_:.5f}  95% CI '
            f'[{be - 1.96 * se_:+.5f}, {be + 1.96 * se_:+.5f}]  p = {res.pvalues[te]:.5f}')
        log(f'    difference between the two: {bf - be:+.5f} (SE {sd:.5f}, z = {z:+.2f}, p = '
            f'{pdiff:.4f})')
        rows7.append({"outcome": out, "label": label, 'estimate': name,
                       "optimiser": met, "llf": float(res.llf),
                       "n_subjects": int(m.PATNO.nunique()),
                       "n_obs": int(len(m)),
                       'beta_between_run': bf, 'se_between_run': sf,
                       'p_between_run': float(res.pvalues[tf]),
                       'beta_within_run': be, 'se_within_run': se_,
                       'p_within_run': float(res.pvalues[te]),
                       'difference': bf - be, 'se_difference': sd,
                       'z_difference': z, 'p_difference': pdiff})
pd.DataFrame(rows7).to_csv(
    os.path.join(OUT, "77_decomposition_run.csv"), index=False)

# ---------------------------------------------------------------------------
# the confounding path: is the run associated with the DAT outcome and with its
# slope?
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("IS THE RUN ASSOCIATED WITH THE DAT OUTCOME AND WITH ITS SLOPE?")
log("=" * 78)
extra_rows = []
for out, label in OUTCOMES.items():
    m = L[["PATNO", "years", out, "run"]].dropna()
    log(f"\n--- {label} ---")
    log(f'  {m.PATNO.nunique()} subjects, {len(m)} observations, {m.run.nunique()} run categories')

    # model with run x years, random intercept and slope.
    # Here the maximum-likelihood estimate is used: the variance components
    # are much more sensitive than the fixed effect to the end point
    # of the optimiser, and with the cascade (first that converges) one would end
    # up comparing two different local optima.
    r, method = fit_ml(f"{out} ~ C(run)*years", m, m.PATNO)
    conv = r is not None
    log(f'  {out} ~ C(run)*anni (random intercept and slope): convergence {conv} ({method}, logL '
        f'{r.llf:.3f})')

    # joint Wald tests on the run terms
    def wald(res, key):
        names = [n for n in res.fe_params.index if key(n)]
        if not names:
            return np.nan, np.nan, 0
        b = res.fe_params[names].to_numpy()
        cov = res.cov_params()
        if isinstance(cov, pd.DataFrame):
            V = cov.loc[names, names].to_numpy()
        else:
            idx = [list(res.fe_params.index).index(n) for n in names]
            V = np.asarray(cov)[np.ix_(idx, idx)]
        chi2 = float(b @ np.linalg.pinv(V) @ b)
        dof = len(names)
        return chi2, float(stats.chi2.sf(chi2, dof)), dof

    c_m, p_m, g_m = wald(r, lambda n: n.startswith("C(run)") and ":" not in n)
    c_i, p_i, g_i = wald(r, lambda n: "C(run)" in n and ":" in n)
    log(f'  main effect of the run (level): chi2 = {c_m:.2f}, df = {g_m}, p = {p_m:.3g}')
    log(f'  run x years (slope):            chi2 = {c_i:.2f}, df = {g_i}, p = {p_i:.3g}')

    # share of variance of the DAT slope explained by the run, in two ways
    # (a) two stages: OLS slope per subject, then eta^2 between runs
    pen = []
    for pid, g in m.groupby("PATNO"):
        if len(g) >= 3 and g.years.nunique() >= 2:
            sl = np.polyfit(g.years.to_numpy(float), g[out].to_numpy(float), 1)[0]
            pen.append({"PATNO": pid, "slope": sl, "run": g.run.iloc[0]})
    P = pd.DataFrame(pen)
    fit_a = smf.ols("slope ~ C(run)", P).fit()
    eta2_a = float(fit_a.rsquared)
    G, nP = P.run.nunique(), len(P)
    expected = (G - 1) / (nP - 1)
    # eta^2 is biased upwards with many groups: epsilon^2 is also
    # reported, which subtracts the share expected by chance
    eps2 = float(eta2_a - (G - 1) / (nP - G) * (1 - eta2_a))
    log(f'  (a) two stages: {nP} subjects with >=3 observations; eta^2 of the run on the slope = '
        f'{eta2_a:.4f} (expected under the null {expected:.4f}; F = {fit_a.fvalue:.2f}, p = '
        f'{fit_a.f_pvalue:.3g}); corrected epsilon^2 = {eps2:.4f}')

    # (b) proportional reduction of the variance of the random slope
    r0, met0 = fit_ml(f"{out} ~ years", m, m.PATNO)
    def var_slope(res):
        cr = res.cov_re
        names = [str(x) for x in cr.index]
        j = names.index("years") if "years" in names else len(names) - 1
        return float(cr.iloc[j, j])
    v0, v1 = var_slope(r0), var_slope(r)
    log(f'  (b) variance of the random slope: {v0:.6g} (without run, {met0}, logL {r0.llf:.3f}) -> '
        f'{v1:.6g} (with run x years); reduction {100 * (v0 - v1) / v0:+.1f}%')

    # level of the DAT at the first observation, by run
    bl_dat = L[["PATNO", "years", out, "run"]].dropna()
    bl_dat = bl_dat.sort_values("years").drop_duplicates("PATNO")
    fit_l = smf.ols(f"{out} ~ C(run)", bl_dat).fit()
    log(f'  DAT level at the first observation: eta^2 of the run = {fit_l.rsquared:.4f} (expected '
        f'{(bl_dat.run.nunique() - 1) / (len(bl_dat) - 1):.4f}; F = {fit_l.fvalue:.2f}, p = '
        f'{fit_l.f_pvalue:.3g}, n = {len(bl_dat)})')

    extra_rows.append({"outcome": out, "label": label,
                       "n_subjects": int(m.PATNO.nunique()),
                       "n_obs": int(len(m)),
                       'run_level_chi2': c_m, 'run_level_dof': g_m,
                       'run_level_p': p_m,
                       'run_x_years_chi2': c_i, 'run_x_years_dof': g_i,
                       'run_x_years_p': p_i,
                       "converged": conv, "optimiser": method,
                       'n_slopes': int(len(P)),
                       'eta2_run_over_slope': eta2_a,
                       'epsilon2_run_over_slope': eps2,
                       'eta2_expected_null': float(expected),
                       'eta2_slope_F': float(fit_a.fvalue),
                       'eta2_slope_p': float(fit_a.f_pvalue),
                       'var_slope_without_run': v0,
                       'var_slope_having_run': v1,
                       'reduction_var_slope_pct': 100 * (v0 - v1) / v0,
                       'eta2_run_over_level': float(fit_l.rsquared),
                       'level_p': float(fit_l.f_pvalue)})

pd.DataFrame(extra_rows).to_csv(
    os.path.join(OUT, "77_run_on_dat.csv"), index=False)

# ---------------------------------------------------------------------------
# aliasing between run and centre
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log('ALIASING BETWEEN ASSAY RUN AND CENTRE (595 subjects)')
log("=" * 78)
tab = pd.crosstab(base.run, base.site)
chi2, pchi, dof, _ = stats.chi2_contingency(tab)
n = len(base)
V = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
# bias-corrected V (Bergsma 2013)
phi2 = chi2 / n
k, rr = tab.shape[1], tab.shape[0]
phi2c = max(0, phi2 - (k - 1) * (rr - 1) / (n - 1))
kc = k - (k - 1) ** 2 / (n - 1)
rc = rr - (rr - 1) ** 2 / (n - 1)
Vc = np.sqrt(phi2c / min(kc - 1, rc - 1))
log(f'\ntable {tab.shape[0]} runs x {tab.shape[1]} centres')
log(f'chi2 = {chi2:.1f}, df = {dof}, p = {pchi:.3g}')
log(f'Cramer V = {V:.4f}   (bias-corrected: {Vc:.4f})')
log(f'a centre perfectly aliased with the run would give V = 1; complete independence V = 0')

# how concentrated each centre is in a single run and vice versa
share_run_in_centre = (tab.max(axis=0) / tab.sum(axis=0))
share_centre_in_run = (tab.max(axis=1) / tab.sum(axis=1))
log(f'\nshare of the subjects of a centre coming from the most frequent run of that centre: median '
    f'{share_run_in_centre.median():.3f}, minimum {share_run_in_centre.min():.3f}, maximum '
    f'{share_run_in_centre.max():.3f}')
log(f'centres entirely contained in a single run: {int((share_run_in_centre == 1).sum())} out of '
    f'{tab.shape[1]} ({int(tab.sum(axis=0)[share_run_in_centre == 1].sum())} subjects)')
log(f'share of the subjects of a run coming from the most frequent centre of that run: median '
    f'{share_centre_in_run.median():.3f}, maximum {share_centre_in_run.max():.3f}')
log(f'runs entirely contained in a single centre: {int((share_centre_in_run == 1).sum())} out of '
    f'{tab.shape[0]}')

tab.to_csv(os.path.join(OUT, "77_run_x_centre.csv"))
log('\ncontingency table run x centre (rows = run, columns = centre):')
log(tab.to_string())

log("\n" + "=" * 78)
log('Files written: outputs/77_fmax_dat.csv, outputs/77_fmax_dat.txt,')
log('              outputs/77_run_on_dat.csv, outputs/77_run_x_centre.csv')
with open(os.path.join(OUT, "77_fmax_dat.txt"), "w") as fh:
    fh.write("\n".join(rep))
