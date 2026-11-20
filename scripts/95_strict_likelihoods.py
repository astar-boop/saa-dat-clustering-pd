"""
95 - The FULL-LIKELIHOOD fits of the spline ladder and the two run x time
     models, redone under the STRICT RULE (selection + restarts + Newton
     decrement).

WHY THIS SCRIPT EXISTS
----------------------
§2.18 declared two exceptions to the strict criterion:

  (d) "Only the restricted-maximum-likelihood fits were re-examined: the
      full-maximum-likelihood refits used for the likelihood-ratio comparisons
      of §3.13 follow the first-converging rule as well."
  (c) "The two run-by-time tests quoted in the text of §3.12 were selected in
      the same way" (that is, selection on the likelihood WITHOUT the round of
      restarts).

The fifth round of review observes that precisely on these models the
manuscript documents an extreme dependence on the optimiser: on the two
run x time models the first optimiser that converges stops 27 and 23
log-likelihood units below the maximum and changes the omnibus statistics
(chi2 36.3 instead of 43.3 on the putamen; 7.1 instead of 29.2 on the striatum).
As long as those fits remain outside the strict criterion, the published p
values are optimiser-dependent descriptions.

This script brings inside the strict criterion BOTH MEMBERS of every ML
likelihood ratio reported in §3.13 and the models that define the two
run x time tests of §3.12, and measures how much the printed statistics
move.

WHICH FITS ENTER WHICH TESTS, AND UNDER WHICH RULE THEY WERE PRODUCED
---------------------------------------------------------------------
A) The spline ladder of the MoCA, 150 h cohort (script 78 -> 83 -> 88_ladder_df).
   The five models, all with random intercept and slope per participant:
     A1  moca ~ T50*years + eta + sesso + durata + istruzione
     A2  moca ~ T50 + years + years2 + T50:years + T50:years2 + covariates
     A3  moca ~ T50 + years + years2 + T50:years + covariates
     A4  moca ~ T50 + s1 + s2 + s3 + T50:s1 + T50:s2 + T50:s3 + covariates
     A5r moca ~ T50 + s1 + s2 + s3 + T50:years + covariates    (full rank)
   plus A5, the parametrisation of 78 with the redundant 'years' column, which
   does not enter any printed test but serves to verify that A5 and A5r are
   the same model.
   The four comparisons printed in the manuscript, all on reml=False fits:
     A1 -> A5r  shape of the mean trajectory         chi2 47.87  2 df
     A5r -> A4  shape of the interaction             chi2  2.33  2 df
     A1 -> A4   the two things together              chi2 50.20  4 df
     A3 -> A2   quadratic restriction                chi2  0.37  1 df
   The degrees of freedom come from the RANK of the fixed-effects matrix, not
   from the number of columns (88_ladder_df.py). The estimation rule was the
   `fit_ml` cascade of 88_ladder_df.py: FIRST optimiser that declares
   convergence, no restart, no stationarity criterion.

B) The two run x time tests of §3.12 (script 77_fmax_dat_batch.py, final lines;
   per optimiser in 85_run_centre.py). The model is
     MIA_PUTAMEN_BILAT ~ C(run)*years        (595 participants, 1,609 observations)
     MIA_CAUDATE_BILAT ~ C(run)*years
   with random intercept and slope, fitted by REML (the function is called
   `fit_ml` in 77 but does not pass reml=False: "ml" stands for "maximum
   likelihood among the optimisers", not for "maximum likelihood"). The
   printed test is a joint Wald test on the 20 terms C(run)[T.*]:years. The
   rule was selection on the likelihood WITHOUT restarts.
   Since the reviewer speaks of "the models defining both run-by-time tests",
   here both the model of the Wald test (REML) and the nested pair that defines
   the analogous ML likelihood ratio, `C(run)*years` against `C(run) + years`,
   which had never been computed, are brought inside the strict criterion.

STRICT RULE (identical to §2.18 and to 88_diagnostics_63.procedura)
-------------------------------------------------------------------
Stage 1 - cascade. The same model is fitted with each of the four
  optimisers of _fit.OTTIMIZZATORI - lbfgs (300), powell (2000),
  nelder-mead (5000), bfgs (2000) - from the statsmodels initial values. For
  each one, `converged` and the profiled log-likelihood are recorded, the
  latter evaluated on A SINGLE model instance so that the four numbers are
  comparable bit by bit.
Stage 2 - selection. Among those that converge the highest log-likelihood
  is kept.
Stage 3 - restarts. From that point all four optimisers restart
  (`start_params`), accepting every gain greater than 1e-10,
  until none gains any more (at most 10 rounds).
Stationarity criterion: Newton decrement
    delta = 1/2 * g' (-H)^{-1} g,      threshold  delta < 1e-4
with g the analytical score in the packed space in which the optimiser works
and H by central differences of the same profiled log-likelihood. The
primitives are COPIED WITHOUT CHANGES from 88_diagnostics_63.py, so that the
numbers are comparable with those already in circulation.

THE RIGHT OBJECTIVE. `MixedLM.fit` writes `self.reml` on the model, so
`md.loglike` and `md.score` return the FULL likelihood on the reml=False
fits and the RESTRICTED one on the REML fits. Each fit is therefore diagnosed
on its own objective: the six models of the spline ladder on ML, the two models
of the run x time Wald test on REML, the four models of the run x time
likelihood ratio on ML. The `objective` column records it row by row.

Output: outputs/95_strict_likelihoods.csv
        outputs/95_strict_likelihoods.txt
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS  # same cascade, same order, same maxiter
from _paths import DATA
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA_FILE = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")

TOL_NEWTON = 1e-4
TOL_SINGULAR = 1e-8
TOL_CORR = 0.999
TOL_GAIN = 1e-10
MAX_ROUNDS = 10
N_MIN = 5           # run pooling threshold, identical to 77

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def ps(p):
    if p is None or not np.isfinite(p):
        return "n/d"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.3e}"


# ============================================================ primitives =====
# Copied verbatim from 88_diagnostics_63.py. The only difference is in `estimate`,
# which accepts the formula of the random effects and the reml flag, and in
# `diagnostics`, where the interaction term is optional: models A2 and
# A4 do not have a single interaction coefficient and the run x time models
# have none. The numerical core (likelihood, score, central
# gradient, profiled Hessian, Newton decrement) is identical character by
# character.
def pack(r):
    md = r.model
    return np.asarray(
        r.params_object.get_packed(use_sqrt=md.use_sqrt, has_fe=False),
        dtype=float)


def _ll(md, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(md.loglike(p))


def _score(md, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(md.score(p), dtype=float)


def grad_central(md, p):
    g = np.empty(len(p))
    for i in range(len(p)):
        h = 1e-5 * max(1.0, abs(p[i]))
        pp, pm = p.copy(), p.copy()
        pp[i] += h
        pm[i] -= h
        g[i] = (_ll(md, pp) - _ll(md, pm)) / (2 * h)
    return g


def hess_profiled(md, p):
    """Central-difference Hessian of the profiled log-likelihood,
    in the same parameter space in which the optimiser works."""
    k = len(p)
    h = np.array([1e-4 * max(1.0, abs(v)) for v in p])
    H = np.zeros((k, k))
    f0 = _ll(md, p)
    for i in range(k):
        pp, pm = p.copy(), p.copy()
        pp[i] += h[i]
        pm[i] -= h[i]
        H[i, i] = (_ll(md, pp) - 2 * f0 + _ll(md, pm)) / h[i] ** 2
        for j in range(i + 1, k):
            a, b, c, d = p.copy(), p.copy(), p.copy(), p.copy()
            a[i] += h[i]; a[j] += h[j]
            b[i] += h[i]; b[j] -= h[j]
            c[i] -= h[i]; c[j] += h[j]
            d[i] -= h[i]; d[j] -= h[j]
            H[i, j] = H[j, i] = (_ll(md, a) - _ll(md, b) - _ll(md, c)
                                 + _ll(md, d)) / (4 * h[i] * h[j])
    return (H + H.T) / 2, f0


def estimate(formula, data, groups, method, it, start=None, reml=True,
          re_formula="~years"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kw = {} if start is None else {"start_params": start}
            return smf.mixedlm(formula, data, groups=groups,
                               re_formula=re_formula).fit(method=method,
                                                          maxiter=it,
                                                          reml=reml, **kw)
        except Exception:
            return None


def procedure(formula, data, groups, reml=True, re_formula="~years"):
    """Stages 1-3 of the strict rule. Copied verbatim from 88, with the only
    change that `reml` and `re_formula` are passed to `estimate`."""
    attempts = {}
    md_ref = None
    for method, it in OPTIMISERS:
        r = estimate(formula, data, groups, method, it, reml=reml,
                  re_formula=re_formula)
        if r is None:
            attempts[method] = (None, np.nan, False)
            continue
        if md_ref is None:
            md_ref = r.model
        p = pack(r)
        attempts[method] = (r, _ll(md_ref, p), bool(r.converged))
    if md_ref is None:
        return None
    conv = [(m, v[1]) for m, v in attempts.items()
            if v[0] is not None and v[2] and np.isfinite(v[1])]
    if conv:
        chosen = max(conv, key=lambda x: x[1])[0]
        between_convergent = True
    else:
        fin = [(m, v[1]) for m, v in attempts.items()
               if v[0] is not None and np.isfinite(v[1])]
        if not fin:
            return None
        chosen = max(fin, key=lambda x: x[1])[0]
        between_convergent = False
    r0, ll0, _ = attempts[chosen]
    best = (r0, '(none)', ll0, pack(r0))
    rounds = 0
    for cycle in range(MAX_ROUNDS):
        improved = False
        for method, it in OPTIMISERS:
            r = estimate(formula, data, groups, method, it, start=best[3],
                      reml=reml, re_formula=re_formula)
            if r is None:
                continue
            p = pack(r)
            ll = _ll(md_ref, p)
            if np.isfinite(ll) and ll > best[2] + TOL_GAIN:
                best = (r, method, ll, p)
                improved = True
        rounds = cycle + 1
        if not improved:
            break
    # the published rule for the spline ladder: first optimiser that converges
    first = None
    for method, _ in OPTIMISERS:
        v = attempts.get(method)
        if v is not None and v[0] is not None and v[2]:
            first = method
            break
    return {'r_cascade': r0, 'opt_chosen': chosen, 'll_choice': ll0,
            'attempts': attempts, 'r_final': best[0],
            'opt_reopt': best[1], 'rounds': rounds, 'll_final': best[2],
            'md_ref': md_ref, 'selection_between_convergent': between_convergent,
            'opt_first_convergent': first}


def diagnostics(r, t=None):
    """Copied verbatim from 88; `t` is optional because not all the models
    here have a single interaction coefficient to report."""
    md = r.model
    p = pack(r)
    g = _score(md, p)
    g_num = grad_central(md, p)
    H, ll = hess_profiled(md, p)
    eig = np.linalg.eigvalsh(H)
    try:
        step = np.linalg.solve(H, -g)
        newton = float(0.5 * g @ step)
    except np.linalg.LinAlgError:
        newton = np.nan
    cov_re = np.asarray(r.cov_re, dtype=float)
    scale = float(r.scale)
    v_int, v_slope = float(cov_re[0, 0]), float(cov_re[1, 1])
    corr = (float(cov_re[0, 1]) / np.sqrt(v_int * v_slope)
            if v_int > 0 and v_slope > 0 else np.nan)
    sing = bool(v_int < TOL_SINGULAR * scale or v_slope < TOL_SINGULAR * scale
                or (np.isfinite(corr) and abs(corr) >= TOL_CORR))
    edge = bool(v_int == 0.0 or v_slope == 0.0)
    cp = np.asarray(r.cov_params(), dtype=float)
    k_fe = int(md.k_fe)
    fe = cp[:k_fe, :k_fe]
    fe = (fe + fe.T) / 2
    eig_fe = np.linalg.eigvalsh(fe)
    return {
        "grad_inf_norm": float(np.max(np.abs(g))),
        "grad_check_analytic_vs_central": float(np.max(np.abs(g - g_num))),
        "newton_decrement": newton,
        "loglike": ll,
        "profile_hessian_neg_definite": bool(eig.max() < 0),
        "profile_hessian_max_eigenvalue": float(eig.max()),
        "re_var_intercept": v_int,
        "re_var_slope": v_slope,
        "re_corr_intercept_slope": corr,
        "residual_scale": scale,
        "singular": sing,
        "boundary": edge,
        "fe_cov_pos_def": bool(eig_fe.min() > 0),
        "fe_cov_cond": float(np.linalg.cond(fe)),
        "beta_interaction": (float(r.params[t]) if t is not None else np.nan),
        "se": (float(r.bse[t]) if t is not None else np.nan),
        "p": (float(r.pvalues[t]) if t is not None else np.nan),
        "converged": bool(r.converged),
    }


def rank(X):
    """Numerical rank with the same explicit tolerance as 88_ladder_df."""
    X = np.asarray(X, dtype=float)
    s = np.linalg.svd(X, compute_uv=False)
    tol = max(X.shape) * np.finfo(float).eps * s[0]
    return int((s > tol).sum()), float(tol)


def wald_run(res):
    """Joint Wald test on the terms C(run)[T.*]:years. Copied verbatim from the
    internal function `wald` of 77_fmax_dat_batch.py (same key, same
    pseudo-inverse)."""
    names = [n for n in res.fe_params.index if "C(run)" in n and ":" in n]
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
    return chi2, float(st.chi2.sf(chi2, dof)), dof


# ================================================================= data ======
log("=" * 78)
log('95 — FULL-LIKELIHOOD FITS AND RUN x TIME MODELS UNDER THE STRICT')
log('     CRITERION')
log("=" * 78)
log(f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}")
log(f'BASE = {BASE}')
log(f'cascade: {OPTIMISERS}')
log(f'stage 1: all the optimisers from the starting values of statsmodels')
log(f"stage 2: among the convergent ones, the highest log-likelihood")
log(f'stage 3: restarts from the chosen point (max {MAX_ROUNDS} rounds, minimum gain {TOL_GAIN:g})')
log(f"criterion: Newton decrement < {TOL_NEWTON:g}, on the objective of the fit")
log("")

cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))

# --- A) the spline ladder of the MoCA, 150 h cohort -------------------------
# Construction identical to 78_longitudinal_sensitivity.py, 83_spline_reconciled.py and
# 88_ladder_df.py.
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete].reset_index(drop=True)
d150["T50"] = np.log(d150.k_T50)
d150["T50"] = (d150.T50 - d150.T50.mean()) / d150.T50.std()

covbl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                   "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base150 = d150[["PATNO", "T50"]].merge(covbl, on="PATNO", how="left")

Lm = cur[cur.PATNO.isin(base150.PATNO)][["PATNO", "YEAR", "moca"]].copy()
Lm["years"] = pd.to_numeric(Lm.YEAR, errors="coerce")
Lm = Lm[Lm.years.notna()].merge(base150, on="PATNO", how="left")

COV = ["age_bl", "SEX", "dur_bl", "EDUCYRS"]
m = Lm[["PATNO", "years", "moca", "T50"] + COV].dropna().reset_index(drop=True)
m["years"] = m.years.astype(float)
m["years2"] = m.years ** 2

SPL_SPEC = "cr(years, df=3, constraints='center')"
_B = dmatrix(SPL_SPEC, {"years": m.years}, return_type="dataframe")
_B = _B.drop(columns=[c for c in _B.columns if c == "Intercept"])
SPL = ["s1", "s2", "s3"]
for i, c in enumerate(_B.columns):
    m[SPL[i]] = _B[c].values

log(f'A) spline scale of the MoCA: {m.PATNO.nunique()} subjects, {len(m)} observations, years '
    f'{m.years.min():.0f}-{m.years.max():.0f}')

# --- B) the two run x time models, 24 h cohort ------------------------------
# Construction identical to 77_fmax_dat_batch.py.
OUTCOMES_DAT = {"MIA_PUTAMEN_BILAT": "DAT putamen, bilateral",
                "MIA_CAUDATE_BILAT": "DAT caudate, bilateral"}
cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
cc = cc[cc.complete_primary]
baseD = cc[["PATNO", "saa_Fmax"]].rename(columns={"saa_Fmax": "Fmax"}).copy()
saa = pd.read_csv(SAA_FILE)
s = saa[(saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")
        & (saa.PATNO.isin(set(baseD.PATNO)))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
s = s[["PATNO", "RUNDATE"]].rename(columns={"RUNDATE": "run_raw"})
baseD = baseD.merge(s, on="PATNO", how="left")
vr = baseD.run_raw.astype(str).value_counts()
baseD["run"] = baseD.run_raw.astype(str).where(
    baseD.run_raw.astype(str).map(vr) >= N_MIN, 'other')

LD = cur[cur.PATNO.isin(baseD.PATNO)][
    ["PATNO", "EVENT_ID", "YEAR"] + list(OUTCOMES_DAT)].copy()
LD["years"] = pd.to_numeric(LD.YEAR, errors="coerce")
LD = LD[LD.years.notna()]
LD = LD.merge(baseD[["PATNO", "run"]], on="PATNO", how="left")

DATA_RUN = {}
for out_ in OUTCOMES_DAT:
    DATA_RUN[out_] = LD[["PATNO", "years", out_, "run"]].dropna()
    log(f'B) run x time, {OUTCOMES_DAT[out_]}: {DATA_RUN[out_].PATNO.nunique()} subjects, '
        f'{len(DATA_RUN[out_])} observations, {DATA_RUN[out_].run.nunique()} run categories')
log("")

# ============================================== the fits to redo =============
# (key, label, formula, data, reml, published rule)
FIT = [
    ("A1", 'spline ladder A1 — linear time (reference)',
     "moca ~ T50*years + age_bl + SEX + dur_bl + EDUCYRS", "moca", False,
     'first converged'),
    ("A2", 'spline ladder A2 — quadratic, interaction on both terms',
     "moca ~ T50 + years + years2 + T50:years + T50:years2 "
     "+ age_bl + SEX + dur_bl + EDUCYRS", "moca", False, 'first converged'),
    ("A3", 'spline ladder A3 — quadratic, interaction on the linear term only',
     "moca ~ T50 + years + years2 + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS", "moca", False, 'first converged'),
    ("A4", 'spline ladder A4 — spline, interaction on the whole basis',
     "moca ~ T50 + s1 + s2 + s3 + T50:s1 + T50:s2 + T50:s3 "
     "+ age_bl + SEX + dur_bl + EDUCYRS", "moca", False, 'first converged'),
    ("A5r", 'spline ladder A5r — spline, interaction on the linear (full rank)',
     "moca ~ T50 + s1 + s2 + s3 + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS", "moca", False, 'first converged'),
    ("A5", "spline ladder A5 — as A5r but with redundant 'years' (auxiliary)",
     "moca ~ T50 + s1 + s2 + s3 + years + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS", "moca", False, 'first converged'),
    ("PUT_full_REML", 'run x time, putamen — model of the Wald (REML)',
     "MIA_PUTAMEN_BILAT ~ C(run)*years", "MIA_PUTAMEN_BILAT", True,
     'selection on the likelihood, without restarts'),
    ("CAU_full_REML", 'run x time, caudate — model of the Wald (REML)',
     "MIA_CAUDATE_BILAT ~ C(run)*years", "MIA_CAUDATE_BILAT", True,
     'selection on the likelihood, without restarts'),
    ("PUT_full_ML", 'run x time, putamen — full model (ML, for the LRT)',
     "MIA_PUTAMEN_BILAT ~ C(run)*years", "MIA_PUTAMEN_BILAT", False,
     '(not published)'),
    ("PUT_red_ML", 'run x time, putamen — nested model (ML, for the LRT)',
     "MIA_PUTAMEN_BILAT ~ C(run) + years", "MIA_PUTAMEN_BILAT", False,
     '(not published)'),
    ("CAU_full_ML", 'run x time, caudate — full model (ML, for the LRT)',
     "MIA_CAUDATE_BILAT ~ C(run)*years", "MIA_CAUDATE_BILAT", False,
     '(not published)'),
    ("CAU_red_ML", 'run x time, caudate — nested model (ML, for the LRT)',
     "MIA_CAUDATE_BILAT ~ C(run) + years", "MIA_CAUDATE_BILAT", False,
     '(not published)'),
]

# which test each fit belongs to, and in which role
MEMBERSHIP = {
    "A1": [('A1 -> A5r (spline, shape of the mean)', 'denominator'),
           ('A1 -> A4 (spline, mean + interaction)', 'denominator')],
    "A5r": [('A1 -> A5r (spline, shape of the mean)', 'numerator'),
            ('A5r -> A4 (spline, shape of the interaction)', 'denominator')],
    "A4": [('A5r -> A4 (spline, shape of the interaction)', 'numerator'),
           ('A1 -> A4 (spline, mean + interaction)', 'numerator')],
    "A3": [('A3 -> A2 (quadratic restriction)', 'denominator')],
    "A2": [('A3 -> A2 (quadratic restriction)', 'numerator')],
    "A5": [('(no test printed: reparametrisation check)',
            'auxiliary')],
    "PUT_full_REML": [('run x time, putamen (Wald, REML)', 'single model')],
    "CAU_full_REML": [('run x time, caudate (Wald, REML)', 'single model')],
    "PUT_full_ML": [('run x time, putamen (LRT, ML)', 'numerator')],
    "PUT_red_ML": [('run x time, putamen (LRT, ML)', 'denominator')],
    "CAU_full_ML": [('run x time, caudate (LRT, ML)', 'numerator')],
    "CAU_red_ML": [('run x time, caudate (LRT, ML)', 'denominator')],
}

rows_fit = []
R = {}
log("-" * 78)
log('REFITS UNDER THE STRICT RULE')
log("-" * 78)
for key, et, formula, datum, reml, rule in FIT:
    dd = m if datum == "moca" else DATA_RUN[datum]
    t0 = time.time()
    P = procedure(formula, dd, dd.PATNO, reml=reml)
    if P is None:
        log(f'  {key}: NO FIT')
        continue
    rf = P['r_final']
    D = diagnostics(rf)
    X = np.asarray(rf.model.exog, dtype=float)
    k_col = int(X.shape[1])
    k_rank, tol_rank = rank(X)
    ll_first = (P['attempts'][P['opt_first_convergent']][1]
                if P['opt_first_convergent'] else np.nan)
    ll_publ = (ll_first if rule == 'first converged' else P['ll_choice'])
    opt_publ = (P['opt_first_convergent'] if rule == 'first converged'
                else P['opt_chosen'])
    R[key] = {"P": P, "D": D, "r": rf, "k_col": k_col, "k_rank": k_rank,
                 "reml": reml, "et": et}
    test_e_roles = MEMBERSHIP[key]
    row = {
        'section': "fit",
        "fit": key,
        'label': et,
        "test": " | ".join(t for t, _ in test_e_roles),
        'role': " | ".join(ru for _, ru in test_e_roles),
        'objective': "REML" if reml else "ML (reml=False)",
        "formula": formula,
        "re_formula": "~years",
        'n_subjects': int(dd.PATNO.nunique()),
        'n_observations': int(len(dd)),
        'rule_pubd': rule,
        'optimiser_rule_pubd': opt_publ,
        'loglike_rule_pubd': ll_publ,
        'optimiser_first_convergent': P['opt_first_convergent'],
        'loglike_first_convergent': ll_first,
        'optimiser_kept': P['opt_chosen'],
        'selection_between_convergent': P['selection_between_convergent'],
        'loglike_earliest_of_restarts': P['ll_choice'],
        'optimiser_restart': P['opt_reopt'],
        'rounds_restart': P['rounds'],
        'loglike_after_the_restarts': P['ll_final'],
        'gain_restarts': P['ll_final'] - P['ll_choice'],
        'gain_vs_rule_pubd': P['ll_final'] - ll_publ,
        "grad_inf_norm": D["grad_inf_norm"],
        "grad_check_analytic_vs_central": D["grad_check_analytic_vs_central"],
        "newton_decrement": D["newton_decrement"],
        "meets_newton_criterion": bool(
            np.isfinite(D["newton_decrement"])
            and D["newton_decrement"] < TOL_NEWTON),
        "singular": D["singular"],
        "boundary": D["boundary"],
        "profile_hessian_neg_definite": D["profile_hessian_neg_definite"],
        "profile_hessian_max_eigenvalue": D["profile_hessian_max_eigenvalue"],
        "re_var_intercept": D["re_var_intercept"],
        "re_var_slope": D["re_var_slope"],
        "re_corr_intercept_slope": D["re_corr_intercept_slope"],
        "residual_scale": D["residual_scale"],
        "fe_cov_pos_def": D["fe_cov_pos_def"],
        "fe_cov_cond": D["fe_cov_cond"],
        'k_effects_fixed': k_col,
        'k_rank': k_rank,
        'deficiency_of_rank': k_col - k_rank,
        'tolerance_rank': tol_rank,
        "converged": D["converged"],
    }
    for method, _ in OPTIMISERS:
        r_, ll_, cv_ = P['attempts'].get(method, (None, np.nan, False))
        row[f'loglike_{method}'] = ll_
        row[f'converged_{method}'] = cv_
    rows_fit.append(row)
    log(f"  {key:<14s} {('ML ' if not reml else 'REML')}  col/rank {k_col}/{k_rank}  "
        f"publ({opt_publ}) {ll_publ:.6f} -> chosen({P['opt_chosen']}) {P['ll_choice']:.6f} -> "
        f"restarts({P['opt_reopt']}, {P['rounds']} rounds) {P['ll_final']:.6f}  |g| "
        f"{D['grad_inf_norm']:.3g}  newton {D['newton_decrement']:.3g}  ({time.time() - t0:.0f}s)")

DF = pd.DataFrame(rows_fit)

# ====================================================== the tests ============
log("")
log("-" * 78)
log('THE TESTS, AS PUBLISHED AND AS RE-ESTIMATED')
log("-" * 78)

rows_test = []


def add_test(name, kind, num, den, chi2_pub, dof_pub, p_pub, source,
                  chi2_new, dof_new, p_new, remark=""):
    rows_test.append({
        'section': "test", "test": name, 'kind': kind,
        'fit_numerator': num, 'fit_denominator': den,
        'source_pubd': source,
        'chi2_published': chi2_pub, 'df_published': dof_pub,
        'p_published': p_pub,
        'chi2_reestimated': chi2_new, 'df_reestimated': dof_new,
        'p_reestimated': p_new,
        "delta_chi2": (abs(chi2_new - chi2_pub)
                       if np.isfinite(chi2_pub) else np.nan),
        "delta_df": (abs(dof_new - dof_pub)
                     if np.isfinite(dof_pub) else np.nan),
        "delta_p": abs(p_new - p_pub) if np.isfinite(p_pub) else np.nan,
        'remark': remark})


# --- A) the four likelihood ratios of the spline ladder ---------------------
# published values read from outputs/88_ladder_df.csv, not copied by hand
gl = pd.read_csv(os.path.join(OUT, "88_ladder_df.csv"))
lrt_pub = gl[gl.section == 'nested comparison (LRT on ML)'].set_index('model')

SPLINE_TEST = [
    ('A1 -> A5r (spline, shape of the mean)', "A5r", "A1", "A1 -> A5r"),
    ('A5r -> A4 (spline, shape of the interaction)', "A4", "A5r", "A5r -> A4"),
    ('A1 -> A4 (spline, mean + interaction)', "A4", "A1", "A1 -> A4"),
    ('A3 -> A2 (quadratic restriction)', "A2", "A3", "A3 -> A2"),
]
for name, num, den, key_pub in SPLINE_TEST:
    pub = lrt_pub.loc[key_pub]
    chi2 = float(2 * (R[num]["P"]['ll_final'] - R[den]["P"]['ll_final']))
    dof = int(R[num]["k_rank"] - R[den]["k_rank"])
    p = float(st.chi2.sf(chi2, dof))
    add_test(name, 'likelihood ratio (ML)', num, den,
                  float(pub.stat_joint), int(pub.df_joint),
                  float(pub.p_joint), "outputs/88_ladder_df.csv",
                  chi2, dof, p)
    log(f'  {name:<44s} published  chi2={pub.stat_joint:10.4f} df={int(pub.df_joint)} '
        f'p={ps(float(pub.p_joint)):>11s}  |  strict chi2={chi2:10.4f} df={dof} p={ps(p):>11s}  '
        f'(|dchi2| {abs(chi2 - float(pub.stat_joint)):.2e})')

# additivity of the decomposition
c15 = float(2 * (R["A5r"]["P"]['ll_final'] - R["A1"]["P"]['ll_final']))
c54 = float(2 * (R["A4"]["P"]['ll_final'] - R["A5r"]["P"]['ll_final']))
c14 = float(2 * (R["A4"]["P"]['ll_final'] - R["A1"]["P"]['ll_final']))
resid_add = abs(c15 + c54 - c14)
log(f"  additivity: {c15:.4f} + {c54:.4f} = {c15 + c54:.4f} against {c14:.4f}   (residual "
    f"{resid_add:.2e})")
rows_test.append({'section': "test", "test": 'additivity of the decomposition',
                   'kind': 'algebraic check',
                   'fit_numerator': "A1 -> A5r + A5r -> A4",
                   'fit_denominator': "A1 -> A4",
                   'chi2_reestimated': c15 + c54, 'chi2_published': c14,
                   "delta_chi2": resid_add,
                   'df_reestimated': 4, 'df_published': 4, "delta_df": 0,
                   'remark': 'sum of the two steps versus the 4 df comparison'})

# reparametrisation check: A5 and A5r are the same model
d_a5 = abs(R["A5"]["P"]['ll_final'] - R["A5r"]["P"]['ll_final'])
log(f"  A5 against A5r (same model space): |dlogL| = {d_a5:.3e}; rank {R['A5']['k_rank']} against "
    f"{R['A5r']['k_rank']}, columns {R['A5']['k_col']} against {R['A5r']['k_col']}")
rows_test.append({'section': "test", "test": 'A5 against A5r',
                   'kind': 'reparametrisation check',
                   'fit_numerator': "A5", 'fit_denominator': "A5r",
                   'chi2_reestimated': d_a5, "delta_chi2": d_a5,
                   'remark': 'same model space: |dlogL| must be ~0'})

# --- B) the two run x time tests -------------------------------------------
publ_run = pd.read_csv(os.path.join(OUT, "77_run_on_dat.csv")).set_index(
    "outcome")
for out_, key, name in [
        ("MIA_PUTAMEN_BILAT", "PUT_full_REML",
         'run x time, putamen (Wald, REML)'),
        ("MIA_CAUDATE_BILAT", "CAU_full_REML",
         'run x time, caudate (Wald, REML)')]:
    pr = publ_run.loc[out_]
    chi2, p, dof = wald_run(R[key]["r"])
    add_test(name, 'joint Wald (REML)', key, '(none)',
                  float(pr.run_x_years_chi2), int(pr.run_x_years_dof),
                  float(pr.run_x_years_p), "outputs/77_run_on_dat.csv",
                  chi2, dof, p)
    log(f'  {name:<44s} published  chi2={pr.run_x_years_chi2:10.4f} df={int(pr.run_x_years_dof)} '
        f'p={ps(float(pr.run_x_years_p)):>11s}  |  strict chi2={chi2:10.4f} df={dof} p={ps(p):>11s}  '
        f'(|dchi2| {abs(chi2 - float(pr.run_x_years_chi2)):.2e})')

for pre, name in [("PUT", 'run x time, putamen (LRT, ML)'),
                  ("CAU", 'run x time, caudate (LRT, ML)')]:
    num, den = f"{pre}_full_ML", f"{pre}_red_ML"
    chi2 = float(2 * (R[num]["P"]['ll_final'] - R[den]["P"]['ll_final']))
    dof = int(R[num]["k_rank"] - R[den]["k_rank"])
    p = float(st.chi2.sf(chi2, dof))
    add_test(name, 'likelihood ratio (ML)', num, den,
                  np.nan, np.nan, np.nan, '(not published)', chi2, dof, p,
                  remark='likelihood analogue of the published Wald')
    log(f'  {name:<44s} published         —                       |  strict chi2={chi2:10.4f} '
        f'df={dof} p={ps(p):>11s}')

DT = pd.DataFrame(rows_test)

# ================================================== saving ==================
COLS_FIT = ['section', "fit", 'label', "test", 'role', 'objective',
            "formula", "re_formula", 'n_subjects', 'n_observations',
            'rule_pubd', 'optimiser_rule_pubd',
            'loglike_rule_pubd', 'optimiser_first_convergent',
            'loglike_first_convergent'] \
    + [f"loglike_{mm}" for mm, _ in OPTIMISERS] \
    + [f"converged_{mm}" for mm, _ in OPTIMISERS] \
    + ['optimiser_kept', 'selection_between_convergent',
       'loglike_earliest_of_restarts', 'optimiser_restart',
       'rounds_restart', 'loglike_after_the_restarts', 'gain_restarts',
       'gain_vs_rule_pubd', "grad_inf_norm",
       "grad_check_analytic_vs_central", "newton_decrement",
       "meets_newton_criterion", "singular", "boundary",
       "profile_hessian_neg_definite", "profile_hessian_max_eigenvalue",
       "re_var_intercept", "re_var_slope", "re_corr_intercept_slope",
       "residual_scale", "fe_cov_pos_def", "fe_cov_cond", 'k_effects_fixed',
       'k_rank', 'deficiency_of_rank', 'tolerance_rank', "converged",
       ]
COLS_TEST = ['section', "test", 'kind', 'fit_numerator', 'fit_denominator',
             'source_pubd', 'chi2_published', 'df_published',
             'p_published', 'chi2_reestimated', 'df_reestimated', 'p_reestimated',
             "delta_chi2", "delta_df", "delta_p", 'remark']
WHOLE = pd.concat([DF.reindex(columns=COLS_FIT),
                   DT.reindex(columns=COLS_TEST)], ignore_index=True)
WHOLE = WHOLE[COLS_FIT + [c for c in COLS_TEST if c not in COLS_FIT]]
WHOLE.to_csv(os.path.join(OUT, "95_strict_likelihoods.csv"), index=False)

# ==================================================== summary ===============
log("")
log("=" * 78)
log('SUMMARY')
log("=" * 78)
log(f"fits redone: {len(DF)}  (ML {int((DF.objective == 'ML (reml=False)').sum())}, REML "
    f"{int((DF.objective == 'REML').sum())})")
log(f'fits that meet the Newton criterion (< {TOL_NEWTON:g}): {int(DF.meets_newton_criterion.sum())} '
    f'out of {len(DF)}')
log(f'Newton decrement — min {DF.newton_decrement.min():.3e}   median '
    f'{DF.newton_decrement.median():.3e}   max {DF.newton_decrement.max():.3e}')
log(f'infinity norm of the gradient — min {DF.grad_inf_norm.min():.3e}   median '
    f'{DF.grad_inf_norm.median():.3e}   max {DF.grad_inf_norm.max():.3e}')
log(f'maximum difference between analytical score and central differences: '
    f'{DF.grad_check_analytic_vs_central.max():.3e}')
log(f'gain from the restarts — max {DF.gain_restarts.max():.3e} (fits with gain > 1e-6: '
    f'{int((DF.gain_restarts > 1e-06).sum())} out of {len(DF)})')
log(f'gain relative to the published rule — max {DF.gain_vs_rule_pubd.max():.6g}')
log(f'singular fits: {int(DF.singular.sum())}; at the boundary: {int(DF.boundary.sum())}; profiled '
    f'Hessian negative definite: {int(DF.profile_hessian_neg_definite.sum())} out of {len(DF)}; '
    f'covariance of the fixed effects positive definite: {int(DF.fe_cov_pos_def.sum())} out of '
    f'{len(DF)}')
log("")
log('detail by optimiser (log-likelihood, and * if it converges):')
log(f"{'fit':<15s} " + " ".join(f"{mm:>16s}" for mm, _ in OPTIMISERS)
    + '   retained')
for _, r_ in DF.iterrows():
    cells = []
    for mm, _ in OPTIMISERS:
        v = r_[f"loglike_{mm}"]
        c = "*" if r_[f"converged_{mm}"] else " "
        cells.append(f"{v:15.4f}{c}" if np.isfinite(v) else f"{'—':>16s}")
    log(f"{r_.fit:<15s} " + " ".join(cells) + f'   {r_.optimiser_kept}')
log("")
log('maximum deviations on the printed tests:')
printed = DT[DT.chi2_published.notna() & (DT.kind != 'algebraic check')
              & (DT.kind != 'reparametrisation check')]
log(f'  |delta chi2| max = {printed.delta_chi2.max():.3e}')
log(f'  |delta df|   max = {printed.delta_df.max():.0f}')
log(f'  |delta p|    max = {printed.delta_p.max():.3e}')
for _, r_ in printed.iterrows():
    log(f'  {r_.test:<44s} chi2 {r_.chi2_published:.4f} -> {r_.chi2_reestimated:.4f}   p '
        f'{ps(r_.p_published)} -> {ps(r_.p_reestimated)}')
log("")

with open(os.path.join(OUT, "95_strict_likelihoods.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
print(f"Saved: {os.path.join(OUT, '95_strict_likelihoods.csv')} ({len(WHOLE)} rows)")
print(f"Saved: {os.path.join(OUT, '95_strict_likelihoods.txt')}")
