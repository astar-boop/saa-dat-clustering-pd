"""
91 - The twelve primary non-motor outcomes redone under the STRICT estimation
     criterion: does anything change in the table of the twelve non-motor
     outcomes (then Table 6, now Table 8)?

WHY THIS SCRIPT EXISTS
----------------------
Two optimisation rules for the mixed models coexist in the manuscript.

  RULE A ("first that converges"). It is that of _fit.fit_mixed: lbfgs ->
  powell -> nm -> bfgs are tried on the same model and the FIRST that reports
  converged=True is returned. It is used by 15_nonmotor_trajectories.py (from
  which the twelve primary non-motor outcomes come), 78_longitudinal_sensitivity.py and
  83_spline_reconciled.py.

  RULE B ("maximum likelihood among the converged fits, then Newton
  decrement < 1e-4"). It is that of scripts 84, 87 and 88, from which the
  estimates of the section on gradients come. The complete procedure is
  documented in the docstring of 88_diagnostics_63.py and reproduced here to
  the letter.

Having two rules in circulation is the same discrepancy between Methods and
Results that the reviewer objects to, in a subtler form. This script does not
resolve it by rewriting the text: it MEASURES it. It redoes all twelve primary
models under Rule B and quantifies by how much beta, standard error, p and
BH-corrected p shift with respect to the values published in
outputs/15_primary_nonmotor.csv.

The question that matters is only one: Table 8 reports that NO outcome
survives the correction for multiple comparisons. Does this conclusion change
under the strict criterion?

PROCEDURE REPRODUCED (identical to 88_diagnostics_63.procedura)
------------------------------------------------------------
Stage 1 - cascade. The same model (random intercept and slope per
  participant, REML) is fitted with each of the four optimisers of
  _fit.OTTIMIZZATORI, from the initial values of statsmodels. For each one
  `converged` and the profiled log-likelihood are recorded, the latter
  evaluated on ONE SINGLE model instance so that the four numbers are
  comparable.
Stage 2 - selection. Among the optimisers that converge the one with the
  highest log-likelihood is taken (not the first that converges).
Stage 3 - re-optimisation. From the chosen point all four optimisers restart
  (`start_params`), accepting every gain greater than 1e-10, until none gains
  any more (at most 10 rounds).

STATIONARITY CRITERION: THE NEWTON DECREMENT
--------------------------------------------
    delta = 1/2 * g' (-H)^{-1} g,      threshold: delta < 1e-4
It is the gain in log-likelihood that would be obtained with a full Newton
step from the reported point. g is the analytical score of statsmodels in the
packed space in which the optimiser works; H is computed by central
differences of the same profiled log-likelihood. The detail of the steps and
of the formulas is in the docstring of 88_diagnostics_63.py: here the
primitives are copied without changes so that the two scripts must give the
same number on the same fit.

CORRECTION FOR MULTIPLE COMPARISONS. The corrected p under Rule B is
RECOMPUTED from scratch with Benjamini-Hochberg within the right family — the
twelve primary outcomes, the same family used by 15_nonmotor_trajectories.py —
starting from the p values of Rule B. The published corrected p is not reused.

REML OR ML. All the estimates are REML (the default of statsmodels), like the
coefficients and the Wald tests of Table 8. The log-likelihood comparisons
made here are between two estimates of the SAME model, so they are legitimate
under REML: they measure how much higher one of the two points is than the
other on the same surface.

Output: outputs/91_table6_strict_criterion.csv,
        outputs/91_table6_strict_criterion.txt
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS, fit_mixed  # same cascade, same order

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []

TOL_NEWTON = 1e-4       # primary criterion: Newton decrement, in loglike
TOL_SINGULAR = 1e-8    # variance < TOL_SINGULAR * residual scale
TOL_CORR = 0.999        # |correlation| >= TOL_CORR
TOL_GAIN = 1e-10        # loglike gain below which it stops
MAX_ROUNDS = 10           # maximum rounds of re-optimisation
ALPHA = 0.05            # threshold of the correction, as in Table 8


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# --------------------------------------------------------------- data -------
# Construction IDENTICAL to 15_nonmotor_trajectories.py: same 150h cohort,
# same predictor z(log k_T50), same baseline covariates.
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete].reset_index(drop=True)
d["T50"] = np.log(d.k_T50)
d["T50"] = (d.T50 - d.T50.mean()) / d.T50.std()

covbl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                   "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base = d[["PATNO", "T50"]].merge(covbl, on="PATNO", how="left")

# (variable, domain, direction, education covariate yes/no) — copied from 15.
PRIMARY = [
    ("moca",                 'Global cognitive',      +1, True),
    ("hvlt_immediaterecall", 'Memory (HVLT)',         +1, True),
    ("SDMTOTAL",             'Executive/attention',   +1, True),
    ("VLTANIM",              'Semantic fluency',      +1, True),
    ("bjlot",                'Visuospatial (BJLOT)',  +1, True),
    ("updrs1_score",         'Global non-motor',    -1, False),
    ("gds",                  'Depression (GDS-15)',   -1, False),
    ("stai",                 'Anxiety (STAI)',           -1, False),
    ("ess",                  'Sleepiness (Epworth)',   -1, False),
    ("rem",                  "RBD (RBDSQ)",            -1, False),
    ("scopa",                'Autonomic (SCOPA-AUT)', -1, False),
    ("quip",                 'Impulse control',      -1, False),
]
SECONDARY_VARS = ["COG_COMPOSITE_INT", "TMT_B", "lexical", "MODBNT",
                  "clockdraw", "nqol_cogns", "MSEADLG"]

ALLV = [v for v, _, _, _ in PRIMARY] + SECONDARY_VARS
L = cur[cur.PATNO.isin(base.PATNO)][["PATNO", "YEAR"] + ALLV].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")


# --------------------------------------------------- primitives (from 88) ------
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
            a, b, c, dd = p.copy(), p.copy(), p.copy(), p.copy()
            a[i] += h[i]; a[j] += h[j]
            b[i] += h[i]; b[j] -= h[j]
            c[i] -= h[i]; c[j] += h[j]
            dd[i] -= h[i]; dd[j] -= h[j]
            H[i, j] = H[j, i] = (_ll(md, a) - _ll(md, b) - _ll(md, c)
                                 + _ll(md, dd)) / (4 * h[i] * h[j])
    return (H + H.T) / 2, f0


def estimate(formula, data, groups, method, it, start=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kw = {} if start is None else {"start_params": start}
            return smf.mixedlm(formula, data, groups=groups,
                               re_formula="~years").fit(method=method,
                                                        maxiter=it, **kw)
        except Exception:
            return None


def procedure(formula, data, groups):
    """Stages 1-3 of Rule B. Identical to 88_diagnostics_63.procedura."""
    # --- stage 1: cascade from the initial values of statsmodels ---------------
    attempts = {}
    md_ref = None
    for method, it in OPTIMISERS:
        r = estimate(formula, data, groups, method, it)
        if r is None:
            attempts[method] = (None, np.nan, False)
            continue
        if md_ref is None:
            md_ref = r.model
        p = pack(r)
        attempts[method] = (r, _ll(md_ref, p), bool(r.converged))
    if md_ref is None:
        return None
    # --- stage 2: among those that converge, the highest likelihood --
    conv = [(m, v[1]) for m, v in attempts.items()
            if v[0] is not None and v[2] and np.isfinite(v[1])]
    if conv:
        chosen = max(conv, key=lambda x: x[1])[0]
    else:
        fin = [(m, v[1]) for m, v in attempts.items()
               if v[0] is not None and np.isfinite(v[1])]
        if not fin:
            return None
        chosen = max(fin, key=lambda x: x[1])[0]
    r0, ll0, _ = attempts[chosen]
    # --- stage 3: re-optimisation from the chosen point ------------------------
    best = (r0, '(none)', ll0, pack(r0))
    rounds = 0
    for cycle in range(MAX_ROUNDS):
        improved = False
        for method, it in OPTIMISERS:
            r = estimate(formula, data, groups, method, it, start=best[3])
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
    return {'r_cascade': r0, 'opt_chosen': chosen, 'll_choice': ll0,
            'attempts': attempts, 'r_final': best[0],
            'opt_reopt': best[1], 'rounds': rounds, 'll_final': best[2],
            'md_ref': md_ref}


def diagnostics(r, t, md_ref=None):
    """Gradient, Newton decrement and structural diagnostics at a point.

    `md_ref` makes it possible to evaluate the log-likelihood always on the
    SAME model instance, so that the values of Rule A and of Rule B are
    comparable bit for bit."""
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
    return {
        "grad_inf_norm": float(np.max(np.abs(g))),
        "grad_check_analytic_vs_central": float(np.max(np.abs(g - g_num))),
        "newton_decrement": newton,
        "loglike": _ll(md_ref, p) if md_ref is not None else ll,
        "profile_hessian_neg_definite": bool(eig.max() < 0),
        "re_var_intercept": v_int,
        "re_var_slope": v_slope,
        "re_corr_intercept_slope": corr,
        "residual_scale": scale,
        "singular": sing,
        "boundary": edge,
        "beta": float(r.params[t]),
        "se": float(r.bse[t]),
        "p": float(r.pvalues[t]),
        "converged": bool(r.converged),
    }


def term(r, var):
    """Name of the interaction term, in whichever order patsy writes it."""
    for cand in ("T50:years", "years:T50"):
        if cand in r.params.index:
            return cand
    return None


# ----------------------------------------------------------- the body -------
log("=" * 78)
log('91 — THE TWELVE PRIMARY NON-MOTOR OUTCOMES UNDER THE STRICT CRITERION')
log("=" * 78)
log(f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}")
log(f'BASE = {BASE}')
log(f'cascade: {OPTIMISERS}')
log('Rule A: first optimiser that converges (_fit.fit_mixed, script 15)')
log(f"Rule B: among the converged ones the highest likelihood, then re-optimisation and Newton "
    f"decrement < {TOL_NEWTON:g} (script 84/87/88)")
log('REML estimation; BH recomputed over the twelve primary outcomes from the p values of Rule B')
log("")

pub = pd.read_csv(os.path.join(OUT, "15_primary_nonmotor.csv"))
log(f"published values read from {os.path.join(OUT, '15_primary_nonmotor.csv')} ({len(pub)} rows)")
log("")

rows = []
for var, dom, direction, educ in PRIMARY:
    cov = ["age_bl", "SEX", "dur_bl"] + (["EDUCYRS"] if educ else [])
    m = L[["PATNO", "years", var, "T50"] + cov].dropna()
    cnt = m.groupby("PATNO").size()
    if len(m) < 200 or (cnt >= 2).sum() < 60:
        log(f'  [skipped] {var}: insufficient coverage')
        continue
    f = f"{var} ~ T50*years + " + " + ".join(cov)
    t0 = time.time()

    # --- Rule A: exactly the call of script 15 ------------------
    rA, optA, fallbacka = fit_mixed(f, m, m.PATNO)
    if rA is None:
        log(f'  [skipped] {var}: no fit under Rule A')
        continue
    tA = term(rA, var)

    # --- Rule B -----------------------------------------------------------
    P = procedure(f, m, m.PATNO)
    if P is None:
        log(f'  [skipped] {var}: no fit under Rule B')
        continue
    rB = P['r_final']
    tB = term(rB, var)
    md_ref = P['md_ref']

    FR = diagnostics(rA, tA, md_ref)
    DB = diagnostics(rB, tB, md_ref)

    row = {
        "domain": dom, "outcome": var,
        "n_subjects": int(m.PATNO.nunique()), "n_obs": int(len(m)),
        # --- published values (Rule A, from 15_primary_nonmotor.csv) -------
        "beta_published": np.nan, "se_published": np.nan,
        "p_published": np.nan, "p_BH_published": np.nan,
        # --- Rule A redone here, to verify reproducibility -------
        "optimiser_ruleA": optA,
        "ruleA_fallback_intercept_only": bool(fallbacka),
        "beta_ruleA": FR["beta"], "se_ruleA": FR["se"], "p_ruleA": FR["p"],
        "loglike_ruleA": FR["loglike"],
        "grad_inf_norm_ruleA": FR["grad_inf_norm"],
        "newton_decrement_ruleA": FR["newton_decrement"],
        "meets_newton_ruleA": bool(np.isfinite(FR["newton_decrement"])
                                   and FR["newton_decrement"] < TOL_NEWTON),
        "converged_flag_ruleA": FR["converged"],
        "singular_ruleA": FR["singular"], "boundary_ruleA": FR["boundary"],
        # --- Rule B --------------------------------------------------------
        "optimiser_ruleB_chosen": P['opt_chosen'],
        "optimiser_ruleB_reoptimisation": P['opt_reopt'],
        "reoptimisation_rounds": P['rounds'],
        "beta_ruleB": DB["beta"], "se_ruleB": DB["se"], "p_ruleB": DB["p"],
        "loglike_ruleB": DB["loglike"],
        "grad_inf_norm_ruleB": DB["grad_inf_norm"],
        "newton_decrement_ruleB": DB["newton_decrement"],
        "meets_newton_ruleB": bool(np.isfinite(DB["newton_decrement"])
                                   and DB["newton_decrement"] < TOL_NEWTON),
        "converged_flag_ruleB": DB["converged"],
        "singular_ruleB": DB["singular"], "boundary_ruleB": DB["boundary"],
        "grad_check_analytic_vs_central_ruleB":
            DB["grad_check_analytic_vs_central"],
        "profile_hessian_neg_definite_ruleB":
            DB["profile_hessian_neg_definite"],
        "re_var_intercept_ruleB": DB["re_var_intercept"],
        "re_var_slope_ruleB": DB["re_var_slope"],
        "re_corr_intercept_slope_ruleB": DB["re_corr_intercept_slope"],
        # --- likelihood gain -------------------------------------
        "loglike_gain_ruleB_minus_ruleA": DB["loglike"] - FR["loglike"],
    }
    # convergence and likelihood of EACH optimiser of the cascade:
    # it is needed to say whether Rule A ever "fell beyond" lbfgs.
    for method, _ in OPTIMISERS:
        r_, ll_, cv_ = P['attempts'].get(method, (None, np.nan, False))
        row[f'loglike_{method}'] = ll_
        row[f'converged_{method}'] = cv_
    rows.append(row)
    print(f"  . {var:22s} A={optA:6s} B={P['opt_chosen']:6s}  |g| {FR['grad_inf_norm']:.3g} -> "
          f"{DB['grad_inf_norm']:.3g}  newton {FR['newton_decrement']:.3g} -> "
          f"{DB['newton_decrement']:.3g}  ({time.time() - t0:.0f}s)",
          flush=True)

D = pd.DataFrame(rows)

# --- the published values, paired by outcome --------------------------------
mp = pub.set_index('outcome')
for i, r in D.iterrows():
    if r["outcome"] in mp.index:
        D.loc[i, "beta_published"] = float(mp.loc[r["outcome"], "beta"])
        D.loc[i, "se_published"] = float(mp.loc[r["outcome"], "se"])
        D.loc[i, "p_published"] = float(mp.loc[r["outcome"], "p"])
        D.loc[i, "p_BH_published"] = float(mp.loc[r["outcome"], "p_BH"])

# --- BH RECOMPUTED within the family of the twelve primary outcomes, from the Rule B p values ---
D["p_BH_ruleB"] = multipletests(D.p_ruleB, method="fdr_bh")[1]
# and, as an internal check, BH recomputed from the p values of Rule A redone:
# if it coincides with the published p_BH, the reproduction of Rule A is exact.
D["p_BH_ruleA_recomputed"] = multipletests(D.p_ruleA, method="fdr_bh")[1]


def variations(df, name, col_pub, col_b):
    df[f'abs_change_{name}'] = (df[col_b] - df[col_pub]).abs()
    df[f'rel_change_{name}'] = np.where(
        df[col_pub].abs() > 0,
        (df[col_b] - df[col_pub]).abs() / df[col_pub].abs(), np.nan)


variations(D, "beta", "beta_published", "beta_ruleB")
variations(D, "se", "se_published", "se_ruleB")
variations(D, "p", "p_published", "p_ruleB")
variations(D, "p_BH", "p_BH_published", "p_BH_ruleB")

D["survives_BH_published"] = D.p_BH_published < ALPHA
D["survives_BH_ruleB"] = D.p_BH_ruleB < ALPHA
D["status_change_BH"] = D.survives_BH_published != D.survives_BH_ruleB
D["nominal_sig_published"] = D.p_published < ALPHA
D["nominal_sig_ruleB"] = D.p_ruleB < ALPHA
D["status_change_nominal"] = D.nominal_sig_published != D.nominal_sig_ruleB

COLS = ["domain", "outcome", "n_subjects", "n_obs",
        "beta_published", "se_published", "p_published", "p_BH_published",
        "optimiser_ruleA", "ruleA_fallback_intercept_only",
        "beta_ruleA", "se_ruleA", "p_ruleA", "p_BH_ruleA_recomputed",
        "loglike_ruleA", "grad_inf_norm_ruleA", "newton_decrement_ruleA",
        "meets_newton_ruleA", "converged_flag_ruleA",
        "singular_ruleA", "boundary_ruleA",
        "optimiser_ruleB_chosen", "optimiser_ruleB_reoptimisation",
        "reoptimisation_rounds",
        "beta_ruleB", "se_ruleB", "p_ruleB", "p_BH_ruleB",
        "loglike_ruleB", "grad_inf_norm_ruleB", "newton_decrement_ruleB",
        "meets_newton_ruleB", "converged_flag_ruleB",
        "singular_ruleB", "boundary_ruleB",
        "grad_check_analytic_vs_central_ruleB",
        "profile_hessian_neg_definite_ruleB",
        "re_var_intercept_ruleB", "re_var_slope_ruleB",
        "re_corr_intercept_slope_ruleB",
        "loglike_gain_ruleB_minus_ruleA"] \
    + [f"loglike_{m}" for m, _ in OPTIMISERS] \
    + [f"converged_{m}" for m, _ in OPTIMISERS] \
    + ["abs_change_beta", "rel_change_beta", "abs_change_se", "rel_change_se",
       "abs_change_p", "rel_change_p", "abs_change_p_BH", "rel_change_p_BH",
       "survives_BH_published", "survives_BH_ruleB", "status_change_BH",
       "nominal_sig_published", "nominal_sig_ruleB", "status_change_nominal"]
D = D[COLS]
D.to_csv(os.path.join(OUT, "91_table6_strict_criterion.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)

log(f'models refitted: {len(D)} of 12')
log("")
log("-" * 78)
log('REPRODUCTION OF THE PUBLISHED VALUES UNDER RULE A')
log("-" * 78)
log("(if these discrepancies are not ~0, the comparison that follows is not clean)")
log(f'  max |refitted beta - published beta| : {(D.beta_ruleA - D.beta_published).abs().max():.3e}')
log(f'  max |refitted SE - published SE|     : {(D.se_ruleA - D.se_published).abs().max():.3e}')
log(f'  max |refitted p - published p|       : {(D.p_ruleA - D.p_published).abs().max():.3e}')
log(f'  max |recomputed BH - published BH|   : '
    f'{(D.p_BH_ruleA_recomputed - D.p_BH_published).abs().max():.3e}')
log("")
log("-" * 78)
log('WHICH OPTIMISER AND WHEN')
log("-" * 78)
log(f'Rule A, optimiser returned: {D.optimiser_ruleA.value_counts().to_dict()}')
log(f'Rule B, optimiser chosen  : {D.optimiser_ruleB_chosen.value_counts().to_dict()}')
for method, _ in OPTIMISERS:
    log(f"  converges in {int(D[f'converged_{method}'].sum())}/{len(D)} fits: {method}")
failed = D[~D.converged_lbfgs]
log(f'outcomes in which lbfgs does NOT converge (Rule A had to go down the cascade): {len(failed)}')
if len(failed):
    log(failed[["outcome", "optimiser_ruleA", "optimiser_ruleB_chosen",
                 "loglike_lbfgs", "loglike_ruleA", "loglike_ruleB"]]
        .to_string(index=False))
log(f'fits in which the re-optimisation of Rule B gains likelihood: '
    f'{int((D.loglike_gain_ruleB_minus_ruleA > 1e-06).sum())}/{len(D)}')
log(f'log-likelihood gain Rule B - Rule A: min {D.loglike_gain_ruleB_minus_ruleA.min():.3e}  max '
    f'{D.loglike_gain_ruleB_minus_ruleA.max():.3e}')
log("")
log("-" * 78)
log("STATIONARITY UNDER THE TWO RULES")
log("-" * 78)
log(f'Newton decrement, Rule A — min {D.newton_decrement_ruleA.min():.3e}  median '
    f'{D.newton_decrement_ruleA.median():.3e}  max {D.newton_decrement_ruleA.max():.3e}  (meets < '
    f'{TOL_NEWTON:g}: {int(D.meets_newton_ruleA.sum())}/{len(D)})')
log(f'Newton decrement, Rule B — min {D.newton_decrement_ruleB.min():.3e}  median '
    f'{D.newton_decrement_ruleB.median():.3e}  max {D.newton_decrement_ruleB.max():.3e}  (meets < '
    f'{TOL_NEWTON:g}: {int(D.meets_newton_ruleB.sum())}/{len(D)})')
log(f'infinity norm of the gradient, Rule A — max {D.grad_inf_norm_ruleA.max():.3e}')
log(f'infinity norm of the gradient, Rule B — max {D.grad_inf_norm_ruleB.max():.3e}')
log(f'maximum discrepancy analytic score vs central differences (Rule B): '
    f'{D.grad_check_analytic_vs_central_ruleB.max():.3e}')
log(f'singular fits: A {int(D.singular_ruleA.sum())}, B {int(D.singular_ruleB.sum())} of {len(D)}')
log(f'fits at the boundary (variance exactly 0): A {int(D.boundary_ruleA.sum())}, B '
    f'{int(D.boundary_ruleB.sum())} of {len(D)}')
log("")
log('Newton decrement of each of the twelve under Rule B:')
log(D[["outcome", "newton_decrement_ruleB", "meets_newton_ruleB",
       "grad_inf_norm_ruleB"]].to_string(index=False))
log("")
log("-" * 78)
log('HOW MUCH THE ESTIMATES MOVE')
log("-" * 78)
log(D[["outcome", "beta_published", "beta_ruleB", "abs_change_beta",
       "se_published", "se_ruleB", "abs_change_se",
       "p_published", "p_ruleB", "abs_change_p",
       "p_BH_published", "p_BH_ruleB", "abs_change_p_BH"]]
    .to_string(index=False))
log("")
for name, label in [("beta", "beta"), ("se", 'standard error'),
                        ("p", "p"), ("p_BH", 'BH-corrected p')]:
    i = D[f'abs_change_{name}'].idxmax()
    log(f"maximum absolute change in {label}: {D.loc[i, f'abs_change_{name}']:.3e}  (outcome "
        f"{D.loc[i, 'outcome']}; relative change {D.loc[i, f'rel_change_{name}']:.3e})")
    j = D[f'rel_change_{name}'].idxmax()
    log(f"  maximum relative change: {D.loc[j, f'rel_change_{name}']:.3e} (outcome "
        f"{D.loc[j, 'outcome']})")
log("")
log("-" * 78)
log(f'DOES THE STATUS CHANGE WITH RESPECT TO THE CORRECTION THRESHOLD (BH < {ALPHA})?')
log("-" * 78)
log(f'outcomes that survive BH in the published values: {int(D.survives_BH_published.sum())} of '
    f'{len(D)}')
log(f'outcomes that survive BH under Rule B           : {int(D.survives_BH_ruleB.sum())} of {len(D)}')
log(f'outcomes that CHANGE status: {int(D.status_change_BH.sum())}')
if int(D.status_change_BH.sum()):
    log(D[D.status_change_BH][["outcome", "p_published", "p_ruleB",
                               "p_BH_published", "p_BH_ruleB"]]
        .to_string(index=False))
# The number quoted in the parenthesis must be the minimum itself, not its
# distance from the threshold: the old form printed `minimo - ALPHA` next to the
# word "resta", and it read as a minimum of 0.119921 in contradiction with
# the value just printed and with the column p_BH_ruleB of the CSV. The margin is
# now declared for what it is, and the orders of magnitude are computed
# instead of hard-coded.
minimum_bh = D.p_BH_ruleB.min()
margin_bh = minimum_bh - ALPHA
shift_bh = D.abs_change_p_BH.max()
orders_bh = np.log10(margin_bh / shift_bh)
log(f'minimum BH-adjusted p — published {D.p_BH_published.min():.6f}, Rule B {minimum_bh:.6f}  (the '
    f'lowest, {minimum_bh:.6f}, stays ABOVE the threshold of {ALPHA}, with a margin of '
    f'{margin_bh:.6f}; the maximum shift of the BH is {shift_bh:.3e}, that is {orders_bh:.1f} orders '
    f'of magnitude less than the margin)')
log(f'nominally significant outcomes (p < {ALPHA}): published {int(D.nominal_sig_published.sum())}, '
    f'Rule B {int(D.nominal_sig_ruleB.sum())}, changes of status '
    f'{int(D.status_change_nominal.sum())}')
if int(D.status_change_nominal.sum()):
    log(D[D.status_change_nominal][["outcome", "p_published", "p_ruleB"]]
        .to_string(index=False))
log("")
log("-" * 78)
log('CONCLUSION')
log("-" * 78)
if int(D.status_change_BH.sum()) == 0:
    log('None of the twelve outcomes changes status with respect to the correction threshold. The '
        'maximum changes observed are:')
    log(f'  beta          <= {D.abs_change_beta.max():.3e} absolute, '
        f'{100 * D.rel_change_beta.max():.4f}% relative')
    log(f'  std error     <= {D.abs_change_se.max():.3e} absolute, '
        f'{100 * D.rel_change_se.max():.4f}% relative')
    log(f'  p             <= {D.abs_change_p.max():.3e} absolute')
    log(f'  BH-adjusted p <= {D.abs_change_p_BH.max():.3e} absolute')
else:
    log('WARNING: at least one outcome changes status with respect to the correction threshold. The '
        'corrected values are in the table above.')
log("")

with open(os.path.join(OUT, "91_table6_strict_criterion.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
log(f"Saved: {os.path.join(OUT, '91_table6_strict_criterion.csv')}")
log(f"Saved: {os.path.join(OUT, '91_table6_strict_criterion.txt')}")
