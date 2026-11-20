"""
88 - Reference diagnostics of the 63 kinetic parameter x time mixed models, under
     the estimation procedure ACTUALLY used for the numbers of §3.12 and Table 7.

WHY THIS SCRIPT EXISTS
----------------------
§2.18 of the manuscript still described the old rule ("one optimiser after
another is tried and the FIRST that converges is kept", _fit.fit_mixed) and
reported four gradient norms between 63 and 98. The numbers published in §3.12
and in Table 7 come instead from the re-optimised procedure of scripts 84 and
87. This script redoes all 63 fits under THAT procedure and produces a
single result object that can be consulted in review, with one row per fit.

PROCEDURE REPRODUCED HERE (the same one declared in §2.18)
---------------------------------------------------------
Stage 1 - cascade. The same model (random intercept and slope per
  participant, REML) is fitted with each of the four optimisers of
  _fit.OTTIMIZZATORI - lbfgs (300 iterations), powell (2000), nelder-mead
  (5000), bfgs (2000) - starting from the statsmodels initial values. For
  each one the `converged` flag and the profile log-likelihood are recorded,
  the latter evaluated on ONE SINGLE model instance so that the four numbers
  are comparable bit for bit.
Stage 2 - selection. Among the optimisers that converge, the one that
  reaches the highest log-likelihood is chosen (not the first that converges).
Stage 3 - re-optimisation and verification. All four optimisers are restarted
  from the chosen point (`start_params`), accepting every gain in
  log-likelihood greater than 1e-10, and this is iterated until none gains any
  more (at most 10 rounds). The final point must satisfy the stationarity
  criterion described below.

STATIONARITY CRITERION: THE NEWTON DECREMENT
--------------------------------------------
    delta = 1/2 * g' (-H)^{-1} g,      threshold: delta < 1e-4
It is the gain in log-likelihood that would be obtained with a full Newton
step starting from the reported point: if it is 1e-4, the point is one
ten-thousandth of a log-likelihood unit away from the local maximum (for
comparison, a 95% profile-likelihood confidence limit is 1.92 units away).
Unlike the gradient norm it is invariant both to the scale of the outcome and
to reparametrisation.

How g and H are computed, exactly:
  - the parameter space is the one in which the optimiser works, that is, the
    packed vector `params_object.get_packed(use_sqrt=md.use_sqrt,
    has_fe=False)`: MixedLM profiles out the fixed effects (GLS) and the
    residual variance, and the free parameters are the elements of the
    Cholesky factor of cov_re/scale;
  - g is the ANALYTICAL SCORE exposed by statsmodels, `md.score(p)`, which is
    the gradient of the profile log-likelihood with respect to that vector.
    No finite difference is therefore needed for the gradient. The
    grad_check column nevertheless reports the maximum discrepancy between
    the analytical score and the central-difference gradient with step
    h_i = 1e-5*max(1,|p_i|), as a check;
  - H is computed BY CENTRAL DIFFERENCES of the same profile log-likelihood,
    because statsmodels does not expose a Hessian in this space
    (md.hessian is in the untransformed space and includes the fixed effects:
    it serves for the standard errors, not for judging the optimum). Step
    h_i = 1e-4*max(1,|p_i|); diagonal
        H_ii = (f(p+h_i e_i) - 2 f(p) + f(p-h_i e_i)) / h_i^2
    and off-diagonal
        H_ij = (f(p+h_i e_i+h_j e_j) - f(p+h_i e_i-h_j e_j)
                - f(p-h_i e_i+h_j e_j) + f(p-h_i e_i-h_j e_j)) / (4 h_i h_j),
    then symmetrised as (H + H')/2;
  - the Newton step for MAXIMISATION is d = -H^{-1} g, solved with
    np.linalg.solve(H, -g); the expected gain is 0.5 * g'd, which coincides
    with 1/2 g'(-H)^{-1} g and is non-negative when H is negative definite.

REML OR ML. All the estimates here are REML (the statsmodels default
setting): the coefficients and the Wald tests of the manuscript come from REML
fits. The likelihood ratios of §3.13 and of Table 9 are instead
computed on full maximum likelihood fits (reml=False), in scripts 78
and 83, because under REML the likelihoods of models with different fixed
effects are not comparable.

CROSS-CHECKS. At the end the script compares row by row the estimates
obtained with the pol_* columns of outputs/84_gradients.csv (from which the
numbers of §3.12 come) and with the M0 rows of outputs/87_amplitude_ladder.csv
(from which the first column of Table 7 comes).

Output: outputs/88_diagnostics_63.csv, outputs/88_diagnostics_63.txt
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
from _fit import OPTIMISERS  # same cascade, same order, same maxiter values

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []

TOL_NEWTON = 1e-4
TOL_SINGULAR = 1e-8
TOL_CORR = 0.999
TOL_GAIN = 1e-10
MAX_ROUNDS = 10


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))

OUTCOMES = {
    "updrs3_score": ("MDS-UPDRS III", ["age_bl", "SEX", "dur_bl"]),
    "updrs_totscore": ("MDS-UPDRS total", ["age_bl", "SEX", "dur_bl"]),
    "updrs2_score": ("MDS-UPDRS II", ["age_bl", "SEX", "dur_bl"]),
    "moca": ("MoCA", ["age_bl", "SEX", "dur_bl", "EDUCYRS"]),
    "MIA_CAUDATE_BILAT": ("DAT caudate, bilateral", ["age_bl", "SEX", "dur_bl"]),
    "MIA_PUTAMEN_BILAT": ("DAT putamen, bilateral", ["age_bl", "SEX", "dur_bl"]),
    "scopa": ("SCOPA-AUT", ["age_bl", "SEX", "dur_bl"]),
}
KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}


def build(cohort_file, kin_map):
    """Identical to 72_longitudinal_refit.build, 80_convergence.build and
    84_gradients.build."""
    d = pd.read_csv(os.path.join(OUT, cohort_file))
    if "complete" in d.columns:
        d = d[d.complete]
    elif "complete_primary" in d.columns:
        d = d[d.complete_primary]
    base = d[["PATNO"] + list(kin_map)].rename(columns=kin_map)
    for k in kin_map.values():
        base[k] = np.log(base[k])
        base[k] = (base[k] - base[k].mean()) / base[k].std()
    bl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                    "duration_yrs"]].rename(
        columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
    base = base.merge(bl, on="PATNO", how="left")
    L = cur[cur.PATNO.isin(base.PATNO)][
        ["PATNO", "EVENT_ID", "YEAR"] + list(OUTCOMES)].copy()
    L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
    L = L[L.years.notna()]
    return L.merge(base, on="PATNO", how="left")


# ------------------------------------------------------------ primitives ----
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
    """Central-difference Hessian of the profile log-likelihood,
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
    """Stages 1-3 of the procedure documented in the docstring.

    Returns (result, chosen optimiser, chosen loglike, per-optimiser
    dictionary, optimiser of the re-optimisation, rounds, final
    loglike)."""
    # --- stage 1: cascade from the statsmodels initial values ---------------
    attempts = {}
    md_ref = None
    ordering = []
    for method, it in OPTIMISERS:
        r = estimate(formula, data, groups, method, it)
        if r is None:
            attempts[method] = (None, np.nan, False)
            continue
        if md_ref is None:
            md_ref = r.model
        p = pack(r)
        attempts[method] = (r, _ll(md_ref, p), bool(r.converged))
        ordering.append(method)
    if md_ref is None:
        return None
    # --- stage 2: among those that converge, the highest likelihood ---------
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
    # --- stage 3: re-optimisation from the chosen point ---------------------
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


def diagnostics(r, t):
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
        "beta_interaction": float(r.params[t]),
        "se": float(r.bse[t]),
        "p": float(r.pvalues[t]),
        "converged": bool(r.converged),
    }


# ----------------------------------------------------------------- body -----
def analyse(L, kins, tag):
    rows = []
    for out, (label, cov) in OUTCOMES.items():
        for k in kins:
            m = L[["PATNO", "years", out, k] + cov].dropna()
            multi = m.groupby("PATNO").size()
            if len(m) < 120 or int((multi >= 2).sum()) < 40:
                continue
            f = f"{out} ~ {k}*years + " + " + ".join(cov)
            t0 = time.time()
            P = procedure(f, m, m.PATNO)
            if P is None:
                log(f'  [{tag}] {out} x {k}: no fit')
                continue
            rc, rf = P['r_cascade'], P['r_final']
            tc = (f"{k}:years" if f"{k}:years" in rc.params.index
                  else f"years:{k}")
            tf = (f"{k}:years" if f"{k}:years" in rf.params.index
                  else f"years:{k}")
            Dc = diagnostics(rc, tc)
            Df = diagnostics(rf, tf)
            row = {
                "cohort": tag, "outcome": out, "label": label, "kinetic": k,
                "n_subjects": int(m.PATNO.nunique()), "n_obs": int(len(m)),
                "optimiser_chosen": P['opt_chosen'],
                "optimiser_reoptimisation": P['opt_reopt'],
                "reoptimisation_rounds": P['rounds'],
                "loglike_chosen": P['ll_choice'],
                "loglike_final": P['ll_final'],
                "delta_loglike_reoptimisation": P['ll_final'] - P['ll_choice'],
            }
            for method, _ in OPTIMISERS:
                r_, ll_, cv_ = P['attempts'].get(method, (None, np.nan, False))
                row[f'loglike_{method}'] = ll_
                row[f'converged_{method}'] = cv_
            for key, val in Df.items():
                row[key] = val
            row["meets_newton_criterion"] = bool(
                np.isfinite(Df["newton_decrement"])
                and Df["newton_decrement"] < TOL_NEWTON)
            # state before the re-optimisation, for the comparison
            row["grad_inf_norm_before"] = Dc["grad_inf_norm"]
            row["newton_decrement_before"] = Dc["newton_decrement"]
            row["beta_before"] = Dc["beta_interaction"]
            row["se_before"] = Dc["se"]
            row["p_before"] = Dc["p"]
            rows.append(row)
            print(f"  . {tag} {out} x {k}  opt={P['opt_chosen']}  |g| {Dc['grad_inf_norm']:.3g} -> "
                  f"{Df['grad_inf_norm']:.3g}  newton {Df['newton_decrement']:.3g}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


log("=" * 78)
log('88 — DIAGNOSTICS OF THE 63 MIXED MODELS UNDER THE PROCEDURE ACTUALLY')
log('     USED FOR §3.12 AND TABLE 7')
log("=" * 78)
log(f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}")
log(f'BASE = {BASE}')
log(f'cascade: {OPTIMISERS}')
log('stage 1: all the optimisers from the starting values of statsmodels')
log("stage 2: among those that CONVERGE, the highest log-likelihood")
log(f'stage 3: re-optimisation from the chosen point (max {MAX_ROUNDS} rounds), criterion Newton '
    f'decrement < {TOL_NEWTON:g}')
log('REML estimation (statsmodels default)')
log("")

L150 = build("08_dataset_150h.csv", KIN150)
L24 = build("analysis_complete_cases.csv", KIN24)
d150 = analyse(L150, list(KIN150.values()), "150h")
d24 = analyse(L24, list(KIN24.values()), "24h")
D = pd.concat([d150, d24], ignore_index=True)

D["p_BH_within_cohort"] = np.nan
for cohort in D.cohort.unique():
    i = D.index[D.cohort == cohort]
    D.loc[i, "p_BH_within_cohort"] = multipletests(D.loc[i, "p"],
                                                   method="fdr_bh")[1]

COLS = ["cohort", "outcome", "label", "kinetic", "n_subjects", "n_obs",
        "optimiser_chosen"] \
    + [f"loglike_{m}" for m, _ in OPTIMISERS] \
    + [f"converged_{m}" for m, _ in OPTIMISERS] \
    + ["loglike_chosen", "optimiser_reoptimisation", "reoptimisation_rounds",
       "loglike_final", "delta_loglike_reoptimisation",
       "grad_inf_norm", "grad_check_analytic_vs_central", "newton_decrement",
       "meets_newton_criterion", "singular", "boundary",
       "re_var_intercept", "re_var_slope", "re_corr_intercept_slope",
       "residual_scale", "profile_hessian_neg_definite",
       "profile_hessian_max_eigenvalue", "fe_cov_pos_def", "fe_cov_cond",
       "beta_interaction", "se", "p", "p_BH_within_cohort", "converged",
       "grad_inf_norm_before", "newton_decrement_before",
       "beta_before", "se_before", "p_before"]
D = D[COLS]
D.to_csv(os.path.join(OUT, "88_diagnostics_63.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

log(f'models fitted: {len(D)}  (150h {len(d150)}, 24h {len(d24)})')
log("")
log("-" * 78)
log("SELECTION OF THE OPTIMISER")
log("-" * 78)
log(f'optimiser chosen (maximum likelihood among the converged ones): '
    f'{D.optimiser_chosen.value_counts().to_dict()}')
for method, _ in OPTIMISERS:
    log(f"  converges in {int(D[f'converged_{method}'].sum())}/{len(D)} fits: {method}")
log(f'fits in which the re-optimisation gains likelihood: '
    f'{int((D.delta_loglike_reoptimisation > 1e-06).sum())}/{len(D)} (maximum gain '
    f'{D.delta_loglike_reoptimisation.max():.6g})')
log("")
log("-" * 78)
log("STATIONARITY")
log("-" * 78)
log(f'Newton decrement — min {D.newton_decrement.min():.3e}   median '
    f'{D.newton_decrement.median():.3e}   max {D.newton_decrement.max():.3e}')
log(f'fits meeting the criterion (< {TOL_NEWTON:g}): {int(D.meets_newton_criterion.sum())} out of '
    f'{len(D)}')
log(f'infinity norm of the gradient — min {D.grad_inf_norm.min():.3e}   median '
    f'{D.grad_inf_norm.median():.3e}   max {D.grad_inf_norm.max():.3e}')
log(f'maximum discrepancy between analytic score and central differences: '
    f'{D.grad_check_analytic_vs_central.max():.3e}')
log("")
log("-" * 78)
log('STRUCTURAL DIAGNOSTICS')
log("-" * 78)
log(f'singular fits: {int(D.singular.sum())} out of {len(D)}')
log(f'fits on the boundary (variance exactly 0): {int(D.boundary.sum())} out of {len(D)}')
log(f'profiled Hessian negative definite: {int(D.profile_hessian_neg_definite.sum())} out of '
    f'{len(D)}')
log(f'covariance of the fixed effects positive definite: {int(D.fe_cov_pos_def.sum())} out of '
    f'{len(D)}')
log(f'condition number of the covariance of the fixed effects: min {D.fe_cov_cond.min():.4g}  median '
    f'{D.fe_cov_cond.median():.4g}  max {D.fe_cov_cond.max():.4g}')
log(f"variance of the random intercept: min {D.re_var_intercept.min():.6g}  max "
    f"{D.re_var_intercept.max():.6g}")
log(f'variance of the random slope    : min {D.re_var_slope.min():.6g}  max '
    f'{D.re_var_slope.max():.6g}')
log(f'|correlation| between the two random effects: min {D.re_corr_intercept_slope.abs().min():.4f}  '
    f'max {D.re_corr_intercept_slope.abs().max():.4f}')
log("")
log("-" * 78)
log("THE FITS THAT STARTED FAR FROM THE OPTIMUM")
log("-" * 78)
gr = D[D.grad_inf_norm_before > 1].sort_values("grad_inf_norm_before",
                                               ascending=False)
log(f'fits with gradient norm > 1 before the re-optimisation: {len(gr)}')
log(gr[["cohort", "label", "kinetic", "n_subjects", "n_obs",
        "optimiser_chosen", "grad_inf_norm_before", "grad_inf_norm",
        "newton_decrement_before", "newton_decrement",
        "beta_before", "beta_interaction", "p_before", "p"]].to_string(
    index=False))
log("")

# ------------------------------------------------------- cross-checks
log("=" * 78)
log('CROSS-CHECK AGAINST THE OUTPUTS FROM WHICH THE PUBLISHED NUMBERS COME')
log("=" * 78)
f84 = os.path.join(OUT, "84_gradients.csv")
if os.path.exists(f84):
    T = pd.read_csv(f84)
    c = D.merge(T[["cohort", "outcome", "kinetic", "pol_beta", "pol_se",
                   "pol_p", "pol_grad_inf_norm", "pol_newton_decrement",
                   "pol_p_BH_within_cohort"]],
                on=["cohort", "outcome", "kinetic"], how="inner")
    log(f'rows matched with 84_gradients.csv (columns pol_*, from which §3.12 comes): {len(c)} out '
        f'of {len(D)}')
    log(f'  max |delta beta| : {(c.beta_interaction - c.pol_beta).abs().max():.3e}')
    log(f'  max |delta SE|   : {(c.se - c.pol_se).abs().max():.3e}')
    log(f'  max |delta p|    : {(c.p - c.pol_p).abs().max():.3e}')
    log(f'  max |delta BH|   : {(c.p_BH_within_cohort - c.pol_p_BH_within_cohort).abs().max():.3e}')
else:
    log(f'WARNING: {f84} does not exist')
log("")
f87 = os.path.join(OUT, "87_amplitude_ladder.csv")
if os.path.exists(f87):
    S = pd.read_csv(f87)
    S = S[S.spec == "M0"].rename(columns={'outcome': "outcome",
                                          'kinetic': "kinetic"})
    c2 = D[D.cohort == "24h"].merge(
        S[["outcome", "kinetic", "beta", "se", "p", 'optimiser']],
        on=["outcome", "kinetic"], how="inner")
    log(f'rows matched with the M0 rows of 87_amplitude_ladder.csv (first column of Table 7): '
        f'{len(c2)}')
    # the merge renames the columns with the same name: *_x from here, *_y from 87
    log(c2[["label", "kinetic", "optimiser_chosen", 'optimiser',
            "beta_interaction", "beta", "se_x", "se_y",
            "p_x", "p_y"]].to_string(index=False))
    log(f'  max |delta beta| : {(c2.beta_interaction - c2.beta).abs().max():.3e}')
    log(f'  max |delta SE|   : {(c2.se_x - c2.se_y).abs().max():.3e}')
    log(f'  max |delta p|    : {(c2.p_x - c2.p_y).abs().max():.3e}')
else:
    log(f'WARNING: {f87} does not exist')
log("")
log("-" * 78)
log('RESULTS THAT PASS BH < 0.05 WITHIN COHORT')
log("-" * 78)
for cohort in ["24h", "150h"]:
    s = D[D.cohort == cohort]
    sig = s[s.p_BH_within_cohort < 0.05]
    log(f'cohort {cohort}: {len(sig)} out of {len(s)} tests')
    if len(sig):
        log(sig[["label", "kinetic", "n_subjects", "n_obs", "beta_interaction",
                 "se", "p", "p_BH_within_cohort"]].to_string(index=False))
log("")

with open(os.path.join(OUT, "88_diagnostics_63.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
log(f"Saved: {os.path.join(OUT, '88_diagnostics_63.csv')}")
log(f"Saved: {os.path.join(OUT, '88_diagnostics_63.txt')}")
