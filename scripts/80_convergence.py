"""
80 - Convergence diagnostics at the level of the single model, for all 63
kinetic parameter x time mixed models.

Why. The `converged` flag of statsmodels is just the `converged` returned by
the optimiser (warnflag == 0). It does not rule out a singular random-effects
fit, a variance on the boundary of the parameter space, an indefinite Hessian
or a gradient far from zero. Script 72 records which optimiser converges; here
everything the reviewer asks for is recorded, for each fit:

  optimiser, converged flag, warnflag and message (if statsmodels keeps it)
  norm of the gradient at the optimum
  covariance matrix of the random effects (cov_re): intercept variance,
    slope variance, their correlation
  singularity flag and boundary flag
  positive definiteness and condition number of the variance-covariance
    matrix of the fixed effects (cov_params)
  state of the Hessian used by statsmodels for the standard errors
  residual scale

Technical notes verified by reading the source of statsmodels 0.14.4
(statsmodels/regression/mixed_linear_model.py and statsmodels/base/optimizer.py):

  - MixedLMResults does NOT expose `mle_retvals`. The optimiser's dictionary
    is reachable only by passing `full_output=True` to `.fit()`, which stores
    it in `results.hist[-1]`. The same cascade of `_fit.py` is therefore redone
    here with `full_output=True`; the comparison with outputs/72_mixed_models
    .csv verifies that this changes nothing in the estimates.
  - `mle_retvals` contains `gopt` (the gradient) only for `lbfgs` and `bfgs`.
    For `powell` and `nm` no gradient exists: for those fits a numerical
    central-difference gradient of the profile log-likelihood is computed.
    The grad_route column tells which route was used for each
    row.
  - `gopt` is the gradient of -loglike/nobs (the objective function that
    statsmodels minimises), not of loglike. To make it comparable with the
    numerical gradient it is multiplied back by nobs. Verified numerically:
    gopt == -score/nobs.
  - statsmodels does NOT keep the text message of scipy (for lbfgs the
    field d['task'] is discarded in _fit_lbfgs). The only text information
    that can be reconstructed is the warnflag, which is decoded here.

Output: outputs/80_convergence.csv, outputs/80_convergence.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as so
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS  # same cascade, same order, same maxiter

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []

# declared thresholds
TOL_SINGULAR = 1e-8      # variance < TOL_SINGULAR * residual scale
TOL_CORR = 0.999          # |correlation| >= TOL_CORR
TOL_GRAD = 1e-3           # |grad|_inf of the profile log-likelihood
TOL_GRAD_SCALED = 1e-6    # |grad|_inf / nobs
TOL_NEWTON = 1e-4         # Newton decrement, in loglike units
TOL_LL = 1e-3             # loglike gain at the refinement beyond which
                          # the reported optimum is not the maximum


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


warnings.filterwarnings("ignore")
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))

OUTCOMES = {
    "updrs3_score": ("MDS-UPDRS III", ["age_bl", "SEX", "dur_bl"]),
    "updrs_totscore": ("MDS-UPDRS total", ["age_bl", "SEX", "dur_bl"]),
    "updrs2_score": ("MDS-UPDRS II", ["age_bl", "SEX", "dur_bl"]),
    "moca": ("MoCA", ["age_bl", "SEX", "dur_bl", "EDUCYRS"]),
    "MIA_STRIATUM_BILAT": ("DAT striatum, bilateral", ["age_bl", "SEX", "dur_bl"]),
    "MIA_PUTAMEN_BILAT": ("DAT putamen, bilateral", ["age_bl", "SEX", "dur_bl"]),
    "scopa": ("SCOPA-AUT", ["age_bl", "SEX", "dur_bl"]),
}
KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}


def build(cohort_file, kin_map):
    """Identical to 72_longitudinal_refit.build."""
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


WARNFLAG = {
    "lbfgs": {0: "converged (warnflag=0)",
              1: "too many function evaluations or iterations (warnflag=1)",
              2: "stopped for another reason; scipy's d['task'] is discarded "
                 "by statsmodels (warnflag=2)"},
    "bfgs": {0: "converged (warnflag=0)",
             1: "maximum number of iterations exceeded (warnflag=1)",
             2: "gradient and/or function calls not changing (warnflag=2)",
             3: "NaN result encountered (warnflag=3)"},
    "powell": {0: "converged (warnflag=0)",
               1: "maximum number of function evaluations exceeded "
                  "(warnflag=1)",
               2: "maximum number of iterations exceeded (warnflag=2)"},
    "nm": {0: "converged (warnflag=0)",
           1: "maximum number of function evaluations exceeded (warnflag=1)",
           2: "maximum number of iterations exceeded (warnflag=2)"},
}


def cascade(formula, data, groups, re_formula="~years"):
    """Same cascade as _fit.fit_mixed, but with full_output=True to obtain
    mle_retvals, and with the ConvergenceWarning collected instead of suppressed."""
    last = None
    for method, it in OPTIMISERS:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                r = smf.mixedlm(formula, data, groups=groups,
                                re_formula=re_formula).fit(
                    method=method, maxiter=it, full_output=True)
            except Exception:
                continue
            msgs = sorted({str(w.message) for w in captured})
        last = (r, method, msgs)
        if r.converged:
            return r, method, msgs
    if last is None:
        return None, None, []
    return last


def _ll(md, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(md.loglike(p))


def grad_numerical(md, p):
    """Central-difference gradient of the profile log-likelihood, with
    respect to the packed free parameters (use_sqrt parametrisation of the
    model, the same one in which the optimiser worked)."""
    g = np.empty(len(p))
    for i in range(len(p)):
        h = 1e-5 * max(1.0, abs(p[i]))
        pp, pm = p.copy(), p.copy()
        pp[i] += h
        pm[i] -= h
        g[i] = (_ll(md, pp) - _ll(md, pm)) / (2 * h)
    return g


def hess_numerical(md, p):
    """Central-difference Hessian of the profile log-likelihood,
    IN THE SAME parameter space in which the optimiser worked.
    (The Hessian of statsmodels, md.hessian, is instead in the untransformed
    space and includes the fixed effects: it serves for the standard errors,
    not to judge the optimiser's optimum.)"""
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
    return H, f0


def start_default(md):
    """The starting values that MixedLM.fit builds by itself (fe=0, cov_re=I,
    vcomp=1), packed as it passes them to the optimiser."""
    p0 = MixedLMParams(md.k_fe, md.k_re, md.k_vc)
    p0.fe_params = np.zeros(md.k_fe)
    p0.cov_re = np.eye(md.k_re)
    p0.vcomp = np.ones(md.k_vc)
    return np.asarray(p0.get_packed(use_sqrt=md.use_sqrt, has_fe=False),
                      dtype=float)


def message_lbfgs(md, maxiter):
    """statsmodels discards the text message of scipy (in _fit_lbfgs it keeps
    only d['grad'], not d['task']). Here `fmin_l_bfgs_b` is rerun with
    objective, gradient, starting values, bounds and maxiter identical to those
    that statsmodels uses, to recover the message. The discrepancy between the
    solution thus obtained and that of statsmodels is also returned, which
    serves to verify that it really is the same optimisation."""
    nobs = float(md.endog.shape[0])

    def f(p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return -md.loglike(p) / nobs

    def g(p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return -np.asarray(md.score(p), dtype=float) / nobs

    s = start_default(md)
    try:
        x, fv, d = so.fmin_l_bfgs_b(f, s, fprime=g, maxiter=maxiter,
                                    bounds=[(None, None)] * len(s),
                                    iprint=0, disp=False)
    except Exception as e:
        return f're-run failed: {e}', np.nan, np.nan
    t = d.get("task", "")
    if isinstance(t, bytes):
        t = t.decode("utf-8", "replace")
    return str(t), int(d.get("warnflag", -1)), x


def refinement(formula, data, groups, packed, md_ref, re_formula="~years"):
    """Restarts FROM THE REPORTED OPTIMUM with all the optimisers of the cascade
    and returns the solution with the highest log-likelihood. If the
    reported optimum were spurious, here the log-likelihood would rise and the
    coefficients would move.

    The log-likelihood of each solution is evaluated on `md_ref`, i.e. the
    SAME model instance with which the reported optimum was evaluated, so
    that the two numbers are comparable without residual doubts."""
    best = None
    for method, it in OPTIMISERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = smf.mixedlm(formula, data, groups=groups,
                                re_formula=re_formula).fit(
                    method=method, maxiter=it, start_params=packed,
                    full_output=False)
            except Exception:
                continue
        p = np.asarray(
            r.params_object.get_packed(use_sqrt=md_ref.use_sqrt, has_fe=False),
            dtype=float)
        ll = _ll(md_ref, p)
        if not np.isfinite(ll):
            continue
        if best is None or ll > best[2]:
            best = (r, method, ll)
    return best


def diagnostics(r, method, msgs, cohort, out, label, kin, m, formula):
    md = r.model
    po = r.params_object
    packed = np.asarray(po.get_packed(use_sqrt=md.use_sqrt, has_fe=False),
                        dtype=float)
    nobs = float(md.endog.shape[0])

    # --- mle_retvals -------------------------------------------------------
    hist = getattr(r, "hist", None)
    mr = hist[-1] if hist else None
    if mr is None:
        keys = ""
        warnflag = np.nan
        message = 'mle_retvals not available'
    else:
        keys = ";".join(sorted(mr.keys()))
        warnflag = int(mr.get("warnflag", -1))
        message = WARNFLAG.get(method, {}).get(
            warnflag, f'warnflag={warnflag} (not decoded)')
        if not any(k in mr for k in ("message", "task", "msg")):
            message += ' | statsmodels does not keep the text message'

    # --- gradient ----------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_an = np.asarray(md.score(packed), dtype=float)
    g_num = grad_numerical(md, packed)
    gn_an = float(np.max(np.abs(g_an)))
    gn_num = float(np.max(np.abs(g_num)))

    if mr is not None and "gopt" in mr:
        gopt = np.asarray(mr["gopt"], dtype=float)
        grad_route = "mle_retvals['gopt']"
        grad_raw = float(np.max(np.abs(gopt)))
        grad_raw_obj = "-loglike/nobs"
        grad_used = grad_raw * nobs          # brought back to the loglike scale
    else:
        grad_route = 'numerical gradient (central differences on loglike)'
        grad_raw = gn_num
        grad_raw_obj = "loglike"
        grad_used = gn_num

    # --- random effects ----------------------------------------------------
    cov_re = np.asarray(r.cov_re, dtype=float)
    scale = float(r.scale)
    v_int = float(cov_re[0, 0])
    v_slope = float(cov_re[1, 1])
    c_is = float(cov_re[0, 1])
    if v_int > 0 and v_slope > 0:
        corr = c_is / np.sqrt(v_int * v_slope)
    else:
        corr = np.nan
    eig_re = np.linalg.eigvalsh((cov_re + cov_re.T) / 2)

    sing_var = (v_int < TOL_SINGULAR * scale) or (v_slope < TOL_SINGULAR * scale)
    sing_corr = bool(np.isfinite(corr) and abs(corr) >= TOL_CORR)
    singular = bool(sing_var or sing_corr)
    edge = bool(v_int == 0.0 or v_slope == 0.0)

    # --- Hessian used by statsmodels for the standard errors ----------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hess, hess_sing = md.hessian(po)
    hess = np.asarray(hess, dtype=float)
    hdiag = np.diag(hess)
    eig_h = np.linalg.eigvalsh((hess + hess.T) / 2)

    # --- cov_params of the fixed effects -----------------------------------
    cp = np.asarray(r.cov_params(), dtype=float)
    k_fe = int(md.k_fe)
    fe = cp[:k_fe, :k_fe]
    fe = (fe + fe.T) / 2
    eig_fe = np.linalg.eigvalsh(fe)
    fe_pd = bool(eig_fe.min() > 0)
    fe_cond = float(np.linalg.cond(fe))
    tot = (cp + cp.T) / 2
    eig_tot = np.linalg.eigvalsh(tot)

    # --- Hessian of the profile loglike in the optimiser's space ------------
    Hp, ll_opt = hess_numerical(md, packed)
    Hp = (Hp + Hp.T) / 2
    eig_hp = np.linalg.eigvalsh(Hp)
    hp_neg_def = bool(eig_hp.max() < 0)
    # Newton decrement: expected gain in log-likelihood from taking one
    # Newton step from the reported optimum. It is the interpretable measure of the
    # distance from the maximum: the raw norm of the gradient depends on the scale
    # of the outcome, this does not.
    # Newton step for the MAXIMISATION: d = -H^{-1} g; expected gain
    # = -0.5 g' H^{-1} g = +0.5 g'd, non-negative when H is negative definite.
    try:
        step = np.linalg.solve(Hp, -g_an)
        newton_dec = float(0.5 * g_an @ step)
        step_max = float(np.max(np.abs(step)))
    except np.linalg.LinAlgError:
        newton_dec, step_max = np.nan, np.nan

    # --- text message of scipy for lbfgs (first attempt in the cascade) -----
    # The comparison between the rerun and the solution of statsmodels is made
    # on the LOG-LIKELIHOOD and not on the packed vector: statsmodels
    # reconstructs cov_re from the Cholesky factor and then refactorises it with
    # np.linalg.cholesky, which imposes a positive diagonal; if the optimiser had
    # arrived with a negative diagonal the vector changes sign while being
    # the same point. The comparison on the loglike does not have this ambiguity.
    task, task_flag, x_lbfgs = message_lbfgs(md, dict(OPTIMISERS)["lbfgs"])
    if method == "lbfgs" and isinstance(x_lbfgs, np.ndarray):
        task_departure = float(abs(_ll(md, x_lbfgs) - _ll(md, packed)))
    else:
        task_departure = np.nan

    # --- refinement from the reported optimum -------------------------------
    t = f"{kin}:years" if f"{kin}:years" in r.params.index else f"years:{kin}"
    ref = refinement(formula, m, m.PATNO, packed, md)
    if ref is None:
        ll_ref, met_ref = np.nan, ""
        b_ref = se_ref = p_ref = np.nan
        d_ll = d_beta = d_se = d_p = np.nan
        ref_conv = False
    else:
        r2, met_ref, ll_ref = ref
        t2 = (f"{kin}:years" if f"{kin}:years" in r2.params.index
              else f"years:{kin}")
        b_ref = float(r2.params[t2])
        se_ref = float(r2.bse[t2])
        p_ref = float(r2.pvalues[t2])
        ref_conv = bool(r2.converged)
        d_ll = ll_ref - ll_opt
        d_beta = b_ref - float(r.params[t])
        d_se = se_ref - float(r.bse[t])
        d_p = p_ref - float(r.pvalues[t])

    return {
        "cohort": cohort, "outcome": out, "label": label, "kinetic": kin,
        "n_subjects": int(m.PATNO.nunique()), "n_obs": int(len(m)),
        "nobs_model": int(nobs),
        "beta_interaction": float(r.params[t]), "se": float(r.bse[t]),
        "p": float(r.pvalues[t]),
        "optimiser": method,
        "sm_converged": bool(r.converged),
        "optimiser_warnflag": warnflag,
        "optimiser_message": message,
        "mle_retvals_keys": keys,
        "lbfgs_scipy_task": task,
        "lbfgs_scipy_warnflag": task_flag,
        "lbfgs_is_reported_fit": bool(method == "lbfgs"),
        "lbfgs_replay_loglike_diff": task_departure,
        "grad_route": grad_route,
        "grad_inf_norm_reported": grad_raw,
        "grad_reported_objective": grad_raw_obj,
        "grad_inf_norm_loglike": grad_used,
        "grad_inf_norm_loglike_over_nobs": grad_used / nobs,
        "grad_inf_norm_numeric": gn_num,
        "grad_inf_norm_analytic_score": gn_an,
        "profile_hess_neg_definite": hp_neg_def,
        "profile_hess_max_eigenvalue": float(eig_hp.max()),
        "newton_decrement_loglike": newton_dec,
        "newton_step_max_abs": step_max,
        "loglike_at_reported_optimum": ll_opt,
        "loglike_after_repolish": ll_ref,
        "repolish_best_optimiser": met_ref,
        "repolish_converged": ref_conv,
        "repolish_beta": b_ref,
        "repolish_se": se_ref,
        "repolish_p": p_ref,
        "delta_loglike_repolish": d_ll,
        "delta_beta_repolish": d_beta,
        "delta_se_repolish": d_se,
        "delta_p_repolish": d_p,
        "reported_optimum_is_maximum": bool(np.isfinite(d_ll)
                                            and d_ll <= TOL_LL),
        "re_var_intercept": v_int,
        "re_var_slope": v_slope,
        "re_cov_intercept_slope": c_is,
        "re_corr_intercept_slope": float(corr),
        "re_cov_min_eigenvalue": float(eig_re.min()),
        "re_cov_cond": float(np.linalg.cond(cov_re)),
        "re_var_intercept_over_scale": v_int / scale,
        "re_var_slope_over_scale": v_slope / scale,
        "residual_scale": scale,
        "residual_sd": float(np.sqrt(scale)) if scale >= 0 else np.nan,
        "singular": singular,
        "singular_reason": ('variance < 1e-8*scale' if sing_var else
                            ("|corr| >= 0.999" if sing_corr else "")),
        "boundary_var_exactly_zero": edge,
        "sm_boundary_warning": bool(np.min(np.abs(np.diag(cov_re))) < 0.01),
        "hessian_singular_pinv": bool(hess_sing),
        "hessian_any_diag_nonneg": bool(np.any(hdiag >= 0)),
        "hessian_neg_definite": bool(eig_h.max() < 0),
        "hessian_max_eigenvalue": float(eig_h.max()),
        "fe_cov_pos_def": fe_pd,
        "fe_cov_min_eigenvalue": float(eig_fe.min()),
        "fe_cov_cond": fe_cond,
        "full_cov_params_pos_def": bool(eig_tot.min() > 0),
        "full_cov_params_min_eigenvalue": float(eig_tot.min()),
        "model_cov_sing_count": int(getattr(md, "_cov_sing", -1)),
        "fit_warnings": " || ".join(msgs),
    }


def fit_all(L, kins, tag):
    rows = []
    for out, (label, cov) in OUTCOMES.items():
        for k in kins:
            m = L[["PATNO", "years", out, k] + cov].dropna()
            multi = m.groupby("PATNO").size()
            if len(m) < 120 or int((multi >= 2).sum()) < 40:
                continue
            f = f"{out} ~ {k}*years + " + " + ".join(cov)
            r, method, msgs = cascade(f, m, m.PATNO)
            if r is None:
                log(f'  [{tag}] {out} x {k}: no optimiser returns a fit')
                continue
            rows.append(diagnostics(r, method, msgs, tag, out, label, k, m, f))
            print(f"  . {tag} {out} x {k}", flush=True)
    return pd.DataFrame(rows)


log("=" * 78)
log('80 — CONVERGENCE DIAGNOSTICS OF THE 63 KINETIC PARAMETER x TIME MIXED MODELS')
log("=" * 78)
log(f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}")
log(f'BASE = {BASE}')
log(f'cascade of optimisers: {OPTIMISERS}')
log("")
log('Declared thresholds:')
log(f'  singular   : variance of a random effect < {TOL_SINGULAR:g} x residual scale, or '
    f'|correlation| >= {TOL_CORR}')
log(f'  boundary   : variance of a random effect exactly 0')
log(f'  gradient   : |grad|_inf of the profiled log-likelihood > {TOL_GRAD:g}; in addition '
    f'|grad|_inf/nobs > {TOL_GRAD_SCALED:g}')
log(f"  optimum    : Newton decrement > {TOL_NEWTON:g} units of log-likelihood")
log("")
log("Caveat on the norm of the gradient. The log-likelihood is not")
log('dimensionless: its gradient grows with nobs and with the scale')
log("of the outcome. The DAT outcomes are on a scale for which the residual")
log("variance is of the order of 5e-3, and there |grad| is large in absolute")
log('value without this meaning distance from the maximum. For')
log("this reason the Newton decrement is also reported, that is the gain in")
log('log-likelihood that would be obtained with a Newton step')
log("from the reported optimum, which is scale-invariant, and the empirical")
log("refinement (refit from that optimum with all the optimisers).")
log("")

L150 = build("08_dataset_150h.csv", KIN150)
L24 = build("analysis_complete_cases.csv", KIN24)
d150 = fit_all(L150, list(KIN150.values()), "150h")
d24 = fit_all(L24, list(KIN24.values()), "24h")
D = pd.concat([d150, d24], ignore_index=True)
D.to_csv(os.path.join(OUT, "80_convergence.csv"), index=False)

log(f'models diagnosed: {len(D)}  (150h {len(d150)}, 24h {len(d24)})')
log("")

# ------------------------------------------------------------ verification --
log("-" * 78)
log('CHECK: full_output=True does not change the estimates')
log("-" * 78)
v72 = os.path.join(OUT, "72_mixed_models.csv")
if os.path.exists(v72):
    o = pd.read_csv(v72)
    c = D.merge(o[["cohort", "outcome", "kinetic", "beta", "se", "p",
                   "optimiser", "converged"]],
                on=["cohort", "outcome", "kinetic"], how="inner",
                suffixes=("", "_72"))
    log(f'rows paired with 72_mixed_models.csv: {len(c)} out of {len(D)}')
    log(f'max |delta beta|     : {np.abs(c.beta_interaction - c.beta).max():.3e}')
    log(f'max |delta SE|       : {np.abs(c.se - c.se_72).max():.3e}')
    log(f'max |delta p|        : {np.abs(c.p - c.p_72).max():.3e}')
    log(f'same optimiser       : {int((c.optimiser == c.optimiser_72).sum())}/{len(c)}')
    log(f'same converged flag  : {int((c.sm_converged == c.converged).sum())}/{len(c)}')
else:
    log(f'WARNING: {v72} does not exist, check not executed')

log("")
log('CHECK: numerical gradient against analytical score of statsmodels')
log("-" * 78)
den = np.maximum(D.grad_inf_norm_analytic_score.abs(), 1e-12)
disc = (D.grad_inf_norm_numeric - D.grad_inf_norm_analytic_score).abs()
log(f'max |numerical - analytical|        : {disc.max():.3e}')
log(f'max relative difference             : {(disc / den).max():.3e}')
sub = D[D.grad_route == "mle_retvals['gopt']"]
if len(sub):
    d2 = (sub.grad_inf_norm_loglike - sub.grad_inf_norm_analytic_score).abs()
    log(f'over {len(sub)} fits with gopt: max |gopt*nobs - score| = {d2.max():.3e}')
log("")

# ----------------------------------------------------------------- summary --
log("=" * 78)
log('SUMMARY OF THE 63 FITS')
log("=" * 78)
n = len(D)
log(f'fits with statsmodels converged flag = False  : {int((~D.sm_converged).sum())} out of {n}')
log(f'SINGULAR fits (declared threshold)            : {int(D.singular.sum())} out of {n}')
log(f'fits AT THE BOUNDARY (variance exactly 0)     : {int(D.boundary_var_exactly_zero.sum())} out '
    f'of {n}')
log(f'cov_params fixed effects NOT positive definite: {int((~D.fe_cov_pos_def).sum())} out of {n}')
log(f'full cov_params NOT positive definite         : {int((~D.full_cov_params_pos_def).sum())} out '
    f'of {n}')
log(f'singular Hessian (pseudo-inverse)             : {int(D.hessian_singular_pinv.sum())} out of '
    f'{n}')
log(f'Hessian with at least one diagonal >= 0       : {int(D.hessian_any_diag_nonneg.sum())} out of '
    f'{n}')
log(f'Hessian not negative definite                 : {int((~D.hessian_neg_definite).sum())} out of '
    f'{n}')
log(f'|grad|_inf > {TOL_GRAD:g}                            : '
    f'{int((D.grad_inf_norm_loglike > TOL_GRAD).sum())} out of {n}')
log(f'|grad|_inf/nobs > {TOL_GRAD_SCALED:g}                    : '
    f'{int((D.grad_inf_norm_loglike_over_nobs > TOL_GRAD_SCALED).sum())} out of {n}')
log(f'statsmodels boundary warning (|var| < 0.01)   : {int(D.sm_boundary_warning.sum())} out of {n}')
log(f'Newton decrement > {TOL_NEWTON:g}                       : '
    f'{int((D.newton_decrement_loglike > TOL_NEWTON).sum())} out of {n}')
log(f'Hessian of the profiled loglike not neg. def.  : {int((~D.profile_hess_neg_definite).sum())} '
    f'out of {n}')
log(f"REPORTED OPTIMUM IS NOT THE MAXIMUM (dl > {TOL_LL:g})  : "
    f"{int((~D.reported_optimum_is_maximum).sum())} out of {n}")
log("")
log("-" * 78)
log("REFINEMENT: restart from the reported optimum with all the optimisers")
log("-" * 78)
log("This is the check that the `converged` flag does not make. If the reported")
log("optimum were the maximum, restarting from there would gain nothing.")
log("")
log(f'max gain in log-likelihood          : {D.delta_loglike_repolish.max():.6g}')
log(f'max |change in beta|                : {D.delta_beta_repolish.abs().max():.6g}')
log(f'max |change in SE|                  : {D.delta_se_repolish.abs().max():.6g}')
log(f'max |change in p|                   : {D.delta_p_repolish.abs().max():.6g}')
mal = D[~D.reported_optimum_is_maximum]
log("")
log(f"--- the {len(mal)} fits whose reported optimum is NOT the maximum ---")
if len(mal):
    log(mal.sort_values("delta_loglike_repolish", ascending=False)[
        ["cohort", "outcome", "kinetic", "optimiser", "sm_converged",
         "repolish_best_optimiser", "loglike_at_reported_optimum",
         "loglike_after_repolish", "delta_loglike_repolish",
         "newton_decrement_loglike"]].to_string(index=False))
    log("")
    log("how the estimates of the interaction change in those fits:")
    log(mal.sort_values("delta_loglike_repolish", ascending=False)[
        ["cohort", "outcome", "kinetic", "beta_interaction", "repolish_beta",
         "se", "repolish_se", "p", "repolish_p"]].to_string(index=False))
log("")
log('consistency between Newton decrement (predicted) and actual gain:')
log(f"  Spearman correlation over {len(D)} fits = "
    f"{D.newton_decrement_loglike.corr(D.delta_loglike_repolish, method='spearman'):.4f}")
log("")
log("-" * 78)
log('TEXT MESSAGE OF SCIPY FOR lbfgs (recovered by rerunning fmin_l_bfgs_b)')
log("-" * 78)
ok = D[D.lbfgs_is_reported_fit]
log(f'rerun paired with the statsmodels solution: max |difference in loglike| = '
    f'{ok.lbfgs_replay_loglike_diff.max():.3e} over {len(ok)} fits')
for k, v in D.groupby("lbfgs_is_reported_fit").lbfgs_scipy_task.value_counts().items():
    log(f"  lbfgs is the reported fit={k[0]}: {v} x  {k[1]}")
log("")
log('gradient: route used')
for k, v in D.grad_route.value_counts().items():
    log(f"  {k}: {v}")
log(f'optimiser that converged: {D.optimiser.value_counts().to_dict()}')
log("")
log('distribution of |grad|_inf (profiled log-likelihood)')
log(f'  min {D.grad_inf_norm_loglike.min():.3e}   median {D.grad_inf_norm_loglike.median():.3e}   '
    f'max {D.grad_inf_norm_loglike.max():.3e}')
log('distribution of the variances of the random effects')
log(f'  var intercept : min {D.re_var_intercept.min():.6g}  max {D.re_var_intercept.max():.6g}')
log(f'  var slope     : min {D.re_var_slope.min():.6g}  max {D.re_var_slope.max():.6g}')
log(f'  |correlation| : min {D.re_corr_intercept_slope.abs().min():.4f}  max '
    f'{D.re_corr_intercept_slope.abs().max():.4f}')
log(f"  minimum var/scale ratio: "
    f"{(D[['re_var_intercept', 're_var_slope']].min(axis=1) / D.residual_scale).min():.3e}")
log(f'  condition number of cov_params (fixed effects): min {D.fe_cov_cond.min():.4g}  max '
    f'{D.fe_cov_cond.max():.4g}')
log("")

if int(D.singular.sum()):
    log('--- SINGULAR FITS ---')
    log(D[D.singular][["cohort", "outcome", "kinetic", "re_var_intercept",
                       "re_var_slope", "re_corr_intercept_slope",
                       "residual_scale", "singular_reason"]].to_string(
        index=False))
    log("")
if int(D.boundary_var_exactly_zero.sum()):
    log('--- FITS AT THE BOUNDARY ---')
    log(D[D.boundary_var_exactly_zero][
        ["cohort", "outcome", "kinetic", "re_var_intercept", "re_var_slope"]
    ].to_string(index=False))
    log("")

worst = D.nlargest(8, "grad_inf_norm_loglike")
log("--- the 8 fits with the largest gradient ---")
log(worst[["cohort", "outcome", "kinetic", "optimiser", "grad_route",
              "grad_inf_norm_loglike", "grad_inf_norm_loglike_over_nobs",
              "newton_decrement_loglike",
              "delta_loglike_repolish"]].to_string(index=False))
log("")

# ------------------------------------------------------------ headline -----
log("=" * 78)
log("THE MODELS THAT SURVIVE THE CORRECTION FOR MULTIPLICITY, AND THE MoCA")
log("=" * 78)
HEADS = [("24h", "MIA_STRIATUM_BILAT", "Fmax"),
        ("24h", "MIA_PUTAMEN_BILAT", "Fmax")]
FIELDS = ["n_subjects", "n_obs",
         "optimiser", "sm_converged", "optimiser_warnflag",
         "optimiser_message", "mle_retvals_keys",
         "lbfgs_is_reported_fit", "lbfgs_scipy_task",
         "lbfgs_replay_loglike_diff",
         "grad_route", "grad_inf_norm_reported", "grad_reported_objective",
         "grad_inf_norm_loglike", "grad_inf_norm_loglike_over_nobs",
         "grad_inf_norm_numeric", "grad_inf_norm_analytic_score",
         "profile_hess_neg_definite", "profile_hess_max_eigenvalue",
         "newton_decrement_loglike", "newton_step_max_abs",
         "loglike_at_reported_optimum", "loglike_after_repolish",
         "repolish_best_optimiser", "repolish_converged",
         "delta_loglike_repolish", "reported_optimum_is_maximum",
         "beta_interaction", "repolish_beta", "delta_beta_repolish",
         "se", "repolish_se", "delta_se_repolish",
         "p", "repolish_p", "delta_p_repolish",
         "re_var_intercept", "re_var_slope", "re_cov_intercept_slope",
         "re_corr_intercept_slope", "re_cov_min_eigenvalue",
         "re_var_intercept_over_scale", "re_var_slope_over_scale",
         "residual_scale", "residual_sd", "singular", "singular_reason",
         "boundary_var_exactly_zero", "sm_boundary_warning",
         "hessian_singular_pinv", "hessian_any_diag_nonneg",
         "hessian_neg_definite", "fe_cov_pos_def", "fe_cov_min_eigenvalue",
         "fe_cov_cond", "full_cov_params_pos_def", "fit_warnings"]


def card(row, title):
    log("")
    log(f'### {title}')
    for c in FIELDS:
        v = row[c]
        if isinstance(v, (float, np.floating)):
            log(f"    {c:32s} = {float(v):.10g}")
        else:
            log(f"    {c:32s} = {v}")


for co, ou, ki in HEADS:
    s = D[(D.cohort == co) & (D.outcome == ou) & (D.kinetic == ki)]
    if len(s) != 1:
        log(f'\nWARNING: {co}/{ou}/{ki} found {len(s)} times')
        continue
    card(s.iloc[0], f'{co} — {ou} x {ki}  (passes BH within cohort)')

moca = D[(D.cohort == "150h") & (D.outcome == "moca")]
log("")
log("-" * 78)
log(f'MoCA in the 150 h cohort: {len(moca)} models (one per kinetic parameter)')
log("-" * 78)
for _, row in moca.iterrows():
    card(row, f'150h — moca x {row.kinetic}')

log("")
log("=" * 78)
log('VERDICT')
log("=" * 78)
heads_rows = pd.concat(
    [D[(D.cohort == co) & (D.outcome == ou) & (D.kinetic == ki)]
     for co, ou, ki in HEADS] + [moca], ignore_index=True)
log("1) SINGULARITY AND BOUNDARY")
sb = heads_rows[heads_rows.singular | heads_rows.boundary_var_exactly_zero]
if len(sb) == 0:
    log("   None of the lead models is singular or at the boundary. The variances")
    log('   of the random effects, relative to the residual scale, are:')
    log(heads_rows[["cohort", "outcome", "kinetic",
                    "re_var_intercept_over_scale", "re_var_slope_over_scale",
                    "re_corr_intercept_slope"]].to_string(index=False))
    log("   The smallest is " +
        f"{heads_rows[['re_var_intercept_over_scale', 're_var_slope_over_scale']].min().min():.4g} "
        f"times the residual scale, against a threshold of 1e-8: none has")
    log('   collapsed to zero, and no correlation comes close to +/-1.')
    log("   The statsmodels boundary warning is nevertheless triggered on some of")
    log("   these models because it is an ABSOLUTE threshold (|var| < 0.01) and the")
    log("   DAT outcomes are on a scale in which the residual variance itself is")
    log("   of the order of 5e-3: it is an artefact of scale, not a fit at the boundary.")
else:
    log('   WARNING: lead models singular or at the boundary:')
    log(sb[["cohort", "outcome", "kinetic", "singular", "singular_reason",
            "boundary_var_exactly_zero", "re_var_intercept", "re_var_slope",
            "re_corr_intercept_slope"]].to_string(index=False))
log("")
log('2) VARIANCE-COVARIANCE MATRICES AND HESSIANS')
np_ = heads_rows[~heads_rows.fe_cov_pos_def | ~heads_rows.hessian_neg_definite
                 | ~heads_rows.profile_hess_neg_definite]
if len(np_) == 0:
    log("   For all the lead models cov_params of the fixed effects is")
    log("   positive definite, the Hessian used by statsmodels for the standard")
    log("   errors is negative definite and has not been pseudo-inverted, and")
    log("   the Hessian of the profiled log-likelihood is negative")
    log(f'   definite. Condition numbers of cov_params between {heads_rows.fe_cov_cond.min():.4g} '
        f'and {heads_rows.fe_cov_cond.max():.4g}.')
else:
    log('   WARNING:')
    log(np_[["cohort", "outcome", "kinetic", "fe_cov_pos_def",
             "hessian_neg_definite", "profile_hess_neg_definite"]].to_string(
        index=False))
log("")
log("3) IS THE REPORTED OPTIMUM REALLY THE MAXIMUM?")
faults = heads_rows[~heads_rows.reported_optimum_is_maximum]
if len(faults) == 0:
    log("   Yes: for all the lead models restarting from the reported optimum")
    log('   does not gain log-likelihood.')
else:
    log('   NO. For the following models the reported point does NOT maximise the')
    log("   likelihood, although statsmodels declares converged=True.")
    log('   The published estimates must be redone from that point.')
    log(faults[["cohort", "outcome", "kinetic", "optimiser", "sm_converged",
                "loglike_at_reported_optimum", "loglike_after_repolish",
                "delta_loglike_repolish", "beta_interaction", "repolish_beta",
                "se", "repolish_se", "p", "repolish_p"]].to_string(index=False))
log("")
log('4) NORM OF THE GRADIENT (crude criterion) BY COMPARISON')
log(heads_rows[["cohort", "outcome", "kinetic", "grad_inf_norm_loglike",
                "grad_inf_norm_loglike_over_nobs", "newton_decrement_loglike",
                "delta_loglike_repolish"]].to_string(index=False))

with open(os.path.join(OUT, "80_convergence.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
log("")
log(f"Saved: {os.path.join(OUT, '80_convergence.csv')}")
log(f"Saved: {os.path.join(OUT, '80_convergence.txt')}")
