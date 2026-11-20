"""
84 - The gradient norm of the 63 mixed models: what makes it large,
     what fixes it, and which stationarity criterion is defensible.

The reviewer observes that four of the 63 fits have an infinity norm of the
gradient of the log-likelihood of up to 98, incompatible with a numerically
tight optimum. The reviewer is right. The explanation that the manuscript
gives in 2.18 ("all on outcomes of large scale") is however wrong, and the
natural correction — standardising the outcome — has no effect at all.

The reason is that statsmodels does not optimise over parameters in units of
the outcome. MixedLM profiles out both the fixed effects (GLS) and the residual
variance, and the free parameters left to the optimiser are the elements of
the Cholesky factor of cov_re / scale, that is, DIMENSIONLESS VARIANCE
RATIOS. Rescaling y -> y/s leaves those parameters unchanged and changes the
REML log-likelihood only by the additive constant (nobs - k_fe) * log(s).
An additive constant has zero gradient, so the gradient with respect to the
optimised parameters is EXACTLY invariant to the scale of the outcome.
The script verifies this numerically on all 63 models.

The large gradient is therefore not an artefact of scale: it is real distance
from the maximum. On those four fits `lbfgs` stops with warnflag = 0 while the
gradient is still 62-98, and restarting from that point gains between 5 and
12.6 units of log-likelihood. The correction is to redo the optimisation:
one restarts from the reported optimum with the whole cascade, iterating until
the log-likelihood no longer rises. Tightening the tolerances of `lbfgs`
(factr, pgtol) does not help: the line search stops at the same point.

Stationarity criterion adopted: the NEWTON DECREMENT
    delta = 1/2 * g' (-H)^{-1} g
evaluated on the profiled log-likelihood, with a threshold of 1e-4 units of
log-likelihood. It is the only one of the quantities considered that is
invariant to how the outcome and the parameters are scaled, and it is expressed
directly in the units that matter for inference (for comparison, a 95%
profile-likelihood confidence interval lies 1.92 units from the maximum).

Output: outputs/84_gradients.csv, outputs/84_gradients.txt
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
from _fit import OPTIMISERS  # same cascade, same order, same maxiter
from _paths import DATA
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []

# --------------------------------------------------------------- thresholds -----
TOL_NEWTON = 1e-4       # primary criterion: Newton decrement, in loglike
TOL_SINGULAR = 1e-8    # variance < TOL_SINGULAR * residual scale
TOL_CORR = 0.999        # |correlation| >= TOL_CORR
TOL_GAIN = 1e-10        # loglike gain below which the polishing stops
MAX_ROUNDS = 10           # maximum rounds of polishing


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))

OUTCOMES = {
    "updrs3_score": ("MDS-UPDRS III", ["age_bl", "SEX", "dur_bl"]),
    "updrs_totscore": ("MDS-UPDRS total", ["age_bl", "SEX", "dur_bl"]),
    "updrs2_score": ("MDS-UPDRS II", ["age_bl", "SEX", "dur_bl"]),
    "moca": ("MoCA", ["age_bl", "SEX", "dur_bl", "EDUCYRS"]),
    "MIA_PUTAMEN_BILAT": ("DAT putamen, bilateral", ["age_bl", "SEX", "dur_bl"]),
    # The caudate had never been included among the outcomes. Adding it enlarges
    # the within-cohort BH family (24h from 28 to 32, 150h from 35 to 40), so the
    # count of survivors is NOT comparable with the published one:
    # it must be reported as such, not as a revision of the previous number.
    "MIA_CAUDATE_BILAT": ("DAT caudate, bilateral", ["age_bl", "SEX", "dur_bl"]),
    "scopa": ("SCOPA-AUT", ["age_bl", "SEX", "dur_bl"]),
}
KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}


def build(cohort_file, kin_map):
    """Identical to 72_longitudinal_refit.build and to 80_convergence.build."""
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


# ------------------------------------------------------------ primitives -----
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


def cascade(formula, data, groups, start=None):
    """First optimiser that converges, like _fit.fit_mixed."""
    last = None
    for method, it in OPTIMISERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                kw = {} if start is None else {"start_params": start}
                r = smf.mixedlm(formula, data, groups=groups,
                                re_formula="~years").fit(method=method,
                                                        maxiter=it, **kw)
            except Exception:
                continue
        last = (r, method)
        if r.converged:
            return r, method
    return (last if last else (None, None))


def polishing(formula, data, groups, r0, md_ref):
    """Restarts from the reported optimum with the WHOLE cascade and iterates until
    the log-likelihood no longer rises. The loglike of each candidate is evaluated
    on the SAME model instance md_ref, so the numbers are comparable.

    This is the correction: not another tolerance, but another
    optimisation starting from the point at which the first one stopped."""
    best_p = pack(r0)
    best = (r0, '(none)', _ll(md_ref, best_p), best_p)
    rounds = 0
    for cycle in range(MAX_ROUNDS):
        improved = False
        for method, it in OPTIMISERS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = smf.mixedlm(formula, data, groups=groups,
                                    re_formula="~years").fit(
                        method=method, maxiter=it, start_params=best[3])
                except Exception:
                    continue
            p = pack(r)
            ll = _ll(md_ref, p)
            if np.isfinite(ll) and ll > best[2] + TOL_GAIN:
                best = (r, method, ll, p)
                improved = True
        rounds = cycle + 1
        if not improved:
            break
    return best[0], best[1], best[2], best[3], rounds


def state(r, t):
    """Scale-independent diagnostics on a fit."""
    md = r.model
    p = pack(r)
    g = _score(md, p)
    H, ll = hess_profiled(md, p)
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
    denom = max(abs(ll), 1.0)
    rel = float(np.max(np.abs(g * p)) / denom)
    return {
        "grad_inf": float(np.max(np.abs(g))),
        "newton": newton,
        "rel_grad": rel,
        "loglike": ll,
        "beta": float(r.params[t]),
        "se": float(r.bse[t]),
        "p": float(r.pvalues[t]),
        "residual_scale": scale,
        "re_var_intercept": v_int,
        "re_var_slope": v_slope,
        "singular": sing,
        "boundary": edge,
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

            # ---- (A) published fit: the cascade as it is ---------------
            r, method = cascade(f, m, m.PATNO)
            if r is None:
                log(f'  [{tag}] {out} x {k}: no fit')
                continue
            t = f"{k}:years" if f"{k}:years" in r.params.index else f"years:{k}"
            A = state(r, t)
            md = r.model
            sd_y = float(m[out].std(ddof=1))

            # ---- (B) same cascade on the standardised outcome --------------
            ms = m.copy()
            ms[out] = ms[out] / sd_y
            rs, method_s = cascade(f, ms, ms.PATNO)
            ts = (f"{k}:years" if f"{k}:years" in rs.params.index
                  else f"years:{k}")
            B = state(rs, ts)
            # back to original units: y' = y/s  =>  beta = s * beta'
            B_beta = B["beta"] * sd_y
            B_se = B["se"] * sd_y
            # equivalent REML loglike: it differs by (nobs - k_fe) * log(s)
            B_ll_equiv = B["loglike"] + (len(m) - int(md.k_fe)) * np.log(sd_y)

            # ---- (C) the correction: redoing the optimisation -------------
            rp, method_p, ll_p, packed_p, rounds = polishing(f, m, m.PATNO, r, md)
            tp = (f"{k}:years" if f"{k}:years" in rp.params.index
                  else f"years:{k}")
            C = state(rp, tp)

            rows.append({
                "cohort": tag, "outcome": out, "label": label, "kinetic": k,
                "n_subjects": int(m.PATNO.nunique()), "n_obs": int(len(m)),
                "outcome_sd": sd_y,
                # --- (A) published
                "pub_optimiser": method,
                "pub_converged": A["converged"],
                "pub_loglike": A["loglike"],
                "pub_grad_inf_norm": A["grad_inf"],
                "pub_grad_over_nobs": A["grad_inf"] / len(m),
                "pub_grad_over_nobs_sd": A["grad_inf"] / (len(m) * sd_y),
                "pub_rel_grad": A["rel_grad"],
                "pub_newton_decrement": A["newton"],
                "pub_beta": A["beta"], "pub_se": A["se"], "pub_p": A["p"],
                "pub_residual_scale": A["residual_scale"],
                "pub_re_var_intercept": A["re_var_intercept"],
                "pub_re_var_slope": A["re_var_slope"],
                "pub_singular": A["singular"], "pub_boundary": A["boundary"],
                "pub_stationary": bool(np.isfinite(A["newton"])
                                       and A["newton"] < TOL_NEWTON),
                # --- (B) standardised outcome
                "std_optimiser": method_s,
                "std_converged": B["converged"],
                "std_grad_inf_norm": B["grad_inf"],
                "std_newton_decrement": B["newton"],
                "std_beta_backtransformed": B_beta,
                "std_se_backtransformed": B_se,
                "std_p": B["p"],
                "std_singular": B["singular"], "std_boundary": B["boundary"],
                "std_loglike_equivalent": B_ll_equiv,
                "std_minus_pub_grad": B["grad_inf"] - A["grad_inf"],
                "std_minus_pub_beta": B_beta - A["beta"],
                "std_minus_pub_se": B_se - A["se"],
                "std_minus_pub_p": B["p"] - A["p"],
                # --- (C) re-optimised
                "pol_optimiser": method_p, "pol_rounds": rounds,
                "pol_converged": C["converged"],
                "pol_loglike": C["loglike"],
                "pol_grad_inf_norm": C["grad_inf"],
                "pol_grad_over_nobs": C["grad_inf"] / len(m),
                "pol_rel_grad": C["rel_grad"],
                "pol_newton_decrement": C["newton"],
                "pol_beta": C["beta"], "pol_se": C["se"], "pol_p": C["p"],
                "pol_residual_scale": C["residual_scale"],
                "pol_singular": C["singular"], "pol_boundary": C["boundary"],
                "pol_stationary": bool(np.isfinite(C["newton"])
                                       and C["newton"] < TOL_NEWTON),
                "delta_loglike": C["loglike"] - A["loglike"],
                "delta_beta": C["beta"] - A["beta"],
                "delta_se": C["se"] - A["se"],
                "delta_p": C["p"] - A["p"],
            })
            print(f"  . {tag} {out} x {k}  |g|pub={A['grad_inf']:.3g} -> |g|pol={C['grad_inf']:.3g}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


log("=" * 78)
log('84 — GRADIENT NORM OF THE 63 MIXED MODELS: DIAGNOSIS, CORRECTION,')
log("     STATIONARITY CRITERION")
log("=" * 78)
log(f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}")
log(f'BASE = {BASE}')
log(f'cascade: {OPTIMISERS}')
log(f"stationarity threshold: Newton decrement < {TOL_NEWTON:g}")
log("")

L150 = build("08_dataset_150h.csv", KIN150)
L24 = build("analysis_complete_cases.csv", KIN24)
d150 = analyse(L150, list(KIN150.values()), "150h")
d24 = analyse(L24, list(KIN24.values()), "24h")
D = pd.concat([d150, d24], ignore_index=True)

# within-cohort BH, before and after the correction
for col_in, col_out in [("pub_p", "pub_p_BH_within_cohort"),
                        ("pol_p", "pol_p_BH_within_cohort")]:
    D[col_out] = np.nan
    for cohort in D.cohort.unique():
        i = D.index[D.cohort == cohort]
        D.loc[i, col_out] = multipletests(D.loc[i, col_in], method="fdr_bh")[1]

D.to_csv(os.path.join(OUT, "84_gradients.csv"), index=False)
log(f'models analysed: {len(D)}  (150h {len(d150)}, 24h {len(d24)})')
log("")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

# --------------------------------------------------------------- STEP 1 ----
log("=" * 78)
log('STEP 1 — WHICH ONES ARE THEY, AND DOES THE GRADIENT NORM DEPEND ON THE SCALE?')
log("=" * 78)
large = D[D.pub_grad_inf_norm > 1].sort_values("pub_grad_inf_norm",
                                                ascending=False)
log(f'fits with |grad|_inf > 1: {len(large)} of {len(D)}  (> 10: '
    f'{int((D.pub_grad_inf_norm > 10).sum())})')
log("")
log(large[["cohort", "label", "kinetic", "n_obs", "outcome_sd",
            "pub_residual_scale", "pub_grad_inf_norm", "pub_grad_over_nobs",
            "pub_optimiser", "pub_converged"]].to_string(index=False))
log("")
log("Standard deviation of the outcome, by outcome (on the sample of each fit):")
log(D.groupby("label").outcome_sd.agg(["min", "max"]).sort_values(
    "min").to_string())
log("")
log("Correlation between gradient norm and scale of the outcome, over the 63 fits:")
for name, x in [('SD of the outcome', D.outcome_sd),
                ('residual scale', D.pub_residual_scale)]:
    rp = np.corrcoef(D.pub_grad_inf_norm, x)[0, 1]
    rs = pd.Series(D.pub_grad_inf_norm).corr(pd.Series(x), method="spearman")
    rl = np.corrcoef(np.log10(D.pub_grad_inf_norm), np.log10(x))[0, 1]
    log(f'  |grad|_inf vs {name:16s}: Pearson {rp:+.4f}   Spearman {rs:+.4f}   Pearson on log10 '
        f'{rl:+.4f}')
log("")
log("The three smallest and the three largest outcome SDs, with their gradient:")
o = D.sort_values("outcome_sd")
log(pd.concat([o.head(3), o.tail(3)])[
    ["cohort", "label", "kinetic", "outcome_sd", "pub_grad_inf_norm"]
].to_string(index=False))
log("")

# --------------------------------------------------------------- STEP 2 ----
log("=" * 78)
log("STEP 2 — STANDARDISING THE OUTCOME CHANGES NOTHING")
log("=" * 78)
log('MixedLM profiles out fixed effects and residual variance: the free')
log('parameters are cov_re/scale, dimensionless ratios. y -> y/s adds to the')
log('REML log-likelihood only the constant (nobs - k_fe)*log(s), which has')
log('zero gradient. Numerical check on all 63 fits:')
log("")
log(f'  max |standardised grad - published grad|    : {D.std_minus_pub_grad.abs().max():.3e}')
log(f'  max |back-transformed beta - published beta|: {D.std_minus_pub_beta.abs().max():.3e}')
log(f'  max |back-transformed SE - published SE|    : {D.std_minus_pub_se.abs().max():.3e}')
log(f'  max |standardised p - published p|          : {D.std_minus_pub_p.abs().max():.3e}')
log(f'  max |equivalent loglike - loglike|          : '
    f'{(D.std_loglike_equivalent - D.pub_loglike).abs().max():.3e}')
log(f'  same optimiser                              : '
    f'{int((D.pub_optimiser == D.std_optimiser).sum())}/{len(D)}')
log(f"  same outcome on singularity                 : "
    f"{int((D.pub_singular == D.std_singular).sum())}/{len(D)}")
log(f'  same outcome on boundary                    : '
    f'{int((D.pub_boundary == D.std_boundary).sum())}/{len(D)}')
log("")
log("The gradient is invariant AT THE SAME POINT. The point at which the")
log('cascade stops, however, is not: dividing y by a constant changes the')
log("last bits of the objective, and on these fits the stopping of `lbfgs` is so")
log('finely balanced that rounding noise decides which optimiser')
log("'converges' first. Where the published fit used `powell` precisely")
log("because `lbfgs` did not converge, on the standardised outcome `lbfgs`")
log('declares convergence instead, and does so at a much worse point.')
log("")
D["_dg"] = D.std_minus_pub_grad.abs()
shift = D[D._dg > 1e-6].sort_values("_dg", ascending=False)
neighbours = D[D._dg <= 1e-6]
log(f'  fits in which standardisation leaves the same point: {len(neighbours)}/{len(D)} (max |delta '
    f'grad| {neighbours._dg.max():.3e}, that is, rounding noise)')
log(f'  fits in which standardisation moves the point      : {len(shift)}/{len(D)}')
log("")
if len(shift):
    log(shift[["cohort", "label", "kinetic", "pub_optimiser", "std_optimiser",
               "pub_grad_inf_norm", "std_grad_inf_norm", "pub_beta",
               "std_beta_backtransformed", "pub_p",
               "std_p"]].to_string(index=False))
    worse = shift[shift.std_grad_inf_norm > shift.pub_grad_inf_norm]
    better = shift[shift.std_grad_inf_norm < shift.pub_grad_inf_norm]
    log("")
    log(f'  standardisation WORSENS the gradient in {len(worse)} fits and improves it in '
        f'{len(better)}.')
    log("  It is therefore not a correction: it is a perturbation of unpredictable sign.")
log("")
log('--- the affected fits, before and after standardisation ---')
log("(beta and SE of the standardised fit are REPORTED IN ORIGINAL UNITS,")
log(" multiplied by the SD of the outcome, hence comparable with the tables)")
log("")
col2 = ["cohort", "label", "kinetic", "outcome_sd",
        "pub_grad_inf_norm", "std_grad_inf_norm",
        "pub_beta", "std_beta_backtransformed",
        "pub_se", "std_se_backtransformed",
        "pub_p", "std_p",
        "pub_singular", "std_singular", "pub_boundary", "std_boundary"]
log(large[col2].to_string(index=False))
log("")
log("THE CORRECTION THAT WORKS: rerun the optimisation from the reported")
log("optimum with the whole cascade, iterating until the log-likelihood")
log("no longer rises.")
log("")
gr = D[D.pub_grad_inf_norm > 1].sort_values("pub_grad_inf_norm",
                                            ascending=False)
col3 = ["cohort", "label", "kinetic", "pub_grad_inf_norm", "pol_grad_inf_norm",
        "pub_newton_decrement", "pol_newton_decrement",
        "pub_loglike", "pol_loglike", "delta_loglike",
        "pub_beta", "pol_beta", "pub_se", "pol_se", "pub_p", "pol_p",
        "pol_optimiser", "pol_rounds",
        "pol_singular", "pol_boundary"]
log(gr[col3].to_string(index=False))
log("")

# --------------------------------------------------------------- STEP 3 ----
log("=" * 78)
log("STEP 3 — STATIONARITY CRITERION APPLIED TO ALL 63")
log("=" * 78)
log("Criterion adopted: NEWTON DECREMENT, delta = 1/2 g'(-H)^-1 g on the")
log("profiled log-likelihood, with threshold 1e-4 — it is the only quantity considered")
log("here that is invariant both to the scale of the outcome and to the")
log("parametrisation, and it is already expressed in the units that matter")
log('(a 95% profile-likelihood confidence limit lies 1.92 away).')
log("")
log("The alternatives, and why not:")
log('  |grad|_inf/nobs        : still depends on the parametrisation;')
log("  |grad|_inf/(nobs x SD) : dividing by the SD is precisely the wrong")
log("                           operation, because the gradient does not depend")
log('                           on the SD (Step 2); it worsens the ranking')
log("                           of the DAT outcomes, which have the smallest SD;")
log('  relative gradient      : reasonable, but it vanishes spuriously when')
log('                           a parameter passes through zero.')
log("")
log('Comparison of the rankings on the 5 fits with |grad| > 1:')
log(gr[["cohort", "label", "kinetic", "outcome_sd", "pub_grad_inf_norm",
        "pub_grad_over_nobs", "pub_grad_over_nobs_sd", "pub_rel_grad",
        "pub_newton_decrement"]].to_string(index=False))
log("")
log(f'--- OUTCOME OF THE CRITERION (threshold {TOL_NEWTON:g}) ---')
log(f'  published fits that satisfy it   : {int(D.pub_stationary.sum())} of {len(D)}')
log(f'  re-optimised fits that satisfy it: {int(D.pol_stationary.sum())} of {len(D)}')
log("")
no = D[~D.pub_stationary].sort_values("pub_newton_decrement", ascending=False)
log(f'--- the {len(no)} published fits that do NOT satisfy the criterion ---')
if len(no):
    log(no[["cohort", "label", "kinetic", "pub_grad_inf_norm",
             "pub_newton_decrement", "pol_newton_decrement",
             "delta_loglike"]].to_string(index=False))
log("")
no_p = D[~D.pol_stationary]
if len(no_p):
    log(f'--- WARNING: {len(no_p)} fits NOT stationary even after re-optimisation ---')
    log(no_p[["cohort", "label", "kinetic", "pol_grad_inf_norm",
               "pol_newton_decrement"]].to_string(index=False))
else:
    log('After re-optimisation all 63 satisfy the criterion.')
log("")
log('Distribution of the Newton decrement over the 63 re-optimised fits:')
log(f'  min {D.pol_newton_decrement.min():.3e}   median {D.pol_newton_decrement.median():.3e}   max '
    f'{D.pol_newton_decrement.max():.3e}')
log('Distribution of |grad|_inf over the 63 re-optimised fits:')
log(f'  min {D.pol_grad_inf_norm.min():.3e}   median {D.pol_grad_inf_norm.median():.3e}   max '
    f'{D.pol_grad_inf_norm.max():.3e}')
log("")
log("Singularity and boundary after re-optimisation:")
log(f'  singular: {int(D.pol_singular.sum())} of {len(D)}   on the boundary: '
    f'{int(D.pol_boundary.sum())} of {len(D)}')
log("")

# --------------------------------------------------------------- STEP 4 ----
log("=" * 78)
log('STEP 4 — DOES ANY CONCLUSION CHANGE?')
log("=" * 78)
log(f'max |change in beta| over the 63: {D.delta_beta.abs().max():.6g}')
log(f'max |change in SE| over the 63  : {D.delta_se.abs().max():.6g}')
log(f'max |change in p| over the 63   : {D.delta_p.abs().max():.6g}')
log(f'fits with |change in p| > 0.001 : {int((D.delta_p.abs() > 0.001).sum())} of {len(D)}')
log("")
for cohort in ["24h", "150h"]:
    s = D[D.cohort == cohort]
    log("-" * 78)
    log(f'COHORT {cohort}: {len(s)} tests, BH correction within cohort')
    log("-" * 78)
    pre = s[s.pub_p_BH_within_cohort < 0.05]
    post = s[s.pol_p_BH_within_cohort < 0.05]
    log(f'  pass BH < 0.05 with the published fits: {len(pre)}')
    log(f'  pass BH < 0.05 after re-optimisation  : {len(post)}')
    show = s[(s.pub_p_BH_within_cohort < 0.20)
               | (s.pol_p_BH_within_cohort < 0.20)].sort_values("pol_p")
    if len(show):
        log("")
        log(show[["label", "kinetic", "n_subjects", "n_obs",
                    "pub_beta", "pol_beta", "pub_se", "pol_se",
                    "pub_p", "pol_p",
                    "pub_p_BH_within_cohort",
                    "pol_p_BH_within_cohort"]].to_string(index=False))
    new = post[post.pub_p_BH_within_cohort >= 0.05]
    lost = pre[pre.pol_p_BH_within_cohort >= 0.05]
    log("")
    if len(new):
        log(f'  *** {len(new)} NEW result(s) above the BH threshold: ***')
        log(new[["label", "kinetic", "pub_p", "pol_p",
                   "pub_p_BH_within_cohort",
                   "pol_p_BH_within_cohort"]].to_string(index=False))
    else:
        log('  no new result passes the BH threshold')
    if len(lost):
        log(f'  *** {len(lost)} LOST result(s): ***')
        log(lost[["label", "kinetic", "pub_p", "pol_p",
                   "pub_p_BH_within_cohort",
                   "pol_p_BH_within_cohort"]].to_string(index=False))
    else:
        log('  no published result falls below the BH threshold')
    log("")

log("-" * 78)
log('APPENDIX — THE SAME PATHOLOGY IN TABLE 7 (script 77)')
log("-" * 78)
log('The technical ladder of Table 7 comes from 77_fmax_dat.csv, which records two')
log('estimates per specification: `beta`/`p`/`llf` from the FIRST optimiser that')
log('converges (the cascade, which an earlier version of the table reported) and')
log("`beta_vmax`/`p_vmax`/`llf_vmax` from the optimiser of MAXIMUM")
log('likelihood among those that converge. Where llf_vmax > llf, the published')
log("row is not at the optimum. The comparison uses numbers already present in the")
log('repository, without re-estimating anything.')
log("")
f77 = os.path.join(OUT, "77_fmax_dat.csv")
if os.path.exists(f77):
    T = pd.read_csv(f77)
    T["delta_llf"] = T.llf_vmax - T.llf
    male = T[T.delta_llf > 1e-3].sort_values("delta_llf", ascending=False)
    log(f'rows with llf_vmax - llf > 1e-3: {len(male)} of {len(T)}')
    log("")
    if len(male):
        log(male[["outcome", "spec", "optimiser", "optimiser_vmax", "llf",
                  "llf_vmax", "delta_llf", "beta", "beta_vmax", "se",
                  "se_vmax", "p", "p_vmax"]].to_string(index=False))
        log("")
        changes = male[(male.p >= 0.05) & (male.p_vmax < 0.05)]
        if len(changes):
            log("*** rows in which the verdict at 0.05 is REVERSED at the true optimum: ***")
            log(changes[["outcome", "spec", "p", "p_vmax"]].to_string(
                index=False))
        else:
            log('no row reverses the verdict at 0.05')
else:
    log(f'WARNING: {f77} does not exist, comparison not carried out')
log("")
log('Independent check: M0, M5, M5a and M6 are rebuilt and the same')
log('polishing used on the 63 is applied. If the polished values coincide with')
log("the *_vmax columns already present in 77_fmax_dat.csv, those columns are")
log("the true optimum and the columns of the cascade are not. Table 7 today")
log('reports the *_vmax columns, by way of 87_amplitude_ladder.csv.')
log("")
SAA_FILE = os.path.join(DATA,
                        "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
N_MIN = 5
DAT = {"MIA_PUTAMEN_BILAT": "DAT putamen",
       "MIA_CAUDATE_BILAT": "DAT caudate"}
COV9 = ["age_bl", "SEX", "dur_bl"]
if os.path.exists(SAA_FILE):
    cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
    cc = cc[cc.complete_primary]
    b9 = cc[["PATNO", "saa_Fmax"]].rename(columns={"saa_Fmax": "Fmax"}).copy()
    b9["logFmax"] = np.log(b9.Fmax)
    b9["Fmax"] = (b9.logFmax - b9.logFmax.mean()) / b9.logFmax.std()
    bl9 = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX",
                                     "duration_yrs"]].rename(
        columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
    b9 = b9.merge(bl9, on="PATNO", how="left")
    saa = pd.read_csv(SAA_FILE)
    sx = saa[(saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")
             & (saa.PATNO.isin(set(b9.PATNO)))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
    b9 = b9.merge(sx[["PATNO", "RUNDATE"]].rename(
        columns={"RUNDATE": "run_raw"}), on="PATNO", how="left")
    b9 = b9.merge(cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO")[
        ["PATNO", "SITE"]], on="PATNO", how="left")
    vr = b9.run_raw.astype(str).value_counts()
    b9["run"] = b9.run_raw.astype(str).where(
        b9.run_raw.astype(str).map(vr) >= N_MIN, 'other')
    vs = b9.SITE.astype(str).value_counts()
    b9["site"] = b9.SITE.astype(str).where(
        b9.SITE.astype(str).map(vs) >= N_MIN, 'other')
    meanrun = b9.groupby("run").logFmax.transform("mean")
    resid = b9.logFmax - meanrun + b9.logFmax.mean()
    b9["Fmax_res"] = (resid - resid.mean()) / resid.std()
    L9 = cur[cur.PATNO.isin(b9.PATNO)][["PATNO", "YEAR"] + list(DAT)].copy()
    L9["years"] = pd.to_numeric(L9.YEAR, errors="coerce")
    L9 = L9[L9.years.notna()].merge(
        b9[["PATNO", "Fmax", "Fmax_res", "run", "site"] + COV9],
        on="PATNO", how="left")
    SPEC9 = [("M0", "Fmax", ""),
             ("M5", "Fmax", " + C(run)*years + C(site)*years"),
             ("M5a", "Fmax", " + C(run)*years"),
             ("M6", "Fmax_res", "")]
    r9 = []
    for out9 in DAT:
        for sig, kin9, extra in SPEC9:
            cols = ["PATNO", "years", out9, kin9] + COV9
            for c in ("run", "site"):
                if f"C({c})" in extra:
                    cols.append(c)
            m9 = L9[list(dict.fromkeys(cols))].dropna()
            f9 = f"{out9} ~ {kin9}*years + " + " + ".join(COV9) + extra
            t9 = f"{kin9}:years"
            ra, meta = cascade(f9, m9, m9.PATNO)
            if ra is None:
                continue
            mda = ra.model
            ga = float(np.max(np.abs(_score(mda, pack(ra)))))
            rb, metb, llb, pkb, _ = polishing(f9, m9, m9.PATNO, ra, mda)
            gb = float(np.max(np.abs(_score(mda, pkb))))
            r9.append({"outcome": out9, "spec": sig,
                       'cascade_opt': meta,
                       'cascade_llf': _ll(mda, pack(ra)),
                       'cascade_grad': ga,
                       'cascade_beta': float(ra.params[t9]),
                       'cascade_se': float(ra.bse[t9]),
                       'cascade_p': float(ra.pvalues[t9]),
                       'polished_opt': metb, 'polished_llf': llb,
                       'polished_grad': gb,
                       'polished_beta': float(rb.params[t9]),
                       'polished_se': float(rb.bse[t9]),
                       'polished_p': float(rb.pvalues[t9])})
            print(f'  . tab9 {out9} {sig}  |g| {ga:.3g} -> {gb:.3g}', flush=True)
    R9 = pd.DataFrame(r9)
    R9["delta_llf"] = R9.polished_llf - R9.cascade_llf
    R9.to_csv(os.path.join(OUT, "84_table9_polished.csv"), index=False)
    log(R9[["outcome", "spec", 'cascade_opt', 'cascade_grad', 'polished_grad',
            'cascade_llf', 'polished_llf', "delta_llf", 'cascade_beta',
            'polished_beta', 'cascade_se', 'polished_se', 'cascade_p',
            'polished_p']].to_string(index=False))
    log("")
    fl = R9[(R9.cascade_p >= 0.05) & (R9.polished_p < 0.05)]
    if len(fl):
        log('*** specifications in which the verdict at 0.05 is REVERSED by bringing the fit to its '
            'maximum: ***')
        log(fl[["outcome", "spec", 'cascade_p', 'polished_p', 'cascade_grad',
                "delta_llf"]].to_string(index=False))
    else:
        log('no specification reverses the verdict at 0.05')
    log("")
    log(f"Saved: {os.path.join(OUT, '84_table9_polished.csv')}")
else:
    log(f'WARNING: {SAA_FILE} not found, check of Table 7 not carried out')
log("")

with open(os.path.join(OUT, "84_gradients.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
log(f"Saved: {os.path.join(OUT, '84_gradients.csv')}")
log(f"Saved: {os.path.join(OUT, '84_gradients.txt')}")
