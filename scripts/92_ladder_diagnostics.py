"""
92 - The technical ladder of Table 7 redone and diagnosed IN FULL under the
     strict rule: all sixteen rows, not only the four M0.

WHY THIS SCRIPT EXISTS
----------------------
The caption of Table 7 declared that the technical ladder had been
computed with SELECTION ON THE LIKELIHOOD only (among the optimisers
that converge, the one with the highest log-likelihood) without the
subsequent round of restarts, and that only the four M0 rows had been
compared with the strict refit. The twelve M4-M6 rows, however, are those that
determine the attenuations reported in §3.12 and §4.7 and the statement that the
two striatal associations remain nominally significant under all the
sensitivity analyses for the recorded technical factors. Their saved object
(outputs/87_amplitude_ladder.csv) records the optimiser chosen and the
likelihood, but not the attempts per optimiser, the gain of the
restarts, the gradient, the Newton decrement, nor the diagnostics of
singularity and of boundary.

This script redoes ALL SIXTEEN rows — four outcome x kinetic parameter pairs
(striatum/putamen x Fmax/SLOPEmax) at the specifications M0, M4, M5 and M6 —
under the complete strict rule, and produces an object with the diagnostics per
row.

STRICT RULE (identical to the one declared in §2.18 and implemented in
88_diagnostics_63.procedura)
----------------------------------------------------------------------
Stage 1 - cascade. The same model (random intercept and random slope per
  subject, REML) is estimated with each of the four optimisers of
  _fit.OTTIMIZZATORI - lbfgs (300 iterations), powell (2000), nelder-mead
  (5000), bfgs (2000) - starting from the initial values of statsmodels. For
  each one `converged` and the profiled log-likelihood are recorded, the latter
  evaluated on A SINGLE model instance so that the four numbers are
  comparable bit for bit.
Stage 2 - selection. Among those that converge the highest log-likelihood
  is kept (not the first that converges).
Stage 3 - restarts. From that point all four optimisers start again
  (`start_params`), accepting every gain above 1e-10,
  until none gains any more (at most 10 rounds).
Stationarity criterion: the Newton decrement
    delta = 1/2 * g' (-H)^{-1} g,     threshold  delta < 1e-4
with g the analytical score of statsmodels in the packed space in which
the optimiser works and H by central differences of the same
profiled log-likelihood. The detail of the formulas is in the docstring of
88_diagnostics_63.py: here the primitives are COPIED WITHOUT CHANGES, so that
the two scripts return the same number on the same fit.

NOTE ON THE SIZE OF THE PROBLEM. M5 has many fixed effects (run x years and
centre x years), but MixedLM profiles out the fixed effects via GLS: the space
in which the optimiser works still has three free parameters (the Cholesky
factor of the 2x2 covariance of the random effects, in the scale of the residual
variance). The cost of M5 lies in the single likelihood computation, not in the
size of the problem. The times are in any case recorded per row.

DATA. The construction of the cohort, the coding of run and centre, the
standardisation of the kinetic parameters and the residualisation by run are
reproduced to the letter from 87_amplitude_ladder.py.

COMPARISON. Each row reports the PUBLISHED value (from
outputs/87_amplitude_ladder.csv) next to the re-estimated one, with absolute
and relative differences on beta, standard error and p.

Output: outputs/92_ladder_diagnostics.csv, outputs/92_ladder_diagnostics.txt
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS  # same cascade, same order, same maxiter
from _paths import DATA
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")

TOL_NEWTON = 1e-4
TOL_SINGULAR = 1e-8
TOL_CORR = 0.999
TOL_GAIN = 1e-10
MAX_ROUNDS = 10

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# ------------------------------------------------------------- primitives ---
# Copied verbatim from 88_diagnostics_63.py.
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
    """Hessian by central differences of the profiled log-likelihood,
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
    """Stages 1-3 of the strict rule. Copied verbatim from 88."""
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
    conv = [(m, v[1]) for m, v in attempts.items()
            if v[0] is not None and v[2] and np.isfinite(v[1])]
    if conv:
        chosen = max(conv, key=lambda x: x[1])[0]
        everyone_conv = True
    else:
        fin = [(m, v[1]) for m, v in attempts.items()
               if v[0] is not None and np.isfinite(v[1])]
        if not fin:
            return None
        chosen = max(fin, key=lambda x: x[1])[0]
        everyone_conv = False
    r0, ll0, _ = attempts[chosen]
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
            'md_ref': md_ref, 'selection_between_convergent': everyone_conv}


def diagnostics(r, t):
    """Copied verbatim from 88."""
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


# ---------------------------------------------------------------- data ------
# Reproduced to the letter from 87_amplitude_ladder.py.
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
saa = pd.read_csv(SAA)
bl_site = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO")[["PATNO", "SITE"]]

s = saa[(saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")
        & (saa.PATNO.isin(set(cc.PATNO)))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
tech = s[["PATNO", "RUNDATE", "InstrumentRep1"]].rename(
    columns={"RUNDATE": "run", "InstrumentRep1": 'instr'}).merge(
    bl_site, on="PATNO", how="left")


def collapse(v, n_min=5):
    v = pd.Series(v).astype(str)
    counts = v.value_counts()
    return v.where(v.map(counts) >= n_min, 'other').to_numpy()


KIN = {"Fmax": "saa_Fmax", "SLOPE": "saa_SLOPEMax"}
base = cc[["PATNO"] + list(KIN.values())].copy()
for k, col in KIN.items():
    base[k] = np.log(base[col].to_numpy(float))
    base[k] = (base[k] - base[k].mean()) / base[k].std()
bl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX",
                                "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base = base.merge(bl, on="PATNO", how="left").merge(tech, on="PATNO", how="left")
base["run"] = collapse(base.run)
base["site"] = collapse(base.SITE)

for k in KIN:
    med = base.groupby("run")[k].transform("mean")
    r_ = base[k] - med + base[k].mean()
    base[k + "_res"] = (r_ - r_.mean()) / r_.std()

OUTCOMES = {"MIA_CAUDATE_BILAT": "DAT caudate, bilateral",
         "MIA_PUTAMEN_BILAT": "DAT putamen, bilateral"}
L = cur[cur.PATNO.isin(set(base.PATNO))][["PATNO", "YEAR"] + list(OUTCOMES)].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")

SPEC = [
    ("M0", 'published', "{k}*years + age_bl + SEX + dur_bl"),
    ("M4", '+ run + centre', "{k}*years + age_bl + SEX + dur_bl + C(run) + C(site)"),
    ("M5", '+ run x years + centre x years',
     "{k}*years + age_bl + SEX + dur_bl + C(run)*years + C(site)*years"),
    ("M6", 'on kinetic parameter residualised for run',
     "{k}_res*years + age_bl + SEX + dur_bl"),
]

PUB = pd.read_csv(os.path.join(OUT, "87_amplitude_ladder.csv"))
PUBI = PUB.set_index(['outcome', 'kinetic', "spec"])

log("=" * 78)
log('92 — THE TECHNICAL LADDER OF TABLE 7, ALL SIXTEEN ROWS UNDER THE')
log('     STRICT RULE, WITH FULL DIAGNOSTICS')
log("=" * 78)
log(f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}")
log(f'BASE = {BASE}')
log(f'cascade: {OPTIMISERS}')
log('stage 1: all the optimisers from the starting values of statsmodels')
log("stage 2: among those that CONVERGE, the highest log-likelihood")
log(f'stage 3: restarts from the chosen point (max {MAX_ROUNDS} rounds), criterion Newton decrement '
    f'< {TOL_NEWTON:g}')
log('REML estimation (statsmodels default); primitives copied from 88_diagnostics_63.py')
log(f'24 h cohort: {L.PATNO.nunique()} subjects; {base.run.nunique()} run categories, '
    f'{base.site.nunique()} centre categories')
log("")

TIMES = []
rows = []
for outcome, lbls in OUTCOMES.items():
    for k in KIN:
        for spec, descr, tpl in SPEC:
            var = f"{k}_res" if spec == "M6" else k
            cols = ["PATNO", "years", outcome, var, "age_bl", "SEX", "dur_bl",
                    "run", "site"]
            m = L[cols].dropna()
            f = f'{outcome} ~ ' + tpl.format(k=k)
            t0 = time.time()
            P = procedure(f, m, m.PATNO)
            if P is None:
                log(f'  {lbls} x {k} {spec}: no fit')
                continue
            rf = P['r_final']
            tf = (f"{var}:years" if f"{var}:years" in rf.params.index
                  else f"years:{var}")
            rc = P['r_cascade']
            tc = (f"{var}:years" if f"{var}:years" in rc.params.index
                  else f"years:{var}")
            Dc = diagnostics(rc, tc)
            Df = diagnostics(rf, tf)
            sec = time.time() - t0
            TIMES.append(sec)

            row = {
                "outcome": outcome, "label": lbls, "kinetic": k,
                "specification": spec, "description": descr,
                "n_subjects": int(m.PATNO.nunique()), "n_obs": int(len(m)),
                "n_fixed_effects": int(rf.model.k_fe),
            }
            for method, _ in OPTIMISERS:
                r_, ll_, cv_ = P['attempts'].get(method, (None, np.nan, False))
                row[f'loglike_{method}'] = ll_
                row[f'converged_{method}'] = cv_
            row["n_optimisers_tried"] = len(OPTIMISERS)
            row["n_optimisers_converged"] = int(sum(
                bool(P['attempts'].get(mm, (None, np.nan, False))[2])
                for mm, _ in OPTIMISERS))
            row["selection_among_converged"] = bool(
                P['selection_between_convergent'])
            row["optimiser_retained"] = P['opt_chosen']
            row["loglike_before_restart"] = P['ll_choice']
            row["loglike_after_restart"] = P['ll_final']
            row["restart_gain"] = P['ll_final'] - P['ll_choice']
            row["optimiser_restart"] = P['opt_reopt']
            row["restart_rounds"] = P['rounds']

            for key, val in Df.items():
                row[key] = val
            row["meets_newton_criterion"] = bool(
                np.isfinite(Df["newton_decrement"])
                and Df["newton_decrement"] < TOL_NEWTON)
            row["grad_inf_norm_before_restart"] = Dc["grad_inf_norm"]
            row["newton_decrement_before_restart"] = Dc["newton_decrement"]

            b, se, p = Df["beta_interaction"], Df["se"], Df["p"]
            row["ci_lo"] = b - 1.96 * se
            row["ci_hi"] = b + 1.96 * se

            pb = PUBI.loc[(outcome, k, spec)]
            row["published_beta"] = float(pb["beta"])
            row["published_se"] = float(pb["se"])
            row["published_ci_lo"] = float(pb["ci_lo"])
            row["published_ci_hi"] = float(pb["ci_hi"])
            row["published_p"] = float(pb["p"])
            row["published_optimiser"] = str(pb['optimiser'])
            row["published_llf"] = float(pb["llf"])
            row["delta_beta_abs"] = b - row["published_beta"]
            row["delta_beta_rel"] = ((b - row["published_beta"])
                                      / abs(row["published_beta"]))
            row["delta_se_abs"] = se - row["published_se"]
            row["delta_se_rel"] = ((se - row["published_se"])
                                    / row["published_se"])
            row["delta_p_abs"] = p - row["published_p"]
            row["delta_p_rel"] = (p - row["published_p"]) / row["published_p"]
            row["delta_loglike_vs_published"] = (P['ll_final']
                                                  - row["published_llf"])
            rows.append(row)
            log(f"  {lbls:24s} x {k:5s} {spec}  opt={P['opt_chosen']:6s} conv "
                f"{row['n_optimisers_converged']}/4  gain {row['restart_gain']:.3g}  |g| "
                f"{Df['grad_inf_norm']:.3g}  newton {Df['newton_decrement']:.3g}  beta {b:+.6f} "
                f"(publ {row['published_beta']:+.6f})  p {p:.5f} (publ {row['published_p']:.5f})  "
                f"[{sec:.0f}s]")

D = pd.DataFrame(rows)

# attenuation relative to M0, within outcome x kinetic parameter pair
D["attenuation_pct_vs_M0"] = np.nan
D["published_attenuation_pct_vs_M0"] = np.nan
for (o, k), g in D.groupby(["outcome", "kinetic"]):
    b0 = float(g.loc[g.specification == "M0", "beta_interaction"].iloc[0])
    p0 = float(g.loc[g.specification == "M0", "published_beta"].iloc[0])
    D.loc[g.index, "attenuation_pct_vs_M0"] = (
        100.0 * (1.0 - g.beta_interaction.abs() / abs(b0)))
    D.loc[g.index, "published_attenuation_pct_vs_M0"] = (
        100.0 * (1.0 - g.published_beta.abs() / abs(p0)))

COLS = ["outcome", "label", "kinetic", "specification", "description",
        "n_subjects", "n_obs", "n_fixed_effects",
        "n_optimisers_tried"] \
    + [f"loglike_{mm}" for mm, _ in OPTIMISERS] \
    + [f"converged_{mm}" for mm, _ in OPTIMISERS] \
    + ["n_optimisers_converged", "selection_among_converged",
       "optimiser_retained", "loglike_before_restart", "loglike_after_restart",
       "restart_gain", "optimiser_restart", "restart_rounds",
       "grad_inf_norm", "grad_check_analytic_vs_central", "newton_decrement",
       "meets_newton_criterion", "grad_inf_norm_before_restart",
       "newton_decrement_before_restart",
       "singular", "boundary", "re_var_intercept", "re_var_slope",
       "re_corr_intercept_slope", "residual_scale",
       "profile_hessian_neg_definite", "profile_hessian_max_eigenvalue",
       "fe_cov_pos_def", "fe_cov_cond", "converged", "loglike",
       "beta_interaction", "se", "ci_lo", "ci_hi", "p",
       "published_beta", "published_se", "published_ci_lo", "published_ci_hi",
       "published_p", "published_optimiser", "published_llf",
       "delta_beta_abs", "delta_beta_rel", "delta_se_abs", "delta_se_rel",
       "delta_p_abs", "delta_p_rel", "delta_loglike_vs_published",
       "attenuation_pct_vs_M0", "published_attenuation_pct_vs_M0"]
D = D[COLS]
D.to_csv(os.path.join(OUT, "92_ladder_diagnostics.csv"), index=False)

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 80)

log("")
log("-" * 78)
log('SELECTION AND RESTARTS')
log("-" * 78)
log(f'rows estimated: {len(D)} (expected 16)')
log(f'optimiser retained: {D.optimiser_retained.value_counts().to_dict()}')
for method, _ in OPTIMISERS:
    log(f"  converges in {int(D[f'converged_{method}'].sum())}/{len(D)} rows: {method}")
log(f'rows in which the selection is made among converged optimisers: '
    f'{int(D.selection_among_converged.sum())}/{len(D)}')
log(f'gain from the restarts — max {D.restart_gain.max():.6g}, rows with gain > 1e-6: '
    f'{int((D.restart_gain > 1e-06).sum())}/{len(D)}')
log(f'restart rounds: {D.restart_rounds.value_counts().to_dict()}')
# The times stay in the .txt, not in the CSV: the package promises that on
# rerunning the same checksums are found, and the wall time would make that false.
log(f'time per row — min {min(TIMES):.0f}s  median {sorted(TIMES)[len(TIMES) // 2]:.0f}s  max '
    f'{max(TIMES):.0f}s  total {sum(TIMES) / 60:.1f} min')
log("")
log("-" * 78)
log("STATIONARITY AND STRUCTURE")
log("-" * 78)
log(f'Newton decrement — min {D.newton_decrement.min():.3e}  median '
    f'{D.newton_decrement.median():.3e}  max {D.newton_decrement.max():.3e}')
log(f'rows that meet the criterion (< {TOL_NEWTON:g}): '
    f'{int(D.meets_newton_criterion.sum())}/{len(D)}')
log(f'infinity norm of the gradient — min {D.grad_inf_norm.min():.3e}  median '
    f'{D.grad_inf_norm.median():.3e}  max {D.grad_inf_norm.max():.3e}')
log(f'maximum discrepancy analytic score vs central differences: '
    f'{D.grad_check_analytic_vs_central.max():.3e}')
log(f'singular rows: {int(D.singular.sum())}/{len(D)};  on the boundary: '
    f'{int(D.boundary.sum())}/{len(D)}')
log(f'profiled Hessian negative definite: {int(D.profile_hessian_neg_definite.sum())}/{len(D)}')
log(f'covariance of the fixed effects positive definite: {int(D.fe_cov_pos_def.sum())}/{len(D)}')
log(f'random intercept variance: min {D.re_var_intercept.min():.6g}  max '
    f'{D.re_var_intercept.max():.6g}')
log(f'random slope variance    : min {D.re_var_slope.min():.6g}  max {D.re_var_slope.max():.6g}')
log(f'|correlation| between random effects: min {D.re_corr_intercept_slope.abs().min():.4f}  max '
    f'{D.re_corr_intercept_slope.abs().max():.4f}')
log("")
log("-" * 78)
log('RE-ESTIMATED AGAINST PUBLISHED (87_amplitude_ladder.csv)')
log("-" * 78)
log(f'max |delta beta| : {D.delta_beta_abs.abs().max():.3e}   max |relative delta| : '
    f'{D.delta_beta_rel.abs().max():.3e}')
log(f'max |delta SE|   : {D.delta_se_abs.abs().max():.3e}   max |relative delta| : '
    f'{D.delta_se_rel.abs().max():.3e}')
log(f'max |delta p|    : {D.delta_p_abs.abs().max():.3e}   max |relative delta| : '
    f'{D.delta_p_rel.abs().max():.3e}')
log(f'max |delta logL| : {D.delta_loglike_vs_published.abs().max():.3e}')
log("")
log(D[["label", "kinetic", "specification", "optimiser_retained",
       "published_optimiser", "beta_interaction", "published_beta",
       "se", "published_se", "p", "published_p",
       "newton_decrement", "restart_gain"]].to_string(index=False))
log("")
log("-" * 78)
log("NOMINAL SIGNIFICANCE BY ROW (uncorrected p)")
log("-" * 78)
for (o, k), g in D.groupby(["outcome", "kinetic"], sort=False):
    g = g.set_index("specification")
    alive = [sp for sp in ("M0", "M4", "M5", "M6")
            if sp in g.index and g.loc[sp, "p"] < 0.05]
    alive_pub = [sp for sp in ("M0", "M4", "M5", "M6")
                if sp in g.index and g.loc[sp, "published_p"] < 0.05]
    log(f"  {g.label.iloc[0]:24s} x {k:6s}  re-estimated: "
        f"{(', '.join(alive) if alive else 'nothing')}   |   published: "
        f"{(', '.join(alive_pub) if alive_pub else 'nothing')}")
log("")
log("-" * 78)
log('ATTENUATION RELATIVE TO M0 (percentage reduction of |beta|)')
log("-" * 78)
log(D[D.specification != "M0"][
    ["label", "kinetic", "specification", "attenuation_pct_vs_M0",
     "published_attenuation_pct_vs_M0"]].to_string(index=False))
log("")

with open(os.path.join(OUT, "92_ladder_diagnostics.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
log(f"Saved: {os.path.join(OUT, '92_ladder_diagnostics.csv')}")
log(f"Saved: {os.path.join(OUT, '92_ladder_diagnostics.txt')}")
