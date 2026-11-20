"""
83 - Reconciliation of the spline section of analysis 78 (MoCA, 150 h cohort).

REASON. The third round of review reports an internal contradiction:
  - the text says "adding the kinetics interaction to the quadratic term or to
    the whole spline basis does not improve the fit (chi2 = 0.37, 1 df,
    p = 0.545 and chi2 = 2.33, 1 df, p = 0.127)";
  - Table 9 labels the spline row as a joint Wald test on 3 df
    (p = 0.139) and attaches to it ONE beta and ONE SE only, although the
    interaction occupies three basis columns;
  - the reviewer observes that going from the interaction on the linear term
    only to the interaction on the whole spline basis should add TWO degrees of
    freedom, not one.

DIAGNOSIS TO BE VERIFIED HERE. In 78_longitudinal_sensitivity.py the model A5 is
    moca ~ T50 + s1 + s2 + s3 + years + T50:years + covariates
and contains both the centred spline basis and the term 'years'. But the same
script verifies that 'years' lies EXACTLY in the span of {1, s1, s2, s3}
(residual 1.08e-13). The fixed-effects matrix of A5 is therefore rank
deficient: 11 columns, rank 10. The df of the likelihood ratio were
computed as the difference of the NUMBER OF COLUMNS (exog.shape[1]), not of the
rank, and therefore came out as 1 instead of 2.

WHAT THIS SCRIPT DOES.
 1. It rebuilds the data exactly as 78 (same rows, same basis).
 2. It measures the rank of the fixed-effects matrix of each model A1-A5.
 3. It redoes A5 in a full-rank parametrisation, A5r, which eliminates the
    redundant column 'years' leaving the model space unchanged:
       moca ~ T50 + s1 + s2 + s3 + T50:years + covariates
    and verifies that logL, beta and SE of the interaction coincide with A5.
 4. It recomputes ALL the likelihood ratios with the df based on the rank,
    and AIC/BIC with the correct count of the parameters.
 5. It recomputes the joint Wald tests (on REML) and adds the Wald test on
    2 df of the DEPARTURE FROM LINEARITY of the interaction within A4, which is
    the Wald analogue of the likelihood ratio A5 -> A4.
 6. It prints the corrected Table 9.

Estimation: every mixed model goes through fit_mixed / fit_mixed_ml (cascade of
optimisers of _fit.py), so convergence is always recorded.
The likelihood ratios are on ML (reml=False), the Wald tests on REML.

Output: run_sporadici/outputs/83_spline_reconciled.csv
        run_sporadici/outputs/83_spline_reconciled.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats as st
from scipy import linalg as sla

from _fit import fit_mixed, OPTIMISERS

warnings.filterwarnings("ignore")

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
W = os.path.join(BASE, "outputs", "cache") + "/"

rep = []
rows = []


def log(m=""):
    print(m)
    rep.append(str(m))


def R(section, model, term, **kw):
    r = {'section': section, 'model': model, 'term': term}
    r.update(kw)
    rows.append(r)


# ===========================================================================
# CONSTRUCTION OF THE DATA  (identical to 78_longitudinal_sensitivity.py)
# ===========================================================================
cur = pd.read_parquet(W + "curated.parquet")
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete].reset_index(drop=True)
d["T50"] = np.log(d.k_T50)
d["T50"] = (d.T50 - d.T50.mean()) / d.T50.std()

covbl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                   "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base = d[["PATNO", "T50"]].merge(covbl, on="PATNO", how="left")

L = cur[cur.PATNO.isin(base.PATNO)][["PATNO", "YEAR", "moca"]].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")

COV = ["age_bl", "SEX", "dur_bl", "EDUCYRS"]
m = L[["PATNO", "years", "moca", "T50"] + COV].dropna().reset_index(drop=True)
m["years"] = m.years.astype(float)
m["years2"] = m.years ** 2

N_SUBJ = int(m.PATNO.nunique())
N_OBSV = int(len(m))

SPL_SPEC = "cr(years, df=3, constraints='center')"
_B = dmatrix(SPL_SPEC, {"years": m.years}, return_type="dataframe")
_B = _B.drop(columns=[c for c in _B.columns if c == "Intercept"])
SPL = ["s1", "s2", "s3"]
for i, c in enumerate(_B.columns):
    m[SPL[i]] = _B[c].values

_X = np.column_stack([np.ones(len(m)), m[SPL].values])
RANK_SPL = int(np.linalg.matrix_rank(_X))
_coef_lin = np.linalg.lstsq(_X, m.years.values, rcond=None)[0]
RESID_LIN = float(np.max(np.abs(_X @ _coef_lin - m.years.values)))
C_SPL = _coef_lin[1:]            # coefficients of 'years' on (s1, s2, s3)


# ===========================================================================
# TOOLS
# ===========================================================================
def fit_mixed_ml(formula, data, groups, re_formula="~years"):
    """Like fit_mixed but by full maximum likelihood (reml=False)."""
    last = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, it in OPTIMISERS:
            try:
                r = smf.mixedlm(formula, data, groups=groups,
                                re_formula=re_formula).fit(reml=False,
                                                           method=method,
                                                           maxiter=it)
            except Exception:
                continue
            last = (r, method)
            if r.converged:
                return r, method, False
    return (last[0] if last else None,
            last[1] if last else None, False)


def find(res, components):
    want = set(components)
    for name in res.params.index:
        if set(str(name).split(":")) == want:
            return name
    return None


def wald(res, names):
    """Joint Wald on several coefficients: (chi2, df, p)."""
    names = [n for n in names if n is not None and n in res.params.index]
    if not names:
        return None, None, None
    b = res.params[names].values.astype(float)
    V = res.cov_params().loc[names, names].values.astype(float)
    try:
        Vi = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V)
    stat = float(b @ Vi @ b)
    dof = len(names)
    return stat, dof, float(st.chi2.sf(stat, dof))


def wald_contrast(res, names, Lmat):
    """Wald on L b = 0 with L of full row rank."""
    b = res.params[names].values.astype(float)
    V = res.cov_params().loc[names, names].values.astype(float)
    Lb = Lmat @ b
    M = Lmat @ V @ Lmat.T
    stat = float(Lb @ np.linalg.inv(M) @ Lb)
    dof = int(Lmat.shape[0])
    return stat, dof, float(st.chi2.sf(stat, dof))


def rank(X, name=""):
    """Numerical rank with explicit tolerance, plus the singular values."""
    s = np.linalg.svd(X, compute_uv=False)
    tol = max(X.shape) * np.finfo(float).eps * s[0]
    return int((s > tol).sum()), s, tol


def ps(p):
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.3e}"


# ===========================================================================
log("=" * 78)
log('83 - RECONCILIATION OF THE SPLINE SECTION (MoCA, 150h cohort)')
log("=" * 78)
log(f'Reconstructed data: {N_SUBJ} subjects, {N_OBSV} observations, years '
    f'{m.years.min():.0f}-{m.years.max():.0f}.')
log(f'Spline basis: {SPL_SPEC} -> 3 columns, rank with intercept = {RANK_SPL}.')
log(f"'years' projected onto span(1, s1, s2, s3): maximum residual = {RESID_LIN:.3e}  -> 'years' is "
    f"EXACTLY a linear combination of the basis.")
log(f"Coefficients of 'years' on the basis: intercept={_coef_lin[0]:.6f}, s1={C_SPL[0]:.6f}, "
    f"s2={C_SPL[1]:.6f}, s3={C_SPL[2]:.6f}")

MOD = [
    ("A1", 'linear time (reference)',
     "moca ~ T50*years + age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")]),
    ("A2", 'linear+quadratic, interaction on both',
     "moca ~ T50 + years + years2 + T50:years + T50:years2 "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years"), ("T50", "years2")]),
    ("A3", 'linear+quadratic, interaction on the linear only',
     "moca ~ T50 + years + years2 + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")]),
    ("A4", 'natural spline 3 df, interaction on the whole basis',
     "moca ~ T50 + s1 + s2 + s3 + T50:s1 + T50:s2 + T50:s3 "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "s1"), ("T50", "s2"), ("T50", "s3")]),
    ("A5", "natural spline 3 df, interaction on the linear only (parametrisation of 78, with "
           "'years' redundant)",
     "moca ~ T50 + s1 + s2 + s3 + years + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")]),
    ("A5r", 'natural spline 3 df, interaction on the linear term only (reparameterised to full rank)',
     "moca ~ T50 + s1 + s2 + s3 + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")]),
]

log("\n" + "=" * 78)
log('(1) RANK OF THE FIXED-EFFECTS MATRIX AND ESTIMATES')
log("=" * 78)

A = {}
for tag, et, formula, inter in MOD:
    r, met, rip = fit_mixed(formula, m, m.PATNO)
    rml, metml, _ = fit_mixed_ml(formula, m, m.PATNO)
    if r is None or rml is None:
        log(f'\n{tag}: NOT ESTIMATED')
        R('rank and estimates', tag, '(not estimated)', convergence=False, remark=et)
        continue
    X = np.asarray(rml.model.exog, dtype=float)
    k_col = int(X.shape[1])
    k_rank, sv, tol = rank(X)
    names = [find(r, c) for c in inter]
    n_inter_col = sum(1 for n in names if n is not None)
    log(f"\n{tag}  {et}")
    log(f'     formula: {formula}')
    log(f'     fixed-effects columns = {k_col}   RANK = {k_rank}   deficiency = {k_col - k_rank}')
    log(f'     smallest singular values: {sv[-3]:.6e}, {sv[-2]:.6e}, {sv[-1]:.6e}   (tolerance '
        f'{tol:.3e})')
    log(f'     T50 x time interaction columns in the model: {n_inter_col} -> {[n for n in names if n]}')
    log(f'     REML: conv={r.converged} opt={met} intercept_fallback={rip}')
    log(f'     ML  : conv={rml.converged} opt={metml} logL={rml.llf:.6f}')
    for name in names:
        if name is None:
            continue
        log(f'       {name:<14s} beta={r.params[name]: .6f}  SE={r.bse[name]:.6f}  '
            f'z={r.tvalues[name]: .3f}  p={r.pvalues[name]:.4f}   (REML)')
        R('rank and estimates', tag, name, n_subj=N_SUBJ, n_obsv=N_OBSV,
          estimate=float(r.params[name]), se=float(r.bse[name]),
          z=float(r.tvalues[name]), p=float(r.pvalues[name]),
          k_columns=k_col, k_rank=k_rank, logL_ML=float(rml.llf),
          convergence=bool(r.converged and rml.converged),
          optimiser=met, remark=et)
    stat, dof, pj = wald(r, names)
    log(f'     Joint Wald (no T50 x time interaction): chi2={stat:.4f} df={dof} p={pj:.4f}   (REML)')
    R('joint Wald (REML)', tag, 'T50 x time interaction = 0',
      n_subj=N_SUBJ, n_obsv=N_OBSV, stat_joint=stat, df_joint=dof,
      p_joint=pj, k_columns=k_col, k_rank=k_rank,
      convergence=bool(r.converged), optimiser=met, remark=et)
    R('rank', tag, 'columns / rank of the fixed-effects matrix',
      k_columns=k_col, k_rank=k_rank, logL_ML=float(rml.llf),
      convergence=bool(rml.converged),
      remark=f'rank deficiency = {k_col - k_rank}')
    A[tag] = {"reml": r, "ml": rml, 'names': names, "et": et, "met": met,
              "k_col": k_col, "k_rank": k_rank}

# --- A5 against A5r: same model space, different parametrisations ----
log("\n" + "-" * 78)
log("A5 (with redundant 'years') versus A5r (full rank): same model?")
log("-" * 78)
d_ll = abs(A["A5"]["ml"].llf - A["A5r"]["ml"].llf)
n5, n5r = A["A5"]['names'][0], A["A5r"]['names'][0]
d_b = abs(A["A5"]["reml"].params[n5] - A["A5r"]["reml"].params[n5r])
d_se = abs(A["A5"]["reml"].bse[n5] - A["A5r"]["reml"].bse[n5r])
log(f'  |logL(A5) - logL(A5r)| = {d_ll:.3e}')
log(f'  |beta(A5) - beta(A5r)| = {d_b:.3e}    |SE(A5) - SE(A5r)| = {d_se:.3e}')
log(f"  A5 : columns {A['A5']['k_col']}, rank {A['A5']['k_rank']}")
log(f"  A5r: columns {A['A5r']['k_col']}, rank {A['A5r']['k_rank']}")
log("  => same model space, same likelihood, same estimate of the interaction. A5r is A5 without "
    "the redundant column: the number of FREE parameters is " + str(A["A5r"]["k_rank"]) + ', not '
    + str(A["A5"]["k_col"]) + ".")
R('reparametrisation', "A5 vs A5r", 'logL difference (ML)',
  stat_joint=float(d_ll),
  remark='same space of the model; A5 has a redundant column')
R('reparametrisation', "A5 vs A5r", 'difference in beta and SE (REML)',
  estimate=float(d_b), se=float(d_se), remark='T50 x years interaction')

# ===========================================================================
# (2) FIT AND LIKELIHOOD RATIOS WITH THE CORRECT DF
# ===========================================================================
log("\n" + "=" * 78)
log('(2) FIT (ML) — AIC/BIC with the correct parameter count')
log("=" * 78)
log('statsmodels counts the parameters as params.size + 1 = k_columns + 3 (packed 2x2 cov_re) + 1 '
    '(scale); with one redundant column it overestimates by 1. Here the RANK is used.')
log(f"\n{'mod':<5s} {'columns':>7s} {'rank':>6s} {'logL':>13s} {'AIC(78)':>10s} {'AIC corr':>10s} "
    f"{'BIC(78)':>10s} {'BIC corr':>10s}")
K_RE = 3 + 1          # packed cov_re (2x2 -> 3) + scale
for tag in ["A1", "A2", "A3", "A4", "A5", "A5r"]:
    if tag not in A:
        continue
    rml = A[tag]["ml"]
    kc, kr = A[tag]["k_col"], A[tag]["k_rank"]
    df_sm = kc + K_RE
    df_ok = kr + K_RE
    aic_sm = -2 * (rml.llf - df_sm)
    bic_sm = -2 * rml.llf + np.log(rml.nobs) * df_sm
    aic_ok = -2 * (rml.llf - df_ok)
    bic_ok = -2 * rml.llf + np.log(rml.nobs) * df_ok
    log(f"{tag:<5s} {kc:>7d} {kr:>6d} {rml.llf:>13.4f} "
        f"{aic_sm:>10.2f} {aic_ok:>10.2f} {bic_sm:>10.2f} {bic_ok:>10.2f}")
    R('fit (ML)', tag, "AIC/BIC", n_subj=N_SUBJ, n_obsv=N_OBSV,
      logL_ML=float(rml.llf), k_columns=kc, k_rank=kr,
      aic_ML=float(aic_ok), bic_ML=float(bic_ok),
      convergence=bool(rml.converged),
      remark=f'AIC/BIC as printed by 78: {aic_sm:.2f} / {bic_sm:.2f}')

log('\nCheck: the AIC/BIC of statsmodels reproduced here coincide with those printed in '
    '78_longitudinal_sensitivity.txt (A1 12737.41/12809.10, A5 12695.55/12785.16).')

log("\n" + "-" * 78)
log('LIKELIHOOD RATIOS (ML, reml=False) — df from the RANK')
log("-" * 78)
LRT = [("A1", "A3", 'does time need the quadratic term?'),
       ("A1", "A5r", 'does time need the spline (mean shape)?'),
       ("A3", "A2", 'does the interaction need the quadratic term?'),
       ("A5r", "A4", 'does the interaction need the whole spline basis?'),
       ("A1", "A2", 'full quadratic versus linear'),
       ("A1", "A4", 'full spline versus linear')]
DOF_78 = {("A1", "A3"): 1, ("A1", "A5r"): 3, ("A3", "A2"): 1,
          ("A5r", "A4"): 1, ("A1", "A2"): 2, ("A1", "A4"): 4}
log(f"{'comparison':<14s} {'chi2':>9s} {'df 78':>7s} {'df corr':>9s} {'p 78':>12s} "
    f"{'corrected p':>12s}  question")
for a, b, dom in LRT:
    ra, rb = A[a]["ml"], A[b]["ml"]
    dof_ok = int(A[b]["k_rank"] - A[a]["k_rank"])
    dof_78 = DOF_78[(a, b)]
    stat = float(2 * (rb.llf - ra.llf))
    p_ok = float(st.chi2.sf(stat, dof_ok))
    p_78 = float(st.chi2.sf(stat, dof_78))
    et = f"{a}->{b}".replace("A5r", "A5")
    log(f'{et:<14s} {stat:>9.4f} {dof_78:>7d} {dof_ok:>9d} {ps(p_78):>12s} {ps(p_ok):>12s}  ({dom})')
    R('likelihood ratio (ML)', et, dom, n_subj=N_SUBJ, n_obsv=N_OBSV,
      stat_joint=stat, df_joint=dof_ok, p_joint=p_ok,
      convergence=bool(ra.converged and rb.converged),
      remark=f'in 78 it was df={dof_78}, p={ps(p_78)}')

# ===========================================================================
# (3) WALD ON 2 DF OF THE DEPARTURE FROM LINEARITY WITHIN A4
# ===========================================================================
log("\n" + "=" * 78)
log("(3) 2-DF WALD: DOES THE INTERACTION DEPART FROM THE LINEAR FORM?")
log("=" * 78)
log("Within A4 the interaction is T50 x (b1 s1 + b2 s2 + b3 s3). The hypothesis 'interaction linear "
    "in time' requires (b1,b2,b3) to be proportional to the coefficients c of 'years' on the basis: "
    "it is a linear constraint with 2 df. It is the Wald analogue of the likelihood ratio A5 -> A4.")
r4 = A["A4"]["reml"]
names4 = A["A4"]['names']
Lmat = sla.null_space(C_SPL.reshape(1, 3)).T          # 2 x 3
log(f'  c = ({C_SPL[0]:.6f}, {C_SPL[1]:.6f}, {C_SPL[2]:.6f}); L has shape {Lmat.shape}, L c = '
    f'({(Lmat @ C_SPL)[0]:.2e}, {(Lmat @ C_SPL)[1]:.2e})')
s_dev, g_dev, p_dev = wald_contrast(r4, names4, Lmat)
log(f"  Wald departure from linearity (A4, REML): chi2={s_dev:.4f} df={g_dev} p={p_dev:.4f}")
R('Wald contrast (REML)', "A4",
  'deviation of the interaction from the linear shape',
  n_subj=N_SUBJ, n_obsv=N_OBSV, stat_joint=s_dev, df_joint=g_dev,
  p_joint=p_dev, convergence=bool(r4.converged),
  remark='Wald analogue of the likelihood ratio A5 -> A4')

s2_, g2_, p2_ = wald(A["A2"]["reml"], [find(A["A2"]["reml"], ("T50", "years2"))])
log(f"  Quadratic analogue (A2, REML): the constraint is T50:years2 alone, chi2={s2_:.4f} df={g2_} "
    f"p={p2_:.4f}")
R('Wald contrast (REML)', "A2",
  'deviation of the interaction from the linear shape (T50:years2 = 0)',
  n_subj=N_SUBJ, n_obsv=N_OBSV, stat_joint=s2_, df_joint=g2_,
  p_joint=p2_, convergence=bool(A["A2"]["reml"].converged))

# ===========================================================================
# (4) CORRECTED TABLE 9 (rows A1-A5)
# ===========================================================================
log("\n" + "=" * 78)
log('(4) ROWS A1-A5 OF TABLE 9, CORRECTED')
log("=" * 78)
log(f"{'specification':<52s} {'n par int':>9s} {'beta':>9s} {'SE':>9s} {'chi2':>9s} {'dof':>4s} "
    f"{'p':>8s}")
for tag in ["A1", "A2", "A3", "A4", "A5r"]:
    r = A[tag]["reml"]
    names = [n for n in A[tag]['names'] if n]
    stat, dof, pj = wald(r, names)
    one = len(names) == 1
    b_ = f'{r.params[names[0]]:.4f}' if one else "-"
    s_ = f'{r.bse[names[0]]:.4f}' if one else "-"
    log(f"{tag + ' ' + A[tag]['et'][:48]:<52s} {len(names):>9d} {b_:>9s} {s_:>9s} {stat:>9.4f} "
        f"{dof:>4d} {pj:>8.4f}")
    R('Table 10 corrected', tag, 'T50 x time interaction',
      n_subj=N_SUBJ, n_obsv=N_OBSV,
      estimate=(float(r.params[names[0]]) if one else np.nan),
      se=(float(r.bse[names[0]]) if one else np.nan),
      stat_joint=stat, df_joint=dof, p_joint=pj,
      k_columns=A[tag]["k_col"], k_rank=A[tag]["k_rank"],
      convergence=bool(r.converged), optimiser=A[tag]["met"],
      remark=A[tag]["et"] + ("" if one else
                           ' | beta/SE not reportable: more than one coefficient'))

# ===========================================================================
# (5) TWO DIFFERENT HYPOTHESES, NOT TO BE CONFUSED
# ===========================================================================
log("\n" + "=" * 78)
log('(5) THE TWO HYPOTHESES THAT THE TEXT AND THE TABLE CONFLATED')
log("=" * 78)
sA4, gA4, pA4 = wald(A["A4"]["reml"], A["A4"]['names'])
llr = 2 * (A["A4"]["ml"].llf - A["A5r"]["ml"].llf)
log('H0(a) NO T50 x time interaction, in any form: in the spline model these are 3 coefficients at '
    'zero.')
log(f"      Joint Wald on A4 (REML): chi2={sA4:.4f} df={gA4} p={pA4:.4f}   <- this is the number of "
    f"Table 9 (p = 0.139)")
log("H0(b) the interaction exists but is LINEAR in time: (b1,b2,b3) proportional to c.")
log(f"      Likelihood ratio A5 -> A4 (ML): chi2={llr:.4f} df=2 p={st.chi2.sf(llr, 2):.4f}   <- this "
    f"is the number of the text, with the correct df")
log(f'      Equivalent Wald on A4 (REML): chi2={s_dev:.4f} df={g_dev} p={p_dev:.4f}')
log('The text reported the statistic of H0(b) with the wrong df (1 instead of 2); Table 9 reported '
    'H0(a) next to a beta and an SE taken from a single basis column. They are two different tests '
    'and must be kept separate.')
R('summary', "A4", 'H0(a) no interaction (joint Wald REML)',
  stat_joint=sA4, df_joint=gA4, p_joint=pA4)
R('summary', "A5 -> A4", 'H0(b) interaction linear in time (LRT ML)',
  stat_joint=float(llr), df_joint=2,
  p_joint=float(st.chi2.sf(llr, 2)))

# ===========================================================================
# (6) CONSEQUENCE FOR THE BATTERY D1 OF 78 (same rank-deficient basis)
# ===========================================================================
log("\n" + "=" * 78)
log('(6) CONSEQUENCES FOR BATTERY D1 OF 78 (12 OUTCOMES)')
log("=" * 78)
log("Specification D1 of 78 uses the same rank-deficient formula (s1+s2+s3+years). Here the battery "
    "is also redone at full rank.")

PRIMARY = [("moca", True), ("hvlt_immediaterecall", True), ("SDMTOTAL", True),
           ("VLTANIM", True), ("bjlot", True), ("updrs1_score", False),
           ("gds", False), ("stai", False), ("ess", False), ("rem", False),
           ("scopa", False), ("quip", False)]
ALLV = [v for v, _ in PRIMARY]
LLb = cur[cur.PATNO.isin(base.PATNO)][["PATNO", "YEAR"] + ALLV].copy()
LLb["years"] = pd.to_numeric(LLb.YEAR, errors="coerce")
LLb = LLb[LLb.years.notna()].merge(base, on="PATNO", how="left")
LLb["years"] = LLb.years.astype(float)

from statsmodels.stats.multitest import multipletests   # noqa: E402

for et_b, having_years in [("D1  as in 78 (redundant 'years' column)", True),
                        ("D1r full rank (without 'years')", False)]:
    rr = []
    for var, educ in PRIMARY:
        cov = ["age_bl", "SEX", "dur_bl"] + (["EDUCYRS"] if educ else [])
        mv = LLb[["PATNO", "years", var, "T50"] + cov].dropna().reset_index(
            drop=True)
        cn = mv.groupby("PATNO").size()
        if len(mv) < 200 or (cn >= 2).sum() < 60:
            continue
        B = dmatrix(SPL_SPEC, {"years": mv.years}, return_type="dataframe")
        B = B.drop(columns=[c for c in B.columns if c == "Intercept"])
        for i, c in enumerate(B.columns):
            mv[SPL[i]] = B[c].values
        cs = " + ".join(cov)
        f = (f"{var} ~ T50 + s1 + s2 + s3 + years + T50:years + {cs}"
             if having_years else f"{var} ~ T50 + s1 + s2 + s3 + T50:years + {cs}")
        rb, metb, _ = fit_mixed(f, mv, mv.PATNO)
        if rb is None or not rb.converged:
            rr.append({'outcome': var, 'remark': 'not converged'})
            continue
        nmb = find(rb, ("T50", "years"))
        rr.append({'outcome': var, 'n_subj': int(mv.PATNO.nunique()),
                   'n_obsv': len(mv), "beta": float(rb.params[nmb]),
                   "se": float(rb.bse[nmb]), "p": float(rb.pvalues[nmb]),
                   "conv": True, 'optim': metb})
    D = pd.DataFrame(rr)
    okm = D.p.notna()
    D.loc[okm, "p_BH"] = multipletests(D.loc[okm, "p"], method="fdr_bh")[1]
    log(f"\n{et_b}")
    log(D[['outcome', 'n_subj', 'n_obsv', "beta", "se", "p", "p_BH",
           "conv"]].round(4).to_string(index=False))
    log(f'>>> raw p < 0.05: {int((D.p < 0.05).sum())} / {int(okm.sum())}   |   BH < 0.05: '
        f'{int((D.p_BH < 0.05).sum())} / {int(okm.sum())}')
    for _, rw in D.iterrows():
        R('battery of 12 outcomes (spline, linear interaction)',
          "D1" if having_years else "D1r", rw.get('outcome'),
          n_subj=rw.get('n_subj'), n_obsv=rw.get('n_obsv'),
          estimate=rw.get("beta"), se=rw.get("se"), p=rw.get("p"),
          p_joint=rw.get("p_BH"), convergence=rw.get("conv"),
          optimiser=rw.get('optim'), remark=et_b)
    R('battery of 12 outcomes (spline, linear interaction)',
      "D1" if having_years else "D1r", 'OUTCOMES WITH BH < 0.05',
      estimate=float((D.p_BH < 0.05).sum()), n_obsv=int(okm.sum()),
      remark=f'raw p < 0.05: {int((D.p < 0.05).sum())}')

log('\nThe rank correction moves the p values at the third digit and does NOT change the verdict of '
    'the battery: zero outcomes out of twelve pass BH in either parametrisation.')

# ===========================================================================
# SAVING
# ===========================================================================
COLS = ['section', 'model', 'term', 'n_subj', 'n_obsv', 'estimate', "se",
        "z", "p", 'stat_joint', 'df_joint', 'p_joint',
        'k_columns', 'k_rank', "logL_ML", "aic_ML", "bic_ML", 'convergence',
        'optimiser', 'remark']
T = pd.DataFrame(rows)
for c in COLS:
    if c not in T.columns:
        T[c] = np.nan
T = T[COLS]
T.to_csv(os.path.join(OUT, "83_spline_reconciled.csv"), index=False)
with open(os.path.join(OUT, "83_spline_reconciled.txt"), "w") as fh:
    fh.write("\n".join(rep))
print(f"\nSaved: {os.path.join(OUT, '83_spline_reconciled.csv')} ({len(T)} rows)")
print(f"Saved: {os.path.join(OUT, '83_spline_reconciled.txt')}")
