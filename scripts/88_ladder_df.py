"""
88 - The full ladder of the nested comparisons A1-A5 on the MoCA trajectory
     (150h cohort), with the degrees of freedom taken from the RANK and not from
     the number of columns, and the decomposition of the 4 df comparison.

REASON. The fourth round of review observes that the manuscript attributes to
the shape of the MEAN trajectory the comparison A1 -> A4 (chi2 = 50.2 on 4 df).
That comparison widens two things at once: the basis of mean time and
the basis of the interaction. It therefore does not isolate the non-linearity
of the mean.

This script does one thing only: it redoes the five models (plus A5r at full
rank), measures for each one columns, numerical rank and logL at full maximum
likelihood, and prints ALL the pairwise nested comparisons with the
correct df, first verifying the nesting as containment of column
spaces. It closes with the additive decomposition of the 4 df comparison.

Estimation convention (identical to 78 and 83): the likelihood ratios are
on ML fits (reml=False), the Wald tests and the coefficients on REML fits. Under
REML the likelihoods of models with different fixed effects are not comparable
and statsmodels indeed returns aic/bic = NaN.

Output: run_sporadici/outputs/88_ladder_df.csv
        run_sporadici/outputs/88_ladder_df.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats as st

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


def R(**kw):
    rows.append(kw)


def ps(p):
    return f"{p:.4f}" if (p is None or not np.isfinite(p) or p >= 1e-4) \
        else f"{p:.3e}"


# ===========================================================================
# DATA (identical to 78_longitudinal_sensitivity.py and 83_spline_reconciled.py)
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


def fit_ml(formula):
    last = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, it in OPTIMISERS:
            try:
                r = smf.mixedlm(formula, m, groups=m.PATNO,
                                re_formula="~years").fit(reml=False,
                                                        method=method,
                                                        maxiter=it)
            except Exception:
                continue
            last = (r, method)
            if r.converged:
                return r, method
    return (last[0] if last else None, last[1] if last else None)


def rank(X):
    s = np.linalg.svd(np.asarray(X, dtype=float), compute_uv=False)
    tol = max(X.shape) * np.finfo(float).eps * s[0]
    return int((s > tol).sum()), tol


def content(Xa, Xb):
    """Is the column space of Xa contained in that of Xb? Maximum residual."""
    Xa = np.asarray(Xa, dtype=float)
    Xb = np.asarray(Xb, dtype=float)
    coef = np.linalg.lstsq(Xb, Xa, rcond=None)[0]
    return float(np.max(np.abs(Xb @ coef - Xa)))


MOD = [
    ("A1", 'linear time (reference)',
     "moca ~ T50*years + age_bl + SEX + dur_bl + EDUCYRS"),
    ("A2", 'linear+quadratic, interaction on both',
     "moca ~ T50 + years + years2 + T50:years + T50:years2 "
     "+ age_bl + SEX + dur_bl + EDUCYRS"),
    ("A3", 'linear+quadratic, interaction on the linear only',
     "moca ~ T50 + years + years2 + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS"),
    ("A4", 'natural spline 3 df, interaction on the whole basis',
     "moca ~ T50 + s1 + s2 + s3 + T50:s1 + T50:s2 + T50:s3 "
     "+ age_bl + SEX + dur_bl + EDUCYRS"),
    ("A5", "natural spline 3 df, interaction on the linear only (parametrisation of 78, 'years' "
           "redundant)",
     "moca ~ T50 + s1 + s2 + s3 + years + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS"),
    ("A5r", 'natural spline 3 df, interaction on the linear term only (full rank)',
     "moca ~ T50 + s1 + s2 + s3 + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS"),
]

log("=" * 78)
log('88 - FULL LADDER OF THE NESTED COMPARISONS (MoCA, 150h cohort)')
log("=" * 78)
log(f'Data: {N_SUBJ} subjects, {N_OBSV} observations, years {m.years.min():.0f}-{m.years.max():.0f}.')
log('Likelihood ratios on ML fits (reml=False); Wald tests and coefficients on REML fits.')

A = {}
log("\n" + "-" * 78)
log(f"{'mod':<5s} {'columns':>8s} {'rank':>6s} {'logL ML':>13s} {'logL REML':>13s}  description")
log("-" * 78)
for tag, et, formula in MOD:
    rml, metml = fit_ml(formula)
    rre, metre, _ = fit_mixed(formula, m, m.PATNO)
    X = np.asarray(rml.model.exog, dtype=float)
    k_col = int(X.shape[1])
    k_rank, tol = rank(X)
    A[tag] = {"X": X, "col": k_col, "rank": k_rank, "ml": rml, "reml": rre,
              "et": et, "formula": formula}
    log(f"{tag:<5s} {k_col:>8d} {k_rank:>6d} {rml.llf:>13.4f} "
        f"{rre.llf:>13.4f}  {et}")
    R(section='models', model=tag, term='columns/rank/logL',
      n_subj=N_SUBJ, n_obsv=N_OBSV, k_columns=k_col, k_rank=k_rank,
      logL_ML=float(rml.llf), logL_REML=float(rre.llf),
      convergence=bool(rml.converged and rre.converged),
      optimiser_ML=metml, optimiser_REML=metre,
      tolerance_rank=float(tol), remark=et, formula=formula)

# --- all the pairwise nested comparisons -----------------------------------
log("\n" + "-" * 78)
log('NESTED COMPARISONS (all the pairs with contained column space)')
log("-" * 78)
log(f"{'comparison':<14s} {'df col':>7s} {'df rank':>9s} {'chi2':>10s} {'p (rank)':>12s}  "
    f"{'nesting residual':>19s}")
tag_ord = [t for t, _, _ in MOD]
for a in tag_ord:
    for b in tag_ord:
        if a == b:
            continue
        res = content(A[a]["X"], A[b]["X"])
        if not (res < 1e-8 and A[b]["rank"] > A[a]["rank"]):
            continue
        dof_col = A[b]["col"] - A[a]["col"]
        dof = A[b]["rank"] - A[a]["rank"]
        chi2 = float(2 * (A[b]["ml"].llf - A[a]["ml"].llf))
        p = float(st.chi2.sf(chi2, dof))
        log(f"{a + ' -> ' + b:<14s} {dof_col:>7d} {dof:>9d} {chi2:>10.4f} {ps(p):>12s}  {res:>19.2e}")
        R(section='nested comparison (LRT on ML)', model=f"{a} -> {b}",
          term='likelihood ratio', n_subj=N_SUBJ, n_obsv=N_OBSV,
          stat_joint=chi2, df_columns=dof_col, df_joint=dof,
          p_joint=p, residual_nesting=res,
          convergence=bool(A[a]["ml"].converged and A[b]["ml"].converged))

# --- decomposition of the 4 df comparison ----------------------------------
log("\n" + "-" * 78)
log('DECOMPOSITION OF THE 4 DF COMPARISON')
log("-" * 78)
c15 = float(2 * (A["A5r"]["ml"].llf - A["A1"]["ml"].llf))
c54 = float(2 * (A["A4"]["ml"].llf - A["A5r"]["ml"].llf))
c14 = float(2 * (A["A4"]["ml"].llf - A["A1"]["ml"].llf))
g15 = A["A5r"]["rank"] - A["A1"]["rank"]
g54 = A["A4"]["rank"] - A["A5r"]["rank"]
g14 = A["A4"]["rank"] - A["A1"]["rank"]
log(f'  A1 -> A5r  shape of the MEAN trajectory            chi2={c15:.4f}  df={g15}  '
    f'p={ps(float(st.chi2.sf(c15, g15)))}')
log(f"  A5r -> A4  shape of the INTERACTION                chi2={c54:.4f}  df={g54}  "
    f"p={ps(float(st.chi2.sf(c54, g54)))}")
log(f'  A1 -> A4   the two together                        chi2={c14:.4f}  df={g14}  '
    f'p={ps(float(st.chi2.sf(c14, g14)))}')
log(f'  additive check: {c15:.4f} + {c54:.4f} = {c15 + c54:.4f} against {c14:.4f}  (difference '
    f'{abs(c15 + c54 - c14):.2e}); {g15} + {g54} = {g15 + g54} against {g14}')
log("  The 4 df comparison enlarges together the basis of the mean time and the basis of the "
    "interaction: it does not isolate the non-linearity of the mean. The comparison that isolates "
    "it is A1 -> A5r, with 2 df.")
R(section='decomposition', model="A1 -> A5r", term='mean shape',
  stat_joint=c15, df_joint=g15,
  p_joint=float(st.chi2.sf(c15, g15)))
R(section='decomposition', model="A5r -> A4", term='interaction shape',
  stat_joint=c54, df_joint=g54,
  p_joint=float(st.chi2.sf(c54, g54)))
R(section='decomposition', model="A1 -> A4", term='mean + interaction',
  stat_joint=c14, df_joint=g14,
  p_joint=float(st.chi2.sf(c14, g14)),
  remark=f'additivity: residual {abs(c15 + c54 - c14):.2e}')

# --- ML versus REML: check of the convention ---------------------------------
log("\n" + "-" * 78)
log("ML AGAINST REML: WHY THE LIKELIHOOD RATIOS ARE ON ML")
log("-" * 78)
log(f"{'mod':<5s} {'AIC ML':>10s} {'BIC ML':>10s} {'AIC REML':>10s} "
    f"{'BIC REML':>10s}")
for tag in tag_ord:
    aml, bml = A[tag]["ml"].aic, A[tag]["ml"].bic
    are, bre = A[tag]["reml"].aic, A[tag]["reml"].bic
    log(f"{tag:<5s} {aml:>10.2f} {bml:>10.2f} {str(are):>10s} "
        f"{str(bre):>10s}")
    R(section="ML vs REML", model=tag, term="AIC/BIC",
      aic_ML=float(aml), bic_ML=float(bml),
      aic_REML=float(are) if np.isfinite(are) else np.nan,
      bic_REML=float(bre) if np.isfinite(bre) else np.nan)
c_lin = float(2 * (A["A5r"]["reml"].llf - A["A1"]["reml"].llf))
log(f"  Under REML the difference of logL between A1 and A5r would be {c_lin:.4f}: it is not a valid "
    f"likelihood ratio, because the two restricted likelihoods are computed on different error "
    f"contrasts.")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "88_ladder_df.csv"),
                           index=False)
with open(os.path.join(OUT, "88_ladder_df.txt"), "w") as f:
    f.write("\n".join(rep) + "\n")
log(f"\nWritten: {os.path.join(OUT, '88_ladder_df.csv')} and .txt")
