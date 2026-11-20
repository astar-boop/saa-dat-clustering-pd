"""
105 - Sensitivity to the optimiser of the run x time test on the caudate.

It reruns, in the primary cohort of sporadic cases only, the same check
already documented for the putamen: the mixed model

    MIA_CAUDATE_BILAT ~ C(run) * years

with random intercept and random slope per subject and REML estimation is
passed, from the initial values of statsmodels, to the four optimisers and in
the same order as `_fit.OTTIMIZZATORI`. For each fit it records convergence,
log-likelihood and joint Wald on the 20 run x years terms. It then compares the
first optimiser that declares convergence with the convergent one that reaches
the highest log-likelihood.

Output:
  outputs/105_caudate_optimiser_sensitivity.csv
  outputs/105_caudate_optimiser_sensitivity.txt
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS
from _paths import DATA
from _assay_run import row_of_kinetics


warnings.filterwarnings("ignore")

BASE = os.environ.get(
    "CLUSTERING_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA_FILE = os.path.join(
    DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
N_MIN = 5
OUTCOME = "MIA_CAUDATE_BILAT"


def collapse_run(s):
    s = s.astype(str)
    frequencies = s.value_counts()
    return s.where(s.map(frequencies) >= N_MIN, 'other')


def wald_run_x_timeval(res):
    names = [n for n in res.fe_params.index if "C(run)" in n and ":" in n]
    b = res.fe_params[names].to_numpy()
    cov = res.cov_params()
    if isinstance(cov, pd.DataFrame):
        V = cov.loc[names, names].to_numpy()
    else:
        idx = [list(res.fe_params.index).index(n) for n in names]
        V = np.asarray(cov)[np.ix_(idx, idx)]
    chi2 = float(b @ np.linalg.pinv(V) @ b)
    dof = len(names)
    return chi2, dof, float(stats.chi2.sf(chi2, dof))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
cc = cc[cc.complete_primary].copy()

saa = pd.read_csv(SAA_FILE)
saa = saa[(saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")
          & (saa.PATNO.isin(set(cc.PATNO)))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
saa = saa[["PATNO", "RUNDATE"]].rename(columns={"RUNDATE": "run_raw"})

base = cc[["PATNO"]].merge(saa, on="PATNO", how="left")
base["run"] = collapse_run(base.run_raw)

long = cur[cur.PATNO.isin(set(cc.PATNO))][
    ["PATNO", "YEAR", OUTCOME]].copy()
long["years"] = pd.to_numeric(long.YEAR, errors="coerce")
long = long.merge(base[["PATNO", "run"]], on="PATNO", how="left")
model_data = long[["PATNO", "years", OUTCOME, "run"]].dropna()

formula = f"{OUTCOME} ~ C(run)*years"
rows = []
results = {}
for optimiser, maxiter in OPTIMISERS:
    try:
        res = smf.mixedlm(
            formula, model_data, groups=model_data.PATNO,
            re_formula="~years").fit(
                method=optimiser, maxiter=maxiter, reml=True)
    except Exception as exc:
        rows.append({
            "optimiser": optimiser,
            "maxiter": maxiter,
            "converged": False,
            "exception": type(exc).__name__,
        })
        continue
    chi2, dof, p = wald_run_x_timeval(res)
    results[optimiser] = res
    rows.append({
        "optimiser": optimiser,
        "maxiter": maxiter,
        "converged": bool(res.converged),
        "loglike_reml": float(res.llf),
        "run_x_time_chi2": chi2,
        "df": dof,
        "p": p,
        "exception": "",
    })

table = pd.DataFrame(rows)
converged = table[table.converged & table.loglike_reml.notna()].copy()
if converged.empty:
    raise RuntimeError('None of the four optimisers converged')

order = [name for name, _ in OPTIMISERS]
first_name = next(name for name in order
                  if name in set(converged.optimiser))
best_name = str(converged.loc[converged.loglike_reml.idxmax(), "optimiser"])
first = converged.set_index("optimiser").loc[first_name]
best = converged.set_index("optimiser").loc[best_name]

table["is_first_converged"] = table.optimiser.eq(first_name)
table["is_highest_likelihood_converged"] = table.optimiser.eq(best_name)
table.to_csv(
    os.path.join(OUT, "105_caudate_optimiser_sensitivity.csv"),
    index=False)

lines = [
    '105 - SENSITIVITY TO THE OPTIMISER: RUN x TIME ON THE CAUDATE',
    f"BASE = {BASE}",
    f'model = {formula}; REML; re_formula = ~years',
    (f'sample = {model_data.PATNO.nunique()} subjects, {len(model_data)} observations, '
     f'{model_data.run.nunique()} run categories'),
    f'order of the optimisers = {order}',
    "",
    table[["optimiser", "converged", "loglike_reml", "run_x_time_chi2",
           "df", "p"]].to_string(index=False),
    "",
    (f'first converged = {first_name}: logL = {first.loglike_reml:.9f}, Wald chi2 = '
     f'{first.run_x_time_chi2:.9f}, df = {int(first.df)}, p = {first.p:.9f}'),
    (f'maximum log-likelihood among the converged = {best_name}: logL = {best.loglike_reml:.9f}, '
     f'Wald chi2 = {best.run_x_time_chi2:.9f}, df = {int(best.df)}, p = {best.p:.9f}'),
    (f'difference in log-likelihood = {best.loglike_reml - first.loglike_reml:.9f}'),
    (f'absolute difference of the Wald chi2 = {abs(best.run_x_time_chi2 - first.run_x_time_chi2):.9f}'),
    (f'absolute difference in p = {abs(best.p - first.p):.9f}'),
]
with open(os.path.join(
        OUT, "105_caudate_optimiser_sensitivity.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f'\nWritten in {OUT}: 105_caudate_optimiser_sensitivity.csv/.txt')
