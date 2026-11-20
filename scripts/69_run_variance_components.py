"""
69 - Variance of the kinetic parameters attributable to the assay run and to
     the instrument, with the SAME estimator in the two cohorts.

Script 23 had computed the variance component of the assay run only for the
150 h cohort; script 66 reports eta squared, which with many groups is
biased upwards. Here both indices are computed in the two cohorts,
so that the comparison is readable:

  eta^2  share of the sum of squares between runs (descriptive, biased
         upwards by about (G-1)/(n-1) under the null hypothesis)
  ICC    variance component from a random-effects model on the run
         (var_run / (var_run + var_residua)), with shrinkage

Output: outputs/69_run_variance_components.csv, .txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from _paths import DATA
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def eta2(y, g):
    g = np.asarray(g)
    tot = np.sum((y - y.mean()) ** 2)
    between = sum(len(y[g == u]) * (y[g == u].mean() - y.mean()) ** 2
              for u in np.unique(g))
    return float(between / tot) if tot > 0 else np.nan


def icc(y, g):
    d = pd.DataFrame({"y": y, "g": pd.Series(g).astype(str)})
    r = smf.mixedlm("y ~ 1", d, groups=d.g).fit(method="lbfgs")
    vr, ve = float(r.cov_re.iloc[0, 0]), float(r.scale)
    return vr / (vr + ve) if (vr + ve) > 0 else np.nan


saa = pd.read_csv(SAA)
cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete]

rows = []
for lab, proj, ids, kin, origin in [
        ("24h (n=595)", 237, set(cc.PATNO),
         {"log Fmax": "saa_Fmax", "log TTT": "saa_TTT", "log AUC": "saa_AUC",
          "log SLOPEmax": "saa_SLOPEMax"}, cc),
        ("150h (n=333)", 155, set(d150.PATNO),
         {"log Fmax": "k_Fmax", "log T50": "k_T50", "log AUC": "k_AUC",
          "log SLOPE": "k_SLOPE"}, d150)]:
    s = saa[(saa.PROJECTID == proj) & (saa.CLINICAL_EVENT == "BL")
            & (saa.PATNO.isin(ids))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
    s = s[["PATNO", "RUNDATE", "InstrumentRep1"]].rename(
        columns={"RUNDATE": "_run", "InstrumentRep1": '_instr'})
    # any columns of the same name already present in the source dataset
    # would be renamed by the merge: dedicated names are used
    duplicates = [c for c in ("_run", '_instr', "RUNDATE", "InstrumentRep1")
              if c in origin.columns]
    m = origin.drop(columns=duplicates).merge(s, on="PATNO", how="inner")
    m = m.rename(columns={"_run": "RUNDATE", '_instr': "InstrumentRep1"})
    log(f'\n{lab}: {len(m)} subjects, {m.RUNDATE.nunique()} runs, {m.InstrumentRep1.nunique()} '
        f'instruments')
    for name, col in kin.items():
        y = np.log(m[col].to_numpy(float))
        rows.append({'cohort': lab, 'parameter': name, "n": len(m),
                      "n_run": int(m.RUNDATE.nunique()),
                      "eta2_run": eta2(y, m.RUNDATE.to_numpy()),
                      "ICC_run": icc(y, m.RUNDATE.to_numpy()),
                      'eta2_instrument': eta2(y, m.InstrumentRep1.to_numpy()),
                      'ICC_instrument': icc(y, m.InstrumentRep1.to_numpy())})

v = pd.DataFrame(rows)
v.to_csv(os.path.join(OUT, "69_run_variance_components.csv"), index=False)
log("\n" + v.round(4).to_string(index=False))
log("\nExpected eta^2 under the null hypothesis of no run effect:")
for lab in v.cohort.unique():
    s = v[v.cohort == lab].iloc[0]
    log(f"  {lab}: (G-1)/(n-1) = {(s.n_run - 1) / (s.n - 1):.3f}")

with open(os.path.join(OUT, "69_run_variance_components.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/69_run_variance_components.csv')
