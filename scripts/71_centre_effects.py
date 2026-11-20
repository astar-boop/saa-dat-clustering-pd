"""
71 - Centre effects on the kinetic parameters, and their relation with the run.

The review observes that centres, runs and imaging pipelines may be
correlated. The biospecimen file does not report the centre, but the curated
data cut does: one can therefore (a) estimate the proportion of variance of the
kinetic parameters attributable to the centre, (b) check whether centre and
assay run are associated, (c) repeat the estimate of the centre within run and
vice versa.

Output: outputs/71_centre_effects.csv, .txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
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


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO")[["PATNO", "SITE"]]
saa = pd.read_csv(SAA)
cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete]

rows = []
for lab, proj, origin, kin in [
        ("24h (n=595)", 237, cc,
         {"log Fmax": "saa_Fmax", "log TTT": "saa_TTT", "log AUC": "saa_AUC",
          "log SLOPEmax": "saa_SLOPEMax"}),
        ("150h (n=333)", 155, d150,
         {"log Fmax": "k_Fmax", "log T50": "k_T50", "log AUC": "k_AUC",
          "log SLOPE": "k_SLOPE"})]:
    s = saa[(saa.PROJECTID == proj) & (saa.CLINICAL_EVENT == "BL")
            & (saa.PATNO.isin(set(origin.PATNO)))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
    s = s[["PATNO", "RUNDATE"]].rename(columns={"RUNDATE": "_run"})
    duplicates = [c for c in ("SITE", "_run", "RUNDATE") if c in origin.columns]
    m = origin.drop(columns=duplicates).merge(s, on="PATNO").merge(bl, on="PATNO")
    m = m[m.SITE.notna()]
    log(f'\n{lab}: {len(m)} subjects, {m.SITE.nunique()} centres, {m._run.nunique()} runs')

    # centres with few participants grouped together: a centre with a single
    # participant would absorb the data point by construction
    little = m.SITE.value_counts()
    site = m.SITE.where(m.SITE.map(little) >= 5, 'other').astype(str)
    log(f'  centres with at least 5 subjects: {int((little >= 5).sum())} (covering '
        f'{int(little[little >= 5].sum())} subjects)')

    # centre-run association
    tab = pd.crosstab(site, m._run)
    chi2, pchi, dof, _ = stats.chi2_contingency(tab)
    cramer = np.sqrt(chi2 / (len(m) * (min(tab.shape) - 1)))
    log(f'  association centre x run: Cramer V = {cramer:.3f} (chi2 = {chi2:.0f}, df = {dof}, p = '
        f'{pchi:.3g})')

    for name, col in kin.items():
        y = np.log(m[col].to_numpy(float))
        # centre within run: residuals on the run, then proportion due to the centre
        resid = y - pd.Series(y).groupby(m._run.to_numpy()).transform("mean").to_numpy() + y.mean()
        rows.append({'cohort': lab, 'parameter': name, "n": len(m),
                      'n_centres': int(site.nunique()),
                      'eta2_centre': eta2(y, site.to_numpy()),
                      'ICC_centre': icc(y, site.to_numpy()),
                      'eta2_centre_within_run': eta2(resid, site.to_numpy()),
                      'V_cramer_centre_run': cramer})

v = pd.DataFrame(rows)
v.to_csv(os.path.join(OUT, "71_centre_effects.csv"), index=False)
log("\n" + v.round(4).to_string(index=False))
log("\nExpected eta^2 under the null hypothesis, (G-1)/(n-1):")
for lab in v.cohort.unique():
    s = v[v.cohort == lab].iloc[0]
    log(f'  {lab}: {(s.n_centres - 1) / (s.n - 1):.3f}')

with open(os.path.join(OUT, "71_centre_effects.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/71_centre_effects.csv')
