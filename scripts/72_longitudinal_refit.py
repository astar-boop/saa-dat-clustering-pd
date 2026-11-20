"""
72 - Re-estimation of the longitudinal mixed models with a convergence check.

Script 11 fits with `lbfgs` only and records `convergence` without using it.
In the resulting table 13 fits out of 63 do not converge, and among these is
one of the two results that pass the correction for multiple comparisons
(Fmax x years on the bilateral putamen, 24-hour cohort). Here the whole grid
is redone with the cascade of optimisers of `_fit.py`, which optimiser
converged is recorded, and the Benjamini-Hochberg correction is recomputed
WITHIN COHORT, which is the family actually used (28 tests in the 24-hour
cohort, 35 in the 150-hour one) and not the 63 tests overall.

Output: outputs/72_mixed_models.csv, outputs/72_refit.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import fit_mixed

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))

OUTCOMES = {
    "updrs3_score": ("MDS-UPDRS III", ["age_bl", "SEX", "dur_bl"]),
    "updrs_totscore": ("MDS-UPDRS total", ["age_bl", "SEX", "dur_bl"]),
    "updrs2_score": ("MDS-UPDRS II", ["age_bl", "SEX", "dur_bl"]),
    "moca": ("MoCA", ["age_bl", "SEX", "dur_bl", "EDUCYRS"]),
    "MIA_STRIATUM_BILAT": ("DAT striatum, bilateral", ["age_bl", "SEX", "dur_bl"]),
    "MIA_PUTAMEN_BILAT": ("DAT putamen, bilateral", ["age_bl", "SEX", "dur_bl"]),
    # The caudate was missing. It had not been discarded: it had never been included.
    # It is the only compartment with appreciably distinct information
    # (rho = 0.58 with the putamen, against 0.94 between striatum and putamen: the
    # striatum CONTAINS the putamen). Without it the striatum/putamen pair is a
    # single measure reported twice, and no anatomical contrast is possible.
    #
    # WARNING: adding it enlarges the within-cohort BH family from 28 to 32
    # tests in the 24h and from 35 to 40 in the 150h. The correction becomes
    # stricter for ALL the results, including those that survived before.
    "MIA_CAUDATE_BILAT": ("DAT caudate, bilateral", ["age_bl", "SEX", "dur_bl"]),
    "scopa": ("SCOPA-AUT", ["age_bl", "SEX", "dur_bl"]),
}
KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}


def build(cohort_file, kin_map):
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


def fit_all(L, kins, tag):
    rows = []
    for out, (label, cov) in OUTCOMES.items():
        for k in kins:
            m = L[["PATNO", "years", out, k] + cov].dropna()
            multi = m.groupby("PATNO").size()
            if len(m) < 120 or int((multi >= 2).sum()) < 40:
                continue
            f = f"{out} ~ {k}*years + " + " + ".join(cov)
            r, method, fellback = fit_mixed(f, m, m.PATNO)
            if r is None:
                log(f'  [{tag}] {out} x {k}: no optimiser converges')
                continue
            t = f"{k}:years" if f"{k}:years" in r.params.index else f"years:{k}"
            rows.append({"cohort": tag, "outcome": out, "label": label,
                         "kinetic": k, "n_subjects": int(m.PATNO.nunique()),
                         "n_obs": int(len(m)),
                         "n_multivisit": int((multi >= 2).sum()),
                         "beta": float(r.params[t]), "se": float(r.bse[t]),
                         "ci_lo": float(r.params[t] - 1.96 * r.bse[t]),
                         "ci_hi": float(r.params[t] + 1.96 * r.bse[t]),
                         "p": float(r.pvalues[t]),
                         "mean_slope": float(r.params["years"]),
                         "converged": bool(r.converged),
                         "optimiser": method})
    d = pd.DataFrame(rows)
    d["p_BH_within_cohort"] = multipletests(d.p, method="fdr_bh")[1]
    return d


log("=" * 78)
log('72 — REFIT OF THE LONGITUDINAL MODELS WITH CONVERGENCE CHECKED')
log("=" * 78)
L150 = build("08_dataset_150h.csv", KIN150)
L24 = build("analysis_complete_cases.csv", KIN24)
d150 = fit_all(L150, list(KIN150.values()), "150h")
d24 = fit_all(L24, list(KIN24.values()), "24h")
everyone = pd.concat([d150, d24], ignore_index=True)
everyone.to_csv(os.path.join(OUT, "72_mixed_models.csv"), index=False)

log(f'\nmodels fitted: {len(everyone)}  (150h {len(d150)}, 24h {len(d24)})')
log(f'not converged after the cascade: {int((~everyone.converged).sum())}')
log(f'optimiser that converged: {everyone.optimiser.value_counts().to_dict()}')

for tag, d in [("150h", d150), ("24h", d24)]:
    sig = d[d.p_BH_within_cohort < 0.05]
    log(f'\n--- {tag}: {len(sig)} of {len(d)} pass BH within cohort ---')
    if len(sig):
        log(sig[["label", "kinetic", "n_subjects", "n_obs", "beta", "se",
                 "ci_lo", "ci_hi", "p", "p_BH_within_cohort",
                 "converged", "optimiser"]].round(5).to_string(index=False))

# comparison with the previous version
prior = pd.read_csv(os.path.join(OUT, "11_mixed_models.csv"))
v = prior.rename(columns={'cohort': "cohort", 'outcome': "outcome",
                            'kinetic': "kinetic",
                            'beta_interaction': "beta_old", "p": "p_old",
                            "p_BH": "p_BH_old", 'convergence': "conv_old"})
cmp = everyone.merge(v[["cohort", "outcome", "kinetic", "beta_old", "p_old",
                     "p_BH_old", "conv_old"]],
                  on=["cohort", "outcome", "kinetic"], how="left")
cmp["delta_beta"] = cmp.beta - cmp.beta_old
changed = cmp[(cmp.conv_old == False)]
log(f'\n--- the {len(changed)} fits that did not converge in the previous version ---')
log(changed[["cohort", "outcome", "kinetic", "beta_old", "p_old", "beta",
              "p", "optimiser"]].round(5).to_string(index=False))
log(f'\nmaximum deviation of beta between the two versions: {cmp.delta_beta.abs().max():.5f}')

with open(os.path.join(OUT, "72_refit.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/72_mixed_models.csv')
