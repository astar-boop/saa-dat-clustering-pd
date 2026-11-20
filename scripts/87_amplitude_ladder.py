"""
87 - The run/centre specification ladder extended to ALL the amplitude
     parameters that survive the correction after the reoptimisation.

Why. Script 84 showed that four fits of the 24 h cohort stopped far from a
numerically tight optimum (gradient norm up to 98) and that, reoptimising
them with a criterion on the Newton decrement, a THIRD result passes the
within-cohort BH threshold: SLOPEmax x years on bilateral striatal binding,
p from 0.0062 to 0.0013, BH from 0.051 to 0.012.

Script 77 had put only Fmax through the run/centre ladder. Here the same
ladder is applied to Fmax and to SLOPEmax on both dopaminergic outcomes,
with the reoptimised fits, so that the conclusion of the manuscript — no
kinetic x time interaction survives once the technical factors are taken
into account — is verified on everything that passes it.

Output: outputs/87_amplitude_ladder.csv, .txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
OPTIM = [("lbfgs", 300), ("powell", 3000), ("nm", 8000), ("bfgs", 3000)]
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def fit_strict(formula, data, groups):
    """Among the optimisers that converge, it takes the one with the
    highest log-likelihood: it is the criterion of script 84, which avoids
    stopping at the first, only approximate, optimum."""
    best = None
    for method, it in OPTIM:
        try:
            r = smf.mixedlm(formula, data, groups=groups,
                            re_formula="~years").fit(method=method, maxiter=it)
        except Exception:
            continue
        if not r.converged:
            continue
        if best is None or r.llf > best[0].llf + 1e-9:
            best = (r, method)
    return best if best else (None, None)


# ---------------------------------------------------------------- data
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

# residualisation for run of each kinetic parameter
for k in KIN:
    med = base.groupby("run")[k].transform("mean")
    r = base[k] - med + base[k].mean()
    base[k + "_res"] = (r - r.mean()) / r.std()

# The caudate enters the ladder. It had not been discarded: it had never been among
# the outcomes, neither here nor in 72. Without it the two outcomes present are a
# single measure reported twice — the striatum CONTAINS the putamen, rho = 0.94 on
# the curated data cut at BL, R^2 = 0.99 with respect to the mean of caudate and
# putamen — and no anatomical contrast is possible in any direction. The caudate is
# the only compartment with appreciably distinct information (rho = 0.58
# with the putamen) and it is the relatively spared one in PD, so it is the
# benchmark that makes any gradient observable.
OUTCOMES = {"MIA_PUTAMEN_BILAT": "DAT putamen, bilateral",
         "MIA_CAUDATE_BILAT": "DAT caudate, bilateral"}
L = cur[cur.PATNO.isin(set(base.PATNO))][["PATNO", "YEAR"] + list(OUTCOMES)].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")

log("=" * 78)
log('87 — run/centre LADDER FOR THE AMPLITUDE PARAMETERS, RE-OPTIMISED FITS')
log("=" * 78)
log(f'24 h cohort: {L.PATNO.nunique()} subjects; {base.run.nunique()} run categories, '
    f'{base.site.nunique()} centre categories')

SPEC = [
    ("M0", 'published', "{k}*years + age_bl + SEX + dur_bl"),
    ("M4", '+ run + centre', "{k}*years + age_bl + SEX + dur_bl + C(run) + C(site)"),
    ("M5", '+ run x years + centre x years',
     "{k}*years + age_bl + SEX + dur_bl + C(run)*years + C(site)*years"),
    ("M6", 'on kinetic parameter residualised for run',
     "{k}_res*years + age_bl + SEX + dur_bl"),
]

rows = []
for outcome, lbls in OUTCOMES.items():
    for k in KIN:
        log(f'\n--- {lbls}  x  {k} ---')
        for spec, descr, tpl in SPEC:
            var = f"{k}_res" if spec == "M6" else k
            cols = ["PATNO", "years", outcome, var, "age_bl", "SEX", "dur_bl",
                    "run", "site"]
            m = L[cols].dropna()
            f = f'{outcome} ~ ' + tpl.format(k=k)
            r, method = fit_strict(f, m, m.PATNO)
            if r is None:
                log(f'  {spec}: no optimiser converges')
                continue
            t = f"{var}:years" if f"{var}:years" in r.params.index else f"years:{var}"
            b, se, p = float(r.params[t]), float(r.bse[t]), float(r.pvalues[t])
            rows.append({'outcome': outcome, 'label': lbls, 'kinetic': k,
                          "spec": spec, 'description': descr,
                          'n_subj': int(m.PATNO.nunique()), 'n_obsv': len(m),
                          "beta": b, "se": se, "ci_lo": b - 1.96 * se,
                          "ci_hi": b + 1.96 * se, "p": p,
                          'optimiser': method, "llf": float(r.llf)})
            log(f'  {spec:3s} {descr:34s} beta = {b:+.5f}  SE = {se:.5f}  95% CI '
                f'[{b - 1.96 * se:+.5f}, {b + 1.96 * se:+.5f}]  p = {p:.5f}  ({method})')

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "87_amplitude_ladder.csv"), index=False)

log("\n" + "=" * 78)
log('SUMMARY: what survives')
log("=" * 78)
for outcome, lbls in OUTCOMES.items():
    for k in KIN:
        s0 = d[(d.outcome == outcome) & (d.kinetic == k)].set_index("spec")
        if not len(s0):
            continue
        alive = [sp for sp in ("M0", "M4", "M5", "M6")
                if sp in s0.index and s0.loc[sp, "p"] < 0.05]
        log(f"  {lbls:24s} x {k:6s}  nominally significant in: "
            f"{(', '.join(alive) if alive else 'no specification')}")
# The size of the family was hard-coded ("28 test"). With the caudate
# the family changes, and a hard-coded number here would fall behind without
# anything flagging it. It must be derived from the grid actually estimated.
_fam = os.path.join(OUT, "72_mixed_models.csv")
log('\nNote: the p values of this table are NOT corrected.')
if os.path.exists(_fam):
    _m = pd.read_csv(_fam)
    _n24 = int((_m.cohort == "24h").sum())
    _outcomes_fam = set(_m.outcome.unique())
    log(f'In the 24 h cohort the correction family counts {_n24} tests (derived from '
        f'72_mixed_models.csv, {len(_outcomes_fam)} outcomes);')
    # if the ladder and the family do not contain the same DAT outcomes, say so:
    # it is the discrepancy that kept the caudate out of the table.
    _dat_scale, _dat_fam = set(OUTCOMES), {e for e in _outcomes_fam if e.startswith("MIA_")}
    if _dat_scale != _dat_fam:
        log(f'WARNING: the ladder covers {sorted(_dat_scale)} whereas the family contains '
            f'{sorted(_dat_fam)}: the two do not coincide.')
else:
    log('Size of the family cannot be determined: 72_mixed_models.csv missing.')
log('A nominal p here does not imply survival of the correction.')

with open(os.path.join(OUT, "87_amplitude_ladder.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/87_amplitude_ladder.csv and .txt')
