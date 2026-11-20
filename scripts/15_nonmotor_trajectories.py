"""
15 - Do the kinetic parameters predict the non-motor trajectories?

Structured design, not a blind sweep:
  - ONE predictor only: z(log T50). TTT and AUC are collinear with T50
    (rho 0.94 and 0.96): three tests for the same information would inflate the
    correction without adding signal.
  - ONE primary measure per domain, chosen a priori: 12 outcomes, correctable
    with Benjamini-Hochberg without destroying power.
  - Extended battery (smaller n but follow-up of 10-11 years) reported
    separately as a second level, with its own correction.

Model: outcome ~ T50 x years + age_BL + sex + duration_BL [+ education],
random intercept and slope per participant. 150h cohort (n = 473).

Sign convention: the "effetto_protettivo_%" column is oriented so
that a POSITIVE value always means "higher T50 (slower seeding)
-> slower progression", whatever the direction of the scale.

Output: outputs/15_*.csv, outputs/15_nonmotor.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from _fit import fit_mixed
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
W = os.path.join(BASE, "outputs", "cache") + "/"
rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


cur = pd.read_parquet(W + "curated.parquet")
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete].reset_index(drop=True)
d["T50"] = np.log(d.k_T50)
d["T50"] = (d.T50 - d.T50.mean()) / d.T50.std()

covbl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                   "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base = d[["PATNO", "T50"]].merge(covbl, on="PATNO", how="left")

# (variable, domain, direction: +1 = high score is BETTER, -1 = worse,
#  education covariate yes/no)
PRIMARY = [
    ("moca",                 'Global cognitive',      +1, True),
    ("hvlt_immediaterecall", 'Memory (HVLT)',         +1, True),
    ("SDMTOTAL",             'Executive/attention',   +1, True),
    ("VLTANIM",              'Semantic fluency',      +1, True),
    ("bjlot",                'Visuospatial (BJLOT)',  +1, True),
    ("updrs1_score",         'Global non-motor',    -1, False),
    ("gds",                  'Depression (GDS-15)',   -1, False),
    ("stai",                 'Anxiety (STAI)',           -1, False),
    ("ess",                  'Sleepiness (Epworth)',   -1, False),
    ("rem",                  "RBD (RBDSQ)",            -1, False),
    ("scopa",                'Autonomic (SCOPA-AUT)', -1, False),
    ("quip",                 'Impulse control',      -1, False),
]
SECONDARY = [
    ("COG_COMPOSITE_INT", 'Cognitive: composite',     +1, True),
    ("TMT_B",             'Trail Making B (seconds)', -1, True),
    ("lexical",           'Lexical fluency (FAS)',  +1, True),
    ("MODBNT",            "Boston Naming",            +1, True),
    ("clockdraw",         "Clock Drawing",            +1, True),
    ("nqol_cogns",        'NeuroQoL cognition',      +1, True),
    ("MSEADLG",           "Schwab & England",         +1, False),
]

ALLV = [v for v, _, _, _ in PRIMARY + SECONDARY]
L = cur[cur.PATNO.isin(base.PATNO)][["PATNO", "YEAR"] + ALLV].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")


def fit_one(var, educ):
    cov = ["age_bl", "SEX", "dur_bl"] + (["EDUCYRS"] if educ else [])
    m = L[["PATNO", "years", var, "T50"] + cov].dropna()
    cnt = m.groupby("PATNO").size()
    if len(m) < 200 or (cnt >= 2).sum() < 60:
        return None
    f = f"{var} ~ T50*years + " + " + ".join(cov)
    r, method, _ = fit_mixed(f, m, m.PATNO)
    if r is None:
        return None
    t = "T50:years" if "T50:years" in r.params.index else "years:T50"
    if t not in r.params.index:
        return None
    return {'n_subj': m.PATNO.nunique(), 'n_obsv': len(m),
            'years_max': m.years.max(),
            'slope_mean': r.params["years"], "beta": r.params[t],
            "se": r.bse[t], "p": r.pvalues[t], "conv": r.converged,
            'optimiser': method}


def run(spec, label):
    rows = []
    for var, dom, direction, educ in spec:
        res = fit_one(var, educ)
        if res is None:
            log(f'  [skipped] {var}: insufficient coverage')
            continue
        sl = res['slope_mean']
        # oriented effect: positive = slow seeding -> slower progression
        prot = direction * res["beta"] / abs(sl) * 100 if sl else np.nan
        rows.append({'domain': dom, 'outcome': var, 'n_subj': res['n_subj'],
                     'n_obsv': res['n_obsv'], 'slope_mean': sl,
                     "beta": res["beta"], "se": res["se"],
                     "effetto_protettivo_%": prot, "p": res["p"],
                     'convergence': res["conv"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_BH"] = multipletests(df.p, method="fdr_bh")[1]
    return df


log("=" * 78)
log('DO THE SAA KINETIC PARAMETERS PREDICT THE NON-MOTOR TRAJECTORIES?')
log("=" * 78)
log('Single predictor: z(log T50), 150h cohort.')
log("Sign: effetto_protettivo_% > 0 = slower seeding -> slower progression.")

log("\n" + "-" * 78)
log('PRIMARY LEVEL — one measure per domain (12 outcomes, BH over 12)')
log("-" * 78)
p1 = run(PRIMARY, 'primary')
log(p1[['domain', 'outcome', 'n_subj', 'n_obsv', 'slope_mean', "beta", "se",
        "effetto_protettivo_%", "p", "p_BH"]].round(4).to_string(index=False))
sig1 = p1[p1.p_BH < 0.05]
log(f'\n>>> significant after BH: {len(sig1)} / {len(p1)}')
if len(sig1):
    log(sig1[['domain', 'outcome', "effetto_protettivo_%", "p", "p_BH"]]
        .round(4).to_string(index=False))
p1.to_csv(os.path.join(OUT, "15_primary_nonmotor.csv"), index=False)

log("\n" + "-" * 78)
log('SECONDARY LEVEL — extended battery (separate BH)')
log("-" * 78)
p2 = run(SECONDARY, 'secondary')
log(p2[['domain', 'outcome', 'n_subj', 'n_obsv', 'slope_mean', "beta", "se",
        "effetto_protettivo_%", "p", "p_BH"]].round(4).to_string(index=False))
sig2 = p2[p2.p_BH < 0.05]
log(f'\n>>> significant after BH: {len(sig2)} / {len(p2)}')
p2.to_csv(os.path.join(OUT, "15_secondary_nonmotor.csv"), index=False)

# ---------------------------------------------------------------------------
# Consistency of signs: is the cognitive signal systematic or isolated to MoCA?
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log('CONSISTENCY OF THE SIGNS IN THE COGNITIVE DOMAIN')
log("=" * 78)
cogn = ["moca", "hvlt_immediaterecall", "SDMTOTAL", "VLTANIM", "bjlot"]
sub = p1[p1.outcome.isin(cogn)]
pos = int((sub["effetto_protettivo_%"] > 0).sum())
log(f'Primary cognitive measures with effect in the expected direction: {pos} / {len(sub)}')
log(sub[['outcome', "effetto_protettivo_%", "p"]].round(4).to_string(index=False))
from scipy import stats as st
if len(sub):
    pb = st.binomtest(pos, len(sub), 0.5).pvalue
    log(f'Binomial test on the concordance of the signs: p = {pb:.4f}')
noncogn = p1[~p1.outcome.isin(cogn)]
posn = int((noncogn["effetto_protettivo_%"] > 0).sum())
log(f'\nNON-cognitive measures with effect in the expected direction: {posn} / {len(noncogn)}')
log(noncogn[['outcome', "effetto_protettivo_%", "p"]].round(4).to_string(index=False))
if len(noncogn):
    log(f'Binomial test: p = {st.binomtest(posn, len(noncogn), 0.5).pvalue:.4f}')

# ---------------------------------------------------------------------------
# Is MoCA explained by the specific cognitive tests?
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log('DOES THE SIGNAL ON MoCA HOLD WHEN ADJUSTING FOR THE SPECIFIC TESTS?')
log("=" * 78)
log('If the signal were carried by a specific domain, adding it as a')
log('time-varying covariate should attenuate the coefficient on MoCA.')
rows = []
b0 = None
for extra in [None, "hvlt_immediaterecall", "SDMTOTAL", "VLTANIM", "bjlot"]:
    cols = ["PATNO", "years", "moca", "T50", "age_bl", "SEX", "dur_bl", "EDUCYRS"]
    f = "moca ~ T50*years + age_bl + SEX + dur_bl + EDUCYRS"
    if extra:
        cols.append(extra)
        f += f" + {extra}"
    m = L[cols].dropna()
    try:
        r = smf.mixedlm(f, m, groups=m.PATNO, re_formula="~years").fit(
            method="lbfgs", maxiter=300)
    except Exception:
        continue
    t = "T50:years" if "T50:years" in r.params.index else "years:T50"
    if b0 is None:
        b0 = r.params[t]
    rows.append({'adjustment': extra or 'none (base)',
                 'n_subj': m.PATNO.nunique(), "beta": r.params[t],
                 "variazione_vs_base_%": 100 * (r.params[t] - b0) / b0 if b0 else np.nan,
                 "p": r.pvalues[t]})
med = pd.DataFrame(rows)
log(med.round(4).to_string(index=False))
med.to_csv(os.path.join(OUT, "15_moca_mediation.csv"), index=False)

with open(os.path.join(OUT, "15_nonmotor.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/15_nonmotor.txt')
