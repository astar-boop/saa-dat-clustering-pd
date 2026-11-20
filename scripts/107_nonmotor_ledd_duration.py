"""
107 - Non-motor outcomes: what changes when the dopaminergic dose (LEDD) and
      disease duration are taken into account.

Why it exists. The first reviewer writes, about the twelve non-motor outcomes
of Table S9, that "the effect of disease duration and LEDD is not considered"
and that "what consists as a significant or not significant change is not
mentioned". In the submitted analysis (15_nonmotor_trajectories.py, redone
under the strict criterion in 91_table6_strict_criterion.py) disease duration
at baseline IS a covariate of every model, but only as a main effect; the
duration x years term had been examined for the MoCA only (13_moca_signal.py)
and time-varying LEDD for the motor outcomes only (11_longitudinal.py, a single
optimiser). This script does on the twelve non-motor outcomes what was missing,
redoes under the strict criterion the two sensitivity analyses already cited in
the supplement (covariate x years on the MoCA; LEDD on the motor outcomes) and
first of all checks that it starts exactly from the numbers already produced by
the pipeline.

What is computed, all on the 150 h cohort (Project 155), sporadic only:

  0. Identity checks before any slow model: baseline covariates, LEDD and
     treatment indicator at baseline against the cohort file; bounds on the
     follow-up LEDD; reproduction of the fifteen motor models with LEDD of
     11_ledd_adjusted.csv (the only reference, an indirect one, for the
     follow-up LEDD).
  1. Reproduction of Table S9. The twelve models
         outcome ~ z(log T50) x years + baseline age + sex + baseline duration
                 [+ education for the cognitive outcomes]
     with random intercept and random slope per subject, REML, fitted (a) with
     the rule with which the table was produced (_fit.fit_mixed: first
     optimiser that converges) and (b) with the strict criterion of the project
     (among the optimisers that converge the highest log-likelihood, restarts
     until none gains, Newton decrement < 1e-4). The script STOPS, outcome by
     outcome, as soon as (a) does not coincide with 15_primary_nonmotor.csv or
     (b) with 91_table6_strict_criterion.csv.
  2. Seven alternative specifications, each on the twelve outcomes, all under
     the strict criterion:
       A  base + time-varying LEDD (per 100 mg/day, at the visit)
       B  base + baseline duration x years
       C  base + LEDD + baseline duration x years
       D  duration at the visit in place of baseline duration
       E  duration from symptoms (baseline age - age at onset) and its
          interaction with years, in place of duration from diagnosis
       F  base + LEDD + LEDD x years
       G  base + treatment indicator PDTRTMNT in place of LEDD
     A, B, C, D and F rest on the SAME observations as the base; E loses the
     subjects without age at onset and G the rows without indicator: the script
     says how many, and for E and G refits the base on the reduced sample, so as
     to separate the effect of the covariate from that of the sample. For each outcome and specification: coefficient
     of the z(T50) x years interaction, standard error, 95% CI, p,
     Benjamini-Hochberg p WITHIN the twelve outcomes of that specification,
     subjects, observations, convergence diagnostics. The coefficients of the
     added covariates are reported alongside, uncorrected and outside the
     family.
  3. Description of LEDD during follow-up: by year, the two definitions of
     'treated' (LEDD > 0; PDTRTMNT = 1) side by side; a descriptive model of
     LEDD as the outcome (LEDD ~ z(T50) x years).
  4. Summary derived from the outputs: how many of the twelve are nominally
     significant, how many survive the correction, which coefficients change
     sign, under each specification.
  5. The significance criterion that the script applies and the scale of the
     effect over ten years.
  6. MoCA: the five covariate x years models of 13_moca_confounders.csv
     redone under the strict criterion, with the log-likelihood gap from the
     point where the archived optimiser had stopped.
  7. MOTOR outcomes with time-varying LEDD: the fifteen models of
     11_ledd_adjusted.csv (3 outcomes x 5 kinetic parameters) redone under the
     strict criterion with and without LEDD, and the count of the visits at
     which the modelled part III is missing.

What the script cannot do, and it says so in the report. Adjusting for a
variable that comes after baseline is a sensitivity analysis, not an estimate
"net of treatment". The LEDD of the curated data cut is a total dose: the drug
class is not among the variables read. Duration "at each visit" is not in the
curated data cut (the column duration_yrs is the baseline duration repeated):
it is derived as baseline duration + elapsed time, and then it is almost
collinear with years; the script measures by how much.

ONLY_SPORADIC does not restrict anything here: it says which working directory
is being read, and the script stops if the variable and the working directory
do not agree or if there is a non-sporadic subject in the cohort.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 scripts/107_nonmotor_ledd_duration.py
Output: outputs/107_nonmotor_specifications.csv, 107_reproduction_S9.csv,
        107_ledd_by_year.csv, 107_moca_covariate_x_years.csv,
        107_motor_ledd.csv, 107_nonmotor_ledd_duration.txt
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS, fit_mixed                        # noqa: E402
from _paths import DATA                                       # noqa: E402

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
CACHE = os.path.join(OUT, "cache", "curated.parquet")
NAME_CURATED = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"

ALPHA = 0.05            # threshold, nominal and corrected, as in Table S9
Z95 = 1.96              # as in 72_longitudinal_refit.py and in Table S9
TOL_NEWTON = 1e-4       # stationarity criterion: Newton decrement
TOL_SINGULAR = 1e-8    # variance < TOL_SINGULAR * residual scale
TOL_CORR = 0.999        # |correlation| of the random effects >= TOL_CORR
TOL_GAIN = 1e-10        # loglike gain below which one stops
MAX_ROUNDS = 10           # maximum rounds of restarts
TOL_ID = 1e-8           # identity with the values already produced by the pipeline
TOL_WALD = 1e-10        # identity between the reported p and the two-sided Wald p
LEDD_UNITS = 100.0      # LEDD enters the non-motor models per 100 mg/day
LEDD_MAX = 10000.0      # mg/day: above this bound a value is not a dose
LEDD_HI = 2000.0      # mg/day: the rows above this dose are counted in the report
N_MIN_YEAR = 20         # below this sample size a yearly share is not commented on
YEARS_FIRSTS = 4          # "first years" in the count of the discordant visits
DECIMALS_PRINTOUT = 3     # decimals to which the supplement prints beta and p of the MoCA

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def listing(v):
    v = list(v)
    return ", ".join(str(x) for x in v) if v else 'noone'


def by_excess(x, digits=2):
    """Rounds UP to `digits` significant figures. It is needed for the values
    reported as maxima ('at most'): rounded to the nearest, a maximum of
    9.707e-4 would become 9.7e-4, which is below the true value."""
    x = float(x)
    if not np.isfinite(x) or x == 0:
        return x
    e = int(np.floor(np.log10(abs(x)))) - (digits - 1)
    return float(np.ceil(x / 10.0 ** e - 1e-9) * 10.0 ** e)


def wald_bilateral(beta, se):
    """p of the two-sided Wald test: 2 * (1 - Phi(|beta / SE|))."""
    return 2 * norm.sf(np.abs(np.asarray(beta, float) / np.asarray(se, float)))


def quantile(x, q):
    return float(x.quantile(q)) if len(x) else np.nan


# (variable, domain, direction: +1 = a high score is better, education
#  covariate yes/no) — the same list as 15_nonmotor_trajectories.py.
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
OUTCOMES = [v for v, _, _, _ in PRIMARY]
N_FAM = len(PRIMARY)

# The five kinetic parameters and the three motor outcomes of 11_longitudinal.py
# (section "Sensitivity 1"): 3 x 5 = the fifteen models of 11_ledd_adjusted.csv.
KIN = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
       "k_SLOPE": "SLOPE"}
KINETICS = list(KIN.values())
ENGINES = [("updrs3_score", "MDS-UPDRS III"),
          ("updrs_totscore", 'MDS-UPDRS total'),
          ("updrs2_score", "MDS-UPDRS II")]
N_FAM_M = len(ENGINES) * len(KINETICS)


def formula(spec, var, educ):
    e = " + EDUCYRS" if educ else ""
    if spec == "S0":
        return f"{var} ~ T50*years + age_bl + SEX + dur_bl{e}"
    if spec == "A":
        return f"{var} ~ T50*years + age_bl + SEX + dur_bl{e} + ledd100"
    if spec == "B":
        return f"{var} ~ T50*years + dur_bl*years + age_bl + SEX{e}"
    if spec == "C":
        return f"{var} ~ T50*years + dur_bl*years + age_bl + SEX{e} + ledd100"
    if spec == "D":
        return f"{var} ~ T50*years + age_bl + SEX + dur_vis{e}"
    if spec == "E":
        return f"{var} ~ T50*years + dur_sym*years + age_bl + SEX{e}"
    if spec == "F":
        return f"{var} ~ T50*years + ledd100*years + age_bl + SEX + dur_bl{e}"
    if spec == "G":
        return f"{var} ~ T50*years + age_bl + SEX + dur_bl{e} + PDTRTMNT"
    raise ValueError(spec)


SPEC = [
    ("S0", 'base, as in Table S9'),
    ("A", 'base + time-varying LEDD'),
    ("B", 'base + baseline duration x years'),
    ("C", 'base + LEDD + baseline duration x years'),
    ("D", 'duration at the visit in place of the baseline duration'),
    ("E", 'duration from symptoms and duration from symptoms x years in place of the duration from '
          'diagnosis'),
    ("F", 'base + LEDD + LEDD x years'),
    ("G", 'base + PDTRTMNT indicator in place of the LEDD'),
]
# added covariates whose coefficient is reported: (specification,
# column label, possible names of the term)
ADDITIONS = {
    "A": [("ledd100", ("ledd100",))],
    "B": [('durxyears', ("dur_bl:years", "years:dur_bl"))],
    "C": [("ledd100", ("ledd100",)),
          ('durxyears', ("dur_bl:years", "years:dur_bl"))],
    "D": [("durvis", ("dur_vis",))],
    "E": [("dursym", ("dur_sym",)),
          ('dursymxyears', ("dur_sym:years", "years:dur_sym"))],
    "F": [("ledd100", ("ledd100",)),
          ('leddxyears', ("ledd100:years", "years:ledd100"))],
    "G": [("pdtrtmnt", ("PDTRTMNT",))],
}
LBLS_AGG = [
    ("ledd100", f'LEDD (per {LEDD_UNITS:g} mg/day)'),
    ('durxyears', 'baseline duration x years (per year of duration)'),
    ("durvis", 'duration at the visit (per year)'),
    ("dursym", 'duration since symptoms (per year)'),
    ('dursymxyears', 'duration since symptoms x years (per year of duration)'),
    ('leddxyears', f'LEDD x years (per {LEDD_UNITS:g} mg/day per year)'),
    ("pdtrtmnt", 'PDTRTMNT indicator (1 against 0)'),
]
NAMES_AGG = [n for n, _ in LBLS_AGG]

# The covariate x years models on the MoCA: keys and formulas of 13_moca_signal.py
# (section c). The last column says whether the sample requires age at onset.
SPEC13 = [
    ('base model',
     "moca ~ T50*years + age_bl + SEX + EDUCYRS + dur_bl", False),
    ('+ age at onset x years',
     "moca ~ T50*years + ageonset*years + age_bl + SEX + EDUCYRS + dur_bl", True),
    ('+ baseline age x years',
     "moca ~ T50*years + age_bl*years + SEX + EDUCYRS + dur_bl", False),
    ('+ education x years',
     "moca ~ T50*years + EDUCYRS*years + age_bl + SEX + dur_bl", False),
    ('+ disease duration x years',
     "moca ~ T50*years + dur_bl*years + age_bl + SEX + EDUCYRS", False),
    ('+ sex x years',
     "moca ~ T50*years + SEX*years + age_bl + EDUCYRS + dur_bl", False),
]
KEY13_BASE = 'base model'
KEY13_DURATION = '+ disease duration x years'


# ------------------------------------------- primitives of the strict criterion
# Copied from 91_table6_strict_criterion.py (in turn from
# 88_diagnostics_63.py) without changes of substance: on the same fit they must
# give the same number, and that is what section 1 checks.
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
    """Central-difference Hessian of the profiled log-likelihood, in the same
    parameter space in which the optimiser works."""
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
            a, b, c, dd = p.copy(), p.copy(), p.copy(), p.copy()
            a[i] += h[i]; a[j] += h[j]
            b[i] += h[i]; b[j] -= h[j]
            c[i] -= h[i]; c[j] += h[j]
            dd[i] -= h[i]; dd[j] -= h[j]
            H[i, j] = H[j, i] = (_ll(md, a) - _ll(md, b) - _ll(md, c)
                                 + _ll(md, dd)) / (4 * h[i] * h[j])
    return (H + H.T) / 2, f0


def estimate(f, data, groups, method, it, start=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kw = {} if start is None else {"start_params": start}
            return smf.mixedlm(f, data, groups=groups,
                               re_formula="~years").fit(method=method,
                                                        maxiter=it, **kw)
        except Exception:
            return None


def procedure(f, data, groups):
    """Cascade, selection on the likelihood among the converged ones, restarts."""
    attempts = {}
    md_ref = None
    for method, it in OPTIMISERS:
        r = estimate(f, data, groups, method, it)
        if r is None:
            attempts[method] = (None, np.nan, False)
            continue
        if md_ref is None:
            md_ref = r.model
        attempts[method] = (r, _ll(md_ref, pack(r)), bool(r.converged))
    if md_ref is None:
        return None
    conv = [(m, v[1]) for m, v in attempts.items()
            if v[0] is not None and v[2] and np.isfinite(v[1])]
    if conv:
        chosen = max(conv, key=lambda x: x[1])[0]
    else:
        fin = [(m, v[1]) for m, v in attempts.items()
               if v[0] is not None and np.isfinite(v[1])]
        if not fin:
            return None
        chosen = max(fin, key=lambda x: x[1])[0]
    r0, ll0, _ = attempts[chosen]
    best = (r0, '(none)', ll0, pack(r0))
    rounds = 0
    for cycle in range(MAX_ROUNDS):
        improved = False
        for method, it in OPTIMISERS:
            r = estimate(f, data, groups, method, it, start=best[3])
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
    return {'opt_chosen': chosen, 'atleast_an_convergent': bool(conv),
            'attempts': attempts, 'r_final': best[0], 'opt_reopt': best[1],
            'rounds': rounds, 'll_final': best[2], 'md_ref': md_ref}


def term(r, names):
    for c in names:
        if c in r.params.index:
            return c
    raise KeyError(f'none of {names} among the model terms')


def diagnostics(r, md_ref):
    """Newton decrement and structural diagnostics at the reported point."""
    md = r.model
    p = pack(r)
    g = _score(md, p)
    H, _ = hess_profiled(md, p)
    eig = np.linalg.eigvalsh(H)
    try:
        newton = float(0.5 * g @ np.linalg.solve(H, -g))
    except np.linalg.LinAlgError:
        newton = np.nan
    cov_re = np.asarray(r.cov_re, dtype=float)
    scale = float(r.scale)
    v_int, v_slope = float(cov_re[0, 0]), float(cov_re[1, 1])
    corr = (float(cov_re[0, 1]) / np.sqrt(v_int * v_slope)
            if v_int > 0 and v_slope > 0 else np.nan)
    return {
        "loglike": _ll(md_ref, p),
        "grad_inf_norm": float(np.max(np.abs(g))),
        "newton": newton,
        'hessian_def_negative': bool(eig.max() < 0),
        'var_intercept': v_int, 'var_slope': v_slope,
        'corr_intercept_slope': corr, 'scale_residual': scale,
        'singular': bool(v_int < TOL_SINGULAR * scale
                          or v_slope < TOL_SINGULAR * scale
                          or (np.isfinite(corr) and abs(corr) >= TOL_CORR)),
        'at_edge': bool(v_int == 0.0 or v_slope == 0.0),
    }


def fit_strict_extended(f, data, kinet="T50"):
    """One model under the strict criterion: (row of results, fit, reference
    model on which the log-likelihoods are comparable), or three None.
    `kinet` is the kinetic parameter whose interaction with years is reported."""
    P = procedure(f, data, data.PATNO)
    if P is None:
        return None, None, None
    r = P['r_final']
    D = diagnostics(r, P['md_ref'])
    t = term(r, (f'{kinet}:years', f'years:{kinet}'))
    b, se = float(r.params[t]), float(r.bse[t])
    row = {"beta": b, "se": se, "ci_lo": b - Z95 * se, "ci_hi": b + Z95 * se,
            "p": float(r.pvalues[t]),
            'optimiser_chosen': P['opt_chosen'],
            'optimiser_restart': P['opt_reopt'],
            'rounds_restart': P['rounds'],
            'atleast_an_convergent': P['atleast_an_convergent']}
    row.update(D)
    row['criterion_newton'] = bool(np.isfinite(D["newton"])
                                   and D["newton"] < TOL_NEWTON)
    # A fit is accepted only if an optimiser has declared convergence
    # AND the reported point is stationary AND the curvature is that of a
    # maximum. The `converged` flag alone is not enough.
    row['accepted'] = bool(row['atleast_an_convergent']
                             and row['criterion_newton']
                             and D['hessian_def_negative']
                             and not D['at_edge'])
    for method, _ in OPTIMISERS:
        _r, ll_, cv_ = P['attempts'].get(method, (None, np.nan, False))
        row[f'loglike_{method}'] = ll_
        row[f'converged_{method}'] = cv_
    return row, r, P['md_ref']


def fit_strict(f, data, kinet="T50"):
    row, r, _md = fit_strict_extended(f, data, kinet)
    return row, r


COLS_DIAG = (['accepted', 'criterion_newton', "newton", "grad_inf_norm", "loglike",
              'atleast_an_convergent', 'optimiser_chosen',
              'optimiser_restart', 'rounds_restart',
              'hessian_def_negative', 'singular', 'at_edge',
              'var_intercept', 'var_slope', 'corr_intercept_slope',
              'scale_residual']
             + [f"loglike_{m}" for m, _ in OPTIMISERS]
             + [f"converged_{m}" for m, _ in OPTIMISERS])


# ====================================================================== data
title("0. WORKING DIRECTORY, POPULATION, DATA, IDENTITY CHECKS")
log(f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}, pandas "
    f"{pd.__version__}")
log(f'optimiser cascade: {OPTIMISERS}')

if os.path.exists(CACHE):
    cur = pd.read_parquet(CACHE)
    origin = "cache/curated.parquet"
else:
    cand = [os.path.join(BASE, NAME_CURATED), os.path.join(DATA, NAME_CURATED)]
    located = [c for c in cand if os.path.exists(c)]
    assert located, f"curated not found either in the cache or in: {cand}"
    cur = pd.read_excel(located[0], sheet_name="20260511")
    origin = os.path.basename(located[0])
cur["PATNO"] = cur["PATNO"].astype("int64")

c150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
c150["PATNO"] = c150["PATNO"].astype("int64")
if "complete" in c150.columns:
    c150 = c150[c150.complete].reset_index(drop=True)
c24 = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
c24["PATNO"] = c24["PATNO"].astype("int64")
if "complete_primary" in c24.columns:
    c24 = c24[c24.complete_primary]
assert not c150.PATNO.duplicated().any(), 'repeated subjects in the 150 h cohort'

blc = cur[cur.EVENT_ID == "BL"]
assert not blc.PATNO.duplicated().any(), "more than one baseline row per subject in the curated data cut"
sub = blc.set_index("PATNO").subgroup
sporadic = set(sub[sub == "Sporadic PD"].index)
p150, p24 = set(c150.PATNO), set(c24.PATNO)
assert (p150 | p24) <= set(sub.index), 'subjects of the cohorts missing from the baseline curated data cut'
only150, only24 = not (p150 - sporadic), not (p24 - sporadic)
assert only150 == only24, ("the two cohorts of the working directory are not of the same kind: one "
                           "is sporadic only, the other is not")
ROOT = 'sporadic only' if only150 else 'full cohort'
assert ONLY == only150, (f"ONLY_SPORADIC={('1' if ONLY else '0')} but the working directory is "
                         f"'{ROOT}': variable and working directory do not agree")
if ONLY:
    no_spor = sorted(p150 - sporadic)
    assert not no_spor, f'non-sporadic subjects in the 150 h cohort: {no_spor[:10]}'
    assert (c150.subgroup == "Sporadic PD").all(), \
        'the subgroup column of 08_dataset_150h.csv contains non-sporadic subjects'
POP = f'150 h cohort, {ROOT}, n = {len(c150)}'
# The name of the working directory's folder does not enter the report: two
# working directories with the same content must give the same file, byte for byte.
log(f"working directory: {ROOT}; ONLY_SPORADIC = {('1' if ONLY else '0')}; curated data cut read "
    f"from {origin}")
log(f'population: {POP} subjects; subgroups from the baseline curated data cut: '
    f'{sub.reindex(sorted(p150)).value_counts().to_dict()}')

# Baseline variables: from the curated data cut, as in 15; they must coincide
# with those that 08_replication_150h.py wrote to the cohort file (same values,
# subject by subject, not only same subjects). Baseline LEDD and PDTRTMNT are in
# the list: the sentence "no one is treated at baseline" rests on them.
VAR_ID = ["age_at_visit", "SEX", "EDUCYRS", "duration_yrs", "ageonset", "LEDD",
          "PDTRTMNT"]
conf = c150[["PATNO"] + VAR_ID].merge(
    blc[["PATNO"] + VAR_ID], on="PATNO", how="left", validate="one_to_one",
    suffixes=('_cohort', "_curated"))
for c in VAR_ID:
    different = ~np.isclose(conf[f'{c}_cohort'].to_numpy(float),
                          conf[f"{c}_curated"].to_numpy(float),
                          atol=1e-9, rtol=0, equal_nan=True)
    assert not different.any(), (
        f'baseline variable different between the curated data cut and 08_dataset_150h.csv: {c}, in '
        f'{int(different.sum())} subjects (PATNO {list(conf.PATNO[different])[:10]})')
log(f'baseline variables ({listing(VAR_ID)}): identical, subject by subject, between the curated '
    f'data cut and 08_dataset_150h.csv')
covbl = blc[["PATNO", "age_at_visit", "SEX", "EDUCYRS", "duration_yrs",
             "ageonset"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})

# Log and standardisation per SUBJECT, as in 11 and in 15. The T50 is the same
# number that the previous version of this script computed by itself.
d = c150[["PATNO"] + list(KIN)].rename(columns=KIN)
for k in KINETICS:
    assert d[k].notna().all() and (d[k] > 0).all(), f'{k} missing or non-positive'
    d[k] = np.log(d[k])
    d[k] = (d[k] - d[k].mean()) / d[k].std()
base = d.merge(covbl, on="PATNO", how="left", validate="one_to_one")
base["dur_sym"] = base.age_bl - base.ageonset        # duration from symptoms, at baseline

OUTCOMES_M = [v for v, _ in ENGINES]
HA_ON = "updrs3_score_on" in cur.columns
L = cur[cur.PATNO.isin(p150)][
    ["PATNO", "EVENT_ID", "YEAR", "age_at_visit", "LEDD", "PDTRTMNT",
     "duration_yrs"] + OUTCOMES + OUTCOMES_M
    + (["updrs3_score_on"] if HA_ON else [])].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left", validate="many_to_one")
L = L.sort_values(["PATNO", "years", "EVENT_ID"]).reset_index(drop=True)
assert not L.duplicated(["PATNO", "years"]).any(), \
    "two rows for the same subject in the same year: the unit is not the subject-year"
assert set(L.PATNO) == p150, 'cohort subjects without visits in the curated data cut'
L["ledd100"] = L.LEDD / LEDD_UNITS
L['elapsed'] = L.age_at_visit - L.age_bl           # actual years from baseline
L["dur_vis"] = L.dur_bl + L.elapsed                # disease duration at the visit
fu = L.groupby("PATNO").years.max()
log(f'visits: {len(L)} subject-year rows, years 0-{L.years.max():.0f}; follow-up (last recorded '
    f'visit) median {fu.median():.1f} years [IQR {fu.quantile(0.25):.1f}-{fu.quantile(0.75):.1f}]')

# ---------------------------------------------- LEDD and indicator: bounds
# For the follow-up LEDD the pipeline has no reference value against which to
# check identity. The bounds are checked, the report states what an altered
# value would shift (maximum, rows above LEDD_HI, zeros with the indicator at
# 1) and, further below, the motor models with LEDD of 11_ledd_adjusted.csv,
# which depend on the follow-up values, are reproduced.
outside = L[L.LEDD.notna() & ((L.LEDD < 0) | (L.LEDD > LEDD_MAX))]
assert len(outside) == 0, (
    f'LEDD outside the limits [0, {LEDD_MAX:g}] mg/day in {len(outside)} rows:\n'
    + outside[["PATNO", "years", "LEDD"]].head(10).to_string(index=False))
val_trt = set(L.PDTRTMNT.dropna().unique())
VAL_TRT = listing(f"{x:g}" for x in sorted(val_trt))
assert val_trt <= {0.0, 1.0}, f'PDTRTMNT has values other than 0 and 1: {VAL_TRT}'
Lfu = L[L.years > 0]
n_zero_trt = int(((L.LEDD == 0) & (L.PDTRTMNT == 1)).sum())
n_zero_trt_fu = int(((Lfu.LEDD == 0) & (Lfu.PDTRTMNT == 1)).sum())
log(f'LEDD (mg/day) on the {len(L)} subject-year rows: within the limits [0, {LEDD_MAX:g}]; minimum '
    f'{L.LEDD.min():.1f}, maximum {L.LEDD.max():.1f};')
log(f'    at follow-up ({len(Lfu)} rows with years > 0) maximum {Lfu.LEDD.max():.1f}, 99th '
    f'percentile {Lfu.LEDD.quantile(0.99):.1f}, rows above {LEDD_HI:g}: '
    f'{int((Lfu.LEDD > LEDD_HI).sum())};')
log(f'    rows with LEDD = 0 and PDTRTMNT = 1: {n_zero_trt} (of which at follow-up {n_zero_trt_fu}). '
    f'PDTRTMNT takes the values {VAL_TRT}.')
log("    Identity of the LEDD values at follow-up: NOT verifiable directly (the pipeline does not")
log("    keep a copy of them). The only check is indirect: the reproduction of the motor models "
    "with LEDD, below.")

# ------------------------------------------------------- samples per outcome
# The sample of each outcome requires ALL the variables used by S0, A, B, C,
# D and F, so that these specifications rest on the same observations. E and G
# use a subset of it (age at onset or indicator not missing). That the sample
# is that of Table S9 is not assumed: section 1 checks it against the published
# counts, before fitting any model.
SAMP, CAMPX = {}, {}
for var, dom, direction, educ in PRIMARY:
    cov = ["age_bl", "SEX", "dur_bl"] + (["EDUCYRS"] if educ else [])
    m_base = L[["PATNO", "years", var, "T50"] + cov].dropna()
    m = L[["PATNO", "years", var, "T50", "ledd100", "dur_vis"] + cov].dropna()
    assert len(m) == len(m_base), (
        f'{var}: LEDD or duration at the visit are missing in {len(m_base) - len(m)} observations of '
        f'the base sample; the specifications would not be on the same observations')
    SAMP[var] = m.reset_index(drop=True)
    CAMPX[var] = L.loc[m.index, list(m.columns) + ["dur_sym", "PDTRTMNT",
                                                    "ageonset"]].reset_index(drop=True)
    assert CAMPX[var][list(m.columns)].equals(SAMP[var]), f'{var}: samples not aligned'


def sample(code, var):
    """The sample on which specification `code` is fitted for outcome `var`."""
    if code == "E":
        x = CAMPX[var].drop(columns=["PDTRTMNT", "ageonset"])
        return x[x.dur_sym.notna()].reset_index(drop=True)
    if code == "G":
        x = CAMPX[var].drop(columns=["dur_sym", "ageonset"])
        return x[x.PDTRTMNT.notna()].reset_index(drop=True)
    return SAMP[var]


# ------------------------------- motor outcomes: samples and reproduction of 11
# The fifteen models of 11_longitudinal.py, "Sensitivity 1": for each of the
# three MDS-UPDRS outcomes and of the five kinetic parameters,
#     outcome ~ z(log kinetic) x years + baseline age + sex + baseline duration + LEDD
# with LEDD in mg/day, random intercept and random slope, REML, lbfgs alone with
# at most 200 iterations. The reproduction is here, before the slow models,
# because it is also the indirect identity check on the follow-up LEDD.
SAMP_M = {}
for out, _lbls in ENGINES:
    cols = ["PATNO", "years", out] + KINETICS + ["age_bl", "SEX", "dur_bl"]
    m0 = L[cols].dropna()
    m1 = L[cols + ["LEDD"]].dropna()
    assert len(m0) == len(m1), (
        f'{out}: LEDD is missing in {len(m0) - len(m1)} observations; the models with and without '
        f'LEDD would not be on the same observations')
    SAMP_M[out] = m1.reset_index(drop=True)


def formula_m(out, k, having_ledd):
    return f"{out} ~ {k}*years + age_bl + SEX + dur_bl" + (" + LEDD" if having_ledd else "")


def estimate_rule11(out, k):
    """The fit as 11_longitudinal.py does it: a single optimiser."""
    m = SAMP_M[out]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula_m(out, k, True), m, groups=m.PATNO,
                           re_formula="~years").fit(method="lbfgs", maxiter=200)


f11 = os.path.join(OUT, "11_ledd_adjusted.csv")
HA_11 = os.path.exists(f11)
RIPR11 = {}
t0 = time.time()
for out, _lbls in ENGINES:
    for k in KINETICS:
        r11 = estimate_rule11(out, k)
        t11 = term(r11, (f"{k}:years", f"years:{k}"))
        RIPR11[(out, k)] = {"r": r11, "beta": float(r11.params[t11]),
                            "se": float(r11.bse[t11]), "p": float(r11.pvalues[t11]),
                            "converged": bool(r11.converged),
                            'n_obsv': int(len(SAMP_M[out]))}
print(f'  . reproduction of the {N_FAM_M} motor models with lbfgs alone ({time.time() - t0:.0f}s)', flush=True)
pbh11 = multipletests([RIPR11[(o, k)]["p"] for o in OUTCOMES_M for k in KINETICS],
                      method="fdr_bh")[1]
for (o, k), q in zip([(o, k) for o in OUTCOMES_M for k in KINETICS], pbh11):
    RIPR11[(o, k)]["p_BH"] = float(q)
if HA_11:
    arch11 = pd.read_csv(f11)
    keys11 = list(zip(arch11.outcome, arch11.kinetic))
    assert len(set(keys11)) == len(keys11), '11_ledd_adjusted.csv has repeated models'
    assert set(keys11) == set(RIPR11), (
        f'the models of 11_ledd_adjusted.csv are not the {N_FAM_M} expected: extra '
        f'{sorted(set(keys11) - set(RIPR11))}, missing {sorted(set(RIPR11) - set(keys11))}')
    arch11 = arch11.set_index(['outcome', 'kinetic'])
    SC11 = {"beta": 0.0, "p": 0.0, "p_BH": 0.0}
    for ch, v in RIPR11.items():
        assert int(arch11.loc[ch, 'n_obsv']) == v['n_obsv'], (
            f"{ch}: {v['n_obsv']} observations against {int(arch11.loc[ch, 'n_obsv'])} in "
            f"11_ledd_adjusted.csv")
        for q in SC11:
            sc_ = abs(v[q] - float(arch11.loc[ch, q]))
            assert sc_ < TOL_ID, (
                f"{ch}: {q} does not reproduce 11_ledd_adjusted.csv (difference {sc_:.3g}); the "
                f"follow-up LEDD, the kinetic parameters or the covariates are not those with which "
                f"the file was produced")
            SC11[q] = max(SC11[q], sc_)
    log(f'11_ledd_adjusted.csv ({N_FAM_M} motor models with LEDD, lbfgs only): same models, same '
        f'observations;')
    log(f"    maximum difference beta {SC11['beta']:.1e}, p {SC11['p']:.1e}, p BH {SC11['p_BH']:.1e} "
        f"(tolerance {TOL_ID:g}). Details in section 7.")
else:
    arch11 = None
    log('11_ledd_adjusted.csv MISSING: reproduction of the motor models with LEDD NOT verified, and '
        'with it')
    log("    the indirect identity of the follow-up LEDD.")

# =========================================================== 1. REPRODUCTION
title('1. REPRODUCTION OF TABLE S9 (before anything else)')
log(f"Population: {POP}. Units: subjects (n) and subject-year observations (obs).")
log("Model: outcome ~ z(log T50) x years + baseline age + sex + baseline duration")
log(f'[+ education for the {sum((e for _, _, _, e in PRIMARY))} cognitive outcomes]; random '
    f'intercept and slope, REML.')
pub = pd.read_csv(os.path.join(OUT, "15_primary_nonmotor.csv")).set_index('outcome')
str91 = pd.read_csv(os.path.join(OUT, "91_table6_strict_criterion.csv")
                    ).set_index("outcome")
assert list(pub.index) == OUTCOMES, 'the outcomes of 15_primary_nonmotor.csv are not the twelve expected'
assert set(str91.index) == set(OUTCOMES), 'the outcomes of 91 are not the twelve expected'
m72 = pd.read_csv(os.path.join(OUT, "72_mixed_models.csv"))
m72 = m72[(m72.cohort == "150h") & (m72.kinetic == "T50")
          & m72.outcome.isin(OUTCOMES)].set_index("outcome")
assert len(m72) > 0, "no outcome in common with 72_mixed_models.csv: the check is empty"

# (0) subjects and observations: it costs nothing, and it is checked for all
# twelve outcomes before fitting the first model.
for var in OUTCOMES:
    m = SAMP[var]
    here = (int(m.PATNO.nunique()), int(len(m)))
    for name, expected in (
            ("15_primary_nonmotor.csv",
             (int(pub.loc[var, 'n_subj']), int(pub.loc[var, 'n_obsv']))),
            ("91_table6_strict_criterion.csv",
             (int(str91.loc[var, "n_subjects"]), int(str91.loc[var, "n_obs"])))):
        assert here == expected, (f'{var}: subjects and observations {here} different from {name} '
                               f'{expected}')

# (a) the rule with which the table was produced. Identity is checked outcome
# by outcome INSIDE the loop: altered data stop the script at the first
# outcome affected, not after all the models have been fitted.
REG15 = {}
for var, dom, direction, educ in PRIMARY:
    m = SAMP[var]
    f = formula("S0", var, educ)
    t0 = time.time()
    rA, optA, _rip = fit_mixed(f, m, m.PATNO)                 # rule of 15
    assert rA is not None and rA.converged, f'{var}: the rule of 15 does not converge'
    tA = term(rA, ("T50:years", "years:T50"))
    vA = {"beta": float(rA.params[tA]), "se": float(rA.bse[tA]),
          "p": float(rA.pvalues[tA]), 'slope_mean': float(rA.params["years"])}
    sc_ = {q: abs(vA[q] - float(pub.loc[var, q])) for q in vA}
    assert max(sc_.values()) < TOL_ID, \
        f'{var}: the rule of 15 does not reproduce 15_primary_nonmotor.csv: differences {sc_}'
    if var in m72.index:
        sc_ = {q: abs(vA[q] - float(m72.loc[var, q])) for q in ("beta", "se", "p")}
        assert max(sc_.values()) < TOL_ID, \
            f'{var}: difference from 72_mixed_models.csv: {sc_}'
    REG15[var] = (vA, optA)
    print(f'  . rule 15 {var:22s} ({time.time() - t0:.0f}s)', flush=True)

# (b) the strict criterion, with the same outcome-by-outcome stop
rows_reprod, rows = [], []
FIT = {}
for var, dom, direction, educ in PRIMARY:
    m = SAMP[var]
    f = formula("S0", var, educ)
    t0 = time.time()
    vA, optA = REG15[var]
    rowb, rB = fit_strict(f, m)
    assert rowb is not None, f'{var}: no fit under the strict criterion'
    sc_ = {q: abs(rowb[q] - float(str91.loc[var, f"{q}_ruleB"]))
           for q in ("beta", "se", "p")}
    assert max(sc_.values()) < TOL_ID, (
        f'{var}: the strict criterion does not reproduce 91_table6_strict_criterion.csv: differences '
        f'{sc_}')
    assert rowb['accepted'], f'{var}: the base model does not meet the strict criterion'
    FIT[("S0", var)] = rowb
    rows_reprod.append({
        'outcome': var, 'domain': dom,
        'n_subj': int(m.PATNO.nunique()), 'n_obsv': int(len(m)),
        'n_subj_published': int(pub.loc[var, 'n_subj']),
        'n_obsv_published': int(pub.loc[var, 'n_obsv']),
        'beta_published': float(pub.loc[var, "beta"]),
        'se_published': float(pub.loc[var, "se"]),
        'p_published': float(pub.loc[var, "p"]),
        'p_BH_published': float(pub.loc[var, "p_BH"]),
        'slope_mean_published': float(pub.loc[var, 'slope_mean']),
        'optimiser_rule15': optA,
        'beta_rule15': vA["beta"], 'se_rule15': vA["se"],
        'p_rule15': vA["p"],
        'slope_mean_rule15': vA['slope_mean'],
        'beta_strict': rowb["beta"], 'se_strict': rowb["se"],
        'p_strict': rowb["p"],
        'beta_strict_91': float(str91.loc[var, "beta_ruleB"]),
        'se_strict_91': float(str91.loc[var, "se_ruleB"]),
        'p_strict_91': float(str91.loc[var, "p_ruleB"]),
        'p_BH_strict_91': float(str91.loc[var, "p_BH_ruleB"]),
        'newton_strict': rowb["newton"], 'accepted': rowb['accepted'],
        'slope_mean_strict': float(rB.params["years"]),
    })
    print(f"  . S0 {var:22s} ({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(rows_reprod)
R['p_BH_rule15'] = multipletests(R.p_rule15, method="fdr_bh")[1]
R['p_BH_strict'] = multipletests(R.p_strict, method="fdr_bh")[1]


def departure(a, b):
    return float((R[a] - R[b]).abs().max())


# Summary of the discrepancies. The corrected p can be compared only now, because
# they depend on all twelve p; the other comparisons repeat over the whole
# table those already passed outcome by outcome.
assert (R.n_subj == R.n_subj_published).all() and (R.n_obsv == R.n_obsv_published).all(), \
    'subjects or observations different from 15_primary_nonmotor.csv'
SC15 = {q: departure(f'{q}_rule15', f'{q}_published')
        for q in ("beta", "se", "p", "p_BH", 'slope_mean')}
assert max(SC15.values()) < TOL_ID, \
    f'the rule of 15 does not reproduce 15_primary_nonmotor.csv: differences {SC15}'
SC91 = {q: departure(f'{q}_strict', f'{q}_strict_91') for q in ("beta", "se", "p", "p_BH")}
assert max(SC91.values()) < TOL_ID, \
    f'the strict criterion does not reproduce 91_table6_strict_criterion.csv: differences {SC91}'
SC72 = {q: float((R.set_index('outcome').loc[m72.index, f'{q}_rule15']
                  - m72[q]).abs().max()) for q in ("beta", "se", "p")}
assert max(SC72.values()) < TOL_ID, \
    f'difference from 72_mixed_models.csv on the common outcomes {list(m72.index)}: {SC72}'
assert R.accepted.all(), \
    f'base models that do not meet the strict criterion: {list(R.outcome[~R.accepted])}'

log("\n(a) Rule with which the table was produced (first optimiser that converges),")
log(f'    against 15_primary_nonmotor.csv, {len(R)} outcomes: identical subjects and observations;')
log(f"    maximum difference beta {SC15['beta']:.1e}, SE {SC15['se']:.1e}, p {SC15['p']:.1e}, p BH "
    f"{SC15['p_BH']:.1e}, mean slope {SC15['slope_mean']:.1e} (tolerance {TOL_ID:g}).")
log(f"(b) Strict criterion against 91_table6_strict_criterion.csv: maximum difference beta "
    f"{SC91['beta']:.1e},")
log(f"    ES {SC91['se']:.1e}, p {SC91['p']:.1e}, p BH {SC91['p_BH']:.1e}.")
log(f"(c) Outcomes also present in 72_mixed_models.csv (150 h, T50): {listing(m72.index)}; maximum "
    f"difference beta {SC72['beta']:.1e}, SE {SC72['se']:.1e}, p {SC72['p']:.1e}.")
SPO = {q: departure(f'{q}_strict', f'{q}_published') for q in ("beta", "se", "p", "p_BH")}
log(f"Between the published table and the strict criterion: maximum difference beta "
    f"{SPO['beta']:.1e}, SE {SPO['se']:.1e},")
log(f"    p {SPO['p']:.1e}, p BH {SPO['p_BH']:.1e}. Newton decrement of the {len(R)} base models: at "
    f"most {by_excess(R.newton_strict.max()):.1e}")
log(f'    (rounded up; threshold {TOL_NEWTON:g}); accepted {int(R.accepted.sum())} out of {len(R)}.')
log("From here on the base specification is the one under the strict criterion, and the")
log('alternatives are estimated with the same criterion.')

# ====================================================== 2. SPECIFICATIONS
title(f'2. LEDD AND DISEASE DURATION: {len(SPEC) - 1} ALTERNATIVE SPECIFICATIONS')
dbl = base.dur_bl
log(f"Disease duration at baseline ({POP}), that is time since diagnosis: median {dbl.median():.2f} "
    f"years")
log(f'    [IQR {dbl.quantile(0.25):.2f}-{dbl.quantile(0.75):.2f}], minimum {dbl.min():.2f}, maximum '
    f'{dbl.max():.2f}.')
dur_constant = bool(L.groupby("PATNO").duration_yrs.nunique().max() == 1)
dur_equal_bl = bool(np.allclose(L.duration_yrs, L.dur_bl, atol=1e-9, rtol=0))
if dur_constant and dur_equal_bl:
    log("The duration_yrs column of the curated data cut is constant within subject and equal to the baseline")
    log("value at every visit: it is NOT a time-varying duration. The duration at the visit is")
    log('obtained as baseline duration + actual years elapsed since baseline (difference')
    log("between age at the visit and baseline age).")
else:
    log('WARNING: the duration_yrs column of the curated data cut varies within subject; the duration')
    log("at the visit used here (baseline duration + actual years elapsed) is not that column.")
assert (L.dur_vis > 0).all(), 'non-positive duration at the visit'
sc = L.elapsed - L.years
r_dv = float(np.corrcoef(L.dur_vis, L.dur_bl + L.years)[0, 1])
log(f'Difference between actual and nominal years of visit ({len(L)} subject-year rows): mean '
    f'{sc.mean():+.3f}, SD {sc.std():.3f},')
log(f'    minimum {sc.min():+.2f}, maximum {sc.max():+.2f} years; correlation between duration at '
    f'the visit and (baseline duration + nominal years) {r_dv:.4f}.')
# That duration at the visit, with NOMINAL time, adds nothing to the base is
# not a result: it is algebra (baseline duration + years is a linear combination
# of two columns that the base already contains). What the data can say, and
# what is measured here outcome by outcome, is how much of the variance of the
# ACTUAL duration at the visit lies outside the space of those columns.
resid = []
for var, dom, direction, educ in PRIMARY:
    m = SAMP[var]
    X = np.column_stack([np.ones(len(m)), m.years, m.dur_bl]).astype(float)
    y = m.dur_vis.to_numpy(float)
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    resid.append(float(res.var() / y.var()))
log('By construction (algebra, not a result): duration at the visit = baseline duration + actual')
log("years since baseline. With NOMINAL years it would be baseline duration + years, that is an")
log("exact linear combination of two columns that the base model already contains, and")
log("specification D would coincide with the reparametrised base. D is therefore identified")
log('only by the calendar deviation of the visits. Share of the variance of the duration at the')
log(f'visit NOT explained by (constant, nominal years, baseline duration), in the samples of the '
    f'{len(resid)} outcomes:')
log(f"    from {100 * min(resid):.3f}% to {100 * max(resid):.3f}%. D is not an independent check, "
    f"and must be read as such.")
r_ly = float(np.corrcoef(L.LEDD, L.years)[0, 1])
log(f'LEDD and years since baseline, on the {len(L)} subject-year rows: Pearson correlation '
    f'{r_ly:.3f}.')
# duration from symptoms (specification E)
sd = base.dur_sym
n_sd_lacks = int(sd.isna().sum())
assert (sd.dropna() > 0).all(), (
    f'non-positive duration since symptoms (age at onset later than baseline age) in '
    f'{int((sd.dropna() <= 0).sum())} subjects')
sd_ok = base[base.dur_sym.notna()]
log(f'Duration since symptoms at baseline (baseline age - age at onset; {POP}): available in '
    f'{len(sd_ok)} subjects,')
log(f'    missing in {n_sd_lacks} (age at onset not recorded); median {sd.median():.2f} years [IQR '
    f'{sd.quantile(0.25):.2f}-{sd.quantile(0.75):.2f}], minimum {sd.min():.2f}, maximum '
    f'{sd.max():.2f};')
log(f"    correlation with the duration since diagnosis, in the {len(sd_ok)}: Pearson "
    f"{float(np.corrcoef(sd_ok.dur_sym, sd_ok.dur_bl)[0, 1]):.3f}, Spearman "
    f"{float(sd_ok.dur_sym.corr(sd_ok.dur_bl, method='spearman')):.3f}.")
n_trt_lacks = int(L.PDTRTMNT.isna().sum())
log(f'PDTRTMNT indicator (specification G): missing in {n_trt_lacks} subject-year rows out of '
    f'{len(L)} ({int(L[L.years == 0].PDTRTMNT.isna().sum())} at baseline,')
log(f'    {int(Lfu.PDTRTMNT.isna().sum())} at follow-up), of '
    f'{int(L[L.PDTRTMNT.isna()].PATNO.nunique())} subjects.')
log("")
log('Specifications (all: random intercept and slope per subject, REML, strict')
log("criterion; reported coefficient: z(log T50) x years, in outcome units per year")
log("per 1 SD of log T50; T50 remains standardised on the subjects of the whole cohort")
log("even where the sample is smaller):")
for code, descr in SPEC:
    log(f'  {code:2s} {descr}')
    log(f"       {formula(code, 'outcome', False).split('~')[1].strip()}")
log(f'LEDD: mg/day at the visit, divided by {LEDD_UNITS:g} in the models. The cognitive outcomes '
    f'additionally have education.')
log("Samples. S0, A, B, C, D and F: the same observations. E: without the subjects lacking age at "
    "onset.")
log("G: without the rows lacking the indicator. For E and G the comparison with the base therefore "
    "mixes the effect")
log('of the covariate and that of the sample; the rows lost are counted in the tables.')

for code, descr in SPEC:
    for var, dom, direction, educ in PRIMARY:
        m = sample(code, var)
        if code == "S0":
            row = dict(FIT[("S0", var)])
            r = None
        else:
            t0 = time.time()
            row, r = fit_strict(formula(code, var, educ), m)
            assert row is not None, f'{code} {var}: no fit'
            print(f"  . {code:2s} {var:22s} newton {row['newton']:.1e} ({time.time() - t0:.0f}s)", flush=True)
        for name in NAMES_AGG:
            for q in ("beta", "se", "p"):
                row[f'{name}_{q}'] = np.nan
        for name, cand in ADDITIONS.get(code, []):
            t = term(r, cand)
            row[f'{name}_beta'] = float(r.params[t])
            row[f'{name}_se'] = float(r.bse[t])
            row[f'{name}_p'] = float(r.pvalues[t])
        n_s_base, n_o_base = int(SAMP[var].PATNO.nunique()), int(len(SAMP[var]))
        # Where the sample is smaller than that of the base (E, G) the base is
        # refitted on the reduced sample: in this way the shift due to the
        # covariate is separated from that due to the lost observations.
        if len(m) == n_o_base:
            rb = FIT[("S0", var)]
        else:
            t0 = time.time()
            rb, _rb = fit_strict(formula("S0", var, educ), m)
            assert rb is not None, f'{code} {var}: no fit of the base on the reduced sample'
            print(f"  . {code:2s} {var:22s} base on the reduced sample, newton {rb['newton']:.1e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        row.update({'beta_base_same_sample': rb["beta"],
                     'se_base_same_sample': rb["se"],
                     'p_base_same_sample': rb["p"],
                     'newton_base_same_sample': rb["newton"],
                     'accepted_base_same_sample': rb['accepted']})
        row.update({"spec": code, 'description': descr, 'outcome': var, 'domain': dom,
                     'direction': direction, 'population': POP,
                     "formula": formula(code, var, educ),
                     'n_subj': int(m.PATNO.nunique()), 'n_obsv': int(len(m)),
                     'n_subj_base': n_s_base, 'n_obsv_base': n_o_base,
                     'n_subj_lost': n_s_base - int(m.PATNO.nunique()),
                     'n_obsv_dropped': n_o_base - int(len(m)),
                     'matching_obsv_of_base': bool(len(m) == n_o_base)})
        rows.append(row)

T = pd.DataFrame(rows)
T['family_n'] = N_FAM
T['family'] = (f'i {N_FAM} primary outcomes within the specification (Benjamini-Hochberg); '
                 f'covariates added outside the family')
T['units'] = (f'beta, SE, CI: units of the outcome per year per 1 SD of log T50; LEDD per '
              f'{LEDD_UNITS:g} mg/day; durations in years')
T['p_BH_within_spec'] = np.nan
for code, _ in SPEC:
    i = T.spec == code
    assert int(i.sum()) == N_FAM, f'{code}: {int(i.sum())} models instead of {N_FAM}'
    T.loc[i, 'p_BH_within_spec'] = multipletests(T.loc[i, "p"], method="fdr_bh")[1]
T['nominal'] = T.p < ALPHA
T['exceeds_BH'] = T.p_BH_within_spec < ALPHA
b0 = T[T.spec == "S0"].set_index('outcome')
T["beta_base"] = T.outcome.map(b0.beta)
T["se_base"] = T.outcome.map(b0.se)
T["delta_beta"] = T.beta - T.beta_base
T['delta_in_SE_base'] = T.delta_beta / T.se_base
T['ratio_SE'] = T.se / T.se_base
T['sign_changes'] = np.sign(T.beta) != np.sign(T.beta_base)
# the same comparisons against the base refitted on the SAME sample (for the
# specifications on the observations of the base the two comparisons coincide)
T['p_BH_base_same_sample'] = np.nan
for code, _ in SPEC:
    i = T.spec == code
    T.loc[i, 'p_BH_base_same_sample'] = multipletests(
        T.loc[i, 'p_base_same_sample'], method="fdr_bh")[1]
T['nominal_base_same_sample'] = T.p_base_same_sample < ALPHA
T['delta_in_SE_base_same_sample'] = ((T.beta - T.beta_base_same_sample)
                                         / T.se_base_same_sample)
T['ratio_SE_same_sample'] = T.se / T.se_base_same_sample
T['sign_changes_same_sample'] = (np.sign(T.beta)
                                     != np.sign(T.beta_base_same_sample))
i = T.matching_obsv_of_base
assert np.array_equal(T.loc[i, 'beta_base_same_sample'].to_numpy(),
                      T.loc[i, "beta_base"].to_numpy()), \
    "base on the same sample different from the base where the sample is the same"
# internal identity: row S0 is the base reproduced in section 1
assert np.allclose(b0.loc[OUTCOMES, "beta"], R.beta_strict, atol=0, rtol=0), \
    "row S0 is not the base of section 1"
# The specifications that must rest on the same observations do so
for code in ("S0", "A", "B", "C", "D", "F"):
    assert T.loc[T.spec == code, 'matching_obsv_of_base'].all(), \
        f'{code}: sample different from that of the base'
# The p reported is the two-sided Wald p, on ALL the rows: that of the
# interaction and those of the added covariates.
SC_WALD = {'z(T50) x years interaction': float(
    np.max(np.abs(T.p - wald_bilateral(T.beta, T.se))))}
for name in NAMES_AGG:
    j = T[f'{name}_p'].notna()
    if j.any():
        SC_WALD[name] = float(np.max(np.abs(
            T.loc[j, f'{name}_p'] - wald_bilateral(T.loc[j, f'{name}_beta'],
                                                    T.loc[j, f'{name}_se']))))
assert max(SC_WALD.values()) < TOL_WALD, \
    f"a reported p is not the two-sided Wald p of its coefficient: {SC_WALD}"
N_WALD = {'non-motor': len(T)}

COLS = (["spec", 'description', 'population', 'units', 'family', 'domain', 'outcome',
         'direction', "formula",
         'n_subj', 'n_obsv', 'n_subj_base', 'n_obsv_base', 'n_subj_lost',
         'n_obsv_dropped', 'matching_obsv_of_base',
         "beta", "se", "ci_lo", "ci_hi", "p",
         'p_BH_within_spec', 'family_n', 'nominal', 'exceeds_BH',
         "beta_base", "delta_beta", 'delta_in_SE_base', 'ratio_SE',
         'sign_changes',
         'beta_base_same_sample', 'se_base_same_sample',
         'p_base_same_sample', 'p_BH_base_same_sample',
         'nominal_base_same_sample', 'delta_in_SE_base_same_sample',
         'ratio_SE_same_sample', 'sign_changes_same_sample',
         'newton_base_same_sample', 'accepted_base_same_sample']
        + [f"{n}_{q}" for n in NAMES_AGG for q in ("beta", "se", "p")]
        + COLS_DIAG)
T = T[COLS]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
for code, descr in SPEC:
    s = T[T.spec == code]
    log(f'\n--- {code}: {descr} ---')
    log(f'[{POP}; uncorrected p; p_BH = Benjamini-Hochberg within the {N_FAM} outcomes of this '
        f'specification]')
    if not s.matching_obsv_of_base.all():
        log(f'[sample reduced relative to the base: subjects lost per outcome '
            f'{int(s.n_subj_lost.min())}-{int(s.n_subj_lost.max())}, observations lost '
            f'{int(s.n_obsv_dropped.min())}-{int(s.n_obsv_dropped.max())};')
        log(f' subjects {int(s.n_subj.min())}-{int(s.n_subj.max())}, observations '
            f'{int(s.n_obsv.min())}-{int(s.n_obsv.max())}]')
    v = s[['outcome', 'n_subj', 'n_obsv', "beta", "se", "ci_lo", "ci_hi", "p",
           'p_BH_within_spec', "newton", 'optimiser_chosen']].copy()
    if not s.matching_obsv_of_base.all():
        v.insert(3, 'obsv_dropped', s.n_obsv_dropped)
    for c in ("beta", "ci_lo", "ci_hi"):
        v[c] = v[c].map(lambda x: f"{x:+.4f}")
    v["se"] = v.se.map(lambda x: f"{x:.4f}")
    v["p"] = v.p.map(lambda x: f"{x:.4f}")
    v['p_BH_within_spec'] = v.p_BH_within_spec.map(lambda x: f"{x:.4f}")
    v["newton"] = v.newton.map(lambda x: f"{x:.1e}")
    log(v.to_string(index=False))
    for name, lbls in LBLS_AGG:
        if s[f'{name}_beta'].notna().any():
            k = int((s[f'{name}_p'] < ALPHA).sum())
            log(f"  coefficient of {lbls}: uncorrected p < {ALPHA} in {k} outcomes out of {len(s)} "
                f"({listing(s.outcome[s[f'{name}_p'] < ALPHA])}); outside the family, descriptive.")

log('\nCoefficients of the added covariates (uncorrected p, outside the family; LEDD per')
log(f'{LEDD_UNITS:g} mg/day, in outcome units; duration x years in outcome units per year per')
log('year of baseline duration), from specifications A, B and D:')
vA = T[T.spec == "A"].set_index('outcome')
vB = T[T.spec == "B"].set_index('outcome')
vD = T[T.spec == "D"].set_index('outcome')
for e in OUTCOMES:
    log(f"   {e:22s} LEDD {vA.loc[e, 'ledd100_beta']:+.4f} (p {vA.loc[e, 'ledd100_p']:.3f})   "
        f"duration x years {vB.loc[e, 'durxyears_beta']:+.4f} (p {vB.loc[e, 'durxyears_p']:.3f})   "
        f"duration at the visit {vD.loc[e, 'durvis_beta']:+.4f} (p {vD.loc[e, 'durvis_p']:.3f})")
log("\nAnd from specifications E, F and G (duration since symptoms x years in outcome units per year")
log(f'per year of duration; LEDD x years in outcome units per year per {LEDD_UNITS:g} mg/day; '
    f'PDTRTMNT in outcome')
log("units, visits with indicator at 1 against visits with indicator at 0):")
vE = T[T.spec == "E"].set_index('outcome')
vF = T[T.spec == "F"].set_index('outcome')
vG = T[T.spec == "G"].set_index('outcome')
for e in OUTCOMES:
    log(f"   {e:22s} duration since symptoms x years {vE.loc[e, 'dursymxyears_beta']:+.4f} (p "
        f"{vE.loc[e, 'dursymxyears_p']:.3f})   LEDD x years {vF.loc[e, 'leddxyears_beta']:+.4f} (p "
        f"{vF.loc[e, 'leddxyears_p']:.3f})   PDTRTMNT {vG.loc[e, 'pdtrtmnt_beta']:+.4f} (p "
        f"{vG.loc[e, 'pdtrtmnt_p']:.3f})")

sigL = vA[vA.ledd100_p < ALPHA]
orient = sigL.direction * sigL.ledd100_beta          # > 0: higher dose, better score
improv, wors = list(sigL.index[orient > 0]), list(sigL.index[orient < 0])
log(f'In the {len(sigL)} outcomes out of {len(vA)} in which the LEDD coefficient has uncorrected p < '
    f'{ALPHA} (specification A),')
log(f'a higher dose goes with a worse score in {len(wors)} ({listing(wors)})')
log(f'and with a better one in {len(improv)} ({listing(improv)}).')
if improv and wors:
    log("A coefficient that changes direction from one group of outcomes to another is compatible with")
    log("a variable that, besides a possible effect of the drug, signals severity and")
    log('prescribing choices; these data do not separate the two.')
sigG = vG[vG.pdtrtmnt_p < ALPHA]
orientG = sigG.direction * sigG.pdtrtmnt_beta
log(f"For the PDTRTMNT indicator (specification G) the outcomes with uncorrected p < {ALPHA} are "
    f"{len(sigG)} out of {len(vG)}:")
log(f'worse score at the visits with indicator at 1 in {int((orientG < 0).sum())} '
    f'({listing(sigG.index[orientG < 0])}),')
log(f'better in {int((orientG > 0).sum())} ({listing(sigG.index[orientG > 0])}).')

# --------------------------------------------------------- diagnostics
title('2bis. CONVERGENCE DIAGNOSTICS OF THE MODELS')
log(f"Models estimated: {len(T)} ({len(SPEC)} specifications x {N_FAM} outcomes). A model is "
    f"accepted if at least one")
log(f"optimiser declares convergence, the Newton decrement at the reported point is < "
    f"{TOL_NEWTON:g},")
log("the profiled Hessian is negative definite and no variance is at zero. The maxima of")
log('Newton and of the gradient are rounded UP to two significant figures.')
for code, _ in SPEC:
    s = T[T.spec == code]
    conv = {m: int(s[f"converged_{m}"].sum()) for m, _ in OPTIMISERS}
    log(f'  {code:2s} accepted {int(s.accepted.sum())}/{len(s)}; Newton at most '
        f'{by_excess(s.newton.max()):.1e}; gradient at most {by_excess(s.grad_inf_norm.max()):.1e}; '
        f'singular {int(s.singular.sum())}; converging {conv}; chosen '
        f'{s.optimiser_chosen.value_counts().sort_index().to_dict()}')
log(f'  All {len(T)} models together: Newton at most {by_excess(T.newton.max()):.1e}; infinity norm '
    f'of the gradient at most {by_excess(T.grad_inf_norm.max()):.1e}; singular '
    f'{int(T.singular.sum())}; at the boundary {int(T.at_edge.sum())}.')
rid = T[~T.matching_obsv_of_base]
if len(rid):
    log(f'  Bases re-estimated on the reduced samples ({listing(sorted(rid.spec.unique()))}): '
        f'{len(rid)} additional models; accepted '
        f'{int(rid.accepted_base_same_sample.sum())}/{len(rid)}; Newton at most '
        f'{by_excess(rid.newton_base_same_sample.max()):.1e}.')
no_acc = T[~T.accepted]
no_acc_rid = T[~T.accepted_base_same_sample]
EVERYONE_ACCEPTED = len(no_acc) == 0 and len(no_acc_rid) == 0
if EVERYONE_ACCEPTED:
    log(f'All {len(T)} models are accepted: the counts that follow rest on stationary fits.')
else:
    log(f'WARNING: {len(no_acc) + len(no_acc_rid)} models NOT accepted; the families that contain '
        f'them are not interpretable:')
    log(no_acc[["spec", 'outcome', "newton", 'atleast_an_convergent',
                 'hessian_def_negative', 'at_edge']].to_string(index=False))
    if len(no_acc_rid):
        log('  bases re-estimated on the reduced sample not accepted: '
            + listing(f"{a} {b}" for a, b in zip(no_acc_rid.spec, no_acc_rid.outcome)))
# how much the rule matters: the first converged one against the reported point
gains = []
for _, r_ in T.iterrows():
    first = next((m for m, _ in OPTIMISERS if r_[f"converged_{m}"]), None)
    gains.append(r_["loglike"] - r_[f'loglike_{first}'] if first else np.nan)
T['gain_vs_first_convergent'] = gains
g_max = T.loc[T.gain_vs_first_convergent.idxmax()]
log(f'Maximum gain in log-likelihood over the first optimiser that converges: '
    f'{g_max.gain_vs_first_convergent:.3g}')
gg = T[T.gain_vs_first_convergent > TOL_NEWTON]
log(f'    (specification {g_max.spec}, outcome {g_max.outcome}); models in which it exceeds '
    f'{TOL_NEWTON:g}: {len(gg)} out of {len(T)}'
    + (": " + ", ".join(f"{a} {b} ({c:.3g})" for a, b, c in
                        zip(gg.spec, gg.outcome, gg.gain_vs_first_convergent))
       if len(gg) else "") + ".")

# ================================================================ 3. LEDD
title("3. LEDD DURING FOLLOW-UP: TWO DEFINITIONS OF 'TREATED'")
bl = L[L.years == 0]
assert set(bl.PATNO) == p150 and len(bl) == len(p150), 'not one baseline row per subject'
n_ledd_bl = int((bl.LEDD > 0).sum())
n_trt_bl = int((bl.PDTRTMNT == 1).sum())
log(f"Population: {POP}. Units: subjects with a visit in the year (one row per subject and year).")
log("Two definitions, always reported side by side: 'LEDD > 0' (non-zero dose recorded at the "
    "visit)")
log("and 'PDTRTMNT = 1' (PPMI treatment indicator).")
log(f'At baseline: LEDD > 0 in {n_ledd_bl} subjects out of {len(bl)}; PDTRTMNT = 1 in {n_trt_bl} out '
    f'of {int(bl.PDTRTMNT.notna().sum())}.')
NO_TREATED_BL = n_ledd_bl == 0 and n_trt_bl == 0
log(f'LEDD missing: {int(L.LEDD.isna().sum())} rows out of {len(L)}; PDTRTMNT missing: '
    f'{int(L.PDTRTMNT.isna().sum())}.')
if int(L.LEDD.isna().sum()) == 0:
    log("In the curated data cut LEDD is never missing: a zero may mean 'no drug'")
    log("or 'dose not recorded', and from this file the two cannot be distinguished.")
both = L[L.PDTRTMNT.notna()]
disc1 = both[(both.PDTRTMNT == 0) & (both.LEDD > 0)]     # dose without indicator
disc2 = both[(both.PDTRTMNT == 1) & (both.LEDD == 0)]    # indicator without dose
n_d1, n_d2 = len(disc1), len(disc2)
if n_d1 + n_d2 > 0:
    log(f'The two definitions do not coincide: out of {len(both)} rows with both, {n_d1} have LEDD > '
        f'0 and')
    log(f'PDTRTMNT = 0, and {n_d2} have PDTRTMNT = 1 and LEDD = 0 '
        f'({100 * (n_d1 + n_d2) / len(both):.1f}% discordant).')
else:
    log(f'The two definitions coincide on all {len(both)} rows that have both.')
if n_d1:
    log(f'The {n_d1} visits with LEDD > 0 and PDTRTMNT = 0: median LEDD {disc1.LEDD.median():.0f} '
        f'mg/day [IQR {disc1.LEDD.quantile(0.25):.0f}-{disc1.LEDD.quantile(0.75):.0f}],')
    log(f'    minimum {disc1.LEDD.min():.0f}, maximum {disc1.LEDD.max():.0f}; median year of visit '
        f'{disc1.years.median():.0f} [IQR '
        f'{disc1.years.quantile(0.25):.0f}-{disc1.years.quantile(0.75):.0f}]; '
        f'{int(((disc1.years >= 1) & (disc1.years <= YEARS_FIRSTS)).sum())} fall in years '
        f'1-{YEARS_FIRSTS}; they concern {disc1.PATNO.nunique()} subjects.')
if n_d2:
    log(f'The {n_d2} visits with PDTRTMNT = 1 and LEDD = 0: median year of visit '
        f'{disc2.years.median():.0f} [IQR '
        f'{disc2.years.quantile(0.25):.0f}-{disc2.years.quantile(0.75):.0f}],')
    log(f'    years {disc2.years.min():.0f}-{disc2.years.max():.0f}; they concern '
        f'{disc2.PATNO.nunique()} subjects. For these visits the zero of LEDD coexists with a '
        f'treatment indicator.')

rows_l = []
for a, s in L.groupby("years", sort=True):
    pos = s[s.LEDD > 0]
    trt = s[s.PDTRTMNT == 1]
    n_known = int(s.PDTRTMNT.notna().sum())
    d1 = s[(s.PDTRTMNT == 0) & (s.LEDD > 0)]
    d2 = s[(s.PDTRTMNT == 1) & (s.LEDD == 0)]
    rows_l.append({
        'year': int(a), 'n_subjects': int(len(s)),
        "n_LEDD_pos": int(len(pos)), 'share_LEDD_pos': len(pos) / len(s),
        'n_PDTRTMNT_known': n_known,
        "n_PDTRTMNT_1": int(len(trt)),
        'share_PDTRTMNT_1': (len(trt) / n_known) if n_known else np.nan,
        'n_PDTRTMNT_missing': int(len(s) - n_known),
        "n_LEDD_pos_PDTRTMNT_0": int(len(d1)),
        "n_LEDD_0_PDTRTMNT_1": int(len(d2)),
        'LEDD_median_LEDD_pos': quantile(pos.LEDD, .5),
        "LEDD_q25_LEDD_pos": quantile(pos.LEDD, .25),
        "LEDD_q75_LEDD_pos": quantile(pos.LEDD, .75),
        'LEDD_median_PDTRTMNT_1': quantile(trt.LEDD, .5),
        "LEDD_q25_PDTRTMNT_1": quantile(trt.LEDD, .25),
        "LEDD_q75_PDTRTMNT_1": quantile(trt.LEDD, .75),
        'LEDD_median_LEDD_pos_PDTRTMNT_0': quantile(d1.LEDD, .5),
        'LEDD_median_everyone': float(s.LEDD.median())})
LD = pd.DataFrame(rows_l)
LD.insert(0, 'population', POP)
LD.insert(1, 'units', 'subjects with a visit in the year; LEDD in mg/day')


def _pct(x):
    return f"{100 * x:.1f}%" if np.isfinite(x) else "-"


def _mg(x):
    return f"{x:.0f}" if np.isfinite(x) else "-"


v = LD[['year', 'n_subjects', "n_LEDD_pos", 'share_LEDD_pos', 'n_PDTRTMNT_known',
        "n_PDTRTMNT_1", 'share_PDTRTMNT_1', 'n_PDTRTMNT_missing',
        "n_LEDD_pos_PDTRTMNT_0", "n_LEDD_0_PDTRTMNT_1"]].copy()
for c in ('share_LEDD_pos', 'share_PDTRTMNT_1'):
    v[c] = v[c].map(_pct)
log('\nShare of treated per year, with the two definitions (share_LEDD_pos = n_LEDD_pos / n_subjects;')
log('share_PDTRTMNT_1 = n_PDTRTMNT_1 / n_PDTRTMNT_known):')
log(v.to_string(index=False))
v = LD[['year', "n_LEDD_pos", 'LEDD_median_LEDD_pos', "LEDD_q25_LEDD_pos",
        "LEDD_q75_LEDD_pos", "n_PDTRTMNT_1", 'LEDD_median_PDTRTMNT_1',
        "LEDD_q25_PDTRTMNT_1", "LEDD_q75_PDTRTMNT_1", "n_LEDD_pos_PDTRTMNT_0",
        'LEDD_median_LEDD_pos_PDTRTMNT_0', 'LEDD_median_everyone']].copy()
for c in v.columns:
    if c.startswith("LEDD_"):
        v[c] = v[c].map(_mg)
log('\nLEDD in mg/day (median and quartiles) among the visits with LEDD > 0, among those with '
    'PDTRTMNT = 1 (zeros')
log('included), among those with LEDD > 0 and PDTRTMNT = 0, and over all:')
log(v.to_string(index=False))
# the first "treated" visit of each subject, with the two definitions: the
# ROW is chosen (sort + drop_duplicates)
earliest = (L[L.LEDD > 0].sort_values(["PATNO", "years"])
         .drop_duplicates("PATNO", keep="first"))
earliest_t = (L[L.PDTRTMNT == 1].sort_values(["PATNO", "years"])
           .drop_duplicates("PATNO", keep="first"))
log(f'\nSubjects with LEDD > 0 recorded at at least one visit: {len(earliest)} out of {len(p150)} '
    f'({100 * len(earliest) / len(p150):.1f}%); never: {len(p150) - len(earliest)}.')
log(f'Subjects with PDTRTMNT = 1 at at least one visit:      {len(earliest_t)} out of {len(p150)} '
    f'({100 * len(earliest_t) / len(p150):.1f}%); never: {len(p150) - len(earliest_t)}.')
log(f'With at least one of the two: {len(set(earliest.PATNO) | set(earliest_t.PATNO))}; with both: '
    f'{len(set(earliest.PATNO) & set(earliest_t.PATNO))}.')
log(f'Year of the first visit with LEDD > 0, among those {len(earliest)}: median '
    f'{earliest.years.median():.0f} [IQR '
    f'{earliest.years.quantile(0.25):.0f}-{earliest.years.quantile(0.75):.0f}];')
log(f'year of the first visit with PDTRTMNT = 1, among those {len(earliest_t)}: median '
    f'{earliest_t.years.median():.0f} [IQR '
    f'{earliest_t.years.quantile(0.25):.0f}-{earliest_t.years.quantile(0.75):.0f}].')
for name, col_n, col_q, col_med in (
        ("LEDD > 0", 'n_subjects', 'share_LEDD_pos', 'LEDD_median_LEDD_pos'),
        ("PDTRTMNT = 1", 'n_PDTRTMNT_known', 'share_PDTRTMNT_1', 'LEDD_median_PDTRTMNT_1')):
    reliab = LD[LD[col_n] >= N_MIN_YEAR]
    a50 = reliab[reliab[col_q] >= 0.5].year.min()
    a90 = reliab[reliab[col_q] >= 0.9].year.min()
    ult = reliab.iloc[-1]
    log(f'{name}, in the years with at least {N_MIN_YEAR} subjects: the share exceeds 50% from year '
        f'{a50:.0f} and 90% from year {a90:.0f};')
    log(f'    at year {ult.year:.0f} (n = {ult[col_n]:.0f}) it is {100 * ult[col_q]:.1f}% with '
        f'median LEDD {ult[col_med]:.0f} mg/day among those visits.')

# Is the exposure associated with the dose? Descriptive model, outside the family.
mL = L[["PATNO", "years", "ledd100", "T50", "age_bl", "SEX", "dur_bl"]].dropna()
t0 = time.time()
rowl, rL = fit_strict("ledd100 ~ T50*years + age_bl + SEX + dur_bl",
                        mL.reset_index(drop=True))
print(f'  . LEDD as outcome ({time.time() - t0:.0f}s)', flush=True)
log('\nLEDD as outcome (descriptive, outside any family, uncorrected p):')
log(f'  LEDD/{LEDD_UNITS:g} ~ z(log T50) x years + baseline age + sex + baseline duration; '
    f'{mL.PATNO.nunique()} subjects, {len(mL)} observations.')
LEDD_OK = rowl is not None and rowl['accepted']
if LEDD_OK:
    sc_ = float(abs(rowl["p"] - wald_bilateral(rowl["beta"], rowl["se"])))
    assert sc_ < TOL_WALD, "LEDD as outcome: the p is not the two-sided Wald p"
    SC_WALD['LEDD as outcome'] = sc_
    N_WALD['LEDD as outcome'] = 1
    log(f"  z(T50) x years: {rowl['beta'] * LEDD_UNITS:+.1f} mg/day per year per SD [95% CI "
        f"{rowl['ci_lo'] * LEDD_UNITS:+.1f}; {rowl['ci_hi'] * LEDD_UNITS:+.1f}], p = "
        f"{rowl['p']:.3f};")
    log(f"  mean slope {float(rL.params['years']) * LEDD_UNITS:+.1f} mg/day per year; Newton "
        f"decrement {rowl['newton']:.1e}.")
    LEDD_ASSOCIATED = rowl["p"] < ALPHA
else:
    log('  the model does not meet the strict criterion: it is not reported.')
    LEDD_ASSOCIATED = None

log("\nWhat the adjustment for LEDD can and cannot show.")
if NO_TREATED_BL:
    log(f"At baseline none of the {len(bl)} subjects is treated, under either of the two")
    log("definitions: LEDD cannot have influenced either T50, measured on baseline")
    log("CSF, or the baseline level of the outcomes. It is therefore not a baseline confounder, and")
    log("everything that LEDD carries arrives AFTER the measurement of the exposure.")
else:
    log(f'WARNING: at baseline {n_ledd_bl} subjects are treated by LEDD > 0 and {n_trt_bl} by '
        f'PDTRTMNT = 1:')
    log("LEDD may also act as a baseline confounder, and it CANNOT be said that everything")
    log("it carries arrives after the measurement of the exposure.")
log("A model with LEDD as a time-varying covariate compares visits at equal")
log('concomitant dose.')
log("It CAN show whether the z(T50) x years coefficient depends on that conditioning:")
log('whether it changes sign, size or status relative to the threshold.')
log("It CANNOT (a) estimate the trajectory that would have occurred without treatment: the")
log("dose coefficient may mix the effect of the drug with the reasons why")
log("it was prescribed, and this script does not observe them; (b) rule out a bias from")
log('conditioning on a variable that comes after baseline, which acts as a mediator or')
log("as a collider if the exposure, or unmeasured causes of the outcome, influence the dose;")
log('(c) distinguish the drug classes: the script reads only the LEDD column of the')
log("curated data cut, which is a total dose; the drug class was not examined.")
if LEDD_ASSOCIATED is True:
    log(f"In these data baseline T50 IS associated with the trajectory of the dose (p = "
        f"{rowl['p']:.3f}): the")
    log('exposure -> dose path exists, and the LEDD-adjusted estimates must be read')
    log('with the caution of point (b).')
elif LEDD_ASSOCIATED is False:
    log(f"In these data no association is detected between baseline T50 and the trajectory of the "
        f"dose (p = {rowl['p']:.3f}):")
    log("the exposure -> dose path, necessary for LEDD to act as a mediator,")
    log("is not detected. Not detected does not mean absent: the interval is the one above.")
log("For this reason the estimates with LEDD, with LEDD x years and with the indicator are sensitivity")
log("analyses and not adjusted primary estimates.")

# ============================================================== 4. SUMMARY
title('4. SUMMARY: HOW MANY SIGNIFICANT, HOW MANY SURVIVE, WHICH CHANGE SIGN')
log(f'Population: {POP}. Family: the {N_FAM} primary outcomes of each specification.')
log(f"'Nominal' = uncorrected p < {ALPHA}; 'passes' = Benjamini-Hochberg p within the family < "
    f"{ALPHA}.")
rows_s = []
for code, descr in SPEC:
    s = T[T.spec == code].set_index('outcome')
    nom = list(s.index[s.nominal])
    upp = list(s.index[s.exceeds_BH])
    seg = list(s.index[s.sign_changes])
    rows_s.append((code, descr, nom, upp, seg, s))
    log(f'\n  {code}  {descr}')
    if s.matching_obsv_of_base.all():
        log(f'     sample: that of the base ({int(s.n_subj.min())}-{int(s.n_subj.max())} subjects, '
            f'{int(s.n_obsv.min())}-{int(s.n_obsv.max())} observations per outcome)')
    else:
        log(f'     reduced sample: {int(s.n_subj.min())}-{int(s.n_subj.max())} subjects, '
            f'{int(s.n_obsv.min())}-{int(s.n_obsv.max())} observations per outcome; relative to the '
            f'base {int(s.n_subj_lost.min())}-{int(s.n_subj_lost.max())} subjects and '
            f'{int(s.n_obsv_dropped.min())}-{int(s.n_obsv_dropped.max())} observations are missing')
    log(f'     nominally significant: {len(nom)} out of {len(s)} ({listing(nom)})')
    pmin = s.p_BH_within_spec.min()
    at_min = list(s.index[np.isclose(s.p_BH_within_spec, pmin, rtol=0, atol=1e-12)])
    log(f'     pass the correction:   {len(upp)} out of {len(s)} ({listing(upp)}); minimum p BH '
        f'{pmin:.3f}, ' + (f'shared by {len(at_min)} outcomes' if len(at_min) > 1
                                       else 'of a single outcome') + f' ({listing(at_min)})')
    if code != "S0":
        log(f'     coefficients that change sign relative to the base: {len(seg)} out of {len(s)} '
            f'({listing(seg)})')
        for e in seg:
            log(f"        {e}: base {s.loc[e, 'beta_base']:+.4f} (|beta|/SE = "
                f"{abs(s.loc[e, 'beta_base'] / s.loc[e, 'se_base']):.2f}) -> {s.loc[e, 'beta']:+.4f} "
                f"(|beta|/SE = {abs(s.loc[e, 'beta'] / s.loc[e, 'se']):.2f})")
        i = s.delta_in_SE_base.abs().idxmax()
        log(f"     maximum shift of the coefficient: {s.loc[i, 'delta_in_SE_base']:+.2f} SE of the "
            f"base ({i}); ratio between SE and SE of the base: "
            f"{s.ratio_SE.min():.2f}-{s.ratio_SE.max():.2f}"
            + ("" if s.matching_obsv_of_base.all()
               else " [sample different from the base: the shift is not due to the covariate alone]"))
        if not s.matching_obsv_of_base.all():
            nomb = list(s.index[s.nominal_base_same_sample])
            segb = list(s.index[s.sign_changes_same_sample])
            ib = s.delta_in_SE_base_same_sample.abs().idxmax()
            log(f'     base re-estimated on the SAME reduced sample: nominal {len(nomb)} out of '
                f'{len(s)} ({listing(nomb)}); minimum p BH {s.p_BH_base_same_sample.min():.3f}')
            log(f"     relative to that base: {len(segb)} out of {len(s)} change sign "
                f"({listing(segb)}); maximum shift {s.loc[ib, 'delta_in_SE_base_same_sample']:+.2f} "
                f"SE ({ib}); ratio between the SEs "
                f"{s.ratio_SE_same_sample.min():.2f}-{s.ratio_SE_same_sample.max():.2f}")

alt = [x for x in rows_s if x[0] != "S0"]
n_upp_max = max(len(x[3]) for x in rows_s)
log("")
if not EVERYONE_ACCEPTED:
    log('VERDICT SUSPENDED: not all models meet the estimation criterion (section 2bis).')
elif n_upp_max == 0:
    log(f'VERDICT. In none of the {len(SPEC)} specifications does an outcome pass the correction '
        f'within the {N_FAM}.')
    log("None of these alternatives brings out an association that survives:")
    for x in alt:
        log(f"   {x[0]}  {x[1]}")
else:
    log('VERDICT. At least one specification has outcomes that pass the correction: '
        + "; ".join(f'{x[0]}: {listing(x[3])}' for x in rows_s if x[3]) + ".")
nom_base = set(rows_s[0][2])
log(f'Outcomes nominally significant in the base: {listing(sorted(nom_base))}. Their uncorrected p '
    f'in the specifications:')
for e in [x for x in OUTCOMES if x in nom_base]:
    log("   " + f"{e:14s}" + "  ".join(
        f'{code} {T[(T.spec == code) & (T.outcome == e)].p.iloc[0]:.3f}' for code, _ in SPEC))
lost = {x[0]: sorted(nom_base - set(x[2])) for x in alt}
new = {x[0]: sorted(set(x[2]) - nom_base) for x in alt}
log("They lose nominal significance: "
    + "; ".join(f'{k}: {listing(v_)}' for k, v_ in lost.items()) + ".")
log('They gain it: ' + "; ".join(f'{k}: {listing(v_)}' for k, v_ in new.items()) + ".")
for k, v_ in new.items():
    for e in v_:
        s_ = T[(T.spec == k) & (T.outcome == e)].iloc[0]
        s0_ = T[(T.spec == "S0") & (T.outcome == e)].iloc[0]
        log(f'   {k} {e}: p {s0_.p:.4f} in the base -> {s_.p:.4f}; beta {s0_.beta:+.4f} -> '
            f'{s_.beta:+.4f}; observations {int(s0_.n_obsv)} -> {int(s_.n_obsv)}; p BH '
            f'{s_.p_BH_within_spec:.3f}'
            + ("." if s_.matching_obsv_of_base else
               f'; base re-estimated on the same {int(s_.n_obsv)} observations: p '
               f'{s_.p_base_same_sample:.4f}, beta {s_.beta_base_same_sample:+.4f}.'))
if any(len(x[4]) for x in alt):
    weak = all(abs(x[5].loc[e, "beta_base"] / x[5].loc[e, "se_base"]) < 1
                 for x in alt for e in x[4])
    log('Sign changes: ' + "; ".join(f'{x[0]}: {listing(x[4])}' for x in alt) + "."
        + (' They concern only coefficients that in the base lie less than one standard error from '
           'zero.' if weak else
           " At least one concerns a coefficient that in the base lies more than one standard error "
           "from zero."))
else:
    log('No coefficient changes sign in any specification.')
log(f'\nMoCA in the {len(SPEC)} specifications (points/year per SD of log T50):')
for code, descr in SPEC:
    s = T[(T.spec == code) & (T.outcome == "moca")].iloc[0]
    log(f'   {code:2s} beta = {s.beta:+.4f}  SE = {s.se:.4f}  95% CI [{s.ci_lo:+.4f}; '
        f'{s.ci_hi:+.4f}]  p = {s.p:.4f}  p BH = {s.p_BH_within_spec:.3f}  (n = {int(s.n_subj)}, obs '
        f'= {int(s.n_obsv)})')

# ====================================== 5. SIGNIFICANCE CRITERION, SCALE
title("5. THE SIGNIFICANCE CRITERION THAT THIS SCRIPT APPLIES, AND THE SCALE")
log('What follows describes the calculations of this script; what the text of the')
log("manuscript says about threshold and test is neither read nor verified here.")
log(f"Family: the {N_FAM} primary outcomes of a specification; each specification is corrected")
log("  on its own, and the alternative specifications are sensitivity analyses of the")
log("  same family, not additional families.")
log("Test: two-sided Wald on the coefficient of the z(log T50) x years interaction, REML estimate.")
log(f'  Verified: on all rows the reported p coincides with 2 x (1 - Phi(|beta/SE|)) within '
    f'{TOL_WALD:g}')
log('  (maximum difference: ' + "; ".join(f"{k} {v_:.1e}" for k, v_ in SC_WALD.items())
    + f"; rows: {N_WALD['non-motor']} non-motor models"
    + (f" and {N_WALD['LEDD as outcome']} of LEDD as outcome"
       if 'LEDD as outcome' in N_WALD else "") + ";")
log("  for MoCA and for the motor outcomes the same check is in sections 6 and 7).")
log(f"Correction: Benjamini-Hochberg within the family; the uncorrected p is reported alongside.")
log(f"Threshold: {ALPHA}. 'Nominal' = uncorrected p < {ALPHA}; 'passes the correction' = corrected p "
    f"< {ALPHA}.")
log(f'Intervals: beta +/- {Z95} x SE.')
log("Effect scale: outcome units per year per 1 SD of standardised log T50;")
log("  effect % = d x beta / |mean annual slope of the outcome|, with d = +1 if a high")
log("  score is better and d = -1 if it is worse.")
log('The script applies no threshold of clinical relevance.')
sR = R.set_index('outcome')
log(f'\nTo give the scale, the base ({POP}) accumulated over 10 years between two patients')
log("1 SD of log T50 apart (95% CI x 10, under the linear model; outcome units):")
for var, dom, direction, educ in PRIMARY:
    s = b0.loc[var]
    pm = sR.loc[var, 'slope_mean_strict']
    log(f'   {var:22s} beta {s.beta:+.4f}  10 years: {10 * s.ci_lo:+.2f} to {10 * s.ci_hi:+.2f}   '
        f'mean slope {pm:+.3f}/year   effect {direction * s.beta / abs(pm) * 100:+.1f}%')

# ============================================ 6. MoCA: COVARIATE x YEARS
title('6. MoCA: THE COVARIATE x YEARS MODELS OF 13_moca_confounders.csv, STRICT CRITERION')
log(f'Population: {POP}; outcome MoCA (points); coefficient: z(log T50) x years, points per')
log('year per 1 SD. Single-outcome models, outside any family: p NOT corrected.')
log('Formulas and keys are those of 13_moca_signal.py (section c), which uses the rule of the')
log('first optimiser that converges. For each model: the archived value; the same')
log('model refitted here with that rule (reproduced or not); the strict criterion; the')
log('log-likelihood gap between the point of the strict criterion and the point at which')
log("the optimiser of the archived rule had stopped.")
f13 = os.path.join(OUT, "13_moca_confounders.csv")
HA_13 = os.path.exists(f13)
if HA_13:
    c13 = pd.read_csv(f13)
    assert not c13.specification.duplicated().any(), \
        '13_moca_confounders.csv has repeated specifications'
    c13 = c13.set_index('specification')
    unknown = sorted(set(c13.index) - {k for k, _, _ in SPEC13})
    if unknown:
        log(f'WARNING: 13_moca_confounders.csv contains specifications that this script does not '
            f'know and does NOT verify: {listing(unknown)}.')
else:
    c13 = None
    log("13_moca_confounders.csv MISSING: the comparison with the archived values is NOT verified;")
    log('the estimates under the strict criterion are reported all the same.')

rows_c = []
for key, f, wants_onset in SPEC13:
    mm = CAMPX["moca"].drop(columns=["PDTRTMNT", "dur_sym"])
    if wants_onset:
        mm = mm[mm.ageonset.notna()].reset_index(drop=True)
    else:
        mm = mm.drop(columns=["ageonset"])
    t0 = time.time()
    row, r, md_ref = fit_strict_extended(f, mm)
    assert row is not None, f'MoCA {key}: no fit under the strict criterion'
    r13, opt13, _ = fit_mixed(f, mm, mm.PATNO)                # rule of 13
    conv13 = bool(r13 is not None and r13.converged)
    if r13 is not None:
        t13 = term(r13, ("T50:years", "years:T50"))
        b13, se13, p13 = (float(r13.params[t13]), float(r13.bse[t13]),
                          float(r13.pvalues[t13]))
        ll13 = _ll(md_ref, pack(r13))
    else:
        b13 = se13 = p13 = ll13 = np.nan
    in_archive = bool(HA_13 and key in c13.index)
    if in_archive:
        a = c13.loc[key]
        a_b, a_se, a_p, a_n = (float(a.beta_T50_x_years), float(a.se), float(a.p),
                               int(a.n_subj))
        reproduced = bool(conv13 and abs(b13 - a_b) < TOL_ID and abs(se13 - a_se) < TOL_ID
                          and abs(p13 - a_p) < TOL_ID
                          and a_n == int(mm.PATNO.nunique()))
    else:
        a_b = a_se = a_p = np.nan
        a_n = -1
        reproduced = False
    rows_c.append({
        'population': POP, 'outcome': "moca", 'specification': key, "formula": f,
        'units': 'beta, SE, CI: MoCA points per year per 1 SD of log T50',
        'family': 'none (uncorrected p values)',
        'n_subj': int(mm.PATNO.nunique()), 'n_obsv': int(len(mm)),
        'in_archive': in_archive,
        'beta_archived': a_b, 'se_archived': a_se, 'p_archived': a_p,
        'n_subj_archived': a_n,
        'optimiser_rule13': opt13, 'converged_rule13': conv13,
        'beta_rule13': b13, 'se_rule13': se13, 'p_rule13': p13,
        'reproduced_rule13': reproduced,
        "beta": row["beta"], "se": row["se"], "ci_lo": row["ci_lo"],
        "ci_hi": row["ci_hi"], "p": row["p"],
        'loglike_rule13': ll13,
        'discrepancy_loglike': row["loglike"] - ll13,
        'delta_beta_vs_archived': row["beta"] - a_b,
        'delta_se_vs_archived': row["se"] - a_se,
        'delta_p_vs_archived': row["p"] - a_p,
        'ratio_se_vs_archived': row["se"] / a_se if in_archive else np.nan,
        f'beta_equal_a_{DECIMALS_PRINTOUT}_decimals': bool(
            in_archive and round(row["beta"], DECIMALS_PRINTOUT)
            == round(a_b, DECIMALS_PRINTOUT)),
        f'p_equal_a_{DECIMALS_PRINTOUT}_decimals': bool(
            in_archive and round(row["p"], DECIMALS_PRINTOUT)
            == round(a_p, DECIMALS_PRINTOUT)),
        **{c: row[c] for c in COLS_DIAG}})
    print(f"  . MoCA {key:28s} newton {row['newton']:.1e} ({time.time() - t0:.0f}s)",
          flush=True)
MC = pd.DataFrame(rows_c)
sc_ = float(np.max(np.abs(MC.p - wald_bilateral(MC.beta, MC.se))))
assert sc_ < TOL_WALD, "MoCA covariate x years: a p is not the two-sided Wald p"
SC_WALD['MoCA covariate x years'] = sc_
N_WALD['MoCA covariate x years'] = len(MC)
log(f'reported p: two-sided Wald, verified on the {len(MC)} rows (maximum difference {sc_:.1e}, '
    f'tolerance {TOL_WALD:g}).')
# internal identities: the base model is row S0 of the MoCA, duration x years
# is row B of the MoCA (same formula, same observations)
mcI = MC.set_index('specification')
# (the formula of the duration is the same string as specification B, and the
# fit must be the same number; that of the base model has the covariates in
# another order, and coincides with S0 only within numerical resolution).
SC_INT = {}
for key, code, tol in ((KEY13_BASE, "S0", TOL_ID), (KEY13_DURATION, "B", 1e-12)):
    ref = T[(T.spec == code) & (T.outcome == "moca")].iloc[0]
    SC_INT[key] = max(abs(mcI.loc[key, q] - ref[q]) for q in ("beta", "se", "p"))
    assert SC_INT[key] < tol and int(mcI.loc[key, 'n_obsv']) == int(ref.n_obsv), \
        (f"MoCA '{key}' does not coincide with row {code} of section 2 (difference "
         f"{SC_INT[key]:.3g}, tolerance {tol:g})")
log(f"Internal identities: '{KEY13_BASE}' against row S0 of MoCA (same model, covariates in a "
    f"different order): maximum difference {SC_INT[KEY13_BASE]:.1e};")
log(f"   '{KEY13_DURATION}' against row B of MoCA (same formula): maximum difference "
    f"{SC_INT[KEY13_DURATION]:.1e}.")
b_col, p_col = (f'beta_equal_a_{DECIMALS_PRINTOUT}_decimals',
                f'p_equal_a_{DECIMALS_PRINTOUT}_decimals')
for _, x in MC.iterrows():
    log(f"\n  {x.specification}   [{x.formula.split('~')[1].strip()}]")
    log(f'     {int(x.n_subj)} subjects, {int(x.n_obsv)} observations')
    if x.in_archive:
        log(f'     archived:          beta = {x.beta_archived:.4f}  SE = {x.se_archived:.4f}  p = '
            f'{x.p_archived:.4f}  (n = {int(x.n_subj_archived)})')
        log(f'     refitted, rule 13 ({x.optimiser_rule13}): '
            + (f'reproduced within {TOL_ID:g}' if x.reproduced_rule13 else
               f'NOT reproduced (beta = {x.beta_rule13:.4f}, SE = {x.se_rule13:.4f}, p = '
               f'{x.p_rule13:.4f}, declared convergence: {x.converged_rule13}); the comparison that '
               f'follows is not between the same fits'))
    else:
        log('     archived:          row MISSING from 13_moca_confounders.csv: comparison NOT '
            'verified')
        log(f'     refitted, rule 13 ({x.optimiser_rule13}): beta = {x.beta_rule13:.4f}, SE = '
            f'{x.se_rule13:.4f}, p = {x.p_rule13:.4f}')
    log(f"     strict criterion:  beta = {x.beta:.4f}  SE = {x.se:.4f}  95% CI [{x.ci_lo:+.4f}; "
        f"{x.ci_hi:+.4f}]  p = {x.p:.4f}  Newton {x.newton:.1e}  accepted: "
        f"{('si' if x.accepted else 'NO')}")
    log(f'     log-likelihood gap from the point of rule 13: {x.discrepancy_loglike:.3g}')
    if x.in_archive:
        log(f'     strict - archived: beta {x.delta_beta_vs_archived:+.2e}, SE '
            f'{x.delta_se_vs_archived:+.2e}, p {x.delta_p_vs_archived:+.2e}; to {DECIMALS_PRINTOUT} '
            f'decimals beta '
            + ('equal' if x[b_col] else 'DIFFERENT') + ", p "
            + ('equal' if x[p_col] else 'DIFFERENT') + ".")
        if abs(x.ratio_se_vs_archived - 1) > 0.1:
            log("     The two standard errors differ by more than 10%: the archived standard")
            log("     error is a property of the point at which the optimiser had")
            log('     stopped, not of the model.')
arc = MC[MC.in_archive]
if len(arc):
    different = arc[~(arc[b_col] & arc[p_col])]
    log(f'\nSummary, {len(arc)} models present in the archive: reproduced with rule 13 '
        f'{int(arc.reproduced_rule13.sum())} out of {len(arc)};')
    log(f'   accepted under the strict criterion {int(arc.accepted.sum())} out of {len(arc)}; equal '
        f'to the archived value to {DECIMALS_PRINTOUT} decimals in both beta and p: '
        f'{len(arc) - len(different)} out of {len(arc)};')
    log('   different: ' + (listing(
        f"'{a}' (beta {b:.3f} -> {c:.3f}, p {d_:.3f} -> {e:.3f}, gap {g:.3g})"
        for a, b, c, d_, e, g in zip(different.specification, different.beta_archived,
                                     different.beta, different.p_archived, different.p,
                                     different.discrepancy_loglike))) + ".")
    i = arc.discrepancy_loglike.idxmax()
    log(f"   maximum log-likelihood gap: {arc.loc[i, 'discrepancy_loglike']:.3g} "
        f"('{arc.loc[i, 'specification']}'); in the others at most "
        f"{by_excess(arc.drop(index=i).discrepancy_loglike.max()):.1e}.")
if not MC.accepted.all():
    log('WARNING: MoCA models that do NOT meet the strict criterion: '
        + listing(MC.specification[~MC.accepted]) + ".")

# ============================================ 7. MOTOR OUTCOMES WITH LEDD
title(f'7. MOTOR OUTCOMES WITH TIME-VARYING LEDD: THE {N_FAM_M} MODELS OF 11, STRICT CRITERION')
log(f'Population: {POP}. The {N_FAM_M} models are {len(ENGINES)} outcomes x {len(KINETICS)} kinetic '
    f'parameters:')
log(f"   outcomes: {listing((f'{v} ({e})' for v, e in ENGINES))};")
log(f'   kinetic parameters: {listing(KINETICS)}, each as a logarithm standardised on the '
    f'{len(c150)} subjects (1 SD).')
log("Model of 11_longitudinal.py ('Sensitivity 1'): outcome ~ z(log kinetic parameter) x years + "
    "baseline age")
log('+ sex + baseline duration + LEDD (mg/day at the visit), random intercept and slope,')
log("REML; in 11 estimated with lbfgs alone (at most 200 iterations), without recording")
log(f'convergence. Family: the {N_FAM_M} models, Benjamini-Hochberg within the {N_FAM_M}, as in 11.')
log('Here: the same models under the strict criterion, WITH LEDD (same formula) and')
log('WITHOUT (same formula minus LEDD), on the same observations.')
for out, lbls in ENGINES:
    m = SAMP_M[out]
    log(f'   {out:15s} {m.PATNO.nunique()} subjects, {len(m)} observations, of which '
        f'{int((m.years > 0).sum())} at follow-up and {int((m.LEDD > 0).sum())} with LEDD > 0')

f63 = os.path.join(OUT, "88_diagnostics_63.csv")
HA_63 = os.path.exists(f63)
if HA_63:
    d63 = pd.read_csv(f63)
    d63 = d63[d63.cohort == "150h"].set_index(["outcome", "kinetic"])
rows_m = []
SC63 = {"beta": 0.0, "se": 0.0, "p": 0.0}
N63 = 0
for out, lbls in ENGINES:
    m = SAMP_M[out]
    for k in KINETICS:
        t0 = time.time()
        pair = {}
        for having_ledd in (False, True):
            f = formula_m(out, k, having_ledd)
            row, r, md_ref = fit_strict_extended(f, m, kinet=k)
            assert row is not None, f'motor {out} {k}: no fit'
            row.update({'population': POP,
                         "spec": "M1" if having_ledd else "M0",
                         'description': ('with time-varying LEDD' if having_ledd
                                         else 'without LEDD'),
                         'outcome': out, 'label': lbls, 'kinetic': k, "formula": f,
                         'n_subj': int(m.PATNO.nunique()), 'n_obsv': int(len(m)),
                         'n_obsv_followup': int((m.years > 0).sum()),
                         'n_obsv_LEDD_pos': int((m.LEDD > 0).sum())})
            for q in ("LEDD_beta", "LEDD_se", "LEDD_p", 'beta_archived_11',
                      'p_archived_11', 'p_BH_archived_11', "beta_lbfgs200",
                      "p_lbfgs200", 'discrepancy_loglike_fr_lbfgs200'):
                row[q] = np.nan
            row["converged_lbfgs200"] = None
            if having_ledd:
                row["LEDD_beta"] = float(r.params["LEDD"])
                row["LEDD_se"] = float(r.bse["LEDD"])
                row["LEDD_p"] = float(r.pvalues["LEDD"])
                v11 = RIPR11[(out, k)]
                row["beta_lbfgs200"] = v11["beta"]
                row["p_lbfgs200"] = v11["p"]
                row["converged_lbfgs200"] = v11["converged"]
                row['discrepancy_loglike_fr_lbfgs200'] = (
                    row["loglike"] - _ll(md_ref, pack(v11["r"])))
                if HA_11:
                    row['beta_archived_11'] = float(arch11.loc[(out, k), "beta"])
                    row['p_archived_11'] = float(arch11.loc[(out, k), "p"])
                    row['p_BH_archived_11'] = float(arch11.loc[(out, k), "p_BH"])
            elif HA_63 and (out, k) in d63.index:
                # the model without LEDD is one of the 63 of the grid: identity
                for q, c in (("beta", "beta_interaction"), ("se", "se"), ("p", "p")):
                    sc_ = abs(row[q] - float(d63.loc[(out, k), c]))
                    assert sc_ < TOL_ID, (f'motor {out} {k} without LEDD: {q} different from '
                                          f'88_diagnostics_63.csv (difference {sc_:.3g})')
                    SC63[q] = max(SC63[q], sc_)
                assert int(d63.loc[(out, k), "n_obs"]) == int(len(m)), \
                    f'motor {out} {k}: observations different from 88_diagnostics_63.csv'
                N63 += 1
            pair[having_ledd] = row
            rows_m.append(row)
        print(f"  . motor {out:15s} {k:6s} newton {pair[False]['newton']:.1e} / "
              f"{pair[True]['newton']:.1e} ({time.time() - t0:.0f}s)", flush=True)

TM = pd.DataFrame(rows_m)
TM['family_n'] = N_FAM_M
TM['family'] = (f'i {N_FAM_M} models ({len(ENGINES)} outcomes x {len(KINETICS)} kinetic '
                  f'parameters) within the specification (Benjamini-Hochberg); LEDD outside the '
                  f'family')
TM['units'] = ('beta, SE, CI: points of the outcome per year per 1 SD of the log of the kinetic '
               'parameter; LEDD_beta: points per mg/day')
TM['p_BH_within_family'] = np.nan
for code in ("M0", "M1"):
    i = TM.spec == code
    assert int(i.sum()) == N_FAM_M, f'{code}: {int(i.sum())} models instead of {N_FAM_M}'
    TM.loc[i, 'p_BH_within_family'] = multipletests(TM.loc[i, "p"], method="fdr_bh")[1]
TM['nominal'] = TM.p < ALPHA
TM['exceeds_BH'] = TM.p_BH_within_family < ALPHA
m0 = TM[TM.spec == "M0"].set_index(['outcome', 'kinetic'])
keys_m = list(zip(TM.outcome, TM.kinetic))
TM['beta_without_LEDD'] = [m0.loc[c, "beta"] for c in keys_m]
TM['se_without_LEDD'] = [m0.loc[c, "se"] for c in keys_m]
TM["delta_beta"] = TM.beta - TM.beta_without_LEDD
TM['delta_in_SE_without_LEDD'] = TM.delta_beta / TM.se_without_LEDD
TM['ratio_SE'] = TM.se / TM.se_without_LEDD
TM['sign_changes'] = np.sign(TM.beta) != np.sign(TM.beta_without_LEDD)
sc_ = float(np.max(np.abs(TM.p - wald_bilateral(TM.beta, TM.se))))
j = TM.LEDD_p.notna()
sc_ = max(sc_, float(np.max(np.abs(
    TM.loc[j, "LEDD_p"] - wald_bilateral(TM.loc[j, "LEDD_beta"], TM.loc[j, "LEDD_se"])))))
assert sc_ < TOL_WALD, "motor: a p is not the two-sided Wald p"
SC_WALD['engines'] = sc_
N_WALD['engines'] = len(TM)
log(f'\nreported p (interaction and LEDD): two-sided Wald, verified on the {len(TM)} rows (maximum '
    f'difference {sc_:.1e}, tolerance {TOL_WALD:g}).')
COLS_M = (['population', 'units', 'family', "spec", 'description', 'outcome',
           'label', 'kinetic',
           "formula", 'n_subj', 'n_obsv', 'n_obsv_followup', 'n_obsv_LEDD_pos',
           "beta", "se", "ci_lo", "ci_hi", "p", 'p_BH_within_family', 'family_n',
           'nominal', 'exceeds_BH', 'beta_without_LEDD', "delta_beta",
           'delta_in_SE_without_LEDD', 'ratio_SE', 'sign_changes',
           "LEDD_beta", "LEDD_se", "LEDD_p",
           'beta_archived_11', 'p_archived_11', 'p_BH_archived_11',
           "beta_lbfgs200", "p_lbfgs200", "converged_lbfgs200",
           'discrepancy_loglike_fr_lbfgs200'] + COLS_DIAG)
TM = TM[COLS_M]

if HA_63:
    log(f'\nModels without LEDD present in 88_diagnostics_63.csv (150 h, strict criterion): {N63} '
        f'out of {N_FAM_M};')
    log(f"   maximum difference beta {SC63['beta']:.1e}, SE {SC63['se']:.1e}, p {SC63['p']:.1e} "
        f"(tolerance {TOL_ID:g}).")
    if N63 < N_FAM_M:
        log(f"   For the other {N_FAM_M - N63} the identity is NOT verified.")
else:
    log("\n88_diagnostics_63.csv MISSING: the identity of the models without LEDD with the grid of "
        "63 is NOT verified.")
for code, descr in (("M0", 'without LEDD'), ("M1", 'with time-varying LEDD')):
    s = TM[TM.spec == code]
    log(f'\n--- {code}: {descr} ---')
    log(f"[{POP}; coefficient: z(log kinetic parameter) x years, outcome points per year per 1 SD; "
        f"uncorrected p;")
    log(f' p_BH = Benjamini-Hochberg within the {N_FAM_M} models of this specification]')
    v = s[['outcome', 'kinetic', 'n_subj', 'n_obsv', "beta", "se", "ci_lo", "ci_hi", "p",
           'p_BH_within_family', "newton", 'optimiser_chosen']].copy()
    for c in ("beta", "ci_lo", "ci_hi"):
        v[c] = v[c].map(lambda x: f"{x:+.4f}")
    v["se"] = v.se.map(lambda x: f"{x:.4f}")
    v["p"] = v.p.map(lambda x: f"{x:.4f}")
    v['p_BH_within_family'] = v.p_BH_within_family.map(lambda x: f"{x:.4f}")
    v["newton"] = v.newton.map(lambda x: f"{x:.1e}")
    log(v.to_string(index=False))
    log(f'  accepted {int(s.accepted.sum())}/{len(s)}; Newton at most '
        f'{by_excess(s.newton.max()):.1e}; gradient at most {by_excess(s.grad_inf_norm.max()):.1e} '
        f'(rounded up); singular {int(s.singular.sum())}; at the boundary {int(s.at_edge.sum())}.')

M1 = TM[TM.spec == "M1"].set_index(['outcome', 'kinetic'])
M0 = TM[TM.spec == "M0"].set_index(['outcome', 'kinetic'])
ENGINES_ACCEPTED = bool(TM.accepted.all())
log('\nSummary of the motor models.')
if not ENGINES_ACCEPTED:
    log('WARNING: not all motor models meet the strict criterion; the counts that follow are not '
        'interpretable:')
    log(TM.loc[~TM.accepted, ["spec", 'outcome', 'kinetic', "newton"]].to_string(index=False))
for code, s in (('M0 without LEDD', M0), ('M1 with LEDD  ', M1)):
    nom = [f"{a} {b}" for a, b in s.index[s.nominal]]
    upp = [f"{a} {b}" for a, b in s.index[s.exceeds_BH]]
    log(f'   {code}: nominally significant {len(nom)} out of {len(s)} ({listing(nom)}); pass the '
        f'correction {len(upp)} out of {len(s)} ({listing(upp)});')
    log(f'                  minimum p {s.p.min():.3f}, minimum p BH {s.p_BH_within_family.min():.3f}.')
seg = [f"{a} {b}" for a, b in M1.index[M1.sign_changes]]
i = M1.delta_in_SE_without_LEDD.abs().idxmax()
log(f'   With LEDD {len(seg)} coefficients out of {len(M1)} change sign ({listing(seg)}).')
for ch in M1.index[M1.sign_changes]:
    log(f"      {ch[0]} {ch[1]}: without LEDD {M0.loc[ch, 'beta']:+.4f} (|beta|/SE = "
        f"{abs(M0.loc[ch, 'beta'] / M0.loc[ch, 'se']):.2f}) -> with LEDD {M1.loc[ch, 'beta']:+.4f} "
        f"(|beta|/SE = {abs(M1.loc[ch, 'beta'] / M1.loc[ch, 'se']):.2f})")
log(f"   Maximum shift of the coefficient: {M1.loc[i, 'delta_in_SE_without_LEDD']:+.3f} SE of the "
    f"model without LEDD ({i[0]}, {i[1]}:")
log(f"      {M0.loc[i, 'beta']:+.4f} -> {M1.loc[i, 'beta']:+.4f}); ratio between the SEs: "
    f"{M1.ratio_SE.min():.3f}-{M1.ratio_SE.max():.3f}.")
kL = int((M1.LEDD_p < ALPHA).sum())
log(f"   LEDD coefficient (outcome points per mg/day; outside the family, uncorrected p): p < "
    f"{ALPHA} in {kL} models out of {len(M1)};")
for out, lbls in ENGINES:
    s = M1.loc[out]
    log(f'      {out:15s} from {LEDD_UNITS * s.LEDD_beta.min():+.3f} to '
        f'{LEDD_UNITS * s.LEDD_beta.max():+.3f} points per {LEDD_UNITS:g} mg/day across the {len(s)} '
        f'kinetic parameters; maximum p {s.LEDD_p.max():.2g}.')
log(f'   Strict criterion against the lbfgs alone of 11 (models with LEDD): convergence declared by '
    f'lbfgs in {int(M1.converged_lbfgs200.astype(bool).sum())} out of {len(M1)};')
log(f'      maximum difference |beta| {float((M1.beta - M1.beta_lbfgs200).abs().max()):.1e}, |p| '
    f'{float((M1.p - M1.p_lbfgs200).abs().max()):.1e}; log-likelihood gap at most '
    f'{by_excess(M1.discrepancy_loglike_fr_lbfgs200.max()):.1e}.')
if HA_11:
    n_nom11 = int((arch11.p < ALPHA).sum())
    n_bh11 = int((arch11.p_BH < ALPHA).sum())
    log(f'   In 11_ledd_adjusted.csv (archived): nominal {n_nom11} out of {len(arch11)}, pass the '
        f'correction {n_bh11} out of {len(arch11)}; minimum p BH {arch11.p_BH.min():.3f}.')
    same = (n_nom11 == int(M1.nominal.sum()) and n_bh11 == int(M1.exceeds_BH.sum()))
    log('   Under the strict criterion the two counts ' + (
        'remain the same.' if same else
        f'CHANGE: nominal {int(M1.nominal.sum())}, pass {int(M1.exceeds_BH.sum())}.'))
else:
    log('   11_ledd_adjusted.csv MISSING: comparison with the archived counts NOT verified.')

# Where is the modelled part III missing? It matters for how the result is read.
log("\nWhere the modelled part III (updrs3_score) is missing. Units: follow-up visits (years > 0)")
log(f'with part II recorded; {POP}.')
fu2 = Lfu[Lfu.updrs2_score.notna()]
for out3 in ("updrs3_score", "updrs_totscore"):
    without = fu2[fu2[out3].isna()]
    con = fu2[fu2[out3].notna()]
    log(f'   {out3}: of the {len(fu2)} follow-up visits with part II, {len(without)} do not have '
        f'{out3} and {len(con)} have it.')
    log(f'      Among the {len(without)} without: LEDD > 0 in {int((without.LEDD > 0).sum())}; '
        f'PDTRTMNT = 1 in {int((without.PDTRTMNT == 1).sum())} (indicator known in '
        f'{int(without.PDTRTMNT.notna().sum())}).')
    log(f'      Among the {len(con)} with:    LEDD > 0 in {int((con.LEDD > 0).sum())}; PDTRTMNT = 1 '
        f'in {int((con.PDTRTMNT == 1).sum())} (indicator known in '
        f'{int(con.PDTRTMNT.notna().sum())}).')
without3 = fu2[fu2.updrs3_score.isna()]
tr_l, untr_l = fu2[fu2.LEDD > 0], fu2[fu2.LEDD == 0]
tr_p, untr_p = fu2[fu2.PDTRTMNT == 1], fu2[fu2.PDTRTMNT == 0]
log('   Share of the follow-up visits with part II that also have updrs3_score:')
log(f'      LEDD > 0:     {int(tr_l.updrs3_score.notna().sum())} out of {len(tr_l)} '
    f'({_pct(tr_l.updrs3_score.notna().mean() if len(tr_l) else np.nan)});   LEDD = 0:     '
    f'{int(untr_l.updrs3_score.notna().sum())} out of {len(untr_l)} '
    f'({_pct(untr_l.updrs3_score.notna().mean() if len(untr_l) else np.nan)})')
log(f'      PDTRTMNT = 1: {int(tr_p.updrs3_score.notna().sum())} out of {len(tr_p)} '
    f'({_pct(tr_p.updrs3_score.notna().mean() if len(tr_p) else np.nan)});   PDTRTMNT = 0: '
    f'{int(untr_p.updrs3_score.notna().sum())} out of {len(untr_p)} '
    f'({_pct(untr_p.updrs3_score.notna().mean() if len(untr_p) else np.nan)})')
if HA_ON:
    log(f'   The curated data cut also has the updrs3_score_on column: at follow-up it is recorded '
        f'in {int(Lfu.updrs3_score_on.notna().sum())} visits, and in '
        f'{int(without3.updrs3_score_on.notna().sum())} of the {len(without3)} without updrs3_score. '
        f'It is not modelled here.')
else:
    log('   The curated data cut that was read does not have the updrs3_score_on column.')
if len(without3) and (without3.LEDD > 0).mean() > 0.5:
    log(f'   The modelled part III is missing mainly at the treated visits: '
        f'{100 * (without3.LEDD > 0).mean():.1f}% of the visits without updrs3_score have LEDD > 0.')
    log('   The models of part III and of the total (which contains it) therefore rest on only')
    log("   a part of the treated visits; the result 'LEDD changes nothing' holds")
    log("   for the visits at which that score was recorded, not for all the treated")
    log(f"   visits. Part II (updrs2_score) is recorded in {len(fu2)} of the {len(Lfu)} follow-up "
        f"visits ({100 * len(fu2) / len(Lfu):.1f}%),")
    log(f'   and in {int(Lfu[Lfu.LEDD > 0].updrs2_score.notna().sum())} of the '
        f'{int((Lfu.LEDD > 0).sum())} with LEDD > 0.')
elif len(without3):
    log(f'   The visits without updrs3_score are not mostly treated '
        f'({100 * (without3.LEDD > 0).mean():.1f}% with LEDD > 0).')
else:
    log('   No follow-up visit with part II lacks updrs3_score.')

# ================================================================== output
T.to_csv(os.path.join(OUT, "107_nonmotor_specifications.csv"), index=False)
R.insert(0, 'population', POP)
R.to_csv(os.path.join(OUT, "107_reproduction_S9.csv"), index=False)
LD.to_csv(os.path.join(OUT, "107_ledd_by_year.csv"), index=False)
MC.to_csv(os.path.join(OUT, "107_moca_covariate_x_years.csv"), index=False)
TM.to_csv(os.path.join(OUT, "107_motor_ledd.csv"), index=False)
with open(os.path.join(OUT, "107_nonmotor_ledd_duration.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
print(f'\nWritten in {OUT}: 107_nonmotor_specifications.csv, 107_reproduction_S9.csv, '
      f'107_ledd_by_year.csv, 107_moca_covariate_x_years.csv, 107_motor_ledd.csv, '
      f'107_nonmotor_ledd_duration.txt')
