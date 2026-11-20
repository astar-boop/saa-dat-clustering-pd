"""
111 - Lateralised motor score (MDS-UPDRS III by side of the body).

ANALYSIS ADDED IN REVIEW: it was not in the plan of the submitted manuscript.

Why it exists. A reviewer asks whether "the most recent calculation" of
Dzialas et al., Ann Neurol 2025;98:120-135 (doi 10.1002/ana.27223) can be used.
Read in full, the article does not recalculate the binding ratios: it shows that
the decline of the LESS affected putamen follows the worsening of the MDS-UPDRS
III of the CONTRALATERAL side of the body. Script 110 has already dealt with the
putaminal side (less and more affected putamen as outcomes of the kinetic
parameters). The other half was missing: the lateralised motor score. An
independent check observed that, if left out, the omission reads as an evasion.
This script builds it with the rules PUBLISHED in the article and uses it in two
ways.

Rules (for each: the passage of the article, or "the article is silent"; where
it is silent the most cautious option is chosen and the alternative is a
sensitivity analysis). The same rules are written, with the derived counts, in
111_rules.csv.

  R1 lateralised items. Methods, "Clinical Scores": the akinetic-rigid items of
     the limbs and the tremor items are separated by side and then added to the
     non-lateralised axial items; the total score is less affected + more
     affected + axial. The list of items is NOT given. Choice: the lateralised
     items are all and only the items that the scale rates separately on the
     right and on the left; "axial" are all the others (the identity total =
     sides + axial requires it). Sensitivity: score of the limbs only.
  R2 tremor. Methods, "Clinical Scores": scores with and without the tremor
     items; the list is NOT given. The means of Table 2 imply that removing
     tremor also lowers the non-lateralised part: all the tremor items are
     removed (3.15-3.18), including jaw and constancy.
  R3 medication state. Methods, "Data", criterion 3: MDS-UPDRS III "in the off
     medication state". How the state is read from the PPMI files is NOT stated.
     Choice: examination declared OFF, or examination of a participant not on
     treatment; it is the definition of updrs3_score in the dictionary of the
     curated data cut, and the identity with that score is verified visit by
     visit.
  R4 items that cannot be rated. The article is silent. Choice: the visit counts
     only if all the items of part III are rated 0-4 (the curated data cut does
     the same). Sensitivity: pro-rating within block with at least 80% of the
     items rated.
  R5 side. Methods, "Dopamine Transporter SPECT" and "Clinical Scores": more
     and less affected hemisphere for each examination date from the asymmetry
     index (S - D)/(S + D), thresholds 0% and 5%; the side of the body
     contralateral to the more affected hemisphere is the "more affected" one.
     For the outcomes of the kinetic parameters the side is fixed ONCE, from the
     baseline uptake (U): declared deviation, because part of the clinical
     visits have no scan (the proportion is in the log) and the side cannot be
     chosen again there.
     Alternative: side of onset of symptoms (D). In the
     positive control the rule of the article applies, at each date (V).
  R6 symmetric. Methods, "Clinical Scores": without asymmetry the mean of the
     two side scores is used. Applied to ties in uptake. With the side of
     onset (D) symmetric or missing, the participant is excluded: the article
     is silent.
  R7 time. Methods, "Defining the Timeline": the scan dates make the timeline,
     clinical examination within +/-3 months, time in months from the first
     scan. Applied in the positive control. For the kinetic parameters the
     visits and the time of the project (YEAR) apply.
  R8 model of the article. "Statistical Analysis": SBR, time, SBR x time;
     antidepressants, LEDD and MoCA time-varying; age, sex, handedness,
     education, duration; random intercept and slope.

What it computes, in the two analysis cohorts of the current working directory:

  1. Kinetic parameter x time on the lateralised scores: side of the body
     contralateral to the less affected putamen, side contralateral to the more
     affected putamen, and their difference as a formal contrast; with and
     without tremor. Model of the project: outcome ~ z(kinetic parameter)*years
     + age + sex + duration at baseline, random intercept and slope, REML,
     strict convergence criterion.
  2. Positive control: is the result of the article reproduced in these data,
     with these binding ratios? Lateralised score on SBR x time for the less
     affected putamen, for the more affected one and for the bilateral ratio,
     next to the published coefficients.
  3. Descriptives: participants and visits usable under each rule, by year;
     observations per participant; how the missingness of the OFF/untreated
     score depends on treatment; agreement between imaging side, side of onset
     and clinically more affected side.

Estimation rule (the same as in 88 and 110): the four optimisers of
_fit.OTTIMIZZATORI start from the statsmodels initial values; among those that
converge the highest log-likelihood is kept; from there all of them restart
until none gains more than 1e-10; the final point must have Newton decrement
< 1e-4, negative definite profiled Hessian, no variance on the boundary. A
model that does not satisfy the criterion is reported as NOT accepted and is
left out of the correction families and of the summary sentences.

Correction. Uncorrected p values and Benjamini-Hochberg WITHIN COHORT. Families:
"111-lati" (kinetic parameters x two sides x with/without tremor, primary rule),
"111-contrasti" (kinetic parameters x with/without tremor), and for each
sensitivity analysis the family of the same form. These tests are ADDITIONAL to
the published family and were specified in review: the "enlarged family"
(published + 111-lati) is given only to show what would change.

Identities with the pipeline, verified BEFORE fitting the new models. (a) The
kinetic parameter x time models on updrs3_score, refitted here, must coincide
with 88_diagnostics_63.csv of the working directory. (b) The published family
(the one with the caudate) is in run_caudate, next to the working directory:
the same anchor models and the p values of the common outcomes must coincide
with its 88, and the bilateral caudate, which 88 of the working directory does
not have, is refitted here (block "still_caudate": it is not a motor score and
does not enter any family) and must coincide with its own for sample size,
beta, SE and p. If a value does not coincide the script stops. If the comparison
cannot be run (run_caudate absent, or from other cohorts) it is written as NOT
verified and the enlarged family is not computed.

Population in each block of the log and in each row of the CSVs: sporadic or
not, and the assay positivity of each cohort (status and type of the sample),
read from the cohort file (24 hours) and from the raw assay file (150 hours),
not hard-coded.

What the script does NOT do is listed in the last section of the log.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 -B scripts/111_lateralised_motor.py
      (N_PROCESSES=n, default 4; the results do not depend on it)
Output: outputs/111_lateralised_motor.txt, 111_rules.csv,
        111_availability.csv, 111_agreement_sides.csv,
        111_scores_by_period.csv, 111_models_kinetics.csv,
        111_check_positive.csv
"""

import os

# A single thread per process, set before importing numpy: two executions
# give the same bytes and the parallel processes do not compete for the cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import hashlib                                                   # noqa: E402
import multiprocessing as mp                                     # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
import warnings                                                  # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS                                   # noqa: E402
from _paths import DATA                                       # noqa: E402

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
UPDRS3 = os.path.join(DATA, "MDS-UPDRS_Part_III_24Apr2026.csv")
DATRAW = os.path.join(DATA, "DAT.csv")
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
CACHE = os.path.join(OUT, "cache", "curated.parquet")
NAME_CURATED = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"
SHEET_DICTIONARY = "Data dictionary"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', min(4, os.cpu_count() or 1))))

# ----------------------------------------------------------- thresholds -----
TOL_NEWTON = 1e-4       # Newton decrement, in log-likelihood units
TOL_SINGULAR = 1e-8
TOL_CORR = 0.999
TOL_GAIN = 1e-10
MAX_ROUNDS = 10
TOL_ID_BETA = 1e-6      # identity with the pipeline: coefficient and standard error
TOL_ID_P = 1e-5         # identity with the pipeline: p
ALPHA = 0.05
N_MIN_OBSV, N_MIN_MULTI = 120, 40      # thresholds of 72/84/88 for fitting a model
THRESHOLD_PRORATA = 0.80   # minimum proportion of rated items for pro-rating (chosen here)
AI_THRESHOLD = 0.05        # alternative asymmetry threshold of the article
WINDOW_MONTHS = 3.0     # article: clinical examination within +/-3 months of the scan
MIN_DATE = 2            # article: at least two examination dates per participant
MIN_DISCORDANT = 20       # minimum discordant pairs to verify the orientation of an item
HOURS_OFF = 6.0           # purely descriptive threshold of the hours since the last dose
Z95 = 1.959963984540054
ROUNDING = 0.005           # half of the last digit of the published coefficients (two decimals)

# Figures published by Dzialas et al., Ann Neurol 2025;98:120-135,
# doi 10.1002/ana.27223. Literals from an external source, reported only to
# set them beside the values computed here (Table 4, text of the Results,
# Table 2 year 0, Methods).
DZ = {
    "n": 719, "date": 1981,
    'less_st': dict(beta=-0.07, p=0.002, ci=(-0.11, -0.02), princ=-3.60),
    'more_st': dict(beta=-0.00, p=0.93, ci=(-0.06, 0.05), princ=-4.87),
    "bil_st": dict(beta=-0.05, p=0.17, ci=(-0.12, 0.02), princ=-5.74),
    'less_ct': dict(beta=-0.08, p=0.003, ci=None),
    'no_aligned_AI0': 0.24,
    # Table 2, year 0: mean (SD)
    "t2": {("ct", "tot"): (22.0, 10.0), ("ct", 'more'): (17.1, 7.2),
           ("ct", 'less'): (11.6, 7.0), ("st", "tot"): (17.2, 8.7),
           ("st", 'more'): (13.2, 6.1), ("st", 'less'): (9.1, 6.1)},
}

KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}
# logarithm column already present in the cohort file, by kinetic parameter (if any)
LOG24 = {"saa_Fmax": "lg_Fmax", "saa_TTT": "lg_TTT", "saa_AUC": "lg_AUC",
         "saa_SLOPEMax": "lg_SLOPE"}
LOG150 = {"k_Fmax": "lg_Fmax", "k_TTT": "lg_TTT", "k_AUC": "lg_AUC", "k_T50": "lg_T50",
          "k_SLOPE": "lg_SLOPE"}
F24 = ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax"]   # first columns of F_primary (02)
TOL_ID_KINET = 1e-9       # identity of the kinetic parameters with the independent copies
COV = ["age_bl", "SEX", "dur_bl"]
COV_ART = ["moca", "ledd_g", "age_bl", "SEX", "handed", "dur_bl", "EDUCYRS"]
F_COV = {"P": "age_bl + SEX + dur_bl",
         "A": "moca + ledd_g + age_bl + SEX + C(handed) + dur_bl + EDUCYRS"}
COV_DESCR = {"A": 'covariates of the article minus the antidepressants (time-varying MoCA and LEDD; age, sex, handedness, education, duration)',
             "P": 'covariates of the project (age, sex, duration at baseline)'}

# Items of part III. D = right (destra), S = left (sinistra).
ITEMS_AR_D = ["NP3RIGRU", "NP3RIGRL", "NP3FTAPR", "NP3HMOVR", "NP3PRSPR", "NP3TTAPR",
             "NP3LGAGR"]
ITEMS_AR_S = ["NP3RIGLU", "NP3RIGLL", "NP3FTAPL", "NP3HMOVL", "NP3PRSPL", "NP3TTAPL",
             "NP3LGAGL"]
ITEMS_TR_D = ["NP3PTRMR", "NP3KTRMR", "NP3RTARU", "NP3RTARL"]
ITEMS_TR_S = ["NP3PTRML", "NP3KTRML", "NP3RTALU", "NP3RTALL"]
ITEMS_AX = ["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RISNG", "NP3GAIT", "NP3FRZGT",
           "NP3PSTBL", "NP3POSTR", "NP3BRADY"]
ITEMS_AX_TR = ["NP3RTALJ", "NP3RTCON"]
EVERY = ITEMS_AR_D + ITEMS_AR_S + ITEMS_TR_D + ITEMS_TR_S + ITEMS_AX + ITEMS_AX_TR
PAIRS = list(zip(ITEMS_AR_D + ITEMS_TR_D, ITEMS_AR_S + ITEMS_TR_S))
# blocks (right, left, non-lateralised) by tremor variant
GROUPS = {"ct": (ITEMS_AR_D + ITEMS_TR_D, ITEMS_AR_S + ITEMS_TR_S, ITEMS_AX + ITEMS_AX_TR),
          "st": (ITEMS_AR_D, ITEMS_AR_S, ITEMS_AX)}
TREM = {"ct": 'with tremor', "st": 'without tremor'}
SIDES = {'less': 'side contralateral to the LESS affected putamen',
        'more': 'side contralateral to the MORE affected putamen',
        "diff": 'difference (less affected - more affected)'}
DEF_SIDE = {"U": 'putaminal uptake at baseline, fixed side (primary)',
            "D": 'side of onset of symptoms (DOMSIDE)',
            "V": 'uptake at each examination date, asymmetry threshold 0 (article)',
            "V5": 'uptake at every examination date, asymmetry threshold 5% (article)'}

rep = []
controls = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def control(name, outcome, detail):
    """Register of the checks: 'superato' or 'NON verificato' (never silent)."""
    assert outcome in ('passed', 'NOT verified')
    controls.append((name, outcome, detail))
    log(f'  [check {outcome}] {name}: {detail}')


# ------------------------------------------------------------- statistics ----
def wilson(k, n, z=Z95):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def share(k, n):
    return f"{k}/{n} = {100 * k / n:.1f}%" if n else f"{k}/{n}"


def bh(p):
    """Benjamini-Hochberg step-up, corrected p values in input order."""
    p = np.asarray(p, float)
    n = p.size
    if n == 0:
        return p
    ordering = np.argsort(p, kind="mergesort")
    q = p[ordering] * n / np.arange(1, n + 1)
    q = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1.0)
    out = np.empty(n)
    out[ordering] = q
    return out


def fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n.d."
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def position(lo, hi, v, halfwidth=ROUNDING):
    """Where a published value, known to two decimals, falls relative to [lo, hi]."""
    if lo <= v - halfwidth and v + halfwidth <= hi:
        return 'within'
    if v + halfwidth < lo or v - halfwidth > hi:
        return 'outside'
    return 'margin'


SENTENCE_POS = {'within': 'falls within the interval estimated here',
             'outside': 'falls outside the interval estimated here',
             'margin': 'is known to two decimals and falls on the edge of the interval estimated here: it cannot be said whether it is inside or outside'}


def one_row(x):
    return " ".join(str(x).split()) if pd.notna(x) else ""


def code(x):
    """Dictionary code as text: 1 and 1.0 are the same code."""
    if isinstance(x, (int, float, np.integer, np.floating)) and float(x) == int(x):
        return str(int(x))
    return one_row(x)


# ------------------------------------------------ mixed models (rule of 88) ----
def pack(r):
    md = r.model
    return np.asarray(
        r.params_object.get_packed(use_sqrt=md.use_sqrt, has_fe=False), dtype=float)


def _ll(md, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(md.loglike(p))


def _score(md, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(md.score(p), dtype=float)


def hess_profiled(md, p):
    """Central-difference Hessian of the profiled log-likelihood,
    in the parameter space of the optimiser (identical to 84, 88 and 110)."""
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


def estimate_mm(formula, data, method, it, start=None):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kw = {} if start is None else {"start_params": start}
            return smf.mixedlm(formula, data, groups=data["PATNO"],
                               re_formula="~years").fit(method=method,
                                                        maxiter=it, **kw)
        except Exception:
            return None


def procedure(formula, data):
    """Cascade from the initial values, choice by likelihood among those that converge,
    restart from the chosen point until no optimiser gains any more."""
    attempts, md_ref = {}, None
    for method, it in OPTIMISERS:
        r = estimate_mm(formula, data, method, it)
        if r is None:
            continue
        if md_ref is None:
            md_ref = r.model
        attempts[method] = (r, _ll(md_ref, pack(r)), bool(r.converged))
    if md_ref is None:
        return None
    conv = [(m, v[1]) for m, v in attempts.items() if v[2] and np.isfinite(v[1])]
    if conv:
        chosen = max(conv, key=lambda x: x[1])[0]
    else:
        fin = [(m, v[1]) for m, v in attempts.items() if np.isfinite(v[1])]
        if not fin:
            return None
        chosen = max(fin, key=lambda x: x[1])[0]
    r0, ll0, _ = attempts[chosen]
    best = (r0, '(none)', ll0, pack(r0))
    rounds = 0
    for cycle in range(MAX_ROUNDS):
        improved = False
        for method, it in OPTIMISERS:
            r = estimate_mm(formula, data, method, it, start=best[3])
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
    return {"r": best[0], 'opt_chosen': chosen, "n_conv": len(conv),
            'opt_reopt': best[1], 'rounds': rounds, 'll_choice': ll0,
            'll_final': best[2]}


def fitmodel(t):
    """Fits a model and returns its estimate and diagnostics (one row)."""
    row = {k: v for k, v in t.items() if k not in ('data', 'term')}
    d = t['data']
    g = d.groupby("PATNO").size()
    row.update(n_subj=int(len(g)), n_obsv=int(len(d)),
                n_multivisit=int((g >= 2).sum()), n_one_single=int((g == 1).sum()),
                obsv_by_subj_median=float(g.median()),
                obsv_by_subj_q1=float(g.quantile(0.25)),
                obsv_by_subj_q3=float(g.quantile(0.75)),
                obsv_by_subj_max=int(g.max()),
                years_max=float(d.years.max()))
    res = procedure(t["formula"], d)
    if res is None:
        row.update(estimated=False, accepted=False)
        return row
    r = res["r"]
    a, b = t['term'].split(":")
    name = f"{a}:{b}" if f"{a}:{b}" in r.params.index else f"{b}:{a}"
    md = r.model
    p = pack(r)
    g_ = _score(md, p)
    H, ll = hess_profiled(md, p)
    eig = np.linalg.eigvalsh(H)
    try:
        newton = float(0.5 * g_ @ np.linalg.solve(H, -g_))
    except np.linalg.LinAlgError:
        newton = np.nan
    cov_re = np.asarray(r.cov_re, dtype=float)
    scale = float(r.scale)
    v_int, v_pen = float(cov_re[0, 0]), float(cov_re[1, 1])
    corr = (float(cov_re[0, 1]) / np.sqrt(v_int * v_pen)
            if v_int > 0 and v_pen > 0 else np.nan)
    singular = bool(v_int < TOL_SINGULAR * scale or v_pen < TOL_SINGULAR * scale
                     or (np.isfinite(corr) and abs(corr) >= TOL_CORR))
    edge = bool(v_int == 0.0 or v_pen == 0.0)
    stationary = bool(np.isfinite(newton) and newton < TOL_NEWTON)
    hess_neg = bool(eig.max() < 0)
    be, se = float(r.params[name]), float(r.bse[name])
    row.update(
        estimated=True, beta=be, se=se, ci_inf=be - Z95 * se, ci_upp=be + Z95 * se,
        p=float(r.pvalues[name]), slope_mean=float(r.params["years"]),
        effect_main=float(r.params[a]), p_main=float(r.pvalues[a]),
        dev_std_residual=float(np.sqrt(scale)),
        opt_chosen=res['opt_chosen'], n_optim_convergent=res["n_conv"],
        opt_reoptimisation=res['opt_reopt'], rounds=res['rounds'],
        loglike=ll, gain_reoptimisation=res['ll_final'] - res['ll_choice'],
        grad_inf=float(np.max(np.abs(g_))), newton=newton, stationary=stationary,
        hess_def_neg=hess_neg, singular=singular, edge=edge,
        accepted=bool(res["n_conv"] >= 1 and stationary and hess_neg
                       and not singular and not edge))
    return row


# ------------------------------------------------------------------ data -----
def load_cohort(name):
    d = pd.read_csv(os.path.join(OUT, name))
    for col in ("complete_primary", "complete"):
        if col in d.columns:
            d = d[d[col]]
    d = d.reset_index(drop=True)
    d["PATNO"] = d["PATNO"].astype("int64")
    assert d.PATNO.is_unique, f'repeated subjects in {name}'
    return d


def load_dictionary():
    """Sheet «Data dictionary» of the curated data cut, if reachable (as in 108)."""
    for p in (os.path.join(BASE, NAME_CURATED), os.path.join(DATA, NAME_CURATED)):
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_excel(p, sheet_name=SHEET_DICTIONARY)
        except ValueError:
            continue
        if not {"Variable", "Description", "Code", "Decode",
                "Derivation Notes"} <= set(d.columns):
            continue
        d = d.copy()
        d['variable'] = d["Variable"].ffill()
        return d, p
    return None, None


def components(u):
    """Sums by block, under the complete rule (C_) and pro-rating (P_).

    C_: defined only if ALL the items of part III are rated 0-4.
    P_: sum of the rated items of the block x (items of the block / rated items),
        if the proportion of rated items of the block is at least THRESHOLD_PRORATA.
    """
    v = u[EVERY]
    ok = v.notna()
    complete = ok.all(axis=1)
    out = {'n_evaluated': ok.sum(axis=1), 'complete': complete}
    for trem, (d, s, ax) in GROUPS.items():
        for name, cols in (("D", d), ("S", s), ("AX", ax)):
            n_ok = ok[cols].sum(axis=1)
            tot = v[cols].sum(axis=1, min_count=1)
            out[f'C_{name}_{trem}'] = tot.where(complete)
            out[f'P_{name}_{trem}'] = (tot * len(cols) / n_ok.where(n_ok > 0)).where(
                n_ok / len(cols) >= THRESHOLD_PRORATA)
    return pd.DataFrame(out, index=u.index)


def outcomes_side(T, side_less, pref, rule="C"):
    """Scores by side. `side_less` = LESS affected side of the body, that is,
    contralateral to the less affected putamen and ipsilateral to the more affected one:
    'S', 'D', '=' (symmetric: mean of the two sides, rule R6) or missing."""
    side = np.asarray(side_less, dtype=object)
    for trem in GROUPS:
        D = T[f'{rule}_D_{trem}'].to_numpy(float)
        S = T[f'{rule}_S_{trem}'].to_numpy(float)
        AX = T[f'{rule}_AX_{trem}'].to_numpy(float)
        mean = (S + D) / 2
        less = np.where(side == "S", S, np.where(side == "D", D,
                        np.where(side == "=", mean, np.nan)))
        more = np.where(side == "S", D, np.where(side == "D", S,
                       np.where(side == "=", mean, np.nan)))
        T[f'{pref}_less_{trem}'] = less + AX            # score of the article
        T[f'{pref}_more_{trem}'] = more + AX
        T[f"{pref}_diff_{trem}"] = less - more            # the axial items cancel out
        T[f'{pref}_limbsless_{trem}'] = less             # limbs only
        T[f"{pref}_artipiu_{trem}"] = more
    return T


def job(L, outcome, rhs, term, extra, skipped, **meta):
    col = ["PATNO", "years", outcome] + sorted(set(extra))
    m = L[col].dropna().sort_values(["PATNO", "years"], kind="mergesort").reset_index(
        drop=True)
    multi = int((m.groupby("PATNO").size() >= 2).sum()) if len(m) else 0
    if len(m) < N_MIN_OBSV or multi < N_MIN_MULTI:
        skipped.append(dict(outcome=outcome, n_obsv=len(m), n_multivisit=multi, **meta))
        return None
    return dict(data=m, formula=f'{outcome} ~ {rhs}', term=term, outcome=outcome, **meta)


# ================================================================== body =====
def main():
    t_start = time.time()
    skipped = []

    # ----------------------------------------------- 0. working directory ----
    title('0. WORKING DIRECTORY, POPULATION, FILES')
    log('ANALYSIS ADDED IN REVIEW (not planned in the submitted manuscript).')
    cur = pd.read_parquet(CACHE)
    cur["PATNO"] = cur["PATNO"].astype("int64")
    assert not cur.duplicated(["PATNO", "EVENT_ID"]).any(), 'repeated visits in the curated data cut'
    sub = cur[cur.EVENT_ID == "BL"].set_index("PATNO").subgroup
    sporadic = set(sub[sub == "Sporadic PD"].index)
    c24 = load_cohort("analysis_complete_cases.csv")
    c150 = load_cohort("08_dataset_150h.csv")
    p24, p150 = set(c24.PATNO), set(c150.PATNO)
    assert (p24 | p150) <= set(sub.index), 'subjects of the cohorts missing from the baseline curated data cut'
    only24, only150 = not (p24 - sporadic), not (p150 - sporadic)
    assert only24 == only150, ("the two cohorts of the working directory are not of the same kind: "
                               "one is sporadic only, the other is not")
    ROOT = 'sporadic only' if only24 else 'full cohort'
    assert ONLY == only24, (f"ONLY_SPORADIC={('1' if ONLY else '0')} but the working directory is '{ROOT}': variable and working directory do not agree")
    POP = ('sporadic PD SAA-positive at baseline' if only24
           else 'PD SAA-positive at baseline, sporadic and carriers')
    COHORTS = {"24h": (c24, KIN24), "150h": (c150, KIN150)}
    union = p24 | p150
    POPULATIONS = {"24h": p24, "150h": p150, 'union': union}
    N_POP = {p: len(s) for p, s in POPULATIONS.items()}

    # Assay positivity of each cohort (positives of one type, or positives of
    # any type): read from the files, not hard-coded. 24 hours: status and type of the
    # sample in the cohort file. 150 hours: positive baseline samples of Project 155
    # in the raw file (the same from which the kinetic parameters are recomputed below).
    def label_saa(states, kinds, without_kind):
        s = "/".join(sorted(states)) if states else 'missing state'
        t = "/".join(sorted(kinds))
        return (f"SAA {s}, " + (f'type {t}' if t else 'without assigned type')
                + (' or without type' if t and without_kind else ""))

    if {"SAA_Status", "SAA_Type"} <= set(c24.columns):
        assert (c24.SAA_Status == "Positive").all(), \
            "24 h cohort: subjects whose sample is not Positive in the cohort file"
        pos24 = label_saa(set(c24.SAA_Status), set(c24.SAA_Type.dropna().astype(str)),
                              bool(c24.SAA_Type.isna().any()))
        pos24_verified = True
    else:
        pos24 = 'status and type of the assay not indicated in the cohort file'
        pos24_verified = False
    sa150 = None
    if os.path.exists(SAA):
        sa = pd.read_csv(SAA, low_memory=False)
        sa["PATNO"] = sa["PATNO"].astype("int64")
        sa150 = sa[(sa.PROJECTID == 155) & (sa.COHORT == "PD") & (sa.CLINICAL_EVENT == "BL")
                   & (sa.SAA_Status == "Positive") & sa.PATNO.isin(p150)].copy()
        assert set(sa150.PATNO) == p150, \
            (f'150 h cohort: {len(p150 - set(sa150.PATNO))} subjects without a Positive baseline '
             f'sample of the 150 h protocol in the raw assay file')
        pos150 = label_saa({"Positive"}, set(sa150.SAA_Type.dropna().astype(str)),
                               bool(sa150.SAA_Type.isna().any()))
    else:
        pos150 = 'positivity not re-verified (raw assay file absent)'
    POS = {"24h": f'24 h assay: {pos24}', "150h": f'assay at 150 h: {pos150}'}
    POS['union'] = f"{POS['24h']}; {POS['150h']}"
    LBLS = {"24h": f'24 h (n = {len(c24)}; {pos24})',
           "150h": f'150 h (n = {len(c150)}; {pos150})'}
    LBLS_POP = dict(LBLS, union=f"union of the two cohorts (n = {len(union)} distinct subjects; {POS['union']})")

    def having_cohort(df):
        """Assay positivity and sample size of the cohort in each row of a CSV."""
        i = list(df.columns).index('population') + 1
        df.insert(i, 'positivity_saa', df['cohort'].map(POS))
        df.insert(i + 2, 'n_cohort', df['cohort'].map(N_POP))
        assert df.positivity_saa.notna().all() and df.n_cohort.notna().all(), \
            "row of a CSV with a cohort without a positivity or sample size label"
        return df

    log(f"working directory: {os.path.basename(os.path.normpath(BASE))} ({ROOT}); ONLY_SPORADIC = {('1' if ONLY else '0')}")
    log(f'population: {POP}')
    log(f"analysis cohorts: {LBLS['24h']}, {LBLS['150h']}; shared {len(p24 & p150)}, distinct {len(union)}")
    for name in ("analysis_complete_cases.csv", "08_dataset_150h.csv"):
        log(f'  {name}: sha256 {fingerprint(os.path.join(OUT, name))[:16]}...')
    log(f'  cached curated data cut: {len(cur)} rows (visits), {cur.PATNO.nunique()} subjects')
    log(f'  {os.path.basename(UPDRS3)}: sha256 {fingerprint(UPDRS3)[:16]}...')
    log(f"assay positivity (derived from the files): {POS['union']}")
    if pos24_verified:
        control('assay positivity, 24 h cohort', 'passed',
                  f'in the cohort file all the {len(c24)} subjects have a sample {pos24}')
    else:
        control('assay positivity, 24 h cohort', 'NOT verified',
                  'the cohort file does not have the columns SAA_Status and SAA_Type')
    if sa150 is not None:
        control('assay positivity, 150 h cohort', 'passed',
                  f'all the {len(c150)} subjects have in the raw file at least one baseline sample of Project 155 with outcome {pos150} ({len(sa150)} samples)')
    else:
        control('assay positivity, 150 h cohort', 'NOT verified',
                  f'{os.path.basename(SAA)} is not in the data directory')
    # identity of the kinetic parameters with the copies the pipeline has already made
    for tag, (c, kin), logcol in (("24h", COHORTS["24h"], LOG24), ("150h", COHORTS["150h"], LOG150)):
        assert np.isfinite(c[list(kin)].to_numpy(float)).all() and (
            c[list(kin)].to_numpy(float) > 0).all(), f'non-positive or missing kinetic parameters ({tag})'
        ok_, no_ = [], []
        for col, lg in logcol.items():
            if lg not in c.columns:
                no_.append(kin[col])
                continue
            sc_ = float(np.abs(np.log(c[col]) - c[lg]).max())
            assert sc_ < TOL_ID_KINET, \
                (f"cohort {tag}: {col} is not consistent with {lg} of the same file (maximum "
                 f"discrepancy {sc_:.2e}): a kinetic parameter value has been altered")
            ok_.append(kin[col])
        if ok_:
            control(f'kinetic parameters and their logarithms in the cohort file, {tag}', 'passed',
                      f"{', '.join(ok_)}: log(kinetic parameter) identical to the column already computed by the pipeline (tolerance {TOL_ID_KINET:g})")
        if no_:
            control(f'kinetic parameters and their logarithms in the cohort file, {tag}', 'NOT verified',
                      f"{', '.join(no_)}: the file does not have the logarithm column: this check does not cover it")
    fnpz = os.path.join(OUT, "features.npz")
    if os.path.exists(fnpz):
        npz = np.load(fnpz)
        assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy()), \
            'features.npz and analysis_complete_cases.csv do not have the same subjects in the same order'
        Mk = np.log(c24[F24].to_numpy(float))
        sc_ = float(np.abs((Mk - Mk.mean(0)) / Mk.std(0) - npz["F_primary"][:, :len(F24)]).max())
        assert sc_ < TOL_ID_KINET, \
            (f'the kinetic parameters of the 24 h cohort are not those of features.npz (maximum '
             f'discrepancy {sc_:.2e} on the standardised values)')
        control('kinetic parameters of the 24 h cohort = features.npz of 02', 'passed',
                  f'same {len(c24)} subjects in the same order; maximum difference {sc_:.1e} on the standardised logarithms')
    else:
        control('kinetic parameters of the 24 h cohort = features.npz of 02', 'NOT verified',
                  'features.npz is not in the working directory')
    # 150-hour cohort: the kinetic parameters are recomputed from the raw file with the
    # rule of 08 (median of the three replicates of the sample, mean across the positive
    # baseline samples)
    if sa150 is not None:
        q = sa150.copy()
        for col in KIN150:
            rip = [f"{col[2:]}_150h_Rep{i}" for i in (1, 2, 3)]
            q[col] = q[rip].apply(pd.to_numeric, errors="coerce").median(axis=1)
        g = q.groupby("PATNO")[list(KIN150)].mean().reindex(c150.PATNO.to_numpy())
        assert g.notna().all().all(), \
            'subjects of the 150 h cohort without a positive baseline sample in the raw file'
        sc_ = float(np.abs(g.to_numpy() / c150[list(KIN150)].to_numpy(float) - 1).max())
        assert sc_ < TOL_ID_KINET, \
            (f'the kinetic parameters of the 150 h cohort are not those recomputed from the raw '
             f'assay file (maximum relative discrepancy {sc_:.2e})')
        control('kinetic parameters of the 150 h cohort = raw file of the assay', 'passed',
                  f"{', '.join(KIN150.values())} recomputed for the {len(c150)} subjects (median of the replicates, mean across the positive baseline samples): maximum relative difference {sc_:.1e}")
    else:
        control('kinetic parameters of the 150 h cohort = raw file of the assay',
                  'NOT verified', f'{os.path.basename(SAA)} is not in the data directory')

    # ------------------------------------------- 1. dictionary and state -----
    title('1. MEDICATION STATE: WHAT THE DICTIONARIES SAY AND WHAT THE DATA SAY')
    dic, p_dic = load_dictionary()
    codes_dom, codes_manual = None, None
    if dic is None:
        control('dictionary of the curated data cut', 'NOT verified',
                  f'{NAME_CURATED} not reachable: no definition cited; the coding of DOMSIDE remains verified only by the concordance of the sides')
    else:
        log(f'Dictionary: sheet «{SHEET_DICTIONARY}» of {os.path.basename(p_dic)}. Verbatim quotations:')
        for var in ("updrs3_score", "PDTRTMNT"):
            r = dic[dic.variable == var]
            if not len(r):
                log(f'  {var}: entry missing from the dictionary')
                continue
            note = one_row(r["Derivation Notes"].iloc[0]).split("Sum (")[0].strip()
            log(f"  {var}: «{one_row(r['Description'].iloc[0])}»"
                + (f' - derivation note: «{note}»' if note else ""))
        r = dic[(dic.variable == "PDTRTMNT") & dic.Code.notna()]
        log('  PDTRTMNT, codes: ' + "; ".join(
            f'{code(a)} = {one_row(b)}' for a, b in zip(r.Code, r.Decode)))
        r = dic[(dic.variable == "DOMSIDE") & dic.Code.notna()]
        codes_dom = {code(a): one_row(b) for a, b in zip(r.Code, r.Decode)}
        log(f"  DOMSIDE («{one_row(dic[dic.variable == 'DOMSIDE'].Description.iloc[0])}»), codes: " + "; ".join(f"{a} = {b}" for a, b in codes_dom.items()))
        assert codes_dom.get("1") == "Left" and codes_dom.get("2") == "Right", \
            f'the coding of DOMSIDE in the dictionary is not 1 = Left, 2 = Right: {codes_dom}'
        control('coding of DOMSIDE', 'passed',
                  'in the dictionary 1 = Left, 2 = Right, as assumed by the code')
        r = dic[(dic.variable == "handed") & dic.Code.notna()]
        codes_manual = {code(a): one_row(b) for a, b in zip(r.Code, r.Decode)}
        log('  handed, codes: ' + "; ".join(f"{a} = {b}" for a, b in codes_manual.items()))
        log("  PDSTATE is not a variable of the curated data cut and the PPMI dictionary for the "
            "raw files is not among the data: its meaning is read from the values.")

    u = pd.read_csv(UPDRS3, low_memory=False)
    u["PATNO"] = u["PATNO"].astype("int64")
    u['row'] = np.arange(len(u))
    items_file = {c for c in u.columns if c.startswith("NP3") and c != "NP3TOT"}
    assert items_file == set(EVERY) and len(set(EVERY)) == len(EVERY), \
        (f'the items of part III in the file are not those assigned by the script: only in the file {sorted(items_file - set(EVERY))}, only in the script {sorted(set(EVERY) - items_file)}')
    for d_, s_ in PAIRS:
        diff = [(a, b) for a, b in zip(d_, s_) if a != b]
        assert len(d_) == len(s_) and diff == [("R", "L")], \
            f"the pair {d_}/{s_} is not the same item on the right and on the left"
    states = sorted(str(x) for x in u.PDSTATE.dropna().unique())
    assert set(states) <= {"ON", "OFF"}, f'unexpected values of PDSTATE: {states}'
    raw = u[EVERY].apply(pd.to_numeric, errors="coerce")
    outside = sorted(set(np.unique(raw.to_numpy()[np.isfinite(raw.to_numpy())]))
                   - {0.0, 1.0, 2.0, 3.0, 4.0})
    assert outside in ([], [101.0]), f'unexpected item values (neither 0-4 nor 101): {outside}'
    u[EVERY] = raw.where((raw >= 0) & (raw <= 4))
    u = u[u.PATNO.isin(union)].copy()
    u["off"] = u.PDSTATE == "OFF"
    u['no_treated'] = u.PDSTATE.isna() & (u.PDTRTMNT == 0)
    u['admitted'] = u.off | u.no_treated
    u = pd.concat([u, components(u)], axis=1)
    log()
    log(f"Raw file {os.path.basename(UPDRS3)}: {len(u)} examinations of the {len(union)} subjects of the two cohorts (unit: rows = examinations). Values of PDSTATE present: {', '.join(states)}, or empty. Item values outside 0-4: {(', '.join((f'{x:g}' for x in outside)) if outside else 'noone')}"
        + (" (in PPMI the code of the item that cannot be rated; its meaning cannot be verified on "
           "the available files: the item is treated as not rated)."
           if outside else "."))
    tab = u.groupby([u.PDSTATE.fillna('(empty)'), u.PDTRTMNT.map(
        {0.0: 'not on treatment', 1.0: 'on treatment'}).fillna('(empty)')]).size()
    for (a, b), n in tab.items():
        log(f'    PDSTATE {a:<8} PDTRTMNT {b:<20} {n:>6} examinations')
    n_off, n_nt = int(u.off.sum()), int(u.no_treated.sum())
    log(f"Rule R3: the examination is admitted if declared OFF ({n_off}) or, with no declared state, "
        f"if it belongs to a subject not on treatment ({n_nt}). Examinations with no state and no "
        f"indication of treatment ({int((u.PDSTATE.isna() & u.PDTRTMNT.isna()).sum())}) and those "
        f"declared ON ({int((u.PDSTATE == 'ON').sum())}) are excluded.")
    off_hours = u[u.off & u.complete & u.HRPOSTMED.notna()]
    log(f'  Descriptive only: among the complete OFF examinations with known hours since the last dose ({len(off_hours)}), median {off_hours.HRPOSTMED.median():.1f} hours; below {HOURS_OFF:g} hours {share(int((off_hours.HRPOSTMED < HOURS_OFF).sum()), len(off_hours))}. The article gives no threshold in hours and none is applied here.')

    # one admitted examination per participant x visit: the most complete first,
    # then the first row
    adm = u[u.admitted].sort_values(["PATNO", "EVENT_ID", 'n_evaluated', 'row'],
                                   ascending=[True, True, False, True], kind="mergesort")
    n_doubles = int(adm.duplicated(["PATNO", "EVENT_ID"]).sum())
    E = adm.drop_duplicates(["PATNO", "EVENT_ID"], keep="first").copy()
    state_v = u.groupby(["PATNO", "EVENT_ID"]).agg(
        n_exams=('row', "size"), some_on=("PDSTATE", lambda s: bool((s == "ON").any())),
    ).reset_index()
    for trem in GROUPS:
        E[f"tot_{trem}"] = E[f"C_D_{trem}"] + E[f"C_S_{trem}"] + E[f"C_AX_{trem}"]
    log(f'  Subject x visit with more than one admitted examination: {n_doubles} (the most complete one is taken, then the first row of the file).')

    # visits of the curated data cut: the grid of the project
    cols_v = ["PATNO", "EVENT_ID", "YEAR", "updrs3_score", "PDTRTMNT", "LEDD", "moca",
              "age_at_visit", "age_datscan", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
              "MIA_PUTAMEN_BILAT", "MIA_CAUDATE_BILAT"]
    V = cur[cur.PATNO.isin(union)][cols_v].rename(columns={"PDTRTMNT": "trt_cur"}).copy()
    V["years"] = pd.to_numeric(V.YEAR, errors="coerce")
    V = V[V.years.notna()]
    c_e = (["PATNO", "EVENT_ID", "PDSTATE", "PDTRTMNT", 'n_evaluated', 'complete']
           + [c for c in E.columns if c[:2] in ("C_", "P_") or c.startswith("tot_")])
    V = V.merge(E[c_e], on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    V = V.merge(state_v, on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
    V['ha_admitted'] = V.n_evaluated.notna()
    V['useful'] = V.complete.fillna(False).astype(bool)
    # identity with the OFF/untreated score of the curated data cut
    bothsides = V.useful & V.updrs3_score.notna()
    different = int((V.loc[bothsides, "tot_ct"] != V.loc[bothsides, "updrs3_score"]).sum())
    only_mine = int((V.useful & V.updrs3_score.isna()).sum())
    incomplete_ma_cur = int((V.ha_admitted & ~V.useful & V.updrs3_score.notna()).sum())
    only_cur = int((~V.ha_admitted & V.updrs3_score.notna()).sum())
    assert int(bothsides.sum()) > 0, 'no visit with a score both in the curated data cut and in the raw file'
    assert different == 0 and only_mine == 0 and incomplete_ma_cur == 0, \
        (f'the total rebuilt from the items does not match updrs3_score of the curated data cut: {different} different values out of {int(bothsides.sum())}; {only_mine} complete visits without a score in the curated data cut; {incomplete_ma_cur} incomplete visits with a score')
    control('rule R3 = OFF/untreated score of the curated data cut', 'passed',
              f'in {int(bothsides.sum())} visits out of {int(bothsides.sum())} the sum of the {len(EVERY)} items of the admitted examination is identical to updrs3_score; no complete visit is missing from the curated data cut and no incomplete one has a score there')
    tr = V[V.ha_admitted & V.trt_cur.notna() & V.PDTRTMNT.notna()]
    assert len(tr) and (tr.trt_cur == tr.PDTRTMNT).all(), \
        "PDTRTMNT of the admitted examination differs from PDTRTMNT of the curated data cut at the same visit"
    control('treatment: raw file = curated data cut', 'passed',
              f'PDTRTMNT identical in {len(tr)} visits out of {len(tr)}')
    log(f'  Visits with a score in the curated data cut but without an admitted examination in the raw file at that visit: {only_cur}. They have no items: they stay out of the lateralised scores.')
    n_inc = int((V.ha_admitted & ~V.useful & (V.n_evaluated > 0)).sum())
    blank = V.ha_admitted & (V.n_evaluated == 0)
    n_blank = int(blank.sum())
    log(f"Rule R4: admitted examinations with some item not rated {n_inc}; admitted examinations with no rated item {n_blank}, of which declared OFF {int((V.loc[blank, 'PDSTATE'] == 'OFF').sum())}. In neither case does the curated data cut give a total (verified above): the complete rule is that of the curated data cut.")

    # ------------------------------------------------ 2. side of the body ----
    title('2. SIDE: IMAGING, ONSET, CLINICAL')
    blc = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                     "duration_yrs", "DOMSIDE", "handed", "MIA_PUTAMEN_L",
                                     "MIA_PUTAMEN_R"]].rename(
        columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl",
                 "MIA_PUTAMEN_L": "L_bl", "MIA_PUTAMEN_R": "R_bl"})
    blc = blc[blc.PATNO.isin(union)].copy()
    # putamen with lower uptake at baseline = more affected; the LESS affected
    # side of the body is the one ipsilateral to it
    blc['lower_bl'] = np.where(blc.L_bl < blc.R_bl, "S",
                               np.where(blc.R_bl < blc.L_bl, "D",
                                        np.where(blc.L_bl == blc.R_bl, "=", None)))
    blc['side_less_U'] = blc.lower_bl
    blc['side_less_D'] = np.where(blc.DOMSIDE == 1, "D",
                                  np.where(blc.DOMSIDE == 2, "S", None))
    # clinical side at baseline (larger sum of the limb items)
    eb = V[(V.EVENT_ID == "BL") & V.useful][["PATNO"] + [f"C_{x}_{t}" for x in 'SD'
                                                         for t in GROUPS]]
    blc = blc.merge(eb, on="PATNO", how="left", validate="one_to_one")
    for trem in GROUPS:
        d_, s_ = blc[f"C_D_{trem}"], blc[f"C_S_{trem}"]
        blc[f"clin_{trem}"] = np.where(d_ > s_, "D", np.where(s_ > d_, "S",
                                       np.where(d_ == s_, "=", None)))
    blc['img_more'] = blc.lower_bl.map({"S": "D", "D": "S"})     # more affected side of the body
    blc['ons'] = blc.DOMSIDE.map({1: "S", 2: "D"})

    # identity of S and D with the raw DAT file
    if os.path.exists(DATRAW):
        dg = pd.read_csv(DATRAW, low_memory=False)
        dg["PATNO"] = dg["PATNO"].astype("int64")
        dg = dg[dg.PATNO.isin(union) & dg.PUTAMEN_L_REF_CWM.notna()
                & dg.PUTAMEN_R_REF_CWM.notna()]
        sc = V[V.MIA_PUTAMEN_L.notna() & V.MIA_PUTAMEN_R.notna()][
            ["PATNO", "EVENT_ID", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]]
        j = sc.merge(dg[["PATNO", "EVENT_ID", "PUTAMEN_L_REF_CWM", "PUTAMEN_R_REF_CWM"]],
                     on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one")
        selfsame = j.PUTAMEN_L_REF_CWM.notna()
        different_ev = int(((j.MIA_PUTAMEN_L != j.PUTAMEN_L_REF_CWM)
                          | (j.MIA_PUTAMEN_R != j.PUTAMEN_R_REF_CWM))[selfsame].sum())
        pairs = set(zip(dg.PATNO, dg.PUTAMEN_L_REF_CWM, dg.PUTAMEN_R_REF_CWM))
        rest = j[~selfsame]
        straight = np.array([(a, b, c) in pairs for a, b, c in
                           zip(rest.PATNO, rest.MIA_PUTAMEN_L, rest.MIA_PUTAMEN_R)], bool)
        flips = np.array([(a, c, b) in pairs for a, b, c in
                             zip(rest.PATNO, rest.MIA_PUTAMEN_L, rest.MIA_PUTAMEN_R)], bool)
        swapped = int((flips & ~straight).sum())
        assert int(selfsame.sum()) + int(straight.sum()) > 0, \
            "no scan of the curated data cut found in DAT.csv: identity check cannot be run"
        assert different_ev == 0 and swapped == 0, \
            (f'left/right putamen of the curated data cut differ from DAT.csv: {different_ev} scans at the same visit, {swapped} scans found only with the sides swapped')
        control('left and right putamen: curated = DAT.csv', 'passed',
                  f'{int(selfsame.sum())} identical scans at the same visit; {int(straight.sum())} found as a pair (left, right) among the scans of the same subject under another visit label; none with swapped sides. Unit: scans of the {len(union)} subjects')
        if int((~straight).sum()):
            control('left and right putamen: scans absent from DAT.csv',
                      'NOT verified',
                      f'{int((~straight).sum())} scans out of {len(j)} do not have in DAT.csv a row of the same subject with the same two values: their side remains verified only by the concordance with onset and clinical side')
    else:
        control('left and right putamen: curated = DAT.csv', 'NOT verified',
                  'DAT.csv is not in the data directory')

    rows_conc = []

    def record(pop, units, item, k, n, remark=""):
        lo, hi = wilson(k, n)
        rows_conc.append(dict(population=POP, cohort=pop, units=units, item=item, k=k,
                               n=n, fraction=(k / n if n else np.nan), ci95_inf=lo,
                               ci95_upp=hi, remark=remark))
        return f'{share(k, n)} [Wilson 95% {100 * lo:.1f}; {100 * hi:.1f}]'

    BASES, LONGS = {}, {}
    for tag, (c, kin) in COHORTS.items():
        base = c[["PATNO"] + list(kin)].rename(columns=kin)
        for k in kin.values():
            base[k] = np.log(base[k])
            base[k] = (base[k] - base[k].mean()) / base[k].std()
        base = base.merge(blc, on="PATNO", how="left", validate="one_to_one")
        for mine, its in (("L_bl", "MIA_PUTAMEN_L"), ("R_bl", "MIA_PUTAMEN_R")):
            assert np.array_equal(base[mine].to_numpy(), c[its].to_numpy()), \
                f'{its} at baseline: the curated data cut does not match the cohort file'
        assert base.lower_bl.notna().all(), 'baseline putaminal uptake missing'
        BASES[tag] = base
        n = len(base)
        log(f'\n[{POP}; {LBLS[tag]} cohort; unit: subjects, at baseline]')
        log(f"  putamen with lower uptake (more affected): left {int((base.lower_bl == 'S').sum())}, right {int((base.lower_bl == 'D').sum())}, tie {int((base.lower_bl == '=').sum())} (mean of the sides, R6)")
        log(f'  side of onset: left {int((base.DOMSIDE == 1).sum())}, right '
            f'{int((base.DOMSIDE == 2).sum())}, symmetric {int((base.DOMSIDE == 3).sum())}, missing '
            f'{int(base.DOMSIDE.isna().sum())}')
        pairs_sides = [('imaging = onset', 'img_more', 'ons'),
                       ('clinical (with tremor) = onset', "clin_ct", 'ons'),
                       ('clinical (with tremor) = imaging', "clin_ct", 'img_more'),
                       ('clinical (without tremor) = onset', "clin_st", 'ons'),
                       ('clinical (without tremor) = imaging', "clin_st", 'img_more')]
        fr = {}
        for name, a, b in pairs_sides:
            sel = base[a].isin(["S", "D"]) & base[b].isin(["S", "D"])
            k_, n_ = int((base.loc[sel, a] == base.loc[sel, b]).sum()), int(sel.sum())
            fr[name] = k_ / n_ if n_ else np.nan
            log(f'  more affected side, {name}: '
                + record(tag, 'subjects at baseline', name, k_, n_,
                           'ties, symmetric and missing excluded')
                + f'; not aligned {share(n_ - k_, n_)}')
        for trem in GROUPS:
            cl = base[f"clin_{trem}"]
            log(f"  clinical side {TREM[trem]}: right {int((cl == 'D').sum())}, left "
                f"{int((cl == 'S').sum())}, tie {int((cl == '=').sum())}, baseline examination not "
                f"usable {int(cl.isna().sum())}")
        sel3 = base.img_more.isin(["S", "D"]) & base.ons.isin(["S", "D"]) & base.clin_ct.isin(
            ["S", "D"])
        k3 = int(((base.img_more == base.ons) & (base.ons == base.clin_ct))[sel3].sum())
        log('  the three sides coincide: '
            + record(tag, 'subjects at baseline', 'imaging = onset = clinical (with tremor)',
                       k3, int(sel3.sum())))
        for name in ('imaging = onset', 'clinical (with tremor) = onset',
                     'clinical (with tremor) = imaging'):
            assert fr[name] > 0.5, \
                (f"cohort {tag}: '{name}' in less than half of the subjects ({100 * fr[name]:.1f}%): right and left are swapped in one of the sources")
        control(f'orientation of the sides, cohort {tag}', 'passed',
                  'imaging, onset and clinical examination indicate the same side in more than half of the subjects for each pair ('
                  + "; ".join(f'{name} {100 * fr[name]:.1f}%' for name in
                              ('imaging = onset', 'clinical (with tremor) = onset',
                               'clinical (with tremor) = imaging')) + ")")
        # long table of the visits of the cohort
        L = V[V.PATNO.isin(set(base.PATNO))].merge(
            base[["PATNO"] + list(kin.values()) + COV + ["DOMSIDE", 'side_less_U',
                                                         'side_less_D', 'lower_bl']],
            on="PATNO", how="left", validate="many_to_one")
        L = outcomes_side(L, L.side_less_U, "U", "C")
        L = outcomes_side(L, L.side_less_D, "D", "C")
        L = outcomes_side(L, L.side_less_U, "UP", "P")
        LONGS[tag] = L
        # stability of the lower-uptake side during follow-up (needed for rule V)
        f = L[(L.years > 0) & L.MIA_PUTAMEN_L.notna() & L.MIA_PUTAMEN_R.notna()]
        lower = np.where(f.MIA_PUTAMEN_L < f.MIA_PUTAMEN_R, "S",
                         np.where(f.MIA_PUTAMEN_R < f.MIA_PUTAMEN_L, "D", "="))
        valid = (lower != "=") & (f.lower_bl != "=").to_numpy()
        ks, ns = int((lower[valid] == f.lower_bl.to_numpy()[valid]).sum()), int(valid.sum())
        log("  follow-up scans in which the putamen with lower uptake is the baseline one: " + record(tag, 'scans after baseline',
                                  'side with lower uptake unchanged', ks, ns))
        assert ns > 0 and ks / ns > 0.5, \
            (f'cohort {tag}: the side with lower uptake remains the baseline one in less than half of the follow-up scans ({share(ks, ns)}): sides swapped after baseline')

    # orientation of each pair of items relative to the side of onset (baseline, union)
    eb2 = V[(V.EVENT_ID == "BL") & V.useful][["PATNO"]].merge(
        E[E.EVENT_ID == "BL"][["PATNO"] + [x for c_ in PAIRS for x in c_]], on="PATNO",
        how="left", validate="one_to_one").merge(blc[["PATNO", 'ons']], on="PATNO",
                                                 how="left", validate="one_to_one")
    eb2 = eb2[eb2.ons.isin(["S", "D"])]
    log(f"\nOrientation of the single items (unit: subjects of the union with a usable baseline "
        f"examination and lateralised onset, n = {len(eb2)}): among the discordant pairs, proportion "
        f"in which the score is higher on the side of onset.")
    for d_, s_ in PAIRS:
        side_ons = np.where(eb2.ons == "D", eb2[d_], eb2[s_])
        side_other = np.where(eb2.ons == "D", eb2[s_], eb2[d_])
        nd, kd = int((side_ons != side_other).sum()), int((side_ons > side_other).sum())
        if nd < MIN_DISCORDANT:
            control(f'orientation {d_}/{s_}', 'NOT verified',
                      f'only {nd} discordant pairs (minimum {MIN_DISCORDANT})')
            continue
        assert kd / nd > 0.5, \
            (f'the item {d_}/{s_} is higher on the side OPPOSITE to onset in {share(nd - kd, nd)} of the discordant pairs: right/left columns swapped')
        control(f'orientation {d_}/{s_}', 'passed', f'{share(kd, nd)} of the discordant pairs')

    # -------------------------------------------------- 3. rules -------------
    title('3. RULES FOR BUILDING THE SCORE')
    t2 = DZ["t2"]
    ax_ct = t2[("ct", 'more')][0] + t2[("ct", 'less')][0] - t2[("ct", "tot")][0]
    ax_st = t2[("st", 'more')][0] + t2[("st", 'less')][0] - t2[("st", "tot")][0]
    n_lat = len(GROUPS["ct"][0])
    rules = [
        ('R1 lateralised items',
         'Methods, Clinical Scores: akinetic-rigid items of the limbs and tremor items separated by side, then summed with the non-lateralised axial items; total = less affected + more affected + axial. The list of items is NOT given.',
         f"lateralised the {n_lat} items per side that the scale rates on the right and on the left ({len(ITEMS_AR_D)} akinetic-rigid, {len(ITEMS_TR_D)} of tremor); not lateralised the other {len(GROUPS['ct'][2])}; side score = items of the side + non-lateralised",
         'score of the limbs only, without non-lateralised items (sens_limbs block)'),
        ('R2 tremor',
         f'Methods, Clinical Scores: scores with and without tremor items; list NOT given. From the published means (Table 2, year 0) the non-lateralised part is worth {ax_ct:.1f} points with tremor and {ax_st:.1f} without'
         + (': non-lateralised tremor items are also removed.' if ax_st < ax_ct else
            ': the published means do NOT indicate that non-lateralised items are removed.'),
         f'without tremor = removed all the {2 * len(ITEMS_TR_D) + len(ITEMS_AX_TR)} tremor items ({len(ITEMS_TR_D)} per side and {len(ITEMS_AX_TR)} not lateralised)',
         'none: both versions are estimated'),
        ('R3 medication state',
         'Methods, Data, criterion 3: MDS-UPDRS III in the off-medication state. How the state is read from the PPMI files and how the untreated are handled is NOT stated.',
         f'examination declared OFF, or without a declared state in a subject not on treatment (definition of updrs3_score of the curated data cut; identity verified on {int(bothsides.sum())} visits)',
         'none estimated; the hours since the last dose are only described'),
        ('R4 items not assessable',
         'The article is silent.',
         f'visit usable only with all the {len(EVERY)} items rated 0-4 (rule of the curated data cut)',
         f'pro rata within block with at least {100 * THRESHOLD_PRORATA:.0f}% of the items rated (sens_prorata block; threshold chosen here)'),
        ('R5 side of the body',
         'Methods, DaT-SPECT and Clinical Scores: more/less affected hemisphere for every examination date from the asymmetry index (L - R)/(L + R), thresholds 0% and 5%; side of the body contralateral to the more affected hemisphere = more affected.',
         'outcomes of the kinetic parameters: side fixed by the baseline putaminal uptake (U), declared deviation; positive control: rule of the article at each date (V)',
         f'kinetic parameters: side of onset (D, sens_onset block); positive control: baseline side (U) and threshold of {100 * AI_THRESHOLD:.0f}% (V5)'),
        ('R6 symmetric subjects',
         'Methods, Clinical Scores: without asymmetry, mean of the two side scores.',
         'tie in uptake: mean of the two sides (difference = 0)',
         'with a symmetric or missing side of onset the subject drops out (the article is silent)'),
        ('R7 timeline',
         'Methods, Defining the Timeline: scan dates; clinical examination within +/-3 months; time in months from the first available scan.',
         f'positive control: visits with scan, |age at the visit - age at the scan| <= {WINDOW_MONTHS:g} months, months from the first usable date; kinetic parameters: visits and time (YEAR) of the project',
         'nothing'),
        ('R8 model',
         'Statistical Analysis: SBR, time, SBR x time; antidepressants, LEDD, MoCA time-varying; age, sex, handedness, education, duration; random intercept and slope; at least two dates per subject.',
         'positive control: the same model WITHOUT antidepressants (the concomitant medication log is not among the data); kinetic parameters: model of the project',
         'positive control with only the covariates of the project (P)'),
    ]
    for name, step, choice, alt in rules:
        log(f'{name}')
        log(f'   article:     {step}')
        log(f'   choice:      {choice}')
        log(f'   alternative: {alt}')

    # -------------------------------------------- 4. availability ------------
    title('4. USABLE SUBJECTS AND VISITS, BY YEAR AND BY RULE')
    log("Unit: visits of the curated data cut with YEAR (the grid of the models of the project). For each")
    log('rule: visits / subjects with at least one visit.')
    rows_disp = []
    REG_DISP = [
        ('visits of the curated data cut', lambda L: pd.Series(True, index=L.index)),
        ('with a part III examination in the raw file', lambda L: L.n_exams.notna()),
        ('with examination OFF or untreated (R3)', lambda L: L.ha_admitted),
        (f'usable: all the {len(EVERY)} items rated (R3 + R4, primary)',
         lambda L: L.useful),
        ('usable with pro-rating, with tremor', lambda L: L.UP_less_ct.notna()),
        ('usable with pro rata, without tremor', lambda L: L.UP_less_st.notna()),
        ('usable and side of onset defined (D)', lambda L: L.D_less_ct.notna()),
        ('usable and with DAT scan at the visit',
         lambda L: L.useful & L.MIA_PUTAMEN_L.notna()),
        ('updrs3_score score present in the curated data cut (reference)',
         lambda L: L.updrs3_score.notna()),
    ]
    for tag in COHORTS:
        L = LONGS[tag]
        log(f'\n[{POP}; {LBLS[tag]} cohort]')
        years = sorted(int(a) for a in L.years.unique())
        for name, f in REG_DISP:
            sel = f(L).fillna(False).astype(bool)
            for a in years + ['everyone']:
                m = sel if a == 'everyone' else sel & (L.years == a)
                rows_disp.append(dict(population=POP, cohort=tag, table='by rule',
                                       year=a, rule=name, treatment="",
                                       n_visits=int(m.sum()),
                                       n_subjects=int(L.loc[m, "PATNO"].nunique())))
            log(f"  {name}: {int(sel.sum())} visits / {int(L.loc[sel, 'PATNO'].nunique())} subjects; by year "
                + ", ".join(f"{a}: {int((sel & (L.years == a)).sum())}" for a in years))
        g = L[L.useful].groupby("PATNO").size().reindex(BASES[tag].PATNO, fill_value=0)
        log(f"  usable observations per subject (primary rule; unit: subjects, n = {len(g)}): median "
            f"{g.median():g}, quartiles {g.quantile(0.25):g}-{g.quantile(0.75):g}, maximum "
            f"{int(g.max())}; none {int((g == 0).sum())}, only one {int((g == 1).sum())}, at least "
            f"two {int((g >= 2).sum())}")
        without_scan = int((L.useful & L.MIA_PUTAMEN_L.isna()).sum())
        log(f'  usable visits without a DAT scan at the same visit: {share(without_scan, int(L.useful.sum()))} (reason for the deviation in R5: the side cannot be chosen again at each visit)')

    title('5. MISSINGNESS OF THE OFF/UNTREATED SCORE AND TREATMENT')
    log("Unit: visits of the curated data cut with YEAR. 'Treatment' is PDTRTMNT of the curated data cut at the")
    log("visit. State of the examination: derived from the rows of the raw file at the same visit.")

    def state_exam(L):
        return np.where(L.useful, 'usable',
               np.where(L.ha_admitted & (L.n_evaluated > 0), 'OFF/untreated, incomplete items',
               np.where(L.ha_admitted, 'OFF/untreated, no item rated',
               np.where(L.some_on.fillna(False).astype(bool), 'ON examination only',
               np.where(L.n_exams.notna(), 'examination without usable state',
                        'no examination in the raw file')))))

    for tag in COHORTS:
        L = LONGS[tag].copy()
        L['state_exam'] = state_exam(L)
        L["trt"] = L.trt_cur.map({0.0: 'not on treatment', 1.0: 'on treatment'}).fillna(
            'treatment not indicated')
        log(f'\n[{POP}; {LBLS[tag]} cohort]')
        for trt in ('not on treatment', 'on treatment', 'treatment not indicated'):
            s = L[L.trt == trt]
            if not len(s):
                continue
            log(f'  {trt}: {len(s)} visits; usable {share(int(s.useful.sum()), len(s))}')
            vc = s.state_exam.value_counts()
            for name in sorted(vc.index):
                log(f'      {name}: {int(vc[name])}')
                rows_disp.append(dict(population=POP, cohort=tag,
                                       table='state of the examination by treatment',
                                       year='everyone', rule=name, treatment=trt,
                                       n_visits=int(vc[name]),
                                       n_subjects=int(s.loc[s.state_exam == name,
                                                            "PATNO"].nunique())))
        years = sorted(int(a) for a in L.years.unique())
        log('  by year (visits on treatment / of which usable; visits not on treatment / of which '
            'usable):')
        for a in years:
            s = L[L.years == a]
            t1, t0 = s[s.trt == 'on treatment'], s[s.trt == 'not on treatment']
            log(f'      year {a:>2}: {len(t1):>4} / {int(t1.useful.sum()):>4}   {len(t0):>4} / {int(t0.useful.sum()):>4}')
            for trt, s_ in (('on treatment', t1), ('not on treatment', t0)):
                rows_disp.append(dict(population=POP, cohort=tag,
                                       table='usable by year and treatment',
                                       year=a, rule='usable (primary)',
                                       treatment=trt, n_visits=int(s_.useful.sum()),
                                       n_subjects=int(s_.loc[s_.useful, "PATNO"].nunique()),
                                       n_visits_totals=len(s_)))
        t1 = L[L.trt == 'on treatment']
        t0 = L[L.trt == 'not on treatment']
        q1, q0 = t1.useful.mean(), t0.useful.mean()
        led_si, led_no = t1[t1.useful].LEDD, t1[~t1.useful].LEDD
        log(f"  Summary: the score is usable in {100 * q0:.1f}% of the visits not on treatment and "
            f"in {100 * q1:.1f}% of those on treatment. "
            + ("After the start of therapy the score exists only if an examination in OFF was "
               "carried out: missingness depends on treatment." if q1 < q0 else
               "The proportion is not lower on treatment."))
        log(f'  Median LEDD at the visits on treatment: {led_si.median():.0f} mg with a usable score '
            f'(n = {int(led_si.notna().sum())}), {led_no.median():.0f} mg without (n = '
            f'{int(led_no.notna().sum())}).')
        us = L[L.useful]
        log(f"  Among the usable visits: {share(int((us.trt == 'on treatment').sum()), len(us))} are examinations in OFF of subjects on treatment, the others of untreated subjects or with treatment not indicated.")

    # ------------------------------ 6. tasks: kinetic parameters x time ------
    tasks_anc, tasks = [], []
    rhs_k = "{k}*years + " + " + ".join(COV)
    for tag, (c, kin) in COHORTS.items():
        L = LONGS[tag]
        for k in kin.values():
            common = dict(cohort=tag, kinetic=k, analysis='kinetics')
            tasks_anc.append(job(
                L, "updrs3_score", rhs_k.format(k=k), f"{k}:years", COV + [k], skipped,
                block='still', definition_side="", rule_items="curated", tremor="ct",
                side='total', **common))
            # the bilateral caudate is in the published family but not in 88 of the working
            # directory: it is refitted to verify the p values read from run_caudate
            tasks_anc.append(job(
                L, "MIA_CAUDATE_BILAT", rhs_k.format(k=k), f"{k}:years", COV + [k], skipped,
                block='still_caudate', definition_side="", rule_items="", tremor="",
                side="", cohort=tag, kinetic=k, analysis='caudate anchor'))
            for trem in GROUPS:
                tasks.append(job(
                    L, f"tot_{trem}", rhs_k.format(k=k), f"{k}:years", COV + [k], skipped,
                    block='total', definition_side="", rule_items='complete',
                    tremor=trem, side='total', **common))
                for block, pref, dl, rv, sides in (
                        ('primary', "U", "U", 'complete', ('less', 'more', "diff")),
                        ('sens_onset', "D", "D", 'complete', ('less', 'more', "diff")),
                        ('sens_limbs', "U", "U", 'complete', ('limbsless', "artipiu")),
                        ("sens_prorata", "UP", "U", "prorata", ('less', 'more', "diff"))):
                    for side in sides:
                        tasks.append(job(
                            L, f'{pref}_{side}_{trem}', rhs_k.format(k=k), f"{k}:years",
                            COV + [k], skipped, block=block, definition_side=dl,
                            rule_items=rv, tremor=trem, side=side.replace('limbs', ""),
                            **common))

    # ------------------------------------- 7. tables of the positive control --
    title('6. POSITIVE CONTROL: CONSTRUCTION OF THE EXAMINATION DATES')
    log("Rules of the article (R5 with V, R6, R7, R8). Unit: examination dates = visits with")
    log('a DAT scan. Time in months from the first usable date of the subject (from the')
    log("ages at scan of the curated data cut); the model is fitted with time in years and the")
    log("interaction coefficient is divided by 12 (same model, different unit).")
    PC, rows_by = {}, []
    for pop, dataset in POPULATIONS.items():
        S = V[V.PATNO.isin(dataset) & V.MIA_PUTAMEN_L.notna() & V.MIA_PUTAMEN_R.notna()].copy()
        S = S.merge(blc[["PATNO", "age_bl", "SEX", "dur_bl", "handed", "EDUCYRS",
                         'lower_bl']], on="PATNO", how="left", validate="many_to_one")
        stream = [('visits with DAT scan (left and right putamen)', len(S),
                   S.PATNO.nunique())]
        S = S[S.useful]
        stream.append(('with usable score at the same visit (R3 + R4)', len(S),
                       S.PATNO.nunique()))
        S['departure_months'] = 12 * (S.age_at_visit - S.age_datscan)
        S = S[S.departure_months.abs() <= WINDOW_MONTHS + 1e-9]
        stream.append((f'clinical examination within +/-{WINDOW_MONTHS:g} months from the scan', len(S),
                       S.PATNO.nunique()))
        S["ledd_g"] = S.LEDD / 1000.0
        S = S.dropna(subset=COV_ART)
        stream.append(('with MoCA, LEDD, handedness, education and baseline covariates',
                       len(S), S.PATNO.nunique()))
        S = S.sort_values(["PATNO", "age_datscan", "EVENT_ID"], kind="mergesort")
        n_selfsame = int(S.duplicated(["PATNO", "age_datscan"]).sum())
        S = S.drop_duplicates(["PATNO", "age_datscan"], keep="first")
        n_date = S.groupby("PATNO").PATNO.transform("size")
        S = S[n_date >= MIN_DATE].copy()
        stream.append((f'subjects with at least {MIN_DATE} dates (same scan repeated on two visits: {n_selfsame} discarded)', len(S), S.PATNO.nunique()))
        S['months'] = 12 * (S.age_datscan - S.groupby("PATNO").age_datscan.transform("min"))
        S["years"] = S.months / 12.0
        s_, d_ = S.MIA_PUTAMEN_L, S.MIA_PUTAMEN_R
        mean = (s_ + d_) / 2
        ai = (s_ - d_) / (s_ + d_)
        # V: side at each date. Left putamen with lower uptake -> less affected side
        # of the body = left.
        side_V = np.where(s_ < d_, "S", np.where(d_ < s_, "D", "="))
        side_V5 = np.where(ai.abs() <= AI_THRESHOLD, "=", side_V)
        S['side_V'], S['side_V5'] = side_V, side_V5
        S['sbr_less_V'], S['sbr_more_V'] = np.maximum(s_, d_), np.minimum(s_, d_)
        S['sbr_less_V5'] = np.where(side_V5 == "=", mean, S.sbr_less_V)
        S['sbr_more_V5'] = np.where(side_V5 == "=", mean, S.sbr_more_V)
        S['sbr_less_U'] = np.where(S.lower_bl == "S", d_, np.where(S.lower_bl == "D", s_, mean))
        S['sbr_more_U'] = np.where(S.lower_bl == "S", s_, np.where(S.lower_bl == "D", d_, mean))
        S["sbr_bil"] = S.MIA_PUTAMEN_BILAT
        S = outcomes_side(S, S.side_V, "V", "C")
        S = outcomes_side(S, S.side_V5, "V5", "C")
        S = outcomes_side(S, S.lower_bl, "U", "C")
        PC[pop] = S
        log(f'\n[{POP}; {LBLS_POP[pop]}]')
        for name, nv, ns in stream:
            log(f'  {name}: {nv} dates / {ns} subjects')
        g = S.groupby("PATNO").size()
        log(f"  dates per subject: median {g.median():g}, quartiles {g.quantile(0.25):g}-{g.quantile(0.75):g}, maximum {int(g.max())}; median follow-up {S.groupby('PATNO').months.max().median():.1f} months (maximum {S.months.max():.1f}). Published: {DZ['n']} subjects, {DZ['date']} dates.")
        log(f"  symmetric dates: {int((side_V == '=').sum())} with threshold 0; {int((side_V5 == '=').sum())} with threshold {100 * AI_THRESHOLD:.0f}%")
        # clinical side and imaging side at the same date (article: 24% not aligned)
        cl = np.where(S.C_D_ct > S.C_S_ct, "D", np.where(S.C_S_ct > S.C_D_ct, "S", "="))
        img = np.where(side_V == "S", "D", np.where(side_V == "D", "S", "="))
        sel = (cl != "=") & (img != "=")
        ka, na = int((cl[sel] == img[sel]).sum()), int(sel.sum())
        log('  clinical side (higher sum, with tremor) = contralateral to the putamen with lower '
            'uptake, at the same date: '
            + record(pop, 'examination dates', 'clinical (with tremor) = imaging at the same date',
                       ka, na, 'rule V, ties excluded')
            + f"; not aligned {share(na - ka, na)} (published: {100 * DZ['no_aligned_AI0']:.0f}%)")
        # scores by period, with the periods of the article
        S['period'] = np.where(S.months <= 6, 'year 0 (0-6 months)',
                       np.where(S.months <= 18, 'year 1 (7-18 months)',
                       np.where(S.months <= 42, 'years 2-3 (19-42 months)', 'year >= 4 (> 42 months)')))
        PC[pop] = S
        for per in ('year 0 (0-6 months)', 'year 1 (7-18 months)', 'years 2-3 (19-42 months)',
                    'year >= 4 (> 42 months)'):
            sp = S[S.period == per]
            for trem in GROUPS:
                for side, col in (("tot", f"tot_{trem}"), ('more', f'V_more_{trem}'),
                                  ('less', f'V_less_{trem}')):
                    pub = t2[(trem, side)] if per.startswith('year 0') else (np.nan, np.nan)
                    rows_by.append(dict(
                        population=POP, cohort=pop, period=per, n_date=len(sp),
                        tremor=TREM[trem], score=side,
                        mean=float(sp[col].mean()) if len(sp) else np.nan,
                        sd=float(sp[col].std()) if len(sp) > 1 else np.nan,
                        mean_pubd_year0=pub[0], sd_pubd_year0=pub[1],
                        units='MDS-UPDRS III points, OFF or untreated; side by rule V'))
            for name, col in (('less affected', 'sbr_less_V'), ('more affected', 'sbr_more_V'),
                              ('bilateral', "sbr_bil")):
                rows_by.append(dict(
                    population=POP, cohort=pop, period=per, n_date=len(sp),
                    tremor="", score=f'SBR putamen {name}',
                    mean=float(sp[col].mean()) if len(sp) else np.nan,
                    sd=float(sp[col].std()) if len(sp) > 1 else np.nan,
                    mean_pubd_year0=np.nan, sd_pubd_year0=np.nan,
                    units='SBR, MIAKAT, cerebral white matter reference'))
        s0 = S[S.period == 'year 0 (0-6 months)']
        log(f"  year 0 ({len(s0)} dates), mean (SD), alongside Table 2 of the article:")
        for trem in GROUPS:
            log(f"    {TREM[trem]}: " + "; ".join(
                f'{name} {s0[col].mean():.1f} ({s0[col].std():.1f}) [publ. {t2[trem, side][0]:.1f} ({t2[trem, side][1]:.1f})]'
                for name, side, col in (('total', "tot", f"tot_{trem}"),
                                        ('more affected side', 'more', f'V_more_{trem}'),
                                        ('less affected side', 'less', f'V_less_{trem}'))))
        # tasks of the positive control
        for cov in ("A", "P"):
            extra = COV_ART if cov == "A" else COV
            for trem in GROUPS:
                common = dict(cohort=pop, analysis='positive control', covariate=cov,
                              tremor=trem, rule_items='complete', block=f"cp_{cov}")
                tasks.append(job(
                    S, f"tot_{trem}", f"sbr_bil*years + {F_COV[cov]}", "sbr_bil:years",
                    extra + ["sbr_bil"], skipped, definition_side="", side='bilateral',
                    predictor="sbr_bil", **common))
                for dl in (("V", "U", "V5") if cov == "A" else ("V", "U")):
                    for side in ('less', 'more'):
                        x = f'sbr_{side}_{dl}'
                        tasks.append(job(
                            S, f'{dl}_{side}_{trem}', f"{x}*years + {F_COV[cov]}",
                            f"{x}:years", extra + [x], skipped, definition_side=dl,
                            side=side, predictor=x, **common))
    n_skipped = len(skipped)
    assert all(t is not None for t in tasks_anc), \
        'an anchoring model does not have enough observations'
    tasks = [t for t in tasks if t is not None]

    # ----------------------------------------------------- 8. estimation -----
    title('7. ESTIMATION')
    log(f'cascade: {OPTIMISERS}; criterion: Newton decrement < {TOL_NEWTON:g}, profiled Hessian negative definite, no variance on the boundary or singular')
    log(f'anchoring models (fitted first): {len(tasks_anc)}; other models: {len(tasks)}; skipped for insufficient sample size (< {N_MIN_OBSV} observations or < {N_MIN_MULTI} subjects with more than one visit): {n_skipped}')
    for s in skipped:
        log(f"  skipped: {s['cohort']} {s['block']} {s['outcome']} x {(s['kinetic'] if 'kinetic' in s else s.get('predictor'))} ({s['n_obsv']} observations, {s['n_multivisit']} subjects with more than one visit)")
    print(f'[{N_PROC} processes]', flush=True)
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        ANC_T = pd.DataFrame(pool.map(fitmodel, tasks_anc, chunksize=1))
        ANC = ANC_T[ANC_T.block == 'still'].reset_index(drop=True)
        ANCC = ANC_T[ANC_T.block == 'still_caudate'].reset_index(drop=True)
        # identity with the pipeline BEFORE fitting the rest
        f88 = os.path.join(OUT, "88_diagnostics_63.csv")
        assert os.path.exists(f88), \
            "88_diagnostics_63.csv missing: the anchoring to the pipeline cannot be run"
        pub = pd.read_csv(f88)
        assert ANC_T.estimated.all(), 'an anchoring model produced no estimate'
        j = ANC.merge(pub[pub.outcome == "updrs3_score"].rename(
            columns={"cohort": 'cohort', "kinetic": 'kinetic'})[
                ['cohort', 'kinetic', "beta_interaction", "se", "p", "n_subjects", "n_obs"]],
            on=['cohort', 'kinetic'], how="left", validate="one_to_one", suffixes=("", "_pip"))
        assert j.beta_interaction.notna().all(), \
            '88_diagnostics_63.csv does not contain all the anchoring models'
        assert (j.n_subj == j.n_subjects).all() and (j.n_obsv == j.n_obs).all(), \
            "anchoring: sample sizes differ from 88_diagnostics_63.csv"
        db = float((j.beta - j.beta_interaction).abs().max())
        sd = float((j.se - j.se).abs().max())
        dp = float((j.p - j.p_pip).abs().max())
        assert db < TOL_ID_BETA and sd < TOL_ID_BETA and dp < TOL_ID_P, \
            (f'anchoring: MDS-UPDRS III x kinetic parameter does not match 88_diagnostics_63.csv (discrepancies: beta {db:.2e}, SE {sd:.2e}, p {dp:.2e}): kinetic parameters, covariates, cohort or visits are not those of the pipeline')
        control('anchoring to the pipeline (kinetic parameters, covariates, visits, estimation)', 'passed',
                  f'i {len(j)} kinetic parameter x time models on updrs3_score re-estimated here coincide with 88_diagnostics_63.csv: same sample sizes, maximum difference in beta {db:.1e}, SE {sd:.1e}, p {dp:.1e}')
        # Published family: that of the manuscript (with the caudate) is in run_caudate.
        # This identity too is verified BEFORE fitting the rest. If the comparison
        # cannot be run (run_caudate absent, or from other cohorts) this is declared and
        # the enlarged family is not computed; if it can be run and the values do not
        # coincide the script stops.
        fcaud = os.path.join(os.path.dirname(os.path.normpath(BASE)), 'run_caudate', "outputs")
        PUBF, reason_all, n_common, only_caud, jc = None, "", 0, [], None
        if not os.path.exists(os.path.join(fcaud, "88_diagnostics_63.csv")):
            reason_all = ('run_caudate/outputs/88_diagnostics_63.csv does not exist beside the working directory')
        else:
            matching = True
            for name, (c, kin) in (("analysis_complete_cases.csv", COHORTS["24h"]),
                                   ("08_dataset_150h.csv", COHORTS["150h"])):
                pth = os.path.join(fcaud, name)
                if not os.path.exists(pth):
                    matching = False
                    continue
                t = pd.read_csv(pth)
                for col in ("complete_primary", "complete"):
                    if col in t.columns:
                        t = t[t[col]]
                t["PATNO"] = t["PATNO"].astype("int64")
                a = t[["PATNO"] + list(kin)].sort_values("PATNO").reset_index(drop=True)
                b = c[["PATNO"] + list(kin)].sort_values("PATNO").reset_index(drop=True)
                matching = matching and a.equals(b)
            pc_ = pd.read_csv(os.path.join(fcaud, "88_diagnostics_63.csv"))
            n_caud_file = int((pc_.outcome == "MIA_CAUDATE_BILAT").sum())
            if not matching:
                reason_all = 'the cohorts of run_caudate are not those of this working directory'
            elif not n_caud_file:
                reason_all = ('88_diagnostics_63.csv of run_caudate does not contain the bilateral caudate: it is not the published family')
            else:
                keys = ['cohort', 'kinetic']
                rin = {"cohort": 'cohort', "kinetic": 'kinetic'}
                jj = ANC.merge(pc_[pc_.outcome == "updrs3_score"].rename(columns=rin)[
                    keys + ["beta_interaction", "p"]],
                    on=keys, how="left", validate="one_to_one", suffixes=("", "_c"))
                assert jj.beta_interaction.notna().all(), \
                    '88_diagnostics_63.csv of run_caudate does not contain all the anchoring models'
                dbj = float((jj.beta - jj.beta_interaction).abs().max())
                dpj = float((jj.p - jj.p_c).abs().max())
                assert dbj < TOL_ID_BETA and dpj < TOL_ID_P, \
                    (f'published family: the anchoring models do not match 88_diagnostics_63.csv of run_caudate (discrepancies: beta {dbj:.2e}, p {dpj:.2e})')
                # the outcomes common to the two files must have the same p values
                cm = pub.merge(pc_, on=["cohort", "outcome", "kinetic"], suffixes=("", "_c"),
                               validate="one_to_one")
                n_common = len(cm)
                assert n_common, \
                    '88_diagnostics_63.csv of the working directory and of run_caudate have no models in common'
                dpc = float((cm.p - cm.p_c).abs().max())
                assert dpc < TOL_ID_P, \
                    (f'published family: the p values of the outcomes common to 88_diagnostics_63.csv of the working directory and of run_caudate do not match (maximum discrepancy {dpc:.2e})')
                only_caud = sorted(set(pc_.outcome) - set(pub.outcome) - {"MIA_CAUDATE_BILAT"})
                # the bilateral caudate is not in 88 of the working directory: refitted here, it
                # must coincide with that of run_caudate (sample size, beta, SE and p)
                jc = ANCC.merge(pc_[pc_.outcome == "MIA_CAUDATE_BILAT"].rename(columns=rin)[
                    keys + ["beta_interaction", "se", "p", "n_subjects", "n_obs"]],
                    on=keys, how="left", validate="one_to_one", suffixes=("", "_c"))
                assert len(jc) == n_caud_file and jc.beta_interaction.notna().all(), \
                    (f'published family: the bilateral caudate models refitted here ({len(jc)}) are not those of 88_diagnostics_63.csv of run_caudate ({n_caud_file})')
                assert (jc.n_subj == jc.n_subjects).all() and (jc.n_obsv == jc.n_obs).all(), \
                    ('published family: bilateral caudate, sample sizes differ from 88_diagnostics_63.csv of run_caudate')
                dbc = float((jc.beta - jc.beta_interaction).abs().max())
                dsc = float((jc.se - jc.se).abs().max())
                dpk = float((jc.p - jc.p_c).abs().max())
                assert dbc < TOL_ID_BETA and dsc < TOL_ID_BETA and dpk < TOL_ID_P, \
                    (f'published family: the bilateral caudate refitted here does not match 88_diagnostics_63.csv of run_caudate (discrepancies: beta {dbc:.2e}, SE {dsc:.2e}, p {dpk:.2e}): a value of the published family has been altered, or the data are not those of that time')
                PUBF = pc_
        if PUBF is None:
            print(f'[published family NOT verified: {reason_all}]', flush=True)
        outcomes = []
        for i, r in enumerate(pool.imap(fitmodel, tasks, chunksize=1)):
            outcomes.append(r)
            if (i + 1) % 20 == 0:
                print(f'  ... {i + 1}/{len(tasks)}  ({time.time() - t_start:.0f} s)',
                      flush=True)
    # order: anchor, new models, at the end the refitted caudate models
    M = pd.concat([ANC, pd.DataFrame(outcomes), ANCC], ignore_index=True)
    n_ns = int((~M.estimated).sum())
    n_no = int((~M.accepted).sum())
    Ms = M[M.estimated]
    log(f'models attempted: {len(M)}; without any estimate: {n_ns}; meeting the criterion: {int(M.accepted.sum())}; NOT accepted: {n_no}')
    log(f"  maximum Newton decrement among the accepted {M.loc[M.accepted, 'newton'].max():.2e}; maximum gain of the restart {Ms.gain_reoptimisation.max():.2e} log-likelihood units; rounds from {int(Ms.rounds.min())} to {int(Ms.rounds.max())}")
    if n_no:
        log('  NOT accepted (reported in the CSV files, excluded from families and summary sentences):')
        log(M.loc[~M.accepted, ['analysis', 'cohort', 'block', 'outcome', 'kinetic',
                                 'predictor', 'estimated', "newton", 'n_optim_convergent',
                                 'singular', 'edge']].to_string(index=False))

    # --------------------------------- 9. kinetic parameters x time ----------
    title('8. KINETIC PARAMETER x TIME ON THE LATERALISED MOTOR SCORES')
    log(f'Population: {POP}. ANALYSIS ADDED IN REVIEW.')
    log("Model: outcome ~ z(kinetic parameter)*years + age + sex + duration at baseline, random intercept and")
    log('random slope by subject, REML. beta = interaction, in MDS-UPDRS III points per')
    log("year per 1 SD of the logarithm of the kinetic parameter (SD over the whole cohort). Examination in OFF or")
    log("of an untreated subject. beta > 0: with a higher kinetic parameter the score grows")
    log("faster (more rapid worsening); beta < 0: more slowly. p = Wald,")
    log("uncorrected. eff.rel. = 100 x beta / mean annual slope of the outcome.")
    log('Correction families, Benjamini-Hochberg WITHIN COHORT:')
    log("  111-lati       kinetic parameters x {less affected side, more affected side} x {with, without")
    log('                 tremor}; side by baseline uptake (U), complete rule;')
    log('  111-contrasti  kinetic parameters x {with, without tremor}: difference between the two sides;')
    log("  sensitivity    each block (onset, limbs only, pro rata) within its own family")
    log('                 of the same form.')
    log('Relation to the published family: these tests are NOT part of it; they are')
    log("additional and specified in review. The column 'BH all.' shows what")
    log('would happen by adding 111-lati to the published family of the cohort.')
    K = M[M.analysis.isin(['kinetics', 'caudate anchor'])].copy()
    # for the difference between the sides the mean slope is close to zero: no ratio
    K['effect_rel_pct'] = (100 * K.beta / K.slope_mean).where(K.side != "diff")
    K['family_BH'], K['n_family'], K["p_BH"] = "", np.nan, np.nan
    K['p_BH_widened'], K['n_family_widened'] = np.nan, np.nan
    NAMES_FAM = {'primary': ('111-sides', "111-contrasti"),
                'sens_onset': ("111-lati-esordio", "111-contrasti-esordio"),
                'sens_limbs': ('111-sides-sole-limbs', None),
                "sens_prorata": ('111-sides-prorata', "111-contrasti-prorata")}
    for tag in COHORTS:
        for block, (f_sides, f_contr) in NAMES_FAM.items():
            for sides, fam in ((('less', 'more'), f_sides), (("diff",), f_contr)):
                i = K.index[(K.cohort == tag) & (K.block == block) & K.side.isin(sides)
                            & K.accepted]
                if fam is None or not len(i):
                    continue
                K.loc[i, 'family_BH'] = fam
                K.loc[i, 'n_family'] = len(i)
                K.loc[i, "p_BH"] = bh(K.loc[i, "p"].to_numpy())
    # (the identity with the published family of run_caudate has already been verified
    # before the estimation: here its outcome, PUBF, is used)
    fam_summ = {}
    if PUBF is None:
        control('published family (for the enlarged family)', 'NOT verified',
                  f'{reason_all}: the enlarged family is NOT computed')
    else:
        for tag, (c, kin) in COHORTS.items():
            pf = PUBF[PUBF.cohort == tag]
            assert set(pf.kinetic) == set(kin.values()) and not pf.duplicated(
                ["outcome", "kinetic"]).any() and len(pf) == pf.outcome.nunique() * len(kin), \
                'the published family of run_caudate does not have the expected grid'
            i = K.index[(K.cohort == tag) & (K.block == 'primary')
                        & K.side.isin(['less', 'more']) & K.accepted]
            everyone = np.concatenate([pf.p.to_numpy(), K.loc[i, "p"].to_numpy()])
            q = bh(everyone)
            K.loc[i, 'p_BH_widened'] = q[len(pf):]
            K.loc[i, 'n_family_widened'] = len(everyone)
            q_only = bh(pf.p.to_numpy())
            names_pub = [f"{a} x {b}" for a, b in zip(pf.outcome, pf.kinetic)]
            fam_summ[tag] = dict(
                n_pub=len(pf), outcomes_pub=sorted(pf.outcome.unique()),
                surv_pub=int((q_only < ALPHA).sum()),
                chi_pub=[f"{n_} (BH {fmt_p(v)})" for n_, v in zip(names_pub, q_only)
                         if v < ALPHA],
                n_all=len(everyone), surv_all_pub=int((q[:len(pf)] < ALPHA).sum()),
                chi_all=[f"{n_} (BH {fmt_p(v)})" for n_, v in zip(names_pub, q[:len(pf)])
                         if v < ALPHA],
                lost=[f'{n_} (BH from {fmt_p(v0)} a {fmt_p(v)})'
                       for n_, v0, v in zip(names_pub, q_only, q[:len(pf)])
                       if v0 < ALPHA <= v],
                surv_all_new=int((q[len(pf):] < ALPHA).sum()))
        control('published family (for the enlarged family)', 'passed',
                  f'cohorts of run_caudate identical to those of the working directory (subjects and kinetic parameters); anchor models coinciding with its 88_diagnostics_63.csv; identical p values in the {n_common} models common to the two files; the {len(jc)} models of the bilateral caudate re-estimated here coincide with its own (same sample sizes, beta, SE and p)')
        if only_caud:
            control('published family: outcomes present only in run_caudate',
                      'NOT verified',
                      f"the p values of {', '.join(only_caud)} are read from run_caudate and neither re-estimated nor compared here")

    def row_k(r):
        return (f'    {r.kinetic:<6} {r.n_subj:>4} {r.n_obsv:>5}  {r.beta:>+8.4f}  [{r.ci_inf:>+7.4f}; {r.ci_upp:>+7.4f}]  {fmt_p(r.p):>8}  {fmt_p(r.p_BH):>8}  {fmt_p(r.p_BH_widened):>8}  '
                + (f'{r.effect_rel_pct:>+7.1f}%' if np.isfinite(r.effect_rel_pct)
                   else "      -"))

    UNITS_CAUD = 'SBR (MIAKAT, CWM) per year per 1 SD of the log of the kinetic parameter'
    head = ('    kinet. subj.  obs.      beta  [95% CI]                   p        BH   BH enl.  rel.eff.')
    for tag in COHORTS:
        g0 = K[(K.cohort == tag)]
        log(f'\n[{POP}; {LBLS[tag]} cohort; unit of beta: points/year per 1 SD]')
        gp_ = g0[(g0.block == 'primary') & g0.side.isin(['less', 'more'])]
        if len(gp_):
            log(f'  observations per subject in the primary models: median {gp_.obsv_by_subj_median.min():g}'
                + ("" if gp_.obsv_by_subj_median.nunique() == 1
                   else f'-{gp_.obsv_by_subj_median.max():g}')
                + f', quartiles {gp_.obsv_by_subj_q1.min():g}-{gp_.obsv_by_subj_q3.max():g}, maximum {int(gp_.obsv_by_subj_max.max())}; subjects with only one observation from {int(gp_.n_one_single.min())} to {int(gp_.n_one_single.max())}, with at least two from {int(gp_.n_multivisit.min())} to {int(gp_.n_multivisit.max())}; last year observed {gp_.years_max.max():g}')
        for block, descr in (('still', 'MDS-UPDRS III of the curated data cut (published family; identity with 88 verified)'),
                              ('total', 'total reconstructed from the items, same visits as the side scores (reference, outside the family)'),
                              ('primary', f"PRIMARY: side by {DEF_SIDE['U']}; full rule"),
                              ('sens_onset', f"sensitivity: side by {DEF_SIDE['D']}"),
                              ('sens_limbs', 'sensitivity: limbs only (without non-lateralised items), side U'),
                              ("sens_prorata", 'sensitivity: non-assessable items pro rata, side U')):
            gb = g0[g0.block == block]
            if not len(gb):
                continue
            log(f"  -- {descr}")
            for trem in GROUPS:
                for side in ('total', 'less', 'more', "diff"):
                    gl = gb[(gb.tremor == trem) & (gb.side == side)]
                    if not len(gl):
                        continue
                    name = 'total' if side == 'total' else SIDES[side]
                    fam = next((x for x in gl.family_BH if x), "")
                    log(f'   {name}, {TREM[trem]}'
                        + (f'  [family {fam}, {int(gl.n_family.max())} tests]' if fam
                           else "  [member of the published family: its BH is in "
                                "88_diagnostics_63.csv]" if block == 'still'
                           else '  [outside any family]'))
                    log(head)
                    for r in gl.itertuples():
                        if r.accepted:
                            log(row_k(r))
                        else:
                            log(f'    {r.kinetic:<6} MODEL NOT ACCEPTED (convergence criterion not met): no estimate reported')
        gc = g0[g0.block == 'still_caudate']
        if len(gc):
            log('  -- bilateral caudate (MIA_CAUDATE_BILAT), member of the published family: refitted here only to verify the p values read from run_caudate ('
                + ("identity verified" if PUBF is not None else
                   "identity NOT verified: see the checks")
                + f'). Unit of beta: {UNITS_CAUD}. It is not a motor score and does not enter any family of this script.')
            log(head)
            for r in gc.itertuples():
                if r.accepted:
                    log(row_k(r))
                else:
                    log(f'    {r.kinetic:<6} MODEL NOT ACCEPTED (convergence criterion not met): no estimate reported')

    # ----------------------------------------- 10. positive control ----------
    title('9. POSITIVE CONTROL: SBR x TIME ON THE LATERALISED SCORE')
    log(f'Population: {POP}. Outcome: MDS-UPDRS III of the side of the body contralateral to the')
    log('putamen indicated (items of the side + non-lateralised), OFF or untreated; for the')
    log('bilateral ratio, the total score. beta = SBR x time interaction, in points')
    log("per MONTH per unit of SBR (negative: with lower SBR, more rapid worsening).")
    log("SBR: MIAKAT, cerebral white matter reference (the article: occipital")
    log("reference). BH within population in the block of six tests (less, more, bilateral x")
    log('with/without tremor) of the same side rule and of the same covariates.')
    log(f"Covariates A: {COV_DESCR['A']}. Covariates P: {COV_DESCR['P']}.")
    C = M[M.analysis == 'positive control'].copy()
    for col in ("beta", 'se', 'ci_inf', 'ci_upp'):
        C[col + '_month'] = C[col] / 12.0
    C["p_BH"], C['n_family'], C['family_BH'] = np.nan, np.nan, ""
    for pop in POPULATIONS:
        for cov in ("A", "P"):
            for dl in (("V", "U", "V5") if cov == "A" else ("V", "U")):
                common = (C.cohort == pop) & (C.covariate == cov) & C.accepted
                i_lat = list(C.index[common & (C.definition_side == dl)])
                i_bil = list(C.index[common & (C.side == 'bilateral')])
                q = pd.Series(bh(C.loc[i_lat + i_bil, "p"].to_numpy()), index=i_lat + i_bil)
                # the two bilateral models are the same in every block: their BH is
                # written once, in the block of the rule of the article (V)
                write = i_lat + (i_bil if dl == "V" else [])
                C.loc[write, "p_BH"] = q[write]
                C.loc[write, 'n_family'] = len(q)
                C.loc[write, 'family_BH'] = f"111-cp-{dl}-{cov}"
    PUBL = {('less', "st"): DZ['less_st'], ('more', "st"): DZ['more_st'],
             ('bilateral', "st"): DZ["bil_st"], ('less', "ct"): DZ['less_ct']}
    C['beta_published'] = [PUBL.get((a, b), {}).get("beta", np.nan)
                            for a, b in zip(C.side, C.tremor)]
    C['p_published'] = [PUBL.get((a, b), {}).get("p", np.nan)
                         for a, b in zip(C.side, C.tremor)]
    NAME_L = {'less': 'less affected putamen -> contralateral side',
              'more': 'more affected putamen -> contralateral side',
              'bilateral': 'bilateral putamen -> total'}
    for pop in POPULATIONS:
        log(f'\n[{POP}; {LBLS_POP[pop]}]')
        for cov, dl in (("A", "V"), ("P", "V"), ("A", "U"), ("P", "U"), ("A", "V5")):
            g = C[(C.cohort == pop) & (C.covariate == cov)
                  & ((C.definition_side == dl) | (C.side == 'bilateral'))]
            log(f'  -- side: {DEF_SIDE[dl]}; covariates {cov}'
                + ("  [the specification closest to the article]"
                   if (cov, dl) == ("A", "V") else ""))
            log('     predictor -> outcome                          trem.  subj. dates  beta/month '
                '[95% CI]              p       BH   published')
            for trem in ("st", "ct"):
                for side in ('less', 'more', 'bilateral'):
                    for r in g[(g.tremor == trem) & (g.side == side)].itertuples():
                        pb = PUBL.get((side, trem))
                        pub_s = (f"{pb['beta']:+.2f} (p = {pb['p']:g})" if pb else 'not given')
                        if not r.accepted:
                            log(f'     {NAME_L[side]:<45} {trem}   MODEL NOT ACCEPTED')
                            continue
                        bhs = fmt_p(r.p_BH) if (r.definition_side == dl or dl == "V") else "  (v. V)"
                        log(f'     {NAME_L[side]:<45} {trem}  {r.n_subj:>5} {r.n_obsv:>5}  {r.beta_month:>+8.4f}  [{r.ci_inf_month:>+7.4f}; {r.ci_upp_month:>+7.4f}]  {fmt_p(r.p):>7}  {bhs:>7}   {pub_s}')

    # -------------------------------------------------- 11. summary ----------
    title("10. SUMMARY (each sentence is derived from the computed values)")
    Ka = K[K.accepted]
    for tag, (c, kin) in COHORTS.items():
        log(f'\nKinetic parameters, {LBLS[tag]} cohort.')
        for block in NAMES_FAM:
            g = Ka[(Ka.cohort == tag) & (Ka.block == block) & Ka.side.isin(['less', 'more'])]
            # models planned by the design: kinetic parameters x two sides x with/without tremor
            expected = len(kin) * 2 * len(GROUPS)
            attempted = int(((K.cohort == tag) & (K.block == block)
                           & K.side.isin(['less', 'more'])).sum())
            if not len(g):
                log(f'  {NAMES_FAM[block][0]}: NO model accepted out of {expected} planned ({expected - attempted} not fitted for insufficient sample size, {attempted} fitted and not accepted): no sentence.')
                continue
            nom, so = g[g.p < ALPHA], g[g.p_BH < ALPHA]

            def chi(x):
                return "; ".join(f'{r.kinetic} side {r.side} {r.tremor} (beta {r.beta:+.3f}, p = {fmt_p(r.p)}, BH {fmt_p(r.p_BH)})' for r in x.itertuples())
            log(f'  {NAMES_FAM[block][0]} ({len(g)} tests accepted out of {expected} planned, {attempted} fitted): nominal {len(nom)}' + (f" [{chi(nom)}]" if len(nom) else "")
                + f'; passing BH {len(so)}' + (f" [{chi(so)}]" if len(so) else "")
                + f'; minimum p {fmt_p(g.p.min())}, minimum BH {fmt_p(g.p_BH.min())}.')
            gd = Ka[(Ka.cohort == tag) & (Ka.block == block) & (Ka.side == "diff")]
            if NAMES_FAM[block][1] is not None and not len(gd):
                log(f'     contrast between the sides ({NAMES_FAM[block][1]}): no model accepted.')
            if len(gd):
                nd, bd = gd[gd.p < ALPHA], gd[gd.p_BH < ALPHA]
                log(f'     contrast between the sides ({NAMES_FAM[block][1]}, {len(gd)} tests): nominal {len(nd)}'
                    + (" [" + "; ".join(f'{r.kinetic} {r.tremor} (beta {r.beta:+.3f}, p = {fmt_p(r.p)})' for r in nd.itertuples()) + "]"
                       if len(nd) else "")
                    + f'; after BH {len(bd)}; minimum p {fmt_p(gd.p.min())}. '
                    + ('No contrast distinguishes the two sides after the correction.'
                       if not len(bd) else
                       'At least one contrast distinguishes the two sides after the correction.'))
        gp = Ka[(Ka.cohort == tag) & (Ka.block == 'primary') & Ka.side.isin(['less', 'more'])]
        # The same test in the other cohort. It is not a replication of the same measure:
        # the assay protocol is a different one, the kinetic parameter only has the same
        # name and the participants are different, apart from the shared participants.
        other = next(t for t in COHORTS if t != tag)
        kin_other = set(COHORTS[other][1].values())
        for r in gp[gp.p < ALPHA].itertuples():
            sel = ((K.cohort == other) & (K.block == 'primary') & (K.side == r.side)
                   & (K.tremor == r.tremor) & (K.kinetic == r.kinetic))
            o = K[sel & K.accepted]
            if r.kinetic not in kin_other:
                outcome_other = (f'the kinetic parameter {r.kinetic} is not among those of that cohort: no corresponding test.')
            elif len(o) == 1:
                outcome_other = (f"beta {o.beta.iloc[0]:+.3f} [95% CI {o.ci_inf.iloc[0]:+.3f}; {o.ci_upp.iloc[0]:+.3f}], p = {fmt_p(o.p.iloc[0])}; sign {('concordant' if o.beta.iloc[0] * r.beta > 0 else 'opposite')}.")
            elif int(sel.sum()):
                outcome_other = 'the corresponding model does not meet the estimation criterion.'
            else:
                outcome_other = ('the corresponding model was not estimated (insufficient sample size).')
            log(f'  The nominal test {r.kinetic}, side {r.side}, {TREM[r.tremor]} (beta {r.beta:+.3f} [95% CI {r.ci_inf:+.3f}; {r.ci_upp:+.3f}]) has as its counterpart, in the {LBLS[other]} cohort (other protocol of the assay, kinetic parameter of the same name; subjects in common between the two cohorts: {len(p24 & p150)}): ' + outcome_other)
        if len(gp):
            log(f'  Width of the intervals (primary): the narrowest 95% CI is {(gp.ci_upp - gp.ci_inf).min():.2f} points/year per SD wide, the widest {(gp.ci_upp - gp.ci_inf).max():.2f}; the mean annual slope of the outcome ranges from {gp.slope_mean.min():+.2f} to {gp.slope_mean.max():+.2f} points/year.')
        if tag in fam_summ:
            fs = fam_summ[tag]
            log(f"  Extended family (published, {fs['n_pub']} tests, plus {fs['n_all'] - fs['n_pub']} accepted tests of 111-lati = {fs['n_all']} tests): passing the correction {fs['surv_all_pub']} among the published and {fs['surv_all_new']} among the new ones; in the published family alone they were {fs['surv_pub']}"
                + (f" [{'; '.join(fs['chi_pub'])}]" if fs["chi_pub"] else "") + ". "
                + ('Adding the lateralised scores does not change the number of published tests '
                   'that survive.' if fs['surv_all_pub'] == fs['surv_pub'] else
                   f"Adding the lateralised scores CHANGES the number of published tests that survive (the procedure is stepwise and the family is larger): dropping out {'; '.join(fs['lost'])}."))
        else:
            log('  Extended family: NOT computed (see the checks).')
    Ca = C[C.accepted]
    log()
    outcome_cp = {}
    for pop in POPULATIONS:
        def one(side, trem, cov="A", dl="V"):
            g = Ca[(Ca.cohort == pop) & (Ca.covariate == cov) & (Ca.tremor == trem)
                   & (Ca.side == side)
                   & ((Ca.definition_side == dl) | (Ca.side == 'bilateral'))]
            return g.iloc[0] if len(g) == 1 else None
        me, pi_, bi = one('less', "st"), one('more', "st"), one('bilateral', "st")
        log(f'Positive control, {LBLS_POP[pop]} (rule V, covariates A, without tremor).')
        if me is None:
            c_ = C[(C.cohort == pop) & (C.covariate == "A") & (C.tremor == "st")
                   & (C.side == 'less') & (C.definition_side == "V")]
            log('  The model on the less affected putamen '
                + ('does not meet the estimation criterion' if len(c_) else
                   "was not fitted (insufficient sample size)")
                + ': no conclusion.')
            outcome_cp[pop] = None
            continue
        pb = DZ['less_st']
        pos = position(me.ci_inf_month, me.ci_upp_month, pb["beta"])
        reproduced = bool(me.beta < 0 and me.p < ALPHA)
        outcome_cp[pop] = (reproduced, pos, bool(me.beta < 0))
        log(f"  Less affected putamen: beta = {me.beta_month:+.4f} points/month per unit of SBR [95% CI {me.ci_inf_month:+.4f}; {me.ci_upp_month:+.4f}], p = {fmt_p(me.p)} ({me.n_subj} subjects, {me.n_obsv} dates). Published: {pb['beta']:+.2f} [{pb['ci'][0]:+.2f}; {pb['ci'][1]:+.2f}], p = {pb['p']:g}. Sign {('concordant' if me.beta < 0 else 'opposite')} to the published one; the published value {SENTENCE_POS[pos]}.")
        log(f"     effect of SBR at time zero: {me.effect_main:+.2f} points per unit of SBR (p = {fmt_p(me.p_main)}); published {pb['princ']:+.2f}. "
            + ("The cross-sectional link between SBR and side score is there."
               if me.effect_main < 0 and me.p_main < ALPHA else
               "The cross-sectional link between SBR and side score is NOT nominal."))
        for name, r, key in (('more affected', pi_, 'more_st'), ('bilateral', bi, "bil_st")):
            if r is None:
                log(f'  Putamen {name}: model not fitted or not accepted.')
            else:
                log(f"  Putamen {name}: beta = {r.beta_month:+.4f} [95% CI {r.ci_inf_month:+.4f}; {r.ci_upp_month:+.4f}], p = {fmt_p(r.p)}. Published: {DZ[key]['beta']:+.2f}, p = {DZ[key]['p']:g}. Effect of SBR at time zero {r.effect_main:+.2f} (p = {fmt_p(r.p_main)}); published {DZ[key]['princ']:+.2f}.")
        more_nom = bool(pi_ is not None and pi_.beta < 0 and pi_.p < ALPHA)
        if reproduced and not more_nom:
            log("  Result: the published association IS REPRODUCED (negative and nominal "
                "interaction on the less affected putamen, not on the more affected one).")
        elif reproduced:
            log("  Result: the interaction on the less affected putamen is negative and nominal, "
                "but so is that on the more affected one: the side specificity is NOT reproduced.")
        else:
            log("  Result: the published association IS NOT REPRODUCED at the nominal level in this "
                "population; the published value " + SENTENCE_POS[pos] + ".")
        everyone_cp = Ca[Ca.cohort == pop]
        pos_ = everyone_cp[(everyone_cp.p < ALPHA) & (everyone_cp.beta > 0)]
        neg_ = everyone_cp[(everyone_cp.p < ALPHA) & (everyone_cp.beta < 0)]
        log(f'  Among all the {len(everyone_cp)} accepted models of this population: negative nominal interactions {len(neg_)}, POSITIVE nominal ones (with higher SBR, more rapid worsening) {len(pos_)}'
            + (" [" + "; ".join(f"{NAME_L[r.side].split(' ->')[0]}, {TREM[r.tremor]}, side {r.definition_side or 'n.a.'}, covariates {r.covariate}: {r.beta_month:+.3f}, p = {fmt_p(r.p)}"
                                for r in pd.concat([neg_, pos_]).itertuples()) + "]"
               if len(pos_) + len(neg_) else "")
            + f'; passing BH in their own block {int((everyone_cp.p_BH < ALPHA).sum())}.')
        others = Ca[(Ca.cohort == pop) & (Ca.side == 'less')
                   & ~((Ca.covariate == "A") & (Ca.definition_side == "V")
                       & (Ca.tremor == "st"))]
        if len(others):
            log(f'  Other {len(others)} specifications of the less affected putamen (tremor, side, covariates): beta/month from {others.beta_month.min():+.4f} to {others.beta_month.max():+.4f}; nominal and negative {int(((others.p < ALPHA) & (others.beta < 0)).sum())} out of {len(others)}; minimum p {fmt_p(others.p.min())}.')
    log()
    for tag in COHORTS:
        e = outcome_cp.get(tag)
        if e is None:
            sentence = 'the positive control cannot be assessed.'
        elif e[0]:
            sentence = ('the lateralised score carries in these data the signal described by the article; the models of the kinetic parameters on it are interpretable as such.')
        elif e[1] == 'within' and e[2]:
            sentence = ('the positive control is INCONCLUSIVE: the estimate has the sign of the published one and an interval that contains it, but is not distinguishable from zero. In this cohort the lateralised score does not have the precision to show the signal of the article; a null result of the kinetic parameters on it is to be read with the same reservation.')
        else:
            sentence = ('the lateralised score does NOT show in this cohort the signal described by the article (estimate of opposite sign, or interval that does not contain the published value with certainty): a null result of the kinetic parameters on it cannot be read as absence of a lateralised effect.')
        log(f'Joint reading, {LBLS[tag]} cohort: {sentence}')

    title('11. CHECKS RUN')
    for name, outcome, det in controls:
        log(f'  [{outcome}] {name}: {det}')
    log(f"  passed {sum((e == 'passed' for _, e, _ in controls))}, NOT verified {sum((e != 'passed' for _, e, _ in controls))}")

    title("12. WHAT THIS SCRIPT DOES NOT DO (deviations from the analysis of the article)")
    for row in (
        'it does not use the occipital-reference SBRs of DaTScan_Analysis.csv (they are not among the data): it uses the MIAKAT ones with cerebral white matter reference from the curated data cut;',
        'it does not include antidepressants among the covariates (the concomitant medication log is missing) and does not recompute the LEDD: it uses that of the curated data cut, in g/day;',
        'does not look for the closest clinical examination among different visits: uses that of the same visit of the curated data cut and applies the +/-3 month window with the ages of the curated data cut;',
        'age and duration at baseline are those of the project (baseline visit, diagnosis-enrolment, in years), not recomputed from the date of the first scan; time zero is the first usable date of the subject, not necessarily the first scan;',
        "the population is that of the two analysis cohorts (SAA-positive sporadic PD), not all the PD of PPMI; the 'mean' putamen of the article is here the bilateral ratio distributed in the curated data cut;",
        'the 5% asymmetry threshold is applied only in the positive control; it does not compare those who complete the follow-up with those who do not;',
        'it does not estimate the models with caudate and striatum, nor the analysis that excludes the subjects with clinical side and imaging side not aligned;',
        'it does not apply the Bonferroni correction of the article: uncorrected p and BH;',
        'in the outcomes of the kinetic parameters the side is fixed at baseline, not re-chosen at every date;',
        'does not adjust the models of the kinetic parameters for LEDD, run or centre and does not estimate the score on the visits of never-treated subjects only;',
        'does not check whether the missingness of the score depends on the kinetic parameters;',
        'does not use R/nlme: statsmodels MixedLM, REML, with the convergence criterion of the project.',
    ):
        log(f'  - {row}')

    # ------------------------------------------------------------ output -----
    UNITS_K = 'MDS-UPDRS III points per year per 1 SD of the log of the kinetic parameter'
    DESCR_SIDE = dict(SIDES, total='total score (no side)',
                      bilateral='bilateral putamen -> total score')
    for T_ in (K, C):
        T_['side_description'] = T_.side.map(DESCR_SIDE)
        T_['tremor'] = T_.tremor.map(TREM)
        T_['definition_side_description'] = T_.definition_side.map(DEF_SIDE).fillna("")
    K.insert(0, 'population', POP)
    K = having_cohort(K)
    caud = K.block == 'still_caudate'
    K['units_beta'] = np.where(caud, UNITS_CAUD, UNITS_K)
    K['state_pharmacological'] = np.where(
        caud, 'not relevant (imaging outcome)',
        np.where(K.block == 'still', 'OFF or untreated (updrs3_score of the curated data cut)',
                 'OFF or untreated (rule R3)'))
    K['relation_having_family_pubd'] = np.where(
        caud, 'member of the published family (version with the caudate, run_caudate): '
              + ('re-estimated here, identity verified' if PUBF is not None else
                 're-estimated here, identity NOT verified'),
        np.where(K.block == 'still', 'member of the published family',
                 'additional, specified at review'))
    K['analysis_added_in_review'] = ~K.block.isin(['still', 'still_caudate'])
    col_k = ['population', 'positivity_saa', 'cohort', 'n_cohort', 'block',
             'definition_side',
             'definition_side_description', 'rule_items', 'tremor', 'side',
             'side_description', 'outcome', 'kinetic', "formula", 'units_beta',
             'state_pharmacological', 'n_subj', 'n_obsv', 'n_multivisit', 'n_one_single',
             'obsv_by_subj_median', 'obsv_by_subj_q1', 'obsv_by_subj_q3', 'obsv_by_subj_max',
             'years_max', 'estimated', 'accepted', "beta", 'se', 'ci_inf', 'ci_upp', "p",
             'family_BH', 'n_family', "p_BH", 'n_family_widened', 'p_BH_widened',
             'relation_having_family_pubd', 'analysis_added_in_review',
             'slope_mean', 'effect_rel_pct', 'effect_main', 'dev_std_residual',
             'opt_chosen', 'n_optim_convergent', 'opt_reoptimisation', 'rounds', "loglike",
             'gain_reoptimisation', "grad_inf", "newton", 'stationary', "hess_def_neg",
             'singular', 'edge']
    K[col_k].to_csv(os.path.join(OUT, "111_models_kinetics.csv"), index=False)
    C.insert(0, 'population', POP)
    C = having_cohort(C).rename(columns={'n_cohort': 'n_population'})
    C['units_beta_month'] = 'MDS-UPDRS III points per month per unit of SBR (MIAKAT, CWM)'
    C['units_beta'] = 'MDS-UPDRS III points per year per unit of SBR (MIAKAT, CWM)'
    C['analysis_added_in_review'] = True
    col_c = ['population', 'positivity_saa', 'cohort', 'n_population', 'block', "covariate",
             'definition_side', 'definition_side_description', 'rule_items', 'tremor',
             'side', 'side_description', 'predictor', 'outcome',
             "formula", 'n_subj', 'n_obsv', 'obsv_by_subj_median', 'obsv_by_subj_max',
             'years_max', 'estimated', 'accepted', 'units_beta_month', 'beta_month', 'se_month',
             'ci_inf_month', 'ci_upp_month', 'units_beta', "beta", 'se', "p", 'family_BH',
             'n_family', "p_BH", 'beta_published', 'p_published', 'effect_main',
             'p_main', 'slope_mean', 'dev_std_residual',
             'analysis_added_in_review', 'opt_chosen', 'n_optim_convergent', 'rounds',
             "loglike", "newton", 'stationary', "hess_def_neg", 'singular', 'edge']
    C[col_c].to_csv(os.path.join(OUT, "111_check_positive.csv"), index=False)
    having_cohort(pd.DataFrame(rows_disp)).to_csv(os.path.join(OUT, "111_availability.csv"),
                                                index=False)
    having_cohort(pd.DataFrame(rows_conc)).to_csv(os.path.join(OUT, "111_agreement_sides.csv"),
                                                index=False)
    having_cohort(pd.DataFrame(rows_by)).to_csv(os.path.join(OUT, "111_scores_by_period.csv"),
                                               index=False)
    R = pd.DataFrame(rules, columns=['rule', 'step_of_article', 'choice', 'alternative'])
    R.insert(0, 'population', POP)
    R.insert(1, 'positivity_saa', POS['union'])
    R.insert(2, 'n_cohorts', "; ".join(f"{t}: {N_POP[t]}" for t in POPULATIONS))
    R.to_csv(os.path.join(OUT, "111_rules.csv"), index=False)
    log('\nSaved in outputs/: 111_lateralised_motor.txt, 111_rules.csv,')
    log('111_availability.csv, 111_agreement_sides.csv, 111_scores_by_period.csv,')
    log('111_models_kinetics.csv, 111_check_positive.csv')
    with open(os.path.join(OUT, "111_lateralised_motor.txt"), "w") as fh:
        fh.write("\n".join(rep))
    print(f'[run time: {time.time() - t_start:.0f} s]', flush=True)


if __name__ == "__main__":
    main()
