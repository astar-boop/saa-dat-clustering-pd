"""
110 - Laterality of DAT binding: less affected putamen and more affected putamen.

Why it exists. A reviewer writes that the striatal binding ratio (SBR) has been
"replaced by a different calculation, which predicts progression better", and
cites Dzialas et al., Ann Neurol 2025;98:120-135 (doi 10.1002/ana.27223,
PMID 40145540, PMC12174767). Read in full, the article does not introduce a new
calculation of the binding ratio: it uses the SBRs distributed by PPMI and
chooses, for each scan, the more and the less affected hemisphere with an
asymmetry index; it finds that it is the decline of the LESS affected putamen
that tracks motor worsening. It is a choice of region and of side, not of
quantification.

The manuscript uses, for the clustering, the putamen with the lower uptake (the
minimum of right and left: by construction the MORE affected) with the bilateral
caudate, and as longitudinal outcomes the bilateral putamen and caudate. The
"side-sensitive" reading had never been tried. This script adds it, as a
supplementary analysis and not as a replacement.

What it computes, in the two analysis cohorts of the current working directory:

  1. Definitions. More and less affected putamen (U) by uptake at baseline,
     side fixed once and for all; (V) by uptake at each visit, which is the
     rule of the article with an asymmetry threshold of zero; (D) by side of
     onset of symptoms (DOMSIDE): contralateral = more affected, ipsilateral
     = less affected; (V5) by uptake at each visit with the 5% asymmetry
     threshold of the article: scans with an asymmetry index within 5%
     (endpoint included, in exact arithmetic on hundredths) receive, for
     both sides, the mean of the two sides; as a sensitivity analysis,
     outside every family, scans with an index exactly equal to 5% are also
     treated as asymmetric (V5x). How often (U) and (D) indicate the same
     putamen, and how often the side with the lower uptake stays the same
     during follow-up.
  2. Mixed models kinetic parameter x time (the same as in 72/84/88: random
     intercept and slope, REML, age, sex and duration at baseline) with the
     less affected and the more affected putamen as outcomes, plus the formal
     contrast between the two sides (outcome = less affected - more affected).
     Bilateral putamen and caudate are re-estimated only to verify identity
     with the values that the pipeline has already produced.
  3. The technical run/centre ladder of 87 (M0, M4, M5, M6) on the same
     outcomes, for Fmax and maximum slope, 24-hour cohort.
  4. The battery at k = 2 of 25 against the Gaussian copula null, with the
     less affected putamen (and, separately, with the bilateral putaminal
     ratio as distributed in the curated data cut, MIA_PUTAMEN_BILAT) in place
     of the putamen with the lower uptake. The functions are those of 25,
     copied; identity is verified by reproducing on the reference branch the
     observed value and the first realisations of the null, and by recomputing
     IN FULL from the null saved by 25 the median, percentiles and p of all
     the statistics.
     For the silhouette at k = 2 only, primary cohort, a larger null
     (B = 4000) for the three putaminal variables: uncorrected p, outside the
     three reproducibility criteria of the rule.
  5. Feasibility of a lateralised motor score from the raw file of the
     MDS-UPDRS III (right and left items), without fitting models to it.

Estimation rule for the mixed models (the same as in 88): the four optimisers of
_fit.OTTIMIZZATORI start from the initial values of statsmodels; among those
that converge the highest log-likelihood is kept; from that point all four
restart until none gains more than 1e-10; the final point must have a Newton
decrement < 1e-4, a negative definite profiled Hessian and must not be
singular. The `converged` flag alone is not enough: an estimate that does not
meet the criterion is reported but flagged, and is left out of the summary
sentences.

Correction. Benjamini-Hochberg WITHIN COHORT, in three declared families:
"pubblicata" (the outcomes of the manuscript), "sostituzione" (the published
family with the bilateral putamen replaced by the lateralised outcome: same
size), "allargata" (the published one plus all the lateralised outcomes). The p
values of the clinical outcomes of the published family are read from
88_diagnostics_63.csv; the dopaminergic ones are re-estimated here and compared
with the pipeline.

Guards (each one stops, or writes NON VERIFICATO if it cannot run):
the coding of DOMSIDE is read from the dictionary of the xlsx file of the
curated data cut and, in any case, the uptake-onset concordance and the
clinical side-onset concordance must exceed 50% in each cohort; the stability
of the side during follow-up must exceed 50% and the left/right values of every
matchable scan are compared with the raw file DAT.csv; the null saved by 25 is
verified in full; the reference models are fitted and compared with the
pipeline BEFORE launching the rest, so that a failed identity stops at once.

What the script does NOT do: it does not recompute the SBRs (the values are
those of the MIAKAT pipeline of the curated data cut, cerebral white matter
reference; the article cited uses the SBRs of DaTScan_Analysis.csv, occipital
reference, which are not among these files); it does not fit models to the
contralateral motor score; it does not redo the curves over k = 2-10.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 scripts/110_dat_laterality.py
      (N_PROCESSES=n for the number of processes; the results do not depend on it)
Output: outputs/110_dat_laterality.txt, 110_laterality_definitions.csv,
        110_models_laterality.csv, 110_ladder_laterality.csv,
        110_battery_k2.csv, 110_battery_k2_nulls.csv,
        110_silhouette_k2_B4000.csv, 110_silhouette_k2_B4000_null.csv
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
import zlib                                                      # noqa: E402
from _legacy import crc32_legacy                                # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402
from scipy.spatial import cKDTree                                # noqa: E402
from scipy.spatial.distance import cdist                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS                                   # noqa: E402
from _paths import DATA                                       # noqa: E402
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
UPDRS3 = os.path.join(DATA, "MDS-UPDRS_Part_III_24Apr2026.csv")
DAT_RAW = os.path.join(DATA, "DAT.csv")
NAME_CURATED = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"
SHEET_DICTIONARY = "Data dictionary"
CACHE = os.path.join(OUT, "cache", "curated.parquet")
N_PROC = max(1, int(os.environ.get('N_PROCESSES', min(8, os.cpu_count() or 1))))

# --------------------------------------------------------------- thresholds -----
TOL_NEWTON = 1e-4       # Newton decrement, in log-likelihood units
TOL_SINGULAR = 1e-8
TOL_CORR = 0.999
TOL_GAIN = 1e-10
MAX_ROUNDS = 10
TOL_ID_BETA = 1e-6      # identity with the pipeline: coefficient and standard error
TOL_ID_P = 1e-5         # identity with the pipeline: p
ALPHA = 0.05
AI_THRESHOLD_PCT = 5       # asymmetry threshold (%) that the article uses as an alternative
THRESHOLD_GUARD = 0.5    # concordances and stability must exceed it in every cohort
TOL_RAW = 1e-9       # identity of the values with the raw file DAT.csv
N_MIN_COLLAPSE = 5       # runs and centres with fewer participants are merged into 'altro'
N_MIN_OBSV, N_MIN_MULTI = 120, 40      # thresholds of 72/84/88 for fitting a model

# battery: the same values as in 25_monte_carlo_nulls.py
SEED_B = 20260805
B_NULL = {"24h": 500, "150h": 300}
N_PROJ = 200
N_SPLIT_PS = 50
K_ID = 3                # realisations of the null regenerated for identity with 25
R_OBSV = 20              # independent streams for the two stochastic statistics
# silhouette at k = 2 only, primary cohort: larger null, own seeds
B_SIL = 4000
BLOCK_SIL = 100        # realisations per task; the result does not depend on it
SEED_SIL = 20261112
COHORT_SIL = "24h"
MORE_STRUCTURE = {"hopkins": True, "silhouette_k2": True,
                 'n_projections_multimodal': True, "dip_pmin_bonf": False,
                 "PS_k2": True, "jaccard_k2": True, 'ARI_algorithms_k2': True}
CRITERIA = ["PS_k2", "jaccard_k2", 'ARI_algorithms_k2']   # primary rule, k = 2
# statistics whose observed value depends on the random stream (the other two
# use only fixed seeds and are deterministic given the matrix)
DEPENDS_FROMTHE_STREAM = {"hopkins": True, "silhouette_k2": False,
                      'n_projections_multimodal': True, "dip_pmin_bonf": True,
                      "PS_k2": True, "jaccard_k2": True, 'ARI_algorithms_k2': False}

# Figures published by Dzialas et al., Ann Neurol 2025;98:120-135,
# doi 10.1002/ana.27223 (PMC12174767). They are literals from an external source,
# reported only to set them beside the counts computed here.
DZIALAS = {"n": 719, 'scans': 1981,
           'no_aligned_AI0': 0.24, 'no_aligned_AI5': 0.31,
           "AI_putamen": 0.17, 'AI_caudate': 0.09,
           'beta_less_without_tremor': -0.07, 'p_less_without_tremor': 0.002,
           'beta_less_having_tremor': -0.08, 'p_less_having_tremor': 0.003,
           'p_more_without_tremor': 0.93}

KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}
COV = ["age_bl", "SEX", "dur_bl"]
REF = {"MIA_PUTAMEN_BILAT": 'bilateral putamen',
       "MIA_CAUDATE_BILAT": 'bilateral caudate'}
DEFINITIONS = {
    "U": 'uptake at baseline, fixed side',
    "V": 'uptake at each visit (rule of the article, zero threshold)',
    "D": 'side of onset of symptoms (DOMSIDE)',
    "V5": (f'uptake at every visit, asymmetry threshold of {AI_THRESHOLD_PCT}% (rule of the article; within threshold the mean of the sides applies)'),
}
SIDES = {'less': 'less affected', 'more': 'more affected', "diff": 'less - more'}
SPEC = [
    ("M0", 'published', "{k}*years + age_bl + SEX + dur_bl"),
    ("M4", '+ run + centre', "{k}*years + age_bl + SEX + dur_bl + C(run) + C(site)"),
    ("M5", '+ run x years + centre x years',
     "{k}*years + age_bl + SEX + dur_bl + C(run)*years + C(site)*years"),
    ("M6", 'on kinetic parameter residualised for run',
     "{k}_res*years + age_bl + SEX + dur_bl"),
]
KIN_SCALE = ["Fmax", "SLOPE"]
NAME_KIN = {"SLOPE": 'maximum slope of the curve (SLOPEmax of the 24 h assay)'}

ITEMS_D = ["NP3RIGRU", "NP3RIGRL", "NP3FTAPR", "NP3HMOVR", "NP3PRSPR", "NP3TTAPR",
          "NP3LGAGR", "NP3PTRMR", "NP3KTRMR", "NP3RTARU", "NP3RTARL"]
ITEMS_S = ["NP3RIGLU", "NP3RIGLL", "NP3FTAPL", "NP3HMOVL", "NP3PRSPL", "NP3TTAPL",
          "NP3LGAGL", "NP3PTRML", "NP3KTRML", "NP3RTALU", "NP3RTALL"]

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


# ------------------------------------------------------------- statistics ----
def wilson(k, n, z=1.959963984540054):
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
    ordering = np.argsort(p, kind="mergesort")
    q = p[ordering] * n / np.arange(1, n + 1)
    q = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1.0)
    out = np.empty(n)
    out[ordering] = q
    return out


def p_mc(v, o, high=True):
    """One-tailed Monte Carlo p as an exact fraction: (numerator, denominator,
    p, standard error). Numerator = #{null >= observed} + 1 (lower tail if
    alto is false), denominator = B + 1; the Monte Carlo standard error is
    the binomial one, square root of p(1 - p)/B."""
    v = np.asarray(v, float)
    k = int(np.sum(v >= o) if high else np.sum(v <= o)) + 1
    den = len(v) + 1
    p = k / den
    return k, den, p, float(np.sqrt(p * (1 - p) / len(v)))


def within_threshold(s, d):
    """'Symmetric' scan according to the article: |S - D|/|S + D| <= 5%, endpoint
    included. The values of the curated data cut have two decimals: the
    comparison is made in integer arithmetic on hundredths, because in floating
    point an index exactly equal to 5% falls now on one side, now on the other.
    Returns (simmetrica, at_border): the second marks the scans with an index
    exactly equal to the threshold. False where a side is missing."""
    s, d = np.asarray(s, float), np.asarray(d, float)
    ok = np.isfinite(s) & np.isfinite(d)
    s0, d0 = np.where(ok, s, 0.0) * 100, np.where(ok, d, 0.0) * 100
    si, of = np.rint(s0).astype(np.int64), np.rint(d0).astype(np.int64)
    assert max(np.abs(s0 - si).max(), np.abs(d0 - of).max()) < 1e-6, \
        "the putaminal values are not to two decimals: the exact threshold is not applicable"
    a, b = 100 * np.abs(si - of), AI_THRESHOLD_PCT * np.abs(si + of)
    return ok & (a <= b), ok & (a == b) & (si != of)


def load_dictionary():
    """'Data dictionary' sheet of the xlsx file of the curated data cut, if reachable.
    Returns (table, path) or (None, None). A variable with several codes
    carries its name only on the first row: it is propagated to the following ones."""
    for fpath in (os.path.join(BASE, NAME_CURATED), os.path.join(DATA, NAME_CURATED)):
        if not os.path.exists(fpath):
            continue
        try:
            d = pd.read_excel(fpath, sheet_name=SHEET_DICTIONARY)
        except Exception:                     # the file is there, the sheet is missing or unreadable
            continue
        if not {"Variable", "Description", "Code", "Decode",
                "Original Variable(s)"} <= set(d.columns):
            continue
        d = d.copy()
        d['variable'] = d["Variable"].ffill()
        return d, fpath
    return None, None


def fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# -------------------------------------------- mixed models (rule of 88) ----
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
    in the parameter space of the optimiser (identical to 84 and 88)."""
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
    """Cascade from the initial values, choice by likelihood among the converged
    fits, restart from the chosen point until no optimiser gains any more."""
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
    row = {k: v for k, v in t.items() if k not in ('data', "formula", 'term', 'kind')}
    d = t['data']
    row.update(n_subj=int(d.PATNO.nunique()), n_obsv=int(len(d)),
                n_multivisit=int((d.groupby("PATNO").size() >= 2).sum()))
    res = procedure(t["formula"], d)
    if res is None:
        row.update(estimated=False, accepted=False)
        return row
    r = res["r"]
    a, b = t['term'].split(":")
    name = f"{a}:{b}" if f"{a}:{b}" in r.params.index else f"{b}:{a}"
    md = r.model
    p = pack(r)
    g = _score(md, p)
    H, ll = hess_profiled(md, p)
    eig = np.linalg.eigvalsh(H)
    try:
        newton = float(0.5 * g @ np.linalg.solve(H, -g))
    except np.linalg.LinAlgError:
        newton = np.nan
    cov_re = np.asarray(r.cov_re, dtype=float)
    scale = float(r.scale)
    v_int, v_pen = float(cov_re[0, 0]), float(cov_re[1, 1])
    corr = (float(cov_re[0, 1]) / np.sqrt(v_int * v_pen)
            if v_int > 0 and v_pen > 0 else np.nan)
    singular = bool(v_int < TOL_SINGULAR * scale or v_pen < TOL_SINGULAR * scale
                     or (np.isfinite(corr) and abs(corr) >= TOL_CORR))
    stationary = bool(np.isfinite(newton) and newton < TOL_NEWTON)
    hess_neg = bool(eig.max() < 0)
    be, se = float(r.params[name]), float(r.bse[name])
    row.update(
        estimated=True, beta=be, se=se, ci_inf=be - 1.96 * se, ci_upp=be + 1.96 * se,
        p=float(r.pvalues[name]), slope_mean=float(r.params["years"]),
        opt_chosen=res['opt_chosen'], n_optim_convergent=res["n_conv"],
        opt_reoptimisation=res['opt_reopt'], rounds=res['rounds'],
        loglike=ll, gain_reoptimisation=res['ll_final'] - res['ll_choice'],
        grad_inf=float(np.max(np.abs(g))), newton=newton, stationary=stationary,
        hess_def_neg=hess_neg, singular=singular,
        edge=bool(v_int == 0.0 or v_pen == 0.0),
        accepted=bool(res["n_conv"] >= 1 and stationary and hess_neg
                       and not singular))
    return row


# --------------------------------------- battery (functions identical to 25) ----
def copula_null(X, rng):
    """Gaussian copula null: empirical marginals and rank dependence."""
    p = X.shape[1]
    rho_s = stats.spearmanr(X).statistic
    R = 2 * np.sin(np.pi * np.atleast_2d(rho_s) / 6)
    w, V = np.linalg.eigh(R)
    R = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    R = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))
    Z = rng.multivariate_normal(np.zeros(p), R, size=len(X))
    U = stats.norm.cdf(Z)
    return np.column_stack([np.quantile(X[:, j], U[:, j]) for j in range(p)])


def hopkins(X, rng, m_frac=0.10, n_rep=30):
    n, p = X.shape
    m = max(5, int(m_frac * n))
    lo, hi = X.min(0), X.max(0)
    tree = cKDTree(X)
    vals = []
    for _ in range(n_rep):
        idx = rng.choice(n, m, replace=False)
        w = tree.query(X[idx], k=2)[0][:, 1]
        u = tree.query(rng.uniform(lo, hi, size=(m, p)), k=1)[0]
        vals.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(vals))


def n_multimodal(X, rng, n_proj=N_PROJ):
    import diptest
    c, pmin = 0, 1.0
    for _ in range(n_proj):
        a = rng.normal(size=X.shape[1])
        a /= np.linalg.norm(a)
        p = diptest.diptest(X @ a)[1]
        c += p < 0.05
        pmin = min(pmin, p)
    return int(c), float(pmin)


def ps_k2(X, rng, n_rep=N_SPLIT_PS):
    from sklearn.cluster import KMeans
    vals = []
    for _ in range(n_rep):
        perm = rng.permutation(len(X))
        a, b = perm[: len(X) // 2], perm[len(X) // 2:]
        ka = KMeans(2, n_init=10, random_state=int(rng.integers(1e6))).fit(X[a])
        lb = KMeans(2, n_init=10, random_state=int(rng.integers(1e6))).fit_predict(X[b])
        pred = np.argmin(cdist(X[b], ka.cluster_centers_), axis=1)
        worst = 1.0
        for c in range(2):
            m = lb == c
            nc = m.sum()
            if nc <= 1:
                worst = 0.0
                continue
            cnt = np.bincount(pred[m], minlength=2)
            worst = min(worst, (cnt * (cnt - 1)).sum() / (nc * (nc - 1)))
        vals.append(worst)
    return float(np.mean(vals))


def jaccard_k2(X, rng, Bb=25):
    from sklearn.cluster import KMeans
    ref = KMeans(2, n_init=15, random_state=SEED_B).fit_predict(X)
    best = {c: [] for c in range(2)}
    for _ in range(Bb):
        ib = rng.choice(len(X), len(X), replace=True)
        km = KMeans(2, n_init=5, random_state=int(rng.integers(1e6))).fit(X[ib])
        lab = np.argmin(cdist(X, km.cluster_centers_), axis=1)
        for c in range(2):
            rs = set(np.where(ref == c)[0])
            best[c].append(max(
                (len(rs & set(np.where(lab == cb)[0])) /
                 len(rs | set(np.where(lab == cb)[0])) for cb in range(2)),
                default=0.0))
    return float(np.mean([np.mean(best[c]) for c in range(2)]))


def ari_algos(X):
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    labs = {
        "kmeans": KMeans(2, n_init=20, random_state=SEED_B).fit_predict(X),
        "ward": AgglomerativeClustering(2, linkage="ward").fit_predict(X),
        "average": AgglomerativeClustering(2, linkage="average").fit_predict(X),
        "gmm": GaussianMixture(2, covariance_type="full", n_init=5,
                               random_state=SEED_B).fit_predict(X)}
    ks = list(labs)
    return float(np.mean([adjusted_rand_score(labs[ks[i]], labs[ks[j]])
                          for i in range(4) for j in range(i + 1, 4)]))


def battery(X, rng):
    """Same order of calls as in 25: the random stream is the same."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    nm, pmin = n_multimodal(X, rng)
    return {"hopkins": hopkins(X, rng),
            "silhouette_k2": float(silhouette_score(
                X, KMeans(2, n_init=15, random_state=SEED_B).fit_predict(X))),
            'n_projections_multimodal': nm,
            "dip_pmin_bonf": min(1.0, pmin * N_PROJ),
            "PS_k2": ps_k2(X, rng),
            "jaccard_k2": jaccard_k2(X, rng),
            'ARI_algorithms_k2': ari_algos(X)}


def streams_obsv(X, label):
    """Observed PS and Jaccard on R_OBSV independent streams of the algorithm."""
    streams = []
    for r in range(R_OBSV):
        g = np.random.default_rng([SEED_B, crc32_legacy(label.encode()), r])
        streams.append({"PS_k2": ps_k2(X, g), "jaccard_k2": jaccard_k2(X, g)})
    return streams


def batt(t):
    """Observed value and B realisations of the null, a single stream as in 25;
    then the two stochastic statistics repeated on R_OBSV independent streams."""
    X = t["X"]
    rng = np.random.default_rng(SEED_B)
    obsv = battery(X, rng)
    nulls = [battery(copula_null(X, rng), rng) for _ in range(t["B"])]
    return {'branch': t['branch'], 'cohort': t['cohort'], 'obsv': obsv, 'nulls': nulls,
            'streams': streams_obsv(X, t['label'])}


def identity_25(t):
    """Reference branch: observed value and first K_ID realisations of the null,
    on the stream of 25, to be compared with the files that 25 saved."""
    rng = np.random.default_rng(SEED_B)
    obsv = battery(t["X"], rng)
    leading = [battery(copula_null(t["X"], rng), rng) for _ in range(K_ID)]
    return {'cohort': t['cohort'], 'obsv': obsv, 'leading': leading}


def silhouette_k2(X):
    """The silhouette at k = 2 of the battery (same k-means, same seed)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(
        X, KMeans(2, n_init=15, random_state=SEED_B).fit_predict(X)))


def null_silhouette(t):
    """A block of realisations of the copula null for the silhouette only. The
    seed depends on branch, cohort and block: the result does not depend on the
    number of processes."""
    rng = np.random.default_rng([SEED_SIL, crc32_legacy(t['label'].encode()),
                                 t['block']])
    return {'branch': t['branch'], 'block': t['block'],
            'values': [silhouette_k2(copula_null(t["X"], rng)) for _ in range(t["n"])]}


def execute(t):
    return {'model': fitmodel, 'batt': batt, 'identity_25': identity_25,
            'streams': lambda q: {'branch': q['branch'], 'cohort': q['cohort'],
                                 'streams': streams_obsv(q["X"], q['label'])},
            "silhouette": null_silhouette}[t['kind']](t)


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


def build(c, kin_map, cur, code_left, code_dx):
    """Long table, like build() of 72/84/88, with the laterality columns."""
    base = c[["PATNO"] + list(kin_map)].rename(columns=kin_map)
    for k in kin_map.values():
        base[k] = np.log(base[k])
        base[k] = (base[k] - base[k].mean()) / base[k].std()
    bl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "duration_yrs",
                                    "DOMSIDE", "SITE", "MIA_PUTAMEN_L",
                                    "MIA_PUTAMEN_R"]].rename(
        columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl",
                 "MIA_PUTAMEN_L": "L_bl", "MIA_PUTAMEN_R": "R_bl"})
    base = base.merge(bl, on="PATNO", how="left", validate="one_to_one")
    # identity with the baseline values that the cohort actually used
    for mine, its in (("L_bl", "MIA_PUTAMEN_L"), ("R_bl", "MIA_PUTAMEN_R")):
        assert np.array_equal(base[mine].to_numpy(), c[its].to_numpy()), \
            f'{its} at baseline: the curated data cut does not match the cohort file'
    base['lower_bl'] = np.where(base.L_bl < base.R_bl, "S",
                                np.where(base.R_bl < base.L_bl, "D", "="))
    L = cur[cur.PATNO.isin(set(base.PATNO))][
        ["PATNO", "EVENT_ID", "YEAR", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]
        + list(REF)].copy()
    L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
    L = L[L.years.notna()].merge(base, on="PATNO", how="left")
    s, d = L.MIA_PUTAMEN_L, L.MIA_PUTAMEN_R
    for col in ("MIA_PUTAMEN_L", "MIA_PUTAMEN_R"):
        assert (L[col].notna() == L.MIA_PUTAMEN_BILAT.notna()).all(), \
            f'{col} and bilateral putamen are not present in the same scans'
    mean = (s + d) / 2
    # (U) side fixed by baseline uptake; in case of a tie the mean of the sides is used
    L['put_less_U'] = np.where(L.lower_bl == "S", d, np.where(L.lower_bl == "D", s, mean))
    L['put_more_U'] = np.where(L.lower_bl == "S", s, np.where(L.lower_bl == "D", d, mean))
    # (V) side chosen at each scan
    L['put_less_V'] = np.maximum(s, d)
    L['put_more_V'] = np.minimum(s, d)
    # (D) onset on the left: the right putamen is the more affected; on the right: the left
    L['put_less_D'] = np.where(L.DOMSIDE == code_left, s,
                               np.where(L.DOMSIDE == code_dx, d, np.nan))
    L['put_more_D'] = np.where(L.DOMSIDE == code_left, d,
                              np.where(L.DOMSIDE == code_dx, s, np.nan))
    # (V5) like V, but scans within the asymmetry threshold take the mean
    L['symmetric'], L['at_border'] = within_threshold(s, d)
    L['put_less_V5'] = np.where(L.symmetric, mean, np.maximum(s, d))
    L['put_more_V5'] = np.where(L.symmetric, mean, np.minimum(s, d))
    # (V5x) sensitivity: scans exactly on the threshold treated as asymmetric
    strict = L.symmetric & ~L.at_border
    L['put_less_V5x'] = np.where(strict, mean, np.maximum(s, d))
    L['put_more_V5x'] = np.where(strict, mean, np.minimum(s, d))
    L["put_diff_V5x"] = L.put_less_V5x - L.put_more_V5x
    for q in DEFINITIONS:
        L[f"put_diff_{q}"] = L[f'put_less_{q}'] - L[f'put_more_{q}']
    return base, L


def collapse(v, n_min=N_MIN_COLLAPSE):
    v = pd.Series(v).astype(str)
    counts = v.value_counts()
    return v.where(v.map(counts) >= n_min, 'other').to_numpy()


def job(L, outcome, k, formula, term, **meta):
    col = ["PATNO", "years", outcome] + sorted(
        set(COV + [k] + [x for x in ("run", "site") if x in formula]))
    m = L[col].dropna().reset_index(drop=True)
    if len(m) < N_MIN_OBSV or int((m.groupby("PATNO").size() >= 2).sum()) < N_MIN_MULTI:
        return None
    return dict(kind='model', data=m, formula=f'{outcome} ~ {formula}',
                term=term, outcome=outcome, kinetic=meta.pop('kinetic', k), **meta)


def matrix(c, columns, n_log=4):
    M = c[columns].to_numpy(float).copy()
    M[:, :n_log] = np.log(M[:, :n_log])
    assert np.isfinite(M).all(), f'missing values in the variables {columns}'
    return (M - M.mean(0)) / M.std(0)


def fmt_p(p, alpha=ALPHA):
    """Four decimals; if rounding would bring the value onto the threshold
    or to the other side of it (0.0499 or 0.05003 printed as 0.0500), decimals
    are added until the side of the threshold is the true one."""
    if pd.isna(p):
        return ""
    if p < 1e-4:
        return f"{p:.1e}"
    for dec in range(4, 12):
        t = f"{p:.{dec}f}"
        if p == alpha or (float(t) != alpha and (float(t) < alpha) == (p < alpha)):
            return t
    return repr(float(p))


def fmt_mc(k, den, p, se):
    """Monte Carlo p as an exact fraction with its standard error."""
    return f'{k}/{den} = {fmt_p(p)} (SE {se:.4f})'


# ================================================================= body =====
def main():
    t_start = time.time()
    rows_def = []

    def record(block, cohort, item, k, n, remark=""):
        lo, hi = wilson(k, n)
        rows_def.append(dict(block=block, cohort=cohort, item=item, k=k, n=n,
                              fraction=(k / n if n else np.nan), ci_inf=lo, ci_upp=hi,
                              remark=remark))

    # ---------------------------------------------------------- 0. working directory ----
    title('0. WORKING DIRECTORY, POPULATION, FILES')
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
    LBLS = {"24h": f'24 h (n = {len(c24)})', "150h": f'150 h (n = {len(c150)})'}
    log(f"working directory: {os.path.basename(os.path.normpath(BASE))} ({ROOT}); ONLY_SPORADIC = {('1' if ONLY else '0')}")
    log(f'population: {POP}')
    log(f"analysis cohorts: {LBLS['24h']}, {LBLS['150h']}; shared participants {len(p24 & p150)}, distinct {len(p24 | p150)}")
    for name in ("analysis_complete_cases.csv", "08_dataset_150h.csv"):
        log(f'  {name}: sha256 {fingerprint(os.path.join(OUT, name))[:16]}...')
    log(f'  cached curated data cut: {len(cur)} rows (visits), {cur.PATNO.nunique()} subjects')
    log("DAT measures: MIA_* variables of the curated data cut (their description is read from the")
    log("dictionary in section 2). The 'baseline' scan is the BL row of the curated data cut.")
    log("Time = YEAR. 'Bilateral putamen' = MIA_PUTAMEN_BILAT, the bilateral putaminal")
    log("ratio as distributed in the curated data cut: it is not the mean of the two sides (section 2).")

    # ------------------------------------------------- 1. the article cited ----
    title("1. WHAT THE ARTICLE CITED BY THE REVIEWER COMPUTES")
    log('Dzialas et al., Ann Neurol 2025;98:120-135, doi 10.1002/ana.27223. The figures in')
    log('this section are PUBLISHED, not computed here.')
    log(f"  {DZIALAS['n']} PD of PPMI, {DZIALAS['scans']} scans. The SBRs are those")
    log('  distributed by PPMI (DaTScan_Analysis.csv, occipital reference): no')
    log('  new computation of the binding ratio. For each region and each date they compute')
    log("  the asymmetry index (L - R)/(L + R) and define the more and the")
    log('  less affected hemisphere (thresholds 0% and 5%; within threshold the mean of the sides holds).')
    log("  The DAT is the PREDICTOR; the outcome is the MDS-UPDRS III of the contralateral")
    log('  body side, in mixed models with a time x DAT interaction.')
    log(f"  Result: only the less affected putamen (beta = {DZIALAS['beta_less_without_tremor']}, p = {DZIALAS['p_less_without_tremor']} without tremor; {DZIALAS['beta_less_having_tremor']}, p = {DZIALAS['p_less_having_tremor']} with tremor);")
    log(f"  more affected putamen p = {DZIALAS['p_more_without_tremor']}. Clinical side and imaging side not aligned in {100 * DZIALAS['no_aligned_AI0']:.0f}% (threshold 0) and in {100 * DZIALAS['no_aligned_AI5']:.0f}% (threshold 5%).")
    log("In the manuscript the DAT is instead a clustering variable and an OUTCOME of the")
    log("kinetic parameter x time models: what can be transposed is the choice of the putamen.")

    # ------------------------------------------------------ 2. definitions ----
    title("2. MORE AND LESS AFFECTED PUTAMEN: DEFINITIONS, CONCORDANCE, GUARDS")
    log("Unit: subjects, at the baseline scan, except where 'scans' is written.")
    # coding of the side of onset: from the dictionary of the curated data cut, if reachable
    dic, f_dic = load_dictionary()
    codes_dom = sorted(float(v) for v in cur.DOMSIDE.dropna().unique())
    NAMES_DOM = {}
    if dic is not None:
        rd = dic[dic.variable == "DOMSIDE"]
        for code, dec in zip(rd.Code, rd.Decode):
            try:
                NAMES_DOM[float(str(code).strip())] = str(dec).strip()
            except ValueError:
                continue                  # non-numeric code ('.' = unknown)
        by_name = {v.lower(): k for k, v in NAMES_DOM.items()}
        assert "left" in by_name and "right" in by_name, \
            'dictionary of the curated data cut: DOMSIDE does not have the entries Left and Right'
        assert set(codes_dom) <= set(NAMES_DOM), \
            (f'DOMSIDE has codes in the data that are missing from the dictionary: {sorted(set(codes_dom) - set(NAMES_DOM))}')
        CODE_LEFT, CODE_DX = by_name["left"], by_name["right"]
        log(f'''DOMSIDE coding READ from the dictionary of the curated data cut '\n            f'({os.path.basename(f_dic)}, sheet '{SHEET_DICTIONARY}'; description: '\n            f'"{' '.join(str(rd.Description.iloc[0]).split())}"): '''
            + ", ".join(f"{k:g} = {v}" for k, v in sorted(NAMES_DOM.items()))
            + f'. Hence onset on the left = {CODE_LEFT:g}, on the right = {CODE_DX:g}.')
        for var in ("MIA_PUTAMEN_L", "MIA_PUTAMEN_R", "MIA_PUTAMEN_BILAT"):
            rv = dic[dic.Variable == var]
            log(f'  {var} in the dictionary: '
                + (f'''"{' '.join(str(rv.Description.iloc[0]).split())}", source variable '
                   f'{rv['Original Variable(s)'].iloc[0]}''' if len(rv)
                   else 'entry missing'))
    else:
        CODE_LEFT, CODE_DX = 1.0, 2.0
        log(f"DOMSIDE coding NOT VERIFIED against the dictionary: {NAME_CURATED} cannot be reached either in the working directory or in the raw data folder. The default value of the script is used (left = {CODE_LEFT:g}, right = {CODE_DX:g}); codes present in the data: {', '.join((f'{v:g}' for v in codes_dom))}.")
    log(f'In any case the coding is subjected to two guards per cohort: the concordance between the lower-uptake putamen and the side of onset, and that between the clinical side (MDS-UPDRS III) and the side of onset, must exceed {100 * THRESHOLD_GUARD:.0f}%; otherwise the script stops.')
    log("Onset on the left -> more affected putamen expected on the right (contralateral).")

    # lateralised motor examination: needed at once, for the guard on the coding
    u = pd.read_csv(UPDRS3, low_memory=False)
    u["PATNO"] = u["PATNO"].astype("int64")
    u['row'] = np.arange(len(u))
    miss = [v for v in ITEMS_D + ITEMS_S if v not in u.columns]
    assert not miss, f'lateralised items missing from the MDS-UPDRS III file: {miss}'
    items = u[ITEMS_D + ITEMS_S].apply(pd.to_numeric, errors="coerce")
    items = items.where((items >= 0) & (items <= 4))            # 101 = not assessable
    u["dx"] = items[ITEMS_D].sum(axis=1, min_count=len(ITEMS_D))
    u["sx"] = items[ITEMS_S].sum(axis=1, min_count=len(ITEMS_S))
    u["on"] = (u.PDSTATE == "ON").astype(int)

    def motor(base):
        """Counts of the lateralised motor score (printed in section 8)."""
        b = u[(u.EVENT_ID == "BL") & u.PATNO.isin(set(base.PATNO))].sort_values(
            ["PATNO", "on", 'row'])
        doubles = int((b.groupby("PATNO").size() > 1).sum())
        b = b.drop_duplicates("PATNO", keep="first")          # the non-ON row, then the first
        m = base.merge(b[["PATNO", "dx", "sx", "PDSTATE"]], on="PATNO", how="left",
                       validate="one_to_one")
        compl = (m.dx.notna() & m.sx.notna()).to_numpy()
        clin = np.where(m.dx > m.sx, "D", np.where(m.sx > m.dx, "S", "="))
        ok = compl & (clin != "=")
        dom = m.DOMSIDE.to_numpy()
        s1 = ok & np.isin(dom, [CODE_LEFT, CODE_DX])
        k1 = int((((clin == "S") & (dom == CODE_LEFT)) | ((clin == "D") & (dom == CODE_DX)))[s1].sum())
        s2 = ok & (m.lower_bl != "=").to_numpy()
        k2 = int((((clin == "S") & (m.lower_bl == "D").to_numpy())
                  | ((clin == "D") & (m.lower_bl == "S").to_numpy()))[s2].sum())
        # longitudinal availability: examinations at the visits with YEAR, participant x visit
        vis = cur[cur.PATNO.isin(set(base.PATNO)) & cur.YEAR.notna()][["PATNO", "EVENT_ID"]]
        e = u.merge(vis, on=["PATNO", "EVENT_ID"], how="inner").sort_values(
            ["PATNO", "EVENT_ID", "on", 'row']).drop_duplicates(["PATNO", "EVENT_ID"],
                                                               keep="first")
        e_ok = e[e.dx.notna() & e.sx.notna()]
        no_on = e_ok[e_ok.on == 0]
        without = no_on[no_on.PDSTATE.isna()]
        treat = None
        if "PDTRTMNT" in u.columns:
            tr = pd.to_numeric(without.PDTRTMNT, errors="coerce")
            treat = (int((tr == 0).sum()), int((tr == 1).sum()), int(tr.isna().sum()))
        return dict(n=len(m), having_exam=int(m.PATNO.isin(set(b.PATNO)).sum()), doubles=doubles,
                    compl=int(compl.sum()), on_bl=int((m.PDSTATE == "ON").sum()),
                    clin_D=int((clin[compl] == "D").sum()),
                    clin_S=int((clin[compl] == "S").sum()),
                    clin_ties=int((clin[compl] == "=").sum()),
                    k1=k1, n1=int(s1.sum()), k2=k2, n2=int(s2.sum()),
                    n_e=len(e), n_eok=len(e_ok), n_no_on=len(no_on),
                    off=int((no_on.PDSTATE == "OFF").sum()), without_state=len(without),
                    treat=treat, multi=int((no_on.groupby("PATNO").size() >= 2).sum()))

    # raw DAT file: benchmark for the left/right values
    COL_G = {"PUTAMEN_L_REF_CWM": "MIA_PUTAMEN_L", "PUTAMEN_R_REF_CWM": "MIA_PUTAMEN_R",
             "PUTAMEN_REF_CWM": "MIA_PUTAMEN_BILAT"}
    raw, n_dup_g = None, 0
    if os.path.exists(DAT_RAW):
        raw = pd.read_csv(DAT_RAW, low_memory=False)
        raw["PATNO"] = raw["PATNO"].astype("int64")
        miss = [v for v in COL_G if v not in raw.columns]
        assert not miss, f'putaminal columns missing from DAT.csv: {miss}'
        for col in COL_G:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        dup = raw.duplicated(["PATNO", "EVENT_ID"], keep=False)
        n_dup_g = int(dup.sum())
        raw = raw.loc[~dup, ["PATNO", "EVENT_ID"] + list(COL_G)]

    def check_raw(L, tag):
        """Left, right and bilateral values of every matchable scan against
        DAT.csv. Key: PATNO and EVENT_ID; the baseline of the curated data cut,
        which has no row of the same name in the raw file, is looked up in the
        screening scan (SC) of the same participant. A single differing value
        stops the script."""
        sc = L.loc[L.MIA_PUTAMEN_L.notna() & L.MIA_PUTAMEN_R.notna(),
                   ["PATNO", "EVENT_ID", "years"] + list(COL_G.values())].reset_index(drop=True)
        followed = (sc.years > 0).to_numpy()
        if raw is None:
            log(f'  left/right values against the raw file: NOT VERIFIED ({os.path.basename(DAT_RAW)} missing from the raw data folder); scans not verified {len(sc)}, of which {int(followed.sum())} after baseline.')
            return
        j = sc.merge(raw, on=["PATNO", "EVENT_ID"], how="left", validate="one_to_one",
                     indicator="_a")
        direct = (j._a == "both").to_numpy()
        gsc = raw[raw.EVENT_ID == "SC"].drop(columns="EVENT_ID")
        jsc = sc[["PATNO"]].merge(gsc, on="PATNO", how="left", validate="many_to_one",
                                  indicator="_b")
        via_sc = (~direct) & (sc.EVENT_ID == "BL").to_numpy() & (jsc._b == "both").to_numpy()
        for col in COL_G:
            j[col] = np.where(via_sc, jsc[col].to_numpy(), j[col].to_numpy())
        having_values = ((direct | via_sc) & j.PUTAMEN_L_REF_CWM.notna().to_numpy()
                      & j.PUTAMEN_R_REF_CWM.notna().to_numpy())
        v = j[having_values]
        for col_g, col_c in COL_G.items():
            both = v[col_g].notna()
            departure = (v.loc[both, col_g] - v.loc[both, col_c]).abs()
            n_div = int((departure > TOL_RAW).sum())
            assert n_div == 0, \
                (f'{tag} cohort: {col_c} of the curated data cut differs from {col_g} of DAT.csv in {n_div} matched scans of {int(both.sum())} (maximum discrepancy {departure.max():.3g}): the left/right values are not those of the raw file')
        n_bil = int(v.PUTAMEN_REF_CWM.notna().sum())
        n_asym = int((v.MIA_PUTAMEN_L != v.MIA_PUTAMEN_R).sum())
        no_row = ~(direct | via_sc)
        log(f'  left/right values against {os.path.basename(DAT_RAW)} (unit: scans; key PATNO + EVENT_ID, baseline looked up in the screening SC): matched and IDENTICAL {share(int(having_values.sum()), len(sc))} ({int((having_values & direct).sum())} with the same EVENT_ID, {int((having_values & via_sc).sum())} baselines via SC; in {n_asym} the two sides differ, and a swap would be visible there; bilateral ratio identical in {n_bil} of {n_bil} with a raw value).')
        log(f'    after baseline: verified {share(int((having_values & followed).sum()), int(followed.sum()))}; at baseline: {share(int((having_values & ~followed).sum()), int((~followed).sum()))}. NOT verifiable {len(sc) - int(having_values.sum())}: without a row in the raw file {int(no_row.sum())} ({int((no_row & ~followed).sum())} baselines, {int((no_row & followed).sum())} after baseline), with a row but without values {int(((direct | via_sc) & ~having_values).sum())}.'
            + (f" Repeated keys in DAT.csv, excluded from the matching: {n_dup_g} rows."
               if n_dup_g else ""))
        record('raw file', tag, 'scans with L and R identical to DAT.csv',
                 int(having_values.sum()), len(sc),
                 'unit: scans; the others cannot be matched (NOT verified)')
        record('raw file', tag, 'same, scans after baseline',
                 int((having_values & followed).sum()), int(followed.sum()), 'unit: scans')

    TAB, BASES, MOT = {}, {}, {}
    for tag, (c, kin) in COHORTS.items():
        base, L = build(c, kin, cur, CODE_LEFT, CODE_DX)
        TAB[tag], BASES[tag] = L, base
        for col, f in (("dat_putamen_low", np.minimum), ("dat_putamen_high", np.maximum)):
            if col in c.columns:
                assert np.array_equal(c[col].to_numpy(),
                                      f(c.MIA_PUTAMEN_L, c.MIA_PUTAMEN_R).to_numpy()), \
                    f"{col} of the cohort file is not the min/max of the two sides"
        # the 'more affected' of definition V is, at baseline, the clustering variable
        bl_r = L[L.EVENT_ID == "BL"]
        assert len(bl_r) == len(base) and np.array_equal(
            bl_r.put_more_V.to_numpy(), np.minimum(bl_r.L_bl, bl_r.R_bl).to_numpy()), \
            "at baseline the more affected putamen (V) is not the lower-uptake putamen"
        n = len(base)
        ai = (np.abs(base.L_bl - base.R_bl) / np.abs(base.L_bl + base.R_bl)).to_numpy()
        ties = int((base.lower_bl == "=").sum())
        sim_bl, _ = within_threshold(base.L_bl, base.R_bl)
        within = int(sim_bl.sum())
        dom = base.DOMSIDE.to_numpy()
        defined = np.isin(dom, [CODE_LEFT, CODE_DX])
        other = pd.notna(dom) & ~defined
        log(f'\n[{POP}; {LBLS[tag]} cohort]')
        log(f"  lower-uptake putamen: left {int((base.lower_bl == 'S').sum())}, right {int((base.lower_bl == 'D').sum())}, tie {ties}")
        log(f"  putaminal asymmetry index |L - R|/|L + R|: median {np.median(ai):.3f}, mean {np.mean(ai):.3f} (published by Dzialas, mean: {DZIALAS['AI_putamen']}); within the threshold of {AI_THRESHOLD_PCT}%, endpoint included: {share(within, n)}")
        log(f"  baseline SBR, mean (SD): less affected {np.maximum(base.L_bl, base.R_bl).mean():.3f} "
            f"({np.maximum(base.L_bl, base.R_bl).std():.3f}), more affected "
            f"{np.minimum(base.L_bl, base.R_bl).mean():.3f} "
            f"({np.minimum(base.L_bl, base.R_bl).std():.3f})")
        log(f'  side of onset: left {int((dom == CODE_LEFT).sum())}, right {int((dom == CODE_DX).sum())}, other code {int(other.sum())}'
            + (" (" + ", ".join(f"{NAMES_DOM.get(float(v), '?')} {int((dom == v).sum())}"
                                for v in sorted(set(dom[other]))) + ")" if other.any() else "")
            + f', missing {int(pd.isna(dom).sum())}')
        record('definitions', tag, 'tie in uptake at baseline', ties, n)
        record('definitions', tag, f'asymmetry within {AI_THRESHOLD_PCT}%', within, n)
        record('definitions', tag, 'side of onset defined (left or right)',
                 int(defined.sum()), n)
        bothsides = defined & (base.lower_bl != "=").to_numpy()
        concordant = (((dom == CODE_LEFT) & (base.lower_bl == "D").to_numpy())
                    | ((dom == CODE_DX) & (base.lower_bl == "S").to_numpy()))
        k, m = int(concordant[bothsides].sum()), int(bothsides.sum())
        lo, hi = wilson(k, m)
        log(f'  the two definitions indicate the same putamen (lower uptake = contralateral to onset): {share(k, m)} [Wilson 95% {100 * lo:.1f}; {100 * hi:.1f}]; not aligned {share(m - k, m)}')
        assert m > 0 and k / m > THRESHOLD_GUARD, \
            (f'{tag} cohort: the lower-uptake putamen is contralateral to the side of onset only in {share(k, m)}: the coding of DOMSIDE (left = {CODE_LEFT:g}, right = {CODE_DX:g}) or the sides of the putamen are inverted')
        record('concordance', tag, 'lower uptake = contralateral to onset', k, m,
                 'subjects with lateralised onset and without ties')
        for name, sel in ((f'asymmetry > {AI_THRESHOLD_PCT}%', ~sim_bl),
                          (f'asymmetry <= {AI_THRESHOLD_PCT}%', sim_bl)):
            s2 = bothsides & sel
            k2, m2 = int(concordant[s2].sum()), int(s2.sum())
            log(f'    with {name}: {share(k2, m2)}')
            record('concordance', tag, f'ditto, {name}', k2, m2)
        log(f"    (Dzialas: not aligned {100 * DZIALAS['no_aligned_AI0']:.0f}%, but the clinical side there is the one with the higher UPDRS III score, not the onset)")
        # guard on the coding, second part: clinical side against side of onset
        MOT[tag] = mo = motor(base)
        assert mo["n1"] > 0 and mo["k1"] / mo["n1"] > THRESHOLD_GUARD, \
            (f"{tag} cohort: the clinical side (MDS-UPDRS III) coincides with the side of onset only in {share(mo['k1'], mo['n1'])}: the coding of DOMSIDE (left = {CODE_LEFT:g}, right = {CODE_DX:g}) is inverted")
        log(f"  guards on the coding of the side of onset PASSED (> {100 * THRESHOLD_GUARD:.0f}%): uptake-onset {share(k, m)}, clinical side-onset {share(mo['k1'], mo['n1'])}")
        # stability of the side during follow-up: unit = scans after baseline
        f = L[(L.years > 0) & L.MIA_PUTAMEN_L.notna()]
        lower = np.where(f.MIA_PUTAMEN_L < f.MIA_PUTAMEN_R, "S",
                         np.where(f.MIA_PUTAMEN_R < f.MIA_PUTAMEN_L, "D", "="))
        valid = (lower != "=") & (f.lower_bl != "=").to_numpy()
        k3, m3 = int((lower[valid] == f.lower_bl.to_numpy()[valid]).sum()), int(valid.sum())
        subj = f[valid].assign(same=lower[valid] == f.lower_bl.to_numpy()[valid])
        changes = int((~subj.groupby("PATNO").same.all()).sum())
        log(f'  follow-up scans in which the lower-uptake side is that of baseline: {share(k3, m3)} (unit: scans; {subj.PATNO.nunique()} subjects, {changes} with at least one inversion)')
        assert m3 > 0 and k3 / m3 > THRESHOLD_GUARD, \
            (f'{tag} cohort: the lower-uptake side remains that of baseline only in {share(k3, m3)} of the follow-up scans: the left and right values after baseline are probably swapped')
        log(f'  guard on the stability of the side in the follow-up PASSED (> {100 * THRESHOLD_GUARD:.0f}%)')
        record('stability', tag, 'side of lower uptake unchanged during follow-up', k3, m3,
                 'unit: scans after baseline, ties excluded')
        record('stability', tag, 'subjects with at least one side inversion', changes,
                 int(subj.PATNO.nunique()))
        check_raw(L, tag)
        # 5% threshold: how many scans are 'symmetric' and take the mean
        having_sides = L.MIA_PUTAMEN_L.notna().to_numpy()
        n_sim, n_sc = int(L.symmetric.sum()), int(having_sides.sum())
        den_neg = int(((L.MIA_PUTAMEN_L + L.MIA_PUTAMEN_R) <= 0).sum())
        n_conf = int(L.at_border.sum())
        log(f'  definition V5: scans with asymmetry index within {AI_THRESHOLD_PCT}% (endpoint included), which receive the mean of the two sides: {share(n_sim, n_sc)} (unit: scans); of these, with index EXACTLY equal to {AI_THRESHOLD_PCT}%: {n_conf}'
            + (' (section 5 reports what changes when they are treated as asymmetric)'
               if n_conf else "")
            + f"; scans with L + R <= 0, in which the index does not have the usual meaning and its "
              f"absolute value is taken: {den_neg}")
        record('definitions', tag, f'scans within {AI_THRESHOLD_PCT}% asymmetry',
                 n_sim, n_sc, 'unit: scans; definition V5')
        # the bilateral ratio of the curated data cut is not the mean of the two sides
        sb = (L.MIA_PUTAMEN_BILAT - (L.MIA_PUTAMEN_L + L.MIA_PUTAMEN_R) / 2).abs()[having_sides]
        log(f'  MIA_PUTAMEN_BILAT (bilateral putaminal ratio as distributed in the curated data cut) against the arithmetic mean of the two sides: equal within 1e-9 in {share(int((sb < 1e-09).sum()), len(sb))} of the scans; median absolute discrepancy {sb.median():.3f}, maximum {sb.max():.3f} SBR units')
        # interval between scan and lumbar puncture at baseline
        if {"age_datscan", "age_LP"} <= set(cur.columns):
            blc = cur[(cur.EVENT_ID == "BL") & cur.PATNO.isin(set(base.PATNO))]
            dd = (blc.age_datscan - blc.age_LP).dropna()
            beyond = int((dd.abs() > 0.5).sum())
            log(f'  scan and lumbar puncture at the baseline row (age at scan - age at puncture, years; unit: subjects): {len(dd)} of {len(blc)} with both ages; median {dd.median():+.3f}; more than 0.5 years apart: {beyond} of {len(dd)} with both ages')
            record('times', tag, 'scan more than 0.5 years from the lumbar puncture',
                     beyond, len(dd), 'subjects with both ages at the baseline row')
        else:
            log('  scan and lumbar puncture at baseline: NOT computed (age_datscan or age_LP '
                'missing from the curated data cut in cache)')
    log("\nMethodological note (NOT computed here; it is a property of the definitions, not")
    log("a result). The definition by uptake chooses the side by looking at the outcome:")
    log("at baseline the 'less affected' putamen is the larger of two noisy measures, and its")
    log("mean slope may be affected by regression towards the mean (definition")
    log("U), whereas choosing the larger at each visit (V, V5) selects upwards at")
    log("each visit. The definition by side of onset (D) does not look at the outcome; it covers")
    log('only the subjects with lateralised onset (counts above).')
    log("The 'more affected' putamen of definition V is the minimum of the two sides at each")
    log("scan: it is the clustering variable of the manuscript (lower-uptake")
    log("putamen) followed over time; at baseline the identity is verified (assertion).")

    # ------------------------------------------------------- tasks -----
    tasks = []
    for tag, (c, kin) in COHORTS.items():
        L = TAB[tag]
        for k in kin.values():
            for outcome in REF:
                tasks.append(job(L, outcome, k, f"{k}*years + " + " + ".join(COV),
                                       f"{k}:years", block='reference', cohort=tag,
                                       definition="", side='bilateral', spec="M0"))
            for q in DEFINITIONS:
                for side in SIDES:
                    tasks.append(job(
                        L, f'put_{side}_{q}', k, f"{k}*years + " + " + ".join(COV),
                        f"{k}:years", block='laterality', cohort=tag, definition=q,
                        side=side, spec="M0"))
        if L.at_border.any():        # sensitivity at the edge of the threshold: outside the families
            for k in kin.values():
                for side in SIDES:
                    tasks.append(job(
                        L, f'put_{side}_V5x', k, f"{k}*years + " + " + ".join(COV),
                        f"{k}:years", block='border', cohort=tag, definition="V5x",
                        side=side, spec="M0"))
    # technical ladder: 24-hour cohort, run and centre built as in 87
    saa = pd.read_csv(SAA, low_memory=False)
    saa["PATNO"] = saa["PATNO"].astype("int64")
    saa['row'] = np.arange(len(saa))
    s237 = saa[(saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")
               & saa.PATNO.isin(p24)].sort_values(["PATNO", 'row'])
    earliest = s237.pipe(row_of_kinetics, OUT).drop_duplicates("PATNO", keep="first")     # the row of the kinetic parameters, as in 87
    assert set(earliest.PATNO) == p24, 'subjects of the 24 h cohort without a baseline assay'
    b24 = BASES["24h"].merge(earliest[["PATNO", "RUNDATE"]].rename(columns={"RUNDATE": "run"}),
                            on="PATNO", how="left", validate="one_to_one")
    n_run_different = None
    if "RUNDATE" in c24.columns:
        n_run_different = int((b24.set_index("PATNO").run
                             != c24.set_index("PATNO").RUNDATE.reindex(b24.PATNO).values).sum())
    b24["run"] = collapse(b24.run)
    b24["site"] = collapse(b24.SITE)
    for k in KIN_SCALE:
        med = b24.groupby("run")[k].transform("mean")
        r = b24[k] - med + b24[k].mean()
        b24[k + "_res"] = (r - r.mean()) / r.std()
    L24s = TAB["24h"].merge(b24[["PATNO", "run", "site"] + [k + "_res" for k in KIN_SCALE]],
                            on="PATNO", how="left", validate="many_to_one")
    outcomes_scale = ["MIA_PUTAMEN_BILAT"] + [f'put_{side}_{q}' for q in DEFINITIONS
                                           for side in ('less', 'more')]
    for outcome in outcomes_scale:
        for k in KIN_SCALE:
            for spec, descr, tpl in SPEC:
                var = f"{k}_res" if spec == "M6" else k
                q = outcome.rsplit("_", 1)[1] if outcome.startswith("put_") else ""
                # in 87 the rows without run or centre are left out of EVERY specification
                tasks.append(job(
                    L24s.dropna(subset=["run", "site"]), outcome, var, tpl.format(k=k),
                    f"{var}:years", block='scale', cohort="24h", definition=q,
                    side=(outcome.split("_")[1] if q else 'bilateral'), spec=spec,
                    kinetic=k, description=descr))
    n_skipped = sum(t is None for t in tasks)
    tasks = [t for t in tasks if t is not None]
    # the models that the pipeline has already produced: fitted and verified first
    firsts = [i for i, t in enumerate(tasks)
             if t['block'] == 'reference' or (t['block'] == 'scale' and t['outcome'] in REF)]
    rest = [i for i in range(len(tasks)) if i not in set(firsts)]

    # battery: branches and matrices
    npz = np.load(os.path.join(OUT, "features.npz"))
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy()), \
        'features.npz and analysis_complete_cases.csv do not have the same subjects in the same order'
    c24m = c24.assign(put_high=np.maximum(c24.MIA_PUTAMEN_L, c24.MIA_PUTAMEN_R),
                      put_lower=np.minimum(c24.MIA_PUTAMEN_L, c24.MIA_PUTAMEN_R))
    c150m = c150.assign(put_high=np.maximum(c150.MIA_PUTAMEN_L, c150.MIA_PUTAMEN_R),
                        put_lower=np.minimum(c150.MIA_PUTAMEN_L, c150.MIA_PUTAMEN_R))
    F24 = ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax"]
    F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE"]
    BRANCHES = {'reference': ('put_lower', 'putamen with lower uptake (manuscript)'),
            'less_affected': ('put_high', 'less affected putamen (higher uptake)'),
            'bilateral': ("MIA_PUTAMEN_BILAT", 'bilateral putaminal ratio as distributed in the curated data cut (MIA_PUTAMEN_BILAT)')}
    X = {}
    for branch, (col, _) in BRANCHES.items():
        X[(branch, "24h")] = matrix(c24m, F24 + [col, "MIA_CAUDATE_BILAT", "age_bl"])
        X[(branch, "150h")] = matrix(c150m, F150 + [col, "MIA_CAUDATE_BILAT", "age_bl"])
    assert np.allclose(X[('reference', "24h")], npz["F_primary"], rtol=0, atol=1e-10), \
        'the matrix of the reference arm does not coincide with F_primary of 02'
    comp_b = [dict(kind='batt', branch=branch, cohort=tag, X=X[(branch, tag)],
                   B=B_NULL[tag], label=f'{branch}|{tag}')
              for branch in ('less_affected', 'bilateral') for tag in COHORTS]
    comp_id = [dict(kind='identity_25', cohort=tag, X=X[('reference', tag)])
               for tag in COHORTS]
    comp_f = [dict(kind='streams', branch='reference', cohort=tag, X=X[('reference', tag)],
                   label=f'reference|{tag}') for tag in COHORTS]
    assert B_SIL % BLOCK_SIL == 0, 'B_SIL must be a multiple of BLOCK_SIL'
    comp_s = [dict(kind="silhouette", branch=branch, X=X[(branch, COHORT_SIL)],
                   label=f'{branch}|{COHORT_SIL}', block=b, n=BLOCK_SIL)
              for branch in BRANCHES for b in range(B_SIL // BLOCK_SIL)]

    def launch(lst, name):
        with mp.get_context("spawn").Pool(N_PROC) as pool:
            out = []
            for i, r in enumerate(pool.imap(execute, lst, chunksize=1)):
                out.append(r)
                if (i + 1) % 20 == 0:
                    print(f'  ... {name}: {i + 1}/{len(lst)}  ({time.time() - t_start:.0f} s)', flush=True)
        return out

    # --------------- 3. identity with the pipeline, before everything else ----
    title("3. IDENTITY WITH THE PIPELINE, VERIFIED BEFORE LAUNCHING THE REST")
    log(f'Only the models that the pipeline has already produced are estimated first ({len(firsts)} of {len(tasks)}: bilateral putamen and caudate and the ladder of the bilateral putamen) and the reference arm of the battery is regenerated; only if every identity holds do the other {len(rest)} models and the nulls start.')
    log(f'cascade: {OPTIMISERS}; criterion: Newton decrement < {TOL_NEWTON:g}')
    f88 = os.path.join(OUT, "88_diagnostics_63.csv")
    assert os.path.exists(f88), "88_diagnostics_63.csv missing: the published family cannot be reconstructed"
    pub = pd.read_csv(f88)

    def consistency_p(t, name):
        """In a file of 88 every p must be the Wald p of its own beta and SE."""
        ok = t.p.notna() & t.beta_interaction.notna() & t.se.notna()
        z = 2 * stats.norm.sf(np.abs(t.beta_interaction[ok] / t.se[ok]))
        sc = float(np.abs(z - t.p[ok]).max())
        assert sc < 1e-9, \
            (f'{name}: at least one p is not the Wald p of its own coefficient and standard error (maximum discrepancy {sc:.2e}): the file has been altered')
        log(f'  {name}: internal consistency of the p (p = 2[1 - Phi(|beta/SE|)]) in {int(ok.sum())} models of {len(t)}, maximum discrepancy {sc:.1e}')

    consistency_p(pub, '88_diagnostics_63.csv of the working directory')
    # the null saved by 25: verified IN FULL, without recomputing anything
    n25 = pd.read_csv(os.path.join(OUT, "25_nulls.csv"))
    SAV, S25 = {}, {}
    for tag in COHORTS:
        sav = pd.read_csv(os.path.join(OUT, f"25_null_distribution_{tag}.csv"))
        assert len(sav) == B_NULL[tag], 'unexpected number of realisations saved by 25'
        assert set(sav.columns) == set(MORE_STRUCTURE), \
            'the statistics of the null saved by 25 are not those expected'
        s25 = n25[n25.cohort.str.startswith(tag)].set_index('statistic')
        assert s25.index.is_unique and set(s25.index) == set(MORE_STRUCTURE), \
            f'25_nulls.csv, {tag} cohort: statistics do not conform'
        for st, high in MORE_STRUCTURE.items():
            v = sav[st].to_numpy(float)
            kk, den, pp, _ = p_mc(v, float(s25.loc[st, 'observed']), high)
            for name, val in (('null_median', float(np.median(v))),
                              ("nullo_IC2.5", float(np.percentile(v, 2.5))),
                              ("nullo_IC97.5", float(np.percentile(v, 97.5))),
                              ("p_montecarlo", pp), ("B", len(v))):
                assert abs(val - float(s25.loc[st, name])) < 1e-12, \
                    (f'{tag} cohort, {st}: {name} recomputed from the saved null ({val!r}) is not that of 25_nulls.csv ({s25.loc[st, name]!r}): the saved distribution and its summary do not correspond')
        SAV[tag], S25[tag] = sav, s25
        log(f'  null saved by 25, {LBLS[tag]} cohort: median, 2.5 and 97.5 percentile, p and B of all {len(MORE_STRUCTURE)} statistics, recomputed from the {len(sav)} saved realisations, coincide with 25_nulls.csv')

    print(f'[{N_PROC} processes]', flush=True)
    outcomes1 = launch(comp_id + [tasks[i] for i in firsts], 'reference')
    result_id = {r['cohort']: r for r in outcomes1[:len(comp_id)]}
    M1 = pd.DataFrame(outcomes1[len(comp_id):])
    assert M1.estimated.all(), (f'{int((~M1.estimated).sum())} reference models without any estimate: no optimiser produced a result')
    for tag in COHORTS:
        leading = pd.DataFrame(result_id[tag]['leading'])
        for st in MORE_STRUCTURE:
            assert abs(result_id[tag]['obsv'][st] - S25[tag].loc[st, 'observed']) < 1e-9, \
                f"{tag}, {st}: the observed value does not reproduce that of 25_nulls.csv"
            assert np.allclose(leading[st].to_numpy(float),
                               SAV[tag][st].to_numpy(float)[:K_ID], rtol=0, atol=1e-9), \
                f'{tag}, {st}: the first {K_ID} realisations of the null do not reproduce 25'
        log(f'  identity of the code with 25, {LBLS[tag]} cohort: the observed value of the {len(MORE_STRUCTURE)} statistics and the first {K_ID} realisations of the null, regenerated here, coincide with the saved files; the null of the reference arm is that saved by 25 ({len(SAV[tag])} realisations).')
    R0 = M1[M1.block == 'reference']

    def compare(mine, its, keys, name, col=("beta_interaction", "se", "p"),
                  num=("n_subjects", "n_obs")):
        """Same models, same sample sizes, same estimates: otherwise it stops."""
        its = its[keys + list(col) + list(num)].rename(columns=dict(
            zip(list(col) + list(num), ["b_pip", 'se_pip', "p_pip", "ns_pip", "no_pip"])))
        j = mine.merge(its, on=keys, how="inner", validate="one_to_one")
        if not len(j):
            return 0
        assert (j.n_subj == j.ns_pip).all() and (j.n_obsv == j.no_pip).all(), \
            f'{name}: sample sizes differ from the pipeline'
        db = float((j.beta - j.b_pip).abs().max())
        sd = float((j.se - j.se_pip).abs().max())
        dp = float((j.p - j.p_pip).abs().max())
        assert db < TOL_ID_BETA and sd < TOL_ID_BETA and dp < TOL_ID_P, \
            (f'{name}: the estimates do not coincide with the pipeline (discrepancies: beta {db:.2e}, SE {sd:.2e}, p {dp:.2e})')
        log(f'  {name}: {len(j)} models, same sample sizes; maximum discrepancy beta {db:.1e}, SE {sd:.1e}, p {dp:.1e}')
        return len(j)

    mine = R0.rename(columns={'cohort': "cohort", 'outcome': "outcome", 'kinetic': "kinetic"})
    n_put = compare(mine, pub, ["cohort", "outcome", "kinetic"],
                      '88_diagnostics_63.csv of the working directory')
    expected_put = int((R0.outcome == "MIA_PUTAMEN_BILAT").sum())
    assert n_put >= expected_put, "the bilateral putamen is not in 88_diagnostics_63.csv"
    # the family of the manuscript has the caudate among the outcomes: with a single
    # chain it is in this same working directory
    fcaud = OUT
    pub_c, caud_verified = None, False
    if os.path.exists(os.path.join(fcaud, "88_diagnostics_63.csv")):
        def table_of(folder, name):
            t = pd.read_csv(os.path.join(folder, name))
            t["PATNO"] = t["PATNO"].astype("int64")
            return t

        matching = all(
            os.path.exists(os.path.join(fcaud, n_))
            and table_of(fcaud, n_).equals(table_of(OUT, n_))
            for n_ in ("analysis_complete_cases.csv", "08_dataset_150h.csv"))
        if matching:
            pub_c = pd.read_csv(os.path.join(fcaud, "88_diagnostics_63.csv"))
            consistency_p(pub_c, '88_diagnostics_63.csv of run_caudate')
            n_c = compare(mine, pub_c, ["cohort", "outcome", "kinetic"],
                            '88_diagnostics_63.csv of run_caudate (same cohorts, verified)')
            caud_verified = n_c == len(R0)
        else:
            log('  run_caudate exists but its cohorts are not those of this working directory: not used.')
    else:
        log('  run_caudate/outputs/88_diagnostics_63.csv NOT found.')
    n_clin_88 = int((~pub.outcome.str.startswith("MIA_")).sum())
    if not caud_verified:
        log('  WARNING, NOT VERIFIED for this working directory:')
        log('    (a) the bilateral caudate has no term of comparison in the pipeline: its estimates '
            'are of this script only;')
        log(f"    (b) the p of the {n_clin_88} models on the clinical outcomes, READ from "
            f"88_diagnostics_63.csv of the working directory and not re-estimated here, have no "
            f"second file to compare them with: only their internal consistency is verified;")
        log("    (c) the 'reconstructed published family' and every corrected p that contains those "
            "p (published, substitution and enlarged families) are NOT verified.")
    f87 = os.path.join(OUT, "87_amplitude_ladder.csv")
    S1 = M1[M1.block == 'scale']
    scale_verified = False
    if os.path.exists(f87):
        n_l = compare(S1, pd.read_csv(f87), ['outcome', 'kinetic', "spec"],
                        '87_amplitude_ladder.csv (scale of the bilateral putamen)',
                        col=("beta", "se", "p"), num=('n_subj', 'n_obsv'))
        scale_verified = n_l == len(S1)
    if not scale_verified:
        log("  WARNING, NOT VERIFIED: the ladder of the bilateral putamen has not been compared in "
            "full with 87_amplitude_ladder.csv.")

    # ------------------------------------------------------- 4. estimation --------
    title('4. ESTIMATION OF THE REST')
    log(f"mixed models: {len(tasks)} in all, of which {len(firsts)} already estimated in section 3 (skipped for insufficient sample size: {n_skipped}); batteries against the copula null: {len(comp_b)} (arms x cohorts), {B_NULL['24h']} and {B_NULL['150h']} realisations; null of the silhouette alone: {B_SIL} realisations for {len(BRANCHES)} putaminal variables, {LBLS[COHORT_SIL]} cohort")
    # the batteries, which take longer, start first
    outcomes2 = launch(comp_b + comp_f + [tasks[i] for i in rest] + comp_s, 'rest')
    result_b = outcomes2[:len(comp_b)]
    result_f = {r['cohort']: r['streams'] for r in outcomes2[len(comp_b):len(comp_b) + len(comp_f)]}
    n0 = len(comp_b) + len(comp_f)
    result_s = outcomes2[n0 + len(rest):]
    rows_m = [None] * len(tasks)
    for i, r in zip(firsts, outcomes1[len(comp_id):]):
        rows_m[i] = r
    for i, r in zip(rest, outcomes2[n0:n0 + len(rest)]):
        rows_m[i] = r
    M = pd.DataFrame(rows_m)
    assert M.estimated.all(), (f'{int((~M.estimated).sum())} models without any estimate: no optimiser produced a result')
    n_no = int((~M.accepted).sum())
    log(f'models estimated: {len(M)}; that satisfy the criterion (a converged optimiser, Newton decrement < {TOL_NEWTON:g}, profiled Hessian negative definite, not singular): {int(M.accepted.sum())}')
    log(f'  maximum Newton decrement {M.newton.max():.2e}; maximum gain of the restart {M.gain_reoptimisation.max():.2e} log-likelihood units; rounds from {int(M.rounds.min())} to {int(M.rounds.max())}')
    if n_no:
        log('  do NOT satisfy the criterion and are excluded from the summary sentences:')
        log(M.loc[~M.accepted, ['block', 'cohort', 'outcome', 'kinetic', "spec", "newton",
                                 'n_optim_convergent', 'singular']].to_string(index=False))
    S = M[M.block == 'scale']
    # internal consistency: M0 of the ladder = base model of the same outcome
    j = S[S.spec == "M0"].merge(M[M.block != 'scale'], on=['cohort', 'outcome', 'kinetic'],
                                suffixes=("", "_b"), validate="one_to_one")
    assert (j.beta - j.beta_b).abs().max() < TOL_ID_BETA and (j.n_obsv == j.n_obsv_b).all(), \
        'M0 of the ladder does not coincide with the base model of the same outcome'
    log(f'  internal consistency: the {len(j)} M0 models of the ladder coincide with the base models '
        f'(same observations, maximum discrepancy of beta {(j.beta - j.beta_b).abs().max():.1e})')
    # every lateralised outcome has all the kinetic parameters of the cohort,
    # maximum slope included
    for tag, (c, kin) in COHORTS.items():
        for q in DEFINITIONS:
            for side in SIDES:
                g = M[(M.block == 'laterality') & (M.cohort == tag)
                      & (M.definition == q) & (M.side == side)]
                assert set(g.kinetic) == set(kin.values()) and len(g) == len(kin), \
                    f'{tag} cohort, {q}, {side}: kinetic parameters are missing among the estimated models'
    for outcome in outcomes_scale:
        g = S[S.outcome == outcome]
        assert set(zip(g.kinetic, g.spec)) == {(k, sp) for k in KIN_SCALE
                                                for sp, _, _ in SPEC}, \
            f'ladder, {outcome}: kinetic parameters or specifications are missing'
    log(f"  completeness: every lateralised outcome has all the kinetic parameters of the cohort ({', '.join(KIN24.values())} at 24 h; {', '.join(KIN150.values())} at 150 h) and the ladder has {', '.join(KIN_SCALE)} in all the {len(SPEC)} specifications")
    if n_run_different is not None:
        log(f'  note on the ladder: as in 87, the run is the run date of the first baseline row of Project 237 in the file; for {n_run_different} subjects of {len(b24)} it is not the assay run of the Type1 sample whose kinetic parameters are in the cohort.')

    # -------------------------------------------------- 5. models -----------
    title("5. KINETIC PARAMETER x TIME ON THE LESS AFFECTED AND MORE AFFECTED PUTAMEN")
    log(f"Population: {POP}. Model: outcome ~ kinetic parameter*years + age + sex + duration,")
    log("random intercept and slope per subject, REML. beta = interaction, in units")
    log("of SBR per year per 1 SD of the logarithm of the kinetic parameter (SD over the whole cohort).")
    log('p = Wald, NOT corrected. BH correction within cohort in three families:')
    log('  BH publ.  = the outcomes of the manuscript (clinical read from 88, dopaminergic here);')
    log("  BH subst. = the same family with the bilateral putamen replaced by the lateralised")
    log('              outcome of the row (same size);')
    log("  BH enl.   = the published family plus ALL the lateralised outcomes (less and")
    log(f"              more affected, {len(DEFINITIONS)} definitions: {', '.join(DEFINITIONS)}).")
    log("The contrast 'less - more' is corrected separately, within cohort, over the contrasts only.")
    log('effect_rel_pct = 100 x beta / mean annual slope of the outcome: percentage')
    log("change of the slope per 1 SD of the kinetic parameter (positive = faster loss).")
    Mb = M[M.block != 'scale'].copy()
    # relative effect: by how much, in per cent, the mean annual slope of the
    # outcome changes per 1 SD of the kinetic parameter (positive = faster loss). It
    # makes comparable two putamina that start from different levels and slopes.
    Mb['effect_rel_pct'] = 100 * Mb.beta / Mb.slope_mean
    for col in ('p_BH_pubd', 'p_BH_substitution', 'p_BH_widened', 'p_BH_contrasts'):
        Mb[col] = np.nan
    fam = {}
    for tag, (c, kin) in COHORTS.items():
        clin = pub[(pub.cohort == tag) & ~pub.outcome.str.startswith("MIA_")]
        assert set(clin.kinetic) == set(kin.values()) and not clin.duplicated(
            ["outcome", "kinetic"]).any() and len(clin) == clin.outcome.nunique() * len(kin), \
            'clinical outcomes of 88_diagnostics_63.csv do not conform to the expected grid'
        if pub_c is not None:
            cc_ = pub_c[(pub_c.cohort == tag) & ~pub_c.outcome.str.startswith("MIA_")]
            jj = clin.merge(cc_, on=["outcome", "kinetic"], suffixes=("", "_c"),
                            validate="one_to_one")
            assert len(jj) == len(clin) and (jj.p - jj.p_c).abs().max() < TOL_ID_P, \
                'the clinical p of the working directory and of run_caudate do not coincide'
        i_ref = Mb.index[(Mb.cohort == tag) & (Mb.block == 'reference')]
        i_put = Mb.index[(Mb.cohort == tag) & (Mb.outcome == "MIA_PUTAMEN_BILAT")]
        i_cau = Mb.index[(Mb.cohort == tag) & (Mb.outcome == "MIA_CAUDATE_BILAT")]
        i_lat = Mb.index[(Mb.cohort == tag) & (Mb.block == 'laterality')
                         & Mb.side.isin(['less', 'more'])]
        i_dif = Mb.index[(Mb.cohort == tag) & (Mb.block == 'laterality')
                         & (Mb.side == "diff")]
        pc = clin.p.to_numpy()
        q = bh(np.concatenate([pc, Mb.loc[i_ref, "p"]]))
        Mb.loc[i_ref, 'p_BH_pubd'] = q[len(pc):]
        n_pub = len(pc) + len(i_ref)
        surv_pub = int((q < ALPHA).sum())
        if pub_c is not None:
            jj = Mb.loc[i_ref].merge(
                pub_c[pub_c.cohort == tag], left_on=['outcome', 'kinetic'],
                right_on=["outcome", "kinetic"], validate="one_to_one")
            assert len(jj) == len(i_ref) and \
                (jj.p_BH_pubd - jj.p_BH_within_cohort).abs().max() < TOL_ID_P and \
                int((pub_c[pub_c.cohort == tag].p_BH_within_cohort < ALPHA).sum()) == surv_pub \
                and int((pub_c.cohort == tag).sum()) == n_pub, \
                'the reconstructed published family does not reproduce that of run_caudate'
        for (qd, side), g in Mb.loc[i_lat].groupby(['definition', 'side']):
            qq = bh(np.concatenate([pc, Mb.loc[i_cau, "p"], g.p]))
            Mb.loc[g.index, 'p_BH_substitution'] = qq[len(pc) + len(i_cau):]
            fam[(tag, qd, side)] = dict(n=len(qq), survivors=int((qq < ALPHA).sum()),
                                        outside_fromthe_putamen=int((qq[:len(pc) + len(i_cau)] < ALPHA).sum()))
        qa = bh(np.concatenate([pc, Mb.loc[i_ref, "p"], Mb.loc[i_lat, "p"]]))
        Mb.loc[i_lat, 'p_BH_widened'] = qa[len(pc) + len(i_ref):]
        names_all = ([f"{o} x {k}" for o, k in zip(clin.outcome, clin.kinetic)]
                    + [f'{REF[e]} x {k}' for e, k in zip(Mb.loc[i_ref, 'outcome'],
                                                         Mb.loc[i_ref, 'kinetic'])]
                    + [f'putamen {SIDES[l_]} ({d_}) x {k}' for l_, d_, k in zip(
                        Mb.loc[i_lat, 'side'], Mb.loc[i_lat, 'definition'],
                        Mb.loc[i_lat, 'kinetic'])])
        Mb.loc[i_dif, 'p_BH_contrasts'] = bh(Mb.loc[i_dif, "p"])
        fam[tag] = dict(surv_all=int((qa < ALPHA).sum()),
                        chi_all=[n_ for n_, v in zip(names_all, qa) if v < ALPHA],
                        n_pub=n_pub, surv_pub=surv_pub, n_all=len(qa), n_clin=len(pc),
                        n_contrasts=len(i_dif), n_sost=len(pc) + len(i_cau) + len(kin),
                        surv_put=int((Mb.loc[i_put, 'p_BH_pubd'] < ALPHA).sum()))

    def table(g, columns):
        t = g[columns].copy()
        for col in t.columns:
            if col in ("beta", 'ci_inf', 'ci_upp', 'slope_mean'):
                t[col] = t[col].map(lambda v: f"{v:+.5f}")
            elif col == 'se':
                t[col] = t[col].map(lambda v: f"{v:.5f}")
            elif col == 'effect_rel_pct':
                t[col] = t[col].map(lambda v: f"{v:+.1f}")
            elif col == "p" or col.startswith("p_BH"):
                t[col] = t[col].map(lambda v: "" if pd.isna(v) else fmt_p(v))
        return t.to_string(index=False)

    for tag in COHORTS:
        g0 = Mb[(Mb.cohort == tag)]
        f0 = fam[tag]
        log(f'\n--- {LBLS[tag]} cohort ---')
        log(f"families: published {f0['n_pub']} tests ({f0['n_clin']} clinical + {f0['n_pub'] - f0['n_clin']} dopaminergic), substitution {f0['n_sost']}, enlarged {f0['n_all']}, contrasts {f0['n_contrasts']}")
        log(f"reconstructed published family: {f0['surv_pub']} of {f0['n_pub']} survive BH < {ALPHA}"
            + (' (identical to that of run_caudate)' if pub_c is not None else
               f" (NOT VERIFIED: run_caudate not available; the {f0['n_clin']} clinical p are read from 88_diagnostics_63.csv of the working directory without a second comparison)"))
        log('\nReference (outcomes of the manuscript, re-estimated):')
        log(table(g0[g0.block == 'reference'],
                    ['outcome', 'kinetic', 'n_subj', 'n_obsv', "beta", 'se', "p",
                     'p_BH_pubd', 'slope_mean', 'effect_rel_pct']))
        for q, descr in DEFINITIONS.items():
            gq = g0[g0.definition == q]
            if not len(gq):
                continue
            log(f'\nDefinition {q} — {descr}  [{int(gq.n_subj.max())} subjects, {int(gq.n_obsv.max())} scans]')
            for side in ('less', 'more'):
                log(f'  putamen {SIDES[side]}:')
                log("  " + table(gq[gq.side == side],
                                   ['kinetic', "beta", 'se', 'ci_inf', 'ci_upp', "p",
                                    'p_BH_substitution', 'p_BH_widened',
                                    'slope_mean', 'effect_rel_pct']
                                   ).replace("\n", "\n  "))
            log("  contrast between the sides (outcome = less affected - more affected):")
            log("  " + table(gq[gq.side == "diff"],
                               ['kinetic', "beta", 'se', 'ci_inf', 'ci_upp', "p",
                                'p_BH_contrasts']).replace("\n", "\n  "))
        gx = g0[g0.block == 'border']
        if len(gx):
            Lt = TAB[tag]
            log(f"\nSensitivity of definition V5 at the boundary of the threshold [{int(gx.n_subj.max())} subjects, {int(gx.n_obsv.max())} scans]. Scans with asymmetry index EXACTLY equal to {AI_THRESHOLD_PCT}%: {int(Lt.at_border.sum())} (of which baseline: {int((Lt.at_border & (Lt.EVENT_ID == 'BL')).sum())}). V5 treats them as symmetric, as the article does; here they are treated as asymmetric (endpoint excluded, 'V5x'). p NOT corrected, OUTSIDE every family:")
            for side in SIDES:
                log(f"  {('putamen ' + SIDES[side] if side != 'diff' else 'contrast between the sides (' + SIDES[side] + ')')}:")
                log("  " + table(gx[gx.side == side],
                                   ['kinetic', "beta", 'se', 'ci_inf', 'ci_upp', "p"]
                                   ).replace("\n", "\n  "))

    # ---------------------------------------------------- 6. ladder -----------
    title('6. TECHNICAL LADDER run/centre ON THE LATERALISED OUTCOMES (24 h cohort)')
    log(f"Population: {POP}, {LBLS['24h']} cohort. Specifications as in 87 and in the")
    log('ladder table of the supplement: M0 published; M4 + run + centre; M5 + run x')
    log('years + centre x years; M6 kinetic parameter residualised for run and restandardised. Run')
    log(f'and centre pooled below {N_MIN_COLLAPSE} subjects: {b24.run.nunique()} categories of run, {b24.site.nunique()} of centre. p NOT corrected (no family).')
    log("beta in units of SBR per year per 1 SD of the logarithm of the kinetic parameter. "
        + "; ".join(f"{k} = {v}" for k, v in NAME_KIN.items()) + ".")
    Sw = S.copy()
    for outcome in outcomes_scale:
        g = Sw[Sw.outcome == outcome]
        if not len(g):
            continue
        q = g.definition.iloc[0]
        name = (REF[outcome] + ' (reference, as in 87)' if outcome in REF else
                f'putamen {SIDES[g.side.iloc[0]]}, definition {q}')
        log(f'\n{name}  [{int(g.n_subj.max())} subjects, {int(g.n_obsv.max())} scans]')
        for k in KIN_SCALE:
            gk = g[g.kinetic == k].set_index("spec")
            log(f"  {k:<6}" + "   ".join(
                f"{sp} {gk.loc[sp, 'beta']:+.4f} (p = {fmt_p(gk.loc[sp, 'p'])})"
                for sp, _, _ in SPEC if sp in gk.index))

    # ------------------------------------------------- 7. battery -----------
    title('7. BATTERY AT k = 2 AGAINST THE GAUSSIAN COPULA NULL')
    log(f'Population: {POP}. Seven standardised variables: four kinetic parameters in')
    log('logarithm, ONE putaminal measure (that of the arm), bilateral caudate of the curated data cut')
    log("(MIA_CAUDATE_BILAT), age. Statistics and null as in 25_monte_carlo_nulls.py.")
    log('p = one-tailed Monte Carlo p, written as the exact fraction')
    log("(#{null >= observed} + 1)/(B + 1), NOT corrected; for the minimum dip the tail is")
    log('the lower one. SE = Monte Carlo standard error of the p, square root of p(1 - p)/B.')
    log("'Reference interval of the null' = from the 2.5 to the 97.5 percentile of the")
    log("realisations of the null: it is a reference interval, NOT a confidence")
    log('interval. The curves over k = 2-10 are not redone.')
    log("'obs. stream 25' = value observed on the random stream of 25 (seed and order of")
    log("calls of 25): it is the value of ONE stream, the same for every arm, and on the same")
    log('stream the realisations of the null are computed.')
    log(f'PS and Jaccard depend on the random stream of the algorithm: they are recomputed on {R_OBSV}')
    log("independent streams, and their mean is given with the p at the mean. The number of streams")
    log("with p below threshold describes the dependence on the stream: it is not a test.")
    log("Methodological notes (NOT computed here). (i) The mean over several streams has less noise")
    log('than a single stream, whereas each realisation of the null carries the noise of a single')
    log("stream: the p at the mean is therefore on the cautious side. (ii) For the reference")
    log('arm the designated estimates of the manuscript come from 03 and 08 (another stream and,')
    log("for Jaccard, more resamplings): the values below are NOT those estimates and")
    log('serve the like-for-like comparison between arms, not to replace them.')
    for tag in COHORTS:
        result_b.append({'branch': 'reference', 'cohort': tag, 'obsv': result_id[tag]['obsv'],
                      'nulls': SAV[tag][list(MORE_STRUCTURE)].to_dict("records"),
                      'streams': result_f[tag], 'fr_25': True})
    COL_B = ['branch', 'variable_putaminal', 'cohort', 'population', "n", 'statistic',
             'criterion_of_rule', 'depends_fromthe_stream', 'observed_stream_25',
             'null_median', 'null_ref_p2_5', 'null_ref_p97_5', "B",
             'p_numerator_stream_25', 'p_denominator', 'p_montecarlo_stream_25',
             'p_se_mc_stream_25', 'n_streams', 'observed_mean_streams',
             'observed_sd_streams', 'observed_min_streams', 'observed_max_streams',
             'p_numerator_mean_streams', 'p_at_mean_streams', 'p_se_mc_mean_streams',
             'streams_having_p_below_alpha', 'ref_observed_stream_25',
             'ref_p_montecarlo_stream_25', 'ref_observed_mean_streams',
             'ref_p_at_mean_streams', 'correction', 'null_fr_25']
    rows_b, rows_n, summary, BY = [], [], {}, {}
    for res in sorted(result_b, key=lambda r: (list(BRANCHES).index(r['branch']),
                                            list(COHORTS).index(r['cohort']))):
        branch, tag = res['branch'], res['cohort']
        N = pd.DataFrame(res['nulls'])
        Fl = pd.DataFrame(res['streams'])
        assert len(Fl) == R_OBSV, 'unexpected number of streams'
        log(f'\n[{BRANCHES[branch][1]}; {POP}; {LBLS[tag]} cohort; B = {len(N)}'
            + ('; null read from 25 and verified in full in section 3'
               if res.get('fr_25') else "") + "]")
        log(f"  {'statistic':<26}{'obs. stream 25':>15}{'null median':>15}{'null ref. interval':>27}   p (exact fraction; Monte Carlo SE)")
        per, sig = {}, []
        for st, high in MORE_STRUCTURE.items():
            v = N[st].to_numpy(float)
            o = float(res['obsv'][st])
            k, den, p, se = p_mc(v, o, high)
            lo, hi = np.percentile(v, [2.5, 97.5])
            row = dict(branch=branch, variable_putaminal=BRANCHES[branch][1], cohort=tag,
                        population=POP, n=len(X[(branch, tag)]), statistic=st,
                        criterion_of_rule=st in CRITERIA,
                        depends_fromthe_stream=DEPENDS_FROMTHE_STREAM[st], observed_stream_25=o,
                        null_median=float(np.median(v)), null_ref_p2_5=float(lo),
                        null_ref_p97_5=float(hi), B=len(v), p_numerator_stream_25=k,
                        p_denominator=den, p_montecarlo_stream_25=p, p_se_mc_stream_25=se,
                        correction='none (uncorrected p)',
                        null_fr_25=bool(res.get('fr_25', False)))
            if st in Fl.columns:
                om = float(Fl[st].mean())
                km, _, pm, esm = p_mc(v, om, high)
                pf = np.array([p_mc(v, x, high)[2] for x in Fl[st]])
                row.update(n_streams=len(Fl), observed_mean_streams=om,
                            observed_sd_streams=float(Fl[st].std(ddof=1)),
                            observed_min_streams=float(Fl[st].min()),
                            observed_max_streams=float(Fl[st].max()),
                            p_numerator_mean_streams=km, p_at_mean_streams=pm,
                            p_se_mc_mean_streams=esm,
                            streams_having_p_below_alpha=int((pf < ALPHA).sum()))
            per[st] = row
            if p < ALPHA:
                sig.append(st)
            log(f"  {st:<26}{o:>15.4f}{np.median(v):>15.4f}{f'[{lo:.3f}; {hi:.3f}]':>27}   {fmt_mc(k, den, p, se)}")
        BY[(branch, tag)] = per
        for st in Fl.columns:
            r_ = per[st]
            log(f"  {st} on {len(Fl)} independent streams: mean {r_['observed_mean_streams']:.4f} (SD {r_['observed_sd_streams']:.4f}; from {r_['observed_min_streams']:.4f} to {r_['observed_max_streams']:.4f}); p at the mean {fmt_mc(r_['p_numerator_mean_streams'], r_['p_denominator'], r_['p_at_mean_streams'], r_['p_se_mc_mean_streams'])}; streams with p < {ALPHA}: {r_['streams_having_p_below_alpha']} of {len(Fl)}")
        if branch != 'reference':
            ref = BY[('reference', tag)]
            log(f"  Like-for-like comparison with the reference arm ({BRANCHES['reference'][1]}), same cohort, same stream of 25, each arm against its own null:")
            for st in MORE_STRUCTURE:
                a, b = per[st], ref[st]
                a.update(ref_observed_stream_25=b['observed_stream_25'],
                         ref_p_montecarlo_stream_25=b['p_montecarlo_stream_25'])
                log(f"    {st:<26} this arm {a['observed_stream_25']:.4f}, p {fmt_mc(a['p_numerator_stream_25'], a['p_denominator'], a['p_montecarlo_stream_25'], a['p_se_mc_stream_25'])} | reference {b['observed_stream_25']:.4f}, p {fmt_mc(b['p_numerator_stream_25'], b['p_denominator'], b['p_montecarlo_stream_25'], b['p_se_mc_stream_25'])}")
                if st in Fl.columns:
                    a.update(ref_observed_mean_streams=b['observed_mean_streams'],
                             ref_p_at_mean_streams=b['p_at_mean_streams'])
                    log(f"    {'':<26} mean of {len(Fl)} streams: this arm {a['observed_mean_streams']:.4f}, p {fmt_mc(a['p_numerator_mean_streams'], a['p_denominator'], a['p_at_mean_streams'], a['p_se_mc_mean_streams'])} | reference {b['observed_mean_streams']:.4f}, p {fmt_mc(b['p_numerator_mean_streams'], b['p_denominator'], b['p_at_mean_streams'], b['p_se_mc_mean_streams'])}")
        rows_b.extend(per.values())
        crit = [st for st in sig if st in CRITERIA]
        crit_m = [st for st in CRITERIA
                  if per[st].get('p_at_mean_streams', per[st]['p_montecarlo_stream_25']) < ALPHA]
        summary[(branch, tag)] = dict(criteria=crit, criteria_mean=crit_m,
                                      rest=[st for st in sig if st not in CRITERIA])
        log(f'  reproducibility criteria of the primary rule (PS, Jaccard, ARI) with p < {ALPHA}: {len(crit)} on the stream of 25' + (f" -> {crit}" if crit else "")
            + f', {len(crit_m)} with PS and Jaccard at the mean of the streams'
            + (f" -> {crit_m}" if crit_m else "")
            + f'; other statistics with p < {ALPHA} (not corrected): '
            + (", ".join(summary[(branch, tag)]['rest']) or 'nothing'))
        if not res.get('fr_25'):
            for b, r in enumerate(res['nulls']):
                rows_n.append(dict(branch=branch, cohort=tag, b=b, **r))
    # silhouette at k = 2: larger null, to resolve its p
    log(f'\n--- Silhouette alone at k = 2, copula null with B = {B_SIL}; {POP}; {LBLS[COHORT_SIL]} cohort ---')
    log(f'Same k-means and same copula null as the battery; {B_SIL} independent')
    log(f'realisations for each of the {len(BRANCHES)} putaminal variables (own seeds, SEED_SIL = {SEED_SIL}).')
    log("p NOT corrected. The silhouette at k = 2 is NOT one of the three reproducibility")
    log("criteria of the rule (PS, Jaccard, ARI between algorithms) and the curves over")
    log('k = 2-10 have not been redone for the variants: this p lies OUTSIDE the rule.')
    rows_s, rows_sn = [], []
    for branch in BRANCHES:
        bl = sorted((r for r in result_s if r['branch'] == branch), key=lambda r: r['block'])
        assert [r['block'] for r in bl] == list(range(B_SIL // BLOCK_SIL)), \
            f'silhouette, {branch}: blocks of the null missing or repeated'
        v = np.concatenate([np.asarray(r['values'], float) for r in bl])
        assert len(v) == B_SIL and np.isfinite(v).all(), 'silhouette: incomplete null'
        o = silhouette_k2(X[(branch, COHORT_SIL)])
        r500 = BY[(branch, COHORT_SIL)]["silhouette_k2"]
        assert abs(o - r500['observed_stream_25']) < 1e-12, \
            f'observed silhouette, {branch}: it is not that of the battery'
        k, den, p, se = p_mc(v, o)
        lo, hi = np.percentile(v, [2.5, 97.5])
        z = (r500['p_montecarlo_stream_25'] - p) / np.sqrt(r500['p_se_mc_stream_25'] ** 2
                                                           + se ** 2)
        rows_s.append(dict(
            branch=branch, variable_putaminal=BRANCHES[branch][1], cohort=COHORT_SIL, population=POP,
            n=len(X[(branch, COHORT_SIL)]), statistic="silhouette_k2", observed=o, B=len(v),
            null_median=float(np.median(v)), null_ref_p2_5=float(lo),
            null_ref_p97_5=float(hi), p_numerator=k, p_denominator=den, p_montecarlo=p,
            p_se_mc=se, correction='none (uncorrected p)',
            in_rule='no: outside the three reproducibility criteria (PS, Jaccard, ARI)',
            B_batt=r500["B"], p_numerator_batt=r500['p_numerator_stream_25'],
            p_denominator_batt=r500['p_denominator'],
            p_montecarlo_batt=r500['p_montecarlo_stream_25'],
            p_se_mc_batt=r500['p_se_mc_stream_25'], z_between_i_two_p=float(z),
            seed=SEED_SIL))
        rows_sn.extend(dict(branch=branch, cohort=COHORT_SIL, b=i, silhouette_k2=x)
                        for i, x in enumerate(v))
        log(f"  {BRANCHES[branch][1]}: observed {o:.4f}; null median {np.median(v):.4f}, reference interval [{lo:.3f}; {hi:.3f}]; p = {fmt_mc(k, den, p, se)}; with B = {r500['B']} (battery above) p = {fmt_mc(r500['p_numerator_stream_25'], r500['p_denominator'], r500['p_montecarlo_stream_25'], r500['p_se_mc_stream_25'])}; difference between the two p in standard errors: {z:+.2f}")
    sil_below = [r for r in rows_s if r["p_montecarlo"] < ALPHA]
    log(f"  With B = {B_SIL} the silhouette at k = 2 has p < {ALPHA} (not corrected) with {len(sil_below)} of the {len(rows_s)} putaminal variables; p from {fmt_p(min((r['p_montecarlo'] for r in rows_s)))} to {fmt_p(max((r['p_montecarlo'] for r in rows_s)))}, maximum Monte Carlo standard error {max((r['p_se_mc'] for r in rows_s)):.4f}.")
    # what the branches already run (102-104) show: the descriptive part only
    f104 = os.path.join(os.path.dirname(os.path.normpath(BASE)), 'run_dat_variants',
                        "104_comparison.csv")
    if os.path.exists(f104):
        v = pd.read_csv(f104)
        v = v[v.population == ('sporadic' if only24 else 'complete')]
        log('\nArms already run by the chain 102-104 (run_dat_variants), same population.')
        log('Observed values only: the comparisons of that file are with a single-realisation')
        log('unimodal null, later withdrawn, and are not reported here.')
        log(v[['measure', 'PS_max_obsv', 'PS_k_of_max', 'gap_k_obsv', 'dip_projections_sig',
               "PS150_max", "gap150_k", "n_150h"]].round(3).to_string(index=False))
        log(f'  the gap chooses k = 1 in all the arms: 24 h {bool((v.gap_k_obsv == 1).all())}, 150 h {bool((v.gap150_k == 1).all())}; maximum observed PS from {v.PS_max_obsv.min():.3f} to {v.PS_max_obsv.max():.3f} (24 h) and from {v.PS150_max.min():.3f} to {v.PS150_max.max():.3f} (150 h)')
    else:
        log('\nrun_dat_variants/104_comparison.csv not found: previous arms not reported.')
    import _dat
    ha_high = any("dat_putamen_high" in v for v in _dat._SET.values())
    log(f"Chain 102: DAT_SET admits {sorted(_dat._SET)}; a set with the higher-uptake putamen {('exists' if ha_high else 'does not exist')}. "
        + ('The arm could therefore be run from the chain as well.' if ha_high else
           'Adding it would require modifying _dat.py.'))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "25_monte_carlo_nulls.py")) as fh:
        fixed = '"dat_putamen_low"' in fh.read()
    log(f"25_monte_carlo_nulls.py {('fixed' if fixed else 'not fixed')} dat_putamen_low in the matrix of the 150 h cohort. "
        + ("For these reasons the battery of the 'less affected' arm is computed here, with the "
           "functions of 25 and the identity verified above." if (fixed or not ha_high) else
           "The battery is in any case computed here, with the identity verified above."))

    # -------------------------------------- 8. motor score --------------
    title("8. LATERALISED MOTOR SCORE: FEASIBILITY")
    log(f'File: {os.path.basename(UPDRS3)}, {len(u)} rows (examinations). Lateralised items: {len(ITEMS_D)} per side (rigidity of upper and lower limb, finger tapping,')
    log("hand movements, pronation-supination, toe tapping, leg agility, postural,")
    log('kinetic and rest tremor of upper and lower limb): a score of the side')
    log("contralateral to the less affected putamen CAN be built. Here the data are counted and")
    log("the clinical side at baseline is compared; no model is estimated on it.")
    for tag in COHORTS:
        mo = MOT[tag]
        log(f'\n[{POP}; {LBLS[tag]} cohort; unit: subjects, baseline examination]')
        log(f"  with baseline examination: {mo['having_exam']} (with two baseline examinations: {mo['doubles']}; the one not ON holds); with the {2 * len(ITEMS_D)} items complete: {share(mo['compl'], mo['n'])}; baseline examinations in ON: {mo['on_bl']}")
        record('motor', tag, 'lateralised items complete at baseline', mo["compl"],
                 mo["n"])
        log(f"  clinical side (larger sum): right {mo['clin_D']}, left {mo['clin_S']}, tie {mo['clin_ties']}")
        for name, k, n_ in (('clinical side = side of onset', mo["k1"], mo["n1"]),
                            ('clinical side contralateral to the putamen with lower uptake',
                             mo["k2"], mo["n2"])):
            lo, hi = wilson(k, n_)
            log(f'  {name}: {share(k, n_)} [Wilson 95% {100 * lo:.1f}; {100 * hi:.1f}]; not aligned {share(n_ - k, n_)}')
            record('motor', tag, name, k, n_)
        log(f"  visits with examination (unit: subject x visit): {mo['n_e']}; with complete items {mo['n_eok']}; of these not in ON {mo['n_no_on']} (declared OFF {mo['off']}, state not indicated {mo['without_state']}); subjects with at least two usable visits: {mo['multi']}")
        if mo['treat'] is not None:
            log(f"    among the examinations with state not indicated (unit: subject x visit): PDTRTMNT = 0 in {mo['treat'][0]}, PDTRTMNT = 1 in {mo['treat'][1]}, PDTRTMNT missing in {mo['treat'][2]} (in the PPMI codebook, NOT read here, 0 = not on treatment for PD and 1 = on treatment)")
        else:
            log('    composition of the examinations with state not indicated: NOT computed '
                '(PDTRTMNT missing from the file)')
        record('motor', tag, 'visits with complete items not in ON', mo['n_no_on'],
                 mo["n_e"], 'unit: subject x visit')
    log('\nMethodological note (NOT computed here). To estimate it one would need to decide the')
    log('pharmacological state admitted at each visit, the rule for the items that cannot be rated and')
    log("the possible exclusion of tremor; then the same kinetic parameter x time models. This")
    log("has not been done.")

    # ---------------------------------------------------- 9. summary ---------
    title("9. SUMMARY (every sentence is derived from the computed values)")
    Ma = Mb[Mb.accepted]
    for tag in COHORTS:
        f0 = fam[tag]
        log(f'\n{LBLS[tag]} cohort.')
        log(f"  Published family: {f0['surv_pub']} of {f0['n_pub']} interactions survive the correction, of which {f0['surv_put']} on the bilateral putamen.")
        for q in DEFINITIONS:
            for side in ('less', 'more'):
                g = Ma[(Ma.cohort == tag) & (Ma.definition == q) & (Ma.side == side)]
                if not len(g):
                    continue
                nom = g[g.p < ALPHA]
                so = g[g.p_BH_substitution < ALPHA]
                at = g[g.p_BH_widened < ALPHA]
                ff = fam[(tag, q, side)]
                signs = ('all negative' if (g.beta < 0).all() else
                         'all positive' if (g.beta > 0).all() else 'of mixed sign')
                log(f'  {q}, putamen {SIDES[side]}: nominal {len(nom)}/{len(g)}'
                    + (f" ({', '.join(nom.kinetic)})" if len(nom) else "")
                    + f'; survive BH in the substitution family {len(so)}'
                    + (f" ({', '.join(so.kinetic)})" if len(so) else "")
                    + f', in the enlarged one {len(at)}'
                    + (f" ({', '.join(at.kinetic)})" if len(at) else "")
                    + f"; coefficients {signs}; in the substitution family {ff['survivors']} of {ff['n']} survive in all ({ff['outside_fromthe_putamen']} outside the putamen).")
        gd = Ma[(Ma.cohort == tag) & (Ma.block == 'laterality') & (Ma.side == "diff")]
        nd, bd = gd[gd.p < ALPHA], gd[gd.p_BH_contrasts < ALPHA]
        log(f'  Contrast between the sides: nominal {len(nd)}/{len(gd)}'
            + (" (" + ", ".join(f"{a} {b}" for a, b in zip(nd.definition, nd.kinetic)) + ")"
               if len(nd) else "")
            + f'; after BH over the {len(gd)} contrasts {len(bd)}. '
            + ('No formal contrast distinguishes the two sides after the correction.'
               if not len(bd) else 'At least one formal contrast distinguishes the two sides.'))
        if len(gd):
            rm = gd.loc[gd.p.idxmin()]
            log(f'    smallest uncorrected p among the contrasts: {rm.kinetic}, definition {rm.definition}, beta {rm.beta:+.5f} [95% CI {rm.ci_inf:+.5f}; {rm.ci_upp:+.5f}], p = {fmt_p(rm.p)}, BH p = {fmt_p(rm.p_BH_contrasts)}.')
            gk = Ma[(Ma.cohort == tag) & (Ma.kinetic == rm.kinetic)
                    & (Ma.block == 'laterality')].set_index(['definition', 'side'])
            for q in DEFINITIONS:
                if all((q, l_) in gk.index for l_ in SIDES):
                    log(f"    {rm.kinetic}, {q}: less affected beta {gk.loc[(q, 'less'), 'beta']:+.5f} (p = {fmt_p(gk.loc[(q, 'less'), 'p'])}), more affected {gk.loc[(q, 'more'), 'beta']:+.5f} (p = {fmt_p(gk.loc[(q, 'more'), 'p'])}), contrast {gk.loc[(q, 'diff'), 'beta']:+.5f} (p = {fmt_p(gk.loc[(q, 'diff'), 'p'])})")
            gx = Ma[(Ma.cohort == tag) & (Ma.kinetic == rm.kinetic)
                    & (Ma.block == 'border')].set_index('side')
            if set(SIDES) <= set(gx.index):
                log(f"    {rm.kinetic}, V5x (scans exactly at {AI_THRESHOLD_PCT}% treated as asymmetric, n = {int(TAB[tag].at_border.sum())}; outside every family): less affected beta {gx.loc['less', 'beta']:+.5f} (p = {fmt_p(gx.loc['less', 'p'])}), more affected {gx.loc['more', 'beta']:+.5f} (p = {fmt_p(gx.loc['more', 'p'])}), contrast {gx.loc['diff', 'beta']:+.5f} (p = {fmt_p(gx.loc['diff', 'p'])})")
    # the ladder: does the less affected putamen behave like the bilateral one?
    Sa = S[S.accepted]
    log(f"\nTechnical ladder, {LBLS['24h']} cohort (p not corrected).")
    for outcome in outcomes_scale:
        g = Sa[Sa.outcome == outcome]
        if not len(g):
            continue
        name = (REF[outcome] if outcome in REF else
                f'{g.definition.iloc[0]}, putamen {SIDES[g.side.iloc[0]]}')
        pieces = []
        for k in KIN_SCALE:
            gk = g[g.kinetic == k].set_index("spec")
            alive = [sp for sp, _, _ in SPEC if sp in gk.index and gk.loc[sp, "p"] < ALPHA]
            pieces.append(f"{k}: nominal in {(', '.join(alive) if alive else 'nothing')}")
        log(f'  {name:<28} ' + "; ".join(pieces))
    put = Sa[(Sa.outcome == "MIA_PUTAMEN_BILAT")].set_index(['kinetic', "spec"]).p
    for q in DEFINITIONS:
        for side in ('less', 'more'):
            g = Sa[(Sa.outcome == f'put_{side}_{q}')].set_index(['kinetic', "spec"]).p
            if not len(g):
                continue
            same = all((g.get((k, sp), np.nan) < ALPHA) == (put.get((k, sp), np.nan) < ALPHA)
                         for k in KIN_SCALE for sp, _, _ in SPEC)
            holds = [k for k in KIN_SCALE
                     if g.get((k, "M5"), 1) < ALPHA and g.get((k, "M6"), 1) < ALPHA]
            log(f"  {q}, {SIDES[side]}: profile of nominal significance {('equal to' if same else 'different from')} that of the bilateral putamen; "
                + (f"it remains nominal both in M5 and in M6 for {', '.join(holds)}." if holds else
                   'no kinetic parameter remains nominal both in M5 and in M6.'))
    bb = Sa[Sa.outcome == "MIA_PUTAMEN_BILAT"].set_index(['kinetic', "spec"]).beta
    for k in KIN_SCALE:
        for q in DEFINITIONS:
            g = Sa[Sa.outcome == f'put_less_{q}'].set_index(['kinetic', "spec"]).p
            if not len(g):
                continue
            if not all(((k, sp) in g.index and (k, sp) in put.index)
                       for sp in ("M0", "M5", "M6")):
                log(f'  {k} on the less affected putamen ({q}): M0, M5 or M6 do not satisfy the '
                    f'estimation criterion; no sentence.')
                continue
            loses = all(g[(k, sp)] >= ALPHA for sp in ("M5", "M6"))
            loses_bil = all(put[(k, sp)] >= ALPHA for sp in ("M5", "M6"))
            bq = Sa[Sa.outcome == f'put_less_{q}'].set_index(['kinetic', "spec"]).beta
            log(f"  {k} on the less affected putamen ({q}): p = {fmt_p(g[k, 'M5'])} in M5, "
                f"{fmt_p(g[k, 'M6'])} in M6; the coefficient in M5 is "
                f"{100 * bq[k, 'M5'] / bq[k, 'M0']:.0f}% of that in M0 (bilateral: p = "
                f"{fmt_p(put[k, 'M5'])} and {fmt_p(put[k, 'M6'])}, "
                f"{100 * bb[k, 'M5'] / bb[k, 'M0']:.0f}%). "
                + ("It loses nominal significance in M5 and in M6"
                   if loses else 'It remains nominal in at least one of M5 and M6')
                + (', like the bilateral putamen.' if loses == loses_bil
                   else ', unlike the bilateral putamen.'))
    for k in KIN_SCALE:
        log(f"\nRelative effect of {k} (% change of the mean slope per 1 SD; ratio of point estimates, without interval: the equality between the sides is not tested), {LBLS['24h']} cohort:")
        for q in DEFINITIONS:
            g = Ma[(Ma.cohort == "24h") & (Ma.definition == q)
                   & (Ma.kinetic == k)].set_index('side')
            if {'less', 'more', "diff"} <= set(g.index):
                log(f"  {q}: less affected {g.loc['less', 'effect_rel_pct']:+.1f}% (beta {g.loc['less', 'beta']:+.4f}, slope {g.loc['less', 'slope_mean']:+.4f}), more affected {g.loc['more', 'effect_rel_pct']:+.1f}% (beta {g.loc['more', 'beta']:+.4f}, slope {g.loc['more', 'slope_mean']:+.4f}); contrast between the sides beta {g.loc['diff', 'beta']:+.4f} [95% CI {g.loc['diff', 'ci_inf']:+.4f}; {g.loc['diff', 'ci_upp']:+.4f}], p = {fmt_p(g.loc['diff', 'p'])}")
        rb = Ma[(Ma.cohort == "24h") & (Ma.outcome == "MIA_PUTAMEN_BILAT") & (Ma.kinetic == k)]
        if len(rb):
            rb = rb.iloc[0]
            log(f'  bilateral putamen: {rb.effect_rel_pct:+.1f}% (beta {rb.beta:+.4f}, slope {rb.slope_mean:+.4f})')
    # conclusions of section 2.6 of the text
    log('\nDo the conclusions of the section on the longitudinal models change?')
    f24, f150 = fam["24h"], fam["150h"]
    m150 = Ma[(Ma.cohort == "150h") & (Ma.block == 'laterality')
              & Ma.side.isin(['less', 'more'])]
    n150_s = int((m150.p_BH_substitution < ALPHA).sum())
    n150_a = int((m150.p_BH_widened < ALPHA).sum())
    log(f"  (a) Replication at 150 h: {n150_s} lateralised interactions survive the correction in the substitution family and {n150_a} in the enlarged one (published: {f150['surv_pub']} of {f150['n_pub']}). "
        + ("The failure to replicate is confirmed." if n150_s == 0 and n150_a == 0 else
           "The failure to replicate is NOT confirmed with every definition: it must be reread."))
    counts = set()
    for side in ('less', 'more'):
        for q in DEFINITIONS:
            g = Ma[(Ma.cohort == "24h") & (Ma.side == side)
                   & (Ma.definition == q)].set_index('kinetic')
            if not len(g):
                continue
            ff = fam[("24h", q, side)]
            counts.add(ff['survivors'])
            log(f"  (b) Primary cohort, putamen {SIDES[side]} ({q}) in place of the bilateral: in all {ff['survivors']} of {ff['n']} survive (published {f24['surv_pub']} of {f24['n_pub']}), of which {ff['outside_fromthe_putamen']} outside the putamen; nominal interactions on this putamen: "
                + (", ".join(f"{k} BH p = {fmt_p(g.loc[k, 'p_BH_substitution'])}"
                             for k in g.index if g.loc[k, "p"] < ALPHA) or 'nothing') + ".")
    log("      "
        + ('The number of interactions that survive the correction does not depend on the putamen chosen.'
           if counts == {f24['surv_pub']} else
           f"The number of interactions that survive the correction DEPENDS on the putamen chosen (observed values: {sorted(counts)}; published {f24['surv_pub']}): the procedure is a step procedure, and the outcomes that remain in the family survive or not according to the p of the putaminal outcome that enters it."))
    for tag in COHORTS:
        log(f"  Enlarged family, {LBLS[tag]} cohort: survive the correction {fam[tag]['surv_all']} of {fam[tag]['n_all']}"
            + (f" ({'; '.join(fam[tag]['chi_all'])})" if fam[tag]["chi_all"] else "") + ".")
    for branch in ('less_affected', 'bilateral'):
        tot = sum(len(summary[(branch, tag)]['criteria']) for tag in COHORTS)
        tot_m = sum(len(summary[(branch, tag)]['criteria_mean']) for tag in COHORTS)
        log(f"  (c) Clustering, arm '{BRANCHES[branch][1]}': reproducibility criteria at k = 2 above the copula null in the two cohorts: {tot} (single stream), {tot_m} (PS and Jaccard at the mean of the streams). "
            + ("The primary rule, for the part at k = 2, is not satisfied."
               if tot == 0 and tot_m == 0 else
               'WARNING: at least one criterion of the primary rule exceeds the null.')
            + ' Other statistics above the null: '
            + "; ".join(f"{LBLS[tag]}: {', '.join(summary[branch, tag]['rest']) or 'nothing'}"
                        for tag in COHORTS) + ".")
    log(f'  (d) Silhouette at k = 2, {LBLS[COHORT_SIL]} cohort, null with B = {B_SIL}, p not corrected and outside the three reproducibility criteria: '
        + "; ".join(f"{r['variable_putaminal']}: p = {fmt_mc(r['p_numerator'], r['p_denominator'], r['p_montecarlo'], r['p_se_mc'])}"
                    for r in rows_s) + ". "
        + (f'It is nominally above the null with all {len(rows_s)} putaminal variables, that of the manuscript included: it does not depend on the choice of the putamen.' if len(sil_below) == len(rows_s) else
           f'It is nominally above the null with {len(sil_below)} putaminal variables of {len(rows_s)}.'))

    # ------------------------------------------------------------ output -----
    Mb.insert(0, 'population', POP)
    Mb.drop(columns=["spec", 'description'], errors="ignore").to_csv(
        os.path.join(OUT, "110_models_laterality.csv"), index=False)
    Left = S.copy()
    Left.insert(0, 'population', POP)
    Left.to_csv(os.path.join(OUT, "110_ladder_laterality.csv"), index=False)
    D = pd.DataFrame(rows_def)
    D.insert(0, 'population', POP)
    D.to_csv(os.path.join(OUT, "110_laterality_definitions.csv"), index=False)
    Bt = pd.DataFrame(rows_b)
    assert set(Bt.columns) == set(COL_B), 'unexpected columns in the battery table'
    Bt[COL_B].to_csv(os.path.join(OUT, "110_battery_k2.csv"), index=False)
    pd.DataFrame(rows_n).to_csv(os.path.join(OUT, "110_battery_k2_nulls.csv"), index=False)
    pd.DataFrame(rows_s).to_csv(os.path.join(OUT, "110_silhouette_k2_B4000.csv"),
                                 index=False)
    pd.DataFrame(rows_sn).to_csv(os.path.join(OUT, "110_silhouette_k2_B4000_null.csv"),
                                  index=False)
    log('\nSaved in outputs/: 110_dat_laterality.txt, 110_laterality_definitions.csv,')
    log('110_models_laterality.csv, 110_ladder_laterality.csv, 110_battery_k2.csv,')
    log('110_battery_k2_nulls.csv, 110_silhouette_k2_B4000.csv,')
    log('110_silhouette_k2_B4000_null.csv')
    with open(os.path.join(OUT, "110_dat_laterality.txt"), "w") as fh:
        fh.write("\n".join(rep))
    print(f'[run time: {time.time() - t_start:.0f} s]', flush=True)


if __name__ == "__main__":
    main()
