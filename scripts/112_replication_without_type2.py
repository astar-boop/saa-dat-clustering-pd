"""
112 - 150 h replication cohort without the two participants whom the 24 h
      assay later classified as Type2: sensitivity analysis at n = 331.

Why it exists. The 150 h protocol (Project 155) does not distinguish Type1 and
Type2: the replication cohort (n = 333, sporadic only) is made of the
participants positive at baseline. For two of them the CSF of the baseline
visit was later analysed with the 24 h protocol (Project 237) and classified
Positive/Type2 (109_coughlin_reconciliation.py). A reviewer will read
Type2 as a possible pattern of a non-PD synucleinopathy, and "the results do
not change" cannot be written without having recomputed them. This script
recomputes them.

What it does.

  1. It identifies the two BY RULE from the raw assay file: member of the
     150 h cohort whose 24 h result run last at the baseline visit is
     Positive/Type2. No PATNO is written in the code. The set is
     compared by identity of PATNO with the one found by script 109
     (109_reconciliation_subjects.csv), if that file exists.
  2. It describes them without identifying them in the csv files: position on
     the seven clustering variables and on the five kinetic parameters (z within
     the cohort), duration of follow-up, rows of the raw file at later visits.
  3. It counts how many of the 333 (and of the remaining 331) have a 24 h result
     at baseline or later, and with what result: it says how many of the 331 are
     NOT typed.
  4. It recomputes, at n = 333 and at n = 331, the results of the replication
     cohort that the manuscript reports:
       (a) the battery at k = 2 against the Gaussian copula null, with the
           functions, the seed and the B of script 25 (Table S5), plus the
           designated observed estimates of script 08 (prediction strength,
           Jaccard, gap, dip) with which script 41 composes the table;
       (b) the family of the 35 kinetic parameter x time tests of the 150 h
           cohort (seven outcomes x five kinetic parameters), under the strict
           criterion;
       (c) the twelve primary non-motor outcomes (T50 x years, Table S9), under
           the strict criterion;
       (d) the battery without the participants shared with the primary cohort
           (script 66, §1.8: n = 288, which becomes 286 if the two are among
           the 288);
       (e) the CSF analyses that use this cohort: the 35 kinetic
           parameter-analyte pairs at the three levels of adjustment (script
           12), the leave-one-analyte-out component (script 36), albumin
           (script 20).
     The n = 333 branch must REPRODUCE the values already saved by the pipeline
     (identity of values, not of counts) before the n = 331 branch is
     used: if it does not reproduce them the script stops without writing
     anything.

What changes at n = 331 besides the two subjects, and it must be said: the
standardisation of the variables is redone on the 331, the copula null is
RE-ESTIMATED on the 331 (marginals and correlations of the 331), the common
component of the analytes is recomputed on the 331. The coefficients of the
mixed models are therefore "per 1 SD of the logarithm of the kinetic parameter
in the population of the branch"; the script also reports the coefficient at
331 rescaled to the SD of the 333, which is an exact rescaling, to separate
the effect of the rescaling from that of the removal of the data. Every
coefficient that shifts by at least 0.10 standard errors is then re-estimated
removing one participant at a time (n = 332), to say to which of the two the
shift must be attributed and why.

Mixed models. Random intercept and random slope per subject, REML, cascade of
_fit.OTTIMIZZATORI. Strict criterion of the project (84, 87, 88, 91, 107):
among the optimisers that converge the highest log-likelihood, restarts
until none gains, Newton decrement < 1e-4, negative definite profiled
Hessian, no variance on the boundary. A model that does not satisfy the
criterion is reported as NOT accepted and excluded from the correction, never
used silently.

Guards. ONLY_SPORADIC and the working directory must agree and the working
directory must be that of the sporadic only. The cohort read from the working
directory must coincide, by identity of PATNO and of values, with the one
rebuilt from the raw file and from the curated data cut. A guard that cannot
run (reference file absent) writes so with "NON verificat" and appears in the
list at the end of the report.

Computation: at most 4 processes, one thread per process. Two executions give
outputs identical byte for byte (fixed seeds, no time written in the files).

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 -B scripts/112_replication_without_type2.py
Output: outputs/112_replication_without_type2.txt, 112_two_participants.csv,
        112_typing_24h.csv, 112_battery.csv, 112_battery_nulls.csv,
        112_kinetics_by_time.csv, 112_not_motor.csv, 112_attribution.csv,
        112_csf_pairs.csv, 112_csf_albumin.csv, 112_summary.csv
"""

import os

# One thread per process, set BEFORE importing numpy: the processes are at
# most four and the result must not depend on the number of cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import hashlib                                                   # noqa: E402
import multiprocessing as mp                                     # noqa: E402
import re                                                        # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
import warnings                                                  # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
import statsmodels.api as sm                                     # noqa: E402
import statsmodels.formula.api as smf                            # noqa: E402
from scipy import stats                                          # noqa: E402
from scipy.spatial import cKDTree                                # noqa: E402
from scipy.spatial.distance import cdist                         # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans      # noqa: E402
from sklearn.decomposition import PCA                            # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture                      # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402
from statsmodels.stats.multitest import multipletests            # noqa: E402
import diptest                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS                                   # noqa: E402
from _paths import DATA                                       # noqa: E402
import _dat                                                      # noqa: E402

warnings.filterwarnings("ignore")

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
# sibling working directory with the "caudate" versions of the longitudinal
# analyses (84/87/88): that is where the family of the 35 tests reported by the
# manuscript is
OUT_CAUDATE = os.path.join(os.path.dirname(os.path.normpath(BASE)),
                           'run_caudate', "outputs")
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
CACHE = os.path.join(OUT, "cache", "curated.parquet")
CACHE_ALB = os.path.join(OUT, "cache", "alb.csv")
SPOR = "Sporadic PD"
N_PROCESSES = 4

# ---- constants of the battery: those of scripts 25, 66 and 08. They are copied,
#      and the n = 333 branch checks by identity of values that they are those.
SEED25 = 20260805        # scripts 25 and 66
B25 = 300                # realisations of the null, 150 h cohort, script 25
B66 = 200                # realisations of the null, script 66 (§1.8)
N_PROJ = 200             # random projections in the battery
N_SPLIT_PS = 50          # partitions for the prediction strength
SEED08 = 20260804        # script 08 (designated observed estimates)
KRANGE = list(range(2, 11))
N_PROJ_08 = 1000

# ---- thresholds
ALPHA = 0.05
Z95 = 1.96
TOL_NEWTON = 1e-4
TOL_SINGULAR = 1e-8
TOL_CORR = 0.999
TOL_GAIN = 1e-10
MAX_ROUNDS = 10
TOL_COHORT = 1e-9        # relative: cohort read against cohort rebuilt
TOL_ID_BATT = 1e-12      # absolute: battery and correlations against the saved values
TOL_ID_MIXED = 1e-8      # absolute: mixed models against the saved values (the
                         # same as in 107_nonmotor_ledd_duration.py)
THRESHOLD_ATTRIB = 0.10     # shift, in SE of the 333 branch, from which a coefficient
                         # is ATTRIBUTED: it is re-estimated removing one
                         # participant at a time

# ---- families of tests (they are the design of the manuscript, not a result)
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}
KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE", "dat_putamen_low",
        "dat_caudate_mean", "age_bl"]          # the seven variables, script 25
NAME7 = {"k_Fmax": "log Fmax", "k_T50": "log T50", "k_AUC": "log AUC",
         "k_SLOPE": "log SLOPEmax", "dat_putamen_low": 'DAT putamen (lower side)',
         "dat_caudate_mean": 'DAT caudate (mean)', "age_bl": 'baseline age'}
# seven outcomes x five kinetic parameters = 35 tests (version with the caudate,
# §1.12)
OUTCOMES35 = {
    "updrs3_score": ("MDS-UPDRS III", ["age_bl", "SEX", "dur_bl"]),
    "updrs_totscore": ('MDS-UPDRS total', ["age_bl", "SEX", "dur_bl"]),
    "updrs2_score": ("MDS-UPDRS II", ["age_bl", "SEX", "dur_bl"]),
    "moca": ("MoCA", ["age_bl", "SEX", "dur_bl", "EDUCYRS"]),
    "MIA_PUTAMEN_BILAT": ('DAT bilateral putamen', ["age_bl", "SEX", "dur_bl"]),
    "MIA_CAUDATE_BILAT": ('DAT bilateral caudate', ["age_bl", "SEX", "dur_bl"]),
    "scopa": ("SCOPA-AUT", ["age_bl", "SEX", "dur_bl"]),
}
# the twelve primary non-motor outcomes (15_nonmotor_trajectories.py, Table S9)
OUTCOMES12 = [
    ("moca", 'Global cognition (MoCA)', True),
    ("hvlt_immediaterecall", 'Memory (HVLT)', True),
    ("SDMTOTAL", 'Executive/attention (SDMT)', True),
    ("VLTANIM", 'Semantic fluency', True),
    ("bjlot", 'Visuospatial (BJLOT)', True),
    ("updrs1_score", 'Global non-motor (MDS-UPDRS I)', False),
    ("gds", 'Depression (GDS-15)', False),
    ("stai", 'Anxiety (STAI)', False),
    ("ess", 'Sleepiness (Epworth)', False),
    ("rem", "RBD (RBDSQ)", False),
    ("scopa", 'Autonomic (SCOPA-AUT)', False),
    ("quip", 'Impulse control (QUIP)', False),
]
ANALYTES = ["abeta", "tau", "ptau", "asyn", "IU_pTau181_CSF",
           "IU_ABeta40_CSF", "IU_ABeta42_CSF"]
STAT = ["hopkins", "silhouette_k2", 'n_projections_multimodal', "dip_pmin_bonf",
        "PS_k2", "jaccard_k2", 'ARI_algorithms_k2']
HIGH = {"hopkins": True, "silhouette_k2": True, 'n_projections_multimodal': True,
        "dip_pmin_bonf": False, "PS_k2": True, "jaccard_k2": True,
        'ARI_algorithms_k2': True}
NAME_STAT = {"hopkins": "Hopkins", "silhouette_k2": "Silhouette (k = 2)",
             'n_projections_multimodal': 'Multimodal projections (of 200)',
             "dip_pmin_bonf": 'Minimum dip (Bonferroni)',
             "PS_k2": "Prediction strength (k = 2)",
             "jaccard_k2": "Jaccard (k = 2)",
             'ARI_algorithms_k2': 'ARI between algorithms (k = 2)'}

rep = []
no_executed = []     # expected guards that could not run in this run


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def no_verified(what):
    no_executed.append(what)


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def listing(s):
    s = list(s)
    return ", ".join(str(x) for x in s) if s else 'noone'


def count(series):
    c = series.value_counts()
    return {k: int(c[k]) for k in sorted(c.index, key=str)}


def load_csv(path, **kw):
    """Reading with exact conversion of the decimals: the identity comparisons
    must not depend on the rounding of the fast pandas parser."""
    return pd.read_csv(path, float_precision="round_trip", **kw)


def bh(p):
    return multipletests(np.asarray(p, float), method="fdr_bh")[1]


# ============================================================================
# BATTERY: the functions of script 25 (identical in 66), copied because those
# modules execute at import time. The n = 333 branch checks their
# identity on the saved values.
# ============================================================================
def copula_null(X, rng):
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
    c, pmin = 0, 1.0
    for _ in range(n_proj):
        a = rng.normal(size=X.shape[1])
        a /= np.linalg.norm(a)
        p = diptest.diptest(X @ a)[1]
        c += p < 0.05
        pmin = min(pmin, p)
    return c, pmin


def ps_k2(X, rng, n_rep=N_SPLIT_PS):
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
    ref = KMeans(2, n_init=15, random_state=SEED25).fit_predict(X)
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
    labs = {
        "kmeans": KMeans(2, n_init=20, random_state=SEED25).fit_predict(X),
        "ward": AgglomerativeClustering(2, linkage="ward").fit_predict(X),
        "average": AgglomerativeClustering(2, linkage="average").fit_predict(X),
        "gmm": GaussianMixture(2, covariance_type="full", n_init=5,
                               random_state=SEED25).fit_predict(X)}
    ks = list(labs)
    return float(np.mean([adjusted_rand_score(labs[ks[i]], labs[ks[j]])
                          for i in range(4) for j in range(i + 1, 4)]))


def battery(X, rng):
    nm, pmin = n_multimodal(X, rng)
    return {"hopkins": hopkins(X, rng),
            "silhouette_k2": float(silhouette_score(
                X, KMeans(2, n_init=15, random_state=SEED25).fit_predict(X))),
            'n_projections_multimodal': float(nm),
            "dip_pmin_bonf": float(min(1.0, pmin * N_PROJ)),
            "PS_k2": ps_k2(X, rng),
            "jaccard_k2": jaccard_k2(X, rng),
            'ARI_algorithms_k2': ari_algos(X)}


def standardise25(M):
    """Standardisation of script 25: mean and SD (ddof = 0) of the population."""
    return (M - M.mean(0)) / M.std(0)


def branch_batt(M, B):
    """Observed battery and B realisations of the copula null, ESTIMATED on the
    matrix passed. A single generator, in the order of script 25."""
    X = standardise25(M)
    rng = np.random.default_rng(SEED25)
    obsv = battery(X, rng)
    nulls = [battery(copula_null(X, rng), rng) for _ in range(B)]
    return {'obsv': obsv, 'nulls': nulls}


# ---------------------------------------------------------------------------
# Designated observed estimates: the functions of script 08 (seed 20260804).
# ---------------------------------------------------------------------------
def fit_labels08(Xm, k):
    return KMeans(k, n_init=50, random_state=SEED08).fit_predict(Xm)


def wk(Xm, lab):
    return sum(((Xm[lab == c] - Xm[lab == c].mean(0)) ** 2).sum()
               for c in np.unique(lab) if (lab == c).sum() > 1)


def gap_stat08(Xm, B=200):
    r = np.random.default_rng(SEED08)
    ks = [1] + KRANGE
    pca = PCA().fit(Xm)
    Xp = pca.transform(Xm)
    lo, hi = Xp.min(0), Xp.max(0)
    gap, sk = [], []
    for k in ks:
        lab = np.zeros(len(Xm), int) if k == 1 else fit_labels08(Xm, k)
        lw = np.log(wk(Xm, lab))
        bs = []
        for _ in range(B):
            Zb = pca.inverse_transform(r.uniform(lo, hi, size=Xp.shape))
            lb = np.zeros(len(Zb), int) if k == 1 else KMeans(
                k, n_init=10, random_state=int(r.integers(1e6))).fit_predict(Zb)
            bs.append(np.log(wk(Zb, lb)))
        bs = np.array(bs)
        gap.append(bs.mean() - lw)
        sk.append(bs.std() * np.sqrt(1 + 1 / B))
    return ks, [float(g) for g in gap], [float(s) for s in sk]


def k_gap(ks, gap, sk):
    """One-standard-error rule, as in 08: the first k with
    gap(k) >= gap(k+1) - s(k+1)."""
    return next((ks[i] for i in range(len(ks) - 1)
                 if gap[i] >= gap[i + 1] - sk[i + 1]), None)


def ps08_k2(Xm, n_rep=50):
    """pred_strength of script 08 at k = 2: it is the first k of the loop, so it
    uses the first draws of the same generator."""
    r = np.random.default_rng(SEED08)
    vals = []
    for _ in range(n_rep):
        perm = r.permutation(len(Xm))
        a, b = perm[: len(Xm) // 2], perm[len(Xm) // 2:]
        ka = KMeans(2, n_init=20, random_state=int(r.integers(1e6))).fit(Xm[a])
        lb = KMeans(2, n_init=20, random_state=int(r.integers(1e6))).fit_predict(Xm[b])
        pred = np.argmin(cdist(Xm[b], ka.cluster_centers_), axis=1)
        worst = 1.0
        for c in range(2):
            mk = lb == c
            nc = mk.sum()
            if nc <= 1:
                worst = 0.0
                continue
            cnt = np.bincount(pred[mk], minlength=2)
            worst = min(worst, (cnt * (cnt - 1)).sum() / (nc * (nc - 1)))
        vals.append(worst)
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(n_rep))


def boot_jaccard08(Xm, k=2, B=200):
    r = np.random.default_rng(SEED08)
    ref = KMeans(k, n_init=25, random_state=SEED08).fit_predict(Xm)
    best = {c: [] for c in range(k)}
    for _ in range(B):
        ib = r.choice(len(Xm), len(Xm), replace=True)
        km = KMeans(k, n_init=10, random_state=int(r.integers(1e6))).fit(Xm[ib])
        lab = np.argmin(cdist(Xm, km.cluster_centers_), axis=1)
        for c in range(k):
            rs = set(np.where(ref == c)[0])
            best[c].append(max(
                (len(rs & set(np.where(lab == cb)[0])) /
                 len(rs | set(np.where(lab == cb)[0])) for cb in range(k)),
                default=0.0))
    return [float(np.mean(best[c])) for c in range(k)]


def proj_dip08(Xm, n_proj=N_PROJ_08):
    r = np.random.default_rng(SEED08)
    pv = np.empty(n_proj)
    for i in range(n_proj):
        a = r.normal(size=Xm.shape[1])
        a /= np.linalg.norm(a)
        pv[i] = diptest.diptest(Xm @ a)[1]
    return pv


def branch_08(M):
    """The designated observed estimates of script 08 on the matrix passed."""
    X = StandardScaler().fit_transform(M)
    ps, ps_se = ps08_k2(X)
    jac = boot_jaccard08(X)
    ks, gap, sk = gap_stat08(X)
    PCs = PCA().fit_transform(X)
    dip_var = [[float(v) for v in diptest.diptest(X[:, j])] for j in range(X.shape[1])]
    dip_pc = [[float(v) for v in diptest.diptest(PCs[:, j])] for j in range(X.shape[1])]
    pv = proj_dip08(X)
    return {"PS_k2": ps, 'PS_k2_se_mc': ps_se,
            'jaccard_avg': float(np.mean(jac)), "jaccard_min": float(np.min(jac)),
            "gap_k": ks, "gap": gap, "gap_sk": sk, "k_gap": k_gap(ks, gap, sk),
            "dip_var": dip_var, "dip_pc": dip_pc,
            "proj_n_sig": int((pv < 0.05).sum()),
            "proj_pmin_bonf": float(min(1.0, pv.min() * N_PROJ_08))}


# ============================================================================
# MIXED MODELS: the primitives of the strict criterion, copied from
# 107_nonmotor_ledd_duration.py (in turn from 91 and from 88) without changes of
# substance. The n = 333 branch checks that they give the numbers already saved.
# ============================================================================
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
    """Cascade, selection on the likelihood among the convergent ones, restarts."""
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
            'r_final': best[0], 'opt_reopt': best[1], 'rounds': rounds,
            'md_ref': md_ref}


FIELDS_FIT = ["beta", "se", "p", 'slope_mean', 'optimiser_chosen',
             'optimiser_restart', 'rounds_restart',
             'atleast_an_convergent', "loglike", "grad_inf_norm", "newton",
             'hessian_def_negative', 'var_intercept', 'var_slope',
             'at_edge', 'singular', 'criterion_newton', 'accepted']


def fit_strict(f, data, kinet):
    """One model under the strict criterion. Returns a dictionary with
    `estimated` False if no optimiser produces a fit."""
    P = procedure(f, data, data.PATNO)
    if P is None:
        return {'estimated': False, 'accepted': False}
    r, md_ref = P['r_final'], P['md_ref']
    md = r.model
    p = pack(r)
    g = _score(md, p)
    H, _f0 = hess_profiled(md, p)
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
    names = (f'{kinet}:years', f'years:{kinet}')
    t = next(c for c in names if c in r.params.index)
    at_edge = bool(v_int == 0.0 or v_slope == 0.0)
    def_neg = bool(eig.max() < 0)
    crit = bool(np.isfinite(newton) and newton < TOL_NEWTON)
    return {
        'estimated': True,
        "beta": float(r.params[t]), "se": float(r.bse[t]),
        "p": float(r.pvalues[t]), 'slope_mean': float(r.params["years"]),
        'optimiser_chosen': P['opt_chosen'],
        'optimiser_restart': P['opt_reopt'],
        'rounds_restart': int(P['rounds']),
        'atleast_an_convergent': bool(P['atleast_an_convergent']),
        "loglike": _ll(md_ref, p),
        "grad_inf_norm": float(np.max(np.abs(g))),
        "newton": newton, 'hessian_def_negative': def_neg,
        'var_intercept': v_int, 'var_slope': v_slope,
        'at_edge': at_edge,
        'singular': bool(v_int < TOL_SINGULAR * scale
                          or v_slope < TOL_SINGULAR * scale
                          or (np.isfinite(corr) and abs(corr) >= TOL_CORR)),
        'criterion_newton': crit,
        # accepted only if an optimiser has declared convergence, the
        # point is stationary, the curvature is that of a maximum and no
        # variance is on the boundary
        'accepted': bool(P['atleast_an_convergent'] and crit and def_neg
                          and not at_edge),
    }


# ============================================================================
# One unit of work for the pool (it must be at module level)
# ============================================================================
def execute(job):
    kind, key, data = job
    t0 = time.time()
    if kind == 'batt':
        result = branch_batt(data["M"], data["B"])
    elif kind == "designate":
        result = branch_08(data["M"])
    elif kind == 'mixed':
        result = fit_strict(data["formula"], data["m"], data['kinet'])
    else:
        raise ValueError(kind)
    return kind, key, result, time.time() - t0


# ============================================================================
# CSF analyses: scripts 12 (three levels), 36 (leave-one-out), 20 (albumin)
# ============================================================================
def partial(y, x, Z):
    """Spearman partial correlation: Pearson between the residuals of the ranks."""
    Zc = sm.add_constant(np.asarray(Z, float))
    ry = sm.OLS(stats.rankdata(y), Zc).fit().resid
    rx = sm.OLS(stats.rankdata(x), Zc).fit().resid
    r, p = stats.pearsonr(ry, rx)
    return float(r), float(p)


def prepare_csf(cohort, kin_map, bl, alb):
    """Like `prep` of scripts 12 and 20: log of the kinetic parameters + analytes,
    age and sex from the baseline of the curated data cut + CSF albumin."""
    k = cohort[["PATNO"] + list(kin_map)].rename(columns=kin_map)
    for c in kin_map.values():
        k[c] = np.log(k[c])
    cols = ["PATNO", "age_at_visit", "SEX"] + ANALYTES
    m = k.merge(bl[cols].rename(columns={"age_at_visit": "age_bl"}),
                on="PATNO", how="left", validate="one_to_one")
    if alb is not None:
        m = m.merge(alb, on="PATNO", how="left", validate="one_to_one")
    return m.reset_index(drop=True)


def factor12(d):
    """`concentration_factor` of script 12: PC1 of the ranks of the analytes."""
    avail = [a for a in ANALYTES if d[a].notna().sum() >= 150]
    m = d[avail].dropna()
    R = np.column_stack([stats.rankdata(m[c]) for c in avail])
    R = (R - R.mean(0)) / R.std(0)
    pca = PCA().fit(R)
    f = pd.Series(PCA(n_components=1).fit_transform(R).ravel(), index=m.index)
    if stats.spearmanr(f, m[avail[0]]).statistic < 0:
        f = -f
    out = pd.Series(np.nan, index=d.index)
    out.loc[m.index] = f
    return out, avail, {"n": int(len(m)),
                        "var_pc1": float(pca.explained_variance_ratio_[0]),
                        "loading_min": float(np.abs(pca.components_[0]).min()),
                        "loading_max": float(np.abs(pca.components_[0]).max()),
                        'loadings_same_sign': bool(
                            (pca.components_[0] > 0).all()
                            or (pca.components_[0] < 0).all())}


def pairs12(d, kins, avail):
    """Section c of script 12: L1 raw, L2 + age and sex, L3 + component."""
    rows = []
    for k in kins:
        for b in avail:
            m = d[[k, b, "age_bl", "SEX", "conc"]].dropna()
            if len(m) < 60:
                continue
            r1, p1 = stats.spearmanr(m[k], m[b])
            r2, p2 = partial(m[b], m[k], m[["age_bl", "SEX"]].to_numpy(float))
            r3, p3 = partial(m[b], m[k], m[["age_bl", "SEX", "conc"]].to_numpy(float))
            rows.append({'kinetic': k, 'biomarker': b, "n": len(m),
                         "rho_L1": float(r1), "p_L1": float(p1),
                         "rho_L2": r2, "p_L2": p2, "rho_L3": r3, "p_L3": p3})
    cc = pd.DataFrame(rows)
    for lv in ("L1", "L2", "L3"):
        cc[f"pBH_{lv}"] = bh(cc[f"p_{lv}"])
    return cc


def factor36(M, cols):
    R = np.column_stack([stats.rankdata(M[c]) for c in cols])
    R = (R - R.mean(0)) / R.std(0)
    f = PCA(n_components=1).fit_transform(R).ravel()
    if np.mean([stats.spearmanr(f, M[c]).statistic for c in cols]) < 0:
        f = -f
    return f


def loo36(d, kins):
    """Script 36: component recomputed, for each pair, without the analyte."""
    M = d.dropna(subset=ANALYTES + ["age_bl", "SEX"])
    f_full = factor36(M, ANALYTES)
    rows = []
    for a in ANALYTES:
        others = [x for x in ANALYTES if x != a]
        f_loo = factor36(M, others)
        rho_ff = float(stats.spearmanr(f_full, f_loo).statistic)
        for kn in kins:
            lr, lp = partial(M[kn], M[a],
                              np.column_stack([M[["age_bl", "SEX"]], f_loo]))
            rows.append({'kinetic': kn, 'biomarker': a, "n_loo": len(M),
                         "rho_LOO": lr, "p_LOO": lp, 'rho_factors': rho_ff})
    res = pd.DataFrame(rows)
    res["pBH_LOO"] = bh(res.p_LOO)
    return res


def kinet_vs(d, kins, what, n_min):
    """partial rho (age, sex) between each kinetic parameter and `what` (scripts 12 and 20)."""
    rows = []
    for k in kins:
        m = d[[k, what, "age_bl", "SEX"]].dropna()
        if len(m) < n_min:
            continue
        r, p = partial(m[what], m[k], m[["age_bl", "SEX"]].to_numpy(float))
        rows.append({'kinetic': k, "n": len(m), 'rho_partial': r, "p": p})
    return pd.DataFrame(rows)


# ============================================================================
def main():
    T0 = time.time()
    title('112 — 150 h COHORT WITHOUT THE TWO Type2 PARTICIPANTS (n = 331)')
    log(f"statsmodels {__import__('statsmodels').__version__}, numpy {np.__version__}, pandas "
        f"{pd.__version__}, scikit-learn {__import__('sklearn').__version__}, scipy "
        f"{__import__('scipy').__version__}")
    log(f'optimiser cascade: {OPTIMISERS}')
    log(f"processes: {N_PROCESSES}; threads per process: {os.environ['OMP_NUM_THREADS']}")

    # ========================================================================
    title('0. FILES, WORKING DIRECTORY, POPULATION')
    assert _dat.variables() == F150[4:6], (
        f'DAT_SET = {_dat.choice()}: the reference values of the pipeline are those of the default '
        f'set {F150[4:6]}; the script is not defined elsewhere')

    sa = pd.read_csv(SAA, low_memory=False)
    assert not sa.duplicated().any(), 'duplicate rows in the SAA file'
    sa["PATNO"] = sa["PATNO"].astype("int64")
    sa["RUNDATE"] = pd.to_datetime(sa["RUNDATE"])
    assert sa.RUNDATE.notna().all(), 'missing run dates in the SAA file'
    assert set(sa.PROJECTID.unique()) == {155, 237}, 'unexpected projects in the SAA file'
    assert set(sa.SAA_Status.unique()) == {"Positive", "Negative", "Inconclusive"}, \
        'unexpected outcomes in the SAA file'
    sa['row'] = np.arange(len(sa))

    assert os.path.exists(CACHE), f'cache of the curated data cut missing: {CACHE}'
    cur = pd.read_parquet(CACHE)
    cur["PATNO"] = cur["PATNO"].astype("int64")
    assert not cur.duplicated(["PATNO", "EVENT_ID"]).any(), 'duplicate visits in the curated data cut'
    blc = cur[cur.EVENT_ID == "BL"]
    assert not blc.PATNO.duplicated().any(), "more than one baseline row per subject"
    sub = blc.set_index("PATNO").subgroup
    sporadic = set(sub[sub == SPOR].index)

    # The cohort files are read as the pipeline reads them (default pandas
    # parser, order of the rows of the file): the battery depends on the order
    # of the rows and on the last digit of the values, and the n = 333 branch must
    # start again from exactly the same matrix as scripts 25, 66 and 08.
    c150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
    c150["PATNO"] = c150["PATNO"].astype("int64")
    if "complete" in c150.columns:
        c150 = c150[c150.complete]
    c150 = c150.reset_index(drop=True)
    c24 = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
    c24["PATNO"] = c24["PATNO"].astype("int64")
    if "complete_primary" in c24.columns:
        c24 = c24[c24.complete_primary]
    c24 = c24.reset_index(drop=True)
    assert c150.PATNO.is_unique and c24.PATNO.is_unique, 'repeated PATNO in the cohort files'
    P333, P24 = set(c150.PATNO), set(c24.PATNO)
    assert (P333 | P24) <= set(sub.index), 'subjects of the cohorts missing from the baseline curated data cut'
    only150, only24 = not (P333 - sporadic), not (P24 - sporadic)
    assert only150 == only24, ("the two cohorts of the working directory are not of the same kind: "
                               "one is sporadic only, the other is not")
    ROOT = 'sporadic only' if only150 else 'full cohort'
    assert ONLY == only150, (f"ONLY_SPORADIC={('1' if ONLY else '0')} but the working directory is "
                             f"'{ROOT}': variable and working directory do not agree")
    if not ONLY:
        sys.exit('112: the analysis is defined for the sporadic population (the reference values '
                 'are those of the sporadic-only working directory). Rerun with ONLY_SPORADIC=1 on '
                 'that working directory.')
    assert (c150.subgroup == SPOR).all(), '08_dataset_150h.csv contains non-sporadic subjects'

    for name, fpath in (('assay (24 and 150 h)', SAA), ('curated data cut (cache)', CACHE),
                       ('150 h cohort', os.path.join(OUT, "08_dataset_150h.csv")),
                       ('24 h cohort', os.path.join(OUT, "analysis_complete_cases.csv"))):
        log(f'{name}: {os.path.basename(fpath)}, {os.path.getsize(fpath)} byte, sha256 '
            f'{fingerprint(fpath)}')
    log(f"working directory: {os.path.basename(os.path.normpath(BASE))} ({ROOT}); ONLY_SPORADIC = 1; "
        f"150 h cohort n = {len(P333)}, 24 h cohort n = {len(P24)} (participants), all with subgroup "
        f"= '{SPOR}'")

    # --- cache of the curated data cut: it is not compared again here with the xlsx;
    #     it is checked to be, byte for byte, the one that 109 compared.
    log109 = os.path.join(OUT, "109_coughlin_reconciliation.txt")
    h_cache = fingerprint(CACHE)
    state = None
    if os.path.exists(log109):
        with open(log109) as fh:
            t109 = fh.read()
        mh = re.search(r"^curated: .*sha256 ([0-9a-f]{64})\s*$", t109, flags=re.M)
        compared = "Guard, cache of the curated data cut: compared with" in t109
        if mh and compared:
            assert mh.group(1) == h_cache, (
                f"the cache of the curated data cut read here is not the one that 109 compared with "
                f"the xlsx (sha256 {h_cache} against {mh.group(1)})")
            state = True
    if state:
        log('Guard, cache of the curated data cut: same sha256 fingerprint as the cache that script 109')
        log("  compared with the xlsx, cell by cell, on the 13 columns it reads (assay")
        log("  outcome, subgroup, age, DAT-SPECT). The columns of the longitudinal outcomes,")
        log('  of the covariates and of the CSF analytes read here are NOT covered by that')
        log("  comparison: for those the check is the reproduction of the values saved by the")
        log('  pipeline (sections 3-7), which reads the same cache.')
    else:
        log("Guard, cache of the curated data cut: NOT verified against the xlsx (the report of 109 is")
        log("  missing or does not report the comparison). The only check is the fingerprint above.")
        no_verified("cache of the curated data cut: correspondence with the xlsx (report of 109 "
                       "missing or without comparison)")

    # --- GUARD: the cohort read is the one rebuilt from the raw files
    #     with the rules of 08_replication_150h.py. Identity of PATNO and of values.
    p = sa[(sa.PROJECTID == 155) & (sa.COHORT == "PD") & (sa.CLINICAL_EVENT == "BL")].copy()
    for f in KIN150.values():
        v = p[[f"{f}_150h_Rep{i}" for i in (1, 2, 3)]].apply(pd.to_numeric, errors="coerce")
        p[f"k_{f}"] = v.median(axis=1)               # median of the three replicates
    pos = p[p.SAA_Status == "Positive"]
    ric = pos.groupby("PATNO")[list(KIN150)].mean()  # mean over the positive samples
    ric = ric.join(blc.set_index("PATNO")[["subgroup", "age_at_visit", "MIA_PUTAMEN_L",
                                           "MIA_PUTAMEN_R", "MIA_CAUDATE_BILAT"]], how="left")
    ric["dat_putamen_low"] = ric[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
    ric["dat_caudate_mean"] = ric["MIA_CAUDATE_BILAT"]
    ric["age_bl"] = ric["age_at_visit"]
    ric = ric[ric.subgroup == SPOR]
    with np.errstate(divide="ignore", invalid="ignore"):
        completes = (np.log(ric[["k_Fmax", "k_T50", "k_AUC", "k_SLOPE"]]).notna().all(axis=1)
                    & ric[["dat_putamen_low", "dat_caudate_mean", "age_bl"]].notna().all(axis=1))
    ric = ric[completes]
    R333 = set(ric.index)
    assert R333 == P333, (
        f'the 150 h cohort of the working directory is not the one reconstructed from the raw files: '
        f'only in the working directory {listing(sorted(P333 - R333))}; only in the reconstruction '
        f'{listing(sorted(R333 - P333))}')
    COL_VAL = list(KIN150) + ["dat_putamen_low", "dat_caudate_mean", "age_bl"]
    a = ric.loc[c150.PATNO, COL_VAL].to_numpy(float)
    b = c150[COL_VAL].to_numpy(float)
    assert np.isfinite(a).all() and np.isfinite(b).all(), 'non-finite values in the cohort'
    departure = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    if not np.allclose(a, b, rtol=TOL_COHORT, atol=0):
        i, j = np.unravel_index(np.argmax(departure), departure.shape)
        raise AssertionError(
            f'150 h cohort: values different from those reconstructed from the raw files (maximum '
            f'relative discrepancy {departure.max():.3g}, PATNO {int(c150.PATNO.iloc[i])}, column '
            f'{COL_VAL[j]})')
    log(f'Guard, 150 h cohort: the {len(P333)} PATNO of 08_dataset_150h.csv are those that')
    log(f'  are reconstructed from the assay file and the curated data cut with the rules of 08, and '
        f'the {len(COL_VAL)} columns')
    log(f"  ({', '.join(COL_VAL)}) match")
    log(f'  ({a.shape[0]} x {a.shape[1]} values, maximum relative discrepancy {departure.max():.2g}, '
        f'tolerance {TOL_COHORT:g}).')

    # ========================================================================
    title('1. THE TWO PARTICIPANTS, IDENTIFIED BY RULE')
    log('Rule: member of the 150 h cohort whose 24 h result (Project 237) run')
    log("LAST at the baseline visit (BL) is Positive/Type2. Unit: participants.")
    h = sa[(sa.PROJECTID == 237) & sa.PATNO.isin(P333)].copy()
    h['outcome'] = np.select(
        [h.SAA_Status == "Negative", h.SAA_Status == "Inconclusive",
         h.SAA_Type == "Type1", h.SAA_Type == "Type2"],
        ['negative', 'inconclusive', "Type1", "Type2"],
        default='positive, undetermined type')
    key = ["PATNO", "CLINICAL_EVENT"]
    mx = h.groupby(key, sort=False).RUNDATE.transform("max")
    ties = h[h.RUNDATE == mx].groupby(key, sort=False).agg(
        k=('row', "size"), n_se=('outcome', "nunique"))
    assert int(((ties.k > 1) & (ties.n_se > 1)).sum()) == 0, (
        "visits with discordant outcomes on the last 24 h run date: the rule does not decide")
    v24 = h.sort_values(key + ["RUNDATE", 'row']).drop_duplicates(key, keep="last")
    bl24 = v24[v24.CLINICAL_EVENT == "BL"]
    TWO = set(bl24[bl24.outcome == "Type2"].PATNO)
    assert TWO <= P333
    log(f'Found by the rule: {len(TWO)} (PATNO {listing(sorted(TWO))}; the PATNO appear only in')
    log('this report, for internal checking; in the csv files they appear as A, B, ...).')
    assert len(TWO) > 0, ("the rule finds no participant: there is no sensitivity analysis to run")

    # --- GUARD: identity with the set found by script 109
    f109 = os.path.join(OUT, "109_reconciliation_subjects.csv")
    if os.path.exists(f109):
        s109 = pd.read_csv(f109)
        s109["PATNO"] = s109["PATNO"].astype("int64")
        q = s109[s109.cohort_150h.astype(bool)
                 & ~s109.row_sPD_pos_earliest_visit.astype(bool)]
        TWO109 = set(q.PATNO)
        assert TWO == TWO109, (
            f'the participants found by the rule are not those of script 109: only here '
            f'{listing(sorted(TWO - TWO109))}; only in 109 {listing(sorted(TWO109 - TWO))}')
        selfsame = bool((q.earliest_visit_code == "BL").all()
                      and (q.earliest_visit_assay == "24h").all()
                      and (q.earliest_visit_outcome == "type2").all())
        log(f'Guard, identity with script 109: the {len(TWO)} PATNO match the members of the 150 h '
            f'cohort')
        log('  that 109_reconciliation_subjects.csv records as not positive/Type I at the first '
            'visit'
            + (' (in 109: first visit BL, 24 h assay, outcome Type II for all).'
               if selfsame else
               '; WARNING: in 109 not all have first visit BL at 24 h with outcome Type II, so the '
               'two sets match but not by the same rule.'))
    else:
        log("Guard, identity with script 109: NOT verified, 109_reconciliation_subjects.csv is "
            "missing from the working directory.")
        no_verified("identity of the Type2 participants with the output of script 109 (file "
                       "missing)")
    in24 = TWO & P24
    log(f'Of the {len(TWO)}, in the primary 24 h cohort: {len(in24)}'
        + (' (consistent: the primary cohort admits only Type1 at baseline).'
           if not in24 else
           ' — UNEXPECTED: the primary cohort admits only Type1 at baseline.'))
    LBLS = {pt: chr(ord("A") + i) for i, pt in enumerate(sorted(TWO))}

    # --- populations
    shared = P333 & P24
    P331 = P333 - TWO
    P288 = P333 - shared
    P286 = P288 - TWO
    two_in_288 = TWO & P288
    POP = {"333": P333, "331": P331, "288": P288, "286": P286}
    NAMEPOP = {"333": f'150 h, sporadic only, n = {len(P333)}',
               "331": f'150 h, sporadic only, without the {len(TWO)} Type2 at baseline, n = '
                      f'{len(P331)}',
               "288": f'150 h, sporadic only, without the {len(shared)} shared with the 24 h cohort, '
                      f'n = {len(P288)}',
               "286": f'150 h, sporadic only, without the shared participants and without the Type2 '
                      f'at baseline, n = {len(P286)}'}
    log(f'Populations (participants): cohort {len(P333)}; without the {len(TWO)}: {len(P331)}; '
        f'shared with the 24 h cohort {len(shared)}; without the shared participants {len(P288)}; '
        f'without the shared participants and without the {len(TWO)}: {len(P286)}.')
    log(f'Of the {len(TWO)}, among the {len(P288)} not shared: {len(two_in_288)}'
        + ("." if len(two_in_288) == len(TWO) else
           f' (the other {len(TWO) - len(two_in_288)} are among the shared participants).'))

    # --- description: z within the cohort
    M333 = c150[F150].to_numpy(float).copy()
    M333[:, :4] = np.log(M333[:, :4])
    X333 = standardise25(M333)
    idx = {pt: int(np.where(c150.PATNO.to_numpy() == pt)[0][0]) for pt in sorted(TWO)}
    S_inv = np.linalg.inv(np.cov(X333, rowvar=False))
    d2 = np.einsum("ij,jk,ik->i", X333, S_inv, X333)        # squared Mahalanobis
    lk = np.log(c150[list(KIN150)].to_numpy(float))
    zk = (lk - lk.mean(0)) / lk.std(0)
    years = pd.to_numeric(cur.YEAR, errors="coerce")
    vis = cur[cur.PATNO.isin(P333) & years.notna()].assign(years=years)
    fu = vis.groupby("PATNO").years.agg(last="max", n_visits="size")
    assert set(fu.index) == P333, 'members of the cohort without dated visits in the curated data cut'
    rows_two = []
    log()
    log(f'Position within the cohort (n = {len(P333)}): z = (value - mean)/SD, SD with denominator n '
        f'as in')
    log(f"script 25; the four kinetic parameters on the log scale. 'rank' = ascending position out "
        f"of {len(P333)}.")
    for pt in sorted(TWO):
        i = idx[pt]
        log(f'\n  Participant {LBLS[pt]} (PATNO {pt})')
        for j, c in enumerate(F150):
            rank = int((X333[:, j] <= X333[i, j]).sum())
            log(f'    {NAME7[c]:<28} z = {X333[i, j]:+.2f}   rank {rank}/{len(P333)}')
            rows_two.append(dict(participant=LBLS[pt], block='seven clustering variables',
                                  quantity=NAME7[c], value=float(X333[i, j]),
                                  units='z within the cohort (n = 333, SD with ddof = 0)',
                                  rank=rank, over=len(P333)))
        rank = int((d2 <= d2[i]).sum())
        log(f'    Mahalanobis distance from the centre, on the seven variables: {np.sqrt(d2[i]):.2f} '
            f'(rank {rank}/{len(P333)}; median of the cohort {np.sqrt(np.median(d2)):.2f})')
        rows_two.append(dict(participant=LBLS[pt], block='seven clustering variables',
                              quantity='Mahalanobis distance from the centre',
                              value=float(np.sqrt(d2[i])), units='multivariate SD',
                              rank=rank, over=len(P333)))
        for j, c in enumerate(KIN150.values()):
            rank = int((zk[:, j] <= zk[i, j]).sum())
            rows_two.append(dict(participant=LBLS[pt], block='five kinetic parameters at 150 h',
                                  quantity=f"log {c}", value=float(zk[i, j]),
                                  units='z within the cohort (n = 333, SD with ddof = 0)',
                                  rank=rank, over=len(P333)))
        log('    kinetic parameters (z of the logarithm): '
            + ", ".join(f"{c} {zk[i, j]:+.2f}" for j, c in enumerate(KIN150.values())))
        u, nv = float(fu.loc[pt, 'last']), int(fu.loc[pt, 'n_visits'])
        n_less = int((fu.last <= u).sum())
        log(f'    follow-up in the curated data cut: last visit at year {u:.0f}, {nv} visits with a '
            f'recorded year; participants of the cohort with a last visit not later: '
            f'{n_less}/{len(P333)}')
        rows_two.append(dict(participant=LBLS[pt], block='follow-up (curated data cut)',
                              quantity='year of the last visit', value=u, units='years',
                              rank=n_less, over=len(P333)))
        rows_two.append(dict(participant=LBLS[pt], block='follow-up (curated data cut)',
                              quantity='visits with recorded year', value=float(nv),
                              units='visits', rank=int((fu.n_visits <= nv).sum()),
                              over=len(P333)))
        log('    rows of the assay file (all the visits, in run order):')
        rr = sa[sa.PATNO == pt].sort_values(["RUNDATE", 'row'])
        for k, r in enumerate(rr.itertuples()):
            prot = {155: '150 h', 237: '24 h'}[r.PROJECTID]
            kind = "" if pd.isna(r.SAA_Type) else f"/{r.SAA_Type}"
            log(f'      {r.CLINICAL_EVENT:<4} {prot:<8} {r.SAA_Status}{kind:<14} run '
                f'{r.RUNDATE.date()}')
            rows_two.append(dict(participant=LBLS[pt], block='assay file',
                                  quantity=f'row {k + 1}: visit {r.CLINICAL_EVENT}, protocol '
                                            f'{prot}, outcome {r.SAA_Status}{kind}',
                                  value=float(r.RUNDATE.year), units='year of the run',
                                  rank=np.nan, over=np.nan))
        after = v24[(v24.PATNO == pt) & (v24.CLINICAL_EVENT != "BL")]
        log('    at 24 h after baseline (last run per visit): '
            + ("; ".join(f'{r.CLINICAL_EVENT} {r.outcome}'
                         for r in after.sort_values("CLINICAL_EVENT").itertuples())
               if len(after) else 'no result'))
    log()
    log(f'Follow-up of the cohort, for comparison (n = {len(P333)}): year of last visit, median '
        f'{fu.last.median():.0f} [IQR {fu.last.quantile(0.25):.0f}-{fu.last.quantile(0.75):.0f}]; '
        f'visits with a recorded year, median {fu.n_visits.median():.0f}.')
    ult_two = fu.loc[sorted(TWO), 'last']
    log(f"The {len(TWO)}: last visit at years {listing((f'{x:.0f}' for x in ult_two))}; visits "
        f"{listing((int(x) for x in fu.loc[sorted(TWO), 'n_visits']))}. To the longitudinal models")
    log("they therefore contribute at most these visits: the exact number of observations they "
        "contribute to each model")
    log("is reported in sections 5 and 6.")
    t1_after = {pt: bool(((v24.PATNO == pt) & (v24.CLINICAL_EVENT != "BL")
                         & (v24.outcome == "Type1")).any()) for pt in sorted(TWO)}
    log(f'With a Type1 at 24 h at a later visit: {sum(t1_after.values())} out of {len(TWO)}.')

    # ========================================================================
    title('2. TYPING AT 24 h IN THE COHORT (at baseline or later)')
    log('Source: assay file, Project 237, all the visits of the members of the cohort. For each')
    log("participant and visit the row run last counts. 'Typed' = positive at 24 h")
    log('with type Type1 or Type2; negatives, inconclusives and positives of undetermined type do not')
    log("have a type. Unit: participants.")
    codes = sorted(set(v24.CLINICAL_EVENT))
    without_pos = [c for c in codes if not (c == "BL" or re.fullmatch(r"V\d+", c))]
    log(f'Visit codes present: {listing(codes)}'
        + ("." if not without_pos else
           f"; of these {listing(without_pos)} have no position relative to baseline and are counted "
           f"among the visits 'after baseline'."))
    OUTCOMES24 = ["Type1", "Type2", 'positive, undetermined type', 'inconclusive', 'negative']
    assert set(v24.outcome) <= set(OUTCOMES24)
    rows_tip = []
    for tag in ("333", "331"):
        vv = v24[v24.PATNO.isin(POP[tag])]
        for code in codes:
            for e in OUTCOMES24:
                rows_tip.append(dict(
                    population=NAMEPOP[tag], table='per visit: outcome of the last run at 24 h', item=f'{code}: {e}',
                    n=int(((vv.CLINICAL_EVENT == code) & (vv.outcome == e)).sum()),
                    units='participants'))
    log()
    log(f"By visit, whole cohort (n = {len(P333)}): participants by outcome of the last run")
    log(f"  {'visit':<7}" + "".join(f"{e[:22]:>24}" for e in OUTCOMES24))
    for code in codes:
        log(f'  {code:<7}' + "".join(
            f'{int(((v24.CLINICAL_EVENT == code) & (v24.outcome == e)).sum()):>24}'
            for e in OUTCOMES24))

    def profile(pop):
        vv = v24[v24.PATNO.isin(pop)]
        per = vv.groupby("PATNO").outcome.agg(lambda s: " + ".join(
            e for e in OUTCOMES24 if e in set(s)))
        blp = vv[vv.CLINICAL_EVENT == "BL"].set_index("PATNO").outcome
        t1 = set(vv[vv.outcome == "Type1"].PATNO)
        t2 = set(vv[vv.outcome == "Type2"].PATNO)
        con = set(vv.PATNO)
        typed = t1 | t2
        return dict(per=per, bl=blp, t1=t1, t2=t2, con=con, typed=typed,
                    tip_bl=set(blp[blp.isin(["Type1", "Type2"])].index))

    PROF = {tag: profile(POP[tag]) for tag in ("333", "331")}
    for tag in ("333", "331"):
        pr, pop = PROF[tag], POP[tag]
        n = len(pop)
        items = [
            ('with at least one 24 h result, at any visit', len(pr["con"])),
            ('without any result at 24 h', n - len(pr["con"])),
            ('with a result at 24 h at baseline', len(pr["bl"])),
        ] + [(f'at baseline: {e}', int((pr["bl"] == e).sum())) for e in OUTCOMES24] + [
            ('typed at baseline (Type1 or Type2)', len(pr["tip_bl"])),
            ('typed only at a later visit', len(pr['typed'] - pr["tip_bl"])),
            ('typed at any visit', len(pr['typed'])),
            ('NEVER typed (no Type1 nor Type2 at any visit)',
             n - len(pr['typed'])),
            ('with at least one Type1 at any visit', len(pr["t1"])),
            ('with at least one Type2 at any visit', len(pr["t2"])),
            ('with Type1 at all visits with a result at 24 h',
             int((pr["per"] == "Type1").sum())),
        ]
        log()
        log(f'[{NAMEPOP[tag]}] participants')
        for vo, k in items:
            log(f"  {vo:<62} {k:>4}")
            rows_tip.append(dict(population=NAMEPOP[tag], table='summary', item=vo, n=k,
                                  units='participants'))
        log('  profile per participant (set of the outcomes at his or her 24 h visits):')
        for vo, k in count(pr["per"]).items():
            log(f"    {vo:<60} {k:>4}")
            rows_tip.append(dict(population=NAMEPOP[tag], table='profile by participant',
                                  item=vo, n=k, units='participants'))
    t2_rows = set(h[h.SAA_Type == "Type2"].PATNO)
    log()
    log(f'Check: participants with ANY Type2 row at 24 h (even if not the last run of the visit): '
        f'{len(t2_rows)}'
        + (", the same as with the last-run rule."
           if t2_rows == PROF["333"]["t2"] else
           f"; with the last-run rule they are {len(PROF['333']['t2'])}."))
    others_t2 = PROF["331"]["t2"]
    pr1 = PROF["331"]
    never = P331 - pr1['typed']
    log()
    log(f"Reading for the {len(P331)} who remain: {len(pr1['tip_bl'])} are typed at baseline "
        f"({int((pr1['bl'] == 'Type1').sum())} Type1, {int((pr1['bl'] == 'Type2').sum())} Type2);")
    log(f"  {len(pr1['typed'] - pr1['tip_bl'])} are not typed at baseline but have a type at a later "
        f"visit; never typed: {len(never)}")
    log(f"  ({len(never & pr1['con'])} with only 24 h results without a type, "
        f"{len(never - pr1['con'])} without any 24 h result).")
    log(f"  With at least one Type1 at some visit: {len(pr1['t1'])} out of {len(P331)}.")
    if others_t2:
        log(f'  WARNING: {len(others_t2)} of the {len(P331)} have a Type2 at a visit after baseline '
            f'(PATNO {listing(sorted(others_t2))}):')
        for pt in sorted(others_t2):
            rr = v24[v24.PATNO == pt].sort_values("CLINICAL_EVENT")
            log(f'    PATNO {pt}: ' + "; ".join(f'{r.CLINICAL_EVENT} {r.outcome} (run '
                                                f'{r.RUNDATE.date()})' for r in rr.itertuples())
                + f"; in the 24 h cohort: {('si' if pt in P24 else 'no')}")
        log(f"  Of these, with Type1 at baseline: "
            f"{int(pr1['bl'].reindex(sorted(others_t2)).eq('Type1').sum())}; shared with the primary "
            f"cohort: {len(others_t2 & P24)}. They are NOT excluded in this analysis, which")
        log("  removes only those who are Type2 at BASELINE: this must be declared.")
    else:
        log(f'  None of the {len(P331)} has a Type2 at a later visit.')
    c35 = cur[cur.PATNO.isin(P333) & (cur.CSFSAA_assay == 3) & cur.CSFSAA.notna()]
    log(f'For completeness, 35 h protocol (categorical outcome of the curated data cut; codes 1 = '
        f'positive/Type I, 3 = Type II): {len(c35)} visits of {c35.PATNO.nunique()} members of the '
        f'cohort, outcomes {count(c35.CSFSAA.astype(int))}; of the {len(TWO)}: '
        f'{int(c35.PATNO.isin(TWO).sum())} visits.')

    # ========================================================================
    # PREPARATION OF THE TASKS
    # ========================================================================
    def matrix(pop):
        d = c150[c150.PATNO.isin(pop)]
        M = d[F150].to_numpy(float).copy()
        M[:, :4] = np.log(M[:, :4])
        return M

    covbl = blc[["PATNO", "age_at_visit", "SEX", "EDUCYRS", "duration_yrs"]].rename(
        columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
    everyone_outcomes = list(dict.fromkeys(list(OUTCOMES35) + [v for v, _, _ in OUTCOMES12]))

    def long(pop):
        """Subject-year table and SD of the log of each kinetic parameter in the
        population. The kinetic parameters are standardised WITHIN the population
        (SD with ddof = 1), as in 72/84/88 and in 15/91/107."""
        d = c150[c150.PATNO.isin(pop)]
        base = d[["PATNO"] + list(KIN150)].rename(columns=KIN150).copy()
        sd = {}
        for k in KIN150.values():
            assert (base[k] > 0).all(), f'{k} not positive'
            lg = np.log(base[k])
            sd[k] = float(lg.std())
            base[k] = (lg - lg.mean()) / lg.std()
        base = base.merge(covbl, on="PATNO", how="left", validate="one_to_one")
        L = cur[cur.PATNO.isin(pop)][["PATNO", "EVENT_ID", "YEAR"] + everyone_outcomes].copy()
        L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
        L = L[L.years.notna()].merge(base, on="PATNO", how="left", validate="many_to_one")
        L = L.sort_values(["PATNO", "years", "EVENT_ID"]).reset_index(drop=True)
        assert not L.duplicated(["PATNO", "years"]).any(), \
            'two rows for the same subject in the same year'
        assert set(L.PATNO) == set(pop), 'subjects without visits in the curated data cut'
        return L, sd

    LONG = {tag: long(POP[tag]) for tag in ("333", "331")}
    # distinct models: (outcome, kinetic parameter). Two of the twelve non-motor
    # outcomes (MoCA and SCOPA-AUT on T50) are also in the family of the 35: they are
    # estimated once.
    MODELS = {}
    for out, (_lab, cov) in OUTCOMES35.items():
        for k in KIN150.values():
            MODELS[(out, k)] = cov
    for var, _dom, educ in OUTCOMES12:
        cov = ["age_bl", "SEX", "dur_bl"] + (["EDUCYRS"] if educ else [])
        if (var, "T50") in MODELS:
            assert MODELS[(var, "T50")] == cov, f'{var}: different covariates in the two families'
        MODELS[(var, "T50")] = cov
    SAMP = {}
    tasks = []
    for tag in ("333", "331"):
        L = LONG[tag][0]
        for (out, k), cov in MODELS.items():
            m = L[["PATNO", "years", out, k] + cov].dropna().reset_index(drop=True)
            multi = m.groupby("PATNO").size()
            SAMP[(tag, out, k)] = dict(
                n_subj=int(m.PATNO.nunique()), n_obsv=int(len(m)),
                n_multi=int((multi >= 2).sum()),
                obsv_of_two={LBLS[pt]: int((m.PATNO == pt).sum()) for pt in sorted(TWO)},
                admitted=bool(len(m) >= 120 and int((multi >= 2).sum()) >= 40))
            if SAMP[(tag, out, k)]['admitted']:
                tasks.append(('mixed', (tag, out, k),
                                dict(formula=f"{out} ~ {k}*years + " + " + ".join(cov),
                                     m=m, kinet=k)))
    # the slow model (MDS-UPDRS I) first, then the batteries, then the rest
    slowones = [c for c in tasks if c[1][1] == "updrs1_score"]
    others = [c for c in tasks if c[1][1] != "updrs1_score"]
    batt = [('batt', tag, dict(M=matrix(POP[tag]), B=B))
            for tag, B in (("333", B25), ("331", B25), ("288", B66), ("286", B66))]
    desi = [("designate", tag, dict(M=matrix(POP[tag]))) for tag in ("333", "331")]
    tasks = slowones + batt + desi + others
    print(f'[computation] {len(tasks)} tasks on {N_PROCESSES} processes '
          f'({len(slowones) + len(others)} mixed models, {len(batt)} battery branches, {len(desi)} '
          f'branches of designated estimates)', flush=True)

    RESULT = {}
    with mp.get_context("spawn").Pool(N_PROCESSES) as pool:
        done = 0
        for kind, key, result, dt in pool.imap_unordered(execute, tasks, chunksize=1):
            RESULT[(kind, key)] = result
            done += 1
            print(f'[computation] {done}/{len(tasks)} {kind} {key} ({dt:.0f}s; elapsed '
                  f'{time.time() - T0:.0f}s)', flush=True)
    assert len(RESULT) == len(tasks)

    # ------------------------------------------------------------------------
    # The two families of mixed models, and the attribution of the shifts
    # ------------------------------------------------------------------------
    def family(keys, name_fam, labels):
        """Comparison of 333 against 331 on a family of models."""
        rows = []
        for (out, k) in keys:
            a = RESULT.get(('mixed', ("333", out, k)))
            b = RESULT.get(('mixed', ("331", out, k)))
            ca, cb = SAMP[("333", out, k)], SAMP[("331", out, k)]
            r = dict(family=name_fam, outcome=out, label=labels[out], kinetic=k,
                     n_subj_333=ca['n_subj'], n_obsv_333=ca['n_obsv'],
                     n_subj_331=cb['n_subj'], n_obsv_331=cb['n_obsv'])
            for et in sorted(LBLS.values()):
                r[f'obsv_of_participant_{et}'] = ca['obsv_of_two'][et]
            assert ca['n_obsv'] - cb['n_obsv'] == sum(ca['obsv_of_two'].values()), \
                f'{out} x {k}: the observations removed are not those of the two'
            assert all(v == 0 for v in cb['obsv_of_two'].values())
            for tag, x in (("333", a), ("331", b)):
                ok = bool(x is not None and x.get('estimated'))
                r[f'estimated_{tag}'] = ok
                for c in FIELDS_FIT:
                    r[f"{c}_{tag}"] = x[c] if ok else np.nan
                r[f'accepted_{tag}'] = bool(ok and x['accepted'])
            r['sd_log_kinetic_333'] = LONG["333"][1][k]
            r['sd_log_kinetic_331'] = LONG["331"][1][k]
            rows.append(r)
        D = pd.DataFrame(rows)
        for tag in ("333", "331"):
            ok = D[f'accepted_{tag}'].to_numpy(bool)
            D[f"p_BH_{tag}"] = np.nan
            if ok.any():
                D.loc[ok, f"p_BH_{tag}"] = bh(D.loc[ok, f"p_{tag}"])
            D[f"ci_lo_{tag}"] = D[f"beta_{tag}"] - Z95 * D[f"se_{tag}"]
            D[f"ci_hi_{tag}"] = D[f"beta_{tag}"] + Z95 * D[f"se_{tag}"]
        D["delta_beta"] = D.beta_331 - D.beta_333
        D['shift_in_SE_333'] = D.delta_beta / D.se_333
        # exact rescaling: z_333 = z_331 * (SD_331/SD_333) + constant
        D['beta_331_by_SD_of_333'] = D.beta_331 * D.sd_log_kinetic_333 / D.sd_log_kinetic_331
        D['shift_in_SE_333_a_scale_fixed'] = (D.beta_331_by_SD_of_333 - D.beta_333) / D.se_333
        D['ratio_SE_331_over_333'] = D.se_331 / D.se_333
        D['nominal_333'] = D.p_333 < ALPHA
        D['nominal_331'] = D.p_331 < ALPHA
        D["BH_333"] = D.p_BH_333 < ALPHA
        D["BH_331"] = D.p_BH_331 < ALPHA
        D['changes_state_nominal'] = D.nominal_333 != D.nominal_331
        D['changes_state_BH'] = D.BH_333 != D.BH_331
        D['changes_sign'] = np.sign(D.beta_333) != np.sign(D.beta_331)
        return D

    keys35 = [(out, k) for out in OUTCOMES35 for k in KIN150.values()]
    keys12 = [(v, "T50") for v, _, _ in OUTCOMES12]
    for name_f, keys in (('35 tests', keys35), ('12 non-motor outcomes', keys12)):
        excluded = [c for c in keys if not (SAMP[("333",) + c]['admitted']
                                             and SAMP[("331",) + c]['admitted'])]
        assert not excluded, (f'{name_f}: models below the minimum sample size of the pipeline (120 '
                             f'observations, 40 subjects with two visits): {excluded}')
    D35 = family(keys35, 'kinetic parameter x time, 150 h, 35 tests',
                   {o: OUTCOMES35[o][0] for o in OUTCOMES35})
    D12 = family(keys12, 'primary non-motor, T50 x years, 12 tests',
                   {v: dom for v, dom, _ in OUTCOMES12})

    # Attribution. Every coefficient that between 333 and 331 shifts by at least
    # THRESHOLD_ATTRIB standard errors is re-estimated removing ONE participant
    # at a time: it says which of those removed shifts it. The shift is read with the
    # kinetic parameter rescaled to the SD of the 333, so it does not contain the
    # rescaling.
    fr_attr = []
    for D in (D35, D12):
        for r in D.itertuples():
            if (r.outcome, r.kinetic) not in fr_attr and (
                    not np.isfinite(r.shift_in_SE_333)
                    or abs(r.shift_in_SE_333) >= THRESHOLD_ATTRIB):
                fr_attr.append((r.outcome, r.kinetic))
    tasks2 = []
    LOO = {}
    for pt in sorted(TWO):
        tag = f'without_{LBLS[pt]}'
        LOO[tag] = long(P333 - {pt})
        for (out, k) in fr_attr:
            m = LOO[tag][0][["PATNO", "years", out, k] + MODELS[(out, k)]].dropna()
            m = m.reset_index(drop=True)
            tasks2.append(('mixed', (tag, out, k),
                             dict(formula=f"{out} ~ {k}*years + "
                                          + " + ".join(MODELS[(out, k)]), m=m, kinet=k)))
    if tasks2:
        print(f'[computation] attribution: {len(tasks2)} models ({len(fr_attr)} coefficients x '
              f'{len(TWO)} participants removed one at a time)', flush=True)
        with mp.get_context("spawn").Pool(N_PROCESSES) as pool:
            for kind, key, result, dt in pool.imap_unordered(execute, tasks2, chunksize=1):
                RESULT[(kind, key)] = result
                print(f'[computation] attribution {key} ({dt:.0f}s; elapsed {time.time() - T0:.0f}s)', flush=True)
    L333 = LONG["333"][0]
    # z of the kinetic parameter per subject in the 333 (one row per subject), as it
    # enters the models
    Z333 = L333.sort_values(["PATNO", "years"]).drop_duplicates("PATNO", keep="first"
                                                               ).set_index("PATNO")[
        list(KIN150.values())]
    assert set(Z333.index) == P333
    rows_attr = []
    for (out, k) in fr_attr:
        fam_name = ('35 kinetic parameter x time tests' if (out, k) in keys35 else "") + (
            " e " if (out, k) in keys35 and (out, k) in keys12 else "") + (
            '12 non-motor outcomes' if (out, k) in keys12 else "")
        a = RESULT[('mixed', ("333", out, k))]
        m3 = L333[["PATNO", "years", out, k] + MODELS[(out, k)]].dropna()
        for pt in sorted(TWO):
            tag = f'without_{LBLS[pt]}'
            x = RESULT[('mixed', (tag, out, k))]
            q = m3[m3.PATNO == pt].sort_values("years")
            ok = bool(x.get('estimated'))
            b_fixed = (x["beta"] * LONG["333"][1][k] / LOO[tag][1][k]) if ok else np.nan
            zc = float(Z333.loc[pt, k])
            rows_attr.append(dict(
                family=fam_name, outcome=out, kinetic=k, participant_removed=LBLS[pt],
                population=f'150 h, sporadic only, without the single participant {LBLS[pt]}, n = '
                            f'{len(P333) - 1}',
                obsv_of_participant_in_model=int(len(q)),
                first_year=float(q.years.min()) if len(q) else np.nan,
                last_year=float(q.years.max()) if len(q) else np.nan,
                change_annual_observed=(
                    float((q[out].iloc[-1] - q[out].iloc[0])
                          / (q.years.iloc[-1] - q.years.iloc[0])) if len(q) >= 2 else np.nan),
                slope_mean_of_model_a_333=a['slope_mean'],
                z_kinetic_of_participant=zc,
                share_of_total_of_squares_of_kinetic=(
                    zc ** 2 / float((Z333[k] ** 2).sum()) if np.isfinite(zc) else np.nan),
                remark_on_p='diagnostic: uncorrected and outside every family of tests',
                beta_333=a["beta"], se_333=a["se"],
                beta_without_the_participant=x["beta"] if ok else np.nan,
                se_without_the_participant=x["se"] if ok else np.nan,
                p_no_corrected_without_the_participant=x["p"] if ok else np.nan,
                accepted_fromthe_criterion_strict=bool(ok and x['accepted']),
                beta_without_the_participant_by_SD_of_333=b_fixed,
                shift_in_SE_333_a_scale_fixed=((b_fixed - a["beta"]) / a["se"])
                if ok else np.nan,
                units_of_beta='units of the outcome per year per 1 SD of the log of the kinetic '
                              'parameter',
                units_of_z='SD of the log of the kinetic parameter in the 333 (ddof = 1)'))
    ATTR = pd.DataFrame(rows_attr)

    # ========================================================================
    title('3. (a) BATTERY AT k = 2 AGAINST THE GAUSSIAN COPULA NULL')
    log(f'Functions, seed and number of realisations of script 25 (seed {SEED25}, B = {B25}), the '
        f'same in 66 (B = {B66}).')
    log('In EVERY branch the standardisation and the copula null (empirical marginals and')
    log(f"Spearman correlations) are estimated on the population of the branch: at n = {len(P331)} "
        f"the null is RE-ESTIMATED")
    log(f"on the {len(P331)}, it is not that of the {len(P333)}. The generator is the same, but with "
        f"a different n the draws")
    log('are not aligned: the two null distributions differ also by Monte Carlo error.')
    log("Monte Carlo p: one-sided in the direction in which the statistic indicates more structure "
        "(for the minimum")
    log("dip the direction is reversed), (number of nulls >= observed + 1)/(B + 1), NOT corrected. "
        "Family: the seven")
    log('statistics of a population; next to the uncorrected p, Bonferroni and Benjamini-Hochberg '
        'over the seven.')

    def summary(obsv, v, stat):
        v = np.asarray(v, float)
        p = ((np.sum(v >= obsv) if HIGH[stat] else np.sum(v <= obsv)) + 1) / (len(v) + 1)
        return dict(observed=float(obsv), null_median=float(np.median(v)),
                    null_RI2_5=float(np.percentile(v, 2.5)),
                    null_RI97_5=float(np.percentile(v, 97.5)),
                    p_montecarlo=float(p), B=int(len(v)))

    NUL = {tag: pd.DataFrame(RESULT[('batt', tag)]['nulls'])[STAT] for tag in POP}
    OBSV = {tag: RESULT[('batt', tag)]['obsv'] for tag in POP}
    DES = {tag: RESULT[("designate", tag)] for tag in ("333", "331")}

    # --- IDENTITY of the 333 branch with the saved values
    def departure_max(a, b):
        return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))

    f25 = os.path.join(OUT, "25_nulls.csv")
    f25d = os.path.join(OUT, "25_null_distribution_150h.csv")
    if os.path.exists(f25) and os.path.exists(f25d):
        r25 = load_csv(f25)
        r25 = r25[r25.cohort == f"150h (n={len(P333)})"].set_index('statistic')
        assert set(r25.index) == set(STAT), (f"25_nulls.csv does not have the seven statistics of "
                                             f"the cohort '150h (n={len(P333)})'")
        d25 = load_csv(f25d)[STAT]
        assert d25.shape == NUL["333"].shape, '25_null_distribution_150h.csv: different shape'
        s_obsv = departure_max([OBSV["333"][s] for s in STAT], r25.loc[STAT, 'observed'])
        s_nul = departure_max(NUL["333"].to_numpy(), d25.to_numpy())
        ria = pd.DataFrame([summary(OBSV["333"][s], NUL["333"][s], s) for s in STAT], index=STAT)
        s_ria = max(departure_max(ria[a], r25.loc[STAT, b]) for a, b in (
            ('null_median', 'null_median'), ('null_RI2_5', "nullo_IC2.5"),
            ('null_RI97_5', "nullo_IC97.5"), ("p_montecarlo", "p_montecarlo")))
        assert max(s_obsv, s_nul, s_ria) < TOL_ID_BATT, (
            f'the branch at n = {len(P333)} does NOT reproduce script 25: maximum discrepancy '
            f'observed {s_obsv:.3g}, nulls {s_nul:.3g}, summaries {s_ria:.3g}')
        log()
        log(f"Identity with script 25 (branch at n = {len(P333)}): the seven observed statistics, "
            f"the {NUL['333'].shape[0]} x {NUL['333'].shape[1]}")
        log('  null realisations and the summaries (median, extremes, p) match 25_nulls.csv and')
        log(f'  25_null_distribution_150h.csv: maximum discrepancy {max(s_obsv, s_nul, s_ria):.1e} '
            f'(tolerance {TOL_ID_BATT:g}).')
    else:
        log()
        log("Identity with script 25: NOT verified, 25_nulls.csv or 25_null_distribution_150h.csv "
            "missing.")
        no_verified("battery: identity of the branch at n = 333 with the outputs of script 25")

    f66 = os.path.join(OUT, "66_replication_without_overlapping.csv")
    if os.path.exists(f66):
        r66 = load_csv(f66)
        sc66 = []
        for name66, tag, nB in (('150h full', "333", B66), ('150h without overlapping',
                                                              "288", B66)):
            q = r66[r66.analysis == name66].set_index('statistic')
            assert set(q.index) == set(STAT) and int(q.n.iloc[0]) == len(POP[tag]), (
                f"66_replication_without_overlapping.csv: '{name66}' is not the expected population")
            ria = pd.DataFrame([summary(OBSV[tag][s], NUL[tag][s].iloc[:nB], s) for s in STAT],
                               index=STAT)
            sc66.append(max(departure_max(ria[a], q.loc[STAT, b]) for a, b in (
                ('observed', 'observed'), ('null_median', 'null_median'),
                ('null_RI2_5', "nullo_IC2.5"), ('null_RI97_5', "nullo_IC97.5"),
                ("p_montecarlo", "p_montecarlo"))))
        assert max(sc66) < TOL_ID_BATT, (
            f'the branches at n = {len(P333)} and n = {len(P288)} do NOT reproduce script 66: '
            f'discrepancies {sc66}')
        log(f"Identity with script 66 (§1.8): the branch at n = {len(P288)} (B = {B66}) and the "
            f"branch at n = {len(P333)} read on the")
        log(f'  first {B66} realisations match 66_replication_without_overlapping.csv: maximum '
            f'discrepancy {max(sc66):.1e}.')
    else:
        log("Identity with script 66: NOT verified, 66_replication_without_overlapping.csv missing.")
        no_verified("battery: identity of the branches at n = 333 and n = 288 with script 66")

    # --- designated estimates (script 08) and unified table (script 41)
    sc08, missing08 = {}, []
    f = os.path.join(OUT, "08_predstrength_150h.csv")
    if os.path.exists(f):
        q = load_csv(f)
        sc08["prediction strength"] = abs(float(q[q.k == 2].PS_obsv.iloc[0]) - DES["333"]["PS_k2"])
    else:
        missing08.append("08_predstrength_150h.csv")
    f = os.path.join(OUT, "08_stability_150h.csv")
    if os.path.exists(f):
        q = load_csv(f)
        q = q[(q.data == 'observed') & (q.k == 2)].iloc[0]
        sc08["Jaccard"] = max(abs(float(q.jaccard_avg) - DES["333"]['jaccard_avg']),
                              abs(float(q.jaccard_min) - DES["333"]["jaccard_min"]))
    else:
        missing08.append("08_stability_150h.csv")
    f = os.path.join(OUT, "08_gap_150h.csv")
    if os.path.exists(f):
        q = load_csv(f)
        assert list(q.k) == DES["333"]["gap_k"], '08_gap_150h.csv: different grid of k'
        sc08["gap"] = max(departure_max(q.gap, DES["333"]["gap"]),
                          departure_max(q.s_k, DES["333"]["gap_sk"]))
    else:
        missing08.append("08_gap_150h.csv")
    f = os.path.join(OUT, "08_diptest_150h.csv")
    if os.path.exists(f):
        q = load_csv(f)
        mine = np.array(DES["333"]["dip_var"] + DES["333"]["dip_pc"])
        assert len(q) == len(mine), '08_diptest_150h.csv: different number of tests'
        sc08['dip on variables and components'] = max(departure_max(q.dip, mine[:, 0]),
                                                    departure_max(q.p, mine[:, 1]))
    else:
        missing08.append("08_diptest_150h.csv")
    f = os.path.join(OUT, "08_replication_150h.txt")
    proj_ok = None
    if os.path.exists(f):
        with open(f) as fh:
            mm = re.search(r"observed: (\d+)/1000", fh.read())
        if mm:
            proj_ok = int(mm.group(1)) == DES["333"]["proj_n_sig"]
            assert proj_ok, ("the number of multimodal projections out of 1000 is not the one "
                             "written in 08_replication_150h.txt")
    if sc08:
        assert max(sc08.values()) < TOL_ID_BATT, (
            f'the branch at n = {len(P333)} does NOT reproduce the estimates of script 08: {sc08}')
        log(f"Identity with script 08 (designated observed estimates, branch at n = {len(P333)}): "
            + "; ".join(f'{k} difference {v:.1e}' for k, v in sc08.items()) + ".")
    for name in missing08:
        log(f'Identity with script 08: NOT verified for {name} (file missing).')
        no_verified(f'designated estimates: identity with {name}')
    if proj_ok:
        log(f'  Multimodal projections out of {N_PROJ_08}: same count written in '
            f'08_replication_150h.txt.')
    else:
        log(f"  Multimodal projections out of {N_PROJ_08}: identity with 08_replication_150h.txt NOT "
            f"verified (file or row missing).")
        no_verified('designated estimates: count of the multimodal projections out of 1000 '
                       'against 08_replication_150h.txt')

    def table_S5(tag, nB):
        """The seven rows of Table S5: five from the battery of script
        25, prediction strength and Jaccard from the designated estimates of script
        08 (composition of script 41), against the null of the branch."""
        rows = []
        for s in STAT:
            if s == "PS_k2":
                obsv, source = DES[tag]["PS_k2"], 'designated estimate, script 08 (50 partitions)'
            elif s == "jaccard_k2":
                obsv, source = (DES[tag]['jaccard_avg'],
                              'designated estimate, script 08 (200 resamplings)')
            else:
                obsv, source = OBSV[tag][s], 'battery, script 25'
            rows.append(dict(statistic=s, source_observed=source,
                              **summary(obsv, NUL[tag][s].iloc[:nB], s)))
        t = pd.DataFrame(rows)
        t["p_bonferroni_7"] = np.minimum(1.0, t.p_montecarlo * len(t))
        t["p_BH_7"] = bh(t.p_montecarlo)
        return t

    def table_batt(tag, nB):
        t = pd.DataFrame([dict(statistic=s, source_observed='battery, script 25/66',
                               **summary(OBSV[tag][s], NUL[tag][s].iloc[:nB], s))
                          for s in STAT])
        t["p_bonferroni_7"] = np.minimum(1.0, t.p_montecarlo * len(t))
        t["p_BH_7"] = bh(t.p_montecarlo)
        return t

    S5 = {tag: table_S5(tag, B25) for tag in ("333", "331")}
    f41 = os.path.join(OUT, "41_unified_nulls.csv")
    if os.path.exists(f41):
        q = load_csv(f41)
        q = q[q.cohort == f"150h (n={len(P333)})"].set_index('statistic')
        assert set(q.index) == set(STAT), '41_unified_nulls.csv: unexpected statistics'
        t = S5["333"].set_index('statistic')
        s41 = max(departure_max(t.loc[STAT, a], q.loc[STAT, b]) for a, b in (
            ('observed', 'observed'), ('null_median', 'null_median'),
            ('null_RI2_5', "nullo_IC2.5"), ('null_RI97_5', "nullo_IC97.5"),
            ("p_montecarlo", "p_montecarlo")))
        assert s41 < TOL_ID_BATT, (f'Table S5 recomposed at n = {len(P333)} does NOT match '
                                   f'41_unified_nulls.csv: discrepancy {s41:.3g}')
        log(f"Identity with script 41 (Table S5, 150 h block): the seven rows recomposed at n = "
            f"{len(P333)}")
        log(f'  match 41_unified_nulls.csv, maximum discrepancy {s41:.1e}.')
    else:
        log("Identity with script 41: NOT verified, 41_unified_nulls.csv missing.")
        no_verified("Table S5: identity with 41_unified_nulls.csv")

    rows_batt = []

    def printout_comparison(name_block, ta, tb, tag_a, tag_b):
        log()
        log(f'[{name_block}]')
        log(f'  A = {NAMEPOP[tag_a]} (B = {int(ta.B.iloc[0])})')
        log(f'  B = {NAMEPOP[tag_b]} (B = {int(tb.B.iloc[0])}; null re-estimated on B)')
        log(f"  "
            f"{'statistic':<31}{'obs. A':>8}{'obs. B':>8}{'med. A':>8}{'med. B':>8}{'null interval B':>20}{'p A':>8}{'p B':>8}{'BH B':>7}")
        for (_, a), (_, b) in zip(ta.iterrows(), tb.iterrows()):
            assert a.statistic == b.statistic
            log(f"  "
                f"{NAME_STAT[a.statistic]:<31}{a.observed:>8.4f}{b.observed:>8.4f}{a.null_median:>8.4f}{b.null_median:>8.4f}{f'[{b.null_RI2_5:.3f}; {b.null_RI97_5:.3f}]':>20}{a.p_montecarlo:>8.4f}{b.p_montecarlo:>8.4f}{b.p_BH_7:>7.3f}")
        for tag, t in ((tag_a, ta), (tag_b, tb)):
            for _, r in t.iterrows():
                rows_batt.append(dict(
                    block=name_block, population=NAMEPOP[tag], n=len(POP[tag]),
                    statistic=NAME_STAT[r.statistic], source_observed=r.source_observed,
                    observed=r.observed, null_median=r.null_median,
                    null_RI2_5=r.null_RI2_5, null_RI97_5=r.null_RI97_5,
                    p_montecarlo_onesided_no_corrected=r.p_montecarlo,
                    p_bonferroni_onthe_7_statistics=r.p_bonferroni_7,
                    p_BH_onthe_7_statistics=r.p_BH_7, B=int(r.B),
                    direction='high values = more structure' if HIGH[r.statistic]
                    else 'low values = more structure',
                    nullref='Gaussian copula estimated on the population of the row'))
        exceed_a = set(ta[ta.p_montecarlo < ALPHA].statistic)
        exceed_b = set(tb[tb.p_montecarlo < ALPHA].statistic)
        log(f'  Exceed the null (uncorrected p < {ALPHA}): A '
            f'{listing((NAME_STAT[s] for s in STAT if s in exceed_a))}; B '
            f'{listing((NAME_STAT[s] for s in STAT if s in exceed_b))}.')
        log(f'  After Bonferroni over the seven: A '
            f'{listing((NAME_STAT[s] for s in ta[ta.p_bonferroni_7 < ALPHA].statistic))}; B '
            f'{listing((NAME_STAT[s] for s in tb[tb.p_bonferroni_7 < ALPHA].statistic))}.')
        FOUR = ["PS_k2", "jaccard_k2", "silhouette_k2", 'ARI_algorithms_k2']
        log(f'  Minimum uncorrected p among prediction strength, Jaccard, silhouette and ARI: A '
            f'{ta[ta.statistic.isin(FOUR)].p_montecarlo.min():.4f}; B '
            f'{tb[tb.statistic.isin(FOUR)].p_montecarlo.min():.4f}.')
        outside = [NAME_STAT[r.statistic] for _, r in tb.iterrows()
                 if not (r.null_RI2_5 <= r.observed <= r.null_RI97_5)]
        log(f'  Statistics of B outside the reference interval of their own null (2.5-97.5 '
            f'percentile): {listing(outside)}.')
        return exceed_a, exceed_b

    eccS5 = printout_comparison('Table S5: battery (script 25) with designated prediction strength and '
                             'Jaccard (script 08), B = 300',
                             S5["333"], S5["331"], "333", "331")
    T25 = {tag: table_batt(tag, B25) for tag in ("333", "331")}
    exceed25 = printout_comparison('Battery of script 25 with all seven statistics from the same '
                             'generator, B = 300',
                             T25["333"], T25["331"], "333", "331")
    log()
    log(f'Designated estimates of script 08 (observed; seed {SEED08}), branch at {len(P333)} and '
        f'branch at {len(P331)}:')
    for name, ch, fmt in (('prediction strength at k = 2 (50 partitions)', "PS_k2", ".4f"),
                          ('  Monte Carlo standard error', 'PS_k2_se_mc', ".4f"),
                          ('Mean Jaccard at k = 2 (200 resamples)', 'jaccard_avg', ".4f"),
                          ('Minimum Jaccard between the two clusters', "jaccard_min", ".4f"),
                          ('k chosen by the gap (1 SE rule)', "k_gap", ""),
                          (f'projections with dip p < 0.05 out of {N_PROJ_08}', "proj_n_sig", "d"),
                          (f'minimum dip over {N_PROJ_08} projections (Bonferroni)',
                           "proj_pmin_bonf", ".4f")):
        log(f"  "
            f"{name:<52}{(format(DES['333'][ch], fmt) if fmt else str(DES['333'][ch])):>10}{(format(DES['331'][ch], fmt) if fmt else str(DES['331'][ch])):>10}")
    dPS = DES["331"]["PS_k2"] - DES["333"]["PS_k2"]
    se_ps = float(np.hypot(DES["333"]['PS_k2_se_mc'], DES["331"]['PS_k2_se_mc']))
    log("  The prediction strength is a mean over 50 random partitions, and at a different n the "
        "partitions are others:")
    log(f'  the difference between the two branches is {dPS:+.4f}, equal to {abs(dPS) / se_ps:.1f} '
        f'times the Monte Carlo standard error of the')
    log(f'  difference ({se_ps:.4f}). It does not separate the effect of the {len(TWO)} from '
        f'simulation noise.')
    ndip = len(DES["333"]["dip_var"]) + len(DES["333"]["dip_pc"])
    dip_bonf = {tag: [min(1.0, x[1] * ndip) for x in DES[tag]["dip_var"] + DES[tag]["dip_pc"]]
                for tag in ("333", "331")}
    log(f"  Dip on {len(DES['333']['dip_var'])} variables and {len(DES['333']['dip_pc'])} principal "
        f"components (Bonferroni over {ndip}): minimum corrected p {min(dip_bonf['333']):.3f} at "
        f"{len(P333)} and {min(dip_bonf['331']):.3f} at {len(P331)}; rejections of unimodality "
        f"{sum((x < ALPHA for x in dip_bonf['333']))} and "
        f"{sum((x < ALPHA for x in dip_bonf['331']))}.")
    for tag in ("333", "331"):
        rows_batt.append(dict(
            block='designated estimates (script 08), without null', population=NAMEPOP[tag],
            n=len(POP[tag]), statistic='k chosen by the gap (1 SE rule)',
            source_observed="script 08, gap_stat, B = 200",
            observed=float("nan") if DES[tag]["k_gap"] is None else float(DES[tag]["k_gap"])))
        rows_batt.append(dict(
            block='designated estimates (script 08), without null', population=NAMEPOP[tag],
            n=len(POP[tag]), statistic=f'projections with dip p < 0.05 out of {N_PROJ_08}',
            source_observed="script 08, proj_dip", observed=float(DES[tag]["proj_n_sig"])))
        rows_batt.append(dict(
            block='designated estimates (script 08), without null', population=NAMEPOP[tag],
            n=len(POP[tag]), statistic=f'dip on variables and components: minimum p, Bonferroni '
                                        f'over {ndip}', source_observed="script 08, diptest",
            observed=float(min(dip_bonf[tag]))))
        rows_batt.append(dict(
            block='designated estimates (script 08), without null', population=NAMEPOP[tag],
            n=len(POP[tag]), statistic='prediction strength at k = 2: Monte Carlo standard error', source_observed='script 08, 50 partitions',
            observed=float(DES[tag]['PS_k2_se_mc'])))

    changes_a = bool(eccS5[0] != eccS5[1] or exceed25[0] != exceed25[1]
                    or DES["333"]["k_gap"] != DES["331"]["k_gap"]
                    or sum(x < ALPHA for x in dip_bonf["333"])
                    != sum(x < ALPHA for x in dip_bonf["331"])
                    or (DES["333"]["proj_pmin_bonf"] < ALPHA)
                    != (DES["331"]["proj_pmin_bonf"] < ALPHA))
    log()
    log('Block (a), in plain words: '
        + (f"at n = {len(P331)} the set of statistics that exceed their own null, the k chosen by "
           f"the gap and the outcome of the dip tests are the same as at n = {len(P333)}. No "
           f"conclusion changes."
           if not changes_a else
           f"at n = {len(P331)} AT LEAST ONE among the set of statistics that exceed the null, the k "
           f"chosen by the gap and the outcome of the dip tests is different from n = {len(P333)}: "
           f"see the rows above."))
    log('Not redone: the fraction of null realisations in which the gap chooses k = 1 (script 45) '
        'and the diagnostic')
    log("curves over k = 2-10; here the gap is computed on the observed data only.")

    # ========================================================================
    title('4. (d) BATTERY WITHOUT THE PARTICIPANTS SHARED WITH THE PRIMARY COHORT (§1.8, script 66)')
    log(f'The manuscript reports the battery without the {len(shared)} shared participants (n = '
        f'{len(P288)}), with B = {B66}. The {len(TWO)} are among the {len(P288)}: {len(two_in_288)} '
        f'out of {len(TWO)}.')
    T66 = {tag: table_batt(tag, B66) for tag in ("288", "286", "333", "331")}
    if two_in_288:
        ecc66 = printout_comparison(f'§1.8: battery of script 66, without the shared participants, B = '
                                 f'{B66}',
                                 T66["288"], T66["286"], "288", "286")
        changes_d = bool(ecc66[0] != ecc66[1])
        log()
        log('Block (d), in plain words: '
            + (f'at n = {len(P286)} the same statistics exceed the null as at n = {len(P288)}. No '
               f'conclusion changes.'
               if not changes_d else
               f"at n = {len(P286)} the set of statistics that exceed the null is different from "
               f"that at n = {len(P288)}: see the rows above."))
    else:
        changes_d = False
        log("Neither of the two is among the non-shared participants: the branch without the shared participants does not change.")
    printout_comparison(f'§1.8, term of comparison: whole cohort read on the first {B66} realisations', T66["333"], T66["331"], "333", "331")

    # ========================================================================
    def report_family(D, name_fam, units):
        n = len(D)
        for tag in ("333", "331"):
            ko = D[~D[f'accepted_{tag}']]
            if len(ko):
                log(f'  WARNING, branch at {len(POP[tag])}: {len(ko)} models out of {n} do NOT meet '
                    f'the strict criterion and are excluded from the correction (effective family '
                    f'{n - len(ko)}): '
                    + listing(f'{r.outcome} x {r.kinetic}' for r in ko.itertuples()))
            else:
                log(f"  Branch at {len(POP[tag])}: all the {n} models meet the strict criterion "
                    f"(maximum Newton decrement {D[f'newton_{tag}'].max():.1e}, threshold "
                    f"{TOL_NEWTON:g}; Hessian negative definite in "
                    f"{int(D[f'hessian_def_negative_{tag}'].sum())}/{n}; on the boundary "
                    f"{int(D[f'at_edge_{tag}'].sum())}; singular {int(D[f'singular_{tag}'].sum())}).")
        log(f'  Unit of beta: {units}.')
        log(f"  "
            f"{'outcome':<22}{'kin.':<6}{'n/obs 333':>10}{'n/obs 331':>10}{'beta 333':>10}{'SE 333':>9}{'p 333':>8}{'BH 333':>8}{'beta 331':>10}{'SE 331':>9}{'p 331':>8}{'BH 331':>8}{'shift':>8}")
        for r in D.itertuples():
            log(f"  "
                f"{r.outcome[:21]:<22}{r.kinetic:<6}{f'{r.n_subj_333}/{r.n_obsv_333}':>10}{f'{r.n_subj_331}/{r.n_obsv_331}':>10}{r.beta_333:>+10.5f}{r.se_333:>9.5f}{r.p_333:>8.4f}{r.p_BH_333:>8.4f}{r.beta_331:>+10.5f}{r.se_331:>9.5f}{r.p_331:>8.4f}{r.p_BH_331:>8.4f}{r.shift_in_SE_333:>+8.3f}")
        log(f"  ('spost.' = (beta at 331 - beta at 333) / SE at 333; uncorrected p; BH = "
            f"Benjamini-Hochberg within the {n} tests of the family, per branch.)")
        i = D.shift_in_SE_333.abs().idxmax()
        j = D.shift_in_SE_333_a_scale_fixed.abs().idxmax()
        log(f'  Maximum shift: {D.shift_in_SE_333[i]:+.3f} SE ({D.outcome[i]} x {D.kinetic[i]}; beta '
            f'{D.beta_333[i]:+.5f} -> {D.beta_331[i]:+.5f}, SE at 333 {D.se_333[i]:.5f}); median of '
            f'the |shifts| {D.shift_in_SE_333.abs().median():.3f} SE.')
        log(f'  With the kinetic parameter brought back to the SD of the {len(P333)} (exact '
            f'rescaling of the coefficient at {len(P331)}): maximum shift '
            f'{D.shift_in_SE_333_a_scale_fixed[j]:+.3f} SE ({D.outcome[j]} x {D.kinetic[j]}).')
        log(f'  Ratio between SE at {len(P331)} and SE at {len(P333)}: from '
            f'{D.ratio_SE_331_over_333.min():.4f} to {D.ratio_SE_331_over_333.max():.4f}.')
        log(f'  Nominally significant (p < {ALPHA}): {int(D.nominal_333.sum())} out of {n} at '
            f'{len(P333)}, {int(D.nominal_331.sum())} out of {n} at {len(P331)}. After BH (< '
            f'{ALPHA}): {int(D.BH_333.sum())} and {int(D.BH_331.sum())}. Minimum BH p: '
            f'{D.p_BH_333.min():.4f} and {D.p_BH_331.min():.4f}.')
        for col, name in (('changes_state_nominal', 'nominal status'),
                          ('changes_state_BH', 'status after BH'), ('changes_sign', 'sign')):
            q = D[D[col]]
            if len(q):
                log(f'  Change {name}: {len(q)} out of {n}:')
                for r in q.itertuples():
                    log(f'    {r.outcome} x {r.kinetic}: beta {r.beta_333:+.5f} -> {r.beta_331:+.5f} '
                        f'({r.shift_in_SE_333:+.3f} SE), p {r.p_333:.4f} -> {r.p_331:.4f}, BH p '
                        f'{r.p_BH_333:.4f} -> {r.p_BH_331:.4f}; distance of p at {len(P333)} from '
                        f'the threshold {abs(r.p_333 - ALPHA):.4f}')
            else:
                log(f'  Change {name}: none out of {n}.')
        large = D[~(D.shift_in_SE_333.abs() < THRESHOLD_ATTRIB)]
        if len(large):
            log(f'  Coefficients that shift by at least {THRESHOLD_ATTRIB} SE: {len(large)} out of '
                f'{n}. For each, the model is re-estimated')
            log(f'  removing ONE participant ONLY (n = {len(P333) - 1}); shifts at fixed scale, in '
                f'SE of the branch at {len(P333)}:')
            for r in large.itertuples():
                q = ATTR[(ATTR.outcome == r.outcome) & (ATTR.kinetic == r.kinetic)]
                assert len(q) == len(TWO), f'attribution missing for {r.outcome} x {r.kinetic}'
                log(f'    {r.outcome} x {r.kinetic}: both removed '
                    f'{r.shift_in_SE_333_a_scale_fixed:+.3f}; '
                    + "; ".join(
                        f'removed only {x.participant_removed} {x.shift_in_SE_333_a_scale_fixed:+.3f}'
                        + ("" if x.accepted_fromthe_criterion_strict else ' (model NOT accepted)')
                        for x in q.itertuples()))
            log('  What the participants removed contribute to these models (observations, years, '
                'observed annual change')
            log(f"  of the outcome between the first and the last observation, against the mean "
                f"slope estimated by the model at {len(P333)}; z of the kinetic parameter):")
            for (se, _g) in large.groupby('outcome', sort=False):
                q = ATTR[ATTR.outcome == se]
                for et in sorted(LBLS.values()):
                    x = q[q.participant_removed == et]
                    x0 = x.iloc[0]
                    var = ('no change defined (fewer than two observations)'
                           if not np.isfinite(x0.change_annual_observed) else
                           f'observed change {x0.change_annual_observed:+.4f}/year against a mean '
                           f'slope of {x0.slope_mean_of_model_a_333:+.4f}/year')
                    no = int(x0.obsv_of_participant_in_model)
                    when = ('no observation' if no == 0 else
                              f'1 observation (year {x0.first_year:.0f})' if no == 1 else
                              f'{no} observations (years {x0.first_year:.0f}-{x0.last_year:.0f})')
                    log(f'    {se}, participant {et}: {when}; {var}; z of the kinetic parameter (and '
                        f'share of the sum of squares of the z in the cohort): '
                        + ", ".join(
                            f'{y.kinetic} {y.z_kinetic_of_participant:+.2f} '
                            f'({100 * y.share_of_total_of_squares_of_kinetic:.1f}%)'
                            for y in x.itertuples()))
            # to whom it is attributed: the participant whose removal alone reproduces
            # most of the total shift
            shares = []
            for r in large.itertuples():
                q = ATTR[(ATTR.outcome == r.outcome) & (ATTR.kinetic == r.kinetic)]
                q = q.assign(d=(q.shift_in_SE_333_a_scale_fixed
                                - r.shift_in_SE_333_a_scale_fixed).abs())
                shares.append(q.sort_values(["d", 'participant_removed']).iloc[0].participant_removed)
            cq = count(pd.Series(shares))
            log("  Participant whose removal alone comes closest to the total shift: "
                + "; ".join(f'{k} in {v} coefficients out of {len(large)}' for k, v in cq.items())
                + ".")
        else:
            log(f'  No coefficient shifts by {THRESHOLD_ATTRIB} SE or more: no attribution to make.')
        cols_two = [c for c in D.columns if c.startswith('obsv_of_participant_')]
        tot = D[cols_two].sum(axis=1)
        log(f'  Observations that the {len(TWO)} contribute to a model: from {int(tot.min())} to '
            f'{int(tot.max())} out of {int(D.n_obsv_333.min())}-{int(D.n_obsv_333.max())}; subjects '
            f'removed per model: from {int((D.n_subj_333 - D.n_subj_331).min())} to '
            f'{int((D.n_subj_333 - D.n_subj_331).max())}.')
        changes = bool(D.changes_state_BH.any() or D.changes_state_nominal.any()
                      or (~D.accepted_333).any() or (~D.accepted_331).any())
        i = D.shift_in_SE_333.abs().idxmax()
        tail = (f'the maximum shift of a coefficient is {abs(D.shift_in_SE_333[i]):.2f} SE '
                f'({D.outcome[i]} x {D.kinetic[i]}, p {D.p_333[i]:.3f} -> {D.p_331[i]:.3f}); change '
                f'sign {int(D.changes_sign.sum())} coefficients out of {n}'
                + (f' (minimum uncorrected p among these, in the two arms: '
                   f'{min(D[D.changes_sign].p_333.min(), D[D.changes_sign].p_331.min()):.3f})'
                   if D.changes_sign.any() else ""))
        return changes, tail

    # ------------------------------------------------------------------ (b)
    title('5. (b) KINETIC PARAMETER x TIME FAMILY OF THE 150 h COHORT (35 tests, §1.12)')
    log(f'Seven outcomes x five kinetic parameters = {len(keys35)} kinetic parameter x years '
        f'interactions; random intercept and slope, REML,')
    log("strict criterion. Covariates: baseline age, sex, baseline duration (+ education for MoCA).")
    log(f"Populations: {NAMEPOP['333']}; {NAMEPOP['331']}.")
    log('Family for the correction: the 35 tests of the 150 h cohort (BH within cohort, as in '
        '§1.12), recomputed per branch.')
    f88 = os.path.join(OUT_CAUDATE, "88_diagnostics_63.csv")
    fc = os.path.join(OUT_CAUDATE, "08_dataset_150h.csv")
    if os.path.exists(f88) and os.path.exists(fc):
        cc = pd.read_csv(fc)
        cc["PATNO"] = cc["PATNO"].astype("int64")
        if "complete" in cc.columns:
            cc = cc[cc.complete]
        assert cc.PATNO.is_unique, 'repeated PATNO in the cohort of run_caudate'
        assert set(cc.PATNO) == P333 and np.allclose(
            cc.set_index("PATNO").loc[c150.PATNO, list(KIN150)].to_numpy(float),
            c150[list(KIN150)].to_numpy(float), rtol=TOL_COHORT, atol=0), (
            'the 150 h cohort of the working directory run_caudate is not that of this working '
            'directory: 88_diagnostics_63.csv is not a valid reference')
        q = load_csv(f88)
        q = q[q.cohort == "150h"].set_index(["outcome", "kinetic"])
        assert set(q.index) == set(keys35), (
            f'the family of 88_diagnostics_63.csv (run_caudate) is not that of the 35 tests: only '
            f'here {sorted(set(keys35) - set(q.index))}; only there '
            f'{sorted(set(q.index) - set(keys35))}')
        q = q.loc[keys35]
        assert (q.n_subjects.to_numpy() == D35.n_subj_333.to_numpy()).all() and (
            q.n_obs.to_numpy() == D35.n_obsv_333.to_numpy()).all(), \
            'subjects or observations different from 88_diagnostics_63.csv'
        sc = {"beta": departure_max(q.beta_interaction, D35.beta_333),
              'SE': departure_max(q.se, D35.se_333), "p": departure_max(q.p, D35.p_333),
              'BH p': departure_max(q.p_BH_within_cohort, D35.p_BH_333)}
        assert max(sc.values()) < TOL_ID_MIXED, (
            f'the branch at n = {len(P333)} does NOT reproduce 88_diagnostics_63.csv: {sc}')
        log(f"Identity with the pipeline (branch at n = {len(P333)}): same 35 (outcome, kinetic "
            f"parameter), same subjects and observations as")
        log('  run_caudate/outputs/88_diagnostics_63.csv (cohort of that working directory '
            'identical to this one for PATNO and kinetic parameters);')
        log('  maximum discrepancy ' + ", ".join(f"{k} {v:.1e}" for k, v in sc.items())
            + f' (tolerance {TOL_ID_MIXED:g}).')
    else:
        log('Identity with the pipeline: NOT verified, run_caudate/outputs/88_diagnostics_63.csv or '
            '08_dataset_150h.csv missing.')
        no_verified('35 tests: identity of the branch at n = 333 with '
                       'run_caudate/outputs/88_diagnostics_63.csv')
    changes_b, tail_b = report_family(
        D35, '35 tests', 'units of the outcome per year per 1 SD of the logarithm of the kinetic '
                        'parameter, SD of the population of the arm')
    log()
    log('Block (b), in plain words: '
        + (f'none of the {len(D35)} interactions passes the correction either at {len(P333)} or at '
           f'{len(P331)} and none changes nominal or corrected status; {tail_b}. No conclusion '
           f'changes.'
           if (not changes_b and not D35.BH_333.any() and not D35.BH_331.any()) else
           f'at least one of the conditions (nominal status unchanged, corrected status unchanged, '
           f'all models accepted, no survivor of the correction) does not hold: see the rows above; '
           f'{tail_b}.'))

    # ------------------------------------------------------------------ (c)
    title('6. (c) THE TWELVE PRIMARY NON-MOTOR OUTCOMES (T50 x years, Table S9)')
    log("Model: outcome ~ z(log T50) x years + baseline age + sex + baseline duration [+ education "
        "for the five")
    log('cognitive outcomes]; random intercept and slope, REML, strict criterion.')
    log(f"Populations: {NAMEPOP['333']}; {NAMEPOP['331']}.")
    log('Family for the correction: the twelve primary outcomes (BH within the twelve, as in Table '
        'S9), recomputed per branch.')
    f91 = os.path.join(OUT, "91_table6_strict_criterion.csv")
    if os.path.exists(f91):
        q = load_csv(f91).set_index("outcome")
        assert set(q.index) == {v for v, _, _ in OUTCOMES12}, \
            'the outcomes of 91_table6_strict_criterion.csv are not the twelve expected'
        q = q.loc[[v for v, _, _ in OUTCOMES12]]
        assert (q.n_subjects.to_numpy() == D12.n_subj_333.to_numpy()).all() and (
            q.n_obs.to_numpy() == D12.n_obsv_333.to_numpy()).all(), \
            'subjects or observations different from 91_table6_strict_criterion.csv'
        sc = {"beta": departure_max(q.beta_ruleB, D12.beta_333),
              'SE': departure_max(q.se_ruleB, D12.se_333), "p": departure_max(q.p_ruleB, D12.p_333),
              'BH p': departure_max(q.p_BH_ruleB, D12.p_BH_333)}
        assert max(sc.values()) < TOL_ID_MIXED, (
            f'the branch at n = {len(P333)} does NOT reproduce 91_table6_strict_criterion.csv: {sc}')
        log(f"Identity with the pipeline (branch at n = {len(P333)}): same subjects and observations "
            f"as 91_table6_strict_criterion.csv")
        log('  (strict criterion); maximum discrepancy '
            + ", ".join(f"{k} {v:.1e}" for k, v in sc.items())
            + f' (tolerance {TOL_ID_MIXED:g}).')
    else:
        log("Identity with the pipeline: NOT verified, 91_table6_strict_criterion.csv missing.")
        no_verified("12 non-motor outcomes: identity of the branch at n = 333 with "
                       "91_table6_strict_criterion.csv")
    changes_c, tail_c = report_family(
        D12, '12 tests', 'units of the outcome per year per 1 SD of the logarithm of T50, SD of the '
                        'population of the arm')
    nom333 = [r.outcome for r in D12.itertuples() if r.nominal_333]
    nom331 = [r.outcome for r in D12.itertuples() if r.nominal_331]
    log(f'  Nominally significant at {len(P333)}: {listing(nom333)}; at {len(P331)}: '
        f'{listing(nom331)}.')
    log()
    log('Block (c), in plain words: '
        + (f'none of the {len(D12)} outcomes passes the correction either at {len(P333)} or at '
           f'{len(P331)}; the nominally significant outcomes are the same; {tail_c}. No conclusion '
           f'changes.'
           if (not changes_c and not D12.BH_333.any() and not D12.BH_331.any()) else
           f'at least one of the conditions (nominal status unchanged, corrected status unchanged, '
           f'all models accepted, no survivor of the correction) does not hold: see the rows above; '
           f'{tail_c}.'))

    # ========================================================================
    title('7. (e) CSF ANALYSES ON THE 150 h COHORT (§1.16-1.17)')
    log('Spearman correlations, raw and partial (Pearson between the residuals of the ranks). The')
    log("common component of the analytes (PC1 of the ranks) is RECOMPUTED in every branch on the "
        "subjects of the branch.")
    blA = blc[["PATNO", "age_at_visit", "SEX"] + ANALYTES]
    alb = None
    if os.path.exists(CACHE_ALB):
        a = pd.read_csv(CACHE_ALB, low_memory=False)
        a["PATNO"] = a["PATNO"].astype("int64")
        a = a[a.CLINICAL_EVENT.isin(["BL", "SC"])].copy()
        a["v"] = pd.to_numeric(a.TESTVALUE, errors="coerce")
        alb = (a[a.TESTNAME == "CSF Albumin"].groupby("PATNO").v.mean()
               .rename("alb_csf").reset_index())
    CSF = {}
    for tag in ("333", "331"):
        d = prepare_csf(c150[c150.PATNO.isin(POP[tag])], KIN150, blA, alb)
        d["conc"], avail, info = factor12(d)
        assert avail == ANALYTES, f'branch at {tag}: available analytes different from the seven'
        cc = pairs12(d, list(KIN150.values()), avail)
        lo = loo36(d, list(KIN150.values()))
        cc = cc.merge(lo, on=['kinetic', 'biomarker'], how="left", validate="one_to_one")
        CSF[tag] = dict(d=d, info=info, cc=cc, fact=kinet_vs(d, list(KIN150.values()), "conc", 60))
    d24 = prepare_csf(c24[~c24.PATNO.isin(TWO)], KIN24, blA, alb)
    d24["conc"], avail24, _info24 = factor12(d24)
    fat24 = kinet_vs(d24, list(KIN24.values()), "conc", 60)

    # identity of the 333 branch
    f12 = os.path.join(OUT, "12_kinetics_biomarkers_adjusted.csv")
    if os.path.exists(f12):
        q = load_csv(f12)
        q = q[q.cohort == "150h"].set_index(['kinetic', 'biomarker'])
        t = CSF["333"]["cc"].set_index(['kinetic', 'biomarker'])
        assert set(q.index) == set(t.index), '12_kinetics_biomarkers_adjusted.csv: different pairs'
        q = q.loc[t.index]
        s12 = max(departure_max(q[c], t[c]) for c in ("n", "rho_L1", "p_L1", "rho_L2", "p_L2",
                                                   "rho_L3", "p_L3", "pBH_L1", "pBH_L2",
                                                   "pBH_L3"))
        assert s12 < TOL_ID_BATT, (f'the branch at n = {len(P333)} does NOT reproduce '
                                   f'12_kinetics_biomarkers_adjusted.csv: discrepancy {s12:.3g}')
        log(f"Identity with script 12 (branch at n = {len(P333)}): the {len(t)} pairs, three levels, "
            f"match 12_kinetics_biomarkers_adjusted.csv; maximum discrepancy {s12:.1e}.")
    else:
        log("Identity with script 12: NOT verified, file missing.")
        no_verified("CSF: identity with 12_kinetics_biomarkers_adjusted.csv")
    f36 = os.path.join(OUT, "36_loo_component.csv")
    if os.path.exists(f36):
        q = load_csv(f36)
        t = CSF["333"]["cc"]
        # 36 writes the analytes with an extended label and in the same order as
        # ANALITI: the comparison is by (kinetic parameter, position of the analyte)
        ordering = {a: i for i, a in enumerate(ANALYTES)}
        tt = t.assign(o=t.biomarker.map(ordering)).sort_values(["o", 'kinetic'])
        q = q.assign(o=np.repeat(np.arange(len(ANALYTES)), len(KIN150))).sort_values(
            ["o", 'kinetic'])
        assert list(q.kinetic) == list(tt.kinetic), '36_loo_component.csv: unexpected order'
        s36 = max(departure_max(q.rho_factor_loo, tt.rho_LOO), departure_max(q.p_factor_loo, tt.p_LOO),
                  departure_max(q.pBH_factor_loo, tt.pBH_LOO), departure_max(q.rho_raw, tt.rho_L1))
        assert s36 < TOL_ID_BATT, (f'the branch at n = {len(P333)} does NOT reproduce '
                                   f'36_loo_component.csv: discrepancy {s36:.3g}')
        log(f"Identity with script 36 (leave-one-analyte-out): maximum discrepancy {s36:.1e}.")
    else:
        log("Identity with script 36: NOT verified, file missing.")
        no_verified("CSF: identity with 36_loo_component.csv")

    rows_alb = []
    f12f = os.path.join(OUT, "12_kinetics_vs_concfactor.csv")
    FACT = {}
    for tag in ("333", "331"):
        t = pd.concat([CSF[tag]['fact'].assign(cohort="150h"), fat24.assign(cohort="24h")],
                      ignore_index=True)
        t["p_BH"] = bh(t.p)
        FACT[tag] = t
    if os.path.exists(f12f):
        q = load_csv(f12f).set_index(['cohort', 'kinetic'])
        t = FACT["333"].set_index(['cohort', 'kinetic'])
        assert set(q.index) == set(t.index)
        q = q.loc[t.index]
        s = max(departure_max(q[c], t[c]) for c in ("n", 'rho_partial', "p", "p_BH"))
        assert s < TOL_ID_BATT, f'12_kinetics_vs_concfactor.csv not reproduced: discrepancy {s:.3g}'
        log(f"Identity with script 12 (kinetic parameters against the component): maximum "
            f"discrepancy {s:.1e}.")
    else:
        log("Identity with 12_kinetics_vs_concfactor.csv: NOT verified, file missing.")
        no_verified("CSF: identity with 12_kinetics_vs_concfactor.csv")

    log()
    log(f"Kinetic parameter-analyte pairs: {len(CSF['333']['cc'])} (five kinetic parameters x seven "
        f"analytes). Family: the 35 pairs of the cohort,")
    log("BH within level of adjustment, recomputed per branch. Unit: participants with the pair "
        "defined.")
    LEV = [("L1", 'raw'), ("L2", 'age and sex'),
           ("L3", 'age, sex and common component'),
           ("LOO", 'age, sex and leave-one-analyte-out component')]
    log(f"  "
        f"{'level':<48}{'BH<0.05 at 333':>14}{'BH<0.05 at 331':>14}{'|rho| max 333':>14}{'|rho| max 331':>14}")
    tallies_csf = {}
    for lv, name in LEV:
        k3 = int((CSF["333"]["cc"][f"pBH_{lv}"] < ALPHA).sum())
        k1 = int((CSF["331"]["cc"][f"pBH_{lv}"] < ALPHA).sum())
        tallies_csf[lv] = (k3, k1)
        log(f"  "
            f"{name:<48}{k3:>14}{k1:>14}{CSF['333']['cc'][f'rho_{lv}'].abs().max():>14.3f}{CSF['331']['cc'][f'rho_{lv}'].abs().max():>14.3f}")
    for tag in ("333", "331"):
        i = CSF[tag]["info"]
        log(f"  Common component, branch at {len(POP[tag])}: {i['n']} participants with the seven "
            f"analytes; variance explained by PC1 {i['var_pc1']:.3f}; |loading| from "
            f"{i['loading_min']:.2f} to {i['loading_max']:.2f}, "
            + ('all of the same sign.' if i['loadings_same_sign'] else 'of MIXED sign.'))
    m = CSF["333"]["cc"].merge(CSF["331"]["cc"], on=['kinetic', 'biomarker'],
                               suffixes=("_333", "_331"), validate="one_to_one")
    for lv, name in LEV:
        dr = (m[f"rho_{lv}_331"] - m[f"rho_{lv}_333"]).abs()
        chg = m[(m[f"pBH_{lv}_333"] < ALPHA) != (m[f"pBH_{lv}_331"] < ALPHA)]
        log(f'  {name}: maximum |difference in rho| between the branches {dr.max():.4f}; pairs that '
            f'change status after BH: {len(chg)}'
            + ("." if not len(chg) else ": " + listing(
                f"{r.kinetic}-{r.biomarker} (BH p {getattr(r, f'pBH_{lv}_333'):.4f} -> "
                f"{getattr(r, f'pBH_{lv}_331'):.4f})" for r in chg.itertuples())))
    two_csf = {LBLS[pt]: bool(CSF["333"]["d"].set_index("PATNO").loc[pt, ANALYTES].notna().all())
               for pt in sorted(TWO)}
    log(f'  Of the {len(TWO)}, with the seven analytes at baseline (hence inside the component at '
        f'{len(P333)}): {sum(two_csf.values())}.')
    log("  Kinetic parameters against the component (rho partial for age and sex; BH over the tests "
        "of the two cohorts together, as in script 12):")
    for tag in ("333", "331"):
        t = FACT[tag][FACT[tag].cohort == "150h"]
        log(f'    branch at {len(POP[tag])} (n = {int(t.n.iloc[0])}): '
            + "; ".join(f'{r.kinetic} {r.rho_partial:+.3f} (p {r.p:.2g}, BH {r.p_BH:.2g})'
                        for r in t.itertuples()))
        for r in t.itertuples():
            rows_alb.append(dict(
                analysis='kinetic parameter against the common component of the analytes (script 12)',
                population=NAMEPOP[tag], kinetic=r.kinetic, n=int(r.n),
                rho_partial_age_sex=r.rho_partial, p_no_corrected=r.p, p_BH=r.p_BH,
                family_BH=f'{len(FACT[tag])} tests: five kinetic parameters at 150 h and four at '
                            f'24 h, together'))

    changes_e_alb = False
    if alb is not None:
        ALBT = {}
        a24 = kinet_vs(d24, list(KIN24.values()), "alb_csf", 40)
        for tag in ("333", "331"):
            t = pd.concat([kinet_vs(CSF[tag]["d"], list(KIN150.values()), "alb_csf", 40)
                           .assign(cohort="150h"), a24.assign(cohort="24h")], ignore_index=True)
            t["p_BH"] = bh(t.p)
            ALBT[tag] = t
        f20 = os.path.join(OUT, "20_kinetics_vs_albumin.csv")
        if os.path.exists(f20):
            q = load_csv(f20).set_index(['cohort', 'kinetic'])
            t = ALBT["333"].set_index(['cohort', 'kinetic'])
            assert set(q.index) == set(t.index), '20_kinetics_vs_albumin.csv: different rows'
            q = q.loc[t.index]
            s = max(departure_max(q[c], t[c]) for c in ("n", 'rho_partial', "p", "p_BH"))
            assert s < TOL_ID_BATT, (f'the branch at n = {len(P333)} does NOT reproduce '
                                     f'20_kinetics_vs_albumin.csv: discrepancy {s:.3g}')
            log(f"Identity with script 20 (CSF albumin): maximum discrepancy {s:.1e}.")
        else:
            log("Identity with 20_kinetics_vs_albumin.csv: NOT verified, file missing.")
            no_verified("CSF: identity with 20_kinetics_vs_albumin.csv")
        log(f"  CSF albumin against the kinetic parameters (rho partial for age and sex; BH over the "
            f"{len(ALBT['333'])} tests of the two cohorts together, as in script 20):")
        for tag in ("333", "331"):
            t = ALBT[tag][ALBT[tag].cohort == "150h"]
            log(f'    branch at {len(POP[tag])} (n = {int(t.n.iloc[0])}): '
                + "; ".join(f'{r.kinetic} {r.rho_partial:+.3f} (p {r.p:.2g}, BH {r.p_BH:.2g})'
                            for r in t.itertuples()))
            for r in t.itertuples():
                rows_alb.append(dict(
                    analysis='kinetic parameter versus CSF albumin (script 20)',
                    population=NAMEPOP[tag], kinetic=r.kinetic, n=int(r.n),
                    rho_partial_age_sex=r.rho_partial, p_no_corrected=r.p, p_BH=r.p_BH,
                    family_BH=f'{len(ALBT[tag])} tests: five kinetic parameters at 150 h and four '
                                f'at 24 h, together'))
        a3 = ALBT["333"][ALBT["333"].cohort == "150h"].set_index('kinetic')
        a1 = ALBT["331"][ALBT["331"].cohort == "150h"].set_index('kinetic')
        sig3 = set(a3[a3.p_BH < ALPHA].index)
        sig1 = set(a1[a1.p_BH < ALPHA].index)
        changes_e_alb = sig3 != sig1
        log(f'    Significant after BH at {len(P333)}: {listing(sorted(sig3))}; at {len(P331)}: '
            f'{listing(sorted(sig1))}; maximum |difference in rho| '
            f'{(a1.rho_partial - a3.rho_partial).abs().max():.4f}.')
    else:
        log('CSF albumin: NOT run, cache/alb.csv missing from the working directory (this script '
            'does not rebuild it).')
        no_verified("CSF: albumin analysis not run (cache/alb.csv missing)")
    changes_e = bool(any(a != b for a, b in tallies_csf.values()) or changes_e_alb)
    log()
    log('Block (e), in plain words: '
        + (f"the number of pairs that pass the correction at each level of adjustment is the same at "
           f"{len(P333)} and at {len(P331)}"
           + (", and the kinetic parameters associated with albumin after correction are the same"
              if alb is not None else "") + '. No conclusion changes.'
           if not changes_e else
           "at least one count of significant pairs or the set of kinetic parameters associated "
           "with albumin changes: see the rows above."))
    log("Not redone: the checks on haemoglobin and Qalb (script 23), the adjustment for albumin of "
        "the pairs")
    log("(script 20, section 3), the substrate hypothesis (script 12, section d).")

    # ========================================================================
    title('8. SUMMARY')
    rows_s = []

    def item(block, quantity, v333, v331, units):
        rows_s.append(dict(block=block, quantity=quantity, value_branch_entire=v333,
                            value_branch_without_i_two=v331, units=units))

    item('population', 'participants', len(P333), len(P331), 'participants')
    item('population', 'participants without those shared with the 24 h cohort', len(P288),
         len(P286), 'participants')
    item('typing at 24 h', 'typed at baseline', len(PROF["333"]["tip_bl"]),
         len(PROF["331"]["tip_bl"]), 'participants')
    item('typing at 24 h', 'typed at any visit', len(PROF["333"]['typed']),
         len(PROF["331"]['typed']), 'participants')
    item('typing at 24 h', 'never typed', len(P333) - len(PROF["333"]['typed']),
         len(P331) - len(PROF["331"]['typed']), 'participants')
    item('typing at 24 h', 'with at least one Type2 at any visit',
         len(PROF["333"]["t2"]), len(PROF["331"]["t2"]), 'participants')
    for s in STAT:
        a = S5["333"].set_index('statistic').loc[s]
        b = S5["331"].set_index('statistic').loc[s]
        item('(a) Table S5', f'{NAME_STAT[s]}: observed', a.observed, b.observed,
             'value of the statistic')
        item('(a) Table S5', f'{NAME_STAT[s]}: Monte Carlo p', a.p_montecarlo, b.p_montecarlo,
             f'uncorrected one-sided p, B = {B25}')
    item('(a) designated estimates', 'k chosen by the gap', DES["333"]["k_gap"], DES["331"]["k_gap"],
         'number of clusters')
    for s in STAT:
        a = T66["288"].set_index('statistic').loc[s]
        b = T66["286"].set_index('statistic').loc[s]
        item('(d) without the shared participants (script 66)', f'{NAME_STAT[s]}: observed', a.observed,
             b.observed, 'value of the statistic')
        item('(d) without the shared participants (script 66)', f'{NAME_STAT[s]}: Monte Carlo p',
             a.p_montecarlo, b.p_montecarlo, f'uncorrected one-sided p, B = {B66}')
    for name, D in (('(b) 35 kinetic parameter x time tests', D35), ('(c) 12 non-motor outcomes', D12)):
        item(name, 'nominally significant tests', int(D.nominal_333.sum()),
             int(D.nominal_331.sum()), f'tests out of {len(D)}, uncorrected p < {ALPHA}')
        item(name, 'tests that pass BH', int(D.BH_333.sum()), int(D.BH_331.sum()),
             f'tests out of {len(D)}, BH p < {ALPHA}')
        item(name, 'minimum BH p', float(D.p_BH_333.min()), float(D.p_BH_331.min()), 'BH p')
        item(name, 'maximum shift of the coefficient', np.nan,
             float(D.shift_in_SE_333.abs().max()), 'SE of the arm at 333')
        item(name, 'maximum shift at fixed scale', np.nan,
             float(D.shift_in_SE_333_a_scale_fixed.abs().max()), 'SE of the arm at 333')
        item(name, 'tests that change nominal status', np.nan,
             int(D.changes_state_nominal.sum()), "test")
        item(name, 'tests that change status after BH', np.nan, int(D.changes_state_BH.sum()),
             "test")
        item(name, 'tests that change sign', np.nan, int(D.changes_sign.sum()), "test")
        item(name, f'coefficients shifted by at least {THRESHOLD_ATTRIB} SE (attributed)', np.nan,
             int((~(D.shift_in_SE_333.abs() < THRESHOLD_ATTRIB)).sum()), "test")
        item(name, 'models accepted by the strict criterion', int(D.accepted_333.sum()),
             int(D.accepted_331.sum()), f'models on {len(D)}')
    for lv, name in LEV:
        item('(e) CSF, 35 pairs', f'pairs with BH p < {ALPHA}, level: {name}',
             tallies_csf[lv][0], tallies_csf[lv][1], 'pairs out of 35')
    item('(e) CSF', 'variance explained by the PC1 of the analytes',
         CSF["333"]["info"]["var_pc1"], CSF["331"]["info"]["var_pc1"], 'fraction')
    item('(e) CSF', 'participants with the seven analytes', CSF["333"]["info"]["n"],
         CSF["331"]["info"]["n"], 'participants')

    outcomes_blocks = [('(a) battery at k = 2 against the copula null', changes_a),
                     ('(b) 35 kinetic parameter x time tests', changes_b),
                     ('(c) 12 non-motor outcomes', changes_c),
                     ('(d) battery without the shared participants', changes_d),
                     ('(e) CSF analyses', changes_e)]
    for name, c in outcomes_blocks:
        log(f"  {name:<52} {('SOMETHING CHANGES: see the block' if c else 'no change of status')}")
        rows_s.append(dict(block=name, quantity='something changes between the two arms '
                                                   '(statuses, acceptance of the models, shifts '
                                                   'beyond threshold)',
                            value_branch_entire=np.nan, value_branch_without_i_two=float(c),
                            units='0 = no, 1 = yes'))
    log()
    log(f'The {len(TWO)} participants removed are those Type2 at 24 h at BASELINE. In the cohort at '
        f'{len(P331)} there remain: with a Type2 at a later')
    log(f"visit {len(PROF['331']['t2'])}; never typed at 24 h "
        f"{len(P331) - len(PROF['331']['typed'])}; with at least one Type1 at some visit "
        f"{len(PROF['331']['t1'])}.")
    log()
    if no_executed:
        log(f'GUARDS NOT RUN IN THIS RUN ({len(no_executed)}):')
        for x in no_executed:
            log(f'  - NOT verified: {x}')
    else:
        log('Guards not run in this run: none.')

    # ========================================================================
    # OUTPUT
    # ========================================================================
    pd.DataFrame(rows_two).to_csv(os.path.join(OUT, "112_two_participants.csv"), index=False)
    pd.DataFrame(rows_tip).to_csv(os.path.join(OUT, "112_typing_24h.csv"), index=False)
    pd.DataFrame(rows_batt).to_csv(os.path.join(OUT, "112_battery.csv"), index=False)
    nn = []
    for tag in ("333", "331", "288", "286"):
        t = NUL[tag].copy()
        t.insert(0, 'realisation', np.arange(1, len(t) + 1))
        t.insert(0, 'population', NAMEPOP[tag])
        nn.append(t)
    pd.concat(nn, ignore_index=True).to_csv(os.path.join(OUT, "112_battery_nulls.csv"),
                                            index=False)
    for D, name, fam, an in (
            (D35, "112_kinetics_by_time.csv",
             '35 kinetic parameter x years tests of the 150 h cohort; BH within the 35, by arm',
             'units of the outcome per year per 1 SD of the log of the kinetic parameter (SD of the '
             'arm)'),
            (D12, "112_not_motor.csv",
             '12 primary non-motor outcomes, T50 x years; BH within the 12, by arm',
             'units of the outcome per year per 1 SD of the log T50 (SD of the arm)')):
        D = D.copy()
        D.insert(1, 'family_of_correction', fam)
        D.insert(2, 'units_of_beta', an)
        D.insert(3, 'population_333', NAMEPOP["333"])
        D.insert(4, 'population_331', NAMEPOP["331"])
        D.to_csv(os.path.join(OUT, name), index=False)
    m.insert(0, 'family_of_correction',
             '35 kinetic parameter-analyte pairs of the 150 h cohort; BH within level, by arm')
    m.insert(1, 'population_333', NAMEPOP["333"])
    m.insert(2, 'population_331', NAMEPOP["331"])
    m.insert(3, 'units', 'Spearman rho (L1) or partial on ranks (L2, L3, LOO); n = participants '
                         'with the pair defined')
    m.to_csv(os.path.join(OUT, "112_csf_pairs.csv"), index=False)
    pd.DataFrame(rows_alb).to_csv(os.path.join(OUT, "112_csf_albumin.csv"), index=False)
    COL_ATTR = ['family', 'outcome', 'kinetic', 'participant_removed', 'population',
                'obsv_of_participant_in_model', 'first_year', 'last_year',
                'change_annual_observed', 'slope_mean_of_model_a_333',
                'z_kinetic_of_participant',
                'share_of_total_of_squares_of_kinetic', "beta_333", "se_333",
                'beta_without_the_participant', 'se_without_the_participant',
                'p_no_corrected_without_the_participant', 'remark_on_p',
                'accepted_fromthe_criterion_strict',
                'beta_without_the_participant_by_SD_of_333',
                'shift_in_SE_333_a_scale_fixed', 'units_of_beta', 'units_of_z']
    ATTR.reindex(columns=COL_ATTR).to_csv(os.path.join(OUT, "112_attribution.csv"),
                                          index=False)
    pd.DataFrame(rows_s).to_csv(os.path.join(OUT, "112_summary.csv"), index=False)
    with open(os.path.join(OUT, "112_replication_without_type2.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")
    print(f'\nSaved in {OUT}: 112_replication_without_type2.txt and the csv files 112_*  '
          f'({time.time() - T0:.0f}s)', flush=True)


if __name__ == "__main__":
    main()
