"""Common functions of the decision rule used by scripts 117-127.

Contains, copied without changes, the functions of the k = 2 battery (prediction strength, Jaccard index,
agreement between algorithms) against the copula reference, the k = 2 partitions of the four algorithms, the
curves over k = 2-10 reusing script 45, the reading of the cohorts, Monte Carlo p values and the
Benjamini-Hochberg correction. It writes no output and is not meant to be run on its own.
"""

import os

# A single thread per process, set before importing numpy: two runs give
# the same bytes and the parallel processes do not compete for the cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"


import sys                                                       # noqa: E402


import warnings                                                  # noqa: E402
import zlib                                                      # noqa: E402
from _legacy import crc32_legacy                                # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402
from scipy.spatial import cKDTree                                # noqa: E402
from scipy.spatial.distance import cdist                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")


# battery: the same values as 25_monte_carlo_nulls.py and 110
SEED_B = 20260805
B_NULL = {"24h": 500, "150h": 300}
N_PROJ = 200
N_SPLIT_PS = 50


R_OBSV = 20              # independent streams for PS and Jaccard (convention of 110)


# second arm of the rule: curves over k = 2-10 with the functions of 45
B_CURVE_C = {"24h": 500, "150h": 300}     # separate cohorts: the values of 45


MORE_STRUCTURE = {"hopkins": True, "silhouette_k2": True,
                 'n_projections_multimodal': True, "dip_pmin_bonf": False,
                 "PS_k2": True, "jaccard_k2": True, 'ARI_algorithms_k2': True}
CRITERIA = ["PS_k2", "jaccard_k2", 'ARI_algorithms_k2']   # primary rule, k = 2
ALGORITHMS = ["kmeans", "ward", "average", "gmm"]


def _load_45():
    """Script 45 as a module: its curve functions are reused."""
    from _modules import load_script
    return load_script("45_diagnostic_curves.py", "curve_45")


C45 = _load_45()
K_CURVE = list(C45.KRANGE)


def p_mc(v, o, high=True):
    """One-sided Monte Carlo p, (#{null >= observed} + 1)/(B + 1)."""
    v = np.asarray(v, float)
    return float(((np.sum(v >= o) if high else np.sum(v <= o)) + 1) / (len(v) + 1))


def se_mc(p, B):
    """Monte Carlo standard error of the p: binomial on B + 1 draws."""
    return float(np.sqrt(p * (1 - p) / (B + 1)))


def p_sup95(v, o):
    """95% upper limit (Clopper-Pearson, one-sided) of the probability
    that a draw of the null is >= the observed value, given c exceedances out of B."""
    v = np.asarray(v, float)
    c, B = int(np.sum(v >= o)), len(v)
    return 1.0 if c >= B else float(stats.beta.ppf(0.95, c + 1, B - c))


def bh(p):
    """Benjamini-Hochberg step-up, in input order. It is the function of
    75_diagnostic_family.py, copied because 75 cannot be imported as a
    module: p_adj_(i) = min_{j >= i} (n / j * p_(j)), truncated at 1."""
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    q = ps * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n, float)
    out[order] = q
    return out


# -------------------------------------- battery (functions identical to 25) ----
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
    """Same order of calls as 25: the random stream is the same."""
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


# ----------------------------------------------- partitions at k = 2 ----------
def labels_k2(X, seed_gmm=SEED_B):
    """The four partitions at k = 2, with the same constructors as ari_algos().
    The identity with ari_algos() is verified in main() on every branch, with the
    seed of 25. A different GMM seed changes only the partition of the GMM."""
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture
    return {
        "kmeans": KMeans(2, n_init=20, random_state=SEED_B).fit_predict(X),
        "ward": AgglomerativeClustering(2, linkage="ward").fit_predict(X),
        "average": AgglomerativeClustering(2, linkage="average").fit_predict(X),
        "gmm": GaussianMixture(2, covariance_type="full", n_init=5,
                               random_state=seed_gmm).fit_predict(X)}


def ari_avg(labs):
    from sklearn.metrics import adjusted_rand_score
    ks = list(labs)
    return float(np.mean([adjusted_rand_score(labs[ks[i]], labs[ks[j]])
                          for i in range(len(ks)) for j in range(i + 1, len(ks))]))


# ---------------------------------------------------------- parallel tasks ----
def job_lone(t):
    """Observed value and B draws of the single-copula null, a single stream
    as in 25 and in 110."""
    X = t["X"]
    rng = np.random.default_rng(SEED_B)
    obsv = battery(X, rng)
    nulls = [battery(copula_null(X, rng), rng) for _ in range(t["B"])]
    return {'kind': 'lone', 'branch': t['branch'], 'obsv': obsv, 'nulls': nulls}


def job_streams(t):
    """PS and Jaccard observed on independent streams (convention of 110).
    Stream r has its own generator: streams r0 to r1 - 1 do not depend on
    which others are computed, and the first R_OBSV are always the same."""
    X = t["X"]
    r0, r1 = t.get("r0", 0), t.get("r1", R_OBSV)
    streams = []
    for r in range(r0, r1):
        g = np.random.default_rng([SEED_B, crc32_legacy(t['label'].encode()), r])
        streams.append({"PS_k2": ps_k2(X, g), "jaccard_k2": jaccard_k2(X, g)})
    return {'kind': 'streams', 'branch': t['branch'], "r0": r0, 'streams': streams}


def job_curve_obsv(t):
    """Observed curves of silhouette and PS over k = 2-10, functions and seed of 45."""
    X = t["X"]
    sil = C45.silhouette_curve(X, seed=C45.SEED)
    ps = C45.ps_curve(X, seed=C45.SEED, order=t['ordering_ps'])
    assert list(sil.k) == K_CURVE and list(ps.k) == K_CURVE, \
        'unexpected grid of k (consistency check of the code, not of the data)'
    return {'kind': 'curve_obsv', 'branch': t['branch'],
            "silhouette": sil.silhouette.to_numpy(float),
            "PS": ps.PS.to_numpy(float),
            'PS_se': ps.PS_sd.to_numpy(float) / np.sqrt(C45.N_SPLIT_PS)}


def job_curve_null(t):
    """Draws b0..b1-1 of the null for the curves. Single-copula null: it is
    one_null() of 45 without the gap (same generator [SEED_NULL, b], same
    copula, same seed s for the curves; the gap uses a generator of its own and
    removing it does not shift the other two). Stratified null: each group from
    its own copula, generator specific to the branch. The PS is computed only
    for b < n_ps."""
    X, code = t["X"], t.get('code')
    sil, ps = [], []
    for b in range(t["b0"], t["b1"]):
        if code is None:
            rng = np.random.default_rng([C45.SEED_NULL, b])
            Y = C45.copula_null(X, rng)
        else:
            rng = np.random.default_rng(
                [C45.SEED_NULL, crc32_legacy((t['branch'] + "|stratificato").encode()), b])
            Y = np.empty_like(X)
            for c in np.unique(code):             # fixed order: increasing codes
                Y[code == c] = C45.copula_null(X[code == c], rng)
        s = int(rng.integers(1e9))
        sil.append(C45.silhouette_curve(Y, seed=s).silhouette.to_numpy(float))
        if b < t["n_ps"]:
            ps.append(C45.ps_curve(Y, seed=s, order=t['ordering_ps']).PS.to_numpy(float))
    return {'kind': 'curve_null', 'branch': t['branch'], 'nullref': t['nullref'],
            "b0": t["b0"], "silhouette": np.array(sil).reshape(-1, len(K_CURVE)),
            "PS": np.array(ps).reshape(-1, len(K_CURVE))}


EXECUTORS = {'lone': job_lone, 'streams': job_streams,
             'curve_obsv': job_curve_obsv, 'curve_null': job_curve_null}


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
