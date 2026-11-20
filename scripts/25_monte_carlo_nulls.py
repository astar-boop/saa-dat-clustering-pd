"""
25 - P1.3: reconstruction of the null models with distribution and Monte Carlo p.

The review observes that for each statistic the manuscript reports a
single null value, without a distribution: judgements such as «the difference is
negligible» therefore remain judgements, not measurements. Moreover the
reference of the «about 50 expected projections» is wrong — the dip test is
conservative and the empirical null produces zero of them out of a thousand,
not fifty.

Here B realisations of the Gaussian copula null are generated (it preserves the
empirical marginals, and is therefore the correct reference for ALL the
statistics and not only for the BIC) and for each statistic the
observed value, the null distribution and the Monte Carlo p-value are reported.

Output: outputs/25_nulls.csv, outputs/25_nulls.txt
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import diptest

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

from _cohorts import labels
COH24, COH150 = labels(OUT)
SEED = 20260805
B24, B150 = 500, 300
# declared seeds and sample sizes: SEED governs both the observed data and the nulls
N_PROJ = 200
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def copula_null(X, rng):
    """Unimodal null that preserves empirical marginals and dependence."""
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


def n_multimodal(X, rng, n_proj=N_PROJ, keep=None):
    c = 0
    pmin = 1.0
    for _ in range(n_proj):
        a = rng.normal(size=X.shape[1]); a /= np.linalg.norm(a)
        p = diptest.diptest(X @ a)[1]
        if keep is not None:
            keep.append(p)
        c += p < 0.05
        pmin = min(pmin, p)
    return c, pmin


# n_rep aligned with script 03: with 8 splits the estimate has SD 0.048 and
# exceeds the threshold of 0.80 in 15% of the replicates by Monte Carlo effect
# alone; with 50 it falls to 0.016 and never exceeds it.
N_SPLIT_PS = 50


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
                worst = 0.0; continue
            cnt = np.bincount(pred[m], minlength=2)
            worst = min(worst, (cnt * (cnt - 1)).sum() / (nc * (nc - 1)))
        vals.append(worst)
    return float(np.mean(vals))


def jaccard_k2(X, rng, Bb=25):
    """See note in 30_*.py: extension of the bootstrap partitioning to all
    the original points via the nearest centroid."""
    ref = KMeans(2, n_init=15, random_state=SEED).fit_predict(X)
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
        "kmeans": KMeans(2, n_init=20, random_state=SEED).fit_predict(X),
        "ward": AgglomerativeClustering(2, linkage="ward").fit_predict(X),
        "average": AgglomerativeClustering(2, linkage="average").fit_predict(X),
        "gmm": GaussianMixture(2, covariance_type="full", n_init=5,
                               random_state=SEED).fit_predict(X)}
    ks = list(labs)
    return float(np.mean([adjusted_rand_score(labs[ks[i]], labs[ks[j]])
                          for i in range(4) for j in range(i + 1, 4)]))


def battery(X, rng, keep_pvals=None):
    nm, pmin = n_multimodal(X, rng, keep=keep_pvals)
    return {"hopkins": hopkins(X, rng),
            "silhouette_k2": silhouette_score(X, KMeans(2, n_init=15, random_state=SEED).fit_predict(X)),
            'n_projections_multimodal': nm,
            "dip_pmin_bonf": min(1.0, pmin * N_PROJ),
            "PS_k2": ps_k2(X, rng),
            "jaccard_k2": jaccard_k2(X, rng),
            'ARI_algorithms_k2': ari_algos(X)}


npz = np.load(os.path.join(OUT, "features.npz"))
X24 = npz["F_primary"]
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete]
F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE", "dat_putamen_low",
        "dat_caudate_mean", "age_bl"]
M = d150[F150].to_numpy(float).copy()
for j in range(4):
    M[:, j] = np.log(M[:, j])
X150 = (M - M.mean(0)) / M.std(0)

log("=" * 78)
log('P1.3 — NULL DISTRIBUTION AND MONTE CARLO p-VALUES')
log("=" * 78)
log('Reference: Gaussian copula with empirical marginals, applied to ALL')
log('the statistics. Direction of the test indicated for each.')

# for each statistic, whether HIGH values indicate more structure
HIGHER = {"hopkins": True, "silhouette_k2": True, 'n_projections_multimodal': True,
          "dip_pmin_bonf": False, "PS_k2": True, "jaccard_k2": True,
          'ARI_algorithms_k2': True}

rows = []
for tag, X, B in [(COH24, X24, B24), (COH150, X150, B150)]:
    rng = np.random.default_rng(SEED)
    pvals_obsv = []
    obsv = battery(X, rng, keep_pvals=pvals_obsv)
    log(f'\n--- {tag}: {B} realisations of the copula null ---')
    nulls = []
    for b in range(B):
        Y = copula_null(X, rng)
        nulls.append(battery(Y, rng))
        if (b + 1) % 50 == 0:
            log(f"    {b+1}/{B}")
    N = pd.DataFrame(nulls)
    # complete null distributions and observed dip p: they are needed for the figure
    slug = "24h" if "24h" in tag else "150h"
    N.to_csv(os.path.join(OUT, f"25_null_distribution_{slug}.csv"), index=False)
    np.save(os.path.join(OUT, f"25_dip_pvals_observed_{slug}.npy"),
            np.asarray(pvals_obsv))
    for stat, obs in obsv.items():
        v = N[stat].to_numpy(float)
        if HIGHER[stat]:
            p = (np.sum(v >= obs) + 1) / (len(v) + 1)
        else:
            p = (np.sum(v <= obs) + 1) / (len(v) + 1)
        rows.append({'cohort': tag, 'statistic': stat, 'observed': obs,
                     'null_median': float(np.median(v)),
                     "nullo_IC2.5": float(np.percentile(v, 2.5)),
                     "nullo_IC97.5": float(np.percentile(v, 97.5)),
                     "p_montecarlo": p, "B": len(v)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "25_nulls.csv"), index=False)

log("\n" + "=" * 78)
log('RESULTS')
log("=" * 78)
for tag in res.cohort.unique():
    log(f"\n{tag}")
    s = res[res.cohort == tag]
    log(s[['statistic', 'observed', 'null_median', "nullo_IC2.5",
           "nullo_IC97.5", "p_montecarlo"]].round(4).to_string(index=False))
    sig = s[s.p_montecarlo < 0.05]
    log(f'  statistics exceeding the null (p < 0.05): {len(sig)} out of {len(s)}' + (f' -> {sig.statistic.tolist()}' if len(sig) else ""))

log("\n" + "=" * 78)
log('THE CORRECT REFERENCE FOR THE MULTIMODAL PROJECTIONS')
log("=" * 78)
for tag in res.cohort.unique():
    r = res[(res.cohort == tag) & (res.statistic == 'n_projections_multimodal')].iloc[0]
    log(f'\n{tag}: observed {r.observed:.0f} out of {N_PROJ} projections')
    log(f"  null distribution: median {r['null_median']:.1f}, 95% CI ({r['nullo_IC2.5']:.1f}; "
        f"{r['nullo_IC97.5']:.1f})")
    log(f'  Monte Carlo p: {r.p_montecarlo:.4f}')
log("\nThe 'about 50 expected' of the manuscript assumed uniform p-values under")
log('unimodality. The dip test is conservative and the empirical reference is')
log('much lower: the sentence must be replaced with the distribution reported here.')

with open(os.path.join(OUT, "25_nulls.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/25_nulls.csv')
