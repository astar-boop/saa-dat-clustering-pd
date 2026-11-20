"""
32 - Specificity of the bootstrap Jaccard index (the control at Delta = 0).

The power analysis of script 22 measures the fraction of replicates in which
each criterion exceeds its own threshold in the presence of an injected cluster
of separation Delta. A power curve, however, cannot be interpreted without the
value at Delta = 0: a criterion that exceeds the threshold even in the absence
of structure has high power because it does not discriminate, not because it is
sensitive.

Here the Jaccard index at k = 2 is computed, with the same implementation used
throughout the battery, on four types of matrix:

  1. the observed data, without any injection;
  2. the Gaussian copula null, which preserves the empirical marginals and the
     dependence structure but contains no groups;
  3. an isotropic Gaussian, with neither groups nor correlation;
  4. the uniform hypercube, the configuration with minimal structure.

The comparison between 2 and 3-4 separates two distinct causes of bootstrap
stability: the presence of groups and the mere geometry of the cloud. If the cut
at k = 2 is determined by the axis of maximum variance, it is reproducible
across resamples even when the data are perfectly unimodal.

Output: outputs/32_jaccard_specificity.csv, outputs/32_jaccard_specificity.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

from _cohorts import labels
COH24, COH150 = labels(OUT)
SEED = 20260805
N_REP = 60
THRESHOLD = 0.75
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def boot_jaccard_k2(Y, rng, B=60):
    """Identical to the corrected implementation used in scripts 03/06/08/09/22/25:
    the partition learnt on the bootstrap is extended to all the original
    points through the nearest centroid (as in fpc::clusterboot)."""
    ref = KMeans(2, n_init=15, random_state=SEED).fit_predict(Y)
    best = {c: [] for c in range(2)}
    for _ in range(B):
        ib = rng.choice(len(Y), len(Y), replace=True)
        km = KMeans(2, n_init=5, random_state=int(rng.integers(1e6))).fit(Y[ib])
        lab = np.argmin(cdist(Y, km.cluster_centers_), axis=1)
        for c in range(2):
            rs = set(np.where(ref == c)[0])
            best[c].append(max(
                (len(rs & set(np.where(lab == cb)[0])) /
                 len(rs | set(np.where(lab == cb)[0])) for cb in range(2)),
                default=0.0))
    return float(np.mean([np.mean(best[c]) for c in range(2)]))


def copula_null(X, rng):
    p = X.shape[1]
    R = 2 * np.sin(np.pi * np.atleast_2d(stats.spearmanr(X).statistic) / 6)
    w, V = np.linalg.eigh(R)
    R = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    R = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))
    U = stats.norm.cdf(rng.multivariate_normal(np.zeros(p), R, size=len(X)))
    return np.column_stack([np.quantile(X[:, j], U[:, j]) for j in range(p)])


def isotropic(X, rng):
    return rng.normal(size=X.shape)


def uniform(X, rng):
    return rng.uniform(0.0, 1.0, size=X.shape)


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
log("SPECIFICITY OF THE BOOTSTRAP JACCARD INDEX (k = 2, Delta = 0)")
log("=" * 78)
log(f'Replicates per cell: {N_REP} | interpretative threshold of Hennig: {THRESHOLD:.2f}')
log("")

GEN = [('observed data (no injection)', None),
       ('Gaussian copula null', copula_null),
       ('isotropic Gaussian', isotropic),
       ('uniform hypercube', uniform)]

rows = []
for tag, X in [(COH24, X24), (COH150, X150)]:
    log(f"--- {tag} ---")
    for name, fn in GEN:
        vals = []
        for r in range(N_REP):
            rng = np.random.default_rng(SEED + 7919 * r)
            Y = X if fn is None else fn(X, rng)
            vals.append(boot_jaccard_k2(Y, rng))
        v = np.array(vals)
        fp = float((v >= THRESHOLD).mean())
        rows.append({'cohort': tag, 'matrix': name, 'jaccard_avg': float(v.mean()),
                     "jaccard_sd": float(v.std()),
                     'CI2.5': float(np.percentile(v, 2.5)),
                     "IC97.5": float(np.percentile(v, 97.5)),
                     'fraction_above_threshold': fp, "n_rep": N_REP})
        log(f'  {name:38s} Jaccard {v.mean():.3f} (95% CI '
            f'{np.percentile(v, 2.5):.3f}–{np.percentile(v, 97.5):.3f})  | exceeds {THRESHOLD:.2f} '
            f'in {100 * fp:.0f}% of the replicates')
    log("")

sp = pd.DataFrame(rows)
sp.to_csv(os.path.join(OUT, "32_jaccard_specificity.csv"), index=False)

log("=" * 78)
log('READING')
log("=" * 78)
log('The copula null — which by construction contains no group — exceeds')
log("the Hennig threshold in almost all the replicates. The threshold of 0.75")
log("is therefore not a criterion for the presence of structure on matrices of this")
log("shape: the cut at k = 2 along the axis of maximum variance is reproducible")
log("across resamples by virtue of the geometry of the cloud, regardless")
log("of the presence of groups. The collapse of the index on the uniform hypercube, where")
log("no direction is privileged and the orientation of the cut is arbitrary,")
log('confirms this reading.')
log("")
log("It follows that the relevant comparison is not with the fixed threshold but with the")
log('null distribution: see the Monte Carlo p-value of script 25.')

with open(os.path.join(OUT, "32_jaccard_specificity.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/32_jaccard_specificity.csv')
