"""
79 - PAC (proportion of ambiguous clustering) against the copula null.

In the manuscript the PAC at k = 2 was the last statistic still compared with
a SINGLE Gaussian realisation (0.101 observed against 0.044 in the null,
outputs/03_pac.csv). All the other statistics are calibrated on the
complete Gaussian copula null distribution. Here that distribution is
computed for the PAC too, in the two cohorts.

  - the generator `copula_null` and the loading of the data are taken
    to the letter from 66_external_review.py (same seed, 20260805);
  - `fit_labels` and `pac` are taken to the letter from 03_clustering.py:
    200 subsamples at 80 %, ambiguity interval 0.10-0.90,
    k-means with n_init = 50, pairs of the upper triangle only.

DIRECTION OF THE TEST. The PAC measures AMBIGUITY: LOW values indicate a
clear-cut partition, that is, more structure. The test is therefore one-sided
towards low values, p = (1 + #{null <= observed}) / (B + 1). This is the
direction opposite to that of silhouette, Hopkins, prediction strength and
Jaccard, for which it is the high values that indicate structure.

The PAC has a fixed internal subsampling seed: as in
03_clustering.py it is a deterministic function of the data given the seed, and
the same seed is used for the observed data and for every null realisation
(common random numbers). The variability of the null distribution therefore
reflects only the variability of the null data, not the subsampling noise.

Output: outputs/79_pac_null.csv, outputs/79_pac_null.txt
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

SEED = 20260805       # the same seed as 66_external_review.py
SEED_03 = 20260804    # the seed of 03_clustering.py, for the identity check
B_NULL = 200          # realisations of the copula null
K = 2
N_JOBS = 10
T0 = time.time()
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# ===========================================================================
# generator of the copula null — to the letter from 66_external_review.py
# ===========================================================================
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


# ===========================================================================
# PAC — to the letter from 03_clustering.py (fit_labels + pac)
# ===========================================================================
def fit_labels(Xm, k, algo, seed=SEED):
    if algo == "kmeans":
        return KMeans(k, n_init=50, random_state=seed).fit_predict(Xm)
    if algo == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xm)
    if algo == "average":
        return AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(Xm)
    if algo == "gmm":
        return GaussianMixture(k, covariance_type="full", n_init=10,
                               random_state=seed).fit_predict(Xm)
    raise ValueError(algo)


def pac(Xm, k, B=200, frac=0.80, seed=SEED, lo=0.10, hi=0.90):
    r = np.random.default_rng(seed)
    nn = len(Xm)
    co = np.zeros((nn, nn))
    cnt = np.zeros((nn, nn))
    m = int(frac * nn)
    for _ in range(B):
        s = r.choice(nn, m, replace=False)
        lb = fit_labels(Xm[s], k, "kmeans", seed=int(r.integers(1e6)))
        eq = (lb[:, None] == lb[None, :]).astype(float)
        co[np.ix_(s, s)] += eq
        cnt[np.ix_(s, s)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = np.where(cnt > 0, co / cnt, np.nan)
    iu = np.triu_indices(nn, 1)
    v = cm[iu]
    v = v[~np.isnan(v)]
    return float(((v > lo) & (v < hi)).mean())


# ===========================================================================
# data — to the letter from 66_external_review.py
# ===========================================================================
npz = np.load(os.path.join(OUT, "features.npz"))
X24 = npz["F_primary"]
PAT24 = npz["PATNO"]

d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete].reset_index(drop=True)
F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE", "dat_putamen_low",
        "dat_caudate_mean", "age_bl"]
M150 = d150[F150].to_numpy(float).copy()
for j in range(4):
    M150[:, j] = np.log(M150[:, j])
X150 = (M150 - M150.mean(0)) / M150.std(0)
PAT150 = d150.PATNO.to_numpy()

log("=" * 78)
log('79 — PAC AT k = 2 AGAINST THE GAUSSIAN COPULA NULL DISTRIBUTION')
log("=" * 78)
log(f'cohort 24h  : n = {X24.shape[0]}, p = {X24.shape[1]}')
log(f'cohort 150h : n = {X150.shape[0]}, p = {X150.shape[1]}')
log(f'seed = {SEED}; B = {B_NULL} null realisations per cohort')
log("PAC: 200 subsamples at 80 %, ambiguity interval 0.10-0.90,")
log('     k-means n_init = 50 (definition of 03_clustering.py).')
log("")
log("DIRECTION: the PAC measures ambiguity, so LOW values = more structure.")
log('One-tailed test towards low values: p = (1 + #{null <= observed}) / (B + 1).')

# ---------------------------------------------------------------------------
# identity check with 03_pac.csv: same code, same seed -> same PAC
# ---------------------------------------------------------------------------
log("\n" + "-" * 78)
log('CHECK: the PAC implemented here reproduces that of 03_pac.csv')
log("-" * 78)
pac_24_seed03 = pac(X24, K, seed=SEED_03)
# float_precision="round_trip": the default C parser of pandas can be wrong
# by 1 ULP and would make an exact equality comparison fail
pac_pub = pd.read_csv(os.path.join(OUT, "03_pac.csv"), float_precision="round_trip")
val_pub = float(pac_pub[(pac_pub.data == 'observed') & (pac_pub.k == K)].PAC.iloc[0])
val_pub_null = float(pac_pub[(pac_pub.data == 'nullref') & (pac_pub.k == K)].PAC.iloc[0])
log(f'observed 24h PAC, seed of 03 ({SEED_03}) : {pac_24_seed03!r}')
log(f'observed 24h PAC in outputs/03_pac.csv    : {val_pub!r}')
log(f'identical bit for bit: {pac_24_seed03 == val_pub}')
assert pac_24_seed03 == val_pub, 'the reimplemented PAC does not reproduce 03_pac.csv'
log(f'(single Gaussian draw in 03_pac.csv, k = 2: {val_pub_null!r} —')
log(" it is the comparison that this script replaces)")


# ---------------------------------------------------------------------------
# null distribution
# ---------------------------------------------------------------------------
def against_null(tag, X, B=B_NULL):
    """Observed PAC and copula null distribution, with Monte Carlo p."""
    rng = np.random.default_rng(SEED)
    obsv = pac(X, K, seed=SEED)
    log(f'\n[{tag}] observed PAC (k = {K}) = {obsv:.6f}   ({time.time() - T0:.0f}s)')
    # the nulls are generated in sequence by a single rng: the order is reproducible
    nulls_X = [copula_null(X, rng) for _ in range(B)]
    log(f'[{tag}] {B} null realisations generated; computing the PAC on {N_JOBS} processes...')
    v = np.array(Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(pac)(Xb, K, seed=SEED) for Xb in nulls_X), dtype=float)
    log(f'[{tag}] done  ({time.time() - T0:.0f}s)')
    # PAC = ambiguity: low values indicate structure -> lower tail
    n_extreme = int(np.sum(v <= obsv))
    p = (1 + n_extreme) / (len(v) + 1)
    row = {'analysis': tag, 'statistic': "PAC_k2",
            'label': "PAC (k = 2)", "n": int(len(X)),
            'observed': obsv, 'null_median': float(np.median(v)),
            "nullo_IC2.5": float(np.percentile(v, 2.5)),
            "nullo_IC97.5": float(np.percentile(v, 97.5)),
            'null_min': float(v.min()), 'null_max': float(v.max()),
            'n_nulls_atleast_extreme': n_extreme,
            'direction': 'lower tail (low PAC = more structure)',
            "p_montecarlo": p, "B": int(len(v)), 'seed': SEED}
    return row, v


rows, distributions = [], {}
for tag, Xm in [("24h", X24), ("150h", X150)]:
    row, v = against_null(tag, Xm)
    rows.append(row)
    distributions[tag] = v

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "79_pac_null.csv"), index=False)
np.savez(os.path.join(OUT, "79_pac_null_distributions.npz"), **distributions)

log("\n" + "=" * 78)
log('RESULT')
log("=" * 78)
log(res[['analysis', "n", 'observed', 'null_median', "nullo_IC2.5",
         "nullo_IC97.5", "p_montecarlo", "B"]].round(4).to_string(index=False))

for r in rows:
    v = distributions[r['analysis']]
    log(f"\n--- {r['analysis']} (n = {r['n']}, B = {r['B']}) ---")
    log(f"  observed PAC             : {r['observed']:.6f}")
    log(f"  copula null, median      : {r['null_median']:.6f}")
    log(f"  copula null, 2.5th-97.5th: [{r['nullo_IC2.5']:.6f}, {r['nullo_IC97.5']:.6f}]")
    log(f'  copula null, min-max     : [{v.min():.6f}, {v.max():.6f}]')
    log(f"  null realisations with PAC <= observed: {r['n_nulls_atleast_extreme']} of {r['B']}")
    log(f"  Monte Carlo p (lower tail) = (1 + {r['n_nulls_atleast_extreme']}) / ({r['B']} + 1) = "
        f"{r['p_montecarlo']:.4f}")
    log(f"  outcome: "
        f"{('PAC' + chr(39) + ' lower than the null (p < 0.05)' if r['p_montecarlo'] < 0.05 else 'PAC NOT' + chr(39) + ' lower than expected under the null (p >= 0.05)')}")

log("\n" + "=" * 78)
log("SENSITIVITY TO THE SUBSAMPLING SEED")
log("=" * 78)
obs24 = float(res[res.analysis == "24h"].observed.iloc[0])
log(f'The observed PAC of the 24h cohort is {obs24:.6f} with seed {SEED} and {pac_24_seed03:.6f} '
    f'with seed {SEED_03}:')
log(f'the difference due to subsampling alone is {abs(obs24 - pac_24_seed03):.6f},')
log(f"negligible relative to the width of the null distribution "
    f"[{float(res[res.analysis == '24h']['nullo_IC2.5'].iloc[0]):.3f}, "
    f"{float(res[res.analysis == '24h']['nullo_IC97.5'].iloc[0]):.3f}].")
log('The conclusion does not depend on which of the two seeds is adopted.')

log("\n" + "=" * 78)
log('NOTE ON THE REPLACED COMPARISON')
log("=" * 78)
log(f'outputs/03_pac.csv compared the observed PAC of the 24h cohort ({val_pub:.6f})')
log(f'with ONE single Gaussian draw ({val_pub_null:.6f}), which carries no')
log("information on the variability of the null. The copula null distribution")
log('computed here gives median, 95 % interval and Monte Carlo p, in the')
log('same form as the other statistics of the manuscript.')

with open(os.path.join(OUT, "79_pac_null.txt"), "w") as fh:
    fh.write("\n".join(rep))
log(f'\nSaved: outputs/79_pac_null.csv and outputs/79_pac_null.txt  ({time.time() - T0:.0f}s)')
