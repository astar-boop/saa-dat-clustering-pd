"""
57 - Calinski-Harabasz and Davies-Bouldin indices for k = 2..10, both cohorts.

The Methods of the manuscript list CH and DB among the criteria for selecting k
and never report a value of them. The observed values of the 24h exist
(03_internal_indices), but the CH of the 150h is missing and, for both indices and
for both cohorts, so is the calibration against the copula null — which is the
reference used in all the other tables of the manuscript.

Mind the direction: for the CH HIGH values indicate more structure, for the
DB LOW values. The Monte Carlo p is one-sided in the right direction for
each.

Provenance check: the observed values of the 24h must reproduce exactly
03_internal_indices.csv (k-means), which is the source already deposited.

Output: outputs/57_ch_db_curves.csv, outputs/57_ch_db_wide.csv, 57_ch_db.txt
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

from _cohorts import labels
COH24, COH150 = labels(OUT)
SEED = 20260804
SEED_NULL = 20260810
KRANGE = list(range(2, 11))
N_JOBS = int(os.environ.get("N_JOBS", "12"))
B24 = int(os.environ.get("B24", "500"))
B150 = int(os.environ.get("B150", "300"))

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


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


def gaussian_null(X, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), size=len(X))


def curve(Xm, seed=SEED):
    """CH and DB of the k-means partition, k = 2..10 (same parameters as 03)."""
    ch, db = [], []
    for k in KRANGE:
        lab = KMeans(k, n_init=50, random_state=seed).fit_predict(Xm)
        ch.append(calinski_harabasz_score(Xm, lab))
        db.append(davies_bouldin_score(Xm, lab))
    return np.array(ch), np.array(db)


def one_null(X, b):
    rng = np.random.default_rng([SEED_NULL, b])
    Y = copula_null(X, rng)
    return curve(Y, seed=int(rng.integers(1e9)))


X24 = np.load(os.path.join(OUT, "features.npz"))["F_primary"]
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete]
F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE", "dat_putamen_low",
        "dat_caudate_mean", "age_bl"]
M = d150[F150].to_numpy(float).copy()
for j in range(4):
    M[:, j] = np.log(M[:, j])
X150 = (M - M.mean(0)) / M.std(0)

t0 = time.time()
log("=" * 78)
log('57 — CALINSKI-HARABASZ AND DAVIES-BOULDIN, k = 2..10, OBSERVED vs COPULA NULL')
log("=" * 78)

rows = []
for tag, X, B in [(COH24, X24, B24), (COH150, X150, B150)]:
    ch_o, db_o = curve(X)
    ch_g, db_g = curve(gaussian_null(X))
    if tag.startswith("24h"):
        ii = pd.read_csv(os.path.join(OUT, "03_internal_indices.csv"))
        km = ii[(ii.algorithm == "kmeans")]
        for name, calc, expected in [
                ('observed CH', ch_o, km[km.data == 'observed'].calinski_harabasz),
                ('DB observed', db_o, km[km.data == 'observed'].davies_bouldin),
                ('CH Gaussian null', ch_g, km[km.data == 'nullref'].calinski_harabasz),
                ('DB Gaussian null', db_g, km[km.data == 'nullref'].davies_bouldin)]:
            d = float(np.max(np.abs(np.asarray(calc) - np.asarray(expected))))
            log(f'  check {name:22s} vs 03_internal_indices: max difference {d:.2e}')

    log(f'\n{tag}: {B} realisations of the copula null on {N_JOBS} processes...')
    res = Parallel(n_jobs=N_JOBS)(delayed(one_null)(X, b) for b in range(B))
    CH = np.vstack([r[0] for r in res])
    DB = np.vstack([r[1] for r in res])
    log(f'  done [{time.time() - t0:.0f}s]')

    for name, obs, NUL, gau, high in [
            ("Calinski-Harabasz", ch_o, CH, ch_g, True),
            ("Davies-Bouldin", db_o, DB, db_g, False)]:
        for i, k in enumerate(KRANGE):
            v = NUL[:, i]
            p = ((np.sum(v >= obs[i]) + 1) / (len(v) + 1) if high
                 else (np.sum(v <= obs[i]) + 1) / (len(v) + 1))
            rows.append({'cohort': tag, 'index': name, "k": k,
                         'direction': 'high = more structure' if high
                                      else 'low = more structure',
                         'observed': obs[i],
                         'null_copula_median': float(np.median(v)),
                         "nullo_copula_IC2.5": float(np.percentile(v, 2.5)),
                         "nullo_copula_IC97.5": float(np.percentile(v, 97.5)),
                         "p_montecarlo": p,
                         'null_gaussian_single': gau[i], "B": len(v)})

long = pd.DataFrame(rows)
long["p_BH"] = np.nan
for ind in long.index.unique():
    m = long.index == ind
    long.loc[m, "p_BH"] = multipletests(long.loc[m, "p_montecarlo"],
                                        method="fdr_bh")[1]
long.to_csv(os.path.join(OUT, "57_ch_db_curves.csv"), index=False)

SERIES = [("observed", 'observed'),
         ("copula null: median", 'null_copula_median'),
         ("copula null: 2.5th pct", "nullo_copula_IC2.5"),
         ("copula null: 97.5th pct", "nullo_copula_IC97.5"),
         ("Monte Carlo p", "p_montecarlo"),
         ("Benjamini-Hochberg adjusted p", "p_BH"),
         ("Gaussian null (single draw)", 'null_gaussian_single')]
wide = []
for ind in long.index.unique():
    for tag in long.cohort.unique():
        s = long[(long.index == ind) & (long.cohort == tag)].set_index("k")
        for et, col in SERIES:
            r = {"index": ind, "cohort": tag, "series": et}
            r.update({f"k={k}": s.loc[k, col] for k in s.index})
            wide.append(r)
pd.DataFrame(wide).to_csv(os.path.join(OUT, "57_ch_db_wide.csv"), index=False)

log("\n" + "=" * 78)
log('RESULTS')
log("=" * 78)
for ind in long.index.unique():
    log(f"\n--- {ind} ---")
    s = long[long.index == ind]
    log(s[['cohort', "k", 'observed', 'null_copula_median', "nullo_copula_IC2.5",
           "nullo_copula_IC97.5", "p_montecarlo", "p_BH"]].round(4).to_string(index=False))
    sig = s[s.p_montecarlo < 0.05]
    log(f'  nominally significant: {len(sig)}/{len(s)}'
        + (f' -> {[(r.cohort, int(r.k)) for r in sig.itertuples()]}' if len(sig) else ""))
    log(f'  after BH: {int((s.p_BH < 0.05).sum())}/{len(s)}')

with open(os.path.join(OUT, "57_ch_db.txt"), "w") as fh:
    fh.write("\n".join(rep))
log(f'\nTotal time: {time.time() - t0:.0f}s')
log('Saved: outputs/57_ch_db_curves.csv')
