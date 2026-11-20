"""
34 - The Delta = 0 point of the power curves, for all the criteria.

The cluster injection of script 22 shifts a share pi of subjects by Delta
standard deviations: at Delta = 0 the matrix coincides with the observed data.
The value of the power curves at that point is therefore not power but the
**false-positive rate**, and without it the curves are not interpretable.

Script 32 measured this quantity for the Jaccard index only. Here it is
extended to the other two criteria with power reported in the figure —
prediction strength and dip test on random projections — using the same
functions, thresholds and numbers of replicates as script 22.

Output: outputs/34_delta_zero.csv, outputs/34_delta_zero.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import diptest

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

from _cohorts import labels
COH24, COH150 = labels(OUT)
SEED = 20260805
N_REP = 60
N_PROJ = 200
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def crit_dip(Y, rng):
    pv = np.empty(N_PROJ)
    for i in range(N_PROJ):
        a = rng.normal(size=Y.shape[1])
        a /= np.linalg.norm(a)
        pv[i] = diptest.diptest(Y @ a)[1]
    return min(1.0, pv.min() * N_PROJ)


def crit_ps(Y, rng, k=2, n_rep=10):
    vals = []
    for _ in range(n_rep):
        perm = rng.permutation(len(Y))
        a, b = perm[: len(Y) // 2], perm[len(Y) // 2:]
        ka = KMeans(k, n_init=10, random_state=int(rng.integers(1e6))).fit(Y[a])
        lb = KMeans(k, n_init=10, random_state=int(rng.integers(1e6))).fit_predict(Y[b])
        pred = np.argmin(cdist(Y[b], ka.cluster_centers_), axis=1)
        worst = 1.0
        for c in range(k):
            m = lb == c
            nc = m.sum()
            if nc <= 1:
                worst = 0.0
                continue
            cnt = np.bincount(pred[m], minlength=k)
            worst = min(worst, (cnt * (cnt - 1)).sum() / (nc * (nc - 1)))
        vals.append(worst)
    return float(np.mean(vals))


def crit_jaccard(Y, rng, k=2, B=40):
    ref = KMeans(k, n_init=15, random_state=SEED).fit_predict(Y)
    best = {c: [] for c in range(k)}
    for _ in range(B):
        ib = rng.choice(len(Y), len(Y), replace=True)
        km = KMeans(k, n_init=5, random_state=int(rng.integers(1e6))).fit(Y[ib])
        lab = np.argmin(cdist(Y, km.cluster_centers_), axis=1)
        for c in range(k):
            rs = set(np.where(ref == c)[0])
            best[c].append(max(
                (len(rs & set(np.where(lab == cb)[0])) /
                 len(rs | set(np.where(lab == cb)[0])) for cb in range(k)),
                default=0.0))
    return float(np.mean([np.mean(best[c]) for c in range(k)]))


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
log('FALSE-POSITIVE RATE AT Delta = 0 (no cluster injected)')
log("=" * 78)
log(f'Replicates: {N_REP} per cohort. Thresholds identical to those of script 22:')
log('dip = at least one significant projection after Bonferroni; PS >= 0.80;')
log('Jaccard >= 0.75.')
log("")

rows = []
for tag, X in [(COH24, X24), (COH150, X150)]:
    dips, pss, jacs = [], [], []
    for r in range(N_REP):
        rng = np.random.default_rng(SEED + 7919 * r)
        dips.append(crit_dip(X, rng))
        pss.append(crit_ps(X, rng))
        jacs.append(crit_jaccard(X, rng))
    dips, pss, jacs = map(np.array, (dips, pss, jacs))
    for crit, val, fp in [("dip", dips, (dips < 0.05).mean()),
                          ("PS", pss, (pss >= 0.80).mean()),
                          ("jaccard", jacs, (jacs >= 0.75).mean())]:
        rows.append({'cohort': tag, 'criterion': crit,
                     'value_avg': float(val.mean()),
                     'false_positives': float(fp), "n_rep": N_REP})
        log(f'  {tag:14s} {crit:8s} mean value {val.mean():.3f}  -> exceeds the threshold in '
            f'{100 * fp:.0f}% of the replicates')
    log("")

dz = pd.DataFrame(rows)
dz.to_csv(os.path.join(OUT, "34_delta_zero.csv"), index=False)

log("=" * 78)
log('READING')
log("=" * 78)
def fp(cohort, crit):
    return 100 * float(dz[(dz.cohort == cohort) & (dz.criterion == crit)]
                       .false_positives.iloc[0])


log(f"The dip keeps a negligible false-positive rate ({fp(COH24, 'dip'):.0f}% and "
    f"{fp(COH150, 'dip'):.0f}%): its low power is conservativeness, not spurious insensitivity.")
log(f"The prediction strength must be read with some caution in the primary cohort, where it exceeds "
    f"the threshold in {fp(COH24, 'PS'):.0f}% of the replicates already in the absence of injection "
    f"(in the replication cohort {fp(COH150, 'PS'):.0f}%): its power curve does not start exactly "
    f"from zero, but remains far from saturation.")
log(f"The Jaccard index exceeds the threshold in {fp(COH24, 'jaccard'):.0f}% and in "
    f"{fp(COH150, 'jaccard'):.0f}% of the replicates: its curve starts from saturation and cannot be "
    f"interpreted as power.")

with open(os.path.join(OUT, "34_delta_zero.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/34_delta_zero.csv')
