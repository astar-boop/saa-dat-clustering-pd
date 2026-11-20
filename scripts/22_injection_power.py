"""
22 - P1.1: quantifying the sensitivity of the battery (cluster injection).

A claim of absence requires a measure of sensitivity. Here synthetic clusters
are injected into the real data and the fraction of replicates in which
each criterion detects them is measured, yielding a power curve and the minimum
detectable separation.

Procedure. On the 687 real participants (and on the 473 of the replication
cohort) a direction is chosen, a proportion pi of participants is drawn and it
is shifted by Delta Mahalanobis standard deviations along that direction. The
resulting matrix has exactly the marginals and the covariance of the real data,
plus a cluster of known separation. Each cell is repeated n_rep times.

Directions: PC1 (where the variance is greatest), PC2, and a random direction.

Output: outputs/22_power.csv, outputs/22_power.txt
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import diptest

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

from _cohorts import labels
COH24, COH150 = labels(OUT)
SEED = 20260805

DELTAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PIS = [0.50, 0.30, 0.15]
N_REP = 60
N_PROJ = 200
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def inject(X, direction, delta, pi, rng):
    """Shifts a proportion pi of participants by delta Mahalanobis units along
    `direction`. On data standardised and whitened along the direction, the
    Mahalanobis distance between the two centres coincides with delta."""
    n = len(X)
    S = np.cov(X, rowvar=False)
    u = direction / np.sqrt(direction @ S @ direction)   # Mahalanobis norm 1
    idx = rng.choice(n, int(round(pi * n)), replace=False)
    Y = X.copy()
    Y[idx] = Y[idx] + delta * (S @ u)
    lab = np.zeros(n, int)
    lab[idx] = 1
    return Y, lab


def crit_dip(Y, rng):
    """Fraction of multimodal random projections and minimum p after Bonferroni."""
    pv = np.empty(N_PROJ)
    for i in range(N_PROJ):
        a = rng.normal(size=Y.shape[1])
        a /= np.linalg.norm(a)
        pv[i] = diptest.diptest(Y @ a)[1]
    return (pv < .05).mean(), min(1.0, pv.min() * N_PROJ)


def crit_ps(Y, rng, k=2, n_rep=10):
    """Prediction strength at k groups."""
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
    """Bootstrap Jaccard index, averaged over the groups. See the note in 30_*.py:
    the partition of the bootstrap is extended to all the original points."""
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
def run_cell(X, direction, delta, pi, seed):
    rng = np.random.default_rng(seed)
    Y, lab = inject(X, direction, delta, pi, rng)
    frac_dip, pmin = crit_dip(Y, rng)
    sil = silhouette_score(Y, KMeans(2, n_init=15, random_state=SEED).fit_predict(Y))
    ps = crit_ps(Y, rng)
    jac = crit_jaccard(Y, rng)
    return {"dip_frac": frac_dip, "dip_pmin_bonf": pmin, "silhouette": sil,
            "PS": ps, "jaccard": jac}


npz = np.load(os.path.join(OUT, "features.npz"))
X24 = npz["F_primary"]
names = json.load(open(os.path.join(OUT, "feature_names.json")))["primary"]

d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete]
F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE", "dat_putamen_low",
        "dat_caudate_mean", "age_bl"]
M = d150[F150].to_numpy(float).copy()
for j in range(4):
    M[:, j] = np.log(M[:, j])
X150 = (M - M.mean(0)) / M.std(0)

log("=" * 78)
log('P1.1 — POWER OF THE BATTERY AGAINST INJECTED CLUSTERS')
log("=" * 78)
log(f'Replicates per cell: {N_REP} | random projections per replicate: {N_PROJ}')
log('Decision thresholds: dip = at least one projection significant after')
log('Bonferroni; prediction strength >= 0.80; Jaccard >= 0.75.')

rows = []
for tag, X in [(COH24, X24), (COH150, X150)]:
    pca = PCA().fit(X)
    dirs = {"PC1": pca.components_[0], "PC2": pca.components_[1]}
    rng0 = np.random.default_rng(SEED)
    v = rng0.normal(size=X.shape[1]); dirs['random'] = v / np.linalg.norm(v)
    for dname, dvec in dirs.items():
        for delta in DELTAS:
            for pi in PIS:
                res = [run_cell(X, dvec, delta, pi, SEED + 1000 * r + int(delta * 10) + int(pi * 100))
                       for r in range(N_REP)]
                R = pd.DataFrame(res)
                rows.append({
                    'cohort': tag, 'direction': dname, "delta": delta, "pi": pi,
                    'power_dip': float((R.dip_pmin_bonf < 0.05).mean()),
                    'power_PS': float((R.PS >= 0.80).mean()),
                    'power_jaccard': float((R.jaccard >= 0.75).mean()),
                    'silhouette_mean': float(R.silhouette.mean()),
                    'PS_mean': float(R.PS.mean()),
                    'jaccard_mean': float(R.jaccard.mean()),
                    'dip_frac_mean': float(R.dip_frac.mean())})
                log(f"  {tag} | {dname} | Δ={delta} π={pi:.2f} -> dip {rows[-1]['power_dip']:.2f} | "
                    f"PS {rows[-1]['power_PS']:.2f} | Jac {rows[-1]['power_jaccard']:.2f} | sil "
                    f"{rows[-1]['silhouette_mean']:.3f}")

pw = pd.DataFrame(rows)
pw.to_csv(os.path.join(OUT, "22_power.csv"), index=False)

log("\n" + "=" * 78)
log('MINIMUM DETECTABLE SEPARATION (power >= 80%)')
log("=" * 78)
for tag in pw.cohort.unique():
    for dname in pw.direction.unique():
        s = pw[(pw.cohort == tag) & (pw.direction == dname)]
        log(f'\n{tag} — direction {dname}')
        for pi in PIS:
            ss = s[s.pi == pi].sort_values("delta")
            out = []
            for crit in ['power_dip', 'power_PS', 'power_jaccard']:
                ok = ss[ss[crit] >= 0.80]
                out.append(f"{crit.replace('power_', '')}: "
                           f"{('Δ ≥ ' + str(ok.delta.min()) if len(ok) else 'never')}")
            log(f"  π = {pi:.2f}  ->  " + " | ".join(out))

with open(os.path.join(OUT, "22_power.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/22_power.csv')
