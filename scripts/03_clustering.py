"""
03 - Clustering: choice of k, comparison between algorithms, stability.

Full battery applied BOTH to the observed data AND to a unimodal null data set
with the same mean and covariance. The comparison with the null is
essential: silhouette, BIC and internal indices always produce an "optimal k"
even when the clusters do not exist.

  1. internal indices (silhouette, Calinski-Harabasz, Davies-Bouldin) k = 2..10
  2. gap statistic of Tibshirani (includes k = 1)
  3. prediction strength of Tibshirani-Walther (out-of-sample reproducibility)
  4. GMM: BIC / ICL, all the covariance structures
  5. bootstrap stability (Jaccard index per cluster, Hennig)
  6. agreement between algorithms (ARI)
  7. consensus clustering: PAC (proportion of ambiguous clustering)

Output: outputs/03_*.csv, outputs/03_clustering_report.txt
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SEED = 20260804
KRANGE = list(range(2, 11))

rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


npz = np.load(os.path.join(OUT, "features.npz"))
names = json.load(open(os.path.join(OUT, "feature_names.json")))
X = npz["F_primary"]
PATNO = npz["PATNO"]
n, p = X.shape
log(f"Primary matrix: n = {n}, p = {p} -> {names['primary']}")

rng = np.random.default_rng(SEED)
# unimodal null: Gaussian with the same mean and covariance as the data
X_null = rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), size=n)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
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


ALGOS = ["kmeans", "ward", "average", "gmm"]


# ---------------------------------------------------------------------------
# 1. Internal indices
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('1. INTERNAL INDICES (observed data vs unimodal null)')
log("=" * 72)

rows = []
for tag, Xm in [('observed', X), ('nullref', X_null)]:
    for algo in ALGOS:
        for k in KRANGE:
            lab = fit_labels(Xm, k, algo)
            if len(np.unique(lab)) < 2:
                continue
            rows.append({
                'data': tag, 'algorithm': algo, "k": k,
                "silhouette": silhouette_score(Xm, lab),
                "calinski_harabasz": calinski_harabasz_score(Xm, lab),
                "davies_bouldin": davies_bouldin_score(Xm, lab),
                "min_cluster_n": int(np.bincount(lab).min()),
            })
idx = pd.DataFrame(rows)
idx.to_csv(os.path.join(OUT, "03_internal_indices.csv"), index=False)

for algo in ALGOS:
    log(f"\n--- {algo} ---")
    piv = idx[idx.algorithm == algo].pivot(index="k", columns='data',
                                           values="silhouette")
    piv.columns = [f"silhouette_{c}" for c in piv.columns]
    pv2 = idx[idx.algorithm == algo].pivot(index="k", columns='data',
                                           values="davies_bouldin")
    pv2.columns = [f"DB_{c}" for c in pv2.columns]
    log(pd.concat([piv, pv2], axis=1).round(3).to_string())

best = idx[idx.data == 'observed'].loc[
    idx[idx.data == 'observed'].groupby('algorithm').silhouette.idxmax()]
log('\nBest silhouette per algorithm (observed data):')
log(best[['algorithm', "k", "silhouette", "min_cluster_n"]].round(3).to_string(index=False))
bn = idx[idx.data == 'nullref'].loc[
    idx[idx.data == 'nullref'].groupby('algorithm').silhouette.idxmax()]
log('\nBest silhouette per algorithm (unimodal NULL):')
log(bn[['algorithm', "k", "silhouette"]].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. Gap statistic
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('2. GAP STATISTIC (Tibshirani et al. 2001, PCA-aligned reference)')
log("=" * 72)


def wk(Xm, lab):
    """Pooled within-cluster dispersion: sum of squares around the centroid."""
    tot = 0.0
    for c in np.unique(lab):
        pts = Xm[lab == c]
        if len(pts) > 1:
            tot += ((pts - pts.mean(0)) ** 2).sum()
    return tot


def gap_statistic(Xm, krange, B=200, seed=SEED):
    r = np.random.default_rng(seed)
    ks = [1] + list(krange)
    # uniform reference in the box aligned with the principal components
    pca = PCA().fit(Xm)
    Xp = pca.transform(Xm)
    lo, hi = Xp.min(0), Xp.max(0)

    logW, logWb_mean, sk = [], [], []
    for k in ks:
        lab = np.zeros(len(Xm), int) if k == 1 else fit_labels(Xm, k, "kmeans")
        logW.append(np.log(wk(Xm, lab)))
        bs = []
        for _ in range(B):
            Zp = r.uniform(lo, hi, size=Xp.shape)
            Zb = pca.inverse_transform(Zp)
            lb = np.zeros(len(Zb), int) if k == 1 else KMeans(
                k, n_init=10, random_state=int(r.integers(1e6))).fit_predict(Zb)
            bs.append(np.log(wk(Zb, lb)))
        bs = np.array(bs)
        logWb_mean.append(bs.mean())
        sk.append(bs.std() * np.sqrt(1 + 1 / B))
    logW, logWb_mean, sk = map(np.array, (logW, logWb_mean, sk))
    gap = logWb_mean - logW
    # 1-SE criterion: the smallest k with Gap(k) >= Gap(k+1) - s(k+1)
    khat = None
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            khat = ks[i]
            break
    return pd.DataFrame({"k": ks, "logW": logW, "logW_null": logWb_mean,
                         "gap": gap, "s_k": sk}), khat


gap_obs, k_obs = gap_statistic(X, KRANGE)
gap_null, k_null = gap_statistic(X_null, KRANGE)
log('\nObserved data:')
log(gap_obs.round(4).to_string(index=False))
log(f'  -> k selected by the 1-SE criterion: {k_obs}')
log('\nUnimodal null (calibration):')
log(gap_null.round(4).to_string(index=False))
log(f'  -> k selected: {k_null}')
gap_obs.to_csv(os.path.join(OUT, "03_gap_observed.csv"), index=False)
gap_null.to_csv(os.path.join(OUT, "03_gap_null.csv"), index=False)


# ---------------------------------------------------------------------------
# 3. Prediction strength
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('3. PREDICTION STRENGTH (Tibshirani & Walther 2005; threshold 0.80)')
log("=" * 72)


def prediction_strength(Xm, krange, n_rep=50, seed=SEED):
    r = np.random.default_rng(seed)
    res = {k: [] for k in krange}
    for _ in range(n_rep):
        perm = r.permutation(len(Xm))
        a, b = perm[: len(Xm) // 2], perm[len(Xm) // 2:]
        Xa, Xb = Xm[a], Xm[b]
        for k in krange:
            ka = KMeans(k, n_init=20, random_state=int(r.integers(1e6))).fit(Xa)
            kb = KMeans(k, n_init=20, random_state=int(r.integers(1e6))).fit(Xb)
            lb = kb.labels_
            # labels of B assigned according to the centroids learnt on A
            pred = np.argmin(cdist(Xb, ka.cluster_centers_), axis=1)
            worst = 1.0
            for c in range(k):
                m = lb == c
                nc = m.sum()
                if nc <= 1:
                    worst = 0.0
                    continue
                pc = pred[m]
                # proportion of pairs also co-assigned by the classifier of A
                cnt = np.bincount(pc, minlength=k)
                same = (cnt * (cnt - 1)).sum()
                worst = min(worst, same / (nc * (nc - 1)))
            res[k].append(worst)
    return pd.DataFrame({"k": krange,
                         'PS_mean': [np.mean(res[k]) for k in krange],
                         "PS_sd": [np.std(res[k]) for k in krange]})


ps_obs = prediction_strength(X, KRANGE)
ps_null = prediction_strength(X_null, KRANGE)
ps = ps_obs.merge(ps_null, on="k", suffixes=('_obsv', '_null'))
log(ps.round(3).to_string(index=False))
ok = ps_obs[ps_obs.PS_mean >= 0.80].k.tolist()
log(f"\n  k with prediction strength >= 0.80 (observed data): "
    f"{(ok if ok else 'NONE -> only k = 1 supported')}")
ps.to_csv(os.path.join(OUT, "03_prediction_strength.csv"), index=False)


# ---------------------------------------------------------------------------
# 4. GMM: BIC / ICL
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('4. GAUSSIAN MIXTURE: BIC and ICL')
log("=" * 72)

gm_rows = []
for tag, Xm in [('observed', X), ('nullref', X_null)]:
    for cov in ["full", "tied", "diag", "spherical"]:
        for k in [1] + KRANGE:
            g = GaussianMixture(k, covariance_type=cov, n_init=10,
                                random_state=SEED, reg_covar=1e-6).fit(Xm)
            resp = g.predict_proba(Xm)
            ent = -np.sum(resp * np.log(np.clip(resp, 1e-300, None)))
            gm_rows.append({'data': tag, "cov": cov, "k": k, "BIC": g.bic(Xm),
                            "ICL": g.bic(Xm) + 2 * ent, 'entropy': ent})
gm = pd.DataFrame(gm_rows)
gm.to_csv(os.path.join(OUT, "03_gmm_bic.csv"), index=False)
for tag in ['observed', 'nullref']:
    sub = gm[gm.data == tag]
    b = sub.loc[sub.groupby("cov").BIC.idxmin()][["cov", "k", "BIC"]]
    i = sub.loc[sub.groupby("cov").ICL.idxmin()][["cov", "k", "ICL"]]
    log(f'\n{tag} - k at minimum BIC per covariance structure:')
    log(b.round(1).to_string(index=False))
    log(f'{tag} - k at minimum ICL:')
    log(i.round(1).to_string(index=False))
log("\nNote: the BIC of a Gaussian mixture measures the fit to the density,")
log("not the existence of separate groups; on unimodal non-Gaussian data")
log("it routinely selects k > 1 to approximate the shape of the density.")


# ---------------------------------------------------------------------------
# 5. Bootstrap stability (Jaccard per cluster)
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log("5. BOOTSTRAP STABILITY - Jaccard index (Hennig 2007)")
log("=" * 72)
log('Interpretation of Hennig: < 0.60 unstable/invalid cluster;')
log('0.60-0.75 weak pattern; > 0.85 stable cluster.')


def boot_jaccard(Xm, k, algo="kmeans", B=200, seed=SEED):
    """Bootstrap Jaccard index according to Hennig.

    The partition learnt on each bootstrap sample is extended to
    all the original subjects through the nearest centroid, as in
    fpc::clusterboot. Without this extension the index would be bounded
    above by the share of distinct subjects present in the bootstrap
    (~0.632) and could not reach Hennig's interpretive thresholds.
    """
    r = np.random.default_rng(seed)
    ref = KMeans(k, n_init=25, random_state=seed).fit_predict(Xm)
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
    return np.array([np.mean(best[c]) for c in range(k)])
jac_rows = []
for tag, Xm in [('observed', X), ('nullref', X_null)]:
    for k in KRANGE:
        j = boot_jaccard(Xm, k)
        jac_rows.append({'data': tag, "k": k, "jaccard_min": j.min(),
                         'jaccard_avg': j.mean(), "jaccard_max": j.max(),
                         "n_cluster_stabili_0.75": int((j >= 0.75).sum()),
                         "n_cluster_stabili_0.85": int((j >= 0.85).sum())})
jac = pd.DataFrame(jac_rows)
jac.to_csv(os.path.join(OUT, "03_bootstrap_jaccard.csv"), index=False)
for tag in ['observed', 'nullref']:
    log(f"\n{tag}:")
    log(jac[jac.data == tag].drop(columns='data').round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Agreement between algorithms
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('6. AGREEMENT BETWEEN ALGORITHMS (ARI)')
log("=" * 72)
ari_rows = []
for tag, Xm in [('observed', X), ('nullref', X_null)]:
    for k in KRANGE:
        labs = {a: fit_labels(Xm, k, a) for a in ALGOS}
        for i in range(len(ALGOS)):
            for j2 in range(i + 1, len(ALGOS)):
                ari_rows.append({'data': tag, "k": k,
                                 'comparison': f"{ALGOS[i]} vs {ALGOS[j2]}",
                                 "ARI": adjusted_rand_score(labs[ALGOS[i]], labs[ALGOS[j2]])})
ari = pd.DataFrame(ari_rows)
ari.to_csv(os.path.join(OUT, "03_algo_agreement.csv"), index=False)
log('\nMean ARI between algorithms per k:')
log(ari.pivot_table(index="k", columns='data', values="ARI").round(3).to_string())
log('\nDetail (observed data):')
log(ari[ari.data == 'observed'].pivot(index="k", columns='comparison',
                                       values="ARI").round(3).to_string())


# ---------------------------------------------------------------------------
# 7. Consensus clustering / PAC
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('7. CONSENSUS CLUSTERING - PAC (proportion of ambiguous clustering)')
log("=" * 72)
log('Low PAC = clear-cut partition; high PAC = ambiguous assignments.')


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


pac_rows = []
for tag, Xm in [('observed', X), ('nullref', X_null)]:
    for k in KRANGE:
        pac_rows.append({'data': tag, "k": k, "PAC": pac(Xm, k)})
pacdf = pd.DataFrame(pac_rows)
pacdf.to_csv(os.path.join(OUT, "03_pac.csv"), index=False)
log(pacdf.pivot(index="k", columns='data', values="PAC").round(3).to_string())

with open(os.path.join(OUT, "03_clustering_report.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/03_clustering_report.txt')
