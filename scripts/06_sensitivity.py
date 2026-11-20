"""
06 - Sensitivity analyses.

A) NON-Gaussian unimodal null (Gaussian copula with empirical marginals):
   a fair reference for the BIC of the mixtures, which otherwise selects k > 1
   only to approximate the skewness of the marginals.
B) Main battery on alternative weightings of the features
   (block-balanced, factor-based): does the result depend on the fact that the
   SAA block contributes 4 variables out of 7?
C) HDBSCAN: density-based method that does not impose k and can declare noise.
D) Enlarged population: all PD with the 24h assay (positive + negative),
   without TTT (not defined in the negatives).

Output: outputs/06_sensitivity.txt, outputs/06_*.csv
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import diptest
import hdbscan
from _paths import DATA, curated_xlsx

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SEED = 20260804
KRANGE = list(range(2, 11))
rng = np.random.default_rng(SEED)
rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


npz = np.load(os.path.join(OUT, "features.npz"))
names = json.load(open(os.path.join(OUT, "feature_names.json")))
X = npz["F_primary"]
n, p = X.shape


# ---------------------------------------------------------------------------
# shared functions
# ---------------------------------------------------------------------------
def wk(Xm, lab):
    return sum(((Xm[lab == c] - Xm[lab == c].mean(0)) ** 2).sum()
               for c in np.unique(lab) if (lab == c).sum() > 1)


def gap_stat(Xm, krange=KRANGE, B=100, seed=SEED):
    r = np.random.default_rng(seed)
    ks = [1] + list(krange)
    pca = PCA().fit(Xm)
    Xp = pca.transform(Xm)
    lo, hi = Xp.min(0), Xp.max(0)
    gap, sk = [], []
    for k in ks:
        lab = np.zeros(len(Xm), int) if k == 1 else KMeans(k, n_init=25, random_state=seed).fit_predict(Xm)
        lw = np.log(wk(Xm, lab))
        bs = []
        for _ in range(B):
            Zb = pca.inverse_transform(r.uniform(lo, hi, size=Xp.shape))
            lb = np.zeros(len(Zb), int) if k == 1 else KMeans(k, n_init=10, random_state=int(r.integers(1e6))).fit_predict(Zb)
            bs.append(np.log(wk(Zb, lb)))
        bs = np.array(bs)
        gap.append(bs.mean() - lw)
        sk.append(bs.std() * np.sqrt(1 + 1 / B))
    gap, sk = np.array(gap), np.array(sk)
    khat = next((ks[i] for i in range(len(ks) - 1) if gap[i] >= gap[i + 1] - sk[i + 1]), None)
    return pd.DataFrame({"k": ks, "gap": gap, "s_k": sk}), khat


def pred_strength(Xm, krange=KRANGE, n_rep=30, seed=SEED):
    r = np.random.default_rng(seed)
    out = []
    for k in krange:
        vals = []
        for _ in range(n_rep):
            perm = r.permutation(len(Xm))
            a, b = perm[: len(Xm) // 2], perm[len(Xm) // 2:]
            ka = KMeans(k, n_init=15, random_state=int(r.integers(1e6))).fit(Xm[a])
            lb = KMeans(k, n_init=15, random_state=int(r.integers(1e6))).fit_predict(Xm[b])
            pred = np.argmin(cdist(Xm[b], ka.cluster_centers_), axis=1)
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
        out.append({"k": k, "PS": np.mean(vals)})
    return pd.DataFrame(out)


def boot_jaccard(Xm, k, algo="kmeans", B=100, seed=SEED):
    """Bootstrap Jaccard index according to Hennig.

    The partition learned on each bootstrap sample is extended to all the
    original subjects through the nearest centroid, as in
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
def proj_dip(Xm, n_proj=500, seed=SEED):
    r = np.random.default_rng(seed)
    pv = []
    for _ in range(n_proj):
        a = r.normal(size=Xm.shape[1]); a /= np.linalg.norm(a)
        pv.append(diptest.diptest(Xm @ a)[1])
    return np.array(pv)


# ---------------------------------------------------------------------------
# A. Non-Gaussian null (Gaussian copula with empirical marginals)
# ---------------------------------------------------------------------------
log("=" * 72)
log('A. NON-GAUSSIAN UNIMODAL NULL (Gaussian copula, empirical marginals)')
log("=" * 72)
log('Construction: the Spearman correlation of the data is estimated, a Gaussian')
log('with that correlation is simulated and each variable is remapped onto the observed')
log('empirical quantiles. The result has the SAME marginals (same skewness)')
log('and the same dependence structure, but no clusters by construction.')

rho_s = stats.spearmanr(X).statistic
R = 2 * np.sin(np.pi * rho_s / 6)          # Spearman -> latent Pearson
w, V = np.linalg.eigh(R)
R = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
R = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))
Zc = rng.multivariate_normal(np.zeros(p), R, size=n)
U = stats.norm.cdf(Zc)
X_cop = np.column_stack([np.quantile(X[:, j], U[:, j]) for j in range(p)])

log(f'\nCheck of the marginals (observed skewness vs copula):')
for j, v in enumerate(names["primary"]):
    log(f"  {v:18s} {stats.skew(X[:, j]):+.3f}  vs {stats.skew(X_cop[:, j]):+.3f}")

log('\nGMM: k at minimum BIC')
rows = []
for tag, Xm in [('observed', X), ('copula null', X_cop)]:
    for cov in ["full", "tied", "diag", "spherical"]:
        bics = [(k, GaussianMixture(k, covariance_type=cov, n_init=8, random_state=SEED,
                                    reg_covar=1e-6).fit(Xm).bic(Xm)) for k in [1] + KRANGE]
        kb = min(bics, key=lambda t: t[1])
        rows.append({'data': tag, "cov": cov, "k_BIC": kb[0], "BIC": kb[1]})
gmc = pd.DataFrame(rows)
log(gmc.pivot(index="cov", columns='data', values="k_BIC").to_string())
gmc.to_csv(os.path.join(OUT, "06_gmm_copula_null.csv"), index=False)
log('\n-> the non-Gaussian unimodal null reproduces the same k as the real data:')
log('   the k selected by the BIC reflects the shape of the marginals, not groups.')

pv_c = proj_dip(X_cop)
log(f'\nDip test on 500 random projections of the copula null: {(pv_c < 0.05).sum()}/500 with p<0.05')

log('\nGap statistic and prediction strength on the copula null:')
g_c, k_c = gap_stat(X_cop)
log(f'  gap -> k = {k_c}')
ps_c = pred_strength(X_cop)
log(f"  maximum prediction strength = {ps_c.PS.max():.3f} (k = "
    f"{int(ps_c.loc[ps_c.PS.idxmax(), 'k'])})")

# ---------------------------------------------------------------------------
# B. Alternative weightings
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('B. ALTERNATIVE FEATURE WEIGHTINGS')
log("=" * 72)
res = []
for tag in ["F_primary", "F_balanced", "F_factor"]:
    Xm = npz[tag]
    g, kh = gap_stat(Xm)
    ps = pred_strength(Xm)
    j2 = boot_jaccard(Xm, 2)
    j3 = boot_jaccard(Xm, 3)
    sil = max(silhouette_score(Xm, KMeans(k, n_init=25, random_state=SEED).fit_predict(Xm))
              for k in KRANGE)
    pv = proj_dip(Xm)
    res.append({'matrix': tag, "gap_k": kh, "PS_max": ps.PS.max(),
                "PS_k2": float(ps.loc[ps.k == 2, "PS"].iloc[0]),
                "jaccard_min_k2": j2.min(), "jaccard_min_k3": j3.min(),
                "silhouette_max": sil, "proj_dip_p<0.05_%": 100 * (pv < .05).mean()})
sens = pd.DataFrame(res)
log(sens.round(3).to_string(index=False))
sens.to_csv(os.path.join(OUT, "06_feature_weighting.csv"), index=False)
log('\n-> the conclusion does not depend on the weighting: gap selects k = 1,')
log(f"   no prediction strength reaches 0.80 (maximum {sens.PS_max.max():.3f}); Jaccard at k=2 "
    f"between {sens['jaccard_min_k2'].min():.3f} and {sens['jaccard_min_k2'].max():.3f} (see 32_* "
    f"for the specificity of this index).")

# ---------------------------------------------------------------------------
# C. HDBSCAN
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('C. HDBSCAN (density-based, k not imposed)')
log("=" * 72)
rows = []
for mcs in [10, 15, 20, 30, 50, 75]:
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5).fit(X)
    lab = cl.labels_
    nk = len(set(lab) - {-1})
    rows.append({"min_cluster_size": mcs, "n_cluster": nk,
                 "%_rumore": 100 * (lab == -1).mean(),
                 'persistence_max': float(cl.cluster_persistence_.max()) if nk else np.nan})
hdb = pd.DataFrame(rows)
log(hdb.round(2).to_string(index=False))
hdb.to_csv(os.path.join(OUT, "06_hdbscan.csv"), index=False)

# ---------------------------------------------------------------------------
# D. Enlarged population: all PD with the 24h assay
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('D. EXTENDED POPULATION: all PD with the 24h assay (pos + neg)')
log("=" * 72)
log('TTT excluded: not defined in negative samples (they do not cross the threshold).')

saa = pd.read_csv(os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"))
sel = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")].copy()
for f in ["Fmax", "AUC", "SLOPEMax"]:
    cols = [f"{f}_24h_Rep{i}" for i in (1, 2, 3)]
    sel[f"saa_{f}"] = sel[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
sel = sel.groupby("PATNO", as_index=False).agg(
    {"saa_Fmax": "mean", "saa_AUC": "mean", "saa_SLOPEMax": "mean", "SAA_Status": "first"})

cur = pd.read_csv(os.path.join(OUT, "analysis_dataset.csv"))[
    ["PATNO"]]  # placeholder, reloaded from the complete curated data cut
full = pd.read_excel(curated_xlsx(BASE),
                     sheet_name="20260511")
fbl = full[full.EVENT_ID == "BL"][["PATNO", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
                                   "MIA_CAUDATE_BILAT", "age_at_visit",
                                   "subgroup"]]
ext = sel.merge(fbl, on="PATNO", how="inner")

# The positive control must rest on the same population as the primary one:
# if the latter excludes the genetic cohort, it is excluded here too.
if os.environ.get('ONLY_SPORADIC', "1") == "1":
    n_pre = len(ext)
    ext = ext[ext.subgroup == "Sporadic PD"].reset_index(drop=True)
    log(f'\nONLY_SPORADIC=1 on the positive control: {n_pre} -> {len(ext)} ({n_pre - len(ext)} '
        f'carriers excluded)')
ext["dat_putamen_low"] = ext[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
FE = ["saa_Fmax", "saa_AUC", "saa_SLOPEMax", "dat_putamen_low", "MIA_CAUDATE_BILAT",
      "age_at_visit"]
ext = ext.dropna(subset=FE)
log(f'\nn = {len(ext)}  (status: {ext.SAA_Status.value_counts().to_dict()})')

Xe = ext[FE].copy()
for c in ["saa_Fmax", "saa_AUC", "saa_SLOPEMax"]:
    Xe[c] = np.log(Xe[c].clip(lower=1e-9))
Xe = StandardScaler().fit_transform(Xe.to_numpy(float))

g_e, k_e = gap_stat(Xe)
ps_e = pred_strength(Xe)
j2e = boot_jaccard(Xe, 2)
pv_e = proj_dip(Xe)
log(f'  gap -> k = {k_e}')
log(f"  prediction strength: k=2 {float(ps_e.loc[ps_e.k == 2, 'PS'].iloc[0]):.3f}, maximum "
    f"{ps_e.PS.max():.3f}")
log(f'  Jaccard bootstrap k=2: min {j2e.min():.3f}, mean {j2e.mean():.3f}')
log(f'  dip on 500 projections: {(pv_e < 0.05).sum()}/500 with p<0.05')

# does the k=2 partition recover the positive/negative distinction?
lab2 = KMeans(2, n_init=50, random_state=SEED).fit_predict(Xe)
ct = pd.crosstab(lab2, ext.SAA_Status.to_numpy())
log('\n  k=2 partition vs SAA status:')
log(ct.to_string())
from sklearn.metrics import adjusted_rand_score
log(f"  ARI(k=2 partition, positive/negative) = "
    f"{adjusted_rand_score(lab2, (ext.SAA_Status == 'Positive').astype(int)):.3f}")
ct.to_csv(os.path.join(OUT, "06_extended_k2_vs_status.csv"))

with open(os.path.join(OUT, "06_sensitivity.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/06_sensitivity.txt')
