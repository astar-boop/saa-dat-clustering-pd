"""
02 - Preparation of the features and CLUSTER TENDENCY tests.

Before looking for k groups one must establish whether a group structure exists.
Here:
  a) the feature matrices are built (primary, block-balanced, factor-based);
  b) the clustering tendency is tested: Hopkins statistic, Hartigan's dip
     test (univariate, on the principal components and on random projections),
     comparison with unimodal null distributions.

Output: outputs/02_*.csv, outputs/02_tendency_report.txt, features_*.npz
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import diptest

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SEED = 20260804
rng = np.random.default_rng(SEED)

rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


# ---------------------------------------------------------------------------
# 1. Data and features
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(OUT, "analysis_dataset.csv"))
d = df[df.complete_primary].reset_index(drop=True).copy()
log(f'Complete cases: n = {len(d)}')

# Transformations: log where the variable is positive and right-skewed.
d["lg_Fmax"] = np.log(d.saa_Fmax)
d["lg_TTT"] = np.log(d.saa_TTT)
d["lg_AUC"] = np.log(d.saa_AUC)
d["lg_SLOPE"] = np.log(d.saa_SLOPEMax)
d["lg_TSmax"] = np.log(d.saa_TSmax)

SAA_ALL = ["lg_Fmax", "lg_TTT", "lg_AUC", "lg_SLOPE"]   # TSmax excluded: rho(TTT)=0.999
# The putaminal (or striatal) measure is parametric: see _dat.py. The default
# 'low' is the original choice, min(L, R) + mean caudate.
import _dat
DAT_ALL = _dat.variables()
AGE = ["age_bl"]

PRIMARY = SAA_ALL + DAT_ALL + AGE
log(f'\nDAT_SET = {_dat.choice()} -> DAT block: {DAT_ALL}')
log(f'\nPrimary set ({len(PRIMARY)}): {PRIMARY}')
log(f'  TSmax excluded a priori: Spearman rho with TTT = '
    f'{stats.spearmanr(d.saa_TTT, d.saa_TSmax).statistic:.4f}')

# ---------------------------------------------------------------------------
# 2. Outliers and correlation structure
# ---------------------------------------------------------------------------
X_raw = d[PRIMARY].to_numpy(float)
# The `complete_primary` flag of 01 is computed on dat_putamen_mean, not on the
# variable selected here: the two lists can diverge. On the current data
# the three DAT variants give the same 687 complete cases, but the coincidence
# must be verified, not assumed, otherwise a NaN enters the matrix silently.
_nan = {v: int(d[v].isna().sum()) for v in PRIMARY if d[v].isna().any()}
assert not _nan, (
    f'NaN in the primary set after the complete_primary filter: {_nan}. The gate of 01 does not '
    f'cover {DAT_ALL}.')
Z = StandardScaler().fit_transform(X_raw)

log('\n=== |z| > 4 by variable ===')
for j, v in enumerate(PRIMARY):
    n_out = int((np.abs(Z[:, j]) > 4).sum())
    log(f"  {v:18s} {n_out}")

# Mahalanobis distance: multivariate outliers
cov = np.cov(Z, rowvar=False)
inv = np.linalg.pinv(cov)
md2 = np.einsum("ij,jk,ik->i", Z - Z.mean(0), inv, Z - Z.mean(0))
crit = stats.chi2.ppf(1 - 0.001, df=len(PRIMARY))
log(f'\nMahalanobis: {int((md2 > crit).sum())} cases beyond chi2_0.999 ({len(PRIMARY)} df, threshold '
    f'{crit:.1f}); max D2 = {md2.max():.1f}')
d["mahalanobis_d2"] = md2

corr = pd.DataFrame(Z, columns=PRIMARY).corr(method="spearman")
log('\n=== Spearman correlations (primary set) ===')
log(corr.round(3).to_string())
corr.to_csv(os.path.join(OUT, "02_corr_primary.csv"))

# ---------------------------------------------------------------------------
# 3. PCA per block (effective dimensionality)
# ---------------------------------------------------------------------------
log('\n=== PCA by block ===')
blocks = {"SAA": SAA_ALL, "DAT": DAT_ALL}
block_scores = {}
for name, cols in blocks.items():
    Zb = StandardScaler().fit_transform(d[cols].to_numpy(float))
    p = PCA().fit(Zb)
    log(f'  {name}: explained variance = {np.round(p.explained_variance_ratio_, 3).tolist()}')
    log(f'       cumulative           = '
        f'{np.round(np.cumsum(p.explained_variance_ratio_), 3).tolist()}')
    for i in range(min(2, len(cols))):
        log(f'       PC{i + 1} loadings: '
            + ", ".join(f"{c}={p.components_[i][j]:+.2f}" for j, c in enumerate(cols)))
    block_scores[name] = p.transform(Zb)[:, :2] if len(cols) >= 2 else p.transform(Zb)

# global PCA on the primary set
pca = PCA().fit(Z)
log('\n=== Global PCA (primary set) ===')
log(f'  explained variance = {np.round(pca.explained_variance_ratio_, 3).tolist()}')
log(f'  cumulative         = {np.round(np.cumsum(pca.explained_variance_ratio_), 3).tolist()}')
PCs = pca.transform(Z)
load = pd.DataFrame(pca.components_.T, index=PRIMARY,
                    columns=[f"PC{i+1}" for i in range(len(PRIMARY))])
log('\nGlobal PCA loadings:')
log(load.round(3).to_string())
load.to_csv(os.path.join(OUT, "02_pca_loadings.csv"))

# ---------------------------------------------------------------------------
# 4. Alternative feature matrices
# ---------------------------------------------------------------------------
# (a) primary: plain z-score -> the SAA block weighs 4/7 of the total variance
F_primary = Z.copy()

# (b) block-balanced: each block (SAA, DAT, age) contributes 1/3
def block_weighted(Zmat, cols, groups):
    W = np.ones(Zmat.shape[1])
    for g in groups:
        idx = [cols.index(c) for c in g]
        W[idx] = np.sqrt(1.0 / (len(groups) * len(idx)))
    return Zmat * W

F_balanced = block_weighted(Z, PRIMARY, [SAA_ALL, DAT_ALL, AGE])

# (c) factor-based: 2 SAA factors + 2 DAT factors + age
F_factor = np.column_stack([
    StandardScaler().fit_transform(block_scores["SAA"]),
    StandardScaler().fit_transform(block_scores["DAT"]),
    StandardScaler().fit_transform(d[["age_bl"]].to_numpy(float)),
])
FACTOR_NAMES = ["SAA_PC1", "SAA_PC2", "DAT_PC1", "DAT_PC2", "age_bl"]

np.savez(os.path.join(OUT, "features.npz"),
         F_primary=F_primary, F_balanced=F_balanced, F_factor=F_factor,
         PATNO=d.PATNO.to_numpy())
with open(os.path.join(OUT, "feature_names.json"), "w") as fh:
    # `dat_set` is the sidecar of the DAT choice: without it, two features.npz files
    # produced by different branches are indistinguishable on disk.
    json.dump({"primary": PRIMARY, "balanced": PRIMARY, "factor": FACTOR_NAMES,
               "saa": SAA_ALL, "dat": DAT_ALL, "dat_set": _dat.choice()},
              fh, indent=2)
d.to_csv(os.path.join(OUT, "analysis_complete_cases.csv"), index=False)

# ---------------------------------------------------------------------------
# 5. CLUSTER TENDENCY
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log('CLUSTER TENDENCY')
log("=" * 70)


def hopkins(X, m_frac=0.10, n_rep=200, seed=0):
    """Hopkins statistic. H ~ 0.5 = no clustering tendency;
    H -> 1 = strong tendency. The reference points are drawn
    uniformly in the hypercube of the data (standard convention)."""
    r = np.random.default_rng(seed)
    n, p = X.shape
    m = max(5, int(m_frac * n))
    lo, hi = X.min(0), X.max(0)
    vals = []
    for _ in range(n_rep):
        idx = r.choice(n, m, replace=False)
        # w: distance from real points to the nearest real neighbour (itself excluded)
        tree_all = cKDTree(X)
        w = tree_all.query(X[idx], k=2)[0][:, 1]
        # u: distance from uniform points to the nearest real neighbour
        U = r.uniform(lo, hi, size=(m, p))
        u = tree_all.query(U, k=1)[0]
        vals.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(vals)), float(np.std(vals))


for label, M in [('primary', F_primary), ("block-balanced", F_balanced),
                 ("factor-based", F_factor)]:
    h, hs = hopkins(M, seed=SEED)
    log(f'\nHopkins ({label}): H = {h:.3f} (sd {hs:.3f})')

# reference: multivariate Gaussian data with the same covariance (unimodal)
Xnull = rng.multivariate_normal(F_primary.mean(0), np.cov(F_primary, rowvar=False),
                                size=len(F_primary))
hn, _ = hopkins(Xnull, seed=SEED)
log(f'Hopkins on a Gaussian null with the same covariance: H = {hn:.3f}')
log('  -> H of the data must be interpreted AGAINST this reference, not against 0.5:')
log("     the statistic is inflated by skewness and tails even without clusters.")

# --- Hartigan's dip test: unimodality ---
log('\n=== Hartigan dip test (H0 = unimodal distribution) ===')
dip_rows = []
for j, v in enumerate(PRIMARY):
    dstat, pval = diptest.diptest(F_primary[:, j])
    dip_rows.append({'space': "feature", "var": v, "dip": dstat, "p": pval})
for j in range(F_primary.shape[1]):
    dstat, pval = diptest.diptest(PCs[:, j])
    dip_rows.append({'space': "PCA", "var": f"PC{j+1}", "dip": dstat, "p": pval})
dip = pd.DataFrame(dip_rows)
dip["p_bonf"] = np.minimum(1, dip.p * len(dip))
log(dip.round(4).to_string(index=False))
dip.to_csv(os.path.join(OUT, "02_diptest.csv"), index=False)

# --- dip test on random projections (multivariate multimodality) ---
log('\n=== Dip test on 1,000 random projections (primary set) ===')
n_proj = 1000
pvals = np.empty(n_proj)
for i in range(n_proj):
    a = rng.normal(size=F_primary.shape[1])
    a /= np.linalg.norm(a)
    pvals[i] = diptest.diptest(F_primary @ a)[1]
log(f'  projections with p < 0.05        : {(pvals < 0.05).sum()} / {n_proj} '
    f'({100 * (pvals < 0.05).mean():.1f}%)')
log(f'  minimum p                        : {pvals.min():.4g}')
log(f'  minimum p after Bonferroni       : {min(1, pvals.min() * n_proj):.4g}')
log("  expected under unimodality ~5% of projections with p<0.05")

# same test on the Gaussian null, as calibration
pnull = np.empty(n_proj)
for i in range(n_proj):
    a = rng.normal(size=Xnull.shape[1])
    a /= np.linalg.norm(a)
    pnull[i] = diptest.diptest(Xnull @ a)[1]
log(f'  [Gaussian null] projections p<0.05: {(pnull < 0.05).sum()} / {n_proj}')

np.save(os.path.join(OUT, "02_proj_dip_pvals.npy"), pvals)

with open(os.path.join(OUT, "02_tendency_report.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/02_tendency_report.txt, features.npz')
