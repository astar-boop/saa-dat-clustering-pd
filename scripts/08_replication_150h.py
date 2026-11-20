"""
08 - Replication of the clustering battery on the 150h protocol (Project 155).

Reason: to verify whether the absence of structure found on the 24h protocol holds
when the assay is changed. The overlap between the two cohorts is only 16%
(81 participants out of 491), so this is an almost independent replication.

Differences from the 24h analysis:
  - the 150h provides T50, absent in the 24h; TTT is excluded because it is redundant
    (Spearman's rho with T50 = 0.96);
  - the replicates are aggregated with the MEDIAN and not with the mean: one well
    reports SLOPE = 9699 against 13 and 50 in the same triplet. The mean is
    reported as a sensitivity analysis;
  - the Type1/Type2 subtyping does not exist in Project 155, so the
    population is the set of the positives.

Output: outputs/08_*.csv, outputs/08_replication_150h.txt
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score)
from statsmodels.stats.multitest import multipletests
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


# ===========================================================================
# 1. CONSTRUCTION OF THE DATASET
# ===========================================================================
log("=" * 72)
log('1. DATASET — 150h protocol (Project 155, Amprion pilot)')
log("=" * 72)

sa = pd.read_csv(os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"))
p = sa[(sa.PROJECTID == 155) & (sa.COHORT == "PD") & (sa.CLINICAL_EVENT == "BL")].copy()
log(f'PD at baseline in the 150h: {len(p)} samples, {p.PATNO.nunique()} subjects')
log(f'  SAA outcome: {p.SAA_Status.value_counts().to_dict()}')

KIN = ["Fmax", "TTT", "AUC", "T50", "SLOPE"]
n_flag = 0
for f in KIN:
    cols = [f"{f}_150h_Rep{i}" for i in (1, 2, 3)]
    v = p[cols].apply(pd.to_numeric, errors="coerce")
    p[f"k_{f}"] = v.median(axis=1)          # primary: median of the replicates
    p[f"kmean_{f}"] = v.mean(axis=1)        # sensitivity: mean
    p[f"kcv_{f}"] = v.std(axis=1, ddof=1) / v.mean(axis=1)
    # count of aberrant replicates: > 5x or < 1/5 of the median of the triplet
    med = v.median(axis=1)
    ratio = v.div(med, axis=0)
    n_flag += int(((ratio > 5) | (ratio < 0.2)).sum().sum())
log(f'  aberrant replicates (>5x or <1/5 of the median of the triplet): {n_flag}')

pos = p[p.SAA_Status == "Positive"].copy()
log(f'  positives: {len(pos)} samples, {pos.PATNO.nunique()} subjects')
agg = {f"k_{f}": "mean" for f in KIN}
agg.update({f"kmean_{f}": "mean" for f in KIN})
pos = pos.groupby("PATNO", as_index=False).agg(agg)

# DAT-SCAN, age and clinical variables from the curated data cut
cur = pd.read_excel(curated_xlsx(BASE),
                    sheet_name="20260511")
cbl = cur[cur.EVENT_ID == "BL"].copy()
CLIN = ["MIA_CAUDATE_L", "MIA_CAUDATE_R", "MIA_CAUDATE_BILAT", "MIA_PUTAMEN_L",
        "MIA_PUTAMEN_R", "MIA_PUTAMEN_BILAT", "MIA_STRIATUM_BILAT",
        "MIA_LOWPUT_EXPECTED", "age_at_visit", "SEX", "EDUCYRS", "BMI",
        "duration_yrs", "ageonset", "agediag", "LEDD", "PDTRTMNT", "hy",
        "updrs1_score", "updrs2_score", "updrs3_score", "updrs_totscore",
        "pigd", "td_pigd", "MSEADLG", "moca", "cogstate", "upsit", "upsit_pctl",
        "ess", "rem", "gds", "stai", "scopa", "quip_any", "abeta", "tau",
        "ptau", "NFL_CSF", "nfl_serum", "urate", "APOE_e4", "fampd_bin",
        "subgroup"]
d = pos.merge(cbl[["PATNO"] + [c for c in CLIN if c in cbl.columns]],
              on="PATNO", how="left", validate="one_to_one")
d["dat_putamen_low"] = d[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
d["dat_putamen_mean"] = d["MIA_PUTAMEN_BILAT"]
d["dat_striatum_mean"] = d["MIA_STRIATUM_BILAT"]
d["dat_caudate_mean"] = d["MIA_CAUDATE_BILAT"]
d["age_bl"] = d["age_at_visit"]

# Which DAT pair enters the primary set: see _dat.py. Default unchanged.
import _dat
DAT_ALL = _dat.variables()
log(f'\nDAT_SET = {_dat.choice()} -> DAT block: {DAT_ALL}')

# Exclusion of the genetic cohort (see 01_build_dataset.py, section 3bis).
# In the 150h its weight is much greater than in the 24h: ~30% of the participants.
if os.environ.get('ONLY_SPORADIC', "1") == "1":
    log('\nONLY_SPORADIC=1 — exclusion of the genetic cohort')
    log(f"  composition before the filter:\n{d.subgroup.fillna('unknown').value_counts().to_string()}")
    n_pre = len(d)
    d = d[d.subgroup == "Sporadic PD"].reset_index(drop=True)
    log(f'  excluded {n_pre - len(d)} carriers -> {len(d)} sporadic')
else:
    log('\nONLY_SPORADIC not active: full cohort (sporadic + carriers)')

log('\nSpearman correlations between the five 150h kinetic parameters (median of the replicates):')
cm = d[[f"k_{f}" for f in KIN]].corr(method="spearman")
cm.index = cm.columns = KIN
log(cm.round(3).to_string())
log(f"\n  TTT excluded: rho with T50 = {cm.loc['TTT', 'T50']:.3f}. AUC retained but strongly "
    f"collinear with T50 (rho = {cm.loc['AUC', 'T50']:.3f}); variant without AUC in sensitivity.")

for f in ["Fmax", "AUC", "T50", "SLOPE"]:
    d[f"lg_{f}"] = np.log(d[f"k_{f}"])
    d[f"lgmean_{f}"] = np.log(d[f"kmean_{f}"])

PRIMARY = ["lg_Fmax", "lg_T50", "lg_AUC", "lg_SLOPE"] + DAT_ALL + ["age_bl"]
d["complete"] = d[PRIMARY].notna().all(axis=1)
log(f'\nComplete cases on the 7 primary features: {int(d.complete.sum())} / {len(d)}')
dc = d[d.complete].reset_index(drop=True).copy()

prev = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
ov = len(set(dc.PATNO) & set(prev.PATNO))
log(f'Overlap with the 24h cohort (n={len(prev)}): {ov} subjects ({100 * ov / len(dc):.0f}% — almost '
    f'independent replication)')

X = StandardScaler().fit_transform(dc[PRIMARY].to_numpy(float))
n, pdim = X.shape
X_null = rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), size=n)
dc.to_csv(os.path.join(OUT, "08_dataset_150h.csv"), index=False)


# ===========================================================================
# functions of the battery (identical to scripts 02-03)
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


ALGOS = ["kmeans", "ward", "average", "gmm"]


def hopkins(Xm, m_frac=0.10, n_rep=200, seed=SEED):
    r = np.random.default_rng(seed)
    nn, pp = Xm.shape
    m = max(5, int(m_frac * nn))
    lo, hi = Xm.min(0), Xm.max(0)
    tree = cKDTree(Xm)
    vals = []
    for _ in range(n_rep):
        idx = r.choice(nn, m, replace=False)
        w = tree.query(Xm[idx], k=2)[0][:, 1]
        u = tree.query(r.uniform(lo, hi, size=(m, pp)), k=1)[0]
        vals.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(vals))


def wk(Xm, lab):
    return sum(((Xm[lab == c] - Xm[lab == c].mean(0)) ** 2).sum()
               for c in np.unique(lab) if (lab == c).sum() > 1)


def gap_stat(Xm, B=200, seed=SEED):
    r = np.random.default_rng(seed)
    ks = [1] + KRANGE
    pca = PCA().fit(Xm)
    Xp = pca.transform(Xm)
    lo, hi = Xp.min(0), Xp.max(0)
    gap, sk = [], []
    for k in ks:
        lab = np.zeros(len(Xm), int) if k == 1 else fit_labels(Xm, k, "kmeans")
        lw = np.log(wk(Xm, lab))
        bs = []
        for _ in range(B):
            Zb = pca.inverse_transform(r.uniform(lo, hi, size=Xp.shape))
            lb = np.zeros(len(Zb), int) if k == 1 else KMeans(
                k, n_init=10, random_state=int(r.integers(1e6))).fit_predict(Zb)
            bs.append(np.log(wk(Zb, lb)))
        bs = np.array(bs)
        gap.append(bs.mean() - lw)
        sk.append(bs.std() * np.sqrt(1 + 1 / B))
    gap, sk = np.array(gap), np.array(sk)
    khat = next((ks[i] for i in range(len(ks) - 1)
                 if gap[i] >= gap[i + 1] - sk[i + 1]), None)
    return pd.DataFrame({"k": ks, "gap": gap, "s_k": sk}), khat


def pred_strength(Xm, n_rep=50, seed=SEED):
    r = np.random.default_rng(seed)
    out = []
    for k in KRANGE:
        vals = []
        for _ in range(n_rep):
            perm = r.permutation(len(Xm))
            a, b = perm[: len(Xm) // 2], perm[len(Xm) // 2:]
            ka = KMeans(k, n_init=20, random_state=int(r.integers(1e6))).fit(Xm[a])
            lb = KMeans(k, n_init=20, random_state=int(r.integers(1e6))).fit_predict(Xm[b])
            pred = np.argmin(cdist(Xm[b], ka.cluster_centers_), axis=1)
            worst = 1.0
            for c in range(k):
                mk = lb == c
                nc = mk.sum()
                if nc <= 1:
                    worst = 0.0
                    continue
                cnt = np.bincount(pred[mk], minlength=k)
                worst = min(worst, (cnt * (cnt - 1)).sum() / (nc * (nc - 1)))
            vals.append(worst)
        out.append({"k": k, "PS": np.mean(vals)})
    return pd.DataFrame(out)


def boot_jaccard(Xm, k, algo="kmeans", B=200, seed=SEED):
    """Bootstrap Jaccard index according to Hennig.

    The partition learned on each bootstrap sample is extended to
    all the original participants through the nearest centroid, as in
    fpc::clusterboot. Without this extension the index would be bounded
    above by the proportion of distinct participants present in the bootstrap
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
def proj_dip(Xm, n_proj=1000, seed=SEED):
    r = np.random.default_rng(seed)
    pv = np.empty(n_proj)
    for i in range(n_proj):
        a = r.normal(size=Xm.shape[1])
        a /= np.linalg.norm(a)
        pv[i] = diptest.diptest(Xm @ a)[1]
    return pv


# ===========================================================================
# 2. CLUSTER TENDENCY
# ===========================================================================
log("\n" + "=" * 72)
log('2. CLUSTERING TENDENCY')
log("=" * 72)
log(f'\nHopkins observed         : {hopkins(X):.3f}')
log(f'Hopkins unimodal null   : {hopkins(X_null):.3f}')

dip_rows = []
PCs = PCA().fit_transform(X)
for j, v in enumerate(PRIMARY):
    sd, pv = diptest.diptest(X[:, j])
    dip_rows.append({'space': "feature", "var": v, "dip": sd, "p": pv})
for j in range(pdim):
    sd, pv = diptest.diptest(PCs[:, j])
    dip_rows.append({'space': "PCA", "var": f"PC{j+1}", "dip": sd, "p": pv})
dip = pd.DataFrame(dip_rows)
dip["p_bonf"] = np.minimum(1, dip.p * len(dip))
log("\nHartigan dip test (H0 = unimodality):")
log(dip.round(4).to_string(index=False))
dip.to_csv(os.path.join(OUT, "08_diptest_150h.csv"), index=False)

pv_obs = proj_dip(X)
pv_nul = proj_dip(X_null)
log(f'\nDip on 1,000 random projections — observed: {(pv_obs < 0.05).sum()}/1000 with p<0.05 '
    f'(expected ~50); minimum p after Bonferroni = {min(1, pv_obs.min() * 1000):.3g}')
log(f'                                    null: {(pv_nul < 0.05).sum()}/1000')

# ===========================================================================
# 3. SELECTION OF k AND STABILITY
# ===========================================================================
log("\n" + "=" * 72)
log("3. SELECTION OF k AND STABILITY")
log("=" * 72)

g_o, k_o = gap_stat(X)
g_n, k_n = gap_stat(X_null)
log(f'\nGap statistic — observed: k = {k_o} | null: k = {k_n}')
log(g_o.round(4).to_string(index=False))
g_o.to_csv(os.path.join(OUT, "08_gap_150h.csv"), index=False)

ps_o, ps_n = pred_strength(X), pred_strength(X_null)
ps = ps_o.merge(ps_n, on="k", suffixes=('_obsv', '_null'))
log('\nPrediction strength (threshold 0.80):')
log(ps.round(3).to_string(index=False))
ok = ps_o[ps_o.PS >= 0.80].k.tolist()
log(f"  k with PS >= 0.80: {(ok if ok else 'NONE -> only k = 1 supported')}")
ps.to_csv(os.path.join(OUT, "08_predstrength_150h.csv"), index=False)

rows = []
for tag, Xm in [('observed', X), ('nullref', X_null)]:
    for k in KRANGE:
        j = boot_jaccard(Xm, k)
        lab = fit_labels(Xm, k, "kmeans")
        labs = {a: fit_labels(Xm, k, a) for a in ALGOS}
        aris = [adjusted_rand_score(labs[ALGOS[i]], labs[ALGOS[j2]])
                for i in range(4) for j2 in range(i + 1, 4)]
        rows.append({'data': tag, "k": k, "silhouette": silhouette_score(Xm, lab),
                     "davies_bouldin": davies_bouldin_score(Xm, lab),
                     "jaccard_min": j.min(), 'jaccard_avg': j.mean(),
                     "n_stabili_0.75": int((j >= 0.75).sum()),
                     'ARI_avg_algorithms': np.mean(aris)})
st = pd.DataFrame(rows)
st.to_csv(os.path.join(OUT, "08_stability_150h.csv"), index=False)
for tag in ['observed', 'nullref']:
    log(f"\n{tag}:")
    log(st[st.data == tag].drop(columns='data').round(3).to_string(index=False))

log('\nHDBSCAN:')
for mcs in [10, 15, 20, 30, 50]:
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5).fit(X)
    nk = len(set(cl.labels_) - {-1})
    log(f'  min_cluster_size={mcs:3d} -> {nk} clusters, {100 * (cl.labels_ == -1).mean():.1f}% noise')

# ===========================================================================
# 4. SENSITIVITY
# ===========================================================================
log("\n" + "=" * 72)
log("4. SENSITIVITY")
log("=" * 72)

# The variants share the DAT block of the current branch: if only the primary
# changed, the sensitivity analysis would measure two things at once.
variants = {
    'primary (median of replicates, with AUC)': PRIMARY,
    'without AUC (collinear with T50)': ["lg_Fmax", "lg_T50", "lg_SLOPE"]
                                      + DAT_ALL + ["age_bl"],
    'TTT in place of T50': None,
    'mean of replicates (as in the 24h)': ["lgmean_Fmax", "lgmean_T50", "lgmean_AUC",
                                      "lgmean_SLOPE"] + DAT_ALL + ["age_bl"],
}
dc["lg_TTT"] = np.log(dc.k_TTT)
variants['TTT in place of T50'] = (["lg_Fmax", "lg_TTT", "lg_AUC", "lg_SLOPE"]
                                   + DAT_ALL + ["age_bl"])

srows = []
for name, cols in variants.items():
    sub = dc[cols].dropna()
    Xv = StandardScaler().fit_transform(sub.to_numpy(float))
    _, kg = gap_stat(Xv, B=100)
    psv = pred_strength(Xv, n_rep=30)
    jv = boot_jaccard(Xv, 2, B=100)
    pvv = proj_dip(Xv, n_proj=500)
    srows.append({'variant': name, "n": len(sub), "gap_k": kg,
                  "PS_max": psv.PS.max(), "PS_k2": float(psv.loc[psv.k == 2, "PS"].iloc[0]),
                  "jaccard_k2": jv.mean(),
                  "silhouette_k2": silhouette_score(Xv, fit_labels(Xv, 2, "kmeans")),
                  "proj_dip_%": 100 * (pvv < .05).mean()})
sens = pd.DataFrame(srows)
log(sens.round(3).to_string(index=False))
sens.to_csv(os.path.join(OUT, "08_sensitivity_150h.csv"), index=False)

# ===========================================================================
# 5. CONTINUOUS AXES AND CLINICAL CORRELATES
# ===========================================================================
log("\n" + "=" * 72)
log('5. CONTINUOUS AXES AND CLINICAL CORRELATES')
log("=" * 72)
pca = PCA().fit(X)
load = pd.DataFrame(pca.components_.T, index=PRIMARY,
                    columns=[f"PC{i+1}" for i in range(pdim)])
log(f'\nExplained variance: {np.round(pca.explained_variance_ratio_[:4], 3).tolist()}')
log('\nLoadings (first 4 components):')
log(load.iloc[:, :4].round(3).to_string())
load.to_csv(os.path.join(OUT, "08_pca_loadings_150h.csv"))

EXT = ["updrs3_score", "updrs2_score", "updrs_totscore", "hy", "pigd", "MSEADLG",
       "moca", "upsit", "upsit_pctl", "updrs1_score", "ess", "rem", "gds",
       "stai", "scopa", "duration_yrs", "ageonset", "agediag", "LEDD",
       "abeta", "tau", "ptau", "nfl_serum", "urate", "SEX", "EDUCYRS", "BMI",
       "MIA_LOWPUT_EXPECTED", "MIA_STRIATUM_BILAT"]
for i in range(4):
    dc[f"PC{i+1}"] = PCs[:, i]
crows = []
for v in EXT:
    if v not in dc.columns or dc[v].notna().sum() < 30:
        continue
    m = dc[v].notna()
    for i in range(4):
        rho, pv = stats.spearmanr(dc.loc[m, f"PC{i+1}"], dc.loc[m, v])
        crows.append({'variable': v, 'axis': f"PC{i+1}", "n": int(m.sum()),
                      "rho": rho, "p": pv})
cont = pd.DataFrame(crows)
cont["p_BH"] = multipletests(cont.p, method="fdr_bh")[1]
cont.to_csv(os.path.join(OUT, "08_axes_correlations_150h.csv"), index=False)
sig = cont[cont.p_BH < 0.05].reindex(
    cont[cont.p_BH < 0.05].rho.abs().sort_values(ascending=False).index)
log(f'\nSignificant correlations (BH-FDR < 0.05): {len(sig)} out of {len(cont)} tests')
log(sig.head(25).round(4).to_string(index=False))

# correlates of the SAA kinetic axis alone
saa_axis = load.iloc[:4].abs().sum(0).idxmax()
log(f'\nAxis dominated by the SAA kinetic parameters: {saa_axis}')
log(cont[cont.axis == saa_axis].reindex(
    cont[cont.axis == saa_axis].rho.abs().sort_values(ascending=False).index)
    .head(8).round(4).to_string(index=False))

with open(os.path.join(OUT, "08_replication_150h.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/08_replication_150h.txt')
