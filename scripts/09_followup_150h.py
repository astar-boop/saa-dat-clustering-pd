"""
09 - Two targeted checks on the 150h replication.

A) The "senza AUC" variant reaches prediction strength 0.874 at k = 2, above the
   threshold of 0.80. It must be compared with a unimodal null built on the SAME
   matrix, and it must be characterised: what separates that partition?

B) In the 150h the PC1 axis (T50 vs AUC) correlates with CSF tau, Abeta and p-tau,
   an association absent in the 24h. It must be tested directly on the single
   kinetic parameters instead of through the principal components, and it must be
   verified whether it survives adjustment for age and sex.

Output: outputs/09_*.csv, outputs/09_followup_150h.txt
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import diptest

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


dc = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
dc = dc[dc.complete].reset_index(drop=True)
log(f"n = {len(dc)}")

NOAUC = ["lg_Fmax", "lg_T50", "lg_SLOPE", "dat_putamen_low",
         "dat_caudate_mean", "age_bl"]
X = StandardScaler().fit_transform(dc[NOAUC].to_numpy(float))
X_null = rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), size=len(X))


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
        out.append({"k": k, "PS": np.mean(vals), "PS_sd": np.std(vals)})
    return pd.DataFrame(out)


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
# ===========================================================================
# A. The variant without AUC
# ===========================================================================
log("\n" + "=" * 72)
# The value to query is that of the sensitivity analysis (08), not a
# number transcribed by hand: 0.874 from a previous round had remained while the
# table below printed another one.
_sens = pd.read_csv(os.path.join(OUT, "08_sensitivity_150h.csv"))
_ps_noauc = float(_sens.loc[_sens.variant.str.startswith('without AUC'), "PS_k2"].iloc[0])
log(f"A. VARIANT WITHOUT AUC: is the prediction strength {_ps_noauc:.3f} (08_sensitivity_150h.csv) "
    f"real?")
log("=" * 72)

ps_o = pred_strength(X)
ps_n = pred_strength(X_null)
ps = ps_o.merge(ps_n, on="k", suffixes=('_obsv', '_null'))
log('\nPrediction strength, matrix without AUC:')
log(ps.round(3).to_string(index=False))
log(f"\n  -> the unimodal null on the SAME matrix reaches {ps_n.PS.max():.3f} at k = "
    f"{int(ps_n.loc[ps_n.PS.idxmax(), 'k'])}.")

j_o = boot_jaccard(X, 2)
j_n = boot_jaccard(X_null, 2)
log(f'\nBootstrap Jaccard k=2 — observed {j_o.mean():.3f} (min {j_o.min():.3f}) | null '
    f'{j_n.mean():.3f} (min {j_n.min():.3f})')
log(f'Silhouette k=2 — observed '
    f'{silhouette_score(X, KMeans(2, n_init=50, random_state=SEED).fit_predict(X)):.3f} | null '
    f'{silhouette_score(X_null, KMeans(2, n_init=50, random_state=SEED).fit_predict(X_null)):.3f}')

pv = np.empty(1000)
for i in range(1000):
    a = rng.normal(size=X.shape[1]); a /= np.linalg.norm(a)
    pv[i] = diptest.diptest(X @ a)[1]
log(f'Dip on 1,000 random projections: {(pv < 0.05).sum()}/1000 with p<0.05 (expected ~50)')

lab = KMeans(2, n_init=100, random_state=SEED).fit_predict(X)
dc["cl_noauc"] = lab
log(f"\nSample sizes of the k=2 partition: {dict(zip(*np.unique(lab, return_counts=True)))}")
log('\nProfile on the construction features (median [IQR]):')
prof = []
for v in NOAUC:
    g = [dc.loc[lab == c, v] for c in (0, 1)]
    h, pval = stats.kruskal(*g)
    eps2 = (h - 1) / (len(dc) - 2)
    prof.append({'variable': v, "g0": f"{g[0].median():.3f}", "g1": f"{g[1].median():.3f}",
                 "p": pval, "epsilon2": eps2})
prof = pd.DataFrame(prof)
log(prof.round(4).to_string(index=False))

log('\nEXTERNAL variables (not used to build the partition), BH-FDR:')
EXT = ["updrs3_score", "updrs2_score", "updrs_totscore", "hy", "pigd", "MSEADLG",
       "moca", "upsit", "updrs1_score", "ess", "rem", "gds", "stai", "scopa",
       "duration_yrs", "LEDD", "abeta", "tau", "ptau", "nfl_serum", "urate",
       "SEX", "EDUCYRS", "BMI", "APOE_e4"]
rows = []
for v in EXT:
    if v not in dc.columns or dc[v].notna().sum() < 30:
        continue
    g = [dc.loc[(lab == c) & dc[v].notna(), v] for c in (0, 1)]
    if min(len(x) for x in g) < 10:
        continue
    # a constant variable does not admit a test: it happens under ONLY_SPORADIC,
    # where LEDD is 0 for everyone (the de novo arm is drug-naive and the treated
    # of the 150h are almost all carriers). It must be flagged, not dropped.
    everyone = pd.concat(g)
    if everyone.nunique() <= 1:
        log(f'  {v}: constant in the cohort (single value {everyone.iloc[0]:g}) — test not defined, '
            f'variable omitted')
        continue
    h, pval = stats.kruskal(*g)
    rows.append({'variable': v, "n": sum(len(x) for x in g),
                 "g0": g[0].median(), "g1": g[1].median(), "p": pval,
                 "epsilon2": (h - 1) / (sum(len(x) for x in g) - 2)})
ext = pd.DataFrame(rows)
ext["p_BH"] = multipletests(ext.p, method="fdr_bh")[1]
log(ext.round(4).to_string(index=False))
log(f'\n  significant after BH: {int((ext.p_BH < 0.05).sum())} / {len(ext)}; maximum epsilon2 '
    f'{ext.epsilon2.max():.4f}')
ext.to_csv(os.path.join(OUT, "09_noauc_k2_external.csv"), index=False)

# does the partition simply reproduce a cut on a single variable?
log('\nARI between the k=2 partition and a median dichotomisation of the single features:')
for v in NOAUC:
    d2 = (dc[v] > dc[v].median()).astype(int)
    log(f'  {v:18s} ARI = {adjusted_rand_score(lab, d2):.3f}')


# ===========================================================================
# B. Direct associations of the 150h kinetic parameters
# ===========================================================================
log("\n" + "=" * 72)
log('B. 150h KINETIC PARAMETERS vs EXTERNAL VARIABLES — direct tests')
log("=" * 72)
log("Raw Spearman and partial correlation adjusted for age and sex.")

KINV = ["lg_Fmax", "lg_T50", "lg_AUC", "lg_SLOPE"]
TARGETS = ["abeta", "tau", "ptau", "nfl_serum", "urate", "updrs3_score",
           "updrs_totscore", "hy", "moca", "upsit", "scopa", "rem", "gds",
           "duration_yrs", "MIA_LOWPUT_EXPECTED", "dat_putamen_low",
           "dat_caudate_mean"]

rows = []
for k in KINV:
    for t in TARGETS:
        if t not in dc.columns:
            continue
        m = dc[[k, t, "age_bl", "SEX"]].dropna()
        if len(m) < 40:
            continue
        rho, pval = stats.spearmanr(m[k], m[t])
        # partial: residuals of the regressions on age and sex (ranks)
        Z = sm.add_constant(m[["age_bl", "SEX"]].to_numpy(float))
        rk = sm.OLS(stats.rankdata(m[k]), Z).fit().resid
        rt = sm.OLS(stats.rankdata(m[t]), Z).fit().resid
        prho, ppval = stats.pearsonr(rk, rt)
        rows.append({'kinetic': k, 'outside': t, "n": len(m), "rho": rho,
                     "p": pval, 'rho_partial': prho, 'p_partial': ppval})
cc = pd.DataFrame(rows)
cc["p_BH"] = multipletests(cc.p, method="fdr_bh")[1]
cc['p_partial_BH'] = multipletests(cc.p_partial, method="fdr_bh")[1]
cc.to_csv(os.path.join(OUT, "09_kinetics_correlations_150h.csv"), index=False)

sig = cc[cc.p_partial_BH < 0.05].reindex(
    cc[cc.p_partial_BH < 0.05].rho_partial.abs().sort_values(ascending=False).index)
log(f"\nSignificant after adjustment for age/sex and BH-FDR: {len(sig)} out of {len(cc)}")
log(sig.round(4).to_string(index=False))
log(f'\nMaximum |partial rho| over all the {len(cc)} pairs: {cc.rho_partial.abs().max():.3f}')

log('\nDetail of the CSF biomarkers (tau, Abeta, p-tau):')
log(cc[cc.outside.isin(["tau", "abeta", "ptau"])].round(4).to_string(index=False))

with open(os.path.join(OUT, "09_followup_150h.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/09_followup_150h.txt')
