"""
04 - Characterisation: profiles of the partitions and continuous axes.

Two complementary readings:
  A) discrete partition (k = 2 and k = 3, k-means on the primary set): profile
     of the construction features + comparison on external clinical variables;
  B) continuous axes (principal components): correlation with the same
     clinical variables.

The variables used to build the groups (SAA kinetic parameters, DAT, age) are
marked as such: the differences on them are construction-dependent and do not
constitute validation. Validation uses only EXTERNAL variables.
Correction for multiple comparisons: Benjamini-Hochberg within domain.

Output: outputs/04_*.csv, outputs/04_characterization_report.txt
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SEED = 20260804

rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


d = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
npz = np.load(os.path.join(OUT, "features.npz"))
names = json.load(open(os.path.join(OUT, "feature_names.json")))
X = npz["F_primary"]
assert (d.PATNO.to_numpy() == npz["PATNO"]).all()
log(f"n = {len(d)}")

# ---------------------------------------------------------------------------
# External variables (NOT used to build the groups)
# ---------------------------------------------------------------------------
DOMAINS = {
    'Motor': ["updrs3_score", "updrs2_score", "updrs_totscore", "hy",
                "pigd", "td_pigd", "MSEADLG"],
    'Cognitive': ["moca", "cogstate"],
    'Olfaction': ["upsit", "upsit_pctl"],
    'Non-motor': ["updrs1_score", "ess", "rem", "gds", "stai", "scopa",
                    "quip_any"],
    'Disease': ["duration_yrs", "ageonset", "agediag", "LEDD", "PDTRTMNT"],
    'CSF/serum biomarkers': ["abeta", "tau", "ptau", "NFL_CSF", "nfl_serum",
                               "urate"],
    'Genetic': ["APOE_e4", "fampd_bin"],
    'DAT not used': ["dat_lowput_pct_expected", "dat_put_asym", "dat_cp_ratio",
                      "dat_striatum_mean", "dat_putamen_mean"],
    'Demographics': ["SEX", "EDUCYRS", "BMI"],
}
CONSTRUCT = names["primary"]        # construction features
BINARY = {"PDTRTMNT", "APOE_e4", "fampd_bin", "SEX", "quip_any", "cogstate"}

log("\nNOTE: 'ageonset' and 'agediag' are strongly correlated with baseline age,")
log("which is a construction feature -> they are not independent validation.")


def summarize(g_all, var, labels):
    """Global test between groups + descriptive statistics per group."""
    groups = [g_all.loc[labels == c, var].dropna() for c in np.unique(labels)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2 or sum(len(g) for g in groups) < 20:
        return None
    if var in BINARY or g_all[var].dropna().nunique() <= 2:
        tab = pd.crosstab(labels, g_all[var])
        if tab.shape[1] < 2 or tab.shape[0] < 2:
            return None
        chi2, pval, _, _ = stats.chi2_contingency(tab)
        stat_name, stat = "chi2", chi2
        desc = {f"g{c}": f"{100*g.mean():.1f}%" for c, g in
                zip(np.unique(labels), groups)}
    else:
        stat, pval = stats.kruskal(*groups)
        stat_name = "H_KruskalWallis"
        desc = {f"g{c}": f"{g.median():.3g} [{g.quantile(.25):.3g}-{g.quantile(.75):.3g}]"
                for c, g in zip(np.unique(labels), groups)}
    # eta^2 (epsilon^2 for Kruskal-Wallis): share of variability explained
    ntot = sum(len(g) for g in groups)
    eps2 = (stat - len(groups) + 1) / (ntot - len(groups)) if stat_name == "H_KruskalWallis" else np.nan
    return {'variable': var, "n": ntot, "test": stat_name, "stat": stat,
            "p": pval, "epsilon2": eps2, **desc}


# ---------------------------------------------------------------------------
# A. Discrete partitions
# ---------------------------------------------------------------------------
partitions = {}
for k in (2, 3):
    lab = KMeans(k, n_init=100, random_state=SEED).fit_predict(X)
    partitions[k] = lab
    d[f"cluster_k{k}"] = lab

for k, lab in partitions.items():
    log("\n" + "=" * 72)
    log(f'A. PARTITION k = {k} (k-means, primary set)')
    log("=" * 72)
    log(f"Sample sizes: {dict(zip(*np.unique(lab, return_counts=True)))}")

    log('\n-- CONSTRUCTION features (differences expected by definition) --')
    rowsc = []
    for v in ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax",
              "dat_putamen_low", "dat_caudate_mean", "age_bl"]:
        r = summarize(d, v, lab)
        if r:
            rowsc.append(r)
    cdf = pd.DataFrame(rowsc)
    log(cdf.drop(columns=["test", "stat"]).round(4).to_string(index=False))
    cdf.to_csv(os.path.join(OUT, f"04_k{k}_construct_profile.csv"), index=False)

    log('\n-- EXTERNAL variables (validation; BH-FDR within domain) --')
    allr = []
    for dom, vs in DOMAINS.items():
        rr = []
        for v in vs:
            if v not in d.columns:
                continue
            r = summarize(d, v, lab)
            if r:
                r['domain'] = dom
                rr.append(r)
        if not rr:
            continue
        sub = pd.DataFrame(rr)
        sub["p_BH"] = multipletests(sub.p, method="fdr_bh")[1]
        allr.append(sub)
    ext = pd.concat(allr, ignore_index=True)
    ext = ext[['domain', 'variable', "n", "p", "p_BH", "epsilon2"]
              + [c for c in ext.columns if c.startswith("g")]]
    log(ext.round(4).to_string(index=False))
    log(f'\n  external variables with p_BH < 0.05: {int((ext.p_BH < 0.05).sum())} / {len(ext)}')
    log(f"  maximum epsilon^2: {ext.epsilon2.max():.4f} "
        f"({(ext.loc[ext.epsilon2.idxmax(), 'variable'] if ext.epsilon2.notna().any() else '-')})")
    ext.to_csv(os.path.join(OUT, f"04_k{k}_external_validation.csv"), index=False)

# ---------------------------------------------------------------------------
# B. Continuous axes
# ---------------------------------------------------------------------------
log("\n" + "=" * 72)
log('B. CONTINUOUS AXES (principal components of the primary set)')
log("=" * 72)
from sklearn.decomposition import PCA

pca = PCA().fit(X)
PCs = pca.transform(X)
load = pd.DataFrame(pca.components_.T, index=CONSTRUCT,
                    columns=[f"PC{i+1}" for i in range(X.shape[1])])
log('\nLoadings (first 4 components):')
log(load.iloc[:, :4].round(3).to_string())
log(f'\nExplained variance: {np.round(pca.explained_variance_ratio_[:4], 3).tolist()}')

for i in range(4):
    d[f"PC{i+1}"] = PCs[:, i]

rows = []
for dom, vs in DOMAINS.items():
    for v in vs:
        if v not in d.columns:
            continue
        s = d[v]
        if s.notna().sum() < 30:
            continue
        for i in range(4):
            m = s.notna()
            rho, pval = stats.spearmanr(d.loc[m, f"PC{i+1}"], s[m])
            rows.append({'domain': dom, 'variable': v, 'axis': f"PC{i+1}",
                         "n": int(m.sum()), "rho": rho, "p": pval})
cont = pd.DataFrame(rows)
cont["p_BH"] = multipletests(cont.p, method="fdr_bh")[1]
cont.to_csv(os.path.join(OUT, "04_continuous_axes_correlations.csv"), index=False)

log('\nSignificant correlations (BH-FDR < 0.05), ordered by |rho|:')
sig = cont[cont.p_BH < 0.05].reindex(
    cont[cont.p_BH < 0.05].rho.abs().sort_values(ascending=False).index)
log(sig.round(4).to_string(index=False))
log(f'\n  total tests: {len(cont)}; significant after BH: {len(sig)}')
log(f'  maximum |rho|: {cont.rho.abs().max():.3f}')

d.to_csv(os.path.join(OUT, "analysis_with_clusters.csv"), index=False)
with open(os.path.join(OUT, "04_characterization_report.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/04_characterization_report.txt, analysis_with_clusters.csv')
