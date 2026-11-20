"""
66 - Additional analyses requested by the external review.

Four blocks, all run on the sporadic population.

  A  Replication free of overlap. The 150 h cohort shares 45
     participants with the primary cohort: the review asks for the battery
     to be repeated after removing them, to separate the replication from
     the correlation between the two samples.

  B  Sensitivity to the run effect. The assay run explains a non-negligible
     share of the variance of the amplitude parameters, which dominate the
     first component of the clustering space. Technical variance can
     mask real structure, not only create spurious structure: the battery
     must therefore be repeated on kinetic parameters residualised by run.

  C  Attenuation of the MoCA coefficient. The "mediation" block of
     script 15 used to fit the models with `lbfgs` only, without checking
     convergence: the base model is precisely the one that _fit.py had
     shown not to converge. The attenuation percentages were therefore
     computed against an estimate that was not an estimate. Here everything
     is redone with the cascade of optimisers and the convergence flag is
     reported.

  D  Accounting of samples, participants and wells. A single table,
     derived from the code, with one column for each unit of analysis.

Output: outputs/66_*.csv, outputs/66_external_review.txt
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import diptest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import fit_mixed
from _paths import DATA
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA_FILE = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")

SEED = 20260805
N_PROJ = 200          # random projections for the multimodality count
N_SPLIT_PS = 50       # splits for the prediction strength
B_NULL = 200          # realisations of the copula null for each comparison
T0 = time.time()
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# ===========================================================================
# battery: the same functions as script 25, reproduced here because that
# module executes at import time
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


def hopkins(X, rng, m_frac=0.10, n_rep=30):
    n, p = X.shape
    m = max(5, int(m_frac * n))
    lo, hi = X.min(0), X.max(0)
    tree = cKDTree(X)
    vals = []
    for _ in range(n_rep):
        idx = rng.choice(n, m, replace=False)
        w = tree.query(X[idx], k=2)[0][:, 1]
        u = tree.query(rng.uniform(lo, hi, size=(m, p)), k=1)[0]
        vals.append(u.sum() / (u.sum() + w.sum()))
    return float(np.mean(vals))


def n_multimodal(X, rng, n_proj=N_PROJ):
    c, pmin = 0, 1.0
    for _ in range(n_proj):
        a = rng.normal(size=X.shape[1]); a /= np.linalg.norm(a)
        p = diptest.diptest(X @ a)[1]
        c += p < 0.05
        pmin = min(pmin, p)
    return c, pmin


def ps_k2(X, rng, n_rep=N_SPLIT_PS):
    vals = []
    for _ in range(n_rep):
        perm = rng.permutation(len(X))
        a, b = perm[: len(X) // 2], perm[len(X) // 2:]
        ka = KMeans(2, n_init=10, random_state=int(rng.integers(1e6))).fit(X[a])
        lb = KMeans(2, n_init=10, random_state=int(rng.integers(1e6))).fit_predict(X[b])
        pred = np.argmin(cdist(X[b], ka.cluster_centers_), axis=1)
        worst = 1.0
        for c in range(2):
            m = lb == c
            nc = m.sum()
            if nc <= 1:
                worst = 0.0
                continue
            cnt = np.bincount(pred[m], minlength=2)
            worst = min(worst, (cnt * (cnt - 1)).sum() / (nc * (nc - 1)))
        vals.append(worst)
    return float(np.mean(vals))


def jaccard_k2(X, rng, Bb=25):
    ref = KMeans(2, n_init=15, random_state=SEED).fit_predict(X)
    best = {c: [] for c in range(2)}
    for _ in range(Bb):
        ib = rng.choice(len(X), len(X), replace=True)
        km = KMeans(2, n_init=5, random_state=int(rng.integers(1e6))).fit(X[ib])
        lab = np.argmin(cdist(X, km.cluster_centers_), axis=1)
        for c in range(2):
            rs = set(np.where(ref == c)[0])
            best[c].append(max(
                (len(rs & set(np.where(lab == cb)[0])) /
                 len(rs | set(np.where(lab == cb)[0])) for cb in range(2)),
                default=0.0))
    return float(np.mean([np.mean(best[c]) for c in range(2)]))


def ari_algos(X):
    labs = {
        "kmeans": KMeans(2, n_init=20, random_state=SEED).fit_predict(X),
        "ward": AgglomerativeClustering(2, linkage="ward").fit_predict(X),
        "average": AgglomerativeClustering(2, linkage="average").fit_predict(X),
        "gmm": GaussianMixture(2, covariance_type="full", n_init=5,
                               random_state=SEED).fit_predict(X)}
    ks = list(labs)
    return float(np.mean([adjusted_rand_score(labs[ks[i]], labs[ks[j]])
                          for i in range(4) for j in range(i + 1, 4)]))


def battery(X, rng):
    nm, pmin = n_multimodal(X, rng)
    return {"hopkins": hopkins(X, rng),
            "silhouette_k2": silhouette_score(
                X, KMeans(2, n_init=15, random_state=SEED).fit_predict(X)),
            'n_projections_multimodal': nm,
            "dip_pmin_bonf": min(1.0, pmin * N_PROJ),
            "PS_k2": ps_k2(X, rng),
            "jaccard_k2": jaccard_k2(X, rng),
            'ARI_algorithms_k2': ari_algos(X)}


HIGHER = {"hopkins": True, "silhouette_k2": True,
          'n_projections_multimodal': True, "dip_pmin_bonf": False,
          "PS_k2": True, "jaccard_k2": True, 'ARI_algorithms_k2': True}

LABEL = {"hopkins": "Hopkins", "silhouette_k2": "Silhouette (k = 2)",
             'n_projections_multimodal': "Multimodal projections (of 200)",
             "dip_pmin_bonf": "Minimum dip (Bonferroni)",
             "PS_k2": "Prediction strength (k = 2)",
             "jaccard_k2": "Jaccard (k = 2)",
             'ARI_algorithms_k2': "Between-algorithm ARI (k = 2)"}


def against_null(tag, X, B=B_NULL):
    """Observed battery and copula null distribution, with Monte Carlo p."""
    rng = np.random.default_rng(SEED)
    obsv = battery(X, rng)
    nulls = []
    for b in range(B):
        nulls.append(battery(copula_null(X, rng), rng))
        if (b + 1) % 50 == 0:
            log(f'    [{tag}] null {b + 1}/{B}  ({time.time() - T0:.0f}s)')
    N = pd.DataFrame(nulls)
    out = []
    for stat, obs in obsv.items():
        v = N[stat].to_numpy(float)
        p = ((np.sum(v >= obs) if HIGHER[stat] else np.sum(v <= obs)) + 1) / (len(v) + 1)
        out.append({'analysis': tag, 'statistic': stat,
                    'label': LABEL[stat], "n": len(X),
                    'observed': obs, 'null_median': float(np.median(v)),
                    "nullo_IC2.5": float(np.percentile(v, 2.5)),
                    "nullo_IC97.5": float(np.percentile(v, 97.5)),
                    "p_montecarlo": p, "B": len(v)})
    return pd.DataFrame(out)


def printout(df):
    for tag in df.analysis.unique():
        s = df[df.analysis == tag]
        log(f"\n--- {tag}  (n = {s.n.iloc[0]}, B = {s.B.iloc[0]}) ---")
        log(s[['label', 'observed', 'null_median', "nullo_IC2.5",
               "nullo_IC97.5", "p_montecarlo"]].round(4).to_string(index=False))
        sig = s[s.p_montecarlo < 0.05]
        log(f'  exceed the null (p < 0.05): {len(sig)} out of {len(s)}'
            + (f' -> {sig.label.tolist()}' if len(sig) else ""))


# ===========================================================================
# data
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

overlapping = sorted(set(PAT24.tolist()) & set(PAT150.tolist()))
log("=" * 78)
log('66 — ADDITIONAL ANALYSES FOR THE EXTERNAL REVIEW')
log("=" * 78)
log(f'24h cohort  : n = {len(X24)}')
log(f'150h cohort : n = {len(X150)}')
log(f'shared subjects: {len(overlapping)}')

# ===========================================================================
# A — replication free of overlap
# ===========================================================================
log("\n" + "=" * 78)
log('A — REPLICATION FREE OF OVERLAP (150 h, shared participants excluded)')
log("=" * 78)
mask = ~np.isin(PAT150, np.asarray(overlapping))
M150_no = M150[mask]
X150_no = (M150_no - M150_no.mean(0)) / M150_no.std(0)
log(f"n after the exclusion: {X150_no.shape[0]}")

res_a = pd.concat([against_null('150h full', X150),
                   against_null('150h without overlapping', X150_no)],
                  ignore_index=True)
printout(res_a)
res_a.to_csv(os.path.join(OUT, "66_replication_without_overlapping.csv"), index=False)

# ===========================================================================
# B — sensitivity to the run effect
# ===========================================================================
log("\n" + "=" * 78)
log("B — SENSITIVITY TO THE RUN EFFECT")
log("=" * 78)
saa = pd.read_csv(SAA_FILE)


def info_run(projectid, patno):
    s = saa[(saa.PROJECTID == projectid) & (saa.CLINICAL_EVENT == "BL")]
    s = s.pipe(row_of_kinetics, OUT).drop_duplicates("PATNO").set_index("PATNO")
    s = s.reindex(patno)
    run = s.RUNDATE.astype(str).to_numpy()
    instr = s.InstrumentRep1.astype(str).to_numpy()
    return run, instr


def share_variance(y, g):
    """Share of variance between groups (eta squared), on the analysis scale."""
    g = np.asarray(g)
    tot = np.sum((y - y.mean()) ** 2)
    between = sum(len(y[g == u]) * (y[g == u].mean() - y.mean()) ** 2
              for u in np.unique(g))
    return float(between / tot) if tot > 0 else np.nan


def residualise(M, run, n_min=5):
    """Centres each kinetic column on the mean of its own run.

    Runs with fewer than n_min samples are gathered into a single class: with
    one or two samples the run mean coincides with the datum and the residual
    would be zero by construction.
    """
    run = pd.Series(run).astype(str)
    little = run.value_counts()
    run = run.where(run.map(little) >= n_min, 'other').to_numpy()
    R = M.copy()
    for j in range(M.shape[1]):
        y = M[:, j]
        med = pd.Series(y).groupby(run).transform("mean").to_numpy()
        R[:, j] = y - med + y.mean()
    return R, run


NAMES_K24 = ["log Fmax", "log TTT", "log AUC", "log SLOPEmax"]
NAMES_K150 = ["log Fmax", "log T50", "log AUC", "log SLOPE"]

# the primary set of the 24h is already standardised: the raw logs are recovered
cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
cc = cc.set_index("PATNO").reindex(PAT24).reset_index()
M24 = np.column_stack([
    np.log(cc["saa_Fmax"].to_numpy(float)),
    np.log(cc["saa_TTT"].to_numpy(float)),
    np.log(cc["saa_AUC"].to_numpy(float)),
    np.log(cc["saa_SLOPEMax"].to_numpy(float)),
    cc["dat_putamen_low"].to_numpy(float),
    cc["dat_caudate_mean"].to_numpy(float),
    cc["age_bl"].to_numpy(float)])
# check: the standardisation of M24 must coincide with F_primary
X24_ric = (M24 - M24.mean(0)) / M24.std(0)
departure = float(np.nanmax(np.abs(X24_ric - X24)))
log(f'check of the reconstruction of the 24h primary set: maximum difference = {departure:.2e}')
assert departure < 1e-8, 'the reconstructed primary set does not coincide with features.npz'

rows_var, blocks = [], []
for lab, proj, M, patno, names in [
        ("24h (n=595)", 237, M24, PAT24, NAMES_K24),
        ("150h (n=333)", 155, M150, PAT150, NAMES_K150)]:
    run, instr = info_run(proj, patno)
    log(f'\n{lab}: {pd.Series(run).nunique()} distinct runs, {pd.Series(instr).nunique()} instruments')
    for j, name in enumerate(names):
        rows_var.append({'cohort': lab, 'parameter': name,
                          'share_variance_run': share_variance(M[:, j], run),
                          'share_variance_instrument': share_variance(M[:, j], instr),
                          "n_run": int(pd.Series(run).nunique())})
    Rk, run_used = residualise(M[:, :4], run)
    M_adj = M.copy()
    M_adj[:, :4] = Rk
    X_adj = (M_adj - M_adj.mean(0)) / M_adj.std(0)
    blocks.append((f'{lab} residualised for run', X_adj))
    for j, name in enumerate(names):
        log(f'   {name:12s} share of variance of the run {share_variance(M[:, j], run):.3f} -> '
            f'{share_variance(Rk[:, j], run_used):.3f} after residualisation')

var = pd.DataFrame(rows_var)
var.to_csv(os.path.join(OUT, "66_variance_run.csv"), index=False)
log('\nShares of variance attributable to the run and to the instrument:')
log(var.round(4).to_string(index=False))

res_b = pd.concat([against_null(tag, X) for tag, X in blocks], ignore_index=True)
printout(res_b)
res_b.to_csv(os.path.join(OUT, "66_run_adjusted.csv"), index=False)

# ===========================================================================
# C — attenuation of the MoCA coefficient, with convergence checked
# ===========================================================================
log("\n" + "=" * 78)
log('C — ATTENUATION OF THE MoCA COEFFICIENT (convergent models)')
log("=" * 78)
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
base = d150[["PATNO", "k_T50", "age_bl", "SEX", "duration_yrs", "EDUCYRS"]].copy()
base["T50"] = stats.zscore(np.log(base.k_T50))
base = base.rename(columns={"duration_yrs": "dur_bl"})
COV = ["hvlt_immediaterecall", "SDMTOTAL", "VLTANIM", "bjlot"]
L = cur[cur.PATNO.isin(set(base.PATNO))][["PATNO", "YEAR", "moca"] + COV].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(
    base[["PATNO", "T50", "age_bl", "SEX", "dur_bl", "EDUCYRS"]],
    on="PATNO", how="left")

rows, b0 = [], None
for extra in [None] + COV:
    cols = ["PATNO", "years", "moca", "T50", "age_bl", "SEX", "dur_bl", "EDUCYRS"]
    f = "moca ~ T50*years + age_bl + SEX + dur_bl + EDUCYRS"
    if extra:
        cols = cols + [extra]
        f += f" + {extra}"
    m = L[cols].dropna()
    r, method, _ = fit_mixed(f, m, m.PATNO)
    if r is None:
        log(f'  [{extra}] no optimiser converges: excluded')
        continue
    t = "T50:years" if "T50:years" in r.params.index else "years:T50"
    b = float(r.params[t])
    if b0 is None:
        b0 = b
    rows.append({'adjustment': extra or 'none (base)',
                  'n_subj': int(m.PATNO.nunique()), 'n_obsv': int(len(m)),
                  "beta": b, "se": float(r.bse[t]),
                  'CI95_inf': b - 1.96 * float(r.bse[t]),
                  'CI95_upp': b + 1.96 * float(r.bse[t]),
                  "p": float(r.pvalues[t]),
                  "variazione_vs_base_%": 100 * (b - b0) / b0,
                  'convergence': bool(r.converged), 'optimiser': method})
att = pd.DataFrame(rows)
att.to_csv(os.path.join(OUT, "66_moca_attenuation.csv"), index=False)
log(att.round(4).to_string(index=False))
log('\nThe corresponding block of script 15 estimated with `lbfgs` alone:')
log('the base model did not converge (beta 0.0573; p 0.384) and the attenuation')
log('percentages referred to that estimate.')

# ===========================================================================
# D — accounting of samples, participants and wells
# ===========================================================================
log("\n" + "=" * 78)
log('D — SAMPLES, SUBJECTS, WELLS')
log("=" * 78)
K24 = ["Fmax", "TTT", "AUC", "TSmax", "SLOPEMax"]
K150 = ["Fmax", "TTT", "AUC", "T50", "SLOPE"]


def count_wells(sub, params, proto):
    tot = ab = 0
    hi = 0.0
    for f in params:
        cols = [f"{f}_{proto}_Rep{i}" for i in (1, 2, 3)]
        v = sub[cols].apply(pd.to_numeric, errors="coerce")
        med = v.median(axis=1)
        for c in cols:
            x = v[c]
            ok = x.notna() & med.notna() & (med != 0)
            tot += int(ok.sum())
            r = x[ok] / med[ok]
            ab += int(((r > 5) | (r < 0.2)).sum())
            if len(r):
                hi = max(hi, float(r.max()))
    return tot, ab, hi


rows_stream = []
s24 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")]
s150 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 155) & (saa.CLINICAL_EVENT == "BL")]
pos24 = s24[s24.SAA_Status == "Positive"]
t1_24 = s24[(s24.SAA_Status == "Positive") & (s24.SAA_Type == "Type1")]
pos150 = s150[s150.SAA_Status == "Positive"]

for lab, sub, params, proto in [
        ('24h, PD at baseline', s24, K24, "24h"),
        ('24h, positives (all types)', pos24, K24, "24h"),
        ('24h, type 1 positive', t1_24, K24, "24h"),
        ('150h, PD at baseline', s150, K150, "150h"),
        ('150h, positives', pos150, K150, "150h")]:
    tot, ab, hi = count_wells(sub, params, proto)
    rows_stream.append({'dataset': lab, 'samples': len(sub),
                         'subjects': int(sub.PATNO.nunique()),
                         'wells_physical': len(sub) * 3,
                         'measurements_parameter_well': tot,
                         'measurements_aberrant': ab,
                         "quota_aberranti_%": round(100 * ab / tot, 2) if tot else np.nan,
                         'max_ratio_at_median': round(hi, 1)})
stream = pd.DataFrame(rows_stream)
stream.to_csv(os.path.join(OUT, "66_flow_wells.csv"), index=False)
log(stream.to_string(index=False))

# outcomes at baseline, to reconcile the 244 negatives cited in the manuscript
log('\nOutcomes of the assay at the baseline visit (samples):')
log(pd.crosstab(s24.SAA_Status, s24.SAA_Type, dropna=False).to_string())
every24 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237)]
log(f"\n24h, ALL visits: {len(every24)} samples, {every24.PATNO.nunique()} subjects; negatives = "
    f"{int((every24.SAA_Status == 'Negative').sum())}")
log(f"24h, at baseline only: negatives = {int((s24.SAA_Status == 'Negative').sum())} samples")

# population of the reliability analysis
icc = pd.read_csv(os.path.join(OUT, "21_reliability_kinetics.csv"))
n24_icc = int(icc[icc.protocol == "24h"].n_subjects.max())
n150_icc = int(icc[icc.protocol == "150h"].n_subjects.max())
measurements = int(icc.n_subjects.sum() * 3)
log(f'\nReliability: {n24_icc} samples in the 24h and {n150_icc} in the 150h; {measurements} '
    f'parameter-well measurements ({icc.n_subjects.sum()} parameter-sample pairs x 3 replicates).')

with open(os.path.join(OUT, "66_external_review.txt"), "w") as fh:
    fh.write("\n".join(rep))
log(f'\nSaved: outputs/66_*.csv and 66_external_review.txt  ({time.time() - T0:.0f}s)')
