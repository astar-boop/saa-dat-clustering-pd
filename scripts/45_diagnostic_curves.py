"""
45 - Complete diagnostic curves for k = 2..10 (request of the reviewer).

The original report gives in full only the curves of bootstrap Jaccard, PAC and
ARI. For gap, silhouette and prediction strength the text gave a single value
(gap: k = 1 and k = 2; silhouette and PS: k = 2 only). The complete curves were
NOT lost: they were already in the outputs of scripts 03 (24h) and 08 (150h), but
calibrated against the simple Gaussian null (mean and covariance of the data).

The reviewer asks for the curves against the **Gaussian copula null** — the
reference of script 25, which preserves the empirical marginals as well as the
dependence structure — and for both cohorts. That null had been
evaluated only at k = 2. Here it is extended to the whole grid k = 2..10.

For each cohort (24h n=687; 150h n=473) and each statistic:
  - observed curve, with the same estimators and seeds as scripts 03 / 08
    (the exact reproduction of the CSVs already deposited is verified);
  - copula null distribution: B realisations, median and 95% CI for each k,
    Monte Carlo p for each k;
  - curve of the single Gaussian null already published (for continuity with
    Figure 5); for the 150h the gap on the null had been computed but not saved,
    and is recomputed here with the same seed.
  - distribution of the k selected by the 1-SE criterion under the null.

Output: outputs/45_*.csv, outputs/45_diagnostic_curves.txt
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

from _cohorts import labels
COH24, COH150 = labels(OUT)
KRANGE = list(range(2, 11))
SEED = 20260804          # seed of scripts 03 and 08 (observed data and Gaussian null)
SEED_NULL = 20260810     # seed of the copula realisations
B_GAP = 200              # uniform references internal to the gap statistic
N_SPLIT_PS = 50          # splits of the prediction strength
N_JOBS = int(os.environ.get("N_JOBS", "12"))
B24 = int(os.environ.get("B24", "500"))
B150 = int(os.environ.get("B150", "300"))

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# ---------------------------------------------------------------------------
# estimators: identical to scripts 03 / 08 / 25
# ---------------------------------------------------------------------------
def wk(Xm, lab):
    """Pooled within-cluster dispersion."""
    tot = 0.0
    for c in np.unique(lab):
        pts = Xm[lab == c]
        if len(pts) > 1:
            tot += ((pts - pts.mean(0)) ** 2).sum()
    return tot


def gap_curve(Xm, B=B_GAP, seed=SEED):
    """Tibshirani's gap statistic, uniform reference in the PCA-aligned box."""
    r = np.random.default_rng(seed)
    ks = [1] + KRANGE
    pca = PCA().fit(Xm)
    Xp = pca.transform(Xm)
    lo, hi = Xp.min(0), Xp.max(0)
    gap, sk = [], []
    for k in ks:
        lab = (np.zeros(len(Xm), int) if k == 1
               else KMeans(k, n_init=50, random_state=seed).fit_predict(Xm))
        lw = np.log(wk(Xm, lab))
        bs = []
        for _ in range(B):
            Zb = pca.inverse_transform(r.uniform(lo, hi, size=Xp.shape))
            lb = (np.zeros(len(Zb), int) if k == 1
                  else KMeans(k, n_init=10,
                              random_state=int(r.integers(1e6))).fit_predict(Zb))
            bs.append(np.log(wk(Zb, lb)))
        bs = np.array(bs)
        gap.append(bs.mean() - lw)
        sk.append(bs.std() * np.sqrt(1 + 1 / B))
    gap, sk = np.array(gap), np.array(sk)
    khat = next((ks[i] for i in range(len(ks) - 1)
                 if gap[i] >= gap[i + 1] - sk[i + 1]), None)
    return pd.DataFrame({"k": ks, "gap": gap, "s_k": sk}), khat


def silhouette_curve(Xm, seed=SEED):
    """Silhouette of k-means (n_init = 50), k = 2..10."""
    out = []
    for k in KRANGE:
        lab = KMeans(k, n_init=50, random_state=seed).fit_predict(Xm)
        out.append({"k": k, "silhouette": silhouette_score(Xm, lab)})
    return pd.DataFrame(out)


def _ps_one(Xm, k, a, b, s1, s2):
    """Prediction strength on a single split (a = training, b = test)."""
    Xa, Xb = Xm[a], Xm[b]
    ka = KMeans(k, n_init=20, random_state=s1).fit(Xa)
    lb = KMeans(k, n_init=20, random_state=s2).fit_predict(Xb)
    pred = np.argmin(cdist(Xb, ka.cluster_centers_), axis=1)
    worst = 1.0
    for c in range(k):
        m = lb == c
        nc = m.sum()
        if nc <= 1:
            return 0.0
        cnt = np.bincount(pred[m], minlength=k)
        worst = min(worst, (cnt * (cnt - 1)).sum() / (nc * (nc - 1)))
    return worst


def ps_curve(Xm, n_rep=N_SPLIT_PS, seed=SEED, order="split_outer"):
    """Tibshirani-Walther prediction strength, k = 2..10.

    `order` replicates the order in which the two original scripts consume the
    random generator, so that the observed curves reproduce exactly the CSVs
    already deposited: "split_outer" = script 03 (24h), "k_outer" = script 08 (150h).
    The two variants differ only in the allocation of the splits, not in the
    estimator.
    """
    r = np.random.default_rng(seed)
    res = {k: [] for k in KRANGE}
    if order == "split_outer":
        for _ in range(n_rep):
            perm = r.permutation(len(Xm))
            a, b = perm[: len(Xm) // 2], perm[len(Xm) // 2:]
            for k in KRANGE:
                s1, s2 = int(r.integers(1e6)), int(r.integers(1e6))
                res[k].append(_ps_one(Xm, k, a, b, s1, s2))
    elif order == "k_outer":
        for k in KRANGE:
            for _ in range(n_rep):
                perm = r.permutation(len(Xm))
                a, b = perm[: len(Xm) // 2], perm[len(Xm) // 2:]
                s1, s2 = int(r.integers(1e6)), int(r.integers(1e6))
                res[k].append(_ps_one(Xm, k, a, b, s1, s2))
    else:
        raise ValueError(order)
    return pd.DataFrame({"k": KRANGE,
                         "PS": [np.mean(res[k]) for k in KRANGE],
                         "PS_sd": [np.std(res[k]) for k in KRANGE]})


def copula_null(X, rng):
    """Unimodal Gaussian copula null: empirical marginals + dependence."""
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
    """Simple Gaussian null of scripts 03 / 08 (same mean and covariance)."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), size=len(X))


# ---------------------------------------------------------------------------
# one null realisation -> the three curves
# ---------------------------------------------------------------------------
def one_null(X, b, ps_order):
    rng = np.random.default_rng([SEED_NULL, b])
    Y = copula_null(X, rng)
    s = int(rng.integers(1e9))
    g, khat = gap_curve(Y, seed=s)
    sil = silhouette_curve(Y, seed=s)
    ps = ps_curve(Y, seed=s, order=ps_order)
    return (g.gap.to_numpy(), sil.silhouette.to_numpy(), ps.PS.to_numpy(),
            -1 if khat is None else khat)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_cohorts():
    X24 = np.load(os.path.join(OUT, "features.npz"))["F_primary"]
    d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
    d150 = d150[d150.complete]
    F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE", "dat_putamen_low",
            "dat_caudate_mean", "age_bl"]
    M = d150[F150].to_numpy(float).copy()
    for j in range(4):
        M[:, j] = np.log(M[:, j])
    X150 = (M - M.mean(0)) / M.std(0)
    return X24, X150


def main():
    t0 = time.time()
    X24, X150 = load_cohorts()
    log("=" * 78)
    log('45 — COMPLETE DIAGNOSTIC CURVES k = 2..10, OBSERVED vs COPULA NULL')
    log("=" * 78)
    log(f"24h : n = {X24.shape[0]}, p = {X24.shape[1]}   |  "
        f"150h: n = {X150.shape[0]}, p = {X150.shape[1]}")
    log(f'Realisations of the null: B(24h) = {B24}, B(150h) = {B150}; n_jobs = {N_JOBS}')
    log(f'Gap: {B_GAP} internal uniform references. PS: {N_SPLIT_PS} splits.')

    # the order of consumption of the generator replicates the original script of
    # each cohort (03 for the 24h, 08 for the 150h): it serves only reproducibility
    cohorts = [(COH24, X24, B24, "split_outer"),
               (COH150, X150, B150, "k_outer")]
    long_rows, khat_rows = [], []

    for tag, X, B, ps_order in cohorts:
        log("\n" + "-" * 78)
        log(f"{tag}   [prediction strength: order '{ps_order}']")
        log("-" * 78)

        # --- observed ------------------------------------------------------
        g_obs, khat_obs = gap_curve(X, seed=SEED)
        sil_obs = silhouette_curve(X, seed=SEED)
        ps_obs = ps_curve(X, seed=SEED, order=ps_order)
        log(f'  observed computed ({time.time() - t0:.0f}s). k selected by the 1-SE criterion on the '
            f'gap: {khat_obs}')

        # --- single Gaussian null (that of Figure 5) -----------------------
        Yg = gaussian_null(X, seed=SEED)
        g_gau, khat_gau = gap_curve(Yg, seed=SEED)
        sil_gau = silhouette_curve(Yg, seed=SEED)
        ps_gau = ps_curve(Yg, seed=SEED, order=ps_order)
        log(f'  single Gaussian null computed ({time.time() - t0:.0f}s). 1-SE k: {khat_gau}')

        # --- check of reproduction of the deposited outputs ----------------
        def chk(name, a, b_):
            d = float(np.max(np.abs(np.asarray(a) - np.asarray(b_))))
            log(f'    check {name:34s} max discrepancy = {d:.2e}')
            return d

        try:
            if tag.startswith("24h"):
                o = pd.read_csv(os.path.join(OUT, "03_gap_observed.csv"))
                chk('observed gap vs 03_gap_observed', g_obs.gap, o.gap)
                n_ = pd.read_csv(os.path.join(OUT, "03_gap_null.csv"))
                chk('Gauss. null gap vs 03_gap_null', g_gau.gap, n_.gap)
                ii = pd.read_csv(os.path.join(OUT, "03_internal_indices.csv"))
                chk('obs. silhouette vs 03_internal', sil_obs.silhouette,
                    ii[(ii.data == 'observed') & (ii.algorithm == "kmeans")].silhouette)
                chk('null silhouette vs 03_internal', sil_gau.silhouette,
                    ii[(ii.data == 'nullref') & (ii.algorithm == "kmeans")].silhouette)
                pp = pd.read_csv(os.path.join(OUT, "03_prediction_strength.csv"))
                chk('PS obs. vs 03_prediction_strength', ps_obs.PS, pp.PS_mean_obsv)
                chk('PS null vs 03_prediction_strength', ps_gau.PS, pp.PS_mean_null)
            else:
                o = pd.read_csv(os.path.join(OUT, "08_gap_150h.csv"))
                chk('observed gap vs 08_gap_150h', g_obs.gap, o.gap)
                st = pd.read_csv(os.path.join(OUT, "08_stability_150h.csv"))
                chk('silhouette obs. vs 08_stability', sil_obs.silhouette,
                    st[st.data == 'observed'].silhouette)
                chk('silhouette null vs 08_stability', sil_gau.silhouette,
                    st[st.data == 'nullref'].silhouette)
                pp = pd.read_csv(os.path.join(OUT, "08_predstrength_150h.csv"))
                chk('obs. PS vs 08_predstrength', ps_obs.PS, pp.PS_obsv)
                chk('null PS vs 08_predstrength', ps_gau.PS, pp.PS_null)
        except Exception as e:  # the check must not block the computation
            log(f'    [check not completed: {e}]')

        # --- copula null: B realisations -----------------------------------
        log(f'  copula null: {B} realisations on {N_JOBS} processes...')
        res = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(one_null)(X, b, ps_order) for b in range(B))
        G = np.vstack([r[0] for r in res])      # B x 10  (k = 1..10)
        S = np.vstack([r[1] for r in res])      # B x 9   (k = 2..10)
        P = np.vstack([r[2] for r in res])      # B x 9
        KH = np.array([r[3] for r in res])
        log(f'  done ({time.time() - t0:.0f}s)')

        # The level of the gap depends on the global shape of the cloud, not on the
        # presence of groups: the informative quantity is the increase relative
        # to k = 1, which is positive only if splitting improves compactness
        # more than it does on a unimodal null.
        ks_gap = [1] + KRANGE
        blocks = [("gap", ks_gap, g_obs.gap.to_numpy(), G,
                   g_gau.gap.to_numpy(), g_obs.s_k.to_numpy()),
                  ('gap_increment_over_k1', ks_gap,
                   g_obs.gap.to_numpy() - g_obs.gap.to_numpy()[0],
                   G - G[:, [0]],
                   g_gau.gap.to_numpy() - g_gau.gap.to_numpy()[0], None),
                  ("silhouette", KRANGE, sil_obs.silhouette.to_numpy(), S,
                   sil_gau.silhouette.to_numpy(), None),
                  ("prediction_strength", KRANGE, ps_obs.PS.to_numpy(), P,
                   ps_gau.PS.to_numpy(), ps_obs.PS_sd.to_numpy() / np.sqrt(N_SPLIT_PS))]

        for stat, ks, obs, NUL, gau, se in blocks:
            for i, k in enumerate(ks):
                v = NUL[:, i]
                p = (np.sum(v >= obs[i]) + 1) / (len(v) + 1)   # high = more structure
                long_rows.append({
                    'cohort': tag, 'statistic': stat, "k": k,
                    'observed': obs[i],
                    'SE_observed': (np.nan if se is None else se[i]),
                    'null_copula_median': float(np.median(v)),
                    "nullo_copula_IC2.5": float(np.percentile(v, 2.5)),
                    "nullo_copula_IC97.5": float(np.percentile(v, 97.5)),
                    "p_montecarlo": p,
                    'null_gaussian_single': gau[i],
                    "B": len(v)})

        u, c = np.unique(KH, return_counts=True)
        khat_rows.append({'cohort': tag, 'k_1SE_observed': khat_obs,
                          'k_1SE_null_gaussian': khat_gau,
                          "nullo_copula_k1_%": 100 * float((KH == 1).mean()),
                          'null_copula_distribution_k':
                              "; ".join(f"k={int(a)}: {b}" for a, b in zip(u, c)),
                          "B": len(KH)})
        log(f'  1-SE criterion on the copula null: k = 1 in {100 * (KH == 1).mean():.1f}% of the '
            f'{len(KH)} realisations')

    long = pd.DataFrame(long_rows)
    long.to_csv(os.path.join(OUT, "45_diagnostic_curves.csv"), index=False)
    pd.DataFrame(khat_rows).to_csv(os.path.join(OUT, "45_k_selected.csv"),
                                   index=False)

    # --- CSV per statistic, requested format (rows = series, columns = k) --
    names = {"gap": "45_gap_curves.csv",
            'gap_increment_over_k1': "45_gap_increment_curves.csv",
            "silhouette": "45_silhouette_curves.csv",
            "prediction_strength": "45_prediction_strength_curves.csv"}
    for stat, fn in names.items():
        sub = long[long.statistic == stat]
        rows = []
        for tag in sub.cohort.unique():
            s = sub[sub.cohort == tag].set_index("k")
            for label, col in [("observed", 'observed'),
                                   ("copula null: median", 'null_copula_median'),
                                   ("copula null: 2.5th pct", "nullo_copula_IC2.5"),
                                   ("copula null: 97.5th pct", "nullo_copula_IC97.5"),
                                   ("Monte Carlo p", "p_montecarlo"),
                                   ("Gaussian null (single draw)",
                                    'null_gaussian_single')]:
                r = {"cohort": tag, "series": label}
                r.update({f"k={int(k)}": s.loc[k, col] for k in s.index})
                rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, fn), index=False)
        log(f'\nSaved: outputs/{fn}')

    log("\n" + "=" * 78)
    log('SUMMARY')
    log("=" * 78)
    for stat in names:
        log(f"\n--- {stat} ---")
        sub = long[long.statistic == stat]
        log(sub[['cohort', "k", 'observed', 'null_copula_median',
                 "nullo_copula_IC2.5", "nullo_copula_IC97.5", "p_montecarlo",
                 'null_gaussian_single']].round(4).to_string(index=False))
    log("\n" + pd.DataFrame(khat_rows).to_string(index=False))
    log(f'\nTotal time: {time.time() - t0:.0f}s')

    with open(os.path.join(OUT, "45_diagnostic_curves.txt"), "w") as fh:
        fh.write("\n".join(rep))
    log('Saved: outputs/45_diagnostic_curves.txt')


if __name__ == "__main__":
    main()
