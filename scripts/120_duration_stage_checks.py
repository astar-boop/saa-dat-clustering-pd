"""
120 - Checks on the shift observed in 118 (duration and stage).

Why it exists. In 118, in the analysis with duration and stage removed (V1),
the 24 h cohort lies closer to p = 0.05 than in the analysis on the unmodified
variables (V0): prediction strength p 0.090 against 0.210, Jaccard 0.084 against
0.521, silhouette at k = 2 p 0.010 against 0.033. The independent verifier
(revisione_R2_rapporti/verifiche_script/riverifica_118.txt) found that the
reading "removing duration and stage brings the criteria closer to the
threshold" does not hold, for four reasons, and measured them with code of its
own:
  (D2) V1 differs from V0 in two things at once: the residualisation and three
       fewer participants, those without stage; and a control with irrelevant
       covariates is missing;
  (D3) the first arm is judged on a single series of random splits;
  (N1) with 2000 realisations the p of the silhouette is an estimate with error.
This script makes those findings reproducible within the project. The rules
are written after reading the verifier's report: they are NOT blind to its
numbers, which serve as the benchmark.

What it computes (24 h cohort, except point 4).
  Sets: V0 (595, unmodified variables); V0r (the same variables on the 592
  with stage, re-standardised); S (592, stage only removed); V1 (592, duration
  and stage removed, the matrix of 118).
  1. Deterministic observed values for the four sets: k-means silhouette at
     k = 2 (constructor of 45) and ARI between the four algorithms
     (constructors of 25). For the ARI, V0 without three participants chosen
     at random (500 draws): it says whether removing ANY THREE PARTICIPANTS
     shifts the ARI.
  2. Prediction strength and Jaccard at k = 2 on 200 independent series of
     random splits, the same for the four sets (paired differences).
     For V0 and V1, against the nulls saved by 118: p at the mean of the
     series and proportion of series with p < 0.05 (that is, how often the
     first arm would be "met" by changing the series only).
  3. Control with irrelevant covariates: duration and stage are permuted
     TOGETHER among the 592 participants (500 permutations), the variables are
     residualised on the permuted covariates, and silhouette at k = 2,
     prediction strength and Jaccard are recomputed (one fixed series, the
     same for each permutation and for V1). The proportion of permutations
     with a value >= that of V1 computed in the same way is reported. If the
     proportion is not small, the shift of V1 cannot be distinguished from
     that of an arbitrary adjustment.
  4. Null of the silhouette extended to 10,000 realisations for V0 and V1, in
     both cohorts: same sequence as 118 (generator [SEED_NULL, b]);
     the first 2000 are those saved, and the first 5 are regenerated here and
     must coincide. p per k, Monte Carlo standard error, and Benjamini-Hochberg
     correction on the 18 tests.

What the script does NOT do: it does not estimate a null for V0r and for S (for
those it gives the observed values only); it does not extend the null of the
prediction strength; it does not address the calibration of the Monte Carlo p
values against a reference fitted to the same data (remark D1 of the verifier,
which concerns the whole rule of the manuscript and requires a separate
analysis).

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 -B scripts/120_duration_stage_checks.py
      N_PROCESSES=n   (default 10; the results do not depend on it)
Output: outputs/120_duration_stage_checks.txt, 120_observed.csv,
        120_series.csv, 120_covariates_permuted.csv, 120_silhouette_B10000.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util                                            # noqa: E402
import multiprocessing as mp                                     # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
import warnings                                                  # noqa: E402
import zlib                                                      # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "10")))


from _modules import load_script as _module                      # noqa: E402


M118 = _module("118_duration_stage.py", 'duration_118')
import _rule_functions                                                    # noqa: E402
C45 = M118.C45
K_CURVE = list(C45.KRANGE)

SEED = 20261113
N_SERIES = 200
BLOCK_SERIES = 25
N_PERM = 500
BLOCK_PERM = 25
N_THREE = 500
B_TOT = 10000
BLOCK_SIL = 250
ALPHA = 0.05
rep = []


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def sil_k2(X):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(X, KMeans(2, n_init=50, random_state=C45.SEED).fit_predict(X)))


def series_fixed(X):
    """PS and Jaccard at k = 2 on one fixed series of random splits."""
    g = np.random.default_rng([SEED, zlib.crc32(b"120|serie fissa")])
    return _rule_functions.ps_k2(X, g), _rule_functions.jaccard_k2(X, g)


def job_perm(t):
    """Permutations b0..b1-1 of the covariates among the participants."""
    X, D = t["X"], t["D"]
    out = []
    for b in range(t["b0"], t["b1"]):
        g = np.random.default_rng([SEED, zlib.crc32(b"120|permutazione"), b])
        Z, _ = M118.residuals(X, D[g.permutation(len(D))])
        ps, ja = series_fixed(Z)
        out.append(dict(b=b, silhouette_k2=sil_k2(Z), PS_k2=ps, jaccard_k2=ja))
    return {'kind': "perm", 'rows': out}


def job_three(t):
    """ARI between algorithms on V0 without three participants chosen at random."""
    R = t["R"]
    out = []
    for b in range(t["b0"], t["b1"]):
        g = np.random.default_rng([SEED, zlib.crc32(b"120|tre soggetti"), b])
        keep = np.sort(g.choice(len(R), len(R) - 3, replace=False))
        out.append(_rule_functions.ari_algos(M118.standardise(R[keep])))
    return {'kind': 'three', 'values': out}


def execute(t):
    if t['kind'] == "perm":
        return job_perm(t)
    if t['kind'] == 'three':
        return job_three(t)
    return _rule_functions.EXECUTORS[t['kind']](t)


def main():
    t0 = time.time()
    title('120 - CHECKS ON THE SHIFT OBSERVED IN 118')
    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy())
    R24 = c24[M118.F24].to_numpy(float).copy()
    R24[:, :4] = np.log(R24[:, :4])
    X0 = npz["F_primary"].astype(float)
    assert np.abs(M118.standardise(R24) - X0).max() < M118.TOL
    dur, hy = c24.duration_yrs.to_numpy(float), c24.hy.to_numpy(float)
    ok = ~np.isnan(hy)
    D = np.column_stack([np.log(dur[ok]), (hy[ok] >= 2).astype(float)])
    X0r = M118.standardise(X0[ok])
    XS, _ = M118.residuals(X0[ok], D[:, [1]])
    X1, _ = M118.residuals(X0[ok], D)
    saved = np.load(os.path.join(OUT, "118_nulls.npz"), allow_pickle=True)
    assert np.abs(saved['V1_24h_matrix'] - X1).max() < 1e-9, \
        'the V1 matrix rebuilt here does not match the one saved by 118'
    assert np.abs(saved['V0_24h_matrix'] - X0).max() < 1e-12
    # the residuals must be uncorrelated with the covariates and have mean 0, SD 1
    A = np.column_stack([np.ones(len(D)), D])
    assert np.abs(A.T @ X1).max() < 1e-8 and np.abs(X1.std(0) - 1).max() < 1e-12
    datasets = {"V0": X0, "V0r": X0r, "S": XS, "V1": X1}
    desc = {"V0": 'variables not modified (595)', "V0r": 'unmodified, only the 592 with stage',
            "S": 'stage alone removed (592)', "V1": 'duration and stage removed (592)'}
    log(f"Declared working directory: {('sporadic only' if ONLY else 'full cohort')}. Identity: V0 "
        f"and V1 match the matrices saved by 118; the residuals of V1 are uncorrelated with the "
        f"covariates. Subjects without stage: {int((~ok).sum())}.")

    tasks = []
    for name, X in datasets.items():
        for r0 in range(0, N_SERIES, BLOCK_SERIES):
            tasks.append(dict(kind='streams', branch=name, X=X, label='120|series common',
                                r0=r0, r1=min(r0 + BLOCK_SERIES, N_SERIES)))
    for b0 in range(0, N_PERM, BLOCK_PERM):
        tasks.append(dict(kind="perm", X=X0[ok], D=D, b0=b0, b1=min(b0 + BLOCK_PERM, N_PERM)))
    for b0 in range(0, N_THREE, 50):
        tasks.append(dict(kind='three', R=R24, b0=b0, b1=min(b0 + 50, N_THREE)))
    for var in ("V0", "V1"):
        for tag in ("24h", "150h"):
            M = saved[f'{var}_{tag}_matrix']
            tasks.append(dict(kind='curve_null', branch=f"{var}|{tag}|id", nullref="copula", X=M,
                                b0=0, b1=5, n_ps=0, ordering_ps=M118.ORDERING_PS[tag]))
            for b0 in range(M118.B_SIL, B_TOT, BLOCK_SIL):
                tasks.append(dict(kind='curve_null', branch=f"{var}|{tag}", nullref="copula", X=M,
                                    b0=b0, b1=min(b0 + BLOCK_SIL, B_TOT), n_ps=0,
                                    ordering_ps=M118.ORDERING_PS[tag]))
    log(f'{len(tasks)} tasks on {N_PROC} processes...')
    result = []
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        for i, r in enumerate(pool.imap_unordered(execute, tasks, chunksize=1)):
            result.append(r)
            if (i + 1) % 40 == 0 or i + 1 == len(tasks):
                print(f'    {i + 1}/{len(tasks)}  ({time.time() - t0:.0f} s)', flush=True)

    # ------------------------------------------------------------------ 1 ----
    title('1. DETERMINISTIC OBSERVED VALUES AT 24 H')
    rows = [dict(dataset=n, description=desc[n], n=len(X), silhouette_k2=sil_k2(X),
                  ARI_algorithms_k2=_rule_functions.ari_algos(X)) for n, X in datasets.items()]
    obsv = pd.DataFrame(rows)
    log(obsv.round(4).to_string(index=False))
    three = np.concatenate([r['values'] for r in result if r['kind'] == 'three'])
    assert len(three) == N_THREE
    a0 = float(obsv.loc[obsv.dataset == "V0", 'ARI_algorithms_k2'].iloc[0])
    a0r = float(obsv.loc[obsv.dataset == "V0r", 'ARI_algorithms_k2'].iloc[0])
    log(f'ARI between algorithms on V0 without three subjects chosen at random ({N_THREE} draws): '
        f'median {np.median(three):.3f}, 2.5-97.5% '
        f'{np.percentile(three, 2.5):.3f}-{np.percentile(three, 97.5):.3f}; proportion with ARI <= '
        f'that of V0 ({a0:.3f}): {(three <= a0 + 1e-12).mean():.3f}; proportion with ARI >= that of '
        f'V0r ({a0r:.3f}): {(three >= a0r - 1e-12).mean():.3f}.')
    obsv.to_csv(os.path.join(OUT, "120_observed.csv"), index=False)

    # ------------------------------------------------------------------ 2 ----
    title(f'2. PREDICTION STRENGTH AND JACCARD ON {N_SERIES} SERIES OF RANDOM SPLITS')
    F = {}
    for name in datasets:
        bl = sorted((r for r in result if r['kind'] == 'streams' and r['branch'] == name),
                    key=lambda r: r["r0"])
        F[name] = pd.DataFrame([x for r in bl for x in r['streams']])
        assert len(F[name]) == N_SERIES
    names_k2 = list(saved['names_k2'])
    rows2 = []
    for name in datasets:
        for stat in ("PS_k2", "jaccard_k2"):
            v = F[name][stat].to_numpy(float)
            r = dict(dataset=name, statistic=stat, mean=float(v.mean()),
                     sd_between_series=float(v.std(ddof=1)))
            if name in ("V0", "V1"):
                nul = saved[f'{name}_24h_null_k2'][:, names_k2.index(stat)]
                p = np.array([_rule_functions.p_mc(nul, x) for x in v])
                r.update(p_at_mean=_rule_functions.p_mc(nul, float(v.mean())),
                         share_series_having_p_below_005=float((p < ALPHA).mean()),
                         p_median=float(np.median(p)), B_null=len(nul))
            rows2.append(r)
    ser = pd.DataFrame(rows2)
    log(ser.round(4).to_string(index=False))
    for name in ("V0", "V1"):
        ps = np.array([_rule_functions.p_mc(saved[f'{name}_24h_null_k2'][:, names_k2.index("PS_k2")], x)
                       for x in F[name].PS_k2])
        ja = np.array([_rule_functions.p_mc(saved[f'{name}_24h_null_k2'][:, names_k2.index("jaccard_k2")], x)
                       for x in F[name].jaccard_k2])
        log(f'{name}: series in which PS or Jaccard has p < 0.05 (first arm met by one of the two): '
            f'{int(((ps < ALPHA) | (ja < ALPHA)).sum())} of {N_SERIES}.')
    log('Paired differences (same series), mean and standard error:')
    for a, b in (("V0r", "V0"), ("S", "V0r"), ("V1", "V0r"), ("V1", "S")):
        for stat in ("PS_k2", "jaccard_k2"):
            d = F[a][stat].to_numpy() - F[b][stat].to_numpy()
            log(f"  {a} - {b}, {stat}: {d.mean():+.4f} (ES {d.std(ddof=1) / np.sqrt(len(d)):.4f})")
    ser.to_csv(os.path.join(OUT, "120_series.csv"), index=False)

    # ------------------------------------------------------------------ 3 ----
    title(f'3. COVARIATES PERMUTED AMONG THE SUBJECTS ({N_PERM} PERMUTATIONS)')
    P = pd.DataFrame([x for r in result if r['kind'] == "perm" for x in r['rows']]).sort_values("b")
    assert len(P) == N_PERM and P.b.is_unique
    ps1, ja1 = series_fixed(X1)
    ps0, ja0 = series_fixed(X0r)
    ref = {"silhouette_k2": (sil_k2(X1), sil_k2(X0r)), "PS_k2": (ps1, ps0), "jaccard_k2": (ja1, ja0)}
    rows3 = []
    for stat, (v1, v0) in ref.items():
        v = P[stat].to_numpy(float)
        q = float((v >= v1 - 1e-12).mean())
        rows3.append(dict(statistic=stat, V0r=v0, V1=v1, permuted_median=float(np.median(v)),
                           permuted_p2_5=float(np.percentile(v, 2.5)),
                           permuted_p97_5=float(np.percentile(v, 97.5)),
                           share_permuted_ge_V1=q,
                           p=float(((v >= v1 - 1e-12).sum() + 1) / (N_PERM + 1))))
    per = pd.DataFrame(rows3)
    log(per.round(4).to_string(index=False))
    log("Reading: 'share_permuted_ge_V1' is the proportion of corrections for covariates with no "
        "link to the subjects that give a value at least equal to the one obtained by removing the "
        "true duration and stage. PS and Jaccard are computed on a single fixed series of splits.")
    per.to_csv(os.path.join(OUT, "120_covariates_permuted.csv"), index=False)

    # ------------------------------------------------------------------ 4 ----
    title(f'4. SILHOUETTE NULL EXTENDED TO {B_TOT} REALISATIONS')
    curve = pd.read_csv(os.path.join(OUT, "118_curves_k2_10.csv"))
    rows4 = []
    for var in ("V0", "V1"):
        for tag in ("24h", "150h"):
            old = saved[f'{var}_{tag}_null_silhouette']
            ident = next(r for r in result if r['kind'] == 'curve_null' and r['branch'] == f"{var}|{tag}|id")
            assert np.abs(ident["silhouette"] - old[:5]).max() < 1e-12, \
                f'{var} {tag}: the first regenerated realisations do not match those of 118'
            new = sorted((r for r in result if r['kind'] == 'curve_null' and r['branch'] == f"{var}|{tag}"),
                           key=lambda r: r["b0"])
            N = np.vstack([old] + [r["silhouette"] for r in new])
            assert N.shape == (B_TOT, len(K_CURVE)), N.shape
            o = curve[(curve.variant == var) & (curve.cohort == tag)
                      & (curve.statistic == "silhouette")].set_index("k")
            for i, k in enumerate(K_CURVE):
                p = _rule_functions.p_mc(N[:, i], float(o.loc[k, 'observed']))
                rows4.append(dict(variant=var, cohort=tag, k=k, observed=float(o.loc[k, 'observed']),
                                   p_B2000=float(o.loc[k, 'p_B_extended']), p_B10000=p,
                                   SE_montecarlo=_rule_functions.se_mc(p, B_TOT), B=B_TOT))
    S = pd.DataFrame(rows4)
    for var in ("V0", "V1"):
        s = S.variant == var
        S.loc[s, "p_BH_B10000"] = _rule_functions.bh(S.loc[s, "p_B10000"].to_numpy())
    S.to_csv(os.path.join(OUT, "120_silhouette_B10000.csv"), index=False)
    log("Identity: the first 5 regenerated realisations match those saved by 118.")
    log(S.round(4).to_string(index=False))
    for var in ("V0", "V1"):
        s = S[S.variant == var]
        i = s.p_BH_B10000.idxmin()
        log(f"{var}: minimum corrected p {s.p_BH_B10000.min():.4f} (with 2000 realisations 118 gave "
            f"{curve[(curve.variant == var) & (curve.statistic == 'silhouette')].p_BH_B_extended.min():.4f}); "
            f"smallest uncorrected p {s.p_B10000.min():.4f} ({s.loc[s.p_B10000.idxmin(), 'cohort']}, "
            f"k = {int(s.loc[s.p_B10000.idxmin(), 'k'])}); surviving BH: "
            f"{int((s.p_BH_B10000 < ALPHA).sum())} of {len(s)}.")

    title('WHAT THIS SCRIPT DOES NOT DO')
    log('- It does not estimate a null for V0r and for S.')
    log('- It does not extend the null of the prediction strength.')
    log('- It does not address the calibration of the Monte Carlo p values against a reference '
        'fitted to the same data (remark D1 of the verifier of 118).')
    log(f'\nTotal time: {time.time() - t0:.0f} s')
    with open(os.path.join(OUT, "120_duration_stage_checks.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
