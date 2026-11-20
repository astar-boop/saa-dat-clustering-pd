"""
125 - Power of the decision rule and recovery of an injected group
(Reviewer 3, point 2).

Why it exists. The reviewer asks whether the decision rule of the manuscript,
the actual one, recovers a group defined by a plausible difference in seeding
time or in DAT binding, even a small one. Supplement §1.11 (script 22)
measures only how often SINGLE criteria exceed conventional thresholds, and
does not look at the labels; script 121 applies the rule to data sets WITHOUT
groups. Here groups are injected into the real data and the procedure of the
manuscript is applied in full (copula reference fitted to the data set
analysed), then it is measured whether the partition finds the injected
participants.

Design, written before the results of this script. Decision of the user of
14/11/2026 (revisione_R3_rapporti/PREPARAZIONE_R3.md, decision 1).
  Data. The two seven-variable matrices of the manuscript (identity checked
  against those of 118 and 121), column order: log Fmax, log TTT (24 h) or
  log T50 (150 h), log AUC, log slope, lower putamen, mean caudate, age.
  Injection. The function of 22 (Supplement §1.11): a proportion pi of
  participants drawn at random is shifted by Delta along a direction, with
  Delta in units of SD along that direction; then the matrix is
  re-standardised as the procedure does. Directions:
    tempo   only the seeding time column (TTT at 24 h, T50 at 150 h),
            towards longer times;
    dat     only the lower putamen, towards lower binding;
    pc1     the first principal component (comparison with §1.11).
  For the directions on one column Delta is the difference between the two
  groups in SD of that variable before the injection.
  Grid: Delta 1.0; 1.5; 2.0; 3.0. pi 0.05; 0.10; 0.20; 0.50. N_REP replicates
  per cell and cohort. Plus one cell Delta = 0 per cohort (the real data, one
  replicate: only the seeds change), which gives the starting point at the
  same B.
  Procedure for each data set (as in 121, functions of 45 via 118 and of _rule_functions):
    first arm    prediction strength, Jaccard and ARI between algorithms at
                 k = 2, each against B_K2 realisations of the fitted copula;
                 one-sided p (#{null >= observed} + 1) / (B + 1);
    second arm   silhouette curve over k = 2-10 against B_SIL realisations.
  The prediction strength curve does NOT enter (about 7 s per realisation, as
  in 121): the power measured is a LOWER BOUND on that of the whole rule,
  which has one more arm joined by "or".
  B_SIL = 299 (the manuscript: 500 at 24 h, 300 at 150 h); B_K2 = 99. With
  B_SIL = 299 the minimum p is 1/300: a single extreme k survives BH over 9
  tests (9/300 = 0.03) but not over 18 (0.06). To be reported next to the
  results.
  Outcome of the rule, two readings:
    per cohort   first arm: at least one of the three p < 0.05; second arm: BH
                 on the 9 k of the cohort, at least one adjusted p < 0.05;
    joint        the group injected in both cohorts (replicates paired by
                 index, same cell): first arm on the 6 p, second arm BH on
                 the 18 tests of the two cohorts. It is the rule of the
                 manuscript.
  Recovery (on the injected and re-standardised data set): k-means at k = 2
  (50 initialisations) and Gaussian mixtures at k = 2; ARI with the injected
  labels; for the injected group, recall and precision of the group of the
  partition that contains most of it.
  Quantities reported: per cell, proportion of data sets with the rule met
  (and per arm), with Clopper-Pearson interval; median ARI and proportion with
  ARI >= 0.5; median recall and precision; separately among the data sets in
  which the rule is met.

Readings, fixed here.
  - "The rule detects a group of this size" only where the proportion of the
    joint reading is >= 0.80 with Clopper-Pearson lower limit >= 0.6.
  - Where the rule is met but the median ARI is < 0.5, the rule signals
    structure without finding the group: it must be said so.
  - The copula reference reproduces the marginals of the data set analysed,
    including the bimodal marginals created by the injection on one column: a
    low power along "tempo" or "dat" is a property of the procedure of the
    manuscript, not a defect of this script, and must be written as such.
  - This script does NOT calibrate the p values (121 does): the proportions
    are on the nominal scale, that of the published rule. Without groups the
    nominal rule is met in about 6% of the data sets (121).
  - Injecting into the real data superimposes the group on the structure
    already present (including the run regime of the TTT at 24 h): it is the
    reviewer's question ("could the rule have found a group in these data"),
    not power in the abstract.

Outputs (working directory run_sporadici, ONLY_SPORADIC=1): outputs/125_outcomes.csv
(one data set per row), outputs/125_summary.csv, outputs/125_rule_power.txt,
outputs/125_partial.pkl (resumption). Environment variables: N_PROCESSES, PROVA=1
(reduced grid, small B, no final file), N_REP.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util                                            # noqa: E402
import multiprocessing as mp                                     # noqa: E402
import pickle                                                    # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
import warnings                                                  # noqa: E402
import zlib                                                      # noqa: E402
from _legacy import crc32_legacy                                # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402
from sklearn.cluster import KMeans                               # noqa: E402
from sklearn.metrics import adjusted_rand_score                  # noqa: E402
from sklearn.mixture import GaussianMixture                      # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
TRIAL = os.environ.get('TRIAL', "0") == "1"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "12")))


from _modules import load_script as _module                      # noqa: E402


M118 = _module("118_duration_stage.py", 'duration_118')
import _rule_functions                                                    # noqa: E402
C45 = M118.C45
K_CURVE = list(C45.KRANGE)

SEED = 20261115
ALPHA = 0.05
if TRIAL:
    DELTAS, PIS, DIRECTIONS = [2.0], [0.10], ['timeval', "dat"]
    N_REP = int(os.environ.get("N_REP", "2"))
    B_SIL, B_K2 = 9, 9
else:
    DELTAS, PIS, DIRECTIONS = [1.0, 1.5, 2.0, 3.0], [0.05, 0.10, 0.20, 0.50], ['timeval', "dat", "pc1"]
    N_REP = int(os.environ.get("N_REP", "40"))
    B_SIL, B_K2 = 299, 99
COL = {'timeval': 1, "dat": 4}
SIGN = {'timeval': +1.0, "dat": -1.0, "pc1": +1.0}
CRITERIA = list(_rule_functions.CRITERIA)
rep = []


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def cp(k, n):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def direction(X, name):
    if name in COL:
        d = np.zeros(X.shape[1])
        d[COL[name]] = 1.0
    else:
        w, V = np.linalg.eigh(np.cov(X, rowvar=False))
        d = V[:, np.argmax(w)]
        d = d * np.sign(d[np.argmax(np.abs(d))])      # sign fixed: largest component positive
    return SIGN[name] * d


def inject(X, d, delta, pi, rng):
    """Like inject() of 22_injection_power.py (Supplement §1.11)."""
    n = len(X)
    S = np.cov(X, rowvar=False)
    u = d / np.sqrt(d @ S @ d)
    idx = rng.choice(n, int(round(pi * n)), replace=False)
    Y = X.copy()
    Y[idx] = Y[idx] + delta * (S @ u)
    lab = np.zeros(n, int)
    lab[idx] = 1
    return Y, lab


def recovery(Y, lab, seed):
    out = {}
    for name, g in (("kmeans", KMeans(2, n_init=50, random_state=seed).fit_predict(Y)),
                    ("gmm", GaussianMixture(2, n_init=10, random_state=seed).fit(Y).predict(Y))):
        out[f'ARI_{name}'] = adjusted_rand_score(lab, g)
        if lab.sum() == 0:
            out[f'recall_{name}'] = out[f'precision_{name}'] = np.nan
            continue
        c = np.bincount(g[lab == 1], minlength=2).argmax()
        out[f'recall_{name}'] = float((g[lab == 1] == c).mean())
        out[f'precision_{name}'] = float((lab[g == c] == 1).mean())
    return out


def job(t):
    tag, dn, of, pj, r = t["tag"], t["dir"], t['of'], t["pj"], t["r"]
    X = t["X"]
    crc = crc32_legacy(tag.encode())
    key = [SEED, crc, DIRECTIONS.index(dn) if dn in DIRECTIONS else 99, of, pj, r]
    t0 = time.time()
    if t["delta"] == 0:
        Y, lab = X.copy(), np.zeros(len(X), int)
    else:
        g = np.random.default_rng(key + [0])
        Y, lab = inject(X, direction(X, dn), t["delta"], t["pi"], g)
    Y = M118.standardise(Y)
    # second arm: silhouette curve
    obsv_s = C45.silhouette_curve(Y, seed=C45.SEED).silhouette.to_numpy(float)
    nul_s = np.empty((B_SIL, len(K_CURVE)))
    for b in range(B_SIL):
        g = np.random.default_rng(key + [1, b])
        nul_s[b] = C45.silhouette_curve(C45.copula_null(Y, g), seed=int(g.integers(1e9))).silhouette.to_numpy(float)
    p_s = ((nul_s >= obsv_s - 1e-12).sum(0) + 1) / (B_SIL + 1)
    # first arm: criteria at k = 2
    obsv_k = np.array([_rule_functions.ps_k2(Y, np.random.default_rng(key + [2])),
                      _rule_functions.jaccard_k2(Y, np.random.default_rng(key + [2])),
                      _rule_functions.ari_algos(Y)])
    nul_k = np.empty((B_K2, len(CRITERIA)))
    for b in range(B_K2):
        g = np.random.default_rng(key + [3, b])
        Z = C45.copula_null(Y, g)
        nul_k[b] = [_rule_functions.ps_k2(Z, g), _rule_functions.jaccard_k2(Z, g), _rule_functions.ari_algos(Z)]
    p_k = ((nul_k >= obsv_k - 1e-12).sum(0) + 1) / (B_K2 + 1)
    res = dict(cohort=tag, direction=dn, delta=t["delta"], pi=t["pi"], replication=r,
               n_injected=int(lab.sum()), seconds=time.time() - t0)
    res.update({f"p_sil_k{k}": float(v) for k, v in zip(K_CURVE, p_s)})
    res.update({f"sil_k{k}": float(v) for k, v in zip(K_CURVE, obsv_s)})
    res.update({f"p_{c}": float(v) for c, v in zip(CRITERIA, p_k)})
    res.update({f'obsv_{c}': float(v) for c, v in zip(CRITERIA, obsv_k)})
    res.update(recovery(Y, lab, seed=int(np.random.default_rng(key + [4]).integers(1e9))))
    return res


def key_res(d):
    return (d['cohort'], d['direction'], d["delta"], d["pi"], d['replication'])


def main():
    t0 = time.time()
    title('125 - POWER OF THE DECISION RULE AND RECOVERY OF AN INJECTED GROUP')
    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy())
    R150 = c150[M118.F150].to_numpy(float).copy()
    R150[:, :4] = np.log(R150[:, :4])
    X = {"24h": npz["F_primary"].astype(float), "150h": M118.standardise(R150)}
    saved = np.load(os.path.join(OUT, "118_nulls.npz"), allow_pickle=True)
    for tag in X:
        assert np.abs(saved[f'V0_{tag}_matrix'] - X[tag]).max() < 1e-12, \
            f'{tag}: the matrix does not coincide with that analysed by 118 (V0)'
    assert ONLY, 'run with ONLY_SPORADIC=1 on the working directory run_sporadici'
    log(f"n = {len(X['24h'])} and {len(X['150h'])}; matrices identical to those of 118 and 121.")
    log(f'Grid: directions {DIRECTIONS}; Delta {DELTAS}; pi {PIS}; {N_REP} replicates. B silhouette '
        f'{B_SIL}, B k = 2 {B_K2}. Processes {N_PROC}.' + ('  [PROVA]' if TRIAL else ""))
    for tag in X:
        for dn in DIRECTIONS:
            d = direction(X[tag], dn)
            log(f'  {tag} direction {dn}: ' + ", ".join(f"{v:+.2f}" for v in d))

    tasks = [dict(tag=tag, dir='nothing', of=0, pj=0, r=0, delta=0.0, pi=0.0, X=X[tag]) for tag in X]
    for r in range(N_REP):
        for dn in DIRECTIONS:
            for of, delta in enumerate(DELTAS):
                for pj, pi in enumerate(PIS):
                    for tag in X:
                        tasks.append(dict(tag=tag, dir=dn, of=of, pj=pj, r=r, delta=delta, pi=pi, X=X[tag]))
    partial = os.path.join(OUT, "125_partial.pkl" if not TRIAL else "125_partial_TRIAL.pkl")
    result = []
    if os.path.exists(partial) and not TRIAL:
        with open(partial, "rb") as fh:
            result = pickle.load(fh)
        done = {key_res(d) for d in result}
        tasks = [c for c in tasks if (c["tag"], c["dir"], c["delta"], c["pi"], c["r"]) not in done]
        log(f'Resumption: {len(result)} data sets already done, {len(tasks)} to do.')
    total = len(tasks) + len(result)
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        for r in pool.imap_unordered(job, tasks, chunksize=1):
            result.append(r)
            if len(result) % 25 == 0 or len(result) == total:
                print(f'    {len(result)}/{total}  ({time.time() - t0:.0f} s)', flush=True)
                with open(partial, "wb") as fh:
                    pickle.dump(result, fh)
    E = pd.DataFrame(result).sort_values(['cohort', 'direction', "delta", "pi", 'replication']).reset_index(drop=True)
    assert len(E) == total and not E.duplicated(['cohort', 'direction', "delta", "pi", 'replication']).any()
    log(f'CPU seconds per data set (median): ' +
        "; ".join(f"{k} {v:.0f}" for k, v in E.groupby('cohort').seconds.median().items()))

    psil = [f"p_sil_k{k}" for k in K_CURVE]
    pk2 = [f"p_{c}" for c in CRITERIA]
    E['branch1'] = (E[pk2] < ALPHA).any(axis=1)
    E["bh9_min"] = [float(_rule_functions.bh(r).min()) for r in E[psil].to_numpy()]
    E['branch2'] = E.bh9_min < ALPHA
    E['rule'] = E.branch1 | E.branch2

    # joint reading: replicates paired by index between the two cohorts
    a = E[E.cohort == "24h"].set_index(['direction', "delta", "pi", 'replication'])
    b = E[E.cohort == "150h"].set_index(['direction', "delta", "pi", 'replication'])
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b)
    J = pd.DataFrame(index=common)
    J['branch1'] = a.loc[common, 'branch1'].to_numpy() | b.loc[common, 'branch1'].to_numpy()
    J["bh18_min"] = [float(_rule_functions.bh(np.r_[x, y]).min())
                     for x, y in zip(a.loc[common, psil].to_numpy(), b.loc[common, psil].to_numpy())]
    J['branch2'] = J.bh18_min < ALPHA
    J['rule'] = J.branch1 | J.branch2
    J = J.reset_index()

    title('STARTING POINT: REAL DATA, NO INJECTION (same B)')
    for _, r in E[E.direction == 'nothing'].iterrows():
        log(f'  {r.cohort}: p k=2 ' + ", ".join(f"{c} {r[f'p_{c}']:.3f}" for c in CRITERIA)
            + f'; silhouette min p {r[psil].min():.4f}, BH over 9 {r.bh9_min:.3f}; rule per cohort '
            + ('satisfied' if r.rule else 'not satisfied'))
    j0 = J[J.direction == 'nothing'].iloc[0]
    log(f'  joint: BH over 18 {j0.bh18_min:.3f}; rule ' + ('satisfied' if j0.rule else 'not satisfied'))

    title('POWER AND RECOVERY BY CELL')
    rows = []
    for (dn, delta, pi), g in E[E.direction != 'nothing'].groupby(['direction', "delta", "pi"]):
        jj = J[(J.direction == dn) & (J.delta == delta) & (J.pi == pi)]
        n, k = len(jj), int(jj.rule.sum())
        lo, hi = cp(k, n)
        row = dict(direction=dn, delta=delta, pi=pi, replicates=n,
                    joint_rule=k / n, joint_CI_lo=lo, joint_CI_hi=hi,
                    joint_branch1=jj.branch1.mean(), joint_branch2=jj.branch2.mean())
        for tag in ("24h", "150h"):
            s = g[g.cohort == tag]
            row.update({f'{tag}_rule': s.rule.mean(), f'{tag}_branch1': s.branch1.mean(),
                         f'{tag}_branch2': s.branch2.mean(),
                         f"{tag}_ARI_kmeans_med": s.ARI_kmeans.median(),
                         f"{tag}_ARI_gmm_med": s.ARI_gmm.median(),
                         f'{tag}_share_ARI_kmeans_ge05': (s.ARI_kmeans >= 0.5).mean(),
                         f'{tag}_recall_kmeans_med': s.recall_kmeans.median(),
                         f'{tag}_precision_kmeans_med': s.precision_kmeans.median(),
                         f'{tag}_ARI_kmeans_med_se_rule': s.loc[s.rule, "ARI_kmeans"].median()
                         if s.rule.any() else np.nan})
        row['detect'] = (row['joint_rule'] >= 0.80) and (lo >= 0.6)
        rows.append(row)
    T = pd.DataFrame(rows)
    for dn in DIRECTIONS:
        log(f'\nDirection {dn}')
        cols = ["delta", "pi", 'joint_rule', 'joint_CI_lo', 'joint_branch1', 'joint_branch2',
                "24h_regola", "150h_regola", "24h_ARI_kmeans_med", "150h_ARI_kmeans_med",
                "24h_richiamo_kmeans_med", "24h_precisione_kmeans_med", 'detect']
        log(T[T.direction == dn][cols].round(3).to_string(index=False))

    title('LIMITATIONS')
    log(f"- The prediction strength curve is not applied: the proportions are a lower bound.")
    log(f'- B silhouette {B_SIL}: minimum p 1/{B_SIL + 1}; a single extreme k does not pass BH over '
        f'18.')
    log("- Nominal scale (not calibrated): without groups the rule is satisfied in about 6% (121).")
    log("- Injection on the real data: the group adds to the structure already present.")
    log(f'\nTotal time: {time.time() - t0:.0f} s')

    if not TRIAL:
        E.to_csv(os.path.join(OUT, "125_outcomes.csv"), index=False)
        T.to_csv(os.path.join(OUT, "125_summary.csv"), index=False)
        with open(os.path.join(OUT, "125_rule_power.txt"), "w") as fh:
            fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
