"""
121 - Calibration of the Monte Carlo p values of the decision rule (double Monte
Carlo).

Why it exists. The decision rule of the manuscript judges each criterion
against a reference (Gaussian copula) FITTED TO THE SAME DATA: empirical
marginals and rank correlations are those of the data set analysed. The
independent verifier of 118 (revisione_R2_rapporti/verifiche_script/
riverifica_118.txt, section 6.4) simulated data sets without groups and found
that, at the nominal level 0.05, the p values so obtained reject much less than
5% of the time (silhouette at k = 2: 0-0.6%; prediction strength 1.0%; Jaccard
1.5%; ARI between algorithms 3.5%), with small simulations and estimators of
its own. If so: (a) "rule not met" is a more cautious outcome than 0.05 says,
and the rule is not "deliberately permissive" as to the level; (b) the nominal
p values of the real data must be read on a calibrated scale. This script
measures it with the functions of the project.

Design, written before the results of this script (the verifier's numbers
were known).
  Truth without groups. For each cohort, the Gaussian copula fitted to the
  real data (seven-variable matrix of the manuscript): each simulated data set
  ("outer") is one realisation of it, re-standardised as the procedure does.
  It is ONE truth without groups among the possible ones: a truth without
  groups but with non-Gaussian dependence would give other numbers.
  Procedure applied to each outer data set, identical to that of the
  manuscript: copula reference fitted to the outer data set; B_INT "inner"
  realisations; one-sided p (#{inner >= observed} + 1)/(B_INT + 1).
  The real data go through the SAME procedure (same B_INT, same generators)
  as data set "-1", so the comparison is like for like; the published p values
  are reported alongside.
  What is calibrated.
    S. Silhouette curve over k = 2-10 (function of 45). N_SIL outer data sets
       per cohort, B_INT_SIL inner realisations.
    K. The three criteria at k = 2 of the first arm: prediction strength,
       Jaccard, ARI between algorithms (functions of 25 via _rule_functions). N_K2 outer
       data sets per cohort, B_INT_K2 inner realisations. The outer data sets
       of K are the first N_K2 of S (same generator), so the two arms can be
       looked at together on the same data sets.
    The prediction strength curve is NOT calibrated: it costs about 7 seconds
    of CPU per realisation and the double Monte Carlo would require tens of
    thousands of hours. The second arm is therefore calibrated for the
    silhouette only.
  Quantities reported.
    1. Proportion of outer data sets with nominal p < 0.05, by criterion and
       by k, with Clopper-Pearson interval; quantiles of the nominal p values.
    2. Calibrated p of the real data, by criterion and by k: proportion of
       outer data sets with nominal p <= that of the real data. Alongside, the
       same thing on the scale of the standardised departure z = (observed -
       median of the inner reference) / SD of the inner reference, which does
       not suffer from the discreteness of the p values.
    3. Second arm, silhouette, as the manuscript applies it: the outer data
       sets of the two cohorts are paired by index (they are independent),
       Benjamini-Hochberg on the 18 tests, and it is counted how often at
       least one adjusted p is < 0.05 without groups. Calibrated p of the real
       data for the whole arm: proportion of outer pairs with minimum adjusted
       p <= that of the real data; and with maximum z >= that of the real
       data.
    4. First arm: proportion of outer data sets in which at least one of the
       three criteria has p < 0.05, by cohort and for the two cohorts
       together; calibrated p of the real data on the minimum p of the three.
  Readings, fixed here.
    - If the proportions of point 1 are well below 0.05, the tests are
      conservative: the phrase "deliberately permissive" must be corrected and
      "not met" must be presented as a cautious outcome.
    - If the calibrated p of the real data at point 3 is < 0.05, the published
      data depart from the reference more than data sets without groups
      generated from that reference do. This does NOT demonstrate groups: it
      says that the fitted Gaussian copula does not describe all the
      dependence between the variables (for example curvature or tails). The
      manuscript should then say that the rule is not met on the nominal p
      values and that, calibrated, the silhouette exceeds the reference, and
      discuss what can produce the departure without groups.
    - If it is >= 0.05, the published result holds when calibrated too.
  The resolution is 1/N: with N_SIL = 400 a calibrated p below 0.0025 cannot
  be distinguished.

CORRECTIONS AFTER THE INDEPENDENT VERIFICATION (14/11/2026). The text above is
left as it was when the script was run and verified (copy in
revisione_R2_rapporti/versioni_script/121_come_eseguito_e_verificato.py); the
code has not changed. The report is in
revisione_R2_rapporti/verifiche_script/riverifica_121.txt. Four points of the
text above are wrong or incomplete.
  1. Direction of "cautious". Point (a) of the first paragraph and the first
     fixed reading say that, if the tests are conservative, "not met" is a
     more cautious outcome. It is the opposite: a test that without groups
     rejects less than 5% of the time also rejects less when the groups are
     there, and a negative result obtained with that test WEIGHS LESS, not
     more. The power of the rule is not measured by this script.
  2. "Deliberately permissive". It must be removed, but not because the rule
     is strict: the individual tests are conservative (silhouette 0.4-3.7%
     depending on k), whereas the whole rule rejects without groups in about
     6% of cases, that is, it is an ordinary 5% test.
  3. Designated statistic. The script does not say which, between the
     calibrated p on the minimum adjusted p and the one on the maximum z,
     decides the reading of point 3; the two and their variants range from
     0.027 to 0.050. The value 0.042 must not be cited on its own: "about
     0.03-0.05, at the threshold". Script 123 fixes a designated statistic
     for the two most extreme tests and corrects them for 18.
  4. p of the real data at 199 realisations. The like-for-like comparison
     uses for the real data a single draw of 199 realisations, which is noisy:
     at 24 h with two groups it gives 0.060 against 0.0345 at 10,000
     realisations; at 150 h with three groups 0.030 against 0.0168. Both must
     be reported.
Other remarks that do not change the numbers: the reference is not
re-standardised as the data are (it is the procedure of the manuscript; at
k = 2 its SD is 1.3-1.6 times wider); the observed silhouette for k >= 5
depends on the k-means seed even with 50 initialisations; with 199
realisations a single test cannot survive the correction over 18.

What the script does NOT do: it does not calibrate the prediction strength
curve; it does not try other truths without groups; it does not inject groups
(power), which is the question of Reviewer 3 at point 2 and requires a separate
analysis; it does not calibrate the review analyses on
residualised variables.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 -B scripts/121_rule_calibration.py
      N_PROCESSES=n   (default 11; the results do not depend on it)
      PROVA=1        few data sets and few realisations, to measure the times;
                     does not write the final outputs
Output: outputs/121_rule_calibration.txt, 121_silhouette_outer.csv,
        121_first_arm_outer.csv, 121_summary.csv; during execution,
        121_partial.pkl
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

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
TRIAL = os.environ.get('TRIAL', "0") == "1"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "11")))


from _modules import load_script as _module                      # noqa: E402


M118 = _module("118_duration_stage.py", 'duration_118')
import _rule_functions                                                    # noqa: E402
C45 = M118.C45
K_CURVE = list(C45.KRANGE)

SEED = 20261114
N_SIL, B_INT_SIL = (400, 199) if not TRIAL else (4, 5)
N_K2, B_INT_K2 = (200, 99) if not TRIAL else (4, 5)
ALPHA = 0.05
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
    """95% Clopper-Pearson interval."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def external(X, tag, j):
    """Outer data set j of the cohort: j = -1 is the real data."""
    if j < 0:
        return X
    g = np.random.default_rng([SEED, crc32_legacy(tag.encode()), j])
    return M118.standardise(C45.copula_null(X, g))


def criteria_k2(X, g):
    return np.array([_rule_functions.ps_k2(X, g), _rule_functions.jaccard_k2(X, g), _rule_functions.ari_algos(X)])


def job(t):
    tag, j, kind, B = t["tag"], t["j"], t['kind'], t["B"]
    Y = external(t["X"], tag, j)
    crc, jj = crc32_legacy(tag.encode()), j + 1
    t0 = time.time()
    if kind == "sil":
        obsv = C45.silhouette_curve(Y, seed=C45.SEED).silhouette.to_numpy(float)
        nul = np.empty((B, len(K_CURVE)))
        for b in range(B):
            g = np.random.default_rng([SEED, crc, jj, 1, b])
            Z = C45.copula_null(Y, g)
            nul[b] = C45.silhouette_curve(Z, seed=int(g.integers(1e9))).silhouette.to_numpy(float)
    else:
        obsv = criteria_k2(Y, np.random.default_rng([SEED, crc, jj, 2]))
        nul = np.empty((B, len(CRITERIA)))
        for b in range(B):
            g = np.random.default_rng([SEED, crc, jj, 3, b])
            nul[b] = criteria_k2(C45.copula_null(Y, g), g)
    p = ((nul >= obsv - 1e-12).sum(0) + 1) / (B + 1)
    z = (obsv - np.median(nul, 0)) / nul.std(0, ddof=1)
    return dict(tag=tag, j=j, kind=kind, obsv=obsv, p=p, z=z, seconds=time.time() - t0)


def main():
    t0 = time.time()
    title('121 - CALIBRATION OF THE p VALUES OF THE DECISION RULE (DOUBLE MONTE CARLO)')
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
    log(f"Declared working directory: {('sporadic only' if ONLY else 'full cohort')}; n = "
        f"{len(X['24h'])} and {len(X['150h'])}. Identity: the matrices coincide with those of 118.")
    log(f'Silhouette: {N_SIL} outer data sets per cohort, {B_INT_SIL} inner realisations. Criteria '
        f'at k = 2: {N_K2} outer data sets, {B_INT_K2} inner realisations. Processes: {N_PROC}.' + ('  [PROVA]' if TRIAL else ""))

    tasks = []
    for tag in X:
        for j in range(-1, N_SIL):
            tasks.append(dict(tag=tag, j=j, kind="sil", B=B_INT_SIL, X=X[tag]))
        for j in range(-1, N_K2):
            tasks.append(dict(tag=tag, j=j, kind="k2", B=B_INT_K2, X=X[tag]))
    # the real data first, then the tasks alternated: the partial results are useful at once
    tasks.sort(key=lambda c: (c["j"] >= 0, c["j"], c['kind'], c["tag"]))
    result = []
    partial = os.path.join(OUT, "121_partial.pkl")
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        for i, r in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
            result.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                print(f'    {i + 1}/{len(tasks)}  ({time.time() - t0:.0f} s)', flush=True)
                if not TRIAL:
                    with open(partial, "wb") as fh:
                        pickle.dump(result, fh)
    sec = pd.DataFrame([dict(kind=r['kind'], tag=r["tag"], s=r['seconds']) for r in result])
    log('CPU seconds per outer data set (median): '
        + "; ".join(f"{k[0]} {k[1]} {v:.0f}" for k, v in sec.groupby(['kind', "tag"]).s.median().items()))

    def table(kind, names):
        rows = []
        for r in result:
            if r['kind'] == kind:
                for i, name in enumerate(names):
                    rows.append(dict(cohort=r["tag"], j=r["j"], item=name, observed=float(r['obsv'][i]),
                                      p=float(r["p"][i]), z=float(r["z"][i])))
        return pd.DataFrame(rows)

    S = table("sil", [f"k={k}" for k in K_CURVE])
    K = table("k2", CRITERIA)
    summary = []

    # ------------------------------------------------------------------ 1-2 --
    for name, T, Bint in (('SILHOUETTE OVER k = 2-10', S, B_INT_SIL),
                          ('CRITERIA AT k = 2 (FIRST ARM)', K, B_INT_K2)):
        title(f'1-2. {name}: LEVEL UNDER THE NULL AND CALIBRATED p OF THE REAL DATA')
        for tag in X:
            est = T[(T.cohort == tag) & (T.j >= 0)]
            true = T[(T.cohort == tag) & (T.j < 0)].set_index('item')
            rows = []
            for item in true.index:
                e = est[est.item == item]
                n, kk = len(e), int((e.p < ALPHA).sum())
                lo, hi = cp(kk, n)
                pv, zv = float(true.loc[item, "p"]), float(true.loc[item, "z"])
                rows.append(dict(
                    cohort=tag, item=item, datasets_external=n, B_internal=Bint,
                    share_p_below_005=kk / n, CI_lo=lo, CI_hi=hi,
                    p_nominal_quantile_1=float(e.p.quantile(0.01)),
                    p_nominal_quantile_5=float(e.p.quantile(0.05)),
                    p_nominal_quantile_10=float(e.p.quantile(0.10)),
                    SD_of_z=float(e.z.std(ddof=1)),
                    data_real_p_nominal=pv,
                    data_real_p_calibrated=float(((e.p <= pv + 1e-12).sum() + 1) / (n + 1)),
                    data_real_z=zv,
                    data_real_p_calibrated_over_z=float(((e.z >= zv - 1e-12).sum() + 1) / (n + 1))))
            t = pd.DataFrame(rows)
            summary.append(t)
            log(f"\n{tag}")
            log(t.drop(columns=['cohort', 'B_internal']).round(4).to_string(index=False))

    # ------------------------------------------------------------------ 3 ----
    title('3. SECOND ARM, SILHOUETTE: BENJAMINI-HOCHBERG ON THE 18 TESTS OF THE TWO COHORTS')

    def branch_sil(j):
        a = S[(S.j == j)].sort_values(['cohort', 'item'])
        assert len(a) == 2 * len(K_CURVE)
        return float(_rule_functions.bh(a.p.to_numpy()).min()), float(a.z.max()), float(a.p.min())

    est = np.array([branch_sil(j) for j in range(N_SIL)])
    true = branch_sil(-1)
    n, kk = len(est), int((est[:, 0] < ALPHA).sum())
    lo, hi = cp(kk, n)
    log(f'Pairs of outer data sets (24 h, 150 h): {n}. At least one corrected p < 0.05 in {kk} '
        f'({100 * kk / n:.2f}%; 95% CI {100 * lo:.2f}-{100 * hi:.2f}%).')
    log(f'Quantiles of the minimum corrected p under the null: 1% '
        f'{np.quantile(est[:, 0], 0.01):.3f}, 5% {np.quantile(est[:, 0], 0.05):.3f}, 10% '
        f'{np.quantile(est[:, 0], 0.1):.3f}, median {np.median(est[:, 0]):.3f}.')
    pc = (int((est[:, 0] <= true[0] + 1e-12).sum()) + 1) / (n + 1)
    pz = (int((est[:, 1] >= true[1] - 1e-12).sum()) + 1) / (n + 1)
    pm = (int((est[:, 2] <= true[2] + 1e-12).sum()) + 1) / (n + 1)
    log(f'Real data, same procedure (inner B {B_INT_SIL}): minimum corrected p {true[0]:.4f}, '
        f'minimum uncorrected p {true[2]:.4f}, maximum z {true[1]:.2f}.')
    log(f"  calibrated p of the whole arm: on the minimum corrected p {pc:.4f}; on the minimum "
        f"uncorrected p {pm:.4f}; on the maximum z {pz:.4f}. Resolution 1/{n + 1} = "
        f"{1 / (n + 1):.4f}.")
    summary.append(pd.DataFrame([dict(
        cohort='both', item='second arm, silhouette (18 tests)', datasets_external=n,
        B_internal=B_INT_SIL, share_p_below_005=kk / n, CI_lo=lo, CI_hi=hi,
        data_real_p_nominal=true[0], data_real_p_calibrated=pc, data_real_z=true[1],
        data_real_p_calibrated_over_z=pz)]))

    # ------------------------------------------------------------------ 4 ----
    title('4. FIRST ARM: AT LEAST ONE OF THE THREE CRITERIA WITH p < 0.05')
    W = K.pivot_table(index=['cohort', "j"], columns='item', values="p")
    pmin = W.min(axis=1)
    for tag in X:
        e, v = pmin[tag][pmin[tag].index >= 0], float(pmin[tag][-1])
        n, kk = len(e), int((e < ALPHA).sum())
        lo, hi = cp(kk, n)
        pc1 = (int((e <= v + 1e-12).sum()) + 1) / (n + 1)
        log(f"{tag}: first arm 'satisfied' without groups in {kk} of {n} ({100 * kk / n:.1f}%; 95% "
            f"CI {100 * lo:.1f}-{100 * hi:.1f}%). Real data: minimum p of the three {v:.3f}; "
            f"calibrated p {pc1:.3f}.")
        summary.append(pd.DataFrame([dict(
            cohort=tag, item='first arm (at least one of the three)', datasets_external=n,
            B_internal=B_INT_K2, share_p_below_005=kk / n, CI_lo=lo, CI_hi=hi,
            data_real_p_nominal=v, data_real_p_calibrated=pc1)]))
    two = pd.concat([pmin["24h"][pmin["24h"].index >= 0].rename("a"),
                     pmin["150h"][pmin["150h"].index >= 0].rename("b")], axis=1).min(axis=1)
    kk = int((two < ALPHA).sum())
    lo, hi = cp(kk, len(two))
    log(f"The two cohorts together (data sets paired by index): first arm 'satisfied' without groups "
        f"in {kk} of {len(two)} ({100 * kk / len(two):.1f}%; 95% CI {100 * lo:.1f}-{100 * hi:.1f}%).")
    # the two arms together, on the same outer data sets
    sil_ok = pd.Series(est[:N_K2, 0] < ALPHA, index=range(N_K2))
    dataset = int(((two.reindex(range(N_K2)) < ALPHA) | sil_ok).sum())
    lo, hi = cp(dataset, N_K2)
    log(f"First arm or second arm (silhouette only), same {N_K2} data sets: rule 'satisfied' without "
        f"groups in {dataset} ({100 * dataset / N_K2:.1f}%; 95% CI {100 * lo:.1f}-{100 * hi:.1f}%). "
        f"The prediction strength curve is not included.")

    title('WHAT THIS SCRIPT DOES NOT DO')
    log('- It does not calibrate the prediction strength curve (second arm calibrated for the '
        'silhouette only).')
    log("- It uses a single truth without groups: the Gaussian copula fitted to the real data.")
    log('- It does not inject groups: it does not measure the power of the rule.')
    log('- It does not calibrate the analyses added in review.')
    log(f'\nTotal time: {time.time() - t0:.0f} s')
    if TRIAL:
        log('PROVA=1: no final output written.')
        return
    S.to_csv(os.path.join(OUT, "121_silhouette_outer.csv"), index=False)
    K.to_csv(os.path.join(OUT, "121_first_arm_outer.csv"), index=False)
    pd.concat(summary, ignore_index=True).to_csv(os.path.join(OUT, "121_summary.csv"), index=False)
    with open(os.path.join(OUT, "121_rule_calibration.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
