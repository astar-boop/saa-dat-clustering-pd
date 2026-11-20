"""
123 - The silhouette of the real data against the reference: how far out it is
on the calibrated scale (two tests, enlarged calibration) and where the
departure comes from.

Why it exists. Script 121 (double Monte Carlo) and its independent
verification (revisione_R2_rapporti/verifiche_script/riverifica_121.txt)
showed that, on the calibrated scale, the silhouette of the real data exceeds
the Gaussian copula reference at two points of the curve: 24 h with two groups
and 150 h with three groups. With 400 (and 700) data sets without groups the
resolution is not enough to say whether those two values withstand the
correction for the 18 tests of the curve, and the p of the real data at 199
realisations is a single noisy draw (0.060 against 0.0345 at 10,000). The
verifier then looked for the origin of the departure with analyses of its own,
after the fact: it lies in the four kinetic parameters, it disappears when
their marginals are normalised, at 24 h it disappears when the 24 participants
with the lowest AUC are removed, there is no bimodality between the centroids.
Those analyses are not in a script of the project. This script redoes them with
its own code and enlarges the calibration of the two tests.

HONESTY ABOUT THE DESIGN. The readings below are written before running this
script, but AFTER having read the verifier's exploratory numbers: the choice of
the two tests, the threshold of -2 SD on the AUC and the four analyses of the
origin come from there. None of these is therefore a prediction. The following
are new, and without numbers known in advance: the calibration at 5,000 data
sets; the like-for-like comparison in blocks of 199 realisations; the
distribution over 100 random removals; the two procedures (reference as in the
manuscript and re-standardised) for each analysis; the positive control of the
dip test; the run date, the between-well variability and the clinical profile
of the weak reactors.

PART 1 - ENLARGED CALIBRATION OF THE TWO TESTS
  Tests: T1 = 24 h, k = 2; T2 = 150 h, k = 3.
  Truth without groups and procedure: those of 121, with its functions and its
  generators. The outer data sets j = 0..399 are therefore the same as in
  121 and must give observed value, p and z identical to
  121_silhouette_outer.csv (identity checked; the script stops if it does
  not hold). The data sets j = 400..N_EST-1 are new. Inner B = 199, as in 121.
  Real data. The reference of the real data is generated with the same scheme
  as 121 (data set "-1") for N_BLOCKS x 199 realisations; the first block is
  that of 121 and must give its same p (identity checked).
  Designated statistic (one only, fixed here): the "like-for-like" calibrated
  p, that is, the mean, over the N_BLOCKS blocks of 199 realisations of the
  reference of the real data, of the proportion of outer data sets with nominal
  p <= that of the block, (count + 1)/(N + 1). It is the expected value of the
  calibrated p for a run at 199 realisations: same B and same discreteness on
  both sides. Interval: 2.5-97.5 percentiles over 2,000 resamplings of the
  outer data sets. Reported alongside, NOT designated: the calibrated p on the
  z (z of the real data from the whole reference, outer z at 199 realisations)
  and the one on the p of the real data at all the realisations.
  Multiplicity. The two tests were chosen because they had the smallest p
  values among the 18 of the curve: the selection is paid for by multiplying
  by 18 (Bonferroni). The whole arm is not recalibrated (all 18 tests would be
  needed for every outer data set).
  Reading, fixed here.
    - designated calibrated p x 18 < 0.05 in at least one of the two tests,
      with the upper limit of the interval x 18 also < 0.05: the silhouette
      of the real data exceeds the reference more than data sets without
      groups generated from that reference do, even after the correction.
    - p x 18 < 0.05 but upper limit x 18 >= 0.05, or p x 18 between 0.05
      and 0.10: "at the threshold"; neither the one nor the other is asserted.
    - p x 18 >= 0.10 in both: the departure does not withstand the correction.
    In no case does this demonstrate groups: it says how well the reference
    describes the data.

PART 2 - WHERE THE DEPARTURE COMES FROM
  For each of the two tests, every analysis computes the k-means silhouette
  (n_init = 50, seed of 45) on a matrix and compares it with B realisations
  of the Gaussian copula fitted to THAT matrix, in two procedures:
    M = as in the manuscript (realisations not re-standardised);
    R = realisations re-standardised like the data (finding 6 of the verifier).
  Procedure M is the one that defines the departure to be explained and is the
  main one; R is reported alongside. One-sided p, z = (observed - median
  of the reference)/SD.
  2a. The seven variables (repeats the published comparison with another stream).
  2b. Blocks: the four kinetic parameters alone; DAT (two variables) and age
      alone. The partition is recomputed on the block. Alongside, for the
      seven-variable partition, the proportion of between-group variance of
      each variable.
  2c. Normalised marginals: each column replaced by the normal scores of the
      ranks (Blom) and re-standardised. Three variants: all the variables; the
      kinetic parameters only; DAT and age only.
      WARNING fixed here: normalisation removes the skewness and the tails
      but also removes any bimodality of a single variable. That the
      departure disappears is therefore not enough to say "shape, not groups":
      2e is needed as well.
  2d. Weak reactors: participants with standardised log AUC below -3, -2.5,
      -2, -1.5 removed; the seven columns re-standardised. Comparison: the same
      number of participants as at the -2 threshold removed at random, 100 times.
  2e. Bimodality: dip statistic (Hartigan) on the projections along the axis
      joining the k-means centroids, maximum over the pairs of centroids;
      compared with the same computation on the realisations of the reference
      (the tabulated p of the dip does not hold after choosing the axis with
      k-means). Positive control: in the realisations of the reference a
      proportion of participants equal to the smallest group of the observed
      partition is shifted by 2, 3 and 4 SD along the axis of the two most
      distant centroids, and one counts how often the dip exceeds the 95th
      percentile of the reference.
      Without this number "no bimodality" cannot be read.
  2f. Who the weak reactors are at 24 h (log AUC < -2 SD): kinetic parameters;
      run date (number of distinct dates and maximum number of participants on
      one date, against 10,000 random subsets of the same sample size; period
      from October 2024); how many remain below -2 SD after centring the
      kinetic parameters on the run date; between-well variability
      (descriptive, with no fixed reading; the absolute dispersion was added
      after the dry run, in which the cv of the AUC turned out high in the
      weak ones); age, duration, stage, scales, DAT, CSF (standardised
      difference and Mann-Whitney, Benjamini-Hochberg over the set); how many
      are among the 45 shared participants and what AUC they have at 150 h.
  Readings, fixed here (per cohort).
    "The departure is shape of the distribution of the kinetic parameters" is
    written only if all hold: (i) block of the kinetic parameters with
    p < 0.05 and DAT-age block with p >= 0.05; (ii) with the kinetic parameters
    normalised p >= 0.05; (iii) dip with p >= 0.05 AND a positive control that
    at 3 SD detects at least 80% of the cases. If the positive control is below
    80%, (iii) is not informative and the sentence becomes "compatible with",
    not "is".
    "The departure at 24 h is carried by a tail of weak reactors" is written
    if, with those below -2 SD removed, p >= 0.05 and at most 5 random removals
    out of 100 give a z that low.
    On the nature of that tail: if the run dates are concentrated (p < 0.05
    on one of the two statistics) it is written that a technical origin is
    plausible; if not, that they are not concentrated by date. In neither of
    the two cases is it ruled out that they are a small subgroup: the
    silhouette does not distinguish the tail of a continuum from a small group.

What the script does NOT do: it does not recalibrate the whole second arm nor
the first; it does not try other truths without groups (the verifier tried
two, on a small scale: the calibrated p values depend on them); it does not
measure the power of the rule; it does not look at the plate (it is not in the
cohort file) nor at the run date at 150 h; it does not follow the weak reactors
over time.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 -B scripts/123_silhouette_departure_origin.py
      N_PROCESSES=n   (default 10; the results do not depend on it)
      PROVA=1        few data sets and few realisations, to measure the timings;
                     does not write the final outputs
Output: outputs/123_silhouette_departure_origin.txt, 123_calibration_two_test.csv,
        123_blocks_data_real.csv, 123_origin.csv, 123_removals_random.csv,
        123_reactors_weak.csv, 123_summary.csv; during execution
        123_partial.pkl
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

import diptest                                                   # noqa: E402
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
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "10")))


from _modules import load_script as _module                      # noqa: E402


M121 = _module("121_rule_calibration.py", 'calibration_121')
M118 = M121.M118
import _rule_functions                                                    # noqa: E402
C45 = M121.C45

SEED = 20261115                      # seed of this script (part 2)
TEST = {"24h": 2, "150h": 3}         # the two tests: cohort -> number of groups
N_TEST_CURVE = 18                    # tests of the silhouette curve (9 k x 2 cohorts)
B_INT = M121.B_INT_SIL if not TRIAL else 199
assert B_INT == 199
N_EST = 5000 if not TRIAL else 12
N_IDENT = 400 if not TRIAL else 6    # outer data sets in common with 121
N_BLOCKS = 100 if not TRIAL else 2
B_ORIG = 1999 if not TRIAL else 19
B_DIP = 999 if not TRIAL else 19
B_RAND, N_RAND = (499, 100) if not TRIAL else (9, 3)
N_CTRL = 200 if not TRIAL else 4
THRESHOLDS = [-3.0, -2.5, -2.0, -1.5]
THRESHOLD = -2.0
SHIFT = [2.0, 3.0, 4.0]
I_AUC = 2
KINET, REST = [0, 1, 2, 3], [4, 5, 6]
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


def crc(s):
    return crc32_legacy(s.encode())


def std(M):
    return M118.standardise(M)


def scores_normal(M, columns):
    """Normal scores of the ranks (Blom) on the given columns; everything re-standardised."""
    N = M.copy()
    n = len(M)
    for c in columns:
        N[:, c] = stats.norm.ppf((stats.rankdata(M[:, c]) - 0.375) / (n + 0.25))
    return std(N)


def kmeans(M, k, seed):
    return C45.KMeans(k, n_init=50, random_state=seed).fit(M)


def stat_sil(M, k, seed):
    return float(C45.silhouette_score(M, kmeans(M, k, seed).labels_))


def stat_dip(M, k, seed):
    """Maximum dip on the projections along the axes joining the centroids."""
    km = kmeans(M, k, seed)
    lab, C = km.labels_, km.cluster_centers_
    out = []
    for a in range(k):
        for b in range(a + 1, k):
            m = (lab == a) | (lab == b)
            d = C[b] - C[a]
            if m.sum() < 10 or np.linalg.norm(d) == 0:
                continue
            out.append(float(diptest.dipstat(M[m] @ d / np.linalg.norm(d))))
    return max(out) if out else np.nan


STAT = {"sil": stat_sil, "dip": stat_dip}


def job(t):
    kind = t['kind']
    t0 = time.time()
    if kind == "cal":
        # identical to 121 ("sil" task), for a single k
        tag, k, j = t["tag"], t["k"], t["j"]
        Y = M121.external(t["X"], tag, j)
        c, jj = crc(tag), j + 1
        obsv = stat_sil(Y, k, C45.SEED)
        nul = np.empty(B_INT)
        for b in range(B_INT):
            g = np.random.default_rng([M121.SEED, c, jj, 1, b])
            Z = C45.copula_null(Y, g)
            nul[b] = stat_sil(Z, k, int(g.integers(1e9)))
        p = ((nul >= obsv - 1e-12).sum() + 1) / (B_INT + 1)
        z = (obsv - np.median(nul)) / nul.std(ddof=1)
        return dict(kind=kind, tag=tag, k=k, j=j, obsv=obsv, p=float(p), z=float(z),
                    seconds=time.time() - t0)
    if kind == 'block':
        # reference of the real data, scheme of 121 for data set "-1" (jj = 0)
        tag, k, i = t["tag"], t["k"], t["i"]
        c = crc(tag)
        nul = np.empty(B_INT)
        for n_, b in enumerate(range(i * B_INT, (i + 1) * B_INT)):
            g = np.random.default_rng([M121.SEED, c, 0, 1, b])
            Z = C45.copula_null(t["X"], g)
            nul[n_] = stat_sil(Z, k, int(g.integers(1e9)))
        return dict(kind=kind, tag=tag, k=k, i=i, nul=nul, seconds=time.time() - t0)
    if kind == "orig":
        M, k, B, f = t["M"], t["k"], t["B"], STAT[t["stat"]]
        obsv = f(M, k, C45.SEED)
        c = crc(t['name'] + "|" + t["tag"])
        nm, nr = np.empty(B), np.empty(B)
        for b in range(B):
            g = np.random.default_rng([SEED, c, b])
            Z = C45.copula_null(M, g)
            s = int(g.integers(1e9))
            nm[b], nr[b] = f(Z, k, s), f(std(Z), k, s)
        return dict(kind=kind, name=t['name'], tag=t["tag"], stat=t["stat"], n=len(M), obsv=obsv,
                    nm=nm, nr=nr, seconds=time.time() - t0)
    if kind == "ctrl":
        # positive control: group injected into the realisations of the reference
        M, k, d, q, u = t["M"], t["k"], t["d"], t["q"], t["u"]
        c = crc("ctrl|" + t["tag"])
        sd = float((M @ u).std())
        dip, sil = np.empty(N_CTRL), np.empty(N_CTRL)
        for b in range(N_CTRL):
            g = np.random.default_rng([SEED, c, int(round(10 * d)), b])
            Z = C45.copula_null(M, g)
            m = g.random(len(Z)) < q
            Z[m] += d * sd * u
            Z = std(Z)
            s = int(g.integers(1e9))
            dip[b], sil[b] = stat_dip(Z, k, s), stat_sil(Z, k, s)
        return dict(kind=kind, tag=t["tag"], d=d, dip=dip, sil=sil, seconds=time.time() - t0)
    raise ValueError(kind)


def p_z(obsv, nul):
    p = ((nul >= obsv - 1e-12).sum() + 1) / (len(nul) + 1)
    return float(p), float((obsv - np.median(nul)) / nul.std(ddof=1))


def main():
    t0 = time.time()
    title('123 - SILHOUETTE AGAINST THE REFERENCE: CALIBRATION OF THE TWO TESTS AND ORIGIN OF THE DEPARTURE')
    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy())
    R150 = c150[M118.F150].to_numpy(float).copy()
    R150[:, :4] = np.log(R150[:, :4])
    X = {"24h": npz["F_primary"].astype(float), "150h": std(R150)}
    saved = np.load(os.path.join(OUT, "118_nulls.npz"), allow_pickle=True)
    for tag in X:
        assert np.abs(saved[f'V0_{tag}_matrix'] - X[tag]).max() < 1e-12, \
            f'{tag}: the matrix does not coincide with that analysed by 118 (V0)'
        assert np.abs(X[tag].mean(0)).max() < 1e-9 and np.abs(X[tag].std(0) - 1).max() < 1e-9
    if ONLY:
        assert (len(X["24h"]), len(X["150h"])) == (595, 333), "unexpected sample sizes for sporadic only"
    log(f"Declared working directory: {('sporadic only' if ONLY else 'full cohort')}; n = {len(X['24h'])} and {len(X['150h'])}. The matrices coincide with those of 118 and of 121.")
    log(f"Tests: 24 h k = {TEST['24h']}; 150 h k = {TEST['150h']}. Outer sets {N_EST}, inner B {B_INT}; reference of the real data {N_BLOCKS} x {B_INT}. Processes: {N_PROC}."
        + ('  [PROVA]' if TRIAL else ""))

    # ------------------------------------------------------------- observed --
    title("0. OBSERVED VALUES: IDENTITY WITH 45 AND WITH 120; STABILITY TO THE k-MEANS SEED")
    s45 = pd.read_csv(os.path.join(OUT, "45_silhouette_curves.csv"))
    s120 = pd.read_csv(os.path.join(OUT, "120_silhouette_B10000.csv"))
    obsv, part, pub = {}, {}, {}
    for tag, k in TEST.items():
        km = kmeans(X[tag], k, C45.SEED)
        obsv[tag] = float(C45.silhouette_score(X[tag], km.labels_))
        part[tag] = km
        r45 = s45[s45.cohort.str.startswith(tag) & (s45.series == "observed")][f"k={k}"]
        r120 = s120[(s120.variant == "V0") & (s120.cohort == tag) & (s120.k == k)].iloc[0]
        assert len(r45) == 1 and abs(float(r45.iloc[0]) - obsv[tag]) < 1e-12, f'{tag}: observed value different from 45'
        assert abs(float(r120.observed) - obsv[tag]) < 1e-12, f'{tag}: observed value different from 120'
        pub[tag] = dict(p2000=float(r120.p_B2000), p10000=float(r120.p_B10000))
        seeds = [stat_sil(X[tag], k, 1000 + s) for s in range(20)]
        sizes = np.bincount(km.labels_)
        log(f"{tag}, k = {k}: observed silhouette {obsv[tag]:.6f} (identical to 45 and to 120). Groups: {', '.join((str(x) for x in sorted(sizes, reverse=True)))}. With 20 other k-means seeds: {min(seeds):.6f}-{max(seeds):.6f}. Published p: {pub[tag]['p2000']:.4f} (2,000 realisations), {pub[tag]['p10000']:.4f} (10,000).")

    # --------------------------------------------------------------- tasks ---
    tasks = []
    for tag, k in TEST.items():
        for i in range(N_BLOCKS):
            tasks.append(dict(kind='block', tag=tag, k=k, i=i, X=X[tag]))
    weak, randoms = {}, {}
    for tag, k in TEST.items():
        M = X[tag]

        def agg(name, Mx, stat="sil", B=B_ORIG):
            tasks.append(dict(kind="orig", name=name, tag=tag, k=k, M=Mx, B=B, stat=stat))

        agg('seven variables', M)
        agg('kinetic parameters only', std(M[:, KINET]))
        agg('only DAT and age', std(M[:, REST]))
        agg('normalised: all', scores_normal(M, KINET + REST))
        agg('normalised: kinetic parameters only', scores_normal(M, KINET))
        agg('normalised: only DAT and age', scores_normal(M, REST))
        for s in THRESHOLDS:
            via = M[:, I_AUC] < s
            weak[(tag, s)] = via
            agg(f'removed AUC < {s:g} SD', std(M[~via]))
        n_via = int(weak[(tag, THRESHOLD)].sum())
        g = np.random.default_rng([SEED, crc(tag), 77])
        randoms[tag] = []
        for r in range(N_RAND):
            via = np.zeros(len(M), bool)
            via[g.choice(len(M), n_via, replace=False)] = True
            randoms[tag].append(via)
            agg(f'removed {n_via} at random #{r}', std(M[~via]), B=B_RAND)
        agg('dip: seven variables', M, stat="dip", B=B_DIP)
        agg('dip: kinetic parameters only', std(M[:, KINET]), stat="dip", B=B_DIP)
        # positive control of the dip: axis of the two most distant centroids
        C = part[tag].cluster_centers_
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        a, b = max(pairs, key=lambda ab: np.linalg.norm(C[ab[1]] - C[ab[0]]))
        u = (C[b] - C[a]) / np.linalg.norm(C[b] - C[a])
        q = float(np.bincount(part[tag].labels_).min() / len(M))
        for d in SHIFT:
            tasks.append(dict(kind="ctrl", tag=tag, k=k, M=M, d=d, q=q, u=u))
    n_fast = len(tasks)
    for tag, k in TEST.items():
        for j in range(-1, N_EST):
            tasks.append(dict(kind="cal", tag=tag, k=k, j=j, X=X[tag]))
    tasks[n_fast:] = sorted(tasks[n_fast:], key=lambda c: (c["j"], c["tag"]))

    result = []
    partial = os.path.join(OUT, "123_partial.pkl")
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        for i, r in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
            result.append(r)
            if (i + 1) % 250 == 0 or i + 1 == len(tasks):
                print(f'    {i + 1}/{len(tasks)}  ({time.time() - t0:.0f} s)', flush=True)
                if not TRIAL:
                    with open(partial, "wb") as fh:
                        pickle.dump(result, fh)
    sec = pd.DataFrame([dict(kind=r['kind'], s=r['seconds']) for r in result])
    log('\nCPU seconds per task (median): '
        + "; ".join(f"{k} {v:.0f}" for k, v in sec.groupby('kind').s.median().items()))

    summary = []

    # ============================================================ PART 1 =====
    title('1. ENLARGED CALIBRATION OF THE TWO TESTS')
    CAL = pd.DataFrame([{k: r[k] for k in ("tag", "k", "j", 'obsv', "p", "z")}
                        for r in result if r['kind'] == "cal"]).sort_values(["tag", "j"])
    s121 = pd.read_csv(os.path.join(OUT, "121_silhouette_outer.csv"))
    blocks_rows = []
    for tag, k in TEST.items():
        log(f"\n{tag}, k = {k}")
        e = CAL[(CAL.tag == tag) & (CAL.j >= 0)].reset_index(drop=True)
        v = CAL[(CAL.tag == tag) & (CAL.j < 0)].iloc[0]
        assert len(e) == N_EST and e.j.tolist() == list(range(N_EST))
        # identity with 121 on the data sets in common and on the real data
        a = s121[(s121.cohort == tag) & (s121.item == f"k={k}") & (s121.j < N_IDENT)].sort_values("j")
        b = CAL[(CAL.tag == tag) & (CAL.j < N_IDENT)].sort_values("j")
        assert a.j.tolist() == b.j.tolist() and len(a) == N_IDENT + 1
        for col_a, col_b in (('observed', 'obsv'), ("p", "p"), ("z", "z")):
            departure = float(np.abs(a[col_a].to_numpy() - b[col_b].to_numpy()).max())
            assert departure < 1e-9, f'{tag}: {col_a} different from 121 (difference {departure:.2e})'
        log(f"  Identity with 121: observed value, p and z coincide on the {N_IDENT} outer sets in "
            f"common and on the real data (difference < 1e-9).")
        assert abs(v.obsv - obsv[tag]) < 1e-12
        # level under the null
        n, kk = len(e), int((e.p < ALPHA).sum())
        lo, hi = M121.cp(kk, n)
        log(f"  Without groups, nominal p < 0.05 in {kk} of {n} ({100 * kk / n:.2f}%; 95% CI {100 * lo:.2f}-{100 * hi:.2f}%). The p values are multiples of 1/{B_INT + 1}: '< 0.05' means <= {(ALPHA * (B_INT + 1) - 1) / (B_INT + 1):.3f}. Quantiles 0.1 / 1 / 5 / 50%: "
            + " / ".join(f"{e.p.quantile(q):.3f}" for q in (0.001, 0.01, 0.05, 0.5))
            + f'. SD of z {e.z.std(ddof=1):.3f}, mean {e.z.mean():+.3f}.')
        # reference of the real data
        B = sorted((r for r in result if r['kind'] == 'block' and r["tag"] == tag), key=lambda r: r["i"])
        assert [r["i"] for r in B] == list(range(N_BLOCKS))
        pool_ = np.concatenate([r["nul"] for r in B])
        p_bl = np.array([((r["nul"] >= obsv[tag] - 1e-12).sum() + 1) / (B_INT + 1) for r in B])
        assert abs(p_bl[0] - v.p) < 1e-12, f'{tag}: the first block does not reproduce the p of 121'
        p_whole, z_whole = p_z(obsv[tag], pool_)
        pe = np.sort(e.p.to_numpy())

        def cal_p(pv, ordered=pe):
            return (np.searchsorted(ordered, pv + 1e-12, side="right") + 1) / (len(ordered) + 1)

        cal_bl = np.array([cal_p(x) for x in p_bl])
        designated = float(cal_bl.mean())
        g = np.random.default_rng([SEED, crc(tag), 5])
        boot = np.empty(2000 if not TRIAL else 50)
        for i in range(len(boot)):
            ps = np.sort(pe[g.integers(0, n, n)])
            boot[i] = np.mean([cal_p(x, ps) for x in p_bl])
        b_lo, b_hi = np.quantile(boot, [0.025, 0.975])
        kz = int((e.z >= z_whole - 1e-12).sum())
        cal_z = (kz + 1) / (n + 1)
        zlo, zhi = M121.cp(kz, n)
        cal_pt = float(cal_p(p_whole))
        log(f'  Real data: reference with {len(pool_)} realisations: p {p_whole:.4f}, z {z_whole:+.3f} (median {np.median(pool_):.4f}, SD {pool_.std(ddof=1):.4f}). The first block of {B_INT} reproduces the p of 121 ({v.p:.3f}). Over the {N_BLOCKS} blocks the p goes from {p_bl.min():.3f} to {p_bl.max():.3f} (median {np.median(p_bl):.3f}).')
        log(f'  DESIGNATED calibrated p (like for like, mean over the blocks): {designated:.5f} (95% CI {b_lo:.5f}-{b_hi:.5f}); per block from {cal_bl.min():.5f} to {cal_bl.max():.5f}.')
        log(f'    x {N_TEST_CURVE} (Bonferroni): {min(1, N_TEST_CURVE * designated):.4f} (upper limit {min(1, N_TEST_CURVE * b_hi):.4f}).')
        log(f'  Not designated: on the z {cal_z:.5f} ({kz} outer sets of {n} with z >= that of the real data; 95% CI of the share {zlo:.5f}-{zhi:.5f}; x {N_TEST_CURVE}: {min(1, N_TEST_CURVE * cal_z):.4f}); on the p with all the realisations {cal_pt:.5f} (x {N_TEST_CURVE}: {min(1, N_TEST_CURVE * cal_pt):.4f}).')
        bonf, bonf_hi = N_TEST_CURVE * designated, N_TEST_CURVE * b_hi
        reading = ('exceeds the reference even after the correction' if bonf < 0.05 and bonf_hi < 0.05
                   else 'at the threshold' if bonf < 0.10 else 'the departure does not survive the correction')
        log(f'  FIXED READING: {reading}.')
        summary.append(dict(portion=1, cohort=tag, k=k, item='share p < 0.05 without groups', value=kk / n,
                            lo=lo, hi=hi, n=n))
        summary.append(dict(portion=1, cohort=tag, k=k, item='designated calibrated p', value=designated,
                            lo=b_lo, hi=b_hi, n=n))
        summary.append(dict(portion=1, cohort=tag, k=k, item='designated calibrated p x 18',
                            value=min(1, bonf), lo=min(1, N_TEST_CURVE * b_lo), hi=min(1, bonf_hi), n=n))
        summary.append(dict(portion=1, cohort=tag, k=k, item='p calibrated on the z', value=cal_z,
                            lo=zlo, hi=zhi, n=n))
        summary.append(dict(portion=1, cohort=tag, k=k, item='p calibrated on the p at all realisations',
                            value=cal_pt, n=n))
        summary.append(dict(portion=1, cohort=tag, k=k, item='p of the real data, all realisations',
                            value=p_whole, n=len(pool_)))
        summary.append(dict(portion=1, cohort=tag, k=k, item='fixed reading', text=reading))
        for i in range(N_BLOCKS):
            blocks_rows.append(dict(cohort=tag, k=k, block=i, p=p_bl[i], p_calibrated=cal_bl[i]))

    # ============================================================ PART 2 =====
    O = {(r["tag"], r['name']): r for r in result if r['kind'] == "orig"}
    rows_orig, rows_rand = [], []

    def row(tag, name, show=True):
        r = O[(tag, name)]
        pm, zm = p_z(r['obsv'], r["nm"])
        pr, zr = p_z(r['obsv'], r["nr"])
        d = dict(cohort=tag, k=TEST[tag], analysis=name, statistic=r["stat"], n=r["n"], B=len(r["nm"]),
                 observed=r['obsv'], median_ref_M=float(np.median(r["nm"])), p_M=pm, z_M=zm,
                 median_ref_R=float(np.median(r["nr"])), p_R=pr, z_R=zr,
                 SD_ref_M_over_R=float(r["nm"].std(ddof=1) / r["nr"].std(ddof=1)))
        if show:
            rows_orig.append(d)
        return d

    readings2 = {}
    for tag, k in TEST.items():
        title(f'2. ORIGIN OF THE DEPARTURE - {tag}, k = {k}')
        M = X[tag]
        lab = part[tag].labels_
        eta = [1 - sum(((M[lab == c, v] - M[lab == c, v].mean()) ** 2).sum() for c in range(k))
               / ((M[:, v] - M[:, v].mean()) ** 2).sum() for v in range(M.shape[1])]
        log('Seven-variable partition: share of between-group variance per variable: '
            + "; ".join(f"{n_} {e_:.2f}" for n_, e_ in zip(M118.NAMES[tag], eta)))
        names = ['seven variables', 'kinetic parameters only', 'only DAT and age', 'normalised: all',
                'normalised: kinetic parameters only', 'normalised: only DAT and age'] \
            + [f'removed AUC < {s:g} SD' for s in THRESHOLDS]
        T = pd.DataFrame([row(tag, n_) for n_ in names])
        log('\n2a-2d. Silhouette against the reference (M = procedure of the manuscript; R = '
            're-standardised reference)')
        log(T[['analysis', "n", 'observed', 'median_ref_M', "p_M", "z_M", "p_R", "z_R",
               'SD_ref_M_over_R']].round(4).to_string(index=False))
        n_via = int(weak[(tag, THRESHOLD)].sum())
        rand = pd.DataFrame([row(tag, f'removed {n_via} at random #{r}', show=False) for r in range(N_RAND)])
        mir = row(tag, f'removed AUC < {THRESHOLD:g} SD', show=False)
        n_low = int((rand.z_M <= mir["z_M"] + 1e-12).sum())
        log(f"\n2d. With {n_via} subjects removed at random, {N_RAND} times (B = {B_RAND}): z_M median {rand.z_M.median():+.2f} (from {rand.z_M.min():+.2f} to {rand.z_M.max():+.2f}); p_M < 0.05 in {int((rand.p_M < ALPHA).sum())} of {N_RAND}. With the {n_via} with AUC < {THRESHOLD:g} SD removed: z_M {mir['z_M']:+.2f}, p_M {mir['p_M']:.4f}. Random removals with z_M as low: {n_low} of {N_RAND}.")
        rand['cohort'] = tag
        rows_rand.append(rand)
        # 2e
        log("\n2e. Bimodality along the axes between the centroids (maximum dip over the pairs)")
        D = pd.DataFrame([row(tag, n_) for n_ in ('dip: seven variables', 'dip: kinetic parameters only')])
        log(D[['analysis', "n", 'observed', 'median_ref_M', "p_M", "z_M", "p_R", "z_R"]]
            .round(4).to_string(index=False))
        ref_dip = O[(tag, 'dip: seven variables')]["nr"]
        ref_sil = O[(tag, 'seven variables')]["nr"]
        s_dip, s_sil = np.quantile(ref_dip, 0.95), np.quantile(ref_sil, 0.95)
        q = float(np.bincount(lab).min() / len(M))
        log(f"Positive control: injected group equal to {100 * q:.0f}% of the subjects, shifted "
            f"along the axis of the two most distant centroids; {N_CTRL} realisations per shift. "
            f"Thresholds: 95th percentile of the re-standardised reference of the real data "
            f"(approximation: the reference is not refitted to each injected set).")
        detected = {}
        for r in sorted((r for r in result if r['kind'] == "ctrl" and r["tag"] == tag), key=lambda r: r["d"]):
            fd, fs = float((r["dip"] > s_dip).mean()), float((r["sil"] > s_sil).mean())
            detected[r["d"]] = fd
            log(f"  shift {r['d']:g} SD: dip above threshold in {100 * fd:.0f}% of cases; silhouette "
                f"above threshold in {100 * fs:.0f}%.")
            summary.append(dict(portion=2, cohort=tag, k=k, item=f"positive control, {r['d']:g} SD: dip",
                                value=fd, n=N_CTRL))
            summary.append(dict(portion=2, cohort=tag, k=k,
                                item=f"positive control, {r['d']:g} SD: silhouette", value=fs, n=N_CTRL))
        # fixed readings
        g_ = T.set_index('analysis')
        c_i = g_.loc['kinetic parameters only', "p_M"] < ALPHA and g_.loc['only DAT and age', "p_M"] >= ALPHA
        c_ii = g_.loc['normalised: kinetic parameters only', "p_M"] >= ALPHA
        c_iii = D.set_index('analysis').loc['dip: seven variables', "p_M"] >= ALPHA
        informative = detected.get(3.0, 0.0) >= 0.80
        if c_i and c_ii and c_iii:
            form = ('the departure is shape of the distribution of the kinetic parameters' if informative else
                     'the departure is compatible with the shape of the distribution of the kinetic parameters (the bimodality test is not informative)')
        else:
            form = 'the conditions for attributing the departure to the shape of the kinetic parameters do NOT all hold'
        tail = mir["p_M"] >= ALPHA and n_low <= 5
        log(f"\nFIXED READINGS - {tag}: (i) kinetic parameters p < 0.05 and DAT-age p >= 0.05: {('si' if c_i else 'no')}; (ii) normalised kinetic parameters p >= 0.05: {('si' if c_ii else 'no')}; (iii) dip p >= 0.05: {('si' if c_iii else 'no')}, positive control at 3 SD >= 80%: {('si' if informative else 'no')}.")
        log(f'  -> {form}.')
        log(f"  -> tail of weak reactors (with AUC < {THRESHOLD:g} SD removed p >= 0.05 and at most 5 random removals out of 100 with z as low): {('si' if tail else 'no')}.")
        readings2[tag] = dict(form=form, tail=tail)
        summary.append(dict(portion=2, cohort=tag, k=k, item='fixed reading: shape', text=form))
        summary.append(dict(portion=2, cohort=tag, k=k, item='fixed reading: tail of weak reactors',
                            text="si" if tail else "no"))

    # ---------------------------------------------------------------- 2f -----
    title(f'2f. THE WEAK REACTORS AT 24 h (log AUC < {THRESHOLD:g} SD)')
    M = X["24h"]
    w = weak[("24h", THRESHOLD)]
    nw = int(w.sum())
    log(f'Subjects: {nw} of {len(M)} ({100 * nw / len(M):.1f}%). Standardised kinetic parameters, mean: '
        + "; ".join(f"{n_} {M[w, i].mean():+.2f}" for i, n_ in enumerate(M118.NAMES['24h'][:4]))
        + f". Group of the two-group k-means: "
          f"{int((part['24h'].labels_[w] == np.bincount(part['24h'].labels_).argmin()).sum())} of "
          f"{nw} in the smaller group.")
    data = pd.to_datetime(c24.RUNDATE)
    assert data.notna().all()
    code = pd.factorize(data)[0]
    n_date = len(np.unique(code))
    log(f'Run dates in the cohort: {n_date}; subjects per date: median {np.median(np.bincount(code)):.0f}, maximum {np.bincount(code).max()}.')

    def conc(m):
        c_ = np.bincount(code[m], minlength=n_date)
        return int((c_ > 0).sum()), int(c_.max())

    obsv_d, obsv_m = conc(w)
    g = np.random.default_rng([SEED, 24, 6])
    NP = 10000 if not TRIAL else 200
    perm = np.array([conc(g.choice(len(M), nw, replace=False)) for _ in range(NP)])
    p_d = ((perm[:, 0] <= obsv_d).sum() + 1) / (NP + 1)
    p_m = ((perm[:, 1] >= obsv_m).sum() + 1) / (NP + 1)
    log(f'Distinct dates among the weak reactors: {obsv_d} (random subsets: median {np.median(perm[:, 0]):.0f}; one-tailed p {p_d:.4f}). Maximum on a single date: {obsv_m} (median {np.median(perm[:, 1]):.0f}; p {p_m:.4f}).')
    late = (data >= pd.Timestamp("2024-10-01")).to_numpy()
    tab = [[int((w & late).sum()), int((w & ~late).sum())], [int((~w & late).sum()), int((~w & ~late).sum())]]
    log(f'Run from October 2024 onwards: {tab[0][0]} of {nw} among the weak, {tab[1][0]} of '
        f'{len(M) - nw} among the others (Fisher p {stats.fisher_exact(tab)[1]:.4f}).')
    Kc = M[:, KINET].copy()
    for c_ in np.unique(code):
        Kc[code == c_] -= Kc[code == c_].mean(0)
    Kc = std(Kc)
    log(f'After centring the kinetic parameters on the mean of the run date and re-standardising: {int((Kc[w, I_AUC] < THRESHOLD).sum())} of the {nw} remain below {THRESHOLD:g} SD (in the whole cohort below threshold: {int((Kc[:, I_AUC] < THRESHOLD).sum())}). Mean of the centred AUC in the {nw}: {Kc[w, I_AUC].mean():+.2f}.')
    concentrate = bool(p_d < ALPHA or p_m < ALPHA)
    log('FIXED READING: ' + ("the run dates are concentrated: a technical origin is plausible."
                              if concentrate else 'the weak reactors are not concentrated by run date.')
        + ' In neither case is a small subgroup excluded.')
    summary.append(dict(portion=2, cohort="24h", k=2, item='weak reactors: distinct dates, p', value=p_d, n=nw))
    summary.append(dict(portion=2, cohort="24h", k=2, item='weak reactors: maximum in one date, p', value=p_m, n=nw))

    var = ["saa_Fmax_cv", "saa_TTT_cv", "saa_AUC_cv", "saa_SLOPEMax_cv", "age_bl", "duration_yrs", "hy",
           "updrs3_score", "updrs_totscore", "moca", "upsit", "LEDD", "dat_putamen_low",
           "dat_caudate_mean", "abeta", "tau", "ptau", "NFL_CSF"]
    # absolute between-well SD of the AUC (cv x value): the cv alone grows when the value
    # is low
    c24 = c24.assign(saa_AUC_sd_wells=pd.to_numeric(c24.saa_AUC_cv, errors="coerce")
                     * pd.to_numeric(c24.saa_AUC, errors="coerce"))
    var.insert(4, 'saa_AUC_sd_wells')
    rows = []
    for v_ in var:
        a = pd.to_numeric(c24.loc[w, v_], errors="coerce").dropna()
        b = pd.to_numeric(c24.loc[~w, v_], errors="coerce").dropna()
        sd = b.std(ddof=1)
        rows.append(dict(variable=v_, n_weak=len(a), n_others=len(b), median_weak=a.median(),
                          median_others=b.median(),
                          diff_standardised=(a.mean() - b.mean()) / sd if sd > 0 else np.nan,
                          p=stats.mannwhitneyu(a, b).pvalue if len(a) > 1 and sd > 0 else np.nan))
    Dv = pd.DataFrame(rows)
    ok = Dv.p.notna()
    Dv.loc[ok, "p_BH"] = _rule_functions.bh(Dv.loc[ok, "p"].to_numpy())
    log('\nWeak against the others (difference of the means in SD of the others; Mann-Whitney; '
        'Benjamini-Hochberg over the rows):')
    log(Dv.round(4).to_string(index=False))
    log('Caveat: the coefficients of variation between wells (cv) grow mechanically when the value is low; saa_AUC_sd_wells = cv x value is the absolute dispersion. Rows with fewer than 5 weak reactors with the datum are not to be read. Descriptive comparisons, not provided for by a fixed reading.')
    if "SEX" in c24.columns:
        t_ = pd.crosstab(w, c24.SEX)
        log(f'Sex (table weak x SEX): {t_.to_numpy().tolist()}; Fisher p '
            f'{stats.fisher_exact(t_.to_numpy())[1]:.4f}.' if t_.shape == (2, 2) else f'Sex: {t_.to_dict()}')
    cond = np.isin(c24.PATNO.to_numpy(), c150.PATNO.to_numpy())
    log(f'\nAmong the {int(cond.sum())} shared with the 150 h cohort: {int((w & cond).sum())} weak '
        f'reactors.')
    if (w & cond).any():
        z150 = pd.Series(X["150h"][:, I_AUC], index=c150.PATNO.to_numpy())
        log('  Their standardised log AUC at 150 h: '
            + ", ".join(f"{z150[p_]:+.2f}" for p_ in c24.PATNO[w & cond]))

    title('WHAT THIS SCRIPT DOES NOT DO')
    log("- It does not recalibrate the whole second arm nor the first: only the two tests, corrected for 18.")
    log("- It uses a single truth without groups (the Gaussian copula fitted to the real data).")
    log('- It does not measure the power of the rule. The positive control of the dip uses '
        'thresholds from the reference of the real data, not refitted.')
    log('- The two tests, the threshold of -2 SD and the analyses of part 2 were chosen after '
        'seeing the exploratory numbers of the verifier of 121: they are not predictions.')
    log("- It does not look at the plate nor at the run date at 150 h; it does not follow the weak reactors over time.")
    log(f'\nTotal time: {time.time() - t0:.0f} s')
    if TRIAL:
        log('PROVA=1: no final output written.')
        return
    CAL.rename(columns={"tag": 'cohort', 'obsv': 'observed'}).to_csv(
        os.path.join(OUT, "123_calibration_two_test.csv"), index=False)
    pd.DataFrame(blocks_rows).to_csv(os.path.join(OUT, "123_blocks_data_real.csv"), index=False)
    pd.DataFrame(rows_orig).to_csv(os.path.join(OUT, "123_origin.csv"), index=False)
    pd.concat(rows_rand, ignore_index=True).to_csv(os.path.join(OUT, "123_removals_random.csv"), index=False)
    Dv.to_csv(os.path.join(OUT, "123_reactors_weak.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "123_summary.csv"), index=False)
    with open(os.path.join(OUT, "123_silhouette_departure_origin.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
