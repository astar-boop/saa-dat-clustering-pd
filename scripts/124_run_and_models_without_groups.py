"""
124 - The departure of the silhouette from the reference: the assay run date and
other models without groups.

Why it exists. Script 123 showed that, on the calibrated scale, the silhouette
of the real data exceeds the Gaussian copula reference at two points (24 h with
two groups, 150 h with three) and that the departure lies in the four kinetic
parameters. Its verifier (revisione_R2_rapporti/verifiche_script/riverifica_123.txt,
findings 1, 2 and 7) found, with exploratory analyses of its own, three things
that 123 does not look at: (1) the departure disappears when the kinetic
parameters are centred on the run date and remains with permuted dates; at 24 h
the weak reactors are in the slow runs; (2) the calibrated p depends heavily on
the model without groups from which the data sets are generated; (3) at 24 h
the departure is absent with the kinetic parameters not log-transformed. None
of these is in a script of the project. This script redoes them with its own
definitions and code.

HONESTY ABOUT THE DESIGN. The readings below are written before running this
script, but AFTER reading the verifier's exploratory numbers (centring on the
date: z +0.70 and +0.93, calibrated p 0.15 and 0.07, 0 permutations out of 20;
calibrated p under other models from 0.003 to 0.20). None is therefore a
prediction. New with respect to the verifier: the definition of the run date
(the sample that gave the kinetic parameters, not the first row of the file);
100 permutations instead of 20; from 1,000 to 2,000 data sets per model instead
of 300-600; the interval with joint resampling of data sets and blocks; the
one-factor models built here; the identity check of the calibration loop
against 121; the statistic without a cut-off for the weak reactors as the
designated one.

RUN DATE (definition, fixed here)
  24 h: RUNDATE of the cohort file, i.e. that of the first positive sample of
  type 1 at baseline (01_build_dataset.py). Checked against the raw file; the
  subjects whose kinetic parameters are the mean of samples run on different
  dates are counted.
  150 h: RUNDATE of the positive sample at baseline in the raw file (project
  155, COHORT PD), the same one from which 08_replication_150h.py takes the kinetic
  parameters; the match is checked by reconstructing Fmax (median of the
  replicates) and comparing it with the cohort file.
  Script 69 uses another definition (first row of the file, whatever the
  status): its between-run variance shares are reproduced with ITS definition
  (identity checked), then recomputed with this one.

PART A - THE RUN DATE
  A1. Centring. From the four kinetic parameters (log, standardised) the mean
      of their own run date is subtracted; the seven columns are
      re-standardised; k-means silhouette against the Gaussian copula fitted
      to that matrix, as in 123 (procedure M = as in the manuscript, main;
      R = re-standardised reference, alongside).
      Alongside, without a fixed reading: centring one kinetic parameter at a
      time.
  A2. Control. The same centring with the dates permuted among the subjects
      (same sample sizes per date), N_PERM times. It tells whether the
      centring lowers the silhouette because it removes the true date or
      because it removes variance anyway.
  A3. Calibration of the centred matrix ("GC"). Data sets without groups from
      the Gaussian copula fitted to the centred matrix, same procedure with
      199 realisations; reference of the centred data in blocks of 199.
      Designated calibrated p as in 123 (mean over the blocks of the
      like-for-like calibrated p).
  A4. A model without groups but with the runs ("GR"). Data sets = Gaussian
      copula fitted to the centred kinetic parameters (not re-standardised) +
      the observed run mean offsets, with the observed run sample sizes; then
      the procedure of the manuscript, which does not know the run. It tells
      whether MEAN differences between runs, without groups of subjects, are
      enough to produce a departure like that of the real data. The p of the
      real data is that of the 100 blocks of 123.
  A5. Weak reactors at 24 h (log AUC < -2 SD, as in 123) and run.
      Designated statistic, without cut-offs: for each subject the mean of the
      standardised log AUC of the OTHER subjects of the same date (the subject
      excluded); difference between the mean of this quantity in the weak
      reactors and in the others; two-sided p on 10,000 permutations of the
      "weak" label. Alongside, not designated: the same for the other three
      kinetic parameters; the verifier's cut-off (runs with mean log TTT > 0),
      which is post hoc.
  A6. At 150 h: association between the three-group partition and the run
      date (chi-square, p by permutation of the labels). Descriptive.
  Readings, fixed here (per cohort).
    "The departure is linked to the run date" is written if, once the kinetic
    parameters are centred, p_M >= 0.05 AND at most 5 permutations out of 100
    give a z this low. If the first holds and the second does not: "the
    centring removes the departure, but it would do so with any dates as
    well". If p_M < 0.05: "the departure remains after the centring".
    Calibrated scale: designated calibrated p of the centred data >= 0.05:
    "after the centring the silhouette no longer departs from the reference";
    < 0.05: "attenuated, not gone" (also reported x 18).
    GR: calibrated p of the real data under GR >= 0.05: "mean differences
    between runs, without groups, are enough to produce a departure of this
    size"; < 0.05: "the run mean offsets alone do not reproduce it".
    A5: p < 0.05: "the weak reactors were analysed in runs in which the other
    subjects also had weaker reactions"; otherwise it is not stated.
    In any case: centring on the date also removes the biological variance
    confounded with the date (samples analysed together were in part
    collected together): it is a limitation, not a correction. "Linked to the
    run" does not mean "artefact".

PART B - OTHER MODELS WITHOUT GROUPS
  For each of the two tests, data sets generated from another model and passed
  through the procedure of the manuscript (Gaussian copula reference fitted to
  the data set, 199 realisations). The p of the real data is that of the 100
  blocks of 123 (re-read from its output; identities checked). Models:
    T4  t copula with 4 degrees of freedom (correlations from Kendall's tau),
        empirical marginals. Stress test: dependent tails in all the
        variables.
    CN  curved one-factor continuum, normal latent: factor = first principal
        component of the normal scores of the four kinetic parameters; each
        kinetic parameter = cubic polynomial of the factor + correlated
        Gaussian residuals; DAT and age linear on the kinetic parameters +
        Gaussian residuals; then each column mapped back to the empirical
        marginals by ranks. The simulated factor is normal: no group by
        construction.
    CE  as CN, but the simulated factor is resampled from the observed values
        of the factor. WARNING fixed here: CE is NOT a model "without groups"
        in the strict sense. It reproduces the observed distribution along
        the main axis of the kinetic parameters, tail included: if the tail
        of the weak reactors were a small group, CE would contain it. CE only
        tells whether a single continuous factor with the observed
        distribution is enough to produce the departure; it does not tell
        what the tail is. The dip test on the factor is reported.
  Alongside, for each model: share of data sets with nominal p < 0.05; share
  of data sets with observed silhouette >= that of the real data.
  Identity of the loop: the same calibration loop, fed with the Gaussian truth
  and the generators of 121, must reproduce observed value, p and z of
  121_silhouette_outer.csv on the first N_ID data sets; the function of the
  calibrated p, applied to the data sets of 123, must reproduce the designated
  value of 123.
  Reading, fixed here (per cohort): with CN and T4 (CE is reported separately
  because of the warning) and the Gaussian copula of 123, "the departure after
  the correction for 18 holds for all the models tried" if p x 18 < 0.05 in
  all; "it holds only for some models" otherwise. In no case does a model in
  which the departure is ordinary prove that that model is true.

PART C - CHECKS ON THE SCALE AND ON THE TAIL
  C1. Kinetic parameters not log-transformed (the four raw columns,
      standardised): silhouette against the reference. Skewness of each
      kinetic parameter on the two scales.
  C2. Targeted removals: as many subjects removed as there are weak reactors
      at -2 SD (24 and 9), chosen as the most extreme of another variable
      (highest AUC; lowest Fmax or slope; longest time; lowest caudate). A
      stricter control than the random removals of 123.
  Descriptive, without a fixed reading.

What the script does NOT do: it does not recalibrate the other sixteen tests
nor the first arm; it does not measure the power of the rule; it does not look
at plate, centre or collection variables; it does not separate, in the run
date, the technical part from the biological one; it does not estimate the
degrees of freedom of the t copula; it does not analyse the contribution of
the single groups to the silhouette (finding 5d of the verifier).

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 -B scripts/124_run_and_models_without_groups.py
      N_PROCESSES=n   (default 10; the results do not depend on it)
      PROVA=1        few data sets and few realisations, to measure the times;
                     does not write the final outputs
Output: outputs/124_run_and_models.txt, 124_calibrations.csv,
        124_blocks_centred.csv, 124_origin.csv, 124_permutations_dates.csv,
        124_dates_run.csv, 124_summary.csv; during execution
        124_partial.pkl
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

import diptest                                                   # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import DATA                                       # noqa: E402
from _assay_run import row_of_kinetics

BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
TRIAL = os.environ.get('TRIAL', "0") == "1"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "10")))
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")


from _modules import load_script as _module                      # noqa: E402


M123 = _module("123_silhouette_departure_origin.py", 'origin_123')
M121, M118, _rule_functions, C45 = M123.M121, M123.M118, M123._rule_functions, M123.C45
std, crc, stat_sil, p_z, scores_normal = (M123.std, M123.crc, M123.stat_sil, M123.p_z,
                                             M123.scores_normal)

SEED = 20261116                      # seed of this script
TEST = dict(M123.TEST)               # {"24h": 2, "150h": 3}
N_TEST_CURVE = M123.N_TEST_CURVE     # 18
B_INT = M123.B_INT
assert B_INT == 199 and TEST == {"24h": 2, "150h": 3} and N_TEST_CURVE == 18
N_VER = ({"GC": 2000, "GR": 1000, "T4": 1000, "CN": 1000, "CE": 1000} if not TRIAL
         else {"GC": 4, "GR": 4, "T4": 4, "CN": 4, "CE": 4})
N_ID = 50 if not TRIAL else 3        # Gaussian data sets for the identity with 121
N_BLOCKS = 100 if not TRIAL else 2  # blocks of the reference of the centred data
B_ORIG = 1999 if not TRIAL else 19
B_PERM, N_PERM = (499, 100) if not TRIAL else (9, 3)
N_PERM_LABELS = 10000 if not TRIAL else 200
N_BOOT = 2000 if not TRIAL else 50
DOF_T = 4
THRESHOLD = M123.THRESHOLD                 # -2 SD on log AUC
I_AUC, KINET, REST = M123.I_AUC, list(M123.KINET), list(M123.REST)
ALPHA = 0.05
NAMES_VER = {"GC": 'Gaussian copula fitted to the kinetic parameters centred on the date',
            "GR": 'Gaussian copula + mean run deviations',
            "T4": 't copula, 4 degrees of freedom',
            "CN": 'curved one-factor continuum, normal latent',
            "CE": 'curved one-factor continuum, latent resampled from the observed'}
rep = []


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def eta2(y, g):
    """Between-group variance share, as in 69."""
    g = np.asarray(g)
    tot = np.sum((y - y.mean()) ** 2)
    between = sum(len(y[g == u]) * (y[g == u].mean() - y.mean()) ** 2 for u in np.unique(g))
    return float(between / tot)


def centre(M, code, columns):
    """Subtracts from the given columns the mean of their own date. Does not re-standardise."""
    A = M.copy()
    for c in np.unique(code):
        m = code == c
        A[np.ix_(m, columns)] -= A[np.ix_(m, columns)].mean(0)
    return A


def quantiles(X, U):
    """Empirical marginals, as in copula_null of 45."""
    return np.column_stack([np.quantile(X[:, j], U[:, j]) for j in range(X.shape[1])])


def corr_kendall(X):
    p = X.shape[1]
    T = np.eye(p)
    for a in range(p):
        for b in range(a + 1, p):
            T[a, b] = T[b, a] = stats.kendalltau(X[:, a], X[:, b]).statistic
    R = np.sin(np.pi * T / 2)
    w, V = np.linalg.eigh(R)
    R = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    return R / np.sqrt(np.outer(np.diag(R), np.diag(R)))


def fitmodel_factor(X):
    """One-factor model: factor = first principal component of the normal scores
    of the kinetic parameters; cubic per kinetic parameter; DAT and age linear on the
    kinetic parameters."""
    n = len(X)
    N = scores_normal(X, list(range(X.shape[1])))
    K, O = N[:, KINET], N[:, REST]
    _, _, vt = np.linalg.svd(K - K.mean(0), full_matrices=False)
    f = K @ vt[0]
    f = (f - f.mean()) / f.std()
    coef = np.array([np.polyfit(f, K[:, c], 3) for c in range(K.shape[1])])
    fit = np.column_stack([np.polyval(coef[c], f) for c in range(K.shape[1])])
    res_k = K - fit
    D = np.column_stack([np.ones(n), K])
    Bo = np.linalg.lstsq(D, O, rcond=None)[0]
    res_o = O - D @ Bo
    r2 = 1 - res_k.var(0) / K.var(0)
    return dict(f=f, coef=coef, Sk=np.cov(res_k, rowvar=False), Bo=Bo, So=np.cov(res_o, rowvar=False),
                r2=r2, asymmetry=float(stats.skew(f)), dip_p=float(diptest.diptest(f)[1]))


def generate_factor(par, X, g, empirical):
    n = len(X)
    f = par["f"][g.integers(0, n, n)] if empirical else g.standard_normal(n)
    K = np.column_stack([np.polyval(par["coef"][c], f) for c in range(len(KINET))])
    K = K + g.multivariate_normal(np.zeros(len(KINET)), par["Sk"], size=n)
    O = np.column_stack([np.ones(n), K]) @ par["Bo"]
    O = O + g.multivariate_normal(np.zeros(len(REST)), par["So"], size=n)
    S = np.empty((n, X.shape[1]))
    S[:, KINET], S[:, REST] = K, O
    U = (np.apply_along_axis(stats.rankdata, 0, S) - 0.5) / n
    return quantiles(X, U)


def generate(t, g):
    """A data set without groups from the model t['ver'], re-standardised."""
    ver, X = t["ver"], t["X"]
    if ver == "GC":
        return std(C45.copula_null(t["Xc"], g))
    if ver == "GR":
        return std(C45.copula_null(t["E"], g) + t["Mu"])
    if ver == "T4":
        n, p = X.shape
        Z = g.multivariate_normal(np.zeros(p), t["Rt"], size=n)
        w = np.sqrt(DOF_T / g.chisquare(DOF_T, size=n))[:, None]
        return std(quantiles(X, stats.t.cdf(Z * w, DOF_T)))
    if ver in ("CN", "CE"):
        return std(generate_factor(t["par"], X, g, empirical=(ver == "CE")))
    raise ValueError(ver)


def job(t):
    kind = t['kind']
    t0 = time.time()
    if kind == "cal":
        # calibration loop: one external data set, B_INT realisations of the reference
        # fitted to THAT data set. With ver = "G" it is the loop of 121, with its generators.
        ver, tag, k, j = t["ver"], t["tag"], t["k"], t["j"]
        if ver == "G":
            base, c = M121.SEED, crc(tag)
            Y = M121.external(t["X"], tag, j)
        else:
            base, c = SEED, crc(ver + "|" + tag)
            Y = generate(t, np.random.default_rng([base, c, j + 1, 0]))
        obsv = stat_sil(Y, k, C45.SEED)
        nul = np.empty(B_INT)
        for b in range(B_INT):
            g = np.random.default_rng([base, c, j + 1, 1, b])
            Z = C45.copula_null(Y, g)
            nul[b] = stat_sil(Z, k, int(g.integers(1e9)))
        p, z = p_z(obsv, nul)
        return dict(kind=kind, ver=ver, tag=tag, k=k, j=j, obsv=obsv, p=p, z=z, seconds=time.time() - t0)
    if kind == 'block':
        # reference of the centred data ("real" data set of GC), in blocks of B_INT
        tag, k, i = t["tag"], t["k"], t["i"]
        c = crc("GC|" + tag)
        nul = np.empty(B_INT)
        for n_, b in enumerate(range(i * B_INT, (i + 1) * B_INT)):
            g = np.random.default_rng([SEED, c, 0, 1, b])
            Z = C45.copula_null(t["Xc"], g)
            nul[n_] = stat_sil(Z, k, int(g.integers(1e9)))
        return dict(kind=kind, tag=tag, k=k, i=i, nul=nul, seconds=time.time() - t0)
    if kind == "orig":
        M, k, B = t["M"], t["k"], t["B"]
        obsv = stat_sil(M, k, C45.SEED)
        c = crc(t['name'] + "|" + t["tag"])
        nm, nr = np.empty(B), np.empty(B)
        for b in range(B):
            g = np.random.default_rng([SEED, c, 7, b])
            Z = C45.copula_null(M, g)
            s = int(g.integers(1e9))
            nm[b], nr[b] = stat_sil(Z, k, s), stat_sil(std(Z), k, s)
        return dict(kind=kind, name=t['name'], tag=t["tag"], family=t['family'], ordering=t['ordering'],
                    n=len(M), B=B,
                    obsv=obsv, nm=nm, nr=nr, seconds=time.time() - t0)
    raise ValueError(kind)


def p_calibrated(p_external, p_blocks):
    """Designated statistic of 123: mean over the blocks of (#{external p <= p of the block} + 1)/(N + 1)."""
    pe = np.sort(np.asarray(p_external, float))
    cal = (np.searchsorted(pe, np.asarray(p_blocks, float) + 1e-12, side="right") + 1) / (len(pe) + 1)
    return float(cal.mean()), cal


def interval(p_external, p_blocks, g):
    """2.5-97.5 percentiles resampling the external data sets and the blocks together."""
    pe, pb = np.asarray(p_external, float), np.asarray(p_blocks, float)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        out[i] = p_calibrated(pe[g.integers(0, len(pe), len(pe))], pb[g.integers(0, len(pb), len(pb))])[0]
    return tuple(float(x) for x in np.quantile(out, [0.025, 0.975]))


def perm_two_code(x, weak, g, n_perm):
    """Difference of the means (weak - others) and two-sided p by permutation of the label."""
    obsv = x[weak].mean() - x[~weak].mean()
    nd, cnt = int(weak.sum()), 0
    for _ in range(n_perm):
        m = np.zeros(len(x), bool)
        m[g.choice(len(x), nd, replace=False)] = True
        cnt += abs(x[m].mean() - x[~m].mean()) >= abs(obsv) - 1e-12
    return float(obsv), (cnt + 1) / (n_perm + 1)


def main():
    t0 = time.time()
    title('124 - THE DEPARTURE OF THE SILHOUETTE: RUN DATE AND OTHER MODELS WITHOUT GROUPS')
    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy())
    R24 = c24[M118.F24].to_numpy(float).copy()
    R150 = c150[M118.F150].to_numpy(float).copy()
    RAW = {"24h": std(R24), "150h": std(R150)}          # kinetic parameters not transformed
    L24, L150 = R24.copy(), R150.copy()
    L24[:, :4], L150[:, :4] = np.log(L24[:, :4]), np.log(L150[:, :4])
    X = {"24h": npz["F_primary"].astype(float), "150h": std(L150)}
    assert np.abs(std(L24) - X["24h"]).max() < 1e-9, '24h: the matrix cannot be reconstructed from the cohort file'
    saved = np.load(os.path.join(OUT, "118_nulls.npz"), allow_pickle=True)
    for tag in X:
        assert np.abs(saved[f'V0_{tag}_matrix'] - X[tag]).max() < 1e-12, \
            f'{tag}: the matrix does not coincide with that analysed by 118 (V0)'
    if ONLY:
        assert (len(X["24h"]), len(X["150h"])) == (595, 333), "unexpected sample sizes for sporadic only"
    cohorts = {"24h": c24, "150h": c150}
    log(f"Declared working directory: {('sporadic only' if ONLY else 'full cohort')}; n = {len(X['24h'])} and {len(X['150h'])}. The matrices coincide with those of 118, 121 and 123. Processes: {N_PROC}."
        + ('  [PROVA]' if TRIAL else ""))

    # observed values: identity with 45; p of the real data: blocks of 123, identity with
    # 121 and with 123
    s45 = pd.read_csv(os.path.join(OUT, "45_silhouette_curves.csv"))
    s121 = pd.read_csv(os.path.join(OUT, "121_silhouette_outer.csv"))
    b123 = pd.read_csv(os.path.join(OUT, "123_blocks_data_real.csv"))
    e123 = pd.read_csv(os.path.join(OUT, "123_calibration_two_test.csv"))
    summ123 = pd.read_csv(os.path.join(OUT, "123_summary.csv"))
    obsv, part, p_bl123, des123 = {}, {}, {}, {}
    for tag, k in TEST.items():
        km = M123.kmeans(X[tag], k, C45.SEED)
        part[tag] = km.labels_
        obsv[tag] = float(C45.silhouette_score(X[tag], km.labels_))
        r45 = s45[s45.cohort.str.startswith(tag) & (s45.series == "observed")][f"k={k}"]
        assert len(r45) == 1 and abs(float(r45.iloc[0]) - obsv[tag]) < 1e-12, f'{tag}: observed value different from 45'
        b = b123[(b123.cohort == tag) & (b123.k == k)].sort_values('block')
        assert b.block.tolist() == list(range(100)), f'{tag}: blocks of 123 incomplete'
        p_bl123[tag] = b.p.to_numpy(float)
        v121 = s121[(s121.cohort == tag) & (s121.item == f"k={k}") & (s121.j == -1)]
        assert len(v121) == 1 and abs(float(v121.p.iloc[0]) - p_bl123[tag][0]) < 1e-12, \
            f"{tag}: the first block of 123 is not the p of 121"
        e = e123[(e123.cohort == tag) & (e123.k == k) & (e123.j >= 0)]
        assert len(e) == 5000
        des123[tag] = p_calibrated(e.p.to_numpy(float), p_bl123[tag])[0]
        r = summ123[(summ123.cohort == tag) & (summ123.item == 'designated calibrated p')]
        assert len(r) == 1 and abs(float(r.value.iloc[0]) - des123[tag]) < 1e-12, \
            f'{tag}: the function of the calibrated p does not reproduce the designated value of 123'
    log(f"Identity: observed values equal to 45; the 100 blocks of 123 reread (the first is the p of "
        f"121); the function of the calibrated p of this script, applied to the 5,000 data sets of "
        f"123, reproduces their designated value ({des123['24h']:.5f} and {des123['150h']:.5f}).")

    # ---------------------------------------------------------- run date ---
    title('0. RUN DATE: DEFINITION, MATCHING, SHARES OF VARIANCE')
    saa = pd.read_csv(SAA, low_memory=False)
    v69 = pd.read_csv(os.path.join(OUT, "69_run_variance_components.csv"))
    CODE, DATE, date_rows = {}, {}, []
    spec = {"24h": dict(proj=237, sel=lambda s: (s.SAA_Status == "Positive") & (s.SAA_Type == "Type1"),
                        rep="Fmax_24h_Rep", col="saa_Fmax", agg="mean", lab69="24h"),
            "150h": dict(proj=155, sel=lambda s: s.SAA_Status == "Positive",
                         rep="Fmax_150h_Rep", col="k_Fmax", agg="median", lab69="150h")}
    for tag, sp in spec.items():
        c = cohorts[tag]
        s = saa[(saa.PROJECTID == sp["proj"]) & (saa.COHORT == "PD") & (saa.CLINICAL_EVENT == "BL")
                & saa.PATNO.isin(c.PATNO)].copy()
        pos = s[sp["sel"](s)].copy()
        v = pos[[f"{sp['rep']}{i}" for i in (1, 2, 3)]].apply(pd.to_numeric, errors="coerce")
        pos["_f"] = v.mean(axis=1) if sp["agg"] == "mean" else v.median(axis=1)
        gr = pos.groupby("PATNO").agg(data=("RUNDATE", "first"), n_date=("RUNDATE", "nunique"),
                                      n_rows=("RUNDATE", "size"), f=("_f", "mean"))
        gr = gr.reindex(c.PATNO.to_numpy())
        assert gr.data.notna().all(), f'{tag}: subjects without a positive sample in the raw file'
        departure = float(np.abs(gr.f.to_numpy() - c[sp["col"]].to_numpy(float)).max()
                       / np.abs(c[sp["col"]].to_numpy(float)).max())
        assert departure < 1e-9, f'{tag}: reconstructed Fmax differs from the cohort file ({departure:.2e})'
        data = pd.to_datetime(gr.data.to_numpy())
        if tag == "24h":
            assert (pd.to_datetime(c.RUNDATE).to_numpy() == data.to_numpy()).all(), \
                '24h: RUNDATE of the cohort file differs from the first positive sample of the raw file'
        DATE[tag] = data
        CODE[tag] = pd.Series(data).astype("category").cat.codes.to_numpy()
        cnt = np.bincount(CODE[tag])
        # identity with 69, with the definition of 69 (first row, whatever the status)
        s69 = saa[(saa.PROJECTID == sp["proj"]) & (saa.CLINICAL_EVENT == "BL")
                  & saa.PATNO.isin(c.PATNO)].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO").set_index("PATNO")
        d69 = s69.RUNDATE.reindex(c.PATNO.to_numpy()).to_numpy()
        L = L24 if tag == "24h" else L150
        r69 = v69[v69.cohort.str.startswith(sp["lab69"])].reset_index(drop=True)
        assert len(r69) == 4
        e69 = [eta2(L[:, j], d69) for j in range(4)]
        assert np.abs(np.array(e69) - r69.eta2_run.to_numpy()).max() < 1e-9, f'{tag}: eta2 differ from 69'
        e_mine = [eta2(L[:, j], CODE[tag]) for j in range(4)]
        different = int((pd.to_datetime(d69) != data).sum())
        log(f'{tag}: {len(c)} subjects, {len(cnt)} run dates (subjects per date: median {np.median(cnt):.0f}, maximum {cnt.max()}, dates with a single subject {int((cnt == 1).sum())}). Fmax reconstructed from the raw file identical to the cohort file. Subjects with more than one positive sample: {int((gr.n_rows > 1).sum())}; with samples on different dates (kinetic parameters averaged across assay runs): {int((gr.n_date > 1).sum())}. With the definition of 69 the dates are {pd.Series(d69).nunique()} and {different} subjects change date.')
        log('  Share of variance between dates (eta2), definition of 69 (identical to 69) | this definition: '
            + "; ".join(f'{M118.NAMES[tag][j]} {e69[j]:.3f} | {e_mine[j]:.3f}' for j in range(4))
            + f'. Expected by chance: {(len(cnt) - 1) / (len(c) - 1):.3f}.')
        for pat, d_, cc_ in zip(c.PATNO, data, CODE[tag]):
            date_rows.append(dict(cohort=tag, PATNO=int(pat), data_run=str(d_.date()), code=int(cc_)))

    # --------------------------------------------------------------- tasks ---
    tasks, perm_code, weak = [], {}, {}
    Xc, E, Mu, par, Rt = {}, {}, {}, {}, {}

    def agg(tag, family, name, Mx, B=B_ORIG):
        tasks.append(dict(kind="orig", tag=tag, k=TEST[tag], family=family, name=name, M=Mx, B=B,
                            ordering=len(tasks)))

    for tag, k in TEST.items():
        M, code = X[tag], CODE[tag]
        E[tag] = centre(M, code, KINET)
        Mu[tag] = M - E[tag]
        Xc[tag] = std(E[tag])
        agg(tag, "A1", 'kinetic parameters centred on the run date', Xc[tag])
        for j in KINET:
            agg(tag, "A1", f'centred only the {M118.NAMES[tag][j]}', std(centre(M, code, [j])))
        g = np.random.default_rng([SEED, crc(tag), 11])
        perm_code[tag] = [g.permutation(code) for _ in range(N_PERM)]
        for r, pc in enumerate(perm_code[tag]):
            agg(tag, "A2", f'centred on permuted dates #{r}', std(centre(M, pc, KINET)), B=B_PERM)
        agg(tag, "C1", 'kinetic parameters not log-transformed', RAW[tag])
        weak[tag] = M[:, I_AUC] < THRESHOLD
        nd = int(weak[tag].sum())
        targeted = [('higher AUC', np.argsort(-M[:, 2])), ('lower Fmax', np.argsort(M[:, 0])),
                  ('lower slope', np.argsort(M[:, 3])), ('longer time', np.argsort(-M[:, 1])),
                  ('lower caudate', np.argsort(M[:, 5]))]
        for name, ordering in targeted:
            via = np.zeros(len(M), bool)
            via[ordering[:nd]] = True
            agg(tag, "C2", f'removed the {nd} with {name}', std(M[~via]))
        for i in range(N_BLOCKS):
            tasks.append(dict(kind='block', tag=tag, k=k, i=i, Xc=Xc[tag]))
        par[tag] = fitmodel_factor(M)
        Rt[tag] = corr_kendall(M)
    n_fast = len(tasks)
    for tag, k in TEST.items():
        common = dict(kind="cal", tag=tag, k=k, X=X[tag])
        for j in range(-1, N_ID):
            tasks.append(dict(common, ver="G", j=j))
        for j in range(N_VER["GC"]):
            tasks.append(dict(common, ver="GC", j=j, Xc=Xc[tag]))
        for j in range(N_VER["GR"]):
            tasks.append(dict(common, ver="GR", j=j, E=E[tag], Mu=Mu[tag]))
        for j in range(N_VER["T4"]):
            tasks.append(dict(common, ver="T4", j=j, Rt=Rt[tag]))
        for ver in ("CN", "CE"):
            for j in range(N_VER[ver]):
                tasks.append(dict(common, ver=ver, j=j, par=par[tag]))
    tasks[n_fast:] = sorted(tasks[n_fast:], key=lambda c: (c["j"], c["ver"], c["tag"]))

    result = []
    partial = os.path.join(OUT, "124_partial.pkl")
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

    CAL = pd.DataFrame([{k: r[k] for k in ("ver", "tag", "k", "j", 'obsv', "p", "z")}
                        for r in result if r['kind'] == "cal"]).sort_values(["ver", "tag", "j"])
    rows_o = []
    for r in result:
        if r['kind'] != "orig":
            continue
        pm, zm = p_z(r['obsv'], r["nm"])
        pr, zr = p_z(r['obsv'], r["nr"])
        rows_o.append(dict(ordering=r['ordering'], cohort=r["tag"], k=TEST[r["tag"]], family=r['family'],
                            analysis=r['name'],
                            n=r["n"], B=r["B"], observed=r['obsv'], median_ref_M=float(np.median(r["nm"])),
                            p_M=pm, z_M=zm, median_ref_R=float(np.median(r["nr"])), p_R=pr, z_R=zr))
    ORIG = pd.DataFrame(rows_o).sort_values('ordering').drop(columns='ordering').reset_index(drop=True)
    summary, blocks_rows = [], []

    def row(portion, tag, item, value=np.nan, lo=np.nan, hi=np.nan, n=np.nan, text=""):
        summary.append(dict(portion=portion, cohort=tag, k=TEST[tag], item=item, value=value, lo=lo, hi=hi,
                            n=n, text=text))

    # ------------------------------------------- identity of the loop -------
    title("IDENTITY OF THE CALIBRATION LOOP WITH 121")
    for tag, k in TEST.items():
        a = s121[(s121.cohort == tag) & (s121.item == f"k={k}") & (s121.j < N_ID)].sort_values("j")
        b = CAL[(CAL.ver == "G") & (CAL.tag == tag)].sort_values("j")
        assert a.j.tolist() == b.j.tolist() and len(a) == N_ID + 1
        for col_a, col_b in (('observed', 'obsv'), ("p", "p"), ("z", "z")):
            sc = float(np.abs(a[col_a].to_numpy() - b[col_b].to_numpy()).max())
            assert sc < 1e-9, f'{tag}: {col_a} differs from 121 (discrepancy {sc:.2e})'
        log(f"{tag}: with the Gaussian truth and the generators of 121 the loop reproduces observed "
            f"value, p and z of 121 on the first {N_ID} data sets and on the real data (discrepancy "
            f"< 1e-9).")

    def orig(tag, name):
        r = ORIG[(ORIG.cohort == tag) & (ORIG.analysis == name)]
        assert len(r) == 1, (tag, name)
        return r.iloc[0]

    COLS = ['analysis', "n", 'observed', 'median_ref_M', "p_M", "z_M", "p_R", "z_R"]
    # =========================================================== PART A ======
    for tag, k in TEST.items():
        title(f'A. THE RUN DATE - {tag}, k = {k}')
        log('A1. Silhouette against the reference (M = procedure of the manuscript; R = re-standardised reference)')
        log(f'  Before centring (from 123, seven variables): observed {obsv[tag]:.4f}.')
        a1 = ORIG[(ORIG.cohort == tag) & (ORIG.family == "A1")]
        log(a1[COLS].round(4).to_string(index=False))
        c1 = orig(tag, 'kinetic parameters centred on the run date')
        pe = ORIG[(ORIG.cohort == tag) & (ORIG.family == "A2")]
        assert len(pe) == N_PERM
        n_low = int((pe.z_M <= c1.z_M + 1e-12).sum())
        log(f'\nA2. Centred on permuted dates, {N_PERM} times (B = {B_PERM}): z_M median {pe.z_M.median():+.2f} (from {pe.z_M.min():+.2f} to {pe.z_M.max():+.2f}); p_M < 0.05 in {int((pe.p_M < ALPHA).sum())} of {N_PERM}. With the true dates: z_M {c1.z_M:+.2f}, p_M {c1.p_M:.4f}. Permutations with z_M this low: {n_low} of {N_PERM}.')
        if c1.p_M < ALPHA:
            reading = 'the departure remains after the centring'
        elif n_low <= 5:
            reading = 'the departure is linked to the run date'
        else:
            reading = 'centring removes the departure, but would do so with arbitrary dates too'
        log(f'  FIXED READING: {reading}.')
        row("A", tag, 'centred on the date: z_M', c1.z_M, n=c1.n)
        row("A", tag, 'centred on the date: p_M', c1.p_M, n=c1.n)
        row("A", tag, 'permuted dates: median z_M', float(pe.z_M.median()), float(pe.z_M.min()),
             float(pe.z_M.max()), N_PERM)
        row("A", tag, 'permuted dates with z_M this low', n_low, n=N_PERM)
        row("A", tag, 'fixed reading: run date', text=reading)

        # A3: calibration of the centred matrix
        e = CAL[(CAL.ver == "GC") & (CAL.tag == tag)].reset_index(drop=True)
        assert len(e) == N_VER["GC"] and e.j.tolist() == list(range(N_VER["GC"]))
        Bk = sorted((r for r in result if r['kind'] == 'block' and r["tag"] == tag), key=lambda r: r["i"])
        assert [r["i"] for r in Bk] == list(range(N_BLOCKS))
        obsv_c = float(stat_sil(Xc[tag], k, C45.SEED))
        assert abs(obsv_c - c1.observed) < 1e-12
        pool_ = np.concatenate([r["nul"] for r in Bk])
        p_bl = np.array([((r["nul"] >= obsv_c - 1e-12).sum() + 1) / (B_INT + 1) for r in Bk])
        p_whole, z_whole = p_z(obsv_c, pool_)
        des, cal_bl = p_calibrated(e.p.to_numpy(), p_bl)
        lo, hi = interval(e.p.to_numpy(), p_bl, np.random.default_rng([SEED, crc("GC|" + tag), 5]))
        kk = int((e.p < ALPHA).sum())
        klo, khi = M121.cp(kk, len(e))
        kz = int((e.z >= z_whole - 1e-12).sum())
        log(f'\nA3. Calibration of the centred matrix: {len(e)} data sets without groups; nominal p < 0.05 in {kk} ({100 * kk / len(e):.2f}%; 95% CI {100 * klo:.2f}-{100 * khi:.2f}%). Centred data: reference with {len(pool_)} realisations: p {p_whole:.4f}, z {z_whole:+.3f}; over the {N_BLOCKS} blocks p from {p_bl.min():.3f} to {p_bl.max():.3f}.')
        log(f'  DESIGNATED calibrated p of the centred data: {des:.4f} (95% CI {lo:.4f}-{hi:.4f}); x {N_TEST_CURVE}: {min(1, N_TEST_CURVE * des):.3f}. On the z: {(kz + 1) / (len(e) + 1):.4f}. Before centring (123): {des123[tag]:.5f}.')
        reading = ('after the centring the silhouette no longer departs from the reference' if des >= ALPHA
                else 'attenuated, not vanished')
        log(f'  FIXED READING: {reading}.')
        row("A", tag, 'GC: share p < 0.05 without groups', kk / len(e), klo, khi, len(e))
        row("A", tag, 'GC: designated calibrated p of the centred data', des, lo, hi, len(e))
        row("A", tag, 'GC: p of the centred data, all realisations', p_whole, n=len(pool_))
        row("A", tag, 'fixed reading: calibrated scale after the centring', text=reading)
        for i, (pb, cb) in enumerate(zip(p_bl, cal_bl)):
            blocks_rows.append(dict(cohort=tag, k=k, block=i, p=pb, p_calibrated=cb))

    # A5: weak reactors and run at 24 h
    title('A5. WEAK REACTORS AT 24 H (log AUC < -2 SD) AND ASSAY RUN')
    tag = "24h"
    M, code, wk = X[tag], CODE[tag], weak[tag]
    cnt = np.bincount(code)
    ok = cnt[code] >= 2
    total = np.zeros((cnt.size, 4))
    np.add.at(total, code, M[:, KINET])
    others = np.full((len(M), 4), np.nan)
    others[ok] = (total[code[ok]] - M[ok][:, KINET]) / (cnt[code[ok]] - 1)[:, None]
    log(f'Weak reactors: {int(wk.sum())} of {len(M)}; on dates with a single subject (excluded from the comparison): {int((wk & ~ok).sum())} weak, {int((~wk & ~ok).sum())} others.')
    g = np.random.default_rng([SEED, crc('weak'), 3])
    for j in [I_AUC] + [x for x in KINET if x != I_AUC]:
        d_, p_ = perm_two_code(others[ok, j], wk[ok], g, N_PERM_LABELS)
        des_ = 'DESIGNATED' if j == I_AUC else 'not designated'
        log(f'  {M118.NAMES[tag][j]} of the other subjects of the same date ({des_}): mean in the weak {others[ok & wk, j].mean():+.3f}, in the others {others[ok & ~wk, j].mean():+.3f}; difference {d_:+.3f}; two-tailed p {p_:.4f} ({N_PERM_LABELS} permutations).')
        row("A", tag, f'weak and run: {M118.NAMES[tag][j]} of the others, difference', d_, n=int(ok.sum()))
        row("A", tag, f'weak and run: {M118.NAMES[tag][j]} of the others, p', p_, n=int(ok.sum()))
        if j == I_AUC:
            p_des = p_
    reading = ('the weak reactors were analysed in runs in which the other subjects also had weaker reactions' if p_des < ALPHA else 'no link between weak reactors and run is claimed')
    log(f'  FIXED READING: {reading}.')
    row("A", tag, 'fixed reading: weak reactors and run', text=reading)
    mean_ttt = np.array([M[code == c, 1].mean() for c in range(cnt.size)])
    slow = mean_ttt[code] > 0
    tab = [[int((wk & slow).sum()), int((wk & ~slow).sum())],
           [int((~wk & slow).sum()), int((~wk & ~slow).sum())]]
    log(f'  Cut of the verifier (post hoc; dates with mean log TTT > 0: {int((mean_ttt > 0).sum())} dates, {int(slow.sum())} subjects): weak {tab[0][0]} in the slow ones and {tab[0][1]} in the others; not weak {tab[1][0]} and {tab[1][1]}; Fisher p {stats.fisher_exact(tab)[1]:.4f}.')
    Cc = Xc[tag]
    log(f'  After centring on the date (columns re-standardised): {int((Cc[wk, I_AUC] < THRESHOLD).sum())} of the {int(wk.sum())} remain below -2 SD; in the whole cohort below threshold {int((Cc[:, I_AUC] < THRESHOLD).sum())}.')

    # A6: three-group partition and date at 150 h
    title('A6. PARTITION AND RUN DATE (descriptive)')
    for tag, k in TEST.items():
        lab, code = part[tag], CODE[tag]
        ct = pd.crosstab(code, lab).to_numpy()
        chi = float(stats.chi2_contingency(ct, correction=False)[0])
        g = np.random.default_rng([SEED, crc("chi|" + tag), 4])
        more = sum(float(stats.chi2_contingency(pd.crosstab(code, g.permutation(lab)).to_numpy(),
                                               correction=False)[0]) >= chi - 1e-9
                  for _ in range(N_PERM_LABELS // 5))
        p_ = (more + 1) / (N_PERM_LABELS // 5 + 1)
        log(f'{tag}, k = {k}: partition x run date, chi-square {chi:.1f}; p by permutation of the labels {p_:.4f} ({N_PERM_LABELS // 5} permutations).')
        row("A", tag, 'partition x date: p by permutation', p_, n=len(lab))

    # =========================================================== PART B ======
    title('B. OTHER MODELS WITHOUT GROUPS (and GR, the model with the assay runs)')
    for tag, k in TEST.items():
        pr = par[tag]
        log(f'\n{tag}, k = {k}. One-factor model: R2 of the cubic on the factor: '
            + "; ".join(f"{M118.NAMES[tag][j]} {pr['r2'][i]:.2f}" for i, j in enumerate(KINET))
            + f". Observed factor: skewness {pr['asymmetry']:+.2f}, dip p {pr['dip_p']:.2f}.")
        tab, pv = [], {"G (123)": des123[tag]}
        tab.append(dict(model='G (123, 5,000 sets)', datasets=5000,
                        rejections_nominals=float(summ123[(summ123.cohort == tag)
                                                      & (summ123.item == 'share p < 0.05 without groups')].value.iloc[0]),
                        p_calibrated=des123[tag], lo=np.nan, hi=np.nan,
                        x18=min(1, N_TEST_CURVE * des123[tag]), silhouette_ge_data=np.nan))
        for ver in ("T4", "CN", "CE", "GR"):
            e = CAL[(CAL.ver == ver) & (CAL.tag == tag)].reset_index(drop=True)
            assert len(e) == N_VER[ver] and e.j.tolist() == list(range(N_VER[ver]))
            des, _ = p_calibrated(e.p.to_numpy(), p_bl123[tag])
            lo, hi = interval(e.p.to_numpy(), p_bl123[tag], np.random.default_rng([SEED, crc(ver + "|" + tag), 5]))
            kk = int((e.p < ALPHA).sum())
            klo, khi = M121.cp(kk, len(e))
            similar = float((e.obsv >= obsv[tag] - 1e-12).mean())
            pv[ver] = des
            tab.append(dict(model=ver, datasets=len(e), rejections_nominals=kk / len(e), p_calibrated=des, lo=lo,
                            hi=hi, x18=min(1, N_TEST_CURVE * des), silhouette_ge_data=similar))
            row("B", tag, f'{ver}: share p < 0.05 without groups', kk / len(e), klo, khi, len(e))
            row("B", tag, f'{ver}: designated calibrated p of the real data', des, lo, hi, len(e))
            row("B", tag, f'{ver}: calibrated p x 18', min(1, N_TEST_CURVE * des), min(1, N_TEST_CURVE * lo),
                 min(1, N_TEST_CURVE * hi), len(e))
            row("B", tag, f'{ver}: share of sets with silhouette >= real data', similar, n=len(e))
        log(pd.DataFrame(tab).round(4).to_string(index=False))
        log('  (resolution floor: 1/(data sets + 1); models: ' +
            "; ".join(f'{v} = {NAMES_VER[v]}' for v in ("T4", "CN", "CE", "GR")) + ")")
        everyone = all(N_TEST_CURVE * pv[v] < ALPHA for v in ("G (123)", "T4", "CN"))
        reading = ('the departure after the correction for 18 holds for all the models tried (G, T4, CN)' if everyone
                else 'the departure after the correction for 18 holds only for some models (G, T4, CN)')
        log(f'  FIXED READING (models without groups G, T4, CN; CE apart): {reading}.')
        row("B", tag, 'fixed reading: models without groups', text=reading)
        reading = ('mean differences between runs, without groups, are enough to produce a departure of this size'
                if pv["GR"] >= ALPHA else 'the mean run offsets alone do not reproduce it')
        log(f'  FIXED READING (GR): {reading}.')
        row("B", tag, 'fixed reading: GR', text=reading)

    # =========================================================== PART C ======
    title('C. CHECKS ON THE SCALE AND ON THE TAIL (descriptive)')
    for tag, k in TEST.items():
        L, G = X[tag], RAW[tag]
        log(f'\n{tag}, k = {k}. Skewness of the kinetic parameters, logarithm | untransformed: '
            + "; ".join(f"{M118.NAMES[tag][j].replace('log ', '')} {stats.skew(L[:, j]):+.2f} | {stats.skew(G[:, j]):+.2f}" for j in KINET))
        cc = ORIG[(ORIG.cohort == tag) & ORIG.family.isin(["C1", "C2"])]
        log(cc[COLS].round(4).to_string(index=False))
        r = orig(tag, 'kinetic parameters not log-transformed')
        row("C", tag, 'untransformed kinetic parameters: z_M', r.z_M, n=r.n)
        row("C", tag, 'untransformed kinetic parameters: p_M', r.p_M, n=r.n)

    title('WHAT THIS SCRIPT DOES NOT DO')
    log("- It does not recalibrate the other sixteen tests or the first arm; it does not measure the power of the rule.")
    log("- It does not distinguish, in the run date, the technical part from the biological one: "
        "centring is a bound, not a correction.")
    log("- The models T4, CN, CE and GR are choices of this script: a model in which the departure "
        "is ordinary is not for that reason true. CE contains the observed tail and is not 'without "
        "groups' in the strict sense.")
    log('- The analyses were chosen after seeing the exploratory numbers of the verifier of 123.')
    log('- It does not look at plate, centre or variables of the sample collection.')
    log(f'\nTotal time: {time.time() - t0:.0f} s')

    if TRIAL:
        log('[PROVA] final outputs not written.')
        return
    CAL.rename(columns={"ver": 'model', "tag": 'cohort', 'obsv': 'observed'}).to_csv(
        os.path.join(OUT, "124_calibrations.csv"), index=False)
    pd.DataFrame(blocks_rows).to_csv(os.path.join(OUT, "124_blocks_centred.csv"), index=False)
    ORIG[ORIG.family != "A2"].to_csv(os.path.join(OUT, "124_origin.csv"), index=False)
    ORIG[ORIG.family == "A2"].to_csv(os.path.join(OUT, "124_permutations_dates.csv"), index=False)
    pd.DataFrame(date_rows).to_csv(os.path.join(OUT, "124_dates_run.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "124_summary.csv"), index=False)
    with open(os.path.join(OUT, "124_run_and_models.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
