"""
117 - Do the participants shared by the two cohorts end up in the same group
under the two protocols?

Why it exists. Reviewer 2 ("Different protocols") asks for the shared
participants to be analysed separately, to see whether the results are
consistent between the two protocols in the same individuals. Script 106 has
already measured the concordance of the individual kinetic parameters
(Table S16). Here the question is about the partitions: do the partitions
estimated on the two cohorts put the same shared participants in the same
groups?

Why not the battery on the shared participants alone. The shared participants
are a few dozen, the variables seven: on a set that small the battery and its
null reference distinguish nothing. Each partition is therefore estimated on
the WHOLE cohort (as in the manuscript) and then read on the shared
participants.

HISTORY OF THE SCRIPT (declared). This is the second draft. The first
(kept in revisione_R2_rapporti/versioni_script/117_prima_della_verifica.py
with its outputs) was verified by an independent checker
(revisione_R2_rapporti/verifiche_script/riverifica_117.txt), who reproduced all
its numbers and found flaws of method. Those taken up here:
  - the permutation p, for two partitions into two groups, is an exact test
    with fixed margins: it is now computed exactly. Some rows cannot give
    p < 0.05 with any table: this is now stated and they do not enter the
    correction;
  - the "95th percentile of chance" was not the significance threshold
    (the distribution is discrete): it is replaced by the minimum number of
    participants in the same group that would give p < 0.05 and by the number
    expected by chance;
  - the interval of the ARI (first bootstrap, then symmetric jackknife) gave
    impossible limits: it is removed; the jackknife standard error remains in
    the CSV;
  - the power of the test was not declared: it is now simulated;
  - the benchmark "shared variables only" is not the right one for the
    partitions at k = 2, which hardly use DAT and age: the "same protocol"
    ceiling is added (two disjoint halves of the same cohort);
  - the agreement at k = 4 on the seven variables was attributed to the
    identical variables only on grounds of plausibility: there is now the
    control that decides it;
  - the principal components were paired by rank and not by content:
    they are replaced by the canonical correlations between the kinetic
    parameters;
  - the shared participants are not a random sample of the 24-hour cohort,
    and at 24 hours the run date matters: they are now described and the
    comparison is repeated after centring the 24-hour kinetic parameters on
    the run date.
All the choices of this draft were made KNOWING the results of the first one
and those of the checker. They are not blind analyses.

What it computes.
  0. The shared participants against the rest of each cohort: difference in
     means in standard deviations for the seven variables (Mann-Whitney),
     duration and stage; at 24 hours the run date (the 150-hour cohort file
     does not contain it).
  1. Partitions at k = 2: the four algorithms and the constructors of 25
     (function labels_k2 of _rule_functions). Identity: the mean of the ARIs between
     the four partitions of each cohort must reproduce "ARI_algorithms_k2" of
     25_nulls.csv. Three sets of variables:
       (a) the seven of the manuscript; (b) the four kinetic parameters only;
       (c) the three shared variables only (DAT and age), a control on
           identical measures: it is not a test.
     For each row, on the shared participants: 2 x 2 table, participants in
     the same group (maximum over the two matchings of the labels), number
     expected by chance, ARI, exact one-tailed p (tables with fixed margins
     with ARI >= observed), minimum attainable p, minimum number of
     participants in the same group to have p < 0.05. A row is "degenere" if
     in one protocol the smaller group has fewer than two shared participants;
     it is "senza potenza" if the minimum attainable p is >= 0.05.
     Benjamini-Hochberg on the rows of (a) and (b) that are neither degenerate
     nor without power. Since at k = 2 the partitions of (a) and of (b) almost
     coincide, the script reports the ARI between the two: the independent
     tests are fewer than those counted.
  2. Power: two normal latent scores with correlation rho between the two
     protocols, cut so as to reproduce the observed group sizes (in each
     protocol the smaller group is that of the higher scores); share of
     replicates with exact p < 0.05. At the first execution of this draft
     group "1" of each protocol was the one with label 1, which is arbitrary:
     in three rows the minority of one protocol was paired with the majority
     of the other and the power came out as zero. Corrected before using the
     numbers.
  3. "Same protocol" ceiling: the non-shared members of a cohort are split
     at random into two halves; the shared participants are added to each;
     the partitions are estimated separately on the two halves and read on
     the shared participants. It says how much agreement is obtained when the
     measure is the same and only the other patients change.
  4. More than two groups: k-means at k = 3 and k = 4 (constructor of 45),
     permutation p. For k = 4 on the seven variables, attribution control:
     among the shared participants a block of variables (the three shared
     ones, or the four kinetic parameters) is permuted in one cohort, the
     partition is re-estimated on the whole cohort and read again on the
     shared participants.
  5. Without partitions: canonical correlations between the four kinetic
     parameters at 24 hours and the four at 150 hours in the shared
     participants, with a permutation test on Pillai's trace; repeated after
     centring the 24-hour kinetic parameters on the mean of their run date
     (means estimated on the whole cohort).

Reading, fixed at the first draft and corrected after the verification. If
there are no groups a partition at k = 2 is an arbitrary cut of a continuous
cloud, and two cuts made on different kinetic variables have no reason to
coincide: agreement at chance level is what is expected BOTH if there are no
groups AND if the two protocols do not order the same patients in the same way.
The result does not distinguish the two things. Agreement above chance IN THE
KINETIC PARAMETERS ALONE would say that something stable passes from one
protocol to the other; agreement in the seven variables can come from DAT and
age, which in the shared participants are the same measures, and must be read
with the control of point 4.

What the script does NOT do: it does not run the battery on the shared
participants alone; it does not use the 229 members of the 150-hour cohort with
a 24-hour sample at one year (they are in 106); it does not correct
the secondary p values; it does not control for the run date at 150 hours; no
interval or p takes into account the uncertainty with which the partitions are
estimated.

Usage:  CLUSTERING_BASE=<radice> ONLY_SPORADIC=1 python3 -B scripts/117_shared_partitions.py
Output: outputs/117_shared_partitions.txt, 117_shared_partitions.csv,
        117_shared_sample.csv, 117_shared_power.csv,
        117_shared_ceiling.csv, 117_shared_k4.csv, 117_shared_canonical.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util                                            # noqa: E402
import sys                                                       # noqa: E402
import warnings                                                  # noqa: E402
import zlib                                                      # noqa: E402
from _legacy import crc32_legacy                                # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans      # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari           # noqa: E402
from sklearn.mixture import GaussianMixture                      # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"


import _rule_functions                                                    # noqa: E402



C45 = _rule_functions.C45
SEED_B = _rule_functions.SEED_B

SEED = 20261113
NPERM = 20000
N_POWER = 4000
N_CEILING = 200
N_K4 = 200
RHO = (0.3, 0.5, 0.7, 0.9)
ALPHA = 0.05
F24 = ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax",
       "dat_putamen_low", "dat_caudate_mean", "age_bl"]
F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE",
        "dat_putamen_low", "dat_caudate_mean", "age_bl"]
NAMES = {"24h": ["log Fmax", "log TTT", "log AUC", "log SLOPEMax",
                'lower putamen', 'mean caudate', 'age'],
        "150h": ["log Fmax", "log T50", "log AUC", "log SLOPE",
                 'lower putamen', 'mean caudate', 'age']}
COMMON = ["dat_putamen_low", "dat_caudate_mean", "age_bl"]
DATASETS = {'a. seven variables': list(range(7)),
           'b. kinetic parameters only': [0, 1, 2, 3],
           'c. shared variables only (DAT, age)': [4, 5, 6]}
TEST = ['a. seven variables', 'b. kinetic parameters only']
ALG = ["kmeans", "ward", "gmm"]          # average linkage is reported separately

rep = []


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def gen(label):
    return np.random.default_rng([SEED, crc32_legacy(label.encode())])


def standardise(M):
    return (M - M.mean(0)) / M.std(0)


def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def parts_k2(X):
    """The four partitions at k = 2 with the constructors of 25 (via _rule_functions)."""
    return _rule_functions.labels_k2(X)


def three_k2(X):
    """k-means, Ward and mixture at k = 2, same constructors (without average linkage)."""
    return {"kmeans": KMeans(2, n_init=20, random_state=SEED_B).fit_predict(X),
            "ward": AgglomerativeClustering(2, linkage="ward").fit_predict(X),
            "gmm": GaussianMixture(2, covariance_type="full", n_init=5,
                                   random_state=SEED_B).fit_predict(X)}


def kmeans_k(X, k):
    return KMeans(k, n_init=50, random_state=C45.SEED).fit_predict(X)


# ------------------------------------------------ exact test for 2 x 2 tables
def tables_exacts(r1, c1, n):
    """All the 2 x 2 tables with fixed margins (r1 participants in group 1 at
    24 hours, c1 in group 1 at 150 hours): for each one n11, hypergeometric
    probability, ARI and number of participants in the same group."""
    x = np.arange(max(0, r1 + c1 - n), min(r1, c1) + 1)
    pr = stats.hypergeom.pmf(x, n, r1, c1)
    la = np.r_[np.ones(r1, int), np.zeros(n - r1, int)]
    out = []
    for xi in x:
        lb = np.r_[np.ones(xi, int), np.zeros(r1 - xi, int),
                   np.ones(c1 - xi, int), np.zeros(n - r1 - c1 + xi, int)]
        acc = int((la == lb).sum())
        out.append((int(xi), float(ari(la, lb)), max(acc, n - acc)))
    t = pd.DataFrame(out, columns=["n11", "ARI", 'same'])
    t["pr"] = pr
    # one-tailed p: probability of the tables with ARI >= that of the table
    t["p"] = [float(t.pr[t.ARI >= a - 1e-12].sum()) for a in t.ARI]
    return t


def exact(la, lb):
    n = len(la)
    r1, c1 = int(la.sum()), int(lb.sum())
    t = tables_exacts(r1, c1, n)
    n11 = int(((la == 1) & (lb == 1)).sum())
    row = t[t.n11 == n11].iloc[0]
    sig = t[t.p < ALPHA]
    return dict(ARI=float(row.ARI), same_group=int(row.same),
                expected_by_chance=float((t.same * t.pr).sum()),
                p_exact=float(row.p), p_minimum_reachable=float(t.p.min()),
                same_group_minimum_by_p05=(int(sig.same.min()) if len(sig) else np.nan),
                rejection_below_the_chance=float(t.pr[t.p < ALPHA].sum())), t


def se_jackknife(la, lb):
    n = len(la)
    jk = np.array([ari(np.delete(la, i), np.delete(lb, i)) for i in range(n)])
    return float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))


# ------------------------------------------------------- canonical correlations
def canonical(A, B):
    """Canonical correlations between two blocks (centred columns)."""
    qa, _ = np.linalg.qr(A - A.mean(0))
    qb, _ = np.linalg.qr(B - B.mean(0))
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)


def pillai_perm(A, B, label):
    obsv = canonical(A, B)
    g = gen(label)
    n = len(A)
    pm = np.array([(canonical(A, B[g.permutation(n)]) ** 2).sum() for _ in range(NPERM)])
    return obsv, float(((pm >= (obsv ** 2).sum() - 1e-12).sum() + 1) / (NPERM + 1))


def main():
    title('117 - THE SHARED PARTICIPANTS IN THE TWO PROTOCOLS: SAME GROUP?')
    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy()), \
        'features.npz and analysis_complete_cases.csv do not have the same subjects in the same order'
    R24 = c24[F24].to_numpy(float).copy()
    R24[:, :4] = np.log(R24[:, :4])
    X24 = npz["F_primary"].astype(float)
    assert np.abs(standardise(R24) - X24).max() < 1e-9, \
        'the 24 h matrix rebuilt from the cohort file does not match features.npz'
    R150 = c150[F150].to_numpy(float).copy()
    R150[:, :4] = np.log(R150[:, :4])
    X150 = standardise(R150)
    X = {"24h": X24, "150h": X150}
    cohort = {"24h": c24, "150h": c150}
    log(f'24 h cohort n = {len(c24)}, 150 h cohort n = {len(c150)}; seven-variable matrices '
        f'standardised as in 25 and 45 (24 h: identical to features.npz).')
    cond = sorted(set(c24.PATNO) & set(c150.PATNO))
    ix = {"24h": c24.set_index("PATNO").index.get_indexer(cond),
          "150h": c150.set_index("PATNO").index.get_indexer(cond)}
    assert (ix["24h"] >= 0).all() and (ix["150h"] >= 0).all()
    ia, ib = ix["24h"], ix["150h"]
    n = len(cond)
    departure = float(np.abs(c24.loc[ia, COMMON].to_numpy(float)
                          - c150.loc[ib, COMMON].to_numpy(float)).max())
    log(f'Shared participants: {n}. Maximum discrepancy in DAT and age of the shared participants '
        f'between the two cohort files, raw values: {departure:.3g}' + (' (identical)' if departure == 0 else ""))
    log(f'  After standardisation within cohort the same three variables differ by up to '
        f'{np.abs(X24[ia][:, 4:] - X150[ib][:, 4:]).max():.2f} SD: the cohorts have different means '
        f'and dispersions.')
    log(f"Declared working directory: {('sporadic only' if ONLY else 'full cohort')}.")
    p25 = pd.read_csv(os.path.join(OUT, "25_nulls.csv"))
    lab7 = {}
    for tag in ("24h", "150h"):
        lab7[tag] = parts_k2(X[tag])
        r = p25[p25.cohort.str.startswith(tag) & (p25.statistic == 'ARI_algorithms_k2')]
        assert len(r) == 1, f'25_nulls.csv: row ARI_algorithms_k2 for {tag} not unique'
        d = abs(_rule_functions.ari_avg(lab7[tag]) - float(r.observed.iloc[0]))
        assert d < 1e-9, f'{tag}: ARI between algorithms does not reproduce 25_nulls.csv (discrepancy {d:.2e})'
        three = three_k2(X[tag])
        assert all(np.array_equal(three[a], lab7[tag][a]) for a in ALG), \
            'the local constructors do not give the partitions of _rule_functions'
        log(f'Identity {tag}: the mean ARI between the four partitions at k = 2 reproduces '
            f'25_nulls.csv ({float(r.observed.iloc[0]):.4f}; discrepancy {d:.1e}).')

    # ------------------------------------------------------------------ 0 ----
    title('0. ARE THE SHARED PARTICIPANTS A PARTICULAR SAMPLE OF THE TWO COHORTS?')
    rows0 = []
    for tag in ("24h", "150h"):
        d, M = cohort[tag], X[tag]
        within = np.zeros(len(d), bool)
        within[ix[tag]] = True
        log(f'\n{tag}: {int(within.sum())} shared participants against {int((~within).sum())} other '
            f'members')
        for j, name in enumerate(NAMES[tag]):
            mw = stats.mannwhitneyu(M[within, j], M[~within, j])
            rows0.append(dict(cohort=tag, variable=name,
                               difference_in_SD=float(M[within, j].mean() - M[~within, j].mean()),
                               p_mann_whitney=float(mw.pvalue)))
        for col, name in (("duration_yrs", 'duration from diagnosis, years (medians)'),
                          ("hy", 'Hoehn and Yahr (means)')):
            a, b = d.loc[within, col].dropna(), d.loc[~within, col].dropna()
            mw = stats.mannwhitneyu(a, b)
            f = np.median if col == "duration_yrs" else np.mean
            rows0.append(dict(cohort=tag, variable=name, shared=float(f(a)),
                               others=float(f(b)), p_mann_whitney=float(mw.pvalue)))
        t = pd.DataFrame([r for r in rows0 if r['cohort'] == tag])
        log(t.drop(columns='cohort').round(4).to_string(index=False))
    date24 = pd.to_datetime(c24.RUNDATE)
    inside24 = np.zeros(len(c24), bool)
    inside24[ia] = True
    months = pd.crosstab(date24.dt.to_period("M"), np.where(inside24, 'shared', 'others'))
    log('\nRun date at 24 h, by month (the months with at least one shared participant):')
    log(months[months.get('shared', 0) > 0].to_string())
    meta = date24.sort_values().iloc[len(date24) // 2]
    late = date24 > meta
    mw = stats.mannwhitneyu(date24[inside24].astype("int64"), date24[~inside24].astype("int64"))
    log(f'Distinct run dates: {date24.nunique()} in the cohort, {date24[inside24].nunique()} among '
        f'the shared participants. Run after the median date of the cohort ({meta.date()}): '
        f'{int(late[inside24].sum())} of {n} shared participants against '
        f'{int(late[~inside24].sum())} of {int((~inside24).sum())} others; Mann-Whitney on the date '
        f'p = {fmt_p(mw.pvalue)}.')
    log('The 150 h cohort file does not contain the run date: not checked here.')
    pd.DataFrame(rows0).to_csv(os.path.join(OUT, "117_shared_sample.csv"), index=False)

    # ------------------------------------------------------------------ 1 ----
    title('1. PARTITIONS AT k = 2 ESTIMATED ON THE WHOLE COHORTS, READ ON THE SHARED PARTICIPANTS')
    rows, tables, tags = [], {}, {}
    for name, col in DATASETS.items():
        la_t = lab7["24h"] if len(col) == 7 else parts_k2(X24[:, col])
        lb_t = lab7["150h"] if len(col) == 7 else parts_k2(X150[:, col])
        tags[name] = (la_t, lb_t)
        log(f'\n{name}')
        for alg in _rule_functions.ALGORITHMS:
            la, lb = la_t[alg][ia], lb_t[alg][ib]
            r, tav = exact(la, lb)
            tables[(name, alg)] = (int(la.sum()), int(lb.sum()))
            degenerate = bool(min(np.bincount(la, minlength=2).min(),
                                np.bincount(lb, minlength=2).min()) < 2)
            without_power = bool(r['p_minimum_reachable'] >= ALPHA)
            assert abs(r["ARI"] - ari(la, lb)) < 1e-12
            r.update(dataset=name, algorithm=alg, k=2, n=n,
                     groups_cohort_24h="/".join(map(str, np.bincount(la_t[alg], minlength=2))),
                     groups_cohort_150h="/".join(map(str, np.bincount(lb_t[alg], minlength=2))),
                     shared_24h="/".join(map(str, np.bincount(la, minlength=2))),
                     shared_150h="/".join(map(str, np.bincount(lb, minlength=2))),
                     SE_jackknife=se_jackknife(la, lb), degenerate=degenerate,
                     without_power=without_power,
                     role="test" if name in TEST else 'check on identical measures')
            rows.append(r)
            mn = r['same_group_minimum_by_p05']
            log(f"  {alg:8s} cohorts {r['groups_cohort_24h']:>8s} and {r['groups_cohort_150h']:>8s}; "
                f"shared {r['shared_24h']:>6s} and {r['shared_150h']:>6s}; same group "
                f"{r['same_group']}/{n} (expected by chance {r['expected_by_chance']:.1f}; "
                f"{('-' if np.isnan(mn) else int(mn))} needed for p < 0.05); ARI {r['ARI']:+.3f}; "
                f"exact p {fmt_p(r['p_exact'])}; minimum attainable p "
                f"{fmt_p(r['p_minimum_reachable'])}"
                + ('  [degenerate]' if degenerate else "")
                + ('  [no power]' if without_power and not degenerate else ""))
    t = pd.DataFrame(rows)
    sel = (t.role == "test") & ~t.degenerate & ~t.without_power
    t["p_BH"] = np.nan
    t.loc[sel, "p_BH"] = _rule_functions.bh(t.loc[sel, 'p_exact'].to_numpy())
    excluded = t[(t.role == "test") & ~sel]
    log(f'\nBenjamini-Hochberg correction over the {int(sel.sum())} tests of (a) and (b) that can '
        f'give p < 0.05 (excluded: '
        + "; ".join(f"{r.dataset[:1]} {r.algorithm}, "
                    f"{('degenerate' if r.degenerate else 'without power')}"
                    for r in excluded.itertuples()) + f"): minimum adjusted p "
                                                     f"{t.loc[sel, 'p_BH'].min():.4f}; tests with "
                                                     f"unadjusted p < 0.05: "
                                                     f"{int((t.loc[sel, 'p_exact'] < ALPHA).sum())}; "
                                                     f"with adjusted p < 0.05: "
                                                     f"{int((t.loc[sel, 'p_BH'] < ALPHA).sum())}.")
    log(f"The test is conservative: under chance the share of tables with p < 0.05 ranges from "
        f"{t.loc[sel, 'rejection_below_the_chance'].min():.3f} to "
        f"{t.loc[sel, 'rejection_below_the_chance'].max():.3f}.")
    log("The partitions of (a) and of (b) are not independent tests. ARI between the partition on "
        "the seven variables and the one on the kinetic parameters alone, on the whole cohort:")
    for alg in ALG:
        log(f'  {alg:8s} 24 h {ari(tags[TEST[0]][0][alg], tags[TEST[1]][0][alg]):+.3f}; 150 h '
            f'{ari(tags[TEST[0]][1][alg], tags[TEST[1]][1][alg]):+.3f}')
    par = t[(t.role != "test") & ~t.degenerate]
    log('Check on identical measures (c), non-degenerate rows: '
        + "; ".join(f'{r.algorithm} {r.same_group}/{n}, ARI {r.ARI:+.3f}, p {fmt_p(r.p_exact)}'
                    for r in par.itertuples()) + ".")

    # ------------------------------------------------------------------ 2 ----
    title('2. POWER OF THE TEST')
    log("Two normal latent scores with correlation rho; in each protocol the subjects with the "
        "highest score form the smaller group, with the observed sizes.")
    rows2 = []
    for name in TEST:
        for alg in ALG:
            r1, c1 = tables[(name, alg)]
            # labels 0 and 1 are arbitrary: the SMALLER group of each protocol is that
            # of the higher scores (the alignment that allows the maximum
            # agreement). The ARI does not depend on the labels.
            r1, c1 = min(r1, n - r1), min(c1, n - c1)
            tav = tables_exacts(r1, c1, n).set_index("n11")
            g = gen(f'117|power|{name}|{alg}')
            row = dict(dataset=name, algorithm=alg, margins=f"{r1}/{n - r1} e {c1}/{n - c1}")
            for rho in RHO:
                z = g.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=(N_POWER, n))
                a = np.argsort(-z[:, :, 0], axis=1)[:, :r1]
                b = np.argsort(-z[:, :, 1], axis=1)[:, :c1]
                n11 = np.array([len(np.intersect1d(x, y)) for x, y in zip(a, b)])
                row[f"rho_{rho}"] = float((tav.p.reindex(n11).to_numpy() < ALPHA).mean())
            rows2.append(row)
    pwr = pd.DataFrame(rows2)
    log(pwr.round(2).to_string(index=False))
    pwr.to_csv(os.path.join(OUT, "117_shared_power.csv"), index=False)
    log(f'Replicates per cell: {N_POWER}. The model is a choice: with another one the values would '
        f'change.')

    # ------------------------------------------------------------------ 3 ----
    title("3. 'SAME PROTOCOL' CEILING: TWO DISJOINT HALVES OF THE SAME COHORT")
    rows3 = []
    for tag in ("24h", "150h"):
        M = X[tag]
        others = np.setdiff1d(np.arange(len(M)), ix[tag])
        g = gen(f'117|ceiling|{tag}')
        acc = {(name, alg): [] for name in DATASETS for alg in ALG}
        acc[(TEST[0], "kmeans k = 4")] = []
        for _ in range(N_CEILING):
            pm = g.permutation(others)
            meta = [np.r_[ix[tag], pm[: len(pm) // 2]], np.r_[ix[tag], pm[len(pm) // 2:]]]
            for name, col in DATASETS.items():
                # each half is re-standardised, as an analysis of its own would do
                P = [three_k2(standardise(M[m][:, col])) for m in meta]
                for alg in ALG:
                    acc[(name, alg)].append(ari(P[0][alg][:n], P[1][alg][:n]))
            Q = [kmeans_k(standardise(M[m]), 4)[:n] for m in meta]
            acc[(TEST[0], "kmeans k = 4")].append(ari(Q[0], Q[1]))
        for (name, alg), v in acc.items():
            v = np.array(v)
            rows3.append(dict(cohort=tag, dataset=name, algorithm=alg,
                               n_by_half=len(ix[tag]) + len(others) // 2,
                               ARI_median=float(np.median(v)),
                               ARI_p5=float(np.percentile(v, 5)),
                               ARI_p95=float(np.percentile(v, 95)), replicates=len(v)))
    tet = pd.DataFrame(rows3)
    log(tet.round(3).to_string(index=False))
    tet.to_csv(os.path.join(OUT, "117_shared_ceiling.csv"), index=False)
    obsv = t[(t.role == "test") & t.algorithm.isin(ALG)]
    below = []
    for r in obsv.itertuples():
        s = tet[(tet.dataset == r.dataset) & (tet.algorithm == r.algorithm)]
        below.append((r.dataset[:1], r.algorithm, bool((r.ARI < s.ARI_p5).all())))
    log("The observed ARI between protocols lies below the fifth percentile of the ceiling in both "
        "cohorts for: " + ", ".join(f"{a} {b}" for a, b, c in below if c)
        + '; not for: ' + (", ".join(f"{a} {b}" for a, b, c in below if not c) or 'nothing') + ".")
    log("The ceiling is optimistic: each half contains the same shared participants with the same values.")

    # ------------------------------------------------------------------ 4 ----
    title("4. MORE THAN TWO GROUPS: k-means AT k = 3 AND k = 4")
    rows4, part_k = [], {}
    for name, col in DATASETS.items():
        for k in (3, 4):
            La, Lb = kmeans_k(X24[:, col], k), kmeans_k(X150[:, col], k)
            part_k[(name, k)] = (La, Lb)
            la, lb = La[ia], Lb[ib]
            o = float(ari(la, lb))
            g = gen(f'117|{name}|kmeans|k{k}')
            pm = np.array([ari(la, g.permutation(lb)) for _ in range(NPERM)])
            p = float((np.sum(pm >= o - 1e-12) + 1) / (NPERM + 1))
            small = bool(min(np.bincount(la, minlength=k).min(),
                               np.bincount(lb, minlength=k).min()) < 2)
            rows4.append(dict(dataset=name, k=k, ARI=o, p_permutation=p,
                               shared_24h="/".join(map(str, np.bincount(la, minlength=k))),
                               shared_150h="/".join(map(str, np.bincount(lb, minlength=k))),
                               group_having_less_of_two_shared=small))
            log(f"  {name}, k = {k}: shared {rows4[-1]['shared_24h']} and "
                f"{rows4[-1]['shared_150h']}; ARI {o:+.3f}; permutation p {fmt_p(p)} (uncorrected)" + ('  [a group with fewer than two shared participants]' if small else ""))
    log("\nAttribution check for k = 4 on the seven variables. Among the shared participants a "
        "block of variables is permuted in ONE cohort, k-means at k = 4 is re-estimated on the "
        "whole cohort and read again on the shared participants.")
    La0, Lb0 = part_k[(TEST[0], 4)]
    obs4 = float(ari(La0[ia], Lb0[ib]))
    rows5 = []
    for tag, other in (("24h", "150h"), ("150h", "24h")):
        held = (La0 if other == "24h" else Lb0)[ix[other]]
        for block, col in (('shared (DAT, age)', [4, 5, 6]), ('kinetics', [0, 1, 2, 3])):
            g = gen(f'117|k4|{tag}|{block}')
            v = []
            for _ in range(N_K4):
                M = X[tag].copy()
                pm = g.permutation(n)
                M[np.ix_(ix[tag], col)] = M[np.ix_(ix[tag][pm], col)]
                v.append(ari(kmeans_k(M, 4)[ix[tag]], held))
            v = np.array(v)
            rows5.append(dict(cohort_permuted=tag, block_permuted=block, ARI_observed=obs4,
                               ARI_avg=float(v.mean()), ARI_p2_5=float(np.percentile(v, 2.5)),
                               ARI_p97_5=float(np.percentile(v, 97.5)),
                               share_above_observed=float((v >= obs4).mean()), replicates=N_K4))
            log(f'  {tag}, {block} permuted: mean ARI {v.mean():+.3f} ({np.percentile(v, 2.5):+.3f}; '
                f'{np.percentile(v, 97.5):+.3f}); share of replicates with ARI >= observed '
                f'({obs4:+.3f}): {(v >= obs4).mean():.3f}')
    k4 = pd.DataFrame(rows5)
    k4.to_csv(os.path.join(OUT, "117_shared_k4.csv"), index=False)
    rc = k4[k4.block_permuted.str.startswith('sharedstrings')].ARI_avg
    rk = k4[k4.block_permuted == 'kinetics'].ARI_avg
    log(f'  With the identity of DAT and age removed the mean ARI falls to {rc.min():+.3f} and '
        f'{rc.max():+.3f}; with that of the kinetic parameters removed it stays at {rk.min():+.3f} '
        f'and {rk.max():+.3f} ({100 * rk.min() / obs4:.0f}% and {100 * rk.max() / obs4:.0f}% of the '
        f'observed).')
    pd.concat([t.assign(section="1. k = 2"),
               pd.DataFrame(rows4).assign(section='4. k-means, k = 3 and 4')],
              ignore_index=True).to_csv(os.path.join(OUT, "117_shared_partitions.csv"),
                                        index=False)

    # ------------------------------------------------------------------ 5 ----
    title('5. WITHOUT PARTITIONS: CANONICAL CORRELATIONS BETWEEN THE KINETIC PARAMETERS IN THE SHARED PARTICIPANTS')
    A, B = X24[ia][:, :4], X150[ib][:, :4]
    c0, p0 = pillai_perm(A, B, '117|canonical|raw')
    log(f"Four kinetic parameters at 24 h against four at 150 h, {n} shared participants: canonical "
        f"correlations {', '.join((f'{x:.2f}' for x in c0))}; Pillai trace {(c0 ** 2).sum():.3f}; "
        f"permutation p {fmt_p(p0)}.")
    share_between = []
    Rc = R24[:, :4].copy()
    for j in range(4):
        s = pd.Series(Rc[:, j])
        m = s.groupby(date24.to_numpy()).transform("mean")
        share_between.append(float(m.var(ddof=0) / s.var(ddof=0)))
        Rc[:, j] = (s - m).to_numpy()
    log('Share of the variance of the 24 h kinetic parameters between run dates (whole cohort): '
        + ", ".join(f"{a} {q:.2f}" for a, q in zip(NAMES["24h"][:4], share_between)) + ".")
    c1, p1 = pillai_perm(Rc[ia], B, "117|canoniche|centrate")
    log(f"After centring the 24 h kinetic parameters on the mean of the run date: canonical "
        f"correlations {', '.join((f'{x:.2f}' for x in c1))}; Pillai trace {(c1 ** 2).sum():.3f}; "
        f"permutation p {fmt_p(p1)}.")
    log(f"With {n} subjects and four variables per block the canonical correlations are inflated "
        f"upwards even under chance: the p counts, not the value.")
    pd.DataFrame([dict(analysis='kinetic parameters as in the cohorts', **{f"r{i + 1}": float(x) for i, x in enumerate(c0)},
                       pillai=float((c0 ** 2).sum()), p_permutation=p0),
                  dict(analysis='24 h centred on the run date', **{f"r{i + 1}": float(x) for i, x in enumerate(c1)},
                       pillai=float((c1 ** 2).sum()), p_permutation=p1)]).to_csv(
        os.path.join(OUT, "117_shared_canonical.csv"), index=False)
    log('\nAt 24 h, does the group of the shared participants follow the run date? (partitions at k '
        '= 2; permutation test on the chi-square of the group x date table, uncorrected)')
    date_c = date24.to_numpy()[ia]
    for name in TEST:
        for alg in ALG:
            la = tags[name][0][alg][ia]
            tab = pd.crosstab(la, date_c).to_numpy()
            chi = stats.chi2_contingency(tab, correction=False)[0]
            g = gen(f'117|run|{name}|{alg}')
            pm = np.array([stats.chi2_contingency(pd.crosstab(g.permutation(la), date_c).to_numpy(),
                                                  correction=False)[0] for _ in range(2000)])
            v = float(np.sqrt(chi / (n * (min(tab.shape) - 1))))
            log(f'  {name[:1]} {alg:8s} Cramer V {v:.2f}; permutation p '
                f'{fmt_p((np.sum(pm >= chi - 1e-09) + 1) / 2001)}')

    title('WHAT THIS SCRIPT DOES NOT DO')
    log('- It does not run the battery on the shared participants alone (too few for seven variables).')
    log("- It does not distinguish 'there are no groups' from 'the two protocols do not order the "
        "same patients in the same way': agreement at chance level is expected in both cases.")
    log("- It does not take into account the uncertainty with which the partitions are estimated.")
    log("- It does not check the run date at 150 h (it is not in the cohort file).")
    log('- The p values of sections 4 and 5 are not corrected.')
    log(f'- Permutations {NPERM}; replicates: power {N_POWER}, ceiling {N_CEILING}, k = 4 {N_K4}; '
        f'seed {SEED}.')
    with open(os.path.join(OUT, "117_shared_partitions.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
