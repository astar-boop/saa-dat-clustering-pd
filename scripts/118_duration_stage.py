"""
118 - Disease duration and clinical stage: do they matter for the search for groups?

Why it exists. Reviewer 2 ("Disease duration and clinical stage") observes
that disease duration is not among the clustering variables, that biomarker
profiles may change with progression, and asks whether duration or stage may
affect the identification of subgroups. The manuscript uses duration only as a
covariate of the longitudinal models.

Note of honesty. On 13/11/2026, before this script was written, the following
were seen in a preliminary way: the correlations of duration and stage with the
seven variables, the share of variance explained, and the ARI between the
k-means partition at k = 2 before and after removing duration and stage (about
0.9). NO result of the battery or of the curves on the residualised variables
was seen: the rules of section 3 are written here before those results.

Clinical variables. Duration: `duration_yrs` of the curated data cut at
baseline (years from diagnosis; it is not the time from symptom onset). Stage:
Hoehn and Yahr at baseline (`hy`), as an indicator "stage 2 or higher" versus
"stage 1", because almost all subjects are at stage 1 or 2.

What it does.
  1. It describes duration and stage in the two cohorts and their association
     with each of the seven variables: Spearman correlation with duration and
     with the stage indicator (14 tests per cohort, corrected with
     Benjamini-Hochberg within cohort), and share of variance of each
     variable explained by [logarithm of duration, stage indicator]
     (R2 of a linear regression).
  2. Do the four partitions at k = 2 of the manuscript (constructors of 25)
     differ by duration (Mann-Whitney) or by stage (Fisher exact)?
     Eight tests per cohort, BH within cohort. The partitions with a group of
     fewer than ten subjects are reported and flagged, and do not enter the
     correction.
  3. Decision rule of the manuscript on the variables after removing
     duration and stage. Three variants for each cohort:
       V0  as published (the seven standardised variables);
       V1  duration and stage removed: from each of the seven standardised
           variables the part linearly explained by
           [logarithm of duration, stage indicator] is subtracted; the
           residuals are re-standardised. The subjects without stage drop out
           (main analysis);
       V2  duration alone removed, on all subjects (sensitivity).
     Age is one of the seven variables and is associated with stage: in V1 the
     part of it related to stage is removed, as for the others. This is
     intended: the question is whether groups appear once what depends on
     duration and stage has been removed.
     For each variant the null reference (Gaussian copula) is fitted to the
     matrix of the variant. Functions, seeds and streams are those of 25
     (criteria at k = 2) and of 45 (curves over k = 2-10), reused through _rule_functions.
     First arm: at least one among prediction strength, Jaccard and ARI between
     algorithms at k = 2 with one-sided Monte Carlo p < 0.05 (uncorrected),
     on the random stream of 25 as in the manuscript. Alongside, PS and Jaccard
     observed at the mean of 20 independent streams (convention of 110).
     Second arm: silhouette or prediction strength over k = 2-10 with corrected
     p < 0.05; Benjamini-Hochberg within statistic over the 18 tests of the two
     cohorts (9 values of k for 2 cohorts), as in the manuscript (75).
     The rule is met if at least one arm is.
     Number of realisations of the null. Criteria at k = 2: 500 (24 h) and 300
     (150 h), as 25. Curves: two readings, both reported.
       "B pubblicato": 500 and 300 for both statistics, as 45. With V0 it
       must reproduce 25_nulls.csv and 45_diagnostic_curves.csv value by
       value; if it does not reproduce them the script stops before V1 and V2.
       "B esteso": silhouette 2000 in both cohorts, prediction strength
       500 in both. Reason, written before the results: with 300
       realisations the minimum attainable p is 1/301 = 0.0033, and with 18
       tests a single value of k cannot survive BH (p <= 0.0028 would be
       needed): in the 150 h cohort the second arm, with B pubblicato, cannot
       be met by a single k. The published realisations are the first of the
       same sequence.
     THE OUTCOME OF THE RULE FOR THIS SCRIPT IS THAT WITH B ESTESO; that with
     B pubblicato is reported alongside for comparison with the manuscript.
     Readings, fixed here:
       - rule not met in V1 and V2, as in V0: removing duration and stage
         does not make groups appear;
       - rule met in V1 or V2 and not in V0: duration or stage were masking
         a structure; it would be a new result, to be reported;
       - rule met in V0 with B esteso: the published outcome depends on the
         number of realisations; to be reported before anything else.

Declared limitations. Residualisation removes differences in MEAN that are
linear in the logarithm of duration and between the two stages; it does not
remove differences in variance or in correlation between stages. The range of
duration and of stage in the two cohorts is narrow: the script says nothing
about later stages. The p of the criteria at k = 2 are not corrected across
criteria, cohorts and variants.

CHANGES AFTER THE INDEPENDENT VERIFICATION (13/11/2026; report in
revisione_R2_rapporti/verifiche_script/riverifica_118.txt). The verifier
reproduced all the numbers and found three checks that did not fire.
Assertions were added in residui() (residuals uncorrelated with the
covariates, with mean 0 and SD 1) and in identity_v0() (the corrected p of 45
must really have been compared in all the curves). The script was NOT rerun
after these additions: they do not change any computation. The version run and
verified is in revisione_R2_rapporti/versioni_script/118_come_eseguito_e_verificato.py.
The verifier also showed that the reading "with duration and stage removed the
criteria come closer to the threshold" does not hold: the related checks are in
script 120. The remark on the calibration of the Monte Carlo p against a
reference fitted to the same data remains open, and does not concern this
script alone.

Checks (identity, not counts). The 24 h matrix rebuilt from the cohort file
must coincide with features.npz; duration and stage of the cohort files must
coincide with the curated data cut at baseline; the cache of the curated data
cut must coincide with the xlsx file in the columns read; V0 must reproduce 25
and 45.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 -B scripts/118_duration_stage.py
      N_PROCESSES=n   number of processes (default 10; the results do not depend on it)
      ONLY_SECTIONS_1_2=1   runs only the sections without Monte Carlo, without writing
Output: outputs/118_duration_stage.txt, 118_associations.csv,
        118_partitions_vs_duration_stage.csv, 118_battery_k2.csv,
        118_curves_k2_10.csv, 118_rule.csv, 118_nulls.npz
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

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
ONLY_12 = os.environ.get('ONLY_SECTIONS_1_2', "0") == "1"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "10")))
FILE_CUR = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"
CACHE = os.path.join(OUT, "cache", "curated.parquet")
COL_CUR = ["PATNO", "EVENT_ID", "duration_yrs", "hy"]


import _rule_functions                                                    # noqa: E402



C45 = _rule_functions.C45
K_CURVE = list(C45.KRANGE)

B_K2 = dict(_rule_functions.B_NULL)                    # 500 and 300: the values of 25
B_PUBL = dict(_rule_functions.B_CURVE_C)               # 500 and 300: the values of 45
B_SIL = 2000                                 # B esteso, silhouette
B_PS = 500                                   # B esteso, prediction strength
BLOCK_PS, BLOCK_SIL = 20, 250
ORDERING_PS = {"24h": "split_outer", "150h": "k_outer"}      # as 45
ALPHA = 0.05
TOL = 1e-9
CRITERIA = list(_rule_functions.CRITERIA)
STAT = {"silhouette": "silhouette", "prediction_strength": "PS"}
F24 = ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax",
       "dat_putamen_low", "dat_caudate_mean", "age_bl"]
F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE",
        "dat_putamen_low", "dat_caudate_mean", "age_bl"]
NAMES = {"24h": ["log Fmax", "log TTT", "log AUC", "log SLOPEMax",
                'lower putamen', 'mean caudate', 'age'],
        "150h": ["log Fmax", "log T50", "log AUC", "log SLOPE",
                 'lower putamen', 'mean caudate', 'age']}
VARIANTS = {"V0": 'as published',
            "V1": 'duration and stage removed',
            "V2": 'duration alone removed'}
GROUP_SMALL = 10

rep = []


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def execute(t):
    """In the child process module _rule_functions is reloaded when this file is imported;
    the functions are those of _rule_functions, not copies."""
    return _rule_functions.EXECUTORS[t['kind']](t)


def standardise(M):
    return (M - M.mean(0)) / M.std(0)


def residuals(X, D):
    """Residuals of each column of X on the matrix D (with intercept),
    re-standardised; it also returns R2 per column."""
    A = np.column_stack([np.ones(len(D)), D])
    assert len(A) == len(X) and np.isfinite(A).all(), 'covariates missing or not aligned'
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    R = X - A @ beta
    r2 = 1 - R.var(0) / X.var(0)
    Z = standardise(R)
    # added after the independent verification: the residuals must be uncorrelated
    # with the covariates (intercept included), with mean 0 and SD 1
    assert np.abs(A.T @ Z).max() < 1e-6 * len(Z), 'residuals correlated with the covariates'
    assert np.abs(Z.mean(0)).max() < 1e-9 and np.abs(Z.std(0) - 1).max() < 1e-9
    return Z, r2


def load_curated():
    cur = pd.read_parquet(CACHE, columns=COL_CUR)
    xlsx = os.path.join(BASE, FILE_CUR)
    if not os.path.exists(xlsx):
        return cur, 'identity of the cache with the xlsx file NOT verified (file not found)'
    x = pd.read_excel(xlsx, sheet_name="20260511", usecols=COL_CUR)[COL_CUR]
    assert len(x) == len(cur), 'cache of the curated data cut: number of rows different from the xlsx file'
    ch = ["PATNO", "EVENT_ID"]
    a = cur.sort_values(ch, kind="mergesort").reset_index(drop=True)
    b = x.sort_values(ch, kind="mergesort").reset_index(drop=True)
    for c in COL_CUR:
        if a[c].dtype.kind in "fiu" and b[c].dtype.kind in "fiu":
            ok = np.array_equal(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)
        else:
            ok = bool((a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all())
        assert ok, f'cache of the curated data cut: the column {c} does not match the xlsx file'
    return cur, (f"the cache coincides with the xlsx file in all the {len(cur)} rows and in the "
                 f"columns read ({', '.join(COL_CUR)})")


def tasks_variant(var, tag, X):
    """The Monte Carlo tasks of one variant in one cohort."""
    branch = f"{var}|{tag}"
    c = [dict(kind='lone', branch=branch, X=X, B=B_K2[tag]),
         dict(kind='streams', branch=branch, X=X, label=f'118|{branch}'),
         dict(kind='curve_obsv', branch=branch, X=X, ordering_ps=ORDERING_PS[tag])]
    for b0 in range(0, B_PS, BLOCK_PS):
        c.append(dict(kind='curve_null', branch=branch, nullref="copula", X=X, b0=b0,
                      b1=min(b0 + BLOCK_PS, B_PS), n_ps=B_PS, ordering_ps=ORDERING_PS[tag]))
    for b0 in range(B_PS, B_SIL, BLOCK_SIL):
        c.append(dict(kind='curve_null', branch=branch, nullref="copula", X=X, b0=b0,
                      b1=min(b0 + BLOCK_SIL, B_SIL), n_ps=B_PS, ordering_ps=ORDERING_PS[tag]))
    return c


def collect(results):
    """Gathers the results of the tasks by arm."""
    out = {}
    for r in results:
        o = out.setdefault(r['branch'], dict(blocks=[]))
        if r['kind'] == 'lone':
            o['obsv'], o['nulls'] = r['obsv'], pd.DataFrame(r['nulls'])
        elif r['kind'] == 'streams':
            o['streams'] = pd.DataFrame(r['streams'])
        elif r['kind'] == 'curve_obsv':
            o['curve_obsv'] = r
        else:
            o['blocks'].append(r)
    for branch, o in out.items():
        bl = sorted(o.pop('blocks'), key=lambda r: r["b0"])
        assert [r["b0"] for r in bl] == sorted({r["b0"] for r in bl}), 'repeated blocks'
        o['null_sil'] = np.vstack([r["silhouette"] for r in bl])
        o['null_ps'] = np.vstack([r["PS"] for r in bl if len(r["PS"])])
        assert o['null_sil'].shape == (B_SIL, len(K_CURVE)), o['null_sil'].shape
        assert o['null_ps'].shape == (B_PS, len(K_CURVE)), o['null_ps'].shape
    return out


def tables_variant(var, data, ns):
    """Rows of the criteria at k = 2 and of the curves for one variant (two cohorts)."""
    rows_b, rows_c = [], []
    for tag in ("24h", "150h"):
        o = data[f"{var}|{tag}"]
        N = o['nulls']
        assert len(N) == B_K2[tag]
        for stat, obsv in o['obsv'].items():
            high = _rule_functions.MORE_STRUCTURE[stat]
            v = N[stat].to_numpy(float)
            r = dict(variant=var, description=VARIANTS[var], cohort=tag, n=ns[(var, tag)],
                     statistic=stat, observed=float(obsv),
                     null_median=float(np.median(v)),
                     null_CI2_5=float(np.percentile(v, 2.5)),
                     null_CI97_5=float(np.percentile(v, 97.5)),
                     p_montecarlo=_rule_functions.p_mc(v, obsv, high), B=len(v),
                     criterion_of_rule=stat in CRITERIA)
            if stat in o['streams'].columns:
                m = float(o['streams'][stat].mean())
                r.update(observed_mean_20_streams=m,
                         observed_sd_streams=float(o['streams'][stat].std(ddof=1)),
                         p_at_mean_of_streams=_rule_functions.p_mc(v, m, high))
            rows_b.append(r)
        for stat, key in STAT.items():
            obsv = o['curve_obsv'][key]
            nul = o['null_sil'] if key == "silhouette" else o['null_ps']
            for i, k in enumerate(K_CURVE):
                vp, ve = nul[:B_PUBL[tag], i], nul[:, i]
                rows_c.append(dict(
                    variant=var, description=VARIANTS[var], cohort=tag, n=ns[(var, tag)],
                    statistic=stat, k=k, observed=float(obsv[i]),
                    null_median_B_published=float(np.median(vp)),
                    null_CI2_5_B_published=float(np.percentile(vp, 2.5)),
                    null_CI97_5_B_published=float(np.percentile(vp, 97.5)),
                    p_B_published=_rule_functions.p_mc(vp, obsv[i]), B_published=len(vp),
                    null_median_B_extended=float(np.median(ve)),
                    p_B_extended=_rule_functions.p_mc(ve, obsv[i]), B_extended=len(ve),
                    p_sup95_B_extended=_rule_functions.p_sup95(ve, obsv[i])))
    b, c = pd.DataFrame(rows_b), pd.DataFrame(rows_c)
    for stat in STAT:                         # BH within statistic, 18 tests
        s = c.statistic == stat
        assert int(s.sum()) == 2 * len(K_CURVE)
        for col in ('p_B_published', 'p_B_extended'):
            c.loc[s, col.replace("p_", "p_BH_")] = _rule_functions.bh(c.loc[s, col].to_numpy())
        c.loc[s, 'n_test_family'] = int(s.sum())
    return b, c


def rule_variant(var, b, c):
    r = dict(variant=var, description=VARIANTS[var])
    first, first_m = False, False
    for tag in ("24h", "150h"):
        s = b[(b.cohort == tag) & b.criterion_of_rule]
        exceed = s[s.p_montecarlo < ALPHA].statistic.tolist()
        pm = s.p_at_mean_of_streams.where(s.p_at_mean_of_streams.notna(),
                                             s.p_montecarlo)
        exceed_m = s[pm < ALPHA].statistic.tolist()
        r[f'first_branch_{tag}_criteria_that_exceed'] = ", ".join(exceed)
        r[f'first_branch_{tag}_p_minimum'] = float(s.p_montecarlo.min())
        r[f'first_branch_{tag}_criteria_that_exceed_at_mean_of_streams'] = ", ".join(exceed_m)
        first |= bool(exceed)
        first_m |= bool(exceed_m)
    r['first_branch_satisfied'] = first
    r['first_branch_satisfied_at_mean_of_streams'] = first_m
    for reading in ('B_extended', 'B_published'):
        sec = False
        for stat in STAT:
            s = c[c.statistic == stat]
            col = f'p_BH_{reading}'
            sop = s[s[col] < ALPHA]
            r[f'second_branch_{stat}_k_that_survive_{reading}'] = "; ".join(
                f"{t} k={k}" for t, k in zip(sop.cohort, sop.k))
            i = s[col].idxmin()
            r[f'second_branch_{stat}_p_BH_minimum_{reading}'] = float(s.loc[i, col])
            r[f'second_branch_{stat}_where_{reading}'] = f"{s.loc[i, 'cohort']} k={s.loc[i, 'k']}"
            sec |= len(sop) > 0
        r[f'second_branch_satisfied_{reading}'] = sec
        r[f'rule_satisfied_{reading}'] = first or sec
    return r


def identity_v0(b, c):
    """V0 must reproduce 25_nulls.csv and 45_diagnostic_curves.csv."""
    p25 = pd.read_csv(os.path.join(OUT, "25_nulls.csv"))
    p45 = pd.read_csv(os.path.join(OUT, "45_diagnostic_curves.csv"))
    worse = 0.0
    bh_compared = []
    for tag in ("24h", "150h"):
        a = p25[p25.cohort.str.startswith(tag)].set_index('statistic')
        q = b[(b.variant == "V0") & (b.cohort == tag)].set_index('statistic')
        assert set(a.index) == set(q.index), 'statistics different from those of 25'
        for stat in a.index:
            for mine, its in (('observed', 'observed'), ('null_median', 'null_median'),
                             ('null_CI2_5', "nullo_IC2.5"), ('null_CI97_5', "nullo_IC97.5"),
                             ("p_montecarlo", "p_montecarlo")):
                d = abs(float(q.loc[stat, mine]) - float(a.loc[stat, its]))
                worse = max(worse, d)
                assert d < TOL, f'V0 {tag} {stat} {mine}: does not reproduce 25 (difference {d:.2e})'
            assert int(q.loc[stat, "B"]) == int(a.loc[stat, "B"])
        for stat in STAT:
            a = p45[p45.cohort.str.startswith(tag) & (p45.statistic == stat)].set_index("k")
            q = c[(c.variant == "V0") & (c.cohort == tag) & (c.statistic == stat)].set_index("k")
            assert list(a.index) == list(q.index) == K_CURVE
            pairs = [('observed', 'observed'),
                      ('null_median_B_published', 'null_copula_median'),
                      ('null_CI2_5_B_published', "nullo_copula_IC2.5"),
                      ('null_CI97_5_B_published', "nullo_copula_IC97.5"),
                      ('p_B_published', "p_montecarlo")]
            if "p_BH" in a.columns and (a.n_test_family == 2 * len(K_CURVE)).all():
                pairs.append(('p_BH_B_published', "p_BH"))
                bh_compared.append((tag, stat))
            for mine, its in pairs:
                d = float(np.abs(q[mine].to_numpy(float) - a[its].to_numpy(float)).max())
                worse = max(worse, d)
                assert d < TOL, f'V0 {tag} {stat} {mine}: does not reproduce 45 (difference {d:.2e})'
            assert (q.B_published.to_numpy() == a.B.to_numpy()).all()
    # the corrected p of 45 is comparable only if its family is of 18 tests:
    # if it has not been so in all four curves, the script stops instead of
    # declaring reproduced a value that it has not compared
    assert len(bh_compared) == 2 * len(STAT), \
        f'corrected p of 45 compared only in {bh_compared}: family different from 18 tests'
    return worse


def main():
    t0 = time.time()
    title('118 - DISEASE DURATION AND CLINICAL STAGE')
    cur, outcome_cur = load_curated()
    cur["PATNO"] = cur["PATNO"].astype("int64")
    bl = cur[cur.EVENT_ID == "BL"].set_index("PATNO")
    log(f'Curated: {outcome_cur}')
    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy())
    R24 = c24[F24].to_numpy(float).copy()
    R24[:, :4] = np.log(R24[:, :4])
    assert np.abs(standardise(R24) - npz["F_primary"]).max() < TOL, \
        'the 24 h matrix rebuilt from the cohort file does not match features.npz'
    R150 = c150[F150].to_numpy(float).copy()
    R150[:, :4] = np.log(R150[:, :4])
    X = {"24h": npz["F_primary"].astype(float), "150h": standardise(R150)}
    cohort = {"24h": c24, "150h": c150}
    clin = {}
    for tag, d in cohort.items():
        for col in ("duration_yrs", "hy"):
            a = d[col].to_numpy(float)
            b = bl.reindex(d.PATNO)[col].to_numpy(float)
            # the cohort file is a CSV: the passage through text can shift the
            # last binary digit (differences observed up to 4e-16)
            assert np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True), \
                f'{tag}: {col} of the cohort file does not coincide with the curated data cut at baseline'
        dur, hy = d.duration_yrs.to_numpy(float), d.hy.to_numpy(float)
        assert np.isfinite(dur).all() and (dur > 0).all(), f'{tag}: duration missing or not positive'
        clin[tag] = dict(duration=dur, hy=hy, stage2=np.where(np.isnan(hy), np.nan, hy >= 2))
    log("Identity: 24 h matrix equal to features.npz; duration and stage of the cohort files equal "
        "to the curated data cut at baseline, subject by subject (relative tolerance 1e-12: the "
        "cohort file is a CSV).")
    log(f"Declared working directory: {('sporadic only' if ONLY else 'full cohort')}; n = {len(c24)} "
        f"(24 h) and {len(c150)} (150 h).")

    # ------------------------------------------------------------------ 1 ----
    title('1. DURATION AND STAGE IN THE TWO COHORTS AND ASSOCIATION WITH THE SEVEN VARIABLES')
    rows_a = []
    for tag in ("24h", "150h"):
        cl, n = clin[tag], len(X[tag])
        dur = cl['duration']
        log(f'\n{tag}, n = {n}. Duration from diagnosis, years: median {np.median(dur):.2f}, '
            f'quartiles {np.percentile(dur, 25):.2f}-{np.percentile(dur, 75):.2f}, range '
            f'{dur.min():.2f}-{dur.max():.2f}; over 1 year {int((dur > 1).sum())}, over 2 years '
            f'{int((dur > 2).sum())}, over 3 years {int((dur > 3).sum())}.')
        hy = pd.Series(cl["hy"])
        log('  Hoehn and Yahr: ' + ", ".join(
            f"stage {('missing' if pd.isna(k) else int(k))}: {v}"
            for k, v in hy.value_counts(dropna=False).sort_index().items()))
        ok = ~np.isnan(cl["hy"])
        D = np.column_stack([np.log(dur[ok]), cl['stage2'][ok]])
        _, r2 = residuals(X[tag][ok], D)
        _, r2d = residuals(X[tag], np.log(dur)[:, None])
        block = []
        for j, name in enumerate(NAMES[tag]):
            a = stats.spearmanr(dur, X[tag][:, j])
            b = stats.spearmanr(cl['stage2'][ok], X[tag][ok, j])
            block.append(dict(cohort=tag, n=n, variable=name,
                               rho_duration=float(a.statistic), p_duration=float(a.pvalue),
                               rho_stage=float(b.statistic), p_stage=float(b.pvalue),
                               n_stage=int(ok.sum()), R2_duration_e_stage=float(r2[j]),
                               R2_single_duration=float(r2d[j])))
        t = pd.DataFrame(block)
        adj = _rule_functions.bh(np.concatenate([t.p_duration.to_numpy(), t.p_stage.to_numpy()]))
        t['p_BH_duration'], t['p_BH_stage'] = adj[:len(t)], adj[len(t):]
        rows_a.append(t)
        log(t[['variable', 'rho_duration', 'p_duration', 'p_BH_duration', 'rho_stage', 'p_stage',
               'p_BH_stage', 'R2_duration_e_stage', 'R2_single_duration']].round(4).to_string(index=False))
        log(f'  Largest correlation in absolute value: duration {t.rho_duration.abs().max():.3f}, '
            f'stage {t.rho_stage.abs().max():.3f}. Tests with corrected p < 0.05 (14 tests): '
            f'duration {int((t.p_BH_duration < ALPHA).sum())}, stage '
            f'{int((t.p_BH_stage < ALPHA).sum())}'
            + (f" ({', '.join(t[t.p_BH_stage < ALPHA].variable)})" if (t.p_BH_stage < ALPHA).any() else "")
            + f". Maximum R2: {t.R2_duration_e_stage.max():.3f} "
              f"({t.loc[t.R2_duration_e_stage.idxmax(), 'variable']}); excluding age "
              f"{t[t.variable != NAMES[tag][6]].R2_duration_e_stage.max():.3f}.")
    assoc = pd.concat(rows_a, ignore_index=True)

    # ------------------------------------------------------------------ 2 ----
    title('2. DO THE k = 2 PARTITIONS OF THE MANUSCRIPT DIFFER BY DURATION OR STAGE?')
    rows_p = []
    for tag in ("24h", "150h"):
        cl = clin[tag]
        lab = _rule_functions.labels_k2(X[tag])
        ok = ~np.isnan(cl["hy"])
        block = []
        for alg in _rule_functions.ALGORITHMS:
            la = lab[alg]
            g = np.bincount(la, minlength=2)
            mw = stats.mannwhitneyu(cl['duration'][la == 0], cl['duration'][la == 1])
            tab = pd.crosstab(la[ok], cl['stage2'][ok]).reindex(
                index=[0, 1], columns=[0.0, 1.0], fill_value=0).to_numpy()
            fi = stats.fisher_exact(tab)
            block.append(dict(
                cohort=tag, algorithm=alg, groups=f"{g[0]}/{g[1]}",
                group_small=bool(g.min() < GROUP_SMALL),
                duration_median_g0=float(np.median(cl['duration'][la == 0])),
                duration_median_g1=float(np.median(cl['duration'][la == 1])),
                p_duration=float(mw.pvalue),
                share_stage2_g0=float(tab[0, 1] / tab[0].sum()),
                share_stage2_g1=float(tab[1, 1] / tab[1].sum()),
                p_stage=float(fi.pvalue)))
        t = pd.DataFrame(block)
        s = ~t.group_small
        adj = _rule_functions.bh(np.concatenate([t.p_duration[s].to_numpy(), t.p_stage[s].to_numpy()]))
        t.loc[s, 'p_BH_duration'], t.loc[s, 'p_BH_stage'] = adj[:int(s.sum())], adj[int(s.sum()):]
        rows_p.append(t)
        log(f"\n{tag}")
        log(t.round(4).to_string(index=False))
        log(f'  Corrected tests: {2 * int(s.sum())}. With corrected p < 0.05: duration '
            f'{int((t.p_BH_duration < ALPHA).sum())}, stage {int((t.p_BH_stage < ALPHA).sum())}'
            + (f" ({', '.join(t[t.p_BH_stage < ALPHA].algorithm)})" if (t.p_BH_stage < ALPHA).any() else "")
            + ".")
    part = pd.concat(rows_p, ignore_index=True)
    if ONLY_12:
        log('\nONLY_SECTIONS_1_2=1: stopping here without writing.')
        return
    assoc.to_csv(os.path.join(OUT, "118_associations.csv"), index=False)
    part.to_csv(os.path.join(OUT, "118_partitions_vs_duration_stage.csv"), index=False)

    # ------------------------------------------------------------------ 3 ----
    title('3. DECISION RULE ON THE VARIABLES AFTER REMOVING DURATION AND STAGE')
    mat, ns, r2v = {}, {}, {}
    for tag in ("24h", "150h"):
        cl = clin[tag]
        ok = ~np.isnan(cl["hy"])
        mat[("V0", tag)] = X[tag]
        mat[("V1", tag)], r2v[("V1", tag)] = residuals(
            X[tag][ok], np.column_stack([np.log(cl['duration'][ok]), cl['stage2'][ok]]))
        mat[("V2", tag)], r2v[("V2", tag)] = residuals(X[tag], np.log(cl['duration'])[:, None])
        for var in VARIANTS:
            ns[(var, tag)] = len(mat[(var, tag)])
        from sklearn.metrics import adjusted_rand_score
        l0 = _rule_functions.labels_k2(X[tag])["kmeans"]
        for var, rows in (("V1", ok), ("V2", np.ones(len(ok), bool))):
            l1 = _rule_functions.labels_k2(mat[(var, tag)])["kmeans"]
            log(f"{tag}, {var} ({VARIANTS[var]}): n = {ns[var, tag]}; R2 removed per variable "
                f"{', '.join((f'{v:.3f}' for v in r2v[var, tag]))}; ARI between the k-means "
                f"partition at k = 2 before and after: {adjusted_rand_score(l0[rows], l1):.3f}")
    log(f"\nRealisations of the null: criteria at k = 2 {B_K2['24h']} and {B_K2['150h']}; curves, "
        f"published B {B_PUBL['24h']} and {B_PUBL['150h']}; extended B silhouette {B_SIL}, "
        f"prediction strength {B_PS}. Processes: {N_PROC}.")

    tab_b, tab_c, rules, raws = [], [], [], {}
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        for phase, variants in (('identity', ["V0"]), ('variants', ["V1", "V2"])):
            tasks = [c for var in variants for tag in ("24h", "150h")
                       for c in tasks_variant(var, tag, mat[(var, tag)])]
            # the long tasks first: the order does not change the results
            tasks.sort(key=lambda c: 0 if c['kind'] == 'lone' else 1)
            log(f"\nPhase '{phase}': {len(tasks)} tasks...")
            result = []
            for i, r in enumerate(pool.imap_unordered(execute, tasks, chunksize=1)):
                result.append(r)
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    print(f'    {i + 1}/{len(tasks)}  ({time.time() - t0:.0f} s)', flush=True)
            data = collect(result)
            for var in variants:
                b, c = tables_variant(var, data, ns)
                tab_b.append(b)
                tab_c.append(c)
                rules.append(rule_variant(var, b, c))
                for tag in ("24h", "150h"):
                    o = data[f"{var}|{tag}"]
                    raws[f'{var}_{tag}_null_silhouette'] = o['null_sil']
                    raws[f'{var}_{tag}_null_PS'] = o['null_ps']
                    raws[f'{var}_{tag}_null_k2'] = o['nulls'].to_numpy(float)
                    raws[f'{var}_{tag}_matrix'] = mat[(var, tag)]
            if phase == 'identity':
                worse = identity_v0(tab_b[0], tab_c[0])
                log(f'Identity: V0 reproduces 25_nulls.csv (observed value, median and range of the '
                    f'null, p) and 45_diagnostic_curves.csv (observed value, median and range of the '
                    f'null, p, corrected p) in both cohorts; maximum difference {worse:.1e}.')
    B = pd.concat(tab_b, ignore_index=True)
    C = pd.concat(tab_c, ignore_index=True)
    Rg = pd.DataFrame(rules)
    B.to_csv(os.path.join(OUT, "118_battery_k2.csv"), index=False)
    C.to_csv(os.path.join(OUT, "118_curves_k2_10.csv"), index=False)
    Rg.to_csv(os.path.join(OUT, "118_rule.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "118_nulls.npz"), names_k2=np.array(
        list(data[next(iter(data))]['nulls'].columns)), **raws)

    title('3a. CRITERIA AT k = 2 (FIRST ARM)')
    for var in VARIANTS:
        for tag in ("24h", "150h"):
            s = B[(B.variant == var) & (B.cohort == tag)]
            log(f'\n{var} ({VARIANTS[var]}), {tag}, n = {ns[var, tag]}, B = {B_K2[tag]}')
            log(s[['statistic', 'observed', 'null_median', 'null_CI2_5', 'null_CI97_5',
                   "p_montecarlo", 'observed_mean_20_streams', 'p_at_mean_of_streams']]
                .round(4).to_string(index=False))
    title('3b. CURVES OVER k = 2-10 (SECOND ARM)')
    for var in VARIANTS:
        for stat in STAT:
            s = C[(C.variant == var) & (C.statistic == stat)]
            log(f'\n{var} ({VARIANTS[var]}), {stat}')
            log(s[['cohort', "k", 'observed', 'null_median_B_extended', 'p_B_published',
                   'p_BH_B_published', 'p_B_extended', 'p_BH_B_extended', 'p_sup95_B_extended']]
                .round(4).to_string(index=False))
    title('3c. OUTCOME OF THE RULE')
    for _, r in Rg.iterrows():
        log(f'\n{r.variant} ({r.description})')
        for tag in ("24h", "150h"):
            log(f"  first arm, {tag}: criteria that exceed the null: "
                f"{r[f'first_branch_{tag}_criteria_that_exceed'] or 'noone'} (minimum p "
                f"{r[f'first_branch_{tag}_p_minimum']:.4f}); at the mean of the streams: "
                f"{r[f'first_branch_{tag}_criteria_that_exceed_at_mean_of_streams'] or 'noone'}")
        for reading in ('B_extended', 'B_published'):
            for stat in STAT:
                log(f"  second arm, {stat}, {reading.replace('_', ' ')}: surviving "
                    f"{r[f'second_branch_{stat}_k_that_survive_{reading}'] or 'noone'}; minimum "
                    f"corrected p {r[f'second_branch_{stat}_p_BH_minimum_{reading}']:.4f} "
                    f"({r[f'second_branch_{stat}_where_{reading}']})")
            log(f"  RULE with {reading.replace('_', ' ')}: "
                + ('SATISFIED' if r[f'rule_satisfied_{reading}'] else 'not satisfied'))
        if r.first_branch_satisfied != r.first_branch_satisfied_at_mean_of_streams:
            log("  WARNING: the outcome of the first arm changes between the stream of 25 and the mean of the streams.")
    v0 = Rg[Rg.variant == "V0"].iloc[0]
    if v0.rule_satisfied_B_extended != v0.rule_satisfied_B_published:
        log("\nWARNING: in V0 the outcome of the rule changes between published B and extended B.")

    title('WHAT THIS SCRIPT DOES NOT DO')
    log('- It removes linear mean differences in the logarithm of duration and between the two '
        'stages; not differences in variance or in correlation.')
    log("- It says nothing about later stages: the range of duration and of stage is that of "
        "section 1.")
    log('- The p values of the criteria at k = 2 are not corrected across criteria, cohorts and variants.')
    log("- The gap is not recomputed (it does not enter the rule).")
    log(f'\nTotal time: {time.time() - t0:.0f} s')
    with open(os.path.join(OUT, "118_duration_stage.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
