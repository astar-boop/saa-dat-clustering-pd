"""
119 - The shared participants analysed on their own, and the two cohorts
without the shared participants.

Why it exists. Reviewer 2 ("Different protocols") asks literally to
"analyse the shared participants separately". Script 117 responds by reading
on the shared participants the partitions estimated on the whole cohorts, and
the draft of the response dismissed the analysis on the shared participants
alone with a sentence without numbers ("it distinguishes nothing"). The checker of
the draft pointed it out
(revisione_R2_rapporti/controlli_bozza/controllo_bozza_v1.txt, B6), and also
pointed out that the sentence "the result does not depend on the shared participants" rested
only on the criteria at k = 2 of the replication cohort (B7). This script does
the two things that were missing.

Rules, written before the results of this script (the results of 117 and of
118 were known).
  A. The shared participants on their own. For each version of the assay, the
     seven variables of the shared participants only, re-standardised within
     the shared participants; battery at k = 2 of 25 (functions of _rule_functions)
     against the Gaussian copula null fitted to the shared participants only;
     500 realisations for both versions. PS and Jaccard observed also at the
     mean of 20 streams. The curves over k = 2-10 are NOT computed: with about
     twenty subjects per half the prediction strength at many groups is not
     defined. Reading: the analysis is reported because the reviewer asks for
     it; to say how informative it is, the width of the reference interval of
     the null is reported alongside each criterion, compared with that of the
     whole cohort (from 118, variant V0). Any outcome must be read with that
     width in view: a "does not exceed" on 45 subjects is not evidence of
     absence, and an "exceeds" with uncorrected p on seven criteria and two
     versions is not evidence of groups.
  B. The two cohorts without the shared participants: primary cohort without
     the shared participants and replication cohort without the shared
     participants, variables re-standardised, whole decision rule (two arms)
     with the functions, the seeds and the numbers of realisations of 118
     (criteria at k = 2: 500 and 300; curves: silhouette 2000, prediction
     strength 500; Benjamini-Hochberg within statistic over the 18 tests of
     the two cohorts). Outcome of the rule with "B esteso", as in 118; that
     with "B pubblicato" alongside. The manuscript (Supplement §1.8) reports
     for the replication cohort without the shared participants only the
     criteria at k = 2, computed with another implementation: the values here
     need not coincide with those, and the report says so.

Checks. The matrices of the whole cohorts are those of 118 (same reading
functions; the 24 h one is compared with features.npz). The sets "without the
shared participants" are verified by identity of the PATNO: cohort minus
intersection. There is no saved value against which to verify the results of
this script: this is declared.

What the script does NOT do: it does not inject groups to measure the power of
the analysis on the shared participants alone; it does not compute the curves
on the shared participants alone; it does not correct the p of the criteria at
k = 2.

  A2. Addition after seeing A (declared): at 24 h one criterion out of three
     fell just below 0.05 in the shared participants alone, and 117 had shown
     that the shared participants were run in two periods. The partition at
     k = 2 of the shared participants alone at 24 h is compared with the run
     period (Fisher exact). It is deterministic; the complete execution of
     13/11/2026 was done before this addition and A2 was run separately with
     ONLY_SECTION_A2=1 (output 119_shared_alone_run.txt and .csv).

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 -B scripts/119_shared_alone.py
      N_PROCESSES=n   (default 10; the results do not depend on it)
      ONLY_SECTION_A2=1   runs section A2 only
Output: outputs/119_shared_alone.txt, 119_shared_alone_k2.csv,
        119_without_shared_k2.csv, 119_without_shared_curves.csv,
        119_without_shared_rule.csv
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

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "10")))
ONLY_A2 = os.environ.get('ONLY_SECTION_A2', "0") == "1"


from _modules import load_script as _module                      # noqa: E402


M118 = _module("118_duration_stage.py", 'duration_118')
import _rule_functions                                                    # noqa: E402
B_SOLE = 500
VAR_B = "S0"
M118.VARIANTS[VAR_B] = 'without the shared participants'
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


def execute(t):
    return _rule_functions.EXECUTORS[t['kind']](t)


def section_a2(c24, X, cond):
    """A2 (addition after seeing A, declared): does the partition at k = 2 of the
    shared participants alone at 24 h coincide with the run period of the
    assay? The shared participants were run in two periods more than a year apart."""
    from scipy import stats
    from sklearn.cluster import KMeans
    title('A2. THE SHARED PARTICIPANTS ALONE AT 24 h: THE PARTITION AT k = 2 AND THE RUN DATE')
    d = c24[c24.PATNO.isin(list(cond))].reset_index(drop=True)
    data = pd.to_datetime(d.RUNDATE)
    lab = KMeans(2, n_init=15, random_state=_rule_functions.SEED_B).fit_predict(X)   # the reference partition of the Jaccard
    months = pd.crosstab(data.dt.to_period("M"), lab)
    log('Group (columns) by month of assay run (rows):')
    log(months.to_string())
    first = data.dt.year == data.dt.year.min()
    tab = pd.crosstab(first, lab).reindex(index=[True, False], columns=[0, 1], fill_value=0)
    fi = stats.fisher_exact(tab.to_numpy())
    log(f'First year of assay runs ({int(data.dt.year.min())}) against the later years, by group: '
        f'{tab.to_numpy().tolist()}; Fisher exact p = {fi.pvalue:.2g}; subjects classified by the '
        f'period {int(max(np.trace(tab.to_numpy()), tab.to_numpy()[::-1].trace()))} of {len(d)}.')
    names = M118.NAMES["24h"]
    diff = X[lab == 1].mean(0) - X[lab == 0].mean(0)
    log('Difference between the two groups in SD (within the shared participants): '
        + ", ".join(f"{a} {v:+.2f}" for a, v in zip(names, diff)))
    pd.DataFrame({"PATNO": d.PATNO, "RUNDATE": d.RUNDATE, 'group_k2': lab}).to_csv(
        os.path.join(OUT, "119_shared_alone_run.csv"), index=False)
    if ONLY_A2:
        with open(os.path.join(OUT, "119_shared_alone_run.txt"), "w") as fh:
            fh.write("\n".join(rep) + "\n")


def main():
    t0 = time.time()
    title('119 - THE SHARED PARTICIPANTS ALONE, AND THE TWO COHORTS WITHOUT THE SHARED '
           'PARTICIPANTS')
    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy())
    R24 = c24[M118.F24].to_numpy(float).copy()
    R24[:, :4] = np.log(R24[:, :4])
    assert np.abs(M118.standardise(R24) - npz["F_primary"]).max() < M118.TOL, \
        'the 24 h matrix rebuilt from the cohort file does not match features.npz'
    R150 = c150[M118.F150].to_numpy(float).copy()
    R150[:, :4] = np.log(R150[:, :4])
    raw = {"24h": R24, "150h": R150}
    cohort = {"24h": c24, "150h": c150}
    cond = set(c24.PATNO) & set(c150.PATNO)
    log(f"Declared working directory: {('sporadic only' if ONLY else 'full cohort')}; n = {len(c24)} "
        f"(24 h) and {len(c150)} (150 h); shared participants {len(cond)}.")
    sole, without, ns = {}, {}, {}
    for tag in ("24h", "150h"):
        within = cohort[tag].PATNO.isin(list(cond)).to_numpy()
        assert set(cohort[tag].PATNO[within]) == cond
        assert set(cohort[tag].PATNO[~within]) == set(cohort[tag].PATNO) - cond
        sole[tag] = M118.standardise(raw[tag][within])
        without[tag] = M118.standardise(raw[tag][~within])
        ns[(VAR_B, tag)] = int((~within).sum())
        log(f'{tag}: shared participants {int(within.sum())}, without the shared participants '
            f'{int((~within).sum())}; variables re-standardised within each set.')

    if ONLY_A2:
        section_a2(c24, sole["24h"], cond)
        return

    tasks = []
    for tag in ("24h", "150h"):
        branch = f'sole|{tag}'
        tasks.append(dict(kind='lone', branch=branch, X=sole[tag], B=B_SOLE))
        tasks.append(dict(kind='streams', branch=branch, X=sole[tag], label=f'119|{branch}'))
        tasks += M118.tasks_variant(VAR_B, tag, without[tag])
    tasks.sort(key=lambda c: 0 if c['kind'] == 'lone' else 1)
    log(f'\n{len(tasks)} tasks on {N_PROC} processes...')
    result = []
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        for i, r in enumerate(pool.imap_unordered(execute, tasks, chunksize=1)):
            result.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f'    {i + 1}/{len(tasks)}  ({time.time() - t0:.0f} s)', flush=True)

    # ------------------------------------------------------------------ A ----
    title('A. THE SHARED PARTICIPANTS ALONE: CRITERIA AT k = 2')
    ref = pd.read_csv(os.path.join(OUT, "118_battery_k2.csv"))
    ref = ref[ref.variant == "V0"]
    rows = []
    for tag in ("24h", "150h"):
        branch = f'sole|{tag}'
        u = next(r for r in result if r['branch'] == branch and r['kind'] == 'lone')
        f = pd.DataFrame(next(r for r in result if r['branch'] == branch and r['kind'] == 'streams')['streams'])
        N = pd.DataFrame(u['nulls'])
        assert len(N) == B_SOLE
        for stat, obsv in u['obsv'].items():
            high = _rule_functions.MORE_STRUCTURE[stat]
            v = N[stat].to_numpy(float)
            q = ref[(ref.cohort == tag) & (ref.statistic == stat)].iloc[0]
            r = dict(version=tag, n=len(sole[tag]), statistic=stat, observed=float(obsv),
                     null_median=float(np.median(v)), null_CI2_5=float(np.percentile(v, 2.5)),
                     null_CI97_5=float(np.percentile(v, 97.5)),
                     p_montecarlo=_rule_functions.p_mc(v, obsv, high), B=len(v),
                     criterion_of_rule=stat in _rule_functions.CRITERIA,
                     width_interval=float(np.percentile(v, 97.5) - np.percentile(v, 2.5)),
                     width_interval_cohort_whole=float(q.null_CI97_5 - q.null_CI2_5),
                     n_cohort_whole=int(q.n))
            if stat in f.columns:
                r.update(observed_mean_20_streams=float(f[stat].mean()),
                         p_at_mean_of_streams=_rule_functions.p_mc(v, float(f[stat].mean()), high))
            rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(OUT, "119_shared_alone_k2.csv"), index=False)
    for tag in ("24h", "150h"):
        s = A[A.version == tag]
        log(f'\n{tag}, {int(s.n.iloc[0])} shared participants, {B_SOLE} realisations of the null')
        log(s[['statistic', 'observed', 'null_median', 'null_CI2_5', 'null_CI97_5',
               "p_montecarlo", 'p_at_mean_of_streams', 'width_interval',
               'width_interval_cohort_whole']].round(4).to_string(index=False))
        c = s[s.criterion_of_rule]
        log(f"  Criteria of the rule with p < 0.05 (uncorrected): "
            f"{', '.join(c[c.p_montecarlo < ALPHA].statistic) or 'noone'}; at the mean of the "
            f"streams: "
            f"{', '.join(c[c.p_at_mean_of_streams.fillna(c.p_montecarlo) < ALPHA].statistic) or 'noone'}. "
            f"Other statistics with p < 0.05: "
            f"{', '.join(s[~s.criterion_of_rule & (s.p_montecarlo < ALPHA)].statistic) or 'nothing'}.")
        rap = (c.width_interval / c.width_interval_cohort_whole)
        log(f"  Reference interval of the null for the three criteria: from {rap.min():.1f} to "
            f"{rap.max():.1f} times wider than in the whole cohort.")

    section_a2(c24, sole["24h"], cond)

    # ------------------------------------------------------------------ B ----
    title('B. THE TWO COHORTS WITHOUT THE SHARED PARTICIPANTS: DECISION RULE')
    data = M118.collect([r for r in result if r['branch'].startswith(VAR_B)])
    b, c = M118.tables_variant(VAR_B, data, ns)
    reg = M118.rule_variant(VAR_B, b, c)
    b.to_csv(os.path.join(OUT, "119_without_shared_k2.csv"), index=False)
    c.to_csv(os.path.join(OUT, "119_without_shared_curves.csv"), index=False)
    pd.DataFrame([reg]).to_csv(os.path.join(OUT, "119_without_shared_rule.csv"), index=False)
    for tag in ("24h", "150h"):
        s = b[b.cohort == tag]
        log(f'\n{tag} without the shared participants, n = {ns[VAR_B, tag]}, B = {M118.B_K2[tag]}')
        log(s[['statistic', 'observed', 'null_median', 'null_CI2_5', 'null_CI97_5',
               "p_montecarlo", 'observed_mean_20_streams', 'p_at_mean_of_streams']]
            .round(4).to_string(index=False))
    for stat in M118.STAT:
        s = c[c.statistic == stat]
        log(f"\n{stat}")
        log(s[['cohort', "k", 'observed', 'null_median_B_extended', 'p_B_published',
               'p_BH_B_published', 'p_B_extended', 'p_BH_B_extended']].round(4).to_string(index=False))
    log("")
    for tag in ("24h", "150h"):
        log(f"first arm, {tag}: criteria that exceed the null: "
            f"{reg[f'first_branch_{tag}_criteria_that_exceed'] or 'noone'} (minimum p "
            f"{reg[f'first_branch_{tag}_p_minimum']:.4f}); at the mean of the streams: "
            f"{reg[f'first_branch_{tag}_criteria_that_exceed_at_mean_of_streams'] or 'noone'}")
    for reading in ('B_extended', 'B_published'):
        for stat in M118.STAT:
            log(f"second arm, {stat}, {reading.replace('_', ' ')}: surviving: "
                f"{reg[f'second_branch_{stat}_k_that_survive_{reading}'] or 'noone'}; minimum "
                f"corrected p {reg[f'second_branch_{stat}_p_BH_minimum_{reading}']:.4f} "
                f"({reg[f'second_branch_{stat}_where_{reading}']})")
        log(f"RULE with {reading.replace('_', ' ')}: "
            + ('SATISFIED' if reg[f'rule_satisfied_{reading}'] else 'not satisfied'))
    log("\nSupplement §1.8 reports for the replication cohort without the shared participants the "
        "criteria at k = 2 computed with another implementation: the values above need not coincide "
        "with those. There is no saved value against which to verify these results.")

    title('WHAT THIS SCRIPT DOES NOT DO')
    log("- It does not inject groups to measure the power of the analysis on the shared "
        "participants alone.")
    log('- It does not compute the curves over k = 2-10 on the shared participants alone.')
    log('- The p values of the criteria at k = 2 are not corrected.')
    log(f'\nTotal time: {time.time() - t0:.0f} s')
    with open(os.path.join(OUT, "119_shared_alone.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
