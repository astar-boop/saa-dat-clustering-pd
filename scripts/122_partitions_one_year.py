"""
122 - Do the two versions of the assay group the same participants in the same
way? The comparison with power: 150 h at baseline, 24 h at one year.

Why it exists. Script 117 compares the partitions of the two cohorts in the 45
shared participants, and can see only a strong agreement. The sporadic members
of the 150 h cohort who also have a Positive/Type1 sample at 24 h at visit V04
(one year later) are many more (229 in 106, Table S16), and in them the single
parameters agree weakly between the versions (time to threshold 0.30; area
under the curve 0.26). The question here: in these participants, do the
partitions obtained with the two versions agree? Decision of the user of
13/11/2026 (decision 8 of the note to the draft of the response to
Reviewer 2).

What was known before these rules were written: the correlations of the single
parameters in the 229 (106) and all the results of 117 on the 45. NO result on
the partitions in the 229 was known.

Data.
  Participants: sporadic members of the 150 h cohort with at least one
  Positive/Type1 sample at 24 h at visit V04. It is the set "coorte a 150 ore,
  sporadici Type1" of 106: the script reproduces its Spearman correlations
  for Fmax, TTT and AUC, which must coincide with 106_agreement_kinetics.csv.
  150 h values: those of the cohort file (baseline; median of the wells):
  Fmax, T50, AUC, SLOPE, that is, the kinetic parameters of the manuscript set.
  24 h values: Positive/Type1 samples of V04, mean of the three wells and then
  mean of the samples (convention of the cohort files and of 106): Fmax, TTT,
  AUC, SLOPEMax, the kinetic parameters of the manuscript set at 24 h.
  DAT and age: at baseline for the 150 h side (cohort file), at V04 for the
  24 h side (curated data cut). They are not
  the same measurements: there is about one year between the two.

Analyses (rules and readings fixed here).
  1. MAIN. Kinetic parameters only, all participants. The variables of each
     version are standardised within these participants and the partition at
     k = 2 is estimated ON THEM, separately per version, with k-means, Ward
     and Gaussian mixture (constructors of 25, via 117). For each algorithm:
     2 x 2 table, participants in corresponding groups (maximum over the two
     correspondences), expected by chance, minimum for p < 0.05, ARI, exact
     one-sided p with fixed margins. Benjamini-Hochberg over the rows in which
     p < 0.05 is attainable. Average linkage is reported only if its smaller
     group has at least ten participants in both versions.
     For each partition the difference between the two groups is reported in
     standard deviations, variable by variable: it says ALONG WHAT it cuts.
  2. Power of the test of point 1 (same model as 117: two latent scores
     with correlation rho, cut at the observed sizes of the groups).
  3. Seven variables, in the participants with DAT and age at V04: as point 1.
     Alongside, DAT and age alone (control: similar measurements one year
     apart, not identical).
  4. Secondary: the 150 h partition estimated on the WHOLE replication cohort
     (that of the manuscript) read on these participants, against the 24 h
     partition of point 1. k-means at k = 3 and k = 4 on the kinetic
     parameters only (permutation p, uncorrected).
  5. Without partitions: canonical correlations between the four kinetic
     parameters at 150 h and the four at 24 h, with a permutation test on
     Pillai's trace; repeated after centring the logarithms of BOTH versions
     on the mean of their run date (means over all the positive PD samples of
     the same version run on that date, as in 106).
  Readings.
    - If the partitions agree above chance in point 1: something stable in the
      participants passes from one version to the other also at the level of
      the groups; the sentence of R2.2 "we did not detect agreement" would
      hold only for the 45.
    - If they do not agree, with a high power for a moderate agreement: within
      the limits of the comparison (one year between the samples, run dates,
      storage) the two versions do not group the same participants in the same
      way, while agreeing weakly on the single parameters. It remains true
      that, if there are no groups, two cuts in half have no reason to
      coincide.
    - Point 5 says whether any linear relation at all exists between the two
      sets of kinetic parameters: it is expected that it does (0.30 and 0.26
      on the single parameters).
  Declared limitations: about one year passes between the two samples; the two
  versions were run in different periods; at 24 h the samples are of type 1 by
  construction and at 150 h the type does not exist; the partitions of point 1
  are estimated on these participants only and are not those of the manuscript.

ADDITIONS AFTER THE INDEPENDENT VERIFICATION (13/11/2026; report in
revisione_R2_rapporti/verifiche_script/riverifica_122.txt; the verified
version is in revisione_R2_rapporti/versioni_script/). The verifier
reproduced all the numbers and showed that the reading "they do not agree, with
high power" is not available, for two reasons that this script now measures:
  6. The power of point 2 assumes ONE score common to the two versions. The
     parameters used instead agree weakly and with mixed signs (Fmax and
     slope negative). Pairs of sets are simulated with the OBSERVED joint
     correlation matrix of the eight kinetic parameters (Gaussian), the two
     partitions are redone and it is counted how often the test gives
     p < 0.05: it is what one expects to see if the only true thing is the
     agreement of the parameters that is observed.
  7. The result depends on how the three wells at 150 h are summarised. The
     cohort file also carries the MEAN of the wells (columns kmean_, the
     sensitivity analysis provided for by 08): point 1 is repeated with that.
     It is a variant chosen after seeing the results, one among several tried
     by the verifier, and its p are not corrected.
Two wordings are also corrected: the correlations reported at the top are now
those of the parameters that enter the partitions (at 150 h the T50, not the
TTT), and the report says plainly what the comparison does not distinguish.

Usage:  CLUSTERING_BASE=<radice> ONLY_SPORADIC=1 python3 -B scripts/122_partitions_one_year.py
Output: outputs/122_partitions_one_year.txt, 122_partitions.csv, 122_power.csv,
        122_groups_variables.csv, 122_canonical.csv, 122_power_realistic.csv,
        122_mean_of_wells.csv
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util                                            # noqa: E402
import sys                                                       # noqa: E402
import warnings                                                  # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402
from sklearn.cluster import AgglomerativeClustering              # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari           # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import DATA                                       # noqa: E402

BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
FILE_SAA = "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"
CACHE = os.path.join(OUT, "cache", "curated.parquet")
VISIT = "V04"


from _modules import load_script as _module                      # noqa: E402


M117 = _module("117_shared_partitions.py", 'shared_117')
import _rule_functions                                                    # noqa: E402
ALPHA, NPERM, N_POWER, RHO = M117.ALPHA, M117.NPERM, M117.N_POWER, M117.RHO
ALG = M117.ALG
K150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE"]
R24 = ["Fmax_24h", "TTT_24h", "AUC_24h", "SLOPEMax_24h"]
N150 = ["log Fmax", "log T50", "log AUC", "log SLOPE"]
N24 = ["log Fmax", "log TTT", "log AUC", "log SLOPEMax"]
NDAT = ['lower putamen', 'mean caudate', 'age']
GROUP_MIN = 10

rep = []
log_p = M117.fmt_p


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def st(M):
    return (M - M.mean(0)) / M.std(0)


def partitions(X):
    P = M117.three_k2(X)
    P["average"] = AgglomerativeClustering(2, linkage="average").fit_predict(X)
    return P


def comparison(PA, PB, label, rows, tables):
    """Compares the partitions of two versions on the same participants."""
    n = len(next(iter(PA.values())))
    for alg in ALG + ["average"]:
        la, lb = PA[alg], PB[alg]
        small = min(np.bincount(la, minlength=2).min(), np.bincount(lb, minlength=2).min())
        if alg == "average" and small < GROUP_MIN:
            log(f"  average  not reported: smaller group of {small} participants ({'/'.join(map(str, np.bincount(la, minlength=2)))} and {'/'.join(map(str, np.bincount(lb, minlength=2)))})")
            continue
        r, _ = M117.exact(la, lb)
        assert abs(r["ARI"] - ari(la, lb)) < 1e-12
        r.update(analysis=label, algorithm=alg, n=n,
                 groups_150h="/".join(map(str, np.bincount(la, minlength=2))),
                 groups_24h="/".join(map(str, np.bincount(lb, minlength=2))),
                 without_power=bool(r['p_minimum_reachable'] >= ALPHA))
        rows.append(r)
        tables[(label, alg)] = (int(la.sum()), int(lb.sum()), n)
        mn = r['same_group_minimum_by_p05']
        log(f"  {alg:8s} groups at 150 h {r['groups_150h']:>8s}, at 24 h {r['groups_24h']:>8s}; in corresponding groups {r['same_group']}/{n} (expected by chance {r['expected_by_chance']:.1f}; {('-' if np.isnan(mn) else int(mn))} needed for p < 0.05); ARI {r['ARI']:+.3f}; exact p {log_p(r['p_exact'])}"
            + ('  [no power]' if r['without_power'] else ""))


def differences(X, lab, names, version, label, alg, rows):
    d = X[lab == 1].mean(0) - X[lab == 0].mean(0)
    for name, v in zip(names, d):
        rows.append(dict(analysis=label, version=version, algorithm=alg, variable=name,
                          difference_in_SD=float(v)))
    return ", ".join(f"{a} {v:+.2f}" for a, v in zip(names, d))


def main():
    title('122 - 150 H AT BASELINE AGAINST 24 H AT ONE YEAR: SAME GROUPS?')
    saa = pd.read_csv(os.path.join(DATA, FILE_SAA), low_memory=False)
    saa["PATNO"] = saa["PATNO"].astype("int64")
    saa["RUNDATE"] = pd.to_datetime(saa["RUNDATE"])
    cur = pd.read_parquet(CACHE, columns=["PATNO", "EVENT_ID", "subgroup", "age_at_visit",
                                          "MIA_PUTAMEN_L", "MIA_PUTAMEN_R", "MIA_CAUDATE_BILAT"])
    cur["PATNO"] = cur["PATNO"].astype("int64")
    bl = cur[cur.EVENT_ID == "BL"].set_index("PATNO")
    v04 = cur[cur.EVENT_ID == VISIT].set_index("PATNO")
    sporadic = set(bl.index[bl.subgroup == "Sporadic PD"])
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv").set_index("PATNO")
    assert set(c150.index) <= sporadic or not ONLY, 'working directory and ONLY_SPORADIC do not agree'

    # ---- 24 h at V04: Positive/Type1 samples, mean of the wells, mean of the samples
    s24 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == VISIT)
              & (saa.SAA_Status == "Positive") & (saa.SAA_Type == "Type1")].copy()
    for r in R24:
        s24[r] = s24[[f"{r}_Rep{i}" for i in (1, 2, 3)]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    k24 = s24.groupby("PATNO")[R24].mean()
    ids = sorted(set(k24.index) & set(c150.index) & sporadic)
    n = len(ids)
    A = np.log(c150.loc[ids, K150].to_numpy(float))            # 150 h, baseline
    B = np.log(k24.loc[ids, R24].to_numpy(float))              # 24 h, V04
    assert np.isfinite(A).all() and np.isfinite(B).all()
    log(f"Declared working directory: {('sporadic only' if ONLY else 'full cohort')}. Members of the 150 h cohort (n = {len(c150)}) with a Positive/Type1 sample at 24 h at {VISIT}: {n}; of these, also in the primary cohort: {len(set(ids) & set(_rule_functions.load_cohort('analysis_complete_cases.csv').PATNO))}.")

    # ---- identity with 106 (Fmax, TTT, AUC: the parameters with the same name)
    p106 = pd.read_csv(os.path.join(OUT, "106_agreement_kinetics.csv"))
    q = p106[p106.block.str.startswith('different visits, between protocols: 150 h BL versus 24 h V04')
             & p106.dataset.str.startswith('150 h cohort, sporadic Type1')
             & (p106.variant == 'mean/median') & p106.quantity.str.startswith("rho [")]
    raw150 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 155) & (saa.CLINICAL_EVENT == "BL")
                 & (saa.SAA_Status == "Positive")].copy()
    for name in ("Fmax", "TTT", "AUC"):
        raw150[name] = raw150[[f'{name}_150h_Rep{i}' for i in (1, 2, 3)]].apply(
            pd.to_numeric, errors="coerce").median(axis=1)
    g150 = raw150.groupby("PATNO")[["Fmax", "TTT", "AUC"]].mean().reindex(ids)
    worse = 0.0
    for name in ("Fmax", "TTT", "AUC"):
        r = q[q.parameter == name]
        assert len(r) == 1 and int(r.n.iloc[0]) == n, \
            f'106: row of {name} not found or different n ({len(r)} rows; n = {n})'
        rho = stats.spearmanr(g150[name], k24.loc[ids, f'{name}_24h']).statistic
        d = abs(rho - float(r.estimate.iloc[0]))
        worse = max(worse, d)
        assert d < 1e-9, f'{name}: Spearman {rho:.6f} does not reproduce 106 ({float(r.estimate.iloc[0]):.6f})'
    assert np.allclose(g150["Fmax"].to_numpy(), c150.loc[ids, "k_Fmax"].to_numpy(), rtol=1e-9), \
        'Fmax at 150 h of the cohort file different from the raw value'
    log(f'Identity: the Spearman correlations of Fmax, TTT and AUC reproduce 106 (n = {n}; maximum difference {worse:.1e}); Fmax at 150 h of the cohort file coincides with the raw value.')

    log('Concordance between versions of the parameters that enter the partitions (Spearman, 150 h '
        'against 24 h): ' + ", ".join(
            f"{a} / {b} {stats.spearmanr(A[:, j], B[:, j]).statistic:+.3f}"
            for j, (a, b) in enumerate(zip(N150, N24))) + ".")
    rows, tables, diff = [], {}, []

    # ------------------------------------------------------------------ 1 ----
    title('1. KINETIC PARAMETERS ONLY: PARTITIONS AT k = 2 ESTIMATED ON THESE PARTICIPANTS')
    XA, XB = st(A), st(B)
    PA, PB = partitions(XA), partitions(XB)
    et1 = '1. kinetic parameters only'
    comparison(PA, PB, et1, rows, tables)
    log('Along what they cut (difference between the two groups, in SD):')
    for alg in ALG:
        log(f"  {alg:8s} 150 h: {differences(XA, PA[alg], N150, '150h', et1, alg, diff)}")
        log(f"  {alg:8s} 24 h:  {differences(XB, PB[alg], N24, '24h', et1, alg, diff)}")

    # ------------------------------------------------------------------ 3 ----
    title("3. SEVEN VARIABLES (PARTICIPANTS WITH DAT AND AGE AT V04)")
    d150 = np.column_stack([c150.loc[ids, "dat_putamen_low"], c150.loc[ids, "dat_caudate_mean"],
                            c150.loc[ids, "age_bl"]]).astype(float)
    w = v04.reindex(ids)
    d24 = np.column_stack([w[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1, skipna=False),
                           w.MIA_CAUDATE_BILAT, w.age_at_visit]).astype(float)
    ok = np.isfinite(d24).all(1) & np.isfinite(d150).all(1)
    log(f'With DAT and age at {VISIT}: {int(ok.sum())} of {n}. Between baseline and {VISIT}: age +{np.median(d24[ok, 2] - d150[ok, 2]):.2f} years (median); Spearman correlation putamen {stats.spearmanr(d150[ok, 0], d24[ok, 0]).statistic:.2f}, caudate {stats.spearmanr(d150[ok, 1], d24[ok, 1]).statistic:.2f}.')
    SA, SB = st(np.column_stack([A[ok], d150[ok]])), st(np.column_stack([B[ok], d24[ok]]))
    et3 = '3. seven variables'
    Q7A, Q7B = partitions(SA), partitions(SB)
    comparison(Q7A, Q7B, et3, rows, tables)
    for alg in ALG:
        log(f"  {alg:8s} 150 h: {differences(SA, Q7A[alg], N150 + NDAT, '150h', et3, alg, diff)}")
        log(f"  {alg:8s} 24 h:  {differences(SB, Q7B[alg], N24 + NDAT, '24h', et3, alg, diff)}")
    log("\nControl: DAT and age only (baseline against V04)")
    etc = '3c. DAT and age only (check)'
    comparison(partitions(st(d150[ok])), partitions(st(d24[ok])), etc, rows, tables)

    # ------------------------------------------------------------------ 4 ----
    title('4. SECONDARY')
    R150 = c150[M117.F150].to_numpy(float).copy()
    R150[:, :4] = np.log(R150[:, :4])
    Xc = st(R150)
    pos = c150.index.get_indexer(ids)
    log("Partition at 150 h estimated on the whole replication cohort (seven variables, that of the "
        "manuscript) read on these participants, against the partition at 24 h of point 1:")
    Pc = {a: v[pos] for a, v in _rule_functions.labels_k2(Xc).items()}
    comparison(Pc, PB, '4. whole cohort at 150 h against point 1 at 24 h', rows, tables)
    log('k-means at k = 3 and k = 4, kinetic parameters only (permutation p, uncorrected):')
    rows_k = []
    for k in (3, 4):
        la, lb = M117.kmeans_k(XA, k), M117.kmeans_k(XB, k)
        o = float(ari(la, lb))
        g = M117.gen(f"122|kmeans|k{k}")
        pm = np.array([ari(la, g.permutation(lb)) for _ in range(NPERM)])
        p = float((np.sum(pm >= o - 1e-12) + 1) / (NPERM + 1))
        rows_k.append(dict(analysis='4. k-means, kinetic parameters only', algorithm="kmeans", k=k, n=n,
                            ARI=o, p_permutation=p))
        log(f"  k = {k}: groups {'/'.join(map(str, np.bincount(la)))} and "
            f"{'/'.join(map(str, np.bincount(lb)))}; ARI {o:+.3f}; p {log_p(p)}")

    t = pd.DataFrame(rows)
    sel = t.analysis.isin([et1, et3]) & ~t.without_power
    t["p_BH"] = np.nan
    t.loc[sel, "p_BH"] = _rule_functions.bh(t.loc[sel, 'p_exact'].to_numpy())
    log(f"\nBenjamini-Hochberg on the {int(sel.sum())} rows of points 1 and 3 in which p < 0.05 is attainable: minimum corrected p {t.loc[sel, 'p_BH'].min():.4f}; with uncorrected p < 0.05: {int((t.loc[sel, 'p_exact'] < ALPHA).sum())}; with corrected p < 0.05: {int((t.loc[sel, 'p_BH'] < ALPHA).sum())}.")
    pd.concat([t, pd.DataFrame(rows_k)], ignore_index=True).to_csv(
        os.path.join(OUT, "122_partitions.csv"), index=False)
    pd.DataFrame(diff).to_csv(os.path.join(OUT, "122_groups_variables.csv"), index=False)

    # ------------------------------------------------------------------ 2 ----
    title('2. POWER OF THE TEST OF POINTS 1 AND 3')
    log("Two normal latent scores with correlation rho; in each version the participants with the "
        "highest score form the smaller group, with the observed sizes.")
    rp = []
    for (label, alg), (r1, c1, m) in tables.items():
        if label not in (et1, et3):
            continue
        r1, c1 = min(r1, m - r1), min(c1, m - c1)
        tav = M117.tables_exacts(r1, c1, m).set_index("n11")
        g = M117.gen(f'122|power|{label}|{alg}')
        row = dict(analysis=label, algorithm=alg, n=m, margins=f"{r1}/{m - r1} e {c1}/{m - c1}")
        for rho in (0.15, 0.3, 0.5, 0.7):
            z = g.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=(N_POWER, m))
            a = np.argsort(-z[:, :, 0], axis=1)[:, :r1]
            b = np.argsort(-z[:, :, 1], axis=1)[:, :c1]
            n11 = np.array([len(np.intersect1d(x, y)) for x, y in zip(a, b)])
            row[f"rho_{rho}"] = float((tav.p.reindex(n11).to_numpy() < ALPHA).mean())
        rp.append(row)
    pwr = pd.DataFrame(rp)
    log(pwr.round(2).to_string(index=False))
    pwr.to_csv(os.path.join(OUT, "122_power.csv"), index=False)

    # ------------------------------------------------------------------ 5 ----
    title('5. WITHOUT PARTITIONS: CANONICAL CORRELATIONS BETWEEN THE KINETIC PARAMETERS')
    c0, p0 = M117.pillai_perm(XA, XB, '122|canonical|raw')
    log(f"Four kinetic parameters at 150 h (baseline) against four at 24 h ({VISIT}), n = {n}: canonical correlations {', '.join((f'{x:.2f}' for x in c0))}; Pillai trace {(c0 ** 2).sum():.3f}; permutation p {log_p(p0)}.")
    # centring on the run date, both versions
    everyone24 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237) & (saa.SAA_Status == "Positive")].copy()
    everyone150 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 155) & (saa.SAA_Status == "Positive")].copy()
    for r in R24:
        everyone24[r] = np.log(everyone24[[f"{r}_Rep{i}" for i in (1, 2, 3)]].apply(
            pd.to_numeric, errors="coerce").mean(axis=1))
        everyone24[r] = everyone24[r] - everyone24.groupby("RUNDATE")[r].transform("mean")
    R150c = ["Fmax_150h", "T50_150h", "AUC_150h", "SLOPE_150h"]
    for r in R150c:
        everyone150[r] = np.log(everyone150[[f"{r}_Rep{i}" for i in (1, 2, 3)]].apply(
            pd.to_numeric, errors="coerce").median(axis=1))
        everyone150[r] = everyone150[r] - everyone150.groupby("RUNDATE")[r].transform("mean")
    a24 = everyone24[(everyone24.CLINICAL_EVENT == VISIT) & (everyone24.SAA_Type == "Type1")] \
        .groupby("PATNO")[R24].mean().reindex(ids).to_numpy(float)
    a150 = everyone150[everyone150.CLINICAL_EVENT == "BL"].groupby("PATNO")[R150c].mean() \
        .reindex(ids).to_numpy(float)
    assert np.isfinite(a24).all() and np.isfinite(a150).all()
    nd24 = everyone24[(everyone24.CLINICAL_EVENT == VISIT) & everyone24.PATNO.isin(ids)].RUNDATE.nunique()
    nd150 = everyone150[(everyone150.CLINICAL_EVENT == "BL") & everyone150.PATNO.isin(ids)].RUNDATE.nunique()
    c1, p1 = M117.pillai_perm(st(a150), st(a24), "122|canoniche|centrate")
    log(f"After centring the logarithms of both versions on the mean of the run date (distinct dates "
        f"among these participants: {nd150} at 150 h, {nd24} at 24 h): canonical correlations "
        f"{', '.join((f'{x:.2f}' for x in c1))}; Pillai trace {(c1 ** 2).sum():.3f}; permutation p "
        f"{log_p(p1)}.")
    log('Partitions at k = 2 on the kinetic parameters centred on the run date:')
    rc, tc = [], {}
    comparison(partitions(st(a150)), partitions(st(a24)), '5. kinetic parameters centred on the run date',
              rc, tc)
    pd.DataFrame([dict(analysis='kinetic parameters as in the cohort files', n=n, pillai=float((c0 ** 2).sum()),
                       p_permutation=p0, **{f"r{i + 1}": float(x) for i, x in enumerate(c0)}),
                  dict(analysis='both centred on the run date', n=n,
                       pillai=float((c1 ** 2).sum()), p_permutation=p1,
                       **{f"r{i + 1}": float(x) for i, x in enumerate(c1)})]).to_csv(
        os.path.join(OUT, "122_canonical.csv"), index=False)
    pd.concat([pd.read_csv(os.path.join(OUT, "122_partitions.csv")), pd.DataFrame(rc)],
              ignore_index=True).to_csv(os.path.join(OUT, "122_partitions.csv"), index=False)

    # ------------------------------------------------------------------ 6 ----
    title('6. REALISTIC POWER: SIMULATED DATA SETS WITH THE OBSERVED CORRELATION BETWEEN THE VERSIONS')
    kmA = np.log(c150.loc[ids, ["kmean_Fmax", "kmean_T50", "kmean_AUC", "kmean_SLOPE"]].to_numpy(float))
    assert np.isfinite(kmA).all()
    N_REAL = 300
    rr = []
    for name, MA in (('median of the wells at 150 h (main analysis)', XA),
                     ('mean of the wells at 150 h', st(kmA))):
        C = np.corrcoef(np.column_stack([MA, XB]), rowvar=False)
        g = M117.gen(f'122|realistica|{name}')
        count = {a: 0 for a in ALG}
        for _ in range(N_REAL):
            Z = g.multivariate_normal(np.zeros(8), C, size=n)
            Pa, Pb = M117.three_k2(st(Z[:, :4])), M117.three_k2(st(Z[:, 4:]))
            for a in ALG:
                e, _ = M117.exact(Pa[a], Pb[a])
                count[a] += e['p_exact'] < ALPHA
        row = dict(dataset=name, replicates=N_REAL, **{a: count[a] / N_REAL for a in ALG})
        rr.append(row)
        log(f'  {name}: proportion of replicates with p < 0.05: '
            + ", ".join(f'{a} {count[a] / N_REAL:.2f}' for a in ALG))
    pd.DataFrame(rr).to_csv(os.path.join(OUT, "122_power_realistic.csv"), index=False)
    log("Reading: if only the observed concordance of the parameters is true (Gaussian, without "
        "groups), these are the proportions of times in which the partitions would agree "
        "detectably. The model is a choice.")

    # ------------------------------------------------------------------ 7 ----
    title('7. POINT 1 WITH THE MEAN OF THE WELLS AT 150 H (POST HOC VARIANT)')
    log('Concordance of the parameters with the mean: ' + ", ".join(
        f"{a} {stats.spearmanr(kmA[:, j], B[:, j]).statistic:+.3f}" for j, a in enumerate(N150)) + ".")
    rm, tm = [], {}
    XM = st(kmA)
    PM = partitions(XM)
    comparison(PM, PB, '7. kinetic parameters only, mean of the wells at 150 h', rm, tm)
    for alg in ALG:
        la, lb = PM[alg], PB[alg]
        # direction of the correspondence: do the groups with longer time coincide?
        slow_a = la == (0 if XM[la == 0, 1].mean() > XM[la == 1, 1].mean() else 1)
        slow_b = lb == (0 if XB[lb == 0, 1].mean() > XB[lb == 1, 1].mean() else 1)
        log(f"  {alg:8s} 'slow' group at 150 h and 'slow' group at 24 h: same participants in {int((slow_a == slow_b).sum())} of {n}")
    pd.DataFrame(rm).to_csv(os.path.join(OUT, "122_mean_of_wells.csv"), index=False)
    log("This variant was chosen after seeing the results; its p values are not corrected.")

    title('WHAT THE COMPARISON DOES NOT DISTINGUISH')
    log("- Agreement at chance level is expected both if there are no groups (two cuts through the "
        "middle of a continuous cloud are arbitrary) and if the two versions do not order the same "
        "participants in the same way; and it is expected also merely because of the weak "
        "concordance of the parameters (point 6). The result does not distinguish these cases.")
    log("- The power of point 2 holds for a score common to the two versions, which is not there "
        "here: for this comparison point 6 holds.")

    title('WHAT THIS SCRIPT DOES NOT DO')
    log('- It does not separate the version of the assay from the time between the two samples '
        '(about one year), from the run date and from storage.')
    log('- The partitions of points 1 and 3 are estimated on these participants only: they are not '
        'those of the manuscript.')
    log("- It does not estimate a term of comparison 'same version at one year': the participants "
        "with the 24 h assay repeated at one year are 27 (106).")
    log("- No p takes account of the uncertainty with which the partitions are estimated; the p "
        "values of point 4 and of point 5 are not corrected.")
    with open(os.path.join(OUT, "122_partitions_one_year.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
