"""
106 - Concordance between the two assay protocols in the same participants.

Two reviewers ask whether the kinetic measures common to the two protocols (24 h,
Project 237; 150 h, Project 155) rank in the same way the people tested with
both; a third proposes pooling the protocols, citing the concordance of the
outcome reported by Coughlin et al. (Mov Disord 2026, 10.1002/mds.70509).
That concordance concerns the categorical outcome: the article states among its
limitations that it did not investigate changes in the amplification parameters
over time. Here both the outcome and the kinetic parameters are examined.

What is computed, all from the raw assay file:

  1. Categorical outcome. Same visit, 150 h against 24 h, for all cohorts
     and for PD only; and a reconstruction of Coughlin's analysis at different
     visits, under two stated rules, PARTIAL because the 35 h protocol is not
     in this file.
  2. Kinetic parameters. Spearman correlation between the 24 h value and the
     150 h value of Fmax, TTT, AUC and slope, in the PD positive in both
     protocols at the baseline visit. The first three parameters have the same
     name in the two protocols; the slope does not (SLOPEMax at 24 h, SLOPE at
     150 h) and the equivalence of the two definitions is not verified. Sets:
     all; with a Type1 pattern at 24 h; sporadic Type1; the participants shared
     by the two analysis cohorts.
  3. Benchmark. The same parameter, in the same 24 h protocol, measured at two
     different visits; and in two different runs of the same visit. For the
     150 h protocol the benchmark does not exist (the script counts the pairs
     and says so).
  4. Estimate of the attenuation ceiling from the reliability of the triplicates.
  5. Exploratory, chosen post hoc: all pairs of parameters between protocols,
     the correlations within protocol, the slope at well level, the values
     centred on the run date, carriers against sporadic.
  6. Kinetic parameters between protocols at DIFFERENT VISITS (added for the
     journal review). The members of the 150 h cohort also have a 24 h result,
     almost always at a later visit: the sample on which the two protocols can
     be compared is larger than the same-visit one. For each pair of visits
     (150 h at visit a, 24 h at visit b) with at least N_MIN participants:
     Spearman with bootstrap interval for the four parameters, times between
     the visits and between the runs, run dates; the 24 h benchmark on the
     same two visits and the difference; the attenuation ceiling; the outcome
     and the type at 24 h in the members of the 150 h cohort; and a
     conclusion, derived from the numbers, on the agreement with the same
     visit. Main comparison: 150 h at baseline against 24 h at V04 in the
     sporadic members of the 150 h cohort with a Type1 at V04. Protocol, time
     between collections, storage and run date remain confounded.
     Next to each estimate of section 6: the same estimate with the last-run
     rule (only the sample run last at each visit counts), the one on complete
     triplets only where a value rests on fewer than three replicates, and
     whether or not the 24 h benchmark on the same two visits exists, with
     its matching on times. A limit of the interval is not read as a sign if
     it lies less than VICINO from zero or less than SE_MC Monte Carlo
     standard errors from it (estimated by batches). The TTT at the same
     visit and at different visits is set side by side in 6.6.

ONLY_SPORADIC does NOT restrict any set here. It only says which working
directory is being read, and the script stops if the variable and the working
directory do not agree. The working directory decides who the "condivisi dalle
due coorti" are; all the other sets come from the raw file and are the same in
the two working directories. The set that corresponds to the population of the
manuscript on sporadic participants is "sporadici Type1"; "tutti" and
"Type1 a 24 ore" also contain carriers, and the composition is printed.

Conventions, the same as in 01_build_dataset.py and 08_replication_150h.py: at 24 h
the triplet is aggregated with the mean, at 150 h with the median; a participant
is positive at a visit if they have at least one positive sample, and their
kinetic parameters are the mean over the positive samples of that visit. One
difference: 01 averages the Positive/Type1 samples only, and so do the "Type1"
sets here; "tutti" averages the positives of every type. For the categorical
outcome, instead, the sample run last within the protocol counts, by analogy
with the rule that Coughlin applies to different versions of the assay on the
same visit.

Three caveats that the script cannot resolve; they are printed in the report
next to the numbers. "Same visit" is the same visit code, not the same
aliquot: the file has no sample identifier. The two runs are not
contemporaneous (the distance in days is printed), so protocol,
storage and run date are confounded. The
bootstrap intervals resample the participants and ignore the clustering by
run date; a variant that resamples the dates is reported as a
sensitivity analysis, but with so few dates no interval is reliable.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 scripts/106_protocol_agreement.py
Output: outputs/106_agreement_kinetics.csv, 106_agreement_outcome.csv,
        106_protocol_agreement.txt
"""

import hashlib
import os
import sys
import warnings
import zlib
from _legacy import crc32_legacy                                # noqa: E402

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA                                       # noqa: E402

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
NAME_CURATED = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"
CACHE = os.path.join(OUT, "cache", "curated.parquet")

SEED = 20261112
NBOOT = 20000         # resamples of the participants
NBOOT_DATE = 5000     # resamples of the run dates (sensitivity)
N_MIN = 10            # below this sample size a correlation is not reported
N_MIN_RIC = 5         # a resample with fewer participants than this is discarded
RUN_MIN = 5           # runs with fewer samples are merged into a single category
NEARBY = 0.02         # a limit closer than this to zero is not read as a sign
# Section 6 only:
NBOOT_DATE_V = 20000  # resamples of the run dates
BATCHES_MC = 20         # batches used to estimate the Monte Carlo error of a limit
SE_MC = 2.0           # a limit within this many Monte Carlo standard errors of zero
                      # is not read as a sign, even if it lies more than VICINO away
RHO_SAME = 0.70     # |rho| from which two parameters carry largely the same information

# Figures published by Coughlin et al., Mov Disord 2026, doi 10.1002/mds.70509:
# participants with different versions of the assay at different visits. They are
# literals from an external source, reported only to set them next to the
# reconstruction.
COUGHLIN = {'concordant': 502, "n": 523, "pos": (432, 448), "neg": (70, 75)}

rep = []
rows_kin = []        # long table of the estimates on the kinetic parameters
rows_outcome = []      # long table of the categorical concordance


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def share(k, n):
    """'k/n = x%' without dividing by zero."""
    return f"{k}/{n} = {100 * k / n:.1f}%" if n else f"{k}/{n}"


def _rho(x, y):
    if len(x) < 3:
        return np.nan
    return np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1]


def _rho_rows(x, y, idx):
    """Spearman on each row of resampled indices, in one block."""
    rx = stats.rankdata(x[idx], axis=1)
    ry = stats.rankdata(y[idx], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
    out = np.full(len(idx), np.nan)
    ok = den > 0
    out[ok] = (rx * ry).sum(axis=1)[ok] / den[ok]
    return out


def _gen(label):
    # Each estimate has its own generator, derived from the seed and the label:
    # adding or removing an analysis does not shift the others.
    return np.random.default_rng([SEED, crc32_legacy(label.encode())])


def _close(n, rho, p, bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n=n, rho=float(rho), lo=float(lo), hi=float(hi), p=float(p), bs=bs,
                limit=bool(min(abs(lo), abs(hi)) < NEARBY))


def estimate(x, y, label):
    """Spearman with percentile bootstrap interval over the participants."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < N_MIN:
        return dict(n=n, rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan, bs=None,
                    limit=False)
    rho, p = stats.spearmanr(x, y)
    idx = _gen(label).integers(0, n, (NBOOT, n))
    return _close(n, rho, p, _rho_rows(x, y, idx))


def estimate_by_date(x, y, date, label, nboot=None):
    """Like stima(), but resampling the run dates instead of the participants."""
    nboot = NBOOT_DATE if nboot is None else nboot
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, date = x[ok], y[ok], np.asarray(date)[ok]
    code = pd.factorize(pd.Series(date).astype(str))[0]
    members = [np.flatnonzero(code == j) for j in range(code.max() + 1)]
    g = _gen(label)
    bs = np.full(nboot, np.nan)
    for b in range(nboot):
        i = np.concatenate([members[j] for j in g.integers(0, len(members), len(members))])
        bs[b] = _rho(x[i], y[i])
    rho = stats.spearmanr(x, y)[0]
    s = _close(len(x), rho, np.nan, bs)     # the asymptotic p ignores the dates
    s["n_date"] = len(members)
    return s


def row(s, what="rho"):
    if not np.isfinite(s["rho"]):
        return f'not estimated (n < {N_MIN})'
    t = f"{what} = {s['rho']:+.3f}  [{s['lo']:+.3f}; {s['hi']:+.3f}]"
    if np.isfinite(s["n"]):
        t = f"n = {s['n']:>3}  " + t
    if np.isfinite(s.get("p", np.nan)):
        t += f"  p = {s['p']:.3g}"
    if s['limit']:
        t += '  (one limit is close to zero: it is not read as a sign)'
    return t


def record(block, dataset, parameter, s, variant="", quantity="rho",
             coincides_having="", subj=None):
    """sogg: the participants on which the row rests; if given, the sporadic ones are counted."""
    rows_kin.append(dict(
        block=block, dataset=dataset, parameter=parameter, variant=variant,
        quantity=quantity, n=s.get("n", np.nan),
        n_sporadic=(np.nan if subj is None else len(set(subj) & sporadic)),
        estimate=s["rho"],
        ci_inf=s["lo"], ci_upp=s["hi"], p=s.get("p", np.nan),
        limit_nearby_a_zero=s.get('limit', False), coincides_having=coincides_having))


def difference_paired(xa, xb, ya, yb, label):
    """rho(xa, xb) - rho(ya, yb) resampling the union of the participants.

    The four vectors are aligned on the union of the participants of the two
    sets, with NaN where the participant does not belong to the set: whoever is
    in both is drawn only once and weighs on both correlations.
    """
    ma, mb = np.isfinite(xa) & np.isfinite(xb), np.isfinite(ya) & np.isfinite(yb)
    count = dict(n_a=int(ma.sum()), n_b=int(mb.sum()), n_common=int((ma & mb).sum()),
                 n_union=int((ma | mb).sum()))
    if ma.sum() < N_MIN or mb.sum() < N_MIN:
        return dict(n=np.nan, rho=np.nan, lo=np.nan, hi=np.nan, bs=None, limit=False,
                    **count)
    point = _rho(xa[ma], xb[ma]) - _rho(ya[mb], yb[mb])
    g = _gen(label)
    nu = len(xa)
    bs = np.full(NBOOT, np.nan)
    for b in range(NBOOT):
        i = g.integers(0, nu, nu)
        a, c = ma[i], mb[i]
        if a.sum() >= N_MIN_RIC and c.sum() >= N_MIN_RIC:
            bs[b] = _rho(xa[i][a], xb[i][a]) - _rho(ya[i][c], yb[i][c])
    s = _close(np.nan, point, np.nan, bs)
    s['discarded'] = float(np.isnan(bs).mean())
    s.update(count)
    return s


def icc_fr(mu, w, k=3):
    """ICC(1,1) and ICC(1,k) from the means and within sums of squares of each triplet."""
    n = mu.shape[-1]
    msb = k * ((mu - mu.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1) / (n - 1)
    msw = w.sum(axis=-1) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw), (msb - msw) / msb


# ------------------------------------------------------------------ data
d = pd.read_csv(SAA, low_memory=False)
assert not d.duplicated().any(), 'duplicate rows in the SAA file'
d["PATNO"] = d["PATNO"].astype("int64")
d["RUNDATE"] = pd.to_datetime(d["RUNDATE"])
assert d.RUNDATE.notna().all(), "missing run dates: the order of the runs is not defined"
assert d.COHORT.notna().all(), 'cohort missing in some row of the SAA file'
assert set(d.PROJECTID.unique()) == {155, 237}, 'unexpected projects in the SAA file'
assert d.groupby("PATNO").COHORT.nunique().max() == 1, 'subject with two cohorts'
d['row'] = np.arange(len(d))


def ordering_visit(e):
    if e == "BL":
        return 0.0
    if e.startswith("V") and e[1:].isdigit():
        return float(int(e[1:]))
    return np.nan                 # ST, U01: chronological position not defined


d["vord"] = d.CLINICAL_EVENT.map(ordering_visit)

# parameter name -> (stem at 24 h, stem at 150 h)
PAR = {"Fmax": ("Fmax_24h", "Fmax_150h"),
       "TTT": ("TTT_24h", "TTT_150h"),
       "AUC": ("AUC_24h", "AUC_150h"),
       'Slope': ("SLOPEMax_24h", "SLOPE_150h")}
P24 = list(PAR)                          # 24 h parameters
P150 = list(PAR) + ["T50"]               # at 150 h there is also T50


def replicates(root, rows=None):
    v = d if rows is None else rows
    return v[[f'{root}_Rep{i}' for i in (1, 2, 3)]].apply(
        pd.to_numeric, errors="coerce")


for name, (a, b) in PAR.items():
    va, vb = replicates(a), replicates(b)
    d[f'm24_{name}'] = va.mean(axis=1)        # 24 h: mean, as in 01
    d[f'md150_{name}'] = vb.median(axis=1)    # 150 h: median, as in 08
    d[f'm150_{name}'] = vb.mean(axis=1)       # sensitivity
    d[f'nr24_{name}'] = va.notna().sum(axis=1)     # replicates available (section 6)
    d[f'nr150_{name}'] = vb.notna().sum(axis=1)
d["md150_T50"] = replicates("T50_150h").median(axis=1)

key = ["PATNO", "CLINICAL_EVENT", "PROJECTID"]
d = d.sort_values(key + ["RUNDATE", 'row']).reset_index(drop=True)

title('0. RAW FILE AND WORKING DIRECTORY')
with open(SAA, "rb") as f:
    fingerprint = hashlib.sha256(f.read()).hexdigest()
log(f'file: {os.path.basename(SAA)}, {os.path.getsize(SAA)} bytes, sha256 {fingerprint}')
log(f'rows (assay results) {len(d)}, subjects {d.PATNO.nunique()}')
for pr, name in ((155, '150 h'), (237, '24 h')):
    s = d[d.PROJECTID == pr]
    log(f'  Project {pr} ({name}): {len(s)} rows, {s.PATNO.nunique()} subjects, runs from '
        f'{s.RUNDATE.min().date()} to {s.RUNDATE.max().date()}')

# curated data cut: genetic subgroup at baseline
if os.path.exists(CACHE):
    cur = pd.read_parquet(CACHE, columns=["PATNO", "EVENT_ID", "subgroup"])
else:
    cand = [os.path.join(BASE, NAME_CURATED), os.path.join(DATA, NAME_CURATED)]
    located = [c for c in cand if os.path.exists(c)]
    assert located, f"curated not found either in the cache or in: {cand}"
    cur = pd.read_excel(located[0], sheet_name="20260511",
                        usecols=["PATNO", "EVENT_ID", "subgroup"])
cur["PATNO"] = cur["PATNO"].astype("int64")
sub = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO").set_index("PATNO").subgroup
sporadic = set(sub[sub == "Sporadic PD"].index)
assert sporadic & set(d.PATNO), 'no sporadic participant of the curated data cut appears in the SAA file'


def composition(patno):
    """Genetic subgroups, from the curated data cut, of a list of participants."""
    return (sub.reindex(list(patno)).fillna('absent from the curated data cut')
            .value_counts().to_dict())


# the two analysis cohorts, read from the outputs of the current working directory
c24 = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
c150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
if "complete_primary" in c24.columns:
    c24 = c24[c24.complete_primary]
if "complete" in c150.columns:
    c150 = c150[c150.complete]
c24["PATNO"] = c24["PATNO"].astype("int64")
c150["PATNO"] = c150["PATNO"].astype("int64")
p24, p150 = set(c24.PATNO), set(c150.PATNO)
shared = p24 & p150
assert (p24 | p150) <= set(sub.index), 'subjects of the cohorts missing from the baseline curated data cut'
only24, only150 = not (p24 - sporadic), not (p150 - sporadic)
assert only24 == only150, ("the two cohorts of the working directory are not of the same kind: one "
                           "is sporadic only, the other is not")
ROOT = 'sporadic only' if only24 else 'full cohort'
assert ONLY == only24, (f"ONLY_SPORADIC={('1' if ONLY else '0')} but the working directory is "
                        f"'{ROOT}': variable and working directory do not agree")
log(f"working directory: {os.path.basename(os.path.normpath(BASE))} ({ROOT}); ONLY_SPORADIC = "
    f"{('1' if ONLY else '0')}")
log(f'analysis cohorts: 24 h n = {len(p24)}, 150 h n = {len(p150)}, shared {len(shared)}, distinct '
    f'{len(p24 | p150)}')
log('ONLY_SPORADIC does not restrict any set of this script: the working directory decides')
log("only who the 'condivisi dalle due coorti' are. The other sets come from the raw")
log("file. The set of the sporadic population is 'sporadici Type1'.")

# ------------------------------------------------------------------ 1. OUTCOME
title('1. CATEGORICAL OUTCOME')
log("Unit: subject x visit x protocol. Outcome of the unit: that of the sample")
log("run last within the protocol. It is an extension, by analogy, of the rule of")
log('Coughlin, which concerns different versions of the assay on the same visit.')
log("CAVEAT: 'same visit' is the same visit code, not the same")
log('aliquot; the file has no sample identifier.')
count = d.groupby(key, sort=False).agg(
    n_samp=('row', "size"), n_outcomes=("SAA_Status", "nunique"),
    each_pos=("SAA_Status", lambda s: bool((s == "Positive").any()))).reset_index()
last_r = d.drop_duplicates(key, keep="last")        # the ROW run last
vis = (last_r[key + ["COHORT", "vord", "SAA_Status", "SAA_Type", "RUNDATE"]]
       .rename(columns={"SAA_Status": 'outcome', "SAA_Type": 'kind', "RUNDATE": 'run'})
       .merge(count, on=key, validate="one_to_one"))
last = set(last_r.row)

mx = d.groupby(key, sort=False).RUNDATE.transform("max")
ties = d[d.RUNDATE == mx].groupby(key, sort=False).agg(
    k=('row', "size"), n_se=("SAA_Status", "nunique"),
    n_kinds=("SAA_Type", lambda s: s.nunique(dropna=False)))
n_ties = int((ties["k"] > 1).sum())
n_ties_disc = int(((ties["k"] > 1) & (ties['n_se'] > 1)).sum())
assert n_ties_disc == 0, f'{n_ties_disc} units with discordant outcomes on the same run date'
# Between two rows with the same last run date the order of the file counts: the
# outcome is the same (check just above), but the TYPE of the row 'run last'
# (rule B of section 1, counts of 6.4) would depend on the order of the rows.
ties_kind = ties[(ties["k"] > 1) & (ties['n_kinds'] > 1)].reset_index()
assert not len(ties_kind), (
    f'{len(ties_kind)} units with two rows on the last run date and different TYPE (of which '
    f'{int(ties_kind.PATNO.isin(p150).sum())} of members of the 150 h cohort): the type of the '
    f'sample run last would depend on the order of the file')
disc = d.merge(vis.loc[vis.n_outcomes > 1, key], on=key).groupby(key).SAA_Status.agg(set)
log(f'\nunits: {len(vis)}; with more than one row: {(vis.n_samp > 1).sum()}')
log(f"  with different outcomes between runs: {len(disc)}, of which with an inconclusive "
    f"{sum(('Inconclusive' in s for s in disc))} and with a positive and a negative "
    f"{sum(({'Positive', 'Negative'} <= s for s in disc))}")
log(f'  with two rows on the last run date: {n_ties} (same outcomes in all; among these the order of '
    f'the file counts)')

ORD = ["Positive", "Negative", "Inconclusive"]


def table(s, col150='outcome_150', col24='outcome_24'):
    return pd.crosstab(s[col150], s[col24]).reindex(index=ORD, columns=ORD, fill_value=0)


def concordance(tab, block, dataset, n_units, n_subj):
    """tab: rows = 150 h, columns = 24 h."""
    pp, nn = int(tab.loc["Positive", "Positive"]), int(tab.loc["Negative", "Negative"])
    pn, npos = int(tab.loc["Positive", "Negative"]), int(tab.loc["Negative", "Positive"])
    n, k, tot = pp + nn + pn + npos, pp + nn, int(tab.values.sum())
    lo, hi = wilson(k, n)
    log(f'\n[{block} — {dataset}: {n_units} units, {n_subj} subjects]  rows = 150 h, columns = 24 h')
    log(tab.to_string())
    log(f'  concordant, excluding the inconclusives: {share(k, n)} [Wilson 95% {100 * lo:.1f}; '
        f'{100 * hi:.1f}]; units with an inconclusive: {tot - n}')
    rows_outcome.append(dict(block=block, dataset=dataset, n_units=n_units,
                            n_subjects=n_subj, share='everyone',
                            concordant=k, n=n, fraction=k / n if n else np.nan,
                            ci_inf=lo, ci_upp=hi))
    for name, a, b in (('positive at 150 h', pp, pp + pn),
                       ('negative at 150 h', nn, nn + npos)):
        l2, h2 = wilson(a, b)
        log(f'    {name} that remain so at 24 h: {share(a, b)} [{100 * l2:.1f}; {100 * h2:.1f}]')
        rows_outcome.append(dict(block=block, dataset=dataset, n_units=n_units,
                                n_subjects=n_subj, share=name,
                                concordant=a, n=b, fraction=a / b if b else np.nan,
                                ci_inf=l2, ci_upp=h2))


vis['outcome_each'] = np.where(vis.each_pos, "Positive", vis.outcome)   # rule of 01 and 08
v24 = vis[vis.PROJECTID == 237].set_index(["PATNO", "CLINICAL_EVENT"])
v150 = vis[vis.PROJECTID == 155].set_index(["PATNO", "CLINICAL_EVENT"])
selfsame = v150.join(v24, how="inner", lsuffix="_150", rsuffix="_24").reset_index()
assert (selfsame.COHORT_150 == selfsame.COHORT_24).all(), 'different cohort in the two protocols'
log(f'\nSubject x visit units with both protocols: {len(selfsame)} ({selfsame.PATNO.nunique()} '
    f'subjects); visits: {selfsame.CLINICAL_EVENT.value_counts().to_dict()}')
log("The Wilson intervals treat the units as independent; some subjects")
log('contribute two.')
for name, s in (('all cohorts', selfsame), ("PD", selfsame[selfsame.COHORT_150 == "PD"])):
    concordance(table(s), 'same visit', name, len(s), s.PATNO.nunique())
    pos24 = s[s.outcome_24 == "Positive"]
    log(f"    type at 24 h of the positives at 24 h: "
        f"{pos24.kind_24.fillna('n.d.').value_counts().to_dict()}")
    equal = table(s).equals(table(s, 'outcome_each_150', 'outcome_each_24'))
    log(f"    with the rule 'at least one positive sample' (that of 01 and 08) the table is "
        f"{('identical' if equal else 'DIFFERING')}")

log('\nPD tested at 150 h at baseline, according to whether the baseline collection was')
log('also run at 24 h or not:')
b150 = v150.reset_index()
b150 = b150[(b150.COHORT == "PD") & (b150.CLINICAL_EVENT == "BL")]
withdr = set(selfsame[(selfsame.COHORT_150 == "PD") & (selfsame.CLINICAL_EVENT == "BL")].PATNO)
u24 = v24.reset_index()
elsewhere = set(u24[u24.CLINICAL_EVENT != "BL"].PATNO)     # 24 h at another visit
doubtful = {}
for name, s in (('with the baseline also run at 24 h', b150[b150.PATNO.isin(withdr)]),
                ('without', b150[~b150.PATNO.isin(withdr)])):
    c = s.outcome.value_counts()
    doubtful[name] = (c.get("Negative", 0) + c.get("Inconclusive", 0)) / len(s)
    log(f"  {name}: n = {len(s)}; at 150 h positives {c.get('Positive', 0)}, negatives "
        f"{share(int(c.get('Negative', 0)), len(s))}, inconclusives {c.get('Inconclusive', 0)}; "
        f"sporadic {share(int(s.PATNO.isin(sporadic).sum()), len(s))}; with a 24 h at another visit "
        f"{int(s.PATNO.isin(elsewhere).sum())}")
log('  The denominators include the inconclusives.')
q_having, q_without = doubtful['with the baseline also run at 24 h'], doubtful['without']
log(f'  Negative or inconclusive at 150 h: {100 * q_having:.1f}% among the former, '
    f'{100 * q_without:.1f}% among the latter.')
if q_having > q_without:
    log("  The concordance at the same visit is therefore measured on a set enriched")
    log('  in non-positive outcomes: it does not estimate that of PD in general and must be read next')
    log('  to that at different visits, below.')
else:
    log('  The two groups do not differ in the direction expected from a selection of the doubtful cases.')

log("\nReconstruction of the analysis of Coughlin at different visits. For each visit")
log("the row run last counts, of whichever protocol; the order is that of the visit")
log("codes (the article uses the actual dates, which are not public).")
log("PARTIAL: the 35 h protocol is not in this file and the download is more")
log("recent. A denominator equal or close to the published one is not a replication.")
vv = (vis[vis.vord.notna()].sort_values(["PATNO", "CLINICAL_EVENT", 'run'])
      .drop_duplicates(["PATNO", "CLINICAL_EVENT"], keep="last").copy())
# category according to the article's definition: positive at 150 h, Type1 at 24 h
vv["cat"] = np.select(
    [vv.outcome == "Negative",
     (vv.outcome == "Positive") & ((vv.PROJECTID == 155) | (vv.kind == "Type1"))],
    ["Negative", "Positive"], default='other')
RULES = {
    "A": 'outcome in the file (positive of any type); inconclusive visits ignored',
    "B": 'positive as the article defines it (at 150 h positive, at 24 h only Type1). The rest is '
         'inferred from the text, not confirmed: inconclusive, Type2 and undetermined count as a '
         'change; those who start from positive or negative are included',
}
for rule, descr in RULES.items():
    log(f'\n  Rule {rule} — {descr}')
    for name, s in (('all cohorts', vv), ("PD", vv[vv.COHORT == "PD"])):
        se = []
        for _, t in s.groupby("PATNO"):
            t = t.sort_values("vord")
            if rule == "A":
                t = t[t.outcome != "Inconclusive"]
                if t.PROJECTID.nunique() < 2:
                    continue
                se.append((t.outcome.iloc[0], bool((t.outcome == t.outcome.iloc[0]).all())))
            else:
                if t.PROJECTID.nunique() < 2 or t.cat.iloc[0] == 'other':
                    continue
                se.append((t.cat.iloc[0], bool((t.cat == t.cat.iloc[0]).all())))
        se = pd.DataFrame(se, columns=['initial', 'constant'])
        k, n = int(se.constant.sum()), len(se)
        lo, hi = wilson(k, n)
        log(f'    {name}: constant {share(k, n)} [{100 * lo:.1f}; {100 * hi:.1f}]')
        rows_outcome.append(dict(block=f'different visits, partial reconstruction, rule {rule}',
                                dataset=name, n_units=np.nan, n_subjects=n,
                                share='everyone', concordant=k, n=n,
                                fraction=k / n if n else np.nan, ci_inf=lo, ci_upp=hi))
        for ini in ("Positive", "Negative"):
            e2 = se[se.initial == ini]
            k2, n2 = int(e2.constant.sum()), len(e2)
            l2, h2 = wilson(k2, n2)
            log(f'      initially {ini}: {share(k2, n2)} [{100 * l2:.1f}; {100 * h2:.1f}]')
            rows_outcome.append(dict(
                block=f'different visits, partial reconstruction, rule {rule}',
                dataset=name, n_units=np.nan, n_subjects=n2,
                share=f'initially {ini}', concordant=k2, n=n2,
                fraction=k2 / n2 if n2 else np.nan, ci_inf=l2, ci_upp=h2))
lo, hi = wilson(COUGHLIN['concordant'], COUGHLIN["n"])
log(f"\n  published (150 h against 24/35 h, all cohorts): "
    f"{share(COUGHLIN['concordant'], COUGHLIN['n'])}; positives "
    f"{COUGHLIN['pos'][0]}/{COUGHLIN['pos'][1]}, negatives {COUGHLIN['neg'][0]}/{COUGHLIN['neg'][1]}")
log(f"  (the article does not give the interval; the Wilson interval computed here is "
    f"[{100 * lo:.1f}; {100 * hi:.1f}])")
log("  The two rules read the same data: the distance between their results is one of")
log('  definition. Neither of the two reproduces all the published cells.')

# ------------------------------------------------------------------ 2. KINETIC PARAMETERS
title('2. KINETIC PARAMETERS BETWEEN PROTOCOLS, SAME VISIT')
log('CAVEATS. Same visit = same visit code, not same aliquot.')
log('The two runs are not contemporaneous (the days that separate them are reported')
log('below): protocol, storage and run date are confounded and these data')
log('do not separate them. The intervals resample the subjects')
log('and ignore the clustering by run date. The p values are not corrected for')
log("multiplicity. The slope is SLOPEMax at 24 h and SLOPE at 150 h: two")
log("definitions whose equivalence is not verified.")
C24 = [f"m24_{n}" for n in P24]
C150 = [f"md150_{n}" for n in P150] + [f"m150_{n}" for n in PAR]


def kinds(s):
    return "+".join(sorted(set(s.fillna("n.d.").astype(str))))


def rows_kinetics(project, cohort="PD", only_type1=False, only_last=False):
    """The samples that enter a participant's value at a visit."""
    s = d[(d.PROJECTID == project) & (d.SAA_Status == "Positive")]
    if cohort:
        s = s[s.COHORT == cohort]
    if only_type1:
        assert project == 237, 'the type exists only in the 24 h protocol'
        s = s[s.SAA_Type == "Type1"]
    if only_last:
        s = s[s.row.isin(last)]
    return s


def kinetics(project, cohort="PD", only_type1=False, only_last=False, columns=None):
    """One row per participant x visit: mean over the POSITIVE samples of the visit."""
    s = rows_kinetics(project, cohort, only_type1, only_last)
    col = columns or (C24 if project == 237 else C150)
    return s.groupby(["PATNO", "CLINICAL_EVENT"]).agg(
        **{c: (c, "mean") for c in col}, n_pos=('row', "size"),
        run=("RUNDATE", "max"), vord=("vord", "first"), kinds=("SAA_Type", kinds))


def pairs(k24, k150, visit="BL"):
    m = k150.join(k24, how="inner", lsuffix="_150", rsuffix="_24").reset_index()
    return m[m.CLINICAL_EVENT == visit] if visit else m


# type of all the positive 24 h samples of the visit, including those that
# a Type1 set does not use
kinds_vis = (d[(d.PROJECTID == 237) & (d.SAA_Status == "Positive")]
            .groupby(["PATNO", "CLINICAL_EVENT"]).SAA_Type.agg(kinds))


def kinds_of(patno, visit):
    return kinds_vis.reindex(pd.MultiIndex.from_arrays([patno, visit])).values


k150 = kinetics(155)
everyone = pairs(kinetics(237), k150)
type1 = pairs(kinetics(237, only_type1=True), k150)
DATASETS = {
    'everyone': everyone,
    'Type1 at 24 h': type1,
    'sporadic Type1': type1[type1.PATNO.isin(sporadic)],
    'shared by the two cohorts': type1[type1.PATNO.isin(shared)],
}
# The two uniqueness checks are structural (one row per participant x visit, a
# single visit): they stay as a reminder, not as a check on the data.
assert everyone.PATNO.is_unique and type1.PATNO.is_unique
miss = shared - set(type1.PATNO)
assert not miss, f'{len(miss)} shared subjects without a Type1/positive pair at baseline'
pp_ = ties[ties["k"] > 1].reset_index()
in_ties = set(pp_[(pp_.CLINICAL_EVENT == "BL")].PATNO) & set(everyone.PATNO)
assert not in_ties, f"{len(in_ties)} subjects of 'tutti' with two rows on the last date"

# Identity with the values that the two cohorts actually used. It is not enough
# that the participants are equal in number: they must carry the same numbers.
# The comparison covers every paired participant who is in one of the two
# cohorts, not only the shared ones.
COMPARISONS = [(type1, c24, '24 h', [("m24_Fmax", "saa_Fmax"), ("m24_TTT", "saa_TTT"),
                                     ("m24_AUC", "saa_AUC"),
                                     ('m24_Slope', "saa_SLOPEMax")]),
             (everyone, c150, '150 h', [("md150_Fmax", "k_Fmax"), ("md150_TTT", "k_TTT"),
                                       ("md150_AUC", "k_AUC"), ("md150_T50", "k_T50"),
                                       ('md150_Slope', "k_SLOPE")])]
IDENT = {}      # identity check -> participants on which it was performed
log()
for mine, cohort, name, columns in COMPARISONS:
    a = mine.set_index("PATNO")
    com = sorted(set(a.index) & set(cohort.PATNO))
    IDENT[name] = len(com)
    if not com:
        log(f'Identity check ({name}) could not be run: no paired subject is in the cohort.')
        continue
    b = cohort.set_index("PATNO").loc[com]
    for x, y in columns:
        assert np.allclose(a.loc[com, x].values, b[y].values, rtol=1e-9, atol=0), \
            f'{x} different from {y} in the paired subjects of the {name} cohort'
    log(f'Identity check ({name}): in {len(com)} paired subjects who are in the cohort, the '
        f'{len(columns)} kinetic values coincide with those saved.')

gap = (everyone.run_24 - everyone.run_150).dt.days
log(f'\nPD positive in both protocols at baseline: {len(everyone)}')
log(f'  distinct run dates: {everyone.run_150.nunique()} at 150 h ({everyone.run_150.min().date()} - '
    f'{everyone.run_150.max().date()}), {everyone.run_24.nunique()} at 24 h '
    f'({everyone.run_24.min().date()} - {everyone.run_24.max().date()})')
more = everyone.run_24.value_counts()
log(f'  the four most numerous run dates at 24 h gather {int(more.head(4).sum())} of the '
    f'{len(everyone)} subjects')
log(f'  days between the two runs: minimum {gap.min()}, median {gap.median():.1f}, maximum '
    f'{gap.max()}; pairs less than one year apart: {int((gap < 365).sum())}')
log(f'  with more than one positive sample: {(everyone.n_pos_24 > 1).sum()} at 24 h, '
    f'{(everyone.n_pos_150 > 1).sum()} at 150 h')

BETWEEN = {}        # (set, parameter) -> estimate, for the later comparisons
ALIAS = {}      # set -> identical set already estimated
log('\nSpearman between 24 h (mean of the triplet) and 150 h (median), same parameter')
for name, m in DATASETS.items():
    log(f'\n  {name} (n = {len(m)})')
    log(f'    subgroups: {composition(m.PATNO)}')
    log(f'    type of all the positive samples at 24 h of the visit: '
        f'{pd.Series(kinds_of(m.PATNO, m.CLINICAL_EVENT)).value_counts().to_dict()}')
    if name == 'everyone':
        log("    (here the value at 24 h is the mean of the positives of every type; 01 averages the")
        log('    Type1 only, as do the Type1 sets below)')
    twin = next((g for g in BETWEEN if g[1] == "Fmax" and g[0] != name
                    and g[0] not in ALIAS
                    and set(DATASETS[g[0]].PATNO) == set(m.PATNO)
                    and np.array_equal(DATASETS[g[0]].sort_values("PATNO")[C24 + C150].values,
                                       m.sort_values("PATNO")[C24 + C150].values,
                                       equal_nan=True)), None)
    if twin:
        ALIAS[name] = twin[0]
        log(f"    coincides, subject by subject, with '{twin[0]}': the estimates are the same and "
            f"are not repeated.")
    for par in PAR:
        if name in ALIAS:
            s = BETWEEN[(ALIAS[name], par)]
        else:
            s = estimate(m[f"m24_{par}"], m[f"md150_{par}"], f'between|{name}|{par}')
            log(f'    {par:<9} {row(s)}')
        BETWEEN[(name, par)] = s
        record('between protocols, same visit', name, par, s, 'mean/median',
                 coincides_having=ALIAS.get(name, ""), subj=m.PATNO)

SENS = ('everyone', 'sporadic Type1')
MEAN2 = {}     # (set, parameter) -> estimate with the mean at 150 h, for section 6
ult_t = pairs(kinetics(237, only_last=True), kinetics(155, only_last=True))
ult_1 = pairs(kinetics(237, only_type1=True, only_last=True),
               kinetics(155, only_last=True))
LATEST = {'everyone': ult_t, 'sporadic Type1': ult_1[ult_1.PATNO.isin(sporadic)]}
for name in SENS:
    m = DATASETS[name]
    log(f"\nSensitivity on '{name}' (n = {len(m)})")
    log('  (a) mean at 150 h in place of the median')
    for par in PAR:
        s = estimate(m[f"m24_{par}"], m[f"m150_{par}"], f'between|mean-mean|{name}|{par}')
        MEAN2[(name, par)] = s
        record('between protocols, same visit', name, par, s, 'mean/mean',
                 subj=m.PATNO)
        log(f'    {par:<9} {row(s)}')
    u = LATEST[name]
    same = set(u.PATNO) == set(m.PATNO)
    col_b = [f"m24_{p}" for p in PAR] + [f"md150_{p}" for p in PAR]
    equals = same and np.array_equal(u.sort_values("PATNO")[col_b].values,
                                       m.sort_values("PATNO")[col_b].values,
                                       equal_nan=True)
    log(f"  (b) only the row run last, which must be positive (n = {len(u)}; "
        f"{('same subjects' if same else 'DIFFERENT SUBJECTS')})")
    if equals:
        log("    coincides with the main estimate: no subject of the set has more")
        log('    than one positive sample, so the values are the same.')
    for par in PAR:
        if equals:
            s = BETWEEN[(name, par)]
        else:
            s = estimate(u[f"m24_{par}"], u[f"md150_{par}"], f'between|last|{name}|{par}')
            log(f'    {par:<9} {row(s)}')
        record('between protocols, same visit', name, par, s,
                 'last run' + ("" if same else ', different subjects'),
                 coincides_having='main estimate' if equals else "", subj=u.PATNO)
    log('  (c) resampling the run dates instead of the subjects. The two factors are')
    log("      crossed and neither of the two intervals considers both; with so")
    log("      few dates they are indicative. Those who have positive samples on two dates are")
    log("      assigned to the more recent.")
    for side, col in (('dates at 24 h', 'run_24'), ('dates at 150 h', 'run_150')):
        for par in PAR:
            s = estimate_by_date(m[f"m24_{par}"].values, m[f"md150_{par}"].values,
                               m[col].values, f'date|{side}|{name}|{par}')
            record('between protocols, same visit', name, par, s,
                     f"resampling {s['n_date']} {side}", subj=m.PATNO)
            log(f"    {side:<15} ({s['n_date']:>2})  {par:<9} {row(s)}")

# ------------------------------------------------------------------ 3. COMPARATOR
title('3. BENCHMARK: THE 24 H PROTOCOL REPEATED')
every24 = (vis[(vis.PROJECTID == 237) & vis.vord.notna()]
           .sort_values(["PATNO", "vord"]).groupby("PATNO").vord.apply(list))


def two_visits(k):
    """The first two orderable visits at which the participant has a positive
    sample (or Type1, depending on the table received). Visits tested but not
    positive in between are skipped; the number of pairs in which this happens
    is printed."""
    k = k.reset_index()
    k = k[k.vord.notna()].sort_values(["PATNO", "vord"]).reset_index(drop=True)
    k = k[k.groupby("PATNO").PATNO.transform("size") >= 2]
    a = k.drop_duplicates("PATNO", keep="first")
    b = k.drop(a.index).drop_duplicates("PATNO", keep="first")
    return (a.set_index("PATNO").join(b.set_index("PATNO"), lsuffix="_a", rsuffix="_b")
            .reset_index())


b_each = two_visits(kinetics(237))
b_t1 = two_visits(kinetics(237, only_type1=True))
COMPARISON = {
    'everyone': ('positive of any type', b_each),
    'Type1 at 24 h': ('Type1 at both visits', b_t1),
    'sporadic Type1': ('sporadic, Type1 at both visits',
                        b_t1[b_t1.PATNO.isin(sporadic)]),
}
log('PD. Same parameter, 24 h protocol, the first two visits with a positive')
log('sample (Type1, in the Type1 sets) of the same subject. They are different')
log('collections: the benchmark also contains the biological change between the visits.')
only1_between = int((kinds_of(everyone.PATNO, everyone.CLINICAL_EVENT) == "Type1").sum())
only1_par = int(((kinds_of(b_each.PATNO, b_each.CLINICAL_EVENT_a) == "Type1")
                 & (kinds_of(b_each.PATNO, b_each.CLINICAL_EVENT_b) == "Type1")).sum())
log(f"In 'tutti', {share(only1_between, len(everyone))} of the subjects between protocols and "
    f"{share(only1_par, len(b_each))} of the pairs of the benchmark have Type1 samples only.")
log('Where the two shares differ the two sets are not comparable by type, and')
log("the like-for-like comparison is that of the Type1 rows.")
WITHIN = {}
for dset, (name, m) in COMPARISON.items():
    log(f'\n  {dset} — {name} (n = {len(m)})')
    if len(m):
        dg = (m.run_b - m.run_a).dt.days
        skip = sum([r.vord_a, r.vord_b] != every24[r.PATNO][:2] for r in m.itertuples())
        log(f'    subgroups: {composition(m.PATNO)}')
        ta_ = pd.Series(kinds_of(m.PATNO, m.CLINICAL_EVENT_a))
        tb_ = pd.Series(kinds_of(m.PATNO, m.CLINICAL_EVENT_b))
        log(f"    type of all the positive samples at the two visits: "
            f"{(ta_ + ' / ' + tb_).value_counts().to_dict()}")
        log(f"    pairs of visits: "
            f"{(m.CLINICAL_EVENT_a + '-' + m.CLINICAL_EVENT_b).value_counts().to_dict()}")
        log(f'    pairs that are not the first two TESTED visits of the subject: {skip}')
        log(f"    days between the two runs, signed: from {dg.min()} to {dg.max()}, median "
            f"{dg.median():.0f}; pairs in which the later visit was run FIRST: {int((dg < 0).sum())}")
        log(f'    in the 24 h cohort: {m.PATNO.isin(p24).sum()}')
    for par in PAR:
        s = estimate(m[f"m24_{par}_a"], m[f"m24_{par}_b"], f'within|{name}|{par}')
        WITHIN[(dset, par)] = s
        record('within the 24 h, two visits', dset, par, s, name, subj=m.PATNO)
        log(f'    {par:<9} {row(s)}')

log('\nDifference: rho within the 24 h (two visits) minus rho between protocols.')
log("Paired bootstrap on the union of the subjects of the two sets. The difference")
log("does NOT isolate the effect of the protocol: the benchmark compares two means at 24 h,")
log("the other side contains a median at 150 h. The repeatabilities of the two protocols")
log("are in section 4: where that of the 150 h is lower, part of the difference is")
log('expected from that alone. In the opposite direction, the benchmark contains the')
log('biological change between two visits, which lowers it.')
for dset, (name, bm) in COMPARISON.items():
    cm = DATASETS[dset]
    com = len(set(bm.PATNO) & set(cm.PATNO))
    log(f'  {dset}: subjects in common between the two sets {com} (out of {len(bm)} of the benchmark '
        f'and {len(cm)} between protocols)')
    an = sorted(set(bm.PATNO) | set(cm.PATNO))
    bi, ci = bm.set_index("PATNO").reindex(an), cm.set_index("PATNO").reindex(an)
    for par in PAR:
        s = difference_paired(bi[f"m24_{par}_a"].values, bi[f"m24_{par}_b"].values,
                                ci[f"m24_{par}"].values, ci[f"md150_{par}"].values,
                                f'difference|{dset}|{par}')
        remark = (f'; Type1 only: {only1_between}/{len(everyone)} between protocols, '
                f'{only1_par}/{len(b_each)} in the comparator' if dset == 'everyone' else "")
        record('difference: within the 24 h minus between protocols', dset, par, s,
                 'paired bootstrap' + remark, quantity='difference in rho', subj=an)
        log(f"    {par:<9} {row(s, 'difference')}"
            + (f"  (resamples discarded: {100 * s['discarded']:.1f}%)"
               if s.get('discarded', 0) > 0 else ""))

k150v = two_visits(kinetics(155))
same_day = int((k150v.run_a == k150v.run_b).sum()) if len(k150v) else 0
log(f"\n150 h protocol, PD positive at two visits: {len(k150v)} subjects ({same_day} with the two "
    f"visits run on the same day). {('Comparison not estimable.' if len(k150v) < N_MIN else '')}")

log("\nSame visit, two runs at 24 h: units with two samples both positive")
log('run on different dates, first run against last. The file does not say whether the two')
log('runs use the same aliquot.')
s24 = d[d.PROJECTID == 237]
u2 = ["PATNO", "CLINICAL_EVENT"]
pos24 = s24[s24.SAA_Status == "Positive"]
date_pos = pos24.groupby(u2).RUNDATE.transform("nunique")
rip = pos24[date_pos > 1].sort_values(u2 + ["RUNDATE", 'row'])
earliest = rip.drop_duplicates(u2, keep="first").set_index(u2)
after = rip.drop_duplicates(u2, keep="last").set_index(u2)
assert earliest.index.equals(after.index), 'first and last run not aligned'
multi = s24[s24.groupby(u2).row.transform("size") > 1].groupby(u2).agg(
    no_pos=("SAA_Status", lambda s: bool((s != "Positive").any())),
    date_pos=("RUNDATE", lambda s: s[s24.loc[s.index, "SAA_Status"] == "Positive"].nunique()))
sel_ = multi.date_pos > 1
rest_np, rest_pp = int((~sel_ & multi.no_pos).sum()), int((~sel_ & ~multi.no_pos).sum())
assert int(sel_.sum()) == len(earliest) and len(earliest) + rest_np + rest_pp == len(multi)
log(f'  units at 24 h with more than one row: {len(multi)} = {len(earliest)} with two positive runs '
    f'on different dates (of which {int((sel_ & multi.no_pos).sum())} also have a non-positive run) '
    f'+ {rest_np} with a single positive date and a non-positive run + {rest_pp} with two positive '
    f'runs on the same date')
for name, sel in (('all cohorts', earliest.COHORT.notna()), ("PD", earliest.COHORT == "PD")):
    pr, ul = earliest[sel.values], after[sel.values]
    gg = (ul.RUNDATE - pr.RUNDATE).dt.days
    indet = int((pr.SAA_Type != "Type1").sum())
    und = int((pr.SAA_Type == "Undetermined").sum())
    log(f'\n  {name} (n = {len(pr)})')
    log(f"    type at the first / last run: "
        f"{(pr.SAA_Type.fillna('n.d.') + ' / ' + ul.SAA_Type.fillna('n.d.')).value_counts().to_dict()}")
    log(f'    first run positive but not Type1: {share(indet, len(pr))}, of which of type '
        f'Undetermined {und}; days between the runs: median {gg.median():.1f}, from {gg.min()} to '
        f'{gg.max()}')
    for par in PAR:
        st = estimate(pr[f"m24_{par}"], ul[f"m24_{par}"], f'tecnica|{name}|{par}')
        record('within the 24 h, same visit, two runs', name, par, st,
                 subj=pr.index.get_level_values("PATNO"))
        log(f'    {par:<9} {row(st)}')
log("  It is a selected set, as the counts above say. With the convention")
log('  of 01 (Type1 only) a first run that is not Type1 does not enter the value of the subject.')
n150rip = d[(d.PROJECTID == 155) & (d.SAA_Status == "Positive")].groupby(u2).RUNDATE.nunique()
log(f"  at 150 h the units with two positive runs on different dates are {int((n150rip > 1).sum())}")

# ------------------------------------------------------------------ 4. CEILING
title('4. ESTIMATE OF THE ATTENUATION CEILING FROM THE TRIPLICATES')
log('One-way ICC on the logarithmic scale, on the positive sample run last')
log("of each subject (Type1 at 24 h in the Type1 sets). The estimate of the ceiling is")
log('the square root of the product of the two ICC(1,3): the maximum correlation expected if the two')
log('protocols measured the same quantity with independent errors.')
log("It is an order of magnitude, for four reasons. ICC(1,3) is the reliability")
log("of the MEAN of three, whereas at 150 h the exposure is the median: it is")
log("an approximation whose direction of error is not known. It measures only the")
log("repeatability within the run: the variance between runs does not enter it and, alone,")
log("would bring the true ceiling lower. The formula holds for Pearson")
log("correlations, the observed one is a Spearman. And it is an estimate on few subjects:")
log("the interval resamples the subjects together in the two protocols, and where a resampled")
log("ICC is <= 0 the ceiling is 0 (share reported).")


def triplicates(project, root, subjects, only_type1, visit="BL"):
    s = d[(d.PROJECTID == project) & (d.SAA_Status == "Positive")
          & (d.COHORT == "PD") & (d.CLINICAL_EVENT == visit) & d.PATNO.isin(subjects)]
    if only_type1 and project == 237:
        s = s[s.SAA_Type == "Type1"]
    s = s.sort_values(["PATNO", "RUNDATE", 'row'])
    ult_ = s.drop_duplicates("PATNO", keep="last")
    # Guard on the DATA: the sample 'run last' must be a single one. If the
    # participant's second-to-last positive sample has the same run date as the
    # last, the triplet chosen would depend on the order of the file.
    pen_ = (s.drop(index=ult_.index).drop_duplicates("PATNO", keep="last")
            .set_index("PATNO").RUNDATE)
    ties_ = int((pen_ == ult_.set_index("PATNO").RUNDATE.reindex(pen_.index)).sum())
    assert ties_ == 0, (f"triplets: subjects with two positive samples on the last run date "
                        f"({('24' if project == 237 else '150')} h, {visit}): {ties_}; the sample "
                        f"run last is not defined")
    s = ult_.set_index("PATNO")
    # CODE CONSISTENCY check, not a check on the data: the caller passes participants
    # obtained with the same filters, so no data can trigger it.
    assert set(s.index) == set(subjects), ("triplets (consistency of the code): subjects different "
                                           "from the set")
    v = replicates(root, s)
    return np.log(v.where(v > 0))


CEILING = {}
ICC11_150 = {}
for name, m in DATASETS.items():
    if name in ALIAS:
        log(f"\n  {name}: coincides with '{ALIAS[name]}'")
        for par in PAR:
            CEILING[(name, par)] = CEILING[(ALIAS[name], par)]
            ICC11_150[(name, par)] = ICC11_150[(ALIAS[name], par)]
        continue
    log(f'\n  {name} (n = {len(m)})')
    for par, (a, b) in PAR.items():
        ta = triplicates(237, a, set(m.PATNO), name != 'everyone')
        tb = triplicates(155, b, set(m.PATNO), False)
        ok = ta.notna().all(axis=1) & tb.reindex(ta.index).notna().all(axis=1)
        ta, tb = ta[ok].values, tb.reindex(ta.index)[ok].values
        n = len(ta)
        if n < N_MIN:
            CEILING[(name, par)], ICC11_150[(name, par)] = np.nan, np.nan
            log(f'    {par:<9} not estimated (n = {n} < {N_MIN})')
            continue
        mua, wa = ta.mean(axis=1), ((ta - ta.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        mub, wb = tb.mean(axis=1), ((tb - tb.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        (i1a, i3a), (i1b, i3b) = icc_fr(mua, wa), icc_fr(mub, wb)
        idx = _gen(f'ceiling|{name}|{par}').integers(0, n, (NBOOT, n))
        ba, bb = icc_fr(mua[idx], wa[idx])[1], icc_fr(mub[idx], wb[idx])[1]
        def_ = (ba > 0) & (bb > 0)
        bs = np.where(def_, np.sqrt(np.where(def_, ba * bb, 0.0)), 0.0)
        estimable = bool(i3a > 0 and i3b > 0)
        point = float(np.sqrt(i3a * i3b)) if estimable else np.nan
        lo, hi = np.percentile(bs, [2.5, 97.5])
        obsv = BETWEEN[(name, par)]
        CEILING[(name, par)] = point
        ICC11_150[(name, par)] = float(i1b)
        text = (f'ceiling = {point:.2f} [{lo:.2f}; {hi:.2f}]' if estimable
                 else 'ICC(1,3) estimated <= 0: ceiling not estimable in this set')
        log(f"    {par:<9} ICC(1,1)/ICC(1,3): 24 h {i1a:+.3f}/{i3a:+.3f}, 150 h "
            f"{i1b:+.3f}/{i3b:+.3f} (n = {n}); {text}; not defined in {100 * (1 - def_.mean()):.0f}% "
            f"of the resamples; observed {obsv['rho']:+.3f} [{obsv['lo']:+.3f}; {obsv['hi']:+.3f}]")
        empty = dict(lo=np.nan, hi=np.nan, n=n)
        for gr, val in (('ICC(1,1) at 24 h', i1a), ('ICC(1,3) at 24 h', i3a),
                        ('ICC(1,1) at 150 h', i1b), ('ICC(1,3) at 150 h', i3b)):
            record('estimate of the attenuation ceiling', name, par,
                     dict(empty, rho=float(val)), quantity=gr, subj=m.PATNO)
        # where the ceiling cannot be estimated, neither can the interval: report and
        # table say the same thing
        record('estimate of the attenuation ceiling', name, par,
                 dict(n=n, rho=point, lo=float(lo) if estimable else np.nan,
                      hi=float(hi) if estimable else np.nan,
                      limit=bool(estimable and lo < NEARBY)),
                 variant=f'not defined in the {100 * (1 - def_.mean()):.1f}% of the resamples',
                 quantity='ceiling', subj=m.PATNO)
log('\n  A ceiling in no case explains a correlation of negative sign.')

# ------------------------------------------------------------------ 5. EXPLORATORY
title("5. EXPLORATORY ANALYSES, CHOSEN POST HOC (not corrected for multiplicity)")
WITHIN_P = {}     # (set, protocol) -> Spearman matrix within the protocol
for name in SENS:
    m = DATASETS[name]
    log(f"\n(a) All the pairs of parameters between protocols — '{name}' (n = {len(m)})")
    log('    rows = 24 h (mean), columns = 150 h (median); Spearman rho')
    mat = pd.DataFrame(index=P24, columns=P150, dtype=float)
    details = []
    for x in P24:
        for y in P150:
            s = estimate(m[f"m24_{x}"], m[f"md150_{y}"], f'incrocio|{name}|{x}|{y}')
            mat.loc[x, y] = s["rho"]
            if x != y:          # the same-name pairs are already in section 2
                et = f'{x} at 24 h versus {y} at 150 h'
                record('exploratory: pairs of parameters between protocols', name, et, s,
                         subj=m.PATNO)
                details.append(f'      {et:<38} {row(s)}')
    log("    " + mat.round(2).to_string().replace("\n", "\n    "))
    log('    With interval, the pairs of different name:')
    for r_ in details:
        log(r_)
    log(f'    Within the protocol, same subjects:')
    for tags, pref, names in (('24 h', "m24", P24), ('150 h', "md150", P150)):
        cm = m[[f"{pref}_{p}" for p in names]].corr(method="spearman")
        cm.index, cm.columns = names, names
        WITHIN_P[(name, tags)] = cm
        log(f'    {tags}\n    ' + cm.round(2).to_string().replace("\n", "\n    "))
        for i_, x in enumerate(names):
            for y in names[i_ + 1:]:
                record('exploratory: within the protocol', name,
                         f'{x} against {y} a {tags}',
                         dict(n=len(m), rho=float(cm.loc[x, y]), lo=np.nan, hi=np.nan),
                         quantity='rho within the protocol', subj=m.PATNO)
    c24_, c150_ = WITHIN_P[(name, '24 h')], WITHIN_P[(name, '150 h')]
    log(f"    AUC in '{name}': at 150 h rho {c150_.loc['AUC', 'TTT']:+.2f} with the TTT and "
        f"{c150_.loc['AUC', 'Fmax']:+.2f} with Fmax; at 24 h {c24_.loc['AUC', 'TTT']:+.2f} with the "
        f"TTT and {c24_.loc['AUC', 'Fmax']:+.2f} with Fmax.")
log("\n    Where the AUC follows the times at 150 h and the amplitude at 24 h, its correlation")
log('    between protocols and those between amplitude at 24 h and times at 150 h are not')
log('    independent findings.')

log('\n(b) Slope and TTT at well level: deviations from the mean of their own')
log('    triplet, logarithmic scale, positive sample run last. It describes')
log('    how the two variables are constructed, not a relation between people.')
for name in SENS:
    m = DATASETS[name]
    for tags, pr, rs, rt in (('24 h', 237, "SLOPEMax_24h", "TTT_24h"),
                              ('150 h', 155, "SLOPE_150h", "TTT_150h")):
        t1 = name != 'everyone'
        a, b = triplicates(pr, rs, set(m.PATNO), t1), triplicates(pr, rt, set(m.PATNO), t1)
        ok = a.notna().all(axis=1) & b.notna().all(axis=1)
        a, b = a[ok].values, b[ok].values
        r = _rho((a - a.mean(axis=1, keepdims=True)).ravel(),
                 (b - b.mean(axis=1, keepdims=True)).ravel())
        log(f'    {name:<16} {tags:<8} rho between wells = {r:+.2f} ({ok.sum()} triplets)')
        record('exploratory: slope and TTT between wells', name,
                 f'Slope against TTT at {tags}',
                 dict(n=int(ok.sum()), rho=float(r), lo=np.nan, hi=np.nan),
                 variant='deviations from the mean of the triplet; n = triplets',
                 quantity='rho between wells', subj=m.PATNO)
log('    ICC(1,1) of the slope at 150 h: '
    + "; ".join(f"{n} {ICC11_150[n, 'Slope']:+.2f}" for n in SENS))
for name in SENS:
    r24 = WITHIN_P[(name, '24 h')].loc['Slope', "TTT"]
    r150 = WITHIN_P[(name, '150 h')].loc['Slope', "TTT"]
    log(f"    Between subjects, '{name}': rho of the slope with the TTT {r24:+.2f} at 24 h and "
        f"{r150:+.2f} at 150 h ({('opposite sign' if r24 * r150 < 0 else 'same sign')}).")
log("    Where that ICC is close to zero the slope at 150 h is almost all well")
log("    noise and the relation between subjects is not estimable: an opposite sign")
log('    then says that the two variables are computed differently, not that')
log('    people are ranked in the opposite order.')

log(f'\n(c) After centring each parameter on its own run date. Logarithmic')
log(f'    scale; run means over all the positive samples of the protocol,')
log(f'    every cohort; runs with fewer than {RUN_MIN} samples in a single category.')
log("    The run date may also reflect when and where the sample was")
log('    collected: the file has no collection dates and does not allow this to be verified.')
pos = d[d.SAA_Status == "Positive"]
for project, pref in ((237, "m24"), (155, "md150")):
    s = pos[pos.PROJECTID == project]
    cat = s.RUNDATE.astype(str)
    cat = cat.where(cat.map(cat.value_counts()) >= RUN_MIN, 'small runs')
    for par in PAR:
        v = np.log(s[f"{pref}_{par}"].where(s[f"{pref}_{par}"] > 0))
        d.loc[s.index, f"c_{pref}_{par}"] = v - v.groupby(cat).transform("mean")
k150c = kinetics(155, columns=[f"c_md150_{p}" for p in PAR])
cen = {'everyone': pairs(kinetics(237, columns=[f"c_m24_{p}" for p in PAR]), k150c)}
c1 = pairs(kinetics(237, only_type1=True, columns=[f"c_m24_{p}" for p in PAR]), k150c)
cen['sporadic Type1'] = c1[c1.PATNO.isin(sporadic)]
CENT = {}       # (set, parameter) -> centred estimate, for section 6
for name in SENS:
    # CODE CONSISTENCY check, not a check on the data: the two sets come from the
    # same row filters and no data can make them differ.
    assert set(cen[name].PATNO) == set(DATASETS[name].PATNO), \
        'different centred set (consistency of the code)'
    log(f'    {name} (n = {len(cen[name])})')
    for par in PAR:
        s = estimate(cen[name][f"c_m24_{par}"], cen[name][f"c_md150_{par}"],
                  f'centrato|{name}|{par}')
        CENT[(name, par)] = s
        record('exploratory: centred on the run date', name, par, s,
                 subj=cen[name].PATNO)
        log(f'      {par:<9} {row(s)}')

carriers = type1[~type1.PATNO.isin(sporadic)]
excludes = []
log(f'\n(d) Carriers against sporadic, Type1 at 24 h. Non-sporadic n = {len(carriers)}: '
    f'{composition(carriers.PATNO)}')
for par in PAR:
    s = estimate(carriers[f"m24_{par}"], carriers[f"md150_{par}"], f'between|no sporadic Type1|{par}')
    record('exploratory: non-sporadic', 'non-sporadic Type1', par, s, 'mean/median',
             subj=carriers.PATNO)
    rs = BETWEEN[('sporadic Type1', par)]
    if s["bs"] is not None and rs["bs"] is not None:
        dd = s["bs"] - rs["bs"]             # disjoint sets: resampled separately
        lo, hi = np.nanpercentile(dd, [2.5, 97.5])
        dif = dict(n=np.nan, rho=s["rho"] - rs["rho"], lo=float(lo), hi=float(hi),
                   limit=bool(min(abs(lo), abs(hi)) < NEARBY))
        record('exploratory: non-sporadic', 'non-sporadic minus sporadic, Type1',
                 par, dif, quantity='difference in rho')
        log(f'    {par:<9} {row(s)}')
        log(f"    {'':<9} difference from the sporadic = {dif['rho']:+.3f} [{dif['lo']:+.3f}; "
            f"{dif['hi']:+.3f}]")
        if dif["lo"] > 0 or dif["hi"] < 0:
            excludes.append(par)
log('    Intervals of the difference that exclude zero: '
    + (", ".join(excludes) if excludes else 'none') + ".")
log('    The estimates applicable to the manuscript on sporadic participants are those of')
log("    'sporadici Type1'.")

# ------------------------------------------------------------------ 6. DIFFERENT VISITS
title('6. KINETIC PARAMETERS BETWEEN PROTOCOLS, DIFFERENT VISITS')
PRINC = ("BL", "V04")         # main comparison: 150 h at baseline, 24 h at V04
NAME_COHORT = '150 h cohort, sporadic Type1'
# In the table each new row carries the population and the unit of measurement
# in full; in the report the sets keep the short name of the other sections.
LBLS_BETWEEN = {
    'everyone': 'all: PD positive in both protocols, any type at 24 h, any genotype',
    'Type1 at 24 h': 'Type1 at 24 h: PD positive at 150 h and Type1 at 24 h, every genotype',
    'sporadic Type1': 'sporadic Type1: sporadic PD, positive at 150 h and Type1 at 24 h',
    NAME_COHORT: NAME_COHORT + ': members of the 150 h cohort, sporadic, positive at 150 h and '
                               'Type1 at 24 h',
}
LBLS_WITHIN = {
    'everyone': 'all: PD positive at 24 h at both visits, every type, every genotype',
    'Type1 at 24 h': 'Type1 at 24 h: PD Type1 at 24 h at both visits, any genotype',
    'sporadic Type1': 'sporadic Type1: sporadic PD, Type1 at 24 h at both visits',
}
FLAGGED = []    # limits close to zero of the estimates recorded in this section
ERR_MC = []       # (SE of the lower limit, SE of the upper) of each recorded interval


def subj_(n):
    return f'{n} subject' if n == 1 else f'{n} subjects'


def error_mc(bs):
    """Monte Carlo standard error of the two percentile limits of a bootstrap.

    Batch estimate: the resamples, independent of one another, are divided into
    BATCHES_MC consecutive batches; in each the two percentiles are recomputed;
    the standard error of the limit computed on all the resamples is the
    standard deviation between batches divided by the square root of the number
    of batches."""
    if bs is None:
        return np.nan, np.nan
    q = np.array([np.nanpercentile(l_, [2.5, 97.5]) if np.isfinite(l_).any()
                  else [np.nan, np.nan]
                  for l_ in np.array_split(np.asarray(bs, float), BATCHES_MC)])
    k = np.isfinite(q).sum(axis=0)
    if k.min() < 2:
        return np.nan, np.nan
    se = np.nanstd(q, axis=0, ddof=1) / np.sqrt(k)
    return float(se[0]), float(se[1])


def mc_of(s):
    if "mc" not in s:
        s["mc"] = error_mc(s.get("bs"))
    return s["mc"]


def having_interval(s):
    """True for the estimates with a bootstrap interval (not for counts and times)."""
    return s.get("bs") is not None or "mc" in s


def neighbours(s):
    """The limits of the interval of s that are not read as a sign: they lie less
    than max(VICINO, SE_MC Monte Carlo standard errors of that limit) from zero."""
    if not having_interval(s) or not np.isfinite(s.get("rho", np.nan)):
        return []
    out = []
    for name_, v, e in zip(('lower', 'upper'), (s["lo"], s["hi"]), mc_of(s)):
        if not np.isfinite(v):
            continue
        threshold = max(NEARBY, SE_MC * e) if np.isfinite(e) else NEARBY
        if abs(v) < threshold:
            out.append(dict(limit=name_, value=float(v), threshold=float(threshold),
                            only_mc=bool(abs(v) >= NEARBY)))
    return out


def remark_nearby(s):
    v = neighbours(s)
    if not v:
        return ""
    if not all(x['only_mc'] for x in v):
        return '  (one limit is close to zero: it is not read as a sign)'
    return (f"  (one limit is within Monte Carlo error of zero [{SE_MC:g} SE = "
            f"{v[0]['threshold']:.3f}]: not read as a sign)")


def row_v(s, what="rho"):
    """riga() with the flagging of the limits close to zero of section 6."""
    if not np.isfinite(s["rho"]):
        return f'not estimated (n < {N_MIN})'
    t = f"{what} = {s['rho']:+.3f}  [{s['lo']:+.3f}; {s['hi']:+.3f}]"
    if np.isfinite(s["n"]):
        t = f"n = {s['n']:>3}  " + t
    if np.isfinite(s.get("p", np.nan)):
        t += f"  p = {s['p']:.3g}"
    return t + remark_nearby(s)


def excludes_zero(s):
    return bool(np.isfinite(s["rho"]) and (s["lo"] > 0 or s["hi"] < 0) and not neighbours(s))


def listing_ez(ss, empty='no parameter', sign=False):
    """ss: {parameter: estimate}. The parameters whose interval excludes zero and,
    next to them, those with a limit close to zero, which count neither as a
    clear exclusion nor as a clear inclusion."""
    ez = [p + (f" ({('positive' if s['rho'] > 0 else 'negative')})" if sign else "")
          for p, s in ss.items() if excludes_zero(s)]
    vz = [f"{p} [{s['lo']:+.3f}; {s['hi']:+.3f}]" for p, s in ss.items() if neighbours(s)]
    return ((", ".join(ez) if ez else empty)
            + (f" (with one limit close to zero, which is not read as a sign: {', '.join(vz)})" if vz else ""))


def record_v(block, dataset, parameter, s, variant="", quantity="rho",
               units='dimensionless', **other):
    """registra() with the unit of measurement written in the 'grandezza' column.
    For the estimates with a bootstrap interval the limit_nearby_a_zero column
    follows the rule of section 6 and one more row carries the Monte Carlo error."""
    if not having_interval(s):
        record(block, dataset, parameter, s, variant, f'{quantity} [{units}]', **other)
        return
    v = neighbours(s)
    record(block, dataset, parameter, dict(s, limit=bool(v)), variant,
             f'{quantity} [{units}]', **other)
    if not np.isfinite(s.get("rho", np.nan)):
        return
    e = mc_of(s)
    record(block, dataset, parameter,
             dict(n=s.get("n", np.nan), rho=np.nan, lo=e[0], hi=e[1]), variant,
             f'Monte Carlo standard error of the limits (ci_inf = SE of the lower limit, ci_upp = SE '
             f'of the upper; {BATCHES_MC} batches) of: {quantity} [{units}]', **other)
    if not other.get('coincides_having'):      # a repeated estimate is counted once
        ERR_MC.append(e)
        for x in v:
            FLAGGED.append(dict(x, block=block, dataset=dataset, parameter=parameter,
                                  variant=variant))


log(f'At the same visit the PD positive in both protocols are {len(everyone)} (section 2).')
log("Here the correlation is between the value at 150 h of one visit and the value at 24 h")
log("of ANOTHER visit of the same subject, for each pair of visits with at least")
log(f"{N_MIN} subjects. Unit: Spearman rho, dimensionless; 95% percentile bootstrap")
log(f'intervals on {NBOOT} resamples of the subjects.')
log('CAVEATS. They are DIFFERENT collections. To protocol, storage and run date,')
log("already confounded at the same visit (section 2), is added the time between the two")
log('collections, and with it the biological change. These data do not separate any')
log('of these factors from the others: for each comparison the time between the')
log("visits and between the runs is reported, and the benchmark is the 24 h protocol repeated")
log('on the same two visits (6.2). Conventions of the kinetic parameters, limits of the')
log('intervals and caveat on the slope: as in section 2. The p values are not corrected')
log("for multiplicity. The 150 h protocol does not assign a type: 'Type1' always")
log('and only refers to the sample at 24 h.')
log(f"LIMITS CLOSE TO ZERO. In this section a limit of the interval is not read as a")
log(f'sign if it lies less than {NEARBY} from zero or less than {SE_MC:g} Monte Carlo standard '
    f'errors (SE)')
log(f"of the limit itself. The Monte Carlo SE is estimated by batches: the resamples")
log(f'are divided into {BATCHES_MC} consecutive batches, the two percentiles are recomputed in each,')
log("and the SE is the standard deviation between batches divided by the square root of the number of")
log('batches. It also holds for the estimates of section 2 when they are recalled here. The SE')
log("of each interval are in the table; the summary is in 6.7.")
log("LAST-RUN RULE. The main convention is that of the cohort files")
log('(mean of the positive samples of the visit; of the Type1 only in the Type1 sets).')
log("Next to each estimate there is the same estimate with the last-run rule: for")
log('each visit and protocol only the sample run last counts, and the subject')
log("enters if that sample is positive (and Type1 at 24 h, in the Type1 sets).")

# dates of the visits, from the curated data cut: they only serve to describe the times
COL_VIS = ["PATNO", "EVENT_ID", "age_at_visit", "age_LP", "visit_date"]
if os.path.exists(CACHE):
    cv = pd.read_parquet(CACHE, columns=COL_VIS)
else:
    cv = pd.read_excel(located[0], sheet_name="20260511", usecols=COL_VIS)
cv["PATNO"] = cv["PATNO"].astype("int64")
assert not cv.duplicated(["PATNO", "EVENT_ID"]).any(), \
    'curated: a subject x visit pair appears twice'
cv['month_visit'] = pd.to_datetime(cv.visit_date.astype(str), format="%m/%Y",
                                   errors="coerce")
cv = cv.set_index(["PATNO", "EVENT_ID"])


def fromthe_curated(patno, visit, col):
    return cv[col].reindex(pd.MultiIndex.from_arrays([patno, visit])).values


def months_cal(fr, a):
    """Calendar months from month 'da' to month 'a' (NaN where a date is missing)."""
    fr, a = pd.DatetimeIndex(fr), pd.DatetimeIndex(a)
    return np.asarray((a.year - fr.year) * 12 + (a.month - fr.month), float)


def times(m, sa, sb, na, nb):
    """Distances between the two halves of each pair, second minus first. sa, sb
    are the column suffixes; na, nb the names with which the two halves are to
    be printed. The time between the month of the visit and the run approximates
    the storage of the sample: the curated data cut gives the visit to the
    month, not the collection date.

    Plausibility guards on the data of the curated data cut: between two
    different visits age must increase in the direction of the visit codes, and
    no sample can have been run before the month of the visit at which it was
    collected."""
    p = m.PATNO.values
    ea, eb = m[f"CLINICAL_EVENT_{sa}"].values, m[f"CLINICAL_EVENT_{sb}"].values
    ca, cb = m[f'run_{sa}'], m[f'run_{sb}']
    cons_a = months_cal(fromthe_curated(p, ea, 'month_visit'), ca)
    cons_b = months_cal(fromthe_curated(p, eb, 'month_visit'), cb)
    between_vis = 12 * (fromthe_curated(p, eb, "age_at_visit") - fromthe_curated(p, ea, "age_at_visit"))
    between_lp = 12 * (fromthe_curated(p, eb, "age_LP") - fromthe_curated(p, ea, "age_LP"))
    direction = np.sign(np.array([ordering_visit(e) for e in eb], float)
                    - np.array([ordering_visit(e) for e in ea], float))
    for col_, x in (("age_at_visit", between_vis), ("age_LP", between_lp)):
        ok = np.isfinite(x) & np.isfinite(direction) & (direction != 0)
        against = int((x[ok] * direction[ok] <= 0).sum())
        assert against == 0, (
            f'curated: in {subj_(against)} {col_} does not increase in the direction of the visits '
            f'({na} -> {nb}): a later visit cannot precede the earlier one')
    for side_, x in ((na, cons_a), (nb, cons_b)):
        earliest_ = int((x[np.isfinite(x)] < 0).sum())
        assert earliest_ == 0, (
            f'curated: {earliest_} samples ({side_}) appear to have been run before the month of the '
            f'visit at which they were collected')
    return [
        (f'days between the two runs ({nb} minus {na})', 0,
         (cb - ca).dt.days.values.astype(float), 'days'),
        (f'months between the two visits, from age_at_visit of the curated data cut ({nb} minus {na})', 1,
         between_vis, 'months'),
        (f'months between the two draws, from age_LP of the curated data cut ({nb} minus {na})', 1, between_lp,
         'months'),
        (f'calendar months between the month of the visit (curated data cut) and the run, {na}', 0,
         cons_a, 'months'),
        (f'calendar months between the month of the visit (curated data cut) and the run, {nb}', 0,
         cons_b, 'months'),
        (f'difference between the two visit-run times, in calendar months ({nb} minus {na})',
         0, cons_b - cons_a, 'months'),
    ]


def distr(x, dec):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return 'NOT available (the curated data cut has the datum for none of the subjects)'
    q = np.percentile(x[ok], [0, 25, 50, 75, 100])
    t = (f'minimum {q[0]:.{dec}f}, quartiles {q[1]:.{dec}f} / {q[2]:.{dec}f} / {q[3]:.{dec}f}, '
         f'maximum {q[4]:.{dec}f}')
    return t + (f' (available for {int(ok.sum())} of {len(x)})' if not ok.all() else "")


def describe_times(m, sa, sb, na, nb, block, dataset):
    """Prints and records the times and the run dates of a set of pairs."""
    for side, s_ in ((na, sa), (nb, sb)):
        c = m[f'run_{s_}']
        log(f'      distinct run dates, {side}: {c.nunique()} (from {c.min().date()} to '
            f'{c.max().date()})')
        record_v(block, dataset, 'run dates',
                   dict(n=len(m), rho=float(c.nunique()), lo=np.nan, hi=np.nan),
                   quantity=f'number of distinct run dates, {side}', units="date",
                   subj=m.PATNO)
    for name_t, dec, x, an_ in times(m, sa, sb, na, nb):
        log(f'      {name_t}: {distr(x, dec)}')
        ok = np.isfinite(x)
        record_v(block, dataset, 'times',
                   dict(n=int(ok.sum()),
                        rho=float(np.median(x[ok])) if ok.any() else np.nan,
                        lo=float(x[ok].min()) if ok.any() else np.nan,
                        hi=float(x[ok].max()) if ok.any() else np.nan),
                   variant='ci_inf = minimum, ci_upp = maximum',
                   quantity=name_t + ': median', units=an_, subj=m.PATNO[ok])


# The three times on which it is judged whether a benchmark is matched to the
# comparison between protocols: position in the list of tempi(), short name,
# decimals.
TIMES_APP = ((1, 'months between the two visits', 1), (0, 'days between the two runs', 0),
             (5, 'difference in storage (months visit-run, second half minus first)', 0))


def pairing(t_par, t_between):
    """For each of the three times: median in the benchmark, median between
    protocols, and whether the two interquartile ranges overlap ('appaiato').
    None where the curated data cut does not have the datum."""
    out = []
    for i_, name_, dec in TIMES_APP:
        xb, xf = (np.asarray(t_[i_][2], float) for t_ in (t_par, t_between))
        xb, xf = xb[np.isfinite(xb)], xf[np.isfinite(xf)]
        if not len(xb) or not len(xf):
            out.append(dict(name=name_, dec=dec, app=None))
            continue
        qb, qf = np.percentile(xb, [25, 50, 75]), np.percentile(xf, [25, 50, 75])
        out.append(dict(name=name_, dec=dec, par=float(qb[1]), between=float(qf[1]),
                        app=bool(max(qb[0], qf[0]) <= min(qb[2], qf[2]))))
    return out


def text_app(a):
    if a["app"] is None:
        return f"{a['name']}: NOT available"
    return (f"{a['name']}: {a['par']:+.{a['dec']}f} against {a['between']:+.{a['dec']}f} "
            f"({('paired' if a['app'] else 'NOT matched')})")


def ord_vis(e):
    v = ordering_visit(e)
    return (v if np.isfinite(v) else 999.0, e)


def a_visit(k, v):
    """The rows of a visit, indexed by participant."""
    return k.xs(v, level="CLINICAL_EVENT")


def visits_of(k):
    return sorted(k.index.get_level_values("CLINICAL_EVENT").unique(), key=ord_vis)


def replicates_min(project, only_type1=False, only_last=False):
    """Per participant x visit, the minimum number of replicates available among
    the samples that enter the value (same row filters as cinetiche())."""
    pref = "nr24" if project == 237 else "nr150"
    return (rows_kinetics(project, "PD", only_type1, only_last)
            .groupby(["PATNO", "CLINICAL_EVENT"])
            .agg(**{f"{pref}_{p}": (f"{pref}_{p}", "min") for p in PAR}))


def having_replicates(project, **selectors):
    return kinetics(project, **selectors).join(replicates_min(project, **selectors),
                                              validate="one_to_one")


# main convention (that of the cohort files) and last-run rule
k24o, k24t = having_replicates(237), having_replicates(237, only_type1=True)
K150 = k150.join(replicates_min(155), validate="one_to_one")
u24o = having_replicates(237, only_last=True)
u24t = having_replicates(237, only_type1=True, only_last=True)
U150 = having_replicates(155, only_last=True)


def pairs_between(k24, va, vb, k150_=None):
    """150 h at visit va, 24 h at visit vb, same participant."""
    m = (a_visit(K150 if k150_ is None else k150_, va)
         .join(a_visit(k24, vb), how="inner", lsuffix="_150", rsuffix="_24")
         .reset_index())
    m["CLINICAL_EVENT_150"], m["CLINICAL_EVENT_24"] = va, vb
    return m


def pairs_within(k, va, vb):
    """The same protocol at visits va and vb, same participant."""
    m = (a_visit(k, va).join(a_visit(k, vb), how="inner", lsuffix="_a", rsuffix="_b")
         .reset_index())
    m["CLINICAL_EVENT_a"], m["CLINICAL_EVENT_b"] = va, vb
    return m


def same_values(m1, m2, col):
    return (set(m1.PATNO) == set(m2.PATNO)
            and np.array_equal(m1.sort_values("PATNO")[col].values,
                               m2.sort_values("PATNO")[col].values, equal_nan=True))


def latest_of(patno, visit, project):
    """Outcome (and type, at 24 h) of the sample run last, for a list of participants."""
    v = vis[(vis.PROJECTID == project) & (vis.CLINICAL_EVENT == visit)
            & vis.PATNO.isin(patno)]
    if project == 155:
        return v.outcome.value_counts().to_dict()
    return (v.outcome + np.where(v.outcome == "Positive", " " + v.kind.fillna("n.d."), "")
            ).value_counts().to_dict()


ULT_REASS = []     # one entry per estimated set: what changes with the last-run rule


def having_latest_run(m, u, colx, coly, seed, block, dataset, princ, sides, t1, key_r):
    """Sets next to the main estimate the same estimate with the last-run rule.
    m: the set in the main convention; u: the same set with the last-run rule;
    colx, coly: parameter -> column; princ: parameter -> main estimate;
    lati: [(visit, project)] of the two halves."""
    pm, pu = set(m.PATNO), set(u.PATNO)
    lost, added = pm - pu, pu - pm
    com = sorted(pm & pu)
    cols = [c[p] for p in PAR for c in (colx, coly)]
    a = m.set_index("PATNO").loc[com, cols].values.astype(float)
    b = u.set_index("PATNO").loc[com, cols].values.astype(float)
    different = int((~((a == b) | (np.isnan(a) & np.isnan(b)))).any(axis=1).sum())
    equals = not lost and not added and different == 0
    log(f"    last-run rule (sample run last at each visit, "
        f"positive{(' and Type1 at 24 h' if t1 else '')}): n = {len(u)}; subjects excluded "
        f"{len(lost)}, added {len(added)}, remaining with a different value {different}"
        + ("" if equals else '  <- DIFFERENT SUBJECTS OR VALUES'))
    if lost:
        log('      sample run last in the excluded: '
            + "; ".join(f"{('150' if pr_ == 155 else '24')} h {v_} {latest_of(lost, v_, pr_)}" for v_, pr_ in sides))
    if equals:
        log('      coincides with the main estimate, value by value.')
    estimates = {}
    for par in PAR:
        if equals:
            s = princ[par]
        else:
            s = estimate(u[colx[par]], u[coly[par]], f'{seed}|{par}')
            log(f'      {par:<9} {row_v(s)}'
                + (f"  [main {princ[par]['rho']:+.3f}, change {s['rho'] - princ[par]['rho']:+.3f}]"
                   if np.isfinite(s["rho"]) and np.isfinite(princ[par]["rho"]) else ""))
        estimates[par] = s
        record_v(block, dataset, par, s,
                   'last run' + ("" if not lost and not added else ', different subjects'),
                   coincides_having='main estimate' if equals else "", subj=u.PATNO)
    r = dict(key=key_r, n0=len(m), n1=len(u), lost=len(lost), added=len(added),
             different=different, equals=equals, estimates=estimates, princ=princ)
    ULT_REASS.append(r)
    return r


def summary_latest(rows_r, where):
    """The sentence, derived, on how many participants and which estimates change."""
    name_r = lambda r: f"'{r['key'][2]}' {r['key'][0]}-{r['key'][1]}"
    cs = [r for r in rows_r if r['lost'] or r['added']]
    sv = [r for r in rows_r if not (r['lost'] or r['added']) and r['different']]
    ug = [r for r in rows_r if r['equals']]
    log(f'\nLast-run rule, summary of {where} (derived). Sets estimated, without those that coincide '
        f'with another: {len(rows_r)}.')
    log(f'  The subjects change in {len(cs)}'
        + (": " + "; ".join(f"{name_r(r)} from n = {r['n0']} a n = {r['n1']} (excluded {r['lost']}, "
                            f"added {r['added']}, left with a different value {r['different']})" for r in cs) if cs else "") + ".")
    log(f'  Only the values change, not the subjects, in {len(sv)}'
        + (": " + "; ".join(f"{name_r(r)} ({subj_(r['different'])} out of {r['n0']})"
                            for r in sv) if sv else "") + ".")
    log(f'  They coincide with the main estimate, value by value: {len(ug)}.')
    var = [(r, p, r['estimates'][p], r["princ"][p]) for r in cs + sv for p in PAR
           if np.isfinite(r['estimates'][p]["rho"]) and np.isfinite(r["princ"][p]["rho"])]
    dropped = [f'{p} in {name_r(r)}' for r in cs + sv for p in PAR
             if np.isfinite(r["princ"][p]["rho"]) and not np.isfinite(r['estimates'][p]["rho"])]
    if not var:
        log('  No estimate changes.')
    else:
        dl = np.array([abs(s["rho"] - s0["rho"]) for _, _, s, s0 in var])
        r, p, s, s0 = var[int(dl.argmax())]
        log(f"  Estimates that change: {len(var)} ({len(PAR)} parameters in {len(cs) + len(sv)} "
            f"sets). Absolute change in rho: median {np.median(dl):.3f}, maximum {dl.max():.3f} ({p} "
            f"in {name_r(r)}: from {s0['rho']:+.3f} to {s['rho']:+.3f}).")
        sign = [f"{p} in {name_r(r)} (from {s0['rho']:+.3f} a {s['rho']:+.3f})"
                 for r, p, s, s0 in var if s["rho"] * s0["rho"] < 0]
        state = [f"{p} in {name_r(r)} (from [{s0['lo']:+.3f}; {s0['hi']:+.3f}] a [{s['lo']:+.3f}; "
                 f"{s['hi']:+.3f}])"
                 for r, p, s, s0 in var if excludes_zero(s) != excludes_zero(s0)]
        log('  Estimates that change sign: ' + ("; ".join(sign) if sign else 'none')
            + ". Intervals that exclude zero in one convention and not in the other: "
            + ("; ".join(state) if state else 'none') + ".")
    if dropped:
        log(f"  Estimates no longer computable (n < {N_MIN}) with the last-run rule: "
            + "; ".join(dropped) + ".")


def replicates_incomplete(m, col24, col150, colx, coly, seed, block, dataset, variant,
                        princ, n_par=None):
    """Where the value of some participant rests on fewer than three replicates it
    says so, parameter by parameter, and sets next to it the estimate on
    complete triplets only.
    col24, col150: parameter -> columns with the minimum number of replicates."""
    found = False
    for par in PAR:
        k24_ = np.zeros(len(m), bool)
        for c in col24[par]:
            k24_ |= (m[c].values < 3)
        k150_ = np.zeros(len(m), bool)
        for c in col150[par]:
            k150_ |= (m[c].values < 3)
        inc = k24_ | k150_
        if not inc.any():
            continue
        found = True
        mc_ = m[~inc]
        s = estimate(mc_[colx[par]], mc_[coly[par]], f'{seed}|{par}')
        log(f'    {par}: the value of {subj_(int(inc.sum()))} rests on fewer than three replicates '
            f'(at 24 h it is the MEAN of the available replicates: {subj_(int(k24_.sum()))}'
            + (f'; at 150 h the MEDIAN: {int(k150_.sum())}' if col150[par] else "")
            + f'). With the complete triplets only: {row_v(s)}')
        record_v(block, dataset, par, s, variant + '; complete triplets only',
                   subj=mc_.PATNO)
        record_v(block, dataset, par,
                   dict(n=len(m), rho=float(inc.sum()), lo=np.nan, hi=np.nan),
                   variant + '; 24 h: mean of the available replicates'
                   + ('; 150 h: median' if col150[par] else ""),
                   quantity='subjects whose value rests on fewer than three replicates',
                   units='subjects', subj=m.PATNO[inc])
    if not found:
        log("    replicates: every value of the set rests on three replicates, for all the "
            "parameters.")


# sets with the last-run rule and sets of the benchmark: built here because
# 6.0 and 6.1 cite them before 6.2 estimates them
def datasets_between(va, vb, latest=False):
    ko, kt, k5 = (u24o, u24t, U150) if latest else (k24o, k24t, K150)
    t, t1 = pairs_between(ko, va, vb, k5), pairs_between(kt, va, vb, k5)
    dset = {'everyone': t, 'Type1 at 24 h': t1,
           'sporadic Type1': t1[t1.PATNO.isin(sporadic)]}
    if va == "BL":       # the values of the 150 h cohort are the baseline ones
        dset[NAME_COHORT] = t1[t1.PATNO.isin(p150 & sporadic)]
    return dset


K_WITHIN = {'everyone': (k24o, u24o), 'Type1 at 24 h': (k24t, u24t),
            'sporadic Type1': (k24t, u24t)}
vis24 = [v for v in visits_of(k24o) if np.isfinite(ordering_visit(v))]
count_d = {(va, vb): len(pairs_within(k24o, va, vb))
           for i_, va in enumerate(vis24) for vb in vis24[i_ + 1:]}
PAIRS_D = [c for c, n in count_d.items() if n >= N_MIN]
INSD, ULTD = {}, {}
for va, vb in PAIRS_D:
    for name, (k, u) in K_WITHIN.items():
        m, mu = pairs_within(k, va, vb), pairs_within(u, va, vb)
        if name == 'sporadic Type1':
            m, mu = m[m.PATNO.isin(sporadic)], mu[mu.PATNO.isin(sporadic)]
        INSD[(va, vb, name)], ULTD[(va, vb, name)] = m, mu
COMPARISON_OF = {'everyone': 'everyone', 'Type1 at 24 h': 'Type1 at 24 h',
               'sporadic Type1': 'sporadic Type1', NAME_COHORT: 'sporadic Type1'}
ties_u = set(map(tuple, pp_[["PATNO", "CLINICAL_EVENT", "PROJECTID"]].values))


def without_ties(m, sides, where):
    """Guard on the DATA: no unit of the set with two rows on the last run date,
    otherwise the last-run rule and the triplet of the ceiling would depend on
    the order of the file. Returns the number (zero)."""
    n_par = sum((p_, v_, pr_) in ties_u for p_ in m.PATNO for v_, pr_ in sides)
    assert n_par == 0, (
        f'{where}: {n_par} units with two rows on the last run date; the sample run last is not '
        f'defined and the last-run rule cannot be applied')
    return n_par


# --- identity with the cohort files
log("\n6.0 Identity with the cohort files")
k150bl = a_visit(k150, "BL")
without150 = p150 - set(k150bl.index)
assert not without150, (f'{len(without150)} members of the 150 h cohort without a positive PD sample '
                      f'at 150 h at baseline in the raw file')
COL150_COHORT = COMPARISONS[1][3]
COL24_COHORT = COMPARISONS[0][3]
ord150 = sorted(p150)
for x, y in COL150_COHORT:
    assert np.allclose(k150bl.loc[ord150, x].values,
                       c150.set_index("PATNO").loc[ord150, y].values, rtol=1e-9, atol=0), \
        f'{x} different from {y} in some member of the 150 h cohort'
log(f'  150 h: all the {len(p150)} members of the cohort have in the raw file a positive')
log(f'  PD sample at 150 h at baseline, and the {len(COL150_COHORT)} kinetic values recomputed here '
    f'coincide with those saved in the cohort.')
# The 24 h checks, with the result of each: (name, performed, participants, reason)
CTRL24 = []
k24bl = a_visit(k24t, "BL")
if p24:
    without24bl = p24 - set(k24bl.index)
    assert not without24bl, (f'{len(without24bl)} members of the 24 h cohort without a PD Type1 sample '
                           f'at 24 h at baseline in the raw file')
    ord24 = sorted(p24)
    for x, y in COL24_COHORT:
        assert np.allclose(k24bl.loc[ord24, x].values,
                           c24.set_index("PATNO").loc[ord24, y].values, rtol=1e-9, atol=0), \
            f'{x} different from {y} in some member of the 24 h cohort'
    log(f'  24 h at baseline: all the {len(p24)} members of the 24 h cohort have in the raw file a '
        f'PD')
    log(f'  Type1 sample at 24 h at baseline, and the {len(COL24_COHORT)} kinetic values recomputed '
        f'here coincide with those saved in the cohort.')
else:
    log("  24 h at baseline: NOT verified, the 24 h cohort of the working directory is empty.")
CTRL24.append(('all members of the 24 h cohort', bool(p24), len(p24),
               'the 24 h cohort is empty'))
CTRL24.append(('matched subjects of section 2', IDENT.get('24 h', 0) > 0,
               IDENT.get('24 h', 0), 'no paired subject is in the 24 h cohort'))
IDENT_D = {}
for va, vb in PAIRS_D:
    m = INSD[(va, vb, 'Type1 at 24 h')].set_index("PATNO")
    com = sorted(set(m.index) & p24)
    if va == "BL" and com:
        for x, y in COL24_COHORT:
            assert np.allclose(m.loc[com, f"{x}_a"].values,
                               c24.set_index("PATNO").loc[com, y].values,
                               rtol=1e-9, atol=0), \
                f'baseline {x} different from {y} in the subjects of the benchmark who are in the cohort'
    IDENT_D[(va, vb)] = len(com) if va == "BL" else 0
    CTRL24.append((f'baseline side of the comparator {va}-{vb} (6.2)',
                   IDENT_D[(va, vb)] > 0, IDENT_D[(va, vb)],
                   'the 24 h cohort contains only baseline values' if va != "BL"
                   else 'no subject of the pair is in the 24 h cohort'))
log('  24 h at a visit other than baseline: NOT verified against a cohort')
log("  file, because no cohort file contains those values.")
done = [f'{n_} ({subj_(k_)})' for n_, ok_, k_, _ in CTRL24 if ok_]
no_done = [f'{n_} ({why_})' for n_, ok_, _, why_ in CTRL24 if not ok_]
if done:
    log('  They are computed by the same function whose result at baseline coincides with')
    log('  the 24 h cohort in the checks run in this execution: '
        + "; ".join(done) + ".")
else:
    log("  The function that computes them has NOT been compared with the 24 h cohort in")
    log('  this execution: no check at baseline could be run.')
log('  Checks at baseline at 24 h NOT run: '
    + ("; ".join(no_done) if no_done else 'none') + ".")
outside = p150 - sporadic
log(f'  members of the 150 h cohort who are not sporadic in the curated data cut: {len(outside)}.')
log(f"  '{NAME_COHORT}' = members of the 150 h cohort, sporadic, with at least one")
log('  Type1 sample at 24 h at the visit indicated.')

# --- 6.1 between protocols
log("\n6.1 Between protocols: 150 h at one visit, 24 h at another")
count_between = {(va, vb): len(pairs_between(k24o, va, vb))
             for va in visits_of(k150) for vb in visits_of(k24o) if va != vb}
PAIRS = [c for c, n in count_between.items() if n >= N_MIN]
log('PD positive in both protocols (every type at 24 h), by pair of different')
log(f"visits '150 h-24 h': with at least {N_MIN} subjects "
    f"{ {f'{a}-{b}': count_between[a, b] for a, b in PAIRS}}")
log(f"  below {N_MIN} (not estimated): "
    f"{ {f'{a}-{b}': n for (a, b), n in count_between.items() if 0 < n < N_MIN}}")
if PRINC in PAIRS:
    more_num = max(PAIRS, key=lambda c: count_between[c])
    log(f"  main comparison, fixed in the code and not chosen on the basis of the estimates: 150 h "
        f"{PRINC[0]} against 24 h {PRINC[1]}, in '{NAME_COHORT}'"
        + ("; it is also the most numerous pair." if more_num == PRINC
           else f'; the most numerous pair is {more_num[0]}-{more_num[1]}.'))
else:
    log(f'  the main comparison (150 h {PRINC[0]} against 24 h {PRINC[1]}) does not reach {N_MIN} '
        f'subjects: NOT estimated.')

COL_BETWEEN = C24 + C150
COLX_F = {p: f"m24_{p}" for p in PAR}
COLY_F = {p: f"md150_{p}" for p in PAR}
NR24_F = {p: [f"nr24_{p}"] for p in PAR}
NR150_F = {p: [f"nr150_{p}"] for p in PAR}
FRAV, INSV, LASTV, ALIASV, WITHOUT_PAR = {}, {}, {}, {}, []
for va, vb in PAIRS:
    et = f'150 h {va} against 24 h {vb}'
    block = f'different visits, between protocols: {et}'
    log(f"\n[{et}]" + ('  <- main comparison' if (va, vb) == PRINC else ""))
    if va != "BL":
        log(f"  '{NAME_COHORT}' is not defined: the values of the cohort are baseline values.")
    # the benchmark within the 24 h protocol on the same two visits: present or not
    vd = tuple(sorted((va, vb), key=ord_vis))
    n_each = count_d.get(vd, 0)
    if vd in PAIRS_D:
        n_d = {nd: len(INSD[vd + (nd,)]) for nd in K_WITHIN}
        below = [nd for nd, n_ in n_d.items() if n_ < N_MIN]
        log(f'  24 h benchmark on the same two visits (24 h {vd[0]} against 24 h {vd[1]}, in 6.2): '
            + ", ".join(f"'{nd}' n = {n_}" for nd, n_ in n_d.items())
            + (f'; NOT estimated (n < {N_MIN}) in: ' + ", ".join(f"'{nd}'" for nd in below)
               if below else "") + ".")
    else:
        n_d = {nd: None for nd in K_WITHIN}
        WITHOUT_PAR.append((va, vb))
        log(f'  24 h benchmark on the same two visits (24 h {vd[0]} against 24 h {vd[1]}): MISSING. '
            f'The PD positive at 24 h at both visits are {n_each}, fewer than {N_MIN}: for this '
            f'comparison between protocols there is NO benchmark within the protocol.')
    dset_u = datasets_between(va, vb, latest=True)
    for name, m in datasets_between(va, vb).items():
        ch = (va, vb, name)
        INSV[ch], LASTV[ch] = m, dset_u[name]
        sides = [(va, 155), (vb, 237)]
        log(f'\n  {name} — {et} (n = {len(m)}, of which sporadic {int(m.PATNO.isin(sporadic).sum())})')
        nd = COMPARISON_OF[name]
        record_v(block, LBLS_BETWEEN[name], 'comparator',
                   dict(n=len(m), rho=float(n_d[nd]) if n_d[nd] is not None else float(n_each),
                        lo=np.nan, hi=np.nan),
                   variant=(f"24 h {vd[0]} against 24 h {vd[1]}, '{nd}': "
                             + ('estimated in 6.2' if n_d[nd] is not None and n_d[nd] >= N_MIN
                                else f'NOT estimated, below N_MIN = {N_MIN}'
                                if n_d[nd] is not None
                                else f'ABSENT: the positives of every type at both visits are below '
                                     f'N_MIN = {N_MIN}')),
                   quantity='subjects of the 24 h comparator on the same two visits (n = subjects '
                             'of the comparison between protocols)',
                   units='subjects')
        if len(m) < N_MIN:
            log(f'    not estimated (n < {N_MIN})')
            for par in PAR:
                FRAV[ch + (par,)] = estimate(m[f"m24_{par}"], m[f"md150_{par}"], "")
                record_v(block, LBLS_BETWEEN[name], par, FRAV[ch + (par,)],
                           'mean/median', subj=m.PATNO)
            continue
        twin = next((g for g in INSV if g[:2] == (va, vb) and g != ch
                        and g not in ALIASV and len(INSV[g]) >= N_MIN
                        and same_values(INSV[g], m, COL_BETWEEN)), None)
        if twin:
            ALIASV[ch] = twin
            log(f"    coincides, subject by subject, with '{twin[2]}': times and estimates are the "
                f"same and are not repeated.")
        else:
            log(f'    subgroups: {composition(m.PATNO)}')
            log(f'    type of all the positive samples at 24 h of visit {vb}: '
                f'{pd.Series(kinds_of(m.PATNO, m.CLINICAL_EVENT_24)).value_counts().to_dict()}')
            log(f'    in the 150 h cohort: {int(m.PATNO.isin(p150).sum())}; in the 24 h cohort: '
                f'{int(m.PATNO.isin(p24).sum())}')
            n_par = without_ties(m, sides, f"'{name}', {et}")
            log(f"    with more than one positive sample at the visit: {int((m.n_pos_24 > 1).sum())} "
                f"at 24 h, {int((m.n_pos_150 > 1).sum())} at 150 h; units with two rows on the last "
                f"run date: {n_par}")
            describe_times(m, "150", "24", f'150 h {va}', f'24 h {vb}', block,
                           LBLS_BETWEEN[name])
        for par in PAR:
            if ch in ALIASV:
                s = FRAV[ALIASV[ch] + (par,)]
            else:
                s = estimate(m[f"m24_{par}"], m[f"md150_{par}"], f'vd|between|{va}|{vb}|{name}|{par}')
                log(f'    {par:<9} {row_v(s)}')
            FRAV[ch + (par,)] = s
            record_v(block, LBLS_BETWEEN[name], par, s, 'mean/median',
                       coincides_having=ALIASV[ch][2] if ch in ALIASV else "", subj=m.PATNO)
        princ_ = {par: FRAV[ch + (par,)] for par in PAR}
        if ch in ALIASV and same_values(LASTV[ALIASV[ch]], LASTV[ch], COL_BETWEEN):
            # with the last-run rule too the set coincides with its twin
            g_ = next(r for r in ULT_REASS if r['key'] == ALIASV[ch])
            for par in PAR:
                record_v(block, LBLS_BETWEEN[name], par, g_['estimates'][par],
                           'last run' + ("" if not g_['lost'] and not g_['added']
                                             else ', different subjects'),
                           coincides_having=ALIASV[ch][2], subj=LASTV[ch].PATNO)
            continue
        if ch in ALIASV:
            log("    with the last-run rule it NO longer coincides with the twin set: estimated "
                "separately.")
        else:
            replicates_incomplete(m, NR24_F, NR150_F, COLX_F, COLY_F,
                                f'vd|complete|between|{va}|{vb}|{name}', block, LBLS_BETWEEN[name],
                                'mean/median', princ_)
        having_latest_run(m, LASTV[ch], COLX_F, COLY_F, f'vd|last|between|{va}|{vb}|{name}',
                         block, LBLS_BETWEEN[name], princ_, sides, name != 'everyone', ch)

R_BETWEEN = list(ULT_REASS)
summary_latest(R_BETWEEN, "6.1")
log('Comparisons between protocols without a 24 h benchmark on the same two visits: ' + (", ".join(f'150 h {a} against 24 h {b}' for a, b in WITHOUT_PAR)
                  if WITHOUT_PAR else 'none')
    + f' (out of {len(PAIRS)} pairs of visits estimated).')

# sensitivity analyses on the main comparison
chp = PRINC + (NAME_COHORT,)
SENSV = {}
mp = INSV.get(chp)
princ_ok = mp is not None and len(mp) >= N_MIN
if princ_ok:
    va, vb = PRINC
    et = f'150 h {va} against 24 h {vb}'
    block = f'different visits, between protocols: {et}'
    members = p150 & sporadic
    log(f"\nSensitivity on the main comparison: '{NAME_COHORT}', {et} (n = {len(mp)})")
    log('  (a) mean at 150 h in place of the median')
    for par in PAR:
        s = estimate(mp[f"m24_{par}"], mp[f"m150_{par}"], f'vd|mean-mean|{va}|{vb}|{par}')
        SENSV[('mean at 150 h', par)] = s
        record_v(block, LBLS_BETWEEN[NAME_COHORT], par, s, 'mean/mean', subj=mp.PATNO)
        log(f'    {par:<9} {row_v(s)}')
    r_p = next(r for r in ULT_REASS if r['key'] == ALIASV.get(chp, chp))
    log(f"  (b) last-run rule: reported in 6.1 next to each set. In the main comparison: n = "
        f"{r_p['n1']} (excluded {r_p['lost']}, added {r_p['added']}, remaining with a different "
        f"value {r_p['different']}); "
        + ('it coincides with the main estimate, value by value.' if r_p['equals']
           else 'it does NOT coincide with the main estimate.'))
    for par in PAR:
        SENSV[('last run', par)] = r_p['estimates'][par]
    log(f'  (c) resampling the run dates instead of the subjects, {NBOOT_DATE_V} resamples (in')
    log(f'      section 2 they are {NBOOT_DATE}). Same reservations as in section 2: the two factors '
        f'are crossed,')
    log('      no interval considers both')
    for side, col in (('dates at 24 h', 'run_24'), ('dates at 150 h', 'run_150')):
        for par in PAR:
            s = estimate_by_date(mp[f"m24_{par}"].values, mp[f"md150_{par}"].values,
                               mp[col].values, f'vd|date|{side}|{va}|{vb}|{par}',
                               nboot=NBOOT_DATE_V)
            SENSV[(f'resampling the {side}', par)] = s
            record_v(block, LBLS_BETWEEN[NAME_COHORT], par, s,
                       f"resampling {s['n_date']} {side}; {NBOOT_DATE_V} resamples",
                       subj=mp.PATNO)
            log(f"    {side:<15} ({s['n_date']:>2})  {par:<9} {row_v(s)}")
    log('  (d) after centring each parameter on its own run date, as in')
    log('      section 5(c): logarithmic scale, run means over all the positives of the')
    log('      protocol')
    mc = pairs_between(kinetics(237, only_type1=True, columns=[f"c_m24_{p}" for p in PAR]),
                    va, vb, kinetics(155, columns=[f"c_md150_{p}" for p in PAR]))
    mc = mc[mc.PATNO.isin(members)]
    # CODE CONSISTENCY check, not a check on the data: the two sets come from the
    # same row filters and no data can make them differ.
    assert set(mc.PATNO) == set(mp.PATNO), \
        'centred set different from the main one (consistency of the code)'
    # Guard on the DATA: the logarithm is not defined for a value <= 0, and whoever
    # has one would drop silently out of the centred estimate only. For each
    # parameter the participants with the centred pair defined must be those of the
    # main estimate.
    a_p, a_c = mp.set_index("PATNO"), mc.set_index("PATNO")
    for par in PAR:
        ok_p = set(a_p.index[np.isfinite(a_p[f"m24_{par}"]) & np.isfinite(a_p[f"md150_{par}"])])
        ok_c = set(a_c.index[np.isfinite(a_c[f"c_m24_{par}"])
                             & np.isfinite(a_c[f"c_md150_{par}"])])
        assert ok_c == ok_p, (
            f'{par}: subjects of the main comparison with a value <= 0 or missing on the logarithmic '
            f'scale, who would drop out of the centred estimate only: {len(ok_p - ok_c)} (the '
            f'opposite case: {len(ok_c - ok_p)})')
    for par in PAR:
        s = estimate(mc[f"c_m24_{par}"], mc[f"c_md150_{par}"], f"vd|centrato|{va}|{vb}|{par}")
        SENSV[('centred on the run date', par)] = s
        record_v(block, LBLS_BETWEEN[NAME_COHORT], par, s, 'centred on the run date',
                   quantity='rho of the values centred on the run date', subj=mc.PATNO)
        log(f'    {par:<9} {row_v(s)}')

# --- 6.2 benchmark on the same visits
log('\n6.2 Benchmark: the same protocol on the same two visits')
log('24 h protocol, PD, earlier visit against later visit (order of the')
log('visit codes). In the Type1 sets the subject must have a Type1 sample at')
log('BOTH visits. How many subjects the benchmark has in common with the comparison')
log("between protocols is reported further below, with the difference.")
log(f"pairs of visits at 24 h with at least {N_MIN} PD positive (every type) at both: "
    f"{ {f'{a}-{b}': count_d[a, b] for a, b in PAIRS_D}}")
log(f"  below {N_MIN} (not estimated): "
    f"{ {f'{a}-{b}': n for (a, b), n in count_d.items() if 0 < n < N_MIN}}")
COLX_D = {p: f"m24_{p}_a" for p in PAR}
COLY_D = {p: f"m24_{p}_b" for p in PAR}
NR24_D = {p: [f"nr24_{p}_a", f"nr24_{p}_b"] for p in PAR}
NR150_D = {p: [] for p in PAR}
DENV = {}
n_earliest_d = len(ULT_REASS)
for va, vb in PAIRS_D:
    et = f'24 h {va} against 24 h {vb}'
    block = f'different visits, within the 24 h: {et}'
    log(f"\n[{et}]")
    for name in K_WITHIN:
        ch = (va, vb, name)
        m = INSD[ch]
        sides = [(va, 237), (vb, 237)]
        log(f'\n  {name} — {et} (n = {len(m)}, of which sporadic {int(m.PATNO.isin(sporadic).sum())})')
        if len(m) >= N_MIN:
            log(f'    subgroups: {composition(m.PATNO)}')
            ta_ = pd.Series(kinds_of(m.PATNO, m.CLINICAL_EVENT_a))
            tb_ = pd.Series(kinds_of(m.PATNO, m.CLINICAL_EVENT_b))
            log(f"    type of all the positive samples at the two visits: "
                f"{(ta_ + ' / ' + tb_).value_counts().to_dict()}")
            dg = (m.run_b - m.run_a).dt.days
            without_ties(m, sides, f"'{name}', {et}")
            log(f"    in the 24 h cohort: {int(m.PATNO.isin(p24).sum())}; in the 150 h cohort: "
                f"{int(m.PATNO.isin(p150).sum())}; pairs in which the later visit was run FIRST: "
                f"{int((dg < 0).sum())}")
            describe_times(m, "a", "b", f'24 h {va}', f'24 h {vb}', block,
                           LBLS_WITHIN[name])
        else:
            log(f'    not estimated (n < {N_MIN})')
        for par in PAR:
            s = estimate(m[f"m24_{par}_a"], m[f"m24_{par}_b"],
                      f'vd|inside24|{va}|{vb}|{name}|{par}')
            DENV[ch + (par,)] = s
            record_v(block, LBLS_WITHIN[name], par, s, 'mean/mean', subj=m.PATNO)
            if len(m) >= N_MIN:
                log(f'    {par:<9} {row_v(s)}')
        if len(m) >= N_MIN:
            princ_ = {par: DENV[ch + (par,)] for par in PAR}
            replicates_incomplete(m, NR24_D, NR150_D, COLX_D, COLY_D,
                                f'vd|complete|inside24|{va}|{vb}|{name}', block,
                                LBLS_WITHIN[name], 'mean/mean', princ_)
            having_latest_run(m, ULTD[ch], COLX_D, COLY_D,
                             f'vd|last|inside24|{va}|{vb}|{name}', block,
                             LBLS_WITHIN[name], princ_, sides, name != 'everyone', ch)
    # identity of the baseline side with the 24 h cohort: performed in 6.0
    if va != "BL":
        log(f"\n  Identity check (24 h {va}): NOT verified, the 24 h cohort contains baseline values "
            f"only.")
    elif not IDENT_D[(va, vb)]:
        log(f"\n  Identity check (24 h {va}): NOT verified, no subject of the pair is in the 24 h "
            f"cohort.")
    else:
        log(f'\n  Identity check (24 h {va}): in {IDENT_D[va, vb]} subjects of the benchmark who are '
            f'in the 24 h cohort, the {len(COL24_COHORT)} baseline values coincide with those saved.')
summary_latest(ULT_REASS[n_earliest_d:], "6.2")

log('\n150 h protocol repeated at two visits (PD positive at both):')
vis150 = [v for v in visits_of(k150) if np.isfinite(ordering_visit(v))]
k150m = k150[[f"md150_{p}" for p in PAR] + ['run']]
count_150 = {(va, vb): len(pairs_within(k150m, va, vb))
             for i_, va in enumerate(vis150) for vb in vis150[i_ + 1:]}
PAIRS_150 = [c for c, n in count_150.items() if n >= N_MIN]
log(f"  subjects by pair of visits: { {f'{a}-{b}': n for (a, b), n in count_150.items() if n > 0}}")
if not PAIRS_150:
    log(f'  No pair of visits reaches {N_MIN} subjects (maximum '
        f'{(max(count_150.values()) if count_150 else 0)}): the benchmark within the 150 h is NOT '
        f'estimable.')
for va, vb in PAIRS_150:
    m = pairs_within(k150m, va, vb)
    et = f'150 h {va} versus 150 h {vb}'
    log(f'  {et} (n = {len(m)}, of which sporadic {int(m.PATNO.isin(sporadic).sum())}; same run day: '
        f'{int((m.run_a == m.run_b).sum())})')
    for par in PAR:
        s = estimate(m[f"md150_{par}_a"], m[f"md150_{par}_b"], f"vd|dentro150|{va}|{vb}|{par}")
        record_v(f'different visits, within the 150 h: {et}',
                   'PD positive at 150 h at both visits, every genotype', par, s,
                   'median/median', subj=m.PATNO)
        log(f'    {par:<9} {row_v(s)}')

log('\nDifference: rho within the 24 h minus rho between protocols, on the same two visits.')
log('Where the subjects with all three measures (150 h at the first visit, 24 h at')
log(f"both) are at least {N_MIN}, the difference is computed on those subjects only")
log('(paired bootstrap). Otherwise the two intervals are set side by side as estimates on')
log("DIFFERENT PARTICIPANTS, and the difference is given by resampling the union of the two")
log('sets (as in section 3: those who are in both weigh on both). The reservations')
log('of section 3 hold: one side contains a median at 150 h, whose')
log("repeatability is in 6.3.")
log("In the table, for the difference on the union, n is the number of distinct subjects")
log('who enter at least one of the two correlations (n_dentro + n_fra - n_common); the')
log("three addends are in the column 'variante'.")
log('Matching on times: for each pair of sets, median in the benchmark against')
log("median between protocols of three times; 'appaiato' = the two interquartile ranges")
log("overlap. The sign is that of 'seconda meta' meno prima'.")
PAIR = {'everyone': ['everyone'], 'Type1 at 24 h': ['Type1 at 24 h'],
          'sporadic Type1': ['sporadic Type1', NAME_COHORT]}
DIFFV, APPV = {}, {}
for va, vb in PAIRS_D:
    if (va, vb) not in PAIRS:
        log(f'\n  {va}-{vb}: no comparison between protocols on these two visits reaches {N_MIN} '
            f'subjects; difference not computed.')
        continue
    for name_d, lst in PAIR.items():
        bm = INSD[(va, vb, name_d)]
        for name_f in lst:
            chf = (va, vb, name_f)
            if chf not in INSV:
                continue
            cm = INSV[chf]
            et = (f"within: 24 h {va} against 24 h {vb}, '{name_d}' (n = {len(bm)}); between: 150 h "
                  f"{va} against 24 h {vb}, '{name_f}' (n = {len(cm)})")
            block = (f'different visits, difference: within the 24 h ({va} against {vb}) minus '
                      f'between protocols (150 h {va} against 24 h {vb})')
            dset_csv = f'within: {LBLS_WITHIN[name_d]} / between: {LBLS_BETWEEN[name_f]}'
            if len(bm) < N_MIN or len(cm) < N_MIN:
                log(f"\n  {et}: one set is below {N_MIN}, difference not computed.")
                continue
            com = sorted(set(bm.PATNO) & set(cm.PATNO))
            an = sorted(set(bm.PATNO) | set(cm.PATNO))
            log(f"\n  {et}")
            log(f'    subjects in common (they have all three measures): {len(com)}; in the union: '
                f'{len(an)}')
            app = pairing(times(bm, "a", "b", f'24 h {va}', f'24 h {vb}'),
                              times(cm, "150", "24", f'150 h {va}', f'24 h {vb}'))
            APPV[(va, vb, name_d, name_f)] = app
            log('    matching on times, benchmark against between protocols: '
                + "; ".join(text_app(a_) for a_ in app) + ".")
            for a_ in app:
                record_v(block, dset_csv, 'matching on times',
                           dict(n=len(an), rho=np.nan if a_["app"] is None else float(a_["app"]),
                                lo=a_.get("par", np.nan), hi=a_.get('between', np.nan)),
                           variant='ci_inf = median in the comparator, ci_upp = median between '
                                    'protocols; n = subjects in the union',
                           quantity=f"{a_['name']}: 1 = matched (overlapping interquartile ranges), "
                                     f"0 = not matched",
                           units='code', subj=an)
            paired = len(com) >= N_MIN
            if paired:
                log(f'    SAME PARTICIPANTS (n = {len(com)}): the two correlations and the '
                    f'difference on the subjects in common only')
                bi, ci = bm.set_index("PATNO").loc[com], cm.set_index("PATNO").loc[com]
                for par in PAR:
                    sd = estimate(bi[f"m24_{par}_a"], bi[f"m24_{par}_b"],
                               f'vd|common|within|{va}|{vb}|{name_d}|{name_f}|{par}')
                    sf = estimate(ci[f"m24_{par}"], ci[f"md150_{par}"],
                               f'vd|common|between|{va}|{vb}|{name_d}|{name_f}|{par}')
                    s = difference_paired(
                        bi[f"m24_{par}_a"].values, bi[f"m24_{par}_b"].values,
                        ci[f"m24_{par}"].values, ci[f"md150_{par}"].values,
                        f'vd|difference|common|{va}|{vb}|{name_d}|{name_f}|{par}')
                    for gr, s_ in (('rho within the 24 h', sd), ('rho between protocols', sf)):
                        record_v(block, dset_csv, par, s_, 'same participants',
                                   quantity=gr, subj=com)
                    record_v(block, dset_csv, par, dict(s, n=s['n_union']),
                               f"same participants, paired bootstrap; n = distinct subjects in at "
                               f"least one of the two correlations; n_dentro = {s['n_a']}; n_fra = "
                               f"{s['n_b']}; n_common = {s['n_common']}",
                               quantity='difference in rho', subj=com)
                    log(f"    {par:<9} within {sd['rho']:+.3f} [{sd['lo']:+.3f}; {sd['hi']:+.3f}]; "
                        f"between {sf['rho']:+.3f} [{sf['lo']:+.3f}; {sf['hi']:+.3f}]; "
                        f"{row_v(s, 'difference')}")
            else:
                log(f"    DIFFERENT PARTICIPANTS: the subjects in common are fewer than {N_MIN}, the "
                    f"paired comparison on the same subjects is not estimable.")
            log("    The two intervals side by side and the difference resampling the union:")
            bi, ci = bm.set_index("PATNO").reindex(an), cm.set_index("PATNO").reindex(an)
            for par in PAR:
                sd, sf = DENV[(va, vb, name_d, par)], FRAV[chf + (par,)]
                s = difference_paired(
                    bi[f"m24_{par}_a"].values, bi[f"m24_{par}_b"].values,
                    ci[f"m24_{par}"].values, ci[f"md150_{par}"].values,
                    f'vd|difference|union|{va}|{vb}|{name_d}|{name_f}|{par}')
                DIFFV[(va, vb, name_d, name_f, par)] = s
                record_v(block, dset_csv, par, dict(s, n=s['n_union']),
                           f"bootstrap on the union; n = distinct subjects in at least one of the "
                           f"two correlations; n_dentro = {s['n_a']}; n_fra = {s['n_b']}; n_common = "
                           f"{s['n_common']}",
                           quantity='difference in rho', subj=an)
                log(f"    {par:<9} within (n = {sd['n']}) {sd['rho']:+.3f} [{sd['lo']:+.3f}; "
                    f"{sd['hi']:+.3f}]; between (n = {sf['n']}) {sf['rho']:+.3f} [{sf['lo']:+.3f}; "
                    f"{sf['hi']:+.3f}]; {row_v(s, 'difference')}"
                    + (f"  (resamples discarded: {100 * s['discarded']:.2f}%)"
                       if s.get('discarded', 0) > 0 else ""))

# --- 6.3 attenuation ceiling
log('\n6.3 Estimate of the attenuation ceiling from the triplicates, for each comparison')
log('Same computation and same reservations as in section 4: one-way ICC on the logarithmic')
log('scale, on the positive sample run last at the visit indicated (Type1 at')
log('24 h in the Type1 sets); ceiling = square root of the product of the two ICC(1,3). It says')
log("how much the repeatability within the run alone limits the correlation; the")
log('change between two visits and the variance between runs do not enter it and, alone,')
log("would bring the true ceiling lower.")


def ceiling_fr(ta, tb, label):
    """The calculation of section 4 on two tables of triplets (log) per participant."""
    tb = tb.reindex(ta.index)
    ok = ta.notna().all(axis=1) & tb.notna().all(axis=1)
    ta, tb = ta[ok].values, tb[ok].values
    n = len(ta)
    if n < N_MIN:
        return dict(n=n, computed=False, point=np.nan, i1b=np.nan)
    mua, wa = ta.mean(axis=1), ((ta - ta.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    mub, wb = tb.mean(axis=1), ((tb - tb.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    (i1a, i3a), (i1b, i3b) = icc_fr(mua, wa), icc_fr(mub, wb)
    idx = _gen(label).integers(0, n, (NBOOT, n))
    ba, bb = icc_fr(mua[idx], wa[idx])[1], icc_fr(mub[idx], wb[idx])[1]
    def_ = (ba > 0) & (bb > 0)
    bs = np.where(def_, np.sqrt(np.where(def_, ba * bb, 0.0)), 0.0)
    estimable = bool(i3a > 0 and i3b > 0)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=n, computed=True, i1a=float(i1a), i3a=float(i3a), i1b=float(i1b),
                i3b=float(i3b), estimable=estimable,
                point=float(np.sqrt(i3a * i3b)) if estimable else np.nan,
                lo=float(lo), hi=float(hi), nondef=float(1 - def_.mean()),
                mc=error_mc(bs))


# The function must be the same calculation as section 4: it is tested on the
# sets of section 4, with the same labels, and must return the same values.
n_trial = 0
for name, m in DATASETS.items():
    if name in ALIAS:
        continue
    for par, (a, b) in PAR.items():
        t = ceiling_fr(triplicates(237, a, set(m.PATNO), name != 'everyone'),
                     triplicates(155, b, set(m.PATNO), False), f'ceiling|{name}|{par}')
        assert np.isclose(t['point'], CEILING[(name, par)], rtol=1e-12, atol=0,
                          equal_nan=True) \
            and np.isclose(t["i1b"], ICC11_150[(name, par)], rtol=1e-12, atol=0,
                           equal_nan=True), \
            f'ceiling_fr does not reproduce section 4 for {name}, {par}'
        n_trial += 1
log('Check: the function used here, applied to the sets of section 4 with the')
log(f'same labels, reproduces their ceiling and ICC(1,1) at 150 h in all the {n_trial} cases')
log('tried (set x parameter).')

CEILINGV, CEILINGD = {}, {}


def estimate_ceiling(t):
    """The ceiling in the form of the other estimates, for the rule on limits close to zero."""
    return dict(n=t["n"], rho=t['point'], lo=t["lo"] if t['estimable'] else np.nan,
                hi=t["hi"] if t['estimable'] else np.nan, mc=t["mc"])


def printout_ceiling(par, t, na, nb, obsv, block, dataset, subj):
    if not t['computed']:
        log(f"    {par:<9} not estimated (n = {t['n']} < {N_MIN})")
        return
    text = (f"ceiling = {t['point']:.2f} [{t['lo']:.2f}; {t['hi']:.2f}]" if t['estimable']
             else 'ICC(1,3) estimated <= 0: ceiling not estimable in this set')
    log(f"    {par:<9} ICC(1,1)/ICC(1,3): {na} {t['i1a']:+.3f}/{t['i3a']:+.3f}, {nb} "
        f"{t['i1b']:+.3f}/{t['i3b']:+.3f} (n = {t['n']}); {text}; not defined in "
        f"{100 * t['nondef']:.0f}% of the resamples; observed {obsv['rho']:+.3f} [{obsv['lo']:+.3f}; "
        f"{obsv['hi']:+.3f}]")
    empty = dict(lo=np.nan, hi=np.nan, n=t["n"])
    for gr, val in ((f"ICC(1,1), {na}", t["i1a"]), (f"ICC(1,3), {na}", t["i3a"]),
                    (f"ICC(1,1), {nb}", t["i1b"]), (f"ICC(1,3), {nb}", t["i3b"])):
        record_v(block, dataset, par, dict(empty, rho=val), quantity=gr, subj=subj)
    record_v(block, dataset, par, estimate_ceiling(t),
               variant=f"not defined in the {100 * t['nondef']:.1f}% of the resamples",
               quantity='ceiling', subj=subj)


for va, vb in PAIRS:
    na, nb = f'24 h {vb}', f'150 h {va}'
    block = f'different visits, attenuation ceiling: 150 h {va} against 24 h {vb}'
    log(f'\n[between protocols: 150 h {va} against 24 h {vb}]')
    for name in ('everyone', 'Type1 at 24 h', 'sporadic Type1', NAME_COHORT):
        ch = (va, vb, name)
        if ch not in INSV:
            continue
        m = INSV[ch]
        if ch in ALIASV:
            log(f"  {name}: coincides with '{ALIASV[ch][2]}'")
            for par in PAR:
                CEILINGV[ch + (par,)] = CEILINGV[ALIASV[ch] + (par,)]
            continue
        log(f'  {name} (n = {len(m)})')
        if len(m) < N_MIN:
            log(f'    not estimated (n < {N_MIN})')
            continue
        for par, (a, b) in PAR.items():
            t = ceiling_fr(triplicates(237, a, set(m.PATNO), name != 'everyone', visit=vb),
                         triplicates(155, b, set(m.PATNO), False, visit=va),
                         f'vd|ceiling|between|{va}|{vb}|{name}|{par}')
            CEILINGV[ch + (par,)] = t
            printout_ceiling(par, t, na, nb, FRAV[ch + (par,)], block, LBLS_BETWEEN[name],
                         m.PATNO)
for va, vb in PAIRS_D:
    na, nb = f'24 h {va}', f'24 h {vb}'
    block = f'different visits, attenuation ceiling: 24 h {va} against 24 h {vb}'
    log(f'\n[within the 24 h: {va} against {vb}]')
    for name in K_WITHIN:
        m = INSD[(va, vb, name)]
        log(f'  {name} (n = {len(m)})')
        if len(m) < N_MIN:
            log(f'    not estimated (n < {N_MIN})')
            continue
        for par, (a, b) in PAR.items():
            t = ceiling_fr(triplicates(237, a, set(m.PATNO), name != 'everyone', visit=va),
                         triplicates(237, a, set(m.PATNO), name != 'everyone', visit=vb),
                         f'vd|ceiling|within|{va}|{vb}|{name}|{par}')
            CEILINGD[(va, vb, name, par)] = t
            printout_ceiling(par, t, na, nb, DENV[(va, vb, name, par)], block,
                         LBLS_WITHIN[name], m.PATNO)
log()
for et_, tt_, side in (('between protocols', CEILINGV, "i3b"), ('within the 24 h', CEILINGD, None)):
    nst = sorted({f"{k[3]} in '{k[2]}' ({k[0]}-{k[1]})" for k, t in tt_.items()
                  if t['computed'] and not t['estimable']})
    log(f"  Comparisons {et_} with an estimated ICC(1,3) <= 0, in which the ceiling is not "
        f"estimable: " + ("; ".join(nst) if nst else 'none') + ".")
    if side:
        neg150 = sorted({f"{k[3]} in '{k[2]}' ({k[0]}-{k[1]})" for k, t in tt_.items()
                         if t['computed'] and t[side] <= 0})
        log("  Of these, those in which it is the ICC(1,3) at 150 h that is <= 0: "
            + ("; ".join(neg150) if neg150 else 'none') + ".")
log('  A ceiling in no case explains a correlation of negative sign.')

# --- 6.4 the type at 24 h in the members of the 150 h cohort
log('\n6.4 Outcome and type at 24 h in the members of the 150 h cohort, by visit')
log("Counts only. Unit: subject x visit; the sample at 24 h run last at the")
log("visit counts. The 150 h does not assign a type, so the 150 h cohort is")
log('made of positives of unknown type; the Type1 sets of this section include those who')
log('have at least one Type1 sample at the visit (column on the right).')
CAT = ['positive Type1', 'positive Type2', 'positive Undetermined', 'negative',
       'inconclusive']
u24c = vis[(vis.PROJECTID == 237) & vis.PATNO.isin(p150)].copy()
u24c["cat"] = np.select(
    [(u24c.outcome == "Positive") & (u24c.kind == "Type1"),
     (u24c.outcome == "Positive") & (u24c.kind == "Type2"),
     (u24c.outcome == "Positive") & (u24c.kind == "Undetermined"),
     u24c.outcome == "Negative", u24c.outcome == "Inconclusive"], CAT, default='other')
assert not (u24c.cat == 'other').any(), 'outcome or type at 24 h outside the expected categories'
rows24 = d[(d.PROJECTID == 237) & d.PATNO.isin(p150)]
without24 = p150 - set(u24c.PATNO)
log(f'members of the 150 h cohort: {len(p150)}; with at least one result at 24 h: '
    f'{len(p150 - without24)}; without: {len(without24)}. Rows at 24 h: {len(rows24)}, in '
    f'{len(u24c)} subject x visit units.')
if not without24:
    log('All the members of the 150 h cohort therefore have at least one result at 24 h.')
ties_m = pp_[(pp_.PROJECTID == 237) & pp_.PATNO.isin(p150)]
log(f'Units at 24 h of members of the cohort with two rows on the last run date: {len(ties_m)}' + ("." if not len(ties_m) else
                        ' (same outcome and type in the two rows: the script stops if they are '
                        'not).'))
t1p = k24t.reset_index()
t1p = t1p[t1p.PATNO.isin(p150)]
rows4 = []
for v_ in sorted(u24c.CLINICAL_EVENT.unique(), key=ord_vis):
    uv = u24c[u24c.CLINICAL_EVENT == v_]
    groups = {c_: set(uv[uv.cat == c_].PATNO) for c_ in CAT}
    groups['at least one Type1'] = set(t1p[t1p.CLINICAL_EVENT == v_].PATNO)
    r_ = {'visit at 24 h': v_ + (' (same visit as the 150 h)' if v_ == "BL" else ""),
          'subjects': len(uv)}
    r_.update({c_: len(groups[c_]) for c_ in CAT})
    r_['more than one row'] = int((uv.n_samp > 1).sum())
    r_['at least one Type1'] = len(groups['at least one Type1'])
    rows4.append(r_)
    for c_, g_ in groups.items():
        record_v('different visits, 24 h outcome in the 150 h cohort',
                   '150 h cohort: all members with a 24 h result at the visit',
                   c_, dict(n=len(uv), rho=float(len(g_)), lo=np.nan, hi=np.nan),
                   variant=f'24 h {v_}; ' + ('sample run last' if c_ in CAT
                                                else 'every sample of the visit'),
                   quantity='number of subjects (n = subjects with a 24 h at the visit)',
                   units='subjects', subj=g_)
log(pd.DataFrame(rows4).set_index('visit at 24 h').to_string())
if PRINC[1] in set(u24c.CLINICAL_EVENT):
    r_ = u24c[u24c.CLINICAL_EVENT == PRINC[1]].cat.value_counts().reindex(CAT, fill_value=0)
    n_v = int(r_.sum())
    log(f'At the visit of the main comparison ({PRINC[1]}): {share(int(r_[CAT[0]]), n_v)} Type1 at '
        f'the sample run last; not Type1 {n_v - int(r_[CAT[0]])}, of which positives of another type '
        f'{int(r_[CAT[1]] + r_[CAT[2]])} and non-positives {int(r_[CAT[3]] + r_[CAT[4]])}.')
    if princ_ok:
        log(f"The main comparison includes {len(mp)} subjects (at least one Type1 sample at "
            f"{PRINC[1]}, sporadic, members of the cohort); 'tutti' includes the positives of every "
            f"type.")

# --- 6.5 comparison with the same visit
log('\n6.5 The two pieces of evidence compared: same visit (section 2) and different visits')
log("By parameter: 'concordano' = estimate of the same sign and overlapping intervals;")
log("'compatibili' = overlapping intervals but estimates of different sign; 'discordano' =")
log('disjoint intervals. The sign of an estimate whose interval includes zero')
log("is not determined: the label only says where the estimate falls. Next to it, the")
log('difference between the two correlations (different visits minus same visit), with')
log("bootstrap interval on the union of the two sets of subjects.")


def verdict(s0, s1):
    if not (np.isfinite(s0["rho"]) and np.isfinite(s1["rho"])):
        return 'not comparable'
    if max(s0["lo"], s1["lo"]) > min(s0["hi"], s1["hi"]):
        return 'disagree'
    return 'agree' if s0["rho"] * s1["rho"] > 0 else 'compatible'


CODE = {'agree': 1.0, 'compatible': 0.0, 'disagree': -1.0,
          'not comparable': np.nan}
AGREEMENT, DIFF0, COM0 = {}, {}, {}
if PRINC in PAIRS:
    for name1, name0 in (('everyone', 'everyone'), ('Type1 at 24 h', 'Type1 at 24 h'),
                         ('sporadic Type1', 'sporadic Type1'),
                         (NAME_COHORT, 'sporadic Type1')):
        ch = PRINC + (name1,)
        if ch not in INSV:
            continue
        m0, m1 = DATASETS[name0], INSV[ch]
        an = sorted(set(m0.PATNO) | set(m1.PATNO))
        a0, a1 = m0.set_index("PATNO").reindex(an), m1.set_index("PATNO").reindex(an)
        in_com = len(set(m0.PATNO) & set(m1.PATNO))
        COM0[name1] = in_com
        log(f"\n  same visit: '{name0}', baseline (n = {len(m0)}); different visits: '{name1}', 150 "
            f"h {PRINC[0]} against 24 h {PRINC[1]} (n = {len(m1)}); subjects in common between the "
            f"two sets: {in_com}")
        for par in PAR:
            s0, s1 = BETWEEN[(name0, par)], FRAV[ch + (par,)]
            v = verdict(s0, s1)
            AGREEMENT[(name1, par)] = v
            zero = [n_ for n_, s_ in (('same visit', s0), ('different visits', s1))
                    if np.isfinite(s_["rho"]) and not excludes_zero(s_)]
            sdif = difference_paired(
                a1[f"m24_{par}"].values, a1[f"md150_{par}"].values,
                a0[f"m24_{par}"].values, a0[f"md150_{par}"].values,
                f'vd|accordo|{name1}|{name0}|{par}')
            DIFF0[(name1, par)] = sdif
            log(f"    {par:<9} same visit {s0['rho']:+.3f} [{s0['lo']:+.3f}; {s0['hi']:+.3f}]; "
                f"different visits {s1['rho']:+.3f} [{s1['lo']:+.3f}; {s1['hi']:+.3f}]: {v}"
                + (f" (interval that does not exclude zero: {', '.join(zero)})"
                   if zero and v != 'not comparable' else "")
                + f"; {row_v(sdif, 'difference')}")
            dset_csv = (f'different visits: {LBLS_BETWEEN[name1]} / same visit (baseline): '
                       f'{LBLS_BETWEEN[name0]}')
            record_v('different visits, agreement with the same visit', dset_csv, par,
                       dict(sdif, n=sdif['n_union']),
                       variant=f"150 h {PRINC[0]} against 24 h {PRINC[1]} minus baseline; bootstrap "
                                f"on the union; n = distinct subjects in at least one of the two "
                                f"correlations; n_visite_diverse = {sdif['n_a']}; n_stessa_visita = "
                                f"{sdif['n_b']}; n_common = {sdif['n_common']}",
                       quantity='difference in rho', subj=an)
            record_v('different visits, agreement with the same visit', dset_csv, par,
                       dict(n=s1["n"], rho=CODE[v], lo=np.nan, hi=np.nan),
                       variant=f"150 h {PRINC[0]} against 24 h {PRINC[1]} (n = {s1['n']}), compared "
                                f"with the baseline (n = {s0['n']})",
                       units='code',
                       quantity='1 = agree (same sign, overlapping intervals); 0 = compatible '
                                 '(overlapping, different sign); -1 = disagree (disjoint)',
                       subj=m1.PATNO)

# --- 6.6 the TTT at the same visit and at different visits
log('\n6.6 TTT: the correlation between protocols at the same visit and at different visits, side by side')
if not princ_ok:
    log("Cannot be formulated: the main comparison is not estimated.")
else:
    va, vb = PRINC
    PAR_K = "TTT"
    m0 = DATASETS['sporadic Type1'].copy()
    m0["CLINICAL_EVENT_150"], m0["CLINICAL_EVENT_24"] = "BL", "BL"
    s0k, s1k = BETWEEN[('sporadic Type1', PAR_K)], FRAV[chp + (PAR_K,)]
    sdk = DIFF0[(NAME_COHORT, PAR_K)]
    p0_, p1_ = set(m0.PATNO), set(mp.PATNO)
    log(f"Numbers side by side only, without explanation. Same visit: 150 h BL against 24 h BL, "
        f"'sporadici Type1' (n = {len(m0)}).")
    log(f"Different visits: 150 h {va} against 24 h {vb}, '{NAME_COHORT}' (n = {len(mp)}). Unit: "
        f"Spearman rho, dimensionless.")
    log(f'  {PAR_K}, same visit:       {row_v(s0k)}')
    log(f'  {PAR_K}, different visits: {row_v(s1k)}')
    bs_ = sdk["bs"][np.isfinite(sdk["bs"])]
    log(f"  difference (different visits minus same visit), bootstrap on the union (n = "
        f"{sdk['n_union']}): {row_v(sdk, 'difference')}")
    log(f'    resamples with difference <= 0: {100 * float((bs_ <= 0).mean()):.1f}% of {len(bs_)}; '
        f'the estimate at different visits '
        + ("lies inside the " if s0k["lo"] <= s1k["rho"] <= s0k["hi"] else "lies OUTSIDE the ")
        + f"same-visit interval [{s0k['lo']:+.3f}; {s0k['hi']:+.3f}]"
        + f', that at the same visit '
        + ("lies inside the " if s1k["lo"] <= s0k["rho"] <= s1k["hi"] else "lies OUTSIDE the ")
        + f"different-visits interval [{s1k['lo']:+.3f}; {s1k['hi']:+.3f}].")
    log(f'  subjects: in both comparisons {len(p0_ & p1_)}; only at the same visit {len(p0_ - p1_)}; '
        f'only at different visits {len(p1_ - p0_)}. Of the {len(p0_)} of the same visit, members of '
        f'the 150 h cohort: {len(p0_ & p150)}; of the 24 h cohort: {len(p0_ & p24)}.')
    log("  In what the two comparisons differ (median [minimum; maximum]; second half minus first):")
    t0_ = times(m0, "150", "24", '150 h BL', '24 h BL')
    t1_ = times(mp, "150", "24", f'150 h {va}', f'24 h {vb}')

    def med_(x, dec):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return (f"{np.median(x):.{dec}f} [{x.min():.{dec}f}; {x.max():.{dec}f}]" if len(x)
                else 'NOT available')

    for i_, name_, dec in ((1, 'months between the two visits (age_at_visit)', 1),
                           (0, 'days between the two runs (24 h minus 150 h)', 0),
                           (3, 'months between the month of the visit and the run, 150 h', 0),
                           (4, 'months between the month of the visit and the run, 24 h', 0),
                           (5, 'difference between the two visit-run times, months', 0)):
        log(f'    {name_}: same visit {med_(t0_[i_][2], dec)}; different visits '
            f'{med_(t1_[i_][2], dec)}')
        for et_, m_, t_ in (("same visit (150 h BL versus 24 h BL), 'sporadic Type1'",
                             m0, t0_),
                            (f"different visits (150 h {va} against 24 h {vb}), '{NAME_COHORT}'", mp, t1_)):
            x_ = np.asarray(t_[i_][2], float)
            ok_ = np.isfinite(x_)
            record_v('different visits, TTT side by side: same visit and different visits',
                       'sporadic Type1 / ' + LBLS_BETWEEN[NAME_COHORT], 'times',
                       dict(n=int(ok_.sum()),
                            rho=float(np.median(x_[ok_])) if ok_.any() else np.nan,
                            lo=float(x_[ok_].min()) if ok_.any() else np.nan,
                            hi=float(x_[ok_].max()) if ok_.any() else np.nan),
                       variant=et_ + '; ci_inf = minimum, ci_upp = maximum',
                       quantity=name_ + ': median', units=t_[i_][3], subj=m_.PATNO[ok_])
    for side_, col_ in (('150 h', 'run_150'), ('24 h', 'run_24')):
        d0_, d1_ = set(m0[col_]), set(mp[col_])
        log(f'    run dates at {side_}: same visit {len(d0_)} (from {min(d0_).date()} to '
            f'{max(d0_).date()}); different visits {len(d1_)} (from {min(d1_).date()} to '
            f'{max(d1_).date()}); dates in common between the two comparisons: {len(d0_ & d1_)}')
        record_v('different visits, TTT side by side: same visit and different visits',
                   'sporadic Type1 / ' + LBLS_BETWEEN[NAME_COHORT], 'run dates',
                   dict(n=len(p0_ | p1_), rho=float(len(d0_ & d1_)), lo=float(len(d0_)),
                        hi=float(len(d1_))),
                   variant='ci_inf = dates at the same visit, ci_upp = dates at different visits; '
                            'n = subjects in the union',
                   quantity=f'run dates at {side_} in common between the two comparisons',
                   units="date", subj=sorted(p0_ | p1_))
    log(f"  The same parameter in the sensitivity analyses of the two comparisons:")
    for et_, a_, b_ in (('mean at 150 h in place of the median',
                         MEAN2[('sporadic Type1', PAR_K)],
                         SENSV[('mean at 150 h', PAR_K)]),
                        ('centred on the run date', CENT[('sporadic Type1', PAR_K)],
                         SENSV[('centred on the run date', PAR_K)])):
        log(f"    {et_}: same visit {a_['rho']:+.3f} [{a_['lo']:+.3f}; "
            f"{a_['hi']:+.3f}]{remark_nearby(a_)}; different visits {b_['rho']:+.3f} "
            f"[{b_['lo']:+.3f}; {b_['hi']:+.3f}]{remark_nearby(b_)}")
    log(f'  The same comparison in the other sets ({PAR_K}):')
    for name1, name0 in (('everyone', 'everyone'), ('Type1 at 24 h', 'Type1 at 24 h'),
                         ('sporadic Type1', 'sporadic Type1')):
        if (name1, PAR_K) not in DIFF0:
            continue
        a_, b_, c_ = BETWEEN[(name0, PAR_K)], FRAV[PRINC + (name1, PAR_K)], DIFF0[(name1, PAR_K)]
        log(f"    '{name1}': same visit (n = {a_['n']}) {a_['rho']:+.3f} [{a_['lo']:+.3f}; "
            f"{a_['hi']:+.3f}]; different visits (n = {b_['n']}) {b_['rho']:+.3f} [{b_['lo']:+.3f}; "
            f"{b_['hi']:+.3f}]; in common {COM0[name1]}; {row_v(c_, 'difference')}")
    # TTT and AUC within the same protocol, in the main set
    r150_ = _rho(mp["md150_TTT"].values, mp["md150_AUC"].values)
    r24_ = _rho(mp["m24_TTT"].values, mp["m24_AUC"].values)
    log(f"  TTT and AUC within the same protocol, '{NAME_COHORT}' (n = {len(mp)}): rho = "
        f"{r150_:+.2f} at 150 h {va}, {r24_:+.2f} at 24 h {vb}."
        + (f' In both |rho| >= {RHO_SAME:.2f}: the correlations between protocols of TTT and of AUC '
           f'are not two independent findings.'
           if min(abs(r150_), abs(r24_)) >= RHO_SAME else
           f' In at least one protocol |rho| < {RHO_SAME:.2f}.'))
    for side_, r_ in ((f'150 h {va}', r150_), (f'24 h {vb}', r24_)):
        record_v('different visits, TTT side by side: same visit and different visits',
                   LBLS_BETWEEN[NAME_COHORT], f'TTT against AUC at {side_}',
                   dict(n=len(mp), rho=float(r_), lo=np.nan, hi=np.nan),
                   quantity='rho within the protocol', subj=mp.PATNO)
    app_k = pairing(t0_, t1_)      # here: same visit against different visits
    d24_0, d24_1 = set(m0['run_24']), set(mp['run_24'])
    log("  Same visit against different visits, matching on times ('appaiato' = overlapping "
        "interquartile ranges): " + "; ".join(text_app(a_) for a_ in app_k) + ".")
    if excludes_zero(sdk):
        log("  Reading, derived: the interval of the difference excludes zero.")
    else:
        log("  Reading, derived: the interval of the difference includes zero"
            + (' (with a limit close to zero)' if neighbours(sdk) else "")
            + ": at this level the difference between the two correlations is not distinguishable "
              "from zero.")
    no_app = [a_['name'] for a_ in app_k if a_["app"] is False]
    si_app = [a_['name'] for a_ in app_k if a_["app"]]
    log('  The two comparisons differ in the visit of the collection at 24 h (by construction)'
        + ('; they are not matched on: ' + "; ".join(no_app) if no_app else "")
        + ('; they are matched on: ' + "; ".join(si_app) if si_app else "")
        + f'; they share {len(d24_0 & d24_1)} run dates at 24 h out of {len(d24_0 | d24_1)} and '
          f'{len(p0_ & p1_)} subjects out of {len(p0_ | p1_)}.')
    log('  These data do not say which of these factors, or whether chance, produces the '
        'difference.')

# --- 6.7 Monte Carlo error
log('\n6.7 Monte Carlo error of the limits of the intervals of section 6')
if ERR_MC:
    e_ = np.array(ERR_MC, float).ravel()
    e_ = e_[np.isfinite(e_)]
    above = int((SE_MC * e_ > NEARBY).sum())
    log(f'Bootstrap intervals recorded (without the repeated estimates): {len(ERR_MC)}; limits with '
        f'a Monte Carlo SE: {len(e_)}.')
    log(f'Monte Carlo SE of a limit: median {np.median(e_):.4f}, 90th percentile '
        f'{np.percentile(e_, 90):.4f}, maximum {e_.max():.4f}. In {above} limits {SE_MC:g} SE exceed '
        f'{NEARBY}: there the threshold is the Monte Carlo error.')
    only_mc = [x for x in FLAGGED if x['only_mc']]
    log(f'Limits that are not read as a sign: {len(FLAGGED)}, of which less than {NEARBY} from zero '
        f'{len(FLAGGED) - len(only_mc)} and flagged only for the Monte Carlo error {len(only_mc)}' + (":" if only_mc else "."))
    for x in only_mc:
        log(f"  {x['parameter']}, {x['variant']} — {x['block']} — {x['dataset']}: limit {x['limit']} "
            f"{x['value']:+.3f}, threshold {x['threshold']:.3f}")
else:
    log('No bootstrap interval recorded in this section.')

log()
if not princ_ok:
    log("CONCLUSION cannot be formulated: the main comparison is not estimated.")
else:
    va, vb = PRINC
    vp = {par: AGREEMENT[(NAME_COHORT, par)] for par in PAR}
    conc = [p for p in PAR if vp[p] == 'agree']
    comp = [p for p in PAR if vp[p] == 'compatible']
    disc = [p for p in PAR if vp[p] == 'disagree']
    s0 = {p: BETWEEN[('sporadic Type1', p)] for p in PAR}
    s1 = {p: FRAV[chp + (p,)] for p in PAR}
    log(f"CONCLUSION, derived from the numbers (main comparison: '{NAME_COHORT}', n = {len(mp)},")
    log(f"150 h {va} against 24 h {vb}; set against 'sporadici Type1' at the same visit, n = "
        f"{len(DATASETS['sporadic Type1'])}).")
    if not disc and not comp:
        log(f"  For all the {len(PAR)} parameters the estimate has the same sign and the intervals "
            f"overlap: the evidence at different visits AGREES with that at the same visit.")
    elif not disc:
        log(f"  No parameter has disjoint intervals: the evidence at different visits does NOT "
            f"CONTRADICT that at the same visit. Same sign: "
            f"{(', '.join(conc) if conc else 'noone')}. Estimate of different sign but overlapping "
            f"intervals: {', '.join(comp)}.")
    else:
        log(f"  DISJOINT intervals for: {', '.join(disc)}. On these parameters the two pieces of "
            f"evidence do NOT agree. Same sign and overlapping intervals: "
            f"{(', '.join(conc) if conc else 'noone')}; overlapping with different sign: "
            f"{(', '.join(comp) if comp else 'noone')}.")
    if comp:
        log("  (" + "; ".join(f"{p}: {s0[p]['rho']:+.3f} at the same visit, {s1[p]['rho']:+.3f} at "
                              f"different visits" for p in comp) + ")")
    fuori0 = [p for p in PAR if not s0[p]["lo"] <= s1[p]["rho"] <= s0[p]["hi"]]
    log('  The difference between the two correlations excludes zero for: '
        + listing_ez({p: DIFF0[(NAME_COHORT, p)] for p in PAR})
        + ". Estimate at different visits outside the same-visit interval: "
        + (", ".join(f"{p} ({s1[p]['rho']:+.3f} versus [{s0[p]['lo']:+.3f}; {s0[p]['hi']:+.3f}])" for p in fuori0) if fuori0 else 'none')
        + ".")
    others = {n_: [p for p in PAR if AGREEMENT.get((n_, p)) == 'disagree']
             for n_ in ('everyone', 'Type1 at 24 h', 'sporadic Type1')}
    log('  In the other sets, parameters with disjoint intervals: '
        + "; ".join(f"'{n_}' {(', '.join(v) if v else 'noone')}" for n_, v in others.items())
        + ".")
    log("  What changes with the larger sample:")
    n0, n1 = len(DATASETS['sporadic Type1']), len(mp)
    in_com = len(set(DATASETS['sporadic Type1'].PATNO) & set(mp.PATNO))
    log(f'  - subjects: from {n0} to {n1} ({n1 / n0:.1f} times); {n1 - in_com} of the {n1} were not '
        f'in the same-visit comparison.')
    w0 = np.mean([s0[p]["hi"] - s0[p]["lo"] for p in PAR])
    w1 = np.mean([s1[p]["hi"] - s1[p]["lo"] for p in PAR])
    log(f"  - mean width of the 95% interval over the {len(PAR)} parameters: from {w0:.2f} to "
        f"{w1:.2f}.")
    for et_, ss in (('at the same visit', s0), ('at different visits', s1)):
        log(f'  - intervals that exclude zero {et_}: '
            + listing_ez(ss, empty='noone', sign=True) + ".")
    # the upper limit: in the main comparison, in the other sets of the same pair of
    # visits and in the sensitivity analyses; and whether it is a gain over the same
    # visit
    p_hi1 = max(PAR, key=lambda p: s1[p]["hi"])
    p_hi0 = max(PAR, key=lambda p: s0[p]["hi"])
    others_hi = {f"'{n_}'": max(FRAV[PRINC + (n_, p)]["hi"] for p in PAR)
                for n_ in ('everyone', 'Type1 at 24 h', 'sporadic Type1')
                if PRINC + (n_, "Fmax") in FRAV
                and np.isfinite(FRAV[PRINC + (n_, "Fmax")]["rho"])}
    others_hi.update({f"sensitivity '{v_}'": max(SENSV[(v_, p)]["hi"] for p in PAR)
                     for v_ in sorted({k_[0] for k_ in SENSV})})
    d_hi = s1[p_hi1]["hi"] - s0[p_hi0]["hi"]
    log("  - upper limit of the interval at different visits: "
        + ", ".join(f"{p} {s1[p]['hi']:+.3f}" for p in PAR)
        + f"; the highest is {s1[p_hi1]['hi']:+.3f} ({p_hi1}). In the other sets of the same pair of "
          f"visits and in the sensitivity analyses the highest goes from "
          f"{min(others_hi.values()):+.3f} to {max(others_hi.values()):+.3f} ("
        + "; ".join(f"{k_} {v_:+.3f}" for k_, v_ in others_hi.items()) + ").")
    log(f"    At the same visit the highest was {s0[p_hi0]['hi']:+.3f} ({p_hi0}): the difference "
        f"between the two maxima is {d_hi:+.3f}, "
        + (f'less than {NEARBY} in absolute value: the upper limit is NOT a gain of the larger sample.' if abs(d_hi) < NEARBY else
           "and the upper limit is lower with the larger sample." if d_hi < 0 else
           "and the upper limit is HIGHER with the larger sample."))
    chd = PRINC + ('sporadic Type1',)
    par_ok = chd in INSD and len(INSD[chd]) >= N_MIN
    if par_ok:
        sd = {p: DENV[chd + (p,)] for p in PAR}
        below_estimate = [p for p in PAR if s1[p]["hi"] < sd[p]["rho"]]
        below_whole = [p for p in PAR if s1[p]["hi"] < sd[p]["lo"]]
        log(f"  - set against the benchmark on the same visits (24 h {va} against 24 h {vb}, "
            f"'sporadici Type1', n = {len(INSD[chd])}): the interval between protocols lies entirely "
            f"below the ESTIMATE of the benchmark for "
            f"{(', '.join(below_estimate) if below_estimate else 'no parameter')}; entirely below "
            f"the INTERVAL of the benchmark for "
            f"{(', '.join(below_whole) if below_whole else 'no parameter')}; the difference "
            f"(bootstrap on the union) excludes zero for "
            + listing_ez({p: DIFFV[PRINC + ('sporadic Type1', NAME_COHORT, p)]
                         for p in PAR}) + ".")
        log(f'    Intervals of the benchmark that exclude zero: '
            + listing_ez(sd, empty='noone', sign=True) + ".")
        wd = np.mean([sd[p]["hi"] - sd[p]["lo"] for p in PAR])
        log(f"    Mean width of the interval: {wd:.2f} in the benchmark (n = {len(INSD[chd])}), "
            f"{w1:.2f} between protocols (n = {n1})."
            + (" It is the benchmark, not the comparison between protocols, that limits the "
               "precision of the difference." if wd > w1 else ""))
    else:
        log(f"  - benchmark on the same visits ('sporadici Type1', 24 h {va} against 24 h {vb}): NOT "
            f"estimable (n < {N_MIN}).")
    for nd_ in ('Type1 at 24 h', 'everyone'):
        chd_ = PRINC + (nd_,)
        if PRINC + (nd_, nd_, "Fmax") in DIFFV:
            log(f"  - the same comparison with the benchmark in '{nd_}' (within n = "
                f"{len(INSD[chd_])}, between n = {len(INSV[chd_])}): the difference excludes zero "
                f"for " + listing_ez({p: DIFFV[PRINC + (nd_, nd_, p)] for p in PAR}) + ".")
    for var_ in sorted({k_[0] for k_ in SENSV}):
        log(f"  - in the sensitivity analysis '{var_}' the intervals that exclude zero are: "
            + listing_ez({p: SENSV[(var_, p)] for p in PAR}, empty='noone', sign=True)
            + ("." if not all(SENSV[(var_, p)] is s1[p] for p in PAR) else
               '. (coincides with the main estimate, value by value)'))
    tt = {p: CEILINGV.get(chp + (p,)) for p in PAR}
    below_ceiling = [p for p in PAR if tt[p] and tt[p]['computed'] and tt[p]['estimable']
                   and s1[p]["hi"] < tt[p]['point']]
    below_ceiling_ci = [p for p in PAR if tt[p] and tt[p]['computed'] and tt[p]['estimable']
                      and s1[p]["hi"] < tt[p]["lo"]]
    no_st = [p for p in PAR if tt[p] and tt[p]['computed'] and not tt[p]['estimable']]
    log(f"  - set against the attenuation ceiling (6.3): the interval lies entirely below the "
        f"ESTIMATE of the ceiling for "
        f"{(', '.join(below_ceiling) if below_ceiling else 'no parameter')}; entirely below the "
        f"INTERVAL of the ceiling for "
        f"{(', '.join(below_ceiling_ci) if below_ceiling_ci else 'no parameter')}; ceiling not "
        f"estimable for {(', '.join(no_st) if no_st else 'no parameter')}.")
    calc = [p for p in PAR if tt[p] and tt[p]['computed']]
    if any(tt[p]['estimable'] for p in calc):
        log('    Ceiling with its interval: '
            + ", ".join(f"{p} {tt[p]['point']:.2f} [{tt[p]['lo']:.2f}; {tt[p]['hi']:.2f}]"
                        for p in calc if tt[p]['estimable'])
            + ". It is an UPPER limit of the true ceiling: only the repeatability within the run "
              "counts.")
    nst0 = [p for p in PAR if not np.isfinite(CEILING[('sporadic Type1', p)])]
    log("  - repeatability at 150 h: at the same visit ('sporadici Type1') the ceiling was not "
        "estimated or not estimable for "
        + (", ".join(nst0) if nst0 else 'no parameter')
        + "; at different visits the ICC(1,3) at 150 h is "
        + (", ".join(f"{p} {tt[p]['i3b']:+.3f}" for p in calc) if calc else 'not computed')
        + ' and the estimate of the ceiling '
        + (", ".join(f"{p} {tt[p]['point']:.2f}" for p in calc if tt[p]['estimable'])
           if any(tt[p]['estimable'] for p in calc) else 'not estimable') + ".")
    log('  What it does NOT add. In the main comparison the two values come from different')
    log('  collections, runs and storage times:')
    tp = times(mp, "150", "24", f'150 h {va}', f'24 h {vb}')
    for i_ in (1, 0, 5):
        log(f"  - {tp[i_][0]}: {distr(tp[i_][2], tp[i_][1])}")
    same_g = int((tp[0][2] == 0).sum())
    log(f'  Pairs with the two runs on the same day: {same_g} out of {len(mp)}.')
    log('  A low correlation between protocols at different visits does not distinguish between')
    log('  protocols that measure different quantities, change between the visits, effect')
    log('  of storage and effect of the run date.')
    if par_ok:
        app = APPV[PRINC + ('sporadic Type1', NAME_COHORT)]
        si = [a_ for a_ in app if a_["app"]]
        no = [a_ for a_ in app if a_["app"] is False]
        nd = [a_ for a_ in app if a_["app"] is None]
        com_p = len(set(INSD[chd].PATNO) & set(mp.PATNO))
        log(f"  The benchmark (6.2: 24 h {va} against 24 h {vb}, 'sporadici Type1', n = "
            f"{len(INSD[chd])}; median in the benchmark against median between protocols):")
        log('  - matched on: ' + ("; ".join(text_app(a_) for a_ in si) if si else 'no time') + ".")
        log('  - NOT matched on: ' + ("; ".join(text_app(a_) for a_ in no) if no else
                                       'no time') + "."
            + (' Cannot be assessed: ' + "; ".join(a_['name'] for a_ in nd) + "." if nd else ""))
        log(f'  It therefore removes from the comparison '
            + ('only: ' + "; ".join(a_['name'] for a_ in si) if si and (no or nd) else
               'all three times' if si else 'none of the three times')
            + f', and only to the extent that its {len(INSD[chd])} subjects ({com_p} in common with '
              f'the {len(mp)} of the main comparison) represent the others.')
    else:
        log("  The benchmark on the same visits is not estimable in the set of the sporadic Type1: "
            "the main comparison remains without a benchmark.")
    if WITHOUT_PAR:
        log('  Without any 24 h benchmark remain the comparisons: '
            + ", ".join(f'150 h {a} against 24 h {b}' for a, b in WITHOUT_PAR) + ".")


# ------------------------------------------------------------------ output
kin = pd.DataFrame(rows_kin)
kin.insert(0, 'root', ROOT)
kin["n"] = kin["n"].astype("Int64")
kin['n_sporadic'] = kin['n_sporadic'].astype("Int64")
kin.to_csv(os.path.join(OUT, "106_agreement_kinetics.csv"), index=False)
outc = pd.DataFrame(rows_outcome)      # does not depend on the working directory: no column
outc['n_units'] = outc['n_units'].astype("Int64")
outc.to_csv(os.path.join(OUT, "106_agreement_outcome.csv"), index=False)
with open(os.path.join(OUT, "106_protocol_agreement.txt"), "w") as f:
    f.write("\n".join(rep) + "\n")
print(f'\nWritten in {OUT}: 106_agreement_kinetics.csv, 106_agreement_outcome.csv, '
      f'106_protocol_agreement.txt')
