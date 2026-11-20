"""
108 - Data availability and timing: what there is, at which visit, for how many.

The first reviewer asks three questions that concern not a result but its
denominators. The reviewer asks, for the CSF analytes, for the "number of available
results for each test in each timepoint"; asks when the assay sample was collected
relative to the collection of the clinical data and "how many data points after SAA
collection were present for each participant"; and observes that the follow-up
of the primary cohort (3 years) may be too short. The manuscript reports n and
observations of the individual models, but not a table that answers these three
questions. It is built here, for each of the two cohorts separately.

What is computed (no model is fitted: these are only counts and quantiles):

  1. CSF analytes of sections 1.16-1.17 and 2.17 of the supplement: the seven
     analytes of the curated data cut (Elecsys Abeta42, t-tau, p-tau181; total
     alpha-synuclein ELISA; p-tau181, Abeta40, Abeta42 from Indiana University)
     and CSF and plasma albumin (Project 181, from the biospecimen file). Per
     analyte and cohort: n with a baseline value, n with the complete panel, n
     with a value at each visit. The per-visit table stays per visit
     (EVENT_ID): each scheduled visit has a single nominal year (this is
     checked, and the script stops if it is not so); the unscheduled visits
     (ST, PW) have one row for each nominal year to which the curated data cut
     assigns them; next to it is the count per nominal year, which includes
     them, as in the longitudinal and retention tables. The denominators
     already printed in the manuscript (complete panel, albumin,
     component-albumin comparison) are recomputed and compared with the
     outputs of 12, 20, 36 and 100; where the comparison is on n only, or is
     missing, this is stated.
  2. Timing at baseline. The assay sample is the one from the lumbar puncture
     of the BL visit (this is checked on the raw file, by recomputing the
     kinetic parameters of the two cohorts; repeated BL results of the same
     protocol are counted before any filter). From the curated data cut:
     interval between lumbar puncture, clinical visit and DAT-SPECT (from the
     ages, which are days / 365: the intervals are rounded to the whole day
     before thresholds and quantiles); time between sample collection and the
     assay run, by cohort and by enrolment phase, without imputing the missing
     age at puncture; difference between the nominal year of the visit, which
     is the time of the models, and the time actually elapsed. The time
     variables that no file of the pipeline carries (in the 150 h cohort: age
     at puncture, age at DAT-SPECT, run date) are declared NOT verified.
  3. For each outcome of the longitudinal models (motor, DAT, the twelve
     primary non-motor ones, the extended neuropsychological battery):
     participants, observations, observations per participant, participants
     with at least 2 and at least 3 observations, years of follow-up. The
     sample is that of the model (outcome and covariates not missing). The
     observation at year 0 is the baseline visit, that is, the visit of the
     sample collection: the counts "after sample collection" are those at
     years >= 1 and are reported next to the totals, which include year 0 as
     in the models. The counts are compared with the outputs of 15, 72, 78 and
     84. For DAT-SPECT: scans at the scheduled years and off schedule.
  3ter. MDS-UPDRS III: visits after baseline with part II but without the
     modelled variable of part III, and how many of these are visits on
     dopaminergic treatment. The definition of the two variables of part III
     is read from the "Data dictionary" sheet of the curated data cut and
     quoted; if the sheet is not found, the OFF/ON reading is declared NOT
     verified.
  4. Time under observation, year by year, in the two cohorts; with the number
     of participants who at the date of the last data point could have reached
     that year, and with the exit from observation among those who could reach
     it: the calendar limit and the exit are both reported.

ONLY_SPORADIC restricts nothing here: it says which working directory is being
read, and the script stops if variable and working directory do not agree.

Warnings also printed in the report. The seven analytes are read from the curated
data cut only: the biospecimen file is not tabulated by visit for these analytes
(it is for albumin only), and whether it contains results at visits that the
curated data cut does not carry is not verified here. The visit date has a
resolution of one month, the ages of one day: the intervals between puncture,
visit and DAT come from the ages; the sample collection date, which is needed
only for the storage time, is approximated to mid-month. Enrolment phase and
visit date, on which "calendar reachability" depends, have no counterpart in the
pipeline unless 70_baseline.csv carries the counts by phase.

Possible side effect: if outputs/cache/alb.csv is missing, _alb.path_alb
rebuilds it from the biospecimen file; the report flags this.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 scripts/108_availability_and_timing.py
Output: outputs/108_csf_baseline.csv, 108_csf_by_visit.csv, 108_times_baseline.csv,
        108_longitudinal_outcomes.csv, 108_longitudinal_by_year.csv,
        108_updrs3_treatment.csv, 108_retention.csv,
        108_availability_and_timing.txt
"""

import hashlib
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA                                       # noqa: E402
from _alb import path_alb                                    # noqa: E402

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
NAME_CURATED = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"
CACHE = os.path.join(OUT, "cache", "curated.parquet")
# the caudate variant of the family of the 63 models lives in a working directory of its own
CAUDATE = os.path.join(os.path.dirname(os.path.normpath(BASE)), 'run_caudate',
                       "outputs")

THRESHOLD_ANALYTES = 150      # rule of 12 and 20: an analyte enters the component if it has
                          # at least this many baseline values in the cohort
THRESHOLDS_DAYS = (30, 90, 180)
YEARS_KEY = (2, 5, 10)
TOL = 1e-8                # tolerance on the value comparisons with the pipeline
SHEET_DICTIONARY = "Data dictionary"   # sheet of the xlsx file of the curated data cut
# off-schedule visits: by definition without a fixed nominal year. Every other
# visit of the curated data cut must have a single nominal year (checked below)
EVENTS_NO_SCHEDULED = ("PW", "ST")
# DAT-SPECT schedule declared here (nominal years): it is a definition of
# this script, not a value derived from the data, and the report says so
YEARS_DAT_SCHEDULED = (0, 1, 2, 4)
PROJECT_ALB = 181        # albumin project declared in the supplement:
                          # compared with the PROJECTID column, never just written down
# the three variables of block 3ter (part II, modelled part III, part III ON)
VAR_II, VAR_III, VAR_III_ON = "updrs2_score", "updrs3_score", "updrs3_score_on"

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def share(k, n):
    """'k/n (x%)' without dividing by zero."""
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else f"{k}/{n}"


def fingerprint(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def quantiles(x):
    """n, median, quartiles and extremes of a series, ignoring missing values."""
    x = pd.Series(x, dtype=float).dropna()
    if not len(x):
        return dict(n=0, median=np.nan, q1=np.nan, q3=np.nan,
                    minimum=np.nan, maximum=np.nan)
    return dict(n=int(len(x)), median=float(x.median()),
                q1=float(x.quantile(.25)), q3=float(x.quantile(.75)),
                minimum=float(x.min()), maximum=float(x.max()))


def fq(q, dec=0):
    """'median [q1; q3], extremes min..max' for the report."""
    return (f"{q['median']:.{dec}f} [{q['q1']:.{dec}f}; {q['q3']:.{dec}f}], extremes "
            f"{q['minimum']:.{dec}f}..{q['maximum']:.{dec}f}")


def partial_ranks(y, x, Z):
    """Partial rank correlation, like `partial` of 12, 20 and 100."""
    Zc = np.column_stack([np.ones(len(y)), np.asarray(Z, float)])
    ry = stats.rankdata(y)
    rx = stats.rankdata(x)
    ry = ry - Zc @ np.linalg.lstsq(Zc, ry, rcond=None)[0]
    rx = rx - Zc @ np.linalg.lstsq(Zc, rx, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def component(M, cols):
    """PC1 of the standardised ranks, oriented as in 12 and 20.

    Returns the scores and the proportion of variance explained.
    """
    R = np.column_stack([stats.rankdata(M[c]) for c in cols])
    R = (R - R.mean(0)) / R.std(0)
    _, s, vt = np.linalg.svd(R, full_matrices=False)
    f = R @ vt[0]
    if stats.spearmanr(f, M[cols[0]]).statistic < 0:
        f = -f
    return f, float(s[0] ** 2 / (s ** 2).sum())


def equals(a, b, what):
    """Identity of value with the pipeline, within the declared tolerance."""
    a, b = float(a), float(b)
    assert abs(a - b) <= TOL * max(1.0, abs(b)), (
        f'{what}: recomputed {a!r}, in the pipeline {b!r}')


def load(name, root=None):
    """A csv of the pipeline, from the current working directory or from the one given."""
    p = os.path.join(root or OUT, name)
    assert os.path.exists(p), f'pipeline output missing: {p}'
    return pd.read_csv(p)


def one_row(x):
    """A text cell on a single line; empty string if missing."""
    return " ".join(str(x).split()) if pd.notna(x) else ""


def load_dictionary():
    """"Data dictionary" sheet of the xlsx file of the curated data cut, if reachable.

    Returns (table, path) or (None, None). In the sheet a variable with
    several codes carries its name on the first row only: it is propagated
    to the rows of the following codes.
    """
    for p in (os.path.join(BASE, NAME_CURATED), os.path.join(DATA, NAME_CURATED)):
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_excel(p, sheet_name=SHEET_DICTIONARY)
        except ValueError:                    # the file is there, the sheet is not
            continue
        if not {"Variable", "Description", "Code", "Decode",
                "Derivation Notes"} <= set(d.columns):
            continue
        d = d.copy()
        d['variable'] = d["Variable"].ffill()
        return d, p
    return None, None


# ------------------------------------------------------------------ data
if os.path.exists(CACHE):
    cur = pd.read_parquet(CACHE)
    source_curated = os.path.realpath(CACHE)
else:
    cand = [os.path.join(BASE, NAME_CURATED), os.path.join(DATA, NAME_CURATED)]
    located = [c for c in cand if os.path.exists(c)]
    assert located, f"curated not found either in the cache or in: {cand}"
    cur = pd.read_excel(located[0], sheet_name="20260511")
    source_curated = located[0]
cur["PATNO"] = cur["PATNO"].astype("int64")
assert not cur.duplicated(["PATNO", "EVENT_ID"]).any(), \
    "curated data cut: more than one row for the same visit of the same subject"
assert cur.YEAR.notna().all(), 'curated data cut: visits without a nominal year'
cur['year'] = cur.YEAR.astype(int)
cur['data_visit'] = pd.to_datetime(cur.visit_date, format="%m/%Y")
assert cur.data_visit.notna().all(), 'curated data cut: visits without a date'
cur['month'] = cur.data_visit.dt.year * 12 + cur.data_visit.dt.month
bl = cur[cur.EVENT_ID == "BL"].set_index("PATNO")
assert bl.index.is_unique
assert (bl.year == 0).all(), 'curated data cut: a BL visit with nominal year other than 0'
sporadic = set(bl.index[bl.subgroup == "Sporadic PD"])

# visit -> nominal year. Each scheduled visit must have ONE nominal year only
# in the whole curated data cut, and each year a single scheduled visit: without
# this the per-visit table would silently assign a wrong year.
_years_ev = cur.groupby("EVENT_ID").year.agg(["min", "max", "size"])
_prog = _years_ev[~_years_ev.index.isin(EVENTS_NO_SCHEDULED)]
_ambiguous = _prog[_prog["min"] != _prog["max"]]
assert not len(_ambiguous), (
    "curated data cut: scheduled visits with more than one nominal year: "
    + "; ".join(f"{ev} years {int(r['min'])}..{int(r['max'])}"
                for ev, r in _ambiguous.iterrows()))
YEAR_OF = {str(ev): int(a) for ev, a in _prog["min"].items()}
assert len(set(YEAR_OF.values())) == len(YEAR_OF), \
    'curated data cut: two scheduled visits with the same nominal year'
EVENT_OF = {a: ev for ev, a in YEAR_OF.items()}
NO_PROG = _years_ev[_years_ev.index.isin(EVENTS_NO_SCHEDULED)]

dic, dic_p = load_dictionary()

saa = pd.read_csv(SAA, low_memory=False)
saa["PATNO"] = saa["PATNO"].astype("int64")
saa["RUNDATE"] = pd.to_datetime(saa["RUNDATE"])
assert not saa.duplicated().any(), 'duplicate rows in the SAA file'

alb_in_cache = os.path.exists(os.path.join(OUT, "cache", "alb.csv"))
alb_p = path_alb(OUT)
alb = pd.read_csv(alb_p)
alb["PATNO"] = alb["PATNO"].astype("int64")
alb["v"] = pd.to_numeric(alb.TESTVALUE, errors="coerce")
assert not alb.duplicated(["PATNO", "CLINICAL_EVENT", "TESTNAME"]).any(), \
    "albumin: more than one row per subject, visit and test"
# the project label comes from the column, and must be the declared one
assert alb.PROJECTID.notna().all(), 'albumin: rows without PROJECTID'
projects_alb = sorted(set(alb.PROJECTID.astype("int64")))
assert projects_alb == [PROJECT_ALB], (
    f'albumin: the PROJECTID column carries the projects {projects_alb}, the supplement declares '
    f'Project {PROJECT_ALB} only')
ET_ALB = f'Project {projects_alb[0]}'

# Table S2 of the pipeline: needed for point 2 (phase, treatment) and for 3bis
p70 = pd.read_csv(os.path.join(OUT, "70_baseline.csv"), index_col=0)
col70 = {"24h": [k for k in p70.columns if k.startswith("Primary")],
         "150h": [k for k in p70.columns if k.startswith("Replication")]}

KIN24 = {"saa_Fmax": "Fmax", "saa_TTT": "TTT", "saa_AUC": "AUC",
         "saa_SLOPEMax": "SLOPE"}
KIN150 = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
          "k_SLOPE": "SLOPE"}
# cohort -> (file, completeness column, kinetic parameters, project, stem of the
#            names in the raw file, aggregation of the triplicate, required type)
COHORTS = {
    "24h": dict(name='24 h (Project 237), primary', file="analysis_complete_cases.csv",
                completes="complete_primary", kin=KIN24, project=237,
                raw={"saa_Fmax": "Fmax_24h", "saa_TTT": "TTT_24h",
                        "saa_AUC": "AUC_24h", "saa_SLOPEMax": "SLOPEMax_24h"},
                triplet="mean", kind="Type1"),
    "150h": dict(name='150 h (Project 155), replication', file="08_dataset_150h.csv",
                 completes="complete", kin=KIN150, project=155,
                 raw={"k_Fmax": "Fmax_150h", "k_TTT": "TTT_150h",
                         "k_AUC": "AUC_150h", "k_T50": "T50_150h",
                         "k_SLOPE": "SLOPE_150h"},
                 triplet="median", kind=None),
}


def load_cohort(root, c):
    d = pd.read_csv(os.path.join(root, COHORTS[c]["file"]))
    if COHORTS[c]['completes'] in d.columns:
        d = d[d[COHORTS[c]['completes']]]
    d = d.copy()
    d["PATNO"] = d["PATNO"].astype("int64")
    assert d.PATNO.is_unique, f'{c}: repeated subjects in the cohort file'
    return d.sort_values("PATNO").reset_index(drop=True)


cohort = {c: load_cohort(OUT, c) for c in COHORTS}
P = {c: set(cohort[c].PATNO) for c in COHORTS}
N = {c: len(P[c]) for c in COHORTS}

# ------------------------------------------------------------------ 0. working directory
title("0. WORKING DIRECTORY, POPULATION AND IDENTITY WITH THE PIPELINE")
log(f'SAA file: {os.path.basename(SAA)}, sha256 {fingerprint(SAA)}')
log(f'curated data cut: {os.path.basename(source_curated)}, sha256 {fingerprint(source_curated)}; '
    f'{len(cur)} rows, {cur.PATNO.nunique()} subjects')
log(f'albumin: {os.path.basename(alb_p)}, sha256 {fingerprint(alb_p)}; {len(alb)} rows; PROJECTID '
    f'column: {projects_alb} (declared: {PROJECT_ALB}; they coincide)')
if not alb_in_cache:
    log('  WARNING, write outside the 108_ prefix: outputs/cache/alb.csv was missing and has been '
        'rebuilt in this run from _alb.path_alb, from the biospecimen file.')
if dic is not None:
    log(f'dictionary of the curated data cut: {os.path.basename(dic_p)}, sheet «{SHEET_DICTIONARY}», '
        f'sha256 of the file {fingerprint(dic_p)}')
else:
    log(f'dictionary of the curated data cut: {NAME_CURATED} with the sheet «{SHEET_DICTIONARY}» not '
        f'found either in the working directory or among the raw data. The checks that use it '
        f'(visit-year, OFF/ON reading of the MDS-UPDRS III) are NOT performed.')

# visits and nominal years: uniqueness has already been checked at loading
_ap = sorted(YEAR_OF.values())
log(f'visits and nominal years, over the whole curated data cut: {len(YEAR_OF)} scheduled visits, '
    f'each with a single nominal year and one per year ('
    + ", ".join(f'{EVENT_OF[a]} = {a}' for a in _ap) + ")")
log('  unscheduled visits, without a fixed year: '
    + ("; ".join(f"{ev}: {int(r['size'])} rows, nominal years from {int(r['min'])} a {int(r['max'])}" for ev, r in NO_PROG.iterrows())
       if len(NO_PROG) else 'none in the curated data cut'))
if dic is not None:
    _de = dic[dic.variable == "EVENT_ID"]
    DECOD_EV = {str(k): one_row(v) for k, v in zip(_de.Code, _de.Decode)}
    assert set(cur.EVENT_ID) <= set(DECOD_EV), (
        f'visits of the curated data cut missing from the dictionary: '
        f'{sorted(set(cur.EVENT_ID) - set(DECOD_EV))}')
    for ev, a in YEAR_OF.items():
        m = re.search(r"\(Year (\d+)\)", DECOD_EV[ev])
        assert m and int(m.group(1)) == a, (
            f'visit {ev}: nominal year {a} in the curated data cut, in the dictionary «{DECOD_EV[ev]}»')
    for ev in NO_PROG.index:
        assert not re.search(r"\(Year \d+\)", DECOD_EV[ev]), (
            f"visit {ev}: treated as unscheduled, but the dictionary gives it a year: "
            f"«{DECOD_EV[ev]}»")
    log(f'  identity with the dictionary: the nominal year of all {len(YEAR_OF)} scheduled visits is '
        f'that of their label; the unscheduled ones have no year in the label ('
        + "; ".join(f"{ev} = «{DECOD_EV[ev]}»" for ev in NO_PROG.index) + ")")
else:
    DECOD_EV = {}
    log("  visit-year correspondence NOT verified against the dictionary (not found): only the "
        "uniqueness of the year per scheduled visit is verified.")

for c in COHORTS:
    assert P[c] <= set(bl.index), f'{c}: subjects missing from the BL visit of the curated data cut'
only = {c: not (P[c] - sporadic) for c in COHORTS}
assert len(set(only.values())) == 1, ("the two cohorts of the working directory are not of the same "
                                      "kind: one is sporadic only, the other is not")
ROOT = 'sporadic only' if only["24h"] else 'full cohort'
assert ONLY == only["24h"], (f"ONLY_SPORADIC={('1' if ONLY else '0')} but the working directory is "
                             f"'{ROOT}': variable and working directory do not agree")
log(f"working directory: {os.path.basename(os.path.normpath(BASE))} ({ROOT}); ONLY_SPORADIC = "
    f"{('1' if ONLY else '0')}")
for c in COHORTS:
    comp = bl.loc[sorted(P[c]), "subgroup"].value_counts().to_dict()
    log(f"  cohort {COHORTS[c]['name']}: n = {N[c]}; subgroups {comp}")
shared = P["24h"] & P["150h"]
log(f"  in both cohorts {len(shared)}, distinct subjects {len(P['24h'] | P['150h'])}")

# the "baseline" of the cohort files is the BL row of the curated data cut: same
# values, but only for the columns that the cohort file carries. `verified`
# records, variable by variable, WHICH comparison was carried out (one sentence;
# absent = none): the verification labels of the report and of the csv files come
# from there, and are not hard-coded.
TIMES_VAR = ("age_at_visit", "age_LP", "age_datscan")
verified = {c: {} for c in COHORTS}
log("\nIdentity of the cohort files with the BL row of the curated data cut, for only the columns "
    "that the cohort file carries (same subjects, same values):")
for c in COHORTS:
    d = cohort[c].set_index("PATNO")
    common = [v for v in TIMES_VAR + ("duration_yrs",
                                      "MIA_PUTAMEN_BILAT", "MIA_CAUDATE_BILAT", "moca",
                                      "abeta", "tau", "ptau") if v in d.columns]
    assert len(common) >= 5, f'{c}: too few common columns for the comparison'
    for v in common:
        a, b = d[v].to_numpy(float), bl.loc[d.index, v].to_numpy(float)
        assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), \
            f'{c}: {v} of the cohort file differs from the BL row of the curated data cut'
        verified[c][v] = f"values identical to {COHORTS[c]['file']}"
    for k in COHORTS[c]["kin"]:
        assert d[k].notna().all(), f'{c}: kinetic parameter {k} missing in a subject of the cohort'
    log(f"  {c}: {len(common)} identical columns on {len(d)} subjects ({', '.join(common)})")
    absent = [v for v in TIMES_VAR if not verified[c].get(v)]
    if absent:
        log(f"       time variables NOT verified in this cohort: {', '.join(absent)} "
            f"({COHORTS[c]['file']} does not carry them: they are read from the curated data cut "
            f"alone, with no counterpart in the pipeline)")
    else:
        log(f"       time variables ({', '.join(TIMES_VAR)}): all among the identical columns")

# the assay sample is the one of the BL visit: the kinetic parameters are recomputed
log("\nThe assay sample is that of the BL visit of the raw file: the kinetic parameters")
log('of the cohorts, recomputed from the BL samples alone, coincide with the cohort files.')
sample = {}
for c, cfg in COHORTS.items():
    s = saa[(saa.PROJECTID == cfg['project']) & (saa.COHORT == "PD")
            & (saa.CLINICAL_EVENT == "BL") & (saa.SAA_Status == "Positive")
            & saa.PATNO.isin(P[c])]
    if cfg['kind']:
        s = s[s.SAA_Type == cfg['kind']]
    s = s.sort_values(["PATNO", "RUNDATE"]).copy()
    assert set(s.PATNO) == P[c], f'{c}: subjects of the cohort without a positive BL sample'
    for k, root_name in cfg['raw'].items():
        v = s[[f'{root_name}_Rep{i}' for i in (1, 2, 3)]].apply(
            pd.to_numeric, errors="coerce")
        s[k] = v.mean(axis=1) if cfg['triplet'] == "mean" else v.median(axis=1)
    ric = s.groupby("PATNO")[list(cfg['raw'])].mean()
    d = cohort[c].set_index("PATNO")
    for k in cfg['raw']:
        assert np.allclose(ric.loc[d.index, k].to_numpy(float), d[k].to_numpy(float),
                           rtol=1e-9, atol=0), \
            f'{c}: {k} recomputed from the BL samples differs from the cohort file'
    by_subj = s.groupby("PATNO").size()
    # the assay run: the ROW run first among the participant's BL samples
    earliest = s.drop_duplicates("PATNO", keep="first").set_index("PATNO")
    sample[c] = earliest.RUNDATE
    selector = 'positives' + (' of type 1' if cfg['kind'] else "")
    log(f"  {c}: {len(s)} BL samples {selector} for {len(by_subj)} subjects (subjects with more than "
        f"one BL sample among those {selector}: {int((by_subj > 1).sum())}); {len(cfg['raw'])} "
        f"identical kinetic parameters; assay runs from {s.RUNDATE.min().date()} to "
        f"{s.RUNDATE.max().date()}")
    # repeated BL results: counted on ALL the BL rows of the protocol for the
    # participants of the cohort, before the filters on clinical cohort, status and type
    everyone_bl = saa[(saa.PROJECTID == cfg['project']) & (saa.CLINICAL_EVENT == "BL")
                   & saa.PATNO.isin(P[c])]
    by_bl = everyone_bl.groupby("PATNO").size()
    extra = everyone_bl[~everyone_bl.index.isin(s.index)].copy()
    log(f'       BL results of the protocol before any filter on status and type: {len(everyone_bl)} '
        f'for {len(by_bl)} subjects; subjects with more than one BL result: {int((by_bl > 1).sum())}')
    if len(extra):
        extra['state_kind'] = (
            extra.SAA_Status.fillna('state not reported').astype(str) + " / "
            + extra.SAA_Type.fillna('type not reported').astype(str))
        departure_g = (extra.RUNDATE - extra.PATNO.map(earliest.RUNDATE)).dt.days
        log(f'       BL results not used: {len(extra)}, of {extra.PATNO.nunique()} subjects; status '
            f'/ type: '
            + "; ".join(f"{k}: {n}" for k, n in
                        extra.groupby('state_kind').size().items())
            + f'; run from {int(departure_g.min())} to {int(departure_g.max())} days relative to the '
              f'result used (negative = before)')
    else:
        log('       no BL result other than those used')
    if "RUNDATE" in d.columns:
        selfsame = (pd.to_datetime(d.RUNDATE) == earliest.RUNDATE.reindex(d.index)).all()
        assert selfsame, f'{c}: run date of the cohort file differs from the raw file'
        verified[c]["RUNDATE"] = f"values identical to {cfg['file']}"
        log('       run date identical to that of the cohort file')
    else:
        log(f"       run date NOT verified in this cohort ({cfg['file']} does not carry it: it is "
            f"read from the raw file alone, with no counterpart in the pipeline)")
    rest = saa[saa.PATNO.isin(P[c]) & (saa.PROJECTID == cfg['project'])
                & (saa.CLINICAL_EVENT != "BL")]
    log(f'       samples of the same protocol at later visits, not used: {len(rest)} '
        f'({rest.PATNO.nunique()} subjects)')

# ------------------------------------------------------------------ 1. CSF
title("1. CSF ANALYTES: AVAILABILITY BY ANALYTE, COHORT AND VISIT")
log("Concerns the exploratory analyses of §1.16-1.17 (methods §2.17). Unit: subjects.")
log("Source of the seven analytes: curated data cut; of albumin: biospecimen file,")
log(f'{ET_ALB} (label read from the PROJECTID column). Values are counted here at')
log("baseline and at the later visits. The identities with the pipeline concern the")
log('baseline samples (12, 20, 36, 100) and the count of the observations with the complete')
log("panel (23): where the comparison is on n alone, or is missing, this is stated row by row.")

ANALYTES = [
    ("abeta", "Abeta42 (Elecsys)", "Roche Elecsys"),
    ("tau", "t-tau (Elecsys)", "Roche Elecsys"),
    ("ptau", "p-tau181 (Elecsys)", "Roche Elecsys"),
    ("asyn", 'total alpha-synuclein (ELISA)', "ELISA"),
    ("IU_pTau181_CSF", "p-tau181 (Indiana)", "Indiana University"),
    ("IU_ABeta40_CSF", "Abeta40 (Indiana)", "Indiana University"),
    ("IU_ABeta42_CSF", "Abeta42 (Indiana)", "Indiana University"),
]
AV = [a for a, _, _ in ANALYTES]
IU = [a for a, _, p in ANALYTES if p == "Indiana University"]
ALB_TEST = {"alb_csf": ("CSF Albumin", 'CSF albumin', ET_ALB),
            "alb_plasma": ("Plasma Albumin", 'plasma albumin', ET_ALB)}

# baseline albumin per participant, as in 20 and 100: mean over the BL or SC rows
ab = alb[alb.CLINICAL_EVENT.isin(["BL", "SC"])]
A = pd.concat([ab[ab.TESTNAME == t].groupby("PATNO").v.mean().rename(k)
               for k, (t, _, _) in ALB_TEST.items()], axis=1)
A["qalb"] = A.alb_csf / A.alb_plasma
more_rows = int((ab.groupby(["PATNO", "TESTNAME"]).size() > 1).sum())
log(f'\nalbumin at baseline: {int(A.alb_csf.notna().sum())} subjects in the whole file; subjects '
    f'with more than one baseline row per test: {more_rows}')

p12 = load("12_kinetics_vs_concfactor.csv")
p20f = load("20_component_vs_albumin.csv")
p20k = load("20_kinetics_vs_albumin.csv")
p36 = load("36_loo_component.csv")
p100 = load("100_checks_cohort_primary.csv")

rows_csf = []       # table 1a: one row per cohort and item
rows_vis = []       # table 1b: one row per cohort, analyte and visit


def item(c, kind, name, n, used_in, check, detail=""):
    rows_csf.append(dict(cohort=c, n_cohort=N[c], kind=kind, item=name, n=int(n),
                          percentage=100 * n / N[c], units='subjects',
                          used_in=used_in, verified_against=check,
                          detail=detail))


for c, cfg in COHORTS.items():
    d = cohort[c][["PATNO"] + list(cfg["kin"])].merge(
        bl[AV + ["age_at_visit", "SEX"]].reset_index(), on="PATNO", how="left",
        validate="one_to_one").merge(A.reset_index(), on="PATNO", how="left",
                                     validate="one_to_one")
    # the kinetic parameters enter as logarithms, as in 12 and 20: the ranks would be
    # the same, but two triplicate means that differ by one rounding unit become a
    # tie after the logarithm, and the identity holds to the ninth decimal place
    for k in cfg["kin"]:
        d[k] = np.log(d[k])
    log(f"\n--- cohort {cfg['name']}, n = {N[c]} ---")
    assert d[["age_at_visit", "SEX"]].notna().all().all(), f"{c}: age or sex missing"

    # (a) baseline value, analyte by analyte
    log('with a value at the baseline visit:')
    nb = {a: int(d[a].notna().sum()) for a in AV}
    for a, et, platf in ANALYTES:
        log(f'  {et:32s} {share(nb[a], N[c])}')
        item(c, 'analyte', et, nb[a], "", "", f'variable {a}; {platf}')
    for k, (_, et, platf) in ALB_TEST.items():
        log(f'  {et:32s} {share(int(d[k].notna().sum()), N[c])}')
    log(f"  {'Qalb (CSF / plasma)':32s} {share(int(d.qalb.notna().sum()), N[c])}")

    # (b) panels
    n7 = int(d[AV].notna().all(axis=1).sum())
    n3 = int(d[IU].notna().all(axis=1).sum())
    used = [a for a in AV if nb[a] >= THRESHOLD_ANALYTES]
    assert len(used) >= 2, f'{c}: fewer than two analytes above the threshold'
    sub = d.dropna(subset=used).reset_index(drop=True)
    f, var1 = component(sub, used)
    sub["comp"] = f
    n_comp = len(sub)
    everyone7 = used == AV
    log(f'complete seven-analyte panel: {share(n7, N[c])}; Indiana three-analyte panel: '
        f'{share(n3, N[c])}')
    log(f"shared component (rule of 12 and 20: analytes with at least {THRESHOLD_ANALYTES} baseline "
        f"values in the cohort): {len(used)} analytes ({', '.join(used)}), n = {n_comp}, variance "
        f"explained by PC1 {var1:.3f}")
    if not everyone7:
        outside = {a: nb[a] for a in AV if a not in used}
        log(f'  analytes below the threshold, present but not used in this cohort: {outside}')

    # identity with 12: same n and same kinetic parameter-component correlation
    s12 = p12[p12.cohort == c]
    assert len(s12) == len(cfg["kin"]), f'{c}: unexpected rows in 12_kinetics_vs_concfactor'
    assert set(s12.n) == {n_comp}, (f'{c}: sample of the component n = {n_comp}, in 12 '
                                    f'{sorted(set(s12.n))}')
    for col, name in cfg["kin"].items():
        equals(partial_ranks(sub.comp, sub[col], sub[["age_at_visit", "SEX"]]),
               s12.loc[s12.kinetic == name, 'rho_partial'].iloc[0],
               f'{c}: partial rho {name}-component')
    log(f"  identity with 12_kinetics_vs_concfactor.csv: n and {len(cfg['kin'])} kinetic "
        f"parameter-component correlations coincide")
    ver7 = '12_kinetics_vs_concfactor.csv (n and rho)'
    if everyone7:
        assert set(p36.n) == {n7}, f'{c}: seven-analyte panel n = {n7}, in 36 {set(p36.n)}'
        log(f"  identity with 36_loo_component.csv: n = {n7} in all {len(p36)} pairs")
        ver7 += "; 36_loo_component.csv (n)"
    item(c, 'panel', 'full seven-analyte panel', n7,
         'shared component and leave-one-analyte-out (§1.16)' if everyone7
         else 'not used in this cohort (below the threshold of the component)',
         ver7 if everyone7 else "", 'all seven analytes at baseline')
    item(c, 'panel', 'three-analyte Indiana panel', n3,
         'shared component (§1.16)' if used == IU
         else 'subset of the panel of seven',
         ver7 if used == IU else "", 'p-tau181, Abeta40, Abeta42 from Indiana')
    item(c, 'panel', 'estimated shared component', n_comp,
         'adjustment for the component (§1.16, Table S12)', ver7,
         f"analytes: {', '.join(used)}; PC1 variance {var1:.4f}")

    # (c) albumin
    n_alb = int(d.alb_csf.notna().sum())
    n_q = int(d.qalb.notna().sum())
    s20 = p20k[p20k.cohort == c]
    assert set(s20.n) == {n_alb}, f'{c}: albumin n = {n_alb}, in 20 {sorted(set(s20.n))}'
    fr = d.dropna(subset=["alb_csf"])
    for col, name in cfg["kin"].items():
        equals(partial_ranks(fr.alb_csf, fr[col], fr[["age_at_visit", "SEX"]]),
               s20.loc[s20.kinetic == name, 'rho_partial'].iloc[0],
               f'{c}: partial rho {name}-albumin')
    ver_alb = '20_kinetics_vs_albumin.csv (n and rho)'
    if c == "24h":
        assert set(p100.n_alb) == {n_alb} and set(p100.n_Qalb) == {n_q}, \
            "24h: n of albumin or of Qalb differs from 100_checks_cohort_primary"
        ver_alb += "; 100_checks_cohort_primary.csv (n)"
    ver_q = "100_checks_cohort_primary.csv (n)" if c == "24h" else ""
    log(f"CSF albumin, with age and sex: n = {n_alb}; Qalb: n = {n_q}")
    log(f"  CSF albumin: n and {len(cfg['kin'])} partial rho identical to 20_kinetics_vs_albumin.csv"
        + ('; n identical to 100_checks_cohort_primary.csv' if c == "24h" else ""))
    log('  Qalb: ' + ("only n is compared, with 100_checks_cohort_primary.csv; the values are NOT "
                      "verified" if ver_q else
                      "NOT verified (no comparison in this script, either of n or of value)"))
    log("  plasma albumin: NOT verified (no comparison in this script, either of n or of value)")
    item(c, 'analyte', 'CSF albumin', n_alb,
         'albumin and kinetic parameters (§1.17, Table S13)', ver_alb,
         f'CSF Albumin, {ET_ALB}, BL or SC rows')
    item(c, 'analyte', 'plasma albumin', int(d.alb_plasma.notna().sum()),
         'denominator of Qalb (§1.17)', "", f"Plasma Albumin, {ET_ALB}")
    item(c, 'analyte', "Qalb", n_q, 'replacement of albumin with Qalb (§1.17)',
         ver_q, 'CSF / plasma albumin')

    # (d) component and albumin together
    both = sub.dropna(subset=["alb_csf"])
    s20f = p20f[p20f.cohort == c]
    assert len(s20f) == 1 and int(s20f.n.iloc[0]) == len(both), \
        f'{c}: component and albumin n = {len(both)}, in 20 {s20f.n.tolist()}'
    equals(stats.spearmanr(both.comp, both.alb_csf).statistic,
           s20f.rho_factor_vs_albumin_CSF.iloc[0], f'{c}: rho component-albumin')
    log(f"component and CSF albumin together: n = {len(both)}; identity with "
        f"20_component_vs_albumin.csv (n and rho)")
    item(c, 'panel', 'shared component and CSF albumin', len(both),
         'component-albumin comparison (§1.16)',
         '20_component_vs_albumin.csv (n and rho)')

    # (e) per visit. The table stays per visit (EVENT_ID). Each scheduled visit
    # has a single nominal year (checked at loading); the unscheduled visits have
    # one row for each nominal year to which the curated data cut assigns them.
    # Next to it, the count per nominal year, which includes the unscheduled
    # ones, as in the longitudinal and retention tables.
    L = cur[cur.PATNO.isin(P[c])].copy()
    assert not L.duplicated(["PATNO", 'year']).any(), \
        f'{c}: two visits of the same subject with the same nominal year'
    L['scheduled'] = ~L.EVENT_ID.isin(EVENTS_NO_SCHEDULED)
    PANEL = 'panel of seven'
    keys = ["EVENT_ID", 'year', 'scheduled']
    pres = L[AV].notna()
    pres[PANEL] = pres[AV].all(axis=1)
    pres[keys] = L[keys]
    cont = pres.groupby(keys)[AV + [PANEL]].sum().astype(int)
    cont_year = pres.groupby('year')[AV + [PANEL]].sum().astype(int)
    vis = L.groupby(keys).size().rename('n_visit').reset_index()
    vis = vis.sort_values(['year', 'scheduled', "EVENT_ID"],
                          ascending=[True, False, True]).reset_index(drop=True)
    vis_year = L.groupby('year').size()
    n_vis_of = {(r.EVENT_ID, int(r.year)): int(r.n_visit) for r in vis.itertuples()}
    for r in vis.itertuples():
        k = (r.EVENT_ID, r.year, r.scheduled)
        for a, et in [(a, et) for a, et, _ in ANALYTES] + [(PANEL, 'seven-analyte panel')]:
            rows_vis.append(dict(
                cohort=c, n_cohort=N[c], analyte=et,
                variable=a if a in AV else 'all seven', source="curated",
                visit=r.EVENT_ID,
                kind_visit='scheduled' if r.scheduled else 'not scheduled',
                year_nominal=int(r.year), n_having_visit=int(r.n_visit),
                n_having_value=int(cont.loc[k, a]),
                n_having_visit_year_nominal=int(vis_year[r.year]),
                n_having_value_year_nominal=int(cont_year.loc[r.year, a]),
                units='subjects'))
    # albumin: the year is that of the row of the curated data cut for the same
    # participant and the same visit; if the curated data cut has no such row, that
    # of the scheduled visit; an unscheduled visit with no row in the curated data
    # cut stays without a year
    sa = alb[alb.PATNO.isin(P[c]) & alb.v.notna()].merge(
        L[["PATNO", "EVENT_ID", 'year']].rename(columns={"EVENT_ID": "CLINICAL_EVENT"}),
        on=["PATNO", "CLINICAL_EVENT"], how="left", validate="many_to_one")
    sa['year'] = sa.year.fillna(sa.CLINICAL_EVENT.map(YEAR_OF))
    for k, (test, et, _) in ALB_TEST.items():
        st = sa[sa.TESTNAME == test]
        by_year = st.groupby('year').PATNO.nunique()
        for (ev, year), g in st.groupby(["CLINICAL_EVENT", 'year'], dropna=False):
            ha_year = pd.notna(year)
            rows_vis.append(dict(
                cohort=c, n_cohort=N[c], analyte=et, variable=test,
                source=f'biological samples, {ET_ALB}', visit=ev,
                kind_visit=('scheduled' if ev in YEAR_OF else
                             'not scheduled' if ev in EVENTS_NO_SCHEDULED else
                             'absent from the curated data cut'),
                year_nominal=int(year) if ha_year else pd.NA,
                n_having_visit=n_vis_of.get((ev, int(year)), 0) if ha_year else 0,
                n_having_value=int(g.PATNO.nunique()),
                n_having_visit_year_nominal=(int(vis_year.get(int(year), 0))
                                            if ha_year else pd.NA),
                n_having_value_year_nominal=(int(by_year[year]) if ha_year else pd.NA),
                units='subjects'))
    T = vis.merge(cont.reset_index(), on=keys, validate="one_to_one")
    T.insert(1, 'kind', np.where(T.scheduled, 'scheduled', 'not scheduled'))
    T = T[T[AV].sum(axis=1) > 0].drop(columns='scheduled')
    log("subjects with a value, by visit of the curated data cut (only the visits with at least one "
        "value). The table is by visit: the scheduled visits have a single nominal")
    log('year each, the unscheduled ones one row per nominal year:')
    log(T.rename(columns={"EVENT_ID": 'visit', **{
        a: a.replace("_CSF", "").replace("IU_", "IU ") for a in AV}})
        .to_string(index=False))
    # where the count per nominal year, which includes the unscheduled visits,
    # differs from that of the scheduled visit of the same year alone
    different = []
    for year in sorted(set(vis.year[~vis.scheduled])):
        kp = (EVENT_OF.get(int(year)), year, True)
        pieces = []
        for a in AV + [PANEL]:
            n_prog = int(cont.loc[kp, a]) if kp in cont.index else 0
            n_year = int(cont_year.loc[year, a])
            if n_year != n_prog:
                pieces.append(f"{a.replace('_CSF', '').replace('IU_', 'IU ')} {n_prog} -> {n_year}")
        if pieces:
            ev_p = EVENT_OF.get(int(year), 'no scheduled visit')
            different.append(f'year {int(year)} ({ev_p}): ' + ", ".join(pieces))
    n_np = int((~L.scheduled).sum())
    if different:
        log(f'unscheduled visits in this cohort: {n_np}. Subjects with a value, scheduled visit only '
            f'-> whole nominal year (unscheduled included), where the two counts differ:')
        for t in different:
            log("  " + t)
    else:
        log(f'unscheduled visits in this cohort: {n_np}, none with a value: for each analyte the '
            f'count per nominal year coincides with that of the scheduled visit.')
    without_year = int(sa.year.isna().sum())
    if without_year:
        log(f'albumin: {without_year} values at visits without a nominal year (unscheduled and with '
            f'no row in the curated data cut)')
    log('albumin, by visit of the biospecimen file: '
        + "; ".join(f"{ev} CSF {int(g[g.TESTNAME == 'CSF Albumin'].PATNO.nunique())}, plasma "
                    f"{int(g[g.TESTNAME == 'Plasma Albumin'].PATNO.nunique())}"
                    for ev, g in sa.groupby("CLINICAL_EVENT")))

    # (f) repeated complete panel
    full = L[L[AV].notna().all(axis=1)]
    per = full.groupby("PATNO").size()
    n_rip = int((per >= 2).sum())
    log(f'seven-analyte panel at any visit: {len(full)} observations of {len(per)} subjects; with at '
        f'least two complete visits: {n_rip}')
    ver = ""
    if everyone7:
        p23 = os.path.join(OUT, "23_review_P17_P19.txt")
        found = None
        if os.path.exists(p23):
            found = re.search(r"all seven analytes: (\d+) \((\d+) subjects\)",
                                open(p23).read())
        if found:
            assert (int(found.group(1)), int(found.group(2))) == (len(full), len(per)), \
                f'{c}: observations/subjects with the complete panel differ from 23'
            ver = '23_review_P17_P19.txt (counts)'
            log(f"  identity with {ver}")
        else:
            log('  no counterpart in the pipeline for this count: NOT verified')
    item(c, 'panel', 'panel of seven, observations at any visit', len(full),
         'stability of the component within subject (§1.16)' if everyone7 else "", ver,
         'unit: observations, not subjects')
    rows_csf[-1]['units'] = 'observations'
    rows_csf[-1]['percentage'] = np.nan
    item(c, 'panel', 'panel of seven, subjects with at least one complete visit',
         len(per), 'stability of the component within subject (§1.16)' if everyone7 else "",
         ver)
    item(c, 'panel', 'seven-analyte panel, subjects with at least two complete visits',
         n_rip, 'stability of the component within subject (§1.16)' if everyone7 else "", "")

# which visits the curated data cut carries for the participants of the two cohorts:
# computed, not hard-coded
_ev = cur[cur.PATNO.isin(P["24h"] | P["150h"])]
_prog_pres = sorted(set(_ev.EVENT_ID) - set(EVENTS_NO_SCHEDULED), key=YEAR_OF.get)
_years_pres = [YEAR_OF[e] for e in _prog_pres]
_np_pres = _ev[_ev.EVENT_ID.isin(EVENTS_NO_SCHEDULED)].groupby("EVENT_ID").size()
log(f"\nVisits of the curated data cut for the subjects of the two cohorts: {len(_prog_pres)} "
    f"scheduled visits ({', '.join(_prog_pres)}), "
    + (f'one for each nominal year from {_years_pres[0]} to {_years_pres[-1]}'
       if _years_pres == list(range(_years_pres[0], _years_pres[-1] + 1))
       else f'at nominal years {_years_pres}')
    + ('; unscheduled visits: '
       + ", ".join(f'{ev} ({int(k)} rows)' for ev, k in _np_pres.items())
       if len(_np_pres) else '; no unscheduled visit') + ".")
_alb_outside = sorted(set(alb.CLINICAL_EVENT[alb.PATNO.isin(P["24h"] | P["150h"])])
                    - set(_ev.EVENT_ID))
log("Albumin visits in the biospecimen file that the curated data cut does not carry for these "
    "subjects: " + (", ".join(_alb_outside) if _alb_outside else 'none') + ".")
log('Limitation: the seven analytes are read from the curated data cut alone. The biospecimen file')
log("is tabulated by visit in this script for albumin alone: whether for the seven")
log("analytes it holds results at visits that the curated data cut does not carry is NOT verified.")

# ------------------------------------------------------------------ 2. TIMING
title('2. TIMING AT BASELINE: LUMBAR PUNCTURE, CLINICAL VISIT, DAT-SPECT, ASSAY RUN')
log("Concerns all the analyses: defines what «baseline» means. Unit: subjects.")
age_col = list(TIMES_VAR)
# all the visits of the participants of the two cohorts, not BL alone: the
# intervals at the later visits (DAT-SPECT) are rounded to the day as well.
# The ages of the curated data cut are days / 365: if one is not, the script stops
# and says which, because thresholds and quantiles in days would no longer make sense.
_age = cur.loc[cur.PATNO.isin(P["24h"] | P["150h"]), ["PATNO", "EVENT_ID"] + age_col]
_outside = _age[age_col].apply(
    lambda s: s.notna() & ((s * 365 - (s * 365).round()).abs() >= 1e-6)).any(axis=1)
assert not _outside.any(), (
    "ages of the curated data cut that are not a whole number of days / 365 (PATNO, EVENT_ID): "
    + ", ".join(f"({int(p)}, {e})" for p, e in
                _age.loc[_outside, ["PATNO", "EVENT_ID"]].head(10).itertuples(index=False))
    + f'; {int(_outside.sum())} rows in all')
DAYS = 365.0
log(f'The ages of the curated data cut multiplied by 365 are all integers in the {len(_age)} visits '
    f'of the subjects of the two cohorts ({int(_age[age_col].notna().sum().sum())} values; the '
    f'script stops if one is not): they are days / 365. The intervals in days are age differences x '
    f'{DAYS:g}, rounded to the whole day before thresholds and quantiles.')
log("The visit has month and year only: the intervals come from the ages, not from the dates.")


def days(delta_age):
    """A difference of ages in whole days (the ages are days / 365).

    Without the rounding the difference carries a noise of 1e-12 and a participant
    at exactly 90 days would be counted "beyond 90 days".
    """
    g = pd.Series(delta_age, dtype=float) * DAYS
    return g.round() + 0.0            # + 0.0: no negative zero in the report


def state_check(c, variables, extra=""):
    """Verification label of a row, from the variables that make it up.

    It says for which variables a comparison with the pipeline was actually
    carried out, and which one (`verified` register), and for which not.
    """
    groups, no = {}, []
    for v in variables:
        et = verified[c].get(v)
        if et:
            groups.setdefault(et, []).append(v)
        else:
            no.append(v)
    t = [f"{et}: {', '.join(vs)}" for et, vs in groups.items()]
    if no:
        t.append(f"NOT verified: {', '.join(no)}")
    if extra:
        t.append(extra)
    return "; ".join(t)


# enrolment phase: comparison with 70_baseline.csv if it carries the counts
def crosscheck_phase(c, counts):
    """True if 70_baseline.csv carries the counts by phase and they coincide; False if
    it does not carry them. It stops if it carries them and they differ, or if they cannot be read."""
    rows = [i for i in p70.index if re.search('phase|phase', str(i), re.I)]
    if not rows or len(col70[c]) != 1:
        return False
    for ph, k in counts.items():
        r = [i for i in rows if re.search(f'(phase|phase)\\s*{ph}\\b', str(i), re.I)]
        assert len(r) == 1, (f'70_baseline.csv names the enrolment phase but the row of phase {ph} '
                             f'is not recognisable: {rows}')
        m = re.match(r"\s*(\d+)", str(p70.loc[r[0], col70[c][0]]))
        assert m and int(m.group(1)) == int(k), (
            f'{c}: phase {ph} n = {k}, in 70_baseline.csv {p70.loc[r[0], col70[c][0]]}')
    return True


for c in COHORTS:
    b = bl.loc[sorted(P[c])]
    verified[c]["enroll_phase"] = (
        'counts by phase identical to 70_baseline.csv'
        if crosscheck_phase(c, b.enroll_phase.value_counts().sort_index().to_dict())
        else "")
log('Enrolment phase (enroll_phase): '
    + ('counts per phase identical to 70_baseline.csv in both cohorts'
       if all(verified[c]["enroll_phase"] for c in COHORTS) else
       '70_baseline.csv does not carry its counts: NOT verified against the pipeline')
    + '. Visit date (visit_date): no pipeline file carries it, NOT verified.')
if dic is not None:
    _df = dic[dic.variable == "enroll_phase"]
    log('  phases according to the dictionary of the curated data cut: '
        + "; ".join(f'{one_row(k)} = «{one_row(v)}»' for k, v in zip(_df.Code, _df.Decode)))
else:
    log('  meaning of the phases NOT verified: dictionary of the curated data cut not found.')

rows_times = []


def row_timeval(c, name, x, units, n_att, dec=0, thresholds=True, remark="", check=""):
    x = pd.Series(x, dtype=float)
    q = quantiles(x)
    r = dict(cohort=c, n_cohort=N[c], interval=name, units=units, n_expected=int(n_att),
             n=q["n"], missing=int(n_att - q["n"]), median=q['median'], q1=q["q1"],
             q3=q["q3"], minimum=q['minimum'], maximum=q['maximum'],
             n_zero=np.nan, n_negatives=np.nan, n_positives=np.nan, remark=remark,
             check=check)
    for g in THRESHOLDS_DAYS:
        r[f'n_beyond_{g}_days'] = np.nan
    if thresholds:
        v = x.dropna()
        r.update(n_zero=int((v == 0).sum()), n_negatives=int((v < 0).sum()),
                 n_positives=int((v > 0).sum()))
        for g in THRESHOLDS_DAYS:
            r[f'n_beyond_{g}_days'] = int((v.abs() > g).sum())
    rows_times.append(r)
    t = f"  {name}: n = {q['n']}, missing {r['missing']}; {fq(q, dec)} {units}"
    if thresholds:
        t += (f"; equal to zero {r['n_zero']}, before {r['n_negatives']}, after {r['n_positives']}; "
              f"beyond "
              + ", ".join(f"{g} days: {r[f'n_beyond_{g}_days']}" for g in THRESHOLDS_DAYS))
    log(t)
    return r


end_data = cur.data_visit.max()
month_end = int(cur.month.max())
times = {}
for c, cfg in COHORTS.items():
    b = bl.loc[sorted(P[c])]
    log(f"\n--- cohort {cfg['name']}, n = {N[c]} ---")
    log('  verification of the time variables of this cohort: '
        + state_check(c, list(TIMES_VAR) + ["RUNDATE", "visit_date", "enroll_phase"]))
    lp = days(b.age_LP - b.age_at_visit)
    dat = days(b.age_datscan - b.age_at_visit)
    r_lp = row_timeval(c, 'baseline lumbar puncture minus baseline clinical visit', lp,
                      'days', N[c], remark='positive = lumbar puncture after the visit',
                      check=state_check(c, ["age_LP", "age_at_visit"]))
    r_dat = row_timeval(c, 'baseline DAT-SPECT minus baseline clinical visit', dat, 'days',
                       N[c], remark='positive = DAT after the visit',
                       check=state_check(c, ["age_datscan", "age_at_visit"]))
    # the same interval among only the participants with the scan before the visit: the
    # signed median over everyone is NOT the median lead of those who had it earlier
    earliest_vis = dat[dat < 0]
    row_timeval(c, 'lead of the baseline DAT-SPECT over the baseline clinical visit, only subjects '
                  'with DAT before the visit', -earliest_vis, 'days',
               len(earliest_vis), thresholds=False,
               remark='subset: subjects with DAT-SPECT before the visit; positive = days in advance',
               check=state_check(c, ["age_datscan", "age_at_visit"]))
    r_dl = row_timeval(c, 'baseline DAT-SPECT minus baseline lumbar puncture',
                      days(b.age_datscan - b.age_LP), 'days', N[c],
                      remark='positive = DAT after the lumbar puncture',
                      check=state_check(c, ["age_datscan", "age_LP"]))
    if r_lp['missing']:
        log(f"    {r_lp['missing']} subjects have an assay result at BL in the raw file but no age "
            f"at puncture in the curated data cut")
    log('    the puncture falls '
        + ("on the same day as the visit in more than half of the subjects"
           if r_lp["n_zero"] > r_lp["n"] / 2 else
           "on a different day from the visit in more than half of the subjects")
        + f" ({share(r_lp['n_zero'], r_lp['n'])}); the DAT-SPECT "
        + ('precedes' if r_dat['n_negatives'] > r_dat["n"] / 2 else 'does not precede')
        + f" the visit in the majority ({share(r_dat['n_negatives'], r_dat['n'])})")

    # storage: from sample collection to the assay run. Those without the age at
    # puncture have no puncture date: it stays missing, it is not imputed
    data_lp = (b.data_visit + pd.Timedelta(days=14)
               + pd.to_timedelta(lp.round(), unit="D"))
    cons = (sample[c].reindex(b.index) - data_lp).dt.days / 365.25
    REMARK_CONS = 'date of the puncture approximated to mid-month (+/- 16 days)'

    def remark_cons(idx):
        k = int(lp.reindex(idx).isna().sum())
        return REMARK_CONS + (f'; {k} without age at lumbar puncture in the curated data cut: '
                            f'excluded, no imputation' if k else "")

    var_cons = ["RUNDATE", "age_LP", "age_at_visit", "visit_date"]
    r_cons = row_timeval(c, 'assay run minus baseline lumbar puncture', cons, 'years',
                        N[c], dec=2, thresholds=False, remark=remark_cons(b.index),
                        check=state_check(c, var_cons))
    if r_cons['missing']:
        log(f"    {r_cons['missing']} subjects without an interval: age at puncture missing in the "
            f"curated data cut; excluded, no imputation")
    times[c] = dict(cons=quantiles(cons), phases={})
    # storage follows the enrolment phase: it is reported by phase
    for ph, g in b.groupby("enroll_phase"):
        times[c]['phases'][ph] = row_timeval(
            c, f'assay run minus baseline lumbar puncture, enrolment phase {ph}',
            cons.reindex(g.index), 'years', len(g), dec=2, thresholds=False,
            remark=remark_cons(g.index),
            check=state_check(c, var_cons + ["enroll_phase"]))

    # treatment at the baseline visit
    trt = pd.to_numeric(b.PDTRTMNT, errors="coerce")
    n_trt, n_trt_na = int((trt == 1).sum()), int(trt.isna().sum())
    ledd_pos = pd.to_numeric(b.LEDD, errors="coerce") > 0
    n_ledd = int(ledd_pos.sum())
    n_one_o_other = int(((trt == 1) | ledd_pos).sum())
    log(f'  dopaminergic treatment at the baseline visit (PDTRTMNT = 1): {share(n_trt, N[c])}, '
        f'missing {n_trt_na}; LEDD > 0 at baseline: {n_ledd}')
    if n_one_o_other != n_trt:
        log(f'    the two variables do not count the same set: {n_ledd} subjects with LEDD > 0, '
            f'{n_trt} with PDTRTMNT = 1, {int(((trt == 1) & ledd_pos).sum())} with both; with one or '
            f'the other {share(n_one_o_other, N[c])}')
    # the count of treated participants has a counterpart in Table S2 of the pipeline
    row_trt = [i for i in p70.index
                if str(i).startswith("On dopaminergic treatment at baseline")]
    ver_trt = 'NOT verified: PDTRTMNT'
    if len(row_trt) == 1 and len(col70[c]) == 1:
        m = re.match(r"\s*(\d+)", str(p70.loc[row_trt[0], col70[c][0]]))
        assert m and int(m.group(1)) == n_trt, (
            f'{c}: {n_trt} treated at baseline, in 70_baseline.csv {p70.loc[row_trt[0], col70[c][0]]}')
        ver_trt = "70_baseline.csv (n)"
    log('    PDTRTMNT = 1 at baseline: '
        + ('n identical to 70_baseline.csv' if ver_trt.startswith("70_") else ver_trt)
        + '; LEDD > 0: NOT verified (no comparison in this script)')
    rows_times.append(dict(cohort=c, n_cohort=N[c],
                            interval='on dopaminergic treatment at baseline (PDTRTMNT = 1)',
                            units='subjects', n_expected=N[c], n=n_trt, missing=n_trt_na,
                            check=ver_trt))
    rows_times.append(dict(cohort=c, n_cohort=N[c], interval='LEDD > 0 at baseline',
                            units='subjects', n_expected=N[c], n=n_ledd,
                            missing=int(pd.to_numeric(b.LEDD, errors="coerce").isna().sum()),
                            check='NOT verified: LEDD'))
    rows_times.append(dict(cohort=c, n_cohort=N[c],
                            interval='PDTRTMNT = 1 or LEDD > 0 at baseline',
                            units='subjects', n_expected=N[c], n=n_one_o_other,
                            check='NOT verified: LEDD'))

    # nominal year versus elapsed time
    L = cur[cur.PATNO.isin(P[c]) & (cur.year > 0)]
    elapsed = L.age_at_visit - L.PATNO.map(b.age_at_visit)
    departure = elapsed - L.year
    q = quantiles(departure)
    distant = int((departure.abs() > 0.5).sum())
    log(f"  time elapsed since the baseline visit minus nominal year of the visit, over the {q['n']} "
        f"later visits: {fq(q, 2)} years; visits more than six months from the nominal year: "
        f"{share(distant, q['n'])}")
    rows_times.append(dict(cohort=c, n_cohort=N[c],
                            interval='time elapsed since baseline minus nominal year',
                            units='years (unit: visits after baseline)',
                            n_expected=len(L), n=q["n"], missing=len(L) - q["n"],
                            median=q['median'], q1=q["q1"], q3=q["q3"],
                            minimum=q['minimum'], maximum=q['maximum'],
                            remark=f'visits more than 0.5 years from the nominal year: {distant}',
                            check='NOT verified: age_at_visit at the visits after baseline'))
    # DAT at the later visits
    Ld = L[L.MIA_PUTAMEN_BILAT.notna()]
    qd = quantiles(days(Ld.age_datscan - Ld.age_at_visit))
    log(f"  DAT-SPECT after baseline: {len(Ld)} scans; scan minus visit {fq(qd)} days (age at scan "
        f"missing in {len(Ld) - qd['n']})")
    rows_times.append(dict(cohort=c, n_cohort=N[c],
                            interval='subsequent DAT-SPECT minus visit of the same year',
                            units='days (unit: examinations after baseline)',
                            n_expected=len(Ld), n=qd["n"], missing=len(Ld) - qd["n"],
                            median=qd['median'], q1=qd["q1"], q3=qd["q3"],
                            minimum=qd['minimum'], maximum=qd['maximum'],
                            check='NOT verified: age_datscan, age_at_visit at the visits after '
                                     'baseline'))

log("\nThe time of the longitudinal models is the nominal year of the visit (0, 1, 2, ...),")
log("not the elapsed time: the difference between the two is the one reported above.")

# ------------------------------------------------------------------ 3. OUTCOMES
title('3. LONGITUDINAL OUTCOMES: SUBJECTS, OBSERVATIONS, FOLLOW-UP')
log("Sample of each row: that of the model, i.e. the visits with the outcome and the")
log("covariates (baseline age, sex, disease duration; education for the cognitive")
log('outcomes) not missing. The kinetic parameters are complete in all the subjects of the')
log('cohorts (verified at point 0), so the sample does not depend on the kinetic parameter.')
log("HOW BASELINE IS COUNTED. Year 0 is the BL visit, i.e. the visit of the sample collection")
log("for the assay. The «total» columns include the observation at year 0, as the")
log('models do; the «after sample collection» columns count only the observations at years >= 1.')

# (variable, label, block, instruction among the covariates)
OUTCOMES = [
    ("updrs3_score", "MDS-UPDRS III", 'motor', False),
    ("updrs_totscore", 'MDS-UPDRS total', 'motor', False),
    ("updrs2_score", "MDS-UPDRS II", 'motor', False),
    ("MIA_PUTAMEN_BILAT", 'DAT bilateral putamen', "DAT", False),
    ("MIA_CAUDATE_BILAT", 'DAT bilateral caudate', "DAT", False),
    ("MIA_STRIATUM_BILAT", 'DAT bilateral striatum', "DAT", False),
    ("moca", "MoCA", 'primary non-motor', True),
    ("hvlt_immediaterecall", 'HVLT immediate recall', 'primary non-motor', True),
    ("SDMTOTAL", "Symbol Digit", 'primary non-motor', True),
    ("VLTANIM", 'Semantic fluency', 'primary non-motor', True),
    ("bjlot", "Benton JLO", 'primary non-motor', True),
    ("updrs1_score", "MDS-UPDRS I", 'primary non-motor', False),
    ("gds", "GDS-15", 'primary non-motor', False),
    ("stai", "STAI", 'primary non-motor', False),
    ("ess", "Epworth", 'primary non-motor', False),
    ("rem", "RBDSQ", 'primary non-motor', False),
    ("scopa", "SCOPA-AUT", 'primary non-motor', False),
    ("quip", "QUIP", 'primary non-motor', False),
    ("COG_COMPOSITE_INT", 'Cognitive composite', 'extended battery', True),
    ("TMT_B", "Trail Making B", 'extended battery', True),
    ("lexical", 'Lexical fluency', 'extended battery', True),
    ("MODBNT", "Boston Naming", 'extended battery', True),
    ("clockdraw", "Clock Drawing", 'extended battery', True),
    ("nqol_cogns", 'NeuroQoL cognition', 'extended battery', True),
    ("MSEADLG", "Schwab & England", 'extended battery', False),
]
assert len({v for v, *_ in OUTCOMES}) == len(OUTCOMES)

# pipeline outputs against which the comparison is made
g84 = load("84_gradients.csv")
sources84 = [("84_gradients.csv", g84)]
if os.path.exists(os.path.join(CAUDATE, "84_gradients.csv")):
    # the caudate working directory must rest on the same participants as this one
    for c in COHORTS:
        dc = load_cohort(CAUDATE, c).set_index("PATNO")
        d0 = cohort[c].set_index("PATNO")
        assert set(dc.index) == P[c], f'run_caudate, {c}: subjects differ from this working directory'
        k0 = list(COHORTS[c]["kin"])[0]
        assert np.allclose(dc.loc[d0.index, k0].to_numpy(float), d0[k0].to_numpy(float),
                           rtol=1e-12, atol=0), f'run_caudate, {c}: kinetic parameters differ'
    sources84.append(('run_caudate/outputs/84_gradients.csv',
                    load("84_gradients.csv", CAUDATE)))
    log('\nworking directory run_caudate found: same subjects and same kinetic parameters as this '
        'working directory.')
else:
    log('\nworking directory run_caudate missing: the caudate counts are NOT verified.')
p72 = load("72_mixed_models.csv")
p15 = {'primary non-motor': load("15_primary_nonmotor.csv"),
       'extended battery': load("15_secondary_nonmotor.csv")}
p78 = load("78_longitudinal_sensitivity.csv")

rows_outcomes = []
rows_year = []
checks = dict(counts=0, values=0)
last_any = {}
for c, cfg in COHORTS.items():
    cov0 = bl.loc[sorted(P[c]), ["age_at_visit", "SEX", "duration_yrs", "EDUCYRS"]]
    L = cur[cur.PATNO.isin(P[c])].merge(
        cov0.rename(columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl",
                             "SEX": "sex_bl", "EDUCYRS": "edu_bl"}).reset_index(),
        on="PATNO", how="left", validate="many_to_one")
    assert not L.duplicated(["PATNO", 'year']).any(), \
        f'{c}: two visits of the same subject with the same nominal year'
    last_any[c] = L.groupby("PATNO").year.max()
    for var, et, block, edu in OUTCOMES:
        cov = ["age_bl", "sex_bl", "dur_bl"] + (["edu_bl"] if edu else [])
        m = L[["PATNO", 'year', var] + cov].dropna()
        per = m.groupby("PATNO").size().reindex(sorted(P[c]), fill_value=0)
        within = per[per > 0]
        post = (m[m.year > 0].groupby("PATNO").size()
                .reindex(within.index, fill_value=0))
        ult = m.groupby("PATNO").year.max()
        n_s, n_o = int(len(within)), int(len(m))
        qt, qp, qf = quantiles(within), quantiles(post), quantiles(ult)
        r = dict(cohort=c, n_cohort=N[c], outcome=var, label=et, block=block,
                 covariate='age, sex, duration' + (', education' if edu else ""),
                 n_subj=n_s, n_obsv=n_o,
                 n_subj_having_baseline=int((m.year == 0).sum()),
                 n_obsv_after_sampling=int((m.year > 0).sum()),
                 n_subj_having_obsv_after_sampling=int((post >= 1).sum()),
                 obsv_by_subj_median=qt['median'], obsv_by_subj_q1=qt["q1"],
                 obsv_by_subj_q3=qt["q3"], obsv_by_subj_min=qt['minimum'],
                 obsv_by_subj_max=qt['maximum'],
                 n_subj_atleast_2_obsv=int((within >= 2).sum()),
                 n_subj_atleast_3_obsv=int((within >= 3).sum()),
                 obsv_after_by_subj_median=qp['median'], obsv_after_by_subj_q1=qp["q1"],
                 obsv_after_by_subj_q3=qp["q3"], obsv_after_by_subj_min=qp['minimum'],
                 obsv_after_by_subj_max=qp['maximum'],
                 n_subj_atleast_2_obsv_after=int((post >= 2).sum()),
                 n_subj_atleast_3_obsv_after=int((post >= 3).sum()),
                 followup_years_median=qf['median'], followup_years_q1=qf["q1"],
                 followup_years_q3=qf["q3"], followup_years_max=qf['maximum'])
        for t in YEARS_KEY:
            r[f'n_subj_latest_obsv_ad_atleast_{t}_years'] = int((ult >= t).sum())

        # ---- identity with the pipeline
        analysis, crosschecks = [], []
        for name84, G in sources84:
            s = G[(G.cohort == c) & (G.outcome == var)]
            if not len(s):
                continue
            assert s.n_subjects.nunique() == 1 and s.n_obs.nunique() == 1, \
                f'{c}, {var}: in {name84} the sample changes with the kinetic parameter'
            assert (int(s.n_subjects.iloc[0]), int(s.n_obs.iloc[0])) == (n_s, n_o), (
                f'{c}, {var}: here {n_s} subjects and {n_o} observations, in {name84} '
                f'{int(s.n_subjects.iloc[0])} and {int(s.n_obs.iloc[0])}')
            equals(m[var].std(ddof=1), s.outcome_sd.iloc[0],
                   f'{c}, {var}: standard deviation of the outcome ({name84})')
            checks['counts'] += 1
            checks['values'] += 1
            analysis.append(f'kinetic parameter x time interactions, {len(s)} models (§1.12)')
            crosschecks.append(f'{name84} (n, observations, SD of the outcome)')
        s = p72[(p72.cohort == c) & (p72.outcome == var)]
        if len(s):
            assert set(s.n_multivisit) == {r['n_subj_atleast_2_obsv']}, \
                f'{c}, {var}: subjects with at least 2 observations differ from 72'
            checks['counts'] += 1
            crosschecks.append('72_mixed_models.csv (subjects with at least 2 observations)')
        if c == "150h" and block in p15:
            s = p15[block][p15[block].outcome == var]
            assert len(s) == 1, f"{var}: missing or repeated in the output of 15"
            assert (int(s.n_subj.iloc[0]), int(s.n_obsv.iloc[0])) == (n_s, n_o), (
                f'150h, {var}: here {n_s} subjects and {n_o} observations, in 15 '
                f'{int(s.n_subj.iloc[0])} and {int(s.n_obsv.iloc[0])}')
            checks['counts'] += 1
            analysis.append('twelve primary non-motor outcomes (§1.13, Table S9)'
                           if block == 'primary non-motor'
                           else 'extended neuropsychological battery (§1.13)')
            crosschecks.append(f"15_{('primary' if block == 'primary non-motor' else 'secondary')}_nonmotor.csv "
                             f"(n, observations)")
        # the same analysis found in two working directories (84 and the caudate 84) is
        # written once
        analysis = list(dict.fromkeys(analysis))
        r['analysis'] = "; ".join(analysis) if analysis else 'not modelled in this cohort'
        r['verified_against'] = "; ".join(crosschecks)
        rows_outcomes.append(r)
        for t, k in m.groupby('year').size().items():
            rows_year.append(dict(cohort=c, n_cohort=N[c], outcome=var, label=et,
                                   block=block, year=int(t), n_subj_having_observation=int(k)))

E = pd.DataFrame(rows_outcomes)
# every outcome modelled by the pipeline must have found its counterpart
for name84, G in sources84:
    expected = set(map(tuple, G[["cohort", "outcome"]].drop_duplicates().to_numpy()))
    assert expected <= set(zip(E.cohort, E.outcome)), f'outcomes of {name84} not covered here'
for block, d15 in p15.items():
    assert set(d15.outcome) == set(E[(E.cohort == "150h") & (E.block == block)].outcome), \
        f"the outcomes of block '{block}' are not those of the output of 15"
log(f"\nIdentity with the pipeline: {checks['counts']} count comparisons and {checks['values']} "
    f"value comparisons (standard deviation of the outcome in the model sample), all passed.")
log('For the outcomes estimated by script 15 alone the pipeline keeps subjects and')
log("observations but no sample statistic: there the comparison is on the two")
_vc = E.verified_against.astype(str)
_both = sorted(set(E[_vc.str.contains("15_", regex=False)
                         & _vc.str.contains('SD of the outcome', regex=False)].label))
log("counts. Among the outcomes of script 15 the following also have the value comparison, because "
    "they are also among the kinetic parameter x time models: "
    + (", ".join(_both) if _both else 'none') + ".")

# retention on the MoCA sample, as in 78
s78 = p78[(p78.section == "C") & (p78.analysis == 'retention') & (p78.model == 'everyone')]
rm = E[(E.cohort == "150h") & (E.outcome == "moca")].iloc[0]
for t in YEARS_KEY:
    s = s78[s78.term == f'at risk at {t} years']
    assert len(s) == 1 and int(s.n_subj.iloc[0]) == rm.n_subj and \
        int(s.estimate.iloc[0]) == rm[f'n_subj_latest_obsv_ad_atleast_{t}_years'], \
        f'MoCA, 150h: subjects still observed at {t} years differ from 78'
s = s78[s78.term == 'median follow-up (years)']
assert len(s) == 1 and float(s.estimate.iloc[0]) == rm.followup_years_median, \
    'MoCA, 150h: median follow-up differs from 78'
log(f"Identity with 78_longitudinal_sensitivity.csv: median follow-up of the MoCA and subjects still "
    f"observed at {', '.join(map(str, YEARS_KEY))} years coincide.")

columns_log = ['label', 'n_subj', 'n_obsv', 'n_obsv_after_sampling',
               'n_subj_having_obsv_after_sampling', 'obsv_by_subj', 'n_subj_atleast_2_obsv',
               'n_subj_atleast_3_obsv', "followup"]
for c, cfg in COHORTS.items():
    s = E[E.cohort == c].copy()
    s['obsv_by_subj'] = [f"{a:g} [{b:g}; {d:g}] {e:g}..{f:g}" for a, b, d, e, f in zip(
        s.obsv_by_subj_median, s.obsv_by_subj_q1, s.obsv_by_subj_q3,
        s.obsv_by_subj_min, s.obsv_by_subj_max)]
    s["followup"] = [f"{a:g} [{b:g}; {d:g}]" for a, b, d in zip(
        s.followup_years_median, s.followup_years_q1, s.followup_years_q3)]
    log(f"\n--- cohort {cfg['name']}, n = {N[c]} ---")
    log('(obsv_by_subj = total observations per subject, median [IQR] min..max; followup = year of '
        'the last observation of the outcome, median [IQR])')
    log(s[columns_log].rename(columns={
        'n_obsv_after_sampling': 'obsv_after', 'n_subj_having_obsv_after_sampling': 'subj_after',
        'n_subj_atleast_2_obsv': '>=2_obsv', 'n_subj_atleast_3_obsv': '>=3_obsv'})
        .to_string(index=False))
    no_mod = s[s.analysis == 'not modelled in this cohort']
    log(f"outcomes described but not modelled in this cohort: {len(no_mod)} "
        f"({(', '.join(no_mod.label) if len(no_mod) else 'noone')})")
    without = s.n_subj - s.n_subj_having_obsv_after_sampling
    worse = s.loc[without.idxmax()]
    log(f'subjects with no observation after sample collection: from {int(without.min())} to '
        f'{int(without.max())} depending on the outcome (maximum: {worse.label})')
    d = s[s.block == "DAT"].iloc[0]
    log(f'DAT-SPECT: {int(d.n_subj_having_obsv_after_sampling)} subjects out of {int(d.n_subj)} have '
        f'at least one scan after baseline, {int(d.n_subj_atleast_3_obsv)} at least three scans in '
        f'all; last scan at year {d.followup_years_median:g} [{d.followup_years_q1:g}; '
        f'{d.followup_years_q3:g}]')
    years_dat = (pd.DataFrame(rows_year).query('cohort == @c and outcome == @d.outcome')
                .set_index('year').n_subj_having_observation)
    log('  scans per nominal year: '
        + ", ".join(f'year {t}: {k}' for t, k in years_dat.items()))
    # scans at the scheduled years and off schedule
    tot_dat = int(years_dat.sum())
    by_prog = {t: int(years_dat.get(t, 0)) for t in YEARS_DAT_SCHEDULED}
    outside = years_dat[~years_dat.index.isin(YEARS_DAT_SCHEDULED)]
    n_in, n_outside = sum(by_prog.values()), int(outside.sum())
    et_prog = ", ".join(map(str, YEARS_DAT_SCHEDULED))
    log(f'  at the scheduled years ({et_prog}: calendar declared in this script, not derived from '
        f'the data) {share(n_in, tot_dat)} scans; off calendar {share(n_outside, tot_dat)}'
        + (": " + ", ".join(f'year {t}: {k}' for t, k in outside.items()) if n_outside else ""))
    ver_dat = ('total of the examinations: sample of the model, '
               + (d.verified_against if d.verified_against else 'NOT verified')
               + '; breakdown by year: NOT verified')
    rows_times.append(dict(
        cohort=c, n_cohort=N[c], interval=f'DAT-SPECT at the scheduled years ({et_prog})',
        units='examinations (sample of the DAT model, all visits)', n_expected=tot_dat,
        n=n_in, remark='by nominal year: ' + "; ".join(f'year {t}: {k}'
                                                       for t, k in by_prog.items()),
        check=ver_dat))
    rows_times.append(dict(
        cohort=c, n_cohort=N[c], interval='DAT-SPECT outside the scheduled years',
        units='examinations (sample of the DAT model, all visits)', n_expected=tot_dat,
        n=n_outside, remark=('by nominal year: ' + "; ".join(f'year {t}: {k}'
                                                           for t, k in outside.items())
                         if n_outside else 'noone'),
        check=ver_dat))

# ------------------------------------------------------------------ 3bis. S2
title('3bis. THE FOLLOW-UP OF TABLE S2')
log("Definition used by 70_baseline_table.py: nominal year of the last visit")
log("present in the curated data cut, whatever data it holds. Unit: subjects.")
row_fu = [i for i in p70.index if i.startswith("Follow-up (years)")]
row_10 = [i for i in p70.index if i.startswith("Follow-up beyond 10")]
assert len(row_fu) == 1 and len(row_10) == 1, 'follow-up rows not found in 70'
fu_summary = {}
for c, cfg in COHORTS.items():
    assert len(col70[c]) == 1, f'column of cohort {c} not found in 70_baseline.csv'
    f = last_any[c]
    text = f"{f.median():.1f} [{f.quantile(.25):.1f}–{f.quantile(.75):.1f}]"
    n_ge, n_gt = int((f >= 10).sum()), int((f > 10).sum())
    assert text == p70.loc[row_fu[0], col70[c][0]], (
        f'{c}: follow-up {text}, in 70 {p70.loc[row_fu[0], col70[c][0]]}')
    assert n_ge == int(p70.loc[row_10[0], col70[c][0]]), (
        f'{c}: {n_ge} subjects at 10 years or more, in 70 {p70.loc[row_10[0], col70[c][0]]}')
    only_bl = int((f == 0).sum())
    fu_summary[c] = dict(text=text, n_ge=n_ge, n_gt=n_gt, only_bl=only_bl)
    log(f'  {c}: median follow-up [IQR] {text} years; last visit at year 10 or beyond: {n_ge}; '
        f'strictly beyond year 10: {n_gt}; with the baseline visit only: {only_bl}. Identical to '
        f'70_baseline.csv.')
    if n_ge != n_gt:
        log(f"       the row «Followed beyond 10 years» of Table S2 counts {n_ge}, i.e. those who "
            f"reached year 10 (>= 10); beyond year 10 there are {n_gt}")

# ------------------------------------------------------------------ 3ter. UPDRS III
title('3ter. MDS-UPDRS III: VISITS WITHOUT THE MODELLED SCORE, AND TREATMENT')
assert {VAR_II, VAR_III} <= {v for v, *_ in OUTCOMES}, \
    'the variables of block 3ter are not among the outcomes of point 3'
log("Unit: visits after baseline (nominal year >= 1) of the subjects of each")
log('cohort; all the rows of the curated data cut, not the model sample.')
for _c in COHORTS:
    _an = E[(E.cohort == _c) & (E.outcome == VAR_III)].analysis.iloc[0]
    log(f"  {_c}: {VAR_III} "
        + (f"is the part III variable modelled in this working directory ({_an})"
           if _an != 'not modelled in this cohort'
           else 'is NOT found to be modelled in this working directory: the counts that follow '
                'describe the variable, not the sample of a model'))
log(f'The curated data cut also carries {VAR_III_ON}.')
if dic is not None:
    log(f'Definitions in the dictionary of the curated data cut ({os.path.basename(dic_p)}, sheet '
        f'«{SHEET_DICTIONARY}»), quoted:')
    descr = {}
    for v in (VAR_III, VAR_III_ON, "PDTRTMNT"):
        r = dic[dic.variable == v]
        assert len(r), f'{v} missing from the dictionary of the curated data cut'
        descr[v] = one_row(r.Description.iloc[0])
        remark = r["Derivation Notes"].iloc[0]
        remark = one_row(str(remark).split("\n\n")[0]) if pd.notna(remark) else ""
        log(f"  {v}: «{descr[v]}»" + (f'; derivation note: «{remark}»' if remark else ""))
    reading_ok = bool(re.search(r"\bOFF\b", descr[VAR_III])
                      and re.search(r"untreated", descr[VAR_III], re.I)
                      and re.search(r"\bON\b", descr[VAR_III_ON])
                      and not re.search(r"\bOFF\b", descr[VAR_III_ON]))
    log('  OFF/ON reading: '
        + (f'verified on the dictionary. The description of {VAR_III} names the OFF state and the '
           f'untreated scores, that of {VAR_III_ON} the ON state.'
           if reading_ok else
           f'NOT verified. The descriptions of the dictionary do not name OFF and «untreated» for '
           f'{VAR_III} and ON for {VAR_III_ON}.'))
    ver_reading = ('OFF/ON reading verified on the dictionary of the curated data cut' if reading_ok
                   else 'OFF/ON reading NOT verified (non-conforming dictionary)')
else:
    log(f"OFF/ON reading of {VAR_III} and {VAR_III_ON}: NOT verified. The dictionary of the curated "
        f"data cut was not found: that {VAR_III} is the OFF or untreated score is not established "
        f"here, and the counts below are to be read without that interpretation.")
    ver_reading = 'OFF/ON reading NOT verified (dictionary not found)'

rows_u3 = []
for c, cfg in COHORTS.items():
    Lc = cur[cur.PATNO.isin(P[c])]
    F = Lc[Lc.year > 0]
    trt = pd.to_numeric(F.PDTRTMNT, errors="coerce")
    ha2, ha3, ha3on = F[VAR_II].notna(), F[VAR_III].notna(), F[VAR_III_ON].notna()
    hole = ha2 & ~ha3
    n_f, n2, n3, n3on, n_hole = (len(F), int(ha2.sum()), int(ha3.sum()),
                                 int(ha3on.sum()), int(hole.sum()))
    b_t, b_0, b_na = (int((hole & (trt == 1)).sum()), int((hole & (trt == 0)).sum()),
                      int((hole & trt.isna()).sum()))
    assert b_t + b_0 + b_na == n_hole, f'{c}: PDTRTMNT has values other than 0, 1, missing'
    b_on = int((hole & ha3on).sum())
    only_on = int((ha3on & ~ha3).sum())
    n_t = int((trt == 1).sum())
    n_t3 = int(((trt == 1) & ha3).sum())
    subj_hole = int(F.PATNO[hole].nunique())
    # check on the data of the reading of the dictionary: without treatment the two
    # variables must carry the same score (baseline included)
    every = Lc[(pd.to_numeric(Lc.PDTRTMNT, errors="coerce") == 0)
               & Lc[VAR_III].notna() & Lc[VAR_III_ON].notna()]
    n_nt, n_nt_equals = len(every), int((every[VAR_III] == every[VAR_III_ON]).sum())
    # the counts of part II and III after baseline against the sample of the model
    ver_mod = {}
    for v, k in ((VAR_II, n2), (VAR_III, n3)):
        e = E[(E.cohort == c) & (E.outcome == v)].iloc[0]
        ver_mod[v] = (f'equal to the observations after baseline of the sample of the model '
                      f'(108_longitudinal_outcomes.csv; {e.verified_against})'
                      if int(e.n_obsv_after_sampling) == k else
                      f'different from the sample of the model ({int(e.n_obsv_after_sampling)} '
                      f'observations after baseline: there the covariates count too)')
    NV = 'NOT verified: no comparison with the pipeline in this script'
    UV = 'visits after baseline'
    for name, n, den, units, ver in [
        ('visits after baseline', n_f, n_f, UV, NV),
        (f'with {VAR_II} (MDS-UPDRS II)', n2, n_f, UV, ver_mod[VAR_II]),
        (f'with {VAR_III} (part III of the models)', n3, n_f, UV, ver_mod[VAR_III]),
        (f'with {VAR_III_ON}', n3on, n_f, UV, NV),
        (f'with {VAR_II} but without {VAR_III}', n_hole, n2, UV, NV),
        (f'with {VAR_II} but without {VAR_III}, of whom PDTRTMNT = 1', b_t, n_hole, UV, NV),
        (f'with {VAR_II} but without {VAR_III}, of which PDTRTMNT = 0', b_0, n_hole, UV, NV),
        (f'with {VAR_II} but without {VAR_III}, of which PDTRTMNT missing', b_na, n_hole, UV, NV),
        (f'with {VAR_II} but without {VAR_III}, of whom with {VAR_III_ON}', b_on, n_hole, UV, NV),
        (f'with {VAR_III_ON} but without {VAR_III}', only_on, n_f, UV, NV),
        ('with PDTRTMNT = 1', n_t, n_f, UV, NV),
        (f'with PDTRTMNT = 1, of which with {VAR_III}', n_t3, n_t, UV, NV),
        (f'subjects with at least one subsequent visit with {VAR_II} but without {VAR_III}',
         subj_hole, N[c], 'subjects', NV),
        (f'visits with PDTRTMNT = 0 and with {VAR_III} e {VAR_III_ON}, of which with the two scores '
         f'equal', n_nt_equals, n_nt, 'visits, baseline included', NV),
    ]:
        rows_u3.append(dict(cohort=c, n_cohort=N[c], item=name, units=units, n=int(n),
                             denominator=int(den),
                             percentage=100 * n / den if den else np.nan,
                             check=ver, reading_of_variables=ver_reading))
    log(f"\n--- cohort {cfg['name']}, n = {N[c]} ---")
    log(f'  visits after baseline: {n_f}; with {VAR_II} {n2}; with {VAR_III} {n3}; with {VAR_III_ON} '
        f'{n3on}')
    log(f"    {VAR_II}: {ver_mod[VAR_II]}")
    log(f"    {VAR_III}: {ver_mod[VAR_III]}")
    log(f'  with {VAR_II} but without {VAR_III}: {share(n_hole, n2)}, of {share(subj_hole, N[c])} '
        f'subjects')
    log(f'    of these visits: PDTRTMNT = 1 in {share(b_t, n_hole)}, PDTRTMNT = 0 in {b_0}, PDTRTMNT '
        f'missing in {b_na}; with {VAR_III_ON} {share(b_on, n_hole)}')
    if n_hole:
        log(f"    the visits with part II and without the part III of the models are in "
            f"{('majority' if b_t > n_hole / 2 else 'minority')} visits with PDTRTMNT = 1")
    log(f'  with {VAR_III_ON} but without {VAR_III}: {share(only_on, n_f)}')
    log(f'  visits with PDTRTMNT = 1: {share(n_t, n_f)}; of these with {VAR_III}: {share(n_t3, n_t)}')
    log(f'  check on the data: at the visits with PDTRTMNT = 0 and with both scores (baseline '
        f'included) the two are equal in {share(n_nt_equals, n_nt)}')
    log('  the other counts of this block: NOT verified (no comparison with the pipeline in this '
        'script)')

# ------------------------------------------------------------------ 4. RETENTION
title('4. RETENTION UNDER OBSERVATION, YEAR BY YEAR')
log("Unit: subjects. «Still under observation at year t» = the last visit of the")
log('curated data cut has nominal year >= t (monotone exit: intermediate gaps count as')
log("retention). «Visit at year t» = a row of the curated data cut exists with that year.")
log("«Reachable» = month of the baseline visit + 12 t months not beyond the last month with")
log(f"data in the whole curated data cut ({end_data.strftime('%m/%Y')}): those who are not reachable "
    f"could not")
log('have had that visit for calendar reasons, whether or not they left the')
log("study earlier. It is an upper bound: the most recent visits may not yet be")
log('in the file, and for this reason the retention percentages among the reachable')
log('are lower bounds.')
log("Verification of the table: retention rests on the last visit, whose summary")
log("is identical to 70_baseline.csv (point 3bis). Reachability by calendar")
log('rests on visit_date, which no pipeline file carries: NOT verified against')
log('the pipeline. Enrolment phase: '
    + ('counts identical to 70_baseline.csv.'
       if all(verified[c]["enroll_phase"] for c in COHORTS) else
       '70_baseline.csv does not carry its counts, NOT verified against the pipeline.'))
rows_withdr = []
for c, cfg in COHORTS.items():
    b = bl.loc[sorted(P[c])]
    f = last_any[c].reindex(b.index)
    L = cur[cur.PATNO.isin(P[c])]
    log(f"\n--- cohort {cfg['name']}, n = {N[c]} ---")
    phase = b.enroll_phase.value_counts().sort_index().to_dict()
    log(f'enrolment phase: {phase} ('
        + ('counts identical to 70_baseline.csv' if verified[c]["enroll_phase"]
           else 'NOT verified against the pipeline')
        + f'); year of the baseline visit: '
          f'{b.data_visit.dt.year.min()}-{b.data_visit.dt.year.max()}; study status at the date of '
          f'the file: {b.study_status.value_counts().to_dict()}; deceased (Death_Status = 1): '
          f'{int((b.Death_Status == 1).sum())}')
    potential = (month_end - b.month) / 12
    qpot = quantiles(potential)
    log(f"follow-up possible by calendar (from the baseline visit to the last month with data): "
        f"{fq(qpot, 1)} years")
    rows = []
    for t in range(0, int(cur.year.max()) + 1):
        reach = (b.month + 12 * t) <= month_end
        still = f >= t
        rows.append(dict(
            cohort=c, n_cohort=N[c], year=t,
            n_having_visit=int(L[L.year == t].PATNO.nunique()),
            n_still_in_observation=int(still.sum()),
            pct_of_cohort=100 * still.mean(),
            n_reachable=int(reach.sum()),
            n_still_between_i_reachable=int((still & reach).sum()),
            pct_of_reachable=(100 * (still & reach).sum() / reach.sum()
                                   if reach.sum() else np.nan),
            n_still_ma_no_reachable=int((still & ~reach).sum())))
    R = pd.DataFrame(rows)
    log(R.drop(columns=['cohort', 'n_cohort']).round(1).to_string(index=False))
    # verification label of the table: derived from the comparisons carried out
    R['check'] = (
        'retention: summary of the last visit identical to 70_baseline.csv (median, IQR, n at year '
        '10 or beyond); reachability by calendar: '
        + ('values identical to the pipeline' if verified[c].get("visit_date") else
           'NOT verified against the pipeline (visit_date without a match)'))
    rows_withdr.append(R)
    for t in YEARS_KEY:
        x = R[R.year == t].iloc[0]
        log(f'  year {t}: still under observation {share(int(x.n_still_in_observation), N[c])}; '
            f'reachable by calendar {share(int(x.n_reachable), N[c])}; among these still under '
            f'observation {share(int(x.n_still_between_i_reachable), int(x.n_reachable))}')
    x5 = R[R.year == 5].iloc[0]
    no_reach = N[c] - int(x5.n_reachable)
    cal = ('the calendar limits to less than 5 years the follow-up of the majority of the cohort' if no_reach > N[c] / 2 else
           'the calendar limits the follow-up of a minority of the cohort to less than 5 years' if no_reach > 0 else
           'the calendar does not limit the follow-up to 5 years')
    log(f'  calendar: {share(no_reach, N[c])} subjects could not have reached year 5 at the date of '
        f'the last data point: {cal}.')
    # the exit from observation (withdrawal, death, or visits not yet in the file) is
    # read among those who COULD be followed: at the last year that more than half
    # of the cohort could have reached
    t_ref = int(R.year[R.n_reachable > N[c] / 2].max())
    xr = R[R.year == t_ref].iloc[0]
    n_r, n_a = int(xr.n_reachable), int(xr.n_still_between_i_reachable)
    exited = n_r - n_a
    states = set(cur.study_status.dropna())
    withdr = (f"; in the whole cohort, withdrawn from the study at the date of the file (study_status = "
           f"Withdrawn): {share(int((b.study_status == 'Withdrawn').sum()), N[c])}"
           if "Withdrawn" in states else
           '; the status «Withdrawn» does not appear in the curated data cut: withdrawn NOT counted')
    withdr += f', deceased (Death_Status = 1): {share(int((b.Death_Status == 1).sum()), N[c])}'
    log(f'  exit from observation: among the {n_r} who could have reached year {t_ref} (the last '
        f'year reachable by more than half of the cohort) {share(n_a, n_r)} are still under '
        f'observation and {share(exited, n_r)} no longer are{withdr}.')
    abb = (f'exit from observation among those who could be followed is present ({exited} out of '
           f'{n_r} at year {t_ref})' if exited > 0 else
           f'none left observation among those who could reach year {t_ref}')
    log(f"  in this cohort {cal}, and {abb}. Both are reported: this script does not establish which "
        f"of the two weighs more on the follow-up.")
    for ph, g in b.groupby("enroll_phase"):
        ff = f.reindex(g.index)
        log(f'  phase {ph}: n = {len(g)}, median follow-up [IQR] {ff.median():.1f} '
            f'[{ff.quantile(0.25):.1f}; {ff.quantile(0.75):.1f}] years')
WITHDR = pd.concat(rows_withdr, ignore_index=True)

# ------------------------------------------------------------------ summary
title('SUMMARY')
for c, cfg in COHORTS.items():
    s = fu_summary[c]
    m = E[(E.cohort == c) & (E.outcome == "moca")].iloc[0]
    u = E[(E.cohort == c) & (E.outcome == "updrs3_score")].iloc[0]
    log(f"{cfg['name']}: n = {N[c]}; follow-up {s['text']} years, {s['n_ge']} at year 10 or beyond, "
        f"{s['only_bl']} with baseline only; CSF storage before the assay run "
        f"{times[c]['cons']['median']:.1f} years (median, n = {times[c]['cons']['n']}); by enrolment "
        f"phase: "
        + "; ".join(f"phase {ph}, n = {r['n']}: {r['median']:.2f} (extremes "
                    f"{r['minimum']:.2f}..{r['maximum']:.2f})"
                    for ph, r in times[c]['phases'].items()) + ".")
    log(f'  MoCA: {int(m.n_subj)} subjects, {int(m.n_obsv)} observations of which '
        f'{int(m.n_obsv_after_sampling)} after sample collection; per subject, after sample '
        f'collection, {m.obsv_after_by_subj_median:g} [{m.obsv_after_by_subj_q1:g}; '
        f'{m.obsv_after_by_subj_q3:g}]; at least 2 after sample collection '
        f'{int(m.n_subj_atleast_2_obsv_after)}, at least 3 {int(m.n_subj_atleast_3_obsv_after)}.')
    log(f'  MDS-UPDRS III: {int(u.n_subj)} subjects, {int(u.n_obsv)} observations of which '
        f'{int(u.n_obsv_after_sampling)} after sample collection; per subject, after sample '
        f'collection, {u.obsv_after_by_subj_median:g} [{u.obsv_after_by_subj_q1:g}; '
        f'{u.obsv_after_by_subj_q3:g}].')

# ------------------------------------------------------------------ writing
pd.DataFrame(rows_csf).to_csv(os.path.join(OUT, "108_csf_baseline.csv"), index=False)
V = pd.DataFrame(rows_vis)
for k in ('year_nominal', 'n_having_visit_year_nominal', 'n_having_value_year_nominal'):
    V[k] = V[k].astype("Int64")          # integers even where a row has no year
V.to_csv(os.path.join(OUT, "108_csf_by_visit.csv"), index=False)

# timing: rows grouped by cohort in the order in which they were created; integer
# counts; the statistics of the rows in days written as integers when they are so
TT = pd.DataFrame(rows_times)
TT = TT.iloc[np.argsort(TT.cohort.map({c: i for i, c in enumerate(COHORTS)}).to_numpy(),
                        kind="stable")].reset_index(drop=True)
COL_CONT = (['n_expected', "n", 'missing', "n_zero", 'n_negatives', 'n_positives']
            + [f'n_beyond_{g}_days' for g in THRESHOLDS_DAYS])
COL_STAT = ['median', "q1", "q3", 'minimum', 'maximum']
for k in COL_CONT:
    TT[k] = TT[k].astype("Int64")
row_in_days = TT.units.str.startswith('days').to_numpy()
for k in COL_STAT:
    TT[k] = pd.Series(
        [np.nan if pd.isna(v) else int(v) if g and float(v).is_integer() else float(v)
         for v, g in zip(TT[k], row_in_days)], dtype=object, index=TT.index)
TT[['remark', 'check']] = TT[['remark', 'check']].fillna("")
assert (TT.check != "").all(), 'a row of the timings without a verification label'
TT = TT[['cohort', 'n_cohort', 'interval', 'units'] + COL_CONT[:3] + COL_STAT
        + COL_CONT[3:6] + ['remark'] + COL_CONT[6:] + ['check']]
TT.to_csv(os.path.join(OUT, "108_times_baseline.csv"), index=False)
pd.DataFrame(rows_u3).to_csv(os.path.join(OUT, "108_updrs3_treatment.csv"), index=False)
E.to_csv(os.path.join(OUT, "108_longitudinal_outcomes.csv"), index=False)
pd.DataFrame(rows_year).to_csv(os.path.join(OUT, "108_longitudinal_by_year.csv"),
                                index=False)
WITHDR.to_csv(os.path.join(OUT, "108_retention.csv"), index=False)
log('\nSaved in outputs/: 108_csf_baseline.csv, 108_csf_by_visit.csv, 108_times_baseline.csv, '
    '108_longitudinal_outcomes.csv, 108_longitudinal_by_year.csv, 108_updrs3_treatment.csv, '
    '108_retention.csv, 108_availability_and_timing.txt')
with open(os.path.join(OUT, "108_availability_and_timing.txt"), "w") as fh:
    fh.write("\n".join(rep))
