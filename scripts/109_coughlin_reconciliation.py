"""
109 - Reconciliation of the counts with Coughlin et al. (Mov Disord 2026).

A reviewer observes that the most recent article on the PPMI data reports 1171
assay positives and the manuscript 928. The two numbers do not count the same
thing, and the difference must be broken down into verifiable pieces instead of
being explained in words. This script does so starting from the public files.

What is computed:

  1. Our flow, per protocol and on the union of the two protocols (distinct
     persons): PD with an assay at 24 or at 150 h -> at baseline -> positive
     at baseline -> admitted pattern (Type1 at 24 h; positive at 150 h) ->
     linked to the curated data cut -> sporadic -> complete on the seven
     variables. The end points are compared by identity of PATNO and of values
     with the two cohort files produced by 01/02 and by 08.
  2. The reconstruction of the PD rows of Table 1 of Coughlin ("All
     participants with CSF SAA results": first available result of each
     participant), with the published figures alongside. Rules of the article:
     the first visit with a result counts, not necessarily baseline; on the
     same visit the assay run last holds; positive = positive at 150 h,
     Type I at 24 and at 35 h. Three sources: the raw file (24 and 150 h), the
     curated data cut (categorical outcome of the three protocols, baseline and
     annual visits) and the file of the kinetic parameters at 35 h (detected
     samples only).
  3. The breakdown of the difference between the positives at the first visit
     of the reconstructed sPD row and our distinct persons, with the assertion
     that the pieces add up.
  4. Our participants whom Coughlin's rule does not count among the positives
     at the first visit, with their runs.

What CANNOT be reconstructed from the public files, and the script prints it
alongside the numbers: the download of the article and that of the files used
here may not coincide (the script compares the dates contained in the files
with the published download date and writes what results from it, it does not
assume it); the article states that it used the participants' real activity
dates, which are restricted data (that it ordered the visits by date is our
inference), and here the order of the visit codes holds (the script counts how
many first visits would change when ordering by the visit month of the curated
data cut, where the month is present); the negatives at 35 h are seen only
through the curated data cut, i.e. only at baseline and at the annual visits;
those who are not in the curated data cut have no subgroup and cannot be
assigned to a row.

The reconciliation is defined for the sporadic population: the term of
comparison is the sPD row of the article. The script stops if ONLY_SPORADIC
and the working directory do not agree, and if the working directory is not
that of the sporadic only.

Guards. The curated data cut is read from the parquet cache; the cache is
compared with the xlsx (same visits by identity of PATNO and EVENT_ID, same
values in the columns read) when the xlsx is reachable. A guard that cannot run
(xlsx absent or not readable by the quick check, analysis_dataset.csv
absent) writes so in the log with "NON verificat" and appears in the list at
the bottom of the summary: it does not disappear. The reason for those removed
at the last step of the flow (kinetic parameters, DAT-SPECT or age) is derived
participant by participant.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 scripts/109_coughlin_reconciliation.py
Output: outputs/109_reconciliation_flow.csv, 109_reconciliation_table1.csv,
        109_reconciliation_departure.csv, 109_reconciliation_subjects.csv,
        109_coughlin_reconciliation.txt
"""

import hashlib
import html
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA                                       # noqa: E402
import _dat                                                      # noqa: E402

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
KIN35 = os.path.join(DATA, "Kinetic_data_for_SAA_Results_24Apr2026.csv")
NAME_CURATED = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"
SHEET_CURATED = "20260511"
CACHE = os.path.join(OUT, "cache", "curated.parquet")
TOLER = 1e-9          # relative tolerance in the comparison of the values with the cohorts

# ---------------------------------------------------------------------------
# Figures PUBLISHED by Coughlin et al., Mov Disord 2026, doi 10.1002/mds.70509,
# Table 1, block "All participants with CSF SAA results", PD rows. They are
# literals from an external source, reported only to set them beside the
# reconstruction; no count of this script depends on them.
# Order: N with a result, positives/Type I at the first visit, negatives,
# Type II, inconclusive; then N with more than one visit with a result.
# ---------------------------------------------------------------------------
DOI = "10.1002/mds.70509"
CAT = ["pos1", "neg", "type2", "inc"]
NAME_CAT = {"pos1": 'positive/Type I', "neg": 'negative', "type2": "Type II",
            "inc": 'inconclusive'}
PUBLISHED = {
    "PD Total":         dict(N=1389, pos1=1171, neg=195, type2=15, inc=8, more=633),
    "sPD":              dict(N=1032, pos1=930,  neg=85,  type2=11, inc=6, more=444),
    "LRRK2 PD":         dict(N=171,  pos1=110,  neg=60,  type2=1,  inc=0, more=125),
    "GBA PD":           dict(N=90,   pos1=82,   neg=6,   type2=1,  inc=1, more=42),
    "Other genetic PD": dict(N=54,   pos1=38,   neg=16,  type2=0,  inc=0, more=22),
    "PD normosmic":     dict(N=42,   pos1=11,   neg=28,  type2=2,  inc=1, more=0),
}
# Download date stated in the article (Data Availability Statement).
DOWNLOAD_PUBLISHED = pd.Timestamp("2026-01-05")
ROWS = list(PUBLISHED)
WITHOUT_ROW = 'absent from the curated data cut'
SPOR = "Sporadic PD"

rep = []
no_executed = []     # guards provided for that could not run in this run


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def no_verified(what):
    """Records a guard that could not run: it also ends up in the summary."""
    no_executed.append(what)


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def listing(s):
    """A set of PATNO as a sorted list, for printing."""
    return ", ".join(str(x) for x in sorted(s)) if s else 'noone'


def count(series):
    """Counts of a series as a dictionary in stable order."""
    c = series.value_counts()
    return {k: int(c[k]) for k in sorted(c.index, key=str)}


def columns_xlsx(path, sheet, columns):
    """The requested columns of an xlsx sheet, read from the XML of the sheet.

    It only serves to compare the parquet cache with the xlsx without opening
    the whole file with openpyxl (tens of seconds): only the cells of the
    requested columns are read. Returns a table with one row per row of the
    sheet (header excluded); empty cells are NaN, numbers float, strings
    str. Raises ValueError if the sheet does not have the expected structure
    (shared strings and numbers, one <v> per cell): the caller must then write
    that the comparison was NOT carried out.
    """
    def attributes(tag):
        return dict(re.findall(r'([\w:]+)="([^"]*)"', tag))

    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for portion in ("xl/workbook.xml", "xl/_rels/workbook.xml.rels"):
            if portion not in names:
                raise ValueError(f'{portion} is missing')
        sheets = [attributes(t) for t in
                 re.findall(r"<sheet\b[^>]*>", z.read("xl/workbook.xml").decode("utf-8"))]
        rid = [f.get("r:id") for f in sheets if f.get("name") == sheet]
        if len(rid) != 1 or not rid[0]:
            raise ValueError(f"sheet '{sheet}' not found")
        rel = [attributes(t) for t in
               re.findall(r"<Relationship\b[^>]*>",
                          z.read("xl/_rels/workbook.xml.rels").decode("utf-8"))]
        target = [r.get("Target") for r in rel if r.get("Id") == rid[0]]
        if len(target) != 1 or f'xl/{target[0]}' not in names:
            raise ValueError(f"part of the sheet '{sheet}' not found")
        sharedstrings = []
        if "xl/sharedStrings.xml" in names:
            text = z.read("xl/sharedStrings.xml").decode("utf-8")
            sharedstrings = ["".join(html.unescape(t) for t in
                                 re.findall(r"<t(?:\s[^>]*)?>([^<]*)</t>", si))
                         for si in re.findall(r"<si>(.*?)</si>", text, flags=re.S)]
        x = z.read(f'xl/{target[0]}')

    def value(kind, body):
        m = re.fullmatch(rb"<v>([^<]*)</v>", body)
        if m is None:
            raise ValueError('cell without a single <v>')
        if kind == b"s":
            return sharedstrings[int(m.group(1))]
        if kind in (b"", b"n"):
            return float(m.group(1))
        raise ValueError(f'unexpected cell type: {kind.decode()}')

    cell = rb'<c r="(%s)(\d+)"([^>]*?)(?:/>|>(.*?)</c>)'
    header = re.search(rb'<row r="1"[^>]*>(.*?)</row>', x, flags=re.S)
    if header is None:
        raise ValueError('header row not found')
    letter = {}
    for m in re.finditer(cell % rb"[A-Z]+", header.group(1), flags=re.S):
        if m.group(4) is not None:
            t = re.search(rb'\st="([^"]*)"', m.group(3))
            name = value(t.group(1) if t else b"", m.group(4))
            if name in letter:
                raise ValueError(f'column repeated in the header: {name}')
            letter[name] = m.group(1).decode()
    absent = [c for c in columns if c not in letter]
    if absent:
        raise ValueError(f'columns missing from the xlsx: {absent}')
    of_letter = {letter[c]: c for c in columns}
    n_rows = len(re.findall(rb"<row\b", x))
    cells = {c: {} for c in columns}
    choice = "|".join(sorted(of_letter)).encode()
    for m in re.finditer(cell % choice, x, flags=re.S):
        row = int(m.group(2))
        if row == 1 or m.group(4) is None:
            continue
        if row > n_rows:
            raise ValueError('non-consecutive row numbers')
        t = re.search(rb'\st="([^"]*)"', m.group(3))
        cells[of_letter[m.group(1).decode()]][row] = value(t.group(1) if t else b"",
                                                             m.group(4))
    return pd.DataFrame({c: pd.Series(cells[c], dtype=object) for c in columns},
                        index=range(2, n_rows + 1))


# the transcription of the published figures must be consistent with itself
for _r, _v in PUBLISHED.items():
    assert sum(_v[c] for c in CAT) == _v["N"], f'inconsistent published figures: {_r}'
for _c in ["N"] + CAT + ['more']:
    assert sum(PUBLISHED[r][_c] for r in ROWS[1:]) == PUBLISHED["PD Total"][_c], (
        f'the published rows do not add up to the total: {_c}')

# ============================================================================
# 0. FILES AND WORKING DIRECTORY
# ============================================================================
title('0. FILES, WORKING DIRECTORY AND POPULATION')

d = pd.read_csv(SAA, low_memory=False)
assert not d.duplicated().any(), 'duplicate rows in the SAA file'
d["PATNO"] = d["PATNO"].astype("int64")
d["RUNDATE"] = pd.to_datetime(d["RUNDATE"])
assert d.RUNDATE.notna().all(), 'missing run dates in the SAA file'
assert d.COHORT.notna().all(), 'missing cohort in the SAA file'
assert set(d.PROJECTID.unique()) == {155, 237}, 'unexpected projects in the SAA file'
assert d.groupby("PATNO").COHORT.nunique().max() == 1, 'subject with two cohorts (SAA)'
assert set(d.SAA_Status.unique()) == {"Positive", "Negative", "Inconclusive"}, (
    'unexpected outcomes in the SAA file')
d['row'] = np.arange(len(d))

k35 = pd.read_csv(KIN35, low_memory=False)
k35["PATNO"] = k35["PATNO"].astype("int64")
k35["RUNDATE"] = pd.to_datetime(k35["RUNDATE"])
assert k35.RUNDATE.notna().all(), 'missing run dates in the 35 h file'
assert set(k35.PROJECTID.unique()) == {299}, 'unexpected projects in the 35 h file'
assert k35.groupby("PATNO").COHORT.nunique().max() == 1, 'subject with two cohorts (35 h)'
OUTCOMES_35 = {"Detected-1": "pos1", "Detected-2": "type2"}
assert set(k35.SAA_Result.unique()) <= set(OUTCOMES_35), (
    "the 35 h file contains outcomes other than Detected-1/Detected-2: the rule 'detected samples "
    "only' no longer holds and the reconstruction must be revised")

COL_CUR = ["PATNO", "EVENT_ID", "COHORT", "subgroup", "CSFSAA", "CSFSAA_assay",
           "visit_date", "age_at_visit", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
           "MIA_PUTAMEN_BILAT", "MIA_CAUDATE_BILAT", "MIA_STRIATUM_BILAT"]
cand = [os.path.join(BASE, NAME_CURATED), os.path.join(DATA, NAME_CURATED)]
located = [c for c in cand if os.path.exists(c)]
FR_CACHE = os.path.exists(CACHE)
if FR_CACHE:
    cur = pd.read_parquet(CACHE, columns=COL_CUR)
    source_cur = os.path.realpath(CACHE)
else:
    assert located, f"curated not found either in the cache or in: {cand}"
    cur = pd.read_excel(located[0], sheet_name=SHEET_CURATED, usecols=COL_CUR)
    source_cur = located[0]
cur["PATNO"] = cur["PATNO"].astype("int64")
assert not cur.duplicated(["PATNO", "EVENT_ID"]).any(), 'duplicate visits in the curated data cut'

# --- GUARD: the parquet cache is the curated xlsx, on the columns read here.
#     Identity of the visits (PATNO, EVENT_ID) and exact equality of the values.
#     If the xlsx is not there or the quick check does not read it, this is written.
KEY_CUR = ["PATNO", "EVENT_ID"]
if not FR_CACHE:
    state_cache = None                       # xlsx read: no cache to verify
elif not located:
    state_cache = (False, f'{NAME_CURATED} absent from both the working directory and the raw data '
                          f'folder')
else:
    try:
        xl = columns_xlsx(located[0], SHEET_CURATED, COL_CUR)
    except (ValueError, zipfile.BadZipFile, KeyError, IndexError) as e:
        state_cache = (False, f'the quick check does not read {NAME_CURATED}: {e}')
    else:
        assert xl[KEY_CUR].notna().all().all(), 'xlsx of the curated data cut: empty PATNO or EVENT_ID'
        xl["PATNO"] = pd.to_numeric(xl["PATNO"]).astype("int64")
        assert not xl.duplicated(KEY_CUR).any(), "duplicate visits in the xlsx of the curated data cut"
        k_xl = set(zip(xl.PATNO, xl.EVENT_ID))
        k_ca = set(zip(cur.PATNO, cur.EVENT_ID))
        assert k_xl == k_ca, (
            f"the cache of the curated data cut and the xlsx do not have the same visits: "
            f"{len(k_ca - k_xl)} only in the cache, {len(k_xl - k_ca)} only in the xlsx")
        xl = xl.set_index(KEY_CUR).loc[list(zip(cur.PATNO, cur.EVENT_ID))]
        for c in [c for c in COL_CUR if c not in KEY_CUR]:
            a, b = xl[c].to_numpy(), cur[c].to_numpy()
            if cur[c].dtype == object:
                va, vb = pd.isna(a), pd.isna(b)
                different = int((va != vb).sum()) + int((a[~va & ~vb] != b[~va & ~vb]).sum())
            else:
                # in the xlsx some numbers (PATNO, COHORT) are text cells
                an = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(float)
                no_numeric = int((~pd.isna(a) & np.isnan(an)).sum())
                b = b.astype(float)
                different = no_numeric + int((~((an == b) | (np.isnan(an) & np.isnan(b)))).sum())
            assert different == 0, (
                f'the cache of the curated data cut differs from the xlsx in the column {c}: '
                f'{different} visits with a different value')
        state_cache = (True, f'{os.path.basename(located[0])}, sha256 {fingerprint(located[0])}')
assert cur.groupby("PATNO").COHORT.nunique().max() == 1, 'subject with two cohorts (curated data cut)'
assert cur.groupby("PATNO").subgroup.nunique(dropna=False).max() == 1, (
    'subgroup not constant within subject in the curated data cut')
COHORT_CUR = {1: "PD", 2: "Control", 3: "SWEDD", 4: "Prodromal"}
assert set(cur.COHORT.unique()) <= set(COHORT_CUR), 'unexpected cohort codes in the curated data cut'
cur['cohort'] = cur.COHORT.map(COHORT_CUR)

for name, fpath in (('assay at 24 and 150 h', SAA), ('kinetic parameters at 35 h', KIN35)):
    log(f'{name}: {os.path.basename(fpath)}, {os.path.getsize(fpath)} byte, sha256 '
        f'{fingerprint(fpath)}')
log(f'curated data cut: {os.path.basename(source_cur)}, {os.path.getsize(source_cur)} bytes, sha256 '
    f'{fingerprint(source_cur)}')
if state_cache is None:
    DESCR_CUR = os.path.basename(source_cur)
    log("  curated data cut read directly from the xlsx: no cache involved.")
elif state_cache[0]:
    DESCR_CUR = (f'{os.path.basename(source_cur)}, cache of {NAME_CURATED} verified on the columns '
                 f'read')
    log(f'  Guard, cache of the curated data cut: compared with {state_cache[1]}.')
    log(f"  Same {len(cur)} visits by identity of (PATNO, EVENT_ID) and identical values in the "
        f"{len(COL_CUR)} columns read here ({len(cur) * len(COL_CUR)} cells, exact equality).")
else:
    DESCR_CUR = (f'{os.path.basename(source_cur)}, cache whose correspondence with {NAME_CURATED} is '
                 f'NOT verified by this script')
    log(f'  Guard, cache of the curated data cut: NOT verified against the xlsx by this script '
        f'({state_cache[1]}).')
    log("  All that follows assumes that the cache is the curated data cut; the only check is")
    log("  the fingerprint printed above.")
    no_verified(f'cache of the curated data cut not compared with the xlsx ({state_cache[1]})')
log(f'  SAA file: {len(d)} rows (assay results), {d.PATNO.nunique()} subjects; 150 h runs from '
    f'{d[d.PROJECTID == 155].RUNDATE.min().date()} to {d[d.PROJECTID == 155].RUNDATE.max().date()}, '
    f'24 h runs from {d[d.PROJECTID == 237].RUNDATE.min().date()} to '
    f'{d[d.PROJECTID == 237].RUNDATE.max().date()}')
log(f'  35 h file: {len(k35)} rows (wells), {k35.PATNO.nunique()} subjects; outcomes present: '
    f'{count(k35.SAA_Result)} -> detected samples only; runs from {k35.RUNDATE.min().date()} to '
    f'{k35.RUNDATE.max().date()}')
log(f'  curated data cut: {len(cur)} rows (subject x visit), {cur.PATNO.nunique()} subjects; visits '
    f'with an assay outcome {int(cur.CSFSAA.notna().sum())}')

# the cohort of a subject must be the same in all the sources in which it appears
co_saa = d.drop_duplicates("PATNO").set_index("PATNO").COHORT
co_k35 = k35.drop_duplicates("PATNO").set_index("PATNO").COHORT
co_cur = cur.drop_duplicates("PATNO").set_index("PATNO").cohort
for a, b, na, nb in ((co_saa, co_cur, "SAA", "curated"), (co_k35, co_cur, "35 h", "curated"),
                     (co_saa, co_k35, "SAA", "35 h")):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    assert (j.a == j.b).all(), f'cohort discordant between {na} and {nb}'
    log(f'  cohort concordant between {na} and {nb} in all the {len(j)} common subjects')

sub = cur.drop_duplicates("PATNO").set_index("PATNO").subgroup     # constant within subject
cbl = cur[cur.EVENT_ID == "BL"].set_index("PATNO")                 # one row per subject
sporadic = set(sub[sub == SPOR].index)

# the two analysis cohorts, read from the outputs of the current working directory
c24 = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
c150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
if "complete_primary" in c24.columns:
    c24 = c24[c24.complete_primary]
if "complete" in c150.columns:
    c150 = c150[c150.complete]
c24["PATNO"] = c24["PATNO"].astype("int64")
c150["PATNO"] = c150["PATNO"].astype("int64")
assert c24.PATNO.is_unique and c150.PATNO.is_unique, 'repeated PATNO in the cohort files'
c24, c150 = c24.set_index("PATNO"), c150.set_index("PATNO")
p24, p150 = set(c24.index), set(c150.index)
assert (p24 | p150) <= set(sub.index), 'subjects of the cohorts missing from the curated data cut'
only24, only150 = not (p24 - sporadic), not (p150 - sporadic)
assert only24 == only150, ("the two cohorts of the working directory are not of the same kind: one "
                           "is sporadic only, the other is not")
ROOT = 'sporadic only' if only24 else 'full cohort'
assert ONLY == only24, (f"ONLY_SPORADIC={('1' if ONLY else '0')} but the working directory is "
                        f"'{ROOT}': variable and working directory do not agree")
if not ONLY:
    sys.exit('109: the reconciliation is defined for the sporadic population (sPD row of the '
             'article). Rerun with ONLY_SPORADIC=1 on the sporadic-only working directory.')
for name, c in (('24 h', c24), ('150 h', c150)):
    assert (c.subgroup == SPOR).all(), f'{name} cohort: not all sporadic'
log(f"working directory: {os.path.basename(os.path.normpath(BASE))} ({ROOT}); ONLY_SPORADIC = 1; all "
    f"the subjects of the two cohorts have subgroup = '{SPOR}'")
log(f'analysis cohorts read from the working directory: 24 h n = {len(p24)}, 150 h n = {len(p150)}')

# ============================================================================
# 1. OUR FLOW
# ============================================================================
title("1. OUR FLOW (unit: participants)")
log('Starting population: PD cohort of the SAA file (24 and 150 h). The rules are')
log("those of 01_build_dataset.py and 08_replication_150h.py: a subject is positive")
log('at baseline if they have at least one positive baseline sample; at 24 h the')
log("admitted pattern is Type1, at 150 h the type does not exist and positive suffices.")

g = d[d.COHORT == "PD"].copy()
DAT150 = _dat.variables()
DAT_FR = {"dat_putamen_low": lambda t: t[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1),
          "dat_putamen_mean": lambda t: t["MIA_PUTAMEN_BILAT"],
          "dat_striatum_mean": lambda t: t["MIA_STRIATUM_BILAT"],
          "dat_caudate_mean": lambda t: t["MIA_CAUDATE_BILAT"]}
STEPS = ['1 with an assay at any visit',
         '2 with an assay at the baseline visit (BL)',
         '3 positive at baseline',
         '4 with the pattern admitted at baseline',
         '5 linked to the curated data cut at baseline',
         '6 sporadic',
         '7 complete on the seven analysis variables']
# Reason for those removed at steps 2-6: it is the definition of the step. The reason
# for those removed at step 7 is NOT a literal: it is derived further down, participant
# by participant, from which of the seven variables is missing (label_incomplete).
REMOVED = ["",
         'assay only on follow-up samples',
         'negative or inconclusive at baseline',
         'positive without an admitted pattern (Type2 or undetermined)',
         'without a baseline row in the curated data cut',
         'variant carriers or other subgroup']
# at 150 h the type does not exist: step 4 does not filter, and this is said
REMOVED_150 = REMOVED[:3] + ['no filter: at 150 h the type does not exist'] + REMOVED[4:]
# the three components of the completeness criterion, in the order in which they are named
KINDS_LACKS = ['kinetics', 'baseline DAT-SPECT', 'baseline age']


def stream(project):
    """The seven sets of PATNO of a protocol, the table of the complete ones, the
    seven variables and, for the sporadic removed at the last step, what is missing."""
    p = g[g.PROJECTID == project]
    bl = p[p.CLINICAL_EVENT == "BL"]
    pos = bl[bl.SAA_Status == "Positive"]
    if project == 237:
        adm = pos[pos.SAA_Type == "Type1"].copy()
        par = {f"saa_{f}": f"{f}_24h" for f in ("Fmax", "TTT", "AUC", "SLOPEMax")}
        for col, root in par.items():          # 24 h: mean of the three replicates
            adm[col] = adm[[f'{root}_Rep{i}' for i in (1, 2, 3)]].apply(
                pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        adm = pos.copy()
        par = {f"k_{f}": f"{f}_150h" for f in ("Fmax", "T50", "AUC", "SLOPE")}
        for col, root in par.items():          # 150 h: median of the three replicates
            adm[col] = adm[[f'{root}_Rep{i}' for i in (1, 2, 3)]].apply(
                pd.to_numeric, errors="coerce").median(axis=1)
    t = adm.groupby("PATNO")[list(par)].mean()   # mean over the admitted samples of the visit
    t = t.join(cbl[["subgroup", "age_at_visit", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
                    "MIA_PUTAMEN_BILAT", "MIA_CAUDATE_BILAT", "MIA_STRIATUM_BILAT"]],
               how="left")
    t["age_bl"] = t["age_at_visit"]
    for name, f in DAT_FR.items():
        t[name] = f(t)
    # the completeness criterion is the same as before, written in its three
    # components: kinetic parameters, DAT-SPECT, age. At 24 h a kinetic parameter is
    # defined if it is not missing; at 150 h if its logarithm is defined.
    if project == 237:
        dat = ["dat_putamen_mean", "dat_caudate_mean"]
        kinet_ok = t[list(par)].notna().all(axis=1)
    else:
        dat = list(DAT150)
        with np.errstate(divide="ignore", invalid="ignore"):
            kinet_ok = np.log(t[list(par)]).notna().all(axis=1)
    seven = list(par) + dat + ["age_bl"]
    dat_ok = t[dat].notna().all(axis=1)
    age_ok = t["age_bl"].notna()
    definite = kinet_ok & dat_ok & age_ok
    s4 = set(adm.PATNO)
    s5 = s4 & set(cbl.index)
    s6 = {x for x in s5 if sub[x] == SPOR}
    s7 = {x for x in s6 if definite[x]}
    datasets = [set(p.PATNO), set(bl.PATNO), set(pos.PATNO), s4, s5, s6, s7]
    for a, b in zip(datasets[1:], datasets[:-1]):
        assert a <= b, 'the steps of the flow are not nested'
    lacks = pd.DataFrame({KINDS_LACKS[0]: ~kinet_ok, KINDS_LACKS[1]: ~dat_ok,
                          KINDS_LACKS[2]: ~age_ok}).loc[sorted(s6 - s7)]
    assert lacks.any(axis=1).all(), "removed at the last step without a missing variable"
    return datasets, t.loc[sorted(s7), seven], seven, lacks


F24, t24, seven24, lacking24 = stream(237)
F150, t150, seven150, missing150 = stream(155)
LACKS = {'24 h': lacking24, '150 h': missing150}
assert set(lacking24.index) == F24[5] - F24[6] and set(missing150.index) == F150[5] - F150[6]


def reason_incomplete(p, protocols=('24 h', '150 h')):
    """What a participant removed at the last step is missing, DERIVED from the data.

    For each protocol in which the participant is sporadic, admitted and
    incomplete the missing components are listed (kinetic parameters, baseline
    DAT-SPECT, baseline age). If the protocols agree it is written only once.
    """
    items = {n: " + ".join(k for k in KINDS_LACKS if LACKS[n].at[p, k])
            for n in protocols if p in LACKS[n].index}
    assert items and all(items.values()), f'PATNO {p}: no missing variable'
    if len(set(items.values())) == 1:
        return 'incomplete for: ' + next(iter(items.values()))
    return 'incomplete for: ' + "; ".join(f"{v} (a {n})" for n, v in items.items())


def label_incomplete(dataset, protocols=('24 h', '150 h')):
    """Label of those removed at the last step, built from the counts of the reasons."""
    if not dataset:
        return 'no participant'
    c = count(pd.Series([reason_incomplete(p, protocols) for p in sorted(dataset)]))
    assert sum(c.values()) == len(dataset), 'the reasons for incompleteness do not add up'
    return 'incomplete for: ' + ", ".join(
        f"{k.removeprefix('incomplete for: ')} {v}" for k, v in c.items())

# --- the end points coincide with the cohort files: same subjects, same values
for name, mine, t, cohort, seven in (('24 h', F24[6], t24, c24, seven24),
                                    ('150 h', F150[6], t150, c150, seven150)):
    expected = set(cohort.index)
    assert mine == expected, (
        f'{name} cohort: the reconstructed flow and the cohort file do not have the same subjects '
        f'(only in the flow: {listing(mine - expected)}; only in the file: '
        f'{listing(expected - mine)})')
    lacking = [c for c in seven if c not in cohort.columns]
    assert not lacking, f'{name} cohort: columns missing from the cohort file: {lacking}'
    a = t[seven].to_numpy(float)
    b = cohort.loc[t.index, seven].to_numpy(float)
    assert np.isfinite(a).all() and np.isfinite(b).all(), f'{name} cohort: non-finite values'
    departure = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    assert np.allclose(a, b, rtol=TOLER, atol=0), (
        f'{name} cohort: values different from the cohort file (max relative difference '
        f'{departure:.3g})')
    log(f'\nGuard, {name} cohort: the {len(mine)} subjects of the last step are the same PATNO as in '
        f'the cohort file and')
    log(f"  the seven variables ({', '.join(seven)}) recomputed from the raw files")
    log(f'  coincide ({len(mine)} x {len(seven)} values, maximum relative difference {departure:.2g}).')
fpath_ad = os.path.join(OUT, "analysis_dataset.csv")
if os.path.exists(fpath_ad):
    ad = pd.read_csv(fpath_ad, usecols=["PATNO"])
    ad["PATNO"] = ad["PATNO"].astype("int64")
    assert set(ad.PATNO) == F24[5], ('24 h: the sporadic participants before the completeness '
                                     'filter are not the same as in analysis_dataset.csv')
    log(f'Guard, 24 h step 6: same {len(F24[5])} PATNO as in analysis_dataset.csv.')
else:
    log(f"Guard, 24 h step 6: analysis_dataset.csv is missing from the working directory. The "
        f"identity of the {len(F24[5])} PATNO of step 6")
    log("  with that file is NOT verified in this run; the end points")
    log('  (step 7) against the cohort files remain verified.')
    no_verified("24 h, step 6: identity of the PATNO with analysis_dataset.csv (file missing "
                   "from the working directory)")

FU = [a | b for a, b in zip(F24, F150)]
for a, b in zip(FU[1:], FU[:-1]):
    assert a <= b, "the steps of the union are not nested"
assert F150[3] == F150[2], 'at 150 h step 4 must not remove anyone'
# the reason for those removed at the last step is derived, block by block
LAST = {"24": label_incomplete(F24[5] - F24[6], ('24 h',)),
          "150": label_incomplete(F150[5] - F150[6], ('150 h',)),
          'union': label_incomplete(FU[5] - FU[6])}
rows_stream = []
for block, F, reasons in (('24 h (Project 237)', F24, REMOVED + [LAST["24"]]),
                          ('150 h (Project 155)', F150, REMOVED_150 + [LAST["150"]]),
                          ('union of the two protocols (distinct persons)', FU,
                           REMOVED + [LAST['union']])):
    log(f'\n[{block}] PD — participants at each step')
    for i, s in enumerate(F):
        removed = len(F[i - 1]) - len(s) if i else np.nan
        r = dict(block=block, step=STEPS[i], n=len(s), removed=removed,
                 reason_of_removed=reasons[i], only_24h=np.nan, only_150h=np.nan,
                 both=np.nan)
        tail = f'  (-{removed}: {reasons[i]})' if i else ""
        if F is FU:
            r.update(only_24h=len(F24[i] - F150[i]), only_150h=len(F150[i] - F24[i]),
                     both=len(F24[i] & F150[i]))
            tail += (f"  [24 h only {r['only_24h']}, 150 h only {r['only_150h']}, both {r['both']}]")
        rows_stream.append(r)
        log(f'  {STEPS[i]:<46} {len(s):>5}{tail}')
log("\nIn the union each step counts those who pass it in AT LEAST one protocol: those removed at")
log("a step of the union are not the sum of those removed in the two protocols.")
log('Step 7, reason for removal: derived for each participant from which component of the')
log("completeness criterion is missing. 'cinetiche' = at least one of the four kinetic")
log("parameters not defined (at 24 h missing; at 150 h missing or negative, because its")
log('logarithm is taken: a zero has logarithm -inf and counts as defined, as in')
log("08_replication_150h.py); 'DAT-SPECT basale' = at least one of the two DAT measures of the")
log("set missing at the baseline of the curated data cut; 'eta' basale' = age at baseline missing. A")
log("participant missing several components is counted once, with the components")
log("joined by '+'. The number after each reason is the number of participants.")
for name, F, M in (('24 h', F24, lacking24), ('150 h', F150, missing150)):
    log(f'  {name}: removed {len(M)}; with kinetic parameters not defined '
        f'{int(M[KINDS_LACKS[0]].sum())}, with baseline DAT-SPECT missing '
        f'{int(M[KINDS_LACKS[1]].sum())}, with baseline age missing {int(M[KINDS_LACKS[2]].sum())} '
        f'(the three items can overlap).')
for name, F in (('24 h', F24), ('150 h', F150)):
    unlinked = F[3] - F[4]
    elsewhere = sorted(x for x in unlinked if x in sub.index)
    log(f'Step 5, {name}: {len(unlinked)} without a baseline row in the curated data cut, of which '
        f'{len(unlinked) - len(elsewhere)} missing from the curated data cut at every visit and '
        f'{len(elsewhere)} present only at other visits. Without a baseline row they have')
    log("  neither subgroup nor DAT-SPECT: they leave here, not at the subgroup step.")
ours = p24 | p150
shared = p24 & p150
assert FU[6] == ours, "the last step of the union does not coincide with the two cohorts"
assert len(p24) + len(p150) - len(shared) == len(ours)
log(f'\nThe two cohorts: {len(p24)} + {len(p150)} = {len(p24) + len(p150)} entries; {len(shared)} '
    f'participants are in both; distinct persons {len(ours)}.')
log("The number to be compared with a count of participants is that of the")
log(f'distinct persons, {len(ours)}, not the sum of the two cohorts.')

# ============================================================================
# 2. RECONSTRUCTION OF TABLE 1 OF COUGHLIN, PD ROWS
# ============================================================================
title('2. RECONSTRUCTION OF THE PD ROWS OF TABLE 1 OF COUGHLIN')
log(f'Source of the figures shown alongside: Coughlin et al., Mov Disord 2026, doi {DOI},')
log("Table 1, 'All participants with CSF SAA results'. They are PUBLISHED LITERALS.")
log("Rules of the article applied here: (a) the FIRST visit with a result counts;")
log('(b) on the same visit the assay run last holds; (c) positive = positive')
log('at 150 h, Type I at 24 and at 35 h.')
log("Unit: participants of the PD cohort.")


def ordering_visit(e):
    if e == "SC":
        return -1.0
    if e == "BL":
        return 0.0
    if isinstance(e, str) and e.startswith("V") and e[1:].isdigit():
        return float(int(e[1:]))
    return np.nan                 # U01, ST, PW: chronological position not defined


# --- coding of the outcome into the categories of Table 1
g['outcome'] = np.select(
    [g.SAA_Status == "Negative",
     g.SAA_Status == "Inconclusive",
     g.PROJECTID == 155,                       # positive at 150 h
     g.SAA_Type == "Type1",
     g.SAA_Type == "Type2"],
    ["neg", "inc", "pos1", "pos1", "type2"],
    default="inc")                             # positive at 24 h of indeterminate type
g['assay'] = g.PROJECTID.map({155: "150h", 237: "24h"})
g['source'] = 'SAA file'
key = ["PATNO", "CLINICAL_EVENT"]

# same visit: the ROW run last holds (selection of rows, not per column)
mx = g.groupby(key, sort=False).RUNDATE.transform("max")
ties = g[g.RUNDATE == mx].groupby(key, sort=False).agg(
    k=('row', "size"), n_se=('outcome', "nunique"), n_pr=("PROJECTID", "nunique"))
assert int(((ties.k > 1) & (ties.n_se > 1)).sum()) == 0, (
    "visits with discordant outcomes on the last run date: the rule does not decide")
assert int((ties.n_pr > 1).sum()) == 0, 'two protocols on the same last run date'
v_saa = (g.sort_values(key + ["RUNDATE", 'row']).drop_duplicates(key, keep="last")
         [key + ['assay', 'outcome', "RUNDATE", 'source']]
         .rename(columns={"CLINICAL_EVENT": 'EVENT', "RUNDATE": 'run'}))

# --- 35 h from the file of the kinetic parameters: one sample = one AMPRION_ID
kp = k35[k35.COHORT == "PD"]
samp = kp.groupby(["PATNO", "CLINICAL_EVENT", "AMPRION_ID"], as_index=False).agg(
    n_se=("SAA_Result", "nunique"), SAA_Result=("SAA_Result", "min"),
    run=("RUNDATE", "max"), run_min=("RUNDATE", "min"))
assert (samp.n_se == 1).all(), '35 h sample with wells of different outcome'
assert (samp.run == samp.run_min).all(), '35 h sample with two run dates'
samp['outcome'] = samp.SAA_Result.map(OUTCOMES_35)
v_k35 = (samp.sort_values(key + ['run', "AMPRION_ID"]).drop_duplicates(key, keep="last")
         [key + ['outcome', 'run']].rename(columns={"CLINICAL_EVENT": 'EVENT'}))
v_k35['assay'] = "35h"
v_k35['source'] = 'kinetic parameters 35 h'

# --- categorical outcome from the curated data cut (PD)
OUTCOMES_CUR = {0: "neg", 1: "pos1", 2: "inc", 3: "type2"}
ASSAYS_CUR = {1: "150h", 2: "24h", 3: "35h"}
cv = cur[(cur.cohort == "PD") & cur.CSFSAA.notna()].copy()
assert set(cv.CSFSAA.unique()) <= set(OUTCOMES_CUR), 'unexpected CSFSAA codes'
assert set(cv.CSFSAA_assay.unique()) <= set(ASSAYS_CUR), 'unexpected CSFSAA_assay codes'
cv['outcome'] = cv.CSFSAA.map(OUTCOMES_CUR)
cv['assay'] = cv.CSFSAA_assay.map(ASSAYS_CUR)
cv['month_visit'] = pd.to_datetime(cv.visit_date, format="%m/%Y", errors="coerce")
v_cur = cv.rename(columns={"EVENT_ID": 'EVENT'})[
    ["PATNO", 'EVENT', 'assay', 'outcome', 'month_visit']].copy()
v_cur['source'] = "curated"
v_cur['run'] = pd.NaT

# --- GUARD: the curated data cut applies to the visits at 24 and 150 h the same
#     rule (last run) and the same coding used here. If it fails, the reconstruction
#     cannot use the curated data cut as the source of the outcome at 35 h.
m = v_cur[v_cur.assay != "35h"].merge(v_saa, on=["PATNO", 'EVENT'], how="left",
                                       suffixes=("_cur", "_saa"))
con = m[m.outcome_saa.notna()]
assert (con.assay_cur == con.assay_saa).all() and (con.outcome_cur == con.outcome_saa).all(), (
    'the curated data cut and the SAA file give a different assay or outcome on the same visit')
without = m[m.outcome_saa.isna()]
log(f'\nGuard on the curated data cut. PD visits of the curated data cut with an outcome at 24 or 150 h: {len(m)}.')
log(f'  {len(con)} have a row in the SAA file with the same visit code: protocol')
log("  and outcome coincide in all of them, with the last-run rule and with the indeterminate")
log('  positives at 24 h counted among the inconclusive.')
log(f'  {len(without)} do not have a row with the same visit code in the SAA file:')
for _, r in without.sort_values("PATNO").iterrows():
    elsewhere = v_saa[v_saa.PATNO == r.PATNO]
    log(f'    PATNO {r.PATNO}: curated data cut {r.EVENT} {r.assay_cur} {NAME_CAT[r.outcome_cur]}; '
        f'in the SAA file: '
        + "; ".join(f'{a.EVENT} {a.assay} {NAME_CAT[a.outcome]}' for a in elsewhere.itertuples()))
recoded = set(without.PATNO)
log("  Same protocol and same outcome, and no other row for the subject: it is in")
log('  all likelihood the same sample, which the curated data cut assigns to baseline')
log('  and the SAA file records with another visit code (the file has no')
log('  sample identifier to confirm it). The reconstruction uses the code of the')
log('  SAA file.')

# --- combined visits: SAA file + kinetic parameters at 35 h + 35 h of the curated data cut
c35 = v_cur[v_cur.assay == "35h"]
# (i) same visit in the SAA file and in the kinetic parameters at 35 h: the last run holds
vk = pd.concat([v_saa, v_k35], ignore_index=True)
vk = vk.sort_values(["PATNO", 'EVENT', 'run']).drop_duplicates(["PATNO", 'EVENT'],
                                                                  keep="last")
# (ii) where the curated data cut has a 35 h, that visit belongs to the curated data cut
keys35 = set(zip(c35.PATNO, c35.EVENT))
vk = vk[[k not in keys35 for k in zip(vk.PATNO, vk.EVENT)]]
# GUARD: the 35 h of the curated data cut is consistent with that of the kinetic
# parameters and, where the date is known, was run after the 24/150 h that it replaces
x = c35.merge(v_k35, on=["PATNO", 'EVENT'], suffixes=("_cur", "_k35"))
assert (x.outcome_cur == x.outcome_k35).all(), '35 h: curated data cut and kinetics file discordant'
s_saa = v_saa[[k in keys35 for k in zip(v_saa.PATNO, v_saa.EVENT)]].merge(
    v_k35[["PATNO", 'EVENT', 'run']], on=["PATNO", 'EVENT'], how="left",
    suffixes=("", "_35"))
verifiable = s_saa[s_saa.run_35.notna()]
assert (verifiable.run_35 > verifiable.run).all(), (
    "a 35 h result of the curated data cut replaces a more recent 24/150 h run")
# (iii) the curated data cut reports the screening collection at baseline: an SC row
#       of the kinetic parameters of someone who has a baseline 35 h in the curated data
#       cut is the same sample
bl35 = c35[c35.EVENT == "BL"].set_index("PATNO").outcome
fold = (vk.source == 'kinetic parameters 35 h') & (vk.EVENT == "SC") & vk.PATNO.isin(bl35.index)
assert (vk[fold].outcome == vk[fold].PATNO.map(bl35)).all(), (
    'a 35 h screening sample and the baseline 35 h result of the curated data cut have different outcomes')
run_sc = vk[fold].set_index("PATNO").run
vk = vk[~fold]
# the run date of a 35 h of the curated data cut is known only if the sample is
# detected
c35 = c35.drop(columns='run').merge(v_k35[["PATNO", 'EVENT', 'run']],
                                      on=["PATNO", 'EVENT'], how="left",
                                      validate="one_to_one")
c35['run'] = c35.run.fillna(c35.PATNO.map(run_sc).where(c35.EVENT == "BL"))
vis = pd.concat([vk, c35], ignore_index=True)
assert not vis.duplicated(["PATNO", 'EVENT']).any(), 'duplicate visits in the combined table'
vis['month_visit'] = pd.to_datetime(vis['month_visit'])
vis["vord"] = vis.EVENT.map(ordering_visit)

log(f'\nPD visits with a result, combined table: {len(vis)} ({vis.PATNO.nunique()} participants); by '
    f'source {count(vis.source)}; by assay {count(vis.assay)}')
log(f'  visits in which a 35 h result of the curated data cut replaces a 24/150 h result of the SAA '
    f'file: {len(s_saa)}; the order of the runs can be verified (35 h detected, hence with a date) '
    f'in {len(verifiable)}, and in all of them the 35 h run is later; in the other '
    f'{len(s_saa) - len(verifiable)} the rule of the curated data cut is relied upon')
log(f'  screening rows (SC) of the 35 h kinetics brought back to the baseline of the curated data '
    f'cut and not counted as a separate visit: {int(fold.sum())}, with identical outcome in all of '
    f'them')
k35_outside = vis[vis.source == 'kinetic parameters 35 h']
log(f'  visits known only from the 35 h kinetics file: {len(k35_outside)} '
    f'({count(k35_outside.EVENT)}); at these visits a negative at 35 h would not be visible')

anomalous = vis[vis.vord.isna()]
more_visits = vis.groupby("PATNO").size()
anomalous_relevant = sorted(p for p in set(anomalous.PATNO) if more_visits[p] > 1)
log(f"\nOrder of the visits: visit code (SC < BL < V01 < ...). PD visits with a code without a "
    f"position ({(count(anomalous.EVENT) if len(anomalous) else 'nothing')}): {len(anomalous)}, in "
    f"{anomalous.PATNO.nunique()} participants;")
if anomalous_relevant:
    log(f'  {len(anomalous_relevant)} of these also have other visits: their visit without a '
        f'position is placed last (PATNO {listing(anomalous_relevant)}).')
else:
    log("  none of these has other visits: the order chosen for those codes is irrelevant.")


def earliest_visit(v):
    """The ROW of the first visit of each participant, plus the number of visits."""
    o = v.sort_values(["PATNO", "vord", 'EVENT'], na_position="last")
    p = o.drop_duplicates("PATNO", keep="first").set_index("PATNO").copy()
    p['n_visits'] = o.groupby("PATNO").size()
    return p


GENES_KNOWN = {"LRRK2", "GBA", "SNCA", "PRKN", "PINK1", "PARK7", "VPS35", "Other GV"}
for label in sub[co_cur == "PD"].dropna().unique():      # the PD cohort only
    if label != SPOR:
        parts = {x.strip() for x in label.split("+")}
        assert parts <= GENES_KNOWN | {"Normosmic"}, f'unexpected subgroup: {label}'


def row_A(sg):
    """Row of Table 1 from the subgroup of the curated data cut (main rule)."""
    if pd.isna(sg):
        return WITHOUT_ROW
    if sg == SPOR:
        return "sPD"
    if sg == "Normosmic":
        return "PD normosmic"
    genes = [x.strip() for x in sg.split("+") if x.strip() != "Normosmic"]
    if genes == ["LRRK2"]:
        return "LRRK2 PD"
    if genes == ["GBA"]:
        return "GBA PD"
    return "Other genetic PD"


def row_B(sg):
    """Variant: every label that contains 'Normosmic' goes among the normosmic."""
    if isinstance(sg, str) and "Normosmic" in sg:
        return "PD normosmic"
    return row_A(sg)


rows_tab = []


def table(earliest, variant, rule=row_A, name_rule="A"):
    f = earliest.copy()
    f['row_t1'] = [rule(sub.get(p, np.nan)) for p in f.index]
    out = {}
    for r in ROWS + [WITHOUT_ROW]:
        s = f if r == "PD Total" else f[f.row_t1 == r]
        v = dict(N=len(s), **{c: int((s.outcome == c).sum()) for c in CAT},
                 more=int((s.n_visits > 1).sum()))
        assert sum(v[c] for c in CAT) == v["N"], 'the categories do not add up to N'
        out[r] = v
        pub = PUBLISHED.get(r, {})
        rows_tab.append(dict(
            variant=variant, rule_subgroup=name_rule, row=r,
            N_reconstructed=v["N"], N_published=pub.get("N", np.nan),
            **{f'{c}_reconstructed': v[c] for c in CAT},
            **{f'{c}_published': pub.get(c, np.nan) for c in CAT},
            more_of_one_visit_reconstructed=v['more'],
            more_of_one_visit_published=pub.get('more', np.nan),
            doi_of_published=DOI))
    assert sum(out[r]["N"] for r in ROWS[1:] + [WITHOUT_ROW]) == out["PD Total"]["N"], (
        'the rows do not add up to the total')
    return out, f


def printout(t):
    log(f"  "
        f"{'row':<22}{'N':>12}{'pos/Type I':>13}{'negatives':>11}{'Type II':>10}{'inconcl.':>10}{'>1 visit':>12}")
    log(f"  {'':<22}" + "".join(f"{'ric|pubbl':>{w}}" for w in (12, 13, 11, 10, 10, 12)))
    for r in ROWS + [WITHOUT_ROW]:
        pub = PUBLISHED.get(r)
        cells = []
        for c, w in (("N", 12), ("pos1", 13), ("neg", 11), ("type2", 10), ("inc", 10),
                     ('more', 12)):
            cells.append(f"{str(t[r][c]) + '|' + (str(pub[c]) if pub else '-'):>{w}}")
        log(f"  {r:<22}" + "".join(cells))


v_saa2 = v_saa.assign(vord=v_saa.EVENT.map(ordering_visit))
v_cur2 = v_cur.assign(vord=v_cur.EVENT.map(ordering_visit))
assert v_cur2.vord.notna().all(), 'PD visits of the curated data cut with a code without a position'
T_saa, _ = table(earliest_visit(v_saa2), 'SAA file only (24 and 150 h)')
T_cur, _ = table(earliest_visit(v_cur2), 'curated data cut only (150, 24 and 35 h; BL and annual '
                                         'visits)')
earliest = earliest_visit(vis)
T, earliest = table(earliest, 'combined')
T_B, _ = table(earliest, 'combined', row_B, "B")

log('\nCOMBINED reconstruction (SAA file + 35 h of the curated data cut + 35 h kinetics),')
log('PD participants by outcome at the first visit with a result. In each cell:')
log(f'reconstructed | published (literal, doi {DOI}).')
printout(T)
# --- the rule that assigns the subgroup labels to the rows, stated beside
#     the row that depends on it (PD normosmic)
labels = pd.Series([sub.get(p, np.nan) for p in earliest.index], index=earliest.index)
compound = labels[labels.str.contains("+", regex=False, na=False)]
log('  Rule of assignment to the rows (rule A, used in this table and throughout the')
log(f"  log): 'sPD' = subgroup '{SPOR}'; 'PD normosmic' = the label 'Normosmic' ALONE;")
log("  a composite label (with '+') is assigned by the genes it contains, ignoring")
log("  'Normosmic': LRRK2 only -> 'LRRK2 PD', GBA only -> 'GBA PD', any other gene or")
log("  combination of genes -> 'Other genetic PD'.")
if len(compound):
    log(f'  PD participants with a result and a composite label: {len(compound)}. By label (n; row '
        f'with rule A; row with rule B):')
    for et, n in count(compound).items():
        log(f'    {et:<40} {n:>3}; {row_A(et)}; {row_B(et)}')
else:
    log("  No PD participant with a result has a composite label.")
d_norm = {c: T_B["PD normosmic"][c] - T["PD normosmic"][c] for c in ["N", "pos1"]}
if d_norm["N"] == 0:
    log("  With rule B (every label that contains 'Normosmic' goes to 'PD normosmic') the")
    log('  PD normosmic row does not change.')
else:
    log("  With rule B (every label that contains 'Normosmic' goes to 'PD normosmic') the")
    log(f"  PD normosmic row would be N {T_B['PD normosmic']['N']}, positive/Type I "
        f"{T_B['PD normosmic']['pos1']}, instead of N {T['PD normosmic']['N']}, positive/Type I "
        f"{T['PD normosmic']['pos1']}")
    log(f"  (published: N {PUBLISHED['PD normosmic']['N']}, positive/Type I "
        f"{PUBLISHED['PD normosmic']['pos1']}). The rows of rule B are in the csv.")
log(f'  first visit by code: {count(earliest.EVENT)}')
log(f'  first visit by assay: {count(earliest.assay)}')
log(f"  '{WITHOUT_ROW}': {T[WITHOUT_ROW]['N']} participants of the SAA file or of the 35 h kinetics "
    f"that the curated data cut does not contain; without a subgroup they cannot be assigned to a "
    f"row")
log("  and the article does not say where it counts them.")
log('\nThe sPD row alone, according to the source used (N with a result | positive/Type I')
log('at the first visit):')
for name, t in (('SAA file only (24 and 150 h)', T_saa),
                ('curated data cut only (BL and annual visits)', T_cur),
                ('combined', T)):
    log(f"  {name:<36} {t['sPD']['N']:>5} | {t['sPD']['pos1']:>4}")
log(f"  {'published':<36} {PUBLISHED['sPD']['N']:>5} | {PUBLISHED['sPD']['pos1']:>4}")
assert T_B["sPD"] == T["sPD"], 'the sPD row must not depend on the rule on normosmic participants'
log('  The sPD row does not depend on how the composite labels are assigned (rule B,')
log('  in the csv): only the split between carriers and normosmic participants changes.')

spd = earliest[earliest.row_t1 == "sPD"]
S = set(spd[spd.outcome == "pos1"].index)          # positives/Type I at the first visit, sPD
d_N = T["sPD"]["N"] - PUBLISHED["sPD"]["N"]
d_pos = T["sPD"]["pos1"] - PUBLISHED["sPD"]["pos1"]
log('\nReading, sPD row (reconstructed minus published): '
    + ", ".join(f"{NAME_CAT.get(c, 'N')} {T['sPD'][c] - PUBLISHED['sPD'][c]:+d}"
                for c in ["N"] + CAT) + ".")
if d_pos == 0:
    log(f'  The reconstructed positive/Type I at the first visit are {len(S)}, the same number')
    log("  as published. It is an equality of counts: the list of participants")
    log("  of the article is not public and the identity of the persons cannot be verified.")
else:
    log(f"  The reconstructed positive/Type I at the first visit are {len(S)}, {abs(d_pos)} "
        f"{('more' if d_pos > 0 else 'fewer')} than published.")
if d_N != 0:
    log(f'  The number of sPD with a result differs by {d_N:+d} from the published one; the')
    log("  difference cannot be attributed with the public files.")
d_tot = {c: T["PD Total"][c] - PUBLISHED["PD Total"][c] for c in ["N"] + CAT}
log('Reading, PD Total row (reconstructed minus published): '
    + ", ".join(f"{NAME_CAT.get(c, 'N')} {d_tot[c]:+d}" for c in ["N"] + CAT) + ".")
no_spd = {c: sum(T[r][c] - PUBLISHED[r][c] for r in ROWS[2:]) + T[WITHOUT_ROW][c]
           for c in ["N"] + CAT}
log(f"  Of this difference, outside the sPD row: N {no_spd['N']:+d}, positive/Type I "
    f"{no_spd['pos1']:+d} (including the {T[WITHOUT_ROW]['N']} without a row).")

# arithmetic of the published figures alone: what the 1171 is made of
pub_tot = PUBLISHED["PD Total"]["pos1"]
pub_no_spd = sum(PUBLISHED[r]["pos1"] for r in ROWS[2:])
assert pub_tot - pub_no_spd == PUBLISHED["sPD"]["pos1"]
log(f"\nPublished figures only (doi {DOI}): {pub_tot} = sPD {PUBLISHED['sPD']['pos1']} + "
    + " + ".join(f"{r} {PUBLISHED[r]['pos1']}" for r in ROWS[2:])
    + f". The non-sporadic participants are {pub_no_spd}; the term of comparison for a sporadic-only "
      f"population is {PUBLISHED['sPD']['pos1']}, not {pub_tot}.")

# --- what the public files do not allow to be reconstructed
log("\nWHAT CANNOT BE RECONSTRUCTED FROM THE PUBLIC FILES")
month_download = DOWNLOAD_PUBLISHED.to_period("M").to_timestamp()
log(f'(1) The download. The article states data obtained on {DOWNLOAD_PUBLISHED.date()} (published '
    f'literal). The dates contained')
log('    in the files read here, compared with that date (all cohorts, all rows):')
# that a file is later than the download of the article is inferred from the dates
# it CONTAINS (last run, last visit month), not from its name
ult_saa, ult_35 = d.RUNDATE.max(), k35.RUNDATE.max()
months_cur = pd.to_datetime(cur.visit_date, format="%m/%Y", errors="coerce")
ult_month = months_cur.max()
ult_month_saa = months_cur[cur.CSFSAA.notna()].max()
proofs = []          # sources with at least one date later than the download
for name, file, ult in (('SAA file (24 and 150 h)', os.path.basename(SAA), ult_saa),
                        ('kinetic parameters at 35 h', os.path.basename(KIN35), ult_35)):
    after_f = bool(ult > DOWNLOAD_PUBLISHED)
    if after_f:
        proofs.append(name)
    log(f"      {name}, {file}: last run {ult.date()}, "
        f"{('LATER than the' if after_f else 'not later than the')} download of the article")
if pd.isna(ult_month):
    log(f'      curated data cut ({DESCR_CUR}): visit month not readable in any row;')
    log("      the comparison with the download of the article is NOT verified for this source")
    no_verified("curated data cut: comparison of the visit month with the download of the "
                   "article (month not readable)")
else:
    after_f = bool(ult_month > month_download)
    if after_f:
        proofs.append("curated")
    log(f'      curated data cut ({DESCR_CUR}):')
    log(f"        last visit month {ult_month.strftime('%m/%Y')} "
        f"({(ult_month_saa.strftime('%m/%Y') if pd.notna(ult_month_saa) else 'noone')} among the "
        f"visits with an assay outcome), {('LATER than the' if after_f else 'not later than the')} "
        f"month of the download ({month_download.strftime('%m/%Y')}); months not readable: "
        f"{int(months_cur.isna().sum())} rows out of {len(cur)}")
DOWNLOAD_POSTERIOR = bool(proofs)
if DOWNLOAD_POSTERIOR:
    without_trial = [n for n in ('SAA file (24 and 150 h)', 'kinetic parameters at 35 h', "curated")
                   if n not in proofs]
    log(f"    They contain dates later than the download of the article: {', '.join(proofs)}. For "
        f"these sources the download")
    log("    used here is therefore later than that of the article."
        + (f" For the others ({', '.join(without_trial)}) the dates" if without_trial else ""))
    if without_trial:
        log("    they contain do not demonstrate it: none is later than the download of the article.")
else:
    log("    No date contained in the files is later than the download of the article: that the")
    log("    download used here is later is NOT demonstrated by the data.")
p35 = earliest[earliest.assay == "35h"].copy()
p35['after_run'] = p35.run > DOWNLOAD_PUBLISHED
# the curated data cut gives only month and year: a month LATER than that of the
# download counts as "after"
p35['after_visit'] = p35.month_visit > month_download
p35['after'] = p35.after_run | p35.after_visit
p35['unknown'] = p35.run.isna() & p35.month_visit.isna()
in_month = p35.month_visit == month_download          # visit in the month of the download
log(f"    PD participants whose first visit with a result is at 35 h: {len(p35)}. Of")
log(f'    these, {int(p35.after.sum())} have a 35 h run later than the download (known only')
log('    for the detected) or a visit month later than that of the download: they')
log("    could not be in the article. It is a MINIMUM: the date on which a result")
log("    entered the database is not public, and a first visit in the month of the download")
log("    counts only if the 35 h run is later than the download. First visits at 35 h")
log(f'    in the month of the download: {int(in_month.sum())}, of which '
    f'{int((in_month & p35.after).sum())} already counted by the run date and '
    f'{int((in_month & ~p35.after).sum())} NOT counted.')
for r in ROWS[1:] + [WITHOUT_ROW]:
    s = p35[p35.row_t1 == r]
    if len(s):
        log(f"      {r:<20} first visit at 35 h {len(s):>4}; certainly after the download "
            f"{int(s.after.sum()):>4}; published N {PUBLISHED.get(r, {}).get('N', '-')}, "
            f"reconstructed N {T[r]['N']}")
# the split by row depends on the rule on the composite labels: this is said
row_B_p35 = pd.Series([row_B(sub.get(p, np.nan)) for p in p35.index], index=p35.index)
after_A = {r: int(p35[p35.row_t1 == r].after.sum()) for r in ROWS[1:] + [WITHOUT_ROW]}
after_B = {r: int(p35[row_B_p35 == r].after.sum()) for r in ROWS[1:] + [WITHOUT_ROW]}
assert sum(after_A.values()) == sum(after_B.values()) == int(p35.after.sum())
if after_A == after_B:
    log("      (rows according to rule A; with rule B the split of the 'certainly after'")
    log("      is the same)")
else:
    log(f"      (rows according to rule A; with rule B the {int(p35.after.sum())} 'certainly after' "
        f"are split: "
        + ", ".join(f"{r} {n}" for r, n in after_B.items() if n) + ")")
n_spd_after = int(p35[(p35.row_t1 == "sPD")].after.sum())
if n_spd_after == 0:
    log("    In the sPD row no first visit at 35 h is certainly later than the")
    log('    download: for this row the difference in download has no demonstrable effect.')
else:
    log(f'    In the sPD row {n_spd_after} first visits at 35 h are certainly later than the download.')
log("(2) The dates. The article states that it used the real dates of activity of the")
log("    participants, a restricted datum; that it ordered the visits by date is an")
log("    inference of ours. Here the order of the visit codes holds.")


# How much the choice matters: the only public date is the visit month of the
# curated data cut (month and year). For each participant with more than one visit
# it is checked whether a visit other than the one chosen by code has an EARLIER
# month (the first visit would change) or the SAME month (the month does not decide).
def comparison_month(v, name):
    """First visit by code against first visit by month of the curated data cut."""
    o = v.sort_values(["PATNO", "vord", 'EVENT'], na_position="last")
    n_vis = o.groupby("PATNO").size()
    more = o[o.PATNO.map(n_vis) > 1]
    known = more.assign(c=more.month.notna()).groupby("PATNO").c.all()
    conf = more[more.PATNO.map(known)]
    first = conf.drop_duplicates("PATNO", keep="first").set_index("PATNO").month
    others = conf[conf.duplicated("PATNO", keep="first")]
    m_others = others.groupby("PATNO").month.min()
    assert set(first.index) == set(m_others.index)
    changes = sorted(first.index[(m_others.reindex(first.index) < first).to_numpy()])
    ties = sorted(first.index[(m_others.reindex(first.index) == first).to_numpy()])
    log(f'    {name}: PD participants with more than one visit {int((n_vis > 1).sum())}; month known '
        f'at all visits in {len(first)}.')
    log(f'      Among these, the first visit by month is a different visit from that by code in '
        f'{len(changes)}'
        + (f' (PATNO {listing(changes)})' if changes else "")
        + f'; in {len(ties)} another visit falls in the same month and the month does not decide'
        + (f' (PATNO {listing(ties)})' if ties else "") + ".")
    if len(first) < int((n_vis > 1).sum()):
        log(f'      For the other {int((n_vis > 1).sum()) - len(first)} the comparison CANNOT be '
            f'verified: at least one visit has no month in the curated data cut.')
    return changes, ties, len(first)


month_of = dict(zip(zip(cur.PATNO, cur.EVENT_ID), months_cur))      # (PATNO, visit) -> month
# stable order: first the codes with a position, then the others in alphabetical order
codes_cur = sorted(set(cur.EVENT_ID),
                    key=lambda e: ((1, 0.0, e) if pd.isna(ordering_visit(e))
                                   else (0, ordering_visit(e), e)))
log('    Check possible with the public data: the visit month of the curated data cut (month and')
log('    year), present only for the visits that the curated data cut contains. Visit codes')
log(f"    of the curated data cut: {', '.join(codes_cur)}.")
changes_cur, ties_cur, n_conf_cur = comparison_month(
    v_cur2.assign(month=v_cur2.month_visit),
    'Visits of the curated data cut with an assay outcome')
vis_m = vis.assign(month=[month_of.get((p, e), pd.NaT) for p, e in zip(vis.PATNO, vis.EVENT)])
changes_vis, ties_vis, n_conf_vis = comparison_month(
    vis_m, 'Combined table (that of the reconstruction)')
if n_conf_cur == 0 and n_conf_vis == 0:
    log("    No participant has the month at all visits: the effect of the order by")
    log("    code on the first visit is NOT verified.")
    no_verified('order of the visits: comparison between visit code and visit month (no '
                   'participant with the month at all visits)')
elif not changes_cur and not changes_vis:
    log("    Where the month is known, ordering by month instead of by code does not change the")
    log("    first visit of any participant. It is not a check on the real dates: the")
    log('    month does not distinguish two visits of the same month and is missing for the visits that the')
    log('    curated data cut does not contain.')
else:
    log("    Where the month is known, ordering by month instead of by code would change the")
    log(f'    first visit of {len(set(changes_cur) | set(changes_vis))} participants: for them the '
        f'outcome at the first visit depends on the choice.')
log('(3) The negatives at 35 h. The 35 h kinetics file contains only detected')
log('    samples; negatives and inconclusives at 35 h are seen only in the curated data cut, which')
log('    covers the baseline and the annual visits of the enrolled participants.')
n35 = vis[vis.assay == "35h"]
log(f"    PD results at 35 h used: {len(n35)}, of which {int((n35.source == 'curated').sum())} from "
    f"the curated data cut ({count(n35[n35.source == 'curated'].outcome)}) and "
    f"{int((n35.source != 'curated').sum())} from the kinetics only.")
log(f"(4) The subgroup of those who are not in the curated data cut ({T[WITHOUT_ROW]['N']} PD "
    f"participants).")
log("A reconstruction close or equal to the published figure in a row is not a replication")
if DOWNLOAD_POSTERIOR:
    log("of the article: it is the same rule applied to a different download.")
else:
    log("of the article: it is the same rule applied to files whose download cannot be")
    log("dated relative to that of the article.")

# ============================================================================
# 3. BREAKDOWN OF THE DIFFERENCE
# ============================================================================
title('3. DECOMPOSITION OF THE DIFFERENCE BETWEEN THE sPD ROW AND OUR DISTINCT PERSONS')
common, only_S, only_ours = S & ours, S - ours, ours - S
assert len(S) == len(common) + len(only_S)
assert len(ours) == len(common) + len(only_ours)
assert len(S) - len(ours) == len(only_S) - len(only_ours)
log(f'Positive/Type I at the first visit, reconstructed sPD row: {len(S)} participants.')
log(f'Our distinct persons (union of the two cohorts):          {len(ours)} participants.')
log(f'  in both sets                                           {len(common):>5}')
log(f'  in the sPD row but not among ours                      {len(only_S):>5}')
log(f'  among ours but not in the sPD row                      {len(only_ours):>5}')
log(f'  check: {len(S)} - {len(ours)} = {len(S) - len(ours)} = {len(only_S)} - {len(only_ours)}')
outside_row = ours - set(spd.index)
log(f'  of ours, outside the reconstructed sPD row (any outcome): {len(outside_row)}')

# Each participant of the sPD row who is not among ours stops at a precise step
# of our flow (union of the protocols): the first one not passed.
# The first six reasons are the definition of the step. The seventh (incomplete) has
# no fixed text: for each participant it is derived from what he or she is missing
# (reason_incomplete), and the label of the piece is built from the counts.
REASONS = [
    'never tested at 24 or 150 h (35 h only)',
    'tested at 24 or 150 h, but not on the baseline sample',
    'baseline at 24 or 150 h not positive',
    'positive baseline without an admitted pattern',
    'without a baseline row in the curated data cut',
    'not sporadic at the baseline of the curated data cut',
]
STEP_INCOMPLETE = len(REASONS)            # index of step 7 in the union
assert STEP_INCOMPLETE == len(FU) - 1


def reason_of_step(p, i):
    """Reason for exclusion of a participant stopped at step i of the union."""
    return reason_incomplete(p) if i == STEP_INCOMPLETE else REASONS[i]


def step_missed(p):
    for i, s in enumerate(FU):
        if p not in s:
            return i
    return len(FU)


det = earliest.loc[sorted(only_S), ['EVENT', 'assay', 'source']].copy()
det['step'] = [step_missed(p) for p in det.index]
assert (det.step < len(FU)).all(), 'a participant outside ours passes all the steps'
det['reason'] = [reason_of_step(p, i) for p, i in zip(det.index, det.step)]
det['earliest_a_35h'] = det.assay == "35h"
rows_departure = []
log(f'\nThe {len(only_S)} of the sPD row who are not among ours, by the first step of')
log("our flow that they do not pass (exclusive categories, in the order of the flow):")
total = 0
for i in range(len(FU)):
    s = det[det.step == i]
    total += len(s)
    # step 7: the label is built from the derived reasons of the participants of the piece
    mot = label_incomplete(set(s.index)) if i == STEP_INCOMPLETE else REASONS[i]
    if i == STEP_INCOMPLETE and not len(s):
        mot = 'incomplete on the seven analysis variables'
    if len(s):
        log(f'  {mot:<52} {len(s):>4}   first visit: '
            + ", ".join(f"{k[1]} a {k[0]} = {v}" for k, v in
                        sorted(s.groupby(['EVENT', 'assay']).size().items())))
    rows_departure.append(dict(dataset='sPD row reconstructed, not among ours',
                             piece=mot, n=len(s),
                             of_which_earliest_visit_a_35h=int(s.earliest_a_35h.sum())))
assert total == len(only_S), 'the pieces of the decomposition do not add up'
log(f"  {'total':<52} {total:>4}   (= {len(only_S)}, asserted)")

# the same breakdown read in three items
n_35 = int(det.earliest_a_35h.sum())
n_no_bl = int((~det.earliest_a_35h & (det.step < 6)).sum())
n_incomplete = int((~det.earliest_a_35h & (det.step == 6)).sum())
assert n_35 + n_no_bl + n_incomplete == len(only_S), 'the three items do not add up'
no_bl = det[~det.earliest_a_35h & (det.step < 6)]
log('\nThe same decomposition in three items (exclusive; the first takes precedence):')
log(f'  (a) first result on the 35 h assay                            {n_35:>4}')
log(f'      of which never tested at 24 or 150 h '
    f'{int((det.earliest_a_35h & (det.step == 0)).sum())}, tested at 24 or 150 h only at a later '
    f'visit {int((det.earliest_a_35h & (det.step == 1)).sum())}, other '
    f'{int((det.earliest_a_35h & (det.step > 1)).sum())}')
log(f'  (b) first result at 24 or 150 h, but outside our flow           {n_no_bl:>4}')
log(f"      by step: {(count(no_bl.reason) if len(no_bl) else 'noone')}")
log(f"      by code of the first visit: {(count(no_bl.EVENT) if len(no_bl) else 'noone')}")
incomplete_c = set(det[~det.earliest_a_35h & (det.step == STEP_INCOMPLETE)].index)
assert len(incomplete_c) == n_incomplete
item_c = '(c) admitted at baseline, excluded because incomplete'
log(f'  {item_c:<64}{n_incomplete:>4}')
log(f'      reason, derived per participant: {label_incomplete(incomplete_c)}')
log(f'  sum                                                            '
    f'{n_35 + n_no_bl + n_incomplete:>4}   (= {len(only_S)}, asserted)')
n_recod = len(set(no_bl.index) & recoded)
if n_recod:
    log(f'  Note on (b): for {n_recod} of these ({listing(set(no_bl.index) & recoded)}) the curated '
        f'data cut reports at baseline')
    log('  an outcome that the SAA file records with another visit code; our')
    log('  flow selects on the code of the SAA file and therefore does not include them.')
incomplete_stream = FU[5] - FU[6]
assert set(det[det.step == 6].index) <= incomplete_stream
log(f'  Note on (c): the admitted and incomplete sporadic participants of our flow are '
    f'{len(incomplete_stream)} in the union ({len(F24[5] - F24[6])} at 24 h, '
    f'{len(F150[5] - F150[6])} at 150 h);')
log(f'  {len(set(det[det.step == 6].index))} of them are positive/Type I at the first visit in the '
    f'sPD row, of which {n_incomplete} with the first result at 24 or 150 h.')

# ============================================================================
# 4. OUR PARTICIPANTS WHOM COUGHLIN'S RULE DOES NOT COUNT AMONG THE POSITIVES
# ============================================================================
title('4. OUR PARTICIPANTS NOT POSITIVE/TYPE I AT THE FIRST VISIT ACCORDING TO COUGHLIN')
log(f'Participants: {len(only_ours)} (PATNO {listing(only_ours)}). For each, all the rows')
log("of the SAA file, the known 35 h results and the outcome at the first visit according to the rule.")
profile = []
for p in sorted(only_ours):
    cohorts = [n for n, s in (('24 h', p24), ('150 h', p150)) if p in s]
    pv = earliest.loc[p]
    log(f"\n  PATNO {p} — in our cohorts: {' e '.join(cohorts)}; row {pv.row_t1}; first visit "
        f"{pv.EVENT}, the assay that holds is {pv.assay} ({pv.source}): {NAME_CAT[pv.outcome]}")
    rr = g[g.PATNO == p].sort_values(["CLINICAL_EVENT", "RUNDATE", 'row'])
    for r in rr.itertuples():
        kind = "" if pd.isna(r.SAA_Type) else f"/{r.SAA_Type}"
        log(f'      {r.CLINICAL_EVENT:<4} {r.assay:<5} {r.SAA_Status}{kind:<14} run {r.RUNDATE.date()}')
    for r in vis[(vis.PATNO == p) & (vis.assay == "35h")].sort_values("vord").itertuples():
        when = 'run date not public' if pd.isna(r.run) else f'run {r.run.date()}'
        log(f'      {r.EVENT:<4} 35h   {NAME_CAT[r.outcome]} ({r.source}; {when})')
    bl = rr[rr.CLINICAL_EVENT == "BL"]
    b150, b24 = bl[bl.PROJECTID == 155], bl[bl.PROJECTID == 237]
    ritest = (len(b150) > 0 and len(b24) > 0 and (b150.SAA_Status == "Positive").any()
              and b24.RUNDATE.max() > b150.RUNDATE.max()
              and b24.sort_values(["RUNDATE", 'row']).iloc[-1].outcome == "type2")
    profile.append(dict(patno=p, in_150=p in p150, in_24=p in p24, ritest=bool(ritest),
                        outcome=pv.outcome, event=pv.EVENT,
                        t1_after=bool(((vis.PATNO == p) & (vis.vord > pv.vord)
                                      & (vis.outcome == "pos1")).any())))
prof = pd.DataFrame(profile)
if len(prof):
    n_withdr = int(prof.ritest.sum())
    log(f'\nSummary: {int((prof.in_150 & ~prof.in_24).sum())} out of {len(prof)} are in the 150 h '
        f'cohort only, {int((prof.in_24 & ~prof.in_150).sum())} in the 24 h cohort only, '
        f'{int((prof.in_24 & prof.in_150).sum())} in both.')
    log(f'  In {n_withdr} out of {len(prof)} the baseline sample, positive at 150 h, was rerun at 24 '
        f'h with outcome Type2: by the last-run rule the first visit counts as Type II.')
    if n_withdr == len(prof):
        log("  They are in our 150 h cohort because our criterion is")
        log("  positivity at 150 h at baseline, a protocol in which the type does not exist;")
        log('  they are not in the 24 h cohort, which admits only Type1.')
    else:
        log('  The others do not fit this pattern: see the rows above.')
    log(f'  At a later visit the combined table records a positive/Type I in '
        f'{int(prof.t1_after.sum())} out of {len(prof)}.')
else:
    log('None: all our distinct persons are positive/Type I at the first visit.')

# how frequent, in the 150 h cohort, the re-run at 24 h of the baseline sample is
bl24 = v_saa[(v_saa.EVENT == "BL") & (v_saa.assay == "24h") & v_saa.PATNO.isin(p150)]
bl24_every = g[(g.CLINICAL_EVENT == "BL") & (g.PROJECTID == 237) & g.PATNO.isin(p150)]
assert set(bl24.PATNO) == set(bl24_every.PATNO)
log(f"\n150 h cohort (n = {len(p150)} participants): the baseline sample was also")
log(f"run at 24 h in {len(bl24)}. Outcome of the last 24 h run: "
    + ", ".join(f'{NAME_CAT[c]} {int((bl24.outcome == c).sum())}' for c in CAT) + ".")
no_t1 = set(bl24[bl24.outcome != "pos1"].PATNO)
assert no_t1 == only_ours & p150, (
    'the participants of the 150 h cohort with a non-Type I baseline at 24 h do not coincide with '
    'those that the rule of Coughlin does not count among the positives')
log(f'  The non-positive/Type I at 24 h are {len(no_t1)}, the same PATNO listed above.')
t1_24 = set(bl24[bl24.outcome == "pos1"].PATNO)
log('  The positive/Type I at 24 h '
    + ('are exactly the participants shared by the two cohorts.'
       if t1_24 == shared else
       f'do not coincide with those shared by the two cohorts: {len(t1_24 - shared)} are not in the '
       f'24 h cohort, {len(shared - t1_24)} shared participants are not Type I at the last run.'))
log(f"  For the other {len(p150) - len(bl24)} the baseline sample was not run at 24 h: for them the "
    f"150 h outcome is the only one available at baseline.")
rows_departure.append(dict(dataset='among ours, not positive/Type I at the first visit (sPD row)',
                         piece='baseline positive at 150 h, re-run at 24 h as Type2',
                         n=int(prof.ritest.sum()) if len(prof) else 0,
                         of_which_earliest_visit_a_35h=int((earliest.loc[sorted(only_ours)].assay
                                                        == "35h").sum())))
rows_departure.append(dict(dataset='among ours, not positive/Type I at the first visit (sPD row)',
                         piece='other',
                         n=int((~prof.ritest).sum()) if len(prof) else 0,
                         of_which_earliest_visit_a_35h=0))
rows_departure.append(dict(dataset='in both sets', piece="", n=len(common),
                         of_which_earliest_visit_a_35h=int((earliest.loc[sorted(common)].assay
                                                        == "35h").sum())))

# ============================================================================
# 5. SUMMARY
# ============================================================================
title('5. SUMMARY')
log(f"Published (doi {DOI}): {PUBLISHED['PD Total']['pos1']} PD positive/Type I at the first visit "
    f"out of {PUBLISHED['PD Total']['N']} with a result, all the")
log(f"  subgroups and the three assays; of these {PUBLISHED['sPD']['pos1']} are sPD.")
# the reconstructed counterpart of the PD total, beside the published one
log(f"PD Total row reconstructed from the public files: {T['PD Total']['pos1']} PD positive/Type I "
    f"at the first visit out of {T['PD Total']['N']} with a result")
if d_tot["N"] == 0 and d_tot["pos1"] == 0:
    log(f"  (published {PUBLISHED['PD Total']['pos1']} out of {PUBLISHED['PD Total']['N']}): the two "
        f"counts coincide with the published ones.")
else:
    log(f"  (published {PUBLISHED['PD Total']['pos1']} out of {PUBLISHED['PD Total']['N']}): "
        f"positive/Type I {d_tot['pos1']:+d}, N {d_tot['N']:+d} relative to the published figures. "
        f"The PD Total row is NOT reproduced.")
log(f'Ours: {len(p24)} + {len(p150)} = {len(p24) + len(p150)} entries, {len(ours)} distinct persons '
    f'({len(shared)} in both cohorts): sporadic,')
log("  baseline visit, 24 h assay (Type1) or 150 h assay, DAT-SPECT and age complete.")
log(f"sPD row reconstructed from the public files: {T['sPD']['N']} with a result, {len(S)} "
    f"positive/Type I at the first visit.")
log(f'  {len(S)} = {len(common)} among ours + {n_35} with the first result at 35 h + {n_no_bl} with '
    f'the first result at 24 or 150 h outside the flow + {n_incomplete} excluded for incomplete '
    f'data.')
log(f'  {len(ours)} = {len(common)} + {len(only_ours)} that the rule of the article does not count '
    f'among the positive/Type I at the first visit.')
assert len(S) == len(common) + n_35 + n_no_bl + n_incomplete
# the guards provided for by the script that could not run in this run
if no_executed:
    log(f'Guards provided for by the script and NOT executed in this run: {len(no_executed)}.')
    for item in no_executed:
        log(f'  NOT verified: {item}')
else:
    log('Guards provided for by the script and NOT executed in this run: none.')

# ------------------------------------------------------------------ output
fl = pd.DataFrame(rows_stream)
for c in ('removed', 'only_24h', 'only_150h', 'both'):
    fl[c] = fl[c].astype("Int64")
fl.insert(0, 'units', 'PD participants')
fl.to_csv(os.path.join(OUT, "109_reconciliation_flow.csv"), index=False)

tb = pd.DataFrame(rows_tab)
for c in tb.columns:
    if c.endswith('_published'):
        tb[c] = tb[c].astype("Int64")
tb.insert(0, 'units', 'PD participants, outcome at the first visit with a result')
tb.to_csv(os.path.join(OUT, "109_reconciliation_table1.csv"), index=False)

sc = pd.DataFrame(rows_departure)
sc.insert(0, 'units', 'participants')
sc.to_csv(os.path.join(OUT, "109_reconciliation_departure.csv"), index=False)

everyone = sorted(S | ours)
sg = pd.DataFrame(dict(
    PATNO=everyone,
    cohort_24h=[p in p24 for p in everyone],
    cohort_150h=[p in p150 for p in everyone],
    row_sPD_pos_earliest_visit=[p in S for p in everyone],
    earliest_visit_code=[earliest.EVENT.get(p, "") for p in everyone],
    earliest_visit_assay=[earliest.assay.get(p, "") for p in everyone],
    earliest_visit_outcome=[earliest.outcome.get(p, "") for p in everyone],
    earliest_visit_source=[earliest.source.get(p, "") for p in everyone],
    steps_of_stream_passed=[step_missed(p) for p in everyone],
    reason_exclusion=[("" if p in ours else reason_of_step(p, step_missed(p)))
                       for p in everyone]))
assert len(sg) == len(S | ours) and sg.PATNO.is_unique
sg.to_csv(os.path.join(OUT, "109_reconciliation_subjects.csv"), index=False)

with open(os.path.join(OUT, "109_coughlin_reconciliation.txt"), "w") as f:
    f.write("\n".join(rep) + "\n")
print(f'\nWritten in {OUT}: 109_reconciliation_flow.csv, 109_reconciliation_table1.csv, '
      f'109_reconciliation_departure.csv, 109_reconciliation_subjects.csv, '
      f'109_coughlin_reconciliation.txt')
