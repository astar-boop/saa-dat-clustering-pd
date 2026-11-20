"""
76 - Exact reconciliation of ALL the denominators (samples, subjects,
     wells, parameter-well measurements) of the reliability analysis.

The review points out four arithmetical discrepancies:

  1. 622 samples (24h) + 336 samples (150h), five parameters, three wells
     -> (622+336)*5 = 4790 parameter-sample pairs and 14370 measurements. The
     manuscript writes 4784 and 14352.
  2. 491*3*5 = 7365, the manuscript writes 7362.
  3. 732*3*5 = 10980, the manuscript writes 10978.
  4. The reliability population of the 24h has 615 subjects, the primary
     sporadic chain has 601 before the restriction to complete cases.

The discrepancy is not a typo: it arises from TWO different counting conventions.

  A. 14352 = 3 x 4784, where 4784 is the sum of the column `n_subjects` of
     21_reliability_kinetics.csv. That column is the number of COMPLETE
     TRIPLETS: `icc_from_reps` in 21_review_P1.py drops entirely the
     rows with at least one unusable well
     (`m = mat[~np.isnan(mat).any(axis=1)]`), after the non-positive values
     have already been made missing by `np.log(np.where(v > 0, v, np.nan))`.
     LISTWISE count, by triplets.

  B. 10978 and 7362 come from `count_wells` in 66_external_review.py, which
     counts the single non-missing wells (`x.notna()`), on the raw scale,
     without dropping the triplet. PER-WELL count.

The script rebuilds both counts from the raw file, parameter by
parameter, and locates every single lost well.

Usage:
    CLUSTERING_BASE=".../run_sporadici" python3 76_denominators.py

Output: outputs/76_denominators.csv, outputs/76_denominators.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from _paths import DATA

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA_FILE = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
CURATED = os.path.join(OUT, "cache", "curated.parquet")

K24 = ["Fmax", "TTT", "AUC", "TSmax", "SLOPEMax"]
K150 = ["Fmax", "TTT", "AUC", "T50", "SLOPE"]

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# ===========================================================================
# raw data
# ===========================================================================
saa = pd.read_csv(SAA_FILE, low_memory=False)
cur = pd.read_parquet(CURATED)
bl = cur[cur.EVENT_ID == "BL"]
assert bl.PATNO.is_unique, 'the curated data cut at BL does not have a unique PATNO'
SPORADIC = set(bl.loc[bl.subgroup == "Sporadic PD", "PATNO"])

s24 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237) &
          (saa.CLINICAL_EVENT == "BL")]
s150 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 155) &
           (saa.CLINICAL_EVENT == "BL")]
pos24 = s24[s24.SAA_Status == "Positive"]
t1_24 = pos24[pos24.SAA_Type == "Type1"]
pos150 = s150[s150.SAA_Status == "Positive"]

POP = [
    ("24h_PD_basale", "24h", s24, K24,
     "COHORT=PD & PROJECTID=237 & CLINICAL_EVENT=BL"),
    ("24h_positivi_tutti_i_tipi", "24h", pos24, K24,
     '... & SAA_Status=Positive  (any SAA_Type)'),
    ("24h_positivi_Type1", "24h", t1_24, K24,
     "... & SAA_Status=Positive & SAA_Type=Type1"),
    ("24h_affidabilita_sporadici", "24h", pos24[pos24.PATNO.isin(SPORADIC)], K24,
     '... & SAA_Status=Positive & subgroup=Sporadic PD  (any SAA_Type)'),
    ("24h_Type1_sporadici", "24h", t1_24[t1_24.PATNO.isin(SPORADIC)], K24,
     "... & Positive & Type1 & subgroup=Sporadic PD"),
    ("150h_PD_basale", "150h", s150, K150,
     "COHORT=PD & PROJECTID=155 & CLINICAL_EVENT=BL"),
    ("150h_positivi", "150h", pos150, K150,
     '... & SAA_Status=Positive  (SAA_Type absent in Project 155)'),
    ("150h_affidabilita_sporadici", "150h", pos150[pos150.PATNO.isin(SPORADIC)],
     K150, "... & SAA_Status=Positive & subgroup=Sporadic PD"),
]


def cols_of(par, proto):
    return [f"{par}_{proto}_Rep{i}" for i in (1, 2, 3)]


def count(sub, par, proto):
    """Complete accounting of one parameter in one population of samples."""
    cols = cols_of(par, proto)
    for c in cols:
        assert c in sub.columns, f'column missing in the raw file: {c}'
    v = sub[cols].apply(pd.to_numeric, errors="coerce")
    arr = v.to_numpy(float)
    n = len(sub)

    miss = np.isnan(arr)                       # well absent from the file
    nonpos = (~miss) & (arr <= 0)              # present but <= 0: lost by the log
    unusable = miss | nonpos                     # unusable on the log scale

    by_samp = miss.sum(axis=1)
    trip_complete = int((~unusable.any(axis=1)).sum())

    idx_miss = sub.index[miss.any(axis=1)]
    idx_nonpos = sub.index[nonpos.any(axis=1)]
    return {
        'samples': n,
        'subjects': int(sub.PATNO.nunique()),
        'parameter': par,
        'wells_nominals': n * 3,
        'wells_no_missing': int((~miss).sum()),
        'wells_missing': int(miss.sum()),
        'samples_having_1_well_missing': int((by_samp == 1).sum()),
        'samples_having_2_wells_missing': int((by_samp == 2).sum()),
        'samples_having_3_wells_missing': int((by_samp == 3).sum()),
        'wells_no_positives': int(nonpos.sum()),
        'triplicates_complete_scale_log': trip_complete,
        'wells_in_triplicates_complete': trip_complete * 3,
        'samples_lost_offthe_triplet': n - trip_complete,
        'PATNO_wells_missing': ";".join(
            str(int(x)) for x in sub.loc[idx_miss, "PATNO"]),
        'PATNO_wells_no_positives': ";".join(
            str(int(x)) for x in sub.loc[idx_nonpos, "PATNO"]),
    }


# ===========================================================================
# table
# ===========================================================================
rows = []
for name, proto, sub, params, definition in POP:
    tot = {'wells_nominals': 0, 'wells_no_missing': 0,
           'wells_missing': 0, 'wells_no_positives': 0,
           'triplicates_complete_scale_log': 0,
           'wells_in_triplicates_complete': 0,
           'samples_lost_offthe_triplet': 0,
           'samples_having_1_well_missing': 0,
           'samples_having_2_wells_missing': 0,
           'samples_having_3_wells_missing': 0}
    for par in params:
        r = count(sub, par, proto)
        r.update({'population': name, 'protocol': proto,
                  'definition': definition})
        rows.append(r)
        for k in tot:
            tot[k] += r[k]
    rows.append({'population': name, 'protocol': proto,
                  'definition': definition, 'parameter': 'TOTAL_5_PARAMETERS',
                  'samples': len(sub), 'subjects': int(sub.PATNO.nunique()),
                  'PATNO_wells_missing': "", 'PATNO_wells_no_positives': "",
                  **tot})

COLS = ['population', 'protocol', 'definition', 'parameter', 'samples',
        'subjects', 'wells_nominals', 'wells_no_missing',
        'wells_missing', 'samples_having_1_well_missing',
        'samples_having_2_wells_missing', 'samples_having_3_wells_missing',
        'wells_no_positives', 'triplicates_complete_scale_log',
        'wells_in_triplicates_complete', 'samples_lost_offthe_triplet',
        'PATNO_wells_missing', 'PATNO_wells_no_positives']
tab = pd.DataFrame(rows)[COLS]
tab.to_csv(os.path.join(OUT, "76_denominators.csv"), index=False)


def block(name):
    return tab[tab.population == name]


def val(name, column, parameter='TOTAL_5_PARAMETERS'):
    s = tab[(tab.population == name) & (tab.parameter == parameter)]
    return int(s[column].iloc[0])


# ===========================================================================
# report
# ===========================================================================
log("=" * 78)
log('76 — RECONCILIATION OF THE DENOMINATORS (samples, subjects, wells)')
log("=" * 78)
log(f'raw file      : {SAA_FILE}')
log(f'curated cache : {CURATED}')
log(f"table         : {os.path.join(OUT, '76_denominators.csv')}")
log("")
log('Two counting conventions coexist in the manuscript and produce')
log('different numbers on the same data:')
log('  [A] LISTWISE by triplet — used for 4784 and 14352. Source:')
log('      icc_from_reps() in scripts/21_review_P1.py, line')
log('        m = mat[~np.isnan(mat).any(axis=1)]')
log('      preceded by  v = np.log(np.where(v > 0, v, np.nan)).')
log("      It drops the WHOLE triplet if even a single well is missing or is")
log("      non-positive (the logarithm is not defined).")
log('  [B] BY WELL — used for 10978, 7362, 10620. Source:')
log('      count_wells() in scripts/66_external_review.py, which sums')
log('      x.notna() well by well on the raw scale.')
log('The two conventions are not in conflict: they measure different things. They')
log("only need to be declared, because 14352 is NOT 3 x (622+336) x 5.")

# --- definition of the reliability population -----------------------------
log("\n" + "=" * 78)
log('1. DEFINITION OF THE POPULATIONS (verified on the code and on the data)')
log("=" * 78)
log("\nReliability (21_review_P1.py, lines 129-137; identical in")
log('38_affidabilita_mediana.py, lines 116-122):')
log("    sel = sa[(sa.PROJECTID == pid) & (sa.COHORT == 'PD') &")
log("             (sa.CLINICAL_EVENT == 'BL') & (sa.SAA_Status == 'Positive')]")
log('    if ONLY_SPORADIC: sel = sel[sel.PATNO.isin(sporadici)]')
log("No filter on SAA_Type. The confirmation of the hypothesis of the review is")
log("therefore POSITIVE: the reliability population of the 24h is the set")
log('of the sporadic positives to ANY type of assay.')
log("")
log('Primary chain (01_build_dataset.py, line 58):')
log("    prim = sel[(sel.SAA_Status == 'Positive') & (sel.SAA_Type == 'Type1')]")
log("The Type1 filter is the only difference between the two populations.")

r_aff = pos24[pos24.PATNO.isin(SPORADIC)]
r_t1 = t1_24[t1_24.PATNO.isin(SPORADIC)]
kinds = r_aff.SAA_Type.value_counts(dropna=False)
set_t1 = set(r_t1.PATNO)
set_no = set(r_aff.loc[r_aff.SAA_Type != "Type1", "PATNO"])
log("")
log(f'  24h, PD at baseline                       : {len(s24):4d} samples, {s24.PATNO.nunique():4d} '
    f'subjects')
log(f'  24h, positives (any type)                 : {len(pos24):4d} samples, '
    f'{pos24.PATNO.nunique():4d} subjects')
log(f"  24h, sporadic positives  = RELIABILITY    : {len(r_aff):4d} samples, "
    f"{r_aff.PATNO.nunique():4d} subjects")
log(f'  24h, Type1 positives                      : {len(t1_24):4d} samples, '
    f'{t1_24.PATNO.nunique():4d} subjects')
log(f'  24h, sporadic Type1 positives = PRIMARY   : {len(r_t1):4d} samples, '
    f'{r_t1.PATNO.nunique():4d} subjects')
log("")
log("  Breakdown of the 622 reliability samples by SAA_Type:")
for k, v in kinds.items():
    log(f'    {str(k):14s} {v:4d} samples')
log(f"  622 = {int(kinds.get('Type1', 0))} Type1 + {int(kinds.get('Type2', 0))} Type2 + "
    f"{int(kinds.get('Undetermined', 0))} Undetermined")
log("")
log('  From samples to subjects:')
log(f'    subjects with at least one Type1 sample    : {len(set_t1)}')
log(f'    subjects with at least one non-Type1 sample: {len(set_no)}')
log(f'    subjects in both classes                   : {len(set_t1 & set_no)}')
log(f'    subjects ONLY non-Type1                    : {len(set_no - set_t1)}')
log(f'    -> {len(set_t1)} + {len(set_no - set_t1)} = {len(set_t1 | set_no)} subjects')
dup = r_aff.PATNO.value_counts()
log(f'    repeated samples at the same visit: {int((dup > 1).sum())} subjects with 2 samples -> '
    f'{len(r_aff)} samples for {r_aff.PATNO.nunique()} subjects')
log("")
log(f"REPLY TO REMARK 4: 615 versus 601. The reliability population")
log(f'does not apply the SAA_Type=Type1 filter. The 21 sporadic non-Type1 samples')
log(f'belong to {len(set_no)} subjects, {len(set_t1 & set_no)} of whom')
log(f"also have a Type1 sample and are already counted among the 601; the remaining")
log(f'{len(set_no - set_t1)} are new. 601 + {len(set_no - set_t1)} = {len(set_t1 | set_no)} '
    f'subjects. In terms of samples: 601 Type1 + 21')
log(f'non-Type1 = 622. No other criterion differs.')

log("")
log("150h (Project 155): the Type1/Type2 subtyping does not exist (SAA_Type is NaN")
log("in all rows), so reliability and primary chain share the")
log('same definition and the 615-vs-601 difference has no analogue:')
log(f'  150h, PD at baseline        : {len(s150):4d} samples, {s150.PATNO.nunique():4d} subjects')
log(f'  150h, positives             : {len(pos150):4d} samples, {pos150.PATNO.nunique():4d} subjects')
r150 = pos150[pos150.PATNO.isin(SPORADIC)]
log(f'  150h, sporadic positives    : {len(r150):4d} samples, {r150.PATNO.nunique():4d} subjects')
log(f'  SAA_Type in Project 155     : {s150.SAA_Type.isna().sum()} rows out of {len(s150)} with a '
    f'missing value')

# --- accounting per parameter ---------------------------------------------
log("\n" + "=" * 78)
log("2. ACCOUNTING BY PROTOCOL AND BY PARAMETER")
log("=" * 78)
SHOW = ['parameter', 'samples', 'wells_nominals', 'wells_no_missing',
          'wells_missing', 'wells_no_positives',
          'triplicates_complete_scale_log', 'wells_in_triplicates_complete']
for name, proto, sub, params, definition in POP:
    b = block(name)
    log(f'\n--- {name} ---')
    log(f'    {definition}')
    log(f'    {len(sub)} samples, {sub.PATNO.nunique()} subjects')
    log(b[SHOW].to_string(index=False))

# --- resolution of the four discrepancies ---------------------------------
log("\n" + "=" * 78)
log('3. RESOLUTION OF THE FOUR ARITHMETIC DISCREPANCIES')
log("=" * 78)

n24 = val("24h_affidabilita_sporadici", 'samples')
n150 = val("150h_affidabilita_sporadici", 'samples')
tc24 = val("24h_affidabilita_sporadici", 'triplicates_complete_scale_log')
tc150 = val("150h_affidabilita_sporadici", 'triplicates_complete_scale_log')

log(f'\n[1] 4790 versus 4784, 14370 versus 14352')
log(f'    Nominal  : ({n24} + {n150}) x 5 = {(n24 + n150) * 5} parameter-sample pairs, '
    f'{(n24 + n150) * 15} wells.')
log(f'    Actual   : {tc24} + {tc150} = {tc24 + tc150} complete triplets, {(tc24 + tc150) * 3} '
    f'wells.')
log(f'    Difference: {(n24 + n150) * 5 - (tc24 + tc150)} parameter-sample pairs = '
    f'{((n24 + n150) * 5 - (tc24 + tc150)) * 3} wells.')
log('    These are the triplets dropped by icc_from_reps. Item by item:')
for name in ["24h_affidabilita_sporadici", "150h_affidabilita_sporadici"]:
    b = block(name)
    for _, r in b[b.parameter != 'TOTAL_5_PARAMETERS'].iterrows():
        lost = int(r.samples_lost_offthe_triplet)
        if lost:
            log(f'      {name:28s} {r.parameter:9s} {int(r.samples)} -> '
                f'{int(r.triplicates_complete_scale_log)}  (-{lost}: {int(r.wells_missing)} missing '
                f'wells, {int(r.wells_no_positives)} non-positive)')
log(f'    Check against outputs/21_reliability_kinetics.csv:')
icc = pd.read_csv(os.path.join(OUT, "21_reliability_kinetics.csv"))
ok = True
for _, r in icc.iterrows():
    name = ("24h_affidabilita_sporadici" if r.protocol == "24h"
            else "150h_affidabilita_sporadici")
    mine = val(name, 'triplicates_complete_scale_log', r.parameter)
    state = "OK" if mine == int(r.n_subjects) else 'DISCORDANT'
    ok &= mine == int(r.n_subjects)
    log(f'      {r.protocol:5s} {r.parameter:9s} n_subjects={int(r.n_subjects):4d}  '
        f'recomputed={mine:4d}  {state}')
log(f"      sum of n_subjects = {int(icc.n_subjects.sum())}, x3 = {int(icc.n_subjects.sum() * 3)}   "
    f"({('coincides' if ok else 'does NOT coincide')} with the recomputation)")
log('    NOTE: the column is called `n_subjects` but contains SAMPLES with')
log('    a complete triplet. In the 24h the 622 samples come from 615 subjects.')

n_pos24 = val("24h_positivi_tutti_i_tipi", 'samples')
w_pos24 = val("24h_positivi_tutti_i_tipi", 'wells_no_missing')
log(f'\n[3] 10980 versus 10978')
log(f'    Nominal  : {n_pos24} x 3 x 5 = {n_pos24 * 15}')
log(f'    Actual   : {w_pos24} non-missing wells (-{n_pos24 * 15 - w_pos24}).')
b = block("24h_positivi_tutti_i_tipi")
for _, r in b[(b.parameter != 'TOTAL_5_PARAMETERS') &
              (b.wells_missing > 0)].iterrows():
    log(f'      {r.parameter}: {int(r.wells_missing)} missing wells in '
        f'{int(r.samples_having_1_well_missing)} samples (one well each) — PATNO '
        f'{r.PATNO_wells_missing}')

n_pos150 = val("150h_positivi", 'samples')
w_pos150 = val("150h_positivi", 'wells_no_missing')
log(f'\n[2] 7365 versus 7362')
log(f'    Nominal  : {n_pos150} x 3 x 5 = {n_pos150 * 15}')
log(f'    Actual   : {w_pos150} non-missing wells (-{n_pos150 * 15 - w_pos150}).')
b = block("150h_positivi")
for _, r in b[(b.parameter != 'TOTAL_5_PARAMETERS') &
              (b.wells_missing > 0)].iterrows():
    log(f'      {r.parameter}: {int(r.wells_missing)} missing wells in '
        f'{int(r.samples_having_1_well_missing)} samples (one well each) — PATNO '
        f'{r.PATNO_wells_missing}')
log('    WARNING: 7362 counts the wells, 14352 counts the complete triplets.')
_lw150 = val("150h_positivi", 'wells_in_triplicates_complete')
_tc150p = val("150h_positivi", 'triplicates_complete_scale_log')
log(f'    On the same population (150h positives) the LISTWISE convention would give {_tc150p} '
    f'complete triplets = {_lw150} wells, not {w_pos150}.')
log(f'    Discrepancy {w_pos150} - {_lw150} = {w_pos150 - _lw150}: the {n_pos150 * 5 - _tc150p} '
    f'triplets dropped (4 on T50, 1 on SLOPE)')
log(f"    are worth {(n_pos150 * 5 - _tc150p) * 3} nominal wells, of which "
    f"{n_pos150 * 15 - w_pos150} were already missing: {(n_pos150 * 5 - _tc150p) * 3} - "
    f"{n_pos150 * 15 - w_pos150} = {w_pos150 - _lw150}")
log("    wells present in themselves that the count by well keeps and")
log('    the count by triplet discards.')

log(f'\n[4] 615 versus 601: resolved at point 1 (SAA_Type=Type1 filter).')

# --- lost wells, one by one -----------------------------------------------
log("\n" + "=" * 78)
log('4. MECHANISM OF MISSINGNESS')
log("=" * 78)
log("In the source populations (all outcomes) missingness is NOT random:")
log("it is almost entirely concentrated in the non-positive samples, where the")
log("threshold parameters are not defined because the curve does not rise.")
log('For each parameter: distribution of the number of missing wells per')
log("sample, and SAA_Status of the samples that lose the whole parameter.")
for lab, sub, params, proto in [('24h, PD at baseline (862)', s24, K24, "24h"),
                                ('150h, PD at baseline (586)', s150, K150, "150h")]:
    log(f"\n  {lab}")
    for par in params:
        v = sub[cols_of(par, proto)].apply(pd.to_numeric, errors="coerce")
        k = v.isna().sum(axis=1)
        dist = k.value_counts().sort_index().to_dict()
        st3 = sub.loc[k == 3, "SAA_Status"].value_counts().to_dict()
        st1 = sub.loc[k.between(1, 2), "SAA_Status"].value_counts().to_dict()
        log(f'    {par:9s} missing wells per sample {dist}')
        log(f"              parameter entirely missing (3/3) in: {(st3 if st3 else 'no sample')}")
        log(f"              partially missing (1-2/3) in: {(st1 if st1 else 'no sample')}")
log('\n  Consequence: restricting to the POSITIVES missingness almost disappears, and')
log('  no positive sample loses a parameter entirely. In the four')
log('  populations of positives the columns samples_having_2_wells_missing and')
log('  samples_having_3_wells_missing of the table are all zero:')
for name in ["24h_positivi_tutti_i_tipi", "24h_positivi_Type1",
             "24h_affidabilita_sporadici", "150h_positivi",
             "150h_affidabilita_sporadici"]:
    log(f"    {name:28s} with2={val(name, 'samples_having_2_wells_missing')}  "
        f"with3={val(name, 'samples_having_3_wells_missing')}  "
        f"with1={val(name, 'samples_having_1_well_missing')}")

log("\n" + "=" * 78)
log('5. EVERY LOST WELL OF THE POPULATIONS OF POSITIVES, ONE BY ONE')
log("=" * 78)
log("In the populations of positives missingness is ALWAYS at the level of a single")
log('well of a single parameter: the physical well exists and reports the')
log('other four parameters.')

for proto, params, sub in [("24h", K24, pos24), ("150h", K150, pos150)]:
    for par in params:
        cols = cols_of(par, proto)
        v = sub[cols].apply(pd.to_numeric, errors="coerce")
        bad = v.isna().any(axis=1) | (v <= 0).any(axis=1)
        if not bad.any():
            continue
        log(f"\n  {proto} / {par}")
        sub_bl = bl.set_index("PATNO").subgroup
        for i in sub.index[bad]:
            row = saa.loc[i]
            values = ", ".join(f'Rep{j}={row[c]}'
                               for j, c in enumerate(cols, start=1))
            grp = ('sporadic' if row.PATNO in SPORADIC
                   else f"excluded/{sub_bl.get(row.PATNO, 'not in the curated data cut')}")
            log(f"    PATNO {int(row.PATNO)}  "
                f"SAA_Type={(row.SAA_Type if pd.notna(row.SAA_Type) else 'NaN'):<12s} "
                f"RUNDATE={row.RUNDATE}  ({grp})")
            log(f'      {par}: {values}')
            # is the same replicate index usable on the other parameters?
            for j, c in enumerate(cols, start=1):
                x = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                if pd.isna(x) or x <= 0:
                    others = ", ".join(
                        f"{p2}={row[f'{p2}_{proto}_Rep{j}']}"
                        for p2 in params if p2 != par)
                    state = ('absent from the file' if pd.isna(x)
                             else f'present but not positive ({x})')
                    log(f'      -> Rep{j} {state}. The same well Rep{j} reports the other four '
                        f'parameters: {others}')
                    log(f'         so the physical well is not missing, only {par} is missing.')

# --- selection chains -----------------------------------------------------
log("\n" + "=" * 78)
log('6. SELECTION CHAINS DOWN TO THE COMPLETE CASES')
log("=" * 78)
ad = pd.read_csv(os.path.join(OUT, "analysis_dataset.csv"))
P24 = ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax",
       "dat_putamen_mean", "dat_caudate_mean", "age_bl"]
log(f'\n24h: {len(t1_24)} Type1 samples -> {t1_24.PATNO.nunique()} subjects')
log(f'     -> {t1_24.PATNO.isin(bl.PATNO).groupby(t1_24.PATNO).first().sum()} subjects present in '
    f'the curated data cut at BL')
log(f'     -> {len(ad)} sporadic (analysis_dataset.csv)')
log(f'     -> {int(ad.complete_primary.sum())} with complete data on the 7 primary variables')
inc = ad[~ad.complete_primary]
log(f'     the {len(inc)} excluded lack:')
for _, r in inc.iterrows():
    log(f'       PATNO {int(r.PATNO)}: {[c for c in P24 if pd.isna(r[c])]}')
_m601 = val("24h_Type1_sporadici", 'wells_missing')
_np601 = val("24h_Type1_sporadici", 'wells_no_positives')
log(f"     Missing kinetic wells among the {val('24h_Type1_sporadici', 'samples')} sporadic Type1 "
    f"samples: {_m601}; non-positive: {_np601}.")
log(f'     Missing variables among the {len(inc)} excluded: DAT-SCAN only '
    f'({sorted({c for _, r in inc.iterrows() for c in P24 if pd.isna(r[c])})}).')
log("     The loss 601 -> 595 is therefore entirely DAT-SCAN, not kinetic.")

# 150h, rebuilt from the raw data
p = s150.copy()
for f in K150:
    p[f"k_{f}"] = p[cols_of(f, "150h")].apply(pd.to_numeric,
                                              errors="coerce").median(axis=1)
agg150 = (p[p.SAA_Status == "Positive"]
          .groupby("PATNO", as_index=False)
          .agg({f"k_{f}": "mean" for f in K150}))
d = agg150.merge(bl[["PATNO", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
                     "MIA_CAUDATE_BILAT", "age_at_visit", "subgroup"]],
                 on="PATNO", how="left")
d = d[d.subgroup == "Sporadic PD"].reset_index(drop=True)
d["dat_putamen_low"] = d[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
for f in ["Fmax", "AUC", "T50", "SLOPE"]:
    d[f"lg_{f}"] = np.log(d[f"k_{f}"])
P150 = ["lg_Fmax", "lg_T50", "lg_AUC", "lg_SLOPE", "dat_putamen_low",
        "MIA_CAUDATE_BILAT", "age_at_visit"]
d["complete"] = d[P150].notna().all(axis=1)
log(f'\n150h: {len(pos150)} positive samples -> {pos150.PATNO.nunique()} subjects')
log(f'      -> {len(d)} sporadic -> {int(d.complete.sum())} with complete data')
for _, r in d[~d.complete].iterrows():
    log(f'       PATNO {int(r.PATNO)}: {[c for c in P150 if pd.isna(r[c])]}')
_pos = d[(d.k_T50 > 0) & (d.k_SLOPE > 0)]
log("      The median of three replicates is defined even with one well")
log('      missing or non-positive (T50 = 0 or SLOPE < 0 in one replicate):')
if len(d):
    log(f"      the medians of T50 and of SLOPE are positive in {len(_pos)} subjects out of "
        f"{len(d)}; the logarithm is defined and")
    log("      whoever has such a well stays in the cohort, given the other data.")
_miss150 = sorted({c for _, r in d[~d.complete].iterrows()
                   for c in P150 if pd.isna(r[c])})
log(f'      Missing variables among the {int((~d.complete).sum())} excluded: {_miss150}.')
log("      The loss 336 -> 333 is therefore entirely DAT-SCAN, not kinetic.")
d150_file = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
log(f"      Check against outputs/08_dataset_150h.csv: {int(d150_file.complete.sum())} complete "
    f"cases "
    f"({('coincides' if int(d150_file.complete.sum()) == int(d.complete.sum()) else 'does NOT coincide')}).")

# --- replacement numbers for the manuscript -------------------------------
log("\n" + "=" * 78)
log('7. EVERY NUMBER OF THE MANUSCRIPT, WITH ITS TABLE ROW')
log("=" * 78)
CHK = [
    ('862 samples, 844 subjects (24h, PD at baseline)',
     val("24h_PD_basale", 'samples'), val("24h_PD_basale", 'subjects')),
    ('732 samples, 722 subjects (24h positives)',
     val("24h_positivi_tutti_i_tipi", 'samples'),
     val("24h_positivi_tutti_i_tipi", 'subjects')),
    ('708 samples, 706 subjects (24h Type1)',
     val("24h_positivi_Type1", 'samples'),
     val("24h_positivi_Type1", 'subjects')),
    ('622 samples, 615 subjects (24h reliability)',
     val("24h_affidabilita_sporadici", 'samples'),
     val("24h_affidabilita_sporadici", 'subjects')),
    ('601 samples/subjects (primary 24h before the complete cases)',
     val("24h_Type1_sporadici", 'samples'),
     val("24h_Type1_sporadici", 'subjects')),
    ('586 samples, 574 subjects (150h PD at baseline)',
     val("150h_PD_basale", 'samples'), val("150h_PD_basale", 'subjects')),
    ('491 samples, 491 subjects (150h positive)',
     val("150h_positivi", 'samples'), val("150h_positivi", 'subjects')),
    ('336 samples, 336 subjects (150h reliability)',
     val("150h_affidabilita_sporadici", 'samples'),
     val("150h_affidabilita_sporadici", 'subjects')),
]
for label, a, b_ in CHK:
    log(f'  {label:62s} samples={a:4d}  subjects={b_:4d}')

log("")
log(f"  2124 physical wells (24h Type1)           : {val('24h_positivi_Type1', 'samples') * 3}")
log(f"  1473 physical wells (150h positives)      : {val('150h_positivi', 'samples') * 3}")
log(f"  10620 measurements (24h Type1, by well)   : {val('24h_positivi_Type1', 'wells_no_missing')}")
log(f"  10978 measurements (24h positives, by well): "
    f"{val('24h_positivi_tutti_i_tipi', 'wells_no_missing')}")
log(f"  7362 measurements (150h positives, by well): {val('150h_positivi', 'wells_no_missing')}")
log(f'  4784 parameter-sample pairs               : {tc24 + tc150}')
log(f"  14352 reliability measurements            : {(tc24 + tc150) * 3}")
log("")
log('  Correct wording for §2.17 and Table 10 (the numbers remain those of the')
log('  manuscript, the sentence that justifies them changes):')
log(f'    «{n24} positive samples at baseline ({r_aff.PATNO.nunique()} subjects) in the 24 h')
log(f'    protocol and {n150} ({r150.PATNO.nunique()} subjects) in the 150 h one. Of the')
log(f'    {(n24 + n150) * 5} parameter-sample combinations, {tc24 + tc150} have the complete')
log(f'    triplet on the logarithmic scale of analysis and form the basis of the')
log(f'    estimates: {(tc24 + tc150) * 3} parameter-well measurements. The '
    f'{(n24 + n150) * 5 - (tc24 + tc150)} excluded')
_wmiss = ((n24 + n150) * 15
          - val("24h_affidabilita_sporadici", 'wells_no_missing')
          - val("150h_affidabilita_sporadici", 'wells_no_missing'))
_wnp = (val("24h_affidabilita_sporadici", 'wells_no_positives')
        + val("150h_affidabilita_sporadici", 'wells_no_positives'))
log(f'    combinations correspond to {_wmiss} wells missing in the source file and to')
log(f"    {_wnp} wells with a non-positive value, for which the logarithm is not")
log(f'    defined.»')
log(f"    (the {(n24 + n150) * 5 - (tc24 + tc150)} discarded triplets are worth "
    f"{((n24 + n150) * 5 - (tc24 + tc150)) * 3} nominal wells: {_wmiss + _wnp} unusable and "
    f"{((n24 + n150) * 5 - (tc24 + tc150)) * 3 - _wmiss - _wnp} valid in")
log(f"    themselves, lost through the listwise deletion of the triplet)")

log("\n" + "=" * 78)
log('8. CAVEATS')
log("=" * 78)
log('- The column `n_subjects` of 21_reliability_kinetics.csv is a count')
log('  of SAMPLES, not of subjects. The manuscript reads it correctly as')
log('  samples (622, 336), but the name of the column is misleading.')
log('- 622 and 336 are the MAXIMA per parameter, not constant values: 66_review')
log('  _esterna.py derives them with .max(). The TTT of the 24h has 620, the T50 of the')
log('  150h 333 and the SLOPE of the 150h 335. A table writing 622 in every row')
log('  of the n column would be incorrect.')
log('- 14352 and 10978/7362 are NOT comparable with each other: different conventions.')
log('- External cross-check on other files of the project: run_sporadici/')
log('  paper_metodi_e_risultati.md, line 23, writes «103 valori su 7.365». In the')
log('  raw file 103 aberrant wells are counted on the 586 PD samples at')
log(f"  baseline of the 150h ({val('150h_PD_basale', 'wells_no_missing')} measurements), not on the "
    f"491 positives")
log(f"  ({val('150h_positivi', 'wells_no_missing')} measurements). Denominator and numerator belong "
    f"to two")
log("  different populations. That file is not among the protected ones but it has")
log("  not been modified: the report is left to whoever maintains it.")

with open(os.path.join(OUT, "76_denominators.txt"), "w") as fh:
    fh.write("\n".join(rep))
log(f"\nSaved: {os.path.join(OUT, '76_denominators.txt')}")
log(f"Saved: {os.path.join(OUT, '76_denominators.csv')}  ({tab.shape[0]} rows x {tab.shape[1]} "
    f"columns)")
