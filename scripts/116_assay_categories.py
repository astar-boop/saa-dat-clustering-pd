"""
116 - Assay categories (status and type) in PD at baseline, along the selection
chain.

Why it exists. Reviewer 2 asks how many "undetermined" and
"inconclusive" samples there are in the primary cohort, how the two
categories are defined, and what having excluded the negatives, the
inconclusives and the other types implies for generalisability (points
"Excluded participants and assay categories" and "Study population and
generalisability"). The manuscript reports the counts by result (Table S1) but
not by type, and does not define the categories.

What it does. It estimates nothing: it counts. The readings are written here,
before the results.

  1. Samples: status (SAA_Status) by type (SAA_Type) in PD at the baseline
     visit, for each protocol.
  2. Rules of the PPMI Data User Guide (on the cover "updated August 6, 2026";
     status criteria on p. 34, type criteria on p. 35) applied to the three
     wells of EVERY row of the file and
     compared with the labels of the file, row by row. With h = wells with
     Fmax >= 45,000 RFU, m = wells with 3,000 <= Fmax < 45,000, l = wells
     with Fmax < 3,000, at 24 h (Project 237):
       Positive      h = 3; or m = 2 or 3; or h = 2 and m = 1
       Negative      h + m = 0 or 1
       Inconclusive  h = 2 and l = 1; or h = 1, m = 1 and l = 1
       Type 1        positive with h = 3
       Type 2        positive with h <= 1 (two or three wells below 45,000)
       Undetermined  positive with h = 2
     At 150 h (Project 155), with c = wells with Fmax >= 5,000 RFU:
       Positive c = 3; Negative c = 0 or 1; Inconclusive c = 2; no type.
     The rules are transcribed from the guide; "m = 2 or 3" is read literally
     ("exactly 2 or 3 replicates have Fmax >= 3,000 RFU & < 45,000 RFU"), without
     conditions on the third well. The ten possible combinations of (h, m, l)
     all receive a status. If the rules do not reproduce a label the
     script does NOT stop: it counts and lists the rows, because the question is
     precisely whether the published definitions describe the file.
  3. Subjects: each subject receives ONE class, in this order (a subject
     can have more than one baseline sample): has a Positive/Type1 (eligible for
     the primary cohort, as in 01); otherwise has a positive of another type
     (Type2 only, Undetermined only, both); otherwise negatives only,
     inconclusives only, or negatives and inconclusives. At 150 h: has a
     positive; otherwise as above. Each class is followed along the chain of the
     manuscript: linked to the curated data cut at baseline, sporadic, complete
     on DAT and age.
  4. Coding of the curated data cut (column CSFSAA) by class, at 24 h and in
     the subjects linked to the curated data cut only.
  5. Extended population of the pipeline check (06, section D; n in the
     manuscript: 678 = 607 positive, 66 negative, 5 inconclusive). 06 gives
     each subject the status of the FIRST baseline row in the order of the file
     (`"SAA_Status": "first"`), which is not a chronological order. Here
     section D of 06 is rebuilt (same selection, same matrix, same
     k-means) and its group x status table must coincide cell by cell
     with 06_extended_k2_vs_status.csv. Then the label of 06 is compared with
     the class of point 3 and with the status of the sample run last, and
     the ARI is recomputed with the latter label.
  6. Direct answer on the primary cohort and on the replication cohort: how many
     baseline samples their members have, and of what category.
  7. The classes of point 3 redone with the rule that the guide recommends (the
     sample with the most recent run date).

CORRECTIONS AFTER THE INDEPENDENT VERIFICATION (13/11/2026; report in
revisione_R2_rapporti/verifiche_script/riverifica_116.txt). The first draft
(a) described the rule of 06 as "status common to all the rows": it was
wrong, 06 takes the first row of the file; the five subjects coincided by
chance; (b) called "(vuoto)" the missing type, which in the file is the string
"NA" (the "not applicable" category of the guide); (c) did not compare the
"collegati" entry of Table S1 at 150 h, which is 491 in the submitted table and
490 in the data; (d) did not anchor the classes of the non-eligible to anything.
Points 6 and 7, the anchoring of the classes to the curated data cut and the
comparison of the fingerprint of the raw file with the register were added
then. No number of the first draft changed.

Checks (identity, not counts).
  - The Type1, sporadic and complete subjects must be THE SAME as those of the
    primary cohort (analysis_complete_cases.csv), and the complete sporadic
    positives at 150 h the same as those of 08_dataset_150h.csv.
  - The group x status table of the extended population, rebuilt here, must
    coincide cell by cell with outputs/06_extended_k2_vs_status.csv.
  - Each class is derived twice, with two functions written in different
    ways, and in the subjects with a single baseline sample it must agree with
    the CSFSAA coding of the curated data cut (independent source).
  - The fingerprint of the raw file must coincide with the register of the
    verification package (INPUT_GREZZI.csv), if the register exists.
  - The columns of the curated data cut read from the cache must coincide with
    the xlsx file.
  - The values of the submitted Table S1 are compared and every discrepancy is
    reported; they are not used to compute anything. One discrepancy is known
    and declared (150 h, "collegati": 491 in the table, 490 in the data; it is
    already among the corrections of the letter to Reviewer 1). Any other
    discrepancy makes the script exit with an error.
  - The thresholds of the rules are NOT fixed by the data (no well falls
    near a threshold): their correctness rests on the PDF of the guide.

What the script does NOT do: it does not use the rule of the sample run last
(it counts all the baseline samples, like Table S1); it does not look at the
later visits; it does not say why a sample is inconclusive; it does not check
the rules of the guide on the values of the single wells beyond Fmax (the guide
defines them on Fmax only).

Usage:  CLUSTERING_BASE=<radice> ONLY_SPORADIC=1 python3 -B scripts/116_assay_categories.py
Output: outputs/116_assay_categories.txt, 116_samples.csv, 116_rules_guide.csv,
        116_rows_not_reproduced.csv, 116_subjects.csv, 116_population_extended.csv,
        116_cohorts_samples.csv, 116_subjects_last_run.csv
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA                                       # noqa: E402

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
FILE_SAA = "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"
FILE_CUR = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"
CACHE = os.path.join(OUT, "cache", "curated.parquet")
COL_CUR = ["PATNO", "EVENT_ID", "COHORT", "subgroup", "CSFSAA", "age_at_visit",
           "MIA_PUTAMEN_L", "MIA_PUTAMEN_R", "MIA_CAUDATE_BILAT"]
THRESHOLD_HI, THRESHOLD_24, THRESHOLD_150 = 45000, 3000, 5000
PROJECT = {"24h": 237, "150h": 155}
# Table S1 of the submitted supplement: only for the comparison, never for the computation
TABLE_S1 = {"24h": {'samples': 862, 'subjects': 844, 'connected': 703,
                      'sporadic': 601, 'completes': 595},
              "150h": {'samples': 586, 'subjects': 574, 'connected': 491,
                       'sporadic': 336, 'completes': 333}}
DEPARTURES_KNOWN = {("150h", 'connected')}       # declared in the docstring
SEED_06 = 20260804                          # seed of 06_sensitivity.py
WITHOUT_KIND = "NA"                           # in the file: the string "NA"
departures = []

rep = []


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def fingerprint(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def share(k, n):
    return f'{k} of {n} ({100 * k / n:.1f}%)'


def load_curated():
    """Cache of the curated data cut, compared with the xlsx file in the columns read."""
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
    return cur, (f'the cache coincides with the xlsx file in all the {len(cur)} rows and in the '
                 f'{len(COL_CUR)} columns read')


def rule_24(F):
    """Status and type according to the guide, from the three Fmax of a 24 h row."""
    if np.isnan(F).any():
        return 'missing well', 'missing well'
    h = int((F >= THRESHOLD_HI).sum())
    m = int(((F >= THRESHOLD_24) & (F < THRESHOLD_HI)).sum())
    lo = int((F < THRESHOLD_24).sum())
    if h == 3 or m in (2, 3) or (h == 2 and m == 1):
        state = "Positive"
    elif h + m <= 1:
        state = "Negative"
    elif (h == 2 and lo == 1) or (h == 1 and m == 1 and lo == 1):
        state = "Inconclusive"
    else:
        state = 'combination not listed'
    if state != "Positive":
        kind = '(none)'
    else:
        kind = "Type1" if h == 3 else ("Undetermined" if h == 2 else "Type2")
    return state, kind


def rule_150(F):
    if np.isnan(F).any():
        return 'missing well'
    c = int((F >= THRESHOLD_150).sum())
    return "Positive" if c == 3 else ("Inconclusive" if c == 2 else "Negative")


def category_24(s):
    if "Positive/Type1" in s:
        return '1. Positive Type 1 (eligible)'
    pos = {x for x in s if x.startswith("Positive/")}
    if pos == {"Positive/Type2"}:
        return '2. Positive, Type 2 only'
    if pos == {"Positive/Undetermined"}:
        return '3. Positive, Undetermined only'
    if pos:
        assert pos == {"Positive/Type2", "Positive/Undetermined"}, f'unexpected types: {pos}'
        return '4. Positive, Type 2 and Undetermined'
    return category_no_positive(s)


def category_150(s):
    if any(x.startswith("Positive") for x in s):
        return '1. Positive (eligible)'
    return category_no_positive(s)


def category_no_positive(s):
    st = {x.split("/")[0] for x in s}
    if st == {"Negative"}:
        return '5. Negative only'
    if st == {"Inconclusive"}:
        return '6. Inconclusive only'
    assert st == {"Negative", "Inconclusive"}, f'unexpected statuses: {st}'
    return '7. Negative and inconclusive'


def category_bis(comb, prot):
    """The same class, derived from the string of the combinations with a
    logic written in another way: it serves to trigger an assertion if one
    of the two derivations is altered."""
    pieces = comb.split(" + ")
    pos = [x for x in pieces if x.startswith("Positive")]
    neg, inc = "Negative" in pieces, "Inconclusive" in pieces
    if prot == "150h":
        if pos:
            return '1. Positive (eligible)'
    else:
        kinds = sorted(x.split("/")[1] for x in pos)
        if "Type1" in kinds:
            return '1. Positive Type 1 (eligible)'
        if kinds == ["Type2"]:
            return '2. Positive, Type 2 only'
        if kinds == ["Undetermined"]:
            return '3. Positive, Undetermined only'
        if kinds == ["Type2", "Undetermined"]:
            return '4. Positive, Type 2 and Undetermined'
        assert not kinds, f'unexpected types: {kinds}'
    if neg and inc:
        return '7. Negative and inconclusive'
    return '5. Negative only' if neg else '6. Inconclusive only'


def quota2(k, n):
    return f'{k} of {n}' + ("" if k == n else f" ({100 * k / n:.2f}%)")


def compare_s1(prot, item, v):
    expected = TABLE_S1[prot][item]
    if v == expected:
        log(f'  {item}: {v}; submitted Table S1: {expected}')
        return
    known = (prot, item) in DEPARTURES_KNOWN
    log(f'  {item}: {v}; submitted Table S1: {expected}  <-- DIFFERENT'
        + (' (known discrepancy, declared in the docstring)' if known else ""))
    if not known:
        departures.append(f'{prot}, {item}: {v} against {expected}')


def latest_run(d):
    """The sample with the most recent run date for each subject."""
    assert not d.duplicated(["PATNO", "RUNDATE"]).any(), \
        'two baseline samples of the same subject with the same run date'
    return d.sort_values(["PATNO", "RUNDATE"]).drop_duplicates("PATNO", keep="last") \
            .set_index("PATNO")


def main():
    title('116 - ASSAY CATEGORIES IN PD AT BASELINE')
    saa_p = os.path.join(DATA, FILE_SAA)
    saa = pd.read_csv(saa_p, low_memory=False)
    saa["PATNO"] = saa["PATNO"].astype("int64")
    saa['row'] = np.arange(len(saa))                 # order of the file
    saa["RUNDATE"] = pd.to_datetime(saa["RUNDATE"])
    assert saa.RUNDATE.notna().all(), 'run dates missing in the assay file'
    cur, outcome_cur = load_curated()
    cur["PATNO"] = cur["PATNO"].astype("int64")
    assert not cur.duplicated(["PATNO", "EVENT_ID"]).any(), 'repeated visits in the curated data cut'
    bl = cur[cur.EVENT_ID == "BL"].set_index("PATNO")
    h = fingerprint(saa_p)
    reg = os.path.join(OUT, "verification_package", "INPUT_GREZZI.csv")
    if os.path.exists(reg):
        r = pd.read_csv(reg).set_index('name').sha256
        if FILE_SAA in r.index:
            assert h == r[FILE_SAA], "the fingerprint of the assay file does not match the register"
            outcome_reg = 'equal to the register of the verification package'
        else:
            outcome_reg = 'identity with the register NOT verified (file not listed)'
    else:
        outcome_reg = 'identity with the register NOT verified (register not found)'
    log(f'Assay file: {FILE_SAA}, {len(saa)} rows, sha256 {h[:16]}..., {outcome_reg}')
    log(f'Curated: {outcome_cur}')
    assert set(saa.SAA_Status.unique()) == {"Positive", "Negative", "Inconclusive"}, \
        'unexpected values of SAA_Status'
    assert set(saa.SAA_Type.dropna().unique()) == {"Type1", "Type2", "Undetermined"}, \
        'unexpected values of SAA_Type'
    assert set(saa.PROJECTID.unique()) == set(PROJECT.values()), 'unexpected projects'
    assert "PD" in set(saa.COHORT.unique()) and "BL" in set(saa.CLINICAL_EVENT.unique())

    c24 = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
    c150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
    c150 = c150[c150.complete]
    p24, p150 = set(c24.PATNO.astype("int64")), set(c150.PATNO.astype("int64"))
    spor_bl = set(bl.index[bl.subgroup == "Sporadic PD"])
    root_spor = not (p24 - spor_bl) and not (p150 - spor_bl)
    assert ONLY == root_spor, ('ONLY_SPORADIC and working directory do not agree: the working '
                                 'directory is '
                                 + ('sporadic only' if root_spor else 'complete'))
    log(f"Working directory: {('sporadic only' if root_spor else 'full cohort')}; 24 h cohort n = "
        f"{len(p24)}, 150 h cohort n = {len(p150)}, in both {len(p24 & p150)}")
    log(f"In the file SAA_Type is the string 'NA' ('not applicable' in the guide) in "
        f"{int(saa.SAA_Type.isna().sum())} rows: all the 150 h rows and the negative 24 h rows.")

    # ------------------------------------------------------------------ 1 ----
    title('1. SAMPLES: STATUS BY TYPE, PD AT THE BASELINE VISIT')
    rows_c, sel = [], {}
    for prot, pid in PROJECT.items():
        d = saa[(saa.COHORT == "PD") & (saa.CLINICAL_EVENT == "BL")
                & (saa.PROJECTID == pid)].copy()
        assert len(d) and (d.COHORT == "PD").all() and (d.CLINICAL_EVENT == "BL").all() \
            and (d.PROJECTID == pid).all(), f'{prot}: selection of PD at baseline altered'
        d['kind'] = d.SAA_Type.fillna(WITHOUT_KIND)
        sel[prot] = d
        ct = pd.crosstab(d.SAA_Status, d.kind, margins=True, margins_name='Total')
        log(f'\n{prot}: {len(d)} samples of {d.PATNO.nunique()} subjects')
        log(ct.to_string())
        for (st, tp), g in d.groupby(["SAA_Status", 'kind']):
            rows_c.append(dict(protocol=prot, state=st, kind=tp, samples=len(g),
                                subjects_having_atleast_an_sample=g.PATNO.nunique()))
        compare_s1(prot, 'samples', len(d))
        compare_s1(prot, 'subjects', d.PATNO.nunique())
    pd.DataFrame(rows_c).to_csv(os.path.join(OUT, "116_samples.csv"), index=False)
    d24 = sel["24h"]
    inc = d24[d24.SAA_Status == "Inconclusive"]
    log(f"\nAt 24 h the Inconclusive rows with SAA_Type = Undetermined are "
        f"{int((inc.kind == 'Undetermined').sum())} out of {len(inc)}; the rows with SAA_Type = "
        f"Undetermined are {int((d24.kind == 'Undetermined').sum())}, of which positive "
        f"{int(((d24.kind == 'Undetermined') & (d24.SAA_Status == 'Positive')).sum())}.")

    # ------------------------------------------------------------------ 2 ----
    title('2. DO THE RULES OF THE GUIDE REPRODUCE THE LABELS OF THE FILE?')
    rows_r, no_reprod = [], []
    for prot, pid in PROJECT.items():
        every = saa[saa.PROJECTID == pid].copy()
        suff = "24h" if prot == "24h" else "150h"
        F = every[[f"Fmax_{suff}_Rep{i}" for i in (1, 2, 3)]].to_numpy(float)
        if prot == "24h":
            pred = [rule_24(f) for f in F]
            every['state_rule'] = [p[0] for p in pred]
            every['kind_rule'] = [p[1] for p in pred]
            every['kind_file'] = np.where(every.SAA_Status == "Positive",
                                          every.SAA_Type.fillna(WITHOUT_KIND), '(none)')
        else:
            every['state_rule'] = [rule_150(f) for f in F]
            every['kind_rule'] = every['kind_file'] = '(none)'
        every['ok_state'] = every.state_rule == every.SAA_Status
        every['ok_kind'] = every.kind_rule == every.kind_file
        for name, sub in (('all cohorts, all visits', every),
                          ('PD at baseline', every[(every.COHORT == "PD")
                                                   & (every.CLINICAL_EVENT == "BL")])):
            n, npos = len(sub), int((sub.SAA_Status == "Positive").sum())
            row = (f'\n{prot}, {name}: {n} rows. Status reproduced in '
                    f'{quota2(int(sub.ok_state.sum()), n)}')
            if prot == "24h":
                row += (f"; type reproduced in the positives in "
                         f"{quota2(int((sub.ok_kind & (sub.SAA_Status == 'Positive')).sum()), npos)}")
            log(row + ".")
            ct = pd.crosstab(sub.SAA_Status, sub.state_rule)
            log('  status in the file (rows) by status according to the guide (columns):')
            log("  " + ct.to_string().replace("\n", "\n  "))
            rows_r.append(dict(protocol=prot, dataset=name, rows=n,
                                state_reproduced=int(sub.ok_state.sum()), positives=npos,
                                kind_reproduced_in_positives=(int(
                                    (sub.ok_kind & (sub.SAA_Status == "Positive")).sum())
                                    if prot == "24h" else np.nan)))
        k = every[~(every.ok_state & every.ok_kind)]
        cols = ["PATNO", "COHORT", "CLINICAL_EVENT", "PROJECTID", "RUNDATE", "SAA_Status",
                "SAA_Type", 'state_rule', 'kind_rule'] + \
               [f"Fmax_{suff}_Rep{i}" for i in (1, 2, 3)]
        no_reprod.append(k[cols].rename(columns={f"Fmax_{suff}_Rep{i}": f"Fmax_Rep{i}"
                                                for i in (1, 2, 3)}))
        if prot == "24h":
            inc_t = every[every.SAA_Status == "Inconclusive"]
            log(f'  Type reported in the file for the Inconclusive rows (the guide does not define '
                f'it): {inc_t.SAA_Type.fillna(WITHOUT_KIND).value_counts().to_dict()}')
        else:
            second = np.sort(F, axis=1)[:, 1]          # the second well by Fmax
            neg2 = (every.SAA_Status == "Negative") & (every.state_rule == "Inconclusive")
            inc2 = every.SAA_Status == "Inconclusive"
            log(f"  The {int(neg2.sum())} Negative rows with two wells >= {THRESHOLD_150} RFU "
                f"({int((neg2 & (every.COHORT == 'PD') & (every.CLINICAL_EVENT == 'BL')).sum())} "
                f"among PD at baseline): median Fmax of the second well "
                f"{np.median(second[neg2.to_numpy()]):.0f} RFU, against "
                f"{np.median(second[inc2.to_numpy()]):.0f} in the {int(inc2.sum())} Inconclusive "
                f"rows. The guide does not state the criterion that distinguishes them.")
    pd.DataFrame(rows_r).to_csv(os.path.join(OUT, "116_rules_guide.csv"), index=False)
    nr = pd.concat(no_reprod, ignore_index=True)
    nr.to_csv(os.path.join(OUT, "116_rows_not_reproduced.csv"), index=False)
    log(f"\nRows with status or type not reproduced by the rules: {len(nr)} (list in "
        f"116_rows_not_reproduced.csv); among PD at baseline: "
        f"{int(((nr.COHORT == 'PD') & (nr.CLINICAL_EVENT == 'BL')).sum())}.")
    log('The thresholds are not fixed by this comparison: no well falls close to a threshold, so '
        'slightly different thresholds would give the same count.')

    # ------------------------------------------------------------------ 3 ----
    title('3. SUBJECTS: ONE CLASS PER SUBJECT, ALONG THE CHAIN')
    CODE = {"Positive/Type1": 1, "Positive/Type2": 3, "Positive/Undetermined": 2,
              "Negative": 0, "Inconclusive": 2}
    rows_s, classes = [], {}
    for prot in PROJECT:
        d = sel[prot].copy()
        d["cat"] = np.where(d.SAA_Status == "Positive",
                            "Positive/" + d.kind, d.SAA_Status)
        sel[prot] = d
        g = d.groupby("PATNO").cat.agg(frozenset)
        cl = g.map(category_24 if prot == "24h" else category_150)
        m = pd.DataFrame({'category': cl, 'samples_baseline': g.map(
            lambda s: " + ".join(sorted(s)))})
        bis = m.samples_baseline.map(lambda c: category_bis(c, prot))
        assert (bis == m.category).all(), f'{prot}: the two derivations of the class do not agree'
        assert len(m) == d.PATNO.nunique()
        m['n_samples'] = d.groupby("PATNO").size().reindex(m.index).to_numpy()
        m['linked'] = m.index.isin(bl.index)
        m['sporadic'] = m.index.isin(spor_bl)
        b = bl.reindex(m.index)
        m['complete'] = b[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R", "MIA_CAUDATE_BILAT",
                           "age_at_visit"]].notna().all(axis=1).to_numpy()
        m['sporadic_complete'] = m.sporadic & m.complete
        m["CSFSAA_curated"] = b.CSFSAA.to_numpy()
        ult = latest_run(d)
        m['latest_run'] = ult.cat.reindex(m.index).to_numpy()
        classes[prot] = m
        if prot == "24h":
            # anchoring to an independent source: the coding of the curated data cut
            one = m[m.linked & (m.n_samples == 1)]
            expected = one.samples_baseline.map(CODE)
            assert (expected == one.CSFSAA_curated).all(), \
                '24h: in the subjects with a single sample the class does not agree with CSFSAA'
            more = m[m.linked & (m.n_samples > 1)]
            ok = int((more.latest_run.map(CODE) == more.CSFSAA_curated).sum())
            log(f'\n24h, anchoring to the curated data cut: in the {len(one)} linked subjects with a '
                f'single baseline sample the category matches CSFSAA in all. In the {len(more)} with '
                f'more than one sample CSFSAA matches the sample run last in {ok}.')
        t = m.groupby('category')[['linked', 'sporadic', 'sporadic_complete']].sum()
        t.insert(0, 'subjects', m.groupby('category').size())
        t.loc['Total'] = t.sum()
        assert int(t.loc['Total', 'subjects']) == d.PATNO.nunique()
        log(f'\n{prot}: subjects by class')
        log(t.to_string())
        for cls, r in t.iterrows():
            rows_s.append(dict(protocol=prot, category=cls, **{k: int(v) for k, v in r.items()}))
        log('  combinations of baseline samples per subject:')
        for comb, n in m.samples_baseline.value_counts().items():
            log(f"    {n:4d}  {comb}")
        elig = set(m.index[m.category.str.startswith("1.") & m.sporadic_complete])
        expectation = p24 if prot == "24h" else p150
        assert elig == expectation, (f'{prot}: the eligible, sporadic and complete subjects are not those '
                              f'of the cohort ({len(elig)} against {len(expectation)}; only here '
                              f'{sorted(elig - expectation)[:5]}, only in the cohort '
                              f'{sorted(expectation - elig)[:5]})')
        log(f'  Identity: the eligible, sporadic and complete subjects are the same '
            f'{len(expectation)} subjects of the cohort analysed.')
        elig_r = t.loc[[i for i in t.index if i.startswith("1.")][0]]
        log('  Eligible subjects along the chain:')
        for item, col in (('connected', 'linked'), ('sporadic', 'sporadic'),
                          ('completes', 'sporadic_complete')):
            compare_s1(prot, item, int(elig_r[col]))
        nonc = sorted(m.index[m.category.str.startswith("1.") & ~m.linked])
        log(f'  Eligible subjects missing from the curated data cut at baseline: {len(nonc)}'
            + (f' (of which missing from the curated data cut at every visit: '
               f'{sum((x not in set(cur.PATNO) for x in nonc))})' if nonc else ""))
        tot = int(t.loc['Total', 'sporadic_complete'])
        log(f'  Sporadic subjects with complete data and a baseline assay: {tot}; the cohort '
            f'analysed keeps {share(len(expectation), tot)} of them.')
        for cls in t.index[1:-1]:
            log(f"    {cls}: {share(int(t.loc[cls, 'sporadic_complete']), tot)}")
        nc = int(t.loc['Total', 'linked'])
        log(f"  Genetic criterion: not sporadic {share(nc - int(t.loc['Total', 'sporadic']), nc)} of "
            f"the linked subjects.")
    pd.DataFrame(rows_s).to_csv(os.path.join(OUT, "116_subjects.csv"), index=False)

    # ------------------------------------------------------------------ 4 ----
    title('4. CODING OF THE CURATED DATA CUT (CSFSAA) BY CLASS: 24 h, LINKED ONLY')
    m24 = classes["24h"]
    k = m24[m24.linked]
    log(pd.crosstab(k.category, k.CSFSAA_curated.fillna(-1).astype(int)).to_string())
    log('Codes (dictionary of the curated data cut, not verified here): 0 negative, 1 positive Type '
        '1, 2 inconclusive, 3 positive Type 2.')
    und = k[k.category.str.startswith("3.")]
    only_und = und[und.samples_baseline == "Positive/Undetermined"]
    log(f'Positive subjects with neither Type 1 nor Type 2 (class 3): {len(und)}, all coded '
        f'{sorted(set(und.CSFSAA_curated.astype(int)))} in the curated data cut. Of these, '
        f'{len(only_und)} have only Positive/Undetermined samples; {len(und) - len(only_und)} '
        f'has/have also an Inconclusive sample.')

    # ------------------------------------------------------------------ 5 ----
    title('5. EXTENDED POPULATION OF THE PIPELINE CHECK (06, SECTION D)')
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    s6 = sel["24h"].sort_values('row').copy()
    for f in ("Fmax", "AUC", "SLOPEMax"):
        s6[f"saa_{f}"] = s6[[f"{f}_24h_Rep{i}" for i in (1, 2, 3)]].apply(
            pd.to_numeric, errors="coerce").mean(axis=1)
    a6 = s6.groupby("PATNO", as_index=False).agg(
        {"saa_Fmax": "mean", "saa_AUC": "mean", "saa_SLOPEMax": "mean", "SAA_Status": "first"})
    ext = a6.merge(bl.reset_index()[["PATNO", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
                                     "MIA_CAUDATE_BILAT", "age_at_visit", "subgroup"]],
                   on="PATNO", how="inner")
    if ONLY:
        ext = ext[ext.subgroup == "Sporadic PD"].reset_index(drop=True)
    ext["dat_putamen_low"] = ext[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
    FE = ["saa_Fmax", "saa_AUC", "saa_SLOPEMax", "dat_putamen_low", "MIA_CAUDATE_BILAT",
          "age_at_visit"]
    ext = ext.dropna(subset=FE).reset_index(drop=True)
    Xe = ext[FE].copy()
    for c in ("saa_Fmax", "saa_AUC", "saa_SLOPEMax"):
        Xe[c] = np.log(Xe[c].clip(lower=1e-9))
    Xe = StandardScaler().fit_transform(Xe.to_numpy(float))
    lab2 = KMeans(2, n_init=50, random_state=SEED_06).fit_predict(Xe)
    ct6 = pd.crosstab(lab2, ext.SAA_Status.to_numpy())
    e = m24[m24.sporadic_complete].copy()
    assert set(ext.PATNO) == set(e.index), \
        'the reconstructed extended population does not match the complete sporadic subjects of point 3'
    f06 = os.path.join(OUT, "06_extended_k2_vs_status.csv")
    if os.path.exists(f06):
        t06 = pd.read_csv(f06, index_col=0)
        cols = sorted(t06.columns)
        a, b = ct6.reindex(columns=cols, fill_value=0).to_numpy(), t06[cols].to_numpy()
        assert np.array_equal(a, b) or np.array_equal(a[::-1], b), \
            f'the reconstructed group x status table does not match 06:\n{ct6}\n{t06}'
        log(f"n = {len(ext)}. Section D of 06 is reconstructed: the group x status table matches "
            f"06_extended_k2_vs_status.csv cell by cell.")
    else:
        log(f"n = {len(ext)}. 06_extended_k2_vs_status.csv not found: identity with 06 NOT verified.")
    log(ct6.to_string())
    ari06 = adjusted_rand_score(lab2, (ext.SAA_Status == "Positive").astype(int))
    log(f'ARI between the k = 2 partition and positive / not positive, label of 06 (first row of the '
        f'file): {ari06:.3f}')
    ext = ext.set_index("PATNO")
    e['label_06'] = ext.SAA_Status.reindex(e.index).to_numpy()
    e['state_latest_run'] = e.latest_run.map(lambda c: c.split("/")[0])
    e['group_k2'] = pd.Series(lab2, index=ext.index).reindex(e.index).to_numpy()
    ct = pd.crosstab(e.label_06, e.category, margins=True, margins_name='Total')
    log('\nLabel of 06 (rows) by class (columns):')
    log(ct.to_string())
    inc5 = e[e.label_06 == "Inconclusive"]
    log(f"\nThe {len(inc5)} subjects that the manuscript calls 'inconclusive', by baseline samples:")
    for comb, n in inc5.samples_baseline.value_counts().items():
        log(f"  {n}  {comb}")
    log(f"  of these, members of the primary cohort: {int(inc5.index.isin(list(p24)).sum())}; with "
        f"Inconclusive samples only: {int((inc5.samples_baseline == 'Inconclusive').sum())}.")
    pos = e[e.label_06 == "Positive"]
    t1 = pos.category.str.startswith("1.")
    log(f"The {len(pos)} 'positive': with at least one Type 1 sample {int(t1.sum())} (of which with "
        f"also a sample of another category "
        f"{int((pos[t1].samples_baseline != 'Positive/Type1').sum())}), without Type 1 "
        f"{int((~t1).sum())} ({pos[~t1].category.value_counts().to_dict()}).")
    log(f"Members of the primary cohort in the extended population: "
        f"{int(e.index.isin(list(p24)).sum())} of {len(p24)}; among the 'positive' of 06: "
        f"{int(pos.index.isin(list(p24)).sum())}.")
    log('\nWith the status of the sample run last (the rule that the guide recommends):')
    log("  " + str(e.state_latest_run.value_counts().to_dict())
        + f'; subjects whose label changes: {int((e.state_latest_run != e.label_06).sum())}')
    ari_u = adjusted_rand_score(e.group_k2, (e.state_latest_run == "Positive").astype(int))
    log(f'  ARI between the same partition and positive / not positive with this label: {ari_u:.3f}')
    log("  " + pd.crosstab(e.group_k2, e.state_latest_run).to_string().replace("\n", "\n  "))
    e.reset_index()[["PATNO", 'category', 'samples_baseline', 'label_06',
                     'state_latest_run', 'group_k2']].assign(
        cohort_primary=e.index.isin(list(p24))).to_csv(
        os.path.join(OUT, "116_population_extended.csv"), index=False)

    # ------------------------------------------------------------------ 6 ----
    title('6. THE BASELINE SAMPLES OF THE MEMBERS OF THE TWO COHORTS ANALYSED')
    rows_k = []
    for prot, members in (("24h", p24), ("150h", p150)):
        d = sel[prot][sel[prot].PATNO.isin(list(members))]
        assert d.PATNO.nunique() == len(members)
        per = d.groupby("PATNO").size()
        log(f'\n{prot}: {len(members)} members, {len(d)} baseline samples; with more than one sample '
            f'{int((per > 1).sum())}.')
        for cat, n in d.cat.value_counts().items():
            ns = d[d.cat == cat].PATNO.nunique()
            log(f'    {cat}: {n} samples of {ns} subjects')
            rows_k.append(dict(protocol=prot, members=len(members), category=cat,
                                samples=int(n), subjects=int(ns)))
        m = classes[prot].loc[sorted(members)]
        others = m[m.samples_baseline.str.contains(r"\+")]
        for comb, n in others.samples_baseline.value_counts().items():
            log(f'    {n} members with: {comb}')
    pd.DataFrame(rows_k).to_csv(os.path.join(OUT, "116_cohorts_samples.csv"), index=False)
    log('\nAt 24 h 01_build_dataset.py keeps only the Positive/Type1 samples before averaging: the '
        'samples of another category of the members do not enter the kinetic values.')

    # ------------------------------------------------------------------ 7 ----
    title('7. THE CLASSES WITH THE RULE OF THE SAMPLE RUN LAST')
    rows_u = []
    for prot in PROJECT:
        m = classes[prot]
        t = m.groupby('latest_run')[['linked', 'sporadic', 'sporadic_complete']].sum()
        t.insert(0, 'subjects', m.groupby('latest_run').size())
        t.loc['Total'] = t.sum()
        log(f'\n{prot}: subjects by category of the baseline sample run last')
        log(t.to_string())
        for cls, r in t.iterrows():
            rows_u.append(dict(protocol=prot, category_latest_run=cls,
                                **{k: int(v) for k, v in r.items()}))
        elig = m[m.category.str.startswith("1.")]
        good = "Positive/Type1" if prot == "24h" else f'Positive/{WITHOUT_KIND}'
        log(f'  Eligible subjects whose sample run last is not {good}: '
            f'{int((elig.latest_run != good).sum())} of {len(elig)}; among the members of the cohort '
            f'analysed: {int((elig[elig.sporadic_complete].latest_run != good).sum())}.')
    pd.DataFrame(rows_u).to_csv(os.path.join(OUT, "116_subjects_last_run.csv"), index=False)

    title('WHAT THIS SCRIPT DOES NOT DO')
    log("- Sections 1-6 count all the baseline samples; the rule of the sample run last is only in "
        "section 7 and in the last part of section 5.")
    log("- It does not look at later visits nor at the retests with the other protocol.")
    log('- The rules of the guide are transcribed by hand from the PDF and the thresholds are not '
        'fixed by the data: the transcription must be checked against the PDF, not against this '
        'file.')
    log('- Section 4 concerns only the subjects linked to the curated data cut and only the 24 h assay.')
    if departures:
        log('\nUNDECLARED DISCREPANCIES FROM TABLE S1: ' + "; ".join(departures))
    with open(os.path.join(OUT, "116_assay_categories.txt"), "w") as fh:
        fh.write("\n".join(rep) + "\n")
    if departures:
        sys.exit(1)


if __name__ == "__main__":
    main()
