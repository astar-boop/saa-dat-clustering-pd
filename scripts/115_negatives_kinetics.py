"""
115 - Kinetic parameters of the assay in the SAA-negative participants of the
positive control.

Question (Reviewer 1, point 8): why groups were not sought in the
SAA-negatives only. This script describes the 24 h kinetic values of the
negatives next to those of the positives, in the SAME population as section D of
script 06 (PD, Project 237, BL visit, 24 h assay; sporadic if
ONLY_SPORADIC=1; complete cases on Fmax, AUC, SLOPEMax, lower putamen,
bilateral caudate, age). It does not estimate models and does not seek groups.

Selection: it is that of 06, reproduced here. For each participant the values
are the mean of the wells and, where a participant has more than one baseline
row, the mean of the rows (as in 06). The status is the one common to all the
baseline rows of the participant; if the rows do not agree the status is
"righe discordi".

Identity guard: the counts by status must coincide with
outputs/06_extended_k2_vs_status.csv (Positive and Negative equal; Inconclusive
of 06 = Inconclusive + righe discordi of here). If the file of 06 is missing,
the comparison is declared NOT verified.

Usage:  CLUSTERING_BASE=<radice> ONLY_SPORADIC=1 python3 -B scripts/115_negatives_kinetics.py
Output: outputs/115_negatives_kinetics.csv, outputs/115_negatives_kinetics.txt
"""

import os
import math
import numpy as np
import pandas as pd
from _paths import DATA

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
FILE_SAA = "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"
FILE_CUR = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"
PARAM = ["Fmax", "AUC", "SLOPEMax", "TTT"]
# Units as declared in the PPMI Data User Guide (6/8/2026), section 8.5.
UNITS = {"Fmax": "RFU", "AUC": 'RFU x h (PPMI Data User Guide)',
         "SLOPEMax": 'RFU per hour (PPMI Data User Guide)', "TTT": 'hours'}

rep = []


def log(s=""):
    print(s)
    rep.append(s)


def down(x, d):
    """Rounds down to d significant digits (for the minima)."""
    if x == 0 or not np.isfinite(x):
        return x
    e = d - 1 - math.floor(math.log10(abs(x)))
    return math.floor(x * 10 ** e) / 10 ** e


def over(x, d):
    """Rounds up to d significant digits (for the maxima)."""
    if x == 0 or not np.isfinite(x):
        return x
    e = d - 1 - math.floor(math.log10(abs(x)))
    return math.ceil(x * 10 ** e) / 10 ** e


# ---------------------------------------------------------------- data
saa = pd.read_csv(os.path.join(DATA, FILE_SAA), low_memory=False)
saa["PATNO"] = saa["PATNO"].astype("int64")
sel = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237)
          & (saa.CLINICAL_EVENT == "BL")].copy()
assert len(sel) > 0, 'no PD / Project 237 / BL row in the assay file'
for f in PARAM:
    cols = [f"{f}_24h_Rep{i}" for i in (1, 2, 3)]
    v = sel[cols].apply(pd.to_numeric, errors="coerce")
    sel[f"v_{f}"] = v.mean(axis=1)
    sel[f'wells_{f}'] = v.notna().sum(axis=1)


def state(x):
    u = sorted(set(x.astype(str)))
    return u[0] if len(u) == 1 else 'discordant rows'


sel = sel.sort_values(["PATNO", "RUNDATE"] if "RUNDATE" in sel.columns else ["PATNO"])
g = sel.groupby("PATNO", as_index=False).agg(
    Fmax=("v_Fmax", "mean"), AUC=("v_AUC", "mean"), SLOPEMax=("v_SLOPEMax", "mean"),
    TTT=("v_TTT", "mean"), wells_TTT=('wells_TTT', "sum"),
    rows=("PATNO", "size"), state=("SAA_Status", state))

cur = pd.read_excel(os.path.join(BASE, FILE_CUR), sheet_name="20260511",
                    usecols=["PATNO", "EVENT_ID", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
                             "MIA_CAUDATE_BILAT", "age_at_visit", "subgroup"])
cur["PATNO"] = cur["PATNO"].astype("int64")
bl = cur[cur.EVENT_ID == "BL"]
assert not bl.PATNO.duplicated().any(), "more than one BL row per participant in the curated data cut"
e = g.merge(bl, on="PATNO", how="inner", validate="one_to_one")
if ONLY:
    e = e[e.subgroup == "Sporadic PD"]
e = e.assign(put=e[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1))
e = e.dropna(subset=["Fmax", "AUC", "SLOPEMax", "put", "MIA_CAUDATE_BILAT",
                     "age_at_visit"]).reset_index(drop=True)
POP = ("PD " + ('sporadic' if ONLY else '(sporadic and carriers)')
       + ', 24 h assay (Project 237), BL visit, complete cases as in 06 section D')

n_state = e.state.value_counts().to_dict()
n_tot = len(e)
log('115 - KINETIC PARAMETERS AT 24 H IN THE SAA-NEGATIVES OF THE POSITIVE CONTROL')
log("=" * 72)
log(f'Population: {POP}')
log(f'n = {n_tot}; by status: ' + ", ".join(f"{k} {v}" for k, v in sorted(n_state.items())))

# ---------------------------------------------------------------- identity with 06
f06 = os.path.join(OUT, "06_extended_k2_vs_status.csv")
if os.path.exists(f06):
    t = pd.read_csv(f06).drop(columns=[c for c in ["row_0"] if c in pd.read_csv(f06).columns])
    tot06 = t.sum(axis=0).to_dict()
    here_inc = n_state.get("Inconclusive", 0) + n_state.get('discordant rows', 0)
    assert int(tot06.get("Positive", -1)) == n_state.get("Positive", 0), \
        f'positives differ from 06: {tot06} against {n_state}'
    assert int(tot06.get("Negative", -1)) == n_state.get("Negative", 0), \
        f'negatives differ from 06: {tot06} against {n_state}'
    assert int(tot06.get("Inconclusive", -1)) == here_inc, \
        f'inconclusives differ from 06: {tot06} against {n_state}'
    assert int(sum(tot06.values())) == n_tot, 'total n differs from 06'
    log(f"Counts by status and total n equal to those of 06 (06 does not save the list of "
        f"participants: the comparison is on the counts, not on the identity). In 06 the "
        f"inconclusives are {int(tot06['Inconclusive'])}: here {n_state.get('Inconclusive', 0)} with "
        f"status Inconclusive and {n_state.get('discordant rows', 0)} with baseline rows of "
        f"different status.")
else:
    log('Comparison of the counts with 06: NOT verified (outputs/06_extended_k2_vs_status.csv missing).')

# ---------------------------------------------------------------- description
neg = e[e.state == "Negative"]
pos = e[e.state == "Positive"]
assert len(neg) > 0 and len(pos) > 0, 'one of the two groups is missing'
rows = []
log("")
log('Values per participant (mean of the wells and of the baseline rows)')
for name, d in (("Negative", neg), ("Positive", pos)):
    log(f'\n  {name}, n = {len(d)}')
    for f in PARAM:
        x = d[f].dropna()
        r = dict(population=POP, visit="BL", state_SAA=name, n_group=len(d),
                 parameter=f, units=UNITS[f], n_having_value=len(x),
                 n_without_value=len(d) - len(x))
        if len(x):
            r.update(minimum=x.min(), q25=x.quantile(.25), median=x.median(),
                     q75=x.quantile(.75), maximum=x.max())
            log(f'    {f:9s} with a value {len(x)} of {len(d)}; minimum {down(x.min(), 3):g}, median '
                f'{x.median():.4g} (quartiles {x.quantile(0.25):.4g}-{x.quantile(0.75):.4g}), '
                f'maximum {over(x.max(), 3):g}  [{UNITS[f]}]')
        else:
            log(f'    {f:9s} with a value 0 of {len(d)}')
        rows.append(r)

log("")
log('Negatives compared with the positives')
n_ttt_no = int((neg.wells_TTT == 0).sum())
log(f'  TTT: no well with a value in {n_ttt_no} of the {len(neg)} negatives; at least one well with '
    f'a value in {len(neg) - n_ttt_no}. Among the positives: no well in '
    f'{int((pos.wells_TTT == 0).sum())} of {len(pos)}.')
rows.append(dict(population=POP, visit="BL", state_SAA="Negative", n_group=len(neg),
                  parameter="TTT", units='participants',
                  measure='without any well with a value', value=n_ttt_no))
con = neg[neg.wells_TTT > 0]
by_n = con.wells_TTT.value_counts().sort_index()
log(f'  Negatives with at least one TTT value, by number of wells with a value (out of '
    f'{(int(con.rows.max()) * 3 if len(con) else 3)} per participant): '
    + ", ".join(f'{int(k)} wells in {int(v)}' for k, v in by_n.items()) + ".")
for k, v in by_n.items():
    rows.append(dict(population=POP, visit="BL", state_SAA="Negative", n_group=len(neg),
                      parameter="TTT", units='participants',
                      measure=f'with value in {int(k)} wells', value=int(v)))
n_everyone = int(((neg.Fmax < pos.Fmax.min()) & (neg.AUC < pos.AUC.min())
               & (neg.SLOPEMax < pos.SLOPEMax.min()) & (neg.wells_TTT == 0)).sum())
log(f'  Negatives without any TTT value and below the minimum of the positives on Fmax, AUC and '
    f'SLOPEMax together: {n_everyone} of {len(neg)}.')
rows.append(dict(population=POP, visit="BL", state_SAA="Negative", n_group=len(neg),
                  parameter="Fmax, AUC, SLOPEMax, TTT", units='participants',
                  measure='without TTT and below the minimum of the positives on the three parameters',
                  value=n_everyone))
log(f'  Fmax: negatives at or above the minimum of the positives '
    f'{int((neg.Fmax >= pos.Fmax.min()).sum())} of {len(neg)}; positives below the maximum of the '
    f'negatives {int((pos.Fmax < neg.Fmax.max()).sum())} of {len(pos)}.')
for f in ["Fmax", "AUC", "SLOPEMax"]:
    below_min = int((neg[f] < pos[f].min()).sum())
    below_p1 = int((neg[f] < pos[f].quantile(.01)).sum())
    ratio = pos[f].median() / neg[f].median()
    log(f'  {f}: negatives below the minimum of the positives {below_min} of {len(neg)}; below the '
        f'1st percentile of the positives {below_p1} of {len(neg)}; median of the positives / median '
        f'of the negatives = {ratio:.1f}')
    for meas, val in (('negatives below the minimum of the positives', below_min),
                     ('negatives below the 1st percentile of the positives', below_p1),
                     ('median of the positives / median of the negatives', ratio)):
        rows.append(dict(population=POP, visit="BL", state_SAA="Negative",
                          n_group=len(neg), parameter=f, units=UNITS[f],
                          measure=meas, value=val))

log("")
log('WHAT THIS SCRIPT DOES NOT DO')
log('- It does not look for groups among the negatives and does not estimate any model.')
log('- It does not use the rule of the sample run last: it reproduces the selection of 06')
log("  (mean of the baseline rows), because it describes the population of that analysis.")
log("- It does not say why the values of the negatives are low: it reports the values.")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "115_negatives_kinetics.csv"), index=False)
with open(os.path.join(OUT, "115_negatives_kinetics.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
print('\nSaved: 115_negatives_kinetics.csv, 115_negatives_kinetics.txt')
