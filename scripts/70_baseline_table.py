"""
70 - Table of baseline characteristics (STROBE) for the two cohorts, and
     comparison between included and excluded for incomplete data.

The review asks for a standard baseline table for each cohort and for the
included against the excluded, with the explicit distinction between mean and median and
with the share of missing data for each variable.

Output: outputs/70_baseline.csv, outputs/70_baseline.txt
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO").set_index("PATNO")

prim = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
everyone = pd.read_csv(os.path.join(OUT, "analysis_dataset.csv"))       # 601
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150c = d150[d150.complete]

excluded = everyone[~everyone.PATNO.isin(set(prim.PATNO))]

# follow-up: last observed year per subject
fu = cur.groupby("PATNO").YEAR.max()


def describe(patno, label):
    idx = [p for p in patno if p in bl.index]
    b = bl.loc[idx]
    f = fu.reindex(idx)
    out = {"cohort": label, "n": len(idx)}

    def cont(col, name, dec=1):
        if col not in b.columns:
            out[f'{name} — mean (SD)'] = "n.a."
            return
        v = pd.to_numeric(b[col], errors="coerce")
        out[f'{name} — mean (SD)'] = f"{v.mean():.{dec}f} ({v.std():.{dec}f})"
        out[f'{name} — median [IQR]'] = (
            f"{v.median():.{dec}f} [{v.quantile(.25):.{dec}f}–{v.quantile(.75):.{dec}f}]")
        out[f'{name} — missing'] = f"{int(v.isna().sum())} ({100*v.isna().mean():.1f}%)"

    cont("age_at_visit", "Age at baseline (years)")
    if "SEX" in b.columns:
        m = pd.to_numeric(b.SEX, errors="coerce")
        out["Male — n (%)"] = f"{int((m == 1).sum())} ({100*(m == 1).mean():.1f}%)"
        out["Male — missing"] = f"{int(m.isna().sum())}"
    cont("EDUCYRS", "Education (years)")
    cont("duration_yrs", "Disease duration (years)", 2)
    cont("updrs3_score", "MDS-UPDRS III")
    cont("updrs_totscore", "MDS-UPDRS total")
    cont("hy", "Hoehn and Yahr", 2)
    cont("moca", "MoCA")
    cont("upsit", "UPSIT")
    if "PDTRTMNT" in b.columns:
        t = pd.to_numeric(b.PDTRTMNT, errors="coerce")
        out["On dopaminergic treatment at baseline — n (%)"] = \
            f"{int((t == 1).sum())} ({100*(t == 1).mean():.1f}%)"
    out["Follow-up (years) — median [IQR]"] = (
        f"{f.median():.1f} [{f.quantile(.25):.1f}–{f.quantile(.75):.1f}]")
    out["Follow-up beyond 10 years — n"] = int((f >= 10).sum())
    return out


rows = [describe(prim.PATNO, "Primary cohort, 24 h (n = 595)"),
         describe(excluded.PATNO, "Excluded for incomplete data, 24 h (n = 6)"),
         describe(d150c.PATNO, "Replication cohort, 150 h (n = 333)")]

tab = pd.DataFrame(rows).set_index("cohort").T
tab.to_csv(os.path.join(OUT, "70_baseline.csv"))
log("=" * 78)
log('BASELINE CHARACTERISTICS OF THE TWO COHORTS')
log("=" * 78)
log(tab.to_string())

# share of subjects in phase 1 / phase 2, if available
for col in ("PPMI_PHASE", "phase", "COHORT_DEFINITION", "ENROLL_STATUS"):
    if col in bl.columns:
        log(f"\n{col}:")
        for label, patno in [("24h", prim.PATNO), ("150h", d150c.PATNO)]:
            idx = [p for p in patno if p in bl.index]
            log(f'  {label}: ' +
                bl.loc[idx, col].value_counts(dropna=False).to_dict().__str__())
        break

with open(os.path.join(OUT, "70_baseline.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/70_baseline.csv')
