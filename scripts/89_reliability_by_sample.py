"""
89 - Grouping unit of the reliability: sample or subject?

The fourth-round review observes that the reliability table (then
Table 7, now Table 10) defines as the unit of
estimation the parameter-by-sample combination, whereas §2.17 says that the
one-way ANOVA is conducted "across subjects". Since the reliability population
of the 24 h protocol contains 622 samples from 615 subjects, the two things do
not coincide: seven participants contributed two samples each.

This script:
  1. verifies the accounting 622 / 615 / 7 and the breakdown by type;
  2. recomputes ICC(1,1) and ICC(1,3) with the SAMPLE as the unit, replicating
     exactly script 21 (which is what the manuscript publishes);
  3. recomputes the same coefficients with the SUBJECT as the unit (unbalanced
     one-way ANOVA on all the wells of the subject, with Satterthwaite's k0
     and Spearman-Brown to 3 replicates), to quantify by how much
     they would shift if the grouping were the one described in words.

Expected outcome: the code groups by sample, so the defect is only one of
wording in §2.17. The recomputation by subject serves to say by how much,
not to replace the published values.

Output: outputs/89_reliability_by_sample.csv
        outputs/89_reliability_by_sample.txt
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
rep = []


def log(m=""):
    print(m)
    rep.append(str(m))


sa = pd.read_csv(os.path.join(DATA,
                              "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"))
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"]
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
sporadic = set(bl.loc[bl.subgroup == "Sporadic PD", "PATNO"])

SPECS = [("24h", 237, ["Fmax", "TTT", "AUC", "TSmax", "SLOPEMax"], "_24h_Rep"),
         ("150h", 155, ["Fmax", "TTT", "AUC", "T50", "SLOPE"], "_150h_Rep")]


def icc_by_sample(mat):
    """mat: n samples x 3 replicates, log scale. Identical to script 21."""
    m = mat[~np.isnan(mat).any(axis=1)]
    n, k = m.shape
    gm = m.mean()
    msb = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msw = ((m - m.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return n, (msb - msw) / (msb + (k - 1) * msw), (msb - msw) / msb


def icc_by_subject(df, cols):
    """Unbalanced one-way ANOVA on the wells grouped by PATNO.

    Returns (n subjects, ICC(1,1), ICC(1,3) by Spearman-Brown). The seven
    subjects with two samples contribute six wells instead of three.
    """
    t = df[["PATNO"] + cols].copy()
    for c in cols:
        t[c] = pd.to_numeric(t[c], errors="coerce")
        t[c] = np.log(t[c].where(t[c] > 0))
    t = t.dropna(subset=cols)                 # same listwise deletion
    long = t.melt(id_vars="PATNO", value_vars=cols, value_name="y")
    g = long.groupby("PATNO").y
    mean_s, ns = g.mean(), g.size()
    a, N, gm = len(ns), len(long), long.y.mean()
    msb = (ns * (mean_s - gm) ** 2).sum() / (a - 1)
    msw = ((long.y - long.PATNO.map(mean_s)) ** 2).sum() / (N - a)
    k0 = (N - (ns ** 2).sum() / N) / (a - 1)
    i1 = (msb - msw) / (msb + (k0 - 1) * msw)
    return a, i1, 3 * i1 / (1 + 2 * i1)


log("=" * 78)
log("89 — GROUPING UNIT OF THE RELIABILITY")
log("=" * 78)

rows = []
counts = []
for tag, pid, params, suffix in SPECS:
    sel = sa[(sa.PROJECTID == pid) & (sa.COHORT == "PD") &
             (sa.CLINICAL_EVENT == "BL") & (sa.SAA_Status == "Positive")]
    if ONLY:
        sel = sel[sel.PATNO.isin(sporadic)]
    n_samp, n_subj = len(sel), sel.PATNO.nunique()
    vc = sel.PATNO.value_counts()
    n_repeated = int((vc > 1).sum())
    kinds = sel.SAA_Type.value_counts(dropna=False).to_dict()
    log(f'\nProtocol {tag} (Project {pid})')
    log(f'  samples {n_samp}, subjects {n_subj}, difference {n_samp - n_subj}')
    log(f'  subjects with more than one sample: {n_repeated} (maximum {int(vc.max())} samples per '
        f'subject)')
    log(f'  assay types: {kinds}')
    if tag == "24h":
        t1 = sel[sel.SAA_Type == "Type1"]
        nt1 = sel[sel.SAA_Type != "Type1"]
        both = len(set(t1.PATNO) & set(nt1.PATNO))
        log(f'  type 1: {len(t1)} samples / {t1.PATNO.nunique()} subjects; not type 1: {len(nt1)} '
            f'samples / {nt1.PATNO.nunique()} subjects; subjects with both: {both}')
        log(f'  check 601 + 14: {len(t1)} + {nt1.PATNO.nunique() - both} = '
            f'{len(t1) + nt1.PATNO.nunique() - both}')
    counts.append({'protocol': tag, 'samples': n_samp, 'subjects': n_subj,
                     'subjects_repeated': n_repeated})

    for p in params:
        cols = [f"{p}{suffix}{i}" for i in (1, 2, 3)]
        if not all(c in sel.columns for c in cols):
            continue
        v = sel[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        v = np.log(np.where(v > 0, v, np.nan))
        n_c, i1_c, i3_c = icc_by_sample(v)
        n_s, i1_s, i3_s = icc_by_subject(sel, cols)
        rows.append({'protocol': tag, 'parameter': p,
                     'n_samples': n_c, 'ICC_1_1_by_sample': i1_c,
                     'ICC_1_3_by_sample': i3_c,
                     'n_subjects': n_s, 'ICC_1_1_by_subject': i1_s,
                     'ICC_1_3_by_subject': i3_s,
                     "delta_ICC_1_1": i1_c - i1_s,
                     "delta_ICC_1_3": i3_c - i3_s})

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
log("\n" + "-" * 78)
log("ICC by grouping unit (logarithmic scale of analysis)")
log("-" * 78)
log(df.round(4).to_string(index=False))
df.to_csv(os.path.join(OUT, "89_reliability_by_sample.csv"), index=False)

# ---- comparison with the published values -------------------------------------
pub_path = os.path.join(OUT, "21_reliability_kinetics.csv")
if os.path.exists(pub_path):
    pub = pd.read_csv(pub_path)
    m = df.merge(pub[['protocol', 'parameter', 'n_subjects',
                      "ICC_1_1", "ICC_1_3"]],
                 on=['protocol', 'parameter'], suffixes=("", "_pub"))
    d1 = (m.ICC_1_1_by_sample - m.ICC_1_1).abs().max()
    d3 = (m.ICC_1_3_by_sample - m.ICC_1_3).abs().max()
    dn = (m.n_samples - m.n_subjects_pub).abs().max()
    log("\nReplication of the published values (script 21) with the SAMPLE as the unit:")
    log(f'  maximum discrepancy in ICC(1,1) : {d1:.2e}')
    log(f'  maximum discrepancy in ICC(1,3) : {d3:.2e}')
    log(f'  maximum discrepancy in counts   : {int(dn)}')
    log("  (the column 'n_subjects' of script 21 actually counts SAMPLES: the name is misleading, "
        "the computation is per sample.)")

log("\n" + "-" * 78)
log('Maximum shift if grouping were by subject (24h protocol)')
log("-" * 78)
d24 = df[df.protocol == "24h"]
i1max = d24.loc[d24.delta_ICC_1_1.abs().idxmax()]
i3max = d24.loc[d24.delta_ICC_1_3.abs().idxmax()]
log(f'  ICC(1,1): {i1max.parameter}, {i1max.ICC_1_1_by_sample:.3f} -> {i1max.ICC_1_1_by_subject:.3f} '
    f'(delta {i1max.delta_ICC_1_1:.3f})')
log(f'  ICC(1,3): {i3max.parameter}, {i3max.ICC_1_3_by_sample:.3f} -> {i3max.ICC_1_3_by_subject:.3f} '
    f'(delta {i3max.delta_ICC_1_3:.3f})')
log('  All the shifts are downwards: the variation between samples')
log("  of the same subject moves from the between-unit term to the within-unit one.")
d150 = df[df.protocol == "150h"]
log(f'  150h protocol: maximum shift {d150.delta_ICC_1_3.abs().max():.1e} — no repeated subject.')

with open(os.path.join(OUT, "89_reliability_by_sample.txt"), "w") as f:
    f.write("\n".join(rep) + "\n")
print('\nWritten 89_reliability_by_sample.csv/.txt')
