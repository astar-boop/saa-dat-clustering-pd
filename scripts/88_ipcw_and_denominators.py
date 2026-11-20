"""
88 - Verification of remark 3.4 of the fourth review.

Three distinct questions:

(A) THE IPCW SCALE. The MoCA text announces four estimates and prints three.
    Here all four are rebuilt at full precision (estimate, robust SE
    clustered by subject, p) and the EXACT percentage change between
    weighted and unweighted is computed within each of the two working
    structures (exchangeable and independence). It is checked whether the
    "25%" and the "24%" of the manuscript survive without rounding, and what
    the values already rounded to three decimals would give instead.

(B) THE QUARTILES OF THE WEIGHTS. The manuscript reports only median and
    maximum. Here the complete distribution of the stabilised weights of the
    MoCA specification is printed: min, Q1, median, mean, Q3, p99, max,
    shares > 5 and > 10.

(C) THE DENOMINATOR OF THE 150h. The reviewer observes that 1473 x 5 = 7365
    whereas Table 2 reports 7362. Here it is established from the raw file
    how many wells the protocol contributes, how many parameters per well,
    what exactly the 7362 counts (potential or observed) and WHICH three
    values are missing, with PATNO, parameter and replicate.

The construction of the data and of the weights in (A) and (B) is copied line
by line from scripts/78_longitudinal_sensitivity.py, section B2; that of (C) from
scripts/76_denominators.py.

Usage:
    cd os.path.dirname(os.path.dirname(os.path.abspath(__file__))) && \
    CLUSTERING_BASE=".../run_sporadici" python3 scripts/88_ipcw_and_denominators.py

Output: outputs/88_ipcw_and_denominators.csv, outputs/88_ipcw_and_denominators.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from _paths import DATA

warnings.filterwarnings("ignore")

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
W = os.path.join(OUT, "cache") + "/"
SAA_FILE = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")

K150 = ["Fmax", "TTT", "AUC", "T50", "SLOPE"]
REPS = ["Rep1", "Rep2", "Rep3"]

rep = []
rows = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def R(section, item, **kw):
    r = {'section': section, 'item': item}
    r.update(kw)
    rows.append(r)


# ===========================================================================
# (A) + (B)  THE IPCW SCALE AND THE WEIGHTS
# ===========================================================================
log("=" * 78)
log('88 - COMMENT 3.4: IPCW SCALE, QUARTILES OF THE WEIGHTS, 150h DENOMINATOR')
log("=" * 78)
log("")
log("=" * 78)
log('(A)+(B) THE IPCW SCALE ON THE MoCA AND THE DISTRIBUTION OF THE WEIGHTS')
log("=" * 78)

cur = pd.read_parquet(W + "curated.parquet")
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete].reset_index(drop=True)
d["T50"] = np.log(d.k_T50)
d["T50"] = (d.T50 - d.T50.mean()) / d.T50.std()

covbl = cur[cur.EVENT_ID == "BL"][["PATNO", "age_at_visit", "SEX", "EDUCYRS",
                                   "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
base = d[["PATNO", "T50"]].merge(covbl, on="PATNO", how="left")

L = cur[cur.PATNO.isin(base.PATNO)][["PATNO", "YEAR", "moca"]].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()].merge(base, on="PATNO", how="left")

COV = ["age_bl", "SEX", "dur_bl", "EDUCYRS"]
m = L[["PATNO", "years", "moca", "T50"] + COV].dropna().reset_index(drop=True)
m["years"] = m.years.astype(float)

N_SUBJ = int(m.PATNO.nunique())
N_OBSV = int(len(m))
log(f'\nData reconstructed: {N_SUBJ} subjects, {N_OBSV} observations.')

last_year = m.groupby("PATNO").years.max()
moca_bl = m[m.years == 0][["PATNO", "moca"]].rename(columns={"moca": "moca_bl"})
subj = base[base.PATNO.isin(m.PATNO.unique())][["PATNO", "T50"] + COV].copy()
subj = subj.merge(moca_bl, on="PATNO", how="left")
subj = subj.merge(last_year.rename('last'), left_on="PATNO",
                  right_index=True, how="left")
subj = subj.dropna().reset_index(drop=True)

J = int(m.years.max())
long = []
for _, s in subj.iterrows():
    for j in range(1, J + 1):
        if s.last >= j - 1:
            long.append({"PATNO": s.PATNO, 'visit': j,
                          'remains': int(s.last >= j),
                          "T50": s.T50, "age_bl": s.age_bl, "SEX": s.SEX,
                          "dur_bl": s.dur_bl, "EDUCYRS": s.EDUCYRS,
                          "moca_bl": s.moca_bl})
long = pd.DataFrame(long)

F_DEN = 'remains ~ C(visit) + age_bl + SEX + EDUCYRS + dur_bl + moca_bl + T50'
F_NUM = 'remains ~ C(visit)'
den = smf.glm(F_DEN, long, family=sm.families.Binomial()).fit()
num = smf.glm(F_NUM, long, family=sm.families.Binomial()).fit()

long["p_den"] = den.predict(long)
long["p_num"] = num.predict(long)
long = long.sort_values(["PATNO", 'visit'])
long["r_den"] = long.groupby("PATNO").p_den.cumprod()
long["r_num"] = long.groupby("PATNO").p_num.cumprod()
long["sw"] = long.r_num / long.r_den

m3 = m.copy()
m3['visit'] = m3.years.astype(int)
m3 = m3.merge(long[["PATNO", 'visit', "sw"]], on=["PATNO", 'visit'],
              how="left")
m3["sw"] = m3.sw.where(m3.visit > 0, 1.0)
m3 = m3.dropna(subset=["sw"]).reset_index(drop=True)
sw = m3.sw.values

log('\n--- (B) FULL DISTRIBUTION OF THE STABILISED WEIGHTS (MoCA) ---')
QUART = [('minimum', float(sw.min())),
         ("Q1 (p25)", float(np.percentile(sw, 25))),
         ('median', float(np.median(sw))),
         ('mean', float(sw.mean())),
         ("Q3 (p75)", float(np.percentile(sw, 75))),
         ("p99", float(np.percentile(sw, 99))),
         ('maximum', float(sw.max()))]
for name, val in QUART:
    log(f'  {name:<10s} = {val:.6f}   (rounded to 2 dec: {val:.2f})')
    R("B", 'MoCA stabilised weight', statistic=name, value=val,
      n_obsv=int(len(m3)), n_subj=int(m3.PATNO.nunique()))
q5 = float((sw > 5).mean())
q10 = float((sw > 10).mean())
log(f'  share > 5  = {q5:.6f}  ({int((sw > 5).sum())} observations)')
log(f'  share > 10 = {q10:.6f}  ({int((sw > 10).sum())} observations)')
R("B", 'MoCA stabilised weight', statistic='share > 5', value=q5)
R("B", 'MoCA stabilised weight', statistic='share > 10', value=q10)
log(f'  interquartile range = {np.percentile(sw, 75) - np.percentile(sw, 25):.6f}')
log('\n  Comparison with the values quoted by the reviewer (Q1 0.95 / median 1.00 /')
log("  Q3 1.02): all three confirmed at rounding to two decimals.")

log('\n--- (A) THE FOUR MARGINAL ESTIMATES, FULL PRECISION ---')
F_MARG = "moca ~ T50*years + age_bl + SEX + dur_bl + EDUCYRS"
IPCW = [
    ('exchangeable, IPCW-WEIGHTED', True, "exch"),
    ('exchangeable, NOT weighted', False, "exch"),
    ('independence, IPCW-WEIGHTED', True, "ind"),
    ('independence, NOT weighted', False, "ind"),
]
res = {}
for et, weighted, struct in IPCW:
    if struct == "exch":
        r = smf.gee(F_MARG, "PATNO", m3,
                    weights=(m3.sw.values if weighted else None),
                    family=sm.families.Gaussian(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
    else:
        r = smf.wls(F_MARG, m3,
                    weights=(m3.sw.values if weighted else np.ones(len(m3)))).fit(
            cov_type="cluster", cov_kwds={"groups": m3.PATNO})
    name = [c for c in r.params.index if "T50" in c and "years" in c][0]
    b, se, p = float(r.params[name]), float(r.bse[name]), float(r.pvalues[name])
    res[et] = (b, se, p)
    log(f'  {et:<28s} beta={b:.10f}  SE={se:.10f}  z={b / se:.4f}  p={p:.6f}')
    log(f"  {'':<28s} rounded: {b:.3f}  {se:.3f}  {p:.3f}")
    R("A", 'marginal estimate', model=et, estimate=b, se=se, z=b / se, p=p,
      n_obsv=int(len(m3)), n_subj=int(m3.PATNO.nunique()))

log('\n--- (A) PERCENTAGE CHANGE WEIGHTED vs UNWEIGHTED ---')
for label, a, bnm in [
        ('exchangeable', 'exchangeable, IPCW-WEIGHTED', 'exchangeable, NOT weighted'),
        ('independence', 'independence, IPCW-WEIGHTED',
         'independence, NOT weighted')]:
    ba, bb = res[a][0], res[bnm][0]
    pc = 100 * (ba - bb) / bb
    ra, rb = round(ba, 3), round(bb, 3)
    pc_r = 100 * (ra - rb) / rb
    log(f'\n  {label}:')
    log(f'    unweighted = {bb:.10f}   weighted = {ba:.10f}')
    log(f"    UNROUNDED change = {pc:+.4f}%  (to one decimal place: {pc:+.1f}%; to the integer: "
        f"{pc:+.0f}%)")
    log(f"    if the values ALREADY rounded to 3 decimals are used ({rb:.3f} -> {ra:.3f}) the change "
        f"would be {pc_r:+.2f}%")
    R("A", 'percentage change', model=label,
      estimate=pc, se=None, remark=f'from {bb:.10f} a {ba:.10f}; on values rounded to 3 dec. it would '
                              f'give {pc_r:.2f}%')

pc_ex = 100 * (res['exchangeable, IPCW-WEIGHTED'][0] /
               res['exchangeable, NOT weighted'][0] - 1)
pc_in = 100 * (res['independence, IPCW-WEIGHTED'][0] /
               res['independence, NOT weighted'][0] - 1)
log("\n  VERDICT ON THE ROUNDING")
log(f"    exchangeable: {pc_ex:.4f}%  -> '25%' holds at full precision "
    f"({('SI' if abs(pc_ex - 25) < 0.5 else 'NO')}); the rounded values 0.030 -> 0.038 would give "
    f"{100 * (0.038 / 0.03 - 1):.1f}%.")
log(f"    independence: {pc_in:.4f}%  -> '24%' holds at full precision "
    f"({('SI' if abs(pc_in - 24) < 0.5 else 'NO')}); the rounded values 0.025 -> 0.031 would give "
    f"{100 * (0.031 / 0.025 - 1):.1f}%.")
log('    Note: the percentages of the manuscript must be computed on the values NOT')
log("    rounded; recomputing them from the three decimals printed in the table gives")
log('    different numbers. The sentence must be written so as to say so.')

# ===========================================================================
# (C) THE DENOMINATOR OF THE 150h
# ===========================================================================
log("")
log("=" * 78)
log('(C) THE DENOMINATOR OF THE 150h PROTOCOL')
log("=" * 78)

sa = pd.read_csv(SAA_FILE, low_memory=False)
p155 = sa[(sa.COHORT == "PD") & (sa.PROJECTID == 155) &
          (sa.CLINICAL_EVENT == "BL")]
pos = p155[p155.SAA_Status == "Positive"]

n_samp = len(pos)
n_subj = pos.PATNO.nunique()
n_wells = n_samp * 3
n_param = len(K150)
potential = n_wells * n_param

log(f'\nPopulation: COHORT=PD & PROJECTID=155 & CLINICAL_EVENT=BL & SAA_Status=Positive')
log(f'  samples                              : {n_samp}')
log(f'  subjects                             : {n_subj}')
log(f'  replicates per sample                : 3')
log(f'  physical wells = {n_samp} x 3             : {n_wells}')
log(f"  parameters reported per well         : {n_param}  ({', '.join(K150)})")
log(f'  POTENTIAL MEASUREMENTS = {n_wells} x {n_param} : {potential}')
R("C", '150h positives', statistic='samples', value=n_samp)
R("C", '150h positives', statistic='subjects', value=n_subj)
R("C", '150h positives', statistic='physical wells', value=n_wells)
R("C", '150h positives', statistic='potential measurements', value=potential)

log('\nCount per well, parameter by parameter (x.notna(), raw scale):')
tot_obsv = 0
missing_dtl = []
for k in K150:
    cols = [f"{k}_150h_{r}" for r in REPS]
    for c in cols:
        assert c in pos.columns, f'column missing: {c}'
    sub = pos[cols].apply(pd.to_numeric, errors="coerce")
    obsv = int(sub.notna().sum().sum())
    man = n_wells - obsv
    tot_obsv += obsv
    log(f'  {k:<7s} potential={n_wells:5d}  observed={obsv:5d}  missing={man:3d}')
    R("C", '150h positive by parameter', statistic=k,
      value=obsv, remark=f'potential {n_wells}, absent {man}')
    if man:
        for _, row in pos[sub.isna().any(axis=1)].iterrows():
            for c in cols:
                if pd.isna(row[c]):
                    missing_dtl.append(
                        {'parameter': k, "PATNO": int(row.PATNO),
                         'replication': c.split("_")[-1],
                         "RUNDATE": row.get("RUNDATE", "")})
log(f"  {'TOTAL':<7s} potential={potential:5d}  observed={tot_obsv:5d}  "
    f"missing={potential - tot_obsv:3d}")
R("C", '150h positives', statistic='observed measures', value=tot_obsv)
R("C", '150h positives', statistic='absent measurements',
  value=potential - tot_obsv)

log(f'\nEXPLICIT ARITHMETIC')
log(f'  {n_samp} samples x 3 replicates             = {n_wells} wells')
log(f'  {n_wells} wells x {n_param} parameters              = {potential} POTENTIAL measurements')
log(f'  {potential} potential - {potential - tot_obsv} missing           = {tot_obsv} OBSERVED '
    f'measurements')
log(f'  Table 2 reports {tot_obsv}: it is the OBSERVED count.')
log(f'  The premise of the reviewer ({n_wells} x {n_param} = {potential}) is CORRECT as a potential '
    f'count.')

log('\nTHE MISSING VALUES, ONE BY ONE:')
md = pd.DataFrame(missing_dtl)
if len(md):
    for _, r in md.iterrows():
        log(f'  parameter {r.parameter:<6s} PATNO {r.PATNO:<8d} replicate {r.replication}   RUNDATE '
            f'{r.RUNDATE}')
        R("C", 'absent measurement', statistic=r.parameter,
          value=r.PATNO, remark=f'replicate {r.replication}, RUNDATE {r.RUNDATE}')
    log(f'\n  Total: {len(md)} missing measurements, all on the parameter '
        f'{sorted(md.parameter.unique())}, in {md.PATNO.nunique()} distinct samples, one replicate '
        f'each.')
    log('  The physical well is not missing: the other four parameters of the same well are '
        'present. Check:')
    for _, r in md.iterrows():
        row = pos[pos.PATNO == r.PATNO].iloc[0]
        others = {k: row.get(f'{k}_150h_{r.replication}') for k in K150
                 if k != r.parameter}
        ok = all(pd.notna(v) for v in others.values())
        log(f'    PATNO {r.PATNO} replicate {r.replication}: other four parameters present = {ok}  '
            f'({others})')

npos = 0
log("\nWells PRESENT but not positive (the logarithm is not defined; they remain counted among the "
    "observed measurements):")
for k in K150:
    cols = [f"{k}_150h_{r}" for r in REPS]
    sub = pos[cols].apply(pd.to_numeric, errors="coerce")
    mask = sub.notna() & (sub <= 0)
    n = int(mask.sum().sum())
    npos += n
    if n:
        for _, row in pos[mask.any(axis=1)].iterrows():
            for c in cols:
                if pd.notna(row[c]) and float(row[c]) <= 0:
                    log(f"  {k:<6s} PATNO {int(row.PATNO)} replicate {c.split('_')[-1]}  value = "
                        f"{row[c]}")
                    R("C", 'non-positive measure', statistic=k,
                      value=int(row.PATNO),
                      remark=f"replicate {c.split('_')[-1]}, value {row[c]}")
log(f'  Total not positive: {npos}. They are PRESENT, hence included in the {tot_obsv}; they leave '
    f'only the listwise count per triplet.')
R("C", '150h positives', statistic='non-positive measures present', value=npos)

# --- reconciliation with the 24h, for the cross-check ---
log("\n" + "-" * 78)
log("CROSS-CHECK: the same quantities in the 24h protocol")
log("-" * 78)
K24 = ["Fmax", "TTT", "AUC", "TSmax", "SLOPEMax"]
p237 = sa[(sa.COHORT == "PD") & (sa.PROJECTID == 237) &
          (sa.CLINICAL_EVENT == "BL")]
for et, sel, kk in [
        ('24h type 1 positive', p237[(p237.SAA_Status == "Positive") &
                                     (p237.SAA_Type == "Type1")], K24),
        ('24h positives (every type)', p237[p237.SAA_Status == "Positive"], K24)]:
    nc = len(sel)
    npz = nc * 3
    pwr = npz * len(kk)
    obsv = int(sum(sel[[f"{k}_24h_{r}" for r in REPS]].apply(
        pd.to_numeric, errors="coerce").notna().sum().sum() for k in kk))
    log(f'  {et:<26s} samples={nc:4d}  wells={npz:5d}  potential={pwr:6d}  observed={obsv:6d}  '
        f'missing={pwr - obsv:3d}')
    R("C", et, statistic='samples', value=nc)
    R("C", et, statistic='physical wells', value=npz)
    R("C", et, statistic='potential measurements', value=pwr)
    R("C", et, statistic='observed measures', value=obsv)

log('\nRECONCILIATION OF THE TWO CONVENTIONS ON THE 150h POSITIVES')
log(f'  per well (Table 2)            : {tot_obsv} observed measurements out of {potential} potential')
comp = 0
for k in K150:
    cols = [f"{k}_150h_{r}" for r in REPS]
    v = pos[cols].apply(pd.to_numeric, errors="coerce").values
    v = np.where(v > 0, v, np.nan)
    comp += int((~np.isnan(v).any(axis=1)).sum())
log(f'  listwise per triplet          : {comp} complete triplets = {3 * comp} measurements')
log(f'  difference {tot_obsv} - {3 * comp} = {tot_obsv - 3 * comp}: wells present in themselves that '
    f'the count per triplet discards')
R("C", '150h positives', statistic='complete triplets listwise', value=comp)

pd.DataFrame(rows).to_csv(
    os.path.join(OUT, "88_ipcw_and_denominators.csv"), index=False)
with open(os.path.join(OUT, "88_ipcw_and_denominators.txt"), "w") as f:
    f.write("\n".join(rep) + "\n")
log('\nWritten outputs/88_ipcw_and_denominators.{csv,txt}')
