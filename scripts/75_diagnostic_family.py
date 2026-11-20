"""
75 - Reconstruction of the diagnostic multiplicity family and recomputation of the
     p values corrected with Benjamini-Hochberg within criterion.

Context. The manuscript (run_sporadici/paper_risultati.md, line 111) states
"Eight of 76 values exceed the null nominally ... none survives
Benjamini-Hochberg correction within criterion (smallest adjusted p = 0.166)".
A reviewer counts 20 + 18 + 18 = 56 and declares the 76 wrong.

This script:
  1. reconstructs the exact composition of the 76 rows of
     outputs/45_diagnostic_curves.csv (statistic x cohort x k);
  2. compares the n_test_family column with the actual number of rows per
     statistic and with the number of non-missing p_BH;
  3. recomputes BH within criterion from p_montecarlo with two denominators:
       - "righe"  : all the rows of the statistic (20/20/18/18 = 76)
       - "come_46": the scheme actually implemented in 46_build_curve_docx.py,
                    which excludes the two degenerate k=1 rows of the increment
                    (20/18/18/18 = 74 tests, 2 rows without p_BH)
     and verifies whether the "come_46" scheme reproduces the p_BH column of the CSV;
  4. independently verifies the 36 rows of outputs/57_ch_db_curves.csv
     (Calinski-Harabasz and Davies-Bouldin).

No value is transcribed by hand: everything is read from the CSVs.

Output: outputs/75_diagnostic_family.csv, outputs/75_diagnostic_family.txt
"""

import os
import numpy as np
import pandas as pd

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

F45 = os.path.join(OUT, "45_diagnostic_curves.csv")
F57 = os.path.join(OUT, "57_ch_db_curves.csv")

rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def bh(p):
    """Benjamini-Hochberg step-up. Returns the corrected p values in input order.

    Explicit implementation (not statsmodels) so as not to inherit its choices:
    p_adj_(i) = min_{j >= i} ( n / j * p_(j) ), truncated at 1.
    """
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    q = ps * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n, float)
    out[order] = q
    return out


# ===========================================================================
# 1. composition of the family
# ===========================================================================
long = pd.read_csv(F45)

log("=" * 78)
log("75 - DIAGNOSTIC FAMILY OF MULTIPLICITY: COMPOSITION AND RECOMPUTED BH")
log("=" * 78)
log(f'Source: {F45}')
log(f'Total rows in the file: {len(long)}')
log("")
log('Composition by statistic x cohort:')

comp = (long.groupby(['statistic', 'cohort'])
        .agg(n_rows=("k", "size"), k_min=("k", "min"), k_max=("k", "max"))
        .reset_index())
log(comp.to_string(index=False))

log("")
log('Composition by statistic:')
by_stat = (long.groupby('statistic')
            .agg(n_rows=("k", "size"),
                 n_cohorts=('cohort', "nunique"),
                 k_min=("k", "min"), k_max=("k", "max"),
                 n_test_family_recorded=('n_test_family', "first"),
                 n_p_BH_no_missing=("p_BH", lambda s: int(s.notna().sum())))
            .reset_index())
by_stat['n_test_family_recorded'] = by_stat['n_test_family_recorded'].astype("Int64")
by_stat['coincides_rows_vs_recorded'] = (
    by_stat.n_rows == by_stat.n_test_family_recorded)
log(by_stat.to_string(index=False))

tot = int(by_stat.n_rows.sum())
log("")
log(f'Sum of the rows by statistic: '
    + " + ".join(str(int(v)) for v in by_stat.n_rows) + f" = {tot}")

three = by_stat[by_stat.statistic != 'gap_increment_over_k1']
log(f'Sum excluding gap_increment_over_k1 (the count of the reviewer): '
    + " + ".join(str(int(v)) for v in three.n_rows) + f' = {int(three.n_rows.sum())}')

# degenerate rows: p exactly 1 by construction
deg = long[(long.statistic == 'gap_increment_over_k1') & (long.k == 1)]
log("")
log("Rows k = 1 of the increment (degenerate by construction: observed = 0 and null increment = 0 "
    "in every realisation):")
log(deg[['cohort', 'statistic', "k", 'observed', 'null_copula_median',
         "p_montecarlo", "p_BH"]].to_string(index=False))

# ===========================================================================
# 2. recomputation of BH within criterion
# ===========================================================================
STATS = list(by_stat.statistic)

long['p_BH_rows'] = np.nan       # family = all the rows of the statistic
long['p_BH_like_46'] = np.nan     # family = rows with k >= 2 for the increment
long['n_fam_rows'] = 0
long['n_fam_like_46'] = 0

for stat in STATS:
    m = long.statistic == stat
    long.loc[m, 'p_BH_rows'] = bh(long.loc[m, "p_montecarlo"].to_numpy())
    long.loc[m, 'n_fam_rows'] = int(m.sum())

    m2 = m & ((long.k >= 2) if stat == 'gap_increment_over_k1' else True)
    long.loc[m2, 'p_BH_like_46'] = bh(long.loc[m2, "p_montecarlo"].to_numpy())
    long.loc[m, 'n_fam_like_46'] = int(m2.sum())

log("")
log("-" * 78)
log('2. BENJAMINI-HOCHBERG DENOMINATORS')
log("-" * 78)
den = (long.groupby('statistic')
       .agg(rows=("k", "size"),
            recorded_in_CSV=('n_test_family', "first"),
            denom_recomputation_rows=('n_fam_rows', "first"),
            denom_recomputation_like_46=('n_fam_like_46', "first"))
       .reset_index())
den['recorded_in_CSV'] = den['recorded_in_CSV'].astype("Int64")
log(den.to_string(index=False))
log("")
log(f'Total rows = {int(den.rows.sum())}; total recorded denominators = '
    f'{int(den.recorded_in_CSV.sum())}; total tests actually corrected in scheme 46 = '
    f'{int(den.denom_recomputation_like_46.sum())}')

# does the "come_46" scheme reproduce the p_BX column of the CSV?
msk = long.p_BH.notna()
d46 = float(np.max(np.abs(long.loc[msk, "p_BH"] - long.loc[msk, 'p_BH_like_46'])))
nan_ok = bool((long.p_BH.isna() == long.p_BH_like_46.isna()).all())
log("")
log(f"Reproduction of the p_BH column of the CSV with the scheme 'come_46': maximum discrepancy = "
    f"{d46:.3e} over {int(msk.sum())} values; pattern of missing values identical: {nan_ok}")

msk_r = long.p_BH.notna()
dR = float(np.max(np.abs(long.loc[msk_r, "p_BH"] - long.loc[msk_r, 'p_BH_rows'])))
log(f"Maximum discrepancy between p_BH of the CSV and the scheme 'righe' (denominators 20/20/18/18): "
    f"{dR:.3e}")

# ===========================================================================
# 3. (a) nominally significant
# ===========================================================================
log("")
log("-" * 78)
log('3. (a) NOMINALLY SIGNIFICANT AT 0.05 (p_montecarlo NOT corrected)')
log("-" * 78)
sig = long[long.p_montecarlo < 0.05].copy()
sig = sig.sort_values("p_montecarlo")
log(f'Number: {len(sig)} out of {len(long)}')
log("")
cols = ['statistic', 'cohort', "k", 'observed', 'null_copula_median',
        "nullo_copula_IC97.5", "p_montecarlo", "p_BH", 'p_BH_rows',
        'p_BH_like_46']
log(sig[cols].to_string(index=False,
                        float_format=lambda v: f"{v:.6f}"))
log("")
log('Breakdown by statistic and cohort:')
log(sig.groupby(['statistic', 'cohort']).k.apply(
    lambda s: ", ".join(f"k={int(x)}" for x in sorted(s))).to_string())

# ===========================================================================
# 4. (b), (c) survivors of BH and minimum p
# ===========================================================================
log("")
log("-" * 78)
log('4. (b) SURVIVORS OF BH AND (c) MINIMUM CORRECTED p')
log("-" * 78)
for label, col in [('CSV as it is (p_BH)', "p_BH"),
                       ('recomputation, family = all the rows (20/20/18/18)',
                        'p_BH_rows'),
                       ('recomputation, scheme 46 (20/18/18/18)', 'p_BH_like_46')]:
    s = long[col]
    n_sig = int((s < 0.05).sum())
    log(f'{label}:')
    log(f'   survive BH 0.05: {n_sig}')
    log(f'   minimum corrected p: {np.nanmin(s.to_numpy()):.6f}')
    imin = s.idxmin()
    r = long.loc[imin]
    log(f'   reached by: {r.statistic}, {r.cohort}, k = {int(r.k)} (p_montecarlo = '
        f'{r.p_montecarlo:.6f})')

log("")
log('minimum corrected p by statistic, in the three schemes:')
mins = (long.groupby('statistic')
        .agg(min_p_montecarlo=("p_montecarlo", "min"),
             min_p_BH_CSV=("p_BH", "min"),
             min_p_BH_rows=('p_BH_rows', "min"),
             min_p_BH_like_46=('p_BH_like_46', "min"))
        .reset_index())
log(mins.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

# ===========================================================================
# 5. (d) does any conclusion change?
# ===========================================================================
log("")
log("-" * 78)
log("5. (d) DIFFERENCES FROM THE VALUES ALREADY IN THE CSV")
log("-" * 78)
diff = long[np.abs(long.p_BH.fillna(-1) - long.p_BH_rows.fillna(-1)) > 1e-12]
log(f'Rows in which p_BH (CSV) and p_BH_rows (recomputation with denominator 20 for the increment) '
    f'differ: {len(diff)}')
if len(diff):
    log(diff[['statistic', 'cohort', "k", "p_montecarlo", "p_BH",
              'p_BH_rows']].to_string(index=False,
                                       float_format=lambda v: f"{v:.6f}"))
log("")
log(f"Number of significant results after BH: CSV = {int((long.p_BH < 0.05).sum())}; denominator "
    f"'righe' = {int((long.p_BH_rows < 0.05).sum())}; scheme 46 = "
    f"{int((long.p_BH_like_46 < 0.05).sum())}")

# ===========================================================================
# 6. independent verification of CH / DB (script 57)
# ===========================================================================
log("")
log("=" * 78)
log('6. INDEPENDENT CHECK OF CALINSKI-HARABASZ AND DAVIES-BOULDIN')
log("=" * 78)
ch = pd.read_csv(F57)
log(f'Source: {F57}')
log(f'Total rows: {len(ch)}')
log("")
log(ch.groupby(['index', 'cohort'])
    .agg(n_rows=("k", "size"), k_min=("k", "min"), k_max=("k", "max"))
    .reset_index().to_string(index=False))

ch['p_BH_index'] = np.nan       # family = index, both cohorts (18)
for ind in ch.index.unique():
    m = ch.index == ind
    ch.loc[m, 'p_BH_index'] = bh(ch.loc[m, "p_montecarlo"].to_numpy())
ch['p_BH_everyone_36'] = bh(ch.p_montecarlo.to_numpy())

d57 = float(np.max(np.abs(ch.p_BH - ch.p_BH_index)))
log("")
log(f'Reproduction of the p_BH column of 57 with family = index (n = 18 per index): maximum '
    f'discrepancy = {d57:.3e}')

sig57 = ch[ch.p_montecarlo < 0.05].sort_values("p_montecarlo")
log("")
log(f'Nominally significant at 0.05: {len(sig57)} out of {len(ch)}')
log(sig57[['index', 'cohort', "k", 'observed', 'null_copula_median',
           "p_montecarlo", "p_BH", 'p_BH_index', 'p_BH_everyone_36']]
    .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
log("")
log(f'Survivors of BH 0.05 (family = index, n = 18): {int((ch.p_BH_index < 0.05).sum())}; minimum '
    f'corrected p = {ch.p_BH_index.min():.6f}')
log(f'Survivors of BH 0.05 (single family of 36): {int((ch.p_BH_everyone_36 < 0.05).sum())}; minimum '
    f'corrected p = {ch.p_BH_everyone_36.min():.6f}')

# ===========================================================================
# saving
# ===========================================================================
keep = ['cohort', 'statistic', "k", 'observed', 'null_copula_median',
        "nullo_copula_IC2.5", "nullo_copula_IC97.5", "p_montecarlo", "B",
        "p_BH", 'n_test_family', 'n_fam_rows', 'p_BH_rows',
        'n_fam_like_46', 'p_BH_like_46']
res = long[keep].copy()
res["nominalmente_significativo_0.05"] = res.p_montecarlo < 0.05
res["significativo_BH_righe_0.05"] = res.p_BH_rows < 0.05
res["significativo_BH_come_46_0.05"] = res.p_BH_like_46 < 0.05
res = res.rename(columns={"p_BH": 'p_BH_in_CSV_45',
                          'n_test_family': 'n_test_family_in_CSV_45'})

ch_out = ch.rename(columns={'index': 'statistic', "p_BH": 'p_BH_in_CSV_57'})
ch_out['n_test_family_in_CSV_57'] = ch_out.groupby('statistic').k.transform("size")
ch_out = ch_out[['cohort', 'statistic', "k", 'observed',
                 'null_copula_median', "nullo_copula_IC2.5",
                 "nullo_copula_IC97.5", "p_montecarlo", "B",
                 'p_BH_in_CSV_57', 'n_test_family_in_CSV_57',
                 'p_BH_index', 'p_BH_everyone_36']]
ch_out["nominalmente_significativo_0.05"] = ch_out.p_montecarlo < 0.05
ch_out["significativo_BH_indice_0.05"] = ch_out.p_BH_index < 0.05

res['family'] = "45_diagnostic_curves"
ch_out['family'] = "57_ch_db_curves"
out = pd.concat([res, ch_out], ignore_index=True, sort=False)
out.to_csv(os.path.join(OUT, "75_diagnostic_family.csv"), index=False)
log("")
log(f"Saved: {os.path.join(OUT, '75_diagnostic_family.csv')} ({len(out)} rows)")

with open(os.path.join(OUT, "75_diagnostic_family.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
print('Saved:', os.path.join(OUT, "75_diagnostic_family.txt"))
