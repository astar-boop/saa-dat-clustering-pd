"""
104 - Comparison of the battery across the three choices of DAT measure.

Six branches: two populations x three putaminal/striatal measures. The `low`
branches were not rerun: they are the outputs already on disk (outputs/ for the
full cohort, run_sporadici/outputs/ for the sporadic participants) and serve
as the reference.

The headline statistics are read from the CSVs, not from the text reports, so
that the comparison does not depend on the parsing of a sentence. The gap k is
RECOMPUTED with Tibshirani's 1-SE criterion instead of being read, and compared
with what the report states: if they diverged, the number in the report would
be a literal and not a computation.

Output: dat_variants/104_comparison.csv, .txt (under the working directory)
"""

import os
import re
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("CLUSTERING_BASE", REPO)
DEST = os.path.join(BASE, "dat_variants")
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# (label, population, measure, outputs folder)
BRANCHES = [
    ('sporadic / low  (ref.)', 'sporadic', "low",      os.path.join(BASE, "outputs")),
    ('sporadic / mean',        'sporadic', "mean",     os.path.join(DEST, "mean", "outputs")),
    ('sporadic / striatum',    'sporadic', "striatum", os.path.join(DEST, "striatum", "outputs")),
]


def k_1se(gap):
    """Tibshirani's 1-SE criterion: the smallest k with gap(k) >= gap(k+1) - s(k+1)."""
    g = gap.sort_values("k").reset_index(drop=True)
    for i in range(len(g) - 1):
        if g.gap[i] >= g.gap[i + 1] - g.s_k[i + 1]:
            return int(g.k[i])
    return int(g.k.iloc[-1])


def load(folder):
    r = {}
    # --- sidecar: which branch actually produced these files
    fn = os.path.join(folder, "feature_names.json")
    if os.path.exists(fn):
        import json
        j = json.load(open(fn))
        r['dat_declared'] = ",".join(j.get("dat", []))
        r["dat_set_sidecar"] = j.get("dat_set", '(absent)')

    # --- prediction strength
    ps = pd.read_csv(os.path.join(folder, "03_prediction_strength.csv"))
    r['PS_max_obsv'] = float(ps.PS_mean_obsv.max())
    r['PS_k_of_max'] = int(ps.loc[ps.PS_mean_obsv.idxmax(), "k"])
    r['PS_max_null'] = float(ps.PS_mean_null.max())
    r["PS_supera_0.80"] = bool((ps.PS_mean_obsv >= 0.80).any())

    # --- gap: k recomputed, not read
    go = pd.read_csv(os.path.join(folder, "03_gap_observed.csv"))
    gn = pd.read_csv(os.path.join(folder, "03_gap_null.csv"))
    r['gap_k_obsv'] = k_1se(go)
    r['gap_k_null'] = k_1se(gn)

    # --- maximum silhouette, observed and under the null
    ii = pd.read_csv(os.path.join(folder, "03_internal_indices.csv"))
    r['sil_max_obsv'] = float(ii[ii.data == 'observed'].silhouette.max())
    r['sil_max_null'] = float(ii[ii.data != 'observed'].silhouette.max())

    # --- agreement between algorithms at k=2.
    # MEAN, not median: it is the statistic that 03 reports and that the manuscript
    # quotes ("0.141 versus 0.246"). With the median the comparison would be with a
    # number that does not appear in the paper.
    aa = pd.read_csv(os.path.join(folder, "03_algo_agreement.csv"))
    r['ARI_k2_obsv'] = float(aa[(aa.data == 'observed') & (aa.k == 2)].ARI.mean())
    r['ARI_k2_null'] = float(aa[(aa.data != 'observed') & (aa.k == 2)].ARI.mean())

    # --- PAC and Jaccard at k=2
    pac = pd.read_csv(os.path.join(folder, "03_pac.csv"))
    r['PAC_k2_obsv'] = float(pac[(pac.data == 'observed') & (pac.k == 2)].PAC.iloc[0])
    bj = pd.read_csv(os.path.join(folder, "03_bootstrap_jaccard.csv"))
    r['jaccard_k2_obsv'] = float(bj[(bj.data == 'observed') & (bj.k == 2)].jaccard_avg.iloc[0])
    r['jaccard_k2_null'] = float(bj[(bj.data != 'observed') & (bj.k == 2)].jaccard_avg.iloc[0])

    # --- Hopkins and dip from the report of 02
    tr = os.path.join(folder, "02_tendency_report.txt")
    if os.path.exists(tr):
        t = open(tr).read()
        m = re.search('Hopkins \\(primary\\): H = ([\\d.]+)', t)
        r["hopkins"] = float(m.group(1)) if m else np.nan
        m = re.search(r"Hopkins on a Gaussian null with the same covariance: H = ([\d.]+)", t)
        r['hopkins_null'] = float(m.group(1)) if m else np.nan
        m = re.search(r"projections with p < 0\.05[^\d]*(\d+) / (\d+)", t)
        if not m:
            m = re.search(r"projections p<0\.05: (\d+) / (\d+)", t)
        r['dip_projections_sig'] = f"{m.group(1)}/{m.group(2)}" if m else "n/d"

    # --- 150h: maximum PS and gap k in the replication
    f150 = os.path.join(folder, "08_predstrength_150h.csv")
    if os.path.exists(f150):
        p150 = pd.read_csv(f150)
        col = 'PS_mean_obsv' if 'PS_mean_obsv' in p150.columns else p150.columns[1]
        r["PS150_max"] = float(p150[col].max())
        r['PS150_k_of_max'] = int(p150.loc[p150[col].idxmax(), "k"])
    g150 = os.path.join(folder, "08_gap_150h.csv")
    if os.path.exists(g150):
        gg = pd.read_csv(g150)
        if {"k", "gap", "s_k"} <= set(gg.columns):
            r["gap150_k"] = k_1se(gg)
    d150 = os.path.join(folder, "08_dataset_150h.csv")
    if os.path.exists(d150):
        dd = pd.read_csv(d150)
        r["n_150h"] = int(dd["complete"].sum()) if "complete" in dd.columns else len(dd)
    return r


rows = []
for label, pop, measure, dirpath in BRANCHES:
    if not os.path.isdir(dirpath):
        log(f'WARNING: folder missing, branch skipped: {dirpath}')
        continue
    try:
        r = load(dirpath)
    except FileNotFoundError as e:
        log(f'WARNING: branch {label} incomplete ({e.filename}): skipped')
        continue
    r.update({'branch': label, 'population': pop, 'measure': measure})
    rows.append(r)

df = pd.DataFrame(rows)

# Each branch must have produced the measure it declares: without this check,
# a branch left behind would go unnoticed.
#
# The check is on the CONTENT — the pair of DAT columns actually used —
# not on the `dat_set` key. The two reference branches were produced
# before the sidecar existed and do not have it: it is the content that says
# which branch they are, and the expected values are derived from _dat.py, not
# rewritten here.
import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
import _dat

log("=" * 78)
log('CHECK: did the branch produce the measure it declares?')
log("=" * 78)
for _, r in df.iterrows():
    expected_env = os.environ.get("DAT_SET")
    os.environ["DAT_SET"] = r['measure']
    expected_dat = ",".join(_dat.variables())
    if expected_env is None:
        del os.environ["DAT_SET"]
    else:
        os.environ["DAT_SET"] = expected_env

    found_dat = r.get('dat_declared', "")
    sidecar = r.get("dat_set_sidecar", '(absent)')
    ok = (found_dat == expected_dat)
    # the sidecar, when it is there, must agree with the content
    ok_sidecar = sidecar in ('(absent)', r['measure'])
    log(f"  {r['branch']:26s} dat={found_dat:42s} sidecar={sidecar:9s} "
        f"{('ok' if ok and ok_sidecar else '<<< MISALIGNED')}")
    assert ok, (f"branch {r['branch']}: uses {found_dat}, expected {expected_dat}")
    assert ok_sidecar, (f"branch {r['branch']}: sidecar {sidecar} contradicts the measure "
                        f"{r['measure']}")

log("\n" + "=" * 78)
log('BATTERY — PRIMARY COHORT 24h')
log("=" * 78)
COLS = ['branch', 'PS_max_obsv', 'PS_max_null', "PS_supera_0.80", 'gap_k_obsv',
        'sil_max_obsv', 'sil_max_null', 'ARI_k2_obsv', 'ARI_k2_null',
        'PAC_k2_obsv', 'jaccard_k2_obsv', "hopkins", 'hopkins_null',
        'dip_projections_sig']
log(df[[c for c in COLS if c in df.columns]].round(3).to_string(index=False))

if "PS150_max" in df.columns:
    log("\n" + "=" * 78)
    log('REPLICATION 150h')
    log("=" * 78)
    c150 = [c for c in ['branch', "n_150h", "PS150_max", 'PS150_k_of_max', "gap150_k"]
            if c in df.columns]
    log(df[c150].round(3).to_string(index=False))
    missing = df[df.PS150_max.isna()].branch.tolist()
    if missing:
        log(f'\n  branches without 150h replication: {missing}')

log("\n" + "=" * 78)
log('DOES THE CONCLUSION MOVE?')
log("=" * 78)
for pop in ['sporadic', 'complete']:
    s = df[df.population == pop]
    if s.empty:
        continue
    log(f"\n{pop}:")
    log(f'  maximum observed PS: ' +
        " | ".join(f'{r.measure}={r.PS_max_obsv:.3f}' for _, r in s.iterrows()))
    log(f"  no branch reaches 0.80: {not s['PS_supera_0.80'].any()}")
    log(f'  gap chooses k=1 in all branches: {(s.gap_k_obsv == 1).all()}  (values: '
        f'{s.gap_k_obsv.tolist()})')
    log(f'  observed silhouette always <= null: {(s.sil_max_obsv <= s.sil_max_null).all()}')

df.to_csv(os.path.join(DEST, "104_comparison.csv"), index=False)
with open(os.path.join(DEST, "104_comparison.txt"), "w") as fh:
    fh.write("\n".join(rep))
log(f"\nSaved: {os.path.join(DEST, '104_comparison.csv')}")
