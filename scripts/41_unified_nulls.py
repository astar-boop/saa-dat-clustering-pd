"""
41 - Table of the Monte Carlo nulls with observed values from a single source.

The review observes that the same statistic appears with different values in the
figure of the selection of k and in the table of the nulls. The cause is not a
different analytical choice but the Monte Carlo variability of the two
stochastic statistics of the battery, prediction strength and Jaccard index,
which script 25 recomputed with fewer repetitions than those used in the main
analysis.

Here the table is rebuilt WITHOUT recomputing anything: the null distributions
saved by script 25 are compared with the observed values taken from the
canonical locations (script 03 for the primary cohort, script 08 for the
replication cohort). The document thus reports a single value per statistic.

Note on the direction of the approximation: for the Jaccard index the observed
value is estimated on 200 resamplings and the null on 25, so that the
null distribution is more dispersed than it would be at equal precision.
For an upper-tail test this makes the p-value conservative.

Output: outputs/41_unified_nulls.csv, outputs/41_unified_nulls.txt
"""

import os
import numpy as np
import pandas as pd

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")

from _cohorts import labels
COH24, COH150 = labels(OUT)
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


HIGHER = {"hopkins": True, "silhouette_k2": True,
          'n_projections_multimodal': True, "dip_pmin_bonf": False,
          "PS_k2": True, "jaccard_k2": True, 'ARI_algorithms_k2': True}
NICE = {"hopkins": "Hopkins", "silhouette_k2": "Silhouette (k = 2)",
        'n_projections_multimodal': 'Multimodal projections',
        "dip_pmin_bonf": 'minimum dip (Bonferroni)',
        "PS_k2": "Prediction strength (k = 2)",
        "jaccard_k2": "Jaccard (k = 2)",
        'ARI_algorithms_k2': 'ARI between algorithms (k = 2)'}

previous = pd.read_csv(os.path.join(OUT, "25_nulls.csv"))

# ---- canonical observed values --------------------------------------------
ps24 = pd.read_csv(os.path.join(OUT, "03_prediction_strength.csv"))
ps24_2 = ps24[ps24.k == 2].iloc[0]
jac24 = pd.read_csv(os.path.join(OUT, "03_bootstrap_jaccard.csv"))
jac24_2 = jac24[(jac24.data == 'observed') & (jac24.k == 2)].iloc[0]

ps150 = pd.read_csv(os.path.join(OUT, "08_predstrength_150h.csv"))
ps150_2 = ps150[ps150.k == 2].iloc[0]
st150 = pd.read_csv(os.path.join(OUT, "08_stability_150h.csv"))
st150_2 = st150[(st150.data == 'observed') & (st150.k == 2)].iloc[0]

N_SPLIT = 50            # partitions for the prediction strength, scripts 03 and 25
B_JAC_OBSV, B_JAC_NULL = 200, 25

CANONICAL = {
    "24h": {"PS_k2": (float(ps24_2.PS_mean_obsv),
                      float(ps24_2.PS_sd_obsv) / np.sqrt(N_SPLIT),
                      "03_prediction_strength.csv"),
            "jaccard_k2": (float(jac24_2.jaccard_avg), np.nan,
                           "03_bootstrap_jaccard.csv")},
    "150h": {"PS_k2": (float(ps150_2.PS_obsv), np.nan,
                       "08_predstrength_150h.csv"),
             "jaccard_k2": (float(st150_2.jaccard_avg), np.nan,
                            "08_stability_150h.csv")},
}

log("=" * 78)
log("TABLE OF THE NULLS WITH OBSERVED VALUES FROM A SINGLE SOURCE")
log("=" * 78)
log('No recomputation: the null distributions are those saved by')
log('script 25; the observed values come from the canonical locations.')
log("")

rows = []
for tag, slug, B in [(COH24, "24h", 500), (COH150, "150h", 300)]:
    N = pd.read_csv(os.path.join(OUT, f"25_null_distribution_{slug}.csv"))
    older = previous[previous.cohort == tag].set_index('statistic')
    for stat in HIGHER:
        can = CANONICAL[slug].get(stat)
        if can is not None:
            obsv, se, source = can
        else:
            obsv, se, source = float(older.loc[stat, 'observed']), np.nan, \
                "25_nulls.csv"
        v = N[stat].to_numpy(float)
        if HIGHER[stat]:
            p = (np.sum(v >= obsv) + 1) / (len(v) + 1)
        else:
            p = (np.sum(v <= obsv) + 1) / (len(v) + 1)
        rows.append({'cohort': tag, 'statistic': stat,
                      'label': NICE[stat], 'observed': obsv,
                      'SE_montecarlo': se, 'source_observed': source,
                      'null_median': float(np.median(v)),
                      "nullo_IC2.5": float(np.percentile(v, 2.5)),
                      "nullo_IC97.5": float(np.percentile(v, 97.5)),
                      "p_montecarlo": p, "B": len(v)})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "41_unified_nulls.csv"), index=False)

for tag in res.cohort.unique():
    s = res[res.cohort == tag]
    log(f"--- {tag} ---")
    log(f"{'statistic':28s} {'observed':>10s} {'MC SE':>8s} {'median':>9s} {'null 95% CI':>20s} "
        f"{'p':>8s}")
    for _, r in s.iterrows():
        se = f'{r.SE_montecarlo:.4f}' if np.isfinite(r.SE_montecarlo) else "—"
        log(f"{r.label:28s} {r.observed:10.4f} {se:>8s} {r['null_median']:9.4f} "
            f"{'[' + format(r['nullo_IC2.5'], '.3f') + '; ' + format(r['nullo_IC97.5'], '.3f') + ']':>20s} "
            f"{r.p_montecarlo:8.4f}")
    sig = s[s.p_montecarlo < 0.05]
    log(f'  exceed the null (p < 0.05): {len(sig)} out of {len(s)}'
        + (f" -> {', '.join(sig.label)}" if len(sig) else ""))
    log("")

log("=" * 78)
log('COMPARISON WITH THE PREVIOUS VERSION OF THE TABLE')
log("=" * 78)
for _, r in res.iterrows():
    old = previous[(previous.cohort == r.cohort) &
                  (previous.statistic == r.statistic)].iloc[0]
    if abs(old.observed - r.observed) > 1e-9:
        log(f'  {r.cohort:13s} {r.label:28s} {old.observed:.4f} -> {r.observed:.4f}   p '
            f'{old.p_montecarlo:.4f} -> {r.p_montecarlo:.4f}')

log("")
log('Precision of the two stochastic statistics:')
log(f"  prediction strength, primary cohort: {N_SPLIT} splits, Monte Carlo SE "
    f"{CANONICAL['24h']['PS_k2'][1]:.4f}")
log(f'  Jaccard index: {B_JAC_OBSV} resamples on the observed data and {B_JAC_NULL} on each null; '
    f'the null distribution is therefore more')
log("  dispersed than necessary and the resulting p-value is conservative.")

with open(os.path.join(OUT, "41_unified_nulls.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/41_unified_nulls.csv')
