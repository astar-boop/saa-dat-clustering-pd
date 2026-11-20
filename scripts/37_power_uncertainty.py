"""
37 - P2: uncertainty and monotonicity of the sensitivity curves (script 22).

Three checks requested by the review.

(a) The injection does NOT preserve the global mean and covariance. Shifting a
    share pi of subjects by a vector delta, the mean moves by pi*delta and
    pi(1-pi) delta delta' is added to the covariance. Here the formula is
    verified numerically against the empirical covariance.

(b) Each cell is a proportion over 60 replicates: the uncertainty must be
    declared. 95% Wilson intervals are reported.

(c) The curves of the figure are not monotone in Delta. Each inversion is
    quantified and it is verified, with a two-proportion test, whether it
    exceeds the expected Monte Carlo variability.

Output: outputs/37_power_uncertainty.csv, outputs/37_power_uncertainty.txt
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SEED = 20260805
N_REP = 60
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------
# (a) what the injection really does to the covariance
# ---------------------------------------------------------------------------
npz = np.load(os.path.join(OUT, "features.npz"))
X24 = npz["F_primary"]

d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete]
F150 = ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE", "dat_putamen_low",
        "dat_caudate_mean", "age_bl"]
M = d150[F150].to_numpy(float).copy()
for j in range(4):
    M[:, j] = np.log(M[:, j])
X150 = (M - M.mean(0)) / M.std(0)


def inject(X, direction, delta, pi, rng):
    n = len(X)
    S = np.cov(X, rowvar=False)
    u = direction / np.sqrt(direction @ S @ direction)
    idx = rng.choice(n, int(round(pi * n)), replace=False)
    Y = X.copy()
    Y[idx] = Y[idx] + delta * (S @ u)
    return Y, delta * (S @ u), idx


log("=" * 78)
log("(a) THE INJECTION DOES NOT PRESERVE THE GLOBAL MEAN AND COVARIANCE")
log("=" * 78)
log('Shifting a share π of subjects by the vector δ, the mean moves by πδ')
log('and π(1−π)δδ′ is added to the covariance. Numerical check:')
log("")
rows_cov = []
for tag, X in [("24h", X24), ("150h", X150)]:
    pca = PCA().fit(X)
    for dname, dvec in [("PC1", pca.components_[0]), ("PC2", pca.components_[1])]:
        for delta in (1.0, 2.0, 3.0):
            for pi in (0.15, 0.30, 0.50):
                rng = np.random.default_rng(SEED)
                Y, dvecfull, idx = inject(X, dvec, delta, pi, rng)
                S0, S1 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
                expected = pi * (1 - pi) * np.outer(dvecfull, dvecfull)
                obsv = S1 - S0
                # variance along the injection direction (normalised)
                w = dvecfull / np.linalg.norm(dvecfull)
                v0, v1 = w @ S0 @ w, w @ S1 @ w
                rows_cov.append({
                    'cohort': tag, 'direction': dname, "delta": delta, "pi": pi,
                    'var_earliest': v0, 'var_after': v1, 'ratio': v1 / v0,
                    'trace_earliest': np.trace(S0), 'trace_after': np.trace(S1),
                    'shift_mean': np.linalg.norm(pi * dvecfull),
                    'error_relative_formula':
                        np.abs(obsv - expected).max() / np.abs(expected).max()})
cov = pd.DataFrame(rows_cov)
log(cov[cov.cohort == "24h"][['direction', "delta", "pi", 'var_earliest',
                              'var_after', 'ratio', 'trace_earliest',
                              'trace_after', 'error_relative_formula']]
    .round(3).to_string(index=False))
log("")
log("Agreement between the observed covariance increment and the formula π(1−π)δδ′")
log("(maximum relative error on the element of largest modulus):")
for delta in sorted(cov.delta.unique()):
    s = cov[cov.delta == delta].error_relative_formula
    log(f'  Δ = {delta:.1f}   median {s.median():.3f}   maximum {s.max():.3f}')
log("")
log('The formula holds in expectation: the residual discrepancy is sampling')
log('variability of the shifted subset, and it is relatively larger at small')
log('Δ, where the added term is small relative to the noise. In any')
log("case the direction and the order of magnitude of the effect are those indicated")
log("by the review, and the injection does NOT preserve the covariance.")
cov.to_csv(os.path.join(OUT, "37_covariance_injection.csv"), index=False)

log("")
log("What the injection preserves is the distribution WITHIN each subgroup:")
log('the marginals of the two groups are those of the real data, and only the')
log('relative position of the two centres changes. The overall matrix instead has')
log('larger variance along the direction of injection, as shown above.')

# ---------------------------------------------------------------------------
# (b) Wilson intervals on the 60 replicates
# ---------------------------------------------------------------------------
pw = pd.read_csv(os.path.join(OUT, "22_power.csv"))
CRIT = [('power_dip', "dip"), ('power_PS', "PS"), ('power_jaccard', "jaccard")]
rows = []
for _, r in pw.iterrows():
    for col, name in CRIT:
        p = float(r[col])
        k = int(round(p * N_REP))
        lo, hi = wilson(k, N_REP)
        rows.append({'cohort': r.cohort, 'direction': r.direction,
                     "delta": r.delta, "pi": r.pi, 'criterion': name,
                     'successes': k, "n": N_REP, 'power': p,
                     'CI95_inf': lo, 'CI95_upp': hi, 'width_CI': hi - lo})
inc = pd.DataFrame(rows)
inc.to_csv(os.path.join(OUT, "37_power_uncertainty.csv"), index=False)

log("\n" + "=" * 78)
log('(b) UNCERTAINTY OF THE POWER ESTIMATES (Wilson 95%, n = 60 replicates)')
log("=" * 78)
log(f'Median width of the interval: {inc.width_CI.median():.3f}  (maximum {inc.width_CI.max():.3f} '
    f'around 0.50)')
log("")
log('Cells that determine the minimum detectable separation (power ≥ 0.80):')
sog = inc[(inc.power >= 0.80)].sort_values(['cohort', 'criterion', "delta"])
for (co, cr), g in sog.groupby(['cohort', 'criterion']):
    g2 = g.loc[g.groupby("pi").delta.idxmin()]
    for _, r in g2.iterrows():
        log(f'  {co:13s} {cr:8s} π={r.pi:.2f}  Δ={r.delta:.1f}  power {r.power:.2f}  95% CI '
            f'[{r.CI95_inf:.2f}; {r.CI95_upp:.2f}]')

log("")
log('With 60 replicates an estimated power of 0.80 has 95% CI [0.68; 0.88]: the')
log('minimum detectable separation is to be read as a point estimate with this')
log('uncertainty, not as a sharp threshold.')

# ---------------------------------------------------------------------------
# (c) monotonicity
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log('(c) INVERSIONS OF THE CURVES (power that falls as Δ increases)')
log("=" * 78)


def two_proportions(k1, k2, n=N_REP):
    p1, p2 = k1 / n, k2 / n
    pp = (k1 + k2) / (2 * n)
    se = np.sqrt(2 * pp * (1 - pp) / n)
    if se == 0:
        return 1.0
    return 2 * (1 - stats.norm.cdf(abs(p1 - p2) / se))


inv = []
for (co, of, pi, cr), g in inc.groupby(['cohort', 'direction', "pi", 'criterion']):
    g = g.sort_values("delta")
    v = g.successes.to_numpy()
    dd = g.delta.to_numpy()
    for i in range(len(v) - 1):
        if v[i + 1] < v[i]:
            p = two_proportions(v[i], v[i + 1])
            inv.append({'cohort': co, 'direction': of, "pi": pi, 'criterion': cr,
                        'fr_delta': dd[i], "a_delta": dd[i + 1],
                        'power_fr': v[i] / N_REP, 'power_a': v[i + 1] / N_REP,
                        'decline': (v[i] - v[i + 1]) / N_REP, "p": p})
INV = pd.DataFrame(inv)
INV.to_csv(os.path.join(OUT, "37_inversions.csv"), index=False)
log(f"Inversions observed: {len(INV)} out of "
    f"{len(inc) - inc.groupby(['cohort', 'direction', 'pi', 'criterion']).ngroups} consecutive "
    f"transitions")
sig = INV[INV.p < 0.05]
log(f'  of which exceed the Monte Carlo variability (two-proportion test, p < 0.05): {len(sig)}')
log("")
if len(sig):
    log(sig.sort_values('decline', ascending=False)
        [['cohort', 'direction', "pi", 'criterion', 'fr_delta', "a_delta",
          'power_fr', 'power_a', "p"]].round(3).to_string(index=False))
log("")
log('By criterion:')
for cr in ("dip", "PS", "jaccard"):
    s = INV[INV.criterion == cr]
    ss = s[s.p < 0.05]
    log(f'  {cr:8s} inversions {len(s):2d}, of which not explainable by chance {len(ss):2d}, maximum '
        f'drop {(s.decline.max() if len(s) else 0):.2f}')

with open(os.path.join(OUT, "37_power_uncertainty.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/37_power_uncertainty.csv, 37_inversions.csv')
