"""
24 - P1.6: single primary family and specification curve of the MoCA coefficient.

The review observes that the same test on the MoCA receives three different
corrected values (0.098 / 0.017 / 0.023) depending on the family, and that the
families neither partition the space nor were chosen a priori. The correct
answer is twofold: to declare a single primary family, and to show the
distribution of the result over all the discretionary choices instead of
choosing one of them.

Choices enumerated:
  - kinetic parameter           : T50, TTT, AUC
  - aggregation of replicates   : median, mean
  - estimation strategy         : mixed model, individual slopes, ANCOVA
  - random slope                : yes, no
  - LEDD as covariate           : no, yes
Total 3 x 2 x 3 x 2 x 2 = 72 specifications.

Output: outputs/24_multiverse.csv, outputs/24_multiverse.txt
"""

import os
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"]
d = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d = d[d.complete].reset_index(drop=True)

covbl = bl[["PATNO", "age_at_visit", "SEX", "EDUCYRS", "duration_yrs"]].rename(
    columns={"age_at_visit": "age_bl", "duration_yrs": "dur_bl"})
L0 = cur[cur.PATNO.isin(d.PATNO)][["PATNO", "YEAR", "moca", "LEDD"]].copy()
L0["years"] = pd.to_numeric(L0.YEAR, errors="coerce")
L0 = L0[L0.years.notna()]

KIN = {"T50": ("k_T50", "kmean_T50"), "TTT": ("k_TTT", "kmean_TTT"),
       "AUC": ("k_AUC", "kmean_AUC")}
SIGN = {"T50": +1, "TTT": +1, "AUC": -1}   # high AUC = fast seeding


def build(kin, agg):
    col = KIN[kin][0 if agg == 'median' else 1]
    b = d[["PATNO", col]].rename(columns={col: "K"}).dropna()
    b["K"] = np.log(b.K)
    b["K"] = (b.K - b.K.mean()) / b.K.std()
    b["K"] = b.K * SIGN[kin]          # oriented: high values = slow seeding
    return L0.merge(b, on="PATNO").merge(covbl, on="PATNO", how="left")


def estimate(L, strategy, slope_random, ledd):
    cov = "age_bl + SEX + EDUCYRS + dur_bl" + (" + LEDD" if ledd else "")
    need = ["PATNO", "years", "moca", "K", "age_bl", "SEX", "EDUCYRS", "dur_bl"]
    if ledd:
        need.append("LEDD")
    m = L[need].dropna()
    if strategy == 'mixed model':
        re = "~years" if slope_random else None
        r = smf.mixedlm(f"moca ~ K*years + {cov}", m, groups=m.PATNO,
                        re_formula=re).fit(method="lbfgs", maxiter=300)
        t = "K:years" if "K:years" in r.params.index else "years:K"
        return r.params[t], r.bse[t], r.pvalues[t], m.PATNO.nunique()
    if strategy == 'individual slopes':
        s = m[["PATNO", "years", "moca"]]
        cnt = s.groupby("PATNO").size()
        s = s[s.PATNO.isin(cnt[cnt >= 3].index)]
        sl = (s.groupby("PATNO").apply(lambda g: np.polyfit(g.years, g.moca, 1)[0])
              .rename("slope").reset_index())
        kk = m.groupby("PATNO").K.first().reset_index()
        mm = sl.merge(kk, on="PATNO")
        rho, p = stats.spearmanr(mm.K, mm.slope)
        return rho, np.nan, p, len(mm)
    # ANCOVA conditional on baseline, years > 0
    b0 = m[m.years == 0][["PATNO", "moca"]].rename(columns={"moca": "base"})
    post = m[m.years > 0].merge(b0, on="PATNO")
    re = "~years" if slope_random else None
    r = smf.mixedlm(f"moca ~ K*years + base + {cov}", post, groups=post.PATNO,
                    re_formula=re).fit(method="lbfgs", maxiter=300)
    t = "K:years" if "K:years" in r.params.index else "years:K"
    return r.params[t], r.bse[t], r.pvalues[t], post.PATNO.nunique()


log("=" * 78)
log('P1.6 — SPECIFICATION CURVE OF THE MoCA COEFFICIENT')
log("=" * 78)
rows = []
combos = list(itertools.product(KIN, ['median', 'mean'],
                                ['mixed model', 'individual slopes', "ANCOVA"],
                                [True, False], [False, True]))
for kin, agg, strat, slope_c, ledd in combos:
    if strat == 'individual slopes' and (slope_c is False or ledd):
        continue          # the two-stage strategy does not have these degrees of freedom
    try:
        L = build(kin, agg)
        b, se, p, n = estimate(L, strat, slope_c, ledd)
    except Exception as e:
        continue
    rows.append({'kinetic': kin, 'aggregation': agg, 'strategy': strat,
                 'slope_random': slope_c, "LEDD": ledd,
                 "beta": b, "se": se, "p": p, 'n_subj': n})
mv = pd.DataFrame(rows)
log(f'\nSpecifications estimated: {len(mv)}')

# corrected p values under the three families discussed in the manuscript
for fam, size in [('family of 6 outcomes', 6), ('family of 12 outcomes', 12),
                  ('full space of 35 tests', 35)]:
    mv[f"pBH_{size}"] = np.minimum(1, mv.p * size / 1)   # BH in the worst case = Bonferroni
mv['direction_expectation'] = mv.beta > 0

log('\nDistribution of the uncorrected p:')
log(f'  median {mv.p.median():.4f} | range {mv.p.min():.4g}–{mv.p.max():.4g}')
log(f'  specifications with p < 0.05         : {(mv.p < 0.05).sum()} / {len(mv)} '
    f'({100 * (mv.p < 0.05).mean():.0f}%)')
log(f'  in the expected direction            : {mv.direction_expectation.sum()} / {len(mv)} '
    f'({100 * mv.direction_expectation.mean():.0f}%)')
log('\nWith conservative correction per family (Bonferroni on the family size):')
for size in [6, 12, 35]:
    log(f"  family of {size:2d} tests: {(mv[f'pBH_{size}'] < 0.05).sum():2d} / {len(mv)} "
        f"specifications survive")

log('\nBy estimation strategy:')
log(mv.groupby('strategy').agg(n=("p", "size"), p_median=("p", "median"),
                                frac_p05=("p", lambda x: (x < .05).mean())).round(4).to_string())
log('\nBy kinetic parameter:')
log(mv.groupby('kinetic').agg(n=("p", "size"), p_median=("p", "median"),
                               frac_p05=("p", lambda x: (x < .05).mean())).round(4).to_string())
log('\nBy aggregation of the replicates:')
log(mv.groupby('aggregation').agg(n=("p", "size"), p_median=("p", "median"),
                                   frac_p05=("p", lambda x: (x < .05).mean())).round(4).to_string())

log('\nSpecifications ordered by p (first 10 and last 5):')
srt = mv.sort_values("p")[['kinetic', 'aggregation', 'strategy',
                           'slope_random', "LEDD", "beta", "p", 'n_subj']]
log(srt.head(10).round(4).to_string(index=False))
log("...")
log(srt.tail(5).round(4).to_string(index=False))

mv.to_csv(os.path.join(OUT, "24_multiverse.csv"), index=False)

log("\n" + "=" * 78)
log('PROPOSED PRIMARY FAMILY')
log("=" * 78)
log('A single family, declared a priori in the review protocol:')
log('  predictor   : standardised log T50, median of the triplicates')
log('  outcomes    : the twelve non-motor domains, one primary measure each')
log('  strategy    : mixed model with random intercept and slope')
log('  correction  : Benjamini-Hochberg on those twelve tests')
log('Everything else — the other kinetic parameters, the other strategies, the motor and')
log('dopaminergic outcomes, the extended battery — must be declared explicitly exploratory.')

with open(os.path.join(OUT, "24_multiverse.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/24_multiverse.csv')
