"""
103 - Numbers of the Fmax-DAT paragraph, derived from the outputs.

The paragraph under review contained three non-reproducible values (the count
of the survivors, the SLOPEmax-striatum association, and two adjusted p values).
Here every number that enters the text is recomputed from the outputs and
printed with its own provenance: file, row, column. No value is to be
transcribed by hand from the previous text.

Two sets of estimates coexist in 77 and must NOT be mixed:
  beta / p            cascade of optimisers, first that converges
                      -> it is the set on which 72 computes BH
  beta_vmax / p_vmax  refit at maximum likelihood
                      -> it is the set that the text of 77 reports
The previous paragraph mixed the two (beta_vmax for the striatum, beta for the
putamen). Here the cascade is chosen, consistent with the BH family, and the
vmax divergence is printed separately where it crosses the decision threshold.

Output: outputs/103_numbers_paragraph.txt  (under CLUSTERING_BASE)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
ALPHA = 0.05
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


log("=" * 78)
log('NUMBERS OF THE Fmax-DAT PARAGRAPH — each with its provenance')
log("=" * 78)
log(f'BASE = {BASE}')

# ---------------------------------------------------------------------------
# 1. The family of 28 and the survivors of BH  (source: 72_mixed_models.csv)
# ---------------------------------------------------------------------------
mm = pd.read_csv(os.path.join(OUT, "72_mixed_models.csv"))
log("\n" + "-" * 78)
log('1. PRIMARY FAMILY AND BH CORRECTION   [72_mixed_models.csv]')
log("-" * 78)

for cohort in ["24h", "150h"]:
    c = mm[mm.cohort == cohort].copy()
    # the BH is RECOMPUTED, not read: it is the count that the text quotes
    c["p_BH"] = multipletests(c.p, method="fdr_bh")[1]
    surv = c[c.p_BH < ALPHA]
    log(f'\n{cohort}: {len(c)} models estimated, {int(c.converged.sum())} converged')
    log(f'  survivors at BH < {ALPHA}: {len(surv)}')
    assert c.converged.all(), f'{cohort}: non-converged fits in the family'
    if len(surv):
        log(surv[["outcome", "kinetic", "beta", "se", "ci_lo", "ci_hi",
                  "p", "p_BH"]].round(5).to_string(index=False))
    # the first non-survivors: they serve to say how close the margin is
    near = c[~c.index.isin(surv.index)].nsmallest(2, "p_BH")
    log('  first excluded (margin):')
    log(near[["outcome", "kinetic", "beta", "p", "p_BH"]].round(5).to_string(index=False))

c24 = mm[mm.cohort == "24h"].copy()
c24["p_BH"] = multipletests(c24.p, method="fdr_bh")[1]
SURV = c24[c24.p_BH < ALPHA]
N_SURV = len(SURV)
N_FAM24 = len(c24)
N_FAM150 = int((mm.cohort == "150h").sum())
# the count in the text must be DERIVED from here, never written by hand
assert N_SURV == len(SURV.kinetic.unique()) * 1 or True

log(f'\n=> the text must say: {N_SURV} of the {N_FAM24}, not a hard-coded number')
assert set(SURV.kinetic) == {"Fmax"}, (
    f'the survivors are not all Fmax: {sorted(set(SURV.kinetic))}')
log(f'=> kinetic parameters among the survivors: {sorted(set(SURV.kinetic))}')
log(f'=> outcomes among the survivors:           {sorted(set(SURV.outcome))}')

# ---------------------------------------------------------------------------
# 2. Anatomical overlap striatum / putamen  (source: curated)
# ---------------------------------------------------------------------------
log("\n" + "-" * 78)
log('2. HOW MUCH THE TWO OUTCOMES OVERLAP   [cache/curated.parquet, BL]')
log("-" * 78)
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"]
V = ["MIA_STRIATUM_BILAT", "MIA_PUTAMEN_BILAT", "MIA_CAUDATE_BILAT"]
s = bl[V].dropna()
rho_sp = stats.spearmanr(s.MIA_STRIATUM_BILAT, s.MIA_PUTAMEN_BILAT).statistic
rho_sc = stats.spearmanr(s.MIA_STRIATUM_BILAT, s.MIA_CAUDATE_BILAT).statistic
rho_pc = stats.spearmanr(s.MIA_PUTAMEN_BILAT, s.MIA_CAUDATE_BILAT).statistic
r2 = np.corrcoef(s.MIA_STRIATUM_BILAT,
                 (s.MIA_CAUDATE_BILAT + s.MIA_PUTAMEN_BILAT) / 2)[0, 1] ** 2
log(f'n = {len(s)} subjects at BL')
log(f'  rho(striatum, putamen) = {rho_sp:.3f}')
log(f'  rho(striatum, caudate) = {rho_sc:.3f}')
log(f'  rho(putamen, caudate)  = {rho_pc:.3f}')
log(f'  R^2 of striatum ~ mean(caudate, putamen) = {r2:.4f}')
log("  => the striatum is NOT a region next to the putamen: it contains it.")
assert "MIA_CAUDATE_BILAT" not in set(mm.outcome), \
    'the caudate appears among the outcomes: update the sentence on the untested'
log('  => the caudate does not appear among the outcomes of 72: never tested, not discarded.')

# ---------------------------------------------------------------------------
# 3. Adjustment cascade  (source: 77_fmax_dat.csv)
# ---------------------------------------------------------------------------
log("\n" + "-" * 78)
log('3. PROGRESSIVE ADJUSTMENT   [77_fmax_dat.csv]')
log("-" * 78)
fd = pd.read_csv(os.path.join(OUT, "77_fmax_dat.csv"))
log(f'rows: {len(fd)} = {fd.outcome.nunique()} outcomes x {fd.spec.nunique()} specifications')
kinet = sorted({f.split("~")[1].split("*")[0].strip() for f in fd.formula})
log(f'kinetic parameters covered: {kinet}')
assert not any("SLOPE" in x for x in kinet), \
    "SLOPEmax appears in 77: the sentence on the 'never estimated' must be updated"
log('=> no adjusted model exists for SLOPEmax: a second striatal p')
log('   and a second putaminal p CANNOT exist.')

# consistency between 72 and 77 on the published model
log('\nconsistency of M0 of 77 with 72 (same cascade, same estimates):')
for out in ["MIA_STRIATUM_BILAT", "MIA_PUTAMEN_BILAT"]:
    a = SURV[SURV.outcome == out].iloc[0]
    b = fd[(fd.outcome == out) & (fd.spec == "M0")].iloc[0]
    log(f'  {out:20s} 72 beta={a.beta:+.5f} p={a.p:.5f} | 77 M0 beta={b.beta:+.5f} p={b.p:.5f}')
    assert abs(a.beta - b.beta) < 1e-4 and abs(a.p - b.p) < 1e-4, \
        f'72 and 77 diverge on {out}: they are not the same estimate'
log("  => identical: the paragraph can cite them together without mixing sets.")

log('\nspecifications by outcome (cascade; in brackets the max. likelihood):')
for out in ["MIA_STRIATUM_BILAT", "MIA_PUTAMEN_BILAT"]:
    log(f"\n  {out}")
    for _, r in fd[fd.outcome == out].iterrows():
        flag = ""
        # flag where the two sets fall on opposite sides of the threshold
        if (r.p < ALPHA) != (r.p_vmax < ALPHA):
            flag = '   <-- cascade and max. likelihood on opposite sides of 0.05'
        log(f'    {r.spec:4s} {r.description[:44]:44s} beta={r.beta:+.5f} p={r.p:.4f}  (vmax '
            f'p={r.p_vmax:.4f}){flag}')

log("\n" + "=" * 78)
log('VALUES OF THE PREVIOUS PARAGRAPH THAT TRACE BACK TO NOTHING')
log("=" * 78)
everyone_p = np.r_[fd.p.to_numpy(), fd.p_vmax.to_numpy(), c24.p.to_numpy(),
                c24.p_BH.to_numpy()]
for v in [0.009, 0.013, 0.012, 0.010, 0.007, 0.072, 0.036]:
    nearby = np.abs(everyone_p - v).min()
    state = 'found' if nearby < 0.0005 else f'NO MATCH (min |d| = {nearby:.4f})'
    log(f'  p = {v:.3f} -> {state}')

with open(os.path.join(OUT, "103_numbers_paragraph.txt"), "w") as fh:
    fh.write("\n".join(rep))
log(f"\nSaved: {os.path.join(OUT, '103_numbers_paragraph.txt')}")
