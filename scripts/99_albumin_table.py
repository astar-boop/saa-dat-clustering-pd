"""
99 - The complete table of the albumin-kinetic parameter associations.

The main text refers to the supplement for «full detail» of the associations
between CSF albumin and kinetic parameters — partial rho, denominators and
corrected p for each parameter — but the supplement does not carry that table:
§1.17 cites some values in prose and Table S10 concerns another analysis,
the counts of kinetic parameter-biomarker associations by level of adjustment.
The cross-reference promises more than it delivers.

Here the table is built from the objects that already exist
(20_kinetics_vs_albumin.csv for the primary estimates, 23_albumin_haemoglobin.csv
for the two controls), in a single, complete form, with the explicit indication
of what is NOT available: the controls on haemoglobin and Qalb exist for the
150 h cohort only.

The script also verifies the figures cited in the main text and in the supplement,
and flags those that do not hold up to rounding.

Usage:  CLUSTERING_BASE=... ONLY_SPORADIC=1 python3 scripts/99_albumin_table.py
Output: outputs/99_albumin_table.csv, .txt
"""

import math
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


NAMES = {"Fmax": "Fmax", "TTT": "TTT", "T50": "T50", "AUC": "AUC",
        "SLOPE": "SLOPEmax"}
COHORT = {"150h": "Replication (150 h)", "24h": "Primary (24 h)"}

base = pd.read_csv(os.path.join(OUT, "20_kinetics_vs_albumin.csv"))
extra = pd.read_csv(os.path.join(OUT, "23_albumin_haemoglobin.csv"))

log("=" * 78)
log('99 — ALBUMIN-KINETIC PARAMETER ASSOCIATIONS, COMPLETE TABLE')
log("=" * 78)

# ----------------------------------------------------------- the correction family
_, bh9, _, _ = multipletests(base.p.values, alpha=0.05, method="fdr_bh")
departure = float(np.abs(bh9 - base.p_BH.values).max())
log(f'\nCorrection family: the {len(base)} tests of the two cohorts TOGETHER.')
log(f'  BH recomputed on the {len(base)}: maximum difference from the saved value {departure:.1e}')
within = []
for c in ("150h", "24h"):
    s = base[base.cohort == c]
    _, b, _, _ = multipletests(s.p.values, alpha=0.05, method="fdr_bh")
    within.append(float(np.abs(b - s.p_BH.values).max()))
log(f'  if it were within cohort the difference would be {within[0]:.2e} and {within[1]:.2e}')
log("  -> the family is the joint one, and the supplement must say so: elsewhere")
log('     the manuscript corrects WITHIN cohort, so the convention changes here.')

# ------------------------------------------------------------------- the table
rows = []
for _, r in base.iterrows():
    e = extra[extra.kinetic == r.kinetic]
    e = e.iloc[0] if len(e) and r.cohort == "150h" else None
    rows.append({
        'cohort': COHORT[r.cohort], 'parameter': NAMES[r.kinetic],
        "n": int(r.n), 'rho_partial': float(r.rho_partial),
        "p": float(r.p), "p_BH": float(r.p_BH),
        'CI_inf': float(e.CI_inf) if e is not None else np.nan,
        'CI_upp': float(e.CI_upp) if e is not None else np.nan,
        'rho_agg_haemoglobin': float(e.rho_agg_haemoglobin) if e is not None else np.nan,
        'p_agg_haemoglobin': float(e.p_agg) if e is not None else np.nan,
        "rho_Qalb": float(e.rho_Qalb) if e is not None else np.nan,
        "p_Qalb": float(e.p_Qalb) if e is not None else np.nan})

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "99_albumin_table.csv"), index=False)


def f(x, dec=3):
    return "—" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{dec}f}"


def fp(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}"


log("")
log("=" * 78)
log('THE TABLE (the control columns exist for the 150 h cohort only)')
log("=" * 78)
log(f"{'cohort':22s} {'par.':9s} {'n':>4s} {'rho':>7s} {'95% CI':>18s} {'p':>9s} {'BH p':>9s} "
    f"{'rho|Hb':>8s} {'p|Hb':>9s} {'rho Qalb':>9s} {'p Qalb':>9s}")
for _, r in d.iterrows():
    ci = (f'[{r.CI_inf:+.3f}, {r.CI_upp:+.3f}]'
          if not math.isnan(r.CI_inf) else "—")
    log(f'{r.cohort:22s} {r.parameter:9s} {r.n:4d} {r.rho_partial:+7.3f} {ci:>18s} {fp(r.p):>9s} '
        f'{fp(r.p_BH):>9s} {f(r.rho_agg_haemoglobin):>8s} {fp(r.p_agg_haemoglobin):>9s} '
        f'{f(r.rho_Qalb):>9s} {fp(r.p_Qalb):>9s}')

# ------------------------------------------- verification of the figures cited
log("")
log("=" * 78)
log('DO THE FIGURES QUOTED IN THE TEXT HOLD?')
log("=" * 78)


def verify(label, printed, measured, dec=3, limit=False):
    if limit:
        ok = measured <= printed
        remark = "" if ok else f'   THE DATA EXCEED IT ({measured:.6f})'
    else:
        expected = round(measured, dec)
        ok = abs(expected - printed) < 10 ** (-dec) / 2
        remark = "" if ok else f'   measured {measured:.6f} -> {expected:+.{dec}f}'
    log(f"  [{('ok ' if ok else 'NO ')}] {label:44s} {printed}{remark}")
    return ok


g = {(r.cohort, r.parameter): r for _, r in d.iterrows()}
R, P = "Replication (150 h)", "Primary (24 h)"
outcomes = [
    verify("150 h, T50, rho", 0.240, g[(R, "T50")].rho_partial),
    verify("150 h, TTT, rho", 0.234, g[(R, "TTT")].rho_partial),
    verify("150 h, SLOPEmax, rho", 0.243, g[(R, "SLOPEmax")].rho_partial),
    verify("150 h, AUC, rho", -0.234, g[(R, "AUC")].rho_partial),
    verify('150 h, maximum BH p of the four', 0.0002,
              max(g[(R, k)].p_BH for k in ("T50", "TTT", "SLOPEmax", "AUC")),
              limit=True),
    verify("24 h, Fmax, rho", -0.514, g[(P, "Fmax")].rho_partial),
    verify("24 h, SLOPEmax, rho", -0.500, g[(P, "SLOPEmax")].rho_partial),
    verify("24 h, AUC, rho", -0.448, g[(P, "AUC")].rho_partial),
    verify('Qalb, maximum p of the four', 0.0011,
              max(g[(R, k)].p_Qalb for k in ("T50", "TTT", "SLOPEmax", "AUC")),
              limit=True),
    verify('150 h, Fmax, p not significant', 0.105, g[(R, "Fmax")].p),
    verify('24 h, TTT, p not significant', 0.137, g[(P, "TTT")].p),
]

log("")
log(f'figures verified: {sum(outcomes)}/{len(outcomes)}')
if not all(outcomes):
    log("")
    log('The two that do not hold, with the correction:')
    log(f"  24 h AUC: rho = {g[P, 'AUC'].rho_partial:.6f} rounds to -0.447, not to -0.448.")
    m = max(g[(R, k)].p_Qalb for k in ("T50", "TTT", "SLOPEmax", "AUC"))
    log(f"  Qalb: the maximum p is {m:.6f}; «all p <= 0.0011» is false, the")
    log(f"        true bound is 0.0012. (It appears in the text and in §1.17.)")

log("")
log("=" * 78)
log("WHAT IS NOT AVAILABLE, AND MUST BE SAID")
log("=" * 78)
log("The two controls — adjustment for the haemoglobin indicator and")
log('substitution with Qalb — exist for the 150 h cohort only (n = 266).')
log('In the primary cohort (n = 55) they were not computed, so the sentence')
log('«The association survived adjustment ... and replicated when Qalb was')
log('substituted» must be referred to the replication, not to both cohorts.')
log('The confidence intervals too exist for the 150 h cohort only.')

with open(os.path.join(OUT, "99_albumin_table.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/99_albumin_table.csv and .txt')
