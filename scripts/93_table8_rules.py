"""
93 - Which optimisation rule lies behind each row of the table of the
     MoCA specifications (current Table 9), and what would change
     under the strict rule.

WHY THIS SCRIPT EXISTS
----------------------
§2.18 states mutually incompatible things about the estimation rules. The rules
in circulation are two, and must be kept distinct:

  RULE A ("first that converges"). _fit.fit_mixed: lbfgs -> powell -> nm ->
  bfgs on the SAME model, the FIRST that reports converged=True is
  returned. No restart, no Newton decrement. It is used by
  15_nonmotor_trajectories.py, 78_longitudinal_sensitivity.py and
  83_spline_reconciled.py.

  RULE B ("maximum likelihood among the converged, then restart, then
  Newton decrement < 1e-4"). Scripts 88 and 91.

  There is in addition a third, intermediate variant, and it must be named:

  RULE C ("maximum likelihood among the converged, without restart").
  87_amplitude_ladder.fit_strict: tries all four optimisers from the
  starting values of statsmodels, discards the non-converged, keeps the one
  with the highest llf. No restart, no Newton decrement.

  And the procedure of the 63 published models (84_gradients.py) is yet another:
  cascade up to the FIRST converged, then restart of the whole cascade from
  THAT point (function `polishing`), iterated until the likelihood no longer
  rises. The Newton decrement is computed and reported.

This script:
  (1) verifies row by row the provenance of the thirteen specifications of the
      MoCA table, matching the values printed in the manuscript with the
      records of 78_longitudinal_sensitivity.csv and 83_spline_reconciled.csv;
  (2) refits under Rule B every row that is a mixed model, and measures by
      how much beta, standard error and p move with respect to the value
      published under Rule A;
  (3) recomputes the two pooling rows from the stratum fits under
      Rule B;
  (4) records which rows are marginal estimates (exchangeable GEE and least
      squares with clustered sandwich), to which the rule on the
      optimisers does not apply: there is no cascade, there is no random
      effect, there is no likelihood to maximise by profile.

Output: outputs/93_table8_rules.csv, outputs/93_table8_rules.txt
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS, fit_mixed

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
rep = []

TOL_NEWTON = 1e-4
TOL_GAIN = 1e-10
MAX_ROUNDS = 10


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


# ============================================================ data (as in 78/83)
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
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
m["years2"] = m.years ** 2

SPL_SPEC = "cr(years, df=3, constraints='center')"
_B = dmatrix(SPL_SPEC, {"years": m.years}, return_type="dataframe")
_B = _B.drop(columns=[c for c in _B.columns if c == "Intercept"])
for i, c in enumerate(_B.columns):
    m[["s1", "s2", "s3"][i]] = _B[c].values

N_SUBJ, N_OBSV = int(m.PATNO.nunique()), int(len(m))

# pattern-mixture strata (definition identical to 78)
last_year = m.groupby("PATNO").years.max()


def stratum_of(a):
    if a < 5:
        return '1: <5 years'
    if a < 10:
        return '2: 5-9 years'
    return '3: >=10 years'


pat = last_year.map(stratum_of).rename("pattern")
m = m.merge(pat, left_on="PATNO", right_index=True, how="left")
LEV = sorted(pat.unique())
cnt = pat.value_counts().reindex(LEV)
PI = (cnt / cnt.sum()).to_dict()


# ================================================================ primitives
def pack(r):
    md = r.model
    return np.asarray(
        r.params_object.get_packed(use_sqrt=md.use_sqrt, has_fe=False),
        dtype=float)


def _ll(md, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(md.loglike(p))


def _score(md, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(md.score(p), dtype=float)


def hess_profiled(md, p):
    k = len(p)
    h = np.array([1e-4 * max(1.0, abs(v)) for v in p])
    H = np.zeros((k, k))
    f0 = _ll(md, p)
    for i in range(k):
        pp, pm = p.copy(), p.copy()
        pp[i] += h[i]
        pm[i] -= h[i]
        H[i, i] = (_ll(md, pp) - 2 * f0 + _ll(md, pm)) / h[i] ** 2
        for j in range(i + 1, k):
            a, b, c, e = p.copy(), p.copy(), p.copy(), p.copy()
            a[i] += h[i]; a[j] += h[j]
            b[i] += h[i]; b[j] -= h[j]
            c[i] -= h[i]; c[j] += h[j]
            e[i] -= h[i]; e[j] -= h[j]
            H[i, j] = H[j, i] = (_ll(md, a) - _ll(md, b) - _ll(md, c)
                                 + _ll(md, e)) / (4 * h[i] * h[j])
    return (H + H.T) / 2


def newton(r):
    md = r.model
    p = pack(r)
    g = _score(md, p)
    H = hess_profiled(md, p)
    try:
        step = np.linalg.solve(H, -g)
        dec = float(0.5 * g @ step)
    except np.linalg.LinAlgError:
        dec = np.nan
    return dec, float(np.max(np.abs(g)))


def estimate(formula, data, groups, method, it, re_formula, start=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kw = {} if start is None else {"start_params": start}
            return smf.mixedlm(formula, data, groups=groups,
                               re_formula=re_formula).fit(method=method,
                                                          maxiter=it, **kw)
        except Exception:
            return None


def rule_B(formula, data, groups, re_formula="~years"):
    """Stages 1-3 as in 88_diagnostics_63.procedura."""
    attempts, md_ref = {}, None
    for method, it in OPTIMISERS:
        r = estimate(formula, data, groups, method, it, re_formula)
        if r is None:
            attempts[method] = (None, np.nan, False)
            continue
        if md_ref is None:
            md_ref = r.model
        attempts[method] = (r, _ll(md_ref, pack(r)), bool(r.converged))
    if md_ref is None:
        return None
    conv = [(k, v[1]) for k, v in attempts.items()
            if v[0] is not None and v[2] and np.isfinite(v[1])]
    pool = conv or [(k, v[1]) for k, v in attempts.items()
                    if v[0] is not None and np.isfinite(v[1])]
    if not pool:
        return None
    chosen = max(pool, key=lambda x: x[1])[0]
    r0, ll0 = attempts[chosen][0], attempts[chosen][1]
    best = (r0, '(none)', ll0, pack(r0))
    rounds = 0
    for cycle in range(MAX_ROUNDS):
        improved = False
        for method, it in OPTIMISERS:
            r = estimate(formula, data, groups, method, it, re_formula,
                      start=best[3])
            if r is None:
                continue
            ll = _ll(md_ref, pack(r))
            if np.isfinite(ll) and ll > best[2] + TOL_GAIN:
                best = (r, method, ll, pack(r))
                improved = True
        rounds = cycle + 1
        if not improved:
            break
    return {"r": best[0], "opt": chosen, "ll0": ll0, "ll": best[2],
            'rounds': rounds, 'reopt': best[1]}


def find(res, components):
    want = set(components)
    for name in res.params.index:
        if set(str(name).split(":")) == want:
            return name
    return None


def wald(res, names):
    names = [n for n in names if n is not None and n in res.params.index]
    if not names:
        return None, None, None
    b = res.params[names].values.astype(float)
    V = res.cov_params().loc[names, names].values.astype(float)
    try:
        Vi = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V)
    stat = float(b @ Vi @ b)
    return stat, len(names), float(st.chi2.sf(stat, len(names)))


# ==================================================== (1) the thirteen rows
log("=" * 78)
log('93 — THE ESTIMATION RULES BEHIND THE TABLE OF THE MoCA SPECIFICATIONS')
log("=" * 78)
log(f'statsmodels {sm.__version__}, numpy {np.__version__}')
log(f'BASE = {BASE}')
log(f'data: {N_SUBJ} subjects, {N_OBSV} observations')
log("")

MS = open(os.path.join(BASE, "paper_risultati.md")).read()
block = MS.split("| Specification | β | SE | p |")[1]
rows_tab = [r for r in block.split("\n") if r.startswith("| ")]
rows_tab = [r for r in rows_tab if r.count("|") == 5]
log("-" * 78)
log('COUNT OF THE SPECIFICATIONS IN THE TABLE OF THE MANUSCRIPT')
log("-" * 78)
for i, r in enumerate(rows_tab, 1):
    log(f"  {i:2d}. {r.split('|')[1].strip()}")
log(f'rows of the table: {len(rows_tab)}  (reference 1 + alternatives {len(rows_tab) - 1})')
log("")

# underlying mixed models, with the table row to which they correspond
MIXED = [
    ("ref", "Linear time (reference)", "A1",
     "moca ~ T50*years + age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")], ("T50", "years"), None),
    ("alt1", "Quadratic time, interaction on both terms", "A2",
     "moca ~ T50 + years + years2 + T50:years + T50:years2 "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years"), ("T50", "years2")], None, None),
    ("alt2", "Quadratic time, interaction on the linear term", "A3",
     "moca ~ T50 + years + years2 + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")], ("T50", "years"), None),
    ("alt3", "Spline time, interaction on the whole basis", "A4",
     "moca ~ T50 + s1 + s2 + s3 + T50:s1 + T50:s2 + T50:s3 "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "s1"), ("T50", "s2"), ("T50", "s3")], None, None),
    ("alt4", "Spline time, interaction on the linear term", "A5r",
     "moca ~ T50 + s1 + s2 + s3 + T50:years "
     "+ age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")], ("T50", "years"), None),
    ("alt5", "Pattern-mixture, stratum as covariate", "PM1",
     "moca ~ T50*years + C(pattern) + age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")], ("T50", "years"), None),
    ("alt6", "Pattern-mixture, stratum x time", "PM2",
     "moca ~ T50*years + C(pattern)*years + age_bl + SEX + dur_bl + EDUCYRS",
     [("T50", "years")], ("T50", "years"), None),
]
LAYERS = [(f"str{i}", f"stratum {k}", k,
           "moca ~ T50*years + age_bl + SEX + dur_bl + EDUCYRS",
           [("T50", "years")], ("T50", "years"), k)
          for i, k in enumerate(LEV, 1)]

out = []
log("-" * 78)
log('(2) EVERY MIXED MODEL UNDER THE TWO RULES')
log("-" * 78)
for tag, label, mod, formula, inter, single, stratum in MIXED + LAYERS:
    dat = m if stratum is None else m[m.pattern == stratum]
    rA, half, ripA = fit_mixed(formula, dat, dat.PATNO,
                               fallback_intercept=(stratum is not None))
    B = rule_B(formula, dat, dat.PATNO)
    rB = B["r"]
    names = [find(rA, c) for c in inter]
    namesb = [find(rB, c) for c in inter]
    if single is not None:
        nA, nB = find(rA, single), find(rB, single)
        bA, seA, pA = float(rA.params[nA]), float(rA.bse[nA]), float(rA.pvalues[nA])
        bB, seB, pB = float(rB.params[nB]), float(rB.bse[nB]), float(rB.pvalues[nB])
        sA = sB = gA = gB = np.nan
    else:
        bA = seA = bB = seB = np.nan
        sA, gA, pA = wald(rA, names)
        sB, gB, pB = wald(rB, namesb)
    decA, gnA = newton(rA)
    decB, gnB = newton(rB)
    out.append({
        'row': tag, 'specification': label, 'model': mod,
        'kind': 'mixed model',
        'rule_pubd': 'A (first converged)',
        'n_subj': int(dat.PATNO.nunique()), 'n_obsv': int(len(dat)),
        "opt_ruleA": half, "opt_ruleB": B["opt"], 'rounds_ruleB': B['rounds'],
        "loglike_ruleA": _ll(rA.model, pack(rA)),
        "loglike_ruleB": B["ll"],
        'gain_loglike_B_less_A': B["ll"] - _ll(rA.model, pack(rA)),
        "beta_ruleA": bA, "beta_ruleB": bB,
        "se_ruleA": seA, "se_ruleB": seB,
        "p_ruleA": pA, "p_ruleB": pB,
        "wald_ruleA": sA, "wald_ruleB": sB, "df": gA,
        "newton_ruleA": decA, "newton_ruleB": decB,
        "grad_ruleA": gnA, "grad_ruleB": gnB,
        'satisfies_newton_ruleA': bool(np.isfinite(decA) and decA < TOL_NEWTON),
        'satisfies_newton_ruleB': bool(np.isfinite(decB) and decB < TOL_NEWTON),
        "abs_delta_beta": abs(bB - bA) if np.isfinite(bA) else np.nan,
        "abs_delta_se": abs(seB - seA) if np.isfinite(seA) else np.nan,
        "abs_delta_p": abs(pB - pA),
    })
    log(f"  {label[:46]:46s} ({mod:3s})  optA={half:6s} optB={B['opt']:6s}  newton A={decA:.2e} "
        f"B={decB:.2e}  |dp|={abs(pB - pA):.2e}")

D = pd.DataFrame(out)

# ---- (3) pooling from the stratum estimates under Rule B --------------------
log("")
log("-" * 78)
log('(3) THE TWO POOLING ROWS, RECOMPUTED FROM THE STRATUM FITS')
log("-" * 78)
S = D[D.row.str.startswith("str")]
w = np.array([PI[k] for k in LEV], dtype=float)
w = w / w.sum()


def pooling_in_common(b, s):
    th = float(np.sum(w * b))
    var = float(np.sum(w ** 2 * s ** 2)
                + (np.sum(w * b ** 2) - th ** 2) / N_SUBJ)
    se = float(np.sqrt(var))
    p = float(2 * st.norm.sf(abs(th / se)))
    wi = 1.0 / s ** 2
    thi = float(np.sum(wi * b) / np.sum(wi))
    sei = float(np.sqrt(1.0 / np.sum(wi)))
    pi_ = float(2 * st.norm.sf(abs(thi / sei)))
    return (th, se, p), (thi, sei, pi_)


qA, ivA = pooling_in_common(S.beta_ruleA.to_numpy(float),
                          S.se_ruleA.to_numpy(float))
qB, ivB = pooling_in_common(S.beta_ruleB.to_numpy(float),
                          S.se_ruleB.to_numpy(float))
for name, a, b in [("Pattern-mixture, pooled by stratum frequency", qA, qB),
                   ("Pattern-mixture, pooled by inverse variance", ivA, ivB)]:
    log(f'  {name}')
    log(f'     Rule A: beta={a[0]:.6f} SE={a[1]:.6f} p={a[2]:.6f}')
    log(f'     Rule B: beta={b[0]:.6f} SE={b[1]:.6f} p={b[2]:.6f}')
    log(f'     discrepancies: |dbeta|={abs(b[0] - a[0]):.3e}  |dSE|={abs(b[1] - a[1]):.3e}  '
        f'|dp|={abs(b[2] - a[2]):.3e}')
    D = pd.concat([D, pd.DataFrame([{
        'row': 'pooling_in_common', 'specification': name,
        'model': "pool", 'kind': 'weighted mean of three stratum fits',
        'rule_pubd': 'A (first converged), across the stratum fits',
        'n_subj': N_SUBJ, 'n_obsv': N_OBSV,
        "beta_ruleA": a[0], "se_ruleA": a[1], "p_ruleA": a[2],
        "beta_ruleB": b[0], "se_ruleB": b[1], "p_ruleB": b[2],
        "abs_delta_beta": abs(b[0] - a[0]), "abs_delta_se": abs(b[1] - a[1]),
        "abs_delta_p": abs(b[2] - a[2])}])], ignore_index=True)

# ---- (4) the four marginal rows --------------------------------------------
log("")
log("-" * 78)
log('(4) THE FOUR MARGINAL ROWS: THE RULE DOES NOT APPLY')
log("-" * 78)
ip = pd.read_csv(os.path.join(OUT, "78_longitudinal_sensitivity.csv"))
marg = ip[(ip.section == "B2") & (ip.analysis == "IPCW")
          & ip.model.str.contains('GEE|LS', regex=True)]
for _, r in marg.iterrows():
    kind = ('Gaussian GEE, exchangeable correlation, clustered sandwich'
            if r.model.startswith("GEE") else
            'least squares (weighted or not), clustered sandwich')
    log(f'  {r.model[:60]:60s} beta={r.estimate:.6f} SE={r.se:.6f} p={r.p:.4f}  [{kind}]')
    log(f"     optimiser recorded in the output: {('(none)' if pd.isna(r.optimiser) else r.optimiser)}")
    D = pd.concat([D, pd.DataFrame([{
        'row': 'marginal', 'specification': r.model, 'model': "-",
        'kind': kind, 'rule_pubd': 'not applicable',
        'n_subj': r.n_subj, 'n_obsv': r.n_obsv,
        "beta_ruleA": r.estimate, "se_ruleA": r.se, "p_ruleA": r.p}])],
        ignore_index=True)
log("  None of these four models has random effects: there is no cascade of optimisers,")
log("  there is no profiled likelihood and there is no Newton decrement to compute.")

# ---- summary ---------------------------------------------------------------
mm = D[D.kind == 'mixed model']
log("")
log("-" * 78)
log('SUMMARY')
log("-" * 78)
log(f'mixed models redone: {len(mm)} ({len(MIXED)} table rows + {len(LAYERS)} stratum fits)')
log(f'satisfy the Newton criterion under Rule A: {int(mm.satisfies_newton_ruleA.sum())} of {len(mm)}')
log(f'satisfy the Newton criterion under Rule B: {int(mm.satisfies_newton_ruleB.sum())} of {len(mm)}')
log(f'Newton decrement under A — median {mm.newton_ruleA.median():.3e}  max '
    f'{mm.newton_ruleA.max():.3e}')
log(f'Newton decrement under B — median {mm.newton_ruleB.median():.3e}  max '
    f'{mm.newton_ruleB.max():.3e}')
log(f'log-likelihood gain B - A: max {mm.gain_loglike_B_less_A.max():.3e}')
every = D[D.abs_delta_p.notna()]
log(f'maximum shift of the coefficient (among the rows with a beta): {D.abs_delta_beta.max():.3e}')
log(f"maximum shift of the standard error: {D.abs_delta_se.max():.3e}")
log(f'maximum shift of the p: {every.abs_delta_p.max():.3e}')
worse = every.loc[every.abs_delta_p.idxmax()]
log(f'  (row with the maximum shift of the p: {worse.specification})')
_seg = D[D.beta_ruleA.notna() & D.beta_ruleB.notna()]
log(f'no row changes the sign of the coefficient: '
    f'{bool((np.sign(_seg.beta_ruleA.to_numpy()) == np.sign(_seg.beta_ruleB.to_numpy())).all())}')
_alt = D[D.row != "ref"]
log(f'excluding the reference row, among the twelve alternatives: |dbeta| max '
    f'{_alt.abs_delta_beta.max():.3e}, |dSE| max {_alt.abs_delta_se.max():.3e}, |dp| max '
    f'{_alt.abs_delta_p.max():.3e}')


# ---- (4b) matching with the printed values and check of the figures --------
log("")
log("-" * 78)
log('(4b) THE VALUES PRINTED IN THE TABLE, AGAINST THE TWO RULES')
log("-" * 78)


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


ORDERING = ["ref", "alt1", "alt2", "alt3", "alt4", "alt5", "alt6"] \
    + ['pooling_in_common', 'pooling_in_common'] + ['marginal'] * 4
idx = {'pooling_in_common': list(D.index[D.row == 'pooling_in_common']),
       'marginal': list(D.index[D.row == 'marginal'])}
used = {'pooling_in_common': 0, 'marginal': 0}
outcomes_digits = []
for row_ms, key in zip(rows_tab, ORDERING):
    fields = [c.strip() for c in row_ms.split("|")[1:-1]]
    if key in idx:
        i = idx[key][used[key]]
        used[key] += 1
    else:
        i = int(D.index[D.row == key][0])
    r = D.loc[i]
    bA, sA_, pA_ = f3(r.beta_ruleA), f3(r.se_ruleA), f3(r.p_ruleA)
    bB, sB_, pB_ = f3(r.beta_ruleB), f3(r.se_ruleB), f3(r.p_ruleB)
    ok_A = (fields[1].startswith(bA) or (fields[1] == "—" and bA == "—")) \
        and fields[3].startswith(pA_)
    ok_B = (bB == "—" or fields[1].startswith(bB)) and \
        (pB_ == "—" or fields[3].startswith(pB_))
    outcomes_digits.append(bool(ok_A and ok_B))
    log(f"  {fields[0][:44]:44s} printed b={fields[1]:>7s} SE={fields[2]:>6s} p={fields[3]:<12s} | A "
        f"{bA}/{sA_}/{pA_} | B {bB}/{sB_}/{pB_} | {('coincide' if ok_A and ok_B else 'DISCREPANT')}")
log(f'rows in which the printed figures coincide under both rules: {sum(outcomes_digits)} of '
    f'{len(outcomes_digits)}')

log("")
log("-" * 78)
log('(5) THE RULE OF THE OTHER TWO SETS, READ FROM THE OUTPUTS')
log("-" * 78)
lad = pd.read_csv(os.path.join(OUT, "87_amplitude_ladder.csv"))
log(f'87_amplitude_ladder.csv: {len(lad)} rows, {lad.spec.nunique()} specifications x '
    f'{len(lad) // lad.spec.nunique()} outcome x kinetic parameter combinations')
log(f'  optimisers retained: {lad.optimiser.value_counts().to_dict()}')
log('  rule of the code (fit_strict): among the converged ones the highest llf, without restart')
g84 = pd.read_csv(os.path.join(OUT, "84_gradients.csv"))
log(f'84_gradients.csv: {len(g84)} fits; the pol_* column (from which the published numbers come)')
log(f"  starts from the FIRST converged one and repeats the cascade from there: rounds "
    f"{g84.pol_rounds.min()}-{g84.pol_rounds.max()}")
log(f'  fits that did not satisfy the criterion before the restart: '
    f'{int((g84.pub_newton_decrement >= TOL_NEWTON).sum())} of {len(g84)}')
log(f'  after the restart: {int((g84.pol_newton_decrement >= TOL_NEWTON).sum())} of {len(g84)}')
dg = pd.read_csv(os.path.join(OUT, "88_diagnostics_63.csv"))
log(f'88_diagnostics_63.csv: {len(dg)} fits under Rule B (restart from the point chosen by '
    f'likelihood)')
c = dg.merge(g84[["cohort", "outcome", "kinetic", "pol_beta", "pol_se",
                  "pol_p"]], on=["cohort", "outcome", "kinetic"])
log(f"  maximum discrepancy between the two end points: beta "
    f"{abs(c.beta_interaction - c.pol_beta).max():.3e}, SE {abs(c.se - c.pol_se).max():.3e}, p "
    f"{abs(c.p - c.pol_p).max():.3e}")

D.to_csv(os.path.join(OUT, "93_table8_rules.csv"), index=False)
with open(os.path.join(OUT, "93_table8_rules.txt"), "w") as fh:
    fh.write("\n".join(rep) + "\n")
log("")
log(f"Saved: {os.path.join(OUT, '93_table8_rules.csv')}")
log(f"Saved: {os.path.join(OUT, '93_table8_rules.txt')}")
