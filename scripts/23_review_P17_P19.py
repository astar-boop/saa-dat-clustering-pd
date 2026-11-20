"""
23 - P1.7 (platform/run confounding on the shared component) and P1.9 (CI).

P1.7 Albumin differs from the other analytes on four collinear dimensions:
     abundance, origin, analytical principle and project/run. A
     non-correlation between measures performed in different runs is what a run
     effect predicts. Five analyses, in order of cost:
       (a) annotate the correlation matrix by platform/project;
       (b) p-tau (Elecsys) / p-tau181 (Project 283) duplicate: same molecule on
           different platforms against different molecules on the same platform;
       (c) decomposition of the variance of the kinetic parameters by run and
           instrument;
       (d) within-participant stability of the component across visits: a
           physiological determinant is a property of the participant, a run
           effect is a property of the run. It is the test that discriminates
           between the two explanations;
       (e) albumin-kinetics association against the haemoglobin indicator
           (P2.18, included here because it uses the same data).

P1.9 Confidence intervals for all the coefficients, and replacement of the null
     tests of the negative controls with intervals/equivalence.

Output: outputs/23_*.csv, outputs/23_review_P17_P19.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
from _alb import path_alb
from _paths import DATA
from _assay_run import row_of_kinetics
ALB = path_alb(OUT)   # rebuilt from the PPMI biospecimen file
SEED = 20260805
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"]
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete].reset_index(drop=True)

ANALYTES = ["abeta", "tau", "ptau", "asyn", "IU_pTau181_CSF",
            "IU_ABeta40_CSF", "IU_ABeta42_CSF"]
# platform and project of origin, from the PPMI data dictionary
PLATFORM = {"abeta": ("Elecsys", "125/159"), "tau": ("Elecsys", "125/159"),
            "ptau": ("Elecsys", "125/159"), "asyn": ("ELISA", "124"),
            "IU_pTau181_CSF": ("Indiana", "283"),
            "IU_ABeta40_CSF": ("Indiana", "283"),
            "IU_ABeta42_CSF": ("Indiana", "283")}

# ===========================================================================
# P1.7 (a) — the correlation matrix annotated by platform
# ===========================================================================
log("=" * 78)
log('P1.7(a) — DO THE CORRELATIONS CLUSTER WITHIN PLATFORM?')
log("=" * 78)
sub = bl[bl.PATNO.isin(d150.PATNO)][ANALYTES].dropna()
log(f'n with all seven analytes: {len(sub)}')
CORR = sub.corr(method="spearman")  # do not call it C: it shadows patsy's C()
within, between, rows = [], [], []
for i, a in enumerate(ANALYTES):
    for b in ANALYTES[i + 1:]:
        same = PLATFORM[a][0] == PLATFORM[b][0]
        r = CORR.loc[a, b]
        (within if same else between).append(r)
        rows.append({'analyte_1': a, 'analyte_2': b, "rho": r,
                     'platform_1': PLATFORM[a][0], 'platform_2': PLATFORM[b][0],
                     'selfsame_platform': same})
pl = pd.DataFrame(rows)
pl.to_csv(os.path.join(OUT, "23_correlations_by_platform.csv"), index=False)
log(f'\npairs on the SAME platform      : n = {len(within):2d}, median rho {np.median(within):.3f}, '
    f'range {min(within):.3f}–{max(within):.3f}')
log(f'pairs on DIFFERENT platforms    : n = {len(between):2d}, median rho {np.median(between):.3f}, '
    f'range {min(between):.3f}–{max(between):.3f}')
u, p = stats.mannwhitneyu(within, between)
log(f'comparison between the two sets : p = {p:.4f}')
log('\nIf the factor were instrumental, the within-platform correlations')
log('should dominate the between-platform ones.')

# ===========================================================================
# P1.7 (b) — the p-tau / p-tau181 duplicate
# ===========================================================================
log("\n" + "=" * 78)
log('P1.7(b) — SAME MOLECULE ON DIFFERENT PLATFORMS')
log("=" * 78)
r_same_mol = CORR.loc["ptau", "IU_pTau181_CSF"]
r_same_plat_elecsys = CORR.loc["tau", "ptau"]
r_same_plat_iu = CORR.loc["IU_ABeta40_CSF", "IU_ABeta42_CSF"]
log(f'p-tau (Elecsys) vs p-tau181 (Indiana)  — same molecule, different platforms : rho = '
    f'{r_same_mol:.3f}')
log(f't-tau vs p-tau (both Elecsys)          — different molecules, same platform : rho = '
    f'{r_same_plat_elecsys:.3f}')
log(f'Abeta40 vs Abeta42 (both Indiana)      — different molecules, same platform : rho = '
    f'{r_same_plat_iu:.3f}')
log('\nIf the correlation between different molecules on the same platform exceeded')
log('that of the same molecule on different platforms, the factor would be')
log('instrumental.')

# ===========================================================================
# P1.7 (c) — variance of the kinetic parameters by run and instrument
# ===========================================================================
log("\n" + "=" * 78)
log('P1.7(c) — DECOMPOSITION OF THE VARIANCE OF THE KINETIC PARAMETERS BY RUN')
log("=" * 78)
sa = pd.read_csv(os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"))
p150 = sa[(sa.PROJECTID == 155) & (sa.COHORT == "PD") & (sa.CLINICAL_EVENT == "BL")
          & (sa.SAA_Status == "Positive")]
p150 = p150[p150.PATNO.isin(d150.PATNO)].pipe(row_of_kinetics, OUT)[["PATNO", "RUNDATE", "InstrumentRep1"]]
kin = d150[["PATNO", "k_Fmax", "k_TTT", "k_AUC", "k_T50", "k_SLOPE"]].merge(
    p150, on="PATNO", how="left")
rows = []
for c in ["k_Fmax", "k_TTT", "k_AUC", "k_T50", "k_SLOPE"]:
    m = kin[[c, "RUNDATE", "InstrumentRep1"]].dropna().copy()
    m["y"] = np.log(m[c])
    # variance component attributable to the run (random-effects model)
    r = smf.mixedlm("y ~ 1", m, groups=m.RUNDATE).fit(method="lbfgs")
    var_run = float(r.cov_re.iloc[0, 0])
    var_res = float(r.scale)
    icc_run = var_run / (var_run + var_res)
    # instrument as a fixed effect
    f = smf.ols("y ~ C(InstrumentRep1)", m).fit()
    rows.append({'parameter': c.replace("k_", ""), "n": len(m),
                 'share_variance_run': icc_run,
                 'p_instrument': f.f_pvalue, 'R2_instrument': f.rsquared})
vr = pd.DataFrame(rows)
log(vr.round(4).to_string(index=False))
vr.to_csv(os.path.join(OUT, "23_variance_run.csv"), index=False)
log(f'\nDistinct runs: {kin.RUNDATE.nunique()} | instruments: {kin.InstrumentRep1.nunique()}')

# ===========================================================================
# P1.7 (d) — within-participant stability of the component across visits
# ===========================================================================
log("\n" + "=" * 78)
log('P1.7(d) — IS THE FACTOR A PROPERTY OF THE SUBJECT OR OF THE RUN?')
log("=" * 78)
log('A physiological determinant is stable within subject across distinct visits;')
log('a run effect is not, since different visits fall in different runs.')

long = cur[cur.PATNO.isin(d150.PATNO)][["PATNO", "EVENT_ID", "YEAR"] + ANALYTES].copy()
long = long.dropna(subset=ANALYTES)
log(f'\nObservations with all seven analytes: {len(long)} ({long.PATNO.nunique()} subjects)')
cnt = long.groupby("PATNO").size()
log(f'subjects with at least two complete visits: {(cnt >= 2).sum()}')

if (cnt >= 2).sum() >= 30:
    R = np.column_stack([stats.rankdata(long[c]) for c in ANALYTES])
    R = (R - R.mean(0)) / R.std(0)
    pca = PCA(n_components=1).fit(R)
    f = pca.transform(R).ravel()
    if stats.spearmanr(f, long[ANALYTES[0]]).statistic < 0:
        f = -f
    long = long.assign(factor=f)
    log(f'variance explained by PC1 over all visits: {pca.explained_variance_ratio_[0]:.3f}')
    m = long[long.PATNO.isin(cnt[cnt >= 2].index)]
    md = smf.mixedlm('factor ~ 1', m, groups=m.PATNO).fit(method="lbfgs")
    vb = float(md.cov_re.iloc[0, 0]); vw = float(md.scale)
    icc = vb / (vb + vw)
    log(f'\nWithin-subject ICC of the factor across visits: {icc:.3f} (n = {m.PATNO.nunique()} '
        f'subjects, {len(m)} observations)')
    # test-retest correlation between the first and second available visit
    first = m.sort_values("YEAR").groupby("PATNO").factor.first()
    second = m.sort_values("YEAR").groupby("PATNO").factor.nth(1)
    tr = pd.concat([first.rename("v1"), second.rename("v2")], axis=1).dropna()
    rho, pv = stats.spearmanr(tr.v1, tr.v2)
    log(f'correlation between first and second visit  : rho = {rho:.3f} (p = {pv:.3g}, n = '
        f'{len(tr)})')
    log('\nReading: a high ICC indicates that the factor is a stable property')
    log('of the subject, incompatible with a pure run effect; an ICC close')
    log('to zero would indicate the opposite.')
    pd.DataFrame([{'ICC_intra_subject': icc, 'n_subjects': m.PATNO.nunique(),
                   "rho_test_retest": rho, 'n_pairs': len(tr)}]).to_csv(
        os.path.join(OUT, "23_stability_component.csv"), index=False)

# ===========================================================================
# P2.18 — albumin against blood contamination
# ===========================================================================
log("\n" + "=" * 78)
log("P2.18 — IS THE ALBUMIN-KINETICS ASSOCIATION BLOOD CONTAMINATION?")
log("=" * 78)
a = pd.read_csv(ALB)
a = a[a.CLINICAL_EVENT.isin(["BL", "SC"])].copy()
a["v"] = pd.to_numeric(a.TESTVALUE, errors="coerce")
alb = a[a.TESTNAME == "CSF Albumin"].groupby("PATNO").v.mean().rename("alb_csf")
plq = a[a.TESTNAME == "Plasma Albumin"].groupby("PATNO").v.mean().rename("alb_pl")
A = pd.concat([alb, plq], axis=1).reset_index()
A["qalb"] = A.alb_csf / A.alb_pl

KIN = {"k_Fmax": "Fmax", "k_TTT": "TTT", "k_AUC": "AUC", "k_T50": "T50",
       "k_SLOPE": "SLOPE"}
D = d150[["PATNO"] + list(KIN)].rename(columns=KIN)
for c in KIN.values():
    D[c] = np.log(D[c])
D = D.merge(A, on="PATNO", how="left").merge(
    bl[["PATNO", "age_at_visit", "SEX", "hemohi"]].rename(
        columns={"age_at_visit": "age_bl"}), on="PATNO", how="left")


def partial(y, x, Z):
    Zc = sm.add_constant(Z)
    ry = sm.OLS(stats.rankdata(y), Zc).fit().resid
    rx = sm.OLS(stats.rankdata(x), Zc).fit().resid
    r, p = stats.pearsonr(ry, rx)
    n = len(y)
    z = np.arctanh(r); se = 1 / np.sqrt(n - Zc.shape[1] - 2)
    return r, p, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), n


rows = []
for k in KIN.values():
    base = D[[k, "alb_csf", "age_bl", "SEX"]].dropna()
    r0, p0, lo0, hi0, n0 = partial(base.alb_csf, base[k],
                                   base[["age_bl", "SEX"]].to_numpy(float))
    m = D[[k, "alb_csf", "age_bl", "SEX", "hemohi"]].dropna()
    r1, p1, lo1, hi1, n1 = partial(m.alb_csf, m[k],
                                   m[["age_bl", "SEX", "hemohi"]].to_numpy(float))
    q = D[[k, "qalb", "age_bl", "SEX"]].dropna()
    r2, p2, lo2, hi2, n2 = partial(q.qalb, q[k], q[["age_bl", "SEX"]].to_numpy(float))
    rows.append({'kinetic': k, "n_alb": n0,
                 'rho_albumin': r0, 'CI_inf': lo0, 'CI_upp': hi0, "p": p0,
                 'rho_agg_haemoglobin': r1, "p_agg": p1, "n_agg": n1,
                 "rho_Qalb": r2, "p_Qalb": p2, "n_Qalb": n2})
ab = pd.DataFrame(rows)
log(ab.round(4).to_string(index=False))
ab.to_csv(os.path.join(OUT, "23_albumin_haemoglobin.csv"), index=False)
log("\nIf the effect followed barrier permeability it should appear on")
log("Qalb; if it were local or due to collection, on CSF albumin but not on Qalb.")

# ===========================================================================
# P1.9 — confidence intervals on the negative controls
# ===========================================================================
log("\n" + "=" * 78)
log('P1.9 — CONFIDENCE INTERVALS ON THE NEGATIVE CONTROLS (§3.16)')
log("=" * 78)
log('The tests of nullity are to be replaced by intervals: a non-significant p')
log("is not evidence of absence if the interval admits relevant effects.")


def ci_rho(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


# The four coefficients are read from 20_component_vs_albumin.csv, which is the
# canonical home of the component/albumin comparison: they had been transcribed
# by hand from an earlier round and stayed the same even when the cohort changed.
_alb_neg = pd.read_csv(os.path.join(OUT, "20_component_vs_albumin.csv")
                       ).set_index('cohort')
_n24 = int(_alb_neg.loc["24h"].n)
for tag, co, col in [
        ('factor vs CSF albumin (150h)', "150h", 'rho_factor_vs_albumin_CSF'),
        ('factor vs CSF albumin (24h)', "24h", 'rho_factor_vs_albumin_CSF'),
        ('factor vs Qalb (150h)', "150h", 'rho_factor_vs_Qalb'),
        ('factor vs Qalb (24h)', "24h", 'rho_factor_vs_Qalb')]:
    r = float(_alb_neg.loc[co][col])
    n = int(_alb_neg.loc[co].n)
    lo, hi = ci_rho(r, n)
    log(f'  {tag:38s} rho = {r:+.3f}  n = {n}  95% CI ({lo:+.3f}; {hi:+.3f})')
log(f"\nWith n = {_n24} the interval is too wide to exclude anything: the")
log('negative control on the 24 h cohort is not informative and must be')
log('declared as such.')

with open(os.path.join(OUT, "23_review_P17_P19.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/23_review_P17_P19.txt')
