"""
85 - The association between assay run and centre, under a single pooling
     rule, for the two cohorts.

A previous version of the manuscript reported two different values for the
same quantity: V = 0.333 in section 3.10 (from outputs/71_centre_effects.csv)
and V = 0.342 with the bias correction 0.222 in section 3.12 (from
outputs/77_fmax_dat.txt). The review had noticed this; the current text
reports only the symmetric rule recomputed here (0.342 and 0.222).

The two values come neither from different cohorts nor from transcription
errors: they are the same table computed with two different pooling rules.

  71_centre_effects.py  pools ONLY the centres with fewer than 5 participants
                        into "altro" (lines 75-76) and leaves the run at its
                        raw granularity, that is, all the distinct values of
                        RUNDATE.
  77_fmax_dat_batch.py  pools BOTH margins at the same threshold N_MIN = 5
                        (lines 80, 131-136), because that is how run and centre
                        enter the adjusted models.

Here everything is recomputed under ONE declared rule: symmetric pooling of the
levels with fewer than 5 participants on both margins, in each cohort
separately. Reason: the chi-square and Cramer's V are biased upwards
when many cells have expected frequencies below 5, and pooling a single
margin lets the almost empty levels of the other one inflate the
statistic; the symmetric threshold at 5 is the conventional rule on the minimum
expected frequency and is the same one at which run and centre enter the
adjusted longitudinal models, so that the association reported is the
association in the parametrisation actually used for the adjustment.

The values under the other two rules (centres only pooled; no pooling) are
also reported, to make the difference between the two numbers of the
manuscript reconstructible.

Output: outputs/85_run_centre.csv, outputs/85_run_centre.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fit import OPTIMISERS
from _paths import DATA
from _assay_run import row_of_kinetics

warnings.filterwarnings("ignore")
OUTCOMES_DAT = {"MIA_STRIATUM_BILAT": 'DAT striatum, bilateral',
                "MIA_PUTAMEN_BILAT": 'DAT putamen, bilateral'}

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SAA_FILE = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
N_MIN = 5           # pooling threshold, identical on the two margins
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def collapse(s, threshold):
    """Levels with fewer than `threshold` observations into a single category."""
    s = s.astype(str)
    v = s.value_counts()
    return s.where(s.map(v) >= threshold, 'other')


def cramer(tab, n):
    """Cramer's V and bias-corrected V (Bergsma 2013).

    V   = sqrt( phi2 / (min(r,c) - 1) ),                phi2 = chi2 / n
    V~  = sqrt( phi2~ / (min(r~, c~) - 1) )
          phi2~ = max(0, phi2 - (r-1)(c-1)/(n-1))
          r~    = r - (r-1)^2/(n-1);   c~ = c - (c-1)^2/(n-1)
    """
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    r, c = tab.shape
    phi2 = chi2 / n
    V = float(np.sqrt(phi2 / (min(r, c) - 1)))
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc_ = c - (c - 1) ** 2 / (n - 1)
    Vc = float(np.sqrt(phi2c / min(rc - 1, cc_ - 1)))
    return float(chi2), int(dof), float(p), V, Vc


def containment(tab):
    """Concentration of each margin in the other.

    Returns (proportion per centre, proportion per run): for each centre the
    proportion of its participants that comes from the most frequent run of that
    centre, and vice versa.
    """
    q_centre = tab.max(axis=0) / tab.sum(axis=0)      # columns = centres
    q_run = tab.max(axis=1) / tab.sum(axis=1)         # rows = runs
    return q_centre, q_run


def mmm(x):
    return float(x.median()), float(x.min()), float(x.max())


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
site_bl = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO")[["PATNO", "SITE"]]
saa = pd.read_csv(SAA_FILE)

cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv"))
cc = cc[cc.complete_primary]
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete]

COHORTS = [('primary 24h', 237, cc), ('replication 150h', 155, d150)]

log("=" * 78)
log('85 — ASSAY RUN x CENTRE, UNDER A SINGLE MERGING RULE')
log("=" * 78)
log("\nDeclared rule: in each cohort, the runs with fewer than 5 subjects are\nmerged into a single "
    "residual category and the centres with fewer than 5 subjects into\nanother residual category — "
    "the same threshold on the two margins. The chi-square and\nthe Cramer V are biased upwards "
    "when many cells have expected\nfrequencies below 5, and merging one margin only lets the "
    "nearly empty\nlevels of the other inflate the statistic; the symmetric threshold of 5 is "
    "the\nconventional rule on the minimum expected frequency and is the same at which run "
    "and\ncentre enter the adjusted longitudinal models.")

rows = []
tables = {}
for lab, proj, origin in COHORTS:
    pats = set(origin.PATNO)
    s = saa[(saa.PROJECTID == proj) & (saa.CLINICAL_EVENT == "BL")
            & (saa.PATNO.isin(pats))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
    s = s[["PATNO", "RUNDATE"]].rename(columns={"RUNDATE": "run_raw"})
    m = origin[["PATNO"]].merge(s, on="PATNO", how="inner").merge(
        site_bl, on="PATNO", how="inner")
    m = m[m.SITE.notna() & m.run_raw.notna()]
    n = len(m)

    log("\n" + "=" * 78)
    log(f'COHORT {lab.upper()}  ({n} subjects)')
    log("=" * 78)
    log(f'subjects in the source: {len(origin)}; with run and centre: {n}')

    vr = m.run_raw.astype(str).value_counts()
    vs = m.SITE.astype(str).value_counts()
    log(f"distinct runs before merging:    {int(vr.size)} (with at least {N_MIN} subjects: "
        f"{int((vr >= N_MIN).sum())}, covering {int(vr[vr >= N_MIN].sum())} subjects; merged: "
        f"{int((vr < N_MIN).sum())} runs, {int(vr[vr < N_MIN].sum())} subjects)")
    log(f"distinct centres before merging: {int(vs.size)} (with at least {N_MIN} subjects: "
        f"{int((vs >= N_MIN).sum())}, covering {int(vs[vs >= N_MIN].sum())} subjects; merged: "
        f"{int((vs < N_MIN).sum())} centres, {int(vs[vs < N_MIN].sum())} subjects)")

    m["run"] = collapse(m.run_raw, N_MIN)
    m["site"] = collapse(m.SITE, N_MIN)
    tab = pd.crosstab(m.run, m.site)
    tables[lab] = tab
    n_run, n_site = tab.shape
    log(f"\ntable after merging: {n_run} run categories x {n_site} centre categories (threshold "
        f"{N_MIN})")

    chi2, dof, p, V, Vc = cramer(tab, n)
    att = stats.chi2_contingency(tab)[3]
    log(f'chi2 = {chi2:.1f}, df = {dof}, p = {p:.3g}')
    log(f'Cramer V = {V:.4f}; bias-corrected (Bergsma) = {Vc:.4f}')
    log(f'cells with expected frequency < 5: {int((att < 5).sum())} out of {att.size} '
        f'({100 * (att < 5).mean():.1f}%); minimum expected {att.min():.3f}')

    # The table stays sparse even after pooling: the threshold acts on the
    # marginal totals, not on the expected cell frequencies, and with 595
    # participants over 882 cells the asymptotic approximation of the chi-square is
    # not guaranteed. A permutation p, which does not require it, is added alongside.
    rng = np.random.default_rng(20260812)
    B = 5000
    rv = m.run.to_numpy()
    sv = m.site.to_numpy()
    nulls = np.empty(B)
    for b_ in range(B):
        nulls[b_] = stats.chi2_contingency(
            pd.crosstab(pd.Series(rng.permutation(rv)), pd.Series(sv)))[0]
    p_perm = float((1 + int((nulls >= chi2).sum())) / (B + 1))
    log(f'permutation p over {B} shuffles of the run label: {p_perm:.4g} (null chi2: median '
        f'{np.median(nulls):.1f}, 99th centile {np.percentile(nulls, 99):.1f}, maximum '
        f'{nulls.max():.1f})')
    V_null = np.sqrt((nulls / n) / (min(tab.shape) - 1))
    log(f'Cramer V expected under independence, at these table dimensions: median '
        f'{np.median(V_null):.4f} (range {V_null.min():.4f}-{V_null.max():.4f}) — this is why the '
        f'corrected V is also reported')

    q_centre, q_run = containment(tab)
    a, b, c = mmm(q_centre)
    log(f"\nshare of the subjects of a centre who come from the most frequent run of that centre: "
        f"median {a:.3f} (range {b:.3f}-{c:.3f})")
    a2, b2, c2 = mmm(q_run)
    log(f"share of the subjects of a run who come from the most frequent centre of that run:    "
        f"median {a2:.3f} (range {b2:.3f}-{c2:.3f})")
    n_c1 = int((q_centre == 1).sum())
    n_r1 = int((q_run == 1).sum())
    log(f'\ncentres entirely contained in a single run: {n_c1} out of {n_site}')
    log(f'runs entirely contained in a single centre: {n_r1} out of {n_run}')

    # the same counts without the residual category, which is artificial
    t2 = tab.drop(index=[i for i in tab.index if i == 'other'],
                  columns=[c_ for c_ in tab.columns if c_ == 'other'])
    q_c2, q_r2 = containment(t2)
    log(f'  (excluding the residual category from both margins: {int((q_c2 == 1).sum())} centres out '
        f'of {t2.shape[1]}, {int((q_r2 == 1).sum())} runs out of {t2.shape[0]})')

    # and on the raw table, without any pooling
    tg = pd.crosstab(m.run_raw.astype(str), m.SITE.astype(str))
    q_cg, q_rg = containment(tg)
    log(f'  (without any merging, {tg.shape[0]} runs x {tg.shape[1]} centres: '
        f'{int((q_cg == 1).sum())} centres out of {tg.shape[1]} in a single run, '
        f'{int((q_rg == 1).sum())} runs out of {tg.shape[0]} in a single centre)')

    row = {'cohort': lab, "n": n, 'threshold_pooling': N_MIN,
            'n_run_raws': int(vr.size), 'n_centres_raws': int(vs.size),
            'n_run_merged': n_run, 'n_centres_merged': n_site,
            'run_below_threshold': int((vr < N_MIN).sum()),
            'subj_run_below_threshold': int(vr[vr < N_MIN].sum()),
            'centres_below_threshold': int((vs < N_MIN).sum()),
            'subj_centres_below_threshold': int(vs[vs < N_MIN].sum()),
            "chi2": chi2, 'dof': dof, "p": p, 'p_permutation': p_perm,
            'n_permutations': B,
            "V_cramer": V, 'V_cramer_corrected': Vc,
            'V_cramer_null_median': float(np.median(V_null)),
            'V_cramer_null_max': float(V_null.max()),
            'share_run_in_centre_median': a,
            'share_run_in_centre_min': b, 'share_run_in_centre_max': c,
            'share_centre_in_run_median': a2,
            'share_centre_in_run_min': b2, 'share_centre_in_run_max': c2,
            'centres_in_an_only_run': n_c1,
            'run_in_an_only_centre': n_r1,
            'centres_in_an_only_run_without_residual': int((q_c2 == 1).sum()),
            'run_in_an_only_centre_without_residual': int((q_r2 == 1).sum()),
            'cells_expected_below_5': int((att < 5).sum()),
            'cells_totals': int(att.size)}

    # ------------------------------------------------------------------
    # the other two rules, to reconstruct the two numbers of the manuscript
    # ------------------------------------------------------------------
    log('\nthe same table under the other merging rules:')
    alt = [('pooled centres only (rule of 71)',
            m.run_raw.astype(str), m.site),
           ('no merging',
            m.run_raw.astype(str), m.SITE.astype(str)),
           ('merged runs only', m.run, m.SITE.astype(str))]
    for name, rr, ss in alt:
        t = pd.crosstab(rr, ss)
        c2_, g2, p2, V2, Vc2 = cramer(t, n)
        log(f'  {name:38s} {t.shape[0]:3d} runs x {t.shape[1]:3d} centres  chi2 = {c2_:7.1f}  df = '
            f'{g2:4d}  p = {p2:9.3g}  V = {V2:.4f}  corrected V = {Vc2:.4f}')
        key = (name.split(" (")[0].replace(" ", "_"))
        row[f'V_{key}'] = V2
        row[f'Vcorr_{key}'] = Vc2
        row[f'chi2_{key}'] = c2_
        row[f'dof_{key}'] = g2
    rows.append(row)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "85_run_centre.csv"), index=False)

# ---------------------------------------------------------------------------
# how the degrees of freedom of the run x time test of 3.12 are reconstructed
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("THE 20 DEGREES OF FREEDOM OF THE RUN x TIME TEST ON THE PUTAMEN (section 3.12)")
log("=" * 78)
tab24 = tables['primary 24h']
log(f"""\nThe test comes from 77_fmax_dat_batch.py, lines 356-393. The run is RUNDATE of the\nraw "
    f"file SAA_Biospecimen_Analysis_Results_24Apr2026.csv, restricted to\nPROJECTID 237 and "
    f"CLINICAL_EVENT == "BL", one row per subject\n(drop_duplicates on PATNO). The "
    f"{tab24.shape[0] - 1} runs with at least {N_MIN} samples stay\ndistinct, the remaining ones are "
    f"merged into a single residual category: in all\n{tab24.shape[0]} categories. The run enters as "
    f"a categorical factor with treatment\ncoding, C(run), in {tab24.shape[0]} - 1 = "
    f"{tab24.shape[0] - 1} indicators.\n\nThe model is MIA_PUTAMEN_BILAT ~ C(run)*years with random "
    f"intercept and slope\nper subject (re_formula="~years"), fitted by maximum likelihood.\nThe "
    f"test is a joint Wald on the interaction terms C(run)[T.*]:years alone,\nwhich are "
    f"{tab24.shape[0] - 1}: hence the {tab24.shape[0] - 1} degrees of freedom.\nThe main effect of "
    f"the run has the same number of degrees of freedom.""")

sud = pd.read_csv(os.path.join(OUT, "77_run_on_dat.csv"))
log('\nvalues currently in outputs/77_run_on_dat.csv:')
for _, r_ in sud.iterrows():
    log(f'  {r_.label:26s} run x years: chi2 = {r_.run_x_years_chi2:.2f}, df = '
        f'{int(r_.run_x_years_dof)}, p = {r_.run_x_years_p:.4g}   |   main effect: chi2 = '
        f'{r_.run_level_chi2:.2f}, df = {int(r_.run_level_dof)}, p = {r_.run_level_p:.4g}')

# ---------------------------------------------------------------------------
# the chi-square of the manuscript is not that of the current outputs
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("WHY SECTION 3.12 REPORTS 43.3 AND 29.2 AND NOT 36.3 AND 7.1")
log("=" * 78)
log('\nAn earlier version of this work reported in section 3.12\n"run x time chi2 = 36.3, 20 df, p '
    '= 0.014" for the putamen and\n"chi2 = 7.1, 20 df, p = 0.996" for the striatum. The outputs '
    'give 43.34\n(p = 0.0018) and 29.20 (p = 0.084), and these are the values that section '
    '3.12\nreports today; the two earlier ones still appear there, but declared for what\nthey are, '
    'i.e. the values of the first optimiser to converge. The degrees of\nfreedom coincide: what '
    'changes is not the parametrisation of the run, but the\npoint the optimiser arrives at. '
    '77_fmax_dat_batch.py uses `fit_ml`, which\namong the converging optimisers keeps the one of '
    'maximum likelihood,\nwhereas `lbfgs` on this model stops at a markedly lower\nlikelihood. '
    'Below, the same quantity computed with each\noptimiser, to make this verifiable.')

L = cur[cur.PATNO.isin(set(cc.PATNO))][
    ["PATNO", "YEAR"] + list(OUTCOMES_DAT)].copy()
L["years"] = pd.to_numeric(L.YEAR, errors="coerce")
L = L[L.years.notna()]
s237 = saa[(saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")
           & (saa.PATNO.isin(set(cc.PATNO)))].pipe(row_of_kinetics, OUT).drop_duplicates("PATNO")
s237 = s237[["PATNO", "RUNDATE"]].rename(columns={"RUNDATE": "run_raw"})
b237 = cc[["PATNO"]].merge(s237, on="PATNO", how="left")
b237["run"] = collapse(b237.run_raw, N_MIN)
L = L.merge(b237[["PATNO", "run"]], on="PATNO", how="left")


def wald(res, key):
    names = [n for n in res.fe_params.index if key(n)]
    b = res.fe_params[names].to_numpy()
    cov = res.cov_params()
    if isinstance(cov, pd.DataFrame):
        Vm = cov.loc[names, names].to_numpy()
    else:
        idx = [list(res.fe_params.index).index(n) for n in names]
        Vm = np.asarray(cov)[np.ix_(idx, idx)]
    c2_ = float(b @ np.linalg.pinv(Vm) @ b)
    return c2_, float(stats.chi2.sf(c2_, len(names))), len(names)


rows_opt = []
for out_, lab_ in OUTCOMES_DAT.items():
    m = L[["PATNO", "years", out_, "run"]].dropna()
    log(f'\n{lab_} ({out_}): {m.PATNO.nunique()} subjects, {len(m)} observations, {m.run.nunique()} '
        f'run categories')
    for method, it in OPTIMISERS:
        try:
            r_ = smf.mixedlm(f"{out_} ~ C(run)*years", m, groups=m.PATNO,
                             re_formula="~years").fit(method=method, maxiter=it)
        except Exception:
            log(f'  {method:7s}: exception')
            continue
        ci, pi, gi = wald(r_, lambda n: "C(run)" in n and ":" in n)
        cm, pm, gm = wald(r_, lambda n: n.startswith("C(run)") and ":" not in n)
        log(f'  {method:7s} conv = {str(r_.converged):5s}  logL = {r_.llf:9.3f}  run x years: chi2 = '
            f'{ci:6.2f}, df = {gi}, p = {pi:.4g}  |  level: chi2 = {cm:6.2f}, df = {gm}, p = '
            f'{pm:.4g}')
        rows_opt.append({"outcome": out_, "label": lab_, "optimiser": method,
                          "converged": bool(r_.converged), "llf": float(r_.llf),
                          'run_x_years_chi2': ci, 'run_x_years_dof': gi,
                          'run_x_years_p': pi, 'run_level_chi2': cm,
                          'run_level_dof': gm, 'run_level_p': pm})
pd.DataFrame(rows_opt).to_csv(
    os.path.join(OUT, "85_run_x_time_by_optimiser.csv"), index=False)

log("\n" + "=" * 78)
log('CONTINGENCY TABLES (rows = run, columns = centre)')
log("=" * 78)
for lab, tab in tables.items():
    log(f"\n{lab}:")
    log(tab.to_string())

with open(os.path.join(OUT, "85_run_centre.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nWritten: outputs/85_run_centre.csv, outputs/85_run_centre.txt')
