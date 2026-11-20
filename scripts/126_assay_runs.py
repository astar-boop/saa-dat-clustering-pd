"""
126 - The assay run: kinetic parameters by run, run against enrolment
period, centre and patient characteristics; run regime of the TTT.
(Reviewer 3, point 1.)

NUMBERING. Written on 14/11/2026 as 126_assay_runs.py, renumbered 126 the
same day at the request of the user because the session of Reviewer 2 has a
script 124 of its own (124_run_and_models_without_groups.py) and the two collided
on the number and on the resume file. Part 5 removed (silhouette with kinetic
parameters centred on the run date): that script 124 does it, more broadly, and
the provisional sentence of R2.3 waits for it. Code and readings of parts 1-4
unchanged.

Decision of the user of 14/11/2026 (revisione_R3_rapporti/PREPARAZIONE_R3.md,
decision 2). HONESTY ABOUT THE DESIGN: the readings below are written before
running this script but AFTER having seen unverified preliminary numbers
(revisione_R3_rapporti/preliminari_R3.txt: two regimes of TTT at 24 hours, ARI
0.009 with the seven-variable k-means) and the exploratory numbers of the
verifier of 123 (centring on the run date the departure disappeared). They are
not predictions.

The run is RUNDATE of the row of the assay file FROM WHICH the kinetic
parameters of the participant COME (recognised by identity of the values: mean
of the wells at 24 hours, median at 150 hours), not the first row of the file.
Script 69 (Table S6) took the first row: for 5 participants at 24 hours with
several rows at baseline the date could be a different one. Part 3 recomputes
Table S6 with the right row and compares it with 69.

PART 1 - Table by run: n, PPMI phase of enrolment, year of the baseline
  visit, storage (years between baseline visit and run), centres, medians of the
  log kinetic parameters. Both cohorts.
PART 2 - Figure: the four kinetic parameters of the matrix (analysis scale, log)
  by run date, one point per participant, colour and symbol by phase; run
  median.
PART 3 - Proportion of variance between runs (eta^2) against 5,000 permutations
  of the run labels, for: the four kinetic parameters (Table S6 redone), DAT
  putamen and caudate, age, sex, duration, UPDRS-III, MoCA. Whole cohort; at 24
  hours also within the second phase. Benjamini-Hochberg on the seven non-assay
  variables, by cohort and together.
  Reading: for the variables with corrected p >= 0.05 it is written that the run
  is not associated with that characteristic beyond chance; for the others
  eta^2 is reported next to that of the kinetic parameters. It is not written
  that the adjustment for run "toglie solo varianza tecnica".
PART 4 - Run regime of the TTT (24 hours). Rule fixed here: two-group k-means
  (1D) on the medians of log TTT of the runs with at least 5 participants; the
  smaller runs go to the nearest centre. Reported: the two
  medians, n, difference in log and in total SDs and d within regime; the regime
  within the second phase and within the centres (linear model with fixed
  effects of centre and phase); pairs of runs <= 7 days apart in different
  regimes; the regime and the weak reactors of 123 (log AUC < -2 SD; Fisher
  exact).
  Recovery: ARI between regime and (a) k-means at k = 2 on the seven variables
  (n_init 50, seed of 45), (b) Gaussian mixtures at k = 2, (c) k-means on
  log TTT alone.
  Readings: "tecnico" is written if the regime explains log TTT within the
  second phase with eta^2 >= 0.3 and if at least one pair of runs <= 7 days
  apart is in different regimes. "La partizione a sette variabili non lo
  ritrova" if the ARI of (a) and (b) is < 0.1.
Outputs (working directory run_sporadici): outputs/126_by_run.csv, 126_eta2.csv,
126_regime.csv, 126_assay_runs.txt, figure outputs/126_kinetics_by_run.png.
From 16/11/2026 also outputs/126_values_by_participant.csv (the values of the figure, one
per participant, without PATNO): export only, no computation changed.
PROVA=1: 200 permutations instead of 5,000, no final file.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util                                            # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
import warnings                                                  # noqa: E402
import zlib                                                      # noqa: E402
from _legacy import crc32_legacy                                # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import DATA                                       # noqa: E402

BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
TRIAL = os.environ.get('TRIAL', "0") == "1"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "2")))
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
CACHE = os.path.join(BASE, "outputs", "cache", "curated.parquet")
SHA_CACHE = "6a8958dea7a8611bc9c0dc19e7c294230fe0324a"


from _modules import load_script as _module                      # noqa: E402


M123 = _module("123_silhouette_departure_origin.py", 'origin_123')
M118 = M123.M118
import _rule_functions                                                    # noqa: E402
C45 = M118.C45

SEED = 20261116
N_PERM_AGE = 5000 if not TRIAL else 200
N_MIN = 5
NAMES = {"24h": ["log Fmax", "log TTT", "log AUC", "log SLOPEmax"],
        "150h": ["log Fmax", "log T50", "log AUC", "log SLOPE"]}
rep = []


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def std(M):
    return M118.standardise(M)


def eta2(y, g):
    y = np.asarray(y, float)
    g = np.asarray(g)
    ok = ~np.isnan(y)
    y, g = y[ok], g[ok]
    u, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv)
    s = np.bincount(inv, weights=y)
    tot = ((y - y.mean()) ** 2).sum()
    return float(((s ** 2 / n).sum() - y.sum() ** 2 / len(y)) / tot)


def eta2_perm(y, g, rng, nperm):
    y = np.asarray(y, float)
    g = np.asarray(g)
    ok = ~np.isnan(y)
    y, g = y[ok], g[ok]
    o = eta2(y, g)
    nul = np.array([eta2(y, rng.permutation(g)) for _ in range(nperm)])
    return o, float(np.median(nul)), float(((nul >= o - 1e-12).sum() + 1) / (nperm + 1)), int(ok.sum())


def collapse(run):
    vc = pd.Series(run).value_counts()
    return np.where(pd.Series(run).map(vc).to_numpy() >= N_MIN, run, 'others (<5)')


def centre(X, run):
    """Removes from the four kinetic parameters the mean of their run; re-standardises."""
    C = X.copy()
    g = collapse(run)
    for u in np.unique(g):
        m = g == u
        C[m, :4] -= C[m, :4].mean(0)
    return std(C)


# ------------------------------------------------------------------- data --
def runs(saa, cohort, proj, col_f, col_t, agg, rep_f, rep_t):
    """RUNDATE of the row whose wells give exactly the kinetic parameters of the participant."""
    s = saa[(saa.PROJECTID == proj) & (saa.CLINICAL_EVENT == "BL") & saa.PATNO.isin(cohort.PATNO)].copy()
    s["_f"] = getattr(s[rep_f], agg)(axis=1)
    s["_t"] = getattr(s[rep_t], agg)(axis=1)
    m = cohort[["PATNO", col_f, col_t]].merge(s[["PATNO", "_f", "_t", "RUNDATE", "InstrumentRep1"]], on="PATNO")
    m = m[np.isclose(m[col_f], m._f, rtol=1e-9) & np.isclose(m[col_t], m._t, rtol=1e-9)]
    assert m.PATNO.is_unique and len(m) == len(cohort), 'link between run and kinetic parameters not unique'
    rows_bl = s.groupby("PATNO").size()
    return m.set_index("PATNO").loc[cohort.PATNO, "RUNDATE"].to_numpy(), int((rows_bl > 1).sum())


def main():
    t0 = time.time()
    title('126 - THE ASSAY RUN')
    import hashlib
    assert hashlib.sha1(open(CACHE, "rb").read()).hexdigest() == SHA_CACHE, 'curated cache altered'
    assert ONLY, 'run with ONLY_SPORADIC=1 on the working directory run_sporadici'
    saa = pd.read_csv(SAA)
    cur = pd.read_parquet(CACHE)
    bl = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO").set_index("PATNO")

    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy())
    R150 = c150[M118.F150].to_numpy(float).copy()
    R150[:, :4] = np.log(R150[:, :4])
    X = {"24h": npz["F_primary"].astype(float), "150h": std(R150)}
    saved = np.load(os.path.join(OUT, "118_nulls.npz"), allow_pickle=True)
    for tag in X:
        assert np.abs(saved[f'V0_{tag}_matrix'] - X[tag]).max() < 1e-12
    cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv")).set_index("PATNO").loc[c24.PATNO].reset_index()
    d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
    d150 = d150[d150.complete].set_index("PATNO").loc[c150.PATNO].reset_index()
    r24, dop24 = runs(saa, cc, 237, "saa_Fmax", "saa_TTT", "mean",
                       [f"Fmax_24h_Rep{i}" for i in (1, 2, 3)], [f"TTT_24h_Rep{i}" for i in (1, 2, 3)])
    r150, dop150 = runs(saa, d150, 155, "k_Fmax", "k_T50", "median",
                         [f"Fmax_150h_Rep{i}" for i in (1, 2, 3)], [f"T50_150h_Rep{i}" for i in (1, 2, 3)])
    # the kinetic columns of the matrix are those of the cohort files (order check)
    assert np.allclose(std(np.log(cc[["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax"]].to_numpy())), X["24h"][:, :4], atol=1e-8)
    RUN = {"24h": r24.astype(str), "150h": r150.astype(str)}
    PAT = {"24h": c24.PATNO.to_numpy(), "150h": c150.PATNO.to_numpy()}
    log(f"n = {len(r24)} and {len(r150)}. Distinct runs: {len(set(r24))} and {len(set(r150))}. "
        f"Subjects with several rows at baseline in the project: {dop24} and {dop150}.")
    source = {"24h": cc, "150h": d150}

    info = {}
    for tag in X:
        b = bl.reindex(PAT[tag])
        vis = pd.to_datetime(b.visit_date.to_numpy())
        rd = pd.to_datetime(RUN[tag])
        info[tag] = pd.DataFrame({
            "PATNO": PAT[tag], 'run': RUN[tag], 'phase': b.enroll_phase.to_numpy(),
            'centre': b.SITE.astype(str).to_numpy(), 'year_visit': vis.year,
            'storage_years': (rd - vis).days / 365.25,
            'sex': pd.to_numeric(b.SEX, errors="coerce").to_numpy(),
            'duration': pd.to_numeric(source[tag].duration_yrs, errors="coerce").to_numpy(),
            "UPDRS-III": pd.to_numeric(b.updrs3_score, errors="coerce").to_numpy(),
            "MoCA": pd.to_numeric(b.moca, errors="coerce").to_numpy()})
        for i, nm in enumerate(NAMES[tag] + ["putamen", 'caudate', 'age']):
            info[tag][nm] = X[tag][:, i]

    # -------------------------------------------------------------- PART 1 --
    title('1. BY RUN')
    tabs = []
    for tag in X:
        d = info[tag]
        t = d.groupby('run').agg(n=("PATNO", "size"),
                                  phase1=('phase', lambda x: int((x == 1).sum())),
                                  phase2=('phase', lambda x: int((x == 2).sum())),
                                  year_visit_median=('year_visit', "median"),
                                  storage_median=('storage_years', "median"),
                                  centres=('centre', "nunique"))
        raw = source[tag]
        cols = {"24h": ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax"],
                "150h": ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE"]}[tag]
        for nm, c in zip(NAMES[tag], cols):
            t[f'median {nm}'] = pd.Series(np.log(raw[c].to_numpy()), index=d.index).groupby(d.run).median()
        t.insert(0, 'cohort', tag)
        tabs.append(t.reset_index())
        log(f'\n{tag}: {len(t)} runs, from {d.run.min()} to {d.run.max()}; phase 1 '
            f'{int((d.phase == 1).sum())}, phase 2 {int((d.phase == 2).sum())}; median storage '
            f'{d.storage_years.median():.2f} years (IQR '
            f'{d.storage_years.quantile(0.25):.2f}-{d.storage_years.quantile(0.75):.2f}).')
        log(t.round(2).to_string())
        r = stats.spearmanr(pd.to_datetime(d.run).map(pd.Timestamp.toordinal),
                            pd.to_datetime(bl.reindex(PAT[tag]).visit_date).map(pd.Timestamp.toordinal))
        log(f'Spearman run date ~ date of the baseline visit: {r.statistic:.3f}')
    T1 = pd.concat(tabs)

    # -------------------------------------------------------------- PART 2 --
    title('2. FIGURE')
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {1: "#eb6834", 2: "#2a78d6"}
    MRK = {1: "s", 2: "o"}
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    rng_j = np.random.default_rng(SEED)
    raws = {"24h": ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax"], "150h": ["k_Fmax", "k_T50", "k_AUC", "k_SLOPE"]}
    for r_, tag in enumerate(X):
        d = info[tag]
        ordering = sorted(d.run.unique())
        pos = d.run.map({c: i for i, c in enumerate(ordering)}).to_numpy()
        for c_, (nm, col) in enumerate(zip(NAMES[tag], raws[tag])):
            a = ax[r_, c_]
            y = np.log(source[tag][col].to_numpy())
            for f in (1, 2):
                m = d.phase.to_numpy() == f
                a.scatter(pos[m] + rng_j.uniform(-.25, .25, m.sum()), y[m], s=9, marker=MRK[f], color=COL[f],
                          alpha=.55, linewidths=0, label=f"PPMI phase {f} (n = {m.sum()})" if c_ == 0 else None)
            med = pd.Series(y).groupby(pos).median()
            a.hlines(med.values, med.index - .35, med.index + .35, color="#0b0b0b", lw=2)
            a.set_xticks(range(len(ordering)))
            a.set_xticklabels([f'{c} (n={int((d.run == c).sum())})' for c in ordering], rotation=90, fontsize=6)
            a.set_title(f"{'24 h' if tag == '24h' else '150 h'} cohort: {nm}", fontsize=9)
            a.grid(axis="y", color="#e6e5e0", lw=.6)
            a.spines[["top", "right"]].set_visible(False)
        ax[r_, 0].legend(fontsize=7, frameon=False, loc="lower left")
        ax[r_, 0].set_ylabel("natural log of the parameter")
    fig.suptitle("Kinetic parameters by assay run date (black bar: run median)", fontsize=11)
    fig.tight_layout()
    fpng = os.path.join(OUT, "126_kinetics_by_run.png" if not TRIAL else "126_TRIAL.png")
    fig.savefig(fpng, dpi=200)
    plt.close(fig)
    log(f'Figure: {fpng}')

    # Export of the plotted values (added on 16/11/2026, at the request of the user):
    # one row per participant, with the correct run, the enrolment phase and the natural
    # logarithm of the four parameters. It serves to redraw Figure S8 without re-reading
    # the raw files. It does not contain the PATNO, does not use random numbers and does
    # not touch any computation or the report: it only writes one more file.
    if not TRIAL:
        _exp = []
        for tag in X:
            _d = info[tag]
            _e = pd.DataFrame({'cohort': tag, 'run': _d.run.to_numpy(), 'phase': _d.phase.to_numpy()})
            for nm, col in zip(NAMES[tag], raws[tag]):
                _e[nm] = np.log(source[tag][col].to_numpy())
            _exp.append(_e)
        _fexp = os.path.join(OUT, "126_values_by_participant.csv")
        pd.concat(_exp).to_csv(_fexp, index=False)
        print(f'Values per participant exported: {_fexp}')

    # -------------------------------------------------------------- PART 3 --
    title('3. SHARE OF VARIANCE BETWEEN RUNS (eta^2) AGAINST PERMUTATION')
    s69 = pd.read_csv(os.path.join(OUT, "69_run_variance_components.csv"))
    rows = []
    rest = ["putamen", 'caudate', 'age', 'sex', 'duration', "UPDRS-III", "MoCA"]
    for tag in X:
        for sub, m in (('whole cohort', np.ones(len(info[tag]), bool)),
                       ('phase 2 only', info[tag].phase.to_numpy() == 2)):
            if tag == "150h" and sub != 'whole cohort':
                continue
            d = info[tag][m]
            rng = np.random.default_rng([SEED, crc32_legacy((tag + sub).encode())])
            for v in NAMES[tag] + rest + ['year_visit', 'storage_years']:
                o, mn, p, n = eta2_perm(d[v].to_numpy(float), d.run.to_numpy(), rng, N_PERM_AGE)
                rows.append(dict(cohort=tag, dataset=sub, variable=v, n=n, runs=d.run.nunique(),
                                  eta2=o, eta2_null_median=mn, p_perm=p,
                                  family='not of the assay' if v in rest else 'assay/period'))
    E3 = pd.DataFrame(rows)
    for (tag, sub), g in E3.groupby(['cohort', 'dataset']):
        mm = g.family == 'not of the assay'
        E3.loc[g.index[mm.to_numpy()], "p_BH_7"] = _rule_functions.bh(g[mm].p_perm.to_numpy())
    log(E3.round(4).to_string(index=False))
    log('\nComparison with Table S6 (script 69, first row of the file):')
    for _, r in s69.iterrows():
        tag = r.cohort.split()[0]
        nm = r.parameter.replace("SLOPEMax", "SLOPEmax")
        q = E3[(E3.cohort == tag) & (E3.dataset == 'whole cohort') & (E3.variable == nm)]
        if len(q):
            log(f'  {tag} {nm}: 69 {r.eta2_run:.4f} ({int(r.n_run)} runs) -> 126 '
                f'{q.eta2.iloc[0]:.4f} ({int(q.runs.iloc[0])} runs)')

    # -------------------------------------------------------------- PART 4 --
    title('4. RUN REGIME OF THE TTT (24 H)')
    d = info["24h"].copy()
    lt = np.log(cc.saa_TTT.to_numpy())
    d["logTTT"] = lt
    medc = d.groupby('run').logTTT.median()
    nc = d.groupby('run').size()
    large = medc[nc >= N_MIN]
    km = C45.KMeans(2, n_init=50, random_state=C45.SEED).fit(large.to_numpy().reshape(-1, 1))
    centres = km.cluster_centers_.ravel()
    slow = int(np.argmax(centres))
    reg_run = {c: int(np.argmin(np.abs(centres - v)) == slow) for c, v in medc.items()}
    d['slow'] = d.run.map(reg_run).astype(int)
    a, b = lt[d.slow == 1], lt[d.slow == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    log(f'Centres (run medians, log hours): {np.sort(centres).round(3)}; slow runs '
        f'{sum(reg_run.values())} of {len(reg_run)}; subjects {len(a)} slow, {len(b)} fast.')
    log(f'Difference in log TTT {a.mean() - b.mean():.3f} = '
        f'{(a.mean() - b.mean()) / lt.std(ddof=1):.2f} total SD; d within regime '
        f'{(a.mean() - b.mean()) / sp:.2f}; median TTT {np.exp(np.median(a)):.2f} against '
        f'{np.exp(np.median(b)):.2f} hours.')
    m2 = d.phase == 2
    e2 = eta2(lt[m2], d.slow[m2])
    log(f'Within the second phase (n = {m2.sum()}): eta^2 of the regime on log TTT {e2:.3f}.')
    import statsmodels.formula.api as smf
    fit = smf.ols('logTTT ~ slow + C(centre) + C(phase)', data=d).fit()
    log(f"With fixed effects of centre and phase: coefficient of the regime {fit.params['slow']:.3f} "
        f"(95% CI {fit.conf_int().loc['slow', 0]:.3f} to {fit.conf_int().loc['slow', 1]:.3f}).")
    date = pd.Series(pd.to_datetime(list(reg_run)), index=list(reg_run)).sort_values()
    pairs = []
    for i, c1 in enumerate(date.index):
        for c2 in date.index[i + 1:]:
            if (date[c2] - date[c1]).days <= 7 and reg_run[c1] != reg_run[c2]:
                pairs.append((c1, c2, int(nc[c1]), int(nc[c2]), round(medc[c1], 2), round(medc[c2], 2)))
    log(f'Pairs of runs <= 7 days apart in different regimes: {len(pairs)}')
    for c in pairs:
        log(f'  {c[0]} (n={c[2]}, median {c[4]}) / {c[1]} (n={c[3]}, median {c[5]})')
    zauc = X["24h"][:, 2]
    weak = zauc < -2.0
    tab = pd.crosstab(weak, d.slow)
    orat, pf = stats.fisher_exact(tab.to_numpy())
    log(f'Weak reactors (log AUC < -2 SD): {weak.sum()}, of which in slow runs '
        f'{int((weak & (d.slow == 1)).sum())}; Fisher exact p = {pf:.2g}.')
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.mixture import GaussianMixture
    g7 = C45.KMeans(2, n_init=50, random_state=C45.SEED).fit_predict(X["24h"])
    gg = GaussianMixture(2, n_init=10, random_state=C45.SEED).fit(X["24h"]).predict(X["24h"])
    g1 = C45.KMeans(2, n_init=50, random_state=C45.SEED).fit_predict(X["24h"][:, [1]])
    rec = dict(ARI_kmeans_7=ari(d.slow, g7), ARI_gmm_7=ari(d.slow, gg), ARI_kmeans_TTT=ari(d.slow, g1))
    log('Recovery: ' + "; ".join(f"{k} {v:.3f}" for k, v in rec.items()))
    log(f'  seven-variable k-means against regime: {pd.crosstab(d.slow, g7).to_numpy().tolist()}')
    technical = (e2 >= 0.3) and len(pairs) >= 1
    no_recovers = rec["ARI_kmeans_7"] < 0.1 and rec["ARI_gmm_7"] < 0.1
    log(f'READING: technical = {technical}; the seven-variable partition does not recover it = '
        f'{no_recovers}.')
    T4 = pd.DataFrame([dict(runs_slow=sum(reg_run.values()), runs=len(reg_run), n_slowones=len(a),
                            n_fast=len(b), diff_log=a.mean() - b.mean(),
                            diff_SD_totals=(a.mean() - b.mean()) / lt.std(ddof=1), d_within=(a.mean() - b.mean()) / sp,
                            eta2_phase2=e2, coef_centre_phase=fit.params['slow'], pairs_7gg=len(pairs),
                            weak=int(weak.sum()), weak_slowones=int((weak & (d.slow == 1)).sum()),
                            fisher_p=pf, technical=technical, no_recovers=no_recovers, **rec)])

    log(f'\nTotal time: {time.time() - t0:.0f} s')
    if not TRIAL:
        T1.to_csv(os.path.join(OUT, "126_by_run.csv"), index=False)
        E3.to_csv(os.path.join(OUT, "126_eta2.csv"), index=False)
        T4.to_csv(os.path.join(OUT, "126_regime.csv"), index=False)
        with open(os.path.join(OUT, "126_assay_runs.txt"), "w") as fh:
            fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
