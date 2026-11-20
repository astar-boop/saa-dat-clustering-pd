"""
127 - Numbers of the letter to Reviewer 3 that so far came only from the verifiers.

Request of the user of 15/11/2026 («risolviamo i problemi dal secondo al quarto»):
bring into a script of the project the numbers that the letter quotes and that had been
computed only by the independent verifiers of 125 and 126, and check the
«ICC centre» column of Table S6, never verified.

Parts
  A. Assay run of each participant: the row of the SAA file whose wells
     reproduce the kinetic parameters analysed (as in 126); check by identity with the
     counts per run of 126.
  B. Table S6 redone in full, column by column:
       eta^2 of run, ICC of run, eta^2 of instrument, eta^2 and ICC of centre.
     The ICC is the proportion of between-group variance estimated by REML in a
     random-intercept model. Primary estimator: EXACT profiled REML likelihood
     (closed form for the one-way model, maximised over lambda = var_g/var_e
     >= 0, boundary included). Check: statsmodels MixedLM with four optimisers;
     the best one must coincide with the profile (|ICC| < 1e-4), otherwise the script
     stops. The ICC of lbfgs alone (that of 69 and 71) and its loss of
     log-likelihood are also reported: this is how one sees whether a submitted value
     was a missed optimum.
     Two definitions of the run: that of 69 (first row of the file, 25 runs at 24 h),
     to reproduce the submitted values, and the correct one (A).
  C. Run x centre (bias-corrected Cramer's V, centres or runs that fall entirely in
     a single category) with the same pooling rule as 85, on the correct
     run.
  D. Run regime of the TTT at 24 h (rule of 126, reproduced and compared by
     identity): coefficient with fixed effects of centre and phase, ordinary 95% CI and
     with standard errors robust by run (cluster), because the regime is assigned
     by run and the participants of a run are not independent.
  E. Check of the mechanism of 125 (groups along a single variable). Cells tempo and
     dat, Delta 3, pi 0.20; replicates 0-19 of 125, by cohort. The injected data sets
     are rebuilt with the seeds of 125 and compared by identity with 125_outcomes.csv
     (observed silhouettes and observed criteria). Then the same rule with a
     different reference, R1: rank dependence of the injected data set (as in 125),
     but marginals taken from the matrix BEFORE the injection. The outcomes with the
     reference of 125 (R0) are read from 125_outcomes.csv. R1 uses knowledge
     of the injection: it is a diagnostic of the mechanism, not a test applicable to
     the real data, and its level in the absence of groups is not estimated here.
     B = 299 (silhouette) and 99 (k = 2), as in 125.

Readings fixed before running:
  - B: a submitted ICC is "sbagliato" if the REML profile exceeds by more than 0.01 in
    log-likelihood the point of lbfgs and the ICC differs by at least 0.005.
  - E: the mechanism of the copula is supported for a direction if the joint
    rule with R1 is satisfied in a larger number of replicates than with R0 with
    two-sided Fisher exact p < 0.05 (20 against 20 replicates).

Execution (from the root of the project):
  CLUSTERING_BASE="$PWD/run_sporadici" ONLY_SPORADIC=1 N_PROCESSES=8 \
      python3 -B scripts/127_reviewer3_letter_numbers.py
  ONLY_TALLIES=1 skips part E (a few seconds).
Output: run_sporadici/outputs/127_reviewer3_letter_numbers.txt, 127_table_S6.csv,
127_run_centre.csv, 127_mechanism.csv.
"""

import hashlib
import importlib.util
import multiprocessing as mp
import os
import sys
import time
import warnings
import zlib
from _legacy import crc32_legacy                                # noqa: E402

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize, stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import DATA                                       # noqa: E402

BASE = os.environ.get("CLUSTERING_BASE", os.path.dirname(HERE))
OUT = os.path.join(BASE, "outputs")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"
ONLY_TALLIES = os.environ.get('ONLY_TALLIES', "0") == "1"
N_PROC = max(1, int(os.environ.get('N_PROCESSES', "8")))
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
CACHE = os.path.join(BASE, "outputs", "cache", "curated.parquet")
SHA_CACHE = "6a8958dea7a8611bc9c0dc19e7c294230fe0324a"
N_MIN = 5
SEED = 20261117
N_REP_E = 20
B_SIL, B_K2 = 299, 99
ALPHA = 0.05
rep = []


from _modules import load_script as _module                      # noqa: E402


M125 = _module("125_rule_power.py", 'power_125')
M118, _rule_functions, C45 = M125.M118, M125._rule_functions, M125.C45
K_CURVE = M125.K_CURVE
CRITERIA = M125.CRITERIA


def log(s=""):
    print(s, flush=True)
    rep.append(str(s))


def title(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


# ----------------------------------------------------------- estimators --
def eta2(y, g):
    g = np.asarray(g)
    tot = np.sum((y - y.mean()) ** 2)
    between = sum(len(y[g == u]) * (y[g == u].mean() - y.mean()) ** 2 for u in np.unique(g))
    return float(between / tot)


def reml_profile(y, g):
    """Profiled REML log-likelihood of the model y = mu + b_g + e, as a function of lambda."""
    g = pd.Series(np.asarray(g).astype(str))
    y = np.asarray(y, float)
    groups = [y[(g == u).to_numpy()] for u in g.unique()]
    n = np.array([len(x) for x in groups], float)
    s1 = np.array([x.sum() for x in groups])
    s2 = np.array([(x ** 2).sum() for x in groups])
    N = n.sum()

    def ll(lam):
        w = lam / (1 + lam * n)
        xvx = np.sum(n - w * n ** 2)               # 1' V0^-1 1
        beta = np.sum(s1 - w * n * s1) / xvx
        # (y - beta)' V0^-1 (y - beta), group by group
        q = np.sum((s2 - 2 * beta * s1 + n * beta ** 2) - w * (s1 - n * beta) ** 2)
        sig2 = q / (N - 1)
        logdet = np.sum(np.log1p(lam * n))
        return -0.5 * ((N - 1) * np.log(sig2) + logdet + np.log(xvx) + (N - 1)), sig2
    return ll


def icc_profile(y, g):
    ll = reml_profile(y, g)
    grid = np.r_[0.0, np.logspace(-8, 3, 2000)]
    values = np.array([ll(l)[0] for l in grid])
    i = int(np.argmax(values))
    if i == 0:
        # checks that the maximum on the boundary is not an artefact of the grid
        r = optimize.minimize_scalar(lambda t: -ll(np.exp(t))[0], bounds=(-30, np.log(grid[1])),
                                     method="bounded", options={"xatol": 1e-10})
        if -r.fun > values[0] + 1e-10:
            lam = float(np.exp(r.x))
        else:
            lam = 0.0
    else:
        a, b = np.log(grid[max(i - 1, 1)]), np.log(grid[min(i + 1, len(grid) - 1)])
        r = optimize.minimize_scalar(lambda t: -ll(np.exp(t))[0], bounds=(a, b), method="bounded",
                                     options={"xatol": 1e-12})
        lam = float(np.exp(r.x))
    llmax, _ = ll(lam)
    return lam / (1 + lam), float(llmax), ll


def icc_statsmodels(y, g):
    d = pd.DataFrame({"y": y, "g": pd.Series(np.asarray(g)).astype(str)})
    out = {}
    for met in ("lbfgs", "bfgs", "powell", "nm"):
        try:
            r = smf.mixedlm("y ~ 1", d, groups=d.g).fit(method=met, reml=True)
            vr, ve = float(r.cov_re.iloc[0, 0]), float(r.scale)
            out[met] = (vr / (vr + ve), float(r.llf), bool(r.converged))
        except Exception as e:                                   # noqa: BLE001
            out[met] = (np.nan, -np.inf, False)
            log(f'    {met}: error {e}')
    return out


def icc_complete(y, g, label):
    icc, llmax, ll = icc_profile(y, g)
    sm = icc_statsmodels(y, g)
    best = max(sm, key=lambda k: sm[k][1])
    # the statsmodels log-likelihood has different constants: the ICC is compared
    assert abs(sm[best][0] - icc) < 1e-4, \
        f'{label}: REML profile {icc:.6f} versus statsmodels {best} {sm[best][0]:.6f}'
    lam_lb = sm["lbfgs"][0] / (1 - sm["lbfgs"][0]) if sm["lbfgs"][0] < 1 else np.inf
    loss = llmax - ll(max(lam_lb, 0.0))[0]
    return dict(ICC=icc, ICC_lbfgs=sm["lbfgs"][0], lbfgs_converged=sm["lbfgs"][2],
                loss_ll_lbfgs=float(loss), optimiser_best=best,
                ICC_bfgs=sm["bfgs"][0], ICC_powell=sm["powell"][0], ICC_nm=sm["nm"][0])


def cramer(tab, n):
    """As in 85: Cramer's V and bias-corrected V (Bergsma 2013)."""
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    r, c = tab.shape
    phi2 = chi2 / n
    V = float(np.sqrt(phi2 / (min(r, c) - 1)))
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc_ = c - (c - 1) ** 2 / (n - 1)
    return V, float(np.sqrt(phi2c / min(rc - 1, cc_ - 1)))


def collapse(s, threshold=N_MIN):
    s = pd.Series(s).astype(str)
    v = s.value_counts()
    return s.where(s.map(v) >= threshold, 'other')


# ----------------------------------------------------------------- data --
def rows_assay(saa, cohort, proj, col_f, col_t, agg, rep_f, rep_t):
    """Row of the SAA file whose wells give the kinetic parameters of the participant (as in 126); plus the first row (as in 69)."""
    s = saa[(saa.PROJECTID == proj) & (saa.CLINICAL_EVENT == "BL") & saa.PATNO.isin(cohort.PATNO)].copy()
    earliest = s.drop_duplicates("PATNO").set_index("PATNO")
    s["_f"] = getattr(s[rep_f], agg)(axis=1)
    s["_t"] = getattr(s[rep_t], agg)(axis=1)
    m = cohort[["PATNO", col_f, col_t]].merge(s[["PATNO", "_f", "_t", "RUNDATE", "InstrumentRep1"]], on="PATNO")
    m = m[np.isclose(m[col_f], m._f, rtol=1e-9) & np.isclose(m[col_t], m._t, rtol=1e-9)]
    assert m.PATNO.is_unique and len(m) == len(cohort), 'link between run and kinetic parameters not unique'
    m = m.set_index("PATNO").loc[cohort.PATNO]
    p = earliest.loc[cohort.PATNO]
    return (m.RUNDATE.astype(str).to_numpy(), m.InstrumentRep1.astype(str).to_numpy(),
            p.RUNDATE.astype(str).to_numpy(), p.InstrumentRep1.astype(str).to_numpy(),
            int((s.groupby("PATNO").size() > 1).sum()))


# ---------------------------------------------------------------- part E --
def copula_marginals(Y, X0, rng):
    """As C45.copula_null, but the quantiles come from X0 (the matrix before the injection)."""
    p = Y.shape[1]
    rho_s = stats.spearmanr(Y).statistic
    R = 2 * np.sin(np.pi * np.atleast_2d(rho_s) / 6)
    w, V = np.linalg.eigh(R)
    R = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    R = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))
    Z = rng.multivariate_normal(np.zeros(p), R, size=len(Y))
    U = stats.norm.cdf(Z)
    return np.column_stack([np.quantile(X0[:, j], U[:, j]) for j in range(p)])


def job_E(t):
    tag, dn, of, pj, r, X = t["tag"], t["dir"], t['of'], t["pj"], t["r"], t["X"]
    crc = crc32_legacy(tag.encode())
    key = [M125.SEED, crc, M125.DIRECTIONS.index(dn), of, pj, r]       # seeds of 125
    g = np.random.default_rng(key + [0])
    Y, lab = M125.inject(X, M125.direction(X, dn), t["delta"], t["pi"], g)
    Y = M118.standardise(Y)
    obsv_s = C45.silhouette_curve(Y, seed=C45.SEED).silhouette.to_numpy(float)
    obsv_k = np.array([_rule_functions.ps_k2(Y, np.random.default_rng(key + [2])),
                      _rule_functions.jaccard_k2(Y, np.random.default_rng(key + [2])),
                      _rule_functions.ari_algos(Y)])
    mia = [SEED, crc, M125.DIRECTIONS.index(dn), of, pj, r]               # own seeds for R1
    nul_s = np.empty((B_SIL, len(K_CURVE)))
    for b in range(B_SIL):
        gg = np.random.default_rng(mia + [1, b])
        nul_s[b] = C45.silhouette_curve(copula_marginals(Y, X, gg),
                                        seed=int(gg.integers(1e9))).silhouette.to_numpy(float)
    nul_k = np.empty((B_K2, len(CRITERIA)))
    for b in range(B_K2):
        gg = np.random.default_rng(mia + [3, b])
        Z = copula_marginals(Y, X, gg)
        nul_k[b] = [_rule_functions.ps_k2(Z, gg), _rule_functions.jaccard_k2(Z, gg), _rule_functions.ari_algos(Z)]
    res = dict(cohort=tag, direction=dn, delta=t["delta"], pi=t["pi"], replication=r, n_injected=int(lab.sum()))
    res.update({f"sil_k{k}": float(v) for k, v in zip(K_CURVE, obsv_s)})
    res.update({f'obsv_{c}': float(v) for c, v in zip(CRITERIA, obsv_k)})
    res.update({f"R1_p_sil_k{k}": float(v) for k, v in zip(K_CURVE, ((nul_s >= obsv_s - 1e-12).sum(0) + 1) / (B_SIL + 1))})
    res.update({f"R1_p_{c}": float(v) for c, v in zip(CRITERIA, ((nul_k >= obsv_k - 1e-12).sum(0) + 1) / (B_K2 + 1))})
    return res


def rule(E, pref):
    psil = [f"{pref}p_sil_k{k}" for k in K_CURVE]
    pk2 = [f"{pref}p_{c}" for c in CRITERIA]
    a = E[E.cohort == "24h"].set_index(['direction', 'replication']).sort_index()
    b = E[E.cohort == "150h"].set_index(['direction', 'replication']).sort_index()
    assert a.index.equals(b.index)
    out = pd.DataFrame(index=a.index)
    for tag, d in (("24h", a), ("150h", b)):
        r1 = (d[pk2] < ALPHA).any(axis=1)
        r2 = np.array([_rule_functions.bh(x).min() for x in d[psil].to_numpy()]) < ALPHA
        out[f'{tag}_rule'] = (r1 | r2).to_numpy()
    r1 = (a[pk2] < ALPHA).any(axis=1).to_numpy() | (b[pk2] < ALPHA).any(axis=1).to_numpy()
    r2 = np.array([_rule_functions.bh(np.r_[x, y]).min() for x, y in zip(a[psil].to_numpy(), b[psil].to_numpy())]) < ALPHA
    out['joint_rule'] = r1 | r2
    return out


# ------------------------------------------------------------------ main --
def main():
    t0 = time.time()
    title('127 - NUMBERS OF THE LETTER TO REVIEWER 3 (ICC, RUN x CENTRE, REGIME, MECHANISM)')
    assert hashlib.sha1(open(CACHE, "rb").read()).hexdigest() == SHA_CACHE, 'curated cache altered'
    assert ONLY, 'run with ONLY_SPORADIC=1 on the working directory run_sporadici'
    saa = pd.read_csv(SAA)
    cur = pd.read_parquet(CACHE)
    bl = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO").set_index("PATNO")
    c24 = _rule_functions.load_cohort("analysis_complete_cases.csv")
    c150 = _rule_functions.load_cohort("08_dataset_150h.csv")
    cc = pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv")).set_index("PATNO").loc[c24.PATNO].reset_index()
    d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
    d150 = d150[d150.complete].set_index("PATNO").loc[c150.PATNO].reset_index()

    # ---------------------------------------------------------------- A --
    title('A. ASSAY RUN OF EACH PARTICIPANT')
    R = {}
    R["24h"] = rows_assay(saa, cc, 237, "saa_Fmax", "saa_TTT", "mean",
                            [f"Fmax_24h_Rep{i}" for i in (1, 2, 3)], [f"TTT_24h_Rep{i}" for i in (1, 2, 3)])
    R["150h"] = rows_assay(saa, d150, 155, "k_Fmax", "k_T50", "median",
                             [f"Fmax_150h_Rep{i}" for i in (1, 2, 3)], [f"T50_150h_Rep{i}" for i in (1, 2, 3)])
    p126 = pd.read_csv(os.path.join(OUT, "126_by_run.csv"))
    for tag in R:
        run, instr, run69, instr69, dop = R[tag]
        counts = pd.Series(run).value_counts().sort_index()
        expected = p126[p126.cohort == tag].set_index('run').n.sort_index()
        assert counts.index.tolist() == expected.index.tolist() and (counts.to_numpy() == expected.to_numpy()).all(), \
            f'{tag}: counts per run differ from 126'
        changes = int((run != run69).sum())
        log(f'{tag}: {len(run)} subjects, {len(set(run))} runs (first row, as in 69: {len(set(run69))}); subjects with more than one row at baseline {dop}; run different from the first row for {changes}; instrument different for {int((instr != instr69).sum())}. Counts per run identical to 126.')
    assert len(set(R["24h"][0])) == 27 and len(set(R["150h"][0])) == 15
    assert len(set(R["24h"][2])) == 25

    # ---------------------------------------------------------------- B --
    title('B. TABLE S6 REDONE (eta^2 and ICC of run, instrument, centre)')
    kinet = {"24h": (cc, {"log Fmax": "saa_Fmax", "log TTT": "saa_TTT", "log AUC": "saa_AUC",
                        "log SLOPEmax": "saa_SLOPEMax"}),
           "150h": (d150, {"log Fmax": "k_Fmax", "log T50": "k_T50", "log AUC": "k_AUC", "log SLOPE": "k_SLOPE"})}
    s69 = pd.read_csv(os.path.join(OUT, "69_run_variance_components.csv"))
    s71 = pd.read_csv(os.path.join(OUT, "71_centre_effects.csv"))
    rows = []
    for tag, (src, kin) in kinet.items():
        run, instr, run69, instr69, _ = R[tag]
        site = bl.reindex(src.PATNO).SITE
        assert site.notna().all()
        centre = collapse(site.astype(str)).to_numpy()
        for name, col in kin.items():
            y = np.log(src[col].to_numpy(float))
            a = icc_complete(y, run69, f'{tag} {name} run 69')
            b = icc_complete(y, run, f'{tag} {name} corrected run')
            c = icc_complete(y, centre, f'{tag} {name} centre')
            r69 = s69[(s69.cohort.str.startswith(tag)) & (s69.parameter == name)].iloc[0]
            r71 = s71[(s71.cohort.str.startswith(tag)) & (s71.parameter == name)].iloc[0]
            # reproduction of the values of 69 and 71 (same definition, same optimiser)
            assert abs(eta2(y, run69) - r69.eta2_run) < 1e-10, f'{tag} {name}: eta2 of 69 not reproduced'
            assert abs(a["ICC_lbfgs"] - r69.ICC_run) < 1e-6, f'{tag} {name}: lbfgs ICC of 69 not reproduced'
            assert abs(eta2(y, instr69) - r69.eta2_instrument) < 1e-10
            assert abs(eta2(y, centre) - r71.eta2_centre) < 1e-10, f'{tag} {name}: centre eta2 of 71 not reproduced'
            assert abs(c["ICC_lbfgs"] - r71.ICC_centre) < 1e-6, f'{tag} {name}: lbfgs centre ICC of 71 not reproduced'
            rows.append(dict(
                cohort=tag, parameter=name,
                eta2_run_69=eta2(y, run69), eta2_run=eta2(y, run),
                ICC_run_69_submitted=a["ICC_lbfgs"], ICC_run_69_optimum=a["ICC"], loss_ll_69=a['loss_ll_lbfgs'],
                ICC_run=b["ICC"], ICC_run_lbfgs=b["ICC_lbfgs"], loss_ll_run=b['loss_ll_lbfgs'],
                eta2_instrument_69=eta2(y, instr69), eta2_instrument=eta2(y, instr),
                eta2_centre=eta2(y, centre),
                ICC_centre_submitted=c["ICC_lbfgs"], ICC_centre_optimum=c["ICC"], loss_ll_centre=c['loss_ll_lbfgs'],
                n_centres=len(set(centre)), n_runs=len(set(run)), n_runs_69=len(set(run69)), n=len(y)))
    S6 = pd.DataFrame(rows)
    S6.to_csv(os.path.join(OUT, "127_table_S6.csv"), index=False)
    log('Reproduction: eta^2 of 69 and 71 and lbfgs ICC of 69 and 71 reproduced (assertions).')
    log('Each ICC: exact REML profile = best of the four statsmodels optimisers within 1e-4 (assertion).')
    log(S6[['cohort', 'parameter', 'eta2_run_69', 'eta2_run', 'ICC_run_69_submitted', 'ICC_run_69_optimum',
            'loss_ll_69', 'ICC_run', 'eta2_instrument_69', 'eta2_instrument']].round(4).to_string(index=False))
    log("")
    log(S6[['cohort', 'parameter', 'eta2_centre', 'ICC_centre_submitted', 'ICC_centre_optimum',
            'loss_ll_centre', 'n_centres']].round(4).to_string(index=False))
    log('\nWrong submitted values (loss of log-likelihood > 0.01 and |difference| >= 0.005):')
    located = 0
    for _, r in S6.iterrows():
        for col_s, col_o, col_p, name in (('ICC_run_69_submitted', 'ICC_run_69_optimum', 'loss_ll_69', 'run ICC'),
                                          ('ICC_centre_submitted', 'ICC_centre_optimum', 'loss_ll_centre', 'centre ICC')):
            if r[col_p] > 0.01 and abs(r[col_s] - r[col_o]) >= 0.005:
                located += 1
                log(f'  {r.cohort} {r.parameter}, {name}: submitted {r[col_s]:.3f}, optimum {r[col_o]:.3f} (log-likelihood +{r[col_p]:.3f})')
    log(f'  total: {located}')
    for tag in R:
        s = S6[S6.cohort == tag].iloc[0]
        log(f'Expected run eta^2 with no effect, (G-1)/(n-1): {tag} {(s.n_runs - 1) / (s.n - 1):.3f} (with the definition of 69: {(s.n_runs_69 - 1) / (s.n - 1):.3f})')

    # ---------------------------------------------------------------- C --
    title('C. RUN x CENTRE (pooling rule of 85, corrected run)')
    s85 = pd.read_csv(os.path.join(OUT, "85_run_centre.csv"))
    rows = []
    for tag, (src, _) in kinet.items():
        run, _, run69, _, _ = R[tag]
        site = bl.reindex(src.PATNO).SITE.astype(str).to_numpy()
        for defn, rr in (('69 (first row)', run69), ('corrected', run)):
            tab = pd.crosstab(collapse(rr).to_numpy(), collapse(site).to_numpy())
            V, Vc = cramer(tab, len(rr))
            q_c = tab.max(axis=0) / tab.sum(axis=0)
            q_r = tab.max(axis=1) / tab.sum(axis=1)
            for run_, q in q_r.items():
                if q == 1:
                    log(f'  {tag}, run {defn}: the run category {run_} ({int(tab.loc[run_].sum())} subjects) comes entirely from centre {tab.loc[run_].idxmax()}')
            for centre_, q in q_c.items():
                if q == 1:
                    log(f'  {tag}, run {defn}: centre {centre_} ({int(tab[centre_].sum())} subjects) lies entirely in run {tab[centre_].idxmax()}')
            rows.append(dict(cohort=tag, run=defn, n=len(rr), categories_run=tab.shape[0],
                              categories_centre=tab.shape[1], V=V, V_corrected=Vc,
                              centres_in_one_single_run=int((q_c == 1).sum()),
                              runs_in_an_only_centre=int((q_r == 1).sum()),
                              centres_in_one_single_run_without_residual=int((q_c.drop('other', errors="ignore") == 1).sum()),
                              runs_in_an_only_centre_without_residual=int((q_r.drop('other', errors="ignore") == 1).sum())))
    TC = pd.DataFrame(rows)
    for tag, lab85 in (("24h", 'primary 24h'), ("150h", 'replication 150h')):
        v85 = s85[s85.cohort == lab85].V_cramer_corrected.iloc[0]
        v = TC[(TC.cohort == tag) & (TC.run == '69 (first row)')].V_corrected.iloc[0]
        assert abs(v - v85) < 1e-10, f'{tag}: corrected V of 85 not reproduced ({v} versus {v85})'
    TC.to_csv(os.path.join(OUT, "127_run_centre.csv"), index=False)
    log('Corrected V of 85 reproduced with the first row (assertion).')
    log(TC.round(4).to_string(index=False))

    # ---------------------------------------------------------------- D --
    title('D. RUN REGIME OF THE TTT (24 H): ORDINARY CI AND CI ROBUST BY RUN')
    run = R["24h"][0]
    b24 = bl.reindex(cc.PATNO)
    d = pd.DataFrame({'run': run, "logTTT": np.log(cc.saa_TTT.to_numpy()),
                      'centre': b24.SITE.astype(str).to_numpy(), 'phase': b24.enroll_phase.to_numpy()})
    medc = d.groupby('run').logTTT.median()
    nc = d.groupby('run').size()
    large = medc[nc >= N_MIN]
    km = C45.KMeans(2, n_init=50, random_state=C45.SEED).fit(large.to_numpy().reshape(-1, 1))
    centres = km.cluster_centers_.ravel()
    slow = int(np.argmax(centres))
    reg = {c: int(np.argmin(np.abs(centres - v)) == slow) for c, v in medc.items()}
    d['slow'] = d.run.map(reg).astype(int)
    assert sum(reg.values()) == 19 and len(reg) == 27 and int(d.slow.sum()) == 309, 'regime differs from 126'
    fit = smf.ols('logTTT ~ slow + C(centre) + C(phase)', data=d).fit()
    ci = fit.conf_int().loc['slow']
    assert round(fit.params['slow'], 3) == 0.321 and round(ci[0], 3) == 0.295 and round(ci[1], 3) == 0.347, \
        'coefficient or ordinary CI differ from 126'
    groups = pd.factorize(d.run)[0]
    rob = smf.ols('logTTT ~ slow + C(centre) + C(phase)', data=d).fit(cov_type="cluster",
                                                                        cov_kwds={"groups": groups})
    cir = rob.conf_int().loc['slow']
    log(f"Regime reproduced (19 slow runs out of 27, 309 slow subjects). Coefficient {fit.params['slow']:.3f}; ordinary 95% CI {ci[0]:.3f} to {ci[1]:.3f} (= 126); robust by run ({d.run.nunique()} groups) {cir[0]:.3f} to {cir[1]:.3f} [unrounded {cir[0]:.5f} to {cir[1]:.5f}] (standard error {fit.bse['slow']:.4f} -> {rob.bse['slow']:.4f}).")

    # ---------------------------------------------------------------- F --
    # Added on the evening of 15/11 after the independent re-reading of the letter
    # (remarks N1-N3): storage by phase and for the 45 present in both cohorts,
    # convention of 126.
    title('F. CSF STORAGE BY PHASE AND IN THE 45 (convention of 126)')
    cons = {}
    for tag, src in (("24h", cc), ("150h", d150)):
        b = bl.reindex(src.PATNO)
        vis = pd.to_datetime(b.visit_date.to_numpy())
        rd = pd.to_datetime(R[tag][0])
        cons[tag] = pd.DataFrame({"PATNO": src.PATNO.to_numpy(), 'run': R[tag][0],
                                  'phase': b.enroll_phase.to_numpy(),
                                  'years': np.asarray((rd - vis).days) / 365.25})
    c = cons["24h"]
    assert round(c.years.median(), 2) == 0.76, 'median storage differs from 126'
    for f in (1, 2):
        x = c[c.phase == f].years
        log(f'24h phase {f} (n = {len(x)}): median {x.median():.2f} years, IQR '
            f'{x.quantile(0.25):.2f}-{x.quantile(0.75):.2f}, range {x.min():.2f}-{x.max():.2f}; more '
            f'than 10 years {int((x > 10).sum())}')
    x = c.years
    log(f'24h whole cohort (n = {len(x)}): median {x.median():.2f}, IQR {x.quantile(0.25):.2f}-{x.quantile(0.75):.2f}')
    common = sorted(set(cons["24h"].PATNO) & set(cons["150h"].PATNO))
    k = c[c.PATNO.isin(common)]
    late = k[pd.to_datetime(k.run) >= pd.Timestamp("2024-10-01")]
    log(f"Present in both cohorts: {len(common)}; at 24 h run from October 2024 onwards: {len(late)} (phase 1 {int((late.phase == 1).sum())}, phase 2 {int((late.phase == 2).sum())}); of these, more than 10 years of storage {int((late.years > 10).sum())}; minimum {late.years.min():.2f}, among the phase 2 only {', '.join((f'{v:.2f}' for v in sorted(late[late.phase == 2].years)))}")
    log('  runs: ' + ", ".join(f"{r} {n}" for r, n in late.run.value_counts().sort_index().items()))

    # ---------------------------------------------------------------- F2 --
    # Added on 16/11 after the re-reading of the numbers of the three letters: R1.10
    # (script 108) counted from the puncture but to the first positive baseline run;
    # part F counts from the visit (first of the month) to the right run. Here: from
    # the puncture (convention of 108: visit at mid-month plus the puncture-visit
    # offset in whole days) to the run of the sample that gives the kinetic parameters.
    title('F2. STORAGE FROM LUMBAR PUNCTURE TO THE RIGHT RUN (puncture as in 108)')
    sp = saa.copy()
    sp["RUNDATE"] = pd.to_datetime(sp.RUNDATE)
    by_run = []
    for tag, src, prog, kind in (("24h", cc, 237, "Type1"), ("150h", d150, 155, None)):
        b = bl.reindex(src.PATNO)
        lp = ((b.age_LP - b.age_at_visit) * 365.0).round() + 0.0
        data_lp = (pd.to_datetime(b.visit_date, format="%m/%Y") + pd.Timedelta(days=14)
                   + pd.to_timedelta(lp, unit="D"))
        right = pd.Series(pd.to_datetime(R[tag][0]), index=src.PATNO.to_numpy())
        # check: with the first positive baseline run (as in 108) 108_times_baseline.csv
        # is reproduced
        s = sp[(sp.PROJECTID == prog) & (sp.COHORT == "PD") & (sp.CLINICAL_EVENT == "BL")
               & (sp.SAA_Status == "Positive") & sp.PATNO.isin(src.PATNO)]
        if kind:
            s = s[s.SAA_Type == kind]
        earliest = s.sort_values(["PATNO", "RUNDATE"]).drop_duplicates("PATNO").set_index("PATNO").RUNDATE
        t108 = pd.read_csv(os.path.join(OUT, "108_times_baseline.csv"))
        t108 = t108[(t108.cohort == tag) & t108.interval.str.startswith('assay run minus lumbar puncture')]
        phase = pd.Series(b.enroll_phase.to_numpy(), index=src.PATNO.to_numpy())
        dlp = pd.Series(data_lp.to_numpy(), index=src.PATNO.to_numpy())
        for f in (1, 2):
            idx = phase.index[phase == f]
            x108 = ((earliest.reindex(idx) - dlp.reindex(idx)).dt.days / 365.25).dropna()
            expected = t108[t108.interval.str.endswith(f'enrolment phase {f}')].iloc[0]
            assert abs(x108.median() - expected.median) < 1e-9 and len(x108) == expected.n, \
                f'{tag} phase {f}: does not reproduce 108 ({x108.median()} versus {expected.median})'
        years = ((right - dlp).dt.days / 365.25)
        changes = int((right != earliest.reindex(right.index)).sum())
        log(f'{tag}: puncture reconstructed as in 108 (the medians by phase of 108 reproduced with the first run); participants whose right run differs from the first: {changes}')
        for f in (1, 2):
            x = years[phase == f].dropna()
            log(f'{tag} phase {f} (n = {len(x)}, missing {int(years[phase == f].isna().sum())}): median {x.median():.2f} years, IQR {x.quantile(0.25):.2f}-{x.quantile(0.75):.2f}, range {x.min():.2f}-{x.max():.2f}; more than 10 years {int((x > 10).sum())}')
        x = years.dropna()
        log(f'{tag} whole cohort (n = {len(x)}): median {x.median():.2f}, IQR '
            f'{x.quantile(0.25):.2f}-{x.quantile(0.75):.2f}')
        if tag == "24h":
            a24 = years
        # F3 (evening of 16/11): median per run, for the storage column of Table S28,
        # which came from 126 (from the visit at the first of the month). Same runs and
        # same n as 126_by_run.csv.
        pc = pd.DataFrame({'run': right.dt.strftime("%Y-%m-%d"), 'years': years.to_numpy()})
        g = pc.groupby('run').agg(n=('years', "size"), n_storage=('years', "count"),
                                    storage_puncture=('years', "median")).reset_index()
        p126 = pd.read_csv(os.path.join(OUT, "126_by_run.csv"))
        p126 = p126[p126.cohort == tag].set_index('run')
        assert list(g.run) == sorted(p126.index) and (g.set_index('run').n == p126.n.reindex(g.run)).all(), \
            f'{tag}: runs or n differ from 126_by_run.csv'
        g['storage_visit_126'] = p126.storage_median.reindex(g.run).to_numpy()
        g.insert(1, 'cohort', tag)
        by_run.append(g)
        log(f'{tag}: storage by run from the puncture, {len(g)} runs; difference from the visit convention (126) between {(g.storage_puncture - g.storage_visit_126).min():+.2f} and {(g.storage_puncture - g.storage_visit_126).max():+.2f} years; missing {int((g.n - g.n_storage).sum())}')
    pd.concat(by_run).to_csv(os.path.join(OUT, "127_storage_by_run.csv"), index=False)
    k = a24.reindex(common)
    late2 = k[pd.to_datetime(c.set_index("PATNO").run.reindex(common)) >= pd.Timestamp("2024-10-01")]
    log(f'The 45, at 24 h run from October 2024 onwards: {len(late2)}; of these, more than 10 years of storage from the puncture {int((late2 > 10).sum())}; minimum {late2.min():.2f}')

    if ONLY_TALLIES:
        log('\nONLY_TALLIES=1: part E skipped.')
        save(t0)
        return

    # ---------------------------------------------------------------- E --
    title("E. MECHANISM OF 125: REFERENCE WITH THE MARGINALS FROM BEFORE THE INJECTION")
    npz = np.load(os.path.join(OUT, "features.npz"), allow_pickle=True)
    assert np.array_equal(npz["PATNO"].astype("int64"), c24.PATNO.to_numpy())
    R150 = c150[M118.F150].to_numpy(float).copy()
    R150[:, :4] = np.log(R150[:, :4])
    X = {"24h": npz["F_primary"].astype(float), "150h": M118.standardise(R150)}
    saved = np.load(os.path.join(OUT, "118_nulls.npz"), allow_pickle=True)
    for tag in X:
        assert np.abs(saved[f'V0_{tag}_matrix'] - X[tag]).max() < 1e-12
    delta, pi = 3.0, 0.20
    of, pj = M125.DELTAS.index(delta), M125.PIS.index(pi)
    tasks = [dict(tag=tag, dir=dn, of=of, pj=pj, r=r, delta=delta, pi=pi, X=X[tag])
               for r in range(N_REP_E) for dn in ('timeval', "dat") for tag in X]
    log(f'{len(tasks)} data sets (2 directions x 2 cohorts x {N_REP_E} replicates), Delta {delta}, pi {pi}; B {B_SIL} and {B_K2}; {N_PROC} processes.')
    result = []
    with mp.get_context("spawn").Pool(N_PROC) as pool:
        for r in pool.imap_unordered(job_E, tasks, chunksize=1):
            result.append(r)
            print(f'    {len(result)}/{len(tasks)}  ({time.time() - t0:.0f} s)', flush=True)
    E = pd.DataFrame(result).sort_values(['cohort', 'direction', 'replication']).reset_index(drop=True)
    e125 = pd.read_csv(os.path.join(OUT, "125_outcomes.csv"))
    e125 = e125[(e125.delta == delta) & (e125.pi == pi) & e125.direction.isin(['timeval', "dat"])
                & (e125.replication < N_REP_E)]
    M = E.merge(e125, on=['cohort', 'direction', 'replication'], suffixes=("", "_125"))
    assert len(M) == len(E) == 4 * N_REP_E
    obsv = [f"sil_k{k}" for k in K_CURVE] + [f'obsv_{c}' for c in CRITERIA]
    departure = max(float(np.abs(M[c] - M[c + "_125"]).max()) for c in obsv)
    assert departure < 1e-9, f'injected data sets not identical to 125 (discrepancy {departure})'
    assert (M.n_injected == M.n_injected_125).all()
    log(f'Injected data sets identical to 125: observed silhouette and criteria match (maximum discrepancy {departure:.1e}).')
    for c in [f"p_sil_k{k}" for k in K_CURVE] + [f"p_{c}" for c in CRITERIA]:
        M[f"R0_{c}"] = M[c]
    rows = []
    for dn in ('timeval', "dat"):
        Md = M[M.direction == dn]
        r0 = rule(Md, "R0_")
        r1 = rule(Md, "R1_")
        for col in ("24h_regola", "150h_regola", 'joint_rule'):
            k0, k1 = int(r0[col].sum()), int(r1[col].sum())
            p = stats.fisher_exact([[k1, N_REP_E - k1], [k0, N_REP_E - k0]])[1]
            rows.append(dict(direction=dn, reading=col.replace('_rule', ""), R0=k0, R1=k1,
                              n=N_REP_E, p_fisher=p))
            log(f"  {dn} Delta 3 pi 0.20, {col.replace('_rule', '')}: reference of 125 {k0}/{N_REP_E}, marginals from before the injection {k1}/{N_REP_E} (Fisher p = {p:.3g})")
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "127_mechanism.csv"), index=False)
    log('\nPre-set reading: mechanism supported if joint R1 > R0 with Fisher p < 0.05.')
    for dn in ('timeval', "dat"):
        r = T[(T.direction == dn) & (T.reading == 'joint')].iloc[0]
        log(f"  {dn}: " + ('supported' if (r.R1 > r.R0 and r.p_fisher < 0.05) else 'not supported')
            + f" ({r.R0} -> {r.R1}, p = {r.p_fisher:.3g})")
    save(t0)


def save(t0):
    assert hashlib.sha1(open(CACHE, "rb").read()).hexdigest() == SHA_CACHE, 'curated cache altered'
    log(f'\nTotal time: {time.time() - t0:.0f} s')
    name = "127_reviewer3_letter_numbers.txt" if not ONLY_TALLIES else "127_reviewer3_letter_numbers_counts.txt"
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
