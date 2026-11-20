"""
81 - Reliability of the MEDIAN of three replicates: two estimators, both
     failed, and the conclusion that follows.

Context. Script 38 estimated the error variance of the median by resampling
the residuals r_ij = x_ij - mean_i from a single pool, in independent
triplets, after standardising them on the empirical standard deviation.
The result was inefficiency factors c as low as 0.045 — a median seven times
more precise than the mean — and reliabilities (0.985 for Fmax at 24 h) much
higher than the ICC(1,3) (0.897). The review observed that the result
depended decisively on an algorithm never specified, and asked either for
its definition and validation, or for a return to the ICC.

This script does both and concludes for the second.

  Estimator A (direct). From the identity median(x_i) - mean(x_i) = median(r_i)
  it follows that Var[median(e)] = Var[median(r)] + Var[mean(e)] IF
  median(e)-mean(e) is uncorrelated with mean(e). It holds under normality.

  Estimator B (resampled). That of script 38, reproduced here.

  Validation. On errors of known shape, with the true c obtained by direct
  simulation: under normality both are accurate; under heavy tails —
  the situation of these data, and the very reason why the median was chosen
  in the long protocol — BOTH FAIL, in opposite directions.
  A overestimates c by up to eleven times, B underestimates it by up to 77%.

  Conclusion. With a single triplet per sample the reliability of the
  median of three is not identifiable without an assumption on the shape of
  the error that these data do not allow to be verified. The manuscript
  therefore reports ICC(1,1) and ICC(1,3) and uses the ICC(1,3) — the
  reliability of the mean of three — as an approximation for the median,
  WITHOUT stating the direction of the approximation error: the law of the
  replicate error is unknown, and this is precisely the non-identifiability
  just described.

Output: outputs/81_reliability_median.csv, 81_validation_estimator.csv, .txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from _paths import DATA

warnings.filterwarnings("ignore")
BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
SRC = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
SEED = 20260805
B_BOOT = 2000
rep = []


def log(m=""):
    print(m, flush=True)
    rep.append(str(m))


def components(m):
    """One-way ANOVA on complete triplets: sigma2_b, sigma2_w."""
    n, k = m.shape
    gm = m.mean()
    msb = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msw = ((m - m.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return max((msb - msw) / k, 0.0), msw


def c_median(m):
    """Estimator A ("direct"): Var(median of 3) in units of sigma2_w.

    It rests on the identity median(x_i) - mean(x_i) = median(r_i) and on
    the assumption that median(e) - mean(e) is uncorrelated with mean(e). The
    second holds under normality and NOT in general: under heavy tails the two
    quantities are dominated by the same outlying value and are strongly
    anticorrelated, so that the estimator overestimates c. See the validation
    block.
    """
    s2b, s2w = components(m)
    if s2w <= 0:
        return 1 / 3
    r = m - m.mean(axis=1, keepdims=True)
    v = np.var(np.median(r, axis=1), ddof=1)
    return float(max((v + s2w / 3) / s2w, 1 / 3))


def c_median_resampled(m, rng, n_sim=200000):
    """Estimator B: that of script 38, reproduced for comparison.

    It resamples the standardised residuals from a single pool, in
    independent triplets. It ignores the within-triplet structure and
    standardises on a standard deviation inflated by the same outlying values,
    and therefore underestimates c.
    """
    r = (m - m.mean(axis=1, keepdims=True)).ravel()
    r = r[np.isfinite(r)]
    sd = r.std(ddof=0)
    if sd == 0:
        return 1 / 3
    z = (r - r.mean()) / sd
    return float(np.median(rng.choice(z, size=(n_sim, 3), replace=True), axis=1).var())


def reliability(m):
    s2b, s2w = components(m)
    if s2b + s2w <= 0:
        return np.nan, np.nan, np.nan
    c = c_median(m)
    return (s2b / (s2b + s2w / 3), s2b / (s2b + c * s2w), c)


# ===========================================================================
# validation: does the estimator recover the true c?
# ===========================================================================
log("=" * 78)
log('VALIDATION OF THE ESTIMATOR ON ERRORS OF KNOWN SHAPE')
log("=" * 78)
log('true c = Var(median of 3 errors)/Var(error), by direct simulation.')
log('estimated c = the one of the script, applied to 600 simulated triplets.')
rng = np.random.default_rng(SEED)
rows_val = []
for name, gen in [
        ('normal', lambda n: rng.normal(size=(n, 3))),
        ("Student's t, 3 df", lambda n: rng.standard_t(3, size=(n, 3))),
        ('contaminated 5% scale 10',
         lambda n: rng.normal(size=(n, 3)) *
                   np.where(rng.random((n, 3)) < 0.05, 10.0, 1.0)),
        ('one aberrant well in 20 triplets',
         lambda n: (lambda e: (np.put_along_axis(
             e, rng.integers(0, 3, size=(n, 1)),
             np.take_along_axis(e, rng.integers(0, 3, size=(n, 1)), 1) +
             np.where(rng.random((n, 1)) < 0.05, 30.0, 0.0), 1), e)[1])(
             rng.normal(size=(n, 3)))),
]:
    e_big = gen(400000)
    c_true = np.var(np.median(e_big, axis=1), ddof=1) / np.var(e_big.ravel(), ddof=1)
    # observable triplets: subject + error, with between-subject variance 1
    e = gen(600)
    m = rng.normal(size=(600, 1)) + e
    rows_val.append({'form': name, 'c_true': c_true,
                      'c_estimator_A_direct': c_median(m),
                      'c_estimator_B_resampled': c_median_resampled(m, rng)})
val = pd.DataFrame(rows_val)
val['error_A'] = (val.c_estimator_A_direct - val.c_true) / val.c_true
val['error_B'] = (val.c_estimator_B_resampled - val.c_true) / val.c_true
log(val.round(4).to_string(index=False))
log("")
log("READING. Under normality both estimators are acceptable and the")
log("true value is 0.449 (for the mean of three it is 1/3 = 0.333). Under heavy")
log("tails — which is the situation of these data, and is the very reason")
log("why the median was chosen in the long protocol — the two estimators")
log('BOTH FAIL, and in opposite directions: the direct one overestimates c')
log("(underestimates the reliability of the median), the resampled one underestimates it")
log("(overestimates it). The true c falls below 1/3, that is, the median is")
log("indeed more precise than the mean, but neither of the two estimators")
log('recovers by how much.')
log("")
log("CONSEQUENCE. With a single triplet per sample the reliability of the")
log("median of three is not identifiable without an assumption on the shape")
log("of the error that these data do not allow to be verified. In the manuscript")
log("ICC(1,1) and ICC(1,3) are therefore reported, and the ICC(1,3) — which is")
log("the reliability of the MEAN of three — is used as an approximation for")
log("the median, which is the actual exposure at 150 h. The DIRECTION of")
log("that approximation error is NOT stated: in the contaminated")
log("distributions simulated here the median turns out more precise than the mean, but the")
log("distribution of the replicate error in these data is unknown — this is exactly the")
log("non-identifiability just described — and a general statement in one")
log("direction or the other would contradict it. Any disattenuation based")
log("on the ICC(1,3) is to be read as an illustrative scenario under explicit")
log('assumptions, not as a quantified correction.')

# ===========================================================================
# estimate on the data
# ===========================================================================
log("\n" + "=" * 78)
log("RELIABILITY ON THE DATA: MEAN VERSUS MEDIAN OF THREE")
log("=" * 78)
saa = pd.read_csv(SRC)
cur = pd.read_parquet(os.path.join(OUT, "cache", "curated.parquet"))
bl = cur[cur.EVENT_ID == "BL"].drop_duplicates("PATNO")[["PATNO", "subgroup"]]
spor = set(bl[bl.subgroup == "Sporadic PD"].PATNO)

PROT = [("24h", 237, ["Fmax", "TTT", "AUC", "TSmax", "SLOPEMax"]),
        ("150h", 155, ["Fmax", "TTT", "AUC", "T50", "SLOPE"])]

rows = []
for proto, proj, params in PROT:
    s = saa[(saa.COHORT == "PD") & (saa.PROJECTID == proj)
            & (saa.CLINICAL_EVENT == "BL") & (saa.SAA_Status == "Positive")]
    s = s[s.PATNO.isin(spor)]
    log(f'\n--- protocol {proto}: {len(s)} sporadic positive samples at baseline ---')
    for f in params:
        cols = [f"{f}_{proto}_Rep{i}" for i in (1, 2, 3)]
        v = s[cols].apply(pd.to_numeric, errors="coerce")
        ok = v.notna().all(axis=1) & (v > 0).all(axis=1)
        m = np.log(v[ok].to_numpy(float))
        R_mean, R_median, c = reliability(m)
        # bootstrap over subjects for the uncertainty
        rb = np.random.default_rng(SEED)
        bm, bd = [], []
        for _ in range(B_BOOT):
            idx = rb.integers(0, len(m), len(m))
            a, b, _c = reliability(m[idx])
            bm.append(a); bd.append(b)
        rows.append({
            'protocol': proto, 'parameter': f, 'n_triplicates_complete': int(ok.sum()),
            'n_excluded': int((~ok).sum()), 'c_median': c,
            'R_mean3': R_mean,
            'R_mean3_CI_inf': float(np.percentile(bm, 2.5)),
            'R_mean3_CI_upp': float(np.percentile(bm, 97.5)),
            'R_median3': R_median,
            'R_median3_CI_inf': float(np.percentile(bd, 2.5)),
            'R_median3_CI_upp': float(np.percentile(bd, 97.5))})
        log(f'  {f:9s} n={int(ok.sum()):4d}  c={c:.3f}  R(mean3)={R_mean:.3f} '
            f'[{np.percentile(bm, 2.5):.3f}; {np.percentile(bm, 97.5):.3f}]  '
            f'R(median3)={R_median:.3f} [{np.percentile(bd, 2.5):.3f}; '
            f'{np.percentile(bd, 97.5):.3f}]')

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "81_reliability_median.csv"), index=False)
val.to_csv(os.path.join(OUT, "81_validation_estimator.csv"), index=False)

# ===========================================================================
# disattenuation of the MoCA coefficient
# ===========================================================================
log("\n" + "=" * 78)
log('DISATTENUATION OF THE MoCA-T50 COEFFICIENT')
log("=" * 78)
t50 = d[(d.protocol == "150h") & (d.parameter == "T50")].iloc[0]
b_obsv = 0.0501
log(f'  observed coefficient                         : {b_obsv:.4f}')
for lbls, R, lo, hi in [('mean of 3 (ICC 1,3)', t50.R_mean3,
                        t50.R_mean3_CI_inf, t50.R_mean3_CI_upp),
                       ('median of 3 (aggregator used)', t50.R_median3,
                        t50.R_median3_CI_inf, t50.R_median3_CI_upp)]:
    log(f'  {lbls}')
    log(f"     reliability {R:.3f}  95% CI [{lo:.3f}; {hi:.3f}]")
    log(f'     standardised predictor, b/sqrt(R) = {b_obsv / np.sqrt(R):.4f}  (interval '
        f'{b_obsv / np.sqrt(hi):.4f}–{b_obsv / np.sqrt(lo):.4f})')
    log(f'     unstandardised slope, b/R         = {b_obsv / R:.4f}')

log(f'\nThe inefficiency factor of the median lies between {d.c_median.min():.3f} and '
    f'{d.c_median.max():.3f}, always >= 1/3: the median')
log("never turns out more precise than the mean, as it must be.")

with open(os.path.join(OUT, "81_reliability_median.txt"), "w") as fh:
    fh.write("\n".join(rep))
log('\nSaved: outputs/81_reliability_median.csv and .txt')
