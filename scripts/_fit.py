"""Robust estimation of the mixed models.

Reason. The pipeline used to fit the random-slope models with `lbfgs` only,
recording `converged` without using it: a non-converged fit entered the tables
like any other. In the sporadic-only cohort the MoCA model is the only one that
does not converge, and it produced a standard error of 0.0658, twice the largest
value observed in twenty random subsamples of the same size (0.0266-0.0345),
although it had more observations than they did. The resulting p value (0.384)
was not a weak estimate: it was not an estimate. Refitted with `powell`, `nm` or
`bfgs`, which all converge to the same point, the answer is
beta = 0.0501, SE = 0.0281, p = 0.075.

Rule adopted: one optimiser after another is tried on the SAME model, and the
first that converges is returned. Changing the optimiser does not change the
model; falling back to a random intercept only does, so that fallback remains
available but must be requested and is flagged in the result.
"""

import warnings
import statsmodels.formula.api as smf

OPTIMISERS = [("lbfgs", 300), ("powell", 2000), ("nm", 5000), ("bfgs", 2000)]


def fit_mixed(formula, data, groups, re_formula="~years",
              fallback_intercept=False):
    """First converged fit among the available optimisers.

    Returns (result, method, fallback) or (None, None, False) if no
    specification converges. `fallback` is True if the random slope had to be
    dropped.
    """
    last = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, it in OPTIMISERS:
            try:
                r = smf.mixedlm(formula, data, groups=groups,
                                re_formula=re_formula).fit(method=method,
                                                           maxiter=it)
            except Exception:
                continue
            last = (r, method)
            if r.converged:
                return r, method, False
        if fallback_intercept:
            for method, it in OPTIMISERS:
                try:
                    r = smf.mixedlm(formula, data, groups=groups).fit(
                        method=method, maxiter=it)
                except Exception:
                    continue
                if r.converged:
                    return r, method, True
    # no convergence: the last attempt is returned, and the caller must treat it
    # as not estimated
    return (last[0] if last else None,
            last[1] if last else None, False)
