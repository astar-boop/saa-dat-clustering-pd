"""Choice of the two DAT measures of the primary set.

The primary set always has two DAT terms: a putaminal (or striatal) measure and
the mean caudate. Which one is the first was hard-coded in `02` and in `08`, and
no sensitivity analysis varied it: it was the only construction choice of the
primary set left without a robustness test.

Here the choice becomes an environment parameter, on the same model as
ONLY_SPORADIC:

    DAT_SET=low        min(putamen L, R) + mean caudate   [DEFAULT, unchanged]
    DAT_SET=mean       bilateral putamen + mean caudate
    DAT_SET=striatum   bilateral striatum + mean caudate

The default reproduces the previous behaviour exactly: without the environment
variable nothing changes.

WARNING on the `striatum` branch. The bilateral striatum CONTAINS the caudate, and
in the 687 complete cases it correlates with the mean caudate at rho = 0.83. The
resulting set therefore carries a largely redundant pair, exactly the defect for
which TSmax was excluded a priori (rho = 0.999 with TTT). The branch is to be read
as a robustness check, not as a legitimate alternative specification.
"""

import os

_SET = {
    "low":      ["dat_putamen_low", "dat_caudate_mean"],
    "mean":     ["dat_putamen_mean", "dat_caudate_mean"],
    "striatum": ["dat_striatum_mean", "dat_caudate_mean"],
}


def choice():
    """The value of DAT_SET, validated. Default 'low'."""
    s = os.environ.get("DAT_SET", "low").strip().lower()
    if s not in _SET:
        raise ValueError(
            f"DAT_SET='{s}' not recognised; expected: {sorted(_SET)}")
    return s


def variables():
    """The two DAT columns of the primary set, in the order in which they enter."""
    return list(_SET[choice()])
