"""Cohort labels derived from the actual sample size.

The string "24h (n=687)" was hard-coded in about ten scripts, where it serves both
as a label and as the join KEY between outputs. With a different population (for
example the sporadic participants only) producers and consumers stay consistent
with one another, but the number shown becomes false and ends up in the tables and
in the manuscript.

Here the label is derived from the data, so that key and label cannot diverge.
"""

import os
import numpy as np
import pandas as pd


def samplesize(OUT):
    """(n of the 24 h cohort, n of the 150 h cohort) read from the outputs."""
    n24 = len(np.load(os.path.join(OUT, "features.npz"))["F_primary"])
    d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
    n150 = (int(d150["complete"].sum()) if "complete" in d150.columns
            else len(d150))
    return n24, n150


def labels(OUT):
    """The two cohort keys, in the format used by all scripts."""
    n24, n150 = samplesize(OUT)
    return f"24h (n={n24})", f"150h (n={n150})"
