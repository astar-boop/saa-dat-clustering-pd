"""CSF and plasma albumin data, rebuilt from the source.

Scripts 20 and 23 used to point to an `alb.csv` produced in a temporary session
directory that has since been deleted: those analyses could no longer be
reproduced on any cohort. Here the file is rebuilt from the PPMI biospecimen
results and cached next to the other data.

Its content does not depend on the population under analysis: it is an extract
of the laboratory data, filtered downstream by the scripts that use it.
"""

import os
import pandas as pd
from _paths import DATA

SRC = os.path.join(DATA, "Current_Biospecimen_Analysis_Results_07May2026.csv")
TESTS = ["CSF Albumin", "Plasma Albumin"]
COLS = ["PATNO", "CLINICAL_EVENT", "TESTNAME", "TESTVALUE", "UNITS",
        "RUNDATE", "PROJECTID"]


def path_alb(OUT):
    """Path of alb.csv, rebuilding it from the source if it is missing."""
    cache = os.path.join(OUT, "cache")
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, "alb.csv")
    if os.path.exists(p):
        return p
    pieces = []
    for ch in pd.read_csv(SRC, usecols=COLS, chunksize=500_000,
                          low_memory=False):
        pieces.append(ch[ch.TESTNAME.isin(TESTS)])
    alb = pd.concat(pieces, ignore_index=True)
    alb.to_csv(p, index=False)
    print(f'[_alb] rebuilt {p}: {len(alb)} rows ({alb.TESTNAME.value_counts().to_dict()})')
    return p
