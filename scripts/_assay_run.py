"""The row of the assay file from which a participant's kinetic parameters come.

Reason. Some participants have more than one baseline row in the assay file (a
sample run twice). The scripts that looked up the run date took the first row
(`drop_duplicates("PATNO")`), which for six participants is not the one whose wells
give the kinetic parameters analysed: three in the 24 h cohort and three in the
150 h cohort.

`row_of_kinetics` removes from a selection of assay rows, for the participants
of the two cohorts, the baseline rows whose wells do NOT give the participant's
kinetic parameters. After that, the first row per participant is the right one. It
does not touch rows of other projects, of other visits or of participants outside
the cohorts, and it must be applied to the run lookup only: applied to the whole
file it would change the sample counts.
"""
import os
import numpy as np
import pandas as pd

_COHORTS = {}


def _cohorts(out):
    if out not in _COHORTS:
        cc = pd.read_csv(os.path.join(out, "analysis_complete_cases.csv"))
        d150 = pd.read_csv(os.path.join(out, "08_dataset_150h.csv"))
        d150 = d150[d150.complete]
        _COHORTS[out] = ((cc, 237, "saa_Fmax", "saa_TTT", "mean", "Fmax_24h_Rep", "TTT_24h_Rep"),
                        (d150, 155, "k_Fmax", "k_T50", "median", "Fmax_150h_Rep", "T50_150h_Rep"))
    return _COHORTS[out]


def row_of_kinetics(s, out):
    remove = []
    for co, proj, cf, ct, agg, rf, rt in _cohorts(out):
        r = s[(s.PROJECTID == proj) & (s.CLINICAL_EVENT == "BL") & s.PATNO.isin(co.PATNO)]
        if r.empty:
            continue
        f = getattr(r[[rf + str(i) for i in (1, 2, 3)]].apply(pd.to_numeric, errors="coerce"), agg)(axis=1)
        t = getattr(r[[rt + str(i) for i in (1, 2, 3)]].apply(pd.to_numeric, errors="coerce"), agg)(axis=1)
        ref = co.set_index("PATNO")
        ok = (np.isclose(f.to_numpy(), ref.loc[r.PATNO, cf].to_numpy(), rtol=1e-9)
              & np.isclose(t.to_numpy(), ref.loc[r.PATNO, ct].to_numpy(), rtol=1e-9))
        good = r[ok]
        assert good.PATNO.is_unique and set(good.PATNO) == set(r.PATNO), \
            'link between run and kinetic parameters not unique'
        remove += list(r.index[~ok])
    return s.drop(index=remove)
