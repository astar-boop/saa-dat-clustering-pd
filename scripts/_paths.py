"""Location of the PPMI raw data, declared in one place only.

Reason. The absolute path of the data folder was written out in twenty-four
scripts. Anyone who obtains the PPMI files could not point the code to them
without editing the scripts one by one. A location declared in twenty-four places
is not one location: it is a repeated copy, and one copy left behind is enough
for two scripts to read from different folders without anything flagging it.

The mechanism is the one CLUSTERING_BASE already provides for the working
directory: the environment variable wins, and a default applies otherwise.
With CLUSTERING_DATA the code reads from the folder it is given; without it, from
the `data` folder next to `scripts`.

The NAMES of the raw files stay where they are, in the scripts that read them,
and are not collected here, so that each script shows which files it needs.
"""

import os

DATA = os.environ.get('CLUSTERING_DATA',
                      os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))


NAME_CURATED = "PPMI_Curated_Data_Cut_Public_20260511.xlsx"


def curated_xlsx(base=None):
    """Path of the curated data cut: in the raw-data folder; failing that, in the
    working directory `base` (its historical location). If it is in neither, the
    path in the data folder is returned, so that the read error names the right place."""
    cand = [os.path.join(DATA, NAME_CURATED)] + ([os.path.join(base, NAME_CURATED)] if base else [])
    return next((p for p in cand if os.path.exists(p)), cand[0])
