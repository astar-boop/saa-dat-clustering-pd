"""
01 - Construction of the analysis dataset.

Population: PD patients (COHORT = PD) at the baseline visit, with the Amprion
24 h SAA assay (PROJECTID 237), result Positive/Type1 -> the kinetic parameters
(in particular TTT) are defined only for the positive samples.

Sources:
  - SAA kinetic parameters : data/SAA_Biospecimen_Analysis_Results_24Apr2026.csv
  - DAT-SCAN, age, clinical data : PPMI_Curated_Data_Cut_Public_20260511.xlsx (sheet 20260511)

Output: outputs/analysis_dataset.csv  +  outputs/01_build_qc.txt
"""

import os
import numpy as np
import pandas as pd
from _paths import DATA, curated_xlsx

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURATED = curated_xlsx(BASE)
SAA = os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv")
OUT = os.path.join(BASE, "outputs")

qc = []


def log(msg):
    print(msg)
    qc.append(str(msg))


# ----------------------------------------------------------------------------
# 1. SAA kinetic parameters (24 h assay, Project 237)
# ----------------------------------------------------------------------------
saa = pd.read_csv(SAA)
log(f'SAA file: {saa.shape[0]} samples, {saa.PATNO.nunique()} subjects')

sel = saa[
    (saa.COHORT == "PD")
    & (saa.PROJECTID == 237)
    & (saa.CLINICAL_EVENT == "BL")
].copy()
log(f'  PD + 24h + BL: {len(sel)} samples, {sel.PATNO.nunique()} subjects')
log(f'  status x type:\n{pd.crosstab(sel.SAA_Status, sel.SAA_Type, dropna=False)}')

KIN = ["Fmax", "TTT", "AUC", "TSmax", "SLOPEMax"]
for f in KIN:
    cols = [f"{f}_24h_Rep{i}" for i in (1, 2, 3)]
    vals = sel[cols].apply(pd.to_numeric, errors="coerce")
    sel[f"saa_{f}"] = vals.mean(axis=1)              # mean of the 3 replicates
    sel[f"saa_{f}_nrep"] = vals.notna().sum(axis=1)  # n valid replicates
    # within-sample CV (technical variability, used only in QC/sensitivity)
    sel[f"saa_{f}_cv"] = vals.std(axis=1, ddof=1) / vals.mean(axis=1)

# primary population: type 1 positive SAA (LBD-like)
prim = sel[(sel.SAA_Status == "Positive") & (sel.SAA_Type == "Type1")].copy()
log(f'  -> Positive/Type1 at BL: {len(prim)} samples, {prim.PATNO.nunique()} subjects')

# a subject can have several samples at the same visit: the mean is kept
dupes = prim.PATNO.duplicated().sum()
log(f'  duplicate samples per subject: {dupes}')
kin_cols = [f"saa_{f}" for f in KIN] + [f"saa_{f}_cv" for f in KIN]
prim = (
    prim.groupby("PATNO", as_index=False)
    .agg({**{c: "mean" for c in kin_cols}, "SEX": "first", "SAA_Status": "first",
          "SAA_Type": "first", "RUNDATE": "first"})
    .rename(columns={"SEX": "saa_sex"})
)
log(f'  unique subjects with complete kinetic parameters: {len(prim)}')

# ----------------------------------------------------------------------------
# 2. Curated data cut: DAT-SCAN, age, clinical variables at BL
# ----------------------------------------------------------------------------
cur = pd.read_excel(CURATED, sheet_name="20260511")
log(f'\nCurated data cut: {cur.shape[0]} rows, {cur.shape[1]} columns')

cur_bl = cur[cur.EVENT_ID == "BL"].copy()
log(f'  BL rows: {len(cur_bl)}')

DAT = ["MIA_CAUDATE_L", "MIA_CAUDATE_R", "MIA_CAUDATE_BILAT",
       "MIA_PUTAMEN_L", "MIA_PUTAMEN_R", "MIA_PUTAMEN_BILAT",
       "MIA_STRIATUM_L", "MIA_STRIATUM_R", "MIA_STRIATUM_BILAT",
       "MIA_LOWPUT_EXPECTED"]
CLIN = ["COHORT", "SEX", "age", "age_at_visit", "age_datscan", "age_LP",
        "EDUCYRS", "agediag", "ageonset", "duration_yrs", "DOMSIDE",
        "PDTRTMNT", "LEDD", "hy", "NHY", "updrs1_score", "updrs2_score",
        "updrs3_score", "updrs_totscore", "pigd", "td_pigd", "moca", "upsit",
        "upsit_pctl", "ess", "rem", "gds", "stai", "scopa", "quip_any",
        "abeta", "tau", "ptau", "NFL_CSF", "nfl_serum", "urate",
        "APOE_e4", "fampd_bin", "subgroup", "NSD_STAGE", "Clinical_Stage",
        "CSFSAA", "CSFSAA_assay", "sym_tremor", "sym_rigid", "sym_brady",
        "sym_posins", "BMI", "cogstate", "MSEADLG"]
keep = ["PATNO"] + [c for c in DAT + CLIN if c in cur_bl.columns]
missing_cols = [c for c in DAT + CLIN if c not in cur_bl.columns]
if missing_cols:
    log(f'  WARNING columns missing in the curated data cut: {missing_cols}')
cur_bl = cur_bl[keep]

# ----------------------------------------------------------------------------
# 3. Merge
# ----------------------------------------------------------------------------
df = prim.merge(cur_bl, on="PATNO", how="left", validate="one_to_one")
log(f'\nMerge: {len(df)} subjects')
log(f'  not found in the curated data cut at BL: {df.COHORT.isna().sum()}')
log(f'  COHORT of the matched: {df.COHORT.value_counts(dropna=False).to_dict()} (1 = PD)')
# consistency of sex between the two sources (SEX curated: 0=Female, 1=Male)
chk = df.dropna(subset=["SEX"])
mism = ((chk.saa_sex == "Male") & (chk.SEX != 1)) | ((chk.saa_sex == "Female") & (chk.SEX != 0))
log(f'  sex discordances SAA-vs-curated: {mism.sum()}')

# ----------------------------------------------------------------------------
# 3bis. Exclusion of the genetic cohort (optional)
#
# ONLY_SPORADIC=1 restricts the population to sporadic PD, excluding all
# the subjects enrolled by genotype (LRRK2, GBA, SNCA, PRKN, other variants and
# combinations). The `subgroup` field of the curated data cut at BL is the source.
# No primary feature is genetic: the population changes, not the method.
# ----------------------------------------------------------------------------
if os.environ.get('ONLY_SPORADIC', "1") == "1":
    log('\nONLY_SPORADIC=1 — exclusion of the genetic cohort')
    log(f"  composition before the filter:\n{df.subgroup.fillna('unknown').value_counts().to_string()}")
    n_pre = len(df)
    df = df[df.subgroup == "Sporadic PD"].reset_index(drop=True)
    log(f'  excluded {n_pre - len(df)} carriers -> {len(df)} sporadic')
else:
    log('\nONLY_SPORADIC not active: full cohort (sporadic + carriers)')

# ----------------------------------------------------------------------------
# 4. Derived DAT features
# ----------------------------------------------------------------------------
df["dat_putamen_low"] = df[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
df["dat_putamen_high"] = df[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].max(axis=1)
df["dat_putamen_mean"] = df["MIA_PUTAMEN_BILAT"]
df["dat_caudate_low"] = df[["MIA_CAUDATE_L", "MIA_CAUDATE_R"]].min(axis=1)
df["dat_caudate_mean"] = df["MIA_CAUDATE_BILAT"]
df["dat_striatum_mean"] = df["MIA_STRIATUM_BILAT"]
# putaminal asymmetry index (0 = symmetric)
df["dat_put_asym"] = (df["dat_putamen_high"] - df["dat_putamen_low"]) / df["MIA_PUTAMEN_BILAT"]
# caudate-putamen gradient (rostro-caudal)
df["dat_cp_ratio"] = df["MIA_CAUDATE_BILAT"] / df["MIA_PUTAMEN_BILAT"]
# % of expected for age/sex (already adjusted for age -> validation only)
df["dat_lowput_pct_expected"] = df["MIA_LOWPUT_EXPECTED"]

df["age_bl"] = df["age_at_visit"]

# ----------------------------------------------------------------------------
# 5. Completeness for the primary feature set
# ----------------------------------------------------------------------------
PRIMARY = ["saa_Fmax", "saa_TTT", "saa_AUC", "saa_SLOPEMax",
           "dat_putamen_mean", "dat_caudate_mean", "age_bl"]
df["complete_primary"] = df[PRIMARY].notna().all(axis=1)
log(f'\nCompleteness of the primary features ({len(PRIMARY)} variables):')
log(df[PRIMARY].notna().sum().to_string())
log(f'  complete cases: {df.complete_primary.sum()} / {len(df)}')

# comparison of complete vs excluded (check of non-random missingness)
log('\nComplete vs excluded (means):')
cmp_vars = ["age_bl", "updrs3_score", "moca", "duration_yrs", "saa_TTT", "saa_Fmax"]
log(df.groupby("complete_primary")[cmp_vars].mean().to_string())

df.to_csv(os.path.join(OUT, "analysis_dataset.csv"), index=False)
log(f'\nSaved: outputs/analysis_dataset.csv  ({df.shape[0]} x {df.shape[1]})')

with open(os.path.join(OUT, "01_build_qc.txt"), "w") as fh:
    fh.write("\n".join(qc))
