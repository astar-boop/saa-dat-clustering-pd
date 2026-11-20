# Clustering of SAA kinetics, DAT imaging and age in Parkinson's disease

Analysis code for a study of whether α-synuclein seed amplification assay (SAA) kinetic parameters,
dopamine transporter (DAT) imaging and age define subgroups among SAA-positive participants with
sporadic Parkinson's disease in the Parkinson's Precision Medicine Initiative (PPMI, formerly the
Parkinson's Progression Markers Initiative) cohort.

The repository contains the code and the aggregate results of the simulations. It contains no PPMI
data and no participant-level results.

## Status

The code was written in Italian and translated into English mechanically, in four checked stages
(comments, file names, printed messages, identifiers and labels). Each stage is verified on the
syntax tree so that nothing but the translated text changes.

- Steps 0–7 of the chain have been re-run from scratch with this English version and reproduce the
  outputs of the original code (32 of 32 output files; five figures).
- The remaining steps have **not yet been re-run** in this version. A complete re-run with a
  comparison of every output is still to be done.
- Fifteen places seed a random stream from the text of a label. `scripts/_legacy.py` rebuilds the
  original label so that the streams do not change; this has not yet been verified on the numbers.
- Some identifiers are awkward because the renaming is mechanical (for example the options `--fr`
  and `--a` of `run_all.py`, which mean "from step" and "to step").

## Simulation results

`simulation_results/` holds the aggregate results of the simulations of the paper (reference
distributions, injected groups, calibration and power of the decision rule, diagnostic curves):
81 files, about 7 MB, listed in `simulation_results/README.md`. They contain no participant-level
data. They come from the run that produced the results of the paper; names and labels were
converted to English, numbers are unchanged.

## Data

The data come from PPMI (www.ppmi-info.org) and cannot be redistributed. After obtaining access,
place these files in a folder named `data/` next to `scripts/`, or point `CLUSTERING_DATA` to the
folder that holds them:

- `PPMI_Curated_Data_Cut_Public_20260511.xlsx`
- `SAA_Biospecimen_Analysis_Results_24Apr2026.csv`
- `Current_Biospecimen_Analysis_Results_07May2026.csv`
- `Kinetic_data_for_SAA_Results_24Apr2026.csv`
- `MDS-UPDRS_Part_III_24Apr2026.csv`
- `DAT.csv`

## Running

    pip install -r requirements-lock.txt
    python3 run_all.py              # the whole chain, in the order of ORDER.csv
    python3 run_all.py --fr 0 --a 7 # steps 0 to 7 only
    python3 run_all.py --listing    # print the steps without running them

Environment variables, all optional:

| Variable | Meaning | Default |
| --- | --- | --- |
| `CLUSTERING_DATA` | folder with the PPMI files | `data/` |
| `CLUSTERING_BASE` | working directory; receives `outputs/`, `figures/`, `logs/` | this folder |
| `ONLY_SPORADIC` | `1` = sporadic participants only (the population of the paper); `0` = all | `1` |
| `N_PROCESSES` | parallel processes for the scripts that use them | script-specific |
| `TRIAL` | `1` = short trial run of the long simulations, no final output | `0` |

The chain stops at the first script that exits with an error; each script writes its own log in
`logs/`. The whole chain takes roughly a day of computing on a 12-core machine; the simulation of
the power of the decision rule (`125_rule_power.py`) alone takes about ten hours.

Results were produced with Python 3.13.5 and the library versions in `requirements-lock.txt`.

## Layout

- `run_all.py` — runs the scripts in order.
- `ORDER.csv` — the order, derived from which script reads the output of which.
- `scripts/` — one script per analysis, numbered; modules whose name starts with `_` are shared helpers.
- `figure/` — the three figures that were redrawn for the revised supplement.
- `simulation_results/` — aggregate results of the simulations (see above).

## Which script produces what

Table and figure numbers are those of the revised manuscript and supplement.

| Step | Script | Produces |
| --- | --- | --- |
| 1 | `01_build_dataset` | — |
| 2 | `02_features_and_tendency` | Table S3 |
| 3 | `03_clustering` | §1.4-1.5 |
| 4 | `04_characterization` | §1.7 |
| 5 | `05_figures` | Figure 1, Figure 3, Figure 4, Figure S2 |
| 6 | `06_sensitivity` | §1.6 |
| 7 | `07_figure_extended` | Figure 6 |
| 8 | `08_replication_150h` | Table S4, §1.8 |
| 9 | `09_followup_150h` | §1.8 |
| 10 | `10_figure_150h` | Figure S3 |
| 11 | `11_longitudinal` | §1.12 |
| 12 | `12_csf_concentration` | Table S12 |
| 13 | `13_moca_signal` | — |
| 14 | `14_figure_longitudinal` | Figure 8 |
| 15 | `15_nonmotor_trajectories` | Table S9 |
| 16 | `16_figure_trajectories` | Figure S4-S5 |
| 17 | `20_albumin` | Table S13 |
| 18 | `21_review_P1` | Figure S6, Table S11 |
| 19 | `22_injection_power` | Table S7 |
| 20 | `23_review_P17_P19` | — |
| 21 | `24_multiverse` | Figure S6 |
| 22 | `25_monte_carlo_nulls` | Table S5 |
| 23 | `28_flow_figure` | Figure S1 |
| 24 | `29_parameters_figure` | Figure S7 |
| 25 | `32_jaccard_specificity` | §1.4-1.5 |
| 26 | `34_delta_zero` | Table S7 |
| 27 | `36_loo_component` | Table S12 |
| 28 | `37_power_uncertainty` | Table S7 |
| 29 | `41_unified_nulls` | Table S5 |
| 30 | `26_review_figures` | Figure 5, Figure 7, Figure S6 |
| 31 | `40_tendency_figure` | Figure 2 |
| 32 | `45_diagnostic_curves` | §1.4-1.5 |
| 33 | `57_ch_db_curves` | §1.4-1.5 |
| 34 | `66_external_review` | Table S1, §1.8 |
| 35 | `69_run_variance_components` | Table S6 |
| 36 | `70_baseline_table` | Table S2 |
| 37 | `71_centre_effects` | Table S6 |
| 38 | `72_longitudinal_refit` | §1.12, Table S9 |
| 39 | `75_diagnostic_family` | §1.4-1.5, Table S14 |
| 40 | `76_denominators` | Table S1, Table S11 |
| 41 | `77_fmax_dat_batch` | §1.12 |
| 42 | `78_longitudinal_sensitivity` | Table S10 |
| 43 | `79_pac_null` | §1.4-1.5 |
| 44 | `80_convergence` | §1.12 |
| 45 | `81_median_reliability_v2` | Table S11 |
| 46 | `83_spline_reconciled` | Table S10 |
| 47 | `84_gradients` | Table S8 |
| 48 | `85_run_centre` | Table S6 |
| 49 | `87_amplitude_ladder` | Table S8 |
| 50 | `88_dat_correlation` | §1.12 |
| 51 | `88_diagnostics_63` | §1.12 |
| 52 | `88_ipcw_and_denominators` | Table S10 |
| 53 | `88_ladder_df` | Table S10 |
| 54 | `89_reliability_by_sample` | Table S11 |
| 55 | `91_table6_strict_criterion` | Table S9 |
| 56 | `92_ladder_diagnostics` | Table S8 |
| 57 | `93_table8_rules` | §1.12 |
| 58 | `95_strict_likelihoods` | §1.12 |
| 59 | `99_albumin_table` | Table S13 |
| 60 | `100_albumin_primary_cohort` | Table S13 |
| 61 | `103_dat_paragraph_numbers` | §1.12 |
| 62 | `104_dat_variants_comparison` | §1.6 |
| 63 | `105_caudate_optimiser_sensitivity` | §1.12 |
| 64 | `106_protocol_agreement` | Table S16 |
| 65 | `107_nonmotor_ledd_duration` | Table S9b |
| 66 | `108_availability_and_timing` | Table 1, Table S20-S21, Table S15 |
| 67 | `109_coughlin_reconciliation` | Table S16 |
| 68 | `110_dat_laterality` | Table S17 |
| 69 | `111_lateralised_motor` | Table S18 |
| 70 | `112_replication_without_type2` | Table S19 |
| 71 | `115_negatives_kinetics` | Results 2.4 |
| 72 | `116_assay_categories` | §1.6, Table S23 |
| 73 | `117_shared_partitions` | Table S22 |
| 74 | `118_duration_stage` | Table S24-S25, Table S26 |
| 75 | `119_shared_alone` | §1.8 |
| 76 | `120_duration_stage_checks` | Table S26 |
| 77 | `121_rule_calibration` | §1.22 |
| 78 | `122_partitions_one_year` | Table S22 |
| 79 | `123_silhouette_departure_origin` | §1.22 |
| 80 | `124_run_and_models_without_groups` | §1.22 |
| 81 | `125_rule_power` | Table S29 |
| 82 | `126_assay_runs` | Table S27-S28, Figure S8 |
| 83 | `127_reviewer3_letter_numbers` | Table S6, Table S29, Table S27-S28 |
| 84 | `figure/fig_S1` | Figure S1 |
| 85 | `figure/fig_S6` | Figure S6 |
| 86 | `figure/fig_S8` | Figure S8 |
