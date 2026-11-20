# Simulation results

Aggregate results of the simulations of the paper: reference (null) distributions, injected groups,
calibration and power of the decision rule, diagnostic curves. They contain no participant-level data.

These files were produced by the run that generated the results of the paper, with the original
(Italian) version of the code. File names, column names and text labels were then converted to
English with the dictionary used for the code; numbers were copied as text and are unchanged.
Labels that the code assembles from several pieces were converted piece by piece, so their wording
can differ slightly from what the scripts in `scripts/` write, and a few Italian words remain.
Files with one row per participant, checkpoint files and the text reports are not included:
running the code regenerates them.

## Injected groups: threshold crossing of the single criteria (Table S7, Figure 7)

- `22_power.csv` (14 kB)

## Copula-null distributions of the battery at k = 2 (Table S5, Figures 2 and 5)

- `25_dip_pvals_observed_150h.npy` (2 kB)
- `25_dip_pvals_observed_24h.npy` (2 kB)
- `25_nulls.csv` (2 kB)
- `25_null_distribution_150h.csv` (30 kB)
- `25_null_distribution_24h.csv` (50 kB)

## Specificity of the bootstrap Jaccard index

- `32_jaccard_specificity.csv` (1 kB)

## Injected groups: the point at separation zero

- `34_delta_zero.csv` (0 kB)

## Injected groups: uncertainty of the curves

- `37_covariance_injection.csv` (5 kB)
- `37_inversions.csv` (4 kB)
- `37_power_uncertainty.csv` (31 kB)

## Observed statistics against the copula null, single source (Table S5)

- `41_unified_nulls.csv` (2 kB)

## Diagnostic curves for k = 2 to 10 against the copula null

- `45_diagnostic_curves.csv` (13 kB)
- `45_gap_curves.csv` (3 kB)
- `45_gap_increment_curves.csv` (3 kB)
- `45_k_selected.csv` (0 kB)
- `45_prediction_strength_curves.csv` (3 kB)
- `45_silhouette_curves.csv` (3 kB)

## Calinski-Harabasz and Davies-Bouldin curves

- `57_ch_db_curves.csv` (7 kB)
- `57_ch_db_wide.csv` (6 kB)

## Proportion of ambiguous clustering against the copula null

- `79_pac_null.csv` (1 kB)
- `79_pac_null_distributions.npz` (3 kB)

## Less and more affected putamen: battery and silhouette against the null (Table S17)

- `110_battery_k2.csv` (15 kB)
- `110_battery_k2_nulls.csv` (193 kB)
- `110_laterality_definitions.csv` (5 kB)
- `110_models_laterality.csv` (60 kB)
- `110_ladder_laterality.csv` (27 kB)
- `110_silhouette_k2_B4000.csv` (2 kB)
- `110_silhouette_k2_B4000_null.csv` (477 kB)

## Replication cohort without the two type 2 participants: battery against the null (Table S19)

- `112_attribution.csv` (11 kB)
- `112_battery.csv` (23 kB)
- `112_battery_nulls.csv` (169 kB)
- `112_kinetics_by_time.csv` (40 kB)
- `112_csf_albumin.csv` (5 kB)
- `112_csf_pairs.csv` (29 kB)
- `112_two_participants.csv` (4 kB)
- `112_not_motor.csv` (14 kB)
- `112_summary.csv` (7 kB)
- `112_typing_24h.csv` (16 kB)

## Shared participants: agreement of the partitions, power and ceiling (Table S22)

- `117_shared_sample.csv` (1 kB)
- `117_shared_canonical.csv` (0 kB)
- `117_shared_k4.csv` (1 kB)
- `117_shared_partitions.csv` (4 kB)
- `117_shared_power.csv` (0 kB)
- `117_shared_ceiling.csv` (2 kB)

## Decision rule after removing disease duration and stage (Table S25)

- `118_associations.csv` (3 kB)
- `118_battery_k2.csv` (6 kB)
- `118_curves_k2_10.csv` (28 kB)
- `118_nulls.npz` (1069 kB)
- `118_partitions_vs_duration_stage.csv` (1 kB)
- `118_rule.csv` (2 kB)

## Shared participants analysed alone; cohorts without them

- `119_shared_alone_k2.csv` (2 kB)
- `119_without_shared_curves.csv` (10 kB)
- `119_without_shared_k2.csv` (2 kB)
- `119_without_shared_rule.csv` (1 kB)

## Checks on the shift without duration and stage (Table S26)

- `120_covariates_permuted.csv` (0 kB)
- `120_observed.csv` (0 kB)
- `120_series.csv` (1 kB)
- `120_silhouette_B10000.csv` (4 kB)

## Calibration of the decision rule (Supplement 1.22)

- `121_first_arm_outer.csv` (76 kB)
- `121_silhouette_outer.csv` (410 kB)
- `121_summary.csv` (5 kB)

## Partitions of the two assay versions one year apart: power

- `122_canonical.csv` (0 kB)
- `122_groups_variables.csv` (4 kB)
- `122_mean_of_wells.csv` (1 kB)
- `122_partitions.csv` (3 kB)
- `122_power.csv` (1 kB)
- `122_power_realistic.csv` (0 kB)

## Origin of the departure of the silhouette (Supplement 1.22)

- `123_blocks_data_real.csv` (7 kB)
- `123_calibration_two_test.csv` (559 kB)
- `123_origin.csv` (4 kB)
- `123_reactors_weak.csv` (2 kB)
- `123_removals_random.csv` (33 kB)
- `123_summary.csv` (3 kB)

## Run date and alternative models without groups (Supplement 1.22)

- `124_blocks_centred.csv` (7 kB)
- `124_calibrations.csv` (700 kB)
- `124_origin.csv` (4 kB)
- `124_permutations_dates.csv` (31 kB)
- `124_summary.csv` (6 kB)

## Power of the whole decision rule with injected groups (Table S29)

- `125_outcomes.csv` (2137 kB)
- `125_summary.csv` (15 kB)
