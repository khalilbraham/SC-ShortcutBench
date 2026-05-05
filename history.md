# Project History

Last updated: 2026-04-24T11:26:00+02:00

## 2026-04-24 NeurIPS rebuild scaffold

Implemented the paper-facing SC-ShortcutBench layout requested for the NeurIPS
rebuild:

- Added clean top-level symlinks/directories: `benchmark/`, `data/`,
  `runs/canonical/`, `results/`, `mitigation/`, `papers/neurips2026/`, and
  `docs/`.
- Canonicalized `large_scale_max5000_20260423` under
  `runs/canonical/large_scale_max5000_20260423` and wrote
  `runs/canonical/large_scale_max5000_20260423.sha256`.
- Added one-command regeneration:
  `bash benchmark/scripts/make_all_results.sh`.
- Added hierarchical source/donor/cell CI script:
  `benchmark/scripts/hierarchical_confidence_intervals.py`.
- Added permutation-null + BH correction script:
  `benchmark/scripts/permutation_null_bh.py`.
- Added initial SAGE CACE-lite mitigation:
  `benchmark/scripts/run_sage_cace.py` and `mitigation/sage/README.md`.
- Generated smoke/full-current outputs under `results/tables/` and
  `results/reports/canonical_result_manifest.md`.

## 2026-04-24 Full canonical experiment pass

Ran the feasible full experiment suite on the canonical max5000 artifacts:

- MOSAIC mitigation:
  `results/tables/mosaic_results.csv` and
  `results/tables/mosaic_trace_curves.csv`.
- SAGE/CACE-lite baseline:
  `results/tables/sage_cace_results.csv`.
- Synthetic shortcut calibration:
  `results/tables/synthetic_shortcut_calibration.csv`.
- Controlled prior dose-response:
  `results/tables/causal_prior_dose_response.csv`.
- Downstream hierarchical CIs with 200 bootstraps:
  `results/tables/hierarchical_ci_downstream.csv`.
- Label-matched permutation-null/BH results with 1000 permutations:
  `results/tables/permutation_pvalues_bh.csv`.

The full direct/native hierarchical CI pass is still an overnight job; the
interactive run was stopped during that stage and the launcher now defaults to
`SKIP_DIRECT_CI=1`.

## Canonical project home

Start every future coding-agent session from:

`/datadisks/datadisk1/khalil/sc_shortcut_project`

This is now the canonical workspace for the single-cell shortcut-learning / bias investigation, benchmark code, benchmark outputs, artifacts, and paper staging.

Compatibility path:

`/datadisks/datadisk1/khalil/DataDrivenDiscovery-v2/sc_shortcut_project`

The compatibility path above is a symlink to the canonical top-level project home.

## Consolidation completed

At `2026-04-23T23:56:09+02:00`, the project was consolidated into one place.

Moved into the canonical home:

- `investigations/shortcut_bias_20260421`
- `benchmark_code/sc_shortcutbench_v2`
- `benchmark_runs/sc_shortcutbench_v2_runs`
- `benchmark_artifacts/sc_shortcutbench_v2_artifacts`
- `papers/paper_staging`

Current sizes:

- `investigations/shortcut_bias_20260421`: `7.1G`
- `benchmark_code/sc_shortcutbench_v2`: `900K`
- `benchmark_runs/sc_shortcutbench_v2_runs`: `27G`
- `benchmark_artifacts/sc_shortcutbench_v2_artifacts`: `14G`
- `papers/paper_staging`: `2.9M`

## Backward-compatible old paths

The old locations were replaced by symlinks so existing scripts and habits still work:

- `/datadisks/datadisk1/khalil/cell2text/investigations/shortcut_bias_20260421`
- `/datadisks/datadisk1/khalil/DataDrivenDiscovery-v2/sc_shortcutbench_v2`
- `/datadisks/datadisk1/khalil/DataDrivenDiscovery-v2/sc_shortcutbench_v2_runs`
- `/datadisks/datadisk1/khalil/DataDrivenDiscovery-v2/sc_shortcutbench_v2_artifacts`
- `/datadisks/datadisk1/khalil/DataDrivenDiscovery-v2/paper_staging`

These old paths are no longer independent copies. They point into the canonical project home above.

## What lives where

### Investigations

Path:

`investigations/shortcut_bias_20260421`

Main contents:

- `benchmark/`
- `paper_neurips2026/`
- `release_package/`
- `repos/`
- `results/`
- `showcase/`
- `hf_cache/`
- `external_data/`

This is the main historical investigation tree, including benchmark construction, audits, older reports, paper material, and cached model/repo assets.

### Benchmark code

Path:

`benchmark_code/sc_shortcutbench_v2`

Main contents:

- benchmark package code
- model interfaces
- tasks
- scripts
- README

This is the active benchmark / analysis code used from the `DataDrivenDiscovery-v2` side.

### Benchmark run outputs

Path:

`benchmark_runs/sc_shortcutbench_v2_runs`

Observed run directories at consolidation time:

- `debug_large_wrapper_cap1`
- `full_gpu_20260423_024152`
- `full_scale_20260423`
- `large_scale_max5000_20260423`
- `missing_encoders_max100_20260423`
- `missing_encoders_smoke_20260423`

This is the main home for generated benchmark tables, predictions, figures, and run-specific outputs.

### Benchmark artifacts

Path:

`benchmark_artifacts/sc_shortcutbench_v2_artifacts`

Main contents:

- `embeddings/`
- `hf/`
- `repos/`
- `uce_model_files/`

This is the main home for reusable model artifacts and downloaded assets used by the benchmark runs.

### Paper staging

Path:

`papers/paper_staging`

Main contents:

- `v2_results_focused/`

This is the current focused paper-staging area used to generate figures/tables for the current argument set.

## Recent agent work before consolidation

### Shortcut-claim diagnosis

The main diagnosis from the recent review was:

- the direct-vs-downstream argument is real, but `conflict_all` mixes shortcut reliance with label-support failure
- `conflict_seen_both` is a cleaner subset because both truth and shortcut labels are present in aligned training
- `decorrelated tissue` remains the strongest clean downstream shortcut argument
- `balanced disease` is weaker as a pure shortcut-collapse argument once label visibility is controlled

### Label-matched downstream probe work

A new strict probe runner was added at:

`benchmark_code/sc_shortcutbench_v2/scripts/run_downstream_label_matched_probe.py`

Intent of that runner:

- keep only conflict rows where both truth and shortcut labels are present in aligned training
- retrain the downstream classifier only on aligned rows whose labels lie in that conflict label universe

Status at consolidation time:

- the first run attempt was interrupted by environment / dependency issues
- the runner still needs a final pass before it can be trusted as the canonical rerun path
- the next agent should inspect and finish this script before using its outputs in the paper

## Recommended starting checklist for the next agent

1. Start in `/datadisks/datadisk1/khalil/sc_shortcut_project`.
2. Read this file first.
3. Verify the label-matched runner in `benchmark_code/sc_shortcutbench_v2/scripts/run_downstream_label_matched_probe.py`.
4. If needed, rerun the strict label-matched downstream protocol on `benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423`.
5. Compare label-matched results against:
   - `conflict_all`
   - `conflict_seen_both`
   - direct representation probes
6. Update the focused paper assets under `papers/paper_staging/v2_results_focused`.

## Practical note

Use the canonical project home for all new edits, new outputs, new notes, and new agent sessions. Treat the old locations as compatibility entry points only.
