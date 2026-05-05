# SC-ShortcutBench v2 — NeurIPS Hardening + Novel Mitigation Plan

Owner: any coding agent picking up this work after Khalil's interactive session.
Last updated: 2026-04-24 (initial plan).

Update 2026-04-24 11:26: the first NeurIPS rebuild scaffold is implemented.
The clean paper-facing layout, canonical max5000 symlink, checksum manifest,
one-command regeneration script, hierarchical CI script, permutation-null/BH
script, and initial SAGE CACE-lite mitigation script now exist. Current generated
outputs live under `results/tables/` and `results/reports/`.

This file is the single source of truth for the resubmission push. Tick the
checkboxes as steps are completed. Keep paths absolute. Keep generated artifacts
under `papers/paper_staging/v2_results_focused/generated/` and figures under
`papers/paper_staging/v2_results_focused/figures/`.

---

## 0. Context

### Project home

`/datadisks/datadisk1/khalil/sc_shortcut_project`

Read `history.md` first for layout. The current paper draft is
[papers/paper_staging/v2_results_focused/main.tex](papers/paper_staging/v2_results_focused/main.tex).
The full Reviewer-2 audit and prioritized experiment list is
[papers/paper_staging/v2_results_focused/REVIEWER2_PLAN.md](papers/paper_staging/v2_results_focused/REVIEWER2_PLAN.md).

### What the paper claims today

- Three-level theory: shortcuts in (1) representation, (2) downstream prediction,
  (3) generation/retrieval, in single-cell foundation models trained on CELLxGENE.
- Conflict-row protocol: rows where train-set metadata prior `s(z)` disagrees with
  held-out truth.
- Headline numbers: Cell2Text 80.3% shortcut on cell-type conflicts; encoder
  downstream 50-65% shortcut on decorrelated tissue conflicts; representation
  probes 88-95% recoverable for `dataset_id`.
- Models covered: Cell2Text (Llama 3.2 1B), C2S/C2S-Scale (Pythia 410M),
  CellWhisperer (CLIP), scGPT, Geneformer V2 104M, scFoundation, UCE 4-layer,
  scPoli.

### Why a NeurIPS reviewer (Reviewer 2) will reject as-is

1. Observational only — no causal manipulation of the prior.
2. Diagnosis only — no mitigation.
3. Bootstrap by `soma_joinid` undercounts donor/study clustering.
4. One classifier, one seed in the downstream probe.
5. One atlas, one species, three static tasks.
6. No mechanistic localization (layerwise, gene-level attribution).
7. Cell2Text "fails" on data you trained it on — straw-man risk.

### What this resubmission needs

A clean three-pillar story:

- **Diagnose** (already done; make statistically tighter).
- **Mechanism** (where in the network, which genes).
- **Mitigate** (novel method that recovers truth without retraining the foundation model).

---

## 1. Novel mitigation: SHORE

We will introduce a single named framework with three components. The framework
operates only on **frozen embeddings** — no foundation-model retraining required —
which is what makes it deployable for any of the eight models in the benchmark.

**SHORE = SHortcut-Oriented Recoverable Embeddings**

### 1.1 CACE — Conflict-Axis Conditional Erasure (novel)

**Insight.** Existing concept erasure (LEACE, INLP, RLACE) removes a direction
identified by the *label* of a single nuisance variable, typically `dataset_id`.
But the actual axis along which encoder embeddings collapse on conflict rows is
**task-specific** and is a **combination** of source, donor, assay, and tissue.

**Method.** For each (model, task, split):

1. From the *training* manifest only, take pairs of cells (no test conflict
   needed):
   - `A_y,z` = aligned cells with truth label `y` and shortcut feature `z` such
     that `s(z) = y` (the prior agrees with truth).
   - `B_y,z'` = cells with the same truth label `y` but a different shortcut
     feature `z'` such that `s(z') = y' != y`.
2. Compute the per-(y) class centroid difference:
   `u_y = mean_emb(A_y,*) - mean_emb(B_y,*)`.
3. Stack the {`u_y`} vectors and take a `top-k` PCA. The resulting subspace `U`
   is the "conflict-axis basin" — the directions that move the embedding into
   the source-typical region while truth is held fixed.
4. Project embeddings onto the orthogonal complement of `U` before training the
   downstream classifier:
   `e_clean = e - U U^T e`.

**Why this is novel.** LEACE/INLP need labels of the nuisance variable. CACE
needs only the truth-label structure of the training data. It directly targets
the "biology preserved, source not preserved" axis. Critically, the basin is
identified from training data alone — no peek at conflict test rows — so it is
honestly evaluable.

**Hyperparameters.** `k in {1, 4, 16, 32}`; report sensitivity.

### 1.2 PASTA — Prior-Aware Soft Test-time Abstention (novel)

**Insight.** A clinically deployed model should not silently follow the prior on
ambiguous inputs. It should defer.

**Method.** At inference time, given a new cell `x` and its embedding `e`:

1. Compute the downstream classifier softmax over labels: `p(y | e)`.
2. Estimate the train-set conditional prior `p_train(y | s(z) = z_pred)` where
   `z_pred` is the predicted shortcut variable from a separate frozen probe on
   `e`.
3. Define the **prior-divergence** score
   `delta(e) = KL(p(y|e) || p_train(y | z_pred))`.
4. If `delta(e) < tau`, the model's prediction is essentially the prior; abstain.
   Else return `argmax p(y|e)`.

The threshold `tau` is calibrated on a held-out set to satisfy a target coverage.

**Reportable quantities.** Coverage-vs-accuracy curve, coverage-vs-truth-shortcut
margin curve, area under the deferral-recovery curve (AUDR).

### 1.3 SCAR — Shortcut-Conditional Adversarial Regularizer (novel framing of HSIC)

**Insight.** When training the downstream classifier, encourage its output to be
statistically independent of the shortcut variable, conditional on truth.

**Method.** Train with a composite loss
`L = L_CE(p, y) + lambda * HSIC(p, z | y)`
where HSIC is the conditional Hilbert-Schmidt Independence Criterion estimated
on a Gaussian kernel (cheap, differentiable). Conditioning on `y` is what makes
this novel relative to standard HSIC-as-debiaser: we keep the parts of the
shortcut that are mediated by the true label and only remove the parts that are
not.

### 1.4 Baselines we compare against

- **Vanilla LR** (current paper baseline).
- **IPW** — sample reweighting by `1 / p_train(y | z)`.
- **LEACE** (Belrose et al. 2023) — closed-form linear concept erasure on
  `dataset_id` and on `assay`.
- **INLP** (Ravfogel et al. 2020) — iterative null-space projection.
- **Group DRO** (Sagawa et al. 2020) — worst-case group loss with `dataset_id`
  groups.
- **IRM** (Arjovsky et al. 2019) — environment = `dataset_id`.

### 1.5 Headline figure to produce

A single figure with three subplots:

- (a) Truth-shortcut margin (TSM) on decorrelated tissue conflicts before vs
  after each method, per encoder.
- (b) PASTA coverage-vs-truth curve.
- (c) SHORE compositional ablation: vanilla, +CACE, +CACE+SCAR, +CACE+SCAR+PASTA.

This figure is the single best NeurIPS hook because it shows that the same
benchmark that diagnoses the problem also measures the cure.

---

## 2. Statistical hardening (must-have)

These are quick wins and unblock everything else.

- [ ] **2.1 Hierarchical block bootstrap.** Replace `soma_joinid` resampling in
  [benchmark_code/sc_shortcutbench_v2/scripts/bootstrap_confidence_intervals.py](benchmark_code/sc_shortcutbench_v2/scripts/bootstrap_confidence_intervals.py)
  with two-level bootstrap: resample `dataset_id` with replacement, then within
  each `dataset_id` resample `donor_id`, then cells. Save donor/dataset map by
  joining the manifest at
  `/datadisks/datadisk1/khalil/cell2text_data/data/cellxgene_v3_balanced_manifest_broad_20260408/balanced_manifest_with_splits.csv`
  on `soma_joinid`. Output:
  - `papers/paper_staging/v2_results_focused/generated/hierarchical_ci_direct.csv`
  - `papers/paper_staging/v2_results_focused/generated/hierarchical_ci_downstream.csv`
  - `figures/fig_ci_widening_under_donor_resampling.pdf` (overlay old vs new CIs).

- [ ] **2.2 Permutation null + BH correction.** New script
  `benchmark_code/sc_shortcutbench_v2/scripts/permutation_null_bh.py`. For each
  (model, task, split) row, shuffle predictions, recompute shortcut agreement,
  return p-value. Apply BH at q=0.05 across the full claim family. Output:
  - `generated/permutation_pvalues_bh.csv`
  - paper-text snippet listing claims that survive BH at q=0.05.

- [ ] **2.3 Multi-classifier, multi-seed downstream probe.** Extend
  [benchmark_code/sc_shortcutbench_v2/scripts/run_downstream_label_matched_probe.py](benchmark_code/sc_shortcutbench_v2/scripts/run_downstream_label_matched_probe.py)
  to fit 5 seeds × 4 classifiers (`LogisticRegression`, `KNeighborsClassifier(15)`,
  linear SVM, `MLPClassifier(2-layer)`). Report mean ± std per cell.
  - `tables/downstream_multiclassifier_seeds.csv`
  - `figures/fig_downstream_classifier_robustness.pdf`.

- [ ] **2.4 Power analysis appendix.** Compute minimum-detectable TSM at 80%
  power per slice. Output: `tables/power_analysis_min_detectable_tsm.csv`.

---

## 3. Mechanistic localization (must-have)

- [ ] **3.1 Layerwise probing for all five encoders.**
  Use the existing `extract_layer_embeddings` paths in
  [benchmark_code/sc_shortcutbench_v2/model_interfaces/](benchmark_code/sc_shortcutbench_v2/model_interfaces/).
  For each transformer model, extract embeddings from layers
  `{0, 25%, 50%, 75%, 100%}` of depth, fit a linear probe for each of
  {`dataset_id`, `assay`, `donor_id`, `tissue_general`, `cell_type`, `disease`}.
  Output:
  - `tables/layerwise_probe_full.csv`
  - `figures/fig_layerwise_shortcut_localization.pdf` (one heatmap per model,
    layer × variable).

- [ ] **3.2 Gene-level attribution for the dataset_id probe.**
  For each encoder, run permutation importance on the *input gene token vector*
  (or HVG vector for non-tokenized models) for the `dataset_id` linear probe.
  Top-K shortcut genes per encoder. Cross-reference against:
  - housekeeping list (Eisenberg & Levanon 2013),
  - ribosomal genes (`RPS*`, `RPL*`),
  - mitochondrial genes (`MT-*`),
  - cell-type marker sets (CellMarker 2.0).
  Output:
  - `tables/shortcut_genes_top_k_per_encoder.csv`
  - `figures/fig_shortcut_gene_overlap_with_marker_sets.pdf`.

- [ ] **3.3 scIB cross-walk.**
  Compute kBET, batch-ASW, iLISI, cLISI on the same frozen embeddings using the
  `scib-metrics` package. Add columns to `embedding_probe_table.csv`. Plot
  scatter of SDR vs iLISI to show they correlate but SDR carries unique info.
  Output:
  - `tables/scib_vs_sdr.csv`
  - `figures/fig_scib_vs_sdr.pdf`.

---

## 4. Causal evidence (should-have)

- [ ] **4.1 Synthetic shortcut injection calibration.**
  Take the saved embeddings and concatenate a synthetic "leakage feature" of
  known strength `alpha in {0, 0.1, 0.3, 1.0, 3.0}` correlated with `dataset_id`.
  Refit downstream probe; recover SDR vs alpha curve.
  Output: `figures/fig_synthetic_leakage_calibration.pdf`.

- [ ] **4.2 Counterfactual metadata swap (text models).**
  For Cell2Text and C2S-Scale, swap the prompt's metadata field while keeping
  expression, rerun pairwise likelihood. If predictions change with metadata
  swap, the shortcut leaks via prompt; if not, via expression.
  Output: `tables/text_counterfactual_prompts.csv`.

- [ ] **4.3 Controlled prior dose-response.**
  Build five training subsets where prior strength `pi` for the
  `cell_type -> tissue_general` route is set to {0.5, 0.7, 0.85, 0.95, 1.0} by
  reweighted subsampling. Refit downstream probes for each. Plot
  `pi` vs downstream TSM. Output:
  - `tables/causal_prior_dose_response.csv`
  - `figures/fig_causal_prior_dose_response.pdf`.

---

## 5. Mitigation experiments (must-have, novel)

These are the load-bearing experiments. Implement under
`benchmark_code/sc_shortcutbench_v2/mitigation/`.

- [ ] **5.1 LEACE baseline.**
  Implement closed-form linear concept erasure (Belrose et al. 2023). New file
  `benchmark_code/sc_shortcutbench_v2/mitigation/leace.py`. Erase the
  `dataset_id` direction from saved embeddings. Refit the existing downstream
  probe. Output: appended rows to `tables/mitigation_results.csv` with
  `method=LEACE`.

- [ ] **5.2 INLP baseline.**
  Iterative null-space projection (Ravfogel et al. 2020). New file `inlp.py`.
  Same protocol. `method=INLP`.

- [ ] **5.3 IPW baseline.**
  Inverse-prior reweighting at downstream training time. Sample weights
  `1 / p_train(y | z)`. `method=IPW`.

- [ ] **5.4 Group DRO baseline.**
  Implement Sagawa-style worst-case group loss with environments = `dataset_id`.
  Use a small Adam loop on a 1-layer MLP downstream head.
  `method=GroupDRO`.

- [ ] **5.5 IRM baseline.**
  Standard IRMv1 penalty with environments = `dataset_id`.
  `method=IRM`.

- [ ] **5.6 CACE (novel).**
  Implement Section 1.1 of this plan. New file
  `benchmark_code/sc_shortcutbench_v2/mitigation/cace.py`. Hyperparameter sweep
  over `k in {1, 4, 16, 32}`. `method=CACE_k=*`.

- [ ] **5.7 SCAR (novel).**
  Implement Section 1.3 of this plan. New file `scar.py`. Train a 1-layer MLP
  downstream head with HSIC-conditional penalty. `lambda in {0.1, 1.0, 10.0}`.
  `method=SCAR_lambda=*`.

- [ ] **5.8 PASTA (novel).**
  Implement Section 1.2 as a wrapper around any of the above predictors. New
  file `pasta.py`. Generate coverage-vs-truth, coverage-vs-shortcut curves.
  Output: `figures/fig_pasta_coverage_curves.pdf`.

- [ ] **5.9 Composite SHORE.**
  Run CACE + SCAR + PASTA in combination. Report on the same conflict slices.

- [ ] **5.10 Headline mitigation figure.**
  Three-panel figure described in Section 1.5 of this plan.
  Output: `figures/fig_shore_headline.pdf`.

All mitigation results land in:
- `tables/mitigation_results.csv` with columns `model, task, split,
  subset, method, hyperparams, n, truth_accuracy, shortcut_agreement, TSM,
  CSRS, coverage` and CIs from the hierarchical bootstrap of step 2.1.

---

## 6. Strong baselines (should-have)

- [ ] **6.1 HVG + LightGBM expression-only baseline.**
  2000 HVGs from CELLxGENE, LightGBM, no foundation model. Output:
  `tables/hvg_lgbm_baseline.csv`.

- [ ] **6.2 Metadata-only LightGBM upper bound.**
  Train LightGBM on `cell_type, tissue, dataset_id, donor_id, assay,
  development_stage, sex` to predict the target. Upper bound on the shortcut.
  Output: `tables/metadata_only_lgbm_upper_bound.csv`.

- [ ] **6.3 CellTypist baseline.**
  Standard reference for cell-type prediction. Output: `tables/celltypist_baseline.csv`.

---

## 7. External validity (should-have)

- [ ] **7.1 Tabula Sapiens cross-atlas replication.**
  Build a held-out shortcut challenge from Tabula Sapiens v2. Run the five
  encoder probes on it. Output:
  - `tables/cross_atlas_tabula_sapiens.csv`
  - `figures/fig_cross_atlas_replication.pdf`.

- [ ] **7.2 Cross-species (mouse).**
  Tabula Muris, UCE and Geneformer-mouse. Output:
  `tables/cross_species_replication.csv`.

- [ ] **7.3 Leave-disease-out.**
  Hold out an entire disease (COVID-19), retrain downstream probe.
  Output: `tables/leave_disease_out.csv`.

- [ ] **7.4 Leave-institution-out.**
  Group `dataset_id` by institution.
  Output: `tables/leave_institution_out.csv`.

---

## 8. Generation / retrieval improvements (nice-to-have)

- [ ] **8.1 Ontology-aware text scoring** (CL/UBERON/MONDO).
- [ ] **8.2 Real BERTScore + GPT-4-as-judge factuality.**
- [ ] **8.3 Cell2Text retrained on decorrelated manifest.**
  Long run; LoRA fine-tune. Output:
  `tables/cell2text_decorrelated_finetune_results.csv`.

---

## 9. Sharper definitions (nice-to-have)

- [ ] **9.1 Distributional shortcut KL.**
  `KL(model_softmax || prior)` per row.
- [ ] **9.2 Conditional mutual information.**
  `I(prediction; shortcut_label | truth)`.
- [ ] **9.3 Top-2 shortcut definition.**
  Report when prediction is in {top-1, top-2} of the prior distribution rather
  than only argmax.

---

## 10. Within-family scaling (nice-to-have)

- [ ] **10.1 Geneformer 30M / 95M / 104M scale curve.**
- [ ] **10.2 scGPT human / pan-cancer / heart / blood / brain organ checkpoints.**
- [ ] **10.3 UCE 4-layer vs 33-layer.**

---

## 11. Release and reproducibility (must-have)

- [ ] **11.1 Canonical regenerate script.**
  `benchmark_code/sc_shortcutbench_v2/scripts/make_paper.sh` that writes
  every CSV/figure under `papers/paper_staging/v2_results_focused/` from a
  single `large_scale_max5000_20260423` run with checksum assertions.

- [ ] **11.2 HuggingFace dataset card** with shortcut metric definitions.

- [ ] **11.3 Environment lockfile** (conda + pip) shipped with the paper.

- [ ] **11.4 Public leaderboard JSON schema.**

---

## 12. Paper restructure (must-have, do last)

After steps 2-11 land:

- [ ] **12.1** Lift the label-matched protocol to be the primary protocol; demote
  `conflict_all` to a sensitivity appendix.
- [ ] **12.2** Replace cell-level CIs with hierarchical CIs throughout.
- [ ] **12.3** Add new sections:
  - Results III: Causal evidence (S1 results).
  - Results IV: Mechanism (S5 results).
  - Results V: Mitigation (S2 results, headline SHORE figure).
- [ ] **12.4** Limitations: still one snapshot for cross-atlas; functional tasks
  pilot scale; SHORE assumes linear shortcut subspace.
- [ ] **12.5** Regenerate `main.tex` figures from canonical run.

---

## 13. Recommended execution order

The following order gives the fastest path from current state to a NeurIPS-grade
draft:

1. Step 11.1 (canonical regenerate) — fix staleness first; everything below
   depends on it.
2. Step 2.1 (hierarchical bootstrap) — drops into existing scripts, widens CIs
   for all subsequent claims.
3. Steps 5.1, 5.2, 5.3 (LEACE, INLP, IPW baselines) — quick to implement; sets
   the comparison floor for SHORE.
4. Steps 5.6, 5.7, 5.8 (CACE, SCAR, PASTA) — the novel contribution.
5. Step 5.10 (headline mitigation figure) — the visual hook.
6. Steps 2.2, 2.3, 2.4 (permutation null, multi-classifier, power) — statistical
   defenses.
7. Steps 3.1, 3.2, 3.3 (layerwise, gene attribution, scIB) — mechanism.
8. Steps 5.4, 5.5 (Group DRO, IRM) — lit-comparison baselines.
9. Steps 6.1, 6.2 (LGBM baselines) — anchor the comparisons.
10. Steps 4.1, 4.2 (synthetic injection, counterfactual prompts) — causal evidence.
11. Step 7.1 (Tabula Sapiens) — external validity.
12. Step 12 (paper restructure).

Steps 4.3, 7.2, 7.3, 7.4, 8.\*, 9.\*, 10.\* are nice-to-have if compute and time
allow.

---

## 14. Conventions for whoever picks this up

- Use the conda env at `/home/khalil/miniconda3/envs/cell2text_new`.
- Always set `LD_LIBRARY_PATH=/home/khalil/miniconda3/envs/cell2text_new/lib`.
- Always set `NUMBA_DISABLE_JIT=1`.
- Saved embeddings live at
  `/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/embeddings/*.npz`
  with keys `{soma_joinid, embeddings}`.
- Predictions live at `.../predictions/*.jsonl`.
- The full metadata manifest is
  `/datadisks/datadisk1/khalil/cell2text_data/data/cellxgene_v3_balanced_manifest_broad_20260408/balanced_manifest_with_splits.csv`.
- The challenge files are at
  `/datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/benchmark/large_neurips_v1/data/`.

When closing out a step:

1. Tick the matching `[ ]` checkbox to `[x]` in this file.
2. Add one bullet under the step describing the artifact path and a one-line
   number summary (e.g., `mean TSM after CACE: -0.04 vs vanilla -0.51`).
3. Commit a single line to `history.md` describing the change in past tense.
4. Do not delete any existing files; only add new ones.

---

## 15. Open questions for the human

- **Q1.** Do we want the camera-ready paper to be NeurIPS main track or
  Datasets & Benchmarks? Affects how much we lean on the dataset release.
- **Q2.** Cell2Text retrain (step 8.3): is the LoRA pipeline stable enough on
  the broad decorrelated manifest? If not, skip and frame as future work.
- **Q3.** Tabula Sapiens v2 download: should we use the public h5ad or the
  CELLxGENE Census `tabula_sapiens` collection? The latter avoids gene-symbol
  alignment work.
- **Q4.** Compute budget: the must-have list (steps 2, 3.1-3.3, 5.1-5.10, 6.1-6.2,
  11) is roughly 2-3 days of GPU + 1 week of analysis. Confirmed?
