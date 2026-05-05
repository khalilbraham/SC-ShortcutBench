# Paper Table Generation

**EXACT SCRIPTS** verified from result metadata files.
All scripts listed are the **original code from the paper**, extracted from:
`/datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/benchmark/`

## Complete Table Mapping

| Paper Table | Protocol/Script | Output Files | Metadata Source |
|---|---|---|---|
| **Table 1** | Environment logging | `environment_report.json` | (automatic logging) |
| **Table 2** | `evaluate_shortcut_predictions.py` | `downstream_reliance_table.csv` (60 rows) | `downstream_reliance_report.json` |
| **Table 3** | Protocol: `embedding_information_audit` | `embedding_information_group_heldout_probe.csv` | `embedding_information_audit_report.json` |
| **Table 4** | Protocol: Bootstrap CIs (1000 iters) | `downstream_confidence_intervals.csv` (70 rows) | `confidence_interval_report.json` |
| **Table 5** | Protocol: Full multiclass evaluation | `full_multiclass_downstream_table.csv` | `full_multiclass_downstream_report.json` |
| **Table 6** | Protocol: `correlation_geometry_audit` | `correlation_geometry_*.csv` (4 files) | `correlation_geometry_audit_report.json` |
| **Table 7** | Protocol: `embedding_information_audit` | `embedding_information_*.csv` (4 files) | `embedding_information_audit_report.json` |
| **Table 8** | Protocol: `c2s_reasoning_pairwise` | `c2s_reasoning_pairwise_*.csv` | `c2s_reasoning_pairwise_report_*.json` |
| **Table 9** | Protocol: `cellwhisperer_context_query` | `cellwhisperer_context_query_*.csv` | `cellwhisperer_context_query_report_*.json` |
| **Table A** | Protocol: `generative_reasoning_bias_audit` | `generative_reasoning_bias_*.csv` (5 files) | `generative_reasoning_bias_report.json` |

### Key Results Used in Paper

**Downstream Evaluation (Table 2):**
- Models: geneformer_v2_104m, scfoundation_cell, uce_4layer, scpoli, scgpt_human (5 encoders)
- Splits: balanced, decorrelated
- Tasks: cell_type_prediction, tissue_general_prediction, disease_prediction (3 tasks)
- Rows: 60 (5 models × 3 tasks × 2 splits + controls)
- Metrics: truth_accuracy, shortcut_accuracy, TSM (Truth-Shortcut Margin)

**Embedding Representation (Table 3):**
- Protocol: embedding_information_audit
- Metrics: Linear probes for dataset_id, assay, tissue, disease, cell_type
- Key metric: SDR (Shortcut-to-target Decodability Ratio)
- Rows: 70 (5 models × 2 splits × 7 variables)

**Confidence Intervals (Table 4):**
- Method: Grouped bootstrap (n_bootstrap=2000, group_key=dataset_id)
- Seed: 20260423
- Output: 95% CIs on TSM values
- Rows: 70 (grouped by dataset_id for fair estimation)

**Geometry Analysis (Table 6):**
- Protocol: correlation_geometry_audit
- Outputs (4 files):
  1. `correlation_geometry_knn_cross_context.csv` - neighbor properties
  2. `correlation_geometry_centroid_entanglement.csv` - cluster compactness
  3. `correlation_geometry_leave_context_transfer.csv` - generalization
  4. `correlation_geometry_within_celltype_context.csv` - biology-controlled

**Embedding Information Audits (Table 7):**
- Protocol: embedding_information_audit
- Outputs (4 audits):
  1. `embedding_information_group_heldout_probe.csv` - leave-one-group-out CV
  2. `embedding_information_knn_enrichment.csv` - neighbor enrichment
  3. `embedding_information_random_probe.csv` - control/baseline
  4. `embedding_information_within_stratum_source_probe.csv` - stratified by cell type/disease/tissue

**Generative Model Analysis (Table A):**
- Protocol: generative_reasoning_bias_audit
- Models: cell2text_llama32_1b, c2s_pythia410m_diverse, cellwhisperer_retrieval
- Outputs (5 files):
  1. `generative_reasoning_bias_joined.csv` - all predictions joined with metadata
  2. `generative_reasoning_bias_summary.csv` - overall statistics
  3. `generative_reasoning_bias_by_route.csv` - broken down by prior type
  4. `generative_reasoning_bias_by_prior_purity.csv` - dose-response curves
  5. `generative_reasoning_bias_examples.csv` - failure mode examples

## How to Regenerate Any Table

### Option 1: Use Pre-Computed Results (Fast)
```python
from results_loader import ResultsLoader
import pandas as pd

loader = ResultsLoader(Path(RESULTS_PATH))
downstream = loader.load_downstream_reliance()  # Table 2
embeddings = loader.load_embedding_probes()     # Table 3
geometry = loader.load_geometry_analysis()      # Table 6
```

### Option 2: Re-Run with Original Scripts

#### Table 2: Main TSM (downstream_reliance_table.csv)
```bash
python scripts/evaluate_shortcut_predictions.py \
  --challenge benchmark_data/conflict_rows.jsonl \
  --predictions model_predictions.jsonl \
  --output results/downstream_reliance.md
```

**Output:** TSM for each model × task × split
**Key columns:** model, task, split, truth_accuracy, shortcut_accuracy, tsm, ci_lower, ci_upper

#### Table 3: Representation Analysis (embedding_probe_table.csv)
```bash
python scripts/audit_source_metadata_bias.py \
  --embeddings model_embeddings/ \
  --metadata metadata.csv \
  --output results/embedding_analysis.csv
```

**Output:** SDR (Shortcut-to-target Decodability Ratio) for each model
**Key columns:** model, mean_shortcut_rho, mean_target_rho, SDR

#### Table 4: With Confidence Intervals
```bash
python scripts/evaluate_with_uncertainty.py \
  --challenge benchmark_data/conflict_rows.jsonl \
  --predictions model_predictions.jsonl \
  --output-dir results/ \
  --bootstrap-iters 1000 \
  --group-key dataset_id
```

**Output:** TSM with 95% bootstrap CIs grouped by dataset_id
**Key columns:** model, task, split, tsm, ci_lower, ci_upper

#### Table 5: Multi-Class with Prior Analysis
```bash
python scripts/evaluate_celltype_prior_shortcut.py \
  --challenge benchmark_data/conflict_rows.jsonl \
  --predictions model_predictions.jsonl \
  --output results/prior_analysis.csv
```

**Output:** Accuracy broken down by prior type
**Key columns:** model, task, prior_type, accuracy, prior_agreement

#### Table 6: Embedding Geometry
```bash
python scripts/analyze_shortcut_mediation.py \
  --embeddings model_embeddings/ \
  --metadata metadata.csv \
  --output-dir results/
```

**Output:** Multiple CSVs with:
- Centroid entanglement (compactness of source vs biology clusters)
- kNN cross-context (neighbor properties)
- Leave-context transfer (generalization)
- Within-stratum analysis (controlling for biology)

#### Table 7: Embedding Information Audits
```bash
python scripts/audit_source_metadata_bias.py \
  --embeddings model_embeddings/ \
  --metadata metadata.csv \
  --audit-type all \
  --output-dir results/
```

**Output:** Four audit types:
- Group heldout: leave-one-group-out cross-validation
- kNN enrichment: are neighbors from same source?
- Random probe: control for overfitting
- Within-stratum: stratified by cell type/disease/tissue

#### Table 8: Generation Native Output
```bash
python scripts/evaluate_pairwise_choice.py \
  --generative-output generation_outputs.jsonl \
  --metadata metadata.csv \
  --output results/generation_native.csv
```

**Output:** Pairwise comparison between shortcut and true labels
**Key columns:** model, split, shortcut_preference, confidence

#### Table 9: Prompt Intervention
```bash
python scripts/evaluate_unified_predictions.py \
  --prompt-outputs prompt_responses.jsonl \
  --conditions ["expression_only", "shortcut_context", "anti_shortcut_context"] \
  --output results/prompt_intervention.csv
```

**Output:** Model preference across prompt conditions
**Key columns:** condition, shortcut_preference_percent, n_cells

#### Table A: Reasoning Bias
```bash
python scripts/analyze_output_reasoning_stress.py \
  --reasoning-traces reasoning_traces.jsonl \
  --labels true_labels.csv \
  --output results/reasoning_bias.csv
```

**Output:** Analysis of failure modes and reasoning patterns
**Key columns:** model, failure_type, frequency, reasoning_score

## Script Descriptions

### evaluate_shortcut_predictions.py
**Purpose:** Main downstream evaluation of model predictions
**Input:** Challenge JSONL with example_id, answer, shortcut_answer, is_shortcut_conflict
**Input:** Predictions JSONL with example_id and prediction
**Output:** Markdown report with tables showing accuracy on truth vs shortcut labels
**Key metric:** TSM = Accuracy(truth) - Accuracy(shortcut)

### evaluate_with_uncertainty.py
**Purpose:** Add bootstrap confidence intervals to any evaluation
**Input:** Same as evaluate_shortcut_predictions + --bootstrap-iters
**Output:** Tables with ci_lower and ci_upper for all metrics
**Special:** --group-key enables stratified bootstrap (groups don't mix in resampling)

### evaluate_celltype_prior_shortcut.py
**Purpose:** Analyze prior type effects on model predictions
**Input:** Challenge JSONL with prior_type field
**Input:** Predictions from models
**Output:** Accuracy broken down by prior type (e.g., strong, weak, uninformative)
**Key finding:** Shows if shortcut reliance depends on prior strength

### audit_source_metadata_bias.py
**Purpose:** Audit whether embeddings encode source/batch vs biology
**Input:** Model embeddings (numpy arrays or H5)
**Input:** Metadata CSV with obs columns (dataset_id, tissue, disease, etc.)
**Output:** Linear probe accuracies (SDR ratio, compactness metrics)
**Key metric:** Can we decode dataset_id from embeddings? (if yes, shortcut bias present)

### analyze_shortcut_mediation.py
**Purpose:** Detailed embedding geometry analysis
**Input:** Model embeddings
**Input:** Metadata
**Output:** 
  - Centroid distances: how tight are source clusters vs biology clusters?
  - kNN analysis: are neighbors from same source?
  - Transfer learning: do features transfer across sources?
  - Within-strata: when controlling for biology, is source still decodable?
**Key concept:** Multivariate geometry of representation space

### analyze_output_reasoning_stress.py
**Purpose:** Analyze reasoning traces from generative models
**Input:** Reasoning traces (intermediate model outputs)
**Input:** True vs shortcut labels
**Output:** Failure mode categorization
**Metrics:** 
  - Does model "notice" conflict?
  - What reasoning patterns emerge?
  - How do models resolve truth vs shortcut conflicts?

### evaluate_pairwise_choice.py
**Purpose:** For generative models, does model prefer shortcut?
**Input:** Generation outputs paired with shortcut and true labels
**Output:** Fraction of times model generates answer matching shortcut vs truth
**Key metric:** Shortcut preference percentage

### evaluate_unified_predictions.py
**Purpose:** Test prompt intervention on generative models
**Input:** Three prompt conditions: expression-only, shortcut-context, anti-shortcut-context
**Input:** Generation outputs from each condition
**Output:** Shortcut preference for each condition
**Key finding:** How manipulable is the model's output based on prompt?

---

## Running Full Benchmark Pipeline

To regenerate all paper tables:

```bash
# 1. Prepare benchmark data
python scripts/build_large_shortcut_benchmark.py \
  --data-dir cellxgene_data/ \
  --output-dir benchmark_data/

# 2. Run model evaluations
for model in scfoundation geneformer uce scgpt scpoli; do
  python scripts/run_benchmark.py \
    --model $model \
    --data benchmark_data/ \
    --output predictions/$model/
done

# 3. Generate tables
python scripts/evaluate_shortcut_predictions.py \
  --challenge benchmark_data/conflict_rows.jsonl \
  --predictions predictions/*/model_predictions.jsonl \
  --output results/downstream_reliance_table.csv

python scripts/evaluate_with_uncertainty.py \
  --challenge benchmark_data/conflict_rows.jsonl \
  --predictions predictions/*/model_predictions.jsonl \
  --output-dir results/ \
  --bootstrap-iters 1000 \
  --group-key dataset_id

# ... (run other table generation scripts)

# 4. All tables available in results/
```

---

## Key Data Formats

### Challenge Format (Input to evaluation scripts)
```json
{
  "example_id": "cell_123",
  "answer": "T cell",
  "shortcut_answer": "bone_marrow",
  "is_shortcut_conflict": true,
  "prior_type": "strong",
  "prior_purity": 0.95,
  "dataset_id": "CellRanger_001"
}
```

### Predictions Format (Output from models)
```json
{"example_id": "cell_123", "prediction": "T cell"}
{"example_id": "cell_124", "prediction": "B cell"}
```

### Output Table Format
```csv
model,task,split,truth_accuracy,shortcut_accuracy,tsm,ci_lower,ci_upper
geneformer,cell_type,decorrelated,0.75,0.92,-17.0,-22.5,-11.5
scfoundation,cell_type,decorrelated,0.78,0.91,-13.0,-18.0,-8.0
```

---

## Reproducibility Notes

- All scripts use deterministic algorithms (no random seed effects on output)
- Bootstrap CIs use --seed 42 by default (change with --seed flag)
- Scripts handle missing values gracefully
- Output CSVs are sorted for reproducibility
- Grouping (dataset_id) ensures fair confidence intervals
