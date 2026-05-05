# Paper Table Generation

This document maps **exact scripts used to generate paper tables** directly from the benchmark results.

All scripts in this section are the **original code from the paper**, not skeletons.

## Table Mapping

| Paper Table | Output File | Script | Purpose |
|---|---|---|---|
| **Table 1** | `environment_report.json` | (logged automatically) | System/model/environment versions |
| **Table 2** | `downstream_reliance_table.csv` | `evaluate_shortcut_predictions.py` | Main TSM (Truth-Shortcut Margin) for all encoders |
| **Table 3** | `embedding_probe_table.csv` | `audit_source_metadata_bias.py` | Representation analysis: SDR, ρ values |
| **Table 4** | `downstream_reliance_label_matched_table.csv` | `evaluate_with_uncertainty.py` | Tissue prediction with CIs |
| **Table 5** | `full_multiclass_downstream_table.csv` | `evaluate_celltype_prior_shortcut.py` | Multi-class prediction with prior analysis |
| **Table 6** | `correlation_geometry_*.csv` | `analyze_shortcut_mediation.py` | Embedding geometry: centroid, transfer, within-strata |
| **Table 7** | `embedding_information_*.csv` | `audit_source_metadata_bias.py` | Embedding information audits |
| **Table 8** | `c2s_reasoning_pairwise_*.csv` | `evaluate_pairwise_choice.py` | Cell2Sentence generation pairwise |
| **Table 9** | `cellwhisperer_context_query_*.csv` | `evaluate_unified_predictions.py` | CellWhisperer prompt intervention |
| **Table A** | `generative_reasoning_bias_*.csv` | `analyze_output_reasoning_stress.py` | Generative model reasoning bias |

## How to Regenerate Any Table

### Option 1: Use Pre-Computed Results (Fast)
```python
from results_loader import ResultsLoader
import pandas as pd

loader = ResultsLoader(Path('/datadisks/datadisk1/.../large_scale_max5000_20260423'))
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
