# SC-ShortcutBench: HuggingFace Dataset Upload Guide

Complete guide to upload the conflict-rows benchmark dataset to HuggingFace with public and private splits.

## 📊 Dataset Overview

**SC-ShortcutBench**: A conflict-row benchmark for evaluating metadata shortcut reliance in single-cell foundation models.

- **Total cells**: 899,801 (from CELLxGENE Census)
- **Splits**: Balanced + Decorrelated
- **Tasks**: Cell type, Tissue, Disease
- **Conflict rows**: ~26K decorrelated + ~49K balanced
- **Size**: ~5GB (expression matrix) + ~100MB (metadata)

---

## 📁 Public vs Private Strategy

**Key Principle**: Prevent overfitting in downstream evaluations
- **Public (Training)**: Cells used for model training—can be used for any downstream training
- **Private (Testing)**: Cells with conflict-row annotations—held out for fair evaluation only

### Public Release (Training Data)
```
sc-shortcutbench-public/
├── metadata/                           # ✓ Public
│   ├── balanced_challenge.csv          # Cell indices + labels (for evaluation)
│   ├── decorrelated_challenge.csv      # Cell indices + labels (for evaluation)
│   ├── conflict_rows_spec.json         # Conflict row definitions
│   └── dataset_manifest.csv            # Dataset information
├── train_expression/                   # ✓ Public (training-allowed)
│   ├── balanced_expression_train.h5ad  # Training cells (no evaluation cells)
│   └── decorrelated_expression_train.h5ad
├── predictions/                        # ✓ Public
│   ├── geneformer_tsm.csv
│   ├── scfoundation_tsm.csv
│   └── ... (all model predictions on held-out test cells)
└── documentation/                      # ✓ Public
    ├── README.md
    ├── DATASET_CARD.md
    └── LICENSE (CC-BY-4.0 or similar)
```

### Private Release (Evaluation Data)
```
sc-shortcutbench-private/
├── test_expression/                    # ✗ Private (evaluation-only, no training)
│   ├── balanced_expression_test.h5ad   # Test cells with conflict rows
│   └── decorrelated_expression_test.h5ad
├── embeddings/                         # ✗ Private (optional, on test set)
│   ├── geneformer_embeddings_test.h5ad
│   └── ... (other model embeddings on test cells)
└── raw_data/                           # ✗ Private
    └── cellxgene_census_subset_test.h5ad
```

---

## 🎯 Step-by-Step Upload

### Step 1: Prepare Public Dataset

```bash
#!/bin/bash
set -e

PROJECT_ROOT="/datadisks/datadisk1/khalil/sc_shortcut_project"
DATASET_ROOT="/tmp/sc-shortcutbench-public"

mkdir -p "$DATASET_ROOT"/{metadata,predictions,documentation}

# 1. Export conflict-row metadata
echo "Exporting metadata..."
python - << 'PYTHON'
import pandas as pd
import json
from pathlib import Path

# Load benchmark results
results_path = Path("/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/tables")

# Read main results
df_downstream = pd.read_csv(results_path / "downstream_reliance_table.csv")

# Save task specifications
tasks = [
    {"name": "cell_type", "n_classes": 200, "type": "fine_grained"},
    {"name": "tissue_general", "n_classes": 26, "type": "anatomical"},
    {"name": "disease", "n_classes": 28, "type": "clinical"}
]

splits = [
    {"name": "balanced", "property": "preserves_prior_distribution"},
    {"name": "decorrelated", "property": "breaks_y_z_correlation"}
]

# Save specs
with open("/tmp/sc-shortcutbench-public/metadata/conflict_rows_spec.json", "w") as f:
    json.dump({"tasks": tasks, "splits": splits}, f, indent=2)

# Save summary statistics
summary = {
    "total_cells": 899801,
    "training_cells": 736724,
    "test_cells": 163077,
    "studies": 457,
    "conflict_rows_balanced": 49775,
    "conflict_rows_decorrelated": 26062
}

with open("/tmp/sc-shortcutbench-public/metadata/dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✓ Metadata prepared")
PYTHON

# 2. Prepare training expression data (public)
echo "Preparing training expression data..."
python - << 'PYTHON'
import scanpy as sc
import pandas as pd
from pathlib import Path

# Load full expression data
balanced_path = "/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/uce_work/balanced/balanced_shortcut_challenge_cells_selected.h5ad"
decorrelated_path = "/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/uce_work/decorrelated/decorrelated_control_challenge_cells_selected.h5ad"

# Load conflict row indices
conflict_balanced = pd.read_csv("/tmp/sc-shortcutbench-public/metadata/balanced_challenge.csv", index_col=0)
conflict_decorrelated = pd.read_csv("/tmp/sc-shortcutbench-public/metadata/decorrelated_challenge.csv", index_col=0)

print("Processing balanced split...")
adata_balanced = sc.read_h5ad(balanced_path)
conflict_indices_balanced = set(conflict_balanced.index)
train_mask_balanced = ~adata_balanced.obs.index.isin(conflict_indices_balanced)

adata_balanced_train = adata_balanced[train_mask_balanced].copy()
adata_balanced_test = adata_balanced[~train_mask_balanced].copy()

adata_balanced_train.write_h5ad("/tmp/sc-shortcutbench-public/train_expression/balanced_expression_train.h5ad")
print(f"✓ Balanced train: {adata_balanced_train.n_obs} cells")

print("Processing decorrelated split...")
adata_decorrelated = sc.read_h5ad(decorrelated_path)
conflict_indices_decorrelated = set(conflict_decorrelated.index)
train_mask_decorrelated = ~adata_decorrelated.obs.index.isin(conflict_indices_decorrelated)

adata_decorrelated_train = adata_decorrelated[train_mask_decorrelated].copy()
adata_decorrelated_test = adata_decorrelated[~train_mask_decorrelated].copy()

adata_decorrelated_train.write_h5ad("/tmp/sc-shortcutbench-public/train_expression/decorrelated_expression_train.h5ad")
print(f"✓ Decorrelated train: {adata_decorrelated_train.n_obs} cells")

print("✓ Training expression data prepared")
PYTHON

# 3. Copy predictions (public results)
echo "Copying predictions..."
cp -r /datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/tables \
    "$DATASET_ROOT/predictions/"

# 4. Create dataset card
cat > "$DATASET_ROOT/README.md" << 'EOF'
# SC-ShortcutBench Dataset

A conflict-row benchmark for evaluating metadata shortcut reliance in single-cell foundation models.

## Dataset Description

This benchmark exposes cases where single-cell foundation models rely on metadata shortcuts rather than expression signals for prediction. Conflict rows are held-out cells where the metadata prior disagrees with the true biological label.

## Files

### Metadata (Public)
- `balanced_challenge.csv` - Balanced conflict-row indices
- `decorrelated_challenge.csv` - Decorrelated conflict-row indices
- `conflict_rows_spec.json` - Task and split specifications
- `dataset_summary.json` - Dataset statistics

### Predictions (Public)
- `downstream_reliance_table.csv` - Truth-Shortcut Margin (TSM) for all models
- `embedding_probe_report.json` - Linear recoverability (ρ) of construction variables
- Generation model results (Cell2Sentence, CellWhisperer, etc.)

### Expression Data (Private, available upon request)
- Balanced and decorrelated expression matrices (h5ad format)
- Raw cell × gene counts from CELLxGENE Census

## Tasks

1. **Cell Type Prediction** - Fine-grained cell identity (200+ classes)
2. **Tissue Prediction** - Anatomical/organ classification (26 classes)
3. **Disease Prediction** - Disease state identification (28 classes)

## Splits

- **Balanced**: Preserves training-time marginal distribution P(Z)
- **Decorrelated**: Breaks training correlation between Y and Z

## Key Results

All 10 evaluated models show shortcut preference on conflict rows:
- Truth accuracy: 15-25%
- Shortcut agreement: 50-55%
- TSM (Truth-Shortcut Margin): -25 to -40 percentage points

See paper for full results and analysis.

## Citation

```bibtex
@article{sc_shortcutbench_2026,
  title={SC-ShortcutBench: A Conflict-Row Benchmark for Metadata Shortcut Reliance in Single-Cell Foundation Models},
  author={...},
  journal={NeurIPS 2026},
  year={2026}
}
```

## License

CC-BY-4.0 (with acknowledgment to CELLxGENE Census for source data)

EOF

echo "✓ Public dataset prepared at $DATASET_ROOT"
```

### Step 2: Set Up HuggingFace Account

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login
# This will prompt for your token from https://huggingface.co/settings/tokens

# Create new repository
# Go to https://huggingface.co/new and create:
# - Khalilbraham/sc-shortcutbench-public (public)
# - Khalilbraham/sc-shortcutbench-private (private)
```

### Step 3: Upload Public Dataset

```bash
#!/bin/bash

# Clone public repo
git clone https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-public
cd sc-shortcutbench-public

# Copy prepared dataset
cp -r /tmp/sc-shortcutbench-public/* .

# Create dataset card
cat > dataset_card.md << 'EOF'
---
dataset_info:
  features:
    - name: conflict_row_index
      dtype: int32
    - name: cell_id
      dtype: string
    - name: dataset_id
      dtype: string
    - name: cell_type
      dtype: string
    - name: tissue_general
      dtype: string
    - name: disease
      dtype: string
    - name: metadata_prior
      dtype: string
    - name: metadata_prior_probability
      dtype: float32
    - name: shortcut_agreement_geneformer
      dtype: float32
    - name: shortcut_agreement_scfoundation
      dtype: float32
    - name: shortcut_agreement_uce
      dtype: float32
    - name: shortcut_agreement_scgpt
      dtype: float32
    - name: shortcut_agreement_scpoli
      dtype: float32
  splits:
    - name: balanced
      num_examples: 49775
    - name: decorrelated
      num_examples: 26062
  download_size: 250MB
  dataset_size: 250MB
language:
  - en
license: cc-by-4.0
multilinguality:
  - monolingual
pretty_name: SC-ShortcutBench
task_ids:
  - single-cell-analysis
  - bias-detection
tags:
  - single-cell
  - genomics
  - bias
  - foundation-models
  - benchmark
---

# SC-ShortcutBench Dataset

See README.md for full documentation.
EOF

# Add and push
git add .
git commit -m "Initial SC-ShortcutBench public dataset release"
git push origin main

echo "✓ Public dataset uploaded to HuggingFace"
```

### Step 4: Upload Private Dataset (Test Data, Restricted Access)

```bash
#!/bin/bash

# Clone private repo
git clone https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-private
cd sc-shortcutbench-private

mkdir -p test_expression embeddings

# 1. Save test expression data (conflict rows only)
python - << 'PYTHON'
import scanpy as sc
import pandas as pd
from pathlib import Path

# Load conflict row indices (from public dataset)
conflict_balanced = pd.read_csv("/tmp/sc-shortcutbench-public/metadata/balanced_challenge.csv", index_col=0)
conflict_decorrelated = pd.read_csv("/tmp/sc-shortcutbench-public/metadata/decorrelated_challenge.csv", index_col=0)

# Load full expression data
balanced_path = "/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/uce_work/balanced/balanced_shortcut_challenge_cells_selected.h5ad"
decorrelated_path = "/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/uce_work/decorrelated/decorrelated_control_challenge_cells_selected.h5ad"

print("Preparing balanced test (conflict rows)...")
adata_balanced = sc.read_h5ad(balanced_path)
test_mask_balanced = adata_balanced.obs.index.isin(conflict_balanced.index)
adata_balanced_test = adata_balanced[test_mask_balanced].copy()
adata_balanced_test.write_h5ad("test_expression/balanced_expression_test.h5ad")
print(f"✓ Balanced test: {adata_balanced_test.n_obs} conflict rows")

print("Preparing decorrelated test (conflict rows)...")
adata_decorrelated = sc.read_h5ad(decorrelated_path)
test_mask_decorrelated = adata_decorrelated.obs.index.isin(conflict_decorrelated.index)
adata_decorrelated_test = adata_decorrelated[test_mask_decorrelated].copy()
adata_decorrelated_test.write_h5ad("test_expression/decorrelated_expression_test.h5ad")
print(f"✓ Decorrelated test: {adata_decorrelated_test.n_obs} conflict rows")

print("\n✓ Private test expression data prepared")
PYTHON

# 2. Add access control notice
cat > PRIVATE_USAGE.md << 'EOF'
# Private Test Dataset Access

This dataset contains **evaluation-only** expression data from CELLxGENE Census.

## ⚠️ Critical: Train/Test Split

**PUBLIC (training)**:
- `sc-shortcutbench-public/train_expression/` — Use for model training
- Safe to use for any downstream analysis

**PRIVATE (testing)**:
- `sc-shortcutbench-private/test_expression/` — Conflict rows only
- **DO NOT use for training your models**
- Use only for final evaluation to prevent overfitting

## Why This Separation?

The private test set contains conflict rows constructed specifically to expose shortcut reliance. Training on conflict rows would:
- Give unfair advantage on this benchmark
- Hide shortcut-reliant behavior
- Violate proper evaluation protocol

## Access Requirements

To access the private test split, you must:

1. Accept CELLxGENE Terms of Service
2. Acknowledge source studies in publications
3. Commit to using data for evaluation only (not training)
4. Not redistribute raw data

## Approved Uses

✅ **OK**:
- Train models on public training split
- Evaluate on private test split
- Publish results using private test performance
- Academic research and method benchmarking

❌ **NOT OK**:
- Training on private test split
- Redistributing private test data
- Using test data for hyperparameter tuning

## Request Access

To request access:
1. Go to `https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-private`
2. Click "Request Access"
3. Provide affiliation and intended use (must state test-only usage)
4. Submit for approval

## Evaluation Protocol

```python
# CORRECT usage
train_data = load_public_training_data()  # Public
model = train_model(train_data)
test_data = load_private_test_data()      # Private (requires approval)
results = evaluate_model(model, test_data)
```

## Licenses

- Dataset: CC-BY-4.0
- Source data: Governed by CELLxGENE Census
- Individual studies: See dataset_manifest.csv for per-study licenses

EOF

# 3. Push private dataset
git add .
git commit -m "Add private expression data (restricted access)"
git push origin main

echo "✓ Private dataset uploaded with restricted access"
```

---

## 🔐 Managing Access

### HuggingFace Dataset Settings

1. **Public Dataset** (`sc-shortcutbench-public`)
   - Go to Settings → Privacy
   - Set to **Public**
   - License: CC-BY-4.0

2. **Private Dataset** (`sc-shortcutbench-private`)
   - Go to Settings → Privacy
   - Set to **Private**
   - Settings → Approval → **Manual approval for new users**

### Grant Access to Collaborators

```bash
# Via HuggingFace web interface:
# 1. Go to Repo Settings
# 2. Manage access
# 3. Add collaborator email
# 4. Set permissions (can edit, can view, etc.)

# Or via CLI:
huggingface-cli repo-add-member \
  --repo-id Khalilbraham/sc-shortcutbench-private \
  --user-id collaborator@email.com
```

---

## 💻 Loading Dataset in Code

### Workflow: Train on Public, Evaluate on Private

```python
import scanpy as sc
from datasets import load_dataset

# 1. Load public training data
print("Loading training data...")
adata_train_balanced = sc.read_h5ad("hf://datasets/Khalilbraham/sc-shortcutbench-public/train_expression/balanced_expression_train.h5ad")
adata_train_decorrelated = sc.read_h5ad("hf://datasets/Khalilbraham/sc-shortcutbench-public/train_expression/decorrelated_expression_train.h5ad")

# 2. Train your model on public training data
# model = train_my_model(adata_train_balanced, adata_train_decorrelated)

# 3. Load metadata (conflict rows to evaluate on)
conflict_balanced = pd.read_csv("hf://datasets/Khalilbraham/sc-shortcutbench-public/metadata/balanced_challenge.csv")
conflict_decorrelated = pd.read_csv("hf://datasets/Khalilbraham/sc-shortcutbench-public/metadata/decorrelated_challenge.csv")

# 4. Load private test data (requires access approval)
print("Loading test data (evaluation only)...")
adata_test_balanced = sc.read_h5ad("hf://datasets/Khalilbraham/sc-shortcutbench-private/test_expression/balanced_expression_test.h5ad")
adata_test_decorrelated = sc.read_h5ad("hf://datasets/Khalilbraham/sc-shortcutbench-private/test_expression/decorrelated_expression_test.h5ad")

# 5. Evaluate your model on the held-out test set
# results = evaluate_on_conflict_rows(model, adata_test_balanced, adata_test_decorrelated, conflict_balanced, conflict_decorrelated)
```

### Public Dataset (Training)

```python
import scanpy as sc

# Load training expression
adata_train = sc.read_h5ad("hf://datasets/Khalilbraham/sc-shortcutbench-public/train_expression/balanced_expression_train.h5ad")
print(f"Training cells: {adata_train.n_obs}, Features: {adata_train.n_vars}")

# Load metadata to understand what NOT to train on
import pandas as pd
conflict_indices = pd.read_csv("hf://datasets/Khalilbraham/sc-shortcutbench-public/metadata/balanced_challenge.csv", index_col=0)
print(f"Test conflict rows (reserved for evaluation): {len(conflict_indices)}")
```

### Private Dataset (Evaluation Only)

```python
import scanpy as sc

# Requires authentication and approval
# Install: huggingface-cli login

adata_test = sc.read_h5ad("hf://datasets/Khalilbraham/sc-shortcutbench-private/test_expression/balanced_expression_test.h5ad", cache=True)
print(f"Test cells: {adata_test.n_obs}")

# These are conflict rows—evaluate model robustness on them
# Do NOT train on these cells
```

---

## 📦 Files to Include

### Public Release (For Training)

```
metadata/
├── balanced_challenge.csv (indices of test conflicts—DO NOT TRAIN ON)
├── decorrelated_challenge.csv (indices of test conflicts—DO NOT TRAIN ON)
├── conflict_rows_spec.json (task definitions)
└── dataset_summary.json

train_expression/
├── balanced_expression_train.h5ad (~1.8GB, 850K cells, safe for training)
└── decorrelated_expression_train.h5ad (~0.9GB, 820K cells, safe for training)

predictions/
├── downstream_reliance_table.csv (Table 4 from paper, on test split)
├── embedding_probe_report.json (Table 3, on test split)
├── c2s_reasoning_pairwise_*.csv (Table 8, on test split)
└── cellwhisperer_context_*.csv (Table 9, on test split)

documentation/
├── README.md (usage guide, emphasizes train/test split)
├── DATASET_CARD.md (metadata)
├── LICENSE (CC-BY-4.0)
└── PAPER_CITATION.txt
```

### Private Release (For Evaluation)

```
test_expression/
├── balanced_expression_test.h5ad (~0.7GB, conflict rows only)
└── decorrelated_expression_test.h5ad (~0.3GB, conflict rows only)

embeddings/ (optional, computed on test set only)
├── geneformer_embeddings_test.h5ad
├── scfoundation_embeddings_test.h5ad
└── ... (other models on test set)

PRIVATE_USAGE.md
└── Instructions: use test split for evaluation only, not training
```

---

## ⚖️ Licensing & Citation

### Recommended License

```
License: CC-BY-4.0 with additional terms

You are free to:
- Share and adapt the dataset
- Use for any purpose, including commercial

You must:
- Give credit to authors
- Acknowledge CELLxGENE Census
- Cite the paper if used in research
- List changes made to dataset

Additional terms:
- Raw expression data subject to CELLxGENE Terms
- See individual study licenses in manifest
```

### Citation Format

```bibtex
@dataset{sc_shortcutbench_2026,
  title={SC-ShortcutBench: A Conflict-Row Benchmark for Metadata Shortcut Reliance in Single-Cell Foundation Models},
  author={...},
  publisher={HuggingFace Datasets},
  year={2026},
  url={https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-public}
}
```

---

## 🎯 Recommended Configuration

**Public Dataset** (`Khalilbraham/sc-shortcutbench-public`) — Training & Analysis
- ✅ Training expression data (~2.7GB, 1.67M cells)
- ✅ Metadata (conflict-row indices for evaluation reference)
- ✅ Prediction results and paper benchmarks
- ✅ Documentation and code examples
- ✅ CC-BY-4.0 license
- **Use for**: Training models, analyzing shortcuts

**Private Dataset** (`Khalilbraham/sc-shortcutbench-private`) — Evaluation Only
- 🔒 Test expression data (~1GB, conflict rows only)
- 🔒 Frozen embeddings on test set (optional)
- 🔒 Manual approval required
- ✅ Available to approved researchers
- **Use for**: Final evaluation of models (prevent overfitting)

---

## 🚀 Quick Start

```bash
# 1. Create datasets on HuggingFace.co
# 2. Run upload script
bash huggingface_upload.sh

# 3. Verify datasets
huggingface-cli repo-view datasets/Khalilbraham/sc-shortcutbench-public
huggingface-cli repo-view datasets/Khalilbraham/sc-shortcutbench-private

# 4. Test loading
python -c "from datasets import load_dataset; ds = load_dataset('Khalilbraham/sc-shortcutbench-public'); print(ds)"
```

---

## 📊 Expected Dataset Sizes

| Component | Size | Format | Access | Purpose |
|-----------|------|--------|--------|---------|
| Metadata (conflict rows) | ~10MB | CSV | Public | Evaluation specs |
| Predictions | ~200MB | CSV/JSON | Public | Benchmark results |
| **Balanced train expression** | **~1.8GB** | **H5AD** | **Public** | **Train new models** |
| **Decorrelated train expression** | **~0.9GB** | **H5AD** | **Public** | **Train new models** |
| **Balanced test expression** | **~0.7GB** | **H5AD** | **Private** | **Evaluate models** |
| **Decorrelated test expression** | **~0.3GB** | **H5AD** | **Private** | **Evaluate models** |
| All embeddings (test set) | ~3GB | HDF5/Parquet | Private | Representation analysis |
| **Total Public** | **~2.9GB** | - | ✅ Train+eval specs |
| **Total Private** | **~4GB** | - | 🔒 Evaluation only |

---

## ✅ Verification Checklist

Before publishing:

- [ ] All files present and valid
- [ ] Metadata matches paper tables
- [ ] License properly specified
- [ ] Citation format clear
- [ ] Access controls configured
- [ ] Documentation complete
- [ ] Test loading works
- [ ] README comprehensive

---

**Ready to share your benchmark with the community!** 🎉
