# SC-ShortcutBench Dataset

## Overview

SC-ShortcutBench is a conflict-row benchmark for evaluating metadata shortcut reliance in single-cell foundation models. It's built on a subset of the CELLxGENE Census with 899,801 human cells across 457 independent studies.

## Dataset Description

### Source
- **CELLxGENE Census** (https://github.com/chanzuckerberg/cellxgene-census/)
- Human cells only (`organism == "homo_sapiens"`)
- Primary data only (`is_primary_data == True`)

### Coverage
| Dimension | Count |
|-----------|-------|
| Cells | 899,801 |
| Studies (datasets) | 457 |
| Tissue types | 26 |
| Disease classes | 28 (including healthy) |
| Assays | 12 |
| Cell types | 200+ (post-harmonization) |

### Train/Test Split
- **Training set**: 736,724 cells from 379 studies
- **Test set**: 163,077 cells from 78 held-out studies
- **Properties**: Zero `dataset_id` overlap between train and test (source-holdout)

## Metadata Fields

All cells have standardized metadata:

| Field | Description | Values |
|-------|-------------|--------|
| `dataset_id` | Source study identifier | ~457 unique values |
| `tissue_general` | Broad tissue category | 26 categories |
| `cell_type` | Cell type annotation | 200+ categories (harmonized to Cell Ontology) |
| `disease` | Disease state or "healthy" | 28 categories |
| `assay` | RNA sequencing assay | 12 categories (10x 3', SMART-seq, etc.) |
| `donor` | Individual donor ID | ~200 unique values |
| `development_stage` | Organism development stage | Multiple stages |
| `sex` | Reported sex | M/F/unknown |

## Conflict-Row Construction

Conflict rows are held-out cells where the metadata prior disagrees with the true biological label:

```
s(z) = argmax_y P_train(Y = y | Z = z)  [metadata prior]
Conflict: s(z) ≠ y_true
Aligned:  s(z) = y_true
```

### Algorithm (Section 4, Algorithm 1)

For each task (cell type, tissue, disease):

1. **Compute prior** P(Y | Z) on training data
2. **Shortcut label** s(z) = argmax_y P_train(Y = y | Z = z)
3. **Prior purity** π(z) = P_train(Y = s(z) | Z = z)
4. **Identify conflicts** where s(z) ≠ y_true in held-out set

### Splits

**Balanced Split**
- Test set marginal frequencies match training P(Z)
- Tests whether models exploit shortcuts when in-distribution
- Larger dataset (~35K balanced tissue conflicts)

**Decorrelated Split**
- Y is made approximately uniform within each Z
- Training-time correlation P(Y, Z) is broken
- Stronger diagnostic; tests true distribution shift
- Smaller dataset (~4K decorrelated tissue conflicts)

## Download & Access

### Option 1: Load from CELLxGENE Census API (automatic)

```python
from scripts.data_loading import load_cellxgene_subset

# Automatically downloads from Census
adata = load_cellxgene_subset(
    data_dir='./data',
    n_cells=None,  # Use all ~900K
    random_seed=42
)
```

Requires: `pip install cellxgene-census`

### Option 2: Load pre-processed h5ad

If you've already downloaded:

```python
import scanpy as sc
adata = sc.read_h5ad('data/cellxgene_census.h5ad')
```

### Option 3: Construct conflict rows from existing data

If you have your own CELLxGENE subset:

```python
from scripts.conflict_rows import ConflictRowBenchmark

benchmark = ConflictRowBenchmark(
    train_data=adata_train,
    test_data=adata_test,
    min_support=5
)

conflict_idx, aligned_idx, priors, stats = benchmark.construct(
    target='cell_type',
    conditioning_vars=['dataset_id', 'tissue_general'],
    split='decorrelated'
)
```

## Conflict-Row Properties

### Sizes (Table 1)

| Split | Task | Conflict Rows | Aligned Rows | Total |
|-------|------|---------------|--------------|-------|
| Balanced | cell type | 264 | 1,474 | 1,738 |
| Balanced | tissue | 14,775 | 20,000 | 34,775 |
| Balanced | disease | 35,000 | 35,000 | 70,000 |
| Decorrelated | cell type | 67 | 17 | 84 |
| Decorrelated | tissue | 4,264 | 3,209 | 7,473 |
| Decorrelated | disease | 13,237 | 6,852 | 20,089 |

### Prior Statistics

Metadata lookups alone achieve high accuracy:

- `dataset_id` → `tissue_general`: **88.4%** accuracy
- `cell_type` + `tissue` → `disease`: **82.5%** accuracy
- `dataset_id` → `assay`: **85.0%** accuracy

(See Table 2 in main text)

## Benchmark Tasks

Three tasks of increasing clinical relevance:

1. **Cell Type** (least clinically relevant)
   - Fine-grained biological classification
   - 26-28 classes depending on split

2. **Tissue** (primary evaluation surface)
   - Anatomical/organ classification
   - 26 tissue categories
   - Strongest shortcut signal due to study design

3. **Disease** (most clinically relevant)
   - Disease state prediction
   - 28 classes (diseases + healthy)
   - Weaker shortcut signal due to lower prior purity

## Usage in Models

### For Linear Probes (Representation Audit)

```python
from sklearn.linear_model import LogisticRegression

# Train probe on aligned cells
clf = LogisticRegression()
clf.fit(embeddings_train[aligned_idx], labels_train[aligned_idx])

# Evaluate on conflict rows
y_pred = clf.predict(embeddings_test[conflict_idx])
truth_acc = (y_pred == labels_test[conflict_idx]).mean()
shortcut_acc = (y_pred == shortcut_labels[conflict_idx]).mean()
tsm = truth_acc - shortcut_acc  # Should be negative if shortcut dominates
```

### For Generative Models

```python
# Score shortcut label vs true label for same cell
shortcut_score = model.score(cell, shortcut_label)
truth_score = model.score(cell, true_label)

# Shortcut preference if shortcut_score > truth_score
```

## Reproducibility

All conflict-row construction uses fixed random seed (`seed=42`) for reproducibility.

To reproduce the exact same splits:

```python
benchmark.construct(..., random_seed=42)
```

## Citation

If you use SC-ShortcutBench, please cite:

```bibtex
@article{sc_shortcutbench_2026,
  title={SC-ShortcutBench: A Conflict-Row Benchmark for Metadata Shortcut Reliance in Single-Cell Foundation Models},
  author={...},
  journal={NeurIPS 2026},
  year={2026}
}
```

## Contact & Questions

- Dataset issues: See [GitHub Issues](https://github.com/...)
- For model-specific evaluation help: See [troubleshooting guide](docs/TROUBLESHOOTING.md)
