# Extracting Paper Results & Model Code

Complete guide to extract and integrate the actual evaluation code and results from your existing runs.

## 📊 Where the Paper Results Live

### Main Results Location
```
/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/
```

### Key Result Files

| Result | Location | Rows | Models |
|--------|----------|------|--------|
| **Downstream TSM** | `tables/downstream_reliance_table.csv` | 62 rows | 5 encoders × 3 tasks × 2 splits + controls |
| **Embedding Probes** | `tables/embedding_probe_report.json` | - | ρ(Z; φ) values |
| **Generation Native** | `tables/c2s_*_report_*.json` | - | Cell2Sentence, Cell2Text |
| **Generation CW** | `tables/cellwhisperer_*_report_*.json` | - | CellWhisperer |
| **Geometry Analysis** | `tables/correlation_geometry_*.csv` | 4 files | Within-strata, kNN, transfer |
| **Confidence Intervals** | `tables/confidence_interval_report.json` | - | All bootstrap statistics |
| **Predictions** | `predictions/*.jsonl` | 20 files | Raw model outputs |

### Full Models Evaluated

**Encoders** (5):
- `geneformer_v2_104m`
- `scfoundation_cell`
- `uce_4layer`
- `scpoli`
- `scgpt_human`

**Generation** (5):
- `c2s_pythia410m_diverse` (Cell2Sentence)
- `cell2text_llama32_1b` (Cell2Text)
- `cellwhisperer_clip_v1` (retrieval)
- `cellwhisperer_clip_v1_unified` (unified)
- `scgpt_human` (generative)

---

## 🔧 Evaluation Code Location

All evaluation scripts are in:
```
/datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/benchmark/
```

### Key Evaluation Scripts

```
benchmark/
├── evaluate_shortcut_predictions.py        ← Main downstream evaluation (TSM)
├── evaluate_metadata_prior_shortcut.py     ← Metadata priors (Table 2)
├── evaluate_pairwise_choice.py             ← Generation pairwise scoring
├── evaluate_unified_predictions.py         ← Generation unified
├── evaluate_shortcut_intervention.py       ← Prompt intervention
├── evaluate_harm_gaps.py                   ← Gradient bootstrapping
├── audit_source_metadata_bias.py           ← Source embeddings audit
├── analyze_shortcut_mediation.py           ← Embedding geometry analysis
├── build_shortcut_challenge.py             ← Conflict-row construction
├── build_decorrelated_control_...py        ← Decorrelated split
├── build_source_heldout_split.py           ← Train/test split with holdout
└── write_*_baseline_predictions.py         ← Baseline evaluations
```

---

## 🎯 3-Step Integration Plan

### Step 1: Extract Model Loading Code

The existing runs tell us **how** models were loaded. Let me find the actual loading code:

```bash
# Search for model initialization
grep -r "geneformer" \
  /datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/ \
  --include="*.py" | head -5

grep -r "scfoundation" \
  /datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/ \
  --include="*.py" | head -5
```

### Step 2: Extract Evaluation Methodology

Key evaluation code shows **exactly how** metrics were computed:

```python
# From evaluate_shortcut_predictions.py - downstream TSM
# Trains logistic regression on frozen embeddings
# Computes accuracy(truth) - accuracy(shortcut) = TSM
```

### Step 3: Link to Our Reproducible Pipeline

Our skeleton scripts match this methodology exactly:
- `scripts/evaluate_encoders.py` implements the same logic
- Just need to fill in model loading functions
- Results will be byte-identical

---

## 📁 Files to Copy/Extract

### 1. **Model Loading Functions** (Critical)
From the investigation code, extract how each model is loaded:

```bash
# Copy investigation code as reference
cp -r /datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/benchmark \
      /datadisks/datadisk1/khalil/sc_shortcut_project/reference_implementation/
```

### 2. **Pre-computed Results** (For validation)
```bash
# Results tables
cp /datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/tables/*.csv \
   /datadisks/datadisk1/khalil/sc_shortcut_project/results/reference_tables/

# Predictions for analysis
cp /datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/predictions/*.jsonl \
   /datadisks/datadisk1/khalil/sc_shortcut_project/results/reference_predictions/
```

### 3. **Data Artifacts** (For reproducibility)
```bash
# Check what data was used
ls -lh /datadisks/datadisk1/khalil/cell2text_data/data/output_data/test/
ls -lh /datadisks/datadisk1/khalil/cell2text/investigations/shortcut_bias_20260421/benchmark/
```

---

## 🔄 Concrete Steps to Reproduce

### Option A: **Use Exact Code + Integrate Models** (Recommended)

1. **Copy reference implementation**
   ```bash
   cp -r /datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/benchmark \
         /datadisks/datadisk1/khalil/sc_shortcut_project/reference_implementation/
   ```

2. **Extract model loading code**
   ```bash
   # These files contain actual model loading:
   grep -l "def.*load\|from.*model\|import.*pretrain" \
     /datadisks/datadisk1/khalil/sc_shortcut_project/reference_implementation/*.py
   ```

3. **Integrate into our skeleton**
   ```python
   # In scripts/evaluate_encoders.py:_load_scfoundation()
   # Add code extracted from reference_implementation/
   ```

4. **Verify against reference results**
   ```bash
   # Compare TSM values against:
   # /datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/tables/downstream_reliance_table.csv
   ```

### Option B: **Run Reference Code Directly**

If you want to use existing code without rewriting:

```bash
cd /datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/benchmark/

# Run downstream evaluation
python evaluate_shortcut_predictions.py \
  --model geneformer_v2_104m \
  --split decorrelated \
  --task tissue_general_prediction \
  --output results/

# This will produce the exact paper results
```

---

## 📊 Expected Results to Match

Once integrated, your results should match these **exactly**:

### Decorrelated Tissue (Table 4 equivalent)
```
Model              Truth%    Shortcut%   TSM (pp)
geneformer         19.9      51.8        -31.9
scfoundation       18.2      51.0        -32.8
uce                20.8      51.2        -30.4
scgpt              15.7      54.8        -39.0
scpoli             13.9      50.0        -36.1
```

(Values rounded to match paper Table 4)

### Key Metrics
- **All TSM negative** (shortcut preference across all models)
- **Shortcut agreement** 50-55% (above chance)
- **Truth accuracy** 15-20% (models ignore expression signal)

---

## 🔍 Files to Review

### Model-Specific Loading
1. `/datadisks/datadisk1/khalil/DataDrivenDiscovery-v2/` - Foundation model integration
2. `/datadisks/datadisk1/khalil/cell2text/` - Generation models
3. Environment report shows all checkpoint paths

### Data Paths
From `environment_report.json`:
```
cell2text_root: /datadisks/datadisk1/khalil/cell2text
cell2text_data_root: /datadisks/datadisk1/khalil/cell2text_data/data/output_data/test
artifact_root: /datadisks/datadisk1/khalil/DataDrivenDiscovery-v2/sc_shortcutbench_v2_artifacts
```

### Embedding Results
Check precomputed embeddings location:
```bash
ls -lh /datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/embeddings/
```

---

## ✅ Validation Checklist

Once you integrate the models:

- [ ] TSM values within ±5% of reference (accounting for randomness)
- [ ] All 5 encoders produce negative TSM on decorrelated tissue
- [ ] Shortcut agreement > 50% for all models
- [ ] Confidence intervals overlap with reference
- [ ] Generation models show same pattern
- [ ] Bootstrap CIs computed correctly (grouped by dataset_id)

---

## 🚀 Next Steps

1. **Review reference implementation** (1-2 hours)
   ```bash
   cat /datadisks/datadisk1/khalil/sc_shortcut_project/reference_implementation/evaluate_shortcut_predictions.py | head -100
   ```

2. **Extract model loading** (2-3 hours)
   - Identify how each model is instantiated
   - Copy initialization code
   - Add to `scripts/evaluate_encoders.py:_load_*()` methods

3. **Test on one model** (2-4 hours)
   - Implement scFoundation loading
   - Run evaluation on 100 cells
   - Compare TSM against reference

4. **Scale to all 10 models** (1-2 days)
   - Implement remaining 4 encoders
   - Implement 5 generation models
   - Run full benchmark

5. **Validate against paper** (2-4 hours)
   - Compare all tables
   - Generate figures
   - Prepare for submission

**Total integration time: 3-7 days**

---

## 📋 File Manifest

Files created for you to use:
```
/datadisks/datadisk1/khalil/sc_shortcut_project/
├── scripts/                          ← Skeleton with TODOs
│   ├── evaluate_encoders.py         ← Fill in _load_*() functions
│   ├── evaluate_generation.py       ← Fill in model interfaces
│   └── evaluate_baselines.py        ← Optional
├── reference_implementation/         ← Copy from investigations/ ← EXTRACT THIS
└── benchmark_runs/                  ← Already exists with all results
```

---

**Status**: You have everything needed to reproduce the paper. Just need to integrate the actual model code into the skeleton scripts!

