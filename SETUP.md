# Setup for Reproducibility

**⚠️ Important:** Pre-computed results (450+ MB) are NOT in the GitHub repo.

## Quick Setup

```bash
pip install -r requirements.txt
```

## Three Reproducibility Options

### Option 1: Code-Only (GitHub)
- Get: evaluation code, evaluation methodology, original scripts
- Miss: pre-computed results (must run evaluations or load from datadisk1)
- Best for: reviewers, understanding methodology

### Option 2: With Pre-Computed Results (datadisk1 access)
- Access results at: `/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423/`
- Load with: `ResultsLoader(Path('/datadisks/datadisk1/...'))`
- Get: instant table generation from 60 downstream rows + 70 embedding metrics
- Best for: teams with datadisk1 access

### Option 3: Full Re-Evaluation
- Re-run all benchmarks using original evaluation scripts
- Produces: new TSM, confidence intervals, geometry analysis
- Best for: verification, new models

## Files in GitHub Repo

```
scripts/
├── evaluate_encoders.py               # Load or run encoder evaluation
├── evaluate_generation.py             # Generative models
├── evaluate_baselines.py              # Baselines
├── analysis.py                        # Tables & figures
├── results_loader.py                  # Load pre-computed results
├── evaluate_shortcut_predictions_original.py    # Original code
├── evaluate_metadata_priors_original.py
├── evaluate_prompt_intervention_original.py
└── analyze_geometry_original.py

configs/
├── benchmark_config.yaml              # Configuration template

requirements.txt                       # Python dependencies
README.md                              # Main documentation
SETUP.md                               # This file
DATASET.md                             # Dataset documentation
```

## Files NOT in GitHub (too large)

```
benchmark_runs/                        # 450+ MB
├── tables/
│   ├── downstream_reliance_table.csv     (60 rows of TSM)
│   ├── embedding_probe_table.csv         (70 rows of SDR)
│   ├── confidence_interval_report.json   (bootstrap CIs)
│   ├── correlation_geometry_*.csv        (geometry analysis)
│   └── ... (many more)
├── embeddings/                        (model embeddings)
├── predictions/                       (model outputs)
└── logs/                              (benchmark logs)

Location: /datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/...
```

## Usage Examples

### Load pre-computed results
```python
from scripts.results_loader import ResultsLoader

loader = ResultsLoader(Path('/datadisks/datadisk1/.../large_scale_max5000_20260423'))
downstream = loader.load_downstream_reliance()  # 60 rows
embeddings = loader.load_embedding_probes()     # 70 rows
```

### Run new evaluations
```python
from scripts.evaluate_encoders import EncoderAudit

audit = EncoderAudit(use_precomputed=False)
results = audit.evaluate_downstream(embeddings, adata_conflict, ...)
```

### Generate tables
```python
from scripts.analysis import StatisticalAnalysis

analysis = StatisticalAnalysis(Path('results'), {})
table = analysis.table_sdr(results)  # Table 3
```

## For NeurIPS Reviewers

- Code is in GitHub (reproducible)
- Results available on datadisk1 (if you have access)
- Original evaluation scripts included (for methodology verification)
- You can re-run evaluations if needed (original code provided)

## Why Results Aren't on GitHub

1. Too large (450+ MB)
2. NeurIPS best practice: code-only submission
3. Encourages independent verification
4. Version control works better without massive binary files

---

See README.md for project overview and datasets.
