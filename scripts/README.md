# SC-ShortcutBench: Reproducible Pipeline

Reproducible evaluation code for "SC-ShortcutBench: A Conflict-Row Benchmark for Metadata Shortcut Reliance in Single-Cell Foundation Models"

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full benchmark pipeline
python scripts/run_benchmark.py --config configs/benchmark_config.yaml

# 3. Generate results and figures
python scripts/run_analysis.py --config configs/analysis_config.yaml
```

## Project Structure

```
scripts/
├── run_benchmark.py           # Main pipeline orchestration
├── data_loading.py            # CELLxGENE Census data loading
├── conflict_rows.py           # Conflict-row construction (Algorithm 1)
├── evaluate_encoders.py       # Linear probes + downstream heads
├── evaluate_generation.py     # Native generative model evaluation
├── evaluate_baselines.py      # Baseline model comparisons
├── analysis.py                # Statistical analysis & visualization
├── run_analysis.py            # Analysis orchestration
└── utils/
    ├── metrics.py             # Evaluation metrics
    ├── statistical.py         # Bootstrap CIs, group statistics
    ├── geometry.py            # Embedding geometry analysis
    └── visualization.py       # Plotting utilities

configs/
├── benchmark_config.yaml      # Benchmark pipeline configuration
├── analysis_config.yaml       # Analysis configuration
└── models.yaml                # Model definitions & parameters

data/
├── raw/                       # CELLxGENE Census (if downloaded)
├── processed/                 # Processed datasets
├── conflict_rows/             # Generated conflict-row datasets
└── metadata/                  # Metadata files

results/
├── embeddings/                # Model embeddings (frozen encoders)
├── predictions/               # Downstream predictions
├── generation/                # Generative model outputs
├── analysis/                  # Statistical analyses
└── figures/                   # Generated figures (paper-ready)

benchmark_runs/               # Historical benchmark execution records
```

## Key Components

### 1. Conflict-Row Construction
```python
from scripts.conflict_rows import ConflictRowBenchmark

benchmark = ConflictRowBenchmark(
    train_data=train_adata,
    test_data=test_adata,
    target='cell_type',
    conditioning_vars=['dataset_id', 'tissue']
)

conflict_rows, aligned_rows, priors = benchmark.construct(
    split='decorrelated',
    min_support=5
)
```

### 2. Encoder Evaluation
```python
from scripts.evaluate_encoders import EncoderAudit

audit = EncoderAudit(model_name='scFoundation')
audit.evaluate(
    conflict_rows=conflict_rows,
    aligned_rows=aligned_rows,
    targets=['cell_type', 'tissue', 'disease']
)
```

### 3. Generative Model Evaluation
```python
from scripts.evaluate_generation import GenerativeAudit

audit = GenerativeAudit(model_name='Cell2Sentence')
audit.evaluate_native_output(conflict_rows)
audit.prompt_intervention(conflict_rows)
```

## Configuration

Edit `configs/benchmark_config.yaml` to customize:
- Data paths and subset sizes
- Model selections and checkpoint paths
- Evaluation metrics and splits
- Hardware settings (GPU allocation, batch size)

## Output Structure

All results organized by model and split:

```
results/
├── encoders/
│   └── scFoundation/
│       ├── decorrelated/
│       │   ├── linear_probes.json
│       │   ├── downstream_predictions.json
│       │   └── embeddings.h5ad
│       └── balanced/
└── generation/
    └── Cell2Sentence/
        ├── native_outputs.json
        └── prompt_intervention.json
```

## Statistical Analysis

All confidence intervals are **95% grouped-bootstrap intervals** with grouping at `dataset_id` to account for source-level dependence.

```python
from scripts.utils.statistical import grouped_bootstrap_ci

ci = grouped_bootstrap_ci(
    values=predictions,
    groups=dataset_ids,
    confidence=0.95,
    metric='accuracy'
)
```

## Reproduction Time Estimates

- Data loading & conflict-row construction: ~2 hours
- Encoder audit (5 models): ~24 hours (GPU)
- Generative audit (5 models): ~12 hours
- Statistical analysis & figures: ~4 hours
- **Total: ~48 hours on GPU hardware**

## NeurIPS Submission Requirements

This pipeline produces:
- ✅ Tables 1-4: Model audit results
- ✅ Figures 1-3: Benchmark overview & mechanism
- ✅ Section 4 results: Detailed comparisons
- ✅ Appendix analyses: Full per-model breakdowns
- ✅ Reproducibility artifacts: Code, configs, results

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

## Support

For issues or questions:
- Check the [FAQ](docs/FAQ.md)
- Review [troubleshooting guide](docs/TROUBLESHOOTING.md)
- Open an issue with reproduction details
