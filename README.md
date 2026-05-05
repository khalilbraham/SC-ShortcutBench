# SC-ShortcutBench: A Benchmark for Metadata Shortcut Reliance in Single-Cell Foundation Models

[![Datasets](https://img.shields.io/badge/Datasets-HuggingFace-yellow)](https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-public)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue)](https://creativecommons.org/licenses/by/4.0/)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026%20D%26B%20Track-blueviolet)](https://neurips.cc/)

## Overview

SC-ShortcutBench is a comprehensive benchmark for evaluating whether single-cell foundation models rely on metadata shortcuts rather than expression signals for prediction.

**Key Results**:
- 899K cells from CELLxGENE Census
- 10 foundation models evaluated
- **All models show shortcut preference** (TSM: -24 to -40 pp)

## Datasets

**Public (Training)**: https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-public
- Expression data (balanced + decorrelated splits)
- Metadata annotations
- Benchmark results

**Private (Evaluation)**: https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-private
- Test expression data (evaluation-only)
- Manual approval required

## Quick Start

**⚠️ Note:** Pre-computed results (450+ MB) are NOT in GitHub. See [SETUP.md](SETUP.md) for options.

```bash
# Install
pip install -r requirements.txt

# Load code and original evaluation methodology
python -c "from scripts.evaluate_encoders import EncoderAudit; print('OK')"

# For full reproducibility, see SETUP.md
```

See [SETUP.md](SETUP.md) for:
- Loading pre-computed results (if you have datadisk1 access)
- Running new evaluations  
- Understanding what's in GitHub vs what's stored separately

## Repository Structure

```
├── scripts/                # Reproducible pipeline
│   ├── run_benchmark.py
│   ├── evaluate_encoders.py
│   ├── evaluate_generation.py
│   └── utils/
├── configs/                # Configuration files
├── reference_implementation/  # Reference code from runs
└── docs/                   # Documentation
    ├── DATASET.md
    ├── IMPLEMENTATION_SUMMARY.md
    └── PAPER_REPRODUCTION_GUIDE.md
```

## Models Evaluated

**Encoders** (5): scFoundation, Geneformer, UCE, scGPT, scPoli
**Generative** (5): Cell2Sentence, Cell2Text, CellWhisperer, scGPT

## Citation

```bibtex
@article{sc_shortcutbench_2026,
  title={SC-ShortcutBench: A Conflict-Row Benchmark for Metadata Shortcut Reliance in Single-Cell Foundation Models},
  author={Your Author Names},
  journal={Proceedings of NeurIPS},
  year={2026},
  note={Datasets and Benchmarks Track}
}
```

## License

CC-BY-4.0

## Datasets on HuggingFace

- **Public**: https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-public
- **Private**: https://huggingface.co/datasets/Khalilbraham/sc-shortcutbench-private

---

Ready for NeurIPS 2026 submission 🎉
