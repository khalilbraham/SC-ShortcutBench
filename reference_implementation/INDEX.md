# Reference Implementation Index

Extracted from: `/datadisks/datadisk1/khalil/sc_shortcut_project/investigations/shortcut_bias_20260421/benchmark/`

## Evaluation Scripts (How metrics were computed)

### Representation Level (Table 3)
- `evaluate_metadata_prior_shortcut.py` - Metadata prior strength (ρ for metadata)
- `audit_source_metadata_bias.py` - Source embeddings audit

### Prediction Level (Table 4-5)
- `evaluate_shortcut_predictions.py` - **MAIN**: Downstream TSM computation
  - Trains logistic regression on frozen embeddings
  - Evaluates on conflict rows
  - Computes truth_accuracy - shortcut_accuracy = TSM
  - Reports confusion matrices and transition maps

### Generation Level (Table 8-9)
- `evaluate_pairwise_choice.py` - Pairwise scoring (Cell2Sentence, etc.)
- `evaluate_unified_predictions.py` - Unified format (CellWhisperer)
- `evaluate_shortcut_intervention.py` - Prompt intervention

### Analysis Scripts
- `analyze_shortcut_mediation.py` - Embedding geometry (Figure 2, Table 7)
- `evaluate_harm_gaps.py` - Dose-response curves (Figure 3, Table 7)

## Data Construction Scripts (Algorithm 1)

- `build_shortcut_challenge.py` - Conflict-row construction
- `build_decorrelated_control_from_manifest.py` - Decorrelated split
- `build_source_heldout_split.py` - Train/test with holdout
- `build_unified_task_spec.py` - Task specifications

## How to Use This

1. **Study the main evaluation**:
   ```bash
   head -200 evaluate_shortcut_predictions.py
   ```

2. **Understand the metrics**:
   ```bash
   grep -A 20 "def.*tsm\|truth.*shortcut\|TSM" evaluate_shortcut_predictions.py
   ```

3. **See how models are loaded**:
   ```bash
   grep -n "import\|from.*load" evaluate_shortcut_predictions.py
   ```

4. **Check the data structure**:
   ```bash
   grep -A 10 "class.*Challenge\|def.*conflict\|conflict_rows" build_shortcut_challenge.py
   ```

## Next Steps

All this code maps directly to our skeleton scripts in `scripts/`:
- `evaluate_shortcut_predictions.py` → `scripts/evaluate_encoders.py`
- `evaluate_pairwise_choice.py` → `scripts/evaluate_generation.py`
- `build_shortcut_challenge.py` → `scripts/conflict_rows.py`

Just extract the model loading code and integrate!

