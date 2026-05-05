"""
Evaluation metrics for SC-ShortcutBench.

Implements paper metrics:
- ρ(Z; φ): Linear recoverability
- π(z): Prior purity
- TSM: Truth-Shortcut Margin
- α_sc: Shortcut agreement
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from typing import Dict, List


def compute_sdr(
    shortcut_accs: Dict[str, float],
    target_accs: Dict[str, float]
) -> float:
    """Compute SDR (Shortcut-to-target Decodability Ratio).

    From Table 3: SDR > 1 for all encoders means construction variables
    are easier to decode than biological targets.

    SDR = mean_ρ(construction) / mean_ρ(targets)

    Args:
        shortcut_accs: Dict of {var: balanced_accuracy} for construction vars
        target_accs: Dict of {var: balanced_accuracy} for biological targets

    Returns:
        sdr: SDR value (>1 indicates shortcut dominance)
    """
    mean_shortcut = np.mean(list(shortcut_accs.values()))
    mean_target = np.mean(list(target_accs.values()))

    sdr = mean_shortcut / mean_target if mean_target > 0 else 0

    return float(sdr)


def compute_tsm(
    y_true: np.ndarray,
    y_shortcut: np.ndarray,
    y_predicted: np.ndarray
) -> float:
    """Compute Truth-Shortcut Margin.

    TSM = P(ŷ = y) - P(ŷ = s(z))

    Negative values indicate shortcut preference.
    From Table 4: TSM ranges from -31.1 pp to -24.7 pp on decorrelated tissue.

    Args:
        y_true: True labels
        y_shortcut: Shortcut/metadata labels
        y_predicted: Predicted labels

    Returns:
        tsm: TSM in percentage points (can be negative)
    """
    truth_acc = np.mean(y_predicted == y_true)
    shortcut_acc = np.mean(y_predicted == y_shortcut)

    tsm = (truth_acc - shortcut_acc) * 100  # Convert to percentage points

    return float(tsm)


def compute_confusion_rates(
    y_true: np.ndarray,
    y_shortcut: np.ndarray,
    y_predicted: np.ndarray
) -> Dict[str, float]:
    """Compute rates: truth, shortcut, neither.

    From Table 4 (decorrelated tissue):
    Truth %, Shortcut %, Neither % sum to 100%.

    Args:
        y_true: True labels
        y_shortcut: Shortcut labels
        y_predicted: Predicted labels

    Returns:
        rates: {'truth': %, 'shortcut': %, 'neither': %}
    """
    n = len(y_true)

    truth_count = np.sum(y_predicted == y_true)
    shortcut_count = np.sum(y_predicted == y_shortcut)
    neither_count = n - truth_count - shortcut_count

    return {
        'truth_percent': 100 * truth_count / n,
        'shortcut_percent': 100 * shortcut_count / n,
        'neither_percent': 100 * neither_count / n,
        'truth_count': int(truth_count),
        'shortcut_count': int(shortcut_count),
        'neither_count': int(neither_count)
    }


def compute_transition_map(
    y_true: np.ndarray,
    y_shortcut: np.ndarray,
    y_predicted: np.ndarray,
    top_k: int = 10
) -> pd.DataFrame:
    """Compute transition map (Table 6).

    Shows top deterministic collapses from true label to predicted label.
    E.g., "cortex -> brain at 99.7%"

    Args:
        y_true: True labels
        y_shortcut: Shortcut labels  (not used, but included for consistency)
        y_predicted: Predicted labels
        top_k: Number of top transitions to return

    Returns:
        df: Dataframe with columns:
            ['true_label', 'predicted_label', 'count', 'within_true_percent']
    """
    # Create transition counts
    transitions = {}

    for true_val, pred_val in zip(y_true, y_predicted):
        key = (true_val, pred_val)
        transitions[key] = transitions.get(key, 0) + 1

    # Compute within-true percentages
    true_counts = {}
    for true_val in y_true:
        true_counts[true_val] = true_counts.get(true_val, 0) + 1

    results = []
    for (true_val, pred_val), count in transitions.items():
        within_true_pct = 100 * count / true_counts[true_val]

        results.append({
            'true_label': str(true_val),
            'predicted_label': str(pred_val),
            'count': int(count),
            'within_true_percent': round(within_true_pct, 1)
        })

    # Sort by count (descending)
    df = pd.DataFrame(results).sort_values('count', ascending=False)

    return df.head(top_k).reset_index(drop=True)


def summarize_results(
    encoder_results: Dict,
    gen_results: Dict,
    baseline_results: Dict
) -> Dict:
    """Create summary statistics across all models.

    Returns high-level summary for paper tables.

    Args:
        encoder_results: Results from encoder audit
        gen_results: Results from generation audit
        baseline_results: Results from baseline audit

    Returns:
        summary: Dict with tables and statistics
    """
    summary = {
        'encoders': {},
        'generation': {},
        'baselines': {}
    }

    # Summarize encoder results
    for model_name, results in encoder_results.items():
        if 'representation' in results:
            rep = results['representation']
            summary['encoders'][model_name] = {
                'mean_shortcut_recoverability': rep.get('mean_shortcut', 0),
                'mean_target_recoverability': rep.get('mean_target', 0),
                'sdr': rep.get('SDR', 0),
                'shortcut_vars': rep.get('shortcut_recoverability', {})
            }

        if 'prediction' in results:
            pred = results['prediction']
            summary['encoders'][model_name]['prediction'] = pred

    # Summarize generation results
    for model_name, results in gen_results.items():
        summary['generation'][model_name] = results

    # Summarize baselines
    for baseline_name, results in baseline_results.items():
        summary['baselines'][baseline_name] = results

    return summary
