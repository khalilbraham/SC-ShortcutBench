"""
Statistical utilities for SC-ShortcutBench.

Implements 95% grouped-bootstrap confidence intervals with grouping at dataset_id
to account for source-level dependence.
"""

import numpy as np
import pandas as pd
from typing import Callable, Tuple, List
from scipy import stats


def grouped_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable = None,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """Compute grouped-bootstrap confidence interval.

    Groups are resampled with replacement; within each group, observations
    are kept intact. This accounts for clustering/dependence within groups.

    From the paper: "All confidence intervals are 95% grouped-bootstrap
    intervals over conflict examples, with grouping at dataset_id to account
    for source-level dependence."

    Args:
        values: (n,) array of values
        groups: (n,) array of group labels (e.g., dataset_id)
        metric_fn: Function to compute metric. If None, returns mean.
                  Should take (values,) and return scalar.
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        (point_estimate, lower_ci, upper_ci)

    Example:
        >>> point, lower, upper = grouped_bootstrap_ci(
        ...     values=predictions == true_labels,
        ...     groups=dataset_ids,
        ...     confidence=0.95,
        ...     n_bootstrap=1000
        ... )
    """
    np.random.seed(random_seed)

    # Compute point estimate
    if metric_fn is None:
        metric_fn = np.mean
    point = metric_fn(values)

    # Get unique groups
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Bootstrap
    bootstrap_metrics = []

    for _ in range(n_bootstrap):
        # Resample groups with replacement
        resampled_groups = np.random.choice(unique_groups, size=n_groups, replace=True)

        # Collect all observations from resampled groups
        resampled_values = []
        for group in resampled_groups:
            group_mask = groups == group
            group_values = values[group_mask]
            resampled_values.extend(group_values)

        # Compute metric on resampled data
        if len(resampled_values) > 0:
            bootstrap_metrics.append(metric_fn(np.array(resampled_values)))

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Compute CIs as percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_ci = np.percentile(bootstrap_metrics, lower_percentile)
    upper_ci = np.percentile(bootstrap_metrics, upper_percentile)

    return float(point), float(lower_ci), float(upper_ci)


def binomial_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportion.

    From Table 7 (generation results): "Significance from exact binomial tests
    with 95% Wilson intervals."

    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level

    Returns:
        (lower_ci, upper_ci)
    """
    from statsmodels.stats.proportion import proportion_confint

    lower, upper = proportion_confint(
        successes, trials,
        alpha=1-confidence,
        method='wilson'
    )

    return float(lower), float(upper)


def tsm_grouped_bootstrap(
    y_truth: np.ndarray,
    y_shortcut: np.ndarray,
    y_predicted: np.ndarray,
    groups: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """Compute TSM (Truth-Shortcut Margin) with grouped bootstrap CI.

    TSM = accuracy(truth) - accuracy(shortcut)

    Negative TSM indicates shortcut preference (main paper result).

    Args:
        y_truth: True labels
        y_shortcut: Shortcut/metadata labels
        y_predicted: Predicted labels
        groups: Group labels (e.g., dataset_id)
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples

    Returns:
        (tsm_point, tsm_lower, tsm_upper)
    """
    def compute_tsm(y_pred):
        """Compute TSM for a subset of predictions."""
        n = len(y_pred)
        if n == 0:
            return 0

        truth_acc = (y_pred == y_truth).mean()
        shortcut_acc = (y_pred == y_shortcut).mean()
        return truth_acc - shortcut_acc

    # Sanity check dimensions
    assert len(y_truth) == len(y_shortcut) == len(y_predicted) == len(groups)

    np.random.seed(42)

    # Compute point TSM
    point_tsm = compute_tsm(y_predicted)

    # Bootstrap over groups
    unique_groups = np.unique(groups)
    bootstrap_tsm = []

    for _ in range(n_bootstrap):
        # Resample groups
        resampled_groups = np.random.choice(
            unique_groups, size=len(unique_groups), replace=True
        )

        # Collect indices for resampled groups
        resampled_idx = []
        for group in resampled_groups:
            group_mask = groups == group
            group_idx = np.where(group_mask)[0]
            resampled_idx.extend(group_idx)

        if resampled_idx:
            resampled_idx = np.array(resampled_idx)
            bootstrap_tsm.append(compute_tsm(y_predicted[resampled_idx]))

    bootstrap_tsm = np.array(bootstrap_tsm)

    # Compute percentile CI
    lower = np.percentile(bootstrap_tsm, 2.5)
    upper = np.percentile(bootstrap_tsm, 97.5)

    return float(point_tsm), float(lower), float(upper)


def prior_purity_dose_response(
    y_truth: np.ndarray,
    y_predicted: np.ndarray,
    prior_purities: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 5,
    confidence: float = 0.95
) -> pd.DataFrame:
    """Compute dose-response of shortcut agreement to prior purity (Fig 7).

    Bins conflict rows by prior purity π(z), then computes truth/shortcut
    agreement in each bin. Shows shortcut agreement scales monotonically
    with prior purity, supporting Proposition 1.

    Args:
        y_truth: True labels
        y_predicted: Predicted labels
        prior_purities: π(z) for each cell
        groups: Group labels (e.g., dataset_id)
        n_bins: Number of purity bins (quintiles, etc.)
        confidence: CI confidence level

    Returns:
        df: Dataframe with per-bin statistics
    """
    # Bin by prior purity
    bins = np.percentile(prior_purities, np.linspace(0, 100, n_bins+1))
    bin_labels = [f'Q{i+1}' for i in range(n_bins)]

    results = []

    for i in range(n_bins):
        # Get cells in this purity bin
        if i < n_bins - 1:
            mask = (prior_purities >= bins[i]) & (prior_purities < bins[i+1])
        else:
            mask = (prior_purities >= bins[i]) & (prior_purities <= bins[i+1])

        if not mask.any():
            continue

        bin_y_truth = y_truth[mask]
        bin_y_pred = y_predicted[mask]
        bin_groups = groups[mask]

        # Compute accuracies with bootstrap CI
        truth_acc, truth_lower, truth_upper = grouped_bootstrap_ci(
            (bin_y_pred == bin_y_truth).astype(int),
            bin_groups,
            confidence=confidence
        )

        shortcut_mask = y_truth == y_predicted  # Shortcut matches prediction
        shortcut_acc, shortcut_lower, shortcut_upper = grouped_bootstrap_ci(
            shortcut_mask[mask].astype(int),
            bin_groups,
            confidence=confidence
        )

        results.append({
            'bin': bin_labels[i],
            'mean_purity': np.mean(prior_purities[mask]),
            'n_cells': np.sum(mask),
            'truth_accuracy': truth_acc,
            'truth_ci_lower': truth_lower,
            'truth_ci_upper': truth_upper,
            'shortcut_agreement': shortcut_acc,
            'shortcut_ci_lower': shortcut_lower,
            'shortcut_ci_upper': shortcut_upper,
            'tsm': truth_acc - shortcut_acc
        })

    return pd.DataFrame(results)


def permutation_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    metric_fn: Callable = None,
    n_permutations: int = 10000
) -> float:
    """Permutation test comparing two predictions.

    Tests null hypothesis that two predictors have equal performance.

    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        metric_fn: Function to compute metric (default: accuracy)
        n_permutations: Number of permutations

    Returns:
        p_value: Two-tailed p-value
    """
    if metric_fn is None:
        metric_fn = lambda y1, y2: (y1 == y2).mean()

    # Observed difference
    obs_diff = metric_fn(y_pred1, y_true) - metric_fn(y_pred2, y_true)

    # Permutation differences
    perm_diffs = []
    np.random.seed(42)

    for _ in range(n_permutations):
        # Randomly assign predictions to models
        mask = np.random.rand(len(y_true)) < 0.5
        y_perm1 = y_pred1.copy()
        y_perm2 = y_pred2.copy()

        # Swap
        y_perm1[mask], y_perm2[mask] = y_pred2[mask], y_pred1[mask]

        perm_diffs.append(
            metric_fn(y_perm1, y_true) - metric_fn(y_perm2, y_true)
        )

    perm_diffs = np.array(perm_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    return float(p_value)
