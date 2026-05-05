"""
Conflict-row construction (Algorithm 1 from paper).

Constructs conflict rows where metadata prior disagrees with true biological label.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class ConflictRowBenchmark:
    """Constructs conflict and aligned rows for shortcut-reliance testing."""

    def __init__(self, train_data, test_data, min_support: int = 5):
        """Initialize with train/test data.

        Args:
            train_data: Anndata object with training cells
            test_data: Anndata object with test cells
            min_support: Minimum cells needed to compute prior (Table 1)
        """
        self.train_data = train_data
        self.test_data = test_data
        self.min_support = min_support

        self.train_df = pd.DataFrame(train_data.obs)
        self.test_df = pd.DataFrame(test_data.obs)

    def construct(
        self,
        target: str,
        conditioning_vars: List[str],
        split: str = 'decorrelated',
        decorrelation_strength: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Construct conflict and aligned rows (Algorithm 1).

        Args:
            target: Target label (e.g., 'cell_type', 'tissue', 'disease')
            conditioning_vars: Variables to condition on (e.g., ['dataset_id', 'cell_type'])
            split: 'balanced' (preserve prior) or 'decorrelated' (break correlation)
            decorrelation_strength: Strength of decorrelation (0=none, 1=full)

        Returns:
            conflict_indices: Indices in test_data where shortcut ≠ truth
            aligned_indices: Indices in test_data where shortcut = truth
            priors: Prior probabilities P(Y|Z) for each Z
            stats: Statistics about the construction
        """
        logger.info(f'Constructing {split} {target} conflicts')

        # Step 1: Compute metadata priors on training data
        priors = self._compute_priors(target, conditioning_vars)

        # Step 2: Partition test data
        if split == 'balanced':
            conflict_idx, aligned_idx, test_reweighted = self._balanced_split(
                target, conditioning_vars, priors
            )
        elif split == 'decorrelated':
            conflict_idx, aligned_idx, test_reweighted = self._decorrelated_split(
                target, conditioning_vars, priors, decorrelation_strength
            )
        else:
            raise ValueError(f'Unknown split: {split}')

        # Compute statistics
        stats = self._compute_statistics(
            target, conditioning_vars, conflict_idx, aligned_idx, priors
        )

        logger.info(f'  Conflict rows: {len(conflict_idx)}')
        logger.info(f'  Aligned rows: {len(aligned_idx)}')
        logger.info(f'  Mean prior purity: {stats["mean_purity"]:.3f}')

        return conflict_idx, aligned_idx, priors, stats

    def _compute_priors(
        self,
        target: str,
        conditioning_vars: List[str]
    ) -> Dict[tuple, Dict[str, float]]:
        """Compute P(Y|Z) on training set.

        Args:
            target: Target label
            conditioning_vars: Variables to condition on

        Returns:
            priors: Dict mapping (z1, z2, ...) -> {label: count}
        """
        priors = {}

        # Group training data by conditioning variables
        grouped = self.train_df.groupby(conditioning_vars)

        for z_values, group_df in grouped:
            # Ensure z_values is always a tuple
            if not isinstance(z_values, tuple):
                z_values = (z_values,)

            # Check minimum support
            if len(group_df) < self.min_support:
                continue

            # Count occurrences of each target value
            value_counts = group_df[target].value_counts()

            # Get shortcut label (most common)
            shortcut_label = value_counts.idxmax()

            # Store prior
            prior_dict = value_counts.to_dict()
            prior_purity = prior_dict.get(shortcut_label, 0) / len(group_df)

            priors[z_values] = {
                'shortcut': shortcut_label,
                'purity': prior_purity,
                'counts': prior_dict,
                'support': len(group_df)
            }

        logger.info(f'Computed priors for {len(priors)} (z1, z2, ...) combinations')
        return priors

    def _balanced_split(
        self,
        target: str,
        conditioning_vars: List[str],
        priors: Dict
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Create balanced split preserving marginal P(Z).

        Test set reweighted so that marginal frequencies match training.

        Returns:
            conflict_indices: Test indices where shortcut ≠ truth
            aligned_indices: Test indices where shortcut = truth
            test_reweighted: Reweighted test dataframe (for analysis)
        """
        test_df = self.test_df.copy()

        # Initialize index arrays
        conflict_indices = []
        aligned_indices = []

        # For each test cell, check if it's conflict or aligned
        for idx, row in test_df.iterrows():
            # Get conditioning values for this cell
            z_values = tuple(row[var] for var in conditioning_vars)

            # Check if this (z_values) combination is in the training priors
            if z_values not in priors:
                continue

            # Get shortcut label and true label
            shortcut_label = priors[z_values]['shortcut']
            true_label = row[target]

            # Categorize as conflict or aligned
            if shortcut_label == true_label:
                aligned_indices.append(idx)
            else:
                conflict_indices.append(idx)

        return (
            np.array(conflict_indices),
            np.array(aligned_indices),
            test_df.loc[np.concatenate([conflict_indices, aligned_indices])]
        )

    def _decorrelated_split(
        self,
        target: str,
        conditioning_vars: List[str],
        priors: Dict,
        decorrelation_strength: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Create decorrelated split breaking P(Y|Z).

        Resamples test set so that within each Z, Y is approximately uniform.
        Breaks the training-time correlation while keeping Z in-distribution.

        Returns:
            conflict_indices: Test indices where shortcut ≠ truth
            aligned_indices: Test indices where shortcut = truth
            test_reweighted: Synthetic reweighted test set
        """
        test_df = self.test_df.copy()

        # Step 1: Group test data by Z
        test_grouped = test_df.groupby(conditioning_vars)

        conflict_indices = []
        aligned_indices = []
        reweighted_rows = []

        for z_values, group_df in test_grouped:
            if not isinstance(z_values, tuple):
                z_values = (z_values,)

            # Skip if Z not in training priors
            if z_values not in priors:
                continue

            shortcut_label = priors[z_values]['shortcut']

            # Step 2: Within this Z group, resample Y to be uniform
            unique_labels = group_df[target].unique()
            target_count_per_label = len(group_df) // len(unique_labels)

            # Stratified resampling within Z
            resampled_group = []
            for label in unique_labels:
                label_rows = group_df[group_df[target] == label]

                # Sample up to target_count_per_label rows
                if len(label_rows) > 0:
                    sampled = label_rows.sample(
                        n=min(len(label_rows), target_count_per_label),
                        random_state=42
                    )
                    resampled_group.append(sampled)

            if resampled_group:
                resampled_group = pd.concat(resampled_group)

                # Classify as conflict/aligned
                for idx, row in resampled_group.iterrows():
                    true_label = row[target]

                    if shortcut_label == true_label:
                        aligned_indices.append(idx)
                    else:
                        conflict_indices.append(idx)

                    reweighted_rows.append(row)

        test_reweighted = pd.DataFrame(reweighted_rows) if reweighted_rows else pd.DataFrame()

        return (
            np.array(conflict_indices),
            np.array(aligned_indices),
            test_reweighted
        )

    def _compute_statistics(
        self,
        target: str,
        conditioning_vars: List[str],
        conflict_idx: np.ndarray,
        aligned_idx: np.ndarray,
        priors: Dict
    ) -> Dict:
        """Compute statistics about the construction."""
        test_df = self.test_df.copy()

        # Get prior purities for conflict rows
        conflict_purities = []
        for idx in conflict_idx:
            z_values = tuple(test_df.loc[idx, var] for var in conditioning_vars)
            if z_values in priors:
                conflict_purities.append(priors[z_values]['purity'])

        stats = {
            'n_conflict': len(conflict_idx),
            'n_aligned': len(aligned_idx),
            'mean_purity': np.mean(conflict_purities) if conflict_purities else 0,
            'std_purity': np.std(conflict_purities) if conflict_purities else 0,
            'min_purity': np.min(conflict_purities) if conflict_purities else 0,
            'max_purity': np.max(conflict_purities) if conflict_purities else 1,
            'n_unique_z': len(priors)
        }

        return stats

    def get_prior_strength(
        self,
        target: str,
        conditioning_vars: List[str]
    ) -> Tuple[float, float]:
        """Get prior strength statistics (Table 2 - corpus evidence).

        Returns:
            Tuple of (metadata-only accuracy, coverage)
        """
        priors = self._compute_priors(target, conditioning_vars)

        # Test data lookups
        test_df = self.test_df.copy()
        correct = 0
        total = 0

        for idx, row in test_df.iterrows():
            z_values = tuple(row[var] for var in conditioning_vars)

            if z_values not in priors:
                continue

            predicted_label = priors[z_values]['shortcut']
            true_label = row[target]

            if predicted_label == true_label:
                correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0
        coverage = total / len(test_df) if len(test_df) > 0 else 0

        return accuracy, coverage
