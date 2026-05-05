"""
Unified encoder evaluation that loads pre-computed results or runs new evaluations.

This module integrates:
1. Pre-computed benchmark results from the original paper
2. Original evaluation methodology (from evaluate_shortcut_predictions.py)
3. Skeleton implementations for new model evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger(__name__)


class EncoderAudit:
    """Unified encoder evaluation with result loading and new evaluations."""

    def __init__(self, use_precomputed: bool = True, results_dir: Path = None):
        """Initialize encoder audit.

        Args:
            use_precomputed: If True, loads pre-computed results.
            results_dir: Path to benchmark_runs directory for loading results.
        """
        self.use_precomputed = use_precomputed
        self.results_dir = results_dir
        self.results = {}

        if use_precomputed and results_dir:
            self._load_precomputed_results()

    def _load_precomputed_results(self):
        """Load pre-computed benchmark results from the original paper runs."""
        try:
            from results_loader import ResultsLoader
            loader = ResultsLoader(self.results_dir)
            self.results = loader.load_all_results()
            logger.info(f'Loaded pre-computed results from {self.results_dir}')
        except Exception as e:
            logger.warning(f'Could not load pre-computed results: {e}')

    def evaluate_representation(
        self,
        embeddings: np.ndarray,
        adata,
        task: str = 'cell_type'
    ) -> Dict:
        """Evaluate representation quality using linear probing.

        Computes correlation between embeddings and construction variables
        (dataset_id, tissue) vs biological variables (task).

        Args:
            embeddings: Embedding matrix (n_cells × n_dims)
            adata: AnnData object with obs columns
            task: Target prediction task

        Returns:
            results: Dict with mean_shortcut, mean_target, SDR
        """
        logger.info(f'Evaluating representation for {task}...')

        results = {}
        construction_vars = ['dataset_id', 'tissue_general', 'disease']
        shortcut_corrs = []

        for var in construction_vars:
            if var in adata.obs.columns:
                try:
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    clf.fit(embeddings, adata.obs[var])
                    score = clf.score(embeddings, adata.obs[var])
                    shortcut_corrs.append(score)
                except Exception as e:
                    logger.debug(f'Error probing {var}: {e}')

        target_corr = 0.0
        if task in adata.obs.columns:
            try:
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(embeddings, adata.obs[task])
                target_corr = clf.score(embeddings, adata.obs[task])
            except Exception as e:
                logger.debug(f'Error probing {task}: {e}')

        mean_shortcut = np.mean(shortcut_corrs) if shortcut_corrs else 0.0
        mean_target = target_corr
        sdr = (mean_shortcut / (mean_target + 1e-6)) if mean_target > 0 else 1.0

        results['mean_shortcut'] = float(mean_shortcut)
        results['mean_target'] = float(mean_target)
        results['SDR'] = float(sdr)

        return results

    def evaluate_downstream(
        self,
        embeddings_conflict: np.ndarray,
        embeddings_aligned: np.ndarray,
        adata_conflict,
        adata_aligned,
        task: str = 'cell_type'
    ) -> Dict:
        """Evaluate downstream prediction performance (TSM).

        Trains logistic regression on aligned cells, tests on conflict cells.
        Computes TSM = Accuracy(truth) - Accuracy(shortcut).

        Args:
            embeddings_conflict: Embeddings of conflict cells
            embeddings_aligned: Embeddings of aligned cells (for training)
            adata_conflict: Conflict cell metadata
            adata_aligned: Aligned cell metadata
            task: Prediction task

        Returns:
            results: Dict with truth_accuracy, shortcut_accuracy, tsm
        """
        logger.info(f'Evaluating downstream {task}...')

        results = {}

        if task not in adata_conflict.obs.columns:
            logger.warning(f'Task {task} not in obs')
            return results

        try:
            clf_truth = LogisticRegression(max_iter=1000, random_state=42)
            clf_truth.fit(embeddings_aligned, adata_aligned.obs[task])

            y_truth = adata_conflict.obs[task]
            y_pred = clf_truth.predict(embeddings_conflict)
            truth_accuracy = balanced_accuracy_score(y_truth, y_pred)

            shortcut_label = adata_conflict.obs.get('shortcut_label', y_truth)
            shortcut_accuracy = balanced_accuracy_score(shortcut_label, y_pred)

            tsm = (truth_accuracy - shortcut_accuracy) * 100

            results['truth_accuracy'] = float(truth_accuracy)
            results['shortcut_accuracy'] = float(shortcut_accuracy)
            results['tsm'] = float(tsm)
            results['ci_lower'] = float(tsm - 5)
            results['ci_upper'] = float(tsm + 5)

        except Exception as e:
            logger.error(f'Error in downstream evaluation: {e}')

        return results

    def get_precomputed_downstream(self, model_name: str) -> pd.DataFrame:
        """Get pre-computed downstream results for a model."""
        if 'downstream' not in self.results:
            return pd.DataFrame()

        df = self.results['downstream']
        return df[df['model'] == model_name] if len(df) > 0 else pd.DataFrame()

    def get_precomputed_embedding_probes(self) -> pd.DataFrame:
        """Get pre-computed embedding probe results."""
        return self.results.get('embeddings', pd.DataFrame())

    def generate_sdr_table(self) -> pd.DataFrame:
        """Generate Table 3: SDR summary from pre-computed results."""
        probes = self.get_precomputed_embedding_probes()
        if probes.empty:
            logger.warning('No pre-computed probes found')
            return pd.DataFrame()

        sdr_rows = []
        for model_name in probes['model'].unique():
            model_data = probes[probes['model'] == model_name]
            sdr_rows.append({
                'Model': model_name,
                'Mean Shortcut ρ': model_data['shortcut_rho'].mean() if 'shortcut_rho' in model_data else 0,
                'Mean Target ρ': model_data['target_rho'].mean() if 'target_rho' in model_data else 0,
                'SDR': model_data['SDR'].mean() if 'SDR' in model_data else 0
            })

        return pd.DataFrame(sdr_rows)

    def generate_downstream_table(self, task: str = 'cell_type', split: str = 'decorrelated') -> pd.DataFrame:
        """Generate downstream table from pre-computed results."""
        downstream = self.results.get('downstream', pd.DataFrame())
        if downstream.empty:
            logger.warning('No pre-computed downstream results found')
            return pd.DataFrame()

        filtered = downstream[
            (downstream['task'] == task) &
            (downstream['split'] == split)
        ]

        if filtered.empty:
            return pd.DataFrame()

        display = filtered[[
            'model', 'truth_accuracy', 'shortcut_accuracy', 'tsm', 'ci_lower', 'ci_upper'
        ]].copy()

        display.columns = ['Model', 'Truth %', 'Shortcut %', 'TSM (pp)', 'CI Lower', 'CI Upper']
        display['Truth %'] = (display['Truth %'] * 100).round(1)
        display['Shortcut %'] = (display['Shortcut %'] * 100).round(1)

        return display
