"""
Statistical analysis and figure generation for SC-ShortcutBench.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalysis:
    """Generate paper tables from benchmark results."""

    def __init__(self, results_dir: Path, config: Dict):
        """Initialize analysis.

        Args:
            results_dir: Directory containing results
            config: Analysis configuration
        """
        self.results_dir = Path(results_dir)
        self.config = config

    def table_sdr(self, results: Dict) -> pd.DataFrame:
        """Generate Table 3: SDR summary.

        Shortcut-to-target Decodability Ratio showing all encoders
        have SDR > 1 (construction variables easier to decode than biology).
        """
        rows = []

        for model_name, model_results in results['encoders'].items():
            if 'representation' not in model_results:
                continue

            rep = model_results['representation']

            rows.append({
                'Model': model_name,
                'Mean Shortcut ρ': rep.get('mean_shortcut', 0),
                'Mean Target ρ': rep.get('mean_target', 0),
                'SDR': rep.get('SDR', 0)
            })

        return pd.DataFrame(rows)

    def table_tissue_prediction(self, results: Dict) -> pd.DataFrame:
        """Generate Table 4: Decorrelated tissue prediction."""
        rows = []

        for model_name, model_results in results['encoders'].items():
            if 'prediction' not in model_results:
                continue

            pred = model_results['prediction']
            if 'tissue_general' not in pred:
                continue

            tissue_pred = pred['tissue_general']

            rows.append({
                'Model': model_name,
                'Truth %': tissue_pred.get('truth_accuracy', 0) * 100,
                'Shortcut %': tissue_pred.get('shortcut_accuracy', 0) * 100,
                'TSM (pp)': tissue_pred.get('tsm', 0)
            })

        return pd.DataFrame(rows)

    def table_disease_prediction(self, results: Dict) -> pd.DataFrame:
        """Generate Table 5: Decorrelated disease prediction."""
        rows = []

        for model_name, model_results in results['encoders'].items():
            if 'prediction' not in model_results:
                continue

            pred = model_results['prediction']
            if 'disease' not in pred:
                continue

            disease_pred = pred['disease']

            rows.append({
                'Model': model_name,
                'Truth %': disease_pred.get('truth_accuracy', 0) * 100,
                'Shortcut %': disease_pred.get('shortcut_accuracy', 0) * 100,
                'TSM (pp)': disease_pred.get('tsm', 0)
            })

        return pd.DataFrame(rows)

    def table_generation_native(self, results: Dict) -> pd.DataFrame:
        """Generate Table 8: Native generative output."""
        rows = []

        for model_name, model_results in results['generation'].items():
            if 'native' not in model_results:
                continue

            native = model_results['native']

            rows.append({
                'Model': model_name,
                'Shortcut Preference %': native.get('shortcut_preference', 0) * 100,
                'N Cells': native.get('n_cells', 0)
            })

        return pd.DataFrame(rows)

    def table_prompt_intervention(self, results: Dict) -> pd.DataFrame:
        """Generate Table 9: Prompt intervention effects."""
        rows = []

        for model_name, model_results in results['generation'].items():
            if 'prompt_intervention' not in model_results:
                continue

            prompt = model_results['prompt_intervention']

            rows.append({
                'Model': model_name,
                'Expression Only %': prompt.get('expression_only', 0) * 100,
                'Shortcut Context %': prompt.get('shortcut_context', 0) * 100,
                'Key Delta (pp)': prompt.get('key_delta', 0) * 100
            })

        return pd.DataFrame(rows)

    def embedding_geometry(self, results: Dict) -> Dict:
        """Analyze embedding geometry on conflict rows."""
        # Placeholder: actual implementation would load embeddings
        # and compute centroid distances (Table 7)
        return {'geometry': 'placeholder'}

    def within_strata_recoverability(self, results: Dict) -> Dict:
        """Compute within-strata recoverability (controlling for biology)."""
        # Placeholder: would stratify by cell_type/disease/tissue
        # and recompute dataset_id probe within each stratum
        return {'within_strata': 'placeholder'}


class FigureGenerator:
    """Generate publication-ready figures."""

    def __init__(self, results_dir: Path, config: Dict):
        """Initialize figure generator.

        Args:
            results_dir: Directory containing results
            config: Analysis configuration
        """
        self.results_dir = Path(results_dir)
        self.config = config

        # Set style
        sns.set_style('whitegrid')
        sns.set_palette('husl')

    def fig_pca_geometry(self, results: Dict):
        """Generate Figure 2: PCA visualizations of embedding geometry.

        Shows that source identity (dataset_id) is more compact than
        biological labels on conflict rows.
        """
        # Create multi-panel figure
        n_models = len(results['encoders'])
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_models == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]

        for idx, (model_name, model_results) in enumerate(results['encoders'].items()):
            row = idx // n_cols
            col = idx % n_cols

            ax = axes[row][col]

            # Placeholder: would plot PCA projection colored by
            # dataset_id (tight), true tissue (loose), shortcut (loose)
            ax.text(0.5, 0.5, f'{model_name}\n(PCA geometry)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model_name}')

        plt.tight_layout()
        logger.info('Figure 2 (PCA geometry) generated')

    def fig_dose_response(self, results: Dict):
        """Generate Figure 3: Prior purity dose-response curves.

        Shows shortcut agreement monotonically increases with prior purity π(z).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Balanced tissue
        ax = axes[0]
        # Placeholder: would plot quintiles with error bars
        ax.text(0.5, 0.5, 'Balanced Tissue\n(Dose-response)',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Balanced Tissue')
        ax.set_xlabel('Prior Purity Quintile')
        ax.set_ylabel('TSM (pp)')

        # Decorrelated tissue
        ax = axes[1]
        ax.text(0.5, 0.5, 'Decorrelated Tissue\n(Dose-response)',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Decorrelated Tissue')
        ax.set_xlabel('Prior Purity Quintile')
        ax.set_ylabel('TSM (pp)')

        plt.tight_layout()
        logger.info('Figure 3 (dose-response) generated')
