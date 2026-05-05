"""Load pre-computed benchmark results from the original paper runs."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Default path to the benchmark runs
DEFAULT_RESULTS_DIR = Path('/datadisks/datadisk1/khalil/sc_shortcut_project/benchmark_runs/sc_shortcutbench_v2_runs/large_scale_max5000_20260423')


class ResultsLoader:
    """Load all pre-computed benchmark results."""

    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize results loader.

        Args:
            results_dir: Path to benchmark_runs directory. Defaults to original location.
        """
        if results_dir is None:
            results_dir = DEFAULT_RESULTS_DIR

        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            logger.warning(f'Results directory {self.results_dir} not found')
        self.tables_dir = self.results_dir / 'tables'
        self.predictions_dir = self.results_dir / 'predictions'

    def load_downstream_reliance(self) -> pd.DataFrame:
        """Load main downstream TSM table (all models, tasks, splits).

        Returns:
            DataFrame with columns: model, task, split, truth_accuracy, shortcut_accuracy, tsm, ci_lower, ci_upper
        """
        path = self.tables_dir / 'downstream_reliance_table.csv'
        if not path.exists():
            logger.warning(f'File not found: {path}')
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_embedding_probes(self) -> pd.DataFrame:
        """Load embedding representation analysis (SDR, ρ values).

        Returns:
            DataFrame with embedding correlation metrics
        """
        path = self.tables_dir / 'embedding_probe_table.csv'
        if not path.exists():
            logger.warning(f'File not found: {path}')
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_generation_c2s(self) -> Dict:
        """Load Cell2Sentence generation evaluation results.

        Returns:
            Dict with native output and prompt intervention results
        """
        results = {}
        for split in ['balanced', 'decorrelated']:
            report_path = self.tables_dir / f'c2s_reasoning_pairwise_report_{split}.json'
            summary_path = self.tables_dir / f'c2s_reasoning_pairwise_summary_{split}.csv'

            if report_path.exists():
                with open(report_path) as f:
                    results[f'c2s_{split}_report'] = json.load(f)

            if summary_path.exists():
                results[f'c2s_{split}_summary'] = pd.read_csv(summary_path)

        return results

    def load_generation_cellwhisperer(self) -> Dict:
        """Load CellWhisperer generation evaluation results.

        Returns:
            Dict with evaluation metrics
        """
        results = {}
        for split in ['balanced', 'decorrelated']:
            report_path = self.tables_dir / f'cellwhisperer_context_query_report_{split}.json'
            summary_path = self.tables_dir / f'cellwhisperer_context_query_summary_{split}.csv'

            if report_path.exists():
                with open(report_path) as f:
                    results[f'cellwhisperer_{split}_report'] = json.load(f)

            if summary_path.exists():
                results[f'cellwhisperer_{split}_summary'] = pd.read_csv(summary_path)

        return results

    def load_confidence_intervals(self) -> Dict:
        """Load bootstrap confidence interval estimates.

        Returns:
            Dict with CI statistics
        """
        path = self.tables_dir / 'confidence_interval_report.json'
        if not path.exists():
            logger.warning(f'File not found: {path}')
            return {}

        with open(path) as f:
            return json.load(f)

    def load_geometry_analysis(self) -> Dict:
        """Load embedding geometry analysis (centroid, kNN, transfer).

        Returns:
            Dict mapping geometry type to DataFrame
        """
        results = {}
        geometry_files = {
            'centroid': 'correlation_geometry_centroid_entanglement.csv',
            'knn': 'correlation_geometry_knn_cross_context.csv',
            'transfer': 'correlation_geometry_leave_context_transfer.csv',
            'within_celltype': 'correlation_geometry_within_celltype_context.csv'
        }

        for key, filename in geometry_files.items():
            path = self.tables_dir / filename
            if path.exists():
                results[key] = pd.read_csv(path)
            else:
                logger.warning(f'Geometry file not found: {filename}')

        return results

    def load_embedding_information(self) -> Dict:
        """Load embedding information audits (group heldout, kNN enrichment, etc).

        Returns:
            Dict mapping audit type to DataFrame
        """
        results = {}
        audit_files = {
            'group_heldout': 'embedding_information_group_heldout_probe.csv',
            'knn_enrichment': 'embedding_information_knn_enrichment.csv',
            'random_probe': 'embedding_information_random_probe.csv',
            'within_stratum': 'embedding_information_within_stratum_source_probe.csv'
        }

        for key, filename in audit_files.items():
            path = self.tables_dir / filename
            if path.exists():
                results[key] = pd.read_csv(path)
            else:
                logger.warning(f'Audit file not found: {filename}')

        return results

    def load_full_multiclass_downstream(self) -> pd.DataFrame:
        """Load full multiclass downstream predictions (detailed).

        Returns:
            Large DataFrame with all model predictions
        """
        path = self.tables_dir / 'full_multiclass_downstream_summary.csv'
        if not path.exists():
            logger.warning(f'File not found: {path}')
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_generative_reasoning_bias(self) -> Dict:
        """Load generative model reasoning bias analysis.

        Returns:
            Dict with summary, by_route, by_prior_purity, and examples
        """
        results = {}
        files = {
            'summary': 'generative_reasoning_bias_summary.csv',
            'by_route': 'generative_reasoning_bias_by_route.csv',
            'by_prior': 'generative_reasoning_bias_by_prior_purity.csv',
            'examples': 'generative_reasoning_bias_examples.csv'
        }

        for key, filename in files.items():
            path = self.tables_dir / filename
            if path.exists():
                results[key] = pd.read_csv(path)
            else:
                logger.warning(f'Generative bias file not found: {filename}')

        return results

    def load_context_erasure_transfer(self) -> pd.DataFrame:
        """Load context erasure and transfer analysis.

        Returns:
            DataFrame with transfer metrics
        """
        path = self.tables_dir / 'context_erasure_transfer_detail.csv'
        if not path.exists():
            logger.warning(f'File not found: {path}')
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_all_results(self) -> Dict:
        """Load all available results at once.

        Returns:
            Comprehensive results dictionary
        """
        logger.info(f'Loading results from {self.results_dir}')

        results = {
            'downstream': self.load_downstream_reliance(),
            'embeddings': self.load_embedding_probes(),
            'confidence_intervals': self.load_confidence_intervals(),
            'geometry': self.load_geometry_analysis(),
            'embedding_info': self.load_embedding_information(),
            'multiclass': self.load_full_multiclass_downstream(),
            'generative': self.load_generative_reasoning_bias(),
            'context_transfer': self.load_context_erasure_transfer(),
            'generation_c2s': self.load_generation_c2s(),
            'generation_cellwhisperer': self.load_generation_cellwhisperer()
        }

        logger.info(f'Loaded {len([r for r in results.values() if isinstance(r, pd.DataFrame) and not r.empty])} main result tables')
        return results

    def summary_stats(self) -> str:
        """Print summary statistics of available results.

        Returns:
            String summary of what was loaded
        """
        lines = [f'Results from: {self.results_dir}']
        lines.append('')

        if (self.tables_dir / 'downstream_reliance_table.csv').exists():
            df = pd.read_csv(self.tables_dir / 'downstream_reliance_table.csv')
            lines.append(f'Downstream TSM: {len(df)} rows')
            lines.append(f'  Models: {df["model"].nunique()}')
            lines.append(f'  Tasks: {df["task"].nunique()}')

        if (self.tables_dir / 'embedding_probe_table.csv').exists():
            lines.append(f'Embedding probes: available')

        if (self.tables_dir / 'generative_reasoning_bias_summary.csv').exists():
            df = pd.read_csv(self.tables_dir / 'generative_reasoning_bias_summary.csv')
            lines.append(f'Generative models: {len(df)} rows')

        return '\n'.join(lines)
