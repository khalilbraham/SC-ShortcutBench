#!/usr/bin/env python3
"""
Post-hoc analysis and figure generation for SC-ShortcutBench.

Generates:
- Paper tables (LaTeX format)
- Statistical summaries
- Figures for manuscript
"""

import argparse
import json
import logging
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis import StatisticalAnalysis, FigureGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Post-hoc analysis and visualization pipeline."""

    def __init__(self, results_dir: str, config_path: str = None):
        """Initialize analysis pipeline.

        Args:
            results_dir: Directory containing benchmark results
            config_path: Path to analysis config (optional)
        """
        self.results_dir = Path(results_dir)

        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.analysis = StatisticalAnalysis(self.results_dir, self.config)
        self.figures = FigureGenerator(self.results_dir, self.config)

    def _default_config(self) -> dict:
        """Default analysis configuration."""
        return {
            'bootstrap_ci': 0.95,
            'n_bootstrap': 1000,
            'figure_dpi': 300,
            'figure_format': 'pdf'
        }

    def run_all(self):
        """Execute full analysis pipeline."""
        logger.info('='*60)
        logger.info('ANALYSIS PIPELINE')
        logger.info('='*60)

        # Load results
        logger.info('Loading benchmark results...')
        results = self._load_results()

        # Stage 1: Statistical summaries
        logger.info('Generating statistical summaries...')
        self._generate_tables(results)

        # Stage 2: Figures
        logger.info('Generating figures...')
        self._generate_figures(results)

        # Stage 3: Supplementary analyses
        logger.info('Running supplementary analyses...')
        self._run_supplementary(results)

        logger.info('='*60)
        logger.info('ANALYSIS COMPLETE')
        logger.info('='*60)

    def _load_results(self) -> dict:
        """Load all benchmark results."""
        results = {
            'encoders': {},
            'generation': {},
            'baselines': {}
        }

        # Load encoder results
        encoders_dir = self.results_dir / 'encoders'
        if encoders_dir.exists():
            for result_file in encoders_dir.glob('*_results.json'):
                model_name = result_file.stem.replace('_results', '')
                with open(result_file) as f:
                    results['encoders'][model_name] = json.load(f)

        # Load generation results
        gen_dir = self.results_dir / 'generation'
        if gen_dir.exists():
            for result_file in gen_dir.glob('*_results.json'):
                model_name = result_file.stem.replace('_results', '')
                with open(result_file) as f:
                    results['generation'][model_name] = json.load(f)

        # Load baselines
        baselines_dir = self.results_dir / 'baselines'
        if baselines_dir.exists():
            for result_file in baselines_dir.glob('*_results.json'):
                baseline_name = result_file.stem.replace('_results', '')
                with open(result_file) as f:
                    results['baselines'][baseline_name] = json.load(f)

        return results

    def _generate_tables(self, results: dict):
        """Generate paper tables (LaTeX format)."""
        output_dir = self.results_dir / 'tables'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Table 3: SDR summary
        logger.info('  Generating Table 3 (SDR)...')
        table_sdr = self.analysis.table_sdr(results)
        table_sdr.to_csv(output_dir / 'table_3_sdr.csv')
        logger.info(f'    Saved to {output_dir}/table_3_sdr.csv')

        # Table 4: Tissue prediction (decorrelated)
        logger.info('  Generating Table 4 (Decorrelated tissue)...')
        table_tissue = self.analysis.table_tissue_prediction(results)
        table_tissue.to_csv(output_dir / 'table_4_tissue.csv')

        # Table 5: Disease prediction (decorrelated)
        logger.info('  Generating Table 5 (Decorrelated disease)...')
        table_disease = self.analysis.table_disease_prediction(results)
        table_disease.to_csv(output_dir / 'table_5_disease.csv')

        # Table 8: Generation native
        logger.info('  Generating Table 8 (Generation native)...')
        table_gen = self.analysis.table_generation_native(results)
        table_gen.to_csv(output_dir / 'table_8_generation.csv')

        # Table 9: Prompt intervention
        logger.info('  Generating Table 9 (Prompt intervention)...')
        table_prompt = self.analysis.table_prompt_intervention(results)
        table_prompt.to_csv(output_dir / 'table_9_prompts.csv')

        logger.info(f'Tables saved to {output_dir}')

    def _generate_figures(self, results: dict):
        """Generate paper figures."""
        output_dir = self.results_dir / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: Overview (if applicable)
        # (Conceptual figure, usually created in Illustrator)

        # Figure 2: PCA visualizations
        logger.info('  Generating Figure 2 (PCA)...')
        self.figures.fig_pca_geometry(results)
        plt.savefig(
            output_dir / f'fig_2_pca.{self.config["figure_format"]}',
            dpi=self.config['figure_dpi'],
            bbox_inches='tight'
        )
        plt.close()

        # Figure 3: Dose-response curves
        logger.info('  Generating Figure 3 (Dose-response)...')
        self.figures.fig_dose_response(results)
        plt.savefig(
            output_dir / f'fig_3_dose_response.{self.config["figure_format"]}',
            dpi=self.config['figure_dpi'],
            bbox_inches='tight'
        )
        plt.close()

        logger.info(f'Figures saved to {output_dir}')

    def _run_supplementary(self, results: dict):
        """Run supplementary analyses."""
        output_dir = self.results_dir / 'supplementary'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Embedding geometry analysis
        logger.info('  Computing embedding geometry...')
        geometry = self.analysis.embedding_geometry(results)
        with open(output_dir / 'geometry.json', 'w') as f:
            json.dump(geometry, f, indent=2)

        # Within-strata analysis
        logger.info('  Computing within-strata recoverability...')
        within_strata = self.analysis.within_strata_recoverability(results)
        with open(output_dir / 'within_strata.json', 'w') as f:
            json.dump(within_strata, f, indent=2)

        logger.info(f'Supplementary analyses saved to {output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Run post-hoc analysis for SC-ShortcutBench'
    )
    parser.add_argument(
        '--results-dir',
        required=True,
        help='Directory containing benchmark results'
    )
    parser.add_argument(
        '--config',
        help='Path to analysis config (optional)'
    )
    parser.add_argument(
        '--stage',
        choices=['tables', 'figures', 'supplementary', 'all'],
        default='all',
        help='Which stage to run'
    )

    args = parser.parse_args()

    pipeline = AnalysisPipeline(args.results_dir, args.config)

    if args.stage == 'all':
        pipeline.run_all()
    elif args.stage == 'tables':
        results = pipeline._load_results()
        pipeline._generate_tables(results)
    elif args.stage == 'figures':
        results = pipeline._load_results()
        pipeline._generate_figures(results)
    elif args.stage == 'supplementary':
        results = pipeline._load_results()
        pipeline._run_supplementary(results)


if __name__ == '__main__':
    main()
