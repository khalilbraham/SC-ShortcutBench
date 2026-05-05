#!/usr/bin/env python3
"""
SC-ShortcutBench: Main pipeline orchestration.

Runs the complete benchmark:
1. Data loading and conflict-row construction
2. Encoder evaluation (linear probes + downstream)
3. Generative model evaluation
4. Baseline comparisons
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import yaml

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loading import load_cellxgene_subset
from conflict_rows import ConflictRowBenchmark
from evaluate_encoders import EncoderAudit
from evaluate_generation import GenerativeAudit
from evaluate_baselines import BaselineAudit
from utils.metrics import summarize_results


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkPipeline:
    """Main benchmark execution pipeline."""

    def __init__(self, config_path: str):
        """Initialize pipeline from config file.

        Args:
            config_path: Path to benchmark_config.yaml
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.data_dir = Path(self.config['paths']['data'])
        self.results_dir = Path(self.config['paths']['results'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create run metadata
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.results_dir / f'benchmark_run_{self.run_id}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config to run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

        logger.info(f'Initialized pipeline. Run ID: {self.run_id}')

    def stage_1_data_loading(self):
        """Load CELLxGENE Census and create train/test split."""
        logger.info('='*60)
        logger.info('STAGE 1: Data Loading')
        logger.info('='*60)

        # Load data
        logger.info('Loading CELLxGENE Census subset...')
        adata = load_cellxgene_subset(
            data_dir=self.data_dir,
            n_cells=self.config['data']['n_cells'],
            tissue_filter=self.config['data'].get('tissue_filter'),
            disease_filter=self.config['data'].get('disease_filter')
        )
        logger.info(f'Loaded {adata.n_obs} cells × {adata.n_vars} genes')

        # Create train/test split
        logger.info(f"Splitting data: {self.config['data']['train_fraction']:.1%} train")
        np.random.seed(self.config['random_seed'])
        train_idx = np.random.choice(
            adata.n_obs,
            size=int(adata.n_obs * self.config['data']['train_fraction']),
            replace=False
        )
        test_idx = np.setdiff1d(np.arange(adata.n_obs), train_idx)

        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()

        logger.info(f'Train: {adata_train.n_obs} cells')
        logger.info(f'Test: {adata_test.n_obs} cells')

        # Save to results
        adata_train.write(self.run_dir / 'data_train.h5ad')
        adata_test.write(self.run_dir / 'data_test.h5ad')

        return adata_train, adata_test

    def stage_2_conflict_rows(self, adata_train, adata_test):
        """Construct conflict rows for benchmarking."""
        logger.info('='*60)
        logger.info('STAGE 2: Conflict-Row Construction')
        logger.info('='*60)

        results = {}
        benchmark = ConflictRowBenchmark(
            train_data=adata_train,
            test_data=adata_test,
            min_support=self.config['conflict_rows']['min_support']
        )

        # Build conflict rows for each task
        for task_config in self.config['conflict_rows']['tasks']:
            task = task_config['target']
            conditioning = task_config['conditioning_vars']

            logger.info(f'Constructing conflict rows for {task}')
            logger.info(f'  Conditioning on: {conditioning}')

            for split_name in ['balanced', 'decorrelated']:
                logger.info(f'  Split: {split_name}')

                conflict_rows, aligned_rows, priors, stats = benchmark.construct(
                    target=task,
                    conditioning_vars=conditioning,
                    split=split_name,
                    decorrelation_strength=self.config['conflict_rows'].get('decorrelation_strength', 1.0)
                )

                # Log statistics
                logger.info(f'    Conflict rows: {len(conflict_rows)}')
                logger.info(f'    Aligned rows: {len(aligned_rows)}')
                logger.info(f'    Prior purity (mean): {stats["mean_purity"]:.3f}')

                # Save to results
                key = f'{split_name}_{task}'
                results[key] = {
                    'conflict_rows': conflict_rows,
                    'aligned_rows': aligned_rows,
                    'priors': priors,
                    'stats': stats
                }

                # Save to disk
                self._save_conflict_rows(
                    conflict_rows, aligned_rows, priors,
                    self.run_dir / 'conflict_rows' / f'{key}_metadata.json'
                )

        return results

    def stage_3_encoder_evaluation(self, conflict_rows_dict, adata_train, adata_test):
        """Evaluate frozen encoders."""
        logger.info('='*60)
        logger.info('STAGE 3: Encoder Evaluation')
        logger.info('='*60)

        encoder_names = self.config['models']['encoders']
        all_results = {}

        for encoder_name in encoder_names:
            logger.info(f'Evaluating {encoder_name}...')

            audit = EncoderAudit(
                model_name=encoder_name,
                device=self.config['hardware']['device'],
                batch_size=self.config['hardware']['batch_size']
            )

            # Evaluate on all conflict-row sets
            encoder_results = {}
            for conflict_key, conflict_data in conflict_rows_dict.items():
                logger.info(f'  {conflict_key}...')

                result = audit.evaluate(
                    conflict_rows=conflict_data['conflict_rows'],
                    aligned_rows=conflict_data['aligned_rows'],
                    adata_train=adata_train,
                    adata_test=adata_test,
                    targets=self.config['conflict_rows']['tasks']
                )
                encoder_results[conflict_key] = result

            all_results[encoder_name] = encoder_results

            # Save intermediate results
            self._save_encoder_results(
                encoder_results,
                self.run_dir / 'encoders' / f'{encoder_name}_results.json'
            )

        return all_results

    def stage_4_generation_evaluation(self, conflict_rows_dict, adata_test):
        """Evaluate generative models."""
        logger.info('='*60)
        logger.info('STAGE 4: Generative Model Evaluation')
        logger.info('='*60)

        gen_models = self.config['models']['generation']
        all_results = {}

        for model_name in gen_models:
            logger.info(f'Evaluating {model_name}...')

            audit = GenerativeAudit(
                model_name=model_name,
                device=self.config['hardware']['device'],
                batch_size=self.config['hardware']['batch_size']
            )

            gen_results = {}
            for conflict_key, conflict_data in conflict_rows_dict.items():
                logger.info(f'  {conflict_key}...')

                # Native output evaluation
                native_result = audit.evaluate_native_output(
                    conflict_rows=conflict_data['conflict_rows'],
                    aligned_rows=conflict_data['aligned_rows'],
                    adata=adata_test
                )

                # Prompt intervention (if applicable)
                prompt_result = None
                if self.config['generation']['evaluate_prompts']:
                    prompt_result = audit.prompt_intervention(
                        conflict_rows=conflict_data['conflict_rows'],
                        adata=adata_test
                    )

                gen_results[conflict_key] = {
                    'native': native_result,
                    'prompt_intervention': prompt_result
                }

            all_results[model_name] = gen_results

            # Save intermediate results
            self._save_generation_results(
                gen_results,
                self.run_dir / 'generation' / f'{model_name}_results.json'
            )

        return all_results

    def stage_5_baseline_evaluation(self, conflict_rows_dict, adata_train, adata_test):
        """Evaluate baseline models."""
        logger.info('='*60)
        logger.info('STAGE 5: Baseline Evaluation')
        logger.info('='*60)

        if not self.config.get('evaluate_baselines', True):
            logger.info('Baseline evaluation disabled in config')
            return {}

        baseline_names = self.config.get('baselines', [])
        all_results = {}

        for baseline_name in baseline_names:
            logger.info(f'Evaluating {baseline_name}...')

            audit = BaselineAudit(model_name=baseline_name)

            baseline_results = {}
            for conflict_key, conflict_data in conflict_rows_dict.items():
                result = audit.evaluate(
                    conflict_rows=conflict_data['conflict_rows'],
                    aligned_rows=conflict_data['aligned_rows'],
                    adata_train=adata_train,
                    adata_test=adata_test
                )
                baseline_results[conflict_key] = result

            all_results[baseline_name] = baseline_results

        return all_results

    def stage_6_summarize(self, encoder_results, gen_results, baseline_results):
        """Create summary statistics and tables."""
        logger.info('='*60)
        logger.info('STAGE 6: Summary & Tables')
        logger.info('='*60)

        summary = summarize_results(
            encoder_results=encoder_results,
            gen_results=gen_results,
            baseline_results=baseline_results
        )

        # Save summary
        with open(self.run_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Create paper tables
        self._create_paper_tables(summary)

        logger.info('Summary statistics saved')
        return summary

    def run(self):
        """Execute full benchmark pipeline."""
        try:
            # Stage 1: Data loading
            adata_train, adata_test = self.stage_1_data_loading()

            # Stage 2: Conflict rows
            conflict_rows_dict = self.stage_2_conflict_rows(adata_train, adata_test)

            # Stage 3: Encoders
            encoder_results = self.stage_3_encoder_evaluation(
                conflict_rows_dict, adata_train, adata_test
            )

            # Stage 4: Generation
            gen_results = self.stage_4_generation_evaluation(
                conflict_rows_dict, adata_test
            )

            # Stage 5: Baselines
            baseline_results = self.stage_5_baseline_evaluation(
                conflict_rows_dict, adata_train, adata_test
            )

            # Stage 6: Summary
            summary = self.stage_6_summarize(encoder_results, gen_results, baseline_results)

            logger.info('='*60)
            logger.info('BENCHMARK COMPLETE')
            logger.info(f'Run ID: {self.run_id}')
            logger.info(f'Results saved to: {self.run_dir}')
            logger.info('='*60)

            return summary

        except Exception as e:
            logger.error(f'Pipeline failed: {e}', exc_info=True)
            raise

    def _save_conflict_rows(self, conflict_rows, aligned_rows, priors, output_path):
        """Save conflict-row metadata."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'n_conflict': len(conflict_rows),
                'n_aligned': len(aligned_rows),
                'priors': priors
            }, f, indent=2)

    def _save_encoder_results(self, results, output_path):
        """Save encoder evaluation results."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _save_generation_results(self, results, output_path):
        """Save generation evaluation results."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _create_paper_tables(self, summary):
        """Create LaTeX tables for paper."""
        tables_dir = self.run_dir / 'tables'
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Generate tables from summary (implement based on paper tables)
        logger.info(f'Paper tables saved to {tables_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Run SC-ShortcutBench pipeline'
    )
    parser.add_argument(
        '--config',
        default='configs/benchmark_config.yaml',
        help='Path to benchmark config file'
    )
    parser.add_argument(
        '--stage',
        choices=['1', '2', '3', '4', '5', '6', 'all'],
        default='all',
        help='Run specific stage (default: all)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = BenchmarkPipeline(args.config)
    summary = pipeline.run()

    # Print summary
    print('\n' + '='*60)
    print('BENCHMARK SUMMARY')
    print('='*60)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
