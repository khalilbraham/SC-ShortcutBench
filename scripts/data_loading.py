"""
CELLxGENE Census data loading and preprocessing.

Loads subset of CELLxGENE Census with metadata for conflict-row construction.
"""

import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def load_cellxgene_subset(
    data_dir: Path,
    n_cells: Optional[int] = None,
    tissue_filter: Optional[List[str]] = None,
    disease_filter: Optional[List[str]] = None,
    random_seed: int = 42
):
    """Load CELLxGENE Census subset with metadata.

    Args:
        data_dir: Directory containing CELLxGENE data
        n_cells: Subsample to this many cells (None = use all)
        tissue_filter: Only include these tissues
        disease_filter: Only include these diseases
        random_seed: Random seed for reproducibility

    Returns:
        adata: Anndata object with expression and metadata
    """
    logger.info(f'Loading CELLxGENE Census from {data_dir}')

    # Try to load from existing h5ad file
    h5ad_file = data_dir / 'cellxgene_census.h5ad'
    if h5ad_file.exists():
        logger.info(f'Loading from {h5ad_file}')
        adata = sc.read_h5ad(h5ad_file)
    else:
        # Load from CELLxGENE API or raw files
        adata = _load_from_census_api(data_dir)

    logger.info(f'Loaded {adata.n_obs} cells × {adata.n_vars} genes')

    # Standardize metadata column names
    adata = _standardize_metadata(adata)

    # Apply filters
    if tissue_filter:
        logger.info(f'Filtering to tissues: {tissue_filter}')
        adata = adata[adata.obs['tissue_general'].isin(tissue_filter)].copy()
        logger.info(f'  Remaining: {adata.n_obs} cells')

    if disease_filter:
        logger.info(f'Filtering to diseases: {disease_filter}')
        adata = adata[adata.obs['disease'].isin(disease_filter)].copy()
        logger.info(f'  Remaining: {adata.n_obs} cells')

    # Subsample if requested
    if n_cells is not None and adata.n_obs > n_cells:
        logger.info(f'Subsampling to {n_cells} cells')
        np.random.seed(random_seed)
        idx = np.random.choice(adata.n_obs, n_cells, replace=False)
        adata = adata[idx].copy()

    # Basic QC
    logger.info(f'Final dataset: {adata.n_obs} cells × {adata.n_vars} genes')
    logger.info(f'Metadata columns: {list(adata.obs.columns)}')

    return adata


def _load_from_census_api(data_dir: Path):
    """Load from CELLxGENE Census API.

    Requires cellxgene_census package. See:
    https://chanzuckerberg.github.io/cellxgene-census/
    """
    try:
        import cellxgene_census
    except ImportError:
        raise ImportError(
            'cellxgene_census required. Install with: pip install cellxgene-census'
        )

    logger.info('Downloading CELLxGENE Census...')

    # Open census
    with cellxgene_census.open_soma() as census:
        # Get human RNA data
        adata = cellxgene_census.get_anndata(
            census=census,
            organism='homo_sapiens',
            obs_value_filter='is_primary_data == True'
        )

    # Save for future use
    adata.write(data_dir / 'cellxgene_census.h5ad')

    return adata


def _standardize_metadata(adata):
    """Standardize metadata column names and values.

    Maps CELLxGENE fields to paper's field names:
    - tissue_general
    - cell_type
    - disease
    - assay
    - donor_id
    - dataset_id
    - sex
    - development_stage
    """
    # Rename columns to standard names
    rename_map = {
        'tissue': 'tissue_general',
        'cell_type_ontology_term_id': 'cell_type_ontology',
        'disease_ontology_term_id': 'disease_ontology',
        'assay_ontology_term_id': 'assay_ontology',
    }

    for old, new in rename_map.items():
        if old in adata.obs.columns and new not in adata.obs.columns:
            adata.obs[new] = adata.obs[old]

    # Ensure required columns exist
    required = ['tissue_general', 'cell_type', 'disease', 'assay', 'dataset_id']
    for col in required:
        if col not in adata.obs.columns:
            # Try to find similar column or create placeholder
            similar = [c for c in adata.obs.columns if col.replace('_', '') in c.replace('_', '')]
            if similar:
                adata.obs[col] = adata.obs[similar[0]]
            else:
                logger.warning(f'Column {col} not found, creating placeholder')
                adata.obs[col] = 'unknown'

    # Clean up string values
    for col in ['tissue_general', 'cell_type', 'disease', 'assay', 'dataset_id']:
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype(str).str.strip()

    return adata


def load_raw_expression_baseline(adata):
    """Extract raw expression for baseline comparisons.

    Creates a copy with raw counts (unprocessed).

    Returns:
        adata_raw: Expression matrix for baseline evaluation
    """
    if 'raw' in adata.layers:
        adata_raw = adata.copy()
        adata_raw.X = adata_raw.layers['raw']
        return adata_raw
    else:
        # If no raw layer, use X as-is
        return adata.copy()


def get_data_splits(
    adata,
    train_fraction: float = 0.8,
    stratify_by: Optional[str] = 'dataset_id',
    random_seed: int = 42
) -> tuple:
    """Create train/test split stratified by dataset_id.

    Ensures no dataset overlap between train and test.

    Args:
        adata: Input data
        train_fraction: Fraction for training set
        stratify_by: Column to stratify on (default: dataset_id for source-holdout)
        random_seed: Random seed

    Returns:
        adata_train, adata_test: Split datasets
    """
    np.random.seed(random_seed)

    if stratify_by and stratify_by in adata.obs.columns:
        logger.info(f'Creating source-holdout split stratified by {stratify_by}')

        # Get unique values
        unique_vals = adata.obs[stratify_by].unique()
        np.random.shuffle(unique_vals)

        # Split unique values
        n_train = int(len(unique_vals) * train_fraction)
        train_vals = set(unique_vals[:n_train])

        # Get indices
        train_mask = adata.obs[stratify_by].isin(train_vals)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(~train_mask)[0]

        logger.info(
            f'Train: {len(train_idx)} cells from {len(train_vals)} '
            f'{stratify_by} values'
        )
        logger.info(
            f'Test: {len(test_idx)} cells from {len(unique_vals) - len(train_vals)} '
            f'{stratify_by} values'
        )

    else:
        logger.info('Creating random 80/20 split')
        idx = np.random.permutation(adata.n_obs)
        split = int(len(idx) * train_fraction)
        train_idx = idx[:split]
        test_idx = idx[split:]

    return adata[train_idx].copy(), adata[test_idx].copy()


def compute_metadata_priors(adata, target: str, conditioning_vars: List[str]):
    """Compute metadata-only priors (Table 2 corpus evidence).

    P(target | conditioning_vars) from the data.

    Args:
        adata: Input data
        target: Target label
        conditioning_vars: Variables to condition on

    Returns:
        accuracy, coverage
    """
    df = pd.DataFrame(adata.obs)

    grouped = df.groupby(conditioning_vars)

    correct = 0
    total = 0

    for z_values, group_df in grouped:
        # Get most common target value (shortcut)
        shortcut = group_df[target].value_counts().idxmax()

        # Count correct predictions
        correct += (group_df[target] == shortcut).sum()
        total += len(group_df)

    accuracy = correct / total if total > 0 else 0
    coverage = total / len(df)

    return accuracy, coverage
