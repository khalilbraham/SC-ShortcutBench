"""
Baseline model comparisons (Appendix: Baseline Comparisons).

Compares scFMs against:
- Raw expression linear probes
- PCA with HVG selection
- scVI (batch correction model)
- Harmony (integration method)
"""

import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from typing import Dict

logger = logging.getLogger(__name__)


class BaselineAudit:
    """Evaluate baseline models."""

    def __init__(self, model_name: str):
        """Initialize baseline audit.

        Args:
            model_name: 'raw_expression', 'PCA_HVG', 'scVI', 'Harmony'
        """
        self.model_name = model_name

    def evaluate(
        self,
        conflict_rows: np.ndarray,
        aligned_rows: np.ndarray,
        adata_train,
        adata_test
    ) -> Dict:
        """Evaluate baseline on conflict rows.

        Args:
            conflict_rows: Indices of conflict cells
            aligned_rows: Indices of aligned cells
            adata_train: Training data
            adata_test: Test data

        Returns:
            results: Evaluation results
        """
        logger.info(f'Evaluating baseline: {self.model_name}')

        if self.model_name == 'raw_expression':
            return self._evaluate_raw_expression(
                conflict_rows, aligned_rows, adata_train, adata_test
            )
        elif self.model_name == 'PCA_HVG':
            return self._evaluate_pca_hvg(
                conflict_rows, aligned_rows, adata_train, adata_test
            )
        elif self.model_name == 'scVI':
            return self._evaluate_scvi(
                conflict_rows, aligned_rows, adata_train, adata_test
            )
        elif self.model_name == 'Harmony':
            return self._evaluate_harmony(
                conflict_rows, aligned_rows, adata_train, adata_test
            )
        else:
            raise ValueError(f'Unknown baseline: {self.model_name}')

    def _evaluate_raw_expression(
        self,
        conflict_rows: np.ndarray,
        aligned_rows: np.ndarray,
        adata_train,
        adata_test
    ) -> Dict:
        """Evaluate raw expression with linear probe (Table A baseline).

        Trains logistic regression directly on raw counts (log-normalized).

        Returns:
            results: Linear recoverability of construction variables
        """
        logger.info('Evaluating raw expression linear probes...')

        # Get expression
        X_train = adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else adata_train.X
        X_test = adata_test.X.toarray() if hasattr(adata_test.X, 'toarray') else adata_test.X

        # Normalize
        X_train = np.log1p(X_train)
        X_test = np.log1p(X_test)

        results = {}

        # Probe for construction variables
        for var in ['dataset_id', 'tissue_general', 'disease', 'assay']:
            if var not in adata_train.obs.columns:
                continue

            # Train probe
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, adata_train.obs[var])

            # Evaluate
            y_pred = clf.predict(X_test)
            acc = balanced_accuracy_score(adata_test.obs[var], y_pred)

            results[var] = float(acc)

        return {'raw_expression_recoverability': results}

    def _evaluate_pca_hvg(
        self,
        conflict_rows: np.ndarray,
        aligned_rows: np.ndarray,
        adata_train,
        adata_test
    ) -> Dict:
        """Evaluate PCA with HVG selection.

        Selects top 2000 highly variable genes, then applies PCA to 50 components.

        Returns:
            results: Linear recoverability on PCA embeddings
        """
        logger.info('Evaluating PCA + HVG baseline...')

        X_train = adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else adata_train.X
        X_test = adata_test.X.toarray() if hasattr(adata_test.X, 'toarray') else adata_test.X

        # Select HVG (variance)
        gene_var = np.var(X_train, axis=0)
        hvg_idx = np.argsort(gene_var)[-2000:]

        X_train_hvg = X_train[:, hvg_idx]
        X_test_hvg = X_test[:, hvg_idx]

        # PCA
        pca = PCA(n_components=50, random_state=42)
        emb_train = pca.fit_transform(X_train_hvg)
        emb_test = pca.transform(X_test_hvg)

        results = {}

        # Probe
        for var in ['dataset_id', 'tissue_general', 'disease', 'assay']:
            if var not in adata_train.obs.columns:
                continue

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(emb_train, adata_train.obs[var])

            y_pred = clf.predict(emb_test)
            acc = balanced_accuracy_score(adata_test.obs[var], y_pred)

            results[var] = float(acc)

        return {'pca_hvg_recoverability': results}

    def _evaluate_scvi(
        self,
        conflict_rows: np.ndarray,
        aligned_rows: np.ndarray,
        adata_train,
        adata_test
    ) -> Dict:
        """Evaluate scVI (batch correction baseline).

        scVI is trained to correct for batch effects (study), so provides
        a strong baseline for checking if batch correction removes shortcuts.

        Returns:
            results: Evaluation metrics
        """
        try:
            import scvi
        except ImportError:
            logger.warning('scvi-tools not installed, skipping scVI baseline')
            return {'scvi': 'not_installed'}

        logger.info('Evaluating scVI baseline...')

        # Would train scVI on adata_train, then evaluate on adata_test
        # For now, placeholder
        logger.warning('scVI baseline training not fully implemented')

        return {'scvi': 'placeholder'}

    def _evaluate_harmony(
        self,
        conflict_rows: np.ndarray,
        aligned_rows: np.ndarray,
        adata_train,
        adata_test
    ) -> Dict:
        """Evaluate Harmony (integration baseline).

        Harmony integrates across studies, providing another batch-correction baseline.

        Returns:
            results: Evaluation metrics
        """
        try:
            from harmony import harmonize
        except ImportError:
            logger.warning('harmonypy not installed, skipping Harmony baseline')
            return {'harmony': 'not_installed'}

        logger.info('Evaluating Harmony baseline...')

        # Would apply Harmony to correct for 'dataset_id'
        # For now, placeholder
        logger.warning('Harmony baseline integration not fully implemented')

        return {'harmony': 'placeholder'}
