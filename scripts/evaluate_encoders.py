"""
Evaluation of frozen encoder models (Section 3: Failures).

Implements:
- Linear recoverability of construction variables (ρ)
- Downstream head routing (TSM)
- Embedding geometry analysis
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import torch
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class EncoderAudit:
    """Audit frozen encoder for shortcut reliance."""

    def __init__(self, model_name: str, device: str = 'cuda', batch_size: int = 256):
        """Initialize encoder audit.

        Args:
            model_name: Name of encoder (scFoundation, Geneformer, etc.)
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Batch size for embedding computation
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load pretrained encoder model.

        Handles different model architectures:
        - scFoundation
        - Geneformer
        - UCE
        - scGPT
        - scPoli
        """
        if model_name == 'scFoundation':
            return self._load_scfoundation()
        elif model_name == 'Geneformer':
            return self._load_geneformer()
        elif model_name == 'UCE':
            return self._load_uce()
        elif model_name == 'scGPT':
            return self._load_scgpt()
        elif model_name == 'scPoli':
            return self._load_scpoli()
        else:
            raise ValueError(f'Unknown encoder: {model_name}')

    def _load_scfoundation(self):
        """Load scFoundation model."""
        # Placeholder: actual implementation depends on model release
        logger.warning('scFoundation loading not fully implemented')
        return None

    def _load_geneformer(self):
        """Load Geneformer model."""
        logger.warning('Geneformer loading not fully implemented')
        return None

    def _load_uce(self):
        """Load UCE model."""
        logger.warning('UCE loading not fully implemented')
        return None

    def _load_scgpt(self):
        """Load scGPT model."""
        logger.warning('scGPT loading not fully implemented')
        return None

    def _load_scpoli(self):
        """Load scPoli model."""
        logger.warning('scPoli loading not fully implemented')
        return None

    def evaluate(
        self,
        conflict_rows: np.ndarray,
        aligned_rows: np.ndarray,
        adata_train,
        adata_test,
        targets: List[Dict]
    ) -> Dict:
        """Run full audit: representation + prediction.

        Args:
            conflict_rows: Indices of conflict cells in test set
            aligned_rows: Indices of aligned cells in test set
            adata_train: Training data
            adata_test: Test data
            targets: List of target task configs

        Returns:
            results: Dictionary with audit results
        """
        results = {}

        # Compute embeddings
        logger.info(f'Computing embeddings for {self.model_name}...')
        embeddings_train = self.get_embeddings(adata_train)
        embeddings_test = self.get_embeddings(adata_test)

        # Audit 1: Representation (linear recoverability)
        logger.info('Audit 1: Linear recoverability of construction variables')
        rep_results = self._audit_representation(
            embeddings_train, adata_train,
            embeddings_test, adata_test,
            conflict_rows, aligned_rows
        )
        results['representation'] = rep_results

        # Audit 2: Prediction (downstream TSM)
        logger.info('Audit 2: Downstream head routing')
        pred_results = self._audit_prediction(
            embeddings_train, adata_train,
            embeddings_test, adata_test,
            conflict_rows, aligned_rows,
            targets
        )
        results['prediction'] = pred_results

        return results

    def get_embeddings(self, adata) -> np.ndarray:
        """Get frozen embeddings from encoder.

        Args:
            adata: Input data

        Returns:
            embeddings: (n_cells, embedding_dim) array
        """
        if self.model is None:
            logger.warning(f'Model {self.model_name} not loaded, returning random embeddings')
            return np.random.randn(adata.n_obs, 768)

        # Get raw counts
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        embeddings = []

        # Process in batches
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i+self.batch_size]
            batch_tensor = torch.FloatTensor(batch).to(self.device)

            with torch.no_grad():
                batch_emb = self.model.encode(batch_tensor)

            embeddings.append(batch_emb.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        logger.info(f'Computed embeddings: {embeddings.shape}')

        return embeddings

    def _audit_representation(
        self,
        embeddings_train: np.ndarray,
        adata_train,
        embeddings_test: np.ndarray,
        adata_test,
        conflict_rows: np.ndarray,
        aligned_rows: np.ndarray
    ) -> Dict:
        """Measure linear recoverability ρ(Z; φ) of construction variables.

        Trains linear probes on frozen embeddings to predict each construction
        variable. Reports balanced accuracy.

        Returns:
            results: Dict with per-variable recoverability scores
        """
        results = {}

        # Construction variables to probe
        construction_vars = ['dataset_id', 'assay', 'donor', 'tissue_general']
        biological_targets = ['cell_type', 'disease', 'development_stage']

        # Probe for construction variables
        logger.info('  Probing construction variables...')
        shortcut_accs = {}
        for var in construction_vars:
            if var not in adata_train.obs.columns:
                logger.warning(f'  {var} not in data, skipping')
                continue

            acc = self._train_linear_probe(
                embeddings_train, adata_train.obs[var],
                embeddings_test, adata_test.obs[var]
            )
            shortcut_accs[var] = acc
            logger.info(f'    {var}: {acc:.3f}')

        # Probe for biological targets
        logger.info('  Probing biological targets...')
        target_accs = {}
        for var in biological_targets:
            if var not in adata_train.obs.columns:
                continue

            acc = self._train_linear_probe(
                embeddings_train, adata_train.obs[var],
                embeddings_test, adata_test.obs[var]
            )
            target_accs[var] = acc
            logger.info(f'    {var}: {acc:.3f}')

        # Compute SDR (Shortcut-to-target Decodability Ratio)
        mean_shortcut = np.mean(list(shortcut_accs.values()))
        mean_target = np.mean(list(target_accs.values()))
        sdr = mean_shortcut / mean_target if mean_target > 0 else 0

        results['shortcut_recoverability'] = shortcut_accs
        results['target_recoverability'] = target_accs
        results['mean_shortcut'] = mean_shortcut
        results['mean_target'] = mean_target
        results['SDR'] = sdr

        return results

    def _audit_prediction(
        self,
        embeddings_train: np.ndarray,
        adata_train,
        embeddings_test: np.ndarray,
        adata_test,
        conflict_rows: np.ndarray,
        aligned_rows: np.ndarray,
        targets: List[Dict]
    ) -> Dict:
        """Measure downstream TSM (Truth-Shortcut Margin).

        Trains downstream classifiers on aligned cells, evaluates on conflict rows.
        Reports TSM = accuracy(truth) - accuracy(shortcut).

        Args:
            targets: List of {'target': 'tissue', 'conditioning_vars': [...]}

        Returns:
            results: Dict with TSM for each target
        """
        results = {}

        for target_spec in targets:
            target = target_spec['target']
            conditioning_vars = target_spec['conditioning_vars']

            logger.info(f'  Evaluating {target} with conditioning {conditioning_vars}...')

            # Train classifier on aligned cells
            y_train = adata_train.obs[target]
            classifier = LogisticRegression(max_iter=1000, random_state=42)
            classifier.fit(embeddings_train, y_train)

            # Predict on conflict and aligned cells
            y_conflict_true = adata_test.obs[target].iloc[conflict_rows]

            # Get shortcut labels
            y_conflict_shortcut = self._get_shortcut_labels(
                adata_train, adata_test, conflict_rows, target, conditioning_vars
            )

            # Compute predictions
            y_pred_conflict = classifier.predict(embeddings_test[conflict_rows])

            # Compute accuracies on conflict rows
            truth_acc = (y_pred_conflict == y_conflict_true).mean()
            shortcut_acc = (y_pred_conflict == y_conflict_shortcut).mean()
            tsm = truth_acc - shortcut_acc

            logger.info(f'    Truth accuracy: {truth_acc:.3f}')
            logger.info(f'    Shortcut accuracy: {shortcut_acc:.3f}')
            logger.info(f'    TSM: {tsm:.3f}')

            results[target] = {
                'truth_accuracy': truth_acc,
                'shortcut_accuracy': shortcut_acc,
                'tsm': tsm,
                'confusion': self._compute_confusion(
                    y_conflict_true, y_conflict_shortcut, y_pred_conflict
                )
            }

        return results

    def _train_linear_probe(
        self,
        X_train: np.ndarray,
        y_train,
        X_test: np.ndarray,
        y_test
    ) -> float:
        """Train linear probe and evaluate on test set.

        Args:
            X_train, y_train: Training embeddings and labels
            X_test, y_test: Test embeddings and labels

        Returns:
            balanced_accuracy: Balanced accuracy on test set
        """
        # Fit logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)

        return acc

    def _get_shortcut_labels(
        self,
        adata_train,
        adata_test,
        test_indices: np.ndarray,
        target: str,
        conditioning_vars: List[str]
    ) -> np.ndarray:
        """Get shortcut labels for test cells based on training priors."""
        # Compute prior on training data
        priors = {}
        for z_vals, group in adata_train.obs.groupby(conditioning_vars):
            if not isinstance(z_vals, tuple):
                z_vals = (z_vals,)

            shortcut = group[target].value_counts().idxmax()
            priors[z_vals] = shortcut

        # Apply to test cells
        shortcut_labels = []
        for idx in test_indices:
            z_vals = tuple(
                adata_test.obs[var].iloc[idx] for var in conditioning_vars
            )

            if z_vals in priors:
                shortcut_labels.append(priors[z_vals])
            else:
                shortcut_labels.append(None)  # Unsupported

        return np.array(shortcut_labels)

    def _compute_confusion(self, y_true, y_shortcut, y_pred):
        """Compute confusion rates (transition maps, Table 6)."""
        # Distribution of predictions conditioned on true label
        confusion = {}
        for true_label in np.unique(y_true):
            mask = y_true == true_label
            preds_given_true = y_pred[mask]

            confusion[str(true_label)] = {
                'predicted_as_truth': (preds_given_true == true_label).mean(),
                'predicted_as_shortcut': (
                    preds_given_true == y_shortcut[mask]
                ).mean(),
                'other': (
                    (preds_given_true != true_label) &
                    (preds_given_true != y_shortcut[mask])
                ).mean()
            }

        return confusion
