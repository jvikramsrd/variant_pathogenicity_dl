"""PyTorch Dataset and position-grouped cross-validation splitters.

Leakage-prevention contract
---------------------------
Variants occurring at the *exact same residue position* must never appear in
both a training and a validation fold. Protein language models propagate local
context, so neighbouring substitutions share almost identical embeddings; an
ordinary random K-fold split therefore inflates validation scores. We use
:class:`~sklearn.model_selection.StratifiedGroupKFold` with ``groups = position``
which simultaneously

1. guarantees position-disjoint folds (no spatial sequence leakage), and
2. stratifies on the label so each fold keeps a similar pathogenic/benign mix.

For multi-protein datasets pass ``groups`` explicitly (e.g. the string
``"<uniprot_accession>:<position>"``) so that identical position numbers in
*different* proteins are treated as independent groups.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VariantDataset(Dataset):
    """Simple feature/label tensor dataset for the MLP head."""

    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
                 sample_weights: Optional[np.ndarray] = None) -> None:
        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        if labels is not None:
            self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        else:  # inference-only usage (e.g. VUS set)
            self.labels = torch.empty(len(self.features), dtype=torch.float32)
        if sample_weights is None:
            sample_weights = np.ones(len(self.features), dtype=np.float32)
        self.sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.sample_weights[idx]


def check_no_position_leakage(
    train_positions: Sequence[int], val_positions: Sequence[int], fold: int = -1
) -> None:
    """Raise AssertionError if any residue position appears in both splits."""
    overlap = set(train_positions) & set(val_positions)
    assert not overlap, (
        "Position leakage detected"
        + (f" in fold {fold}" if fold >= 0 else "")
        + f": {sorted(overlap)[:10]}"
    )


def make_position_group_folds(
    positions: np.ndarray,
    labels: np.ndarray,
    k_folds: int = 5,
    seed: int = 42,
    groups: Optional[Sequence[str]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build *k_folds* group-disjoint, label-stratified (train, val) splits.

    Parameters
    ----------
    positions:
        1-based residue positions; the default grouping key so that all
        variants at one residue stay inside a single fold.
    labels:
        Binary labels used for stratification across folds.
    groups:
        Optional explicit grouping keys (e.g. ``"P04637:273"``) that override
        *positions*.  Required for multi-protein pooling where raw position
        numbers collide across proteins.

    Returns
    -------
    list of ``(train_indices, val_indices)`` numpy arrays, verified to contain
    zero group overlap.
    """
    labels = np.asarray(labels)
    group_keys = np.asarray(groups) if groups is not None else np.asarray(positions)
    n_groups = len(np.unique(group_keys))
    k = min(k_folds, n_groups)
    if k < 2:
        raise ValueError(
            f"Need at least 2 distinct grouping keys to cross-validate; "
            f"got {n_groups}."
        )
    if k != k_folds:
        logger.warning("Reduced k_folds from %d to %d (only %d unique groups).",
                       k_folds, k, n_groups)

    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    dummy_x = np.zeros((len(labels), 1), dtype=np.float32)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(dummy_x, y=labels, groups=group_keys)):
        check_no_position_leakage(group_keys[train_idx], group_keys[val_idx], fold=fold_idx)
        splits.append((train_idx, val_idx))
        logger.info(
            "fold %d: %d train / %d val variants (%d / %d unique groups)",
            fold_idx, len(train_idx), len(val_idx),
            len(np.unique(group_keys[train_idx])), len(np.unique(group_keys[val_idx])),
        )
    return splits
