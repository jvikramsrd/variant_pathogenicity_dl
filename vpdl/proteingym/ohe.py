"""The one-hot baseline, in the same hypothesis space as ProteinGym's.

ProteinNPT's ``OHE_not_augmented`` is a linear model on the flattened one-hot
encoding of the whole mutated sequence. For a single substitution at position
``p`` from ``wt`` to ``mut``, that vector is the wild-type encoding (identical
for every variant, so absorbed by the intercept) plus ``+1`` at ``(p, mut)`` and
``-1`` at ``(p, wt)``. This module builds exactly those two non-zeros per row,
so the model class is the same; only the optimiser differs (closed-form ridge
here, AdamW with weight decay 5e-3 there).

Why it works at all under random folds — and fails under the other two. Each
exact substitution occurs once per assay, so its ``(p, mut)`` weight is never
seen in training and shrinks to zero. What remains is the ``(p, wt)`` weight,
shared by every substitution at that position: the model predicts **how
sensitive the position is**. Random folds leave other substitutions at the
same position in training, so that works; modulo and contiguous folds hold out
whole positions, so it cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vpdl.proteingym.data import AMINO_ACIDS

__all__ = ["design_matrix", "OneHotRidge"]

_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}


def design_matrix(frame: pd.DataFrame, width_positions: int):
    """Sparse ``n x (width_positions * 20)`` matrix: +1 at (p, mut), -1 at (p, wt)."""
    from scipy.sparse import csr_matrix

    positions = frame["position"].to_numpy() - 1
    mut_columns = positions * len(AMINO_ACIDS) + frame["mut"].map(_INDEX).to_numpy()
    wt_columns = positions * len(AMINO_ACIDS) + frame["wt"].map(_INDEX).to_numpy()

    n = len(frame)
    rows = np.repeat(np.arange(n), 2)
    columns = np.column_stack([mut_columns, wt_columns]).ravel()
    values = np.tile([1.0, -1.0], n)
    return csr_matrix((values, (rows, columns)),
                      shape=(n, width_positions * len(AMINO_ACIDS)))


class OneHotRidge:
    """Ridge regression on :func:`design_matrix` features, for one assay."""

    def __init__(self, width_positions: int, alpha: float = 1.0) -> None:
        self.width_positions = width_positions
        self.alpha = alpha
        self.model = None

    def fit(self, frame: pd.DataFrame) -> "OneHotRidge":
        from sklearn.linear_model import Ridge

        self.model = Ridge(alpha=self.alpha).fit(
            design_matrix(frame, self.width_positions),
            frame["DMS_score"].to_numpy(dtype=float),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(design_matrix(frame, self.width_positions))
