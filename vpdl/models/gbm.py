"""Gradient-boosted trees over the curated feature table.

Expected to be the strong arm, and the reason is empirical rather than
aesthetic: v1's own grid showed a curated-feature head with no protein-language
input beating every ESM cell it was compared against. Small-n heterogeneous
tabular data with informative missingness is what boosted trees are for.

Backends are tried in order (xgboost, lightgbm, sklearn) so the arm still runs
on a machine where the ARM wheels for the first two are awkward — which the DGX
Spark may well be.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vpdl.models.base import derive_seed

logger = logging.getLogger(__name__)

__all__ = ["GBMClassifier", "available_backend"]


def available_backend(preferred: str | None = None) -> str:
    order = [preferred] if preferred else []
    order += ["xgboost", "lightgbm", "sklearn"]
    for name in order:
        if name is None:
            continue
        if name == "sklearn":
            return "sklearn"
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return "sklearn"


class GBMClassifier:
    """Boosted-tree arm. Honours the same seeding contract as every other model."""

    def __init__(
        self,
        seed: int = 42,
        split_index: int = 0,
        backend: str | None = None,
        n_estimators: int = 600,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        **kwargs: Any,
    ) -> None:
        # Seed first, construct second -- see derive_seed's docstring for the
        # incident this ordering exists to prevent.
        self.seed = derive_seed(seed, split_index)
        np.random.seed(self.seed % (2**32))

        self.backend = available_backend(backend)
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            **kwargs,
        )
        self.model = self._build()

    def _build(self):
        if self.backend == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                random_state=self.seed,
                eval_metric="logloss",
                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,
                **self.params,
            )
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                random_state=self.seed, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=5, verbose=-1, **self.params,
            )

        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            random_state=self.seed,
            max_iter=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            max_depth=self.params["max_depth"],
        )

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        logger.info("GBM backend=%s seed=%d n=%d", self.backend, self.seed, len(y))
        if self.backend == "sklearn":
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> np.ndarray | None:
        return getattr(self.model, "feature_importances_", None)
