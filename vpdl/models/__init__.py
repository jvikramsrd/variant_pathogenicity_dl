"""Model registry. One axis of the experiment matrix.

Arms are constructed by name so the CLI, the sweep driver and the tests all
reach them the same way, and adding one touches no other layer.

    gbm     boosted trees over curated features -- expected strong arm
    mlp     residual MLP head -- continuity with every v1 number
    bilstm  recurrent baseline over sequence windows -- included to be beaten,
            see vpdl/models/bilstm.py for why that is the honest framing
"""

from __future__ import annotations

from typing import Any, Callable

from vpdl.models.base import (
    CHECKPOINT_FORMAT,
    FittedModel,
    Model,
    align_features_to_checkpoint,
    derive_seed,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "FittedModel",
    "Model",
    "MODELS",
    "build_model",
    "align_features_to_checkpoint",
    "derive_seed",
    "load_checkpoint",
    "save_checkpoint",
]


def _gbm(**kwargs: Any):
    from vpdl.models.gbm import GBMClassifier
    return GBMClassifier(**kwargs)


def _mlp(**kwargs: Any):
    from vpdl.models.mlp import MLPClassifier
    return MLPClassifier(**kwargs)


def _bilstm(**kwargs: Any):
    from vpdl.models.bilstm import BiLSTMClassifier
    return BiLSTMClassifier(**kwargs)


# Lazily constructed: importing the registry must not require torch, so a
# tabular-only run works on a box with no CUDA stack installed.
MODELS: dict[str, Callable[..., Any]] = {
    "gbm": _gbm,
    "mlp": _mlp,
    "bilstm": _bilstm,
}


def build_model(name: str, **kwargs: Any):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Known: {sorted(MODELS)}")
    return MODELS[name](**kwargs)
