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


def _window(encoder: str) -> Callable[..., Any]:
    def build(**kwargs: Any):
        from vpdl.models.seqwin import WindowEncoderClassifier
        return WindowEncoderClassifier(encoder=encoder, **kwargs)
    return build


def _fusion(**kwargs: Any):
    from vpdl.dl.fusion import FusionClassifier
    return FusionClassifier(**kwargs)


def _plm_finetune(**kwargs: Any):
    from vpdl.dl.plm.finetune import PLMFinetuneClassifier
    return PLMFinetuneClassifier(**kwargs)


# Lazily constructed: importing the registry must not require torch, so a
# tabular-only run works on a box with no CUDA stack installed.
MODELS: dict[str, Callable[..., Any]] = {
    "gbm": _gbm,
    "mlp": _mlp,
    "bilstm": _bilstm,
    # DL branch (docs/dl/DL_ARCHITECTURE.md). Window baselines share the
    # bilstm arm's inputs; fusion takes tabular + embedding modalities;
    # plm_finetune trains a protein language model end to end with PEFT.
    "aa_mlp": _window("aa_mlp"),
    "cnn": _window("cnn"),
    "bilstm_attn": _window("bilstm_attn"),
    "transformer": _window("transformer"),
    "fusion": _fusion,
    "plm_finetune": _plm_finetune,
}


def build_model(name: str, **kwargs: Any):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Known: {sorted(MODELS)}")
    return MODELS[name](**kwargs)
