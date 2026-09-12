"""Batched inference over a saved stage-2 "priors-only" transfer head.

Not executed in this environment; no training, dataset processing, or model
download happens here. This module and its CLI (``scripts/predict.py``) are
static, portable code, meant to be run on the machine that has the trained
checkpoint and pandas/PyTorch installed. See ``docs/INFERENCE.md`` for exact
commands.

Scope (deliberately narrow -- see "Not supported" below)
----------------------------------------------------------
:func:`src.transfer.load_transfer_head` already versions the checkpoint
format (``TRANSFER_HEAD_FORMAT`` / ``TransferHeadFormatError``) and persists
everything a *tabular* head needs to score again: the architecture config,
per-view :class:`~sklearn.preprocessing.StandardScaler` statistics, and the
exact feature-column schema. This module wraps that for the ``arch="priors"``
case only -- a single feature view of named prior/score columns, no ESM
backbone forward pass required. That is the case a plain CSV of prior-score
features can be scored against without downloading or running an ESM-2 model.

Design points, each answering one requirement of the "design a correct and
efficient inference pipeline" objective:

* **Explicit validated input schema** -- :func:`validate_input_frame` names
  every missing identifying (``MASTER_KEY``) or feature column by name,
  rather than letting a missing column surface as a KeyError deep inside a
  ``.to_numpy()`` call.
* **Preprocessing + model versioned together** -- :func:`load_inference_model`
  delegates to :func:`src.transfer.load_transfer_head`, which already
  refuses a checkpoint written by an incompatible format
  (:class:`~src.transfer.TransferHeadFormatError`) rather than guessing at
  its payload shape.
* **Gene/feature alignment via canonical IDs** -- :func:`align_genes` routes
  every ``uniprot_id`` through :func:`src.gene_aliases.resolve_gene_id`, so
  ``"MLH1"`` and ``"P40692"`` score identically and an unresolvable/misspelled
  gene raises :class:`~src.gene_aliases.UnknownGeneError` instead of silently
  scoring under the wrong identity.
* **Only train-time-fitted preprocessing is applied** -- :func:`predict`
  uses the checkpoint's own stored scaler statistics
  (``loaded.scale_views``) and never fits anything on the frame being
  scored. Fitting on the inference input itself would be the
  preprocessing-fit-after-split leakage documented in
  ``docs/HOLDOUT_PROTOCOL.md``, just moved to inference time.
* **Explicit missing-feature behaviour** -- a NaN in any required feature
  column after alignment is a named :class:`InferenceInputError`, not a
  silent ``nan`` prediction; a caller must impute upstream using values
  consistent with training (the checkpoint does not carry a canonical
  impute policy for arbitrary future inputs, only for the columns it saw at
  fit time).
* **Batching, CPU/GPU selection, output schema, model/version metadata** --
  see :func:`predict` and :class:`PredictionBatch`.

Not supported (flagged, not silently wrong)
--------------------------------------------
* ``arch in ("esm", "concat", "gatewave")`` checkpoints, whose feature view(s)
  include ESM-2 embeddings computed from a raw sequence: scoring them
  requires loading the ESM-2 backbone named in the checkpoint's config and
  running a forward pass per example, which is a materially larger
  dependency and out of scope for this pass. :func:`load_inference_model`
  raises :class:`UnsupportedArchitectureError` naming the arch rather than
  attempting a partial score.
* Calibration (temperature/isotonic, ``src/calibration.py``) is not applied
  here; ``probability`` is the model's raw sigmoid output. Calibrating it is
  a separate, documented step -- see the "Remaining risks" list in this
  audit's final report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .gene_aliases import ACCESSION_TO_SYMBOL, resolve_gene_id
from .transfer import load_transfer_head, predict_logits

logger = logging.getLogger(__name__)

#: Columns that identify a row across every source table in this project
#: (src/extended_builder.py's MASTER_KEY). Required on every inference input
#: so predictions can be attached back to a specific variant.
MASTER_KEY: Tuple[str, ...] = ("uniprot_id", "position", "wt_aa", "mut_aa")

#: The only architecture this module scores. See module docstring "Scope".
SUPPORTED_ARCH = "priors"


class InferenceInputError(ValueError):
    """The input frame does not satisfy the model's declared schema."""


class UnsupportedArchitectureError(ValueError):
    """The checkpoint's architecture requires a feature view this module
    does not compute (an ESM-2 backbone forward pass)."""


@dataclass(frozen=True)
class LoadedInferenceModel:
    """A checkpoint ready to score input frames, plus everything
    :func:`predict`'s output needs to describe where a prediction came from.
    """
    model: torch.nn.Module
    scale_views: object  # Callable[[Sequence[np.ndarray]], List[np.ndarray]]
    feature_columns: Tuple[str, ...]
    config: Dict[str, object]
    threshold: Optional[float]
    checkpoint_path: str
    device: torch.device


def load_inference_model(checkpoint_path: Path,
                         device: Optional[torch.device] = None) -> LoadedInferenceModel:
    """Load a stage-2 ``arch="priors"`` transfer head for inference.

    Raises :class:`~src.transfer.TransferHeadFormatError` (propagated
    unchanged from ``load_transfer_head``) for an incompatible checkpoint
    format, and :class:`UnsupportedArchitectureError` for a checkpoint whose
    architecture needs an ESM-2 forward pass this module does not perform.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scale_views, payload = load_transfer_head(checkpoint_path, device=device)
    arch = payload["config"].get("arch")
    if arch != SUPPORTED_ARCH:
        raise UnsupportedArchitectureError(
            f"{checkpoint_path}: arch={arch!r} requires an ESM-2 embedding "
            f"feature view; this module only scores arch={SUPPORTED_ARCH!r} "
            "checkpoints (a single named-column feature view). Score this "
            "checkpoint with the training-time script instead, or extend "
            "this module once an embedding-computation path is added and "
            "tested against a real ESM-2 model.")
    return LoadedInferenceModel(
        model=model, scale_views=scale_views,
        feature_columns=tuple(payload["feature_columns"]),
        config=dict(payload["config"]), threshold=payload.get("threshold"),
        checkpoint_path=str(checkpoint_path), device=device)


def validate_input_frame(df: pd.DataFrame, loaded: LoadedInferenceModel) -> pd.DataFrame:
    """Check and reorder ``df`` to the checkpoint's declared schema.

    Column *order* in ``df`` is irrelevant -- a caller only needs to supply
    every named column, in any order; this returns a frame whose columns are
    ``MASTER_KEY`` followed by ``loaded.feature_columns`` in the checkpoint's
    own order, never the order pandas happened to read a CSV in.

    Raises :class:`InferenceInputError` naming every missing column, both
    identifying and feature-level -- never a bare ``KeyError``.
    """
    if len(df) == 0:
        raise InferenceInputError("Input frame has zero rows.")
    missing_key = [c for c in MASTER_KEY if c not in df.columns]
    if missing_key:
        raise InferenceInputError(
            f"Input frame is missing identifying column(s) {missing_key}; "
            f"every row must be identifiable by {list(MASTER_KEY)} so "
            "predictions can be attached back to a specific variant.")
    missing_features = [c for c in loaded.feature_columns if c not in df.columns]
    if missing_features:
        raise InferenceInputError(
            f"Input frame is missing {len(missing_features)} feature "
            f"column(s) the checkpoint expects: {missing_features[:10]}"
            + (" ..." if len(missing_features) > 10 else "") +
            f". Full expected schema ({len(loaded.feature_columns)} columns): "
            f"{list(loaded.feature_columns)}")
    return df[list(MASTER_KEY) + list(loaded.feature_columns)].reset_index(drop=True)


def align_genes(df: pd.DataFrame,
                known_genes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Resolve ``df``'s ``uniprot_id`` column to canonical accessions.

    Raises :class:`~src.gene_aliases.UnknownGeneError` for anything
    unresolvable -- a misspelled or unaliased gene identifier must fail
    loudly, never pass through as an opaque string that happens not to match
    anything downstream. If ``known_genes`` is given, additionally raises
    :class:`InferenceInputError` naming any gene outside that set: an
    entirely unseen gene should be an explicit, documented rejection, not a
    silent score computed from features the model was never trained to
    interpret for that protein.
    """
    resolved = df["uniprot_id"].map(resolve_gene_id)
    if known_genes is not None:
        known = {resolve_gene_id(g) for g in known_genes}
        unseen = sorted(set(resolved) - known)
        if unseen:
            readable = sorted(ACCESSION_TO_SYMBOL.get(acc, acc) for acc in unseen)
            raise InferenceInputError(
                f"Input contains gene(s) outside the model's known set: "
                f"{readable}. Pass known_genes=None to score genes outside "
                "the training panel, if that is intentional.")
    out = df.copy()
    out["uniprot_id"] = resolved
    return out


@dataclass(frozen=True)
class PredictionBatch:
    """The output of :func:`predict`: predictions plus enough metadata to
    trace a probability back to the exact checkpoint and threshold that
    produced it."""
    predictions: pd.DataFrame
    model_config: Dict[str, object]
    checkpoint_path: str
    threshold: float


def predict(loaded: LoadedInferenceModel, df: pd.DataFrame, *,
           known_genes: Optional[Sequence[str]] = None,
           batch_size: int = 1024,
           threshold: Optional[float] = None) -> PredictionBatch:
    """Score every row of ``df`` with ``loaded``, batched on ``loaded.device``.

    Accepts a single-row or multi-row frame identically -- there is no
    separate single-item code path, so behaviour cannot silently diverge
    between the two shapes.

    Applies ONLY ``loaded``'s own stored scaler statistics
    (:func:`~src.transfer.load_transfer_head`'s ``scale_views``); never fits
    anything on ``df``. See the module docstring's "Only train-time-fitted
    preprocessing is applied" point for why.

    Raises :class:`InferenceInputError` for a NaN in any required feature
    column after alignment -- inference does not impute silently.
    """
    validated = validate_input_frame(df, loaded)
    aligned = align_genes(validated, known_genes=known_genes)
    raw = aligned[list(loaded.feature_columns)].to_numpy(dtype=np.float64)
    nan_rows = np.isnan(raw).any(axis=1)
    if nan_rows.any():
        bad_keys = aligned.loc[nan_rows, list(MASTER_KEY)].head(20)
        raise InferenceInputError(
            f"{int(nan_rows.sum())} row(s) contain NaN in a required "
            "feature column after alignment; inference does not impute -- "
            "impute upstream using the checkpoint's training-time impute "
            f"values, or drop the row before calling predict(). First "
            f"offending keys: {bad_keys.to_dict('records')}")
    scaled = loaded.scale_views([raw])[0]
    logits = predict_logits(loaded.model, [scaled], loaded.device, batch_size=batch_size)
    proba = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    thr = threshold if threshold is not None else (
        loaded.threshold if loaded.threshold is not None else 0.5)
    out = aligned[list(MASTER_KEY)].copy()
    out["probability"] = proba
    out["predicted_label"] = (proba >= thr).astype(int)
    out["threshold_used"] = thr
    out["checkpoint"] = loaded.checkpoint_path
    return PredictionBatch(predictions=out, model_config=loaded.config,
                           checkpoint_path=loaded.checkpoint_path, threshold=float(thr))


__all__ = [
    "MASTER_KEY", "SUPPORTED_ARCH",
    "InferenceInputError", "UnsupportedArchitectureError",
    "LoadedInferenceModel", "PredictionBatch",
    "load_inference_model", "validate_input_frame", "align_genes", "predict",
]
