"""The model contract, checkpoint format discipline, and schema alignment.

Adding a model means implementing :class:`Model` and registering it in
:mod:`vpdl.models`. No other layer changes — which is what makes the
model axis of the experiment matrix cheap to extend.

Satisfies regression landmines L8 (feature schemas never silently truncate) and
L14 (checkpoint format tags are validated on load).
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:  # numpy is used only in annotations here, so it stays optional:
    import numpy as np  # checkpoint and schema logic must load without a model stack

__all__ = [
    "CHECKPOINT_FORMAT",
    "Model",
    "FittedModel",
    "align_features_to_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "derive_seed",
]

CHECKPOINT_FORMAT = "vpdl-v2"


@runtime_checkable
class Model(Protocol):
    """Structural contract shared by gbm, mlp and bilstm arms."""

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "Model": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class FittedModel:
    model: Any
    columns: list[str]
    seed: int
    split_index: int
    extra: dict[str, Any]


def derive_seed(seed: int, split_index: int | str) -> int:
    """Per-split seed, derived deterministically from the run seed.

    Exists because v1 constructed the model — and so initialised the head —
    *before* anything seeded the RNG. The first split of every process therefore
    drew its weights from process-start entropy. MMR_GENES is ordered
    (MLH1, ...), so MLH1, and only MLH1, was unseeded in every run ever done;
    two runs of one cell at one commit disagreed by 0.021 AUROC on it while the
    other three genes came back bit-identical.

    Every model's ``__init__`` must call this BEFORE constructing any parameter.

    `split_index` should be a **stable identifier for the split** — the held-out
    gene name — not its position in the iteration. A positional index shifts
    whenever a gene drops out of the panel, silently changing every downstream
    initialisation: the same cell at the same seed would then produce different
    numbers depending on which genes happened to have data that build. This
    project lost an entire gene to a build flag once already, so that is not
    hypothetical. Integers remain accepted for callers with genuinely ordinal
    splits.

    String keys are hashed with SHA-256 rather than :func:`hash`, whose salt
    changes between processes and would make runs irreproducible across
    invocations — the precise bug this function exists to prevent.
    """
    if isinstance(split_index, str):
        digest = hashlib.sha256(split_index.encode("utf-8")).digest()
        split_index = int.from_bytes(digest[:4], "big")
    return (int(seed) * 1_000_003 + int(split_index) * 9_176 + 17) % (2**31 - 1)


def align_features_to_checkpoint(
    available_columns: Sequence[str],
    checkpoint_columns: Sequence[str],
) -> list[str]:
    """Return the checkpoint's column order, or raise naming what would be lost.

    Weight transfer genuinely requires pinning feature order to the
    checkpoint's. v1 did that correctly and then silently *discarded* the eight
    richer columns the newer table carried — including ``gnomad_log10_af``,
    the one feature PROJECT_PLAN Phase 3 explicitly requires as an input.
    Fixing the mismatch moved mean ROC-AUC 0.9229 -> 0.9445, so this was never
    a cosmetic difference.

    Silent truncation is the failure mode; an explicit error naming the columns
    is the fix.
    """
    checkpoint = list(checkpoint_columns)
    available = list(available_columns)

    absent = [column for column in checkpoint if column not in available]
    if absent:
        raise ValueError(
            f"Checkpoint expects columns the table does not have: {absent}. "
            "Refusing to substitute or impute them."
        )

    discarded = [column for column in available if column not in checkpoint]
    if discarded:
        raise ValueError(
            f"Feature schema mismatch would discard {len(discarded)} column(s) "
            f"the table provides: {discarded}. The checkpoint was trained on "
            f"{len(checkpoint)} features and the table offers {len(available)}. "
            "Either retrain on the full schema or drop the extras explicitly — "
            "do not let a warm start quietly shrink the feature space."
        )

    return checkpoint


def save_checkpoint(
    path: Path | str,
    state: Any,
    fmt: str | None = CHECKPOINT_FORMAT,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint inside a format envelope."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"format": fmt, "metadata": dict(metadata or {}), "state": state}
    with open(path, "wb") as handle:
        pickle.dump(envelope, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a checkpoint, rejecting a foreign format tag.

    Deliberately asymmetric: a MISSING tag is accepted with a warning, because
    real checkpoints predating the field exist and rejecting them would break
    loading of everything already produced. A tag that is PRESENT and WRONG is
    rejected — that is a foreign or future file, and accepting it fails later,
    deep inside model construction, with an error naming neither the file nor
    its format.
    """
    path = Path(path)
    with open(path, "rb") as handle:
        envelope = pickle.load(handle)

    if not isinstance(envelope, dict) or "state" not in envelope:
        raise ValueError(f"{path} is not a vpdl checkpoint envelope.")

    recorded = envelope.get("format")
    if recorded is not None and recorded != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} has checkpoint format {recorded!r}, expected "
            f"{CHECKPOINT_FORMAT!r}. Refusing to load a foreign format."
        )

    return envelope
