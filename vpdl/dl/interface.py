"""The DL branch's output contract — frozen here for the future reasoning branch.

What is fixed now:

* :class:`ModalityRepresentation` — the one shape every branch emits per
  variant: an embedding, an optional score and uncertainty, and metadata.
* :class:`DLRepresentation` — the DL branch's instance of it.
* :data:`DL_OUTPUT_SCHEMA` — JSON Schema for one exported record (the JSONL the
  DL branch writes), validated by :func:`validate_dl_record` with no
  third-party dependency.
* :class:`ReasoningRepresentation` — the SAME shape, named, so a future fusion
  layer can accept both without special cases. It is a type only: nothing in
  this repository produces one, and nothing here implements reasoning.

What the future fusion layer is expected to do (not implemented): take a
``DLRepresentation`` and a ``ReasoningRepresentation`` for the same
``variant_id`` and register each as a modality of
:func:`vpdl.dl.fusion.build_fusion_net` — modalities are a name and a width, so
adding one requires no change to the DL encoders, which stay frozen.

Semantics that are part of the contract:

* ``pathogenicity_score`` is a probability-like score in [0, 1], higher = more
  likely pathogenic, produced by a model that did NOT train on this variant's
  gene (out-of-gene prediction from its leave-one-gene-out fold). It is a
  research output, not a clinical classification.
* ``uncertainty`` is the standard deviation of that probability (MC dropout or
  a seed ensemble; ``uncertainty_method`` says which).
* ``embedding`` is the fused penultimate representation, float32, of width
  ``embedding_dim``; in JSONL it is either an inline list or a reference
  ``{"file": ..., "row": ...}`` into an ``.npy`` written beside it.
* ``model_version`` / ``feature_version`` identify the checkpoint and the
  feature-store entries; ``dataset_version`` the canonical table's sha256.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vpdl.dl import DL_SCHEMA_VERSION

__all__ = ["ModalityRepresentation", "DLRepresentation", "ReasoningRepresentation",
           "DL_OUTPUT_SCHEMA", "validate_dl_record", "write_dl_outputs", "read_dl_outputs"]


@dataclass
class ModalityRepresentation:
    variant_id: str
    embedding: np.ndarray
    score: float | None = None
    uncertainty: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    modality: str = "unspecified"

    def __post_init__(self) -> None:
        self.embedding = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        if self.score is not None and not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(f"{self.variant_id}: score {self.score} outside [0, 1]")
        if self.uncertainty is not None and float(self.uncertainty) < 0:
            raise ValueError(f"{self.variant_id}: negative uncertainty")


@dataclass
class DLRepresentation(ModalityRepresentation):
    modality: str = "dl"

    def to_record(self, embedding_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
        meta = dict(self.metadata)
        record = {
            "schema_version": DL_SCHEMA_VERSION,
            "variant_id": self.variant_id,
            "gene": meta.pop("gene"),
            "embedding": dict(embedding_ref) if embedding_ref else self.embedding.tolist(),
            "embedding_dim": int(self.embedding.shape[0]),
            "pathogenicity_score": None if self.score is None else float(self.score),
            "uncertainty": None if self.uncertainty is None else float(self.uncertainty),
            "uncertainty_method": meta.pop("uncertainty_method", None),
            "model_version": meta.pop("model_version"),
            "feature_version": meta.pop("feature_version"),
            "dataset_version": meta.pop("dataset_version", None),
            "fold": meta.pop("fold", None),
            "metadata": meta,
        }
        validate_dl_record(record)
        return record


@dataclass
class ReasoningRepresentation(ModalityRepresentation):
    """Placeholder TYPE for the future reasoning branch. Produced by nothing here."""

    modality: str = "reasoning"


DL_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "vpdl DL branch output record",
    "type": "object",
    "required": ["schema_version", "variant_id", "gene", "embedding", "embedding_dim",
                 "pathogenicity_score", "uncertainty", "model_version", "feature_version"],
    "properties": {
        "schema_version": {"const": DL_SCHEMA_VERSION},
        "variant_id": {"type": "string", "pattern": r"^[A-Z0-9]+:[0-9]+:[A-Z]>[A-Z]$"},
        "gene": {"type": "string"},
        "embedding": {"oneOf": [
            {"type": "array", "items": {"type": "number"}},
            {"type": "object", "required": ["file", "row"],
             "properties": {"file": {"type": "string"}, "row": {"type": "integer"}}}]},
        "embedding_dim": {"type": "integer", "minimum": 1},
        "pathogenicity_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "uncertainty": {"type": ["number", "null"], "minimum": 0},
        "uncertainty_method": {"type": ["string", "null"]},
        "model_version": {"type": "string"},
        "feature_version": {"type": "string"},
        "dataset_version": {"type": ["string", "null"]},
        "fold": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}


def validate_dl_record(record: Mapping[str, Any]) -> None:
    """Check one record against :data:`DL_OUTPUT_SCHEMA` (dependency-free subset)."""
    import re

    schema = DL_OUTPUT_SCHEMA
    missing = [k for k in schema["required"] if k not in record]
    if missing:
        raise ValueError(f"DL record missing {missing}")
    extra = set(record) - set(schema["properties"])
    if extra:
        raise ValueError(f"DL record has undeclared fields {sorted(extra)}")
    if record["schema_version"] != DL_SCHEMA_VERSION:
        raise ValueError(f"schema_version {record['schema_version']!r} != {DL_SCHEMA_VERSION}")
    if not re.match(schema["properties"]["variant_id"]["pattern"], str(record["variant_id"])):
        raise ValueError(f"malformed variant_id {record['variant_id']!r}")
    embedding = record["embedding"]
    if isinstance(embedding, list):
        if len(embedding) != record["embedding_dim"]:
            raise ValueError("embedding length != embedding_dim")
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in embedding):
            raise ValueError("embedding contains non-finite values")
    elif not (isinstance(embedding, dict) and {"file", "row"} <= set(embedding)):
        raise ValueError("embedding must be a list or a {file, row} reference")
    score = record["pathogenicity_score"]
    if score is not None and not (0.0 <= float(score) <= 1.0):
        raise ValueError(f"pathogenicity_score {score} outside [0, 1]")
    if record["uncertainty"] is not None and float(record["uncertainty"]) < 0:
        raise ValueError("uncertainty must be >= 0")
    for key in ("model_version", "feature_version"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"{key} must be a non-empty string")


def write_dl_outputs(representations: Sequence[DLRepresentation], out_dir: Path | str,
                     name: str = "dl_outputs", inline: bool = False) -> dict[str, Path]:
    """JSONL records + (unless `inline`) one ``.npy`` of embeddings they point into."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"jsonl": out_dir / f"{name}.jsonl"}
    if not inline:
        paths["embeddings"] = out_dir / f"{name}.embeddings.npy"
        np.save(paths["embeddings"], np.stack([r.embedding for r in representations])
                if representations else np.zeros((0, 0), dtype=np.float32))
    with open(paths["jsonl"], "w", encoding="utf-8") as handle:
        for row, representation in enumerate(representations):
            ref = None if inline else {"file": paths["embeddings"].name, "row": row}
            handle.write(json.dumps(representation.to_record(ref)) + "\n")
    (out_dir / f"{name}.schema.json").write_text(json.dumps(DL_OUTPUT_SCHEMA, indent=2))
    return paths


def read_dl_outputs(jsonl: Path | str) -> list[DLRepresentation]:
    """Round-trip reader, resolving embedding references."""
    jsonl = Path(jsonl)
    cache: dict[str, np.ndarray] = {}
    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        validate_dl_record(record)
        embedding = record["embedding"]
        if isinstance(embedding, dict):
            array = cache.setdefault(embedding["file"],
                                     np.load(jsonl.parent / embedding["file"], mmap_mode="r"))
            embedding = np.asarray(array[embedding["row"]])
        meta = dict(record.get("metadata") or {})
        meta.update({k: record[k] for k in ("gene", "model_version", "feature_version",
                                            "dataset_version", "fold", "uncertainty_method")})
        out.append(DLRepresentation(record["variant_id"], embedding,
                                    record["pathogenicity_score"], record["uncertainty"], meta))
    return out
