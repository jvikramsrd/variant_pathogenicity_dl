"""The SLM's output contract, and how it reads the DL branch's — the only link between them.

The DL branch froze its side first (``vpdl.dl.interface``): one shape per
variant (``ModalityRepresentation``), its own instance (``DLRepresentation``),
a JSON Schema, and a NAMED but unimplemented ``ReasoningRepresentation`` for
"the future reasoning branch". This is that branch, so :class:`SLMRepresentation`
IS a ``ReasoningRepresentation`` — a future fusion layer can register both
modalities without a special case, exactly as the DL branch designed.

Semantics, in the DL contract's terms:

* ``score`` = ``pathogenic_probability`` = p(Pathogenic) + p(Likely pathogenic),
  after calibration, in [0, 1]. It is NOT renormalised against VUS: a variant
  with p(VUS)=0.9 has a low score on purpose.
* ``uncertainty`` = predictive entropy of the five-class distribution, in nats,
  or the ensemble / MC-dropout standard deviation when ``uncertainty_method``
  says so.
* ``embedding`` = the fused representation (text + structured + optional DL),
  float32, width ``slm_embedding_dim``.
* ``evidence_embedding`` = mean of the evidence units' vectors, or null when the
  evidence heads were off.

Reading the DL side (:class:`DLInput`) enforces two things the leakage report
depends on: dimensions come from the records themselves (never hard-coded), and
a variant with no DL record gets a zero vector WITH ``mask = 0``, never an
imputed embedding.

This module does not fuse anything. Fusion is out of scope for this phase
(docs/slm/GENOMIC_SLM_DL_INTERFACE.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vpdl.dl.interface import ReasoningRepresentation
from vpdl.slm.schema import CLASSES, SLM_SCHEMA_VERSION

__all__ = ["SLMRepresentation", "SLM_OUTPUT_SCHEMA", "validate_slm_record", "write_slm_outputs",
           "read_slm_outputs", "DLInput"]


@dataclass
class SLMRepresentation(ReasoningRepresentation):
    """One variant's SLM output. ``modality`` stays "reasoning" — the DL branch's name for it."""

    class_probabilities: dict[str, float] = field(default_factory=dict)
    evidence_embedding: np.ndarray | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    acmg: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = False
    quality_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.evidence_embedding is not None:
            self.evidence_embedding = np.asarray(self.evidence_embedding, dtype=np.float32).reshape(-1)
        if self.class_probabilities:
            missing = [c for c in CLASSES if c not in self.class_probabilities]
            if missing:
                raise ValueError(f"{self.variant_id}: class_probabilities missing {missing}")
            total = sum(self.class_probabilities.values())
            if not 0.99 <= total <= 1.01:
                raise ValueError(f"{self.variant_id}: class probabilities sum to {total:.3f}")

    @property
    def pathogenic_probability(self) -> float | None:
        if not self.class_probabilities:
            return None
        return self.class_probabilities["pathogenic"] + self.class_probabilities["likely_pathogenic"]

    @property
    def benign_probability(self) -> float | None:
        if not self.class_probabilities:
            return None
        return self.class_probabilities["benign"] + self.class_probabilities["likely_benign"]

    def to_record(self, embedding_ref: Mapping[str, Any] | None = None,
                  evidence_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
        meta = dict(self.metadata)
        record = {
            "schema_version": SLM_SCHEMA_VERSION,
            "variant_id": self.variant_id,
            "protein_variant_id": meta.pop("protein_variant_id", None),
            "gene": meta.pop("gene", ""),
            "slm_embedding": dict(embedding_ref) if embedding_ref else self.embedding.tolist(),
            "slm_embedding_dim": int(self.embedding.shape[0]),
            "evidence_embedding": (dict(evidence_ref) if evidence_ref else
                                   (self.evidence_embedding.tolist()
                                    if self.evidence_embedding is not None else None)),
            "class_probabilities": {k: float(v) for k, v in self.class_probabilities.items()},
            "pathogenic_probability": self.pathogenic_probability,
            "benign_probability": self.benign_probability,
            "vus_probability": self.class_probabilities.get("vus") if self.class_probabilities else None,
            "uncertainty": None if self.uncertainty is None else float(self.uncertainty),
            "uncertainty_method": meta.pop("uncertainty_method", None),
            "abstained": bool(self.abstained),
            "evidence": list(self.evidence),
            "acmg": list(self.acmg),
            "explanation": self.explanation,
            "model_version": meta.pop("model_version"),
            "feature_version": meta.pop("feature_version"),
            "dataset_version": meta.pop("dataset_version", None),
            "split": meta.pop("split", None),
            "calibration": meta.pop("calibration", None),
            "quality_flags": list(self.quality_flags),
            "metadata": meta,
        }
        validate_slm_record(record)
        return record


SLM_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "vpdl genomic SLM output record",
    "type": "object",
    "required": ["schema_version", "variant_id", "gene", "slm_embedding", "slm_embedding_dim",
                 "class_probabilities", "pathogenic_probability", "uncertainty", "model_version",
                 "feature_version"],
    "properties": {
        "schema_version": {"const": SLM_SCHEMA_VERSION},
        "variant_id": {"type": "string", "pattern": r"^clinvar:[0-9]+$|^[A-Z0-9]+:[0-9]+:[A-Z]>[A-Z]$"},
        "protein_variant_id": {"type": ["string", "null"],
                               "pattern": r"^[A-Z0-9]+:[0-9]+:[A-Z]>[A-Z]$"},
        "gene": {"type": "string"},
        "slm_embedding": {"oneOf": [{"type": "array", "items": {"type": "number"}},
                                    {"type": "object", "required": ["file", "row"]}]},
        "slm_embedding_dim": {"type": "integer", "minimum": 1},
        "evidence_embedding": {"type": ["array", "object", "null"]},
        "class_probabilities": {"type": "object"},
        "pathogenic_probability": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "benign_probability": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "vus_probability": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "uncertainty": {"type": ["number", "null"], "minimum": 0},
        "uncertainty_method": {"type": ["string", "null"]},
        "abstained": {"type": "boolean"},
        "evidence": {"type": "array"},
        "acmg": {"type": "array"},
        "explanation": {"type": "string"},
        "model_version": {"type": "string"},
        "feature_version": {"type": "string"},
        "dataset_version": {"type": ["string", "null"]},
        "split": {"type": ["string", "null"]},
        "calibration": {"type": ["string", "object", "null"]},
        "quality_flags": {"type": "array"},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}


def validate_slm_record(record: Mapping[str, Any]) -> None:
    """Dependency-free check against :data:`SLM_OUTPUT_SCHEMA` (mirrors the DL validator)."""
    import re

    schema = SLM_OUTPUT_SCHEMA
    missing = [k for k in schema["required"] if k not in record]
    if missing:
        raise ValueError(f"SLM record missing {missing}")
    extra = set(record) - set(schema["properties"])
    if extra:
        raise ValueError(f"SLM record has undeclared fields {sorted(extra)}")
    if record["schema_version"] != SLM_SCHEMA_VERSION:
        raise ValueError(f"schema_version {record['schema_version']!r} != {SLM_SCHEMA_VERSION}")
    if not re.match(schema["properties"]["variant_id"]["pattern"], str(record["variant_id"])):
        raise ValueError(f"malformed variant_id {record['variant_id']!r}")
    protein = record.get("protein_variant_id")
    if protein is not None and not re.match(schema["properties"]["protein_variant_id"]["pattern"], str(protein)):
        raise ValueError(f"malformed protein_variant_id {protein!r}")
    embedding = record["slm_embedding"]
    if isinstance(embedding, list):
        if len(embedding) != record["slm_embedding_dim"]:
            raise ValueError("slm_embedding length != slm_embedding_dim")
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in embedding):
            raise ValueError("slm_embedding contains non-finite values")
    elif not (isinstance(embedding, dict) and {"file", "row"} <= set(embedding)):
        raise ValueError("slm_embedding must be a list or a {file, row} reference")
    probabilities = record["class_probabilities"]
    if probabilities:
        if sorted(probabilities) != sorted(CLASSES):
            raise ValueError(f"class_probabilities must hold exactly {sorted(CLASSES)}")
        total = sum(probabilities.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"class probabilities sum to {total:.3f}")
    for key in ("pathogenic_probability", "benign_probability", "vus_probability"):
        value = record.get(key)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{key} {value} outside [0, 1]")
    if record["uncertainty"] is not None and float(record["uncertainty"]) < 0:
        raise ValueError("uncertainty must be >= 0")
    for key in ("model_version", "feature_version"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"{key} must be a non-empty string")


def write_slm_outputs(representations: Sequence[SLMRepresentation], out_dir: Path | str,
                      name: str = "slm_outputs", inline: bool = False) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"jsonl": out_dir / f"{name}.jsonl"}
    if not inline and representations:
        paths["embeddings"] = out_dir / f"{name}.embeddings.npy"
        np.save(paths["embeddings"], np.stack([r.embedding for r in representations]))
        if all(r.evidence_embedding is not None for r in representations):
            paths["evidence_embeddings"] = out_dir / f"{name}.evidence.npy"
            np.save(paths["evidence_embeddings"],
                    np.stack([r.evidence_embedding for r in representations]))
    with open(paths["jsonl"], "w", encoding="utf-8") as handle:
        for row, representation in enumerate(representations):
            reference = None if inline else {"file": paths["embeddings"].name, "row": row}
            evidence_reference = ({"file": paths["evidence_embeddings"].name, "row": row}
                                  if "evidence_embeddings" in paths else None)
            handle.write(json.dumps(representation.to_record(reference, evidence_reference)) + "\n")
    (out_dir / f"{name}.schema.json").write_text(json.dumps(SLM_OUTPUT_SCHEMA, indent=2))
    return paths


def read_slm_outputs(jsonl: Path | str) -> list[SLMRepresentation]:
    jsonl = Path(jsonl)
    cache: dict[str, np.ndarray] = {}

    def resolve(value):
        if isinstance(value, dict):
            array = cache.setdefault(value["file"], np.load(jsonl.parent / value["file"], mmap_mode="r"))
            return np.asarray(array[value["row"]])
        return None if value is None else np.asarray(value, dtype=np.float32)

    out = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        validate_slm_record(record)
        metadata = dict(record.get("metadata") or {})
        metadata.update({k: record[k] for k in ("gene", "protein_variant_id", "model_version",
                                                "feature_version", "dataset_version", "split",
                                                "uncertainty_method", "calibration")})
        out.append(SLMRepresentation(
            variant_id=record["variant_id"], embedding=resolve(record["slm_embedding"]),
            score=record["pathogenic_probability"], uncertainty=record["uncertainty"],
            metadata=metadata, class_probabilities=record["class_probabilities"],
            evidence_embedding=resolve(record["evidence_embedding"]),
            evidence=record.get("evidence", []), explanation=record.get("explanation", ""),
            acmg=record.get("acmg", []), abstained=bool(record.get("abstained", False)),
            quality_flags=list(record.get("quality_flags", []))))
    return out


@dataclass
class DLInput:
    """DL representations, read through the DL branch's own reader, keyed by protein variant id."""

    by_variant: dict[str, np.ndarray]
    scores: dict[str, float]
    folds: dict[str, str | None]
    dimension: int
    versions: dict[str, Any]

    @classmethod
    def from_outputs(cls, jsonl: Path | str) -> "DLInput":
        from vpdl.dl.interface import read_dl_outputs
        representations = read_dl_outputs(jsonl)
        if not representations:
            raise ValueError(f"{jsonl}: no DL representations")
        dimensions = {int(r.embedding.shape[0]) for r in representations}
        if len(dimensions) != 1:
            raise ValueError(f"{jsonl}: DL embeddings of different widths {sorted(dimensions)}")
        versions = {"model_version": sorted({r.metadata.get("model_version") for r in representations}),
                    "feature_version": sorted({r.metadata.get("feature_version") for r in representations}),
                    "dataset_version": sorted({r.metadata.get("dataset_version") for r in representations}),
                    "source": str(jsonl)}
        return cls({r.variant_id: r.embedding for r in representations},
                   {r.variant_id: r.score for r in representations if r.score is not None},
                   {r.variant_id: r.metadata.get("fold") for r in representations},
                   dimensions.pop(), versions)

    def matrix(self, protein_variant_ids: Iterable[str | None]) -> tuple[np.ndarray, np.ndarray]:
        """(embeddings [N, D], mask [N]) — a variant without a DL record gets zeros and mask 0."""
        ids = list(protein_variant_ids)
        embeddings = np.zeros((len(ids), self.dimension), dtype=np.float32)
        mask = np.zeros(len(ids), dtype=np.float32)
        for row, identifier in enumerate(ids):
            vector = self.by_variant.get(identifier) if isinstance(identifier, str) else None
            if vector is not None:
                embeddings[row] = vector
                mask[row] = 1.0
        return embeddings, mask

    def fold_check(self, protein_variant_ids: Iterable[str | None], genes: Iterable[str]) -> dict[str, Any]:
        """Are the DL scores out-of-fold for these variants? (Leakage audit input.)

        A DL record's ``fold`` is the gene held out when it was produced. If it
        equals the variant's gene, that variant's own label did not train the DL
        model. Anything else is reported, not silently accepted.
        """
        matched = mismatched = unknown = 0
        for identifier, gene in zip(protein_variant_ids, genes):
            if not isinstance(identifier, str) or identifier not in self.by_variant:
                continue
            fold = self.folds.get(identifier)
            if fold is None:
                unknown += 1
            elif str(fold) == str(gene):
                matched += 1
            else:
                mismatched += 1
        return {"out_of_fold": matched, "in_fold_or_other": mismatched, "fold_unknown": unknown,
                "note": "'in_fold_or_other' means the DL model that produced the score may have "
                        "trained on this variant's gene: its score is not out-of-gene for it."}
