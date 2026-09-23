"""Structured genomic features: typed, fitted on training rows only, leakage-labelled.

Structured fields go to the model as their own inputs (embeddings for
categories, standardised numbers with a missing-value mask) — not pasted into
the text — so an ablation can switch them off cleanly (EXP-011) and the
leakage audit can see exactly which ones a run used.

Each feature says whether it is on by default and why. Off by default:

    gene              a gene-identity shortcut (per-gene label rates differ hugely);
                      on only for ablations, never under gene holdout
    collection_method "literature only" submissions skew pathogenic — a provenance shortcut
    review_status, stars, number_submitters, submitter
                      describe how the LABEL was reviewed, not the variant:
                      forbidden by the leakage policy, cannot be switched on
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from vpdl.slm.schema import NEVER_INPUT

__all__ = ["FeatureSpec", "FEATURES", "DEFAULT_FEATURES", "StructuredEncoder", "feature_frame"]

FORBIDDEN = frozenset({"review_status", "stars", "number_submitters", "submitter"}) | NEVER_INPUT


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str                  # categorical | numeric | binary
    default: bool
    note: str


FEATURES: dict[str, FeatureSpec] = {f.name: f for f in [
    FeatureSpec("consequence", "categorical", True, "notation-level consequence (vpdl.slm.variants)"),
    FeatureSpec("variant_type", "categorical", True, "ClinVar Type"),
    FeatureSpec("origin", "categorical", True, "germline / somatic / ..."),
    FeatureSpec("chromosome_class", "categorical", True, "autosome / X / Y / MT (inheritance context)"),
    FeatureSpec("has_protein_change", "binary", True, "ClinVar name carries a p. change"),
    FeatureSpec("log10_af", "numeric", True, "gnomAD allele frequency where joined; missing mask otherwise"),
    FeatureSpec("dl_score", "numeric", True, "DL branch out-of-gene score where available (missense panel)"),
    FeatureSpec("gene", "categorical", False, "gene-identity shortcut: ablation only"),
    FeatureSpec("collection_method", "categorical", False, "provenance shortcut: ablation only"),
]}
DEFAULT_FEATURES = tuple(name for name, spec in FEATURES.items() if spec.default)


def _chromosome_class(value: Any) -> str:
    text = str(value or "").upper().replace("CHR", "")
    if text in ("X", "Y"):
        return text
    if text in ("MT", "M"):
        return "MT"
    return "autosome" if text.isdigit() else "unknown"


def feature_frame(examples: pd.DataFrame, variants: pd.DataFrame,
                  documents: pd.DataFrame | None = None,
                  frequencies: pd.DataFrame | None = None,
                  dl_scores: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Raw feature values per example (not yet encoded)."""
    v = variants.set_index("variant_id")
    frame = pd.DataFrame(index=examples.index)
    ids = examples["variant_id"]
    frame["consequence"] = ids.map(v["consequence"]).fillna("unknown")
    frame["variant_type"] = ids.map(v["variant_type"]).fillna("unknown")
    frame["origin"] = ids.map(v["origin"]).fillna("unknown")
    frame["chromosome_class"] = ids.map(v["chromosome"]).map(_chromosome_class)
    frame["has_protein_change"] = ids.map(v["hgvs_p"]).fillna("").astype(str).str.len().gt(3).astype(float)
    frame["gene"] = examples["gene"].fillna("")
    if documents is not None and "document_id" in examples:
        frame["collection_method"] = examples["document_id"].map(
            documents.set_index("document_id")["collection_method"]).fillna("unknown")
    else:
        frame["collection_method"] = "unknown"
    if frequencies is not None:
        af = ids.map(frequencies.set_index("variant_id")["allele_frequency"])
        frame["log10_af"] = np.log10(af.clip(lower=1e-8))
    else:
        frame["log10_af"] = np.nan
    protein = ids.map(v["protein_variant_id"])
    frame["dl_score"] = protein.map(dict(dl_scores or {})) if dl_scores else np.nan
    return frame


@dataclass
class StructuredEncoder:
    """Vocabularies and scalers fitted on TRAINING rows; unknown values map to index 0."""

    features: tuple[str, ...] = DEFAULT_FEATURES
    vocab: dict[str, list[str]] = field(default_factory=dict)
    scale: dict[str, tuple[float, float]] = field(default_factory=dict)
    fitted_on: str = ""

    def __post_init__(self) -> None:
        bad = [f for f in self.features if f in FORBIDDEN]
        if bad:
            raise ValueError(f"features {bad} are forbidden inputs (they describe the label's review)")
        unknown = [f for f in self.features if f not in FEATURES]
        if unknown:
            raise ValueError(f"unknown features {unknown}; known {sorted(FEATURES)}")

    @property
    def categorical(self) -> list[str]:
        return [f for f in self.features if FEATURES[f].kind == "categorical"]

    @property
    def numeric(self) -> list[str]:
        return [f for f in self.features if FEATURES[f].kind in ("numeric", "binary")]

    def fit(self, frame: pd.DataFrame, split: pd.Series | None = None) -> "StructuredEncoder":
        rows = frame if split is None else frame.loc[split.to_numpy() == "train"]
        if split is not None and rows.empty:
            raise ValueError("no training rows to fit the structured encoder on")
        self.fitted_on = "train" if split is not None else "all rows given"
        for name in self.categorical:
            self.vocab[name] = ["<unk>"] + sorted(rows[name].astype(str).unique().tolist())
        for name in self.numeric:
            values = pd.to_numeric(rows[name], errors="coerce").dropna()
            mean = float(values.mean()) if len(values) else 0.0
            std = float(values.std()) if len(values) > 1 else 1.0
            self.scale[name] = (mean, std if std and math.isfinite(std) else 1.0)
        return self

    def cardinalities(self) -> list[int]:
        return [len(self.vocab[name]) for name in self.categorical]

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        categories = np.zeros((len(frame), len(self.categorical)), dtype=np.int64)
        for j, name in enumerate(self.categorical):
            lookup = {value: i for i, value in enumerate(self.vocab[name])}
            categories[:, j] = [lookup.get(str(v), 0) for v in frame[name]]
        numbers = np.zeros((len(frame), len(self.numeric)), dtype=np.float32)
        mask = np.zeros_like(numbers)
        for j, name in enumerate(self.numeric):
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            present = np.isfinite(values)
            mean, std = self.scale[name]
            numbers[present, j] = ((values[present] - mean) / std).astype(np.float32)
            mask[:, j] = present
        return {"categorical": categories, "numeric": numbers, "numeric_mask": mask}

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({"features": list(self.features), "vocab": self.vocab,
                                          "scale": self.scale, "fitted_on": self.fitted_on}, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "StructuredEncoder":
        data = json.loads(Path(path).read_text())
        return cls(tuple(data["features"]), data["vocab"],
                   {k: tuple(v) for k, v in data["scale"].items()}, data.get("fitted_on", ""))

    def width(self) -> int:
        return len(self.numeric) * 2


def check_features(names: Iterable[str]) -> None:
    bad = [n for n in names if n in FORBIDDEN]
    if bad:
        raise ValueError(f"forbidden features {bad}")
