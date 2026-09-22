"""The DL branch's hooks into :func:`vpdl.experiment.run_cell`.

`run_cell` owns the protocol — folds, inner split, threshold, metrics,
provenance. This module supplies only what the DL models add to it:

* **embedding blocks** from the feature store, appended to the tabular matrix
  and standardised with statistics from the fold's inner-TRAINING rows only
  (the same rule :class:`vpdl.features.FeatureMatrix` follows);
* **modality maps and availability masks** for the fusion model, derived from
  the feature groups in :mod:`vpdl.features` (so an ablation group and a fusion
  modality are the same set of columns);
* **extra-row scoring and export** — held-out-gene rows beyond the labelled
  test set (functional-validation variants, or everything, for the DL output),
  with MC-dropout uncertainty and fused representations where the model has them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.dl.feature_store import FeatureStore
from vpdl.features import group_of
from vpdl.splits import variant_keys

logger = logging.getLogger(__name__)

__all__ = ["MODALITY_OF_GROUP", "EMBEDDING_MODALITY", "DLContext", "Augmenter",
           "score_extra_rows", "modality_of_column"]

# Feature group -> fusion modality. `prior_scores` (AlphaMissense) is another
# model's output, not a DL modality: it is kept apart so the population
# ablation (Model A vs B) is not blocked by the L11 proxy guard, and so "all DL
# modalities" means exactly that.
MODALITY_OF_GROUP: dict[str | None, str] = {
    "gnomad": "population", "structure": "structure", "genomic": "genomic",
    "domains": "annotation", "prior_scores": "external_priors", None: "other",
}
EMBEDDING_MODALITY = {"esm1b": "sequence_plm", "esm2": "sequence_plm",
                      "zeroshot": "zeroshot", "structure": "structure_embedding"}


def modality_of_column(column: str) -> str:
    return MODALITY_OF_GROUP.get(group_of(column), "other")


@dataclass
class DLContext:
    store_root: str | Path = "features"
    functional_keys: frozenset[str] = frozenset()
    export_dir: str | Path | None = None
    sequences: Mapping[str, str] | None = None
    uncertainty_samples: int = 30
    store: FeatureStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = FeatureStore(self.store_root)
        self.functional_keys = frozenset(self.functional_keys)

    def resolve(self, spec: str, fold=None):
        """``(entry, representation)`` — per-fold specs resolve against `fold`.

        ``perfold=<mapping.json>:<representation>`` names a JSON file mapping
        each fold (held-out gene) to ``family/model_tag/key`` — one entry per
        strict-mode pretrained backbone. The fold's backbone must have been
        pretrained with that fold's gene held out, or this raises.
        """
        if not spec.startswith("perfold="):
            return self.store.resolve(spec)
        location, _, representation = spec.rpartition(":")
        mapping = json.loads(Path(location[len("perfold="):]).read_text())
        if fold is None or fold.name not in mapping:
            raise ValueError(f"{spec}: no entry for fold {getattr(fold, 'name', None)!r}; "
                             f"mapping has {sorted(mapping)}")
        entry, _ = self.store.resolve(f"{mapping[fold.name]}:{representation}")
        _check_pretraining_exposure(entry, fold)
        return entry, representation

    def embeddings(self, spec: str, ids: Sequence[str], fold=None) -> np.ndarray:
        entry, representation = self.resolve(spec, fold)
        if representation in entry.blocks:
            values = entry.lookup(representation, ids)
        else:
            from vpdl.dl.plm.embed import derive
            values = derive(entry.blocks_for(ids), representation)
        return values.reshape(len(ids), -1)

    def feature_version(self, specs: Iterable[str]) -> str | None:
        specs = list(specs)
        versions = []
        for spec in specs:
            if spec.startswith("perfold="):
                location = spec.rpartition(":")[0]
                mapping = json.loads(Path(location[len("perfold="):]).read_text())
                versions.append("perfold{" + ",".join(f"{k}={v}" for k, v in
                                                      sorted(mapping.items())) + "}")
            else:
                versions.append(spec.partition(":")[0])
        return ";".join(versions) if versions else None

    def augmenter(self, config, columns: Sequence[str], fit_frame: pd.DataFrame,
                  fold=None) -> "Augmenter":
        return Augmenter(self, config, list(columns), fit_frame, fold)


def _check_pretraining_exposure(entry, fold) -> None:
    """A strict-mode backbone must have held out exactly this fold's gene."""
    from vpdl.dl.leakage import LeakageError
    from vpdl.sources.uniprot import MMR_ACCESSIONS

    adaptation = (entry.meta.get("identity") or {}).get("adaptation") or {}
    if not adaptation:
        return                                   # P0: the original model, nothing adapted
    mode, holdout = adaptation.get("pretrain_corpus_mode"), adaptation.get("pretrain_holdout")
    expected = {MMR_ACCESSIONS[g][0] for g in fold.test_genes if g in MMR_ACCESSIONS}
    if mode == "strict" and holdout not in expected:
        raise LeakageError(
            f"fold {fold.name}: its embeddings come from a backbone pretrained with "
            f"{holdout!r} held out, not the fold's gene {sorted(expected)}. That backbone saw "
            "this fold's protein family during pretraining.")


class Augmenter:
    """Appends standardised embedding blocks (and fusion masks) to a matrix."""

    def __init__(self, context: DLContext, config, columns: list[str],
                 fit_frame: pd.DataFrame, fold=None) -> None:
        self.context = context
        self.config = config
        self.columns = columns
        self.fold = fold
        self.fusion = config.model == "fusion"
        ids = list(variant_keys(fit_frame))
        self.stats: list[tuple[str, np.ndarray, np.ndarray]] = []
        for spec in config.embedding_blocks:
            values = context.embeddings(spec, ids, fold)
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
            self.stats.append((spec, mean, np.where(scale > 1e-8, scale, 1.0)))
        # Modality layout of the augmented matrix: tabular groups, then blocks.
        self.modalities: dict[str, list[int]] = {}
        for index, column in enumerate(columns):
            self.modalities.setdefault(modality_of_column(column), []).append(index)
        offset = len(columns)
        for spec, mean, _ in self.stats:
            family = (context.resolve(spec, fold)[0].meta["family"]
                      if spec.startswith("perfold=") else spec.split("/")[0])
            name = EMBEDDING_MODALITY.get(family, family)
            if name in self.modalities:
                name = f"{name}:{spec.partition(':')[2]}"
            self.modalities[name] = list(range(offset, offset + len(mean)))
            offset += len(mean)
        self.mask_modalities = [m for m in self.modalities
                                if all(i < len(columns) for i in self.modalities[m])]
        self.masks = ({m: offset + i for i, m in enumerate(self.mask_modalities)}
                      if self.fusion else {})

    def __call__(self, frame: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        ids = list(variant_keys(frame))
        parts = [np.asarray(X, dtype=np.float32)]
        for spec, mean, scale in self.stats:
            parts.append(((self.context.embeddings(spec, ids, self.fold) - mean) / scale)
                         .astype(np.float32))
        if self.fusion:
            # Availability from RAW values: after imputation every row looks
            # available, which is exactly what the mask must not believe.
            raw = frame.reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce")
            for modality in self.mask_modalities:
                cols = [self.columns[i] for i in self.modalities[modality]]
                parts.append(raw[cols].notna().any(axis=1).to_numpy(np.float32)[:, None])
        return np.concatenate(parts, axis=1)

    def fusion_kwargs(self) -> dict[str, Any]:
        return {"modalities": self.modalities, "masks": self.masks}


def score_extra_rows(table: pd.DataFrame, fold, config, model, matrix, augment, score,
                     functional_keys: set[str], context: DLContext | None,
                     out_dir: Path | str, test_frame: pd.DataFrame,
                     X_test: np.ndarray) -> pd.DataFrame:
    """Score held-out-gene rows (and export representations) for one fold."""
    if config.score_rows == "none":
        rows, X = test_frame, X_test
    else:
        rows = table[table["gene"].isin(fold.test_genes)]
        if config.score_rows == "functional":
            rows = rows[np.isin(variant_keys(rows), list(functional_keys))]
        rows = rows.reset_index(drop=True)
        X = matrix.transform(rows)
        if augment is not None:
            X = augment(rows, X)
    if not len(rows):
        return pd.DataFrame()
    scores = np.asarray(score(rows, X), dtype=float)
    uncertainty = np.full(len(rows), np.nan)
    method = None
    if hasattr(model, "predict_with_uncertainty"):
        mean, uncertainty = model.predict_with_uncertainty(
            X, context.uncertainty_samples if context else None)
        method = "mc_dropout"
    out = pd.DataFrame({"cell": config.slug, "fold": fold.name,
                        "gene": rows["gene"].to_numpy(), "variant_key": variant_keys(rows),
                        "score": scores, "uncertainty": uncertainty,
                        "uncertainty_method": method})

    if context is not None and context.export_dir:
        representation = None
        if hasattr(model, "represent"):
            representation = model.represent(X)
        elif hasattr(model, "represent_frames") and context.sequences:
            representation = model.represent_frames(rows, X, context.sequences)
        if representation is not None:
            export = Path(context.export_dir)
            export.mkdir(parents=True, exist_ok=True)
            token = fold.name.replace(":", "-")
            np.savez_compressed(export / f"representations_{config.slug}__{token}.npz",
                                variant_keys=out["variant_key"].to_numpy(),
                                gene=out["gene"].to_numpy(), representation=representation,
                                score=scores, uncertainty=uncertainty)
    return out
