"""Tokenisation cache and batch assembly.

Tokenised inputs are cached as ``.npy`` under a key that includes the
tokenizer's fingerprint, the maximum length and a hash of the exact texts —
unchanged data is never tokenised twice, and a cache built with another
tokenizer or other texts can never be picked up (the key differs).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.slm.build.examples import class_index, code_multi_hot
from vpdl.slm.modeling.backbone import Backbone, tokenizer_fingerprint
from vpdl.slm.schema import CLASSES, as_list
from vpdl.slm.text.acmg import CODES
from vpdl.slm.text.evidence import EVIDENCE_TYPES, POLARITIES

__all__ = ["tokenize_cached", "TaskTensors", "UnitTensors", "assemble_documents", "assemble_units",
           "to_batch"]


def _texts_digest(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def tokenize_cached(texts: Sequence[str], backbone: Backbone, max_length: int,
                    cache_dir: Path | str | None, name: str) -> dict[str, np.ndarray]:
    texts = [t or "" for t in texts]
    key = hashlib.sha256(f"{tokenizer_fingerprint(backbone.tokenizer)}|{max_length}|"
                         f"{_texts_digest(texts)}".encode()).hexdigest()[:16]
    folder = Path(cache_dir) / f"{name}-{key}" if cache_dir else None
    if folder is not None and (folder / "meta.json").exists():
        meta = json.loads((folder / "meta.json").read_text())
        if meta.get("key") == key and meta.get("n") == len(texts):
            return {"input_ids": np.load(folder / "input_ids.npy"),
                    "attention_mask": np.load(folder / "attention_mask.npy"), "meta": meta}
    encoded = backbone.tokenizer(texts, truncation=True, padding="max_length", max_length=max_length,
                                 return_tensors="np")
    ids = encoded["input_ids"].astype(np.int32)
    mask = encoded["attention_mask"].astype(np.int8)
    lengths = [len(x) for x in backbone.tokenizer(texts, truncation=False)["input_ids"]] if texts else []
    meta = {"key": key, "n": len(texts), "max_length": max_length,
            "tokenizer": tokenizer_fingerprint(backbone.tokenizer),
            "truncated_fraction": float(np.mean([l > max_length for l in lengths])) if lengths else 0.0,
            "median_tokens": float(np.median(lengths)) if lengths else 0.0}
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "input_ids.npy", ids)
        np.save(folder / "attention_mask.npy", mask)
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"input_ids": ids, "attention_mask": mask, "meta": meta}


@dataclass
class TaskTensors:
    example_ids: np.ndarray
    split: np.ndarray
    genes: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    class_target: np.ndarray                       # [N, 5] probability targets (0 rows = none)
    class_index: np.ndarray                        # [N] hard index or -1
    binary: np.ndarray                             # [N] 1 / 0 / nan
    categorical: np.ndarray | None = None
    numeric: np.ndarray | None = None
    numeric_mask: np.ndarray | None = None
    dl_embedding: np.ndarray | None = None
    dl_mask: np.ndarray | None = None
    acmg_target: np.ndarray | None = None
    acmg_supervised: np.ndarray | None = None
    acmg_applicable: np.ndarray | None = None
    token_meta: dict[str, Any] = field(default_factory=dict)

    def rows(self, split: str | Sequence[str]) -> np.ndarray:
        wanted = [split] if isinstance(split, str) else list(split)
        return np.flatnonzero(np.isin(self.split, wanted))


@dataclass
class UnitTensors:
    example_ids: np.ndarray
    split: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    types: np.ndarray                              # [U, T]
    polarity: np.ndarray                           # [U] index or -1

    def rows(self, split: str | Sequence[str]) -> np.ndarray:
        wanted = [split] if isinstance(split, str) else list(split)
        return np.flatnonzero(np.isin(self.split, wanted))


def assemble_documents(examples: pd.DataFrame, backbone: Backbone, max_length: int,
                       cache_dir: Path | str | None = None,
                       structured: Mapping[str, np.ndarray] | None = None,
                       dl: tuple[np.ndarray, np.ndarray] | None = None,
                       acmg_applicable: np.ndarray | None = None) -> TaskTensors:
    tokens = tokenize_cached(examples["input_text"].tolist(), backbone, max_length, cache_dir, "documents")
    n = len(examples)
    target = np.zeros((n, len(CLASSES)), dtype=np.float32)
    index = class_index(examples.get("target_label", pd.Series([None] * n)).tolist())
    for i, (hard, soft) in enumerate(zip(index, examples.get("target_soft", pd.Series([None] * n)))):
        if hard >= 0:
            target[i, hard] = 1.0
        elif soft is not None and not (isinstance(soft, float) and np.isnan(soft)):
            target[i] = np.asarray(soft, dtype=np.float32)
    tensors = TaskTensors(
        example_ids=examples["example_id"].astype(str).to_numpy(),
        split=examples["split"].astype(str).to_numpy(), genes=examples["gene"].fillna("").astype(str).to_numpy(),
        input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"], class_target=target,
        class_index=index, binary=pd.to_numeric(examples.get("target_binary", pd.Series([np.nan] * n)),
                                                errors="coerce").to_numpy(dtype=float),
        token_meta=tokens["meta"])
    if structured is not None:
        tensors.categorical = structured["categorical"]
        tensors.numeric = structured["numeric"]
        tensors.numeric_mask = structured["numeric_mask"]
    if dl is not None:
        tensors.dl_embedding, tensors.dl_mask = dl
    if "target_codes" in examples:
        tensors.acmg_target = code_multi_hot(examples["target_codes"].tolist(), CODES)
        tensors.acmg_supervised = (tensors.acmg_target.sum(1) > 0).astype(np.float32)
    else:
        tensors.acmg_target = np.zeros((n, len(CODES)), dtype=np.float32)
        tensors.acmg_supervised = np.zeros(n, dtype=np.float32)
    tensors.acmg_applicable = (acmg_applicable if acmg_applicable is not None
                               else np.ones((n, len(CODES)), dtype=np.float32))
    return tensors


def assemble_units(units: pd.DataFrame, backbone: Backbone, max_length: int,
                   cache_dir: Path | str | None = None) -> UnitTensors:
    tokens = tokenize_cached(units["input_text"].tolist(), backbone, max_length, cache_dir, "units")
    types = np.zeros((len(units), len(EVIDENCE_TYPES)), dtype=np.float32)
    lookup = {name: i for i, name in enumerate(EVIDENCE_TYPES)}
    for r, values in enumerate(units["target_types"]):
        for value in as_list(values):
            if value in lookup:
                types[r, lookup[value]] = 1.0
    polarity_lookup = {name: i for i, name in enumerate(POLARITIES)}
    polarity = np.array([polarity_lookup.get(p, -1) for p in units["target_polarity"]], dtype=np.int64)
    return UnitTensors(units["example_id"].astype(str).to_numpy(), units["split"].astype(str).to_numpy(),
                       tokens["input_ids"], tokens["attention_mask"], types, polarity)


def to_batch(tensors: TaskTensors, indices: np.ndarray, device) -> dict[str, Any]:
    import torch
    batch = {"input_ids": torch.as_tensor(tensors.input_ids[indices], dtype=torch.long, device=device),
             "attention_mask": torch.as_tensor(tensors.attention_mask[indices], dtype=torch.long, device=device),
             "class_target": torch.as_tensor(tensors.class_target[indices], device=device),
             "acmg_target": torch.as_tensor(tensors.acmg_target[indices], device=device),
             "acmg_supervised": torch.as_tensor(tensors.acmg_supervised[indices], device=device),
             "acmg_applicable": torch.as_tensor(tensors.acmg_applicable[indices], device=device)}
    if tensors.categorical is not None:
        batch["categorical"] = torch.as_tensor(tensors.categorical[indices], dtype=torch.long, device=device)
        batch["numeric"] = torch.as_tensor(tensors.numeric[indices], device=device)
        batch["numeric_mask"] = torch.as_tensor(tensors.numeric_mask[indices], device=device)
    if tensors.dl_embedding is not None:
        batch["dl_embedding"] = torch.as_tensor(tensors.dl_embedding[indices], device=device)
        batch["dl_mask"] = torch.as_tensor(tensors.dl_mask[indices], device=device)
    return batch


def unit_batch(units: UnitTensors, indices: np.ndarray, device) -> dict[str, Any]:
    import torch
    return {"unit_input_ids": torch.as_tensor(units.input_ids[indices], dtype=torch.long, device=device),
            "unit_attention_mask": torch.as_tensor(units.attention_mask[indices], dtype=torch.long, device=device),
            "unit_types": torch.as_tensor(units.types[indices], device=device),
            "unit_polarity": torch.as_tensor(units.polarity[indices], device=device)}
