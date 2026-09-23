"""Versioned feature store: generate once, memory-map forever, never mismatch.

Layout::

    features/<family>/<model_tag>/<entry_key>/
        meta.json          identity + provenance (below)
        variant_ids.txt    row order, one variant id per line
        <block>.npy        one array per block, row-aligned to variant_ids.txt

    family     esm1b | esm2 | zeroshot | structure | genomic | ...
    model_tag  e.g. esm2_650m@P0 (backbone + pretraining arm)
    entry_key  hash of everything that changes the numbers: model version,
               sequence versions, context policy, layer, radius, dataset version

Every entry records ``variant_id`` order, ``model``, ``model_version``,
``sequence_version``, ``dtype``, ``shape`` (per block), ``dataset_version``,
``generation_date``, the generating config and the git commit. Two rules come
from paid-for bugs:

* the key includes the schema (landmine L5: a cache keyed only on gene and
  model silently served features built without the prior columns);
* a lookup that asks for a variant the entry does not hold raises — it never
  zero-fills (v1's row-count-only cache check could not see a reordered
  table and returned other variants' features).

Arrays are ``.npy`` so :func:`numpy.load` can ``mmap_mode="r"`` them: a DGX
process reads only the rows a fold needs, and many workers share one page cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["FEATURE_STORE_FORMAT", "entry_key", "FeatureEntry", "FeatureStore"]

FEATURE_STORE_FORMAT = "vpdl-feature-store-v1"


def entry_key(identity: Mapping[str, Any]) -> str:
    """16-hex hash of an identity mapping, order-independent."""
    payload = json.dumps(identity, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class FeatureEntry:
    path: Path
    meta: dict[str, Any]
    variant_ids: list[str]
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {v: i for i, v in enumerate(self.variant_ids)}

    @property
    def blocks(self) -> list[str]:
        return list(self.meta["shapes"])

    def array(self, block: str, mmap: bool = True) -> np.ndarray:
        if block not in self.meta["shapes"]:
            raise KeyError(f"{self.path}: no block {block!r}; has {self.blocks}")
        return np.load(self.path / f"{block}.npy", mmap_mode="r" if mmap else None)

    def rows(self, variant_ids: Sequence[str]) -> np.ndarray:
        missing = [v for v in variant_ids if v not in self._index]
        if missing:
            raise KeyError(
                f"{len(missing)} variant(s) not in feature entry {self.path.name} "
                f"(e.g. {missing[:3]}). Extract them; missing features are never zero-filled.")
        return np.array([self._index[v] for v in variant_ids], dtype=np.int64)

    def lookup(self, block: str, variant_ids: Sequence[str]) -> np.ndarray:
        """Rows for `variant_ids` in the order given, as float32."""
        return np.asarray(self.array(block)[self.rows(variant_ids)], dtype=np.float32)

    def blocks_for(self, variant_ids: Sequence[str]) -> dict[str, np.ndarray]:
        rows = self.rows(variant_ids)
        return {b: np.asarray(self.array(b)[rows], dtype=np.float32) for b in self.blocks}


class FeatureStore:
    def __init__(self, root: Path | str = "features") -> None:
        self.root = Path(root)

    def entry_dir(self, family: str, model_tag: str, key: str) -> Path:
        return self.root / family / model_tag / key

    def write(self, family: str, model_tag: str, identity: Mapping[str, Any],
              variant_ids: Sequence[str], blocks: Mapping[str, np.ndarray],
              dtype: str = "float16", extra: Mapping[str, Any] | None = None) -> FeatureEntry:
        """Write an entry atomically (to a temp dir, then rename)."""
        required = {"model", "model_version", "sequence_version", "dataset_version"}
        if missing := required - set(identity):
            raise ValueError(f"feature identity lacks {sorted(missing)}")
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("duplicate variant ids in a feature entry")
        for name, array in blocks.items():
            if len(array) != len(variant_ids):
                raise ValueError(f"block {name} has {len(array)} rows for "
                                 f"{len(variant_ids)} variant ids")
        key = entry_key(identity)
        final = self.entry_dir(family, model_tag, key)
        if final.exists():
            raise FileExistsError(f"{final} exists — identical identity already stored; "
                                  "load it instead of regenerating")
        staging = final.with_name(f".{key}.tmp-{os.getpid()}-{int(time.time())}")
        staging.mkdir(parents=True, exist_ok=False)
        shapes = {}
        for name, array in blocks.items():
            stored = np.asarray(array, dtype=dtype)
            np.save(staging / f"{name}.npy", stored)
            shapes[name] = list(stored.shape)
        (staging / "variant_ids.txt").write_text("\n".join(variant_ids) + "\n")
        from vpdl.provenance import _git_state
        meta = {"format": FEATURE_STORE_FORMAT, "family": family, "model_tag": model_tag,
                "key": key, "identity": dict(identity), "dtype": dtype, "shapes": shapes,
                "n_variants": len(variant_ids),
                "variant_ids_sha256": hashlib.sha256("\n".join(variant_ids).encode())
                .hexdigest(),
                "generation_date": datetime.now(timezone.utc).isoformat(),
                "git": _git_state(), **dict(extra or {})}
        (staging / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        if final.exists():                    # another writer finished first
            shutil.rmtree(staging, ignore_errors=True)
            raise FileExistsError(f"{final} exists — identical identity already stored; "
                                  "load it instead of regenerating")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        return self.load(family, model_tag, key)

    def load(self, family: str, model_tag: str, key: str) -> FeatureEntry:
        path = self.entry_dir(family, model_tag, key)
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("format") != FEATURE_STORE_FORMAT:
            raise ValueError(f"{path}: format {meta.get('format')!r}, expected "
                             f"{FEATURE_STORE_FORMAT!r}")
        ids = (path / "variant_ids.txt").read_text().splitlines()
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if digest != meta["variant_ids_sha256"]:
            raise ValueError(f"{path}: variant_ids.txt does not match meta.json (stale entry)")
        return FeatureEntry(path, meta, ids)

    def find(self, family: str, model_tag: str, identity: Mapping[str, Any]) -> FeatureEntry | None:
        key = entry_key(identity)
        path = self.entry_dir(family, model_tag, key)
        return self.load(family, model_tag, key) if (path / "meta.json").exists() else None

    def entries(self, family: str | None = None) -> list[dict[str, Any]]:
        pattern = f"{family}/*/*/meta.json" if family else "*/*/*/meta.json"
        return [json.loads(p.read_text()) for p in sorted(self.root.glob(pattern))
                if not p.parent.name.startswith(".")]          # skip staging dirs

    def resolve(self, spec: str) -> tuple[FeatureEntry, str]:
        """``family/model_tag/key:block`` -> (entry, block). Used by embedding blocks."""
        location, _, block = spec.partition(":")
        parts = location.split("/")
        if len(parts) != 3 or not block:
            raise ValueError(f"feature spec must be family/model_tag/key:block, got {spec!r}")
        return self.load(*parts), block
