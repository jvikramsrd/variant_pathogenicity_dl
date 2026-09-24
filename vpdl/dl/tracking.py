"""Experiment tracking for DGX runs — built on vpdl.provenance, not beside it.

Every DL run appends one record to ``runs/dl/registry.jsonl`` (append-only, one
JSON object per line) with:

    experiment_id    deterministic: same config + data + seed + commit => same id,
                     so a re-run is recognisable as a replicate
    run_id           unique per execution
    git_commit       and whether the code that ran was dirty (vpdl.provenance)
    dataset_version  sha256 of the table read
    feature_version  feature-store entries read
    model_version    backbone + checkpoint identity
    command          the command line that ran
    config, seed, hardware (vpdl.device), metrics, checkpoint, timings

The registry never replaces per-cell artefacts (results/predictions/summary);
it indexes them, so "which run produced this number" is one grep.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = ["experiment_id", "run_record", "append_registry", "REGISTRY"]

REGISTRY = Path("runs/dl/registry.jsonl")


def experiment_id(kind: str, config: Mapping[str, Any], dataset_version: str | None,
                  seed: int | None, commit: str | None) -> str:
    payload = json.dumps({"kind": kind, "config": config, "dataset": dataset_version,
                          "seed": seed, "commit": commit}, sort_keys=True, default=str)
    return f"{kind}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def run_record(kind: str, config: Mapping[str, Any], seed: int | None = None,
               dataset_path: Path | str | None = None, feature_version: str | None = None,
               model_version: Mapping[str, Any] | str | None = None,
               metrics: Mapping[str, Any] | None = None, checkpoint: str | None = None,
               artefacts: Mapping[str, str] | None = None, started: float | None = None,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from vpdl.device import detect
    from vpdl.provenance import _git_state, _library_versions, file_sha256

    git = _git_state()
    dataset_version = (file_sha256(dataset_path) if dataset_path and Path(dataset_path).exists()
                       else None)
    return {
        "experiment_id": experiment_id(kind, config, dataset_version, seed, git.get("commit")),
        "run_id": uuid.uuid4().hex[:12],
        "kind": kind,
        "git_commit": git.get("commit"), "git_dirty": git.get("dirty"),
        "git_dirty_paths": git.get("dirty_paths"),
        "dataset_path": Path(dataset_path).as_posix() if dataset_path else None,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "model_version": model_version,
        "config": dict(config), "seed": seed,
        # The process's command line (a pool worker inherits its parent's). Not part of
        # experiment_id: the same config typed two ways is still the same experiment.
        "command": list(sys.argv),
        "hardware": detect().as_dict(),
        "platform": {"machine": platform.machine(), "python": platform.python_version()},
        "versions": _library_versions(),
        "metrics": dict(metrics or {}), "checkpoint": checkpoint,
        "artefacts": dict(artefacts or {}),
        "started_utc": (datetime.fromtimestamp(started, timezone.utc).isoformat()
                        if started else None),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(time.time() - started, 1) if started else None,
        **dict(extra or {}),
    }


def append_registry(record: Mapping[str, Any], registry: Path | str = REGISTRY) -> Path:
    """Append one record. Locked on POSIX, because `vpdl-dl train --jobs N`
    has N processes finishing cells at once and a record (hardware block
    included) is longer than the size the OS guarantees to append atomically."""
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(registry, "a", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError:                         # Windows: single-process use
            handle.write(line)
            return registry
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    return registry
