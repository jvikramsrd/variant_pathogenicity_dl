"""Run identity, artefact checksums, and the gate that blocks bad pooling.

Two runs that read different tables are indistinguishable from their metrics
alone. In September 2026 a 28-cell result set was pooled from 16 cells built on
one dataset and 12 on another; the metric deltas reached 0.47 MCC at identical
seeds and read as model variance until the hashes were checked. Everything here
exists to make that failure loud.

Satisfies regression landmines L10 (manifest describes the file on disk) and
L13 (incomparable runs are not pooled).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "COMPARABILITY_KEYS",
    "REPLICATE_KEYS",
    "file_sha256",
    "schema_hash",
    "split_hash",
    "provenance_record",
    "write_manifest",
    "verify_manifest",
    "assert_comparable",
]

# Required to pool any two runs at all: they must have read the same table and
# partitioned it the same way.
COMPARABILITY_KEYS: tuple[str, ...] = ("dataset_sha256", "split_hash")

# Additionally required among seeds of ONE arm. Feature schema is deliberately
# absent from COMPARABILITY_KEYS: an ablation arm differs from its comparator in
# exactly that field, and a gate forbidding it would forbid the ablation table.
REPLICATE_KEYS: tuple[str, ...] = COMPARABILITY_KEYS + ("feature_schema",)

_CHUNK = 1 << 20


def file_sha256(path: Path | str) -> str:
    """SHA-256 of a file's bytes, read in chunks so large tables are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def schema_hash(columns: Iterable[str]) -> str:
    """Order-independent hash of a feature schema.

    Order-independent because column order is not a modelling decision; two runs
    reading the same columns in a different order are the same experiment.
    """
    payload = "\n".join(sorted(set(columns)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def split_hash(held_out_keys: Iterable[str]) -> str:
    """Hash of the held-out variant KEYS, not their count.

    Two splits of equal size holding different variants are different splits;
    recording only `n_holdout` cannot tell them apart.
    """
    payload = "\n".join(sorted(str(k) for k in held_out_keys))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _git_state(repo: Path | None = None) -> dict[str, Any]:
    """Commit, and whether the CODE that ran differs from it — and where.

    "Dirty" means the code that ran is not the committed code: a modified
    tracked file anywhere, or an untracked file under ``vpdl/``. Untracked
    outputs (``runs/``, ``docs/v2/HARDWARE.md``) do not count — v1 counted
    them, so every one of its 28 summaries said ``dirty: true`` and the field
    carried no information. The paths are recorded too, because v1's other
    gap was that *what* was uncommitted at run time was never on the record.
    """
    root = str(repo) if repo else None

    def git(*args: str) -> str:
        # NOT stripped: porcelain lines start with a meaningful space (" M path"),
        # and stripping the whole output ate it on the first line only, so the
        # first recorded path lost its first character.
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL,
        )

    try:
        commit = git("rev-parse", "HEAD").strip()
        changed = git("status", "--porcelain", "--untracked-files=no")
        untracked_code = git("status", "--porcelain", "--untracked-files=all",
                             "--", "vpdl")
        paths = sorted({line[3:] for line in (changed + "\n" + untracked_code)
                        .splitlines() if line.strip()})
        return {"commit": commit, "dirty": bool(paths), "dirty_paths": paths[:25]}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Absent git is recorded as absent, never silently filled in.
        return {"commit": None, "dirty": None, "dirty_paths": None}


# Distribution names, read from installed metadata rather than by importing:
# the import name differs from the distribution name (scikit-learn is
# `sklearn`, which an earlier version got wrong, recording null for a package
# that was installed), and importing torch just to read a version costs seconds.
_TRACKED_DISTRIBUTIONS = ("numpy", "pandas", "scikit-learn", "torch",
                          "xgboost", "lightgbm")


def _library_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in _TRACKED_DISTRIBUTIONS:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def provenance_record(
    dataset_path: Path | str,
    feature_columns: Iterable[str],
    held_out_keys: Iterable[str],
    sources: Sequence[str],
    repo: Path | None = None,
) -> dict[str, Any]:
    """The block every run summary must carry.

    `sources` is v2-specific and load-bearing for the paper: a run trained on
    ClinVar alone and a run trained on everything are not the same experiment,
    and the artefact has to say which it was.
    """
    return {
        "dataset_sha256": file_sha256(dataset_path),
        "feature_schema": schema_hash(feature_columns),
        "split_hash": split_hash(held_out_keys),
        "sources": sorted(sources),
        "git": _git_state(repo),
        "versions": _library_versions(),
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
        },
    }


@dataclass
class Manifest:
    artefacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def write_manifest(
    path: Path | str,
    artefacts: Sequence[Path | str],
    meta: Mapping[str, Any] | None = None,
) -> Manifest:
    """Record each artefact's checksum and size, measured from disk right now.

    Call this AFTER the final write of every file it names. v1's two-phase build
    wrote the manifest, then rewrote the CSV, so the recorded digest described a
    file that no longer existed: the manifest claimed 37.8 MB where 48.4 MB sat
    on disk, and flagged gnomAD disabled on a table carrying gnomAD columns.
    """
    path = Path(path)
    manifest = Manifest(meta=dict(meta or {}))
    for artefact in artefacts:
        artefact = Path(artefact)
        manifest.artefacts[artefact.name] = {
            "path": artefact.as_posix(),        # "/" on every OS: verifiable on the other one
            "sha256": file_sha256(artefact),
            "bytes": artefact.stat().st_size,
        }
    path.write_text(json.dumps(
        {"artefacts": manifest.artefacts, "meta": manifest.meta}, indent=2
    ))
    return manifest


def verify_manifest(path: Path | str) -> None:
    """Re-measure every artefact and raise if the manifest has gone stale."""
    path = Path(path)
    recorded = json.loads(path.read_text())
    problems: list[str] = []

    for name, entry in recorded.get("artefacts", {}).items():
        artefact = Path(entry["path"])
        if not artefact.exists():
            problems.append(f"{name}: recorded but missing from disk")
            continue
        actual = file_sha256(artefact)
        if actual != entry["sha256"]:
            problems.append(
                f"{name}: stale checksum — manifest says {entry['sha256'][:12]}…, "
                f"disk says {actual[:12]}…"
            )

    if problems:
        raise ValueError(
            "Manifest does not describe the files on disk:\n  "
            + "\n  ".join(problems)
        )


def assert_comparable(
    runs: Sequence[Mapping[str, Any]],
    as_replicates: bool = False,
) -> None:
    """Refuse to pool runs that are not the same experiment.

    `as_replicates=True` additionally requires an identical feature schema, which
    is the correct gate among seeds of one arm and the wrong one across arms.
    """
    if len(runs) < 2:
        return

    required = REPLICATE_KEYS if as_replicates else COMPARABILITY_KEYS

    for index, run in enumerate(runs):
        absent = [key for key in required if run.get(key) is None]
        if absent:
            raise ValueError(
                f"Run {index} is missing provenance: {', '.join(absent)}. "
                "A run without a provenance block is reported as unknown, never "
                "treated as matching — that is precisely the case needing a flag."
            )

    for key in required:
        distinct = {run[key] for run in runs}
        if len(distinct) > 1:
            raise ValueError(
                f"Runs disagree on {key}: {sorted(distinct)}. "
                "These are different experiments and must not be pooled."
            )
