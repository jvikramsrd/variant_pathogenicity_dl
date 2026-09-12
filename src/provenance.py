"""Run identity: what table, what features, what splits, what code.

A results artifact that records only hyperparameters cannot be compared with
another such artifact, because the two may have been computed on different
dataset builds, different feature schemas or different splits and nothing in
either file would say so. That is not hypothetical here: the priors-only
baseline was withheld from the manuscript for two days on the *suspicion* that
it had been scored on a superseded table, and neither the summary JSON nor the
results CSV could settle the question. It had not been -- but only re-running it
proved that, at the cost of the paper's headline comparison in the meantime.

So every run records:

* ``dataset_sha256``     -- the bytes actually read
* ``feature_schema``     -- a hash of the sorted feature-column list
* ``split_definition``   -- a hash of the held-out key assignment
* ``git_commit``         -- code state, with a dirty flag
* ``libraries``          -- versions of the libraries that move numbers

and :func:`assert_comparable` refuses to aggregate records that disagree on the
dataset or the split. It deliberately does *not* require a matching feature
schema: an ablation arm differs from its comparator in exactly that field, and a
gate that forbade it would forbid the ablation table. Schema equality is required
only among replicates of one arm -- three seeds of one cell that read different
columns are not three seeds of one cell -- which is :data:`REPLICATE_KEYS`.

Hashes are truncated to 16 hex characters: long enough that a collision is not a
practical concern for a handful of runs, short enough to paste into a table.
"""
from __future__ import annotations

import hashlib
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

HASH_CHARS = 16
_TRACKED_LIBRARIES = ("torch", "transformers", "scikit-learn", "numpy", "pandas")


def _short(digest: "hashlib._Hash") -> str:
    return digest.hexdigest()[:HASH_CHARS]


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Full SHA-256 of a file, read in chunks.

    Not truncated: this one is quoted in the manuscript and checked against
    ``Get-FileHash`` on the build machine, so it must be the real digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


def feature_schema_hash(columns: Sequence[str]) -> str:
    """Hash of the feature set, order-independent.

    Sorted before hashing because column order is an artifact of how the frame
    was assembled, not part of the schema: two runs reading the same columns in
    a different order are comparable and must hash alike.
    """
    joined = "\n".join(sorted(str(c) for c in columns))
    return _short(hashlib.sha256(joined.encode("utf-8")))


def split_definition_hash(assignments: Mapping[str, Iterable[str]]) -> str:
    """Hash of a split: each split name against its sorted member keys.

    Takes the keys themselves rather than counts. Two splits of equal size
    holding different variants are different splits, and a count-based hash
    would call them identical -- precisely the failure this module exists to
    prevent.
    """
    parts = []
    for name in sorted(assignments):
        members = sorted(str(k) for k in assignments[name])
        parts.append(f"{name}\t{len(members)}\t" + ",".join(members))
    return _short(hashlib.sha256("\n".join(parts).encode("utf-8")))


def library_versions(names: Sequence[str] = _TRACKED_LIBRARIES) -> Dict[str, str]:
    """Installed versions of the libraries whose behaviour moves results."""
    out: Dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def git_state(repo: Optional[Path] = None) -> Dict[str, object]:
    """Commit SHA and whether the working tree was dirty when the run started.

    The dirty flag matters more than the SHA: a clean commit identifies the code
    exactly, while a dirty tree means the commit is a lower bound on what ran and
    the run is not reproducible from git alone.
    """
    cwd = str(repo or Path(__file__).resolve().parents[1])

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(("git", *args), cwd=cwd, capture_output=True,
                                 text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status.strip()),
    }


def provenance_record(dataset_path: Path,
                      feature_columns: Sequence[str],
                      split_assignments: Mapping[str, Iterable[str]],
                      extra: Optional[Mapping[str, object]] = None,
                      ) -> Dict[str, object]:
    """The full identity block written into a run summary."""
    record: Dict[str, object] = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(Path(dataset_path)),
        "feature_schema": feature_schema_hash(feature_columns),
        "n_features": len(feature_columns),
        "split_definition": split_definition_hash(split_assignments),
        "git": git_state(),
        "libraries": library_versions(),
    }
    if extra:
        record.update(dict(extra))
    return record


class IncomparableRuns(ValueError):
    """Raised when runs that must not be pooled are about to be pooled."""


#: Must match for any pooling at all -- same table, same held-out assignment.
COMPARABILITY_KEYS = ("dataset_sha256", "split_definition")
#: Additionally required among seed replicates of a single arm.
REPLICATE_KEYS = COMPARABILITY_KEYS + ("feature_schema",)
#: Additionally required to trust that identical dataset/split/schema were
#: also produced by identical code. Two runs can agree on all of
#: COMPARABILITY_KEYS and REPLICATE_KEYS while one predates a correctness
#: fix and one postdates it -- exactly the seeding-bug situation
#: MISSING_EVIDENCE.md item 12 had to resolve by manually reading commit
#: messages across the 28-summary set, because nothing in assert_comparable
#: itself would have caught a pre-fix/post-fix mix. Not folded into
#: REPLICATE_KEYS by default: a seed re-run one commit later to pick up an
#: unrelated fix is still the same arm, and requiring an identical commit
#: would make that ordinary case unaggregatable. Pass
#: ``keys=REPLICATE_KEYS + CODE_KEYS`` for the stronger guarantee.
CODE_KEYS = ("git",)


def _hashable(value: object) -> object:
    """Coerce a provenance value into something usable as a grouping key.

    Every COMPARABILITY_KEYS/REPLICATE_KEYS value is a hashable string, but
    ``git`` (see :data:`CODE_KEYS`) is a nested ``{"commit": ..., "dirty":
    ...}`` dict, which ``dict.setdefault`` cannot use as a key directly.
    Sorted items rather than ``str(dict)``: item order is not part of the
    value being compared.
    """
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def assert_comparable(records: Mapping[str, Mapping[str, object]],
                      keys: Sequence[str] = COMPARABILITY_KEYS) -> None:
    """Refuse to aggregate runs whose dataset or split assignment disagree.

    ``records`` maps a run label (a cell slug) to its provenance block. Runs
    missing a key are reported rather than skipped: an artifact written before
    this module existed is exactly the case that needs flagging, and silently
    treating "unknown" as "matching" would reintroduce the failure.

    Pass ``keys=REPLICATE_KEYS`` when the runs are meant to be seeds of one arm,
    which additionally requires an identical feature schema. Pass
    ``keys=REPLICATE_KEYS + CODE_KEYS`` to additionally require an identical
    git commit (see :data:`CODE_KEYS`).
    """
    if len(records) < 2:
        return
    problems = []
    for key in keys:
        seen: Dict[object, list] = {}
        for label, rec in records.items():
            seen.setdefault(_hashable(rec.get(key, "<missing>")), []).append(label)
        if len(seen) > 1:
            groups = "; ".join(
                f"{value!r}: {', '.join(sorted(labels))}"
                for value, labels in sorted(seen.items(), key=lambda kv: str(kv[0]))
            )
            problems.append(f"{key} differs across runs -- {groups}")
    if problems:
        raise IncomparableRuns(
            "These runs cannot be aggregated:\n  " + "\n  ".join(problems)
            + "\nAggregating them would compare arms across different data or "
              "splits. Re-run the odd one out, or aggregate the matching subset "
              "explicitly."
        )


def assert_no_dirty_code(records: Mapping[str, Mapping[str, object]]) -> None:
    """Name runs whose working tree was dirty when they started.

    A dirty run's recorded ``git.commit`` is only a lower bound on what code
    actually executed -- the every-summary-in-the-current-grid case
    docs/RUNLOG.md 2026-09-07 already flags as "not recoverable from the
    artifacts." This is opt-in and separate from :func:`assert_comparable`
    (not folded into COMPARABILITY_KEYS/REPLICATE_KEYS/CODE_KEYS) because it
    is a statement about one run's own trustworthiness, not about whether two
    runs agree with each other -- a single dirty run should be flagged even
    when nothing else is being pooled against it. Records with no ``git``
    block are skipped, not flagged: that is :func:`assert_comparable`'s
    ``"<missing>"`` case via :data:`CODE_KEYS`, not this function's job.
    """
    dirty = [label for label, rec in records.items()
            if isinstance(rec.get("git"), dict) and rec["git"].get("dirty")]
    if dirty:
        raise IncomparableRuns(
            "These runs had an uncommitted working tree at run time, so "
            "their recorded git commit is only a lower bound on what code "
            "executed: " + ", ".join(sorted(dirty)))
