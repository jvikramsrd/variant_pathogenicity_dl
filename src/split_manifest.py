"""Reproducible gene-level split manifests.

Complements, and does not replace, ``src.mmr_dataset.write_leave_one_gene_out_manifest``
(which records the leave-one-gene-out rotation by gene *symbol*, with no
dataset checksum or seed). :class:`SplitManifest` adds the three guarantees
the user-facing reproducibility requirement needs on top of that: canonical
(alias-proof) gene IDs via ``src.gene_aliases``, a dataset fingerprint via
``src.provenance.sha256_file``, and a versioned on-disk format that fails
loudly on a future incompatible schema rather than silently misparsing it --
the same checkpoint-version-rejection principle applied to splits instead of
model weights.

A manifest written by this module is the single artifact a second machine
needs to reproduce *which rows* a run trained, validated, tested and held
out on: given the same dataset file, :func:`verify_against_dataset` proves
byte-for-byte identity before anything is trained against it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from .gene_aliases import assert_disjoint_gene_sets, resolve_gene_ids
from .provenance import sha256_file

#: Bumped whenever the on-disk JSON schema changes incompatibly. A manifest
#: written by a future version with a different schema must fail to load
#: here rather than be silently misinterpreted as this version's shape.
MANIFEST_VERSION = 1


class ManifestVersionError(ValueError):
    """Raised when a loaded manifest's version does not match this module's."""


class DatasetMismatchError(ValueError):
    """Raised when a dataset file's checksum does not match a manifest's."""


@dataclass(frozen=True)
class SplitManifest:
    """A reproducible record of which genes went into which partition.

    All four gene tuples are stored as canonical UniProt accessions (see
    ``src.gene_aliases``); construct via :func:`build_split_manifest` rather
    than the constructor directly if your inputs may be raw HGNC symbols.
    """

    train_genes: Tuple[str, ...]
    val_genes: Tuple[str, ...]
    test_genes: Tuple[str, ...]
    holdout_genes: Tuple[str, ...]
    seed: int
    dataset_sha256: str
    dataset_path: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    manifest_version: int = MANIFEST_VERSION

    def validate(self) -> None:
        """Raise if any gene is unresolvable or appears in more than one partition.

        Resolution itself (via ``assert_disjoint_gene_sets`` ->
        ``resolve_gene_id``) already raises ``UnknownGeneError`` for any
        entry that is not a known symbol/accession, so a manifest that
        passes this call is guaranteed to name only real, disjoint genes.
        """
        assert_disjoint_gene_sets(self.train_genes, self.val_genes,
                                  self.test_genes, self.holdout_genes)

    def to_json(self, path: Path) -> Path:
        """Write this manifest as indented JSON. Does not create parent dirs
        silently for arbitrary paths outside the run's own output tree --
        the caller is expected to have already created its output directory,
        matching this project's convention of keeping derived artifacts out
        of raw-data locations."""
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path

    @classmethod
    def from_json(cls, path: Path) -> "SplitManifest":
        payload = json.loads(Path(path).read_text())
        version = payload.get("manifest_version")
        if version != MANIFEST_VERSION:
            raise ManifestVersionError(
                f"{path}: manifest_version {version!r} != expected "
                f"{MANIFEST_VERSION}. Refusing to load a split manifest "
                "written by an incompatible format rather than guessing "
                "at its shape.")
        return cls(
            train_genes=tuple(payload["train_genes"]),
            val_genes=tuple(payload["val_genes"]),
            test_genes=tuple(payload["test_genes"]),
            holdout_genes=tuple(payload["holdout_genes"]),
            seed=int(payload["seed"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            dataset_path=str(payload["dataset_path"]),
            created_at_utc=str(payload["created_at_utc"]),
            manifest_version=version,
        )


def build_split_manifest(*, train_genes=(), val_genes=(), test_genes=(),
                         holdout_genes=(), seed: int,
                         dataset_path: Path) -> SplitManifest:
    """Construct and validate a :class:`SplitManifest` from raw gene names.

    Accepts any mix of HGNC symbols and UniProt accessions in each partition
    (resolved via ``src.gene_aliases.resolve_gene_ids``) and computes the
    dataset fingerprint itself, so a caller cannot forget to hash the file
    the split actually describes.
    """
    manifest = SplitManifest(
        train_genes=resolve_gene_ids(train_genes),
        val_genes=resolve_gene_ids(val_genes),
        test_genes=resolve_gene_ids(test_genes),
        holdout_genes=resolve_gene_ids(holdout_genes),
        seed=seed,
        dataset_sha256=sha256_file(Path(dataset_path)),
        dataset_path=str(dataset_path),
    )
    manifest.validate()
    return manifest


def verify_against_dataset(manifest: SplitManifest, dataset_path: Path) -> None:
    """Raise :class:`DatasetMismatchError` if *dataset_path* is not the exact
    file this manifest's split was computed against.

    This is the reproducibility guarantee itself: a manifest names genes,
    not row indices, so replaying a split correctly requires re-deriving
    row membership from the dataset -- which is only valid if the dataset
    is byte-for-byte the one the manifest was built from. A different build
    of "the same" table (a newer ClinVar snapshot, a re-run gnomAD join)
    can silently assign different rows to the same gene names.
    """
    actual = sha256_file(Path(dataset_path))
    if actual != manifest.dataset_sha256:
        raise DatasetMismatchError(
            f"{dataset_path}: sha256 {actual} != manifest's recorded "
            f"{manifest.dataset_sha256}. This is not the dataset this split "
            "was computed against; row membership for the named genes is "
            "not guaranteed to match. Re-derive the split from this file, "
            "or locate the original dataset build.")


__all__ = [
    "MANIFEST_VERSION", "SplitManifest", "ManifestVersionError",
    "DatasetMismatchError", "build_split_manifest", "verify_against_dataset",
]
