"""Synthetic-data tests for src.gene_aliases and src.split_manifest.

docs/HOLDOUT_PROTOCOL.md documents this file as covering both modules with
in-memory data only and no execution against a real dataset. These tests use
a tiny temp file standing in for a dataset CSV (so sha256_file has real bytes
to hash) and MMR_UNIPROT's real gene symbols/accessions (so resolve_gene_id
exercises the actual alias table rather than a fake one) -- neither reads,
downloads, or processes any real dataset.
"""
from __future__ import annotations

import json

import pytest

from src.gene_aliases import (CANONICAL_GENE_IDS, UnknownGeneError,
                              assert_disjoint_gene_sets, resolve_gene_id,
                              resolve_gene_ids)
from src.split_manifest import (MANIFEST_VERSION, DatasetMismatchError,
                                ManifestVersionError, SplitManifest,
                                build_split_manifest, verify_against_dataset)

# Two real MMR genes, so resolve_gene_id exercises the actual alias table.
_SYMBOL_A, _ACCESSION_A = next(iter(CANONICAL_GENE_IDS.items()))
_OTHER = [(s, a) for s, a in CANONICAL_GENE_IDS.items() if s != _SYMBOL_A]
_SYMBOL_B, _ACCESSION_B = _OTHER[0]


@pytest.fixture()
def dataset_csv(tmp_path):
    """A small, deterministic stand-in for a real dataset table."""
    path = tmp_path / "fake_dataset.csv"
    path.write_text("uniprot_id,position,wt_aa,mut_aa,label\nP1,1,A,G,1\n")
    return path


# --------------------------------------------------------------------------- #
# src.gene_aliases
# --------------------------------------------------------------------------- #
def test_resolve_gene_id_accepts_symbol_case_insensitively():
    assert resolve_gene_id(_SYMBOL_A.lower()) == _ACCESSION_A
    assert resolve_gene_id(_SYMBOL_A.upper()) == _ACCESSION_A


def test_resolve_gene_id_accepts_canonical_accession_directly():
    assert resolve_gene_id(_ACCESSION_A) == _ACCESSION_A


def test_resolve_gene_id_rejects_unknown_name():
    with pytest.raises(UnknownGeneError):
        resolve_gene_id("NOT_A_REAL_GENE")


def test_resolve_gene_id_rejects_empty_or_non_string():
    with pytest.raises(UnknownGeneError):
        resolve_gene_id("")
    with pytest.raises(UnknownGeneError):
        resolve_gene_id(None)  # type: ignore[arg-type]


def test_resolve_gene_ids_preserves_order():
    assert resolve_gene_ids([_SYMBOL_B, _SYMBOL_A]) == (_ACCESSION_B, _ACCESSION_A)


def test_assert_disjoint_gene_sets_passes_when_no_overlap():
    assert_disjoint_gene_sets(train_genes=[_SYMBOL_A], holdout_genes=[_SYMBOL_B])


def test_assert_disjoint_gene_sets_catches_a_same_spelling_collision():
    with pytest.raises(ValueError):
        assert_disjoint_gene_sets(train_genes=[_SYMBOL_A], holdout_genes=[_SYMBOL_A])


def test_assert_disjoint_gene_sets_catches_a_cross_spelling_collision():
    """The exact case a raw string-equality check would miss: the same gene
    spelled as a symbol in one partition and its accession in another."""
    with pytest.raises(ValueError):
        assert_disjoint_gene_sets(train_genes=[_SYMBOL_A],
                                  holdout_genes=[_ACCESSION_A])


def test_assert_disjoint_gene_sets_allows_all_empty_partitions():
    assert_disjoint_gene_sets()


# --------------------------------------------------------------------------- #
# src.split_manifest
# --------------------------------------------------------------------------- #
def test_build_split_manifest_resolves_symbols_to_canonical_accessions(dataset_csv):
    manifest = build_split_manifest(
        train_genes=[_SYMBOL_A], holdout_genes=[_SYMBOL_B],
        seed=42, dataset_path=dataset_csv)
    assert manifest.train_genes == (_ACCESSION_A,)
    assert manifest.holdout_genes == (_ACCESSION_B,)
    assert manifest.seed == 42
    assert manifest.manifest_version == MANIFEST_VERSION


def test_build_split_manifest_rejects_a_gene_in_two_partitions(dataset_csv):
    with pytest.raises(ValueError):
        build_split_manifest(
            train_genes=[_SYMBOL_A], holdout_genes=[_SYMBOL_A],
            seed=42, dataset_path=dataset_csv)


def test_build_split_manifest_rejects_an_unknown_gene(dataset_csv):
    with pytest.raises(UnknownGeneError):
        build_split_manifest(
            train_genes=["NOT_A_REAL_GENE"], holdout_genes=[_SYMBOL_B],
            seed=42, dataset_path=dataset_csv)


def test_manifest_round_trips_through_json(tmp_path, dataset_csv):
    manifest = build_split_manifest(
        train_genes=[_SYMBOL_A], holdout_genes=[_SYMBOL_B],
        seed=7, dataset_path=dataset_csv)
    out = manifest.to_json(tmp_path / "split_manifest.json")
    loaded = SplitManifest.from_json(out)
    assert loaded == manifest


def test_from_json_rejects_an_incompatible_manifest_version(tmp_path):
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps({
        "train_genes": [], "val_genes": [], "test_genes": [],
        "holdout_genes": [], "seed": 0, "dataset_sha256": "x",
        "dataset_path": "x", "created_at_utc": "x",
        "manifest_version": MANIFEST_VERSION + 1,
    }))
    with pytest.raises(ManifestVersionError):
        SplitManifest.from_json(path)


def test_verify_against_dataset_passes_for_the_exact_file(dataset_csv):
    manifest = build_split_manifest(
        train_genes=[_SYMBOL_A], holdout_genes=[_SYMBOL_B],
        seed=1, dataset_path=dataset_csv)
    verify_against_dataset(manifest, dataset_csv)  # must not raise


def test_verify_against_dataset_fails_for_a_different_file(dataset_csv, tmp_path):
    manifest = build_split_manifest(
        train_genes=[_SYMBOL_A], holdout_genes=[_SYMBOL_B],
        seed=1, dataset_path=dataset_csv)
    other = tmp_path / "different_dataset.csv"
    other.write_text("uniprot_id,position,wt_aa,mut_aa,label\nP2,2,C,T,0\n")
    with pytest.raises(DatasetMismatchError):
        verify_against_dataset(manifest, other)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
