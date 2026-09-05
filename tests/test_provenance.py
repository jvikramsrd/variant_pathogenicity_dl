"""Tests for run-identity recording and the comparability gate."""
from __future__ import annotations

import hashlib

import pytest

from src.provenance import (COMPARABILITY_KEYS, HASH_CHARS, REPLICATE_KEYS,
                            IncomparableRuns, assert_comparable,
                            feature_schema_hash, library_versions,
                            provenance_record, sha256_file,
                            split_definition_hash)


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "table.csv"
    payload = b"gene,position\nMLH1,1\n" * 5000        # spans the read chunk
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_dataset_hash_is_not_truncated(tmp_path):
    """It is quoted in the manuscript and checked against Get-FileHash."""
    p = tmp_path / "t.csv"
    p.write_bytes(b"x")
    assert len(sha256_file(p)) == 64


def test_feature_schema_hash_ignores_column_order():
    a = feature_schema_hash(["alphamissense", "af_plddt", "in_domain"])
    b = feature_schema_hash(["in_domain", "alphamissense", "af_plddt"])
    assert a == b
    assert len(a) == HASH_CHARS


def test_feature_schema_hash_distinguishes_a_dropped_column():
    full = ["alphamissense", "af_plddt", "in_domain"]
    assert feature_schema_hash(full) != feature_schema_hash(full[:-1])


def test_split_hash_distinguishes_equal_sized_different_splits():
    """The failure a count-based hash would miss."""
    a = {"MLH1": ["P40692:1:M>I", "P40692:2:A>V"]}
    b = {"MLH1": ["P40692:3:R>C", "P40692:4:K>E"]}
    assert len(next(iter(a.values()))) == len(next(iter(b.values())))
    assert split_definition_hash(a) != split_definition_hash(b)


def test_split_hash_ignores_member_order():
    a = {"MLH1": ["P40692:1:M>I", "P40692:2:A>V"]}
    b = {"MLH1": ["P40692:2:A>V", "P40692:1:M>I"]}
    assert split_definition_hash(a) == split_definition_hash(b)


def test_split_hash_distinguishes_which_gene_is_held_out():
    a = {"MLH1": ["P40692:1:M>I"], "MSH2": ["P43246:1:M>V"]}
    b = {"MSH2": ["P40692:1:M>I"], "MLH1": ["P43246:1:M>V"]}
    assert split_definition_hash(a) != split_definition_hash(b)


def test_library_versions_reports_missing_rather_than_raising():
    out = library_versions(["numpy", "a-package-that-does-not-exist"])
    assert out["a-package-that-does-not-exist"] == "not installed"
    assert out["numpy"] != "not installed"


def test_provenance_record_has_every_identity_field(tmp_path):
    p = tmp_path / "table.csv"
    p.write_bytes(b"gene\nMLH1\n")
    rec = provenance_record(p, ["alphamissense"], {"MLH1": ["P40692:1:M>I"]})
    for key in ("dataset_path", "dataset_sha256", "feature_schema",
                "n_features", "split_definition", "git", "libraries"):
        assert key in rec, key
    assert rec["n_features"] == 1
    assert set(rec["git"]) == {"commit", "dirty"}


def _rec(**over):
    base = {"dataset_sha256": "aa", "feature_schema": "bb",
            "split_definition": "cc"}
    base.update(over)
    return base


def test_assert_comparable_passes_on_matching_runs():
    assert_comparable({"cell_a": _rec(), "cell_b": _rec()})


def test_assert_comparable_allows_a_single_run():
    assert_comparable({"only": _rec(dataset_sha256="whatever")})


@pytest.mark.parametrize("key", COMPARABILITY_KEYS)
def test_assert_comparable_rejects_each_disagreement(key):
    with pytest.raises(IncomparableRuns) as exc:
        assert_comparable({"cell_a": _rec(), "cell_b": _rec(**{key: "different"})})
    assert key in str(exc.value)
    assert "cell_a" in str(exc.value) and "cell_b" in str(exc.value)


def test_missing_provenance_is_flagged_not_assumed_matching():
    """A pre-provenance artifact is the case that most needs catching."""
    with pytest.raises(IncomparableRuns) as exc:
        assert_comparable({"new_cell": _rec(), "old_cell": {}})
    assert "<missing>" in str(exc.value)


def test_error_names_every_run_in_each_group():
    with pytest.raises(IncomparableRuns) as exc:
        assert_comparable({"a": _rec(), "b": _rec(),
                           "c": _rec(dataset_sha256="other")})
    msg = str(exc.value)
    assert "a, b" in msg and "c" in msg


def test_ablation_arms_pool_despite_different_feature_schemas():
    """The gate must not forbid the comparison the ablation table is made of."""
    assert_comparable({"comparator": _rec(feature_schema="27cols"),
                       "ablate_structure": _rec(feature_schema="25cols")})


def test_replicate_keys_reject_a_schema_mismatch_between_seeds():
    """Three seeds that read different columns are not three seeds of one arm."""
    with pytest.raises(IncomparableRuns):
        assert_comparable({"seed42": _rec(feature_schema="27cols"),
                           "seed43": _rec(feature_schema="25cols")},
                          keys=REPLICATE_KEYS)


def test_feature_schema_is_not_a_default_comparability_key():
    assert "feature_schema" not in COMPARABILITY_KEYS
    assert "feature_schema" in REPLICATE_KEYS
