"""Composable, dependency-light checks for dataset merges and preprocessing.

Pure functions over caller-supplied pandas DataFrames/Series -- nothing here
reads a file, calls a network API, or executes the real build pipeline. They
exist to be called *by* a merge/preprocessing step (in ``src/extended_builder.py``,
``src/gnomad.py``, ``src/mavedb.py``, etc.) so a defect surfaces as a named,
actionable exception at the point it happens, rather than as a silently wrong
row count discovered later.

Every check below is motivated by a real defect class already found in this
codebase's history, cited by file:line where verified in this checkout:

* None of the 11 ``.merge()`` calls in ``src/extended_builder.py`` (lines
  275, 496-497, 699-752, 823, 832) pass ``validate=``, unlike
  ``src/mmr_dataset.py:342`` and ``src/gnomad.py:436``, which already do.
  ``validate_merge_cardinality`` gives a pre-merge diagnostic in the same
  spirit, without changing the merges themselves (a change to production
  merge semantics was deliberately deferred until there is a working
  environment to verify against -- see docs/PIPELINE_MAP.md section 3).
* The zero-shot score join (``src/extended_builder.py:756-762``) already
  fails loudly with ``logger.error`` when it matches zero rows -- the one
  merge in that file with any runtime cardinality check. ``check_unmatched_ids``
  generalises that specific, already-proven pattern into a reusable check.
* ``manifest.json``'s ``stats.master_label_counts`` field silently described
  a stale, pre-mutation row count for as long as
  ``src/extended_builder.py::refresh_manifest`` shallow-merged caller updates
  instead of recomputing from disk (root-caused this session; now fixed).
  ``check_no_unexpected_row_loss`` generalises "a transform silently changed
  which rows/keys are present" into a reusable pre/post check.
* Dataset builds join labels from three sources with different conventions
  (ClinVar star ratings, ProteinGym clinical, DMS bins); ``_resolve_binary_evidence``
  (``src/extended_builder.py:260``) already excludes same-key contradictory
  labels rather than guessing a resolution. ``check_no_duplicate_keys``
  generalises "does this key column actually identify one row" as a
  standalone, callable assertion for any table, not only the ones that
  already route through ``_resolve_binary_evidence``.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


class SchemaValidationError(ValueError):
    """A DataFrame's columns or dtypes don't match what a stage expects."""


class MergeCardinalityError(ValueError):
    """A merge's key columns don't satisfy the declared cardinality."""


class RowLossError(ValueError):
    """A transform dropped or added keys beyond the declared tolerance."""


class DuplicateKeyError(ValueError):
    """A supposed-unique key column has duplicate values."""


def validate_schema(df: pd.DataFrame, expected_columns: Sequence[str],
                    dtypes: Optional[Mapping[str, object]] = None) -> None:
    """Raise :class:`SchemaValidationError` naming any missing or
    mismatched-dtype column.

    Extra columns beyond ``expected_columns`` are not an error: a stage that
    only reads a subset of a wider table (e.g. ``master[key + [...]]`` calls
    throughout ``src/extended_builder.py``) must not be forced to enumerate
    every column that happens to also be present.
    """
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing expected column(s): {missing}. "
            f"Columns present: {list(df.columns)}")
    if dtypes:
        problems = []
        for col, expected in dtypes.items():
            if col not in df.columns:
                continue  # already reported above if it's in expected_columns
            actual = df[col].dtype
            if actual != np.dtype(expected) if isinstance(expected, (str, type)) \
                    else actual != expected:
                problems.append(f"{col}: expected {expected}, got {actual}")
        if problems:
            raise SchemaValidationError(
                "Dtype mismatch: " + "; ".join(problems))


def validate_merge_cardinality(left: pd.DataFrame, right: pd.DataFrame,
                               on: Sequence[str],
                               expected: str = "one_to_one") -> None:
    """Pre-merge diagnostic for the cardinality a caller is about to assume.

    ``pd.merge(..., validate=...)`` already enforces this, but only *after*
    pandas has done the join and raises a ``MergeError`` whose message names
    neither side's row/key counts. This runs first and reports both sides'
    duplicate-key counts up front, so a caller sees *why* a merge would fail
    rather than that it did. Does not perform the merge itself.

    ``expected`` must be one of ``"one_to_one"``, ``"one_to_many"``
    (left keys may repeat on the right), or ``"many_to_one"`` (right keys
    may repeat... i.e. left keys may repeat, right must not).
    """
    on = list(on)
    if expected not in ("one_to_one", "one_to_many", "many_to_one"):
        raise ValueError(f"Unknown expected cardinality: {expected!r}")

    def _dup_count(df: pd.DataFrame) -> int:
        return int(df.duplicated(subset=on).sum())

    left_dups = _dup_count(left)
    right_dups = _dup_count(right)
    left_may_repeat = expected in ("one_to_many",)
    right_may_repeat = expected in ("many_to_one",)
    problems = []
    if left_dups and not left_may_repeat:
        problems.append(
            f"left has {left_dups} duplicate key row(s) on {on}, but "
            f"expected={expected!r} requires unique left keys")
    if right_dups and not right_may_repeat:
        problems.append(
            f"right has {right_dups} duplicate key row(s) on {on}, but "
            f"expected={expected!r} requires unique right keys")
    if problems:
        raise MergeCardinalityError(
            f"Merge on {on} would violate declared cardinality {expected!r}: "
            + "; ".join(problems))


def check_no_unexpected_row_loss(before_df: pd.DataFrame, after_df: pd.DataFrame,
                                 key_cols: Sequence[str], context: str = "",
                                 tolerance: int = 0) -> None:
    """Raise :class:`RowLossError` if keys were dropped or added beyond
    ``tolerance``, naming exactly which ones.

    Motivated by the manifest-staleness defect this session found: a
    transform (the PMS2 homology gate, ``src/mmr_dataset.py:263-275``)
    changed which rows carry a label, and nothing checked or recorded that
    change at the point it happened -- the only record of it
    (``stats.master_label_counts``) went stale until a caller explicitly
    recomputed it. ``tolerance`` exists because some drops are expected and
    documented (e.g. the homology gate is *supposed* to null 21 rows) --
    pass the documented count rather than 0 to assert the *expected* size of
    a known, intentional change instead of merely disabling the check.
    """
    key_cols = list(key_cols)
    before_keys = set(map(tuple, before_df[key_cols].itertuples(index=False, name=None)))
    after_keys = set(map(tuple, after_df[key_cols].itertuples(index=False, name=None)))
    dropped = before_keys - after_keys
    added = after_keys - before_keys
    changed = len(dropped) + len(added)
    if changed > tolerance:
        prefix = f"{context}: " if context else ""
        sample_dropped = sorted(dropped)[:20]
        sample_added = sorted(added)[:20]
        raise RowLossError(
            f"{prefix}{len(dropped)} key(s) dropped, {len(added)} key(s) added "
            f"(tolerance={tolerance}). Dropped sample: {sample_dropped}. "
            f"Added sample: {sample_added}.")


def check_no_duplicate_keys(df: pd.DataFrame, key_cols: Sequence[str]) -> None:
    """Raise :class:`DuplicateKeyError` listing duplicate key values
    (first 20) if ``key_cols`` does not uniquely identify each row.

    Generalises the invariant ``_resolve_binary_evidence``
    (``src/extended_builder.py:260``) already enforces for label sources
    specifically -- "a key with contradictory evidence is excluded, never
    silently resolved by row order" -- into a plain uniqueness assertion
    callable against any table before it is trusted as one-row-per-key.
    """
    key_cols = list(key_cols)
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        dup_keys = (df.loc[dup_mask, key_cols]
                   .drop_duplicates()
                   .head(20)
                   .itertuples(index=False, name=None))
        raise DuplicateKeyError(
            f"{int(dup_mask.sum())} row(s) share a duplicate key on "
            f"{key_cols} (showing up to 20 distinct keys): {list(dup_keys)}")


def check_unmatched_ids(left_ids: Iterable[object], right_ids: Iterable[object],
                        context: str = "") -> Dict[str, object]:
    """Report (never raises) IDs present on only one side of a join.

    Returns ``{"context", "left_only_count", "right_only_count",
    "left_only_sample", "right_only_sample"}``. Does not raise, unlike the
    other checks here: this codebase has real, by-design one-sided cases
    (VUS rows with no clinical label deliberately have no match in a
    label-only table) that a hard failure would wrongly block. Generalises
    the zero-match guard already proven at
    ``src/extended_builder.py:756-762`` (the zero-shot score join, which
    logs an error only when *every* row is unmatched) into a report a
    caller can log unconditionally or assert against a known tolerance.
    """
    left_set = set(left_ids)
    right_set = set(right_ids)
    left_only = left_set - right_set
    right_only = right_set - left_set
    return {
        "context": context,
        "left_only_count": len(left_only),
        "right_only_count": len(right_only),
        "left_only_sample": sorted(map(str, left_only))[:20],
        "right_only_sample": sorted(map(str, right_only))[:20],
    }


def check_missing_value_policy(df: pd.DataFrame, column: str,
                               max_missing_frac: Optional[float] = None) -> float:
    """Return the missing-value fraction for ``column``; raise if it exceeds
    ``max_missing_frac``.

    Motivated by this codebase's own review finding that most ``zs_*``
    (published-model score) columns in the master table are ~99% missing by
    design (documented inline at ``scripts/run_mmr_transfer.py:249``, "most
    zs_* columns are ~99% missing"), which makes a bare ``.isna().sum()``
    uninformative without a documented expected threshold to compare
    against -- this makes that threshold an explicit, checkable argument
    instead of a comment.
    """
    if column not in df.columns:
        raise SchemaValidationError(f"Column {column!r} not present.")
    if len(df) == 0:
        return 0.0
    frac = float(df[column].isna().mean())
    if max_missing_frac is not None and frac > max_missing_frac:
        raise ValueError(
            f"Column {column!r} is {frac:.1%} missing, exceeding the "
            f"declared maximum of {max_missing_frac:.1%}.")
    return frac


def check_class_balance(labels: pd.Series,
                        min_minority_frac: Optional[float] = None) -> Dict[object, float]:
    """Return each class's fraction of non-null ``labels``; raise if the
    smallest class falls under ``min_minority_frac``.

    Motivated by this codebase's own documented finding
    (``MISSING_EVIDENCE.md`` item 9, the error-analysis writeup) that *MSH2*
    carries 144 of the panel's 180 ProteinGym-clinical benign labels against
    *MLH1*'s 6 -- a per-gene class-imbalance effect discovered only after
    training, by inspecting false positives. This makes that kind of
    imbalance checkable before training, on any label series (a full table,
    a single gene's partition, or a fold), rather than after the fact.
    """
    present = labels.dropna()
    if len(present) == 0:
        raise ValueError("No non-null labels to compute class balance from.")
    counts = present.value_counts(normalize=True)
    fractions = {k: float(v) for k, v in counts.items()}
    if min_minority_frac is not None:
        smallest = min(fractions.values())
        if smallest < min_minority_frac:
            raise ValueError(
                f"Smallest class fraction {smallest:.1%} is below the "
                f"declared minimum {min_minority_frac:.1%}. "
                f"Full distribution: {fractions}")
    return fractions


__all__ = [
    "SchemaValidationError", "MergeCardinalityError", "RowLossError",
    "DuplicateKeyError", "validate_schema", "validate_merge_cardinality",
    "check_no_unexpected_row_loss", "check_no_duplicate_keys",
    "check_unmatched_ids", "check_missing_value_policy", "check_class_balance",
]
