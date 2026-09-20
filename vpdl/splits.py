"""Evaluation partitions, and the leakage invariant they exist to protect.

Leave-one-gene-out is the primary protocol: with four genes it is the only
honest estimate of transfer to an unseen gene, and it is what every v1 number
was reported on.

Satisfies regression landmine L3 (leakage groups are protein-qualified).
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["group_keys", "lopo_splits", "assert_no_group_straddle", "variant_keys"]


def group_keys(df: pd.DataFrame) -> np.ndarray:
    """Leakage-group key per row: ``uniprot:position``.

    The protein qualifier is not decoration. v1 grouped on the bare residue
    `position`, so pooling proteins collapsed e.g. P40692:175 and P43246:175
    into one group — folds were neither leakage-free nor the sizes they
    reported (CODE_REVIEW D2).
    """
    missing = {"uniprot_id", "position"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot build leakage groups without {sorted(missing)}. "
            "Grouping on position alone silently merges distinct proteins."
        )
    return (
        df["uniprot_id"].astype(str) + ":" + df["position"].astype(int).astype(str)
    ).to_numpy()


def variant_keys(df: pd.DataFrame) -> np.ndarray:
    """Fully-qualified variant identity, used for split hashing."""
    return (
        df["uniprot_id"].astype(str)
        + ":" + df["position"].astype(int).astype(str)
        + ":" + df["wt_aa"].astype(str)
        + ">" + df["mut_aa"].astype(str)
    ).to_numpy()


def lopo_splits(
    df: pd.DataFrame,
    genes: Sequence[str] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_positions, test_positions)`` holding out one gene at a time.

    Positions are positional indices for ``.iloc``, not labels, so callers cannot
    accidentally index by a stale DataFrame index.

    A gene with no rows yields nothing and is reported by the caller. v1 logged
    a warning here that nobody read, and 16 grid cells trained on a three-gene
    panel believing it was four (MISSING_EVIDENCE item 14) — so
    :func:`vpdl.assemble.assert_genes_present` is the real guard, and this
    function stays honest about what it actually produced.
    """
    if "gene" not in df.columns:
        raise ValueError("Leave-one-gene-out requires a 'gene' column.")

    if genes is not None:
        order = list(genes)
    else:
        # Blank/NaN gene names are dropped rather than becoming a fold. Feature-
        # only sources (AlphaMissense, gnomAD) can emit rows whose gene did not
        # resolve, and a "" fold would silently appear in the results table.
        present = df["gene"].dropna().unique()
        order = sorted(str(g) for g in present if str(g).strip())
        blank = len(present) - len(order)
        if blank:
            logger.warning(
                "%d row group(s) have a blank gene name and are excluded from "
                "leave-one-gene-out. Check the source's gene resolution.", blank,
            )

    positions = np.arange(len(df))
    gene_column = df["gene"].to_numpy()

    for gene in order:
        held_out = gene_column == gene
        if not held_out.any():
            continue
        yield positions[~held_out], positions[held_out]


def assert_no_group_straddle(
    df: pd.DataFrame,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
) -> None:
    """Assert at runtime that no leakage group spans the split.

    Cheap, and the one check that would have caught a whole class of
    silently-inflated results.
    """
    train_groups = set(group_keys(df.iloc[train_positions]))
    test_groups = set(group_keys(df.iloc[test_positions]))
    shared = train_groups & test_groups
    if shared:
        sample = sorted(shared)[:5]
        raise ValueError(
            f"{len(shared)} leakage group(s) appear in both train and test, "
            f"e.g. {sample}. Variants at one (protein, residue) must stay in "
            "one partition."
        )
