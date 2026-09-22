"""Evaluation folds beyond leave-one-gene-out.

``logo`` is the primary protocol and reproduces :func:`vpdl.experiment.run_cell`'s
original loop exactly (same rows, same seed key = the held-out gene), so every
existing result keeps its meaning. The others answer different questions:

    logo           MLH1+MSH2+MSH6 -> PMS2, etc. Transfer to an unseen gene.
                   Paralogs of the held-out gene stay in training.
    logo_purged    logo, minus training rows at residues homologous to any test
                   residue (shared ``cluster_id``). Transfer with no aligned
                   paralog position left to copy from.
    family         MutL (MLH1, PMS2) vs MutS (MSH2, MSH6). The sequence-cluster
                   split: nothing at >= ~26% identity crosses it (measured with
                   vpdl.dl.homology on the four sequences; cross-family <= 3%).
    random_debug   residue-grouped random K-fold within all genes. Leaks gene
                   identity and paralog context by design. Debugging only —
                   never a reported result.

Every scheme returns :class:`Fold` objects with positional indices into the
same working frame, so the rest of the cell (inner split, threshold, metrics,
provenance) is shared code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from vpdl.dl.homology import PROTEIN_FAMILIES, family_of
from vpdl.splits import group_keys

__all__ = ["SPLIT_SCHEMES", "REPORTABLE_SCHEMES", "GENE_DISJOINT_SCHEMES",
           "Fold", "make_folds", "describe_folds"]

SPLIT_SCHEMES = ("logo", "logo_purged", "family", "random_debug")
REPORTABLE_SCHEMES = frozenset({"logo", "logo_purged", "family"})
GENE_DISJOINT_SCHEMES = frozenset({"logo", "logo_purged", "family"})


@dataclass(frozen=True)
class Fold:
    """One outer fold. `name` is the seed key (the gene, for logo)."""

    name: str
    train_positions: np.ndarray
    test_positions: np.ndarray
    test_genes: tuple[str, ...]
    purged: int = 0


def _hash_fold(key: str, k: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % k


def make_folds(
    work: pd.DataFrame,
    scheme: str = "logo",
    train_column: str = "_train",
    eval_column: str = "_eval",
    families: Mapping[str, str] = PROTEIN_FAMILIES,
    n_folds: int = 5,
    seed: int = 42,
) -> list[Fold]:
    """Outer folds over `work` (rows carrying a training or evaluation label)."""
    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"unknown split scheme {scheme!r}; known: {SPLIT_SCHEMES}")
    trainable = work[train_column].notna().to_numpy()
    evaluable = work[eval_column].notna().to_numpy()
    genes = work["gene"].to_numpy()

    def eval_values(column: np.ndarray) -> list[str]:
        return sorted(str(v) for v in pd.unique(column[evaluable])
                      if pd.notna(v) and str(v).strip())

    folds: list[Fold] = []
    if scheme in ("logo", "logo_purged"):
        for gene in eval_values(genes):
            train = np.where((genes != gene) & trainable)[0]
            test = np.where((genes == gene) & evaluable)[0]
            folds.append(Fold(gene, train, test, (gene,)))
        if scheme == "logo_purged":
            folds = [_purge(work, fold) for fold in folds]
        return folds

    if scheme == "family":
        family = work["uniprot_id"].map(lambda acc: family_of(acc, families)).to_numpy()
        for name in eval_values(family):
            train = np.where((family != name) & trainable)[0]
            test = np.where((family == name) & evaluable)[0]
            folds.append(Fold(f"family:{name}", train, test,
                              tuple(sorted(set(genes[test])))))
        return folds

    groups = group_keys(work)
    assignment = np.array([_hash_fold(g, n_folds, seed) for g in groups])
    for k in range(n_folds):
        train = np.where((assignment != k) & trainable)[0]
        test = np.where((assignment == k) & evaluable)[0]
        if len(test):
            folds.append(Fold(f"random_debug:{k}", train, test,
                              tuple(sorted(set(genes[test])))))
    return folds


def _purge(work: pd.DataFrame, fold: Fold) -> Fold:
    """Drop training rows sharing a homology cluster with any test row."""
    if "cluster_id" not in work.columns:
        raise ValueError(
            "logo_purged needs a `cluster_id` column (build the canonical table "
            "with vpdl.dl.canonical, which assigns homology clusters).")
    clusters = work["cluster_id"].to_numpy()
    test_clusters = set(clusters[fold.test_positions])
    keep = np.array([clusters[i] not in test_clusters for i in fold.train_positions],
                    dtype=bool)
    return Fold(fold.name, fold.train_positions[keep], fold.test_positions,
                fold.test_genes, purged=int((~keep).sum()))


def describe_folds(work: pd.DataFrame, folds: Sequence[Fold],
                   train_column: str = "_train", eval_column: str = "_eval") -> pd.DataFrame:
    """Row counts and class balance per fold — printed before any training."""
    rows = []
    for fold in folds:
        train = work.iloc[fold.train_positions]
        test = work.iloc[fold.test_positions]
        rows.append({
            "fold": fold.name, "test_genes": "+".join(fold.test_genes),
            "n_train": len(train), "train_pathogenic": int((train[train_column] == 1).sum()),
            "n_test": len(test), "test_pathogenic": int((test[eval_column] == 1).sum()),
            "purged_homologous_train_rows": fold.purged,
        })
    return pd.DataFrame(rows)
