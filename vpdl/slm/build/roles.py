"""Data roles, the evaluation sets reserved up front, and what pretraining must not see.

Roles (``vpdl.slm.schema.DATA_ROLES``) are assigned per document from a split,
with one override: documents of variants that have INDEPENDENT functional data
are INDEPENDENT_VALIDATION whatever the split says.

Reserved evaluation sets are fixed BEFORE pretraining, because a pretraining
corpus is built once and reused: the MMR test set, the temporal test set, the
functional holdout and the unseen-gene panel. Their narratives are never
pretraining text, and in strict mode neither are the publications ClinVar
cites for them (``pretraining_exclusions``).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

__all__ = ["SPLIT_TO_ROLE", "assign_roles", "reserved_variants", "pretraining_exclusions"]

SPLIT_TO_ROLE = {"train": "TRAINING", "mmr_train": "TRAINING", "val": "VALIDATION",
                 "mmr_val": "VALIDATION", "test": "TEST", "broad_test": "TEST", "excluded": "EXCLUDED"}


def assign_roles(split: pd.DataFrame, independent_variants: Iterable[str] = ()) -> pd.Series:
    roles = split["split"].map(SPLIT_TO_ROLE).fillna("EXCLUDED")
    independent = set(independent_variants)
    if independent:
        roles = roles.where(~split["variant_id"].isin(independent), "INDEPENDENT_VALIDATION")
    return roles


def reserved_variants(splits: Mapping[str, pd.DataFrame],
                      independent_variants: Iterable[str] = ()) -> set[str]:
    """Every variant that is VALIDATION / TEST / INDEPENDENT in any fixed evaluation split."""
    reserved = set(independent_variants)
    for frame in splits.values():
        held = frame.loc[~frame["split"].isin(["train", "mmr_train", "excluded"]), "variant_id"]
        reserved |= set(held.astype(str))
    return reserved


def pretraining_exclusions(tables: Mapping[str, pd.DataFrame], reserved: set[str],
                           holdout_publications: Iterable[str] = (),
                           strict_literature: bool = True) -> dict[str, Any]:
    documents = tables["documents"]
    excluded_documents = set(documents.loc[documents["variant_id"].isin(reserved), "document_id"])
    pmids: set[str] = set()
    if strict_literature:
        citations = tables.get("citations", pd.DataFrame())
        if not citations.empty:
            ids = citations["variation_id"].map(lambda v: f"clinvar:{int(v)}")
            chosen = citations.loc[ids.isin(reserved) & (citations["citation_source"] == "PubMed")]
            pmids |= set(chosen["citation_id"].astype(str))
        in_text = documents.loc[documents["variant_id"].isin(reserved), "text_pmids"]
        pmids |= {p for values in in_text for p in values}
    for identifier in holdout_publications:
        kind, _, value = identifier.partition(":")
        if kind.upper() == "PMID":
            pmids.add(value)
    return {"kind": "exclusions", "documents": sorted(excluded_documents), "pmids": sorted(pmids),
            "reserved_variants": len(reserved), "strict_literature": strict_literature}
