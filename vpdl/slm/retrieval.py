"""Evidence retrieval that cannot reach into the evaluation, and keeps provenance.

The knowledge base (``vpdl.kb``) retrieves passages for a reader model; this
retrieves EVIDENCE UNITS and documents for grounding a variant's explanation —
the units another variant's submitters wrote, gene and disease descriptions,
literature passages.

Two rules, enforced in code rather than by convention:

1. **Role filter.** An index built for training refuses documents whose role is
   VALIDATION, TEST or INDEPENDENT_VALIDATION (:class:`RoleFilterError`). The
   leakage audit's ``retrieval_contamination`` check re-verifies it from the
   built index.
2. **As-of date.** A temporal experiment may only retrieve what existed then;
   ``as_of`` drops later items.

Every retrieved item carries source_id, document_id, variant_id, gene, disease,
publication, timestamp, evidence type, the text and the retrieval score, so an
explanation can cite it and a reader can check it.

Search is exact-word BM25 over the unit text — ``vpdl.kb.search``'s
implementation, imported unchanged, because genetics queries are full of
identifiers. An embedding stage can be added the same way the KB does it;
it is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from vpdl.kb.search import BM25, reciprocal_rank_fusion
from vpdl.slm.schema import as_list

__all__ = ["RetrievedItem", "EvidenceIndex", "RoleFilterError"]

BLOCKED_ROLES = ("VALIDATION", "TEST", "INDEPENDENT_VALIDATION")


class RoleFilterError(ValueError):
    """An index for training was given documents it must not hold."""


@dataclass
class RetrievedItem:
    evidence_id: str
    document_id: str
    variant_id: str
    gene: str
    disease: str | None
    source_id: str
    publication: str | None
    timestamp: str | None
    evidence_types: tuple[str, ...]
    text: str
    retrieval_score: float
    provenance: str
    role: str

    def as_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass
class EvidenceIndex:
    frame: pd.DataFrame
    for_training: bool = True
    bm25: BM25 | None = field(default=None, repr=False)

    @classmethod
    def build(cls, units: pd.DataFrame, roles: pd.Series | None = None, for_training: bool = True,
              documents: pd.DataFrame | None = None, roles_by_document: pd.Series | None = None,
              keep_roles: Iterable[str] = ("TRAINING", "PRETRAINING", "KNOWLEDGE_BASE", "REFERENCE_ONLY")
              ) -> "EvidenceIndex":
        frame = units.loc[units["sentence_role"].isin(["evidence", "code_list"])].copy()
        if roles_by_document is not None:
            frame["role"] = frame["document_id"].map(roles_by_document).fillna("UNKNOWN")
        elif roles is not None:
            frame["role"] = roles.reindex(frame.index).fillna("UNKNOWN")
        else:
            frame["role"] = "UNKNOWN"
        if for_training:
            blocked = frame.loc[frame["role"].isin(BLOCKED_ROLES)]
            if len(blocked):
                raise RoleFilterError(
                    f"{len(blocked)} evidence unit(s) with roles "
                    f"{sorted(set(blocked['role']))} were given to a training-time index; "
                    "build it from TRAINING documents only (docs/slm/GENOMIC_SLM_DATA_LEAKAGE_REPORT.md).")
            frame = frame.loc[frame["role"].isin(list(keep_roles) + ["UNKNOWN"])]
        if documents is not None and "disease_names" in documents:
            lookup = documents.set_index("document_id")["disease_names"]
            frame["disease"] = frame["document_id"].map(
                lambda d: (lookup.get(d) or [None])[0] if d in lookup.index else None)
        else:
            frame["disease"] = None
        frame = frame.reset_index(drop=True)
        return cls(frame, for_training, BM25(frame["text_span"].fillna("").tolist()))

    def document_ids(self) -> list[str]:
        return sorted(set(self.frame["document_id"].astype(str)))

    def search(self, query: str, k: int = 6, as_of: str | None = None,
               exclude_variants: Sequence[str] = ()) -> list[RetrievedItem]:
        if self.bm25 is None:
            self.bm25 = BM25(self.frame["text_span"].fillna("").tolist())
        scores = self.bm25.scores(query)
        order = [int(i) for i in scores.argsort()[::-1] if scores[i] > 0]
        ranked = reciprocal_rank_fusion([order[:100]])
        excluded = set(exclude_variants)
        items: list[RetrievedItem] = []
        for index in ranked:
            row = self.frame.iloc[index]
            if as_of and isinstance(row.get("date"), str) and row["date"] > as_of:
                continue
            if row["variant_id"] in excluded:
                continue
            items.append(RetrievedItem(
                evidence_id=row["evidence_id"], document_id=row["document_id"],
                variant_id=row["variant_id"], gene=row.get("gene", ""), disease=row.get("disease"),
                source_id=row.get("source_id", ""),
                publication=(as_list(row.get("pmids")) + [None])[0],
                timestamp=row.get("date"), evidence_types=tuple(as_list(row.get("evidence_types"))),
                text=row["text_span"], retrieval_score=float(scores[index]),
                provenance=row.get("provenance", ""), role=row.get("role", "UNKNOWN")))
            if len(items) >= k:
                break
        return items
