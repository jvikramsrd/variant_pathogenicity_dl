"""Context tiers and third-party text restrictions.

Tiers say what KIND of text a document is, so results can be broken down by
it and a model can be checked for leaning on one kind. They are not a truth
ranking: an expert-panel summary can be out of date, and a single lab's
narrative can hold the only functional data on a variant.

    1  expert-curated interpretation: ClinGen expert panels (VCEP), practice guidelines
    2  clinical-laboratory interpretation with assertion criteria and a narrative
    3  literature-derived or research submissions (case reports, functional studies)
    4  biomedical literature and reviews (PubMed, PMC, GeneReviews, MedlinePlus)
    5  weakly structured / low-context (no criteria, no or very short narrative)

Restricted text: ClinVar redistributes what submitters send, but some
submitters' text carries its source's terms. OMIM's allelic-variant
descriptions reach ClinVar as "literature only" submissions from OMIM, and
OMIM's terms forbid incorporating its content into software without a licence
(docs/kb/SOURCES.md). Such documents are kept (they can be counted and cited)
but flagged ``restricted`` and kept out of model weights by default.
"""

from __future__ import annotations

__all__ = ["TIERS", "assign_tier", "restriction", "RESTRICTED_SUBMITTERS", "MIN_NARRATIVE_CHARS"]

TIERS = {1: "expert-curated interpretation", 2: "clinical laboratory interpretation",
         3: "literature-derived / research submission", 4: "biomedical literature and reviews",
         5: "weakly structured / low context"}

LITERATURE_SOURCES = {"pubmed", "pmc", "genereviews", "medlineplus", "orphanet", "clingen_gene_validity"}
EXPERT_SOURCES = {"clingen_erepo"}
MIN_NARRATIVE_CHARS = 40

RESTRICTED_SUBMITTERS = {
    "omim": "OMIM terms: no incorporation into software without a Johns Hopkins licence "
            "(docs/kb/SOURCES.md, section 3)",
}


def assign_tier(source_id: str, review_status: str | None = None,
                collection_method: str | None = None, text: str | None = None) -> int:
    source = (source_id or "").lower()
    if source in EXPERT_SOURCES:
        return 1
    if source in LITERATURE_SOURCES:
        return 4
    status = (review_status or "").lower()
    method = (collection_method or "").lower()
    narrative = len((text or "").strip()) >= MIN_NARRATIVE_CHARS
    if "expert panel" in status or "practice guideline" in status:
        return 1
    if not narrative or "no assertion criteria" in status or "no classification" in status:
        return 5
    if "literature only" in method or "research" in method or "curation" in method:
        return 3
    if "criteria provided" in status:
        return 2
    return 5


def restriction(submitter: str | None) -> str | None:
    """Why this submitter's text is kept out of weights, or None."""
    name = (submitter or "").strip().lower()
    return RESTRICTED_SUBMITTERS.get(name)
