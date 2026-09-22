"""PubMed abstracts from NLM's annual baseline (``pubmed*.xml.gz``).

Streams each file once; memory stays flat however many files there are.

What is kept: English records with an abstract — title plus abstract text,
with structured-abstract labels kept ("METHODS: ...").

What is dropped, and counted: records without an abstract, non-English
records, and — because this model is for clinical questions — **retracted
papers, retraction notices and expressions of concern**. NLM marks a retracted
article with the "Retracted Publication" type and a ``RetractionIn`` link;
either one is enough to drop it.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

__all__ = ["Abstract", "EXCLUDED_TYPES", "iter_abstracts"]

EXCLUDED_TYPES = frozenset({
    "Retracted Publication", "Retraction of Publication", "Expression of Concern",
})


@dataclass(frozen=True)
class Abstract:
    pmid: str
    year: str
    text: str


def _text(element: ET.Element | None) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _parse(article: ET.Element, stats: Counter) -> Abstract | None:
    citation = article.find("MedlineCitation")
    if citation is None:
        stats["malformed"] += 1
        return None
    body = citation.find("Article")
    if body is None:
        stats["malformed"] += 1
        return None

    types = {t.text for t in body.findall("PublicationTypeList/PublicationType")}
    retracted = any(c.get("RefType") == "RetractionIn"
                    for c in citation.findall("CommentsCorrectionsList/CommentsCorrections"))
    if retracted or types & EXCLUDED_TYPES:
        stats["retracted_or_concern"] += 1
        return None

    languages = {(language.text or "").lower() for language in body.findall("Language")}
    if "eng" not in languages:
        stats["not_english"] += 1
        return None

    parts = []
    for node in body.findall("Abstract/AbstractText"):
        text = _text(node)
        label = (node.get("Label") or "").strip()
        if text:
            parts.append(f"{label}: {text}" if label and label.upper() != "UNLABELLED" else text)
    if not parts:
        stats["no_abstract"] += 1
        return None

    title = _text(body.find("ArticleTitle"))
    issue = body.find("Journal/JournalIssue/PubDate")
    year = ""
    if issue is not None:
        year = issue.findtext("Year") or (issue.findtext("MedlineDate") or "")[:4]
    stats["kept"] += 1
    return Abstract(pmid=citation.findtext("PMID") or "", year=year,
                    text="\n".join([title, *parts]) if title else "\n".join(parts))


def iter_abstracts(path: Path | str, stats: Counter | None = None) -> Iterator[Abstract]:
    """Yield the kept abstracts of one baseline file; ``stats`` counts every decision."""
    stats = stats if stats is not None else Counter()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        context = ET.iterparse(handle, events=("start", "end"))
        _, root = next(context)
        for event, element in context:
            if event == "end" and element.tag == "PubmedArticle":
                abstract = _parse(element, stats)
                if abstract is not None:
                    yield abstract
                root.clear()                 # keep memory flat across 30k records
