"""Evidence-grounded explanations, and the check that keeps them grounded.

Output shape (docs/slm/GENOMIC_SLM_REASONING.md):

    classification     one of the five classes, or "abstain"
    probabilities      five calibrated probabilities
    key_evidence       the evidence units used, verbatim, each with its id [E#] and source
    evidence_polarity  counts of pathogenic / benign / neutral / mixed units
    acmg_context       codes the SOURCE stated (with the unit) and codes the MODEL predicted
                       (marked predicted, masked where the gene's specification excludes them)
    uncertainty        entropy, abstention and why
    explanation        sentences, each ending in citations [E#] — or [MODEL] for a statement
                       about the model's own output
    missing_evidence   evidence types not found in the provided text (not "absent": unseen)
    sources            document, submitter, date for every cited unit

:func:`explain` builds this deterministically from evidence units, so it is
grounded by construction; it states only what a unit's words say. There is no
free-text chain of thought.

:func:`check_grounding` scores ANY explanation (a teacher's, a future
generative head's) against the evidence it was given. A sentence is
unsupported when it cites nothing, cites an id not provided, states a number
no cited unit contains, names an ACMG code neither stated nor predicted,
claims an evidence type no cited unit shows, or states the opposite direction
of every unit it cites.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vpdl.slm.schema import CLASSES, as_list
from vpdl.slm.text.acmg import find_codes
from vpdl.slm.text.evidence import _BENIGN_CUES, _PATHOGENIC_CUES, _TYPE_RULES
from vpdl.slm.text.sentences import split_sentences

__all__ = ["CORE_EVIDENCE_TYPES", "explain", "check_grounding", "GroundingReport", "DISCLAIMER"]

DISCLAIMER = ("RESEARCH USE ONLY. A research model's reading of submitted evidence; not a "
              "clinical classification and not a substitute for expert curation.")
CORE_EVIDENCE_TYPES = ("population", "functional", "segregation", "computational", "case",
                       "phenotype", "de_novo", "allelic", "splicing")
_CITATION = re.compile(r"\[(E\d+|MODEL)\]")
_NUMBER = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?%?")
# Type words a sentence must be able to back with a cited unit.
_CLAIM_TYPES = {name: _TYPE_RULES[name] for name in
                ("population", "functional", "segregation", "de_novo", "allelic", "splicing")}


def _truncate(text: str, limit: int = 240) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def explain(probs: Sequence[float], units: Sequence[Mapping[str, Any]],
            predicted_codes: Sequence[str] = (), inapplicable_codes: Sequence[str] = (),
            uncertainty: Mapping[str, Any] | None = None, max_units: int = 6,
            abstained: bool = False) -> dict[str, Any]:
    """Deterministic explanation from evidence units (dicts as in the evidence_units table)."""
    probs = np.asarray(probs, dtype=float)
    usable = [u for u in units if u.get("sentence_role") in ("evidence", "code_list")]
    ranked = sorted(usable, key=lambda u: (u.get("evidence_polarity") == "neutral",
                                           -float(u.get("confidence", 0))))[:max_units]
    key, sentences = [], []
    for number, unit in enumerate(ranked, start=1):
        tag = f"E{number}"
        key.append({"id": tag, "evidence_id": unit.get("evidence_id"), "text": unit.get("text_span"),
                    "types": as_list(unit.get("evidence_types")),
                    "polarity": unit.get("evidence_polarity"),
                    "acmg_codes": as_list(unit.get("acmg_codes")),
                    "source_id": unit.get("source_id"), "document_id": unit.get("document_id")})
        types = ", ".join(as_list(unit.get("evidence_types"))) or "unclassified"
        direction = {"pathogenic": "points toward pathogenicity",
                     "benign": "points toward a benign effect",
                     "mixed": "contains evidence in both directions",
                     "neutral": "has no clear direction"}.get(unit.get("evidence_polarity"), "")
        sentences.append(f"The source states ({types}; {direction}): "
                         f"\"{_truncate(unit.get('text_span') or '')}\" [{tag}].")
    order = np.argsort(-probs)
    top = CLASSES[int(order[0])]
    sentences.append(f"The model assigns the highest probability to {top.replace('_', ' ')} "
                     f"({probs[order[0]]:.2f}); the next is {CLASSES[int(order[1])].replace('_', ' ')} "
                     f"({probs[order[1]]:.2f}) [MODEL].")
    if abstained:
        sentences.append("The model abstains: its confidence or the available evidence is below "
                         "the validated threshold [MODEL].")
    seen_types = {t for u in usable for t in as_list(u.get("evidence_types"))}
    stated = [{"code": c, "status": "stated_by_source", "evidence_id": u.get("evidence_id")}
              for u in usable for c in as_list(u.get("acmg_codes"))]
    predicted = [{"code": c, "status": "predicted", "gene_applicable": c.split("_")[0] not in inapplicable_codes}
                 for c in predicted_codes]
    polarity = {p: sum(1 for u in usable if u.get("evidence_polarity") == p)
                for p in ("pathogenic", "benign", "neutral", "mixed")}
    return {
        "classification": "abstain" if abstained else top,
        "probabilities": {name: round(float(p), 4) for name, p in zip(CLASSES, probs)},
        "key_evidence": key, "evidence_polarity": polarity,
        "acmg_context": stated + [p for p in predicted if p["gene_applicable"]],
        "uncertainty": dict(uncertainty or {}) | {"abstained": bool(abstained)},
        "explanation": " ".join(sentences),
        "missing_evidence": [t for t in CORE_EVIDENCE_TYPES if t not in seen_types],
        "sources": sorted({(u.get("source_id") or "", u.get("document_id") or "") for u in ranked}),
        "disclaimer": DISCLAIMER,
    }


@dataclass
class GroundingReport:
    sentences: int = 0
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    invalid_citations: int = 0
    number_mismatches: int = 0
    code_hallucinations: int = 0
    invented_types: int = 0
    contradictions: int = 0
    cited: set[str] = field(default_factory=set)
    evidence_ids: set[str] = field(default_factory=set)

    @property
    def unsupported_claim_rate(self) -> float:
        return len(self.unsupported) / self.sentences if self.sentences else float("nan")

    @property
    def evidence_coverage(self) -> float:
        return len(self.cited & self.evidence_ids) / len(self.evidence_ids) if self.evidence_ids else float("nan")

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cited"] = sorted(self.cited)
        out["evidence_ids"] = sorted(self.evidence_ids)
        out["unsupported_claim_rate"] = self.unsupported_claim_rate
        out["evidence_coverage"] = self.evidence_coverage
        out["contradiction_rate"] = self.contradictions / self.sentences if self.sentences else float("nan")
        return out


_ONLY_CITATIONS = re.compile(r"^\s*(?:\[(?:E\d+|MODEL)\]\s*)+[.;,]?\s*$")


def _sentences_with_citations(text: str) -> list[str]:
    """Sentences, with a trailing citation-only fragment joined to the sentence it cites.

    A quoted passage ends in its own full stop, so "… database." [E1]." splits
    into two. The citation belongs to the sentence before it (the knowledge
    base's citation reader handles the same case the same way).
    """
    out: list[str] = []
    for sentence in split_sentences(text):
        if out and _ONLY_CITATIONS.match(sentence.text):
            out[-1] = f"{out[-1]} {sentence.text.strip()}"
        else:
            out.append(sentence.text)
    return out


def check_grounding(explanation: str, evidence: Mapping[str, Mapping[str, Any]],
                    predicted_codes: Sequence[str] = ()) -> GroundingReport:
    """`evidence`: {"E1": {"text": ..., "acmg_codes": [...], "polarity": ...}, ...}."""
    report = GroundingReport(evidence_ids=set(evidence))
    allowed_codes = {c.split("_")[0] for c in predicted_codes}
    for text in _sentences_with_citations(explanation or ""):
        tags = _CITATION.findall(text)
        body = _CITATION.sub("", text)
        if not re.search(r"[A-Za-z]{3,}", body):
            continue
        report.sentences += 1
        problems = []
        if not tags:
            problems.append("no citation")
        cited = [t for t in tags if t != "MODEL"]
        missing = [t for t in cited if t not in evidence]
        if missing:
            report.invalid_citations += len(missing)
            problems.append(f"cites unknown {missing}")
        report.cited.update(t for t in cited if t in evidence)
        cited_units = [evidence[t] for t in cited if t in evidence]
        if cited and not missing:
            support_text = " ".join(u.get("text", "") for u in cited_units)
            numbers = set(_NUMBER.findall(body)) - set(_NUMBER.findall(support_text))
            if numbers:
                report.number_mismatches += 1
                problems.append(f"numbers not in cited evidence {sorted(numbers)}")
            stated = {c.split("_")[0] for u in cited_units for c in as_list(u.get("acmg_codes"))}
            # In an explanation a code is always meant as a code, so no context is
            # required on either side ("It meets PS3" must be checkable).
            stated |= {m.code for m in find_codes(support_text, require_context=False)}
            codes = {m.code for m in find_codes(body, require_context=False)} - stated - allowed_codes
            if codes:
                report.code_hallucinations += 1
                problems.append(f"ACMG codes not stated or predicted {sorted(codes)}")
            quoted = set(re.findall(r'"([^"]+)"', body))
            claim_text = body if not quoted else re.sub(r'"[^"]+"', " ", body)
            for name, rule in _CLAIM_TYPES.items():
                if rule.search(claim_text) and not rule.search(support_text):
                    report.invented_types += 1
                    problems.append(f"claims {name} evidence the cited units do not show")
            says_path = bool(_PATHOGENIC_CUES.search(claim_text))
            says_benign = bool(_BENIGN_CUES.search(claim_text))
            polarities = {u.get("polarity") for u in cited_units}
            if (says_path and not says_benign and polarities and polarities <= {"benign"}) or \
               (says_benign and not says_path and polarities and polarities <= {"pathogenic"}):
                report.contradictions += 1
                problems.append("direction contradicts every cited unit")
        elif tags == ["MODEL"] or (tags and set(tags) == {"MODEL"}):
            problems = []                         # a statement about the model's own output
        if problems:
            report.unsupported.append({"sentence": text, "problems": problems})
    return report
