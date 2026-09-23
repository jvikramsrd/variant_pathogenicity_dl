"""Conclusion leakage: find the sentences that state the answer, remove only those.

A lab narrative usually ends by giving its verdict ("Therefore, this variant
has been classified as Pathogenic."). A model shown that sentence learns to
read verdicts, not evidence. Deleting the words "pathogenic", "benign" and
"VUS" everywhere is the wrong fix: "in trans with a pathogenic variant",
"benign in silico predictions" and "pathogenic variants in MLH1 cause Lynch
syndrome" are evidence and background, and would be destroyed.

So the unit is the SENTENCE, and a sentence is removed only when it matches a
conclusion pattern:

    classification_statement   "... is classified as likely pathogenic", "we classify
                               this variant as ...", "this variant is benign",
                               "reclassified from VUS to likely benign"
    criteria_statement         "meets ACMG criteria for pathogenic",
                               "likely benign according to ACMG guidelines"
    insufficient_evidence      "the available evidence is currently insufficient to
                               determine the role of this variant in disease"
                               (a VUS verdict in all but name)
    external_classification    "has been classified as pathogenic by other
                               laboratories / in ClinVar / by the expert panel"
                               (someone else's verdict — the consensus leaks)
    heading                    "Classification: Pathogenic", a bare "Likely benign."

Everything else is kept verbatim, including sentences that mention a class
word as evidence. :func:`residual_assertions` re-scans masked text for
assertive class statements the patterns missed; the leakage audit reports that
rate and fails a build above its threshold, so a new lab template that slips
through is caught by a number, not by luck.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vpdl.slm.text.sentences import Sentence, split_sentences

__all__ = ["CONCLUSION_CATEGORIES", "MaskedSpan", "MaskResult", "classify_sentence",
           "mask_conclusions", "residual_assertions", "LABEL_TERM"]

CONCLUSION_CATEGORIES = ("classification_statement", "criteria_statement",
                         "insufficient_evidence", "external_classification", "heading")

LABEL_TERM = (r"(?:likely[\s-]+pathogenic|likely[\s-]+benign|pathogenic|benign|"
              r"(?:a\s+)?variants?\s+of\s+(?:uncertain|unknown|undetermined)\s+(?:clinical\s+)?significance|"
              r"uncertain\s+(?:clinical\s+)?significance|unknown\s+significance|VUS|VOUS)")
_L = LABEL_TERM

_SUBJECT = (r"(?:this|the|that)\s+(?:(?:[\w.>*+-]+\s+){0,3})?"
            r"(?:variant|alteration|change|sequence\s+change|substitution|mutation|allele|deletion|"
            r"duplication|insertion|missense\s+change)|it|c\.[^\s,;]+|p\.[^\s,;]+")
_VERB = (r"(?:is|was|are|were|has\s+been|have\s+been|had\s+been|be|being|remains?)\s+"
         r"(?:(?:therefore|thus|currently|now|hereby|also|consequently|still|initially|"
         r"previously|herein|best)\s+)*")
_JUDGED = r"(?:re)?(?:classified|interpreted|considered|categori[sz]ed|assessed|curated|designated|reported|called|scored|assigned)"

_EXTERNAL = re.compile(
    r"\b(?:by|from|in|per)\s+(?:(?:multiple|several|other|many|\d+|two|three|four|five|"
    r"independent|different|additional|clinical)\s+)*(?:laborator(?:y|ies)|labs?|submitters?|"
    r"clinical\s+laborator(?:y|ies)|groups|centers|centres|databases?)\b"
    r"|\bin\s+clinvar\b|\bby\s+clinvar\b|clinvar\s+(?:submitters?|entr(?:y|ies)\s+\w+\s+" + _L + r")"
    r"|\bother\s+(?:laborator(?:y|ies)|submitters?|groups)\b"
    r"|\bby\s+(?:the\s+)?(?:insight|clingen|expert\s+panel|vcep|enigma)\b"
    r"|\b(?:insight|clingen|expert\s+panel|vcep|enigma)\s+(?:has|have)\s+(?:classified|interpreted)",
    re.I)

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("heading", re.compile(
        r"^\s*(?:acmg\s+|final\s+|overall\s+|clinical\s+)?(?:classification|interpretation|"
        r"conclusion|assessment|clinical\s+significance|verdict)\s*[:\-]", re.I)),
    ("heading", re.compile(r"^\s*[\"'(]?" + _L + r"[\"')]?\s*[.!]?\s*$", re.I)),
    ("insufficient_evidence", re.compile(
        r"(?:evidence|data|information)\s+(?:is|are|was|were)\s+(?:currently\s+|presently\s+|"
        r"at\s+this\s+time\s+)?(?:insufficient|inadequate|not\s+sufficient|limited)\s+to\s+"
        r"(?:determine|establish|classify|conclude|assess)", re.I)),
    ("insufficient_evidence", re.compile(
        r"(?:clinical\s+significance|role|pathogenicity|significance)\s+of\s+(?:this|the)\s+"
        r"(?:variant|alteration|change)\s+(?:in\s+disease\s+)?(?:is|remains)\s+"
        r"(?:currently\s+)?(?:unclear|uncertain|unknown|undetermined)", re.I)),
    ("criteria_statement", re.compile(
        r"(?:meets?|met|fulfils?|fulfills?|satisf(?:y|ies))\s+(?:the\s+)?(?:\w+\s+){0,5}?"
        r"(?:criteria|guidelines?|rules?|requirements?)\b.{0,60}?\b(?:for|as|to\s+be\s+(?:classified|"
        r"considered)\s+as)\s+(?:a\s+|an\s+)?" + _L, re.I)),
    ("criteria_statement", re.compile(
        _L + r"\s+(?:variant\s+)?(?:per|according\s+to|using|based\s+on|under|following)\s+"
        r"(?:the\s+)?(?:\w+\s+){0,3}?(?:acmg|amp|insight|clingen|guidelines?|criteria|"
        r"classification\s+(?:rules|scheme|system))", re.I)),
    ("classification_statement", re.compile(
        r"\b(?:we|our\s+(?:laboratory|lab))\s+(?:have\s+|has\s+)?(?:therefore\s+|thus\s+|now\s+)?"
        r"(?:re)?(?:classify|classified|interpret(?:ed)?|consider(?:ed)?|categori[sz]e[sd]?|"
        r"curate[sd]?|call(?:ed)?)\b.{0,80}?\b" + _L, re.I)),
    ("classification_statement", re.compile(
        _VERB + _JUDGED + r"\s+(?:to\s+be\s+|as\s+)?(?:a\s+|an\s+)?(?:\w+\s+){0,2}?" + _L, re.I)),
    ("classification_statement", re.compile(
        r"(?:re)?classif(?:ied|ication)\s+(?:was\s+|has\s+been\s+)?(?:changed|updated|revised|"
        r"downgraded|upgraded|moved)?\s*(?:from\s+" + _L + r"\s+)?to\s+" + _L, re.I)),
    ("classification_statement", re.compile(
        r"(?:downgraded|upgraded|reclassified)\s+(?:from\s+.{0,40}?\s+)?to\s+" + _L, re.I)),
    ("classification_statement", re.compile(
        r"(?:^|[,;:]\s*|\b(?:therefore|thus|hence|consequently|overall|in\s+summary|in\s+conclusion|"
        r"taken\s+together|collectively|together),?\s+)(?:" + _SUBJECT + r")\s+" + _VERB +
        r"(?:a\s+|an\s+)?(?:\w+\s+)?" + _L + r"(?:\s+variant)?\s*(?:[.;,]|$|\s+(?:and|based|given|"
        r"because|due|for|in|with))", re.I)),
    ("classification_statement", re.compile(
        r"(?:overall|available|the|this|these|collective)\s+(?:\w+\s+){0,2}?(?:evidence|data)\s+"
        r"(?:\w+\s+){0,2}?(?:supports?|indicates?|suggests?|is\s+consistent\s+with|favou?rs?)\s+"
        r"(?:a|an|its|the)?\s*(?:\w+\s+){0,2}?(?:" + _L + r"|classification\s+as\s+" + _L + r")", re.I)),
    ("classification_statement", re.compile(
        r"(?:classification|interpretation)\s+(?:of|as)\s+(?:a\s+|an\s+)?" + _L, re.I)),
]

_ASSERTIVE = re.compile(
    r"(?:\b(?:is|was|as|be|been)\s+(?:a\s+|an\s+)?(?:likely\s+)?(?:pathogenic|benign)\b(?!\s+(?:variant|"
    r"missense|allele|mutation|change)s?\s+(?:in|at|within|of|is|are|was|were|have|has|that|which)))"
    r"|\bclassified\s+as\b|\binterpreted\s+as\b"
    r"|\b(?:is|was|as)\s+(?:a\s+)?(?:VUS|VOUS|variant\s+of\s+uncertain\s+significance)\b", re.I)


@dataclass(frozen=True)
class MaskedSpan:
    start: int
    end: int
    category: str
    text: str


@dataclass(frozen=True)
class MaskResult:
    text: str
    spans: tuple[MaskedSpan, ...]
    n_sentences: int

    @property
    def n_masked(self) -> int:
        return len(self.spans)

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for span in self.spans:
            counts[span.category] = counts.get(span.category, 0) + 1
        return counts


# "Other missense changes at this residue have been reported as pathogenic" is
# evidence about OTHER variants (ACMG PM5), not this variant's verdict.
_OTHER_VARIANTS = re.compile(
    r"\b(?:other|different|another|additional|several|multiple|similar|nearby|neighbou?ring)\s+"
    r"(?:\w+\s+){0,2}?(?:variants?|changes?|substitutions?|alleles?|mutations?)\b", re.I)


def classify_sentence(sentence: str) -> str | None:
    """The conclusion category of one sentence, or None if it is not a conclusion."""
    text = sentence.strip()
    if not text:
        return None
    for category, pattern in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if category == "classification_statement" and _OTHER_VARIANTS.search(text[:match.start() + 1]):
            continue
        if category == "classification_statement" and _EXTERNAL.search(text):
            return "external_classification"
        return category
    if _EXTERNAL.search(text) and re.search(_L, text, re.I) and re.search(
            r"(?:classif|interpret|report|submit|assert|call)", text, re.I):
        return "external_classification"
    return None


def mask_conclusions(text: str, keep_external: bool = False,
                     replacement: str = "") -> MaskResult:
    """Remove conclusion sentences; everything else is kept byte for byte.

    ``keep_external`` keeps other submitters' verdicts (an ablation: "does
    knowing the consensus help?" is a different question from evidence reading).
    """
    sentences: list[Sentence] = split_sentences(text or "")
    spans: list[MaskedSpan] = []
    for sentence in sentences:
        category = classify_sentence(sentence.text)
        if category is None or (keep_external and category == "external_classification"):
            continue
        spans.append(MaskedSpan(sentence.start, sentence.end, category, sentence.text))
    if not spans:
        return MaskResult(text or "", (), len(sentences))
    pieces, cursor = [], 0
    for span in spans:
        pieces.append(text[cursor:span.start])
        if replacement:
            pieces.append(replacement)
        cursor = span.end
    pieces.append(text[cursor:])
    masked = re.sub(r"[ \t]{2,}", " ", "".join(pieces))
    masked = re.sub(r"\n\s*\n+", "\n", masked).strip()
    return MaskResult(masked, tuple(spans), len(sentences))


def residual_assertions(text: str) -> list[str]:
    """Assertive class statements still present — the leakage audit's tripwire.

    Deliberately broader than the masking patterns (it also fires on
    "the variant was benign in silico"), so it over-counts. The audit reports
    it as an upper bound on residual leakage; a jump in it after a data
    refresh means a new template is getting through.
    """
    return [m.group(0) for m in _ASSERTIVE.finditer(text or "")]
