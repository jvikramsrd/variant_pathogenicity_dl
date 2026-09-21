"""Question -> passages -> local model -> citation check -> answer, or "not found".

The model sees only the retrieved passages and must cite one for every
sentence. :func:`check_citations` enforces that mechanically: an answer with an
uncited sentence, or a citation to a passage that was not retrieved, is
withheld — not trimmed. Deleting the offending sentence could delete the
"not" that the rest depends on.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from vpdl.kb.chunks import Chunk
from vpdl.kb.search import KnowledgeIndex
from vpdl.kb.variants import EvidenceTable, VariantDB, VariantReport

__all__ = ["SYSTEM_PROMPT", "NOT_FOUND", "DISCLAIMER", "build_messages",
           "check_citations", "Answer", "ask", "render"]

NOT_FOUND = "NOT_FOUND"
DISCLAIMER = ("RESEARCH USE ONLY. Decision support for qualified experts; "
              "not a medical device and not a diagnosis.")

SYSTEM_PROMPT = f"""You answer questions from clinical genetics researchers and clinicians, using ONLY the numbered source passages you are given.

Rules:
1. Every sentence must end with at least one citation such as [S1] or [S2][S3], naming the passage(s) that state it.
2. Use only what the passages state. Do not add anything from memory, even if you believe it is true.
3. If the passages do not answer the question, reply with exactly: {NOT_FOUND}
4. Never say whether a specific variant is pathogenic, benign or uncertain. Variant classifications are reported separately from the database.
5. Be concise. Copy numbers, ages, intervals and gene names exactly as written in the passages."""


class ModelClient(Protocol):
    def chat(self, messages: list[dict], model: str) -> str: ...
    def embed(self, texts: Sequence[str], model: str): ...


def build_messages(question: str, passages: Sequence[Chunk]) -> list[dict]:
    blocks = [f"[S{i}] ({p.doc_title} - {p.section})\n{p.text}"
              for i, p in enumerate(passages, start=1)]
    user = "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# -- citation check -----------------------------------------------------------

_CITATION = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\]")
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "vs.", "approx.", "Fig.", "No.", "Dr.", "ca.")
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LEADING_CITATIONS = re.compile(r"^((?:\[[^\]]*\]\s*)+)(.*)$", re.DOTALL)


@dataclass
class CitationCheck:
    ok: bool
    reason: str = ""          # not_found | uncited_sentence | invalid_citation | empty
    detail: str = ""          # the offending text: for reviewers, never shown to readers
    cited: list[int] = field(default_factory=list)


# What a reader is told when an answer is withheld. It names the failure and
# deliberately does not quote it — quoting an unsupported sentence would show
# the reader exactly the claim the check exists to stop.
_WITHHELD = {
    "uncited_sentence": "a sentence had no citation",
    "invalid_citation": "it cited a passage that was not retrieved",
    "empty": "it gave no answer",
}


def _sentences(text: str) -> list[str]:
    protected = text
    for i, abbreviation in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbreviation, f"\x00{i}\x00")
    sentences: list[str] = []
    for line in protected.splitlines():
        for piece in _SPLIT.split(line.strip()):
            if not piece:
                continue
            # "... every 1-2 years. [S1]" — a citation after the full stop belongs
            # to the sentence before it.
            leading = _LEADING_CITATIONS.match(piece)
            if leading and sentences:
                sentences[-1] += " " + leading.group(1).strip()
                piece = leading.group(2).strip()
            if piece:
                sentences.append(piece)
    restored = []
    for sentence in sentences:
        for i, abbreviation in enumerate(_ABBREVIATIONS):
            sentence = sentence.replace(f"\x00{i}\x00", abbreviation)
        restored.append(sentence)
    return restored


def check_citations(answer: str, n_passages: int) -> CitationCheck:
    text = answer.strip()
    if not text or NOT_FOUND in text:
        return CitationCheck(False, "not_found")
    cited: list[int] = []
    claims = 0
    for sentence in _sentences(text):
        words = re.findall(r"[A-Za-z]{2,}", _CITATION.sub("", sentence))
        if len(words) < 4 or sentence.rstrip(" *").endswith(":"):
            continue                         # a heading or fragment, not a claim
        claims += 1
        numbers = [int(n) for group in _CITATION.findall(sentence)
                   for n in re.findall(r"\d+", group)]
        if not numbers:
            return CitationCheck(False, "uncited_sentence", sentence[:200])
        bad = [n for n in numbers if not 1 <= n <= n_passages]
        if bad:
            return CitationCheck(False, "invalid_citation", f"S{bad[0]}")
        cited.extend(n for n in numbers if n not in cited)
    if claims == 0:
        return CitationCheck(False, "empty")
    return CitationCheck(True, cited=cited)


# -- the pipeline ---------------------------------------------------------------

@dataclass
class Answer:
    question: str
    model: str
    passages: list[Chunk]
    raw: str
    check: CitationCheck
    variants: VariantReport | None
    seconds: float

    @property
    def answered(self) -> bool:
        return self.check.ok


def ask(
    question: str,
    index: KnowledgeIndex,
    client: ModelClient,
    model: str,
    embed_model: str | None = None,
    k: int = 6,
    variant_db: VariantDB | None = None,
    evidence: EvidenceTable | None = None,
) -> Answer:
    started = time.time()
    variants = variant_db.lookup(question) if variant_db is not None else None
    if variants is not None and evidence is not None:
        for record in variants.records:
            lines = evidence.lines(record)
            if lines:
                variants.evidence[record.variation_id] = lines

    embed = None
    if index.embeddings is not None:
        embed_model = embed_model or index.embed_model

        def embed(texts):
            return client.embed(texts, embed_model)

    passages = index.search(question, k=k, embed=embed)
    if not passages:
        return Answer(question, model, [], "", CitationCheck(False, "not_found"),
                      variants, time.time() - started)
    # Variant facts are deliberately NOT in the prompt: the model cannot restate
    # (or misstate) a classification it was never shown.
    raw = client.chat(build_messages(question, passages), model)
    return Answer(question, model, passages, raw, check_citations(raw, len(passages)),
                  variants, time.time() - started)


def render(answer: Answer, show_retrieved: bool = False) -> str:
    out = [DISCLAIMER, ""]

    if answer.variants is not None:
        report = answer.variants
        out.append("VARIANT LOOKUP (ClinVar record as published; not a model output)")
        for record in report.records:
            stars = f"{record.stars} star{'s' if record.stars != 1 else ''}"
            out += [f"  {record.name}",
                    f"    ClinVar classification: {record.classification}",
                    f"    Review status: {record.review_status} ({stars}); "
                    f"last evaluated: {record.last_evaluated or 'not given'}; "
                    f"submitters: {record.submitters}",
                    f"    Condition(s): {record.conditions or 'not given'}",
                    f"    {record.url}"]
            out += [f"    {line}" for line in report.evidence.get(record.variation_id, [])]
        out += [f"  {note}" for note in report.notes]
        out += [f"  Source: {report.source}", ""]

    if answer.check.ok:
        out += [f"ANSWER (model: {answer.model}; from the sources below only)", answer.raw, ""]
        out.append("SOURCES (verbatim excerpts)")
        for number in answer.check.cited:
            passage = answer.passages[number - 1]
            out.append(f"[S{number}] {passage.heading}")
            if passage.private:
                out.append("  (private source: excerpt not displayed)")
            else:
                out += ["  " + line for line in passage.text.splitlines()]
            out += [f"  {passage.attribution}", ""]
    elif answer.check.reason == "not_found":
        out += ["NOT FOUND IN THE SOURCES. The knowledge base does not answer this "
                "question; nothing was generated from the model's own memory.", ""]
    else:
        out += [f"ANSWER WITHHELD: the model's answer failed the citation check "
                f"({_WITHHELD.get(answer.check.reason, answer.check.reason)}), so it is "
                "not shown.", ""]

    if show_retrieved or not answer.check.ok:
        if answer.passages:
            out.append("Retrieved passages (for your own reading):")
            out += [f"  - {p.heading}  {p.url}" for p in answer.passages]
    return "\n".join(out).rstrip() + "\n"
