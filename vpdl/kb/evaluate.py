"""Score candidate models on a fixed question set. The best-measured model wins.

Question types (``docs/kb/eval_questions.jsonl``, written before any tuning):

* ``answerable`` — the sources answer it. Checked: did search find the right
  section (``expected_sections``); did the model answer with valid citations;
  does the answer contain the key facts (``answer_keywords``: every group must
  match, any phrase within a group).
* ``refusal`` — the sources do not answer it. Correct behaviour is "not found".
  These matter as much as the rest: a model that always answers scores well on
  the answerable questions and is dangerous on these.
* ``variant`` — checks the lookup, not the model: is the named record found?

Faithfulness — is each sentence really supported by the passage it cites? —
needs a human reader. Every answer is written to ``answers_<model>.jsonl``
(passage ids, not passage text) for that review.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from vpdl.kb.answer import Answer, ask
from vpdl.kb.search import KnowledgeIndex
from vpdl.kb.variants import VariantDB

logger = logging.getLogger(__name__)

__all__ = ["load_questions", "keywords_match", "evaluate"]


def load_questions(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle
                     if line.strip() and not line.lstrip().startswith("//")]
    for question in questions:
        if question.get("type") not in ("answerable", "refusal", "variant"):
            raise ValueError(f"{question.get('id')}: unknown type {question.get('type')!r}")
    return questions


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("–", "-").replace("—", "-"))


def keywords_match(answer: str, groups: Sequence[Sequence[str]]) -> bool:
    text = _normalise(answer)
    return all(any(_normalise(option) in text for option in group) for group in groups)


def evaluate(
    questions: list[dict],
    index: KnowledgeIndex,
    client,
    models: Sequence[str],
    out_dir: Path | str,
    embed_model: str | None = None,
    k: int = 6,
    variant_db: VariantDB | None = None,
) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []

    variant_questions = [q for q in questions if q["type"] == "variant"]
    variant_hits = None
    if variant_questions and variant_db is not None:
        variant_hits = 0
        for question in variant_questions:
            report = variant_db.lookup(question["question"])
            expected = question.get("expected_name_contains", "")
            found = bool(report and any(expected in r.name for r in report.records))
            variant_hits += int(found == question.get("expect_found", True))

    for model in models:
        rows, answers = [], []
        for question in questions:
            if question["type"] == "variant":
                continue
            answer: Answer = ask(question["question"], index, client, model,
                                 embed_model=embed_model, k=k)
            sections = {p.section_id for p in answer.passages}
            row = {
                "id": question["id"], "type": question["type"],
                "answered": answer.answered, "reason": answer.check.reason,
                "detail": answer.check.detail,
                "seconds": round(answer.seconds, 1),
                "retrieved": [p.chunk_id for p in answer.passages],
                "cited": answer.check.cited,
            }
            if question["type"] == "answerable":
                expected = set(question.get("expected_sections", []))
                row["retrieval_hit"] = bool(expected & sections) if expected else None
                row["keywords_ok"] = answer.answered and keywords_match(
                    answer.raw, question.get("answer_keywords", []))
            rows.append(row)
            answers.append({**row, "question": question["question"], "answer": answer.raw})
            logger.info("%s %s: %s", model, question["id"],
                        "answered" if answer.answered else answer.check.reason[:60])

        with (out / f"answers_{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.jsonl").open(
                "w", encoding="utf-8") as handle:
            for record in answers:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        answerable = [r for r in rows if r["type"] == "answerable"]
        refusals = [r for r in rows if r["type"] == "refusal"]
        hits = [r["retrieval_hit"] for r in answerable if r["retrieval_hit"] is not None]

        def share(values) -> float | None:
            values = list(values)
            return round(sum(values) / len(values), 3) if values else None

        summaries.append({
            "model": model,
            "answerable": len(answerable),
            "retrieval_hit": share(hits),
            "answered": share(r["answered"] for r in answerable),
            "correct_facts": share(r["keywords_ok"] for r in answerable),
            "refusals": len(refusals),
            "refused_correctly": share(not r["answered"] for r in refusals),
            "withheld_for_citations": sum(r["reason"] not in ("", "not_found")
                                          for r in rows),
            "variant_lookup_correct": (f"{variant_hits}/{len(variant_questions)}"
                                       if variant_hits is not None else None),
            "mean_seconds": share(r["seconds"] for r in rows),
            # Where the model actually ran, as Ollama reported it (1.0 = all on GPU).
            "gpu_share": getattr(client, "gpu_share", {}).get(model),
        })
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries
