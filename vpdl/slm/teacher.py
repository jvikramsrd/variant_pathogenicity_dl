"""Teacher distillation: a larger local model labelling TRAINING examples — noisy, filtered, traced.

A teacher's output is noisy supervision, never ground truth. Everything here
is built so that can be checked later:

* only TRAINING examples are ever sent (:func:`prompts_for`), so a teacher
  cannot see an evaluation variant — the leakage audit re-checks it;
* the prompt contains the model input only (conclusions already masked), never
  the label;
* every output records teacher model AND its version/digest, the prompt hash,
  the parse result, a confidence and the time — so a distilled run can be
  compared with the same run without a teacher (the point of EXP-017);
* outputs are kept only if they parse, cite only the evidence ids they were
  given (:func:`filter_outputs` uses the grounding check) and reach the
  confidence threshold. Everything dropped is counted.

Licences: ``qwen3:32b`` is Apache-2.0 and its outputs may train another model
(docs/slm/PLAN.md). Whether MedGemma's Health AI Developer Foundations terms
allow training on its outputs has NOT been checked, so a medgemma teacher
requires ``accept_terms=True`` and says so in every record.

Nothing here has been run: no teacher output exists in this repository
(NOT RUN — REQUIRES DGX SPARK).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from vpdl.slm.schema import as_list

__all__ = ["TeacherConfig", "TeacherRecord", "build_prompt", "prompts_for", "parse_output",
           "filter_outputs", "run_teacher", "unit_problems", "TEACHER_SYSTEM_PROMPT"]

TEACHER_SYSTEM_PROMPT = """You label clinical-genetics evidence for a research dataset.

You are given numbered evidence sentences [E1], [E2], ... taken from a variant
interpretation, with the interpreter's conclusion already removed.

Return ONLY a JSON object:
{"units": [{"id": "E1", "types": ["population"], "polarity": "pathogenic",
            "acmg_codes": ["PM2"], "confidence": 0.0-1.0}, ...],
 "summary": "one sentence per claim, each ending with the ids it rests on, e.g. [E1][E3]",
 "missing_evidence": ["segregation", ...],
 "confidence": 0.0-1.0}

Rules:
1. Use only the sentences given. Do not add facts, numbers, codes or studies.
2. Every sentence of "summary" must end with at least one [E#] id that you were given.
3. Do not state or guess the variant's classification.
4. If a sentence carries no evidence, give it types [] and polarity "neutral".
"""


@dataclass
class TeacherConfig:
    model: str = "qwen3:32b"
    accept_terms: bool = False           # required for models whose terms are unchecked
    min_confidence: float = 0.6
    max_examples: int | None = None
    temperature: float = 0.0
    seed: int = 0
    url: str | None = None               # loopback only (vpdl.kb.ollama enforces it)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


UNCHECKED_TERMS = {"medgemma": "MedGemma's Health AI Developer Foundations terms have not been "
                               "checked for training other models on its outputs"}


@dataclass
class TeacherRecord:
    example_id: str
    variant_id: str
    teacher_model: str
    teacher_version: str
    prompt_sha256: str
    raw: str
    parsed: dict[str, Any] | None
    confidence: float | None
    grounded: bool
    reject_reason: str | None
    created: str
    split_at_generation: str
    licence_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prompt(input_text: str, units: Sequence[Mapping[str, Any]], gene: str = "",
                 consequence: str = "") -> tuple[str, dict[str, dict[str, Any]]]:
    """(user prompt, evidence map {E1: {...}}) — the model input only, never the label."""
    evidence = {}
    lines = []
    for number, unit in enumerate(units, start=1):
        tag = f"E{number}"
        text = unit.get("text_span") or unit.get("text") or ""
        evidence[tag] = {"text": text, "acmg_codes": as_list(unit.get("acmg_codes")),
                         "polarity": unit.get("evidence_polarity"),
                         "evidence_id": unit.get("evidence_id")}
        lines.append(f"[{tag}] {text}")
    context = f"Gene: {gene or 'not given'}. Variant consequence: {consequence or 'not given'}."
    prompt = f"{context}\n\nEvidence sentences:\n" + "\n".join(lines)
    if not lines:
        prompt += f"\n(no evidence sentences)\n\nInterpretation text:\n{input_text}"
    return prompt, evidence


def prompts_for(examples, units_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
                allowed_splits: Iterable[str] = ("train", "mmr_train")) -> list[dict[str, Any]]:
    """Prompts for TRAINING examples only; anything else is refused loudly."""
    allowed = set(allowed_splits)
    bad = sorted(set(examples["split"]) - allowed)
    rows = []
    for record in examples.itertuples(index=False):
        if record.split not in allowed:
            continue
        prompt, evidence = build_prompt(record.input_text,
                                        units_by_document.get(record.document_id, []),
                                        getattr(record, "gene", ""))
        rows.append({"example_id": record.example_id, "variant_id": record.variant_id,
                     "document_id": record.document_id, "split": record.split,
                     "prompt": prompt, "evidence": evidence})
    if bad:
        import logging
        logging.getLogger(__name__).info(
            "teacher prompts: %d example(s) skipped because their split is %s (only %s are sent)",
            int((~examples["split"].isin(list(allowed))).sum()), bad, sorted(allowed))
    return rows


def parse_output(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, "no JSON object in the answer"
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        return None, f"invalid JSON ({error.msg})"
    if not isinstance(parsed, dict) or "units" not in parsed:
        return None, "JSON without a 'units' list"
    return parsed, None


def unit_problems(parsed: Mapping[str, Any], evidence: Mapping[str, Any]) -> list[str]:
    """What is wrong with the answer's ``units`` list: ids never given, labels outside the vocabulary.

    The summary's grounding check does not look at the units, and the units are what
    would train the evidence heads, so they are checked on their own.
    """
    from vpdl.slm.text.evidence import EVIDENCE_TYPES, POLARITIES

    units = parsed.get("units")
    if not isinstance(units, list):
        return ["'units' is not a list"]
    problems = []
    for unit in units:
        if not isinstance(unit, Mapping):
            problems.append(f"unit {unit!r} is not an object")
            continue
        if unit.get("id") not in evidence:
            problems.append(f"unit id {unit.get('id')!r} was not given")
        unknown = [t for t in as_list(unit.get("types")) if t not in EVIDENCE_TYPES]
        if unknown:
            problems.append(f"unit {unit.get('id')!r}: unknown evidence types {unknown}")
        if unit.get("polarity") is not None and unit.get("polarity") not in POLARITIES:
            problems.append(f"unit {unit.get('id')!r}: unknown polarity {unit.get('polarity')!r}")
    return problems


def _confidence(parsed: Mapping[str, Any] | None) -> float | None:
    """The answer's own confidence, or None when it is absent or not a number in [0, 1]."""
    value = (parsed or {}).get("confidence")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def filter_outputs(records: Sequence[TeacherRecord], config: TeacherConfig) -> dict[str, Any]:
    kept, counts = [], {"parsed": 0, "grounded": 0, "confident": 0, "kept": 0, "total": len(records)}
    for record in records:
        if record.parsed is None:
            continue
        counts["parsed"] += 1
        if not record.grounded:
            continue
        counts["grounded"] += 1
        # No stated confidence cannot pass a confidence threshold.
        if record.confidence is None or record.confidence < config.min_confidence:
            continue
        counts["confident"] += 1
        kept.append(record)
    counts["kept"] = len(kept)
    return {"kept": kept, "counts": counts, "config": config.as_dict()}


def run_teacher(prompts: Sequence[Mapping[str, Any]], config: TeacherConfig,
                client=None) -> list[TeacherRecord]:
    """Ask the local teacher for labels. Requires a loopback Ollama; not run in this repository."""
    from vpdl.slm.evaluation.explain import check_grounding

    note = ""
    for name, reason in UNCHECKED_TERMS.items():
        if name in config.model.lower():
            if not config.accept_terms:
                raise PermissionError(f"teacher {config.model!r}: {reason}. Pass accept_terms=True "
                                      "only after checking the terms, or use an Apache-2.0 model "
                                      "such as qwen3:32b.")
            note = reason
    if client is None:
        from vpdl.kb.ollama import LocalOllama
        if config.temperature != 0.0 or config.seed != 0:
            # LocalOllama always sends temperature 0 and seed 0; accepting other values would
            # store a provenance record that misstates what the model was asked with.
            raise ValueError("the local Ollama client runs at temperature 0, seed 0; "
                             f"TeacherConfig asks for temperature {config.temperature}, "
                             f"seed {config.seed}")
        client = LocalOllama(config.url)
    records = []
    for item in prompts[: config.max_examples or len(prompts)]:
        messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": item["prompt"]}]
        raw = client.chat(messages, config.model)
        parsed, reason = parse_output(raw)
        grounded = False
        if parsed is not None:
            report = check_grounding(parsed.get("summary", ""), item["evidence"])
            grounded = report.unsupported_claim_rate in (0.0,) or not report.unsupported
            if not grounded:
                reason = f"ungrounded summary ({len(report.unsupported)} sentence(s))"
            problems = unit_problems(parsed, item["evidence"])
            if problems:
                grounded = False
                reason = f"invalid units: {problems[:3]}"
        records.append(TeacherRecord(
            example_id=item["example_id"], variant_id=item["variant_id"],
            teacher_model=config.model,
            teacher_version=getattr(client, "model_versions", {}).get(config.model, "unrecorded"),
            prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest()[:16], raw=raw,
            parsed=parsed, confidence=_confidence(parsed), grounded=grounded,
            reject_reason=reason, created=time.strftime("%Y-%m-%d %H:%M:%S"),
            split_at_generation=item["split"], licence_note=note))
    return records
