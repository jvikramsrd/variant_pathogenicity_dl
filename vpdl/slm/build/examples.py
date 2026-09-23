"""Task examples: what the model reads and what it must predict, per task policy.

The input text is built by :func:`model_input`, in this order:

1. conclusion sentences removed (the verdict)                    policy.mask_conclusions
2. other submitters' verdicts removed                            policy.mask_external_classifications
3. sentences that only list ACMG codes dropped; codes masked     policy.mask_acmg_codes
4. sentences citing an INDEPENDENT_VALIDATION publication removed  functional holdout mode
5. restricted documents never become examples

Every example keeps its provenance (document, variant, source, submitter,
date, tier) and how much was removed, so the leakage audit can check the
masking actually happened and a reader can trace any prediction to the words
the model saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from vpdl.slm.labels import normalize_classification
from vpdl.slm.parallel import pmap
from vpdl.slm.schema import CLASSES, NEVER_INPUT, as_list
from vpdl.slm.text.acmg import CODES, TaskPolicy, find_codes, mask_codes, task_policy
from vpdl.slm.text.conclusion import classify_sentence
from vpdl.slm.text.sentences import split_sentences

__all__ = ["InputStats", "model_input", "build_examples", "TASKS", "holdout_patterns",
           "class_index", "code_multi_hot", "summarize_masking"]

TASKS = ("classify", "classify_from_codes", "acmg_codes", "evidence_type", "evidence_polarity")


@dataclass(frozen=True)
class InputStats:
    sentences: int
    conclusions_removed: int
    external_removed: int
    code_lists_removed: int
    codes_masked: int
    holdout_removed: int


def holdout_patterns(identifiers: Iterable[str]) -> re.Pattern | None:
    """'PMID:33357406' / 'DOI:10.1016/...' -> one regex matching either written form."""
    parts = []
    for identifier in identifiers:
        kind, _, value = identifier.partition(":")
        if kind.upper() == "PMID":
            parts.append(rf"(?<!\d){re.escape(value)}(?!\d)")
        elif kind.upper() == "DOI":
            parts.append(re.escape(value))
    return re.compile("|".join(parts), re.I) if parts else None


def model_input(text: str, policy: TaskPolicy, holdout: re.Pattern | None = None) -> tuple[str, InputStats]:
    kept, conclusions, external, code_lists, holdout_removed = [], 0, 0, 0, 0
    sentences = split_sentences(text or "")
    all_codes = find_codes(text or "")
    masked_codes = 0
    for sentence in sentences:
        category = classify_sentence(sentence.text)
        if category == "external_classification" and policy.mask_external_classifications:
            external += 1
            continue
        if category is not None and category != "external_classification" and policy.mask_conclusions:
            conclusions += 1
            continue
        if holdout is not None and holdout.search(sentence.text):
            holdout_removed += 1
            continue
        piece = sentence.text
        codes = [c for c in all_codes if sentence.start <= c.start < sentence.end]
        if codes and policy.mask_acmg_codes:
            shifted = [type(c)(c.code, c.strength, c.normalized, c.start - sentence.start,
                               c.end - sentence.start, c.met) for c in codes]
            piece, n = mask_codes(piece, shifted)
            masked_codes += n
            if len(re.findall(r"[A-Za-z]{3,}", piece)) <= 6 and re.search(
                    r"criteri|applied|acmg|codes?|met\b", piece, re.I):
                code_lists += 1          # "The following criteria were applied:" with nothing left
                continue
        kept.append(piece)
    return " ".join(kept), InputStats(len(sentences), conclusions, external, code_lists,
                                      masked_codes, holdout_removed)


def _input_worker(payload: tuple) -> tuple[str, InputStats]:
    text, policy, holdout = payload
    return model_input(text, policy, holdout)


def _document_examples(tables: Mapping[str, pd.DataFrame], split: pd.DataFrame, policy: TaskPolicy,
                       holdout: re.Pattern | None, min_chars: int,
                       workers: int | None = 1) -> pd.DataFrame:
    documents = tables["documents"]
    frame = documents.merge(split[["document_id", "split"]], on="document_id", how="inner")
    frame = frame.loc[~frame["restricted"].fillna(False).astype(bool)]
    frame = frame.loc[frame["label_category"].isin(["five_class", "pair"])]
    inputs = pmap(_input_worker, [(text or "", policy, holdout) for text in frame["text"]], workers)
    rows = []
    for record, (text, stats) in zip(frame.itertuples(index=False), inputs):
        if len(text) < min_chars:
            continue
        rows.append({"example_id": record.document_id, "document_id": record.document_id,
                     "variant_id": record.variant_id, "gene": record.gene, "split": record.split,
                     "source_id": record.source_id, "submitter": record.submitter, "date": record.date,
                     "tier": record.tier, "input_text": text, **stats.__dict__})
    examples = pd.DataFrame(rows)
    if examples.empty:
        return examples
    labels = frame.set_index("document_id")[["classification_raw"]]
    info = labels.loc[examples["document_id"], "classification_raw"].map(normalize_classification)
    examples["target_label"] = [i.label5 for i in info]
    examples["target_soft"] = [list(i.soft) if i.soft else None for i in info]
    examples["target_binary"] = [i.binary for i in info]
    examples["target_category"] = [i.category for i in info]
    return examples


def build_examples(tables: Mapping[str, pd.DataFrame], split: pd.DataFrame, task: str,
                   holdout_publications: Iterable[str] = (), min_chars: int = 20,
                   features: Iterable[str] = (), workers: int | None = 1) -> pd.DataFrame:
    """Examples for `task` on `split`; refuses answer-describing features.

    `workers` processes clean the narratives (None: every core); the result is
    identical for any worker count.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known {TASKS}")
    policy = task_policy(task)
    forbidden = (set(features) & (NEVER_INPUT | set(policy.forbidden_features)))
    if forbidden:
        raise ValueError(f"task {task!r}: features {sorted(forbidden)} describe the answer or the "
                         "label's review, not the evidence; they cannot be inputs (leakage policy).")
    holdout = holdout_patterns(holdout_publications)

    if task == "classify":
        examples = _document_examples(tables, split, policy, holdout, min_chars, workers)
    elif task == "classify_from_codes":
        units = tables["evidence_units"]
        codes = units.groupby("document_id")["acmg_codes"].agg(
            lambda values: sorted({c for v in values for c in v}))
        codes = codes[codes.map(len) > 0]
        base = _document_examples(tables, split, task_policy("classify"), holdout, 0, workers)
        base = base.loc[base["document_id"].isin(codes.index)].copy()
        base["input_text"] = base["document_id"].map(lambda d: " ".join(codes[d]))
        examples = base
    elif task == "acmg_codes":
        base = _document_examples(tables, split, policy, holdout, min_chars, workers)
        units = tables["evidence_units"]
        met =units.groupby("document_id")["acmg_codes"].agg(lambda v: sorted({c for x in v for c in x}))
        not_met = units.groupby("document_id")["acmg_codes_not_met"].agg(
            lambda v: sorted({c for x in v for c in x}))
        gold = tables.get("acmg_labels")
        if gold is not None and not gold.empty:
            met = pd.concat([met, gold.set_index("document_id")["codes_met"]])
            not_met = pd.concat([not_met, gold.set_index("document_id")["codes_not_met"]])
            met = met[~met.index.duplicated(keep="last")]
            not_met = not_met[~not_met.index.duplicated(keep="last")]
        base = base.loc[base["document_id"].isin(met[met.map(len) > 0].index)].copy()
        base["target_codes"] = base["document_id"].map(lambda d: [c.split("_")[0] for c in met[d]])
        base["target_codes_strength"] = base["document_id"].map(lambda d: list(met[d]))
        base["target_codes_not_met"] = base["document_id"].map(
            lambda d: [c.split("_")[0] for c in not_met.get(d, [])])
        base["code_source"] = base["document_id"].map(
            lambda d: "expert_panel_structured" if gold is not None and not gold.empty
            and d in set(gold["document_id"]) else "narrative_explicit")
        examples = base
    else:                                   # unit-level tasks
        units = tables["evidence_units"].merge(split[["document_id", "split"]], on="document_id")
        documents = tables["documents"].set_index("document_id")
        restricted = documents["restricted"].fillna(False).astype(bool)
        units = units.loc[units["sentence_role"].isin(["evidence", "code_list", "description", "other"])
                          & ~units["document_id"].map(restricted).fillna(False)]
        rows = []
        for u in units.itertuples(index=False):
            if holdout is not None and holdout.search(u.text_span):
                continue
            text, n = mask_codes(u.text_span) if policy.mask_acmg_codes else (u.text_span, 0)
            if len(text) < 10:
                continue
            rows.append({"example_id": u.evidence_id, "document_id": u.document_id,
                         "variant_id": u.variant_id, "gene": u.gene, "split": u.split,
                         "source_id": u.source_id, "date": u.date, "input_text": text,
                         "codes_masked": n, "target_types": list(u.evidence_types),
                         "target_polarity": u.evidence_polarity, "target_role": u.sentence_role,
                         "label_provenance": u.provenance, "label_confidence": u.confidence})
        examples = pd.DataFrame(rows)
    if not examples.empty:
        examples["task"] = task
    return examples.reset_index(drop=True)


def class_index(labels: Iterable[str | None]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(CLASSES)}
    return np.array([lookup.get(label, -1) if label else -1 for label in labels])


def code_multi_hot(code_lists: Iterable[Iterable[str]], vocabulary: tuple[str, ...] = CODES) -> np.ndarray:
    index = {code: i for i, code in enumerate(vocabulary)}
    rows = list(code_lists)
    out = np.zeros((len(rows), len(vocabulary)), dtype=np.float32)
    for r, codes in enumerate(rows):
        for code in as_list(codes):
            base = code.split("_")[0]
            if base in index:
                out[r, index[base]] = 1.0
    return out


def summarize_masking(examples: pd.DataFrame) -> dict[str, Any]:
    columns = ["conclusions_removed", "external_removed", "code_lists_removed", "codes_masked",
               "holdout_removed"]
    present = [c for c in columns if c in examples.columns]
    return {c: int(examples[c].sum()) for c in present} | {"examples": int(len(examples))}
