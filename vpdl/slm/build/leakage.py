"""The SLM leakage audit: fifteen checks, and a gate that stops training on a critical one.

Report objects are the DL branch's (``vpdl.dl.leakage.LeakageFinding`` /
``LeakageReport`` / ``LeakageError``, imported unchanged), so both branches'
reports read the same.

    1  label_leakage            answer-describing columns used as inputs
    2  conclusion_leakage       verdict sentences left in model inputs
    3  template_leakage         laboratory template families across train/test
    4  exact_duplicates         identical input text across train/test
    5  near_duplicates          near-identical input text across train/test
    6  variant_duplicates       one variant group (VariationID / genomic change /
                                protein change) on both sides
    7  literature_leakage       a test variant's publications cited for training variants
    8  gene_shortcut            how well the training genes' label rates alone rank test
    9  disease_shortcut         the same for diseases
    10 functional_leakage       independent functional variants, assay values or holdout
                                publications reaching training
    11 temporal_leakage         post-cutoff or undated documents in a temporal train set
    12 transcript_leakage       one genomic change under two ids on both sides
    13 teacher_contamination    teacher outputs generated for non-training examples
    14 benchmark_contamination  evaluation narratives / publications in pretraining text
    15 retrieval_contamination  evaluation documents in a training-time retrieval index

Severities are decided per scheme: variant duplicates are expected under
``random`` (info) and critical everywhere else; near-duplicates are critical
under the text and laboratory schemes (which promise isolation) and a warning
elsewhere, with the rate, so a reader knows what the number rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from vpdl.dl.leakage import LeakageError, LeakageFinding, LeakageReport
from vpdl.evaluate import roc_auc
from vpdl.slm.build.examples import holdout_patterns
from vpdl.slm.build.features import FORBIDDEN
from vpdl.slm.build.splits import variant_groups
from vpdl.slm.clinvar_text import specific_condition
from vpdl.slm.text.conclusion import classify_sentence, residual_assertions
from vpdl.slm.text.sentences import split_sentences

__all__ = ["AuditInputs", "run_audit", "leakage_gate", "LeakageError", "LeakageReport", "CHECKS"]

CHECKS = ("label_leakage", "conclusion_leakage", "template_leakage", "exact_duplicates",
          "near_duplicates", "variant_duplicates", "literature_leakage", "gene_shortcut",
          "disease_shortcut", "functional_leakage", "temporal_leakage", "transcript_leakage",
          "teacher_contamination", "benchmark_contamination", "retrieval_contamination")
ISOLATING = ("text", "laboratory")
FUNCTIONAL_COLUMNS = ("cimra_value", "cimra_strength", "dms_score", "mavedb_score")
EVALUATED = ("test", "val", "broad_test", "mmr_val")


@dataclass
class AuditInputs:
    scheme: str
    examples: pd.DataFrame                           # split, input_text, document_id, variant_id, gene
    variants: pd.DataFrame | None = None             # genomic_key, hgvs_p, protein_variant_id
    documents: pd.DataFrame | None = None            # disease_names, text_pmids
    citations: pd.DataFrame | None = None
    features: tuple[str, ...] = ()
    functional_variants: tuple[str, ...] = ()        # protein ids with independent assay data
    holdout_publications: tuple[str, ...] = ()
    cutoff: str | None = None
    pretraining: Mapping[str, Iterable[str]] | None = None   # {"documents": [...], "pmids": [...]}
    teacher: pd.DataFrame | None = None              # example_id, split_at_generation
    retrieval_documents: Iterable[str] | None = None # document ids in a training-time index
    strict_literature: bool = False
    strict_functional: bool = True
    residual_warning_rate: float = 0.05
    shortcut_warning_auc: float = 0.8
    extra: dict[str, Any] = field(default_factory=dict)


def _train_test(frame: pd.DataFrame, column: str) -> tuple[set, set]:
    values = frame[[column, "split"]].dropna()
    train = set(values.loc[values["split"].isin(["train", "mmr_train"]), column])
    test = set(values.loc[values["split"].isin(EVALUATED), column])
    return train, test


def _crossing(frame: pd.DataFrame, column: str) -> tuple[int, int]:
    if column not in frame.columns:
        return -1, 0
    train, test = _train_test(frame, column)
    shared = train & test
    rows = int(frame.loc[frame["split"].isin(EVALUATED), column].isin(shared).sum())
    return rows, len(shared)


def _check_label(inputs: AuditInputs) -> LeakageFinding:
    bad = sorted(set(inputs.features) & FORBIDDEN)
    if bad:
        return LeakageFinding("label_leakage", "critical",
                              f"features {bad} describe the label or its review", len(bad))
    return LeakageFinding("label_leakage", "info", "no answer-describing feature is an input", 0,
                          {"features": list(inputs.features)})


def _check_conclusions(inputs: AuditInputs) -> list[LeakageFinding]:
    texts = inputs.examples["input_text"].fillna("")
    residual_conclusions = 0
    for text in texts:
        if any(classify_sentence(s.text) for s in split_sentences(text)):
            residual_conclusions += 1
    assertive = texts.map(lambda t: bool(residual_assertions(t))).mean() if len(texts) else 0.0
    findings = [LeakageFinding(
        "conclusion_leakage", "critical" if residual_conclusions else "info",
        f"{residual_conclusions} example input(s) still contain a conclusion sentence"
        if residual_conclusions else "no conclusion sentence in any model input",
        residual_conclusions)]
    findings.append(LeakageFinding(
        "conclusion_leakage", "warning" if assertive > inputs.residual_warning_rate else "info",
        f"{assertive:.1%} of inputs contain an assertive class phrase (broad tripwire; over-counts "
        "evidence such as 'benign in silico')", int(round(assertive * len(texts))),
        {"rate": round(float(assertive), 4)}))
    return findings


def _check_crossing(inputs: AuditInputs, check: str, column: str, what: str) -> LeakageFinding:
    rows, groups = _crossing(inputs.examples, column)
    if rows < 0:
        return LeakageFinding(check, "warning", f"{column} not available: run `vpdl-slm dedup` "
                              "and join clusters before auditing", 0)
    if rows == 0:
        return LeakageFinding(check, "info", f"no {what} crosses train/evaluation", 0)
    severity = "critical" if inputs.scheme in ISOLATING else (
        "info" if inputs.scheme == "random" else "warning")
    n_eval = int(inputs.examples["split"].isin(EVALUATED).sum())
    return LeakageFinding(check, severity,
                          f"{rows} evaluation example(s) ({rows / max(1, n_eval):.1%}) share a {what} "
                          f"with training ({groups} groups)", rows, {"groups": groups})


def _check_variants(inputs: AuditInputs) -> list[LeakageFinding]:
    examples = inputs.examples
    if inputs.variants is not None:
        columns = [c for c in ("genomic_key", "hgvs_p", "protein_variant_id") if c in inputs.variants]
        frame = examples.merge(inputs.variants[["variant_id", *columns]], on="variant_id", how="left")
    else:
        frame = examples.assign(genomic_key=None, hgvs_p=None, protein_variant_id=None)
    frame = frame.assign(gene=frame["gene"].fillna(""))
    frame["_group"] = variant_groups(frame)
    rows, groups = _crossing(frame, "_group")
    severity = "info" if inputs.scheme == "random" else ("critical" if rows else "info")
    findings = [LeakageFinding(
        "variant_duplicates", severity,
        (f"{rows} evaluation example(s) belong to a variant group also in training"
         + (" (expected: the random split does not isolate variants)" if inputs.scheme == "random" else ""))
        if rows else "no variant group crosses train/evaluation", rows, {"groups": groups})]
    extra = 0
    if "genomic_key" in frame:
        keyed = frame.dropna(subset=["genomic_key"])
        train_ids = keyed.loc[keyed["split"].isin(["train", "mmr_train"])].groupby("genomic_key")["variant_id"].agg(set)
        test_ids = keyed.loc[keyed["split"].isin(EVALUATED)].groupby("genomic_key")["variant_id"].agg(set)
        extra = sum(1 for key in train_ids.index.intersection(test_ids.index)
                    if len(train_ids[key] | test_ids[key]) > 1)
    findings.append(LeakageFinding(
        "transcript_leakage", "critical" if extra and inputs.scheme != "random" else "info",
        f"{extra} genomic change(s) appear under different ids on both sides" if extra
        else "no genomic change is split across ids", extra))
    return findings


def _check_literature(inputs: AuditInputs) -> LeakageFinding:
    pmids_by_variant: dict[str, set[str]] = {}
    if inputs.citations is not None and not inputs.citations.empty:
        for variation, source, identifier in inputs.citations[
                ["variation_id", "citation_source", "citation_id"]].itertuples(index=False):
            if source == "PubMed":
                pmids_by_variant.setdefault(f"clinvar:{int(variation)}", set()).add(str(identifier))
    if inputs.documents is not None and "text_pmids" in inputs.documents:
        for variant, pmids in inputs.documents[["variant_id", "text_pmids"]].itertuples(index=False):
            pmids_by_variant.setdefault(variant, set()).update(pmids)
    if not pmids_by_variant:
        return LeakageFinding("literature_leakage", "warning",
                              "no citations supplied: literature leakage NOT checked", 0)
    split_by_variant = inputs.examples.groupby("variant_id")["split"].first()
    train = {p for v, s in split_by_variant.items() if s in ("train", "mmr_train")
             for p in pmids_by_variant.get(v, ())}
    evaluated = [v for v, s in split_by_variant.items() if s in EVALUATED]
    shared = [v for v in evaluated if pmids_by_variant.get(v, set()) & train]
    rate = len(shared) / max(1, len(evaluated))
    severity = "critical" if shared and inputs.strict_literature else ("warning" if shared else "info")
    return LeakageFinding("literature_leakage", severity,
                          f"{len(shared)} evaluation variant(s) ({rate:.1%}) share a cited publication "
                          "with a training variant", len(shared), {"rate": round(rate, 4)})


def _shortcut(inputs: AuditInputs, check: str, keys: pd.Series) -> LeakageFinding:
    examples = inputs.examples.assign(_key=keys.to_numpy())
    if "target_binary" not in examples:
        return LeakageFinding(check, "info", "no binary target: shortcut not measured", 0)
    labelled = examples.dropna(subset=["target_binary", "_key"])
    train = labelled.loc[labelled["split"].isin(["train", "mmr_train"])]
    test = labelled.loc[labelled["split"].isin(["test", "broad_test"])]
    if train.empty or test.empty:
        return LeakageFinding(check, "info", "not measurable on this split", 0)
    rates = train.groupby("_key")["target_binary"].mean()
    prior = float(train["target_binary"].mean())
    scores = test["_key"].map(rates).fillna(prior)
    seen = float(test["_key"].isin(rates.index).mean())
    auc = roc_auc(test["target_binary"].astype(int).to_numpy(), scores.to_numpy())
    severity = "warning" if np.isfinite(auc) and auc >= inputs.shortcut_warning_auc else "info"
    return LeakageFinding(check, severity,
                          f"training label rate per {check.split('_')[0]} alone gives test ROC-AUC "
                          f"{auc:.3f} ({seen:.0%} of test rows have a seen value) — models must beat "
                          "this to claim they read evidence", 0,
                          {"auc": None if not np.isfinite(auc) else round(float(auc), 4),
                           "seen_fraction": round(seen, 4)})


def _check_functional(inputs: AuditInputs) -> LeakageFinding:
    problems = []
    bad_features = sorted(set(inputs.features) & set(FUNCTIONAL_COLUMNS))
    if bad_features:
        problems.append(f"functional values as features {bad_features}")
    if inputs.functional_variants and inputs.variants is not None:
        protein = inputs.examples["variant_id"].map(
            inputs.variants.set_index("variant_id")["protein_variant_id"])
        trained = inputs.examples["split"].isin(["train", "mmr_train"]) & protein.isin(
            set(inputs.functional_variants))
        if trained.any():
            problems.append(f"{int(trained.sum())} training example(s) of independent functional variants")
    pattern = holdout_patterns(inputs.holdout_publications)
    if pattern is not None:
        train_text = inputs.examples.loc[inputs.examples["split"].isin(["train", "mmr_train"]), "input_text"]
        cited = int(train_text.fillna("").map(lambda t: bool(pattern.search(t))).sum())
        if cited:
            problems.append(f"{cited} training input(s) cite a holdout functional publication")
    if not problems:
        return LeakageFinding("functional_leakage", "info",
                              "no independent functional variant, value or publication reaches training", 0)
    severity = "critical" if inputs.strict_functional else "warning"
    return LeakageFinding("functional_leakage", severity, "; ".join(problems), len(problems))


def _check_temporal(inputs: AuditInputs) -> LeakageFinding:
    if inputs.scheme != "temporal":
        return LeakageFinding("temporal_leakage", "info", "not a temporal split", 0)
    if not inputs.cutoff or "date" not in inputs.examples:
        return LeakageFinding("temporal_leakage", "critical", "temporal split without cutoff/dates", 1)
    train = inputs.examples.loc[inputs.examples["split"] == "train"]
    dates = train["date"].fillna("")
    late = int((dates >= inputs.cutoff).sum())
    undated = int((dates == "").sum())
    if late or undated:
        return LeakageFinding("temporal_leakage", "critical",
                              f"{late} training example(s) dated on/after the cutoff, {undated} undated",
                              late + undated)
    return LeakageFinding("temporal_leakage", "info",
                          f"all training examples predate {inputs.cutoff}; publication dates of cited "
                          "papers are NOT checked (needs PubMed years)", 0)


def _check_teacher(inputs: AuditInputs) -> LeakageFinding:
    if inputs.teacher is None or inputs.teacher.empty:
        return LeakageFinding("teacher_contamination", "info", "no teacher outputs used", 0)
    split = inputs.examples.set_index("example_id")["split"]
    at = inputs.teacher["example_id"].map(split)
    bad = int(at.isin(EVALUATED).sum() + (~inputs.teacher["example_id"].isin(split.index)).sum())
    return LeakageFinding("teacher_contamination", "critical" if bad else "info",
                          f"{bad} teacher output(s) belong to non-training or unknown examples"
                          if bad else "teacher outputs exist only for training examples", bad)


def _evaluation_pmids(inputs: AuditInputs) -> set[str]:
    evaluated = set(inputs.examples.loc[inputs.examples["split"].isin(EVALUATED), "variant_id"])
    cited: set[str] = set()
    if inputs.citations is not None and not inputs.citations.empty:
        rows = inputs.citations.loc[inputs.citations["variation_id"].map(
            lambda v: f"clinvar:{int(v)}").isin(evaluated)]
        cited |= set(rows.loc[rows["citation_source"] == "PubMed", "citation_id"].astype(str))
    if inputs.documents is not None and "text_pmids" in inputs.documents:
        rows = inputs.documents.loc[inputs.documents["variant_id"].isin(evaluated), "text_pmids"]
        cited |= {str(p) for values in rows for p in values}
    return cited


def _check_pretraining(inputs: AuditInputs) -> LeakageFinding:
    """Two kinds of file are accepted, and they mean opposite things.

    ``kind: "exclusions"`` (written by ``vpdl-slm roles``) lists what pretraining
    must NOT contain: the check is then that it covers THIS split's evaluation
    set — an exclusion list built for another split is the common way to think
    you are protected and not be.
    ``kind: "corpus"`` lists what pretraining DOES contain: the check is
    membership.
    """
    if inputs.pretraining is None:
        return LeakageFinding("benchmark_contamination", "warning",
                              "no pretraining manifest given: contamination NOT checked", 0)
    documents = set(inputs.pretraining.get("documents", ()))
    pmids = {str(p) for p in inputs.pretraining.get("pmids", ())}
    evaluated_documents = set(inputs.examples.loc[inputs.examples["split"].isin(EVALUATED), "document_id"])
    cited = _evaluation_pmids(inputs)
    kind = str(inputs.pretraining.get("kind", "corpus"))

    if kind == "exclusions":
        uncovered = evaluated_documents - documents
        if uncovered:
            return LeakageFinding(
                "benchmark_contamination", "critical",
                f"{len(uncovered)} evaluation narrative(s) of this split are NOT in the pretraining "
                "exclusion list (was the list built from a different split?)", len(uncovered),
                {"examples": sorted(uncovered)[:3]})
        missing_pmids = cited - pmids
        severity = ("critical" if missing_pmids and inputs.strict_literature else
                    "warning" if missing_pmids else "info")
        return LeakageFinding(
            "benchmark_contamination", severity,
            f"{len(missing_pmids)} publication(s) cited for evaluation variants are not excluded "
            "from pretraining text" if missing_pmids
            else "the exclusion list covers every evaluation narrative and cited publication",
            len(missing_pmids))

    overlap = len(documents & evaluated_documents)
    if overlap:
        return LeakageFinding("benchmark_contamination", "critical",
                              f"{overlap} evaluation narrative(s) are in the pretraining corpus", overlap)
    pmid_overlap = len(pmids & cited)
    severity = ("critical" if pmid_overlap and inputs.strict_literature else
                "warning" if pmid_overlap else "info")
    return LeakageFinding("benchmark_contamination", severity,
                          f"{pmid_overlap} publication(s) cited for evaluation variants are in pretraining text"
                          if pmid_overlap else "no evaluation narrative or cited publication in pretraining",
                          pmid_overlap)


def _check_retrieval(inputs: AuditInputs) -> LeakageFinding:
    if inputs.retrieval_documents is None:
        return LeakageFinding("retrieval_contamination", "info", "no training-time retrieval index", 0)
    index = set(inputs.retrieval_documents)
    evaluated = set(inputs.examples.loc[inputs.examples["split"].isin(EVALUATED), "document_id"])
    overlap = len(index & evaluated)
    return LeakageFinding("retrieval_contamination", "critical" if overlap else "info",
                          f"{overlap} evaluation document(s) are retrievable during training"
                          if overlap else "the training retrieval index holds no evaluation document",
                          overlap)


def run_audit(inputs: AuditInputs) -> LeakageReport:
    report = LeakageReport(scheme=inputs.scheme)
    examples = inputs.examples
    if "exact_group" not in examples and "input_text" in examples:
        from vpdl.slm.text.dedup import exact_group
        inputs.examples = examples = examples.assign(exact_group=examples["input_text"].map(exact_group))
    report.add(_check_label(inputs), *_check_conclusions(inputs),
               _check_crossing(inputs, "template_leakage", "template_cluster", "laboratory template"),
               _check_crossing(inputs, "exact_duplicates", "exact_group", "identical input text"),
               _check_crossing(inputs, "near_duplicates", "near_dup_cluster", "near-duplicate text"),
               *_check_variants(inputs), _check_literature(inputs))
    if inputs.scheme == "gene":
        report.add(LeakageFinding("gene_shortcut", "info", "gene holdout: test genes are unseen", 0))
    else:
        report.add(_shortcut(inputs, "gene_shortcut", examples["gene"]))
    if inputs.documents is not None and "disease_names" in inputs.documents:
        disease = examples["document_id"].map(inputs.documents.set_index("document_id")["disease_names"].map(
            lambda names: next((n for n in names if specific_condition(n)), None)))
        report.add(_shortcut(inputs, "disease_shortcut", disease))
    else:
        report.add(LeakageFinding("disease_shortcut", "info", "no documents given: not measured", 0))
    report.add(_check_functional(inputs), _check_temporal(inputs), _check_teacher(inputs),
               _check_pretraining(inputs), _check_retrieval(inputs))
    return report


def leakage_gate(inputs: AuditInputs) -> LeakageReport:
    report = run_audit(inputs)
    report.assert_clean()
    return report
