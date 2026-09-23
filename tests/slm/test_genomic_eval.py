"""Evaluation, outputs and the interfaces around them: metrics, calibration, uncertainty,
VUS, grounded explanations, the DL/SLM contract, retrieval, the teacher, baselines, configs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vpdl.slm.baselines import run_baselines
from vpdl.slm.config import config_problems, load_finetune_config
from vpdl.slm.evaluation.calibration import (TemperatureScaling, VectorScaling, calibration_summary,
                                             classwise_ece, fit_calibrator, grouped_calibration,
                                             softmax, top_label_ece)
from vpdl.slm.evaluation.explain import check_grounding, explain
from vpdl.slm.evaluation.metrics import (binary_metrics, five_class_metrics, multilabel_metrics,
                                         pathogenic_score, span_f1, validation_threshold)
from vpdl.slm.evaluation.uncertainty import abstain, entropy, evaluate_uncertainty, mutual_information
from vpdl.slm.evaluation.vus import ranking_metrics, reclassification_outcomes, subtier_agreement, vus_scores
from vpdl.slm.experiments import EXPERIMENTS, HYPOTHESES, STATUS, experiments_markdown
from vpdl.slm.hardware import INTENDED_TARGET, hardware_report
from vpdl.slm.interface import DLInput, SLMRepresentation, read_slm_outputs, write_slm_outputs
from vpdl.slm.retrieval import EvidenceIndex, RoleFilterError
from vpdl.slm.schema import CLASSES
from vpdl.slm.teacher import TeacherConfig, TeacherRecord, build_prompt, filter_outputs, parse_output, prompts_for


def _probs(rows: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return softmax(rng.normal(size=(rows, len(CLASSES))))


# -- metrics -------------------------------------------------------------------------

def test_five_class_metrics_report_every_class_and_a_confusion_matrix():
    y = np.array([0, 1, 2, 3, 4, 0, 4])
    probs = np.eye(5)[y] * 0.8 + 0.04
    report = five_class_metrics(y, probs)
    assert report["accuracy"] == 1.0 and report["mcc"] > 0.9
    assert set(report["per_class"]) == set(CLASSES)
    assert np.array(report["confusion"]).shape == (5, 5)


def test_the_binary_view_collapses_the_scale_and_ignores_vus():
    probs = np.array([[0.7, 0.2, 0.05, 0.03, 0.02], [0.02, 0.03, 0.05, 0.2, 0.7],
                      [0.1, 0.1, 0.6, 0.1, 0.1]])
    scores = pathogenic_score(probs)
    assert scores[0] > 0.9 and scores[1] < 0.1
    assert 0.4 < scores[2] < 0.6                     # a VUS-heavy row sits in the middle
    report = binary_metrics([1.0, 0.0], scores[:2], threshold=0.5)
    assert report["roc_auc"] == 1.0 and report["balanced_accuracy"] == 1.0


def test_scoring_test_data_without_a_validation_threshold_is_refused():
    with pytest.raises(ValueError, match="validation"):
        binary_metrics([1.0, 0.0, 1.0], [0.9, 0.1, 0.8], threshold=None, is_validation=False)
    assert validation_threshold([1.0, 0.0], [0.9, 0.1]) <= 0.9


def test_multi_label_and_span_scores_behave():
    truth = np.array([[1, 0, 1], [0, 1, 0]])
    predicted = np.array([[0.9, 0.2, 0.8], [0.1, 0.9, 0.4]])
    report = multilabel_metrics(truth, predicted, ["a", "b", "c"])
    assert report["micro_f1"] == 1.0
    gold = [("d1", 0, 20, "population")]
    assert span_f1([("d1", 2, 20, "population")], gold)["f1"] == 1.0
    assert span_f1([("d1", 30, 40, "population")], gold)["matched"] == 0
    assert span_f1([("d1", 0, 20, "functional")], gold)["matched"] == 0


# -- calibration -----------------------------------------------------------------------

def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, size=300)
    logits = np.eye(5)[y] * 6 + rng.normal(scale=2.0, size=(300, 5))    # sharp and often wrong
    before = top_label_ece(softmax(logits), y)
    scaler = TemperatureScaling().fit(logits, y)
    after = top_label_ece(scaler.transform(logits), y)
    assert after < before and scaler.temperature != 1.0


def test_vector_scaling_fits_per_class_and_stays_a_distribution():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, size=200)
    logits = np.eye(5)[y] * 3 + rng.normal(size=(200, 5))
    probs = VectorScaling().fit(logits, y).transform(logits)
    assert np.allclose(probs.sum(1), 1.0)


def test_calibration_cannot_be_fitted_on_test_rows():
    logits, y = _probs(10), np.arange(10) % 5
    with pytest.raises(ValueError, match="validation"):
        fit_calibrator("temperature", logits, y, ["test"] * 10)
    calibrator = fit_calibrator("temperature", logits, y, ["val"] * 10)
    assert calibrator.fitted_on == "val"


def test_calibration_is_summarised_pooled_and_per_group():
    y = np.arange(20) % 5
    probs = _probs(20)
    summary = calibration_summary(probs, y)
    assert {"top_label_ece", "classwise_ece", "brier", "nll", "reliability"} <= set(summary)
    assert set(classwise_ece(probs, y)) == set(CLASSES)
    grouped = grouped_calibration(probs, y, ["MLH1"] * 10 + ["BRCA1"] * 10, min_n=5)
    assert set(grouped) == {"MLH1", "BRCA1"}


# -- uncertainty ------------------------------------------------------------------------

def test_uncertainty_signals_are_scored_by_how_well_they_predict_errors():
    y = np.arange(60) % 5
    confident_right = np.eye(5)[y] * 0.9 + 0.02
    result = evaluate_uncertainty(confident_right, y)
    assert result["error_rate"] == 0.0
    noisy = np.full((60, 5), 0.2)
    mixed = np.vstack([confident_right[:30], noisy[30:]])
    y_mixed = np.concatenate([y[:30], (y[30:] + 1) % 5])
    scored = evaluate_uncertainty(mixed, y_mixed)
    assert scored["error_detection_auroc"]["entropy"] > 0.9
    assert 0 <= scored["risk_coverage"]["aurc"] <= 1


def test_disagreement_between_passes_is_its_own_signal():
    agree = np.stack([np.full((5, 5), 0.2)] * 3)
    disagree = np.stack([np.eye(5), np.eye(5)[::-1], np.full((5, 5), 0.2)])
    assert float(mutual_information(agree).mean()) < float(mutual_information(disagree).mean())
    assert entropy(np.full((1, 5), 0.2))[0] == pytest.approx(np.log(5))


def test_abstention_reports_why_it_abstained():
    probs = np.array([[0.9, 0.05, 0.02, 0.02, 0.01], [0.3, 0.25, 0.2, 0.15, 0.1]])
    mask, reasons = abstain(probs, min_confidence=0.5, evidence_count=[3, 0], min_evidence=1)
    assert not mask[0] and mask[1]
    assert set(reasons[1]) == {"low_confidence", "no_informative_evidence"}


# -- VUS ---------------------------------------------------------------------------------

def test_vus_priority_ranks_pathogenic_leaning_variants_first():
    probs = np.array([[0.4, 0.3, 0.2, 0.05, 0.05], [0.02, 0.03, 0.2, 0.35, 0.4]])
    scores = vus_scores(probs)
    assert scores["priority"].iloc[0] > scores["priority"].iloc[1]
    assert scores["benign_priority"].iloc[1] > scores["benign_priority"].iloc[0]


def test_reclassification_outcomes_come_from_two_releases():
    old = pd.DataFrame({"variant_id": ["clinvar:1", "clinvar:2", "clinvar:3"],
                        "clinvar_classification": ["Uncertain significance"] * 3})
    new = pd.DataFrame({"variant_id": ["clinvar:1", "clinvar:2", "clinvar:3"],
                        "clinvar_classification": ["Pathogenic", "Likely benign",
                                                   "Uncertain significance"]})
    outcomes = reclassification_outcomes(old, new)
    assert outcomes.set_index("variant_id")["outcome"].to_dict() == {
        "clinvar:1": 1.0, "clinvar:2": 0.0, "clinvar:3": None} or \
        outcomes["outcome"].isna().sum() == 1


def test_vus_ranking_reports_enrichment_over_the_base_rate():
    priority = np.linspace(1, 0, 20)
    outcome = np.array([1] * 5 + [0] * 15)
    report = ranking_metrics(priority, outcome, ks=(5,))
    assert report["roc_auc"] == 1.0
    assert report["top_k"]["5"]["enrichment"] == pytest.approx(4.0)


def test_clinvars_own_vus_tiers_can_be_compared_with_the_ranking():
    agreeing = subtier_agreement([0.9, 0.5, 0.1], [["vus_high"], ["vus_mid"], ["vus_low"]])
    assert agreeing["spearman"] == pytest.approx(1.0)
    disagreeing = subtier_agreement([0.1, 0.5, 0.9], [["vus_high"], ["vus_mid"], ["vus_low"]])
    assert disagreeing["spearman"] == pytest.approx(-1.0)
    assert "note" in subtier_agreement([0.5], [["vus_high"]])


# -- explanations --------------------------------------------------------------------------

def _units():
    return [
        {"evidence_id": "SCV1#1", "document_id": "SCV1", "source_id": "clinvar_submission_summary",
         "sentence_role": "evidence", "text_span": "The variant is absent from gnomAD.",
         "evidence_types": ["population"], "evidence_polarity": "pathogenic", "acmg_codes": ["PM2"],
         "confidence": 0.8},
        {"evidence_id": "SCV1#2", "document_id": "SCV1", "source_id": "clinvar_submission_summary",
         "sentence_role": "evidence", "text_span": "It segregated with disease in 3 relatives.",
         "evidence_types": ["segregation"], "evidence_polarity": "pathogenic", "acmg_codes": [],
         "confidence": 0.6},
    ]


def test_a_built_explanation_cites_everything_it_says():
    probs = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    explanation = explain(probs, _units(), predicted_codes=["PP1"])
    report = check_grounding(explanation["explanation"],
                             {e["id"]: {"text": e["text"], "acmg_codes": e["acmg_codes"],
                                        "polarity": e["polarity"]} for e in explanation["key_evidence"]},
                             predicted_codes=["PP1"])
    assert report.unsupported == [] and report.evidence_coverage == 1.0
    assert explanation["classification"] == "pathogenic"
    assert "functional" in explanation["missing_evidence"]
    assert explanation["disclaimer"].startswith("RESEARCH USE ONLY")


def test_an_explanation_that_invents_evidence_is_caught():
    evidence = {"E1": {"text": "The variant is absent from gnomAD.", "acmg_codes": ["PM2"],
                       "polarity": "pathogenic"}}
    report = check_grounding(
        "The variant segregated with disease in 7 families [E1]. "
        "A functional assay showed loss of function [E1]. "
        "It meets PS3 [E1]. "
        "This is stated without any citation.", evidence)
    problems = " ".join(p for item in report.unsupported for p in item["problems"])
    assert "numbers not in cited evidence" in problems
    assert "segregation" in problems or "functional" in problems
    assert "PS3" in problems
    assert "no citation" in problems
    assert report.unsupported_claim_rate == 1.0


def test_a_claim_in_the_opposite_direction_of_its_source_is_a_contradiction():
    evidence = {"E1": {"text": "A functional assay showed normal activity.", "acmg_codes": [],
                       "polarity": "benign"}}
    report = check_grounding("The assay showed markedly reduced activity [E1].", evidence)
    assert report.contradictions == 1


# -- the DL / SLM contract --------------------------------------------------------------------

def _representation(variant_id="clinvar:1", **kwargs):
    probabilities = kwargs.pop("class_probabilities",
                               {"pathogenic": 0.5, "likely_pathogenic": 0.2, "vus": 0.2,
                                "likely_benign": 0.05, "benign": 0.05})
    return SLMRepresentation(
        variant_id=variant_id, embedding=np.arange(4, dtype=np.float32), score=0.7, uncertainty=0.3,
        metadata={"gene": "MLH1", "model_version": "m1", "feature_version": "f1",
                  "protein_variant_id": "P40692:67:G>R", "uncertainty_method": "predictive_entropy"},
        class_probabilities=probabilities, explanation="grounded text [E1].", **kwargs)


def test_slm_records_round_trip_through_files(tmp_path):
    paths = write_slm_outputs([_representation(), _representation("clinvar:2")], tmp_path)
    back = read_slm_outputs(paths["jsonl"])
    assert [r.variant_id for r in back] == ["clinvar:1", "clinvar:2"]
    assert back[0].pathogenic_probability == pytest.approx(0.7)
    assert back[0].metadata["protein_variant_id"] == "P40692:67:G>R"
    assert json.loads((tmp_path / "slm_outputs.schema.json").read_text())["title"]


def test_the_slm_representation_is_the_dl_branchs_reasoning_modality():
    from vpdl.dl.interface import ModalityRepresentation, ReasoningRepresentation
    representation = _representation()
    assert isinstance(representation, ReasoningRepresentation)
    assert isinstance(representation, ModalityRepresentation)
    assert representation.modality == "reasoning"


def test_impossible_probabilities_are_refused():
    with pytest.raises(ValueError, match="sum"):
        _representation(class_probabilities={"pathogenic": 0.9, "likely_pathogenic": 0.9,
                                             "vus": 0.0, "likely_benign": 0.0, "benign": 0.0})
    with pytest.raises(ValueError, match="missing"):
        _representation(class_probabilities={"pathogenic": 1.0})


def test_dl_representations_are_read_with_their_own_dimensions(tmp_path):
    from vpdl.dl.interface import DLRepresentation, write_dl_outputs
    representations = [
        DLRepresentation("P40692:67:G>R", np.ones(6, dtype=np.float32), 0.8, 0.1,
                         {"gene": "MLH1", "fold": "MLH1", "model_version": "dl1",
                          "feature_version": "f1", "dataset_version": "d1"}),
        DLRepresentation("P43246:1:A>C", np.zeros(6, dtype=np.float32), 0.2, 0.2,
                         {"gene": "MSH2", "fold": "MLH1", "model_version": "dl1",
                          "feature_version": "f1", "dataset_version": "d1"})]
    paths = write_dl_outputs(representations, tmp_path)
    dl = DLInput.from_outputs(paths["jsonl"])
    assert dl.dimension == 6
    matrix, mask = dl.matrix(["P40692:67:G>R", None, "unknown"])
    assert matrix.shape == (3, 6) and list(mask) == [1.0, 0.0, 0.0]
    assert matrix[1].sum() == 0.0                      # missing is zeros + mask 0, never imputed
    folds = dl.fold_check(["P40692:67:G>R", "P43246:1:A>C"], ["MLH1", "MSH2"])
    assert folds["out_of_fold"] == 1 and folds["in_fold_or_other"] == 1


# -- retrieval ---------------------------------------------------------------------------------

def _units_frame():
    return pd.DataFrame({
        "evidence_id": ["a#1", "b#1", "c#1"], "document_id": ["a", "b", "c"],
        "variant_id": ["clinvar:1", "clinvar:2", "clinvar:3"], "gene": ["MLH1", "MSH2", "BRCA1"],
        "source_id": ["clinvar_submission_summary"] * 3, "sentence_role": ["evidence"] * 3,
        "text_span": ["absent from gnomAD in MLH1", "segregated with disease in MSH2",
                      "a functional assay in BRCA1 showed loss of function"],
        "evidence_types": [["population"], ["segregation"], ["functional"]],
        "pmids": [[], ["123456"], []], "date": ["2020-01-01", "2024-01-01", "2024-01-01"],
        "provenance": ["rule:evidence-lexicon/v1"] * 3})


def test_a_training_index_refuses_evaluation_documents():
    roles = pd.Series({"a": "TRAINING", "b": "TEST", "c": "TRAINING"})
    with pytest.raises(RoleFilterError, match="training-time index"):
        EvidenceIndex.build(_units_frame(), roles_by_document=roles, for_training=True)
    index = EvidenceIndex.build(_units_frame().iloc[[0, 2]], roles_by_document=roles)
    assert index.document_ids() == ["a", "c"]


def test_retrieval_returns_provenance_and_respects_an_as_of_date():
    index = EvidenceIndex.build(_units_frame(), for_training=False)
    items = index.search("functional assay loss of function", k=2)
    assert items and items[0].evidence_id == "c#1"
    assert items[0].provenance and items[0].source_id
    old = index.search("segregated with disease", k=2, as_of="2021-01-01")
    assert all(item.timestamp <= "2021-01-01" for item in old)
    assert not index.search("absent from gnomAD", k=2, exclude_variants=["clinvar:1"]) or \
        all(i.variant_id != "clinvar:1" for i in index.search("absent from gnomAD", k=2,
                                                              exclude_variants=["clinvar:1"]))


# -- teacher ------------------------------------------------------------------------------------

def test_only_training_examples_get_a_teacher_prompt():
    examples = pd.DataFrame({"example_id": ["e1", "e2"], "variant_id": ["clinvar:1", "clinvar:2"],
                             "document_id": ["d1", "d2"], "split": ["train", "test"],
                             "input_text": ["absent from gnomAD", "segregated with disease"],
                             "gene": ["MLH1", "MSH2"]})
    prompts = prompts_for(examples, {"d1": _units()})
    assert [p["example_id"] for p in prompts] == ["e1"]
    assert "E1" in prompts[0]["evidence"] and "gnomAD" in prompts[0]["prompt"]


def test_the_prompt_never_contains_the_label():
    prompt, evidence = build_prompt("input", _units(), gene="MLH1")
    assert "pathogenic" not in prompt.lower() and "classified" not in prompt.lower()
    assert set(evidence) == {"E1", "E2"}


def test_teacher_answers_are_parsed_and_filtered():
    parsed, problem = parse_output('noise {"units": [{"id": "E1"}], "confidence": 0.9} tail')
    assert problem is None and parsed["units"][0]["id"] == "E1"
    assert parse_output("not json")[0] is None
    records = [
        TeacherRecord("e1", "clinvar:1", "qwen3:32b", "v", "hash", "raw", {"units": []}, 0.9, True,
                      None, "now", "train"),
        TeacherRecord("e2", "clinvar:2", "qwen3:32b", "v", "hash", "raw", {"units": []}, 0.2, True,
                      None, "now", "train"),
        TeacherRecord("e3", "clinvar:3", "qwen3:32b", "v", "hash", "raw", None, None, False,
                      "invalid JSON", "now", "train")]
    result = filter_outputs(records, TeacherConfig(min_confidence=0.6))
    assert result["counts"] == {"total": 3, "parsed": 2, "grounded": 2, "confident": 1, "kept": 1}


def test_a_teacher_with_unchecked_terms_needs_an_explicit_decision():
    with pytest.raises(PermissionError, match="terms"):
        from vpdl.slm.teacher import run_teacher
        run_teacher([], TeacherConfig(model="medgemma:27b"), client=object())


# -- baselines ------------------------------------------------------------------------------------

def test_baselines_run_on_the_same_splits_and_metrics(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    labels = rng.choice(["pathogenic", "benign"], size=n)
    texts = ["absent from gnomAD and segregated with disease" if label == "pathogenic"
             else "present at high frequency in the population and predicted tolerated"
             for label in labels]
    examples = pd.DataFrame({
        "example_id": [f"e{i}" for i in range(n)], "document_id": [f"d{i}" for i in range(n)],
        "variant_id": [f"clinvar:{i}" for i in range(n)], "gene": ["G"] * n,
        "split": ["train"] * 40 + ["val"] * 10 + ["test"] * 10, "input_text": texts,
        "target_label": labels, "target_soft": [None] * n,
        "target_binary": [1.0 if label == "pathogenic" else 0.0 for label in labels]})
    results = run_baselines(examples, tmp_path, ("majority", "tfidf_lr", "tfidf_svm"))
    assert (tmp_path / "baselines.json").exists()
    for kind in ("majority", "tfidf_lr", "tfidf_svm"):
        assert "five_class" in results[kind]["test"]
    assert results["tfidf_lr"]["test"]["five_class"]["accuracy"] >= \
        results["majority"]["test"]["five_class"]["accuracy"]


# -- experiments, hardware, configuration -----------------------------------------------------------

def test_the_experiment_matrix_is_complete_and_unrun():
    identifiers = [e.id for e in EXPERIMENTS]
    assert identifiers == sorted(identifiers) and len(identifiers) == 25
    assert all(e.status == STATUS and "NOT RUN" in e.status for e in EXPERIMENTS)
    assert all(e.command for e in EXPERIMENTS)
    for experiment in EXPERIMENTS:
        assert experiment.hypothesis is None or all(
            part in HYPOTHESES for part in experiment.hypothesis.split(","))
    markdown = experiments_markdown()
    assert "EXP-025" in markdown and "Clinical-text ablations" in markdown


def test_the_hardware_report_separates_what_is_here_from_what_is_intended():
    report = hardware_report()
    assert report["intended_target"] == INTENDED_TARGET
    assert isinstance(report["running_on_intended_target"], bool)
    if not report["running_on_intended_target"]:
        assert any("NOT the intended target" in note for note in report["recommendations"])
    assert "pyarrow" in report["checks"]


def test_a_typo_in_a_config_is_an_error_not_a_default(tmp_path):
    good = tmp_path / "good.toml"
    good.write_text('[finetune]\nrun_name = "x"\nexamples = "missing.parquet"\nmax_length = 128\n')
    config = load_finetune_config(good)
    assert config.max_length == 128
    assert any("examples" in problem for problem in config_problems(config))
    bad = tmp_path / "bad.toml"
    bad.write_text('[finetune]\nmax_lenght = 128\n')
    with pytest.raises(ValueError, match="unknown setting"):
        load_finetune_config(bad)
